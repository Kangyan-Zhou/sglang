//! In-process, ZMQ-fed cache-aware load balancing policy.
//!
//! `CacheAwareZmqPolicy` ties together the four primitives in
//! [`super::kv_events`]:
//!
//! * `wire` — msgpack decoding of SGLang `KVEventBatch` payloads
//! * `hash` — `compute_block_hashes` matches the publisher's hash chain
//! * `tree` — `HashTree` per model, fed by decoded events
//! * `subscriber` — `KvEventSubscriberRegistry` owns one ZMQ SUB per
//!   `(worker_url, dp_rank)` and forwards `WorkerEvent`s over an mpsc
//!   channel
//!
//! This file owns the indexer side: one consumer task drains the mpsc
//! channel and dispatches each event to the right per-model `HashTree`.
//! [`CacheAwareZmqPolicy::add_worker`]/[`remove_worker`] keep the subscriber
//! registry and a small `worker_url -> model_id` side-table in sync. On the
//! request path, [`select_worker`] tokenizes the request, computes a chain
//! of block hashes, queries the right tree, and either routes by best
//! prefix match (if above the cache threshold and load is balanced) or
//! falls back to min-load.
//!
//! # Concurrency
//!
//! The policy is `Send + Sync`. Per-model trees are `Arc<HashTree>`s
//! held inside a `DashMap<String /*model_id*/, Arc<HashTree>>` so:
//!
//! * The consumer task (event side) and request-path code (router side)
//!   share trees without serializing on the policy itself.
//! * Each `HashTree` has its own internal `RwLock`; matches are read-only
//!   under that lock and apply path is write-only — that's the lock that
//!   actually serialises, not anything in this file.
//!
//! `worker_models` is a `DashMap<String /*url*/, WorkerInfo>` (model_id
//! plus the dp_size recorded at registration) used by the consumer to
//! learn which tree an event applies to and by `remove_worker` to clear
//! every rank that was ever indexed.
//!
//! # Tokenizer-not-loaded behaviour
//!
//! `TokenizerRegistry` is async-loaded. If the tokenizer for the routing
//! model is not yet present (or encoding fails), `select_worker` falls
//! back to min-load routing rather than dropping the request — so a
//! gateway brought up before all tokenizers finished loading still routes
//! traffic, just without cache-aware affinity.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::kv_events::{
    compute_block_hashes,
    discovery::{fetch_event_config, EventConfig},
    subscriber::{KvEventSubscriberRegistry, WorkerEvent},
    tree::{HashTree, KvWorkerId, MatchResult},
    wire::KvCacheEvent,
};
use super::{
    get_healthy_worker_indices, normalize_model_key, utils::PeriodicTask, CacheAwareConfig,
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;
use crate::tokenizer::TokenizerRegistry;

/// Per-worker bookkeeping the policy keeps so the consumer task can
/// locate the right tree for an incoming event and so `remove_worker`
/// can clear *all* dp ranks the worker was registered with — not just
/// the ranks reported by the worker's *current* `dp_size` (which may
/// have changed since registration).
#[derive(Clone, Debug)]
struct WorkerInfo {
    /// Normalized model_id this worker was registered against.
    model_key: String,
    /// `dp_size` recorded at registration time. Used by `remove_worker`
    /// to walk every rank we may have indexed.
    dp_size: u32,
}

/// Cache-aware policy fed by SGLang KV-cache events received over ZMQ.
///
/// One [`HashTree`] per model_id, one [`KvEventSubscriberRegistry`] for
/// the gateway as a whole. See the module-level docs for the wiring
/// diagram.
///
/// `Debug` is implemented manually because `KvEventSubscriberRegistry`
/// and `TokenizerRegistry` are not `Debug` themselves; printing only the
/// shallow fields is enough for tracing/logging.
pub struct CacheAwareZmqPolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<HashTree>>>,
    /// Reverse index: `worker_url -> WorkerInfo`. Populated by
    /// [`add_worker`] and consulted by the consumer task to find the right
    /// tree for an incoming event. The wire payload does not carry a
    /// model_id, so the gateway must remember it per-worker. We also
    /// remember the `dp_size` recorded at registration so a later
    /// `remove_worker` clears every indexed rank even if the worker now
    /// reports a smaller `dp_size`.
    worker_models: Arc<DashMap<String, WorkerInfo>>,
    subscribers: Arc<KvEventSubscriberRegistry>,
    tokenizer_registry: Arc<TokenizerRegistry>,
    /// HTTP client used to introspect each worker's `/server_info` at
    /// `add_worker` time. Reused across workers so we keep a single
    /// connection pool. A failure here is non-fatal: the policy falls
    /// back to constructing an `EventConfig` from the global config.
    http_client: reqwest::Client,
    /// Owns the consumer task that drains [`mpsc::Receiver<WorkerEvent>`]
    /// and applies events to the trees. `Some` until `shutdown()`.
    consumer_task: parking_lot::Mutex<Option<JoinHandle<()>>>,
    /// `Some` if the eviction interval was non-zero. The `PeriodicTask`
    /// is dropped on policy drop, which signals the worker thread.
    _eviction_task: Option<PeriodicTask>,
    cancel: CancellationToken,
}

impl CacheAwareZmqPolicy {
    /// Construct a policy and start its background consumer + (optional)
    /// eviction tasks. `tokenizer_registry` is shared with the rest of the
    /// gateway; this policy only reads from it.
    pub fn new(config: CacheAwareConfig, tokenizer_registry: Arc<TokenizerRegistry>) -> Self {
        let trees: Arc<DashMap<String, Arc<HashTree>>> = Arc::new(DashMap::new());
        let worker_models: Arc<DashMap<String, WorkerInfo>> = Arc::new(DashMap::new());
        let cancel = CancellationToken::new();

        let (tx, rx) = mpsc::channel::<WorkerEvent>(config.event_channel_capacity.max(1));
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx));

        // Spawn the consumer task. It pulls WorkerEvents off the channel
        // and dispatches each batch to the right per-model tree.
        let consumer_task = spawn_consumer_task(
            rx,
            Arc::clone(&trees),
            Arc::clone(&worker_models),
            cancel.clone(),
        );

        // Spawn the periodic eviction task if configured.
        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "ZmqCacheEviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        let evicted = tree.evict_lru(max_tree_size);
                        if evicted > 0 {
                            debug!(
                                model = %model_id,
                                evicted,
                                max_tree_size,
                                "ZMQ cache eviction"
                            );
                        }
                    }
                },
            ))
        } else {
            None
        };

        // Build the HTTP client once. The discovery layer applies a
        // per-request timeout; this builder-level timeout is a safety net
        // against hangs in malformed clients (e.g. a worker that accepts
        // the connection then never writes a response).
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .expect("reqwest::Client should build with default config");

        info!(
            sync_mode = "zmq",
            event_port = config.event_port,
            block_size = config.block_size,
            event_channel_capacity = config.event_channel_capacity,
            "cache-aware ZMQ indexer policy initialized; awaiting worker subscriptions"
        );

        Self {
            config,
            trees,
            worker_models,
            subscribers,
            tokenizer_registry,
            http_client,
            consumer_task: parking_lot::Mutex::new(Some(consumer_task)),
            _eviction_task: eviction_task,
            cancel,
        }
    }

    /// Register a worker. Records its model_id and dp_size in
    /// `worker_models` and opens ZMQ subscriptions for each DP rank
    /// (defaults to 1 rank if the worker is not DP-aware). Idempotent:
    /// re-registering the same worker is safe.
    ///
    /// If the worker was previously registered under a *different* model
    /// id, its entries in the old tree are cleared first so a stale
    /// prefix from the old assignment cannot be returned for the new
    /// model.
    pub async fn add_worker(&self, worker: &dyn Worker) {
        let new_model_key = normalize_model_key(worker.model_id()).to_string();
        let url = worker.url().to_string();
        let worker_dp_size = worker.dp_size().unwrap_or(1).max(1) as u32;

        // Resolve the per-worker `EventConfig`. Discovery is best-effort:
        // a worker that doesn't expose `/server_info` (older SGLang), or
        // one that's unreachable at registration time, falls back to a
        // config built from the worker URL + the gateway's globally
        // configured `event_port`/`block_size`.
        let event_cfg = match fetch_event_config(&url, &self.http_client).await {
            Ok(Some(mut cfg)) => {
                if cfg.block_size as usize != self.config.block_size {
                    warn!(
                        worker_url = %url,
                        worker_block_size = cfg.block_size,
                        gateway_block_size = self.config.block_size,
                        "kv-events discovery: block_size mismatch between worker /server_info and gateway config; \
                         routing prefix matches will silently miss. Use the gateway block_size for now."
                    );
                    cfg.block_size = self.config.block_size as u32;
                }
                if cfg.dp_size != worker_dp_size {
                    debug!(
                        worker_url = %url,
                        worker_dp_size,
                        discovered_dp_size = cfg.dp_size,
                        "kv-events discovery: dp_size differs from worker.dp_size(); trusting discovery"
                    );
                }
                debug!(
                    worker_url = %url,
                    host = %cfg.host,
                    port_base = cfg.port_base,
                    dp_size = cfg.dp_size,
                    "kv-events discovery: using per-worker EventConfig"
                );
                cfg
            }
            Ok(None) => fallback_event_config(&url, worker_dp_size, &self.config),
            Err(e) => {
                warn!(
                    worker_url = %url,
                    error = %e,
                    "kv-events discovery: failed to resolve worker URL; falling back to static event_port"
                );
                fallback_event_config(&url, worker_dp_size, &self.config)
            }
        };

        let recorded_dp_size = event_cfg.dp_size;

        // If the worker was previously registered under a different
        // model, clear its entries from the old tree so future matches
        // for the old model don't return stale URLs. We use the
        // *historical* dp_size from `worker_models` so we walk every
        // rank we may have indexed.
        if let Some(prev_info) = self.worker_models.get(&url).map(|e| e.value().clone()) {
            if prev_info.model_key != new_model_key {
                if let Some(old_tree) = self
                    .trees
                    .get(&prev_info.model_key)
                    .map(|e| e.value().clone())
                {
                    for dp_rank in 0..prev_info.dp_size {
                        old_tree.clear_worker(&KvWorkerId {
                            url: url.clone(),
                            dp_rank,
                        });
                    }
                }
            }
        }

        // Make sure the per-model tree exists so events that arrive before
        // the first request do not fall into the gap.
        self.trees
            .entry(new_model_key.clone())
            .or_insert_with(|| Arc::new(HashTree::new()));

        self.worker_models.insert(
            url.clone(),
            WorkerInfo {
                model_key: new_model_key,
                dp_size: recorded_dp_size,
            },
        );

        // Log the resolved ZMQ endpoints at debug — operators can grep this
        // during onboarding to confirm the gateway is connecting to the
        // ports they expect. info! would be too noisy on a fleet with many
        // workers.
        let endpoints: Vec<String> = (0..event_cfg.dp_size)
            .map(|r| format!("tcp://{}:{}", event_cfg.host, event_cfg.port_base as u32 + r))
            .collect();
        debug!(
            worker_url = %url,
            dp_size = event_cfg.dp_size,
            endpoints = ?endpoints,
            "cache-aware-zmq adding worker subscriptions"
        );

        self.subscribers.add_worker(&url, &event_cfg).await;
    }

    /// Remove a worker. Cancels all of its ZMQ subscriptions, drops its
    /// `worker_url -> WorkerInfo` entry, and clears it from the per-model
    /// tree so future requests do not pick a worker that no longer holds
    /// the cache.
    ///
    /// We clear the worker from the tree using the *historical* dp_size
    /// recorded at `add_worker` time. The worker's currently-reported
    /// `dp_size` may be smaller (e.g. after a reconfig), which would
    /// leak the higher ranks if we trusted it.
    pub async fn remove_worker(&self, worker: &dyn Worker) {
        let url = worker.url().to_string();
        // Drop subscriptions first so no further events arrive for this
        // worker — otherwise a racing event could re-populate the tree
        // we are about to clear.
        self.subscribers.remove_worker(&url).await;

        let info = self.worker_models.remove(&url).map(|(_, v)| v);

        // Clear the worker (every DP rank we may have indexed) from its
        // model tree. We use the dp_size recorded at registration so a
        // worker that has since lowered its dp_size still gets every
        // rank cleared. If we never saw this URL (e.g. a duplicate
        // remove call), fall back to the worker's current values; this
        // is also the path for a pre-existing tree that was loaded out
        // of band.
        let (model_key, dp_size) = match info {
            Some(WorkerInfo { model_key, dp_size }) => (model_key, dp_size),
            None => (
                normalize_model_key(worker.model_id()).to_string(),
                worker.dp_size().unwrap_or(1).max(1) as u32,
            ),
        };

        if let Some(tree) = self.trees.get(&model_key).map(|e| e.value().clone()) {
            for dp_rank in 0..dp_size {
                tree.clear_worker(&KvWorkerId {
                    url: url.clone(),
                    dp_rank,
                });
            }
        }
    }

    /// Cancel the consumer task and shut down all subscribers. Idempotent
    /// and safe to call multiple times. Awaiting twice does not hang
    /// because the second call sees the consumer task already taken.
    pub async fn shutdown(&self) {
        // Tell the consumer task to exit; it will also exit naturally
        // when the registry is dropped, but cancellation gets it out
        // promptly even if the channel still has pending items.
        self.cancel.cancel();
        self.subscribers.shutdown().await;
        let join = {
            let mut guard = self.consumer_task.lock();
            guard.take()
        };
        if let Some(handle) = join {
            if let Err(e) = handle.await {
                warn!(error = %e, "consumer task did not join cleanly");
            }
        }
    }

    /// Apply one decoded `WorkerEvent` to the right tree. Pulled out so
    /// tests can inject events without spinning up a real socket.
    /// Production traffic flows through [`spawn_consumer_task`], which
    /// calls [`apply_event_to_trees`] directly to avoid an `Arc` clone
    /// per event; this method is kept as a method for the test hook.
    #[cfg(test)]
    fn apply_event(&self, ev: WorkerEvent) {
        apply_event_to_trees(&self.trees, &self.worker_models, ev);
    }

    /// Look up the tree for a model. Uses the same `normalize_model_key`
    /// that the policy uses on insert, so empty model_ids work too.
    fn tree_for_model(&self, model_id: &str) -> Option<Arc<HashTree>> {
        let key = normalize_model_key(model_id);
        self.trees.get(key).map(|e| e.value().clone())
    }

    /// Tokenize `text` for `model_id`. Returns `None` if no tokenizer
    /// has been loaded yet (so caller should fall back to min-load) or
    /// if encoding errors. Errors are logged at debug level — they are
    /// not fatal to routing.
    fn tokenize(&self, model_id: &str, text: &str) -> Option<Vec<u32>> {
        let tokenizer = self.tokenizer_registry.get(model_id)?;
        match tokenizer.encode(text, /* add_special_tokens */ false) {
            Ok(enc) => Some(enc.token_ids().to_vec()),
            Err(e) => {
                debug!(model = %model_id, error = %e, "tokenizer encode failed; falling back");
                None
            }
        }
    }

    /// Min-load routing helper used both for "no tree" and "imbalanced"
    /// fallbacks. Mirrors the existing [`super::CacheAwarePolicy`] shape.
    fn pick_min_load(workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> Option<usize> {
        healthy_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()
    }

    /// Pick the best healthy worker among the matched URLs. If multiple
    /// gateway workers map to the same URL (rare but possible — e.g. one
    /// `Worker` per DP rank with DP-aware routing), the lowest-loaded one
    /// wins.
    fn pick_matched_worker(
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        matched: &MatchResult,
    ) -> Option<usize> {
        if matched.workers.is_empty() {
            return None;
        }
        let urls: std::collections::HashSet<&str> =
            matched.workers.iter().map(|w| w.url.as_str()).collect();
        healthy_indices
            .iter()
            .copied()
            .filter(|&idx| urls.contains(workers[idx].url()))
            .min_by_key(|&idx| workers[idx].load())
    }

    /// Test-only hook: route an event into the trees synchronously so
    /// tests don't need to spin up a real ZMQ pair. Use the same
    /// dispatch logic the consumer task uses.
    #[cfg(test)]
    pub(crate) fn apply_worker_event_for_test(&self, ev: WorkerEvent) {
        self.apply_event(ev);
    }

    /// Test inspection: total node count across all per-model trees.
    /// Exposed for integration tests in `tests/routing/` that need to
    /// observe the indexer's state to assert that ordered events landed
    /// in the tree. Production code should not depend on this.
    #[doc(hidden)]
    pub fn total_node_count_for_test(&self) -> usize {
        self.trees.iter().map(|e| e.value().node_count()).sum()
    }

    /// Test inspection: node count for one model's tree, or `None` if
    /// the tree has not been created yet (no events applied, no worker
    /// added).
    #[doc(hidden)]
    pub fn node_count_for_model_test(&self, model_id: &str) -> Option<usize> {
        self.tree_for_model(model_id).map(|t| t.node_count())
    }
}

impl std::fmt::Debug for CacheAwareZmqPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAwareZmqPolicy")
            .field("config", &self.config)
            .field("tree_count", &self.trees.len())
            .field("worker_count", &self.worker_models.len())
            .finish()
    }
}

impl Drop for CacheAwareZmqPolicy {
    fn drop(&mut self) {
        // Best-effort cancel so neither a stray consumer task nor any
        // per-worker subscriber task is left running when the policy is
        // dropped without an explicit `shutdown()` call.
        //
        // * `self.cancel.cancel()` exits the consumer task.
        // * `self.subscribers.cancel_all()` signals every per-worker
        //   subscriber's own `CancellationToken` so an idle subscriber
        //   blocked on `sub.recv()` (no traffic) will exit on its next
        //   yield point instead of leaking along with its ZMQ socket.
        //
        // Note: this does not await — the JoinHandles are dropped, which
        // detaches the tasks. If the caller wants a clean shutdown (await
        // join), they must call `shutdown()` explicitly.
        self.cancel.cancel();
        self.subscribers.cancel_all();
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwareZmqPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id()).to_string();

        // Compute (min, max) load in a single pass — same as the existing
        // mesh policy.
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(mn, mx), w| {
            let l = w.load();
            (mn.min(l), mx.max(l))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            let idx = Self::pick_min_load(workers, &healthy_indices)?;
            workers[idx].increment_processed();
            return Some(idx);
        }

        // Try the cache-aware path. Any failure (no request text, no
        // tokenizer, no tree, low match-rate, no healthy match) falls
        // through to min-load — never an error to the caller.
        let request_text = match info.request_text {
            Some(t) if !t.is_empty() => t,
            _ => {
                let idx = Self::pick_min_load(workers, &healthy_indices)?;
                workers[idx].increment_processed();
                return Some(idx);
            }
        };

        let tokens = match self.tokenize(&model_id, request_text) {
            Some(t) if !t.is_empty() => t,
            _ => {
                debug!(model = %model_id, "no tokenizer / empty tokens; min-load fallback");
                let idx = Self::pick_min_load(workers, &healthy_indices)?;
                workers[idx].increment_processed();
                return Some(idx);
            }
        };

        let block_size = self.config.block_size.max(1);
        let block_hashes = compute_block_hashes(&tokens, block_size);
        if block_hashes.is_empty() {
            let idx = Self::pick_min_load(workers, &healthy_indices)?;
            workers[idx].increment_processed();
            return Some(idx);
        }

        let tree = match self.tree_for_model(&model_id) {
            Some(t) => t,
            None => {
                let idx = Self::pick_min_load(workers, &healthy_indices)?;
                workers[idx].increment_processed();
                return Some(idx);
            }
        };

        let matched = tree.match_prefix(None, &block_hashes);
        let match_rate = matched.matched_blocks as f32 / block_hashes.len() as f32;

        let chosen = if match_rate > self.config.cache_threshold {
            Self::pick_matched_worker(workers, &healthy_indices, &matched)
        } else {
            None
        }
        .or_else(|| Self::pick_min_load(workers, &healthy_indices))?;

        workers[chosen].increment_processed();
        Some(chosen)
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        if !success {
            debug!(worker_url, "request failed for cache-aware-zmq policy");
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware_zmq"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    async fn on_add_worker(&self, worker: &dyn Worker) {
        // Forward to the inherent async `add_worker` so the worker-add hook
        // in the registration workflow flows through here. The inherent
        // method is kept public for tests and direct callers.
        self.add_worker(worker).await;
    }

    async fn on_remove_worker(&self, worker: &dyn Worker) {
        self.remove_worker(worker).await;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build an `EventConfig` from the gateway's global `CacheAwareConfig`
/// when per-worker discovery fails (or the worker doesn't expose the
/// `/server_info` `kv_events` block). The host is parsed out of the
/// worker URL; the port and other tuning fields come from the global
/// config. Falls back to `127.0.0.1` if the URL has no recognizable host
/// — the subscriber will then fail to connect and the policy stays in
/// the load-balanced default routing, which is the existing pre-discovery
/// behavior.
fn fallback_event_config(
    worker_url: &str,
    worker_dp_size: u32,
    config: &CacheAwareConfig,
) -> EventConfig {
    let host = url::Url::parse(worker_url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_else(|| "127.0.0.1".to_string());
    EventConfig {
        host,
        port_base: config.event_port,
        topic: String::new(),
        block_size: config.block_size as u32,
        dp_size: worker_dp_size,
    }
}

/// Spawn the background consumer that drains `rx` and dispatches every
/// `WorkerEvent` to the right per-model tree. Exits when:
///   * the cancellation token fires, OR
///   * the channel is closed (all senders dropped — i.e. the registry
///     was dropped without an explicit shutdown).
fn spawn_consumer_task(
    mut rx: mpsc::Receiver<WorkerEvent>,
    trees: Arc<DashMap<String, Arc<HashTree>>>,
    worker_models: Arc<DashMap<String, WorkerInfo>>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    debug!("cache-aware-zmq consumer cancelled");
                    return;
                }
                maybe = rx.recv() => match maybe {
                    Some(ev) => apply_event_to_trees(&trees, &worker_models, ev),
                    None => {
                        debug!("cache-aware-zmq consumer: channel closed; exiting");
                        return;
                    }
                },
            }
        }
    })
}

/// Single-event dispatch. Called both by the consumer task and (in
/// tests) by [`CacheAwareZmqPolicy::apply_worker_event_for_test`]. Pure
/// function over the trees + worker_models maps so test injection and
/// production share the exact same logic.
fn apply_event_to_trees(
    trees: &DashMap<String, Arc<HashTree>>,
    worker_models: &DashMap<String, WorkerInfo>,
    ev: WorkerEvent,
) {
    let WorkerEvent {
        worker, batch, seq, ..
    } = ev;

    let model_id = match worker_models.get(&worker.url) {
        Some(info) => info.value().model_key.clone(),
        None => {
            // We received an event from a worker we don't know — could
            // happen during a race with `remove_worker`, or if a worker
            // was added to the subscriber registry through a path that
            // bypassed `add_worker`. Drop it; debug-log so it shows up in
            // troubleshooting but doesn't spam at info.
            debug!(
                worker_url = %worker.url,
                dp_rank = worker.dp_rank,
                seq,
                "dropping event from unknown worker"
            );
            return;
        }
    };

    let tree = trees
        .entry(model_id)
        .or_insert_with(|| Arc::new(HashTree::new()))
        .value()
        .clone();

    for event in batch.events {
        match event {
            KvCacheEvent::BlockStored(stored) => {
                tree.insert(&worker, stored.parent_block_hash, &stored.block_hashes);
            }
            KvCacheEvent::BlockRemoved(removed) => {
                tree.remove(&worker, &removed.block_hashes);
            }
            KvCacheEvent::AllBlocksCleared => {
                tree.clear_worker(&worker);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::core::{BasicWorkerBuilder, Worker, WorkerType};
    use crate::policies::kv_events::wire::{BlockRemoved, BlockStored, KvCacheEvent, KvEventBatch};
    use llm_tokenizer::{
        traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer},
        TokenizerRegistry,
    };

    /// Fixed-tokens-by-model fake tokenizer for tests. `encode(text)`
    /// returns the configured token sequence regardless of input — that's
    /// fine because the tests below only care that "same text → same
    /// hashes" and that hashes are reproducible. Per-model tokenizers
    /// are registered under the model's name in [`TokenizerRegistry`].
    struct FixedTokenizer {
        tokens: Vec<TokenIdType>,
    }

    impl FixedTokenizer {
        fn new(tokens: Vec<TokenIdType>) -> Self {
            Self { tokens }
        }
    }

    impl Encoder for FixedTokenizer {
        fn encode(&self, _input: &str, _special: bool) -> anyhow::Result<Encoding> {
            Ok(Encoding::Plain(self.tokens.clone()))
        }
        fn encode_batch(&self, inputs: &[&str], special: bool) -> anyhow::Result<Vec<Encoding>> {
            inputs.iter().map(|i| self.encode(i, special)).collect()
        }
    }

    impl Decoder for FixedTokenizer {
        fn decode(&self, _ids: &[TokenIdType], _skip: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl Tokenizer for FixedTokenizer {
        fn vocab_size(&self) -> usize {
            65535
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            static EMPTY: std::sync::OnceLock<SpecialTokens> = std::sync::OnceLock::new();
            EMPTY.get_or_init(SpecialTokens::default)
        }
        fn token_to_id(&self, _t: &str) -> Option<TokenIdType> {
            None
        }
        fn id_to_token(&self, _id: TokenIdType) -> Option<String> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Build a registry with a single model_id -> fixed tokens mapping.
    /// Uses `load()` (the production API) with an in-memory loader.
    /// `register()` is `#[cfg(test)]` inside `llm-tokenizer` and not
    /// reachable from this crate.
    async fn registry_with(model_id: &str, tokens: Vec<TokenIdType>) -> Arc<TokenizerRegistry> {
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();
        let tokens_clone = tokens.clone();
        registry
            .load(&id, model_id, "fake://test", move || async move {
                Ok(Arc::new(FixedTokenizer::new(tokens_clone)) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("load tokenizer");
        Arc::new(registry)
    }

    /// Empty registry (no tokenizers loaded) — exercises the "no
    /// tokenizer; fall back to load" path.
    fn empty_registry() -> Arc<TokenizerRegistry> {
        Arc::new(TokenizerRegistry::new())
    }

    fn build_worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .label("model_id", model_id)
                .build(),
        )
    }

    fn make_stored_event(
        worker_url: &str,
        block_hashes: Vec<i64>,
        token_ids: Vec<u32>,
    ) -> WorkerEvent {
        WorkerEvent {
            worker: KvWorkerId {
                url: worker_url.to_string(),
                dp_rank: 0,
            },
            seq: 1,
            batch: KvEventBatch {
                ts: 0.0,
                events: vec![KvCacheEvent::BlockStored(BlockStored {
                    block_hashes,
                    parent_block_hash: None,
                    token_ids,
                    block_size: 4,
                    lora_id: None,
                    medium: Some("GPU".to_string()),
                })],
                attn_dp_rank: Some(0),
            },
        }
    }

    fn make_removed_event(worker_url: &str, block_hashes: Vec<i64>) -> WorkerEvent {
        WorkerEvent {
            worker: KvWorkerId {
                url: worker_url.to_string(),
                dp_rank: 0,
            },
            seq: 2,
            batch: KvEventBatch {
                ts: 0.0,
                events: vec![KvCacheEvent::BlockRemoved(BlockRemoved {
                    block_hashes,
                    medium: Some("GPU".to_string()),
                })],
                attn_dp_rank: Some(0),
            },
        }
    }

    /// Hash chain that compute_block_hashes produces for the registry
    /// fixture (`tokens=[1,2,3,4]`, block_size=4 → single-block chain).
    /// Reuses the cross-language golden in `kv_events::hash` tests.
    const FIXTURE_TOKENS: [u32; 4] = [1, 2, 3, 4];
    fn fixture_hashes() -> Vec<i64> {
        compute_block_hashes(&FIXTURE_TOKENS, 4)
    }

    fn cfg_for_test() -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0, // disable eviction thread
            max_tree_size: 10000,
            block_size: 4,
            event_port: 5557,
            event_channel_capacity: 16,
        }
    }

    #[tokio::test]
    async fn cache_hit_routes_to_holding_worker() {
        let model = "modelA";
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        let w0 = build_worker("http://w0:30000", model);
        let w1 = build_worker("http://w1:30000", model);
        policy.add_worker(w0.as_ref()).await;
        policy.add_worker(w1.as_ref()).await;

        // Inject an event saying w0 holds the request's hash chain.
        let hashes = fixture_hashes();
        policy.apply_worker_event_for_test(make_stored_event(
            w0.url(),
            hashes.clone(),
            FIXTURE_TOKENS.to_vec(),
        ));

        let workers = vec![w0.clone(), w1.clone()];
        let chosen = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("ignored, FixedTokenizer is constant"),
                    ..Default::default()
                },
            )
            .await
            .expect("should select");
        assert_eq!(workers[chosen].url(), w0.url());

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn no_tokenizer_falls_back_to_min_load() {
        let model = "modelA";
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), empty_registry());

        let w0 = build_worker("http://w0:30000", model);
        let w1 = build_worker("http://w1:30000", model);
        policy.add_worker(w0.as_ref()).await;
        policy.add_worker(w1.as_ref()).await;

        // Bump w0's load so min-load fallback should pick w1.
        w0.increment_load();
        w0.increment_load();

        let workers = vec![w0.clone(), w1.clone()];
        let chosen = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hi"),
                    ..Default::default()
                },
            )
            .await
            .expect("should select");
        assert_eq!(workers[chosen].url(), w1.url());

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn low_match_rate_falls_back_to_min_load() {
        let model = "modelA";
        // Tokens whose hash chain we will NOT insert into the tree.
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let mut cfg = cfg_for_test();
        cfg.cache_threshold = 0.99; // require near-total match
        let policy = CacheAwareZmqPolicy::new(cfg, registry);

        let w0 = build_worker("http://w0:30000", model);
        let w1 = build_worker("http://w1:30000", model);
        policy.add_worker(w0.as_ref()).await;
        policy.add_worker(w1.as_ref()).await;

        // Insert only an UNRELATED chain on w0 — match rate will be 0.
        policy.apply_worker_event_for_test(make_stored_event(
            w0.url(),
            vec![999_i64, 998_i64],
            vec![100, 101, 102, 103, 104, 105, 106, 107],
        ));

        // Bump w0's load so min-load picks w1, distinguishing it from a
        // happy cache hit.
        w0.increment_load();
        w0.increment_load();

        let workers = vec![w0.clone(), w1.clone()];
        let chosen = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("anything"),
                    ..Default::default()
                },
            )
            .await
            .expect("should select");
        assert_eq!(workers[chosen].url(), w1.url());

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn imbalanced_load_skips_cache_check() {
        let model = "modelA";
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0, // would normally always trigger cache routing
                balance_abs_threshold: 5,
                balance_rel_threshold: 2.0,
                eviction_interval_secs: 0,
                max_tree_size: 10000,
                block_size: 4,
                event_port: 5557,
                event_channel_capacity: 16,
            },
            registry,
        );

        let w0 = build_worker("http://w0:30000", model);
        let w1 = build_worker("http://w1:30000", model);
        policy.add_worker(w0.as_ref()).await;
        policy.add_worker(w1.as_ref()).await;

        // w0 holds the prefix...
        let hashes = fixture_hashes();
        policy.apply_worker_event_for_test(make_stored_event(
            w0.url(),
            hashes,
            FIXTURE_TOKENS.to_vec(),
        ));
        // ...but its load is much higher.
        for _ in 0..20 {
            w0.increment_load();
        }

        let workers = vec![w0.clone(), w1.clone()];
        for _ in 0..3 {
            let chosen = policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        request_text: Some("x"),
                        ..Default::default()
                    },
                )
                .await
                .expect("should select");
            assert_eq!(workers[chosen].url(), w1.url(), "imbalance must dominate");
        }

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn remove_worker_clears_tree_entry() {
        let model = "modelA";
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        let w0 = build_worker("http://w0:30000", model);
        let w1 = build_worker("http://w1:30000", model);
        policy.add_worker(w0.as_ref()).await;
        policy.add_worker(w1.as_ref()).await;

        let hashes = fixture_hashes();
        policy.apply_worker_event_for_test(make_stored_event(
            w0.url(),
            hashes.clone(),
            FIXTURE_TOKENS.to_vec(),
        ));
        // Sanity: tree has w0 holding the chain.
        let tree = policy.tree_for_model(model).expect("tree");
        assert!(!tree.match_prefix(None, &hashes).workers.is_empty());

        // Remove w0; the tree should no longer attribute the prefix to it.
        policy.remove_worker(w0.as_ref()).await;
        let m = tree.match_prefix(None, &hashes);
        assert!(
            m.workers.is_empty() || !m.workers.iter().any(|kw| kw.url == w0.url()),
            "after remove, w0 must not appear in match result; got {:?}",
            m.workers
        );

        // Subsequent select_worker should pick w1 (only healthy worker).
        let workers = vec![w1.clone()];
        let chosen = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("x"),
                    ..Default::default()
                },
            )
            .await
            .expect("should select");
        assert_eq!(workers[chosen].url(), w1.url());

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn block_removed_event_drops_worker_from_node() {
        let model = "modelA";
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        let w0 = build_worker("http://w0:30000", model);
        policy.add_worker(w0.as_ref()).await;

        let hashes = fixture_hashes();
        policy.apply_worker_event_for_test(make_stored_event(
            w0.url(),
            hashes.clone(),
            FIXTURE_TOKENS.to_vec(),
        ));

        let tree = policy.tree_for_model(model).expect("tree");
        assert_eq!(
            tree.match_prefix(None, &hashes).matched_blocks,
            hashes.len()
        );

        // BlockRemoved drops w0 from the chain; subsequent match must
        // not list w0.
        policy.apply_worker_event_for_test(make_removed_event(w0.url(), hashes.clone()));
        let m = tree.match_prefix(None, &hashes);
        assert!(
            !m.workers.iter().any(|kw| kw.url == w0.url()),
            "after BlockRemoved, w0 must be gone from match: {:?}",
            m.workers
        );

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn multi_model_isolation() {
        // Two workers, two distinct models. Events for modelA must not
        // affect modelB's tree, and vice versa.
        let model_a = "modelA";
        let model_b = "modelB";

        // Register both tokenizers in one registry via the public
        // async `load()` API (the test-only `register()` is not exported
        // from `llm-tokenizer` as a public symbol).
        let registry = TokenizerRegistry::new();
        let id_a = TokenizerRegistry::generate_id();
        let id_b = TokenizerRegistry::generate_id();
        registry
            .load(&id_a, model_a, "fake://a", || async {
                Ok(Arc::new(FixedTokenizer::new(FIXTURE_TOKENS.to_vec())) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("load a");
        registry
            .load(&id_b, model_b, "fake://b", || async {
                Ok(Arc::new(FixedTokenizer::new(FIXTURE_TOKENS.to_vec())) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("load b");
        let registry = Arc::new(registry);

        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        let wa = build_worker("http://wa:30000", model_a);
        let wb = build_worker("http://wb:30000", model_b);
        policy.add_worker(wa.as_ref()).await;
        policy.add_worker(wb.as_ref()).await;

        let hashes = fixture_hashes();
        // Inject event saying wa (model_a) holds the chain.
        policy.apply_worker_event_for_test(make_stored_event(
            wa.url(),
            hashes.clone(),
            FIXTURE_TOKENS.to_vec(),
        ));

        // Tree for model_a has wa.
        let tree_a = policy.tree_for_model(model_a).expect("tree a");
        let m_a = tree_a.match_prefix(None, &hashes);
        assert!(
            m_a.workers.iter().any(|kw| kw.url == wa.url()),
            "wa must hold prefix in tree_a"
        );

        // Tree for model_b should not see wa's blocks.
        let tree_b = policy.tree_for_model(model_b).expect("tree b");
        let m_b = tree_b.match_prefix(None, &hashes);
        assert!(
            m_b.workers.is_empty(),
            "tree_b must not see wa's blocks; got {:?}",
            m_b.workers
        );

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_is_idempotent() {
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), empty_registry());
        // First shutdown — must not hang.
        let r1 = tokio::time::timeout(Duration::from_secs(2), policy.shutdown()).await;
        assert!(r1.is_ok(), "first shutdown timed out");
        // Second shutdown — must not hang either.
        let r2 = tokio::time::timeout(Duration::from_secs(2), policy.shutdown()).await;
        assert!(r2.is_ok(), "second shutdown timed out");
    }

    /// Tokenizer present but `encode()` returns Err: must fall back to
    /// min-load (no panic, no error to caller). Mirrors the
    /// "no tokenizer" test, but exercises the encoder-error branch of
    /// [`CacheAwareZmqPolicy::tokenize`].
    #[tokio::test]
    async fn tokenizer_encode_error_falls_back_to_min_load() {
        struct FailingTokenizer;
        impl Encoder for FailingTokenizer {
            fn encode(&self, _input: &str, _special: bool) -> anyhow::Result<Encoding> {
                Err(anyhow::anyhow!("encode failed"))
            }
            fn encode_batch(
                &self,
                _inputs: &[&str],
                _special: bool,
            ) -> anyhow::Result<Vec<Encoding>> {
                Err(anyhow::anyhow!("encode failed"))
            }
        }
        impl Decoder for FailingTokenizer {
            fn decode(&self, _ids: &[TokenIdType], _skip: bool) -> anyhow::Result<String> {
                Ok(String::new())
            }
        }
        impl Tokenizer for FailingTokenizer {
            fn vocab_size(&self) -> usize {
                65535
            }
            fn get_special_tokens(&self) -> &SpecialTokens {
                static EMPTY: std::sync::OnceLock<SpecialTokens> = std::sync::OnceLock::new();
                EMPTY.get_or_init(SpecialTokens::default)
            }
            fn token_to_id(&self, _t: &str) -> Option<TokenIdType> {
                None
            }
            fn id_to_token(&self, _id: TokenIdType) -> Option<String> {
                None
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let model = "modelA";
        let registry = TokenizerRegistry::new();
        let id = TokenizerRegistry::generate_id();
        registry
            .load(&id, model, "fake://failing", || async {
                Ok(Arc::new(FailingTokenizer) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("load failing tokenizer");
        let registry = Arc::new(registry);

        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        let w0 = build_worker("http://w0:30000", model);
        let w1 = build_worker("http://w1:30000", model);
        policy.add_worker(w0.as_ref()).await;
        policy.add_worker(w1.as_ref()).await;

        // Bump w0 so min-load fallback distinguishably picks w1.
        w0.increment_load();
        w0.increment_load();

        let workers = vec![w0.clone(), w1.clone()];
        let chosen = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("anything"),
                    ..Default::default()
                },
            )
            .await
            .expect("should select");
        assert_eq!(workers[chosen].url(), w1.url());

        policy.shutdown().await;
    }

    /// Worker re-registered under a different model: stale entries in
    /// the old tree must not survive the reassignment. Without the
    /// `add_worker` walk-and-clear, a `match_prefix` against model A
    /// could still return the worker URL even though it now serves
    /// model B.
    #[tokio::test]
    async fn add_worker_model_reassignment_clears_old_tree() {
        let model_a = "modelA";
        let model_b = "modelB";

        let registry = TokenizerRegistry::new();
        let id_a = TokenizerRegistry::generate_id();
        let id_b = TokenizerRegistry::generate_id();
        registry
            .load(&id_a, model_a, "fake://a", || async {
                Ok(Arc::new(FixedTokenizer::new(FIXTURE_TOKENS.to_vec())) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("load a");
        registry
            .load(&id_b, model_b, "fake://b", || async {
                Ok(Arc::new(FixedTokenizer::new(FIXTURE_TOKENS.to_vec())) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("load b");
        let registry = Arc::new(registry);

        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        // First registration: URL u maps to model A.
        let w_a = build_worker("http://u:30000", model_a);
        policy.add_worker(w_a.as_ref()).await;

        // Inject events on tree A for this URL.
        let hashes = fixture_hashes();
        policy.apply_worker_event_for_test(make_stored_event(
            w_a.url(),
            hashes.clone(),
            FIXTURE_TOKENS.to_vec(),
        ));
        let tree_a = policy.tree_for_model(model_a).expect("tree a");
        assert!(
            tree_a
                .match_prefix(None, &hashes)
                .workers
                .iter()
                .any(|kw| kw.url == w_a.url()),
            "sanity: tree_a should hold the prefix before reassignment"
        );

        // Now re-register the same URL under model B. The old tree
        // entries on model A must be cleared.
        let w_b = build_worker("http://u:30000", model_b);
        policy.add_worker(w_b.as_ref()).await;

        let m_a = tree_a.match_prefix(None, &hashes);
        assert!(
            !m_a.workers.iter().any(|kw| kw.url == w_a.url()),
            "after reassignment, tree_a must not return the URL; got {:?}",
            m_a.workers
        );

        policy.shutdown().await;
    }

    /// `remove_worker` must use the dp_size recorded at `add_worker`
    /// time, not the worker's current `dp_size`. Otherwise events
    /// already indexed under higher-rank `(url, dp_rank)` pairs leak
    /// in the tree.
    #[tokio::test]
    async fn remove_worker_uses_historical_dp_size() {
        use crate::core::worker::DPAwareWorker;

        let model = "modelA";
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        // Build two DP-aware "views" over the same URL: one with
        // dp_size=4 (registration time) and a second with dp_size=2
        // (later, simulating a config change). Both share the same
        // BasicWorker URL so `worker_url` is identical.
        let url = "http://w0:30000";
        let basic_4 = BasicWorkerBuilder::new(url)
            .label("model_id", model)
            .build();
        let dp_4 = DPAwareWorker::with_base_worker(basic_4, url.to_string(), 0, 4);

        policy.add_worker(&dp_4 as &dyn Worker).await;

        // Inject events on every rank we registered.
        for rank in 0..4 {
            let ev = WorkerEvent {
                worker: KvWorkerId {
                    url: url.to_string(),
                    dp_rank: rank,
                },
                seq: rank as i64 + 1,
                batch: KvEventBatch {
                    ts: 0.0,
                    events: vec![KvCacheEvent::BlockStored(BlockStored {
                        block_hashes: vec![100 + rank as i64],
                        parent_block_hash: None,
                        token_ids: vec![1, 2, 3, 4],
                        block_size: 4,
                        lora_id: None,
                        medium: Some("GPU".to_string()),
                    })],
                    attn_dp_rank: Some(rank),
                },
            };
            policy.apply_worker_event_for_test(ev);
        }

        // Sanity: tree contains entries for rank 2 and rank 3.
        let tree = policy.tree_for_model(model).expect("tree");
        let m2_before = tree.match_prefix(None, &[102]);
        assert!(
            m2_before
                .workers
                .iter()
                .any(|kw| kw.url == url && kw.dp_rank == 2),
            "rank 2 should be indexed before remove_worker"
        );
        let m3_before = tree.match_prefix(None, &[103]);
        assert!(
            m3_before
                .workers
                .iter()
                .any(|kw| kw.url == url && kw.dp_rank == 3),
            "rank 3 should be indexed before remove_worker"
        );

        // Now remove with a worker that *currently* reports dp_size=2.
        // The fix: we must still clear ranks 2..=3 because the registry
        // recorded dp_size=4 at add_worker time.
        let basic_2 = BasicWorkerBuilder::new(url)
            .label("model_id", model)
            .build();
        let dp_2 = DPAwareWorker::with_base_worker(basic_2, url.to_string(), 0, 2);
        policy.remove_worker(&dp_2 as &dyn Worker).await;

        let m2_after = tree.match_prefix(None, &[102]);
        assert!(
            !m2_after
                .workers
                .iter()
                .any(|kw| kw.url == url && kw.dp_rank == 2),
            "rank 2 must be cleared after remove_worker; got {:?}",
            m2_after.workers
        );
        let m3_after = tree.match_prefix(None, &[103]);
        assert!(
            !m3_after
                .workers
                .iter()
                .any(|kw| kw.url == url && kw.dp_rank == 3),
            "rank 3 must be cleared after remove_worker; got {:?}",
            m3_after.workers
        );

        policy.shutdown().await;
    }

    #[tokio::test]
    async fn unknown_worker_event_is_dropped() {
        // Direct exercise of the apply path — an event from a worker we
        // never registered must be ignored without panic.
        let model = "modelA";
        let registry = registry_with(model, FIXTURE_TOKENS.to_vec()).await;
        let policy = CacheAwareZmqPolicy::new(cfg_for_test(), registry);

        // No add_worker call — worker_models is empty.
        let hashes = fixture_hashes();
        policy.apply_worker_event_for_test(make_stored_event(
            "http://stranger:1",
            hashes.clone(),
            FIXTURE_TOKENS.to_vec(),
        ));
        // No tree was populated.
        assert!(policy
            .tree_for_model(model)
            .is_none_or(|t| t.node_count() == 0));

        policy.shutdown().await;
    }
}
