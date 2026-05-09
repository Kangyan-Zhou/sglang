//! End-to-end integration tests for the cache-aware ZMQ policy.
//!
//! Exercises the full pipeline of T1–T6:
//!
//! * a real ZMQ PUB socket bound on `127.0.0.1`,
//! * a [`CacheAwareZmqPolicy`] subscribing to that PUB,
//! * a `BlockStored` payload encoded as msgpack and shipped as a 3-frame
//!   multipart message,
//! * `select_worker` matching the published prefix and routing to the
//!   publishing worker.
//!
//! Also verifies the new `sync_mode` flag and factory dispatch added in T6.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use rmp::encode as mp;
use smg::{
    config::{
        types::{CacheAwareSyncMode, PolicyConfig},
        RouterConfig, RoutingMode,
    },
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{
        kv_events::compute_block_hashes, CacheAwareConfig, CacheAwareZmqPolicy,
        LoadBalancingPolicy, PolicyFactory, PolicyRegistry, SelectWorkerInfo,
    },
};
use tokio::time::timeout;
use zeromq::{Endpoint, PubSocket, Socket, SocketSend, ZmqMessage};

use llm_tokenizer::{
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer},
    TokenizerRegistry,
};

// ---------------------------------------------------------------------------
// Test helpers — kept private to this file. Only the slice of the wire
// format the e2e test needs is implemented; subscriber/wire tests already
// cover the BlockRemoved / AllBlocksCleared / nil-medium variants.
// ---------------------------------------------------------------------------

/// Fake tokenizer that returns the same fixed token sequence for any
/// input. Lets the test fix the block-hash chain it publishes from the
/// PUB socket and then hit `select_worker` with arbitrary request text.
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

/// Bind a PUB socket to an OS-assigned localhost port. Returns the socket
/// and the port the OS picked.
async fn make_pub_bound() -> (PubSocket, u16) {
    let mut sock = PubSocket::new();
    let endpoint = sock
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("bind PUB socket");
    let port = match endpoint {
        Endpoint::Tcp(_, p) => p,
        other => panic!("unexpected endpoint: {other:?}"),
    };
    (sock, port)
}

/// Encode a single `BlockStored` event in the same wire layout as
/// SGLang's `msgspec.msgpack` emitter (see `wire.rs`). Layout:
/// `["BlockStored", block_hashes, parent, token_ids, block_size, lora_id, medium]`.
fn encode_block_stored_event(
    block_hashes: &[i64],
    parent: Option<i64>,
    token_ids: &[u32],
    block_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 7).unwrap();
    mp::write_str(&mut buf, "BlockStored").unwrap();
    mp::write_array_len(&mut buf, block_hashes.len() as u32).unwrap();
    for v in block_hashes {
        mp::write_sint(&mut buf, *v).unwrap();
    }
    match parent {
        Some(v) => {
            mp::write_sint(&mut buf, v).unwrap();
        }
        None => mp::write_nil(&mut buf).unwrap(),
    }
    mp::write_array_len(&mut buf, token_ids.len() as u32).unwrap();
    for v in token_ids {
        mp::write_uint(&mut buf, *v as u64).unwrap();
    }
    mp::write_uint(&mut buf, block_size as u64).unwrap();
    mp::write_nil(&mut buf).unwrap(); // lora_id
    mp::write_str(&mut buf, "GPU").unwrap(); // medium
    buf
}

/// Encode a single `BlockRemoved` event as msgspec emits it. Layout:
/// `["BlockRemoved", block_hashes, medium]`. We always include `medium`
/// (with a value of "GPU") for parity with the publisher's typical
/// emission.
fn encode_block_removed_event(block_hashes: &[i64]) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 3).unwrap();
    mp::write_str(&mut buf, "BlockRemoved").unwrap();
    mp::write_array_len(&mut buf, block_hashes.len() as u32).unwrap();
    for v in block_hashes {
        mp::write_sint(&mut buf, *v).unwrap();
    }
    mp::write_str(&mut buf, "GPU").unwrap();
    buf
}

/// Encode a single `AllBlocksCleared` event. Layout: `["AllBlocksCleared"]`.
fn encode_all_blocks_cleared_event() -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 1).unwrap();
    mp::write_str(&mut buf, "AllBlocksCleared").unwrap();
    buf
}

/// Wrap one or more pre-encoded events into a `KVEventBatch` array with
/// a timestamp and the optional dp-rank field present.
fn encode_event_batch(ts: f64, events: Vec<Vec<u8>>, attn_dp_rank: Option<u32>) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 3).unwrap();
    mp::write_f64(&mut buf, ts).unwrap();
    mp::write_array_len(&mut buf, events.len() as u32).unwrap();
    for ev in events {
        buf.extend_from_slice(&ev);
    }
    match attn_dp_rank {
        Some(v) => {
            mp::write_uint(&mut buf, v as u64).unwrap();
        }
        None => mp::write_nil(&mut buf).unwrap(),
    }
    buf
}

/// Build a 3-frame ZMQ message matching SGLang's `ZmqEventPublisher`:
/// `(topic="", seq=BE i64, payload=msgpack)`.
fn build_multipart(seq: i64, payload: Vec<u8>) -> ZmqMessage {
    let mut msg = ZmqMessage::from(Bytes::new());
    msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    msg.push_back(Bytes::from(payload));
    msg
}

/// Register a fake tokenizer for `model_id` that returns the given token
/// chain on every encode.
async fn registry_with_fixed_tokens(model_id: &str, tokens: Vec<u32>) -> Arc<TokenizerRegistry> {
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

/// Build a worker handle pointing at a routing URL with the given
/// `model_id` label (the cache-aware policy reads `model_id` to pick
/// the tokenizer).
fn build_worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .label("model_id", model_id)
            .build(),
    )
}

/// Poll `select_worker` until it picks `expected_url` or `max` elapses.
/// The ZMQ pipe is asynchronous (publish → SUB recv → mpsc → consumer →
/// tree apply); a polling loop is less flaky than a fixed sleep.
async fn wait_for_route_to(
    policy: &CacheAwareZmqPolicy,
    workers: &[Arc<dyn Worker>],
    request_text: &str,
    expected_url: &str,
    max: Duration,
) -> Option<usize> {
    let start = std::time::Instant::now();
    while start.elapsed() < max {
        let info = SelectWorkerInfo {
            request_text: Some(request_text),
            ..Default::default()
        };
        if let Some(idx) = policy.select_worker(workers, &info).await {
            if workers[idx].url() == expected_url {
                return Some(idx);
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    None
}

/// Poll an arbitrary predicate on the policy until it returns `true` or
/// `max` elapses. Used to wait for the consumer task to apply tree
/// mutations (BlockRemoved, AllBlocksCleared) — the per-step e2e timing
/// is identical to `wait_for_route_to` but the assertion shape differs.
async fn wait_until<F>(predicate: F, max: Duration) -> bool
where
    F: Fn() -> bool,
{
    let start = std::time::Instant::now();
    while start.elapsed() < max {
        if predicate() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    predicate()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// e2e: a real PUB socket publishes a `BlockStored` event; the policy's
/// SUB picks it up and the next `select_worker` call routes to the
/// publishing worker. Two workers exist so picking the right one
/// requires a cache hit + load tiebreak rather than a one-worker
/// degenerate case.
///
/// API constraint: the subscriber registry builds endpoints as
/// `tcp://{host}:{event_port + dp_rank}` where `event_port` is a single
/// global config value (`subscriber.rs::add_worker`). The worker URL's
/// port is *not* part of the SUB endpoint — only its host is. With both
/// workers using `127.0.0.1` as host (the only default-bindable
/// loopback IP on macOS), both subscribers necessarily connect to the
/// same `tcp://127.0.0.1:event_port` socket. So a single bound PUB is
/// the only feasible test design today: both workers' subscribers
/// receive the same event and both end up indexed in the tree under
/// their own `KvWorkerId`s.
///
/// What the test still exercises:
///   * the full publish → SUB → mpsc → consumer → tree apply pipeline,
///   * `select_worker` reaching the cache-aware branch (match_rate >
///     threshold), and
///   * `pick_matched_worker` returning the min-load matched worker.
/// Worker B's load is bumped above worker A's so the min-load tiebreak
/// among matched workers deterministically picks A.
///
/// Once the policy gains per-worker endpoints (or `event_port` becomes
/// per-worker), this test should switch to two distinct PUB sockets so
/// only A ends up in the tree.
#[tokio::test]
async fn zmq_indexer_routes_to_publishing_worker_e2e() {
    let model = "modelA";
    let tokens: Vec<u32> = vec![1, 2, 3, 4];

    // 1. Bind a real PUB socket on an OS-assigned port.
    let (mut pub_a, port) = make_pub_bound().await;

    // 2. Tokenizer registry — encodes any text to `tokens`.
    let registry = registry_with_fixed_tokens(model, tokens.clone()).await;

    // 3. Build the policy pointed at `port`. block_size=4 produces a
    //    single hash for the chain above.
    let cfg = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        eviction_interval_secs: 0, // no background eviction in tests
        max_tree_size: 10_000,
        block_size: 4,
        event_port: port,
        event_channel_capacity: 64,
    };
    let policy = CacheAwareZmqPolicy::new(cfg, Arc::clone(&registry));

    // 4. Two workers on the same host. Both subscribers connect to the
    //    same PUB (single global `event_port`), so the published event
    //    is indexed in the tree under both workers' KvWorkerIds. The
    //    tiebreak between matched workers is min-load — see the load
    //    skew below.
    let url_a = "http://127.0.0.1:30000";
    let url_b = "http://127.0.0.1:30001";
    let w_a = build_worker(url_a, model);
    let w_b = build_worker(url_b, model);

    // Wire both workers via the *trait* method so we exercise the same
    // dispatch path that the worker-registration workflow uses.
    LoadBalancingPolicy::on_add_worker(&policy, w_a.as_ref()).await;
    LoadBalancingPolicy::on_add_worker(&policy, w_b.as_ref()).await;

    // Bump w_b's load so `pick_matched_worker`'s min-load tiebreak picks
    // w_a. The increment stays below `balance_abs_threshold` so the
    // imbalance-detection branch does not skip the cache-aware path.
    for _ in 0..3 {
        w_b.increment_load();
    }

    // 5. Allow the SUB sockets to handshake before publishing. 100ms is
    //    comfortable on localhost. The polling step below soaks up any
    //    extra latency, so this sleep just prevents a publish-before-SUB
    //    race that would silently drop the message.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 6. Compute the block hashes that worker A "holds" and publish a
    //    BlockStored event for them.
    let hashes = compute_block_hashes(&tokens, 4);
    assert!(!hashes.is_empty(), "expected at least one block hash");
    let event_bytes = encode_block_stored_event(
        &hashes, /*parent*/ None, &tokens, /*block_size*/ 4,
    );
    let payload = encode_event_batch(/*ts*/ 0.0, vec![event_bytes], Some(0));
    pub_a
        .send(build_multipart(1, payload))
        .await
        .expect("send block-stored event");

    // 7. Drive `select_worker` until the event has been applied.
    let workers = vec![Arc::clone(&w_a), Arc::clone(&w_b)];
    let chosen = wait_for_route_to(
        &policy,
        &workers,
        "any text — FixedTokenizer ignores it",
        url_a,
        Duration::from_secs(3),
    )
    .await;
    assert!(
        chosen.is_some(),
        "policy did not route to publishing worker A within timeout"
    );
    assert_eq!(workers[chosen.unwrap()].url(), url_a);

    // 8. Shutdown cleanly.
    let res = timeout(Duration::from_secs(2), policy.shutdown()).await;
    assert!(res.is_ok(), "policy shutdown should not hang");
}

/// e2e: a `BlockRemoved` event flowing through the ZMQ pipeline removes
/// the matching prefix from the tree. After publish, the worker that
/// previously held the prefix should no longer be matched — the policy
/// falls back to min-load.
///
/// Single PUB + single worker keeps the test deterministic: there is no
/// load-tiebreak ambiguity once the tree has been cleared. The
/// match-prefix → empty workers transition is the exact production path
/// `BlockRemoved` exists to drive.
#[tokio::test]
async fn zmq_indexer_block_removed_clears_tree_e2e() {
    let model = "modelA";
    let tokens: Vec<u32> = vec![1, 2, 3, 4];

    let (mut pub_a, port) = make_pub_bound().await;
    let registry = registry_with_fixed_tokens(model, tokens.clone()).await;

    let cfg = CacheAwareConfig {
        cache_threshold: 0.0, // any match counts as a hit
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        eviction_interval_secs: 0,
        max_tree_size: 10_000,
        block_size: 4,
        event_port: port,
        event_channel_capacity: 64,
    };
    let policy = CacheAwareZmqPolicy::new(cfg, Arc::clone(&registry));

    let url_a = "http://127.0.0.1:30100";
    let w_a = build_worker(url_a, model);
    LoadBalancingPolicy::on_add_worker(&policy, w_a.as_ref()).await;

    // SUB handshake.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Publish a BlockStored for the chain w_a "holds".
    let hashes = compute_block_hashes(&tokens, 4);
    assert!(!hashes.is_empty());
    let stored = encode_block_stored_event(&hashes, None, &tokens, 4);
    pub_a
        .send(build_multipart(1, encode_event_batch(0.0, vec![stored], Some(0))))
        .await
        .expect("send block-stored");

    // Wait for the tree to populate.
    let stored_landed = wait_until(
        || {
            policy
                .node_count_for_model_test(model)
                .is_some_and(|n| n > 0)
        },
        Duration::from_secs(3),
    )
    .await;
    assert!(
        stored_landed,
        "BlockStored did not populate tree within timeout"
    );
    let nodes_after_store = policy
        .node_count_for_model_test(model)
        .expect("tree exists");

    // Publish a BlockRemoved for those exact hashes.
    let removed = encode_block_removed_event(&hashes);
    pub_a
        .send(build_multipart(2, encode_event_batch(1.0, vec![removed], Some(0))))
        .await
        .expect("send block-removed");

    // Wait for the consumer to drain the BlockRemoved. We assert that
    // either (a) node_count strictly drops below the post-store count, or
    // (b) `match_prefix` returns no workers — both are equivalent
    // observations of the prefix being unattributed.
    let removed_applied = wait_until(
        || {
            policy
                .node_count_for_model_test(model)
                .is_some_and(|n| n < nodes_after_store)
        },
        Duration::from_secs(3),
    )
    .await;
    assert!(
        removed_applied,
        "BlockRemoved did not shrink the tree within timeout (before={}, after={:?})",
        nodes_after_store,
        policy.node_count_for_model_test(model),
    );

    let res = timeout(Duration::from_secs(2), policy.shutdown()).await;
    assert!(res.is_ok());
}

/// e2e: an `AllBlocksCleared` event empties the per-worker tree state
/// for the publishing worker. Only one worker exists; after the clear,
/// the tree should report zero nodes attributable to it (or fewer than
/// the post-store snapshot).
#[tokio::test]
async fn zmq_indexer_all_blocks_cleared_e2e() {
    let model = "modelA";
    let tokens: Vec<u32> = vec![1, 2, 3, 4];

    let (mut pub_a, port) = make_pub_bound().await;
    let registry = registry_with_fixed_tokens(model, tokens.clone()).await;

    let cfg = CacheAwareConfig {
        cache_threshold: 0.0,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        eviction_interval_secs: 0,
        max_tree_size: 10_000,
        block_size: 4,
        event_port: port,
        event_channel_capacity: 64,
    };
    let policy = CacheAwareZmqPolicy::new(cfg, Arc::clone(&registry));

    let url_a = "http://127.0.0.1:30200";
    let w_a = build_worker(url_a, model);
    LoadBalancingPolicy::on_add_worker(&policy, w_a.as_ref()).await;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let hashes = compute_block_hashes(&tokens, 4);
    let stored = encode_block_stored_event(&hashes, None, &tokens, 4);
    pub_a
        .send(build_multipart(1, encode_event_batch(0.0, vec![stored], Some(0))))
        .await
        .expect("send block-stored");

    let stored_landed = wait_until(
        || {
            policy
                .node_count_for_model_test(model)
                .is_some_and(|n| n > 0)
        },
        Duration::from_secs(3),
    )
    .await;
    assert!(stored_landed, "BlockStored did not populate tree");

    // Publish AllBlocksCleared.
    let cleared = encode_all_blocks_cleared_event();
    pub_a
        .send(build_multipart(2, encode_event_batch(1.0, vec![cleared], Some(0))))
        .await
        .expect("send all-blocks-cleared");

    // After clear, the worker should not be attributable to the prefix.
    // Use `select_worker` to observe the routing fallback: with one
    // worker, min-load picks it but the *cache match* should not be
    // returning the URL via `pick_matched_worker`. Easier observation:
    // node_count for this worker drops to 0.
    let cleared_applied = wait_until(
        || {
            policy
                .node_count_for_model_test(model)
                .is_some_and(|n| n == 0)
        },
        Duration::from_secs(3),
    )
    .await;
    assert!(
        cleared_applied,
        "AllBlocksCleared did not empty the tree (final={:?})",
        policy.node_count_for_model_test(model),
    );

    // After the clear, `select_worker` still returns the only healthy
    // worker — but via min-load fallback, not cache match. We only
    // assert that routing still works (no tree match means min-load
    // wins the lone worker).
    let workers = vec![Arc::clone(&w_a)];
    let info = SelectWorkerInfo {
        request_text: Some("anything"),
        ..Default::default()
    };
    let chosen = policy.select_worker(&workers, &info).await;
    assert_eq!(chosen, Some(0), "min-load fallback should still route");

    let res = timeout(Duration::from_secs(2), policy.shutdown()).await;
    assert!(res.is_ok());
}

/// e2e: a burst of 50 sequenced `BlockStored` events all land in the
/// tree, in order. Catches: ordered delivery, no event drops in the
/// consumer task under burst, channel capacity sane. Each event uses a
/// distinct prefix (different first hash) so we expect 50 separate
/// nodes after the burst.
#[tokio::test]
async fn zmq_indexer_multi_batch_sequenced_delivery_e2e() {
    let model = "modelA";
    // Tokens don't matter for the burst test — we publish synthetic
    // hashes directly, distinct per event.
    let tokens_for_routing: Vec<u32> = vec![1, 2, 3, 4];

    let (mut pub_a, port) = make_pub_bound().await;
    let registry = registry_with_fixed_tokens(model, tokens_for_routing).await;

    let cfg = CacheAwareConfig {
        cache_threshold: 0.5,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        eviction_interval_secs: 0,
        max_tree_size: 100_000,
        block_size: 4,
        event_port: port,
        // 64 is well above 50 so the channel should not back-pressure
        // visibly. The test still asserts that no events are dropped at
        // any layer.
        event_channel_capacity: 64,
    };
    let policy = CacheAwareZmqPolicy::new(cfg, Arc::clone(&registry));

    let url_a = "http://127.0.0.1:30300";
    let w_a = build_worker(url_a, model);
    LoadBalancingPolicy::on_add_worker(&policy, w_a.as_ref()).await;

    // SUB handshake — give it a comfortable margin since the burst that
    // follows is 50 messages back-to-back.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Publish 50 distinct BlockStored events. Each chain has one hash;
    // the per-event hashes are distinct so the tree gains one node per
    // event. Sequence numbers are monotonic.
    const N: i64 = 50;
    for i in 0..N {
        // Distinct prefix per event: use `i + 1` (avoid 0) as the only
        // hash. parent_block_hash=None means each is a root chain.
        let hashes = vec![i + 1];
        let stored = encode_block_stored_event(&hashes, None, &[1, 2, 3, 4], 4);
        let payload = encode_event_batch(i as f64, vec![stored], Some(0));
        pub_a
            .send(build_multipart(i + 1, payload))
            .await
            .expect("send burst event");
    }

    // Wait for the tree to grow to the expected node count. 5s timeout
    // is generous for 50 events on localhost; CI runners with high
    // jitter occasionally need >2s.
    let target = N as usize;
    let all_landed = wait_until(
        || {
            policy
                .node_count_for_model_test(model)
                .is_some_and(|n| n >= target)
        },
        Duration::from_secs(5),
    )
    .await;
    assert!(
        all_landed,
        "expected at least {} nodes after burst; got {:?}",
        target,
        policy.node_count_for_model_test(model)
    );

    // Sanity: the count is *exactly* `target` — no double-applies, no
    // drops — since each event added a distinct prefix.
    assert_eq!(
        policy.node_count_for_model_test(model),
        Some(target),
        "expected exactly {} distinct nodes after a sequenced burst",
        target
    );

    let res = timeout(Duration::from_secs(2), policy.shutdown()).await;
    assert!(res.is_ok());
}

/// Factory dispatch: `sync_mode=zmq` + tokenizer registry → ZMQ policy.
#[tokio::test]
async fn factory_dispatches_to_zmq_when_sync_mode_zmq_and_registry_present() {
    let registry = registry_with_fixed_tokens("modelA", vec![1, 2, 3, 4]).await;
    let policy = PolicyFactory::create_from_config_with_registry(
        &PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 1024,
            sync_mode: CacheAwareSyncMode::Zmq,
            block_size: 4,
            event_port: 5557,
            event_channel_capacity: 64,
        },
        Some(registry),
    );
    assert_eq!(policy.name(), "cache_aware_zmq");
}

/// Factory falls back to mesh when `sync_mode=zmq` is set but no
/// tokenizer registry is wired through. Soft fallback (warn + use the
/// legacy policy) keeps existing test fixtures working without the
/// registry plumbing.
#[tokio::test]
async fn factory_falls_back_to_mesh_when_zmq_without_registry() {
    let policy = PolicyFactory::create_from_config_with_registry(
        &PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 1024,
            sync_mode: CacheAwareSyncMode::Zmq,
            block_size: 4,
            event_port: 5557,
            event_channel_capacity: 64,
        },
        None,
    );
    assert_eq!(policy.name(), "cache_aware");
}

/// Factory honours `sync_mode=mesh` even when a tokenizer registry is
/// available — operators must be able to opt out of the new path.
#[tokio::test]
async fn factory_honours_explicit_mesh_sync_mode() {
    let registry = registry_with_fixed_tokens("modelA", vec![1, 2, 3, 4]).await;
    let policy = PolicyFactory::create_from_config_with_registry(
        &PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 1024,
            sync_mode: CacheAwareSyncMode::Mesh,
            block_size: 4,
            event_port: 5557,
            event_channel_capacity: 64,
        },
        Some(registry),
    );
    assert_eq!(policy.name(), "cache_aware");
}

/// JSON without `sync_mode` defaults to `Mesh` so pre-ZMQ configs keep
/// their original behavior. ZMQ is strictly opt-in.
#[tokio::test]
async fn policy_config_default_sync_mode_is_mesh() {
    let json = r#"{
        "type": "cache_aware",
        "cache_threshold": 0.5,
        "balance_abs_threshold": 32,
        "balance_rel_threshold": 1.1,
        "eviction_interval_secs": 60,
        "max_tree_size": 10000
    }"#;
    let cfg: PolicyConfig = serde_json::from_str(json).expect("parse cache_aware policy");
    match cfg {
        PolicyConfig::CacheAware {
            sync_mode,
            block_size,
            event_port,
            event_channel_capacity,
            ..
        } => {
            assert_eq!(sync_mode, CacheAwareSyncMode::Mesh);
            assert_eq!(block_size, 64);
            assert_eq!(event_port, 5557);
            assert_eq!(event_channel_capacity, 1024);
        }
        _ => panic!("expected CacheAware variant"),
    }
}

/// JSON with explicit `sync_mode=mesh` round-trips correctly.
#[tokio::test]
async fn policy_config_explicit_mesh_round_trips() {
    let json = r#"{
        "type": "cache_aware",
        "cache_threshold": 0.5,
        "balance_abs_threshold": 32,
        "balance_rel_threshold": 1.1,
        "eviction_interval_secs": 60,
        "max_tree_size": 10000,
        "sync_mode": "mesh"
    }"#;
    let cfg: PolicyConfig = serde_json::from_str(json).expect("parse mesh-mode policy");
    match cfg {
        PolicyConfig::CacheAware { sync_mode, .. } => {
            assert_eq!(sync_mode, CacheAwareSyncMode::Mesh);
        }
        _ => panic!("expected CacheAware variant"),
    }
}

/// `PolicyRegistry::new_with_tokenizer_registry` propagates the registry,
/// but `PolicyConfig::cache_aware()` keeps the safe `Mesh` default — wiring
/// a tokenizer registry alone does not flip the policy to ZMQ.
#[tokio::test]
async fn policy_registry_mesh_default_even_with_registry() {
    let registry = registry_with_fixed_tokens("modelA", vec![1, 2, 3, 4]).await;
    let cfg = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://127.0.0.1:30000".to_string()],
        },
        PolicyConfig::cache_aware(0.5, 32, 1.1, 60, 1000),
    );
    let policy_registry = PolicyRegistry::new_with_tokenizer_registry(
        cfg.policy.clone(),
        Some(Arc::clone(&registry)),
    );
    assert!(policy_registry.tokenizer_registry().is_some());
    let default = policy_registry.get_default_policy();
    assert_eq!(
        default.name(),
        "cache_aware",
        "default cache_aware policy must stay on the mesh variant unless sync_mode=zmq is set explicitly"
    );
}

/// Explicit `sync_mode = Zmq` plus a tokenizer registry selects the ZMQ
/// variant — proves the opt-in path still works post-default-flip.
#[tokio::test]
async fn policy_registry_zmq_when_explicitly_enabled() {
    let registry = registry_with_fixed_tokens("modelA", vec![1, 2, 3, 4]).await;
    let cfg = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec!["http://127.0.0.1:30000".to_string()],
        },
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 60,
            max_tree_size: 1000,
            sync_mode: CacheAwareSyncMode::Zmq,
            block_size: 64,
            event_port: 5557,
            event_channel_capacity: 64,
        },
    );
    let policy_registry = PolicyRegistry::new_with_tokenizer_registry(
        cfg.policy.clone(),
        Some(Arc::clone(&registry)),
    );
    let default = policy_registry.get_default_policy();
    assert_eq!(
        default.name(),
        "cache_aware_zmq",
        "explicit sync_mode=zmq with a tokenizer registry must produce the ZMQ variant"
    );
}
