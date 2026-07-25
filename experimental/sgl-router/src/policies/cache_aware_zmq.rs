// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware-ZMQ selection policy.
//!
//! Combines the KV-event-fed [`HashTree`] with active-load scoring and
//! tokenizer-driven block-hash lookup to pick the worker most likely to
//! already hold the request's prefix in its KV cache.
//!
//! # Selection algorithm
//!
//! Given `workers` (already filtered to healthy + matching pool by the
//! caller) and a `SelectionContext` carrying the JSON request body and the
//! ingress-precomputed routing tokens:
//!
//! Every load comparison below uses [`WorkerLoads::load_of`]: where a fresh
//! [`super::engine_load::EngineLoadTable`] snapshot exists, the
//! engine-reported queue depth (`num_running + num_waiting`) PLUS this
//! worker's dispatches since that snapshot was taken (the engine hasn't had
//! a chance to report them yet); otherwise the router-side in-flight
//! counter `Worker::active_load()`. The "since that snapshot" bound matters:
//! the engine only refreshes its gauge every few seconds
//! ([`super::engine_load::EngineLoadTable`]'s freshness window), which at
//! sustained request rates is several selections' worth of staleness, and
//! using it unadjusted lets a burst of back-to-back decisions all read the
//! same "worker looks idle" number and all pile onto it before the gauge
//! catches up. Adding back only the
//! not-yet-reported dispatches (not the worker's full `active_load`, which
//! can include long-held slots from slow-draining streaming responses)
//! self-corrects within the same burst without overcorrecting away from
//! workers that are idle on the engine side but still draining a finished
//! stream to a slow client.
//!
//! 1. **Routing tokens.** Prefer the ingress-precomputed ids
//!    (`ctx.request_tokens()`); fall back to tokenizing the body here
//!    (chat-encoder-aware for chat traffic, raw `prompt`/`text` otherwise)
//!    for callers that didn't pre-tokenize. On any failure (no tokens, no
//!    tokenizer, encode error, empty), fall through to step 4 (min-load).
//! 2. **Hash + match.** Compute block hashes via
//!    [`super::kv_events::compute_block_hashes`], query the shared hash tree
//!    for the longest matching prefix. If `match_rate > cache_threshold`,
//!    take the lowest-load worker whose `url` appears in the match result as
//!    the cache candidate. Otherwise, fall through to step 4.
//! 3. **Rotation check.** Rotate the prefix onto the least-backlogged worker
//!    when that worker owes enough less prefill to pay for the cache being
//!    given up — see [`CacheAwareZmqPolicy::should_rotate`].
//! 4. **Min-load fallback.** Pick the lowest-load worker.
//!
//! WHY the rotation decision is per-candidate and in tokens: cache-aware
//! routing deliberately *concentrates* traffic onto the workers holding a
//! prefix, and rotation exists to relieve that by seeding fresh replicas. The
//! only quantities that bear on a request are its own cache candidate's backlog
//! and the lightest worker's — never the fleet-wide spread, which would let one
//! unrelated saturated worker divert every request to the globally coldest
//! worker, by definition the one least likely to hold anything. And both sides
//! are measured in *prefill tokens*, so the comparison needs no conversion
//! factor: what the cache saves and what the queue costs are the same unit.
//!
//! Rotation is meant to be self-limiting: it seeds a replica, the seeded worker
//! publishes the prefix, the holder set grows, and the next request's candidate
//! is drawn from a larger set. That only closes if the seeded worker keeps the
//! prefix long enough to publish it — under eviction pressure rotation can
//! re-seed indefinitely without the holder set ever growing, and every request
//! pays a cold prefill. The `holders` field on the rotation decision log is what
//! separates the two.
//!
//! The implementation never returns `None` for a non-empty `workers` slice;
//! a misconfigured tree or tokenizer degrades to a min-load pick, not a
//! routing failure.

use crate::config::CacheAwareConfig;

use crate::policies::active_load::{spawn_sweeper, JanitorHandle};
use crate::policies::engine_load::{EngineLoadTable, FreshLoad};
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::{request_tokens_for, Policy, SelectionContext};
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::Worker;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Selection policy that scores candidates by tree-overlap with the
/// request's prefix and falls back to load-based picking when the tree
/// doesn't have useful signal.
pub struct CacheAwareZmqPolicy {
    config: CacheAwareConfig,
    /// Per-process KV-event hash tree, fed by the indexer. Cheap to
    /// clone an `Arc`; we never write to the tree from here.
    tree: Arc<HashTree>,
    /// Tokenizer registry — selection reads `model_id` from the context
    /// and looks up the per-model tokenizer.
    tokenizers: Arc<TokenizerRegistry>,
    /// Worker-sourced block size, shared with the `KvEventIndex` that
    /// seeds it on worker registration. Read once per request; if
    /// `None` (no worker has reported a `page_size` yet) the policy
    /// degrades to min-load — the router cannot hash a prompt without
    /// a block size that matches what the worker publishes.
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Engine-reported per-worker load (running + waiting), shared with the
    /// `KvEventIndex` load subscriber. Read once per selection; a worker with
    /// a fresh snapshot uses it in place of the router-side in-flight counter
    /// (`Worker::active_load`), falling back to that counter when the snapshot
    /// is stale or absent (cold start / worker predates load publishing).
    engine_load: Arc<EngineLoadTable>,
    /// Optional metrics sink. Set via [`Self::with_metrics`] by the policy
    /// factory for the production policy; `None` in unit tests and
    /// non-cache-aware call sites. When set, each scored cache-aware selection
    /// records its block metrics — `sgl_router_overlap_blocks` (matched),
    /// `sgl_router_prompt_blocks_total` (total), and the per-request
    /// `sgl_router_expected_hit_rate`. Set once via [`Self::with_metrics`]
    /// (tests) or the `Policy::attach_metrics` hook (production, called by
    /// `PolicyRegistry::attach_metrics` after the registry is built).
    metrics: OnceLock<Arc<MetricsRegistry>>,
    /// Rolling per-model accumulator behind the periodic expected-hit-rate
    /// summary log — see [`HitRateStats`] for the outcome taxonomy. Drained and
    /// reset by `_summary_janitor` every [`SUMMARY_INTERVAL`].
    stats: Arc<HitRateStats>,
    /// Background task that logs + resets [`Self::stats`] on
    /// [`SUMMARY_INTERVAL`]. `None` when constructed outside a Tokio runtime
    /// (unit tests); dropping it cancels the task, so it lives exactly as long
    /// as the policy.
    _summary_janitor: Option<JanitorHandle>,
}

impl std::fmt::Debug for CacheAwareZmqPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAwareZmqPolicy")
            .field("config", &self.config)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

/// Both operands of the rotation decision plus the worker a rotation would seed,
/// all from ONE [`WorkerLoads`] pass, produced only by
/// [`CacheAwareZmqPolicy::backlogs`].
///
/// Named fields rather than a tuple because a swap of the two token counts fails
/// SILENTLY: the fold seeds with the candidate, so `floor_tokens <=
/// candidate_tokens` always holds and a transposed subtraction is 0 for every
/// request — rotation off fleet-wide, with nothing in the logs.
struct PrefillBacklogs {
    /// The worker a rotation seeds the prefix onto: the least-backlogged one,
    /// ties broken by the lower [`WorkerLoads::load_of`].
    floor: Arc<Worker>,
    /// That worker's queued prefill, in tokens.
    floor_tokens: usize,
    /// The cache candidate's queued prefill, from the same pass.
    candidate_tokens: usize,
}

/// Distinct worker URLs holding the matched prefix.
///
/// NOT `matched.workers.len()`: [`super::kv_events::tree::KvWorkerId`] is
/// `{url, dp_rank}`, so a dp-attention engine contributes one entry per rank and
/// the raw length overstates replica count by up to `dp_size`. The `holders` log
/// field is read as "how many replicas hold this prefix" — the module docs make
/// it the discriminator between rotation seeding successfully and rotation
/// re-seeding under eviction pressure — so an inflated count breaks the one
/// diagnostic it exists for.
fn distinct_holder_urls(matched: &super::kv_events::MatchResult) -> HashSet<&str> {
    matched.workers.iter().map(|kw| kw.url.as_str()).collect()
}

/// One worker's standing in the rotation-floor fold, during
/// [`CacheAwareZmqPolicy::backlogs`].
///
/// Named fields rather than a tuple because `tokens` and `load` are the same
/// Rust type in DIFFERENT units — prefill tokens vs in-flight request count —
/// and [`Self::key`] is the single place they may be ordered together. A
/// transposed key would silently promote the request count to primary, so the
/// floor would be chosen by load with tokens as the tiebreak: rotation still
/// fires, still logs, still names a real worker, but `floor_tokens` is not the
/// minimum.
struct FloorCandidate<'a> {
    worker: &'a Arc<Worker>,
    /// Queued prefill, in tokens. Primary — this is the quantity the rotation
    /// arithmetic consumes.
    tokens: usize,
    /// `WorkerLoads::load_of` — engine-reported depth plus this router's
    /// since-snapshot dispatches. Tiebreak ONLY; see [`Self::key`].
    load: usize,
}

impl<'a> FloorCandidate<'a> {
    /// Ordering key: fewer queued tokens wins; among workers tied on tokens, the
    /// lower `load_of` wins. The unit boundary lives here and nowhere
    /// else.
    fn key(&self) -> (usize, usize) {
        (self.tokens, self.load)
    }
}

/// Per-selection load lookup. Built once per `select` from a single
/// [`EngineLoadTable::fresh_worker_state`] pass: a worker with a fresh
/// engine-reported snapshot uses its queue depth (`num_running +
/// num_waiting`) plus its own dispatches acquired since that snapshot's
/// timestamp (see [`Self::load_of`]); otherwise it falls back to the
/// router-side in-flight counter (`Worker::active_load`). Holding the
/// snapshot keeps every per-worker `load_of` an O(1) map lookup.
struct WorkerLoads {
    /// url -> that worker's fused engine snapshot.
    fresh: HashMap<String, FreshLoad>,
}

impl WorkerLoads {
    /// Build the per-selection snapshot from one `fresh_worker_state` pass.
    /// The single construction chokepoint guarantees every comparison in a
    /// given `select` sees one consistent view of load.
    fn from_engine(table: &EngineLoadTable, now: Instant) -> Self {
        Self {
            fresh: table.fresh_worker_state(now),
        }
    }

    /// A worker's current load: the engine-reported queue depth as of the
    /// last fresh snapshot, plus this worker's own dispatches made *since*
    /// that snapshot's timestamp — i.e. exactly the requests the engine
    /// hasn't had a chance to report back on yet. This is deliberately not
    /// the worker's full `active_load()`: that counter also includes
    /// long-held slots from slow-draining streaming responses (see
    /// `crate::proxy::Proxy::forward_streaming_to`'s `stream_guards` doc)
    /// that the engine's own last report has likely already accounted for —
    /// adding the full counter on top would bias selection away from workers
    /// that are idle on the engine side but still slowly draining a finished
    /// stream to a client.
    ///
    /// This correction is per-router-process: it only sees dispatches THIS
    /// router pod made. It closes the single-pod stale-gauge herd, but does
    /// not coordinate with other router replicas — two pods can still both
    /// read the same stale engine number and independently pile onto the
    /// same worker within one gauge-refresh window. Closing that would need
    /// cross-replica state sharing, which this fix does not attempt.
    fn load_of(&self, w: &Worker) -> usize {
        match self.fresh.get(w.url.as_str()) {
            // `saturating_add`, not an assertable invariant: both operands
            // are bounded by real admission/concurrency limits (a worker's
            // in-flight count is capped well below `usize::MAX` by
            // `SlotRegistry::try_claim`'s admission cap), so overflow here
            // is unreachable from real traffic — reaching it would mean a
            // problem (memory exhaustion, a corrupt engine payload) that is
            // already symptomatic elsewhere, not something worth a panic on
            // this per-request hot path.
            Some(f) => f.depth.saturating_add(w.slots_acquired_since(f.at)),
            None => w.active_load(),
        }
    }

    /// A worker's queued prefill in tokens, as of this selection's snapshot.
    ///
    /// `None` when the engine does not report it — no fallback is offered on
    /// purpose. The router-side substitutes (`active_load`, queue depth) are
    /// request counts, and silently swapping a request count in where a token
    /// count is expected is exactly the unit confusion that makes a routing knob
    /// mean different things on different fleets. Callers decline to rotate
    /// instead.
    ///
    /// No since-snapshot correction is applied: the engine's own radix match is
    /// what makes this number meaningful, and this router cannot compute what a
    /// dispatch it just made will actually cost in uncached tokens on the far
    /// side. The staleness bound is the publish cadence.
    fn pending_prefill_tokens_of(&self, w: &Worker) -> Option<usize> {
        self.fresh.get(w.url.as_str())?.pending_prefill_tokens
    }

    /// Number of workers whose load came from the engine (vs the router-side
    /// fallback). Used only to annotate the per-selection load-inputs debug
    /// line.
    fn engine_worker_count(&self) -> usize {
        self.fresh.len()
    }

    /// How many of `workers` report a prefill-token backlog — i.e. how many are
    /// eligible as a rotation destination for THIS selection.
    ///
    /// Scoped to the candidate slice, not to the whole snapshot: one
    /// [`EngineLoadTable`] is shared by every registered worker of every role, so
    /// a fleet-wide count can read healthy while none of the workers actually
    /// being routed over report anything. In PD that is the normal case — a
    /// prefill pool mid-rollout against decode pods that already publish.
    ///
    /// Zero for a whole fleet whenever the engines don't publish a prefill backlog
    /// — the default `--schedule-policy fcfs` case, so the default case — and below
    /// the candidate count mid-rollout or behind a wedged publisher. Logged beside
    /// `candidates` on the per-selection line, which is the only outside signal
    /// that rotation is deciding from a partial view.
    fn backlog_reporting_count(&self, workers: &[Arc<Worker>]) -> usize {
        workers
            .iter()
            .filter(|w| self.pending_prefill_tokens_of(w).is_some())
            .count()
    }
}

/// Cadence of the expected-hit-rate summary log (see [`HitRateStats`]). One
/// aggregate INFO line per model per window keeps the router's own hit-rate view
/// visible without relying on the per-request decision lines, which sit on the
/// silenceable `cache_hit_rate` target and each cover one outcome.
const SUMMARY_INTERVAL: Duration = Duration::from_secs(30);

/// Rolling per-model accumulator behind the periodic expected-hit-rate
/// summary. Every cache-aware selection records exactly one outcome here:
/// [`Self::record_scored`] (matched + total prompt blocks were computed — see
/// [`ScoredOutcome`] for the three destinations) or [`Self::record_unscored`]
/// (routing couldn't score it: no tokens, no block size, or empty hashes). The
/// window mean is block/token-weighted (`Σ matched / Σ total`), comparable to
/// the engine's cached/prompt ratio.
#[derive(Default)]
struct HitRateStats {
    windows: Mutex<HashMap<String, HitRateWindow>>,
}

/// Where a scored selection actually went. Three states, not a bool: the two an
/// operator most needs to tell apart are "the cache had nothing for this request"
/// and "the cache had something and we gave it up".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoredOutcome {
    /// Pinned to a live worker holding the matched prefix.
    Routed,
    /// Abandoned the prefix for the least-backlogged worker — see
    /// [`CacheAwareZmqPolicy::should_rotate`].
    Rotated,
    /// No cache-matched worker was eligible at all.
    FellBack,
}

/// One model's counts for the current summary window. Drained (taken + reset)
/// each tick.
#[derive(Default, Clone, Copy)]
struct HitRateWindow {
    /// Scored selections that pinned to a cache-matched live worker.
    routed: u64,
    /// Scored selections that had a usable cache candidate and gave it up for the
    /// least-backlogged worker. Counted separately from `fell_back` because the
    /// two have opposite meanings for an operator: `fell_back` climbing says the
    /// cache isn't matching, `rotated` climbing says it is and the holders are
    /// too backlogged to use.
    rotated: u64,
    /// Scored selections with no eligible cache-matched worker: the predicted
    /// rate was at or below the ratio, or no cache-holding worker was a live
    /// candidate.
    fell_back: u64,
    /// Selections that could not be scored at all (no tokens, no block size, or
    /// empty hashes) — no predicted rate exists for them.
    unscored: u64,
    /// Matched (cache-hit) prompt blocks summed over the scored selections.
    /// Numerator of the block/token-weighted window mean.
    sum_matched_blocks: u64,
    /// Total prompt blocks summed over the scored selections. Denominator of
    /// the block/token-weighted window mean.
    sum_total_blocks: u64,
}

impl HitRateWindow {
    /// Selections that produced a predicted rate this window.
    ///
    /// Destructured exhaustively so adding a counter to [`HitRateWindow`] fails to
    /// compile here rather than silently under-reporting the denominator that
    /// [`Self::mean_rate`]'s contract depends on. (The match in
    /// [`HitRateStats::record_scored`] covers the other direction — a new
    /// [`ScoredOutcome`] variant.)
    fn scored(&self) -> u64 {
        let Self {
            routed,
            rotated,
            fell_back,
            unscored: _,
            sum_matched_blocks: _,
            sum_total_blocks: _,
        } = self;
        routed + rotated + fell_back
    }

    /// Mean predicted hit rate over the scored selections, or 0.0 when nothing
    /// was scored. BLOCK-weighted — `Σ matched_blocks / Σ total_blocks` over the
    /// window — not a per-request mean, so a short brand-new request (few blocks,
    /// low match) and a deep-conversation request (many blocks, high match)
    /// contribute in proportion to their size. This is the canonical WHY anchor
    /// for the block-weighted meter (the `prompt_blocks_total` metric + the
    /// `select` observe site reference it).
    ///
    /// Same weighting SHAPE as the engine's `sglang_cached_tokens /
    /// sglang_prompt_tokens`, so the two track closely — but only approximately,
    /// for three reasons: (a) each request's FINAL block is partial
    /// (`compute_block_hashes` uses `div_ceil`), so a whole tail block counts for
    /// `< block_size` tokens in the denominator. Matched blocks are always full
    /// leading-prefix blocks, so this skew is one-sided and bounded by
    /// `~block_size / prompt_len` — sub-1% for the multi-block prompts that
    /// dominate the sum. (b) The router counters accumulate ONLY on the scored
    /// path — unscored selections are in neither numerator nor denominator —
    /// while the engine ratio covers every request. (c) A
    /// [`ScoredOutcome::Rotated`] selection keeps its matched blocks in the
    /// numerator although it was routed AWAY from the holder, so this meter reads
    /// high against the engine by however much rotation is giving up; `rotated`
    /// quantifies that gap. So read it as "comparable in shape/weighting", not an
    /// exact match.
    ///
    /// An unscored-only window has no defined mean; the summary's `scored=0` field
    /// is what distinguishes it from a genuine all-miss window (where `scored > 0`,
    /// `sum_total_blocks > 0`, and `sum_matched_blocks == 0`).
    fn mean_rate(&self) -> f64 {
        if self.sum_total_blocks == 0 {
            0.0
        } else {
            self.sum_matched_blocks as f64 / self.sum_total_blocks as f64
        }
    }
}

impl HitRateStats {
    fn record_scored(
        &self,
        model: &str,
        matched_blocks: u64,
        total_blocks: u64,
        outcome: ScoredOutcome,
    ) {
        let mut windows = self.windows.lock();
        let w = windows.entry(model.to_owned()).or_default();
        match outcome {
            ScoredOutcome::Routed => w.routed += 1,
            ScoredOutcome::Rotated => w.rotated += 1,
            ScoredOutcome::FellBack => w.fell_back += 1,
        }
        w.sum_matched_blocks += matched_blocks;
        w.sum_total_blocks += total_blocks;
    }

    fn record_unscored(&self, model: &str) {
        let mut windows = self.windows.lock();
        windows.entry(model.to_owned()).or_default().unscored += 1;
    }

    /// Read a model's current window without draining it — lets tests assert
    /// the block sums a sequence of `record_scored` calls produced.
    #[cfg(test)]
    fn hit_rate_window_for(&self, model: &str) -> HitRateWindow {
        self.windows.lock().get(model).copied().unwrap_or_default()
    }

    /// Take and reset the window, returning per-model entries sorted by model
    /// id (stable log order across ticks).
    fn drain_sorted(&self) -> Vec<(String, HitRateWindow)> {
        let taken = std::mem::take(&mut *self.windows.lock());
        let mut entries: Vec<(String, HitRateWindow)> = taken.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    }
}

/// Log one summary line per model that saw traffic this window (draining +
/// resetting the accumulator as a side effect). Factored out of the sweeper
/// closure so the emit logic is unit-testable without a running ticker.
fn log_hit_rate_summary(stats: &HitRateStats) {
    for (model, w) in stats.drain_sorted() {
        tracing::info!(
            model = %model,
            window_secs = SUMMARY_INTERVAL.as_secs(),
            scored = w.scored(),
            routed = w.routed,
            rotated = w.rotated,
            fell_back = w.fell_back,
            unscored = w.unscored,
            mean_hit_rate = w.mean_rate(),
            "cache-aware expected-hit-rate summary",
        );
    }
}

impl CacheAwareZmqPolicy {
    pub fn new(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        block_size_oracle: Arc<BlockSizeOracle>,
        engine_load: Arc<EngineLoadTable>,
    ) -> Self {
        let stats = Arc::new(HitRateStats::default());
        // `spawn_sweeper` needs a runtime; the factory builds policies inside
        // `main`'s Tokio runtime. Guard so sync constructions (unit tests)
        // don't panic — they just don't get the periodic summary (the
        // per-request accumulation still runs, it's simply never drained).
        let _summary_janitor = if tokio::runtime::Handle::try_current().is_ok() {
            let stats = Arc::clone(&stats);
            Some(spawn_sweeper(
                move || {
                    // The closure emits its own per-model summary lines; return
                    // 0 so the janitor's generic "removed entries" line stays
                    // quiet.
                    log_hit_rate_summary(&stats);
                    0
                },
                SUMMARY_INTERVAL,
                "cache-hit-rate",
            ))
        } else {
            // Only reached by sync construction (unit tests); in production the
            // factory builds policies inside `main`'s runtime. Log it so a
            // future off-runtime construction that silently disables the
            // summary is greppable (mirrors `StickyPolicy`).
            tracing::debug!(
                "CacheAwareZmqPolicy constructed outside a Tokio runtime; hit-rate summary is disabled"
            );
            None
        };
        Self {
            config,
            tree,
            tokenizers,
            block_size_oracle,
            engine_load,
            metrics: OnceLock::new(),
            stats,
            _summary_janitor,
        }
    }

    /// Attach a metrics sink so each cache-aware selection records the
    /// prefix-overlap block count into `sgl_router_overlap_blocks`. Builder
    /// form used by tests; production wiring goes through the
    /// `Policy::attach_metrics` hook.
    pub fn with_metrics(self, metrics: Arc<MetricsRegistry>) -> Self {
        let _ = self.metrics.set(metrics);
        self
    }

    /// Lowest-load worker by the per-selection load lookup — ties broken by
    /// stable iteration order (the order the registry returned, i.e.
    /// dashmap-undefined). For production traffic the ties are rare; tests
    /// pin the load skew.
    fn pick_min_load(workers: &[Arc<Worker>], loads: &WorkerLoads) -> Option<Arc<Worker>> {
        workers
            .iter()
            .min_by_key(|w| loads.load_of(w))
            .map(Arc::clone)
    }

    /// Record an "unscored" selection (routing could not compute a
    /// `match_rate` — no tokens, no block size, or empty hashes) and return
    /// the min-load pick. Consolidates the accumulation so every no-score
    /// early return in [`Self::select`] records the same stat once, in one
    /// place, rather than repeating it at each fall-through site.
    fn unscored_min_load(
        &self,
        workers: &[Arc<Worker>],
        loads: &WorkerLoads,
        model: &str,
    ) -> Option<Arc<Worker>> {
        self.stats.record_unscored(model);
        Self::pick_min_load(workers, loads)
    }

    /// Read a model's current summary window without draining it — lets tests
    /// assert the accounting a `select` produced.
    #[cfg(test)]
    fn hit_rate_window(&self, model: &str) -> HitRateWindow {
        self.stats
            .windows
            .lock()
            .get(model)
            .copied()
            .unwrap_or_default()
    }

    /// The least-backlogged worker (the one a rotation would seed) together with
    /// its prefill backlog and `candidate`'s, all read in ONE pass.
    ///
    /// `None` only when the CANDIDATE's own backlog is unknown: it is absent from
    /// the fresh table (never published, or any of its ranks stale — see
    /// [`EngineLoadTable::fresh_worker_state`]), or present with a rank that does
    /// not report the field. There is no second operand to compare against, so
    /// there is no decision to make.
    ///
    /// Every OTHER worker that doesn't report is skipped rather than failing the
    /// whole decision. Deciding from a partial view is safe here and refusing to
    /// is not: an unmeasurable worker is simply never named the floor, so it can
    /// never be rotated onto, while the workers that do report still yield a
    /// locally valid trade (`candidate - floor > matched` holds whatever the
    /// unmeasured workers are doing). Failing closed instead would let one silent
    /// replica disable rotation for the entire pool, on every request.
    ///
    /// Both operands come from the same pass because they must be comparable.
    /// Sampling one before tokenization and the other after would compare
    /// different instants, and could name the candidate itself as the rotation
    /// destination. Seeding the fold with the candidate makes this total, and
    /// makes `floor == candidate` imply a zero difference, hence no rotation.
    fn backlogs<'a>(
        workers: &'a [Arc<Worker>],
        loads: &WorkerLoads,
        candidate: &'a Arc<Worker>,
    ) -> Option<PrefillBacklogs> {
        let candidate_tokens = loads.pending_prefill_tokens_of(candidate)?;
        // (tokens, load) compared lexicographically: fewer queued tokens wins, and
        // among workers TIED on tokens the lower `load_of` wins. The tiebreak
        // carries the load: a tie at zero is the common case (every fresh replica
        // reports 0), `pending_prefill_tokens` is frozen between engine publishes,
        // and `load_of` is what counts this router's dispatches since the
        // snapshot — so without it every rotation in a publish window stampedes
        // onto whichever tied worker sorts first. Request counts only order
        // workers already equal in tokens.
        let mut floor = FloorCandidate {
            worker: candidate,
            tokens: candidate_tokens,
            load: loads.load_of(candidate),
        };
        for w in workers {
            // Identity is the URL: the registry keys workers by it, and a
            // re-registered worker is a fresh `Arc` for the same URL. Skipping the
            // candidate keeps its backlog a single read — read twice, a backlog
            // that shrinks in between would show the candidate as its own floor
            // with a nonzero difference, i.e. a "rotation" onto the worker we were
            // already going to use.
            if w.url == candidate.url {
                continue;
            }
            let Some(tokens) = loads.pending_prefill_tokens_of(w) else {
                continue;
            };
            let c = FloorCandidate {
                worker: w,
                tokens,
                load: loads.load_of(w),
            };
            if c.key() < floor.key() {
                floor = c;
            }
        }
        Some(PrefillBacklogs {
            floor: Arc::clone(floor.worker),
            floor_tokens: floor.tokens,
            candidate_tokens,
        })
    }

    /// Whether to abandon a cached prefix and seed it onto the least-backlogged
    /// worker instead.
    ///
    /// ```text
    /// rotate  iff  candidate_backlog - floor_backlog > matched_tokens
    /// ```
    ///
    /// Routing to the candidate means waiting behind its backlog and then
    /// prefilling `total - matched` tokens; routing to the floor means waiting
    /// behind a smaller backlog and prefilling all `total`. The prompt's total
    /// length cancels, leaving exactly the line above: rotate when the backlog you
    /// skip exceeds the prefill you give up. Only two workers are comparable this
    /// way — [`HashTree::match_prefix`] returns just the deepest matched node's
    /// worker set, so every other worker scores as zero overlap whatever shallower
    /// prefix it may hold.
    ///
    /// Known bias on dp-attention fleets: both backlog operands are per-rank means
    /// (see [`FreshLoad::pending_prefill_tokens`]), but `matched_tokens` is NOT
    /// divided by the rank count even though the same argument applies to it — the
    /// router picks a URL, the engine's DP controller picks the rank prefix-blind,
    /// so the cached prefix sits on roughly one rank in `D` and the expected saving
    /// is `matched_tokens / D`. Leaving it undivided makes the rule ~`D`× harder to
    /// trip than the derivation above. Deliberate: that direction keeps the cache,
    /// and a request that lands on the holding rank realises the full saving.
    ///
    /// Known limitation: `pending_prefill_tokens` counts the waiting queue and
    /// the in-flight chunk, so a worker saturated with DECODE but holding an
    /// empty prefill queue reports a small backlog. Rotation is deliberately
    /// blind to that — it trades prefill against prefill — and the engine's own
    /// admission limits are what bound decode saturation.
    fn should_rotate(b: &PrefillBacklogs, matched_tokens: usize) -> bool {
        b.candidate_tokens.saturating_sub(b.floor_tokens) > matched_tokens
    }
}

impl Policy for CacheAwareZmqPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }

        // Per-selection load lookup: engine-reported queue depth where fresh,
        // else the router-side in-flight counter. One snapshot pass serves
        // every comparison below (min-load fallback, matched-set tiebreak,
        // rotation check).
        let loads = WorkerLoads::from_engine(&self.engine_load, Instant::now());

        tracing::debug!(
            model = %ctx.model(),
            // Fleet-wide (the load table spans every role); compare against
            // `candidates` below rather than against each other.
            engine_load_workers = loads.engine_worker_count(),
            engine_load_expected = self.engine_load.expected_count(),
            candidates = workers.len(),
            // Of THIS selection's candidates, how many can be a rotation
            // destination. Below `candidates` means rotation is deciding from a
            // partial view; zero means it cannot fire at all. Zero is EXPECTED on a
            // fleet running the default `--schedule-policy fcfs` — the engine only
            // publishes a prefill backlog under `lpm`/`dfs-weight` — and also
            // occurs mid-rollout or behind a wedged load publisher. Check the
            // engine's schedule policy before chasing publisher health.
            backlog_reporting_workers = loads.backlog_reporting_count(workers),
            "cache-aware-zmq: load inputs considered",
        );

        // 1. Routing tokens. Prefer the ids computed once at ingress; fall
        //    back to tokenizing the body here so the policy stays usable for
        //    callers that don't pre-tokenize (e.g. unit tests). In production
        //    the ingress always pre-tokenizes, so this is a single tokenize.
        let fallback_ids;
        let tokens: &[u32] = match ctx.request_tokens() {
            Some(t) if !t.is_empty() => t,
            _ => {
                let body = match ctx.request_body() {
                    Some(b) if !b.is_empty() => b,
                    _ => return self.unscored_min_load(workers, &loads, ctx.model().0.as_str()),
                };
                let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
                    return self.unscored_min_load(workers, &loads, ctx.model().0.as_str());
                };
                let Some(rt) = request_tokens_for(&self.tokenizers, ctx.model(), &value) else {
                    return self.unscored_min_load(workers, &loads, ctx.model().0.as_str());
                };
                fallback_ids = rt.ids;
                &fallback_ids
            }
        };

        // 2. Hash + match.
        // Source block_size from the worker — the router can only hash
        // prompts at the block size the workers publish at. If no worker
        // has registered yet (oracle empty), cache-aware routing has no
        // ground truth to score against; fall back to min-load.
        let Some(block_size) = self.block_size_oracle.get() else {
            tracing::debug!(
                model = %ctx.model(),
                "cache-aware-zmq: block size unknown (no worker page_size yet), falling back to min-load",
            );
            return self.unscored_min_load(workers, &loads, ctx.model().0.as_str());
        };
        // EAGLE-family workers hash KV blocks over token bigrams; the query
        // hashes must match the worker's stored hashes or the tree lookup
        // always misses (overlap stays 0). The oracle carries the worker-
        // reported flag.
        let is_bigram = self.block_size_oracle.is_bigram();
        let block_hashes = if is_bigram {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if block_hashes.is_empty() {
            return self.unscored_min_load(workers, &loads, ctx.model().0.as_str());
        }
        let matched = self.tree.match_prefix(None, &block_hashes);
        let match_rate = matched.matched_blocks as f32 / block_hashes.len() as f32;
        tracing::debug!(
            model = %ctx.model(),
            hashing = if is_bigram { "bigram" } else { "unigram" },
            n_blocks = block_hashes.len(),
            matched_blocks = matched.matched_blocks,
            match_rate,
            cache_threshold = self.config.cache_threshold,
            "cache-aware-zmq match_prefix",
        );
        // Record overlap + expected-hit-rate here, before the routing decision,
        // so the histograms capture the full distribution of the predicted rate
        // — including the selections that clear no useful prefix and fall to
        // min-load. This is the quantitative signal that cache-aware routing is
        // matching prefixes at all.
        if let Some(m) = self.metrics.get() {
            m.observe_overlap_blocks(ctx.model().0.as_str(), matched.matched_blocks as u64);
            m.observe_prompt_blocks(ctx.model().0.as_str(), block_hashes.len() as u64);
            m.observe_expected_hit_rate(ctx.model().0.as_str(), match_rate as f64);
            // overlap (Σ matched blocks) + prompt (Σ total blocks) are recorded
            // as an unconditional pair here so `rate(overlap_blocks_sum) /
            // rate(prompt_blocks_total)` is the block-weighted expected hit rate
            // (see `HitRateWindow::mean_rate` for the weighting + its caveats).
            // The `expected_hit_rate` histogram keeps the per-request distribution.
        }
        let holder_urls = distinct_holder_urls(&matched);

        // The actual destination decides the [`ScoredOutcome`], not the rate
        // alone: pin to a matched worker only when the rate clears the ratio AND
        // one of the matched workers is a live candidate. A rate above the ratio
        // whose matched workers have all left the healthy set still lands on
        // min-load — a real fall-through (the degraded-fleet case this summary
        // exists to surface), so it must count as one, not as a routed hit.
        //
        // Every holder in this set holds the SAME prefix, so choosing between them
        // trades no cache — it only asks which will serve soonest. Ordered by
        // queued prefill tokens (matching `Self::backlogs`) so the holder pick and
        // the rotation floor can't disagree: picking a holder with a 50k-token
        // queue over another holder of the same prefix with 100 queued, then
        // correctly declining to rotate, would park the request behind the larger
        // queue and record an ordinary `Routed` selection with nothing logged.
        //
        // The ordering must stay HOMOGENEOUS across the set. Mixing a reported
        // token count against a sentinel for an unreported one makes "runs an
        // engine new enough to publish the field" the primary key — an image
        // property, not a load property — so a saturated reporting holder would
        // beat an idle silent one holding the identical prefix. So: token-ordered
        // only when every holder reports, else `load_of` for all of them, which is
        // the pre-rotation behaviour and correct here because no cache is at stake.
        // The homogeneity test is `collect::<Option<Vec<_>>>()` itself — one silent
        // holder yields `None` for the whole set — and it is applied ABOVE the
        // comparator, so each arm carries exactly one key and the unmeasurable case
        // has no sentinel whose direction could be wrong.
        //
        // Operational consequence worth knowing: when holders report inconsistently
        // this degrades to `load_of`, and if a silent holder wins that ordering then
        // `backlogs` finds no backlog for it and rotation is skipped for the request
        // entirely. So adding one older-image node can stop rotation firing for a
        // model. Conservative (the cache is kept), but it reads as a regression.
        let best_matched: Option<Arc<Worker>> = if match_rate > self.config.cache_threshold {
            let holders: Vec<&Arc<Worker>> = workers
                .iter()
                .filter(|w| holder_urls.contains(w.url.as_str()))
                .collect();
            let measured: Option<Vec<FloorCandidate<'_>>> = holders
                .iter()
                .map(|w| {
                    loads
                        .pending_prefill_tokens_of(w)
                        .map(|tokens| FloorCandidate {
                            worker: w,
                            tokens,
                            load: loads.load_of(w),
                        })
                })
                .collect();
            match measured {
                Some(cands) => cands
                    .into_iter()
                    .min_by_key(FloorCandidate::key)
                    .map(|c| Arc::clone(c.worker)),
                None => holders
                    .into_iter()
                    .min_by_key(|w| loads.load_of(w))
                    .map(Arc::clone),
            }
        } else {
            None
        };
        let Some(chosen) = best_matched else {
            // Fell through to min-load: the predicted rate was at or below the
            // ratio, or no cache-holding worker is currently a live candidate.
            // Logged on the dedicated `cache_hit_rate` target so a cold-cache
            // flood can be silenced independently (`RUST_LOG=cache_hit_rate=warn`)
            // without losing the periodic summary, which stays on the default
            // target.
            let reason = if match_rate <= self.config.cache_threshold {
                "below_threshold"
            } else {
                "no_live_matched_worker"
            };
            self.stats.record_scored(
                ctx.model().0.as_str(),
                matched.matched_blocks as u64,
                block_hashes.len() as u64,
                ScoredOutcome::FellBack,
            );
            tracing::info!(
                target: "cache_hit_rate",
                model = %ctx.model(),
                hit_rate = match_rate,
                matched_blocks = matched.matched_blocks,
                n_blocks = block_hashes.len(),
                cache_threshold = self.config.cache_threshold,
                holders = holder_urls.len(),
                reason,
                "cache-aware-zmq: no cache-matched worker selected — routing to min-load",
            );
            return Self::pick_min_load(workers, &loads);
        };

        // 3. Rotation check. Skipped when the chosen worker reports no prefill
        //    backlog; without the measurement, cache locality wins.
        let matched_tokens = matched.matched_blocks.saturating_mul(block_size as usize);
        match Self::backlogs(workers, &loads, &chosen) {
            Some(b) if Self::should_rotate(&b, matched_tokens) => {
                self.stats.record_scored(
                    ctx.model().0.as_str(),
                    matched.matched_blocks as u64,
                    block_hashes.len() as u64,
                    ScoredOutcome::Rotated,
                );
                // Same `cache_hit_rate` target as the fall-through line: both are
                // per-request cache-routing decisions, so one filter reaches both
                // and neither can be silenced without the other.
                tracing::info!(
                    target: "cache_hit_rate",
                    model = %ctx.model(),
                    worker = %b.floor.url,
                    matched_worker = %chosen.url,
                    matched_tokens,
                    candidate_backlog_tokens = b.candidate_tokens,
                    floor_backlog_tokens = b.floor_tokens,
                    // Rotation is meant to GROW this; flat while rotations mount is
                    // the eviction-pressure failure in the module docs.
                    holders = holder_urls.len(),
                    "cache-aware-zmq: floor owes less prefill than the cache is worth — seeding prefix onto it",
                );
                return Some(b.floor);
            }
            Some(_) => {}
            None => {
                // The one remaining way rotation goes silent. Logged per request
                // at debug (not info: on a fleet that never reports the field this
                // is every request) so "rotation isn't firing" is diagnosable
                // without reading the code to discover the precondition.
                tracing::debug!(
                    target: "cache_hit_rate",
                    model = %ctx.model(),
                    worker = %chosen.url,
                    backlog_reporting_workers = loads.backlog_reporting_count(workers),
                    "cache-aware-zmq: rotation not evaluated — chosen worker reports no prefill backlog",
                );
            }
        }

        self.stats.record_scored(
            ctx.model().0.as_str(),
            matched.matched_blocks as u64,
            block_hashes.len() as u64,
            ScoredOutcome::Routed,
        );
        tracing::debug!(
            model = %ctx.model(),
            worker = %chosen.url,
            matched_blocks = matched.matched_blocks,
            matched_tokens,
            "cache-aware-zmq: selected worker by cache overlap",
        );
        Some(chosen)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CacheAwareConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::engine_load::LoadStat;
    use crate::policies::kv_events::tree::KvWorkerId;
    use crate::policies::kv_events::HashTree;
    use crate::tokenizer::adapter;
    use std::time::Duration;

    fn cfg_default() -> CacheAwareConfig {
        CacheAwareConfig::default()
    }

    /// Any match counts, so tests can drive the rotation check without also
    /// satisfying the match-rate ratio.
    fn cfg_zero_threshold() -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.0,
        }
    }

    /// Helper: build a `BlockSizeOracle` already primed to the test's
    /// canonical block size (4). Mirrors what `KvEventIndex::add_worker`
    /// would do when the first real worker registers.
    fn oracle_for_tests(block_size: u32) -> Arc<BlockSizeOracle> {
        let o = BlockSizeOracle::new();
        o.try_set(block_size)
            .expect("fresh oracle accepts first set");
        o
    }

    fn worker(url: &str, model_id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(model_id.into())],
            bootstrap_port: None,
        }))
    }

    /// Build a policy with a fresh (empty) engine-load table, so selection
    /// reads the router-side `active_load` counter — matching the
    /// pre-load-aware behaviour these tests assert.
    fn new_policy(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        oracle: Arc<BlockSizeOracle>,
    ) -> CacheAwareZmqPolicy {
        CacheAwareZmqPolicy::new(config, tree, tokenizers, oracle, EngineLoadTable::new())
    }

    /// Build a policy with an explicit engine-load table, for tests that
    /// exercise engine-reported load overriding the router-side counter.
    fn new_policy_with_load(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        oracle: Arc<BlockSizeOracle>,
        engine_load: Arc<EngineLoadTable>,
    ) -> CacheAwareZmqPolicy {
        CacheAwareZmqPolicy::new(config, tree, tokenizers, oracle, engine_load)
    }

    fn load_stat(running: u64, waiting: u64) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_tokens: 0,
            max_total_num_tokens: 0,
            pending_prefill_tokens: None,
        }
    }

    /// A `LoadStat` from an engine that reports its queued prefill, in tokens.
    fn load_stat_backlog(running: u64, prefill_tokens: u64) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: 0,
            num_tokens: 0,
            max_total_num_tokens: 0,
            pending_prefill_tokens: Some(prefill_tokens),
        }
    }

    fn tokenizer_registry_with_tiny() -> Arc<TokenizerRegistry> {
        let cfg = crate::config::Config {
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                tokenizer_shards: 1,
                tokenizer_backend: Default::default(),
                tokenizer_l1_cache_mb: 0,
                policy: crate::config::PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
            admission: crate::config::AdmissionConfig::default(),
            retry: crate::config::RetryConfig::default(),
        };
        Arc::new(TokenizerRegistry::load_from_config(&cfg).expect("load tiny tokenizer"))
    }

    /// Empty workers list returns None (parity with other policies).
    #[test]
    fn empty_workers_returns_none() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(b"{\"prompt\":\"hi\"}"));
        assert!(policy.select(&[], &ctx).is_none());
    }

    /// Empty tree: no overlap signal anywhere, fall through to min-load.
    #[test]
    fn empty_tree_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0's load so min-load picks w1 deterministically.
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello world"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Tree contains w0's prefix; cache-aware selection picks w0 even though w1
    /// has lower router-side load. Load skew alone never diverts a cached
    /// request — only a reported prefill-token backlog can (see
    /// `rotates_when_the_floor_owes_enough_less_prefill`), and no engine reports
    /// one here.
    #[test]
    fn non_empty_tree_highest_overlap_wins() {
        let tree = Arc::new(HashTree::new());
        // Insert w0's tokens into the tree. The tiny tokenizer's hash
        // chain for our input is whatever `compute_block_hashes` returns;
        // we mimic the policy's hashing path so the test stays
        // deterministic against tokenizer changes.
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world"; // longer → more blocks
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(
            !hashes.is_empty(),
            "tiny tokenizer must produce at least one full block",
        );
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0, // any match counts
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// The cache-aware path records the matched prefix-overlap block count
    /// into `sgl_router_overlap_blocks`. Regression: the metric was defined
    /// but never observed in production, so the histogram stayed empty and
    /// gave no signal that cache-aware routing was matching anything.
    #[test]
    fn records_overlap_blocks_metric() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));

        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let _ = policy.select(&workers, &ctx).expect("must pick");

        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "overlap_blocks histogram must be observed on a cache-aware selection; got:\n{rendered}"
        );
    }

    /// Production wiring path: the policy is stored as `Arc<dyn Policy>` in a
    /// `PolicyRegistry`, then `PolicyRegistry::attach_metrics` injects the
    /// registry — exactly what `AppContext::with_active_load` does at startup.
    /// Exercises trait dispatch (the default no-op vs the `CacheAwareZmqPolicy`
    /// override) and the registry fan-out, neither of which the `with_metrics`
    /// builder test covers.
    #[test]
    fn attach_metrics_via_registry_records_overlap() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            toks,
            oracle_for_tests(4),
        );
        let model = ModelId("tiny".into());
        let registry = crate::policies::PolicyRegistry::default();
        registry.insert(model.clone(), Arc::new(policy));

        // The production injection point — not the `with_metrics` builder.
        let metrics = MetricsRegistry::new();
        registry.attach_metrics(Arc::clone(&metrics));

        let chosen_policy = registry.get(&model).unwrap();
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let _ = chosen_policy.select(&workers, &ctx).expect("must pick");

        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "PolicyRegistry::attach_metrics must wire overlap recording through the trait; got:\n{rendered}"
        );
    }

    /// The overlap observation is recorded *before* the cache-threshold branch,
    /// so low-overlap selections that fall back to min-load are still counted.
    /// `cache_threshold: 1.0` forces the fallback (match_rate is always <= 1.0)
    /// even on a full prefix match; assert the histogram is still observed AND
    /// the pick came from min-load (w1), not the cache-overlap worker (w0).
    #[test]
    fn overlap_recorded_even_when_selection_falls_back() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 1.0, // match_rate <= 1.0 always -> always fall back
            },
            tree,
            toks,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));

        // Bump w0's load so min-load picks w1 — distinguishing a min-load
        // fallback from the cache-overlap pick (which would be w0). Two guards
        // mirror `empty_tree_falls_back_to_min_load`; router-side load never
        // diverts the cache-aware path on its own, so the pick still reaches it.
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w1:30000",
            "cache_threshold 1.0 must force a min-load fallback (w1), not the overlap worker (w0)"
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "overlap must be recorded even on the below-threshold fallback; got:\n{rendered}"
        );
        assert!(
            rendered.contains("sgl_router_expected_hit_rate_count{model_id=\"tiny\"}"),
            "expected-hit-rate must be recorded even on the below-threshold fallback; got:\n{rendered}"
        );
        assert!(
            rendered.contains("sgl_router_prompt_blocks_total{model_id=\"tiny\"}"),
            "prompt-blocks denominator must be recorded on the scored path even on fallback; got:\n{rendered}"
        );
    }

    /// End-to-end wiring guard for the block-weighted meter: with a PARTIAL prefix
    /// match (only the first N of M blocks cached), the numerator (matched) and
    /// denominator (total) must be DIFFERENT values, and each must land in the
    /// right series/field. Every other metrics-attached select test uses a full
    /// match (`matched == total`), which can't tell them apart — so a swapped arg
    /// at the `observe_prompt_blocks` / `record_scored` call sites would peg the
    /// ratio at 1.0 or invert `mean_rate()` with no other test failing.
    #[test]
    fn select_records_partial_match_numerator_and_denominator_distinctly() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        let m = hashes.len();
        assert!(m >= 2, "need a multi-block prompt to have a partial match");
        let n = m - 1; // cache only the leading N of M blocks -> matched N < total M
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &hashes[..n],
        );

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0, // route on the partial match (don't fall back)
            },
            tree,
            toks,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        policy.select(&workers, &ctx).expect("must pick");

        // record_scored wiring (arg ORDER): matched=N goes to sum_matched_blocks,
        // total=M to sum_total_blocks; a swap would make mean_rate() = M/N > 1.
        let w = policy.hit_rate_window("tiny");
        assert_eq!(w.sum_matched_blocks, n as u64, "matched blocks");
        assert_eq!(w.sum_total_blocks, m as u64, "total blocks");
        assert!((w.mean_rate() - n as f64 / m as f64).abs() < 1e-9);

        // observe_* wiring (which COUNT into which SERIES): prompt_blocks_total
        // must be the TOTAL (M), not the matched (N) — else the ratio pins at 1.0.
        let rendered = metrics.render();
        let val = |prefix: &str| -> f64 {
            rendered
                .lines()
                .find(|l| l.starts_with(prefix))
                .and_then(|l| l.rsplit(' ').next())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or_else(|| panic!("no parseable metric line `{prefix}` in:\n{rendered}"))
        };
        let prompt_total = val("sgl_router_prompt_blocks_total{model_id=\"tiny\"}");
        let overlap_sum = val("sgl_router_overlap_blocks_sum{model_id=\"tiny\"}");
        assert_eq!(prompt_total, m as f64, "denominator = total blocks");
        assert_eq!(overlap_sum, n as f64, "numerator = matched blocks");
        assert_ne!(
            overlap_sum, prompt_total,
            "a partial match must keep numerator and denominator distinct"
        );
    }

    #[test]
    fn hit_rate_stats_accounts_scored_unscored_and_resets_on_drain() {
        let stats = HitRateStats::default();
        // Model "m": two routed (above threshold), one below-threshold fall-
        // back, one rotated, one unscored. Recorded as (matched_blocks,
        // total_blocks). The rotated one carries a HIGH match — that is the shape
        // that distinguishes it from a fall-back, and it stays in the block sums
        // (see `HitRateWindow::mean_rate` caveat (c)).
        stats.record_scored("m", 9, 10, ScoredOutcome::Routed);
        stats.record_scored("m", 8, 10, ScoredOutcome::Routed);
        stats.record_scored("m", 0, 1, ScoredOutcome::FellBack);
        stats.record_scored("m", 10, 10, ScoredOutcome::Rotated);
        stats.record_unscored("m");
        // A second model, to prove the split is per-model.
        stats.record_scored("other", 4, 4, ScoredOutcome::Routed);

        let drained = stats.drain_sorted();
        assert_eq!(drained.len(), 2, "one entry per model that saw traffic");

        // Sorted by model id: "m" before "other".
        let (m, w) = &drained[0];
        assert_eq!(m, "m");
        assert_eq!(w.scored(), 4, "routed + rotated + fell_back");
        assert_eq!(w.routed, 2);
        assert_eq!(w.rotated, 1);
        assert_eq!(
            w.fell_back, 1,
            "a rotation must NOT be folded into fell_back: one means the cache had \
             nothing, the other means it had something we gave up",
        );
        assert_eq!(w.unscored, 1);
        // Window mean is BLOCK-weighted over the scored selections:
        // Σ matched / Σ total = (9 + 8 + 0 + 10) / (10 + 10 + 1 + 10) = 27/31.
        assert!(
            (w.mean_rate() - 27.0 / 31.0).abs() < 1e-6,
            "block-weighted mean over scored only; got {}",
            w.mean_rate(),
        );

        let (other, w2) = &drained[1];
        assert_eq!(other, "other");
        assert_eq!(w2.scored(), 1);
        assert_eq!(w2.routed, 1);
        assert_eq!(w2.rotated, 0);
        assert_eq!(w2.fell_back, 0);
        assert!(
            (w2.mean_rate() - 1.0).abs() < 1e-6,
            "fully matched -> 1.0; got {}",
            w2.mean_rate(),
        );

        // Draining resets the window: a second drain sees nothing.
        assert!(
            stats.drain_sorted().is_empty(),
            "window must reset after drain",
        );
    }

    #[test]
    fn hit_rate_window_mean_guards_zero_scored() {
        // An unscored-only window (block size never known, say) has scored == 0
        // and no block sums. The mean must be the 0.0 sentinel, never a NaN
        // from 0/0.
        let mut w = HitRateWindow {
            unscored: 3,
            ..Default::default()
        };
        assert_eq!(w.scored(), 0);
        assert_eq!(w.sum_total_blocks, 0);
        assert_eq!(w.mean_rate(), 0.0, "no NaN when nothing was scored");

        // With scored data the mean is Σ matched / Σ total.
        w.routed = 2;
        w.fell_back = 1;
        w.sum_matched_blocks = 6;
        w.sum_total_blocks = 10;
        assert_eq!(w.scored(), 3);
        assert!((w.mean_rate() - 0.6).abs() < 1e-6, "got {}", w.mean_rate());
    }

    #[test]
    fn hit_rate_window_mean_is_block_weighted_not_request_weighted() {
        // Size-skewed inputs where the two weightings disagree: a tiny brand-
        // new request (0 matched / 1 total, request-rate 0.0) and a large
        // deep-conversation request (8 matched / 10 total, request-rate 0.8).
        let stats = HitRateStats::default();
        stats.record_scored("m", 0, 1, ScoredOutcome::FellBack);
        stats.record_scored("m", 8, 10, ScoredOutcome::Routed);
        let w = stats.hit_rate_window_for("m");

        // Block/token-weighted: Σ matched / Σ total = 8 / 11 ≈ 0.727.
        let block_weighted = 8.0 / 11.0;
        assert!(
            (w.mean_rate() - block_weighted).abs() < 1e-6,
            "block-weighted mean; got {}",
            w.mean_rate(),
        );
        // Request-weighted would be (0.0 + 0.8) / 2 = 0.4 — the small request
        // drags it down as hard as the big one. The block-weighted mean must
        // NOT equal that (asserted against the exact value, not a hand-tuned
        // distance, so tweaking the inputs can't silently flip the guard).
        let request_weighted = 0.4;
        assert!(
            (w.mean_rate() - request_weighted).abs() > 1e-6,
            "block-weighted ({}) must differ from request-weighted ({})",
            w.mean_rate(),
            request_weighted,
        );
    }

    /// A genuine all-miss window — every scored selection matched 0 blocks but
    /// had a nonzero prompt — must read `mean_rate() == 0.0` AND stay
    /// distinguishable from an unscored-only window (`scored > 0`,
    /// `sum_total_blocks > 0`). The `mean_rate` doc leans on exactly this
    /// distinction; only the unscored-only side was pinned before.
    #[test]
    fn hit_rate_window_all_miss_but_scored_is_zero_yet_distinguishable() {
        let stats = HitRateStats::default();
        stats.record_scored("m", 0, 5, ScoredOutcome::FellBack);
        stats.record_scored("m", 0, 8, ScoredOutcome::Routed);
        let w = stats.hit_rate_window_for("m");
        assert_eq!(w.mean_rate(), 0.0, "all-miss window means 0.0");
        assert!(
            w.scored() > 0 && w.sum_total_blocks > 0,
            "all-miss must be distinguishable from unscored-only (scored={}, total={})",
            w.scored(),
            w.sum_total_blocks,
        );
    }

    #[test]
    fn select_records_routed_then_unscored_in_stats() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(cfg_default(), tree, toks, oracle_for_tests(4));
        let w0 = worker("http://w0:30000", "tiny");
        let workers = vec![Arc::clone(&w0)];
        let model = ModelId("tiny".into());

        // Prefix fully in the tree and its matched worker is live: match_rate
        // 1.0 > threshold -> routed.
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "should pin to the cache worker"
        );
        let w = policy.hit_rate_window("tiny");
        assert_eq!(w.routed, 1, "matched live worker above ratio -> routed");
        assert_eq!(w.fell_back, 0);
        assert_eq!(w.unscored, 0);

        // A body with no routable content can't be tokenized -> unscored.
        let junk = serde_json::to_vec(&serde_json::json!({ "unexpected": 1 })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&junk));
        let _ = policy
            .select(&workers, &ctx)
            .expect("must still pick min-load");
        let w = policy.hit_rate_window("tiny");
        assert_eq!(w.unscored, 1, "unroutable body -> unscored");
        assert_eq!(
            w.routed, 1,
            "routed count unchanged by the unscored request"
        );
    }

    #[test]
    fn select_counts_fell_back_when_matched_worker_not_live() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(cfg_default(), tree, toks, oracle_for_tests(4));
        // The cache-holding worker (w0) is NOT in the live candidate set; only
        // w1 is. match_rate clears the ratio but no matched worker is live, so
        // the request lands on min-load and must count as fell_back, not routed.
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "routes to the only live worker"
        );
        let w = policy.hit_rate_window("tiny");
        assert_eq!(
            w.fell_back, 1,
            "above ratio but no live matched worker -> fell_back"
        );
        assert_eq!(w.routed, 0, "must not be counted as a routed hit");
    }

    /// End-to-end bigram wiring (the fix that takes `overlap_blocks_sum` from
    /// 0 to non-zero for EAGLE models): an EAGLE worker publishes its blocks
    /// under BIGRAM hashes. Only a router whose oracle reports `is_bigram` —
    /// and thus hashes its query with the bigram hasher — matches them, so
    /// overlap is non-zero and it picks the cached worker. A unigram-hashing
    /// router against the SAME tree matches nothing (overlap recorded as 0).
    #[test]
    fn bigram_routing_matches_only_with_bigram_hashing() {
        fn overlap_sum(rendered: &str) -> f64 {
            rendered
                .lines()
                .find(|l| l.starts_with("sgl_router_overlap_blocks_sum{model_id=\"tiny\"}"))
                .and_then(|l| l.split_whitespace().last())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(-1.0)
        }

        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        // The EAGLE worker publishes BIGRAM block hashes.
        let bigram_hashes = compute_block_hashes_bigram(&ids, block_size as usize);
        assert!(!bigram_hashes.is_empty());
        assert_ne!(
            bigram_hashes,
            compute_block_hashes(&ids, block_size as usize),
            "bigram and unigram hashes must differ for this prefix"
        );
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();

        // Bigram-aware router (oracle.is_bigram == true): query hashes match
        // the bigram tree -> overlap > 0 and it picks the matched worker w0.
        {
            let tree = Arc::new(HashTree::new());
            tree.insert(
                &KvWorkerId::new("http://w0:30000".into(), 0),
                None,
                &bigram_hashes,
            );
            let oracle = BlockSizeOracle::new();
            oracle.try_set(block_size).unwrap();
            oracle.set_bigram(true);
            let metrics = MetricsRegistry::new();
            let policy = new_policy(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                },
                tree,
                Arc::clone(&registry),
                oracle,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            let chosen = policy.select(&workers, &ctx).expect("must pick");
            assert_eq!(
                chosen.url, "http://w0:30000",
                "bigram-aware router must match w0's bigram-hashed prefix"
            );
            assert!(
                overlap_sum(&metrics.render()) > 0.0,
                "overlap_blocks_sum must be > 0 once the router hashes with bigram"
            );
        }

        // Unigram router (default is_bigram == false) vs the SAME bigram tree:
        // query hashes never match -> overlap recorded as 0.
        {
            let tree = Arc::new(HashTree::new());
            tree.insert(
                &KvWorkerId::new("http://w0:30000".into(), 0),
                None,
                &bigram_hashes,
            );
            let oracle = BlockSizeOracle::new();
            oracle.try_set(block_size).unwrap();
            let metrics = MetricsRegistry::new();
            let policy = new_policy(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                },
                tree,
                Arc::clone(&registry),
                oracle,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            let _ = policy.select(&workers, &ctx).expect("must pick");
            assert_eq!(
                overlap_sum(&metrics.render()),
                0.0,
                "unigram hashing matches nothing in a bigram tree -> overlap_sum == 0"
            );
        }
    }

    /// A chat-completions request on a model with a chat template must route by
    /// the **chat-templated** tokens (BOS + role markers + content) — the tokens
    /// the engine actually cached — not by the raw joined content. Worker w0
    /// published its blocks under the templated tokens; only a router that
    /// renders the same template hashes a matching query. Hashing the raw
    /// content instead would match nothing, leaving live `overlap_blocks_sum`
    /// at 0 for chat traffic.
    #[test]
    fn chat_request_routes_by_templated_tokens() {
        let registry = tokenizer_registry_with_tiny();
        let template = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>",
            "bos_token": "<s>",
        });
        registry.attach_chat_template_for_test("tiny", &template);

        let messages = serde_json::json!([{"role":"user","content":"hello world hello world"}]);
        // Engine-side blocks are keyed on tokenize(render(messages)).
        let templated_tokens = registry
            .encode_chat(
                "tiny",
                &messages,
                None,
                crate::tokenizer::dsv4::RenderOpts::chat(),
            )
            .unwrap();
        let block_size = 4u32;
        let templated_hashes = compute_block_hashes(&templated_tokens, block_size as usize);
        assert!(
            !templated_hashes.is_empty(),
            "templated prompt must produce at least one block"
        );

        let tree = Arc::new(HashTree::new());
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &templated_hashes,
        );

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(block_size),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": messages,
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "chat request must route by chat-templated tokens to the worker holding that prefix"
        );
    }

    /// Templated and raw-content hashings must genuinely differ, confirming
    /// the chat-template path does real work (a no-op template would make this
    /// assertion fail, and raw-content hashes would miss the engine's
    /// templated blocks).
    #[test]
    fn chat_templated_hashes_differ_from_raw_content_hashes() {
        let registry = tokenizer_registry_with_tiny();
        let template = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>",
            "bos_token": "<s>",
        });
        registry.attach_chat_template_for_test("tiny", &template);
        let content = "hello world hello world";
        let messages = serde_json::json!([{"role":"user","content":content}]);

        let templated = registry
            .encode_chat(
                "tiny",
                &messages,
                None,
                crate::tokenizer::dsv4::RenderOpts::chat(),
            )
            .unwrap();
        let raw = adapter::encode(&registry.get("tiny").unwrap(), content).unwrap();
        assert_ne!(
            compute_block_hashes(&templated, 4),
            compute_block_hashes(&raw, 4),
            "templated and raw-content block hashes must differ"
        );
    }

    /// The DeepSeek-V4 built-in encoder is dispatched for chat requests when a
    /// model has it (no Jinja template). The query tokens come from the V4
    /// encoder, so a worker holding that encoded prefix is matched. (The V4
    /// markers aren't special tokens in the tiny fixture, but the dispatch +
    /// routing wiring is what's under test; byte-exact V4 token parity is pinned
    /// by `dsv4`'s string goldens and validated live.)
    #[test]
    fn chat_request_routes_via_dsv4_encoder() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_encoder_for_test("tiny", crate::tokenizer::ChatEncoder::DeepSeekV4);
        assert!(registry.has_chat_encoder("tiny"));

        let messages =
            serde_json::json!([{"role":"user","content":"hello world hello world hello world"}]);
        let encoded = registry
            .encode_chat(
                "tiny",
                &messages,
                None,
                crate::tokenizer::dsv4::RenderOpts::chat(),
            )
            .unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&encoded, block_size as usize);
        assert!(!hashes.is_empty());

        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(block_size),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "messages": messages })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "dsv4 chat request must route by the V4-encoded prefix"
        );
    }

    /// Helper: a tree holding `content`'s RAW-tokenized block hashes on w0, the
    /// two workers, and a policy — the fixture the raw-fallback routing tests
    /// share. Returns (policy, workers, model).
    fn raw_prefix_fixture(
        registry: Arc<TokenizerRegistry>,
        content: &str,
    ) -> (CacheAwareZmqPolicy, Vec<Arc<Worker>>, ModelId) {
        let raw_tokens = adapter::encode(&registry.get("tiny").unwrap(), content).unwrap();
        let hashes = compute_block_hashes(&raw_tokens, 4);
        assert!(
            !hashes.is_empty(),
            "raw content must produce at least one block"
        );
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        (policy, workers, ModelId("tiny".into()))
    }

    /// Graceful degradation: a model that HAS a chat template whose render fails
    /// (here it always raises) must fall back to hashing the RAW content and
    /// still route by prefix — not error, not blindly min-load. Exercises the
    /// `request_tokens_for` fall-through that the leaf `encode_chat`-returns-None
    /// tests don't reach at the routing level.
    #[test]
    fn chat_render_failure_falls_back_to_raw_routing() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ raise_exception('boom') }}",
                "bos_token": "<s>",
            }),
        );
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": [{"role": "user", "content": content}],
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "a failed template render must degrade to raw-content routing"
        );
    }

    /// A chat request on a model WITHOUT a chat template routes by the raw
    /// joined `messages[*].content` — the common config where the model ships
    /// no `chat_template`. Covers the `request_tokens_for` path that skips the
    /// template block entirely for a `messages` body.
    #[test]
    fn chat_on_template_less_model_routes_by_raw_content() {
        let registry = tokenizer_registry_with_tiny(); // no template attached
        assert!(!registry.has_chat_encoder("tiny"));
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": [{"role": "user", "content": content}],
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// A `/v1/completions` (`prompt`) request on a model that DOES have a chat
    /// template must still use the raw path — the template applies only to
    /// `messages` traffic. Guards the `messages`-presence gate in
    /// `request_tokens_for`.
    #[test]
    fn completions_prompt_on_templated_model_uses_raw_path() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
                "bos_token": "<s>",
            }),
        );
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        // `prompt` body (no `messages`) -> raw path, so it matches the raw tree.
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": content })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// Two workers both hold the prefix; the lower-load one wins.
    #[test]
    fn tie_break_by_lowest_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        // Both workers hold the prefix.
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0 to load=1; w1 is at 0 — tiebreak picks w1.
        let _g = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Without a reported prefill backlog there is no rotation, however skewed the
    /// request counts are — cache locality wins by default. w0 holds the prefix at
    /// router-side load 20 against an idle w1 and still wins.
    #[test]
    fn no_reported_backlog_means_no_rotation() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(cfg_zero_threshold(), tree, registry, oracle_for_tests(4));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let mut guards = Vec::new();
        for _ in 0..20 {
            guards.push(w0.load_guard());
        }
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
            "no backlog measurement -> keep the cache",
        );
    }

    /// The rotation decision, over the operand grid. Both sides are token counts,
    /// so the rule is one subtraction against the matched prefix and there is no
    /// knob to sweep.
    #[test]
    fn should_rotate_compares_backlog_difference_against_matched_tokens() {
        let b = |candidate_tokens: usize, floor_tokens: usize| PrefillBacklogs {
            floor: worker("http://floor:30000", "tiny"),
            floor_tokens,
            candidate_tokens,
        };
        // candidate backlog, floor backlog, matched tokens, expected, why
        let cases: &[(usize, usize, usize, bool, &str)] = &[
            (
                0,
                0,
                4096,
                false,
                "candidate is the floor — nothing to skip",
            ),
            (500, 500, 4096, false, "equal backlogs — nothing to skip"),
            (
                5000,
                1000,
                4096,
                false,
                "skipping 4000 tokens of queue does not pay for 4096 of prefill",
            ),
            (
                5097,
                1000,
                4096,
                true,
                "skipping 4097 does — one token over the cached prefix",
            ),
            (
                5096,
                1000,
                4096,
                false,
                "exactly equal is not greater: ties keep the cache",
            ),
            (
                100_000,
                0,
                4096,
                true,
                "a huge backlog difference rotates regardless of depth",
            ),
            (1, 0, 0, true, "no matched tokens — nothing to protect"),
        ];
        for &(cand, floor, matched, want, why) in cases {
            assert_eq!(
                CacheAwareZmqPolicy::should_rotate(&b(cand, floor), matched),
                want,
                "should_rotate(candidate {cand}, floor {floor}, matched {matched}): {why}",
            );
        }
    }

    /// End to end: the holder's backlog exceeds the floor's by more than the
    /// cached prefill is worth, so the prefix is seeded onto the floor.
    #[test]
    fn rotates_when_the_floor_owes_enough_less_prefill() {
        let chosen_for = |holder_backlog: u64| -> (String, HitRateWindow) {
            let tree = Arc::new(HashTree::new());
            let registry = tokenizer_registry_with_tiny();
            let text = "hello world hello world hello world";
            let tok = registry.get("tiny").unwrap();
            let ids = adapter::encode(&tok, text).unwrap();
            let hashes = compute_block_hashes(&ids, 4);
            tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

            let engine_load = EngineLoadTable::new();
            let now = Instant::now();
            engine_load.set(
                "http://w0:30000",
                0,
                load_stat_backlog(1, holder_backlog),
                now,
            );
            engine_load.set("http://w1:30000", 0, load_stat_backlog(1, 0), now);

            let policy = new_policy_with_load(
                cfg_zero_threshold(),
                tree,
                registry,
                oracle_for_tests(4),
                engine_load,
            );
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let model = ModelId("tiny".into());
            let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
            let ctx = SelectionContext::new(&model, Some(&body));
            let url = policy
                .select(&workers, &ctx)
                .expect("must pick")
                .url
                .clone();
            (url, policy.hit_rate_window("tiny"))
        };

        // 9 blocks x 4 tokens = 36 matched tokens.
        let (url, w) = chosen_for(36);
        assert_eq!(
            url, "http://w0:30000",
            "a 36-token backlog difference exactly equals the cached prefill: keep it",
        );
        assert_eq!(
            (w.routed, w.rotated, w.fell_back),
            (1, 0, 0),
            "keeping the cache is a Routed outcome",
        );

        let (url, w) = chosen_for(37);
        assert_eq!(
            url, "http://w1:30000",
            "one token more and the floor is worth seeding",
        );
        // The outcome accounting is the operator's ONLY aggregate view of
        // rotation, so it has to be pinned at the `select` level: recording this
        // as `Routed` changes no routing behaviour and would otherwise pass the
        // whole suite.
        assert_eq!(
            (w.routed, w.rotated, w.fell_back),
            (0, 1, 0),
            "a rotation must be counted as Rotated, not folded into routed/fell_back",
        );
    }

    /// The cache candidate is chosen with the same key as the rotation floor.
    ///
    /// Both workers hold the prefix. w1 has the smaller prefill queue; w0 has the
    /// lower router-side load. Ordering holders by load alone picks w0 and then
    /// declines to rotate (the 10-token gap is under the 36-token prefix), parking
    /// the request behind the LARGER queue and recording an ordinary `Routed`
    /// selection with nothing logged. Ordering by tokens picks w1 outright.
    #[test]
    fn cache_candidate_is_chosen_by_the_same_key_as_the_floor() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat_backlog(0, 50_000), now);
        engine_load.set("http://w1:30000", 0, load_stat_backlog(0, 49_990), now);

        let policy = new_policy_with_load(
            cfg_zero_threshold(),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // w1 carries the router-side load, so a load-only key prefers w0.
        let _guards: Vec<_> = (0..5).map(|_| w1.load_guard()).collect();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
            "the holder with the shorter prefill queue must win, not the one with \
             fewer in-flight requests",
        );
        let w = policy.hit_rate_window("tiny");
        assert_eq!(
            (w.routed, w.rotated),
            (1, 0),
            "picking the right holder directly is Routed — not a rotation away from \
             the wrong one",
        );
    }

    /// When holders report inconsistently, the whole matched set falls back to
    /// `load_of` ordering — reporting-ness must never become the sort key.
    ///
    /// Both workers hold the identical prefix, so no cache is traded either way.
    /// Both directions are asserted, because the two ways of getting this wrong fail
    /// OPPOSITE ways and one fixture cannot see both: a `usize::MAX` sentinel always
    /// prefers the reporting holder, `unwrap_or(0)` always prefers the silent one.
    /// Only ordering the whole set by `load_of` gets both cases right.
    #[test]
    fn mixed_reporting_holders_fall_back_to_load_ordering() {
        let winner_when = |silent_depth: u64, loud_depth: u64, loud_backlog: u64| -> String {
            let tree = Arc::new(HashTree::new());
            let registry = tokenizer_registry_with_tiny();
            let text = "hello world hello world hello world";
            let tok = registry.get("tiny").unwrap();
            let ids = adapter::encode(&tok, text).unwrap();
            let hashes = compute_block_hashes(&ids, 4);
            tree.insert(
                &KvWorkerId::new("http://silent:30000".into(), 0),
                None,
                &hashes,
            );
            tree.insert(
                &KvWorkerId::new("http://loud:30000".into(), 0),
                None,
                &hashes,
            );

            let engine_load = EngineLoadTable::new();
            let now = Instant::now();
            engine_load.set("http://silent:30000", 0, load_stat(silent_depth, 0), now);
            engine_load.set(
                "http://loud:30000",
                0,
                load_stat_backlog(loud_depth, loud_backlog),
                now,
            );

            let policy = new_policy_with_load(
                cfg_zero_threshold(),
                tree,
                registry,
                oracle_for_tests(4),
                engine_load,
            );
            let workers = vec![
                worker("http://silent:30000", "tiny"),
                worker("http://loud:30000", "tiny"),
            ];
            let model = ModelId("tiny".into());
            let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
            let ctx = SelectionContext::new(&model, Some(&body));
            policy
                .select(&workers, &ctx)
                .expect("must pick")
                .url
                .clone()
        };

        // The silent holder is the idle one, so it wins. A MAX sentinel would send
        // the request to the saturated holder purely because it publishes the field.
        assert_eq!(
            winner_when(0, 200, 90_000),
            "http://silent:30000",
            "publishing the field must not beat being idle",
        );
        // The reporting holder is the idle one, so it wins. `unwrap_or(0)` would read
        // the silent holder as having an empty backlog and send the request there.
        assert_eq!(
            winner_when(200, 0, 100),
            "http://loud:30000",
            "not publishing the field must not read as an empty backlog",
        );
    }

    /// `matched_tokens` is built from MATCHED blocks, not the prompt's total.
    ///
    /// Every other rotation fixture inserts the whole prompt, making the two
    /// indistinguishable. Here the tree holds only the first 5 of 9 blocks, so
    /// matched = 20 tokens and total = 36; the 21-token backlog gap rotates on the
    /// former and not on the latter.
    #[test]
    fn rotation_weighs_the_matched_prefix_not_the_whole_prompt() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert_eq!(hashes.len(), 9, "fixture assumes a 9-block prompt");
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &hashes[..5],
        );

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat_backlog(0, 21), now);
        engine_load.set("http://w1:30000", 0, load_stat_backlog(0, 0), now);

        let policy = new_policy_with_load(
            cfg_zero_threshold(),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
            "21 > 20 matched tokens rotates; against the 36-token total it would not",
        );
        assert_eq!(policy.hit_rate_window("tiny").rotated, 1);
    }

    /// `holders` counts distinct URLs, so a multi-rank engine reads as one holder.
    #[test]
    fn distinct_holder_urls_collapses_ranks_of_one_worker() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, "hello world hello world hello world").unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        for rank in 0..4 {
            tree.insert(
                &KvWorkerId::new("http://dp:30000".into(), rank),
                None,
                &hashes,
            );
        }
        let matched = tree.match_prefix(None, &hashes);
        assert_eq!(
            matched.workers.len(),
            4,
            "the tree really does key holders per rank",
        );
        assert_eq!(
            distinct_holder_urls(&matched).len(),
            1,
            "one dp-4 replica is ONE holder, not four",
        );
    }

    /// `backlog_reporting_count` is scoped to the candidate slice. The load table
    /// spans every registered worker of every role, so a fleet-wide count can read
    /// healthy while none of the workers being routed over report anything.
    #[test]
    fn backlog_reporting_count_counts_candidates_not_the_whole_table() {
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        // Two reporting workers in the table, only one of them a candidate.
        engine_load.set("http://reporting:30000", 0, load_stat_backlog(0, 10), now);
        engine_load.set("http://elsewhere:30000", 0, load_stat_backlog(0, 10), now);
        engine_load.set("http://silent:30000", 0, load_stat(0, 0), now);
        let loads = WorkerLoads::from_engine(&engine_load, now);

        let candidates = vec![
            worker("http://reporting:30000", "tiny"),
            worker("http://silent:30000", "tiny"),
            worker("http://absent:30000", "tiny"),
        ];
        assert_eq!(
            loads.engine_worker_count(),
            3,
            "the table is fleet-wide: three workers published",
        );
        assert_eq!(
            loads.backlog_reporting_count(&candidates),
            1,
            "only one of these three candidates can be a rotation destination",
        );
    }

    /// The floor is ordered by queued TOKENS first; router-side load only breaks
    /// ties. Here the holder has the largest backlog but the LOWEST load — a
    /// realistic shape (long prefill queue, few in-flight requests) — so a key
    /// that put load first would name the holder its own floor and switch
    /// rotation off. No other fixture separates the two orderings.
    #[test]
    fn floor_ordering_is_tokens_primary_not_load_primary() {
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://holder:30000", 0, load_stat_backlog(0, 50_000), now);
        engine_load.set("http://light:30000", 0, load_stat_backlog(0, 100), now);
        let loads = WorkerLoads::from_engine(&engine_load, now);

        let holder = worker("http://holder:30000", "tiny");
        let light = worker("http://light:30000", "tiny");
        // `light` carries the higher router-side load, so a load-primary key would
        // prefer the holder.
        let _guards: Vec<_> = (0..5).map(|_| light.load_guard()).collect();
        let workers = vec![Arc::clone(&holder), Arc::clone(&light)];

        let b = CacheAwareZmqPolicy::backlogs(&workers, &loads, &holder).expect("both report");
        assert_eq!(
            b.floor.url, "http://light:30000",
            "fewer queued TOKENS wins, whatever the request counts say",
        );
        assert_eq!((b.candidate_tokens, b.floor_tokens), (50_000, 100));
    }

    /// A worker silent about its backlog is never a rotation DESTINATION, but it
    /// does not veto the decision either:
    ///
    /// 1. The only alternative to the holder is silent → no floor exists → the
    ///    holder keeps the request. A missing backlog must never read as "idle".
    /// 2. A silent third worker alongside a reporting one → rotation still
    ///    happens, onto the reporting one. Otherwise one engine mid-rollout would
    ///    take the whole pool's rotation with it.
    #[test]
    fn a_silent_worker_is_never_the_floor_but_never_vetoes_rotation() {
        let chosen_for = |extra: &[(&str, LoadStat)]| -> String {
            let tree = Arc::new(HashTree::new());
            let registry = tokenizer_registry_with_tiny();
            let text = "hello world hello world hello world";
            let tok = registry.get("tiny").unwrap();
            let ids = adapter::encode(&tok, text).unwrap();
            let hashes = compute_block_hashes(&ids, 4);
            tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

            let engine_load = EngineLoadTable::new();
            let now = Instant::now();
            // The holder is deep in prefill debt: it rotates the moment any
            // measurable worker is lighter.
            engine_load.set("http://w0:30000", 0, load_stat_backlog(1, 100_000), now);
            let mut workers = vec![worker("http://w0:30000", "tiny")];
            for (url, stat) in extra {
                engine_load.set(url, 0, stat.clone(), now);
                workers.push(worker(url, "tiny"));
            }

            let policy = new_policy_with_load(
                cfg_zero_threshold(),
                tree,
                registry,
                oracle_for_tests(4),
                engine_load,
            );
            let model = ModelId("tiny".into());
            let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
            let ctx = SelectionContext::new(&model, Some(&body));
            policy
                .select(&workers, &ctx)
                .expect("must pick")
                .url
                .clone()
        };

        assert_eq!(
            chosen_for(&[("http://w1:30000", load_stat(1, 0))]),
            "http://w0:30000",
            "the only alternative is unmeasurable: it must not be treated as the idle floor",
        );
        assert_eq!(
            chosen_for(&[
                ("http://w1:30000", load_stat(1, 0)),
                ("http://w2:30000", load_stat_backlog(1, 0)),
            ]),
            "http://w2:30000",
            "a silent worker must not disable rotation onto a worker that DOES report",
        );
    }

    /// Workers tied on prefill backlog are ordered by router-side load, not by
    /// slice position — see `backlogs` for why the tie case is the one that
    /// matters.
    #[test]
    fn floor_breaks_backlog_ties_on_router_side_load() {
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://holder:30000", 0, load_stat_backlog(1, 9_000), now);
        // Two candidate floors, both reporting an empty prefill queue.
        engine_load.set("http://first:30000", 0, load_stat_backlog(0, 0), now);
        engine_load.set("http://second:30000", 0, load_stat_backlog(0, 0), now);
        let loads = WorkerLoads::from_engine(&engine_load, now);

        let holder = worker("http://holder:30000", "tiny");
        let first = worker("http://first:30000", "tiny");
        let second = worker("http://second:30000", "tiny");
        // `first` sorts earlier and would win on token count alone; give it the
        // router-side load that the frozen engine snapshot cannot yet show.
        let _g1 = first.load_guard();
        let _g2 = first.load_guard();
        let workers = vec![Arc::clone(&holder), Arc::clone(&first), Arc::clone(&second)];

        let b = CacheAwareZmqPolicy::backlogs(&workers, &loads, &holder).expect("all report");
        assert_eq!(b.floor_tokens, 0, "both floors are tied at zero tokens");
        assert_eq!(
            b.floor.url, "http://second:30000",
            "the tie must break to the worker this router has dispatched less to, \
             not to slice order",
        );
    }

    /// `backlogs` reads both operands in one pass, so a candidate that IS the
    /// least-backlogged worker shows a zero difference and cannot rotate onto
    /// itself.
    #[test]
    fn backlogs_reads_both_operands_in_one_pass() {
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat_backlog(1, 400), now);
        engine_load.set("http://w1:30000", 0, load_stat_backlog(1, 900), now);
        let loads = WorkerLoads::from_engine(&engine_load, now);

        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        let b = CacheAwareZmqPolicy::backlogs(&workers, &loads, &w0).expect("both report");
        assert_eq!(b.floor.url, "http://w0:30000");
        assert_eq!((b.floor_tokens, b.candidate_tokens), (400, 400));
        assert_eq!(
            b.candidate_tokens.saturating_sub(b.floor_tokens),
            0,
            "the candidate is its own floor: no difference, so no rotation",
        );

        let b = CacheAwareZmqPolicy::backlogs(&workers, &loads, &w1).expect("both report");
        assert_eq!(b.floor.url, "http://w0:30000");
        assert_eq!((b.floor_tokens, b.candidate_tokens), (400, 900));

        // A candidate absent from the load snapshot has no backlog to compare, so
        // there is no pair to return.
        let unknown = worker("http://elsewhere:30000", "tiny");
        assert!(
            CacheAwareZmqPolicy::backlogs(&workers, &loads, &unknown).is_none(),
            "a candidate with no reported backlog yields no comparison",
        );
    }

    /// Fresh engine-reported load drives the matched-set tiebreak instead of the
    /// router-side in-flight counter. Both workers hold the prefix and have zero
    /// router-side load, so without engine load the tiebreak would pick w0
    /// (stable order). Engine load says w0 is hot (50) and w1 is light (1), so the
    /// lowest-load holder is w1.
    #[test]
    fn engine_load_overrides_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(50, 0), now);
        engine_load.set("http://w1:30000", 0, load_stat(1, 0), now);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        // Router-side counters are both 0 — only engine load is skewed.
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "engine-reported load must drive selection",
        );
    }

    /// The matched-set tiebreak uses engine load, not `active_load()`: both
    /// workers hold the prefix, engine load says w1 is lighter → w1 wins.
    /// (Guards against a regression that reverted the tiebreak to
    /// `active_load()`.) Neither engine reports a prefill backlog, so rotation
    /// never enters the picture and the tiebreak is what is under test.
    #[test]
    fn matched_set_tiebreak_uses_engine_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(10, 0), now);
        engine_load.set("http://w1:30000", 0, load_stat(2, 0), now);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "matched-set tiebreak must use engine load",
        );
    }

    /// Recent dispatches made AFTER the engine's last snapshot are added on
    /// top of the reported load. Without this, repeated `select` calls in
    /// the same burst would all read the same "worker looks idle" engine
    /// number and all pile onto it before the gauge catches up. w0 looks
    /// lighter by the raw engine numbers alone (1 vs 3), but three slots
    /// claimed on w0 after the snapshot flip the effective load in w1's
    /// favor (1+3=4 > 3+0=3).
    #[test]
    fn recent_dispatches_are_added_on_top_of_engine_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let snapshot_at = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(1, 0), snapshot_at);
        engine_load.set("http://w1:30000", 0, load_stat(3, 0), snapshot_at);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Three requests dispatched to w0 AFTER the engine's snapshot —
        // exactly the "burst the engine hasn't reported back on yet" shape.
        let _g1 = w0.load_guard();
        let _g2 = w0.load_guard();
        let _g3 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "w0's effective load (1 engine + 3 recent = 4) must exceed w1's \
             (3 engine + 0 recent = 3), even though the raw engine numbers \
             alone favor w0",
        );
    }

    /// `load_of` must use the OLDEST rank's timestamp as the "since" cutoff
    /// for a multi-rank worker, not the newest — this pins the end-to-end
    /// wiring of the choice `EngineLoadTable::fresh_worker_state` makes (see
    /// its doc comment). A regression to "newest" would silently treat the
    /// dispatch below as already covered by rank1's later snapshot, even
    /// though rank0's older snapshot doesn't reflect it.
    #[test]
    fn load_of_uses_oldest_rank_timestamp_for_multi_rank_worker() {
        let engine_load = EngineLoadTable::new();
        let earlier = Instant::now();
        let w = worker("http://w:30000", "tiny");
        // Real sleeps, not synthetic `Instant` offsets: the dispatch's
        // timestamp is captured internally by `load_guard()` and isn't
        // injectable (see `worker.rs`'s `slots_acquired_since` tests for the
        // same reasoning).
        std::thread::sleep(Duration::from_millis(5));
        let _g = w.load_guard(); // dispatched strictly between earlier/later
        std::thread::sleep(Duration::from_millis(5));
        let later = Instant::now();
        engine_load.set("http://w:30000", 0, load_stat(1, 0), earlier);
        engine_load.set("http://w:30000", 1, load_stat(1, 0), later);

        let loads = WorkerLoads::from_engine(&engine_load, later);
        assert_eq!(
            loads.load_of(&w),
            3,
            "depth (1+1=2) plus the one dispatch made after the OLDEST \
             rank's timestamp = 3; using the newest rank's timestamp \
             instead would exclude that dispatch and wrongly give 2",
        );
    }

    /// A stale engine snapshot falls back to PURE `active_load()` — the
    /// recent-dispatch correction only applies alongside a fresh snapshot
    /// (see `load_of`'s `Some` branch). A regression that added
    /// `slots_acquired_since` to the fallback branch too would double-count
    /// this worker's own in-flight guards.
    #[test]
    fn load_of_fallback_does_not_add_recent_dispatches_on_top_of_active_load() {
        let engine_load = EngineLoadTable::new();
        let stale = Instant::now() - Duration::from_secs(3600);
        engine_load.set("http://w:30000", 0, load_stat(50, 0), stale);
        let w = worker("http://w:30000", "tiny");
        let _g1 = w.load_guard();
        let _g2 = w.load_guard();

        let loads = WorkerLoads::from_engine(&engine_load, Instant::now());
        assert_eq!(
            loads.load_of(&w),
            2,
            "must equal active_load() exactly (2) — not the stale depth \
             (50) plus anything, and not active_load() plus a second \
             correction",
        );
    }

    /// A stale engine snapshot is ignored: selection falls back to the
    /// router-side `active_load` counter. w0's (stale) engine load is high,
    /// but w1 carries a router-side guard, so fallback picks w0.
    #[test]
    fn stale_engine_load_falls_back_to_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        // Past the default freshness window; an hour ago is comfortably stale.
        let engine_load = EngineLoadTable::new();
        let stale = Instant::now() - Duration::from_secs(3600);
        engine_load.set("http://w0:30000", 0, load_stat(50, 0), stale);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Router-side: w1 has one in-flight request, w0 has none. With the
        // stale engine load ignored, the tiebreak picks w0 (load 0 < 1).
        let _g = w1.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "stale engine load must be ignored in favour of active_load",
        );
    }

    /// Tokenizer is missing for the requested model → fall back to
    /// min-load (no panic, no error).
    #[test]
    fn missing_tokenizer_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let empty_registry = Arc::new(TokenizerRegistry::default());
        let policy = new_policy(cfg_default(), tree, empty_registry, oracle_for_tests(4));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Missing body → fall back to min-load.
    #[test]
    fn missing_request_body_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Body present but no recognizable prompt field → fall back.
    #[test]
    fn body_without_prompt_field_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"frobnicate":42}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Body has a non-text shape that yields zero tokens → fall back.
    /// (Tokenizer always returns ≥0 ids; an empty string yields the
    /// empty vec, then `compute_block_hashes` returns empty too.)
    #[test]
    fn empty_text_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":""}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Match rate below the threshold → fall back. Threshold = 0.99
    /// means the tree must match every single block; we insert an
    /// UNRELATED chain so the rate is 0.
    #[test]
    fn low_match_rate_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        // Tree contains a chain unrelated to the test's request.
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &[999, 998, 997],
        );

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.99,
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello world hello world hello world"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Byte-slice helper over the shared `extract_prompt_text_from_value` free
    /// function, so the extraction-shape tests below stay terse.
    fn extract_prompt_text(body: &[u8]) -> Option<String> {
        let v: serde_json::Value = serde_json::from_slice(body).ok()?;
        crate::policies::extract_prompt_text_from_value(&v)
    }

    /// Chat completions shape with `messages[*].content` string.
    #[test]
    fn extract_prompt_chat_string_content() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hello"}]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "hello");
    }

    /// Chat completions shape with multimodal content blocks (text parts).
    #[test]
    fn extract_prompt_chat_block_content() {
        let body = br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":"x"}]}]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "hi");
    }

    /// `/v1/completions` array form is joined with newlines.
    #[test]
    fn extract_prompt_completions_array() {
        let body = br#"{"prompt":["a","b","c"]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "a\nb\nc");
    }

    /// SGLang native `text` field.
    #[test]
    fn extract_prompt_sglang_text_field() {
        let body = br#"{"text":"abc"}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "abc");
    }

    /// Unknown shape → None.
    #[test]
    fn extract_prompt_unknown_shape_returns_none() {
        let body = br#"{"frobnicate":42}"#;
        assert!(extract_prompt_text(body).is_none());
    }

    /// Lifecycle: removing a worker from the tree via `clear_worker`
    /// makes subsequent matches miss; the policy then falls back to
    /// min-load.
    #[test]
    fn lifecycle_clear_worker_removes_overlap() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        let kw0 = KvWorkerId::new("http://w0:30000".into(), 0);
        tree.insert(&kw0, None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree.clone(),
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();

        // Before clear: w0 wins.
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");

        // After clear: tree no longer attributes the prefix to w0.
        tree.clear_worker(&kw0);
        // Bump w0's load so min-load fallback distinguishes from w1.
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let chosen2 = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen2.url, "http://w1:30000");
    }

    /// `request_tokens_for` flags chat-encoder output as engine-equivalent (safe
    /// to forward to the engine as `input_ids`): the ids match what the engine
    /// tokenizes from its own chat template.
    #[test]
    fn request_tokens_chat_encoder_is_engine_equivalent() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
                "bos_token": "<s>",
            }),
        );
        let messages = serde_json::json!([{"role":"user","content":"hello world"}]);
        let expected = registry
            .encode_chat(
                "tiny",
                &messages,
                None,
                crate::tokenizer::dsv4::RenderOpts::chat(),
            )
            .unwrap();

        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "model": "tiny", "messages": messages });
        let rt = request_tokens_for(&registry, &model, &value).expect("tokens");
        assert!(
            rt.engine_equivalent,
            "chat-encoder ids must be engine-equivalent"
        );
        assert_eq!(rt.ids, expected);
    }

    /// End-to-end: `request_tokens_for` threads the request's thinking mode through
    /// `resolve_render_opts` → `encode_chat` → the dsv4 encoder, so a thinking-mode
    /// body produces DIFFERENT routing tokens (the generation prompt opens `<think>`)
    /// than the same messages in chat mode. Without this the whole feature's wiring
    /// is untested: every other routing test uses a chat-mode body, so a refactor
    /// that hardcoded `RenderOpts::chat()` in `request_tokens_for` would stay green
    /// while silently routing thinking-mode endpoints on chat-mode tokens.
    #[test]
    fn request_tokens_for_threads_thinking_mode() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_encoder_for_test("tiny", crate::tokenizer::ChatEncoder::DeepSeekV4);
        let model = ModelId("tiny".into());
        let messages = serde_json::json!([{"role":"user","content":"hello world"}]);

        let chat_ids = request_tokens_for(
            &registry,
            &model,
            &serde_json::json!({ "messages": messages.clone() }),
        )
        .expect("tokens")
        .ids;
        let thinking_ids = request_tokens_for(
            &registry,
            &model,
            &serde_json::json!({
                "messages": messages.clone(),
                "chat_template_kwargs": {"thinking": true}
            }),
        )
        .expect("tokens")
        .ids;

        assert_ne!(
            chat_ids, thinking_ids,
            "request_tokens_for must thread chat_template_kwargs.thinking into the render"
        );
        // The thinking body matches the encoder driven with thinking opts directly.
        let expected_thinking = registry
            .encode_chat(
                "tiny",
                &messages,
                None,
                crate::tokenizer::dsv4::RenderOpts {
                    thinking: true,
                    reasoning_effort: crate::tokenizer::dsv4::ReasoningEffort::None,
                },
            )
            .unwrap();
        assert_eq!(thinking_ids, expected_thinking);
    }

    /// `request_tokens_for` on the raw-prompt path (no chat encoder) is NOT
    /// engine-equivalent — the engine would still apply its template, so the
    /// router's raw ids must not be forwarded as `input_ids`.
    #[test]
    fn request_tokens_raw_prompt_not_engine_equivalent() {
        let registry = tokenizer_registry_with_tiny(); // no template attached
        assert!(!registry.has_chat_encoder("tiny"));
        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "prompt": "hello world" });
        let rt = request_tokens_for(&registry, &model, &value).expect("tokens");
        assert!(!rt.engine_equivalent);
        assert!(!rt.ids.is_empty());
    }

    /// `request_tokens_for` returns `None` when there is no routable prompt
    /// field — the handler then forwards nothing and the engine tokenizes as
    /// usual.
    #[test]
    fn request_tokens_none_for_unroutable_body() {
        let registry = tokenizer_registry_with_tiny();
        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "frobnicate": 42 });
        assert!(request_tokens_for(&registry, &model, &value).is_none());
    }

    /// `select` consumes the ingress-precomputed tokens and does NOT
    /// re-tokenize the body: the body here tokenizes to an unrelated prefix
    /// (which the tree does not hold), but the ctx tokens point at w0's cached
    /// prefix, so w0 wins. If `select` re-tokenized the body it would miss and
    /// fall back to min-load (w1).
    #[test]
    fn select_prefers_ingress_tokens_over_body() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let tree_ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&tree_ids, 4);
        assert!(!hashes.is_empty());
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
            },
            tree,
            registry,
            oracle_for_tests(4),
            EngineLoadTable::new(),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Load w0 so a min-load fallback would pick w1 — distinguishes "used
        // ctx tokens (w0)" from "re-tokenized the body and missed (w1)".
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        // Body tokenizes to an unrelated prefix the tree does NOT hold.
        let body = serde_json::to_vec(&serde_json::json!({"prompt":"zzz unrelated"})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body)).with_request_tokens(Some(&tree_ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "select must use ctx tokens (w0's prefix), not re-tokenize the body"
        );
    }
}
