// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine-reported runtime load, fed by the load subscriber.
//!
//! Workers publish a [`LoadStat`] gauge on their dedicated load socket (see
//! `python/sglang/srt/managers/scheduler_components/load_publisher.py`). The
//! load subscriber routes those into this table, keyed per
//! `(worker_url, dp_rank)`; the
//! cache-aware-zmq policy reads the freshest aggregate per worker as a
//! truthful load signal, falling back to the router-side in-flight counter
//! when no fresh snapshot exists (cold start, stale publisher, or a worker
//! that predates load publishing).
//!
//! Load is a *gauge*, not a delta: last value wins, no sequence/replay
//! semantics. Entries older than [`EngineLoadTable::freshness`] are ignored.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::{DashMap, DashSet};
use serde::de::{self, Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde::Deserialize;

/// Per-scheduler runtime load snapshot. Mirrors the Python `LoadStat` in
/// `managers/scheduler_components/load_publisher.py`, published on the
/// worker's dedicated load socket (separate from KV-cache events).
///
/// Wire shape (msgspec `tag=True` + `array_like`):
/// `["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
/// max_total_num_tokens, attn_dp_rank, num_waiting_uncached_tokens?]`. The
/// encoding is POSITIONAL, so fields are only ever appended; we read the prefix
/// we know and drain the rest. `attn_dp_rank` is ignored (the router keys load
/// by the subscriber's socket rank, not the payload) but must still be consumed
/// in order.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadStat {
    /// Requests currently running on the engine.
    pub num_running_reqs: u64,
    /// Requests queued waiting to run.
    pub num_waiting_reqs: u64,
    /// KV tokens currently in use.
    pub num_tokens: u64,
    /// KV-cache token capacity; 0 when unknown.
    pub max_total_num_tokens: u64,
    /// Uncached input tokens the engine still owes prefill compute for, net of its
    /// own device+host radix match.
    ///
    /// `None` is the COMMON case, not an exotic one: the engine publishes this only
    /// when it actually computes the radix discount, which needs a cache-aware
    /// `--schedule-policy` (`lpm` / `dfs-weight`) and the radix cache enabled. The
    /// default is `fcfs`, so a default fleet sends `None` and everything here
    /// declines to act. An engine predating the field also sends `None`.
    ///
    /// Known residual on a fleet that DOES publish: the engine's gate is a startup
    /// property, so a worker saturated enough to skip its scheduling pass (or an
    /// `lpm` worker with a waiting queue over 128) reports raw sequence length
    /// under a discounted label — biased HIGH, i.e. toward rotating away from it.
    /// See `_uncached_tokens_are_discounted` in
    /// `python/sglang/srt/managers/scheduler_components/load_publisher.py`.
    ///
    /// `None` is deliberately distinguished from `Some(0)` ("nothing queued"): a
    /// router that cannot see this number must not pretend the backlog is empty.
    /// Everything that consumes it declines to act rather than guessing.
    ///
    /// A DECODE-role engine in PD disaggregation reports a structural `Some(0)`:
    /// it does no prefill, so the zero is permanent rather than a measurement.
    /// Safe for the router today because PD pools are selected separately, so a
    /// decode pool is uniformly zero and no comparison inside it can move — but a
    /// future consumer that mixes roles in one comparison must not read that zero
    /// as "idle".
    pub pending_prefill_tokens: Option<u64>,
}

impl<'de> Deserialize<'de> for LoadStat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LoadStatVisitor;

        impl<'de> Visitor<'de> for LoadStatVisitor {
            type Value = LoadStat;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a tagged msgpack array [\"LoadStat\", ...fields]")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<LoadStat, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let tag: String = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("event tag"))?;
                if tag != "LoadStat" {
                    return Err(de::Error::custom(format!(
                        "expected \"LoadStat\" tag, got {tag:?}"
                    )));
                }
                // Counts are always emitted (no Python defaults), but default
                // missing fields to 0 and drain trailing fields (attn_dp_rank,
                // future additions) for forward-compatibility.
                let num_running_reqs: u64 = seq.next_element()?.unwrap_or(0);
                let num_waiting_reqs: u64 = seq.next_element()?.unwrap_or(0);
                let num_tokens: u64 = seq.next_element()?.unwrap_or(0);
                let max_total_num_tokens: u64 = seq.next_element()?.unwrap_or(0);
                // `attn_dp_rank` sits between the counts and the fields added
                // after it. Consumed positionally and discarded: without this read
                // the rank's VALUE would land in `pending_prefill_tokens` — a dp
                // rank silently reported as a token backlog. Pinned by
                // `wire_seven_element_payload_reads_the_token_field_not_the_rank`.
                //
                // Read as `IgnoredAny`, not `u64`: msgspec's `omit_defaults` does
                // NOT trim trailing defaults on an `array_like` struct, so a
                // Python-side `None` reaches us as an explicit msgpack nil rather
                // than as a short array. Typing this `u64` would reject the whole
                // frame on a nil — dropping the snapshot and losing the worker's
                // load entirely, which is strictly worse than not knowing one
                // field.
                let _attn_dp_rank: Option<IgnoredAny> = seq.next_element()?;
                // Absent on engines that predate it; `None` means "unknown", never
                // "zero". `.flatten()` folds the two ways it can be absent —
                // array ended, or an explicit nil — into the same `None`, so the
                // Rust `Option` means what the Python `Optional[int]` declares.
                let pending_prefill_tokens: Option<u64> =
                    seq.next_element::<Option<u64>>()?.flatten();
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(LoadStat {
                    num_running_reqs,
                    num_waiting_reqs,
                    num_tokens,
                    max_total_num_tokens,
                    pending_prefill_tokens,
                })
            }
        }

        deserializer.deserialize_seq(LoadStatVisitor)
    }
}

/// Decode a single load frame's msgpack payload into a [`LoadStat`].
pub fn decode_load_stat(payload: &[u8]) -> Result<LoadStat, rmp_serde::decode::Error> {
    rmp_serde::from_slice(payload)
}

/// One worker's fused load, as of a [`EngineLoadTable::fresh_worker_state`] pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FreshLoad {
    /// Queue depth summed across the worker's ranks (`running + waiting`). A
    /// relative quantity — only ever compared against other workers' depths, so
    /// the sum is fine.
    pub(crate) depth: usize,
    /// Queued prefill tokens, MEAN over the worker's ranks. `None` when any rank
    /// does not report it (an engine predating the field) — never zero.
    ///
    /// Mean, not sum, because this one IS compared against an absolute quantity:
    /// a request's matched prefix. Under dp-attention a URL fronts `D`
    /// independent schedulers and a dispatch lands on exactly ONE of them, so the
    /// wait a request actually pays is one rank's backlog — the mean is its
    /// expected value. Summing would scale both rotation operands by `D` while
    /// the matched prefix stayed fixed, making rotation fire ~`D`× more eagerly
    /// than the derivation on `should_rotate` claims. Depth above escapes this
    /// because `D` is constant across a pool and cancels in a min.
    pub(crate) pending_prefill_tokens: Option<usize>,
    /// Oldest contributing rank's timestamp, so a caller correcting for
    /// since-snapshot dispatches uses the conservative cutoff.
    pub(crate) at: Instant,
}

/// A per-rank load snapshot older than this is treated as stale, so a silent
/// or slow publisher degrades to the router-side load signal rather than
/// pinning a worker at its last reported value.
const DEFAULT_FRESHNESS: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
struct LoadEntry {
    load: LoadStat,
    at: Instant,
}

/// Per-`(worker_url, dp_rank)` engine-reported load. Written by the load
/// subscriber pump, read by the cache-aware-zmq policy. Shared out of
/// [`super::kv_events::index::KvEventIndex`] the same way the hash tree is.
#[derive(Debug)]
pub struct EngineLoadTable {
    by_rank: DashMap<(String, u32), LoadEntry>,
    /// Worker URLs that advertised a load topic and so are *expected* to
    /// publish load. Lets the router distinguish "load-aware routing active"
    /// from "silently degraded to the in-flight counter" (expected but no
    /// fresh snapshot) — see [`Self::expected_count`].
    expected: DashSet<String>,
    freshness: Duration,
}

impl EngineLoadTable {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness: DEFAULT_FRESHNESS,
        })
    }

    #[cfg(test)]
    pub fn with_freshness(freshness: Duration) -> Arc<Self> {
        Arc::new(Self {
            by_rank: DashMap::new(),
            expected: DashSet::new(),
            freshness,
        })
    }

    /// Record the latest load for one `(worker_url, dp_rank)`.
    pub fn set(&self, url: &str, dp_rank: u32, load: LoadStat, at: Instant) {
        self.by_rank
            .insert((url.to_string(), dp_rank), LoadEntry { load, at });
    }

    /// Mark a worker as expected to publish load (it advertised a load topic).
    pub fn mark_expected(&self, url: &str) {
        self.expected.insert(url.to_string());
    }

    /// Number of workers expected to publish load. Compared against the size
    /// of [`Self::snapshot_fresh`] to surface a dead/misconfigured publisher
    /// (expected > 0 but no fresh snapshots) in logs.
    pub fn expected_count(&self) -> usize {
        self.expected.len()
    }

    /// Shared accumulation pass behind [`Self::snapshot_fresh`] (and used
    /// directly by `cache_aware_zmq::WorkerLoads`, which needs both halves):
    /// one walk of the table, per worker URL, producing the summed queue
    /// depth (`num_running_reqs + num_waiting_reqs`) across that worker's
    /// ranks and the OLDEST snapshot timestamp among them — **but only for
    /// workers whose every known rank is fresh**. A worker with any stale
    /// rank is omitted, so the caller falls back to its own load signal.
    /// (Summing only the fresh ranks would make a worker whose other ranks
    /// went silent look misleadingly idle and draw *more* traffic.)
    /// `snapshot_fresh` and any other consumer walking this same pass can
    /// never disagree with each other about which workers count as fresh.
    ///
    /// The oldest (not newest) rank's timestamp is deliberately what's kept
    /// alongside the depth: a caller using it as a "dispatches not yet
    /// reflected in this number" cutoff (see
    /// `crate::policies::cache_aware_zmq::WorkerLoads::load_of`) needs a
    /// bound that never treats an unreported dispatch as already-covered —
    /// the freshest rank's timestamp could do exactly that for whichever
    /// rank published less recently. This conservatism is one-sided, not
    /// free: for a multi-rank worker with skewed publish times, a dispatch
    /// that landed on (and was already reported by) the FRESHER rank can
    /// get re-added by the caller's cutoff-based correction anyway, since
    /// that correction has no way to attribute a dispatch to a specific
    /// rank. That's an accepted, bounded over-count (it biases the wrong
    /// direction relative to the under-count this method exists to avoid,
    /// not a correctness hole) rather than something this method can close
    /// on its own — closing it would require per-rank dispatch attribution,
    /// which the router-side slot tracking below doesn't have.
    pub(crate) fn fresh_worker_state(&self, now: Instant) -> HashMap<String, FreshLoad> {
        /// Per-URL fold state. Named rather than a tuple: it carries two `bool`s
        /// and three counters, and a positional mix-up between the `bool`s would
        /// make a stale worker report as fresh.
        struct Acc {
            depth: usize,
            /// Summed over ranks, then divided by `ranks` on the way out — see
            /// [`FreshLoad::pending_prefill_tokens`].
            tokens: u64,
            ranks: u32,
            all_fresh: bool,
            all_report_tokens: bool,
            oldest_at: Instant,
        }
        let mut acc: HashMap<String, Acc> = HashMap::new();
        for entry in self.by_rank.iter() {
            let at = entry.value().at;
            let fresh = now.duration_since(at) <= self.freshness;
            let l = &entry.value().load;
            let depth = (l.num_running_reqs.saturating_add(l.num_waiting_reqs)) as usize;
            let slot = acc.entry(entry.key().0.clone()).or_insert(Acc {
                depth: 0,
                tokens: 0,
                ranks: 0,
                all_fresh: true,
                all_report_tokens: true,
                oldest_at: at,
            });
            slot.depth = slot.depth.saturating_add(depth);
            slot.ranks = slot.ranks.saturating_add(1);
            // Usable only if EVERY rank reports: summing a subset would understate
            // the backlog, and understating it is the direction that wrongly makes
            // a worker look like the idle floor.
            match l.pending_prefill_tokens {
                Some(t) => slot.tokens = slot.tokens.saturating_add(t),
                None => slot.all_report_tokens = false,
            }
            slot.all_fresh = slot.all_fresh && fresh;
            slot.oldest_at = slot.oldest_at.min(at);
        }
        acc.into_iter()
            .filter_map(|(url, a)| {
                a.all_fresh.then(|| {
                    (
                        url,
                        FreshLoad {
                            depth: a.depth,
                            pending_prefill_tokens: a
                                .all_report_tokens
                                .then(|| (a.tokens / a.ranks.max(1) as u64) as usize),
                            at: a.oldest_at,
                        },
                    )
                })
            })
            .collect()
    }

    /// Per worker URL, the summed queue depth (`num_running_reqs +
    /// num_waiting_reqs`) across that worker's ranks, for workers whose
    /// every known rank is fresh. Computed once per selection so per-worker
    /// lookups are O(1). See [`Self::fresh_worker_state`] for the freshness
    /// gate behind this.
    pub fn snapshot_fresh(&self, now: Instant) -> HashMap<String, usize> {
        self.fresh_worker_state(now)
            .into_iter()
            .map(|(url, fresh)| (url, fresh.depth))
            .collect()
    }

    /// Drop every rank entry (and the expected mark) for a worker. Called on
    /// worker removal so a re-added worker does not leave stale load behind.
    pub fn forget_worker(&self, url: &str) {
        self.by_rank.retain(|k, _| k.0 != url);
        self.expected.remove(url);
    }

    #[cfg(test)]
    pub fn entry_count(&self) -> usize {
        self.by_rank.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build the msgpack payload the engine's load socket carries:
    /// `["LoadStat", running, waiting, num_tokens, max_total, attn_dp_rank?,
    /// num_waiting_uncached_tokens?]`.
    ///
    /// Byte-level on purpose. Every other test in this file constructs `LoadStat`
    /// in memory, which exercises none of the positional deserializer — and the
    /// positional deserializer is the entire cross-version contract, which the
    /// in-memory `LoadStat` literals every other test builds do not touch.
    fn wire_load_stat(
        counts: [u64; 4],
        attn_dp_rank: Option<Option<u64>>,
        pending: Option<Option<u64>>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        let n = 5 + attn_dp_rank.is_some() as u32 + pending.is_some() as u32;
        rmp::encode::write_array_len(&mut buf, n).unwrap();
        rmp::encode::write_str(&mut buf, "LoadStat").unwrap();
        for c in counts {
            rmp::encode::write_uint(&mut buf, c).unwrap();
        }
        for slot in [attn_dp_rank, pending].into_iter().flatten() {
            match slot {
                Some(v) => {
                    rmp::encode::write_uint(&mut buf, v).unwrap();
                }
                None => rmp::encode::write_nil(&mut buf).unwrap(),
            }
        }
        buf
    }

    /// An engine that predates `num_waiting_uncached_tokens` sends 6 elements.
    /// The field must read as UNKNOWN, not as an empty backlog — `Some(0)` here
    /// would make every old-image worker look like the perfect rotation floor.
    #[test]
    fn wire_six_element_payload_reads_the_token_field_as_unknown() {
        let got = decode_load_stat(&wire_load_stat([7, 3, 0, 0], Some(Some(0)), None))
            .expect("6-element payload decodes");
        assert_eq!((got.num_running_reqs, got.num_waiting_reqs), (7, 3));
        assert_eq!(
            got.pending_prefill_tokens, None,
            "absent field is unknown, never zero",
        );
    }

    /// The positional-skip regression. `attn_dp_rank` is deliberately NONZERO and
    /// the token count deliberately ZERO, so dropping the skip in `Deserialize`
    /// yields `Some(3)` instead of `Some(0)` — a dp rank read as a token backlog.
    /// A rank of 0 (the common case) would make that mutation invisible.
    #[test]
    fn wire_seven_element_payload_reads_the_token_field_not_the_rank() {
        let got = decode_load_stat(&wire_load_stat([1, 0, 0, 0], Some(Some(3)), Some(Some(0))))
            .expect("7-element payload decodes");
        assert_eq!(
            got.pending_prefill_tokens,
            Some(0),
            "must read the token field, not attn_dp_rank",
        );

        let got = decode_load_stat(&wire_load_stat(
            [1, 0, 0, 0],
            Some(Some(3)),
            Some(Some(4096)),
        ))
        .expect("7-element payload decodes");
        assert_eq!(got.pending_prefill_tokens, Some(4096));
    }

    /// `omit_defaults` does not trim trailing defaults on an `array_like` msgspec
    /// struct, so a Python-side `None` arrives as an explicit nil rather than as a
    /// short array. Neither nil may cost us the whole snapshot: losing every count
    /// for a worker is worse than not knowing one field.
    #[test]
    fn wire_explicit_nils_degrade_to_unknown_not_to_a_dropped_frame() {
        let got = decode_load_stat(&wire_load_stat([5, 2, 0, 0], Some(None), Some(Some(900))))
            .expect("nil attn_dp_rank must not reject the frame");
        assert_eq!((got.num_running_reqs, got.num_waiting_reqs), (5, 2));
        assert_eq!(got.pending_prefill_tokens, Some(900));

        let got = decode_load_stat(&wire_load_stat([5, 2, 0, 0], Some(Some(0)), Some(None)))
            .expect("nil token count must not reject the frame");
        assert_eq!(
            got.pending_prefill_tokens, None,
            "an explicit nil is unknown, same as absent",
        );
    }

    /// Fields appended after ours are drained, so a newer engine does not break an
    /// older router.
    #[test]
    fn wire_trailing_unknown_fields_are_ignored() {
        let mut buf = wire_load_stat([1, 2, 0, 0], Some(Some(0)), Some(Some(64)));
        buf[0] = 0x90 | 9; // widen the array header to 9 elements
        rmp::encode::write_uint(&mut buf, 111).unwrap();
        rmp::encode::write_str(&mut buf, "future").unwrap();
        let got = decode_load_stat(&buf).expect("trailing fields are drained");
        assert_eq!(got.pending_prefill_tokens, Some(64));
    }

    fn load(running: u64, waiting: u64) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_tokens: 0,
            max_total_num_tokens: 0,
            pending_prefill_tokens: None,
        }
    }

    #[test]
    fn sums_queue_depth_across_ranks() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.set("http://w:30000", 0, load(5, 1), now);
        t.set("http://w:30000", 1, load(3, 2), now);
        let fresh = t.snapshot_fresh(now);
        // (5+1) + (3+2) = 11
        assert_eq!(fresh.get("http://w:30000").copied(), Some(11));
    }

    #[test]
    fn stale_entries_are_dropped_from_snapshot() {
        let t = EngineLoadTable::with_freshness(Duration::from_millis(10));
        let old = Instant::now();
        t.set("http://w:30000", 0, load(9, 9), old);
        // A read far in the future sees the entry as stale -> worker absent.
        let later = old + Duration::from_secs(60);
        assert!(!t.snapshot_fresh(later).contains_key("http://w:30000"));
    }

    #[test]
    fn forget_worker_clears_all_ranks() {
        let t = EngineLoadTable::new();
        let now = Instant::now();
        t.set("http://w:30000", 0, load(1, 0), now);
        t.set("http://w:30000", 1, load(1, 0), now);
        t.set("http://other:30000", 0, load(1, 0), now);
        t.forget_worker("http://w:30000");
        assert_eq!(t.entry_count(), 1);
        assert!(!t.snapshot_fresh(now).contains_key("http://w:30000"));
        assert!(t.snapshot_fresh(now).contains_key("http://other:30000"));
    }

    /// A worker with any stale rank is omitted entirely (not summed over only
    /// its fresh ranks), so a partially-silent worker falls back to the
    /// router-side counter instead of looking misleadingly idle.
    #[test]
    fn partial_freshness_excludes_worker() {
        let t = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let now = Instant::now();
        let stale = now - Duration::from_secs(3600);
        t.set("http://w:30000", 0, load(5, 1), now); // fresh
        t.set("http://w:30000", 1, load(9, 9), stale); // stale
        assert!(
            !t.snapshot_fresh(now).contains_key("http://w:30000"),
            "any stale rank must drop the whole worker from the snapshot"
        );
    }

    #[test]
    fn fresh_worker_state_picks_the_earliest_rank_timestamp() {
        let t = EngineLoadTable::new();
        let earlier = Instant::now() - Duration::from_secs(2);
        let later = earlier + Duration::from_secs(1);
        t.set("http://w:30000", 0, load(5, 1), later);
        t.set("http://w:30000", 1, load(3, 2), earlier);
        let now = later + Duration::from_millis(1);
        let fused = t
            .fresh_worker_state(now)
            .get("http://w:30000")
            .copied()
            .expect("worker present");
        assert_eq!(fused.depth, 11, "depth sums across ranks");
        assert_eq!(
            fused.at, earlier,
            "must expose the OLDEST rank's timestamp, not the newest"
        );
    }

    /// The freshness cutoff is inclusive, and the default window is 5s.
    ///
    /// Pinned because staleness costs more than a missing depth: a worker dropped
    /// from this pass loses its token backlog too, so it silently stops being
    /// eligible as a rotation destination. Widening the window would keep routing
    /// on numbers the engine has stopped confirming; narrowing it turns rotation
    /// off under normal publish jitter. Neither is visible without a boundary
    /// assertion.
    #[test]
    fn freshness_cutoff_is_inclusive_at_the_default_window() {
        assert_eq!(
            DEFAULT_FRESHNESS,
            Duration::from_secs(5),
            "the default window is a routing input; changing it is not a refactor",
        );
        let t = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let published = Instant::now();
        t.set("http://w:30000", 0, load(1, 0), published);

        // Exactly at the window: still fresh (the comparison is `<=`).
        assert!(
            t.fresh_worker_state(published + Duration::from_secs(5))
                .contains_key("http://w:30000"),
            "a snapshot exactly at the cutoff must count as fresh",
        );
        // One tick past: gone, and gone entirely — not merely token-less.
        assert!(
            t.fresh_worker_state(published + Duration::from_secs(5) + Duration::from_nanos(1))
                .is_empty(),
            "one tick past the cutoff drops the worker from the pass",
        );
    }

    /// The token backlog is the MEAN over ranks, not the sum.
    ///
    /// A dispatch to a dp-attention URL lands on one of its `D` schedulers, so the
    /// wait it pays is one rank's backlog. Summing would inflate both rotation
    /// operands by `D` while the matched prefix they are compared against stayed
    /// fixed — rotation firing ~`D`× too eagerly on every dp fleet.
    #[test]
    fn pending_prefill_tokens_is_the_mean_over_ranks() {
        let now = Instant::now();
        let with_tokens = |tokens: u64| LoadStat {
            num_running_reqs: 1,
            num_waiting_reqs: 0,
            num_tokens: 0,
            max_total_num_tokens: 0,
            pending_prefill_tokens: Some(tokens),
        };
        let t = EngineLoadTable::new();
        t.set("http://w:30000", 0, with_tokens(400), now);
        t.set("http://w:30000", 1, with_tokens(600), now);
        let fused = t.fresh_worker_state(now);
        let f = fused.get("http://w:30000").expect("worker present");
        assert_eq!(
            f.pending_prefill_tokens,
            Some(500),
            "mean over the two ranks, NOT the 1000-token sum",
        );
        // Depth keeps summing — it is only ever compared against other workers.
        assert_eq!(f.depth, 2, "depth stays a sum across ranks");

        // Single-rank workers are unaffected: mean of one is that one.
        let single = EngineLoadTable::new();
        single.set("http://s:30000", 0, with_tokens(777), now);
        assert_eq!(
            single
                .fresh_worker_state(now)
                .get("http://s:30000")
                .unwrap()
                .pending_prefill_tokens,
            Some(777),
        );
    }

    /// A worker's token backlog is usable only when EVERY rank reports one.
    /// Summing a subset would understate the backlog, and understating it is what
    /// makes a busy worker look like the idle floor.
    #[test]
    fn pending_prefill_tokens_needs_every_rank_to_report() {
        let now = Instant::now();
        let with_tokens = |running: u64, tokens: Option<u64>| LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: 0,
            num_tokens: 0,
            max_total_num_tokens: 0,
            pending_prefill_tokens: tokens,
        };

        let t = EngineLoadTable::new();
        t.set("http://w:30000", 0, with_tokens(1, Some(400)), now);
        t.set("http://w:30000", 1, with_tokens(1, Some(600)), now);
        assert_eq!(
            t.fresh_worker_state(now)
                .get("http://w:30000")
                .unwrap()
                .pending_prefill_tokens,
            Some(500),
            "both ranks report -> fused (see pending_prefill_tokens_is_the_mean_over_ranks)",
        );

        let mixed = EngineLoadTable::new();
        mixed.set("http://w:30000", 0, with_tokens(1, Some(400)), now);
        mixed.set("http://w:30000", 1, with_tokens(1, None), now);
        assert_eq!(
            mixed
                .fresh_worker_state(now)
                .get("http://w:30000")
                .unwrap()
                .pending_prefill_tokens,
            None,
            "one silent rank -> the whole worker is unknown, NOT the partial sum",
        );
    }

    #[test]
    fn fresh_worker_state_agrees_with_snapshot_fresh_on_which_workers_are_present() {
        let t = EngineLoadTable::with_freshness(Duration::from_secs(5));
        let now = Instant::now();
        let stale = now - Duration::from_secs(3600);
        t.set("http://fresh:30000", 0, load(1, 0), now);
        t.set("http://mixed:30000", 0, load(1, 0), now);
        t.set("http://mixed:30000", 1, load(1, 0), stale);

        let depths = t.snapshot_fresh(now);
        let state = t.fresh_worker_state(now);
        assert!(depths.contains_key("http://fresh:30000"));
        assert!(state.contains_key("http://fresh:30000"));
        assert!(!depths.contains_key("http://mixed:30000"));
        assert!(!state.contains_key("http://mixed:30000"));
    }

    #[test]
    fn expected_count_tracks_marked_workers_and_forget() {
        let t = EngineLoadTable::new();
        assert_eq!(t.expected_count(), 0);
        t.mark_expected("http://w:30000");
        t.mark_expected("http://w:30000"); // idempotent
        t.mark_expected("http://other:30000");
        assert_eq!(t.expected_count(), 2);
        t.forget_worker("http://w:30000");
        assert_eq!(t.expected_count(), 1);
    }
}
