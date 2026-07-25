"""Per-scheduler load reporting for load-aware routing.

Independent of KV-cache events: each scheduler publishes a periodic
[`LoadStat`] gauge on its own ZMQ PUB socket (a dedicated port range,
distinct from the KV-event publisher's) so load-aware routers — e.g. the
experimental sgl-router `cache_aware_zmq` policy — can route on the
engine's true queue depth / KV occupancy instead of inferring load from a
router-side in-flight counter.

The only thing borrowed is the generic ZMQ PUB transport
(`ZmqEventPublisher`) from `sglang.srt.utils.event_publisher`; the load wire
format and publishing cadence live here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec

from sglang.srt.utils.event_publisher import (
    KVEventsConfig,
    NullEventPublisher,
    ZmqEventPublisher,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState

logger = logging.getLogger(__name__)

# ZMQ topic the load publisher tags its frames with. The load socket carries
# only load, so subscribers can subscribe-all; the topic is cosmetic/self-
# documenting.
LOAD_TOPIC = "load"

# Publish a load snapshot at most once every this many `publish_load_stat`
# calls, unless `force=True` (extend/prefill batches, where load changes
# most). Load is a gauge consumed for routing, so per-decode-step publishing
# is wasteful.
LOAD_PUBLISH_INTERVAL = 5

# Re-warn about publish failures every this many consecutive failures, so a
# permanent failure (e.g. a renamed field) keeps a live breadcrumb instead of
# going silent after the first warning, without flooding the log.
LOAD_PUBLISH_FAIL_WARN_EVERY = 60


def _policy_discounts_uncached_tokens(
    schedule_policy: str, *, disable_radix_cache: bool
) -> bool:
    """Pure half of `_uncached_tokens_are_discounted`, split out so the rule can be
    tested without standing up a `ServerArgs`.

    Both conditions are necessary, and the second is the one that fails OPEN if
    forgotten: `SchedulePolicy._validate_and_adjust_policy` rewrites a cache-aware
    policy to `CacheAgnosticPolicy.FCFS` whenever the tree cache is disabled, so
    `--schedule-policy lpm --disable-radix-cache` requests a cache-aware policy and
    runs a cache-agnostic one.
    """
    from sglang.srt.managers.schedule_policy import CacheAwarePolicy

    if disable_radix_cache:
        return False
    return schedule_policy in {p.value for p in CacheAwarePolicy}


def _uncached_tokens_are_discounted() -> bool:
    """Whether this engine subtracts its radix match from the queued prefill it
    reports, i.e. whether `num_waiting_uncached_tokens` means "uncached" rather
    than "raw queued sequence length".

    `Req.num_matched_prefix_tokens` — the term
    `SchedulerLoadInquirer.get_num_waiting_uncached_tokens` subtracts — is
    populated on two paths, and NEITHER is the default:

    * a cache-aware `--schedule-policy` (`CacheAwarePolicy`) with the radix cache
      enabled, via `SchedulePolicy._compute_prefix_matches`; or
    * a cache-agnostic one whose `tree_cache.supports_fast_match_prefix()` is
      True — which no cache implements today
      (`BasePrefixCache.supports_fast_match_prefix` returns False and is never
      overridden), so this path cannot currently be taken. If a cache ever
      implements it, widen the predicate to match, or the field goes on the wire
      as None when it did not need to.

    Under the default `--schedule-policy fcfs` the term stays 0, so the count is
    undiscounted. Publishing it anyway would hand a consumer a number in the wrong
    unit with no way to tell — so we publish None and let the consumer decline.
    Undiscounted counts remain on `/v1/loads`, where the field is a plain gauge
    rather than an operand.

    KNOWN RESIDUAL, not closable from startup state: this answers "is the discount
    configured", not "was it computed for the queue being reported".
    `SchedulePolicy._determine_active_policy` degrades `lpm` to FCFS for any pass
    where the waiting queue exceeds 128, and `calc_priority` — the only caller that
    populates the term — is skipped entirely when the running batch is full. In
    both cases newly queued requests keep the term at 0 while this predicate still
    says "discounted", so a saturated worker over-reports. Closing it needs a
    per-request "match was computed" flag threaded to
    `get_num_waiting_uncached_tokens`; until then treat a saturated `lpm` worker's
    count as an upper bound.
    """
    try:
        # Imported lazily: this module is imported from scheduler startup and this
        # pulls in a heavier config graph.
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        # `disable_radix_cache` is keyword-only: this is the sole site where a
        # policy string and a bool are ordered together, and transposing them
        # silently withholds the field forever (a non-empty policy string is
        # truthy, so the disable check would always fire).
        return _policy_discounts_uncached_tokens(
            args.schedule_policy, disable_radix_cache=args.disable_radix_cache
        )
    except Exception:
        # Never block scheduler startup on this; withholding is the safe direction
        # (consumers decline rather than mis-unit).
        logger.warning(
            "load-publisher: could not determine whether queued prefill is "
            "radix-discounted; withholding num_waiting_uncached_tokens",
            exc_info=True,
        )
        return False


class LoadStat(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,  # type: ignore[call-arg]
):
    """Per-scheduler runtime load snapshot.

    Wire shape (tag + array_like): ``["LoadStat", num_running_reqs,
    num_waiting_reqs, num_tokens, max_total_num_tokens, attn_dp_rank,
    num_waiting_uncached_tokens?]``. A router reads the prefix it knows and
    ignores the rest, so adding a field is compatible in both directions.
    `attn_dp_rank` exists so the snapshot can be published directly through
    `ZmqEventPublisher.publish` (which stamps it — see `event_publisher`, so it
    is always present on the wire); the router keys load by the subscriber's
    socket rank, not this field.

    WHY `num_waiting_uncached_tokens` is appended AFTER `attn_dp_rank` rather
    than beside the other counts: this is a positional encoding, so a field
    inserted mid-struct would be read as `attn_dp_rank` by any router built
    before it. Appended last, an older payload simply ends early and a newer
    router sees the field as absent — distinguishable from a genuine zero, which
    matters because the router only makes token-based routing decisions when it
    knows the engine actually reports this.
    """

    num_running_reqs: int
    num_waiting_reqs: int
    # KV tokens currently in use, from the engine's KV pool.
    num_tokens: int
    # KV-cache token capacity; 0 when unknown.
    max_total_num_tokens: int
    attn_dp_rank: Optional[int] = None
    # Input tokens queued for prefill compute — the waiting queue's
    # `seqlen - num_matched_prefix_tokens` plus the chunked request's remainder.
    #
    # WARNING, load-bearing for any consumer doing arithmetic against a prefix
    # match: whether this is NET of the engine's own radix match depends on
    # engine flags. `num_matched_prefix_tokens` is populated only by a
    # cache-aware `--schedule-policy` (lpm, dfs-weight), or by a cache-agnostic
    # one whose tree_cache implements `supports_fast_match_prefix()` — which no
    # cache does today (`BasePrefixCache.supports_fast_match_prefix` returns
    # False and is never overridden). Under the DEFAULT `--schedule-policy fcfs`
    # it is 0, so this field is the raw queued sequence length. See the comment
    # in `SchedulerLoadInquirer.get_num_waiting_uncached_tokens`.
    #
    # Optional so an omitted value stays distinguishable from zero.
    #
    # Caveat for consumers: a DECODE-role scheduler in PD disaggregation reports
    # a structural 0 here (see `SchedulerLoadInquirer`), not a measurement — it
    # does no prefill. So `0` means "owes no prefill compute", which on a decode
    # scheduler is permanently true. Anything comparing this ACROSS roles has to
    # account for that; comparing within one role is well defined.
    num_waiting_uncached_tokens: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class SchedulerLoadPublisher:
    """Owns one scheduler's dedicated load PUB socket and the throttled,
    best-effort `publish_load_stat` path.

    Enabled on the same condition as KV-event publishing (a `kv_events_config`
    on the attn-TP/CP-rank-0 scheduler), and binds the load port range packed
    immediately after the KV-event range (`kv_base + dp_size`). Stays a no-op
    (a `NullEventPublisher`) when disabled or when the KV config has no usable
    ZMQ endpoint.
    """

    kv_events_config: Optional[str]
    ps: ParallelState
    # Number of attention-DP ranks (= the KV port range width); the load port
    # range starts at kv_base + dp_size.
    dp_size: int
    enable: bool = False
    publisher: Any = None
    _publish_counter: int = 0
    # Consecutive publish failures, reset on success (drives the periodic warn).
    _fail_count: int = 0
    # Whether this engine's `num_waiting_uncached_tokens` is genuinely net of the
    # radix match; see `_uncached_tokens_are_discounted`. When False the field is
    # published as None rather than as an undiscounted count, so a consumer that
    # needs the discounted quantity declines to act instead of silently doing
    # arithmetic in the wrong unit.
    _report_uncached_tokens: bool = False

    def __post_init__(self) -> None:
        self.publisher = NullEventPublisher()
        self.enable = bool(
            self.kv_events_config
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        if not self.enable:
            return
        # Resolved only once publishing is known to be on. Above the enable check
        # it warns with a traceback on every scheduler whose backend never sets the
        # global `ServerArgs` (the MLX runner, unit tests) even though nothing will
        # be published.
        self._report_uncached_tokens = _uncached_tokens_are_discounted()
        try:
            cfg = KVEventsConfig.from_cli(self.kv_events_config)
        except Exception:
            # Malformed config — the KV publisher init would have failed too;
            # stay a no-op rather than raising at scheduler startup.
            return
        if cfg.publisher == "null" or not cfg.endpoint:
            return
        load_endpoint = ZmqEventPublisher.offset_endpoint_port(
            cfg.endpoint, self.dp_size
        )
        if load_endpoint is None:
            return
        # Dedicated load socket: own port, replay disabled, unbuffered (load is
        # a gauge, not a replayable delta).
        self.publisher = ZmqEventPublisher(
            self.ps.attn_dp_rank,
            endpoint=load_endpoint,
            replay_endpoint=None,
            buffer_steps=0,
            topic=LOAD_TOPIC,
        )

    def publish_load_stat(self, load_provider: Callable, force: bool = False) -> None:
        """Publish a load snapshot, throttled to [`LOAD_PUBLISH_INTERVAL`]
        calls unless `force`.

        `load_provider` returns a live load snapshot (a `GetLoadsReqOutput`)
        read directly from scheduler state — used instead of metrics stats,
        whose values are only populated under `--enable-metrics`. Invoked only
        after the throttle passes, so the snapshot is computed only when
        actually publishing.

        Best-effort: a failure here must never crash the scheduler loop —
        routers fall back to their own in-flight counter. Failures re-warn
        every [`LOAD_PUBLISH_FAIL_WARN_EVERY`] consecutive failures.
        """
        if not self.enable:
            return

        self._publish_counter += 1
        if not force and self._publish_counter < LOAD_PUBLISH_INTERVAL:
            return
        self._publish_counter = 0

        try:
            load = load_provider()
            self.publisher.publish(
                LoadStat(
                    num_running_reqs=load.num_running_reqs,
                    num_waiting_reqs=load.num_waiting_reqs,
                    num_tokens=load.num_used_tokens,
                    max_total_num_tokens=load.max_total_num_tokens,
                    num_waiting_uncached_tokens=(
                        load.num_waiting_uncached_tokens
                        if self._report_uncached_tokens
                        else None
                    ),
                )
            )
            self._fail_count = 0
        except Exception:
            if self._fail_count % LOAD_PUBLISH_FAIL_WARN_EVERY == 0:
                logger.warning(
                    "load-publisher: publish_load_stat failed (%d consecutive); "
                    "load-aware routers fall back to their in-flight load signal",
                    self._fail_count + 1,
                    exc_info=True,
                )
            self._fail_count += 1
