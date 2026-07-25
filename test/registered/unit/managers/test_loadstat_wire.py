"""Wire-contract test for the LoadStat load-snapshot event.

Locks the msgpack array shape the Rust router decoder depends on
(experimental/sgl-router/src/policies/engine_load.rs, `LoadStat`):

    ["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
     max_total_num_tokens, attn_dp_rank, num_waiting_uncached_tokens]

The encoding is POSITIONAL, so fields are only ever APPENDED — inserting one
mid-struct would be read as the field after it by any router built before the
change. The Rust side hand-encodes these bytes in its own tests
(`engine_load.rs`, the `wire_*` tests); this verifies the Python publisher
actually *emits* that shape, so the two sides can't drift.

Note `omit_defaults=True` does NOT trim trailing defaults under
`array_like=True` — it applies only to the dict-like encoding — so every
snapshot carries all declared fields and a `None` goes on the wire as an explicit
msgpack nil. Cross-version compatibility comes from an older engine's struct
having FEWER FIELDS, not from trimming. `test_none_is_emitted_not_omitted`
pins that, because the Rust decoder's tolerance of nil depends on it.

CPU-only: just exercises the msgspec encoding.
"""

import types
import unittest

import msgspec.msgpack

from sglang.srt.managers.scheduler_components.load_publisher import (
    LoadStat,
    SchedulerLoadPublisher,
    _policy_discounts_uncached_tokens,
    _uncached_tokens_are_discounted,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLoadStatWire(CustomTestCase):
    def test_loadstat_msgpack_array_shape(self):
        # `attn_dp_rank` is stamped by ZmqEventPublisher.publish in production;
        # set it here to assert the full on-the-wire shape.
        stat = LoadStat(
            num_running_reqs=7,
            num_waiting_reqs=3,
            num_tokens=1024,
            max_total_num_tokens=8192,
            attn_dp_rank=2,
            num_waiting_uncached_tokens=4096,
        )
        # Same encoder the publisher thread uses (msgspec.msgpack.Encoder).
        raw = msgspec.msgpack.Encoder().encode(stat)
        decoded = msgspec.msgpack.Decoder().decode(raw)

        # tag=True + array_like → [tag, *fields] in declaration order.
        self.assertEqual(
            decoded,
            ["LoadStat", 7, 3, 1024, 8192, 2, 4096],
            "LoadStat wire shape must match the Rust decoder's expectation",
        )

    def test_uncached_tokens_follow_attn_dp_rank(self):
        # The position is the compatibility contract: a router built before this
        # field reads slot 5 as attn_dp_rank and drains the rest. If the two ever
        # swap, every such router reads a token backlog as a dp rank and vice
        # versa. Distinct values so a swap can't hide.
        decoded = msgspec.msgpack.Decoder().decode(
            msgspec.msgpack.Encoder().encode(
                LoadStat(1, 2, 3, 4, attn_dp_rank=5, num_waiting_uncached_tokens=6)
            )
        )
        self.assertEqual(decoded[5], 5, "slot 5 must remain attn_dp_rank")
        self.assertEqual(
            decoded[6], 6, "num_waiting_uncached_tokens must be appended AFTER the rank"
        )

    def test_none_is_emitted_not_omitted(self):
        # omit_defaults does not trim under array_like. A withheld token count is
        # an explicit nil in slot 6, never a shorter array — the Rust decoder
        # folds nil and absent into the same "unknown", and must keep doing so.
        decoded = msgspec.msgpack.Decoder().decode(
            msgspec.msgpack.Encoder().encode(
                LoadStat(1, 2, 3, 4, attn_dp_rank=0, num_waiting_uncached_tokens=None)
            )
        )
        self.assertEqual(len(decoded), 7, "trailing None is emitted, not trimmed")
        self.assertIsNone(decoded[6])

    def test_loadstat_tag_is_class_name(self):
        # The Rust decoder matches the literal tag string "LoadStat"; guard
        # against an accidental msgspec `tag=` override or class rename.
        raw = msgspec.msgpack.Encoder().encode(LoadStat(0, 0, 0, 0))
        decoded = msgspec.msgpack.Decoder().decode(raw)
        self.assertEqual(decoded[0], "LoadStat")


class TestUncachedTokenDiscountGate(CustomTestCase):
    """`num_waiting_uncached_tokens` is only meaningful when the engine actually
    subtracts its radix match. Under the default `--schedule-policy fcfs` it does
    not, and the field is the raw queued sequence length — so the publisher
    withholds it rather than hand a consumer a number in the wrong unit.
    """

    def test_cache_agnostic_policies_withhold_the_field(self):
        # fcfs is the DEFAULT schedule policy, so this is the common deployment.
        for policy in ("fcfs", "lof", "random", "routing-key"):
            with self.subTest(policy=policy):
                self.assertFalse(
                    _policy_discounts_uncached_tokens(
                        policy, disable_radix_cache=False
                    ),
                    f"{policy} does not populate num_matched_prefix_tokens, so the "
                    "count is undiscounted and must not be published",
                )

    def test_cache_aware_policies_publish_the_field(self):
        for policy in ("lpm", "dfs-weight"):
            with self.subTest(policy=policy):
                self.assertTrue(
                    _policy_discounts_uncached_tokens(
                        policy, disable_radix_cache=False
                    ),
                    f"{policy} populates num_matched_prefix_tokens via "
                    "_compute_prefix_matches, so the count is genuinely uncached",
                )

    def test_disabled_radix_cache_withholds_even_under_a_cache_aware_policy(self):
        # `_validate_and_adjust_policy` rewrites a cache-aware policy to FCFS when
        # the tree cache is disabled, so the REQUESTED policy is not the effective
        # one. Missing this check fails OPEN: raw sequence lengths published under a
        # discounted label.
        for policy in ("lpm", "dfs-weight"):
            with self.subTest(policy=policy):
                self.assertFalse(
                    _policy_discounts_uncached_tokens(policy, disable_radix_cache=True),
                    f"--schedule-policy {policy} --disable-radix-cache runs FCFS, "
                    "so nothing computes the discount",
                )

    def test_every_cli_schedule_policy_is_classified(self):
        from sglang.srt.managers.schedule_policy import (
            CacheAgnosticPolicy,
            CacheAwarePolicy,
        )

        aware = {p.value for p in CacheAwarePolicy}
        agnostic = {p.value for p in CacheAgnosticPolicy}
        self.assertEqual(aware & agnostic, set(), "a policy cannot be both")

        # A MANUAL MIRROR of the `--schedule-policy` choices at
        # `ServerArgs.schedule_policy`, not derived from them — the choices live in
        # `Annotated` metadata that isn't cheaply readable from here. So this
        # catches a policy being removed from an enum, but NOT a new CLI choice
        # added without an enum home. Update it alongside that flag.
        #
        # `priority` is deliberately in neither enum: `_validate_and_adjust_policy`
        # raises on it, so it can never reach a running scheduler.
        self.assertEqual(
            {"lpm", "dfs-weight", "fcfs", "lof", "random", "routing-key", "priority"}
            - (aware | agnostic),
            {"priority"},
            "a --schedule-policy choice needs an enum home, or the load publisher "
            "withholds the prefill backlog for it forever",
        )
        for policy in aware | agnostic:
            self.assertEqual(
                _policy_discounts_uncached_tokens(policy, disable_radix_cache=False),
                policy in aware,
                f"{policy} must be gated by its enum membership",
            )


class _CapturingPublisher:
    def __init__(self):
        self.published = []

    def publish(self, stat):
        self.published.append(stat)


class TestWithheldFieldReachesTheWire(CustomTestCase):
    """The gate is only worth anything if it reaches the payload. Asserting the
    predicate alone leaves an unconditional publish passing every test.
    """

    def _publish_once(self, *, report: bool) -> LoadStat:
        # `kv_events_config=None` makes __post_init__ bail early (no ZMQ), then we
        # substitute a capturing publisher and drive one forced publish.
        pub = SchedulerLoadPublisher(
            kv_events_config=None,
            ps=types.SimpleNamespace(attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0),
            dp_size=1,
        )
        pub.enable = True
        pub.publisher = _CapturingPublisher()
        pub._report_uncached_tokens = report
        load = types.SimpleNamespace(
            num_running_reqs=2,
            num_waiting_reqs=1,
            num_used_tokens=64,
            max_total_num_tokens=1024,
            num_waiting_uncached_tokens=4096,
        )
        pub.publish_load_stat(lambda: load, force=True)
        self.assertEqual(len(pub.publisher.published), 1, "exactly one snapshot")
        return pub.publisher.published[0]

    def test_undiscounted_engine_publishes_none(self):
        stat = self._publish_once(report=False)
        self.assertIsNone(
            stat.num_waiting_uncached_tokens,
            "an undiscounted count must be withheld, not published in the wrong unit",
        )
        # The other counts still flow — withholding one field is not a blackout.
        self.assertEqual(
            (stat.num_running_reqs, stat.num_waiting_reqs, stat.num_tokens),
            (2, 1, 64),
        )

    def test_discounted_engine_publishes_the_count(self):
        stat = self._publish_once(report=True)
        self.assertEqual(stat.num_waiting_uncached_tokens, 4096)

    def test_field_default_is_to_withhold(self):
        # Fail-closed by construction. Note this asserts the FIELD DEFAULT, not the
        # resolved gate: with publishing disabled `__post_init__` returns before
        # resolving anything, so the flag must already be safe. (An earlier version
        # of this test claimed to prove "fcfs does not discount" — it did not; in a
        # unit-test process the global ServerArgs are unset, so it was passing via
        # the except path. `test_unresolvable_policy_withholds` covers that, and
        # `test_cache_agnostic_policies_withhold_the_field` covers fcfs.)
        pub = SchedulerLoadPublisher(
            kv_events_config=None,
            ps=types.SimpleNamespace(attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0),
            dp_size=1,
        )
        self.assertFalse(
            pub._report_uncached_tokens,
            "a publisher that never resolved the policy must not publish the field",
        )

    def test_resolved_cache_aware_policy_turns_the_field_on(self):
        # The branch that turns the feature ON. Without it every other test here is
        # satisfied by a publisher that withholds unconditionally — rotation would be
        # dead on every fleet, including `lpm`, with the whole suite green.
        import sglang.srt.server_args as server_args

        original = server_args.get_global_server_args
        server_args.get_global_server_args = lambda: types.SimpleNamespace(
            schedule_policy="lpm", disable_radix_cache=False
        )
        try:
            self.assertTrue(
                _uncached_tokens_are_discounted(),
                "lpm with the radix cache enabled must publish the backlog",
            )
        finally:
            server_args.get_global_server_args = original

    def test_unresolvable_policy_withholds(self):
        # The except path exists so a startup-order surprise can't turn into a
        # published number in the wrong unit. Force it by breaking the lookup.
        import sglang.srt.server_args as server_args

        original = server_args.get_global_server_args

        def boom():
            raise RuntimeError("global server args not set yet")

        server_args.get_global_server_args = boom
        try:
            self.assertFalse(
                _uncached_tokens_are_discounted(),
                "an unresolvable policy must withhold, never publish",
            )
        finally:
            server_args.get_global_server_args = original


if __name__ == "__main__":
    unittest.main()
