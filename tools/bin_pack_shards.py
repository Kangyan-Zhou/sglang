#!/usr/bin/env python3
"""Phase-3 manifest builder: bin-pack Bazel test targets into shards.

Reads `bazel cquery --output=jsonproto ...` output (a list of targets with
their tags), groups by suite tag, and bin-packs each group into N shards
of ~equal estimated duration using the longest-processing-time-first (LPT)
heuristic. Writes a JSON manifest keyed by suite name; each value is a list
of `{"id": int, "targets": str}` consumable as a GitHub Actions matrix.

This replaces `python/sglang/test/ci/ci_register.py:auto_partition` once
Stage 3 is live. The algorithm is the same — sort by est_time desc,
greedily place each test on the currently-shortest shard — just operating
on Bazel's view of the world (test targets) instead of run_suite.py's
test paths.

Usage
-----
    bazel cquery 'kind(py_test, //test/...)' --output=jsonproto > targets.jsonproto
    python3 tools/bin_pack_shards.py targets.jsonproto \
        --shards stage-b-test-1-gpu-small=8,stage-b-test-1-gpu-large=8 \
        > manifest.json

Stdin / stdout convention: targets.jsonproto on argv, manifest.json on
stdout (so the GHA `echo "shards=$(jq -c . manifest.json)" >> $GITHUB_OUTPUT`
pattern works without a temp file).

Output schema
-------------
    {
      "stage-b-test-1-gpu-small": [
        {"id": 0, "targets": "//test/A:foo //test/B:bar"},
        {"id": 1, "targets": "//test/C:baz"},
        ...
      ],
      "stage-b-test-1-gpu-large": [...],
      ...
    }

Suites with no matching targets are omitted entirely so `if:` guards on
shard jobs can skip cleanly.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SUITE_TAG_PREFIX = "sgl-suite-"
_EST_TIME_TAG_PREFIX = "est_time:"


@dataclass
class _Target:
    label: str
    suite: str
    est_time: int


@dataclass(order=True)
class _Shard:
    """Heap-ordered by total est_time so we always place into the
    currently-shortest shard. `id` breaks ties deterministically."""

    total: int = 0
    id: int = 0
    targets: list[str] = field(default_factory=list, compare=False)


def _parse_shards_arg(arg: str) -> dict[str, int]:
    """Parse `name=K[,name=K...]` into {suite: shard_count}."""
    out: dict[str, int] = {}
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            sys.exit(f"--shards entry {piece!r} missing '='")
        name, _, count = piece.partition("=")
        try:
            n = int(count)
        except ValueError:
            sys.exit(f"--shards entry {piece!r}: count must be an int")
        if n < 1:
            sys.exit(f"--shards entry {piece!r}: count must be >= 1")
        out[name.strip()] = n
    return out


def _extract_targets(jsonproto_path: Path) -> list[_Target]:
    """Pull (label, suite, est_time) tuples from a `bazel cquery
    --output=jsonproto` payload. Targets without both `sgl-suite-*` and
    `est_time:*` tags are silently skipped (they're not part of any
    runnable suite under our convention)."""
    payload = json.loads(jsonproto_path.read_text())
    out: list[_Target] = []
    for target in payload.get("results", []):
        rule = target.get("target", {}).get("rule", {})
        label = rule.get("name", "")
        if not label:
            continue
        # rule.attribute is a list of attr dicts; tags live in the one
        # whose name == "tags". Bazel's jsonproto output uses
        # string_list_value for repeated string attrs.
        tags: list[str] = []
        for attr in rule.get("attribute", []):
            if attr.get("name") == "tags":
                tags = list(attr.get("stringListValue", []))
                break
        suite = ""
        est_time = -1
        for tag in tags:
            if tag.startswith(_SUITE_TAG_PREFIX):
                suite = tag[len(_SUITE_TAG_PREFIX) :]
            elif tag.startswith(_EST_TIME_TAG_PREFIX):
                try:
                    est_time = int(tag[len(_EST_TIME_TAG_PREFIX) :])
                except ValueError:
                    sys.stderr.write(f"WARN: {label} has malformed {tag!r}; skipping\n")
                    est_time = -1
        if not suite or est_time < 0:
            continue
        out.append(_Target(label=label, suite=suite, est_time=est_time))
    return out


def _bin_pack(targets: list[_Target], shard_count: int) -> list[_Shard]:
    """Greedy LPT bin-packing: sort tests by est_time desc, push each
    onto the shard with the lowest running total. Same algorithm as
    sglang's existing run_suite.py auto_partition."""
    shards = [_Shard(id=i) for i in range(shard_count)]
    heapq.heapify(shards)
    for t in sorted(targets, key=lambda x: x.est_time, reverse=True):
        shortest = heapq.heappop(shards)
        shortest.total += t.est_time
        shortest.targets.append(t.label)
        heapq.heappush(shards, shortest)
    # Restore deterministic id order; matrix consumers want shard 0 to
    # exist regardless of which tests landed in it.
    return sorted(shards, key=lambda s: s.id)


def _build_manifest(
    targets: list[_Target], shard_counts: dict[str, int], errors: list[str]
) -> dict[str, list[dict]]:
    by_suite: dict[str, list[_Target]] = {}
    for t in targets:
        by_suite.setdefault(t.suite, []).append(t)

    # Suites with targets but no shard config are skipped with a stderr
    # warning — caller controls which suites to emit shards for, and
    # not every suite the codebase declares maps to a runnable CI suite
    # in this run. (E.g. nightlies on a per-commit run.)
    for suite, items in by_suite.items():
        if suite not in shard_counts:
            sys.stderr.write(
                f"INFO: suite {suite!r} has {len(items)} target(s) but no "
                f"--shards entry; skipped (add to --shards to include)\n"
            )

    # Configured shards with zero matching targets ARE an error — the
    # caller asked for them and got nothing. Likely a typo or stale tag.
    for suite in shard_counts:
        if suite not in by_suite:
            errors.append(
                f"--shards configured {suite!r}=N but no targets carry "
                f"the sgl-suite-{suite} tag (typo? stale config?)"
            )

    manifest: dict[str, list[dict]] = {}
    for suite, items in by_suite.items():
        if suite not in shard_counts:
            continue
        shards = _bin_pack(items, shard_counts[suite])
        manifest[suite] = [
            {"id": s.id, "targets": " ".join(s.targets), "est_time": s.total}
            for s in shards
            if s.targets  # drop empty shards (shard_count > len(targets))
        ]
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "jsonproto",
        type=Path,
        help="Path to `bazel cquery --output=jsonproto ...` output",
    )
    p.add_argument(
        "--shards",
        required=True,
        help="Comma-separated suite=count pairs, e.g. "
        "stage-b-test-1-gpu-small=8,stage-b-test-1-gpu-large=8",
    )
    args = p.parse_args()

    shard_counts = _parse_shards_arg(args.shards)
    targets = _extract_targets(args.jsonproto)

    errors: list[str] = []
    manifest = _build_manifest(targets, shard_counts, errors)

    if errors:
        sys.stderr.write(f"ERROR: {len(errors)} bin-pack issue(s):\n")
        for err in errors:
            sys.stderr.write(f"  {err}\n")
        return 1

    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
