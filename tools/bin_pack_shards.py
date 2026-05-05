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


def _extract_targets(jsonproto_path: Path, errors: list[str]) -> list[_Target]:
    """Pull (label, suite, est_time) tuples from a `bazel cquery
    --output=jsonproto` payload.

    Failure modes we treat as hard errors (vs silent skip):
    - Suite tag present but est_time missing or malformed. The target
      would be reachable via `bazel test --test_tag_filters=sgl-suite-*`
      but invisible to the manifest — the worst kind of "looks-wired,
      isn't-wired" failure.
    - The `tags` attribute exists but isn't `stringListValue`-shaped
      (Bazel jsonproto schema change worth failing loudly on).
    - Top-level payload is missing the `results` key (truncated cquery
      output, error envelope, etc.).

    Targets that lack a suite tag entirely are legitimately skipped —
    they're not claimed by any CI run.
    """
    try:
        raw = jsonproto_path.read_text()
    except OSError as e:
        sys.exit(f"ERROR: cannot read {jsonproto_path}: {e}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(
            f"ERROR: {jsonproto_path} is not valid JSON ({e}); "
            f"size={len(raw)} bytes. Did `bazel cquery` fail mid-write?"
        )

    if "results" not in payload:
        sys.exit(
            f"ERROR: {jsonproto_path} has no 'results' key. Bazel jsonproto "
            f'for zero matching targets is `{{"results": []}}`, never `{{}}`. '
            f"Probable truncated cquery output; size={len(raw)} bytes."
        )

    out: list[_Target] = []
    for target in payload["results"]:
        rule = target.get("target", {}).get("rule", {})
        label = rule.get("name", "")
        if not label:
            continue
        # rule.attribute is a list of attr dicts; tags live in the one
        # whose name == "tags". Bazel's jsonproto output uses
        # `stringListValue` (camelCase) per protobuf-to-JSON canonical
        # mapping for the `string_list_value` proto field.
        tags: list[str] | None = None
        for attr in rule.get("attribute", []):
            if attr.get("name") == "tags":
                if "stringListValue" not in attr:
                    errors.append(
                        f"{label}: 'tags' attribute is not stringListValue-"
                        f"shaped; Bazel jsonproto schema may have changed"
                    )
                    break
                tags = list(attr["stringListValue"])
                break
        if tags is None:
            continue  # no tags attr; legitimately not in any CI suite

        suite = ""
        est_time: int | None = None
        for tag in tags:
            if tag.startswith(_SUITE_TAG_PREFIX):
                suite = tag[len(_SUITE_TAG_PREFIX) :]
            elif tag.startswith(_EST_TIME_TAG_PREFIX):
                try:
                    est_time = int(tag[len(_EST_TIME_TAG_PREFIX) :])
                except ValueError:
                    errors.append(
                        f"{label}: malformed {tag!r} (expected "
                        f"{_EST_TIME_TAG_PREFIX}<int>)"
                    )

        if not suite:
            # Legitimate skip: target isn't claimed by any CI suite.
            continue
        if est_time is None:
            # Asymmetric with missing-suite: a test wearing a suite tag
            # that has no est_time would silently disappear from the
            # manifest while still appearing in `bazel query` results.
            # Hard error — codegen is out of sync or the BUILD's tags
            # were edited by hand.
            errors.append(
                f"{label}: has tag '{_SUITE_TAG_PREFIX}{suite}' but no "
                f"'{_EST_TIME_TAG_PREFIX}N' tag — codegen out of sync? "
                f"Re-run scripts/ci/generate_bazel_tags.py."
            )
            continue
        if est_time < 0:
            # Already errored above (malformed); skip without re-erroring.
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

    # Suites with targets but no shard config are reported via GHA
    # workflow annotations (on stderr — stdout is reserved for the JSON
    # manifest). Caller controls which suites to emit shards for, and
    # not every suite the codebase declares maps to a runnable CI suite
    # in this run (e.g. nightlies on a per-commit run). The `::warning::`
    # syntax surfaces drift on the PR Checks page so a stale --shards
    # arg doesn't silently lose a suite.
    for suite, items in by_suite.items():
        if suite not in shard_counts:
            sys.stderr.write(
                f"::warning title=Bazel manifest::"
                f"suite {suite!r} has {len(items)} target(s) but no "
                f"--shards entry; skipped\n"
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

    errors: list[str] = []
    targets = _extract_targets(args.jsonproto, errors)
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
