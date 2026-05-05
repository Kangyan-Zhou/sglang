#!/usr/bin/env python3
"""AST-parse register_*_ci(...) decorators from test files; emit a Starlark
registry that tools/pytest.bzl reads to apply suite/est_time tags to each
generated py_test target.

This is the Phase-0.5 codegen tool that bridges sglang's existing
test/run_suite.py registry (in python/sglang/test/ci/ci_register.py) to
Bazel's tag-based selection.

Usage
-----
    python3 scripts/ci/generate_bazel_tags.py
        # writes tools/sglang_test_registry.bzl

    python3 scripts/ci/generate_bazel_tags.py --check
        # exits 1 if the file is out of date (for pre-commit / CI gating)

In both modes the script also exits non-zero if any individual file fails
to parse, has non-literal kwargs, or has wrong-type kwargs in a register_*_ci
call. Silent skip is a load-bearing failure mode for this codegen — a
test that disappears from the registry silently disappears from CI tag
filtering, so we never let those errors hide.

Output schema
-------------
    SGLANG_TEST_REGISTRY = {
        "test/registered/quant/test_int8_kernel.py": {
            "suite": "stage-b-test-1-gpu-small",
            "est_time": 15,
            "nightly": False,
        },
        ...
    }

Limitations
-----------
- Only parses literal arguments. `register_cuda_ci(est_time=foo)` (variable
  reference) is treated as an error: the file is excluded from the registry
  AND `main()` exits non-zero so it can't be ignored.
- When a file has both register_cuda_ci and register_amd_ci, the cuda
  registration wins. AMD-specific suites are lost in the demo registry.
- `disabled=True` registrations are skipped entirely (those tests don't run).
- Registrations not at module scope (inside `if`, `try`, function bodies)
  produce a warning. They're never matched by the registry, but the warning
  surfaces them so contributors don't accidentally hide tests behind a
  conditional registration.

Any change to ``_format_starlark`` (indentation, key ordering, ...) requires
regenerating ``tools/sglang_test_registry.bzl`` in the same commit, since
the ``--check`` byte-equality compare will otherwise fail until regen.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Hardware-specific decorators we care about, in priority order. Cuda first
# so CI tests that have both cuda+amd registrations get cuda metadata.
_HW_FUNCS = (
    "register_cuda_ci",
    "register_cpu_ci",
    "register_amd_ci",
    "register_npu_ci",
)

# Positional argument order, mirroring _PARAM_ORDER in
# python/sglang/test/ci/ci_register.py. Keep in sync with that file.
_PARAM_ORDER = ("est_time", "suite", "nightly", "disabled")

# Roots to scan. These match test/run_suite.py's `collect_tests` discovery.
_TEST_ROOTS = ("test/registered", "test/srt", "test/manual")


@dataclass
class _Reg:
    suite: str
    est_time: int
    nightly: bool


def _eval_call_args(
    call: ast.Call, file: Path, errors: list[str]
) -> dict[str, object] | None:
    """Extract positional + keyword arguments from a register_*_ci call.

    Returns the resolved name → value dict (positionals bound by
    _PARAM_ORDER, then keywords on top), or None if any value is a
    non-literal. Non-literal kwargs are a hard error — silently dropping
    the registration would make the test disappear from the Bazel registry.
    """
    out: dict[str, object] = {}

    # Positional args bind by _PARAM_ORDER, matching ci_register.py.
    for i, arg in enumerate(call.args):
        if i >= len(_PARAM_ORDER):
            errors.append(
                f"{file}:{arg.lineno}: register_*_ci has more positional args "
                f"({len(call.args)}) than _PARAM_ORDER ({len(_PARAM_ORDER)})"
            )
            return None
        try:
            out[_PARAM_ORDER[i]] = ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            errors.append(
                f"{file}:{arg.lineno}: register_*_ci positional arg "
                f"'{_PARAM_ORDER[i]}' is not a literal"
            )
            return None

    for kw in call.keywords:
        if kw.arg is None:  # **kwargs splat
            errors.append(
                f"{file}:{kw.lineno}: register_*_ci uses **kwargs splat; "
                f"cannot resolve"
            )
            return None
        if kw.arg in out:
            errors.append(
                f"{file}:{kw.lineno}: register_*_ci passes '{kw.arg}' both "
                f"positionally and as keyword"
            )
            return None
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            errors.append(
                f"{file}:{kw.lineno}: register_*_ci kwarg '{kw.arg}' is not a literal"
            )
            return None
    return out


def _warn_non_top_level_registrations(
    tree: ast.AST, file: Path, errors: list[str]
) -> None:
    """Flag register_*_ci calls that aren't at module top level.

    Non-top-level registrations (inside `if`, `try`, function bodies) are
    invisible to `_extract_first_registration` and would silently disappear
    from the Bazel registry. This is treated as an error rather than a
    warning so contributors can't accidentally hide tests behind a
    conditional.
    """
    top_level_calls = {
        id(node.value)
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in _HW_FUNCS:
            continue
        if id(node) not in top_level_calls:
            errors.append(
                f"{file}:{node.lineno}: {node.func.id}(...) is not at module top "
                f"level — would be invisible to the Bazel registry. Move it to "
                f"the top of the file or wrap it in a no-op for the registry."
            )


def _extract_first_registration(file: Path, errors: list[str]) -> _Reg | None:
    """Find the first register_<hw>_ci(...) call at module scope.

    Returns None if the file has no registration, or if every registration
    has `disabled=True`. Any structural error (parse failure, non-literal
    kwarg, wrong types) appends to ``errors`` so main() can exit non-zero.
    """
    try:
        source = file.read_text()
    except UnicodeDecodeError as e:
        errors.append(f"{file}: encoding error reading file: {e}")
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"{file}:{e.lineno}: SyntaxError: {e.msg}")
        return None

    _warn_non_top_level_registrations(tree, file, errors)

    by_hw: dict[str, _Reg] = {}
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id not in _HW_FUNCS:
            continue
        bound = _eval_call_args(call, file, errors)
        if bound is None:
            continue
        if bound.get("disabled"):
            continue
        suite = bound.get("suite")
        est_time = bound.get("est_time")
        if not isinstance(suite, str):
            errors.append(
                f"{file}:{call.lineno}: {call.func.id}(suite={suite!r}) — "
                f"suite must be a string literal"
            )
            continue
        # Accept both int and float for est_time (sglang uses 1.0 for some
        # cpu tests, integer seconds for cuda). Reject bool, which subclasses
        # int but is never a sensible duration. Round float to int for the
        # registry — bin-packing operates on whole-second granularity anyway.
        if isinstance(est_time, bool) or not isinstance(est_time, (int, float)):
            errors.append(
                f"{file}:{call.lineno}: {call.func.id}(est_time={est_time!r}) — "
                f"est_time must be an int or float literal"
            )
            continue
        by_hw[call.func.id] = _Reg(
            suite=suite,
            est_time=int(round(est_time)),
            nightly=bool(bound.get("nightly", False)),
        )

    # Cuda > cpu > amd > npu; pick the highest-priority entry that exists.
    for hw in _HW_FUNCS:
        if hw in by_hw:
            return by_hw[hw]
    return None


def _format_starlark(registry: dict[str, _Reg]) -> str:
    """Render the registry as a Starlark `.bzl` source file.

    Uses ``json.dumps`` for every string value to escape quotes, backslashes,
    and any future-funky characters — JSON's string subset is valid Starlark,
    so this is the cheap defense-in-depth route.
    """
    lines = [
        '"""GENERATED by scripts/ci/generate_bazel_tags.py — do not edit by hand.',
        "",
        "Maps each test file's repo-relative path to its register_*_ci metadata,",
        "consumed by tools/pytest.bzl to set per-target Bazel tags.",
        '"""',
        "",
        "SGLANG_TEST_REGISTRY = {",
    ]
    for path in sorted(registry):
        r = registry[path]
        lines.append(f"    {json.dumps(path)}: {{")
        lines.append(f'        "suite": {json.dumps(r.suite)},')
        lines.append(f'        "est_time": {r.est_time},')
        lines.append(f'        "nightly": {str(r.nightly)},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _build_registry(repo_root: Path, errors: list[str]) -> dict[str, _Reg]:
    registry: dict[str, _Reg] = {}
    for root in _TEST_ROOTS:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        for file in root_path.rglob("test_*.py"):
            reg = _extract_first_registration(file, errors)
            if reg is None:
                continue
            registry[str(file.relative_to(repo_root))] = reg
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail with exit 1 if the output file is stale (for pre-commit / CI).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    errors: list[str] = []
    registry = _build_registry(repo_root, errors)
    output = _format_starlark(registry)
    target = repo_root / "tools" / "sglang_test_registry.bzl"

    # Errors from the AST scan are fatal in both modes — silent skip is the
    # exact failure we're guarding against.
    if errors:
        sys.stderr.write(
            f"ERROR: {len(errors)} register_*_ci issue(s) in test sources:\n"
        )
        for err in errors:
            sys.stderr.write(f"  {err}\n")
        sys.stderr.write(
            "Fix the source(s) above; the affected tests would silently "
            "disappear from the Bazel registry.\n"
        )
        return 1

    if args.check:
        existing = target.read_text() if target.exists() else ""
        if existing != output:
            sys.stderr.write(
                f"ERROR: {target.relative_to(repo_root)} is out of date. "
                f"Re-run `python3 scripts/ci/generate_bazel_tags.py`.\n"
            )
            return 1
        sys.stderr.write(f"OK: {target.relative_to(repo_root)} matches sources\n")
        return 0

    target.write_text(output)
    sys.stderr.write(
        f"Wrote {target.relative_to(repo_root)} ({len(registry)} entries)\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
