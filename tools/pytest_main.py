"""py_test entry point that re-invokes pytest with the args Bazel passed in.

Two startup checks before pytest runs:

  1. SGLANG_REQUIRED_VENV_DEPS — venv_deps from the macro. Validates the
     listed module names import; fails fast with a clear message if not.

  2. SGLANG_DECLARED_DIRS + SGLANG_KNOWN_SUBDIRS — populated by the
     macro from tools/sglang_modules.bzl. The audit hook installed on
     sys.addaudithook() watches every `import` event; if a `sglang.X`
     import escapes the declared set, the test exits with an attributable
     error rather than letting Bazel cache a stale result.

Why the audit hook matters: sglang has cross-submodule lazy imports
(e.g. python/sglang/srt/constrained/xgrammar_backend.py:331 does
`from sglang.test.test_utils import ...` inside a method). Without the
audit, a test scoped to `:srt` could pass against a runfiles tree that's
missing `:test_helpers` (because the editable install in dev environments
falls back to the host source tree), which would Bazel-cache a green
result while genuinely depending on files Bazel doesn't see.

The submodule taxonomy lives in tools/sglang_modules.bzl. This file
holds no hardcoded list of submodules — both the "what's allowed for
this test" and "what counts as a submodule directory" come in via env
vars set by the macro.
"""

from __future__ import annotations

import importlib
import os
import sys


def _check_venv_deps() -> None:
    raw = os.environ.get("SGLANG_REQUIRED_VENV_DEPS", "")
    missing: list[str] = []
    for name in (n.strip() for n in raw.split(",") if n.strip()):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        sys.stderr.write(
            "ERROR: required host-venv deps not importable: "
            + ", ".join(missing)
            + "\nRun scripts/ci/cuda/ci_install_dependency.sh (and any side-script "
            "like ci_install_deepep.sh) in a venv before `bazel test`, or remove "
            "the missing names from this target's `venv_deps`.\n"
        )
        sys.exit(2)


def _install_import_audit() -> None:
    raw_declared = os.environ.get("SGLANG_DECLARED_DIRS", "").strip()
    if not raw_declared:
        return
    declared = frozenset(s for s in raw_declared.split(",") if s)

    if "__all__" in declared:
        return  # aggregate :sglang_srcs disables the audit

    raw_known = os.environ.get("SGLANG_KNOWN_SUBDIRS", "").strip()
    if not raw_known:
        sys.stderr.write(
            "ERROR: SGLANG_DECLARED_DIRS is set but SGLANG_KNOWN_SUBDIRS is "
            "empty. tools/pytest.bzl wiring is broken — both env vars should "
            "be set together.\n"
        )
        sys.exit(2)
    known_subdirs = frozenset(s for s in raw_known.split(",") if s)

    allow_top_level = "__top__" in declared
    allowed_subdirs = declared - {"__top__"}

    def hook(event: str, args: tuple) -> None:
        if event != "import":
            return
        mod = args[0]
        if not isinstance(mod, str) or not mod.startswith("sglang"):
            return
        if mod == "sglang":
            return  # the package init itself
        parts = mod.split(".", 2)
        if len(parts) < 2 or parts[0] != "sglang":
            return
        sub = parts[1]
        if sub in known_subdirs:
            if sub not in allowed_subdirs:
                raise ImportError(
                    f"Test imported '{mod}' but sglang_modules did not include "
                    f"the //python:* target for sglang.{sub}. Allowed: "
                    f"{sorted(allowed_subdirs)}. Either widen sglang_modules "
                    f"or remove the import."
                )
        else:
            # sglang.X where X is a top-level file (utils, version, ...)
            if not allow_top_level:
                raise ImportError(
                    f"Test imported top-level '{mod}' but sglang_modules did "
                    f"not include //python:core. Declared: {sorted(declared)}."
                )

    sys.addaudithook(hook)


def main() -> int:
    _check_venv_deps()
    _install_import_audit()
    import pytest  # late import — after audit hook is installed

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
