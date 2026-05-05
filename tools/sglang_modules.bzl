"""Single source of truth for the //python:* submodule taxonomy.

Adding a new py_library target in python/BUILD.bazel? Add an entry here
too. tools/pytest.bzl uses this mapping to compute the env vars that
drive the import audit hook in tools/pytest_main.py — keeping all three
files in agreement.
"""

# Maps a //python:* target's trailing name → list of sglang.X directory
# names under python/sglang/. Special sentinels:
#   "__top__"  → top-level .py files in python/sglang/ (sglang.utils, etc.)
#   "__all__"  → audit disabled (the :sglang_srcs aggregate)
LABEL_TO_DIRS = {
    "core": ["__top__"],
    "srt": ["srt"],
    "jit_kernel": ["jit_kernel"],
    "lang": ["lang"],
    "test_helpers": ["test"],
    "multimodal_gen": ["multimodal_gen"],
    "misc": ["cli", "benchmark", "eval"],
    "sglang_srcs": ["__all__"],
}

# All real sglang.X submodule directory names (sentinels stripped).
# Constant per macro version; used by the import audit to distinguish
# `sglang.srt.foo` (a submodule) from `sglang.utils` (a top-level file).
# Starlark has no set comprehension — dict-keys is the dedupe idiom.
ALL_SUBMODULE_DIRS = sorted({
    d: None
    for dirs in LABEL_TO_DIRS.values()
    for d in dirs
    if d not in ("__top__", "__all__")
}.keys())

# Common bundle for tests that exercise sglang.srt + use CustomTestCase
# from sglang.test. Everything except :multimodal_gen is here because of
# cross-submodule eager imports we can't avoid:
#
#   python/sglang/__init__.py:30+   from sglang.lang.api import ...
#   python/sglang/test/test_utils.py:34   from sglang.bench_serving import ...
#       (bench_serving is at top-level → :core, but it imports
#        sglang.benchmark.* which is in :misc)
#
# Removing :lang or :misc would make the audit hook fail every test that
# touches sglang. The narrow-win for non-diffusion tests is escaping
# :multimodal_gen (479 files); diffusion tests should use :sglang_srcs.
SGLANG_RUNTIME = [
    "//python:srt",
    "//python:test_helpers",
    "//python:jit_kernel",
    "//python:lang",
    "//python:misc",
    "//python:core",
]
