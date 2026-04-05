#!/usr/bin/env python3
"""Resolve the correct version tag for setuptools-scm.

Called by setuptools-scm via git_describe_command in pyproject.toml.
Outputs either a bare tag (e.g., "v0.5.10") for exact-match commits,
or a `git describe --long` string (e.g., "v0.5.10-2-gabcdef0") for
untagged commits. Both formats are accepted by setuptools-scm.

This two-step approach avoids a strverscmp bug where
`git tag --sort=-version:refname` sorts v0.5.10rc0 above v0.5.10,
which would cause CI to build the wrong version.

Strategy:
1. If the current commit has an exact version tag, use it directly.
   This handles CI release builds (both stable and rc).
2. Otherwise, find the highest version tag across all branches
   and describe relative to it. This handles local dev installs
   from main where release tags only exist on release branches.
"""

import subprocess
import sys


def run_git(*args: str, allow_failure: bool = False) -> str:
    """Run a git command and return stripped stdout.

    Args:
        allow_failure: If True, return "" on non-zero exit (expected for
            commands like --exact-match that legitimately fail).
            If False, log stderr on failure before returning "".
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"ERROR: Failed to run 'git {' '.join(args)}': {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        if not allow_failure:
            stderr_msg = result.stderr.strip()
            print(
                f"WARNING: git {' '.join(args)} failed "
                f"(exit {result.returncode}): {stderr_msg}",
                file=sys.stderr,
            )
        return ""

    return result.stdout.strip()


def get_exact_version_tag() -> str:
    """Return the version tag name if HEAD has an exact version tag, or empty string."""
    return run_git(
        "describe", "--tags", "--exact-match", "--match", "v*", allow_failure=True
    )


def get_latest_version_tag_describe() -> str:
    """Find the highest version tag across all branches and describe relative to it.

    Uses `git tag --sort=-version:refname` (which internally uses strverscmp).
    Note: strverscmp sorts rc/alpha suffixes ABOVE the bare version
    (e.g., v0.5.10rc0 > v0.5.10). This is acceptable for the fallback
    path because it only runs for untagged dev commits where the exact
    base version is less important (.devN+gHASH suffix is appended).
    """
    # List all version tags sorted descending by version. Try each one
    # with git describe until we find one that is an ancestor of HEAD.
    tags_raw = run_git("tag", "--list", "--sort=-version:refname", "v*.*.*")
    if not tags_raw:
        print("WARNING: No version tags (v*.*.*) found in repo", file=sys.stderr)
        return ""
    tag_list = tags_raw.splitlines()
    for tag in tag_list:
        result = run_git(
            "describe", "--tags", "--long", "--match", tag, "HEAD", allow_failure=True
        )
        if result:
            return result
    print(
        f"WARNING: Found {len(tag_list)} version tags but none are ancestors of HEAD. "
        f"Is this a shallow clone? Try: git fetch --unshallow --tags",
        file=sys.stderr,
    )
    return ""


def get_version_describe() -> str:
    """Main entry point: resolve the version describe string."""
    # Prefer exact match — correct for both stable and pre-release tags
    exact = get_exact_version_tag()
    if exact:
        return exact

    # Fallback for untagged commits (e.g., dev install from main)
    return get_latest_version_tag_describe()


def main() -> None:
    result = get_version_describe()
    if not result:
        print(
            "ERROR: Could not determine version from git tags.\n"
            "Possible causes:\n"
            "  - No version tags (v*.*.*) exist: run 'git fetch --tags'\n"
            "  - Shallow clone without tags: run 'git fetch --unshallow --tags'\n"
            "  - Git safe.directory issue: run 'git config --global --add safe.directory <repo>'\n"
            "  - Not inside a git repository\n"
            "setuptools-scm will fall back to version 0.0.0.dev0",
            file=sys.stderr,
        )
        sys.exit(1)
    print(result)


if __name__ == "__main__":
    main()
