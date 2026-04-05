#!/usr/bin/env python3
"""Tests for get_version_tag.py using temporary git repos."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from get_version_tag import (
    get_exact_version_tag,
    get_latest_version_tag_describe,
    get_version_describe,
    main,
)

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "get_version_tag.py")


class GitRepoFixture:
    """Context manager that creates a temporary git repo and chdir into it.

    The temporary directory is cleaned up on exit.
    """

    def __init__(self):
        self._tmpdir: str | None = None
        self._orig_dir: str | None = None

    def __enter__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_version_")
        self._orig_dir = os.getcwd()
        os.chdir(self._tmpdir)
        self._run("git", "init", "-b", "main")
        self._run("git", "config", "user.email", "test@test.com")
        self._run("git", "config", "user.name", "Test")
        return self

    def __exit__(self, *args):
        if self._orig_dir:
            os.chdir(self._orig_dir)
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run(self, *cmd: str):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command {cmd} failed (exit {result.returncode}):\n{result.stderr}"
            )

    def commit(self, msg="commit"):
        self._run("git", "commit", "--allow-empty", "-m", msg)

    def tag(self, name):
        self._run("git", "tag", name)

    def branch(self, name):
        self._run("git", "checkout", "-b", name)

    def checkout(self, ref):
        self._run("git", "checkout", ref)


class TestGetVersionTag(unittest.TestCase):
    """Test version tag resolution logic."""

    def test_exact_match_stable_tag(self):
        """On a commit tagged v0.5.10, should return v0.5.10."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("release")
            repo.tag("v0.5.10")

            result = get_version_describe()
            self.assertEqual(result, "v0.5.10")

    def test_exact_match_rc_tag(self):
        """On a commit tagged v0.5.10rc0, should return v0.5.10rc0."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("rc release")
            repo.tag("v0.5.10rc0")

            result = get_version_describe()
            self.assertEqual(result, "v0.5.10rc0")

    def test_stable_tag_when_rc_also_exists(self):
        """On v0.5.10 commit, should return v0.5.10 even if v0.5.10rc0 exists."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("rc")
            repo.tag("v0.5.10rc0")
            repo.commit("release")
            repo.tag("v0.5.10")

            result = get_version_describe()
            self.assertEqual(result, "v0.5.10")

    def test_rc_tag_when_stable_also_exists(self):
        """On v0.5.10rc0 commit, should return v0.5.10rc0 even if v0.5.10 exists later."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("rc")
            repo.tag("v0.5.10rc0")
            repo.commit("release")
            repo.tag("v0.5.10")
            # Go back to the rc commit
            repo.checkout("v0.5.10rc0")

            result = get_version_describe()
            self.assertEqual(result, "v0.5.10rc0")

    def test_untagged_commit_falls_back(self):
        """On an untagged commit, should describe relative to the latest ancestor tag."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("some work")
            repo.commit("more work")

            result = get_version_describe()
            # Should be like "v0.5.9-2-g<hash>"
            self.assertTrue(result.startswith("v0.5.9-2-g"), f"Got: {result}")

    def test_untagged_commit_on_main_finds_nearest_ancestor_tag(self):
        """Dev install from main should describe relative to the nearest ancestor tag,
        even when newer tags exist on unreachable release branches."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")

            # Create a release branch with a newer tag
            repo.branch("release-0.5.10")
            repo.commit("release work")
            repo.tag("v0.5.10")

            # Go back to main and add untagged commits
            repo.checkout("main")
            repo.commit("dev work on main")

            result = get_version_describe()
            # v0.5.10 is NOT an ancestor of main HEAD, so should use v0.5.9
            self.assertTrue(
                result.startswith("v0.5.9-1-g"),
                f"Expected describe relative to v0.5.9 (ancestor of HEAD), got: {result}",
            )

    def test_post_release_tag(self):
        """On a post-release tag like v0.5.8.post1, should return it exactly."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.8")
            repo.commit("hotfix")
            repo.tag("v0.5.8.post1")

            result = get_version_describe()
            self.assertEqual(result, "v0.5.8.post1")

    def test_no_tags_at_all(self):
        """With no version tags, should return empty string."""
        with GitRepoFixture() as repo:
            repo.commit("initial")

            result = get_version_describe()
            self.assertEqual(result, "")

    def test_exact_match_preferred_over_fallback(self):
        """Exact match should be used even if fallback would give a different result."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("rc")
            repo.tag("v0.5.10rc0")
            repo.commit("release")
            repo.tag("v0.5.10")
            repo.commit("rc2 for next version")
            repo.tag("v0.5.11rc0")

            result = get_version_describe()
            self.assertEqual(result, "v0.5.11rc0")

    def test_multiple_tags_on_same_commit(self):
        """If both v0.5.10 and v0.5.10rc0 are on the same commit, either is acceptable."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.9")
            repo.commit("release")
            repo.tag("v0.5.10rc0")
            repo.tag("v0.5.10")

            # git describe --exact-match has undefined ordering for multiple
            # tags on the same commit, so either is acceptable
            result = get_version_describe()
            self.assertIn(result, ["v0.5.10", "v0.5.10rc0"])

    def test_fallback_picks_highest_ancestor_tag(self):
        """When multiple tags are ancestors, fallback should pick the highest version."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.8")
            repo.commit("second")
            repo.tag("v0.5.9")
            repo.commit("untagged work")
            repo.commit("more work")

            result = get_version_describe()
            self.assertTrue(
                result.startswith("v0.5.9-2-g"),
                f"Expected describe relative to v0.5.9 (highest ancestor), got: {result}",
            )


class TestGetExactVersionTag(unittest.TestCase):
    """Test the exact-match helper in isolation."""

    def test_returns_empty_on_untagged(self):
        with GitRepoFixture() as repo:
            repo.commit("initial")

            self.assertEqual(get_exact_version_tag(), "")

    def test_returns_tag_on_tagged(self):
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v1.0.0")

            self.assertEqual(get_exact_version_tag(), "v1.0.0")

    def test_ignores_non_version_tags(self):
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("some-other-tag")

            # --match v* should skip non-version tags
            self.assertEqual(get_exact_version_tag(), "")


class TestGetLatestVersionTagDescribe(unittest.TestCase):
    """Test the fallback sorted-tag helper in isolation."""

    def test_strverscmp_sorts_rc_above_stable(self):
        """Document that strverscmp sorts rc above stable (known limitation)."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v0.5.10")
            repo.commit("after")
            repo.tag("v0.5.10rc0")
            repo.commit("untagged")

            result = get_latest_version_tag_describe()
            # strverscmp puts v0.5.10rc0 first, so fallback describes
            # relative to v0.5.10rc0 — this is the known limitation
            # that exact-match avoids for tagged commits
            self.assertTrue(
                result.startswith("v0.5.10rc0-"),
                f"Expected fallback to use rc0 due to strverscmp, got: {result}",
            )

    def test_ignores_non_version_tags(self):
        """Non-version tags should not be picked up by the fallback path."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v1.0.0")
            repo.commit("second")
            repo.tag("release-candidate")
            repo.commit("third")

            result = get_latest_version_tag_describe()
            self.assertTrue(
                result.startswith("v1.0.0-2-g"),
                f"Expected describe relative to v1.0.0, got: {result}",
            )


class TestMain(unittest.TestCase):
    """Test the main() entry point behavior."""

    def test_main_prints_version_to_stdout(self):
        """main() should print the version to stdout on success."""
        with GitRepoFixture() as repo:
            repo.commit("initial")
            repo.tag("v1.0.0")

            result = subprocess.run(
                [sys.executable, SCRIPT_PATH],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "v1.0.0")

    def test_main_exits_nonzero_when_no_tags(self):
        """main() should exit(1) and print error to stderr when no tags exist."""
        with GitRepoFixture() as repo:
            repo.commit("initial")

            result = subprocess.run(
                [sys.executable, SCRIPT_PATH],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Could not determine version", result.stderr)


if __name__ == "__main__":
    unittest.main()
