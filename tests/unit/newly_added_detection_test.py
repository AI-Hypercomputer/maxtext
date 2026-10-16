# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the line-range test-change detector in ``newly_added_detection``."""

import subprocess
import sys
import textwrap

import pytest

from tests.utils import newly_added_detection
from tests.utils.newly_added_detection import _build_diff_commands
from tests.utils.newly_added_detection import _is_test_file
from tests.utils.newly_added_detection import changed_tests_from_diff
from tests.utils.newly_added_detection import diff_against_base
from tests.utils.newly_added_detection import get_changed_tests
from tests.utils.newly_added_detection import main
from tests.utils.newly_added_detection import parse_changed_line_map
from tests.utils.newly_added_detection import touched_test_names
from tests.utils.newly_added_detection import touched_tests

_SCRIPT = newly_added_detection.__file__


# --- _is_test_file -----------------------------------------------------------


def test_is_test_file_accepts_test_suffixes_under_tests():
  assert _is_test_file("tests/unit/a_test.py")
  assert _is_test_file("tests/integration/b_tests.py")


def test_is_test_file_rejects_non_tests_and_non_suffix():
  assert not _is_test_file("tests/utils/test_helpers.py")  # helper, not *_test.py
  assert not _is_test_file("src/maxtext/foo_test.py")  # outside tests/
  assert not _is_test_file("tests/unit/helper.py")  # not a test suffix


# --- _build_diff_commands ----------------------------------------------------


def test_build_diff_commands_uses_only_merge_base_threedot_ranges():
  # Only three-dot (merge-base) ranges: remote first (CI / local-with-origin),
  # then local (developer without an origin remote). Two-dot tip-vs-tip ranges are
  # excluded because they over-report every commit the base gained past the fork
  # point, dragging unrelated scheduled_only tests into pre-submit.
  cmds = _build_diff_commands("main")
  assert cmds == [
      ["git", "diff", "--unified=0", "origin/main...HEAD"],
      ["git", "diff", "--unified=0", "main...HEAD"],
  ]
  # Guard against a two-dot form sneaking back in: every range arg is three-dot.
  range_args = [cmd[-1] for cmd in cmds]
  assert all("..." in arg for arg in range_args)


def test_build_diff_commands_honours_non_main_base():
  cmds = _build_diff_commands("release/v2")
  assert cmds == [
      ["git", "diff", "--unified=0", "origin/release/v2...HEAD"],
      ["git", "diff", "--unified=0", "release/v2...HEAD"],
  ]


def test_build_diff_commands_never_contains_twodot_ranges():
  for base in ["main", "origin/main", "release/v1"]:
    cmds = _build_diff_commands(base)
    for cmd in cmds:
      range_arg = cmd[-1]
      assert "..." in range_arg, f"Range arg {range_arg} must use three-dot merge-base syntax"
      assert ".." not in range_arg.replace("...", ""), f"Range arg {range_arg} contains two-dot syntax"


# --- parse_changed_line_map --------------------------------------------------


def test_line_map_added_region_from_header_range():
  diff = (
      "diff --git a/tests/unit/a_test.py b/tests/unit/a_test.py\n"
      "--- a/tests/unit/a_test.py\n"
      "+++ b/tests/unit/a_test.py\n"
      "@@ -0,0 +1,3 @@\n"
      "+line1\n+line2\n+line3\n"
  )
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {1, 2, 3}}


def test_line_map_single_line_default_length():
  diff = "+++ b/tests/unit/a_test.py\n" "@@ -10 +12 @@\n" "+changed\n"
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {12}}


def test_line_map_pure_deletion_marks_boundary():
  # `+4,0` = pure deletion; the join sits between new-file lines 4 and 5.
  diff = "+++ b/tests/unit/a_test.py\n" "@@ -5,2 +4,0 @@\n" "-gone1\n-gone2\n"
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {4, 5}}


def test_line_map_accumulates_multiple_hunks_and_files():
  diff = (
      "+++ b/tests/unit/a_test.py\n"
      "@@ -0,0 +1,1 @@\n"
      "+x\n"
      "@@ -8,0 +10,2 @@\n"
      "+y\n+z\n"
      "+++ b/tests/unit/b_test.py\n"
      "@@ -0,0 +3,1 @@\n"
      "+w\n"
  )
  assert parse_changed_line_map(diff) == {
      "tests/unit/a_test.py": {1, 10, 11},
      "tests/unit/b_test.py": {3},
  }


def test_line_map_ignores_deleted_file():
  diff = "+++ /dev/null\n" "@@ -1,2 +0,0 @@\n" "-a\n-b\n"
  assert not parse_changed_line_map(diff)


# --- tests_touching_lines ----------------------------------------------------

_SOURCE = textwrap.dedent(
    """\
    import pytest


    class TestAlpha:

      @pytest.mark.scheduled_only
      def test_existing(self):
        x = 1
        assert x == 1

      def test_untouched(self):
        assert True


    def test_top_level_untouched():
      assert True


    async def test_async_new():
      assert True
    """
)
# Line numbers in _SOURCE:
#  6 @pytest.mark.scheduled_only
#  7 def test_existing        (span 6-9, decorator included)
# 11 def test_untouched       (span 11-12)
# 15 def test_top_level_untouched (span 15-16)
# 19 async def test_async_new (span 19-20)


def test_touching_body_line_flags_that_test():
  assert touched_test_names(_SOURCE, {9}) == {"test_existing"}


def test_touching_decorator_line_flags_that_test():
  assert touched_test_names(_SOURCE, {6}) == {"test_existing"}


def test_touching_async_test_is_detected():
  assert touched_test_names(_SOURCE, {20}) == {"test_async_new"}


def test_untouched_tests_are_not_flagged():
  # Editing test_existing must not drag in the neighbouring untouched tests.
  assert touched_test_names(_SOURCE, {8, 9}) == {"test_existing"}


def test_lines_outside_any_test_flag_nothing():
  assert touched_test_names(_SOURCE, {1, 2}) == set()  # imports / blank lines


def test_empty_touched_set_returns_empty():
  assert touched_test_names(_SOURCE, set()) == set()


def test_unparseable_source_returns_empty_gracefully():
  # pytest collection surfaces the syntax error itself; the parser must not raise.
  assert touched_test_names("def broken(:\n", {1}) == set()


def test_insertion_after_untouched_test_does_not_flag_it():
  # The regression that git's hunk-header heuristic caused: a new test added
  # right after an untouched test must flag ONLY the new test.
  source = textwrap.dedent(
      """\
      def test_old():
        assert True


      def test_new():
        assert True
      """
  )
  # test_old spans lines 1-2; test_new spans lines 5-6. The added lines are 5-6.
  assert touched_test_names(source, {5, 6}) == {"test_new"}


# --- diff_against_base / get_changed_tests / CLI -----------------------------


def _git(repo, *args):
  """Run git in ``repo`` with a fixed identity so commits work on a bare CI runner."""
  subprocess.run(
      ["git", "-c", "user.name=ci", "-c", "user.email=ci@example.com", "-c", "commit.gpgsign=false", *args],
      cwd=repo,
      check=True,
      capture_output=True,
  )


@pytest.fixture(name="pr_repo")
def pr_repo_fixture(tmp_path):
  """A repo whose ``main`` holds one test file and whose HEAD (branch ``feature``) adds a test to it."""
  repo = tmp_path / "repo"
  repo.mkdir()
  _git(repo, "init", "-q")
  _git(repo, "checkout", "-q", "-b", "main")
  test_file = repo / "tests" / "unit" / "sample_test.py"
  test_file.parent.mkdir(parents=True)
  test_file.write_text("def test_old():\n  assert True\n", encoding="utf-8")
  _git(repo, "add", ".")
  _git(repo, "commit", "-q", "-m", "base")
  _git(repo, "checkout", "-q", "-b", "feature")
  test_file.write_text("def test_old():\n  assert True\n\n\ndef test_new():\n  assert True\n", encoding="utf-8")
  _git(repo, "commit", "-q", "-am", "add test_new")
  return repo


def test_get_changed_tests_reports_only_the_touched_test(pr_repo, monkeypatch):
  monkeypatch.chdir(pr_repo)
  # There is no origin remote, so detection falls through to the local ``main...HEAD`` range.
  assert get_changed_tests("main") == {("tests/unit/sample_test.py", "test_new")}


def test_changed_tests_from_diff_reads_sources_relative_to_cwd(pr_repo, monkeypatch):
  monkeypatch.chdir(pr_repo)
  diff = "+++ b/tests/unit/sample_test.py\n@@ -2,0 +3,4 @@\n"
  assert changed_tests_from_diff(diff) == {("tests/unit/sample_test.py", "test_new")}


def test_diff_against_base_returns_none_outside_a_git_work_tree(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  assert diff_against_base("main") is None


def test_main_returns_two_and_explains_when_diff_is_unavailable(tmp_path, monkeypatch, capsys):
  monkeypatch.chdir(tmp_path)
  assert main(["--base", "main"]) == 2
  captured = capsys.readouterr()
  assert captured.out == ""
  assert "cannot diff" in captured.err


def test_script_runs_without_the_tests_package(pr_repo):
  # `-I` (isolated mode) keeps the script directory, cwd and PYTHONPATH out of sys.path, so this
  # passes only if the script needs nothing but the standard library. analyze_code_changes.sh
  # depends on that: it runs on a bare runner where importing the ``tests`` package fails.
  result = subprocess.run(
      [sys.executable, "-I", _SCRIPT, "--base", "main"],
      cwd=pr_repo,
      capture_output=True,
      text=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr
  assert result.stdout.splitlines() == ["tests/unit/sample_test.py::test_new"]


# --- markers -----------------------------------------------------------------

_MARKED_SOURCE = textwrap.dedent(
    """\
    import pytest

    pytestmark = [pytest.mark.integration_test, pytest.mark.tpu_backend("arg")]


    @pytest.mark.tpu_only
    class TestOnTpu:

      def test_inherits_class_marker(self):
        assert True

      @pytest.mark.skip_on_tpu7x
      def test_skipped_on_tpu7x(self):
        assert True


    @pytest.mark.tpu_only
    @pytest.mark.parametrize("x", [1])
    def test_function_marker(x):
      assert x


    def test_unmarked():
      assert True
    """
)
_ALL_LINES = set(range(1, 40))


def test_touched_tests_collects_markers_from_module_class_and_function():
  found = touched_tests(_MARKED_SOURCE, _ALL_LINES)
  assert found["test_inherits_class_marker"] == {"integration_test", "tpu_backend", "tpu_only"}
  assert found["test_skipped_on_tpu7x"] == {"integration_test", "tpu_backend", "tpu_only", "skip_on_tpu7x"}
  assert found["test_function_marker"] == {"integration_test", "tpu_backend", "tpu_only", "parametrize"}
  assert found["test_unmarked"] == {"integration_test", "tpu_backend"}


def test_touched_tests_ignores_non_pytest_decorators():
  source = "from unittest import mock\n\n\n@mock.patch('os.getcwd')\ndef test_patched(_):\n  assert True\n"
  assert touched_tests(source, {5}) == {"test_patched": frozenset()}


def test_changed_tests_from_diff_marker_filters_match_the_tpu7x_rule(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  unit_dir = tmp_path / "tests" / "unit"
  unit_dir.mkdir(parents=True)
  (unit_dir / "marked_test.py").write_text(_MARKED_SOURCE, encoding="utf-8")
  diff = "+++ b/tests/unit/marked_test.py\n@@ -0,0 +1,30 @@\n"
  path = "tests/unit/marked_test.py"
  assert changed_tests_from_diff(diff) == {
      (path, "test_inherits_class_marker"),
      (path, "test_skipped_on_tpu7x"),
      (path, "test_function_marker"),
      (path, "test_unmarked"),
  }
  # The CI rule: runnable on TPU7X = carries tpu_only and not skip_on_tpu7x.
  assert changed_tests_from_diff(diff, require_marker="tpu_only", exclude_marker="skip_on_tpu7x") == {
      (path, "test_inherits_class_marker"),
      (path, "test_function_marker"),
  }
  assert changed_tests_from_diff(diff, require_marker="gpu_only") == set()


def test_script_marker_flags_filter_output(pr_repo):
  # pr_repo's test_new carries no markers, so requiring tpu_only yields nothing but still exits 0.
  result = subprocess.run(
      [
          sys.executable,
          "-I",
          _SCRIPT,
          "--base",
          "main",
          "--require-marker",
          "tpu_only",
          "--exclude-marker",
          "skip_on_tpu7x",
      ],
      cwd=pr_repo,
      capture_output=True,
      text=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr
  assert result.stdout == ""
