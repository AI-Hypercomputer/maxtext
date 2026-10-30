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

"""Unit tests for the line-range test-change detector in ``newly_added_detection``.

Most tests run the command-line tool the way CI does: a bare ``python3 -I -S`` (standard library only), inside a
temporary git repository whose ``main`` was pushed to a bare ``origin`` remote, so the ``origin/main...HEAD`` range
that CI uses is the one exercised.

Never define a ``unittest.TestCase`` or absl ``parameterized.TestCase`` subclass at module level in this file: pytest
collects TestCase subclasses whatever their name, so the Development Tests job would run its methods as tests. Build such
classes inside a test function, or write them out as source strings, as the tests below do.
"""

import configparser
import importlib.util
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap
import types

import pytest

# The repository root: tests/dev/<this file>.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Loaded from its path rather than imported as tests.utils.newly_added_detection: the Development Tests job runs
# without MaxText's dependencies, which importing the tests.utils package needs.
_spec = importlib.util.spec_from_file_location(
    "newly_added_detection", _REPO_ROOT / "tests/utils/newly_added_detection.py"
)
newly_added_detection = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = newly_added_detection
_spec.loader.exec_module(newly_added_detection)

# pylint: disable=protected-access
TPU7X_FLAVORS = newly_added_detection.TPU7X_FLAVORS
TPU7X_PR_MATRIX = newly_added_detection.TPU7X_PR_MATRIX
_TESTS_ROOT = newly_added_detection._TESTS_ROOT
_TEST_SUFFIXES = newly_added_detection._TEST_SUFFIXES
_build_diff_commands = newly_added_detection._build_diff_commands
_is_test_file = newly_added_detection._is_test_file
# pylint: enable=protected-access
changed_tests_from_diff = newly_added_detection.changed_tests_from_diff
diff_against_base = newly_added_detection.diff_against_base
get_changed_tests = newly_added_detection.get_changed_tests
is_changed_item = newly_added_detection.is_changed_item
item_test_names = newly_added_detection.item_test_names
main = newly_added_detection.main
marker_expression_matches = newly_added_detection.marker_expression_matches
parse_changed_line_map = newly_added_detection.parse_changed_line_map
touched_test_names = newly_added_detection.touched_test_names
touched_tests = newly_added_detection.touched_tests

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

# Every option keeps the output independent of the user's git config (color, external diff and textconv drivers,
# prefixes, rename detection, inter-hunk context, path quoting); the ``-- tests/`` pathspec keeps files that can never
# hold a test, whatever their encoding, out of the diff.
_HARDENED_DIFF = [
    "git",
    "-c",
    "core.quotePath=false",
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--unified=0",
    "--inter-hunk-context=0",
    "-M",
    "--full-index",
    "--src-prefix=a/",
    "--dst-prefix=b/",
]


def _range_arg(cmd):
  """Return the revision range of a diff command: the argument before the ``--`` that starts the pathspec."""
  return cmd[cmd.index("--") - 1]


def test_build_diff_commands_uses_only_merge_base_threedot_ranges():
  # Only three-dot (merge-base) ranges: remote first (CI / local-with-origin),
  # then local (developer without an origin remote). Two-dot tip-vs-tip ranges are
  # excluded because they over-report every commit the base gained past the fork
  # point, dragging unrelated scheduled_only tests into pre-submit.
  cmds = _build_diff_commands("main")
  assert cmds == [
      [*_HARDENED_DIFF, "origin/main...HEAD", "--", "tests/"],
      [*_HARDENED_DIFF, "main...HEAD", "--", "tests/"],
  ]
  # Guard against a two-dot form sneaking back in: every range arg is three-dot.
  range_args = [_range_arg(cmd) for cmd in cmds]
  assert all("..." in arg for arg in range_args)


def test_build_diff_commands_honours_non_main_base():
  cmds = _build_diff_commands("release/v2")
  assert cmds == [
      [*_HARDENED_DIFF, "origin/release/v2...HEAD", "--", "tests/"],
      [*_HARDENED_DIFF, "release/v2...HEAD", "--", "tests/"],
  ]


def test_build_diff_commands_never_contains_twodot_ranges():
  for base in ["main", "origin/main", "release/v1"]:
    cmds = _build_diff_commands(base)
    for cmd in cmds:
      range_arg = _range_arg(cmd)
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


def test_line_map_reads_hunk_bodies_by_count():
  # An added line whose content starts with "++ " shows up as "+++ ..." in the diff. It is content, not a file header.
  diff = "+++ b/tests/unit/a_test.py\n" "@@ -0,0 +1,2 @@\n" "+++ not a header\n" "+x\n" "@@ -9,0 +12 @@\n" "+y\n"
  assert parse_changed_line_map(diff) == {"tests/unit/a_test.py": {1, 2, 12}}


def test_line_map_unquotes_c_quoted_paths():
  # Even with core.quotePath=false, git C-quotes a path that holds '"', '\' or a control character.
  diff = (
      '+++ "b/tests/unit/a\\"b_test.py"\n'
      "@@ -0,0 +1 @@\n"
      "+x\n"
      '+++ "b/tests/unit/r\\303\\251sum\\303\\251_test.py"\n'
      "@@ -0,0 +2 @@\n"
      "+y\n"
  )
  assert parse_changed_line_map(diff) == {'tests/unit/a"b_test.py': {1}, "tests/unit/r\u00e9sum\u00e9_test.py": {2}}


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
  # Line 6 is also the first line of TestAlpha's body, so it is not part of the class header.
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


_GROUPED_SOURCE = textwrap.dedent(
    """\
    import pytest

    pytestmark = pytest.mark.tpu_only


    @pytest.mark.integration_test
    class TestGroup(
        object,
    ):
      # The class header runs from the class line to the line before the first body statement.
      pytestmark = [pytest.mark.post_training]
      flag = True

      def test_one(self):
        assert self.flag

      def helper(self):
        return self.flag

      def test_two(self):
        assert self.helper()


    def test_free():
      assert True
    """
)
# Line numbers in _GROUPED_SOURCE:
#  3 module pytestmark
#  6 class decorator
#  7-10 class header (class line, base list, comment)
# 11 class-body pytestmark
# 12 another class attribute
# 14 def test_one (span 14-15)
# 17 def helper (span 17-18)
# 20 def test_two (span 20-21)
# 24 def test_free (span 24-25)


@pytest.mark.parametrize(
    "lines, expected",
    [
        pytest.param({3}, {"test_one", "test_two", "test_free"}, id="module-pytestmark"),
        pytest.param({6}, {"test_one", "test_two"}, id="class-decorator"),
        pytest.param({8}, {"test_one", "test_two"}, id="class-header"),
        pytest.param({10}, {"test_one", "test_two"}, id="comment-in-class-header"),
        pytest.param({11}, {"test_one", "test_two"}, id="class-pytestmark"),
        pytest.param({12}, set(), id="other-class-attribute"),
        pytest.param({18}, set(), id="helper-method"),
        pytest.param({15}, {"test_one"}, id="test-body"),
    ],
)
def test_marker_lines_touch_every_test_they_apply_to(lines, expected):
  # A class's decorators, header and class-body pytestmark apply to all of its tests, and the module pytestmark to
  # all tests in the module, so an edit there touches each of them.
  assert touched_test_names(_GROUPED_SOURCE, lines) == expected


def test_touched_tests_include_class_body_pytestmark_markers():
  assert touched_tests(_GROUPED_SOURCE, {15}) == {
      "test_one": frozenset({"tpu_only", "integration_test", "post_training"})
  }


def test_test_names_follow_pytest_default_prefix():
  # pytest's default python_functions is "test", so `def test(self)` and `def testFoo(self)` are tests.
  source = textwrap.dedent(
      """\
      import unittest


      class Benchmark(unittest.TestCase):

        def test(self):
          pass

        def testFoo(self):
          pass

        def helper_test(self):
          pass

        def _test_private(self):
          pass

        def Test_capitalized(self):
          pass


      def testbar():
        pass
      """
  )
  assert touched_test_names(source, set(range(1, 30))) == {"test", "testFoo", "testbar"}


_COLLECTION_SOURCE = textwrap.dedent(
    """\
    import unittest


    class TestPlain:

      def test_plain(self):
        pass


    class TestWithInit:

      def __init__(self):
        self.value = 1

      def test_with_init(self):
        pass


    class TestDisabled:
      __test__ = False

      def test_disabled(self):
        pass


    class Helper:

      def test_in_helper(self):
        pass


    class Mixin:

      def test_in_mixin(self):
        pass


    class TestUsesMixin(Mixin):

      def test_own(self):
        pass


    class Suite(unittest.TestCase):

      def test_in_suite(self):
        pass
    """
)


def test_tests_that_pytest_never_collects_are_not_reported():
  # pytest skips a Test* class with __init__, a class with another name unless it may be a TestCase, and anything
  # under `__test__ = False`. Reporting such a test would start a TPU7X runner that then selects no test.
  assert touched_test_names(_COLLECTION_SOURCE, set(range(1, 60))) == {
      "test_plain",
      "test_in_mixin",  # Collected through TestUsesMixin.
      "test_own",
      "test_in_suite",
  }
  disabled_module = (
      "__test__ = False\n\n\ndef test_free():\n  pass\n\n\nclass TestGroup:\n\n  def test_method(self):\n    pass\n"
  )
  assert touched_test_names(disabled_module, set(range(1, 20))) == set()


_DELETION_BASE = textwrap.dedent(
    """\
    import pytest


    @pytest.mark.skip_on_tpu7x
    class TestGroup:

      def test_one(self):
        assert True

      @pytest.mark.skip_on_tpu7x
      def test_two(self):
        assert True

      def test_three(self):
        assert True


    def test_free():
      assert True
    """
)
# Line numbers in _DELETION_BASE:
#  4 class decorator
# 10 test_two's decorator (span 10-12, then a blank line 13)


def _without_lines(source, lines):
  """Return ``source`` with the given 1-based line numbers removed."""
  return "".join(line for number, line in enumerate(source.splitlines(keepends=True), 1) if number not in lines)


@pytest.mark.parametrize(
    "deleted, expected",
    [
        pytest.param({4}, {"test_one", "test_two", "test_three"}, id="class-decorator"),
        pytest.param({10}, {"test_two"}, id="test-decorator"),
        pytest.param({10, 11, 12, 13}, set(), id="whole-test"),
    ],
)
def test_deleted_lines_are_attributed_on_the_base_side(deleted, expected):
  new_source = _without_lines(_DELETION_BASE, deleted)
  assert touched_test_names(new_source, set(), base_source=_DELETION_BASE, deleted_lines=deleted) == expected


# --- diff_against_base / get_changed_tests / CLI -----------------------------


def _git(repo, *args):
  """Run git in ``repo`` with a fixed identity, and without hooks or the user's and system's git config.

  That keeps commits working, and fast, on a bare CI runner and on a workstation with hooks configured.
  """
  subprocess.run(
      [
          "git",
          "-c",
          "user.name=ci",
          "-c",
          "user.email=ci@example.com",
          "-c",
          "commit.gpgsign=false",
          "-c",
          "core.hooksPath=/dev/null",
          *args,
      ],
      cwd=repo,
      env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
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


def _write_files(root, files):
  """Write ``files`` under ``root``. Each repo-relative path maps to text, bytes, or None to delete the file."""
  for relative, content in files.items():
    path = root / relative
    if content is None:
      path.unlink()
      continue
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
      path.write_bytes(content)
    else:
      path.write_text(content, encoding="utf-8")


def _make_pr_repo(tmp_path, base, change):
  """Return a checkout shaped like CI's: ``main`` holds ``base``, and HEAD is one commit that applies ``change``.

  ``main`` is pushed to a bare ``origin`` remote and the local ``main`` branch is deleted, as in a CI checkout, so the
  detector can only diff against ``origin/main``. ``base`` and ``change`` map repo-relative paths to contents (see
  ``_write_files``); an empty ``change`` makes an empty commit.
  """
  repo = tmp_path / "repo"
  origin = tmp_path / "origin.git"
  repo.mkdir()
  _git(repo, "init", "-q")
  _git(repo, "checkout", "-q", "-b", "main")
  _write_files(repo, base)
  _git(repo, "add", "-A")
  _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
  _git(tmp_path, "init", "-q", "--bare", str(origin))
  _git(repo, "remote", "add", "origin", str(origin))
  _git(repo, "push", "-q", "origin", "main")
  _git(repo, "checkout", "-q", "-b", "feature")
  _git(repo, "branch", "-q", "-D", "main")
  _write_files(repo, change)
  _git(repo, "add", "-A")
  _git(repo, "commit", "-q", "--allow-empty", "-m", "change")
  return repo


def _run_cli(cwd, *args, env=None, stdin_text=None):
  """Run the detector the way CI does: with ``python3 -I -S``, which can import only the standard library.

  ``-I`` keeps the script's directory, the cwd, ``PYTHON*`` variables and the user site-packages off ``sys.path``.
  ``-S`` also drops site-packages, which ``-I`` alone keeps. ``stdin_text`` is written to the detector's stdin.
  """
  return subprocess.run(
      [sys.executable, "-I", "-S", _SCRIPT, *args],
      cwd=cwd,
      env=env,
      input=stdin_text,
      capture_output=True,
      encoding="utf-8",
      errors="replace",
      check=False,
  )


def _ok(result):
  """Assert that a detector run exited 0, and return its stdout lines."""
  assert result.returncode == 0, f"exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  return result.stdout.splitlines()


def _compact(flavors):
  return json.dumps(list(flavors), separators=(",", ":"))


def _eligible(repo, *args):
  """Return what ``--eligible-flavors`` prints, after checking that stdout is exactly one compact JSON array."""
  lines = _ok(_run_cli(repo, "--base", "main", "--eligible-flavors", *args))
  assert len(lines) == 1, lines
  flavors = json.loads(lines[0])
  assert lines[0] == _compact(flavors)
  return flavors


def _src(*marks, module_mark=None, name="test_case"):
  """Return a test module whose one test, ``name``, has a ``@pytest.mark.<mark>`` decorator for each of ``marks``."""
  lines = ["import pytest", ""]
  if module_mark is not None:
    lines += [f"pytestmark = {module_mark}", ""]
  lines += ["", *(f"@pytest.mark.{mark}" for mark in marks), f"def {name}():", "  assert True", ""]
  return "\n".join(lines)


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
  # `-I` (isolated mode) keeps the script directory, cwd and PYTHONPATH out of sys.path, and `-S` drops
  # site-packages too, so this passes only if the script needs nothing but the standard library.
  # analyze_code_changes.sh depends on that: it runs on a bare runner where importing the ``tests`` package fails.
  result = _run_cli(pr_repo, "--base", "main")
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


def test_changed_tests_from_diff_reports_every_touched_test(tmp_path, monkeypatch):
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


def test_changed_tests_from_diff_flavor_keeps_only_eligible_tests(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  unit_dir = tmp_path / "tests" / "unit"
  unit_dir.mkdir(parents=True)
  (unit_dir / "marked_test.py").write_text(_MARKED_SOURCE, encoding="utf-8")
  diff = "+++ b/tests/unit/marked_test.py\n@@ -0,0 +1,30 @@\n"
  path = "tests/unit/marked_test.py"
  # The module pytestmark makes every test an integration_test. test_unmarked has no hardware marker, so it is
  # cpu_only, and TPU7X skips test_skipped_on_tpu7x.
  assert changed_tests_from_diff(diff, flavor="tpu7x-integration") == {
      (path, "test_inherits_class_marker"),
      (path, "test_function_marker"),
  }
  assert changed_tests_from_diff(diff, flavor="tpu7x-unit") == set()
  with pytest.raises(ValueError):
    changed_tests_from_diff(diff, flavor="tpu7x-no-such-flavor")


# --- CLI exit status ----------------------------------------------------------

# analyze_code_changes.sh treats any non-zero exit as "detection failed" and runs the full TPU7X matrix; 2 must
# mean "cannot diff" and nothing else.


@pytest.mark.parametrize(
    "mode", [[], ["--eligible-flavors"], ["--flavor", "tpu7x-unit"]], ids=["list", "eligible-flavors", "flavor"]
)
def test_cli_exits_two_when_the_base_does_not_resolve(tmp_path, mode):
  repo = _make_pr_repo(tmp_path, {"tests/unit/sample_test.py": _src()}, {"tests/unit/sample_test.py": _src("tpu_only")})
  result = _run_cli(repo, "--base", "no-such-branch", *mode)
  assert result.returncode == 2, result
  assert result.stdout == ""
  assert "cannot diff" in result.stderr


@pytest.mark.parametrize("mode", [[], ["--eligible-flavors"]], ids=["list", "eligible-flavors"])
def test_cli_exits_two_outside_a_git_repository(tmp_path, mode):
  outside = tmp_path / "outside"
  outside.mkdir()
  # GIT_CEILING_DIRECTORIES keeps git from finding a repository above tmp_path.
  env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
  result = _run_cli(outside, "--base", "main", *mode, env=env)
  assert result.returncode == 2, result
  assert result.stdout == ""
  assert "cannot diff" in result.stderr


def test_cli_never_passes_a_base_that_looks_like_an_option_to_git(tmp_path):
  # In the local range "<base>...HEAD", this base would make `git diff` write its output to a file and exit 0.
  repo = _make_pr_repo(tmp_path, {}, {"tests/unit/sample_test.py": _src("tpu_only")})
  result = _run_cli(repo, f"--base=--output={tmp_path / 'leak'}")
  assert result.returncode == 2, result
  assert "cannot diff" in result.stderr
  assert not list(tmp_path.glob("leak*"))


def test_cli_base_is_the_flag_then_github_base_ref_then_main(tmp_path):
  path = "tests/unit/sample_test.py"
  repo = _make_pr_repo(tmp_path, {}, {path: _src()})
  env = {key: value for key, value in os.environ.items() if key != "GITHUB_BASE_REF"}
  listed = [f"{path}::test_case"]
  assert _ok(_run_cli(repo, "--base", "main", env={**env, "GITHUB_BASE_REF": "no-such-branch"})) == listed
  assert _ok(_run_cli(repo, env={**env, "GITHUB_BASE_REF": "main"})) == listed
  assert _run_cli(repo, env={**env, "GITHUB_BASE_REF": "no-such-branch"}).returncode == 2
  assert _ok(_run_cli(repo, env=env)) == listed


@pytest.mark.parametrize(
    "args",
    [["--flavor", "tpu7x-no-such-flavor"], ["--flavor", "tpu7x-unit", "--eligible-flavors"], ["--no-such-option"]],
    ids=["unknown-flavor", "flavor-with-eligible-flavors", "unknown-option"],
)
def test_cli_usage_errors_exit_64_not_2(tmp_path, args):
  # 2 means "cannot diff"; a usage error must never be mistaken for it.
  repo = _make_pr_repo(tmp_path, {}, {"tests/unit/sample_test.py": _src("tpu_only")})
  result = _run_cli(repo, "--base", "main", *args)
  assert result.returncode == 64, result
  assert result.stdout == ""


def test_main_returns_64_for_a_usage_error_instead_of_raising(capsys):
  assert main(["--flavor", "tpu7x-no-such-flavor"]) == 64
  assert capsys.readouterr().out == ""


# --- CLI marker filters, flavors and eligible flavors -------------------------


def test_cli_filters_keep_only_tests_that_run_on_tpu7x(tmp_path):
  # A PR adds a tpu_only test and a tpu_only test that TPU7X skips. Only the first may start a TPU7X runner.
  path = "tests/unit/sample_test.py"
  old = "import pytest\n\n\ndef test_old():\n  assert True\n"
  new = old + textwrap.dedent(
      """\


      @pytest.mark.tpu_only
      def test_tpu():
        assert True


      @pytest.mark.tpu_only
      @pytest.mark.skip_on_tpu7x
      def test_tpu_skipped():
        assert True
      """
  )
  repo = _make_pr_repo(tmp_path, {path: old}, {path: new})
  assert _ok(_run_cli(repo, "--base", "main")) == [f"{path}::test_tpu", f"{path}::test_tpu_skipped"]
  only_tpu = [f"{path}::test_tpu"]
  assert _ok(_run_cli(repo, "--base", "main", "--flavor", "tpu7x-unit")) == only_tpu
  assert _eligible(repo) == ["tpu7x-unit"]


_PYTEST_INI = textwrap.dedent(
    """\
    [pytest]
    testpaths =
        tests
    addopts =
        -rf --import-mode=importlib --strict-markers
        --ignore=tests/unit/ignored_test.py
        --ignore tests/unit/spaced_ignored_test.py
        --ignore=tests/unit/legacy
    """
)
_CLASS_DECORATOR_SOURCE = (
    "import pytest\n\n\n@pytest.mark.tpu_only\nclass TestGroup:\n\n  def test_case(self):\n    pass\n"
)
_CLASS_PYTESTMARK_SOURCE = (
    "import pytest\n\n\nclass TestGroup:\n  pytestmark = [pytest.mark.tpu_only]\n\n  def test_case(self):\n    pass\n"
)
_UNIT = "tests/unit/case_test.py"


@pytest.mark.parametrize(
    "path, source, ini, flavors",
    [
        pytest.param(_UNIT, _src("tpu_only"), _PYTEST_INI, ["tpu7x-unit"], id="unit-tpu_only"),
        pytest.param(_UNIT, _src("tpu_only"), None, ["tpu7x-unit"], id="no-pytest-ini"),
        pytest.param(
            "tests/integration/case_test.py",
            _src("integration_test", "tpu_only"),
            _PYTEST_INI,
            ["tpu7x-integration"],
            id="integration",
        ),
        pytest.param(
            "tests/post_training/unit/case_test.py",
            _src("post_training", "tpu_only"),
            _PYTEST_INI,
            ["tpu7x-post-training-unit"],
            id="post-training-unit",
        ),
        pytest.param(
            _UNIT,
            _src("post_training", "tpu_only"),
            _PYTEST_INI,
            ["tpu7x-post-training-unit"],
            id="post-training-in-unit",
        ),
        pytest.param(
            "tests/post_training/integration/case_test.py",
            _src("post_training", "integration_test", "tpu_only"),
            _PYTEST_INI,
            [],
            id="post-training-integration-not-in-matrix",
        ),
        pytest.param(
            "tests/post_training/unit/case_test.py", _src("tpu_only"), _PYTEST_INI, [], id="post-training-path-only"
        ),
        pytest.param(
            "tests/integration/case_test.py",
            _src("post_training", "tpu_only"),
            _PYTEST_INI,
            [],
            id="post-training-marker-outside-its-paths",
        ),
        pytest.param(
            _UNIT,
            _src("gpu_only", module_mark="[pytest.mark.tpu_only]"),
            _PYTEST_INI,
            [],
            id="gpu_only-in-tpu_only-module",
        ),
        pytest.param("tests/unit/ignored_test.py", _src("tpu_only"), _PYTEST_INI, [], id="ignored-file"),
        pytest.param("tests/unit/spaced_ignored_test.py", _src("tpu_only"), _PYTEST_INI, [], id="ignored-file-two-args"),
        pytest.param("tests/unit/legacy/case_test.py", _src("tpu_only"), _PYTEST_INI, [], id="ignored-directory"),
        pytest.param(
            "tests/unit/legacy_api_test.py", _src("tpu_only"), _PYTEST_INI, ["tpu7x-unit"], id="ignore-is-not-a-prefix"
        ),
        pytest.param(
            "tests/integration/case_test.py",
            _src("integration_test", "tpu_only"),
            "[pytest]\ntestpaths = tests/unit\n",
            [],
            id="outside-testpaths",
        ),
        pytest.param(_UNIT, _src(), _PYTEST_INI, [], id="unmarked"),
        pytest.param(_UNIT, _src("cpu_only"), _PYTEST_INI, [], id="cpu_only"),
        pytest.param(_UNIT, _src("tpu_backend"), _PYTEST_INI, [], id="tpu_backend-is-not-a-hardware-marker"),
        pytest.param(_UNIT, _src("tpu_only", "skip_on_tpu7x"), _PYTEST_INI, [], id="skip_on_tpu7x"),
        pytest.param(_UNIT, _src("tpu_only", "skip"), _PYTEST_INI, [], id="skip"),
        pytest.param(_UNIT, _src("tpu_only", 'skip(reason="broken")'), _PYTEST_INI, [], id="skip-with-reason"),
        pytest.param(_UNIT, _src("tpu_only", 'skipif(True, reason="x")'), _PYTEST_INI, ["tpu7x-unit"], id="skipif"),
        pytest.param(_UNIT, _src("tpu_only", "scheduled_only"), _PYTEST_INI, ["tpu7x-unit"], id="scheduled_only"),
        pytest.param(
            _UNIT, _src(module_mark="pytest.mark.tpu_only"), _PYTEST_INI, ["tpu7x-unit"], id="single-marker-pytestmark"
        ),
        pytest.param(
            _UNIT,
            _src(module_mark="(pytest.mark.tpu_only, pytest.mark.scheduled_only)"),
            _PYTEST_INI,
            ["tpu7x-unit"],
            id="tuple-pytestmark",
        ),
        pytest.param(_UNIT, _CLASS_DECORATOR_SOURCE, _PYTEST_INI, ["tpu7x-unit"], id="class-decorator"),
        pytest.param(_UNIT, _CLASS_PYTESTMARK_SOURCE, _PYTEST_INI, ["tpu7x-unit"], id="class-body-pytestmark"),
    ],
)
def test_cli_eligible_flavors_follow_markers_paths_and_ignores(tmp_path, path, source, ini, flavors):
  # The PR matrix runs only the flavors that collect and run a touched test.
  repo = _make_pr_repo(tmp_path, {"README.md": "base\n"} if ini is None else {"pytest.ini": ini}, {path: source})
  assert _eligible(repo) == flavors


_MIXED_PR = {
    "tests/unit/mixed_test.py": textwrap.dedent(
        """\
        import pytest


        @pytest.mark.tpu_only
        def test_unit():
          assert True


        @pytest.mark.tpu_only
        @pytest.mark.skip_on_tpu7x
        def test_unit_skipped_on_tpu7x():
          assert True


        def test_unit_cpu():
          assert True


        @pytest.mark.tpu_only
        @pytest.mark.post_training
        def test_unit_post_training():
          assert True
        """
    ),
    "tests/integration/flow_test.py": _src("integration_test", "tpu_only", name="test_integration"),
    "tests/post_training/unit/trainer_test.py": _src("post_training", "tpu_only", name="test_post_training_unit"),
    "tests/post_training/integration/rl_test.py": _src(
        "post_training", "integration_test", "tpu_only", name="test_post_training_integration"
    ),
}


@pytest.fixture(name="mixed_repo", scope="module")
def mixed_repo_fixture(tmp_path_factory):
  """A PR that adds a test for every TPU7X flavor, plus tests that no TPU7X flavor runs."""
  return _make_pr_repo(tmp_path_factory.mktemp("mixed"), {"README.md": "base\n"}, _MIXED_PR)


def test_cli_lists_every_touched_test_without_filters(mixed_repo):
  assert _ok(_run_cli(mixed_repo, "--base", "main")) == [
      "tests/integration/flow_test.py::test_integration",
      "tests/post_training/integration/rl_test.py::test_post_training_integration",
      "tests/post_training/unit/trainer_test.py::test_post_training_unit",
      "tests/unit/mixed_test.py::test_unit",
      "tests/unit/mixed_test.py::test_unit_cpu",
      "tests/unit/mixed_test.py::test_unit_post_training",
      "tests/unit/mixed_test.py::test_unit_skipped_on_tpu7x",
  ]


@pytest.mark.parametrize(
    "flavor, expected",
    [
        ("tpu7x-unit", ["tests/unit/mixed_test.py::test_unit"]),
        ("tpu7x-integration", ["tests/integration/flow_test.py::test_integration"]),
        (
            "tpu7x-post-training-unit",
            [
                "tests/post_training/unit/trainer_test.py::test_post_training_unit",
                "tests/unit/mixed_test.py::test_unit_post_training",
            ],
        ),
        (
            "tpu7x-post-training-integration",
            ["tests/post_training/integration/rl_test.py::test_post_training_integration"],
        ),
    ],
)
def test_cli_flavor_lists_only_the_tests_that_flavor_runs(mixed_repo, flavor, expected):
  assert _ok(_run_cli(mixed_repo, "--base", "main", "--flavor", flavor)) == expected


def test_cli_eligible_flavors_prints_one_compact_json_line_in_matrix_order(mixed_repo):
  # tpu7x-post-training-integration has an eligible test as well, but it is not in the PR matrix.
  result = _run_cli(mixed_repo, "--base", "main", "--eligible-flavors")
  assert _ok(result) == ['["tpu7x-unit","tpu7x-integration","tpu7x-post-training-unit"]']
  assert result.stderr.splitlines() == [
      "tpu7x-unit: tests/unit/mixed_test.py::test_unit",
      "tpu7x-integration: tests/integration/flow_test.py::test_integration",
      "tpu7x-post-training-unit: tests/post_training/unit/trainer_test.py::test_post_training_unit",
      "tpu7x-post-training-unit: tests/unit/mixed_test.py::test_unit_post_training",
  ]


# --- marker_expression_matches ------------------------------------------------


@pytest.mark.parametrize(
    "expression, markers, expected",
    [
        ("a or b and c", {"a"}, True),  # `and` binds tighter than `or`: a or (b and c)
        ("a or b and c", {"b"}, False),
        ("(a or b) and c", {"a"}, False),
        ("(a or b) and c", {"a", "c"}, True),
        ("not a and b", {"a"}, False),  # `not` binds tighter than `and`: (not a) and b
        ("not a and b", {"b"}, True),
        ("not a or b", {"a", "b"}, True),  # ... and tighter than `or`: (not a) or b
        ("not (a and b)", {"a"}, True),
        ("not not a", {"a"}, True),
        ("a and not (b or c)", {"a", "c"}, False),
        ("((a))", {"a"}, True),
        ("", set(), True),  # Like pytest, an empty expression selects everything.
    ],
)
def test_marker_expression_matches_follows_pytest_precedence(expression, markers, expected):
  assert marker_expression_matches(expression, markers) is expected


@pytest.mark.parametrize("expression", ["a and", "(a", "a b", "not", "a & b", "and a", "a or or b", ")", "a)"])
def test_marker_expression_matches_rejects_malformed_expressions(expression):
  with pytest.raises(ValueError):
    marker_expression_matches(expression, {"a", "b"})


def test_marker_expression_matches_agrees_with_pytest():
  # Every flavor marker, and two more expressions, on every combination of the markers they use. pytest's own `-m`
  # evaluator is private API, so this cross-check skips if it moves.
  expression_module = pytest.importorskip("_pytest.mark.expression")
  expressions = [flavor.marker for flavor in TPU7X_FLAVORS.values()]
  assert len(expressions) == 4
  expressions += ["cpu_only or gpu_only and tpu_only", "not (cpu_only or tpu_only) and not gpu_only"]
  names = ("cpu_only", "gpu_only", "tpu_only", "integration_test", "post_training", "newly_added")
  for expression in expressions:
    compiled = expression_module.Expression.compile(expression)
    for mask in range(2 ** len(names)):
      markers = frozenset(name for bit, name in enumerate(names) if mask >> bit & 1)
      assert marker_expression_matches(expression, markers) == compiled.evaluate(markers.__contains__), (
          expression,
          sorted(markers),
      )


@pytest.mark.parametrize("separator", [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\xa0"])
def test_marker_expression_whitespace_follows_pytest(separator):
  # pytest's `-m` lexer separates tokens with spaces and tabs only, and rejects other whitespace such as a newline.
  # The detector must accept and reject exactly the same expressions, or it would pick tests pytest never runs.
  expression_module = pytest.importorskip("_pytest.mark.expression")
  # pytest 9 raises SyntaxError; older versions raise their own ParseError.
  rejected = (SyntaxError, getattr(expression_module, "ParseError", SyntaxError))
  expression = f"tpu_only{separator}and not cpu_only"
  try:
    expression_module.Expression.compile(expression)
  except rejected:
    with pytest.raises(ValueError):
      marker_expression_matches(expression, {"tpu_only"})
  else:
    assert marker_expression_matches(expression, {"tpu_only"})


# --- TPU7X_FLAVORS and TPU7X_PR_MATRIX against the workflows ------------------


def _load_workflow(name):
  """Return ``.github/workflows/<name>`` parsed; skip without PyYAML or the file (e.g. outside a full checkout)."""
  yaml = pytest.importorskip("yaml")
  path = _REPO_ROOT / ".github" / "workflows" / name
  if not path.is_file():
    pytest.skip(f"{path} is not in this checkout")
  return yaml.safe_load(path.read_text(encoding="utf-8"))


def _from_json_map(expression):
  """Return the JSON object of a ``${{ fromJSON('{...}')[inputs.flavor] }}`` workflow expression."""
  match = re.search(r"fromJSON\('(\{.*\})'\)", expression, re.DOTALL)
  assert match is not None, expression
  return json.loads(match.group(1))


def _ignore_paths(args):
  """Return the ``--ignore`` paths of a pytest argument string; fail on any argument the detector does not model."""
  tokens = shlex.split(args)
  ignores = []
  while tokens:
    token = tokens.pop(0)
    if token.startswith("--ignore="):
      ignores.append(token[len("--ignore=") :])
    elif token == "--ignore" and tokens:
      ignores.append(tokens.pop(0))
    else:
      pytest.fail(f"newly_added_detection does not model the pytest argument {token!r}")
  return tuple(ignores)


def test_tpu7x_flavors_mirror_run_tests_coordinator():
  inputs = _load_workflow("run_tests_coordinator.yml")["jobs"]["execute-test-package"]["with"]
  markers = _from_json_map(inputs["pytest_marker"])
  addopts = _from_json_map(inputs["pytest_addopts"])
  extra_args = _from_json_map(inputs["pytest_extra_args"])
  assert sorted(name for name in markers if name.startswith("tpu7x-")) == sorted(TPU7X_FLAVORS)
  for name, flavor in TPU7X_FLAVORS.items():
    paths = tuple(shlex.split(addopts[name]))
    assert not any(path.startswith("-") for path in paths), f"{name}: the detector does not model {addopts[name]!r}"
    assert flavor.marker == markers[name], name
    assert flavor.paths == paths, name
    assert flavor.ignores == _ignore_paths(extra_args[name]), name


def test_tpu7x_pr_matrix_is_the_ci_pipeline_fallback_matrix():
  # On PRs the matrix comes from analyze_code_changes; the literal list after `||` is used whenever it does not.
  flavor = _load_workflow("ci_pipeline.yml")["jobs"]["tpu7x-tests"]["strategy"]["matrix"]["flavor"]
  match = re.search(r"\|\|\s*'(\[[^']*\])'", flavor)
  assert match is not None, f"no literal fallback list after || in {flavor!r}"
  assert tuple(json.loads(match.group(1))) == TPU7X_PR_MATRIX
  assert set(TPU7X_PR_MATRIX) <= set(TPU7X_FLAVORS)


def _normalized(expression):
  """Return a workflow expression with its whitespace collapsed, so YAML line folding does not matter."""
  return " ".join(expression.split())


def test_ci_pipeline_wires_run_tpu7x_label():
  # Whole expressions, not substrings, so that a negation or an `&&` in place of `||` fails here.
  jobs = _load_workflow("ci_pipeline.yml")["jobs"]
  entry_if = (
      "github.event_name != 'pull_request' || "
      "(github.event.action != 'labeled' && github.event.action != 'unlabeled') || "
      "github.event.label.name == 'scheduled-only' || github.event.label.name == 'run-tpu7x' || "
      "github.event.label.name == 'skip-ci-tests'"
  )
  # A label event that skipped one of them would skip the build (and every test), or All Required Tests Passed would
  # take the run for one started by an unrelated label (it looks for Code Quality).
  for entry_job in ("analyze_code_changes", "code_quality_check", "docs_build_check", "dev_tests"):
    assert _normalized(jobs[entry_job]["if"]) == entry_if, entry_job
  has_label = "contains(github.event.pull_request.labels.*.name, 'run-tpu7x')"
  check_step = next(step for step in jobs["analyze_code_changes"]["steps"] if step.get("id") == "check")
  assert _normalized(check_step["env"]["RUN_TPU7X"]) == f"${{{{ {has_label} }}}}"
  tpu7x_job = jobs["tpu7x-tests"]
  # The heavy-test gate applies to every TPU7X run, the run-tpu7x and scheduled-only labels included.
  assert _normalized(tpu7x_job["if"]) == (
      "!cancelled() && needs.gate_test_run.result == 'success' && "
      "needs.analyze_code_changes.outputs.run_heavy_tests == 'true' && (github.event_name != 'pull_request' || "
      f"contains(github.event.pull_request.labels.*.name, 'scheduled-only') || {has_label} || "
      "needs.analyze_code_changes.outputs.has_new_tests == 'true')"
  )
  assert _normalized(tpu7x_job["with"]["is_pull_request"]) == (
      f"${{{{ github.event_name == 'pull_request' && !{has_label} && "
      "!contains(github.event.pull_request.labels.*.name, 'scheduled-only') }}"
  )


def test_ci_pipeline_runs_started_by_other_labels_cancel_nothing():
  # A run started by adding or removing a label the entry jobs ignore skips every test job. In the PR's concurrency
  # group it would cancel the PR's in-progress run, e.g. a long run-tpu7x run when someone adds gemini-review, so it
  # falls through to a group of its own (its run_id).
  workflow = _load_workflow("ci_pipeline.yml")
  entry_if = workflow["jobs"]["analyze_code_changes"]["if"]
  entry_labels = set(re.findall(r"github\.event\.label\.name == '([^']+)'", entry_if))
  group = _normalized(workflow["concurrency"]["group"])
  assert set(re.findall(r"github\.event\.label\.name == '([^']+)'", group)) == entry_labels
  assert entry_labels == {"scheduled-only", "run-tpu7x", "skip-ci-tests"}
  assert group == (
      "${{ github.event_name == 'pull_request' && "
      "((github.event.action != 'labeled' && github.event.action != 'unlabeled') || "
      "github.event.label.name == 'scheduled-only' || github.event.label.name == 'run-tpu7x' || "
      "github.event.label.name == 'skip-ci-tests') && "
      "format('{0}-pr-{1}', github.workflow, github.event.pull_request.number) || "
      "github.event_name == 'schedule' && format('{0}-schedule', github.workflow) || github.run_id }}"
  )


def _run_analyze_script(workdir, *, env=None, detector_rc=1, detector_stdout=""):
  """Run analyze_code_changes.sh on a one-commit PR, with a fake ``python3`` in place of newly_added_detection.

  The fake records its arguments, prints ``detector_stdout`` and exits ``detector_rc``. Returns the finished process,
  the GITHUB_OUTPUT writes as ``(key, value)`` pairs in order (GitHub keeps the last write of a key), and the
  arguments of every fake ``python3`` call.
  """
  script = _REPO_ROOT / ".github" / "scripts" / "analyze_code_changes.sh"
  if not script.is_file():
    pytest.skip(f"{script} is not in this checkout")
  try:
    bash_major = subprocess.run(
        ["bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True, check=False
    ).stdout.strip()
  except FileNotFoundError:
    bash_major = ""
  if not bash_major.isdigit() or int(bash_major) < 4:
    pytest.skip("analyze_code_changes.sh needs bash 4 or later (it lowercases with ${VAR,,})")
  workdir.mkdir()
  repo = _make_pr_repo(workdir, {"README.md": "base\n"}, {"README.md": "updated\n"})
  fake_bin = workdir / "bin"
  fake_bin.mkdir()
  calls = workdir / "detector-calls"
  stdout = workdir / "detector-stdout"
  stdout.write_text(detector_stdout, encoding="utf-8")
  (fake_bin / "python3").write_text(
      f"#!/bin/sh\necho \"$@\" >> '{calls}'\ncat '{stdout}'\nexit {detector_rc}\n", encoding="utf-8"
  )
  (fake_bin / "python3").chmod(0o755)
  gh_output = workdir / "github_output"
  gh_output.write_text("", encoding="utf-8")
  # Drop the CI job's own values of the variables that the script reads, so that a test sets them or gets the default.
  read_by_script = ("PR_NUMBER", "GITHUB_REPOSITORY", "FORCE_ALL_TESTS", "RUN_TPU7X")
  read_by_heavy_gate = ("EVENT_ACTION", "IS_DRAFT", "HAS_SKIP_CI_TESTS_LABEL", "IS_COPYBARA_PR")
  run_env = {key: value for key, value in os.environ.items() if key not in read_by_script + read_by_heavy_gate}
  run_env.update(
      PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
      EVENT_NAME="pull_request",
      GITHUB_BASE_REF="main",
      GITHUB_OUTPUT=str(gh_output),
      **(env or {}),
  )
  result = subprocess.run(["bash", str(script)], cwd=repo, env=run_env, capture_output=True, text=True, check=False)
  writes = [tuple(line.split("=", 1)) for line in gh_output.read_text(encoding="utf-8").splitlines() if "=" in line]
  return result, writes, calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def _written(writes, key):
  return [value for written_key, value in writes if written_key == key]


@pytest.mark.parametrize(
    "detector_rc, detector_stdout, has_new_tests, tpu7x_flavors",
    [
        pytest.param(2, "", "true", None, id="detector-cannot-diff"),
        pytest.param(1, "Traceback (most recent call last):\n", "true", None, id="detector-crashed"),
        pytest.param(0, "not json\n", "true", None, id="malformed-output"),
        pytest.param(0, '[]\n["tpu7x-unit"]\n', "true", None, id="two-lines"),
        pytest.param(0, '["gpu-unit"]\n', "true", None, id="not-a-tpu7x-flavor"),
        pytest.param(0, "[]\n", "false", None, id="no-flavor"),
        pytest.param(0, '["tpu7x-unit"]\n', "true", '["tpu7x-unit"]', id="one-flavor"),
        pytest.param(
            0,
            '["tpu7x-unit","tpu7x-integration","tpu7x-post-training-unit"]\n',
            "true",
            '["tpu7x-unit","tpu7x-integration","tpu7x-post-training-unit"]',
            id="every-flavor",
        ),
    ],
)
def test_analyze_code_changes_maps_detector_output_to_the_tpu7x_outputs(
    tmp_path, detector_rc, detector_stdout, has_new_tests, tpu7x_flavors
):
  # A failed or malformed detection runs the full matrix (fail-safe), [] skips tpu7x-tests, and a list picks flavors.
  result, writes, calls = _run_analyze_script(tmp_path / "pr", detector_rc=detector_rc, detector_stdout=detector_stdout)
  assert result.returncode == 0, result.stdout + result.stderr
  assert calls == ["tests/utils/newly_added_detection.py --base main --eligible-flavors"]
  assert _written(writes, "has_new_tests") == [has_new_tests]
  assert _written(writes, "tpu7x_flavors") == ([tpu7x_flavors] if tpu7x_flavors else [])


def test_analyze_code_changes_run_tpu7x_bypasses_detector_and_enables_all_flavors(tmp_path):
  # Case-insensitive RUN_TPU7X=True bypasses newly_added_detection, sets has_new_tests=true without
  # tpu7x_flavors, and enables run_tests even when only an excluded file (README.md) changed.
  result, writes, calls = _run_analyze_script(tmp_path / "label", env={"RUN_TPU7X": "True"})
  assert result.returncode == 0, result.stdout + result.stderr
  assert not calls
  assert _written(writes, "has_new_tests") == ["true"]
  assert not _written(writes, "tpu7x_flavors")
  assert _written(writes, "run_tests")[-1] == "true"
  assert _written(writes, "run_heavy_tests") == ["true"]
  assert _written(writes, "heavy_tests_gated") == ["false"]


@pytest.mark.parametrize(
    "env, detector_stdout, heavy_tests_gated",
    [
        # run-tpu7x asks for TPU7X tests, which the gate then skips: All Required Tests Passed must fail.
        pytest.param({"RUN_TPU7X": "true"}, "", "true", id="run-tpu7x"),
        # Nothing heavy: the README.md change selects no suite, and the detector finds no TPU7X test.
        pytest.param({}, "[]\n", "false", id="no-tpu7x-test"),
    ],
)
def test_analyze_code_changes_heavy_gate_counts_tpu7x_tests(tmp_path, env, detector_stdout, heavy_tests_gated):
  # tpu7x-tests is a heavy job: with skip-ci-tests it is skipped (run_heavy_tests=false), even with run-tpu7x, and
  # a change that needs it is reported as gated.
  result, writes, _ = _run_analyze_script(
      tmp_path / "pr",
      env={"HAS_SKIP_CI_TESTS_LABEL": "true", "EVENT_ACTION": "synchronize", **env},
      detector_rc=0,
      detector_stdout=detector_stdout,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  assert _written(writes, "run_heavy_tests") == ["false"]
  assert _written(writes, "heavy_tests_gated") == [heavy_tests_gated]
  message = _written(writes, "heavy_tests_gate_message")
  assert message and ("'skip-ci-tests' label" in message[0]) == (heavy_tests_gated == "true")


def test_detector_models_every_pytest_ini_setting_that_selects_tests():
  # The detector hardcodes python_files and the tests/ root, and reads only testpaths and --ignore from addopts. Any
  # other collection setting (--ignore-glob, --deselect, -k, python_functions, ...) would make it list tests that
  # pytest never runs, and the TPU7X job would then fail on pytest exit code 5.
  path = _REPO_ROOT / "pytest.ini"
  if not path.is_file():
    pytest.skip(f"{path} is not in this checkout")
  parser = configparser.ConfigParser(interpolation=None)
  parser.read(path, encoding="utf-8")
  ini = parser["pytest"]
  assert ini.get("python_files", "").split() == [f"*{suffix}" for suffix in _TEST_SUFFIXES]
  assert ini.get("testpaths", "").split() == [_TESTS_ROOT.rstrip("/")]
  for key in ("python_classes", "python_functions", "norecursedirs"):
    assert key not in ini, f"pytest.ini sets {key}, which newly_added_detection does not model"
  # Options that do not change which tests are collected or selected.
  harmless = {"-rf", "--import-mode=importlib", "--strict-markers"}
  args = shlex.split(ini.get("addopts", ""))
  index = 0
  while index < len(args):
    arg = args[index]
    if arg == "--ignore":
      index += 1  # its path
    else:
      assert arg.startswith("--ignore=") or arg in harmless, f"newly_added_detection does not model {arg!r}"
    index += 1


# --- What counts as touched, end to end ---------------------------------------


def _grouped_module(module_lines=(), class_decorators=(), class_line="class TestTarget:", class_lines=()):
  """Return a module with TestTarget (test_one, test_two), TestBystander (test_three) and a module-level test_free.

  ``module_lines`` follow the import. ``class_decorators`` and ``class_line`` open TestTarget, and ``class_lines``
  start its body.
  """
  return "\n".join(
      [
          "import pytest",
          "",
          *module_lines,
          "",
          "",
          *class_decorators,
          class_line,
          *class_lines,
          "",
          "  def test_one(self):",
          "    assert True",
          "",
          "  def test_two(self):",
          "    assert True",
          "",
          "",
          "class TestBystander:",
          "",
          "  def test_three(self):",
          "    assert True",
          "",
          "",
          "def test_free():",
          "  assert True",
          "",
      ]
  )


_TPU_ONLY_CLASS = {"class_decorators": ["@pytest.mark.tpu_only"]}
_TPU_ONLY_MODULE = "pytestmark = pytest.mark.tpu_only"


@pytest.mark.parametrize(
    "base, new, expected",
    [
        pytest.param({}, _TPU_ONLY_CLASS, ["test_one", "test_two"], id="class-tpu_only-added"),
        pytest.param(
            {"class_decorators": ["@pytest.mark.tpu_only", "@pytest.mark.skip_on_tpu7x"]},
            _TPU_ONLY_CLASS,
            ["test_one", "test_two"],
            id="class-skip_on_tpu7x-removed",  # A pure deletion: @@ -N +M,0 @@.
        ),
        pytest.param(
            {"class_lines": ["  pytestmark = [pytest.mark.tpu_only, pytest.mark.skip_on_tpu7x]"]},
            {"class_lines": ["  pytestmark = [pytest.mark.tpu_only]"]},
            ["test_one", "test_two"],
            id="class-body-pytestmark-edited",
        ),
        pytest.param(
            _TPU_ONLY_CLASS,
            {**_TPU_ONLY_CLASS, "class_line": "class TestTarget(object):"},
            ["test_one", "test_two"],
            id="class-header-edited",
        ),
        pytest.param(
            {**_TPU_ONLY_CLASS, "class_lines": ["  __test__ = False"]},
            _TPU_ONLY_CLASS,
            ["test_one", "test_two"],
            id="class-__test__-False-removed",
        ),
        pytest.param(
            {"module_lines": ["pytestmark = [pytest.mark.tpu_only, pytest.mark.skip_on_tpu7x]"]},
            {"module_lines": [_TPU_ONLY_MODULE]},
            ["test_free", "test_one", "test_three", "test_two"],
            id="module-pytestmark-edited",
        ),
        pytest.param(
            {"module_lines": [_TPU_ONLY_MODULE, "__test__ = False"]},
            {"module_lines": [_TPU_ONLY_MODULE]},
            ["test_free", "test_one", "test_three", "test_two"],
            id="module-__test__-False-removed",
        ),
    ],
)
def test_cli_class_and_module_marker_edits_touch_every_test_they_apply_to(tmp_path, base, new, expected):
  # Turning a whole class or module on for TPU7X must run its tests before merge, and nothing else.
  path = "tests/unit/grouped_test.py"
  repo = _make_pr_repo(
      tmp_path,
      {path: _grouped_module(**base), "tests/unit/untouched_test.py": _src("tpu_only")},
      {path: _grouped_module(**new)},
  )
  assert _ok(_run_cli(repo, "--base", "main")) == [f"{path}::{name}" for name in expected]
  assert _eligible(repo) == ["tpu7x-unit"]


def test_cli_edits_outside_tests_and_marker_lines_touch_nothing(tmp_path):
  # Helpers, fixtures, setUp, other class attributes, module constants, non-test files (conftest.py too), src/ and
  # pytest.ini never count.
  path = "tests/unit/support_test.py"
  old = textwrap.dedent(
      """\
      import pytest

      LIMIT = 1


      @pytest.fixture
      def resource():
        return LIMIT


      def helper():
        return LIMIT


      @pytest.mark.tpu_only
      class TestGroup:
        attribute = LIMIT

        def setUp(self):
          self.value = LIMIT

        def helper_method(self):
          return self.value

        def test_one(self):
          assert self.helper_method()


      @pytest.mark.tpu_only
      def test_free(resource):
        assert helper() == resource
      """
  )
  repo = _make_pr_repo(
      tmp_path,
      {
          path: old,
          "tests/conftest.py": "LIMIT = 1\n",
          "tests/unit/helpers.py": "def test_lookalike():\n  return 1\n",
          "src/pkg/module.py": "LIMIT = 1\n",
          "pytest.ini": "[pytest]\ntestpaths = tests\n",
      },
      {
          path: old.replace("LIMIT", "CAP"),
          "tests/conftest.py": "LIMIT = 2\n",
          "tests/unit/helpers.py": "def test_lookalike():\n  return 2\n",
          "src/pkg/module.py": "LIMIT = 2\n",
          "pytest.ini": "[pytest]\ntestpaths =\n    tests\n",
      },
  )
  assert not _ok(_run_cli(repo, "--base", "main"))


_FIRST = "def test_first():\n  assert 1\n"
_MIDDLE = textwrap.dedent(
    """\
    @pytest.mark.tpu_only
    @pytest.mark.skip_on_tpu7x
    def test_middle():
      value = 2
      assert value
      assert value > 1
    """
)
_LAST = "def test_last():\n  assert 3\n"
# Unchanged tests that keep a renamed copy similar enough for git to pair it with the original (-M needs 50%).
_KEPT = tuple(f"def test_kept_{index}():\n  assert {index}\n" for index in range(6))


def _join_tests(*tests):
  """Return a test module holding the given test sources, two blank lines apart."""
  return "import pytest\n\n\n" + "\n\n".join(tests)


_THREE_TESTS = {"tests/unit/three_test.py": _join_tests(_FIRST, _MIDDLE, _LAST)}
_SAME_NAME_IN_TWO_CLASSES = textwrap.dedent(
    """\
    import pytest


    @pytest.mark.tpu_only
    class TestX:

      def test_x_only(self):
        assert True

      def test_same(self):
        x_value = 1
        assert x_value


    @pytest.mark.tpu_only
    class TestY:

      def test_same(self):
        assert True
    """
)
_X_TEST_SAME = "\n  def test_same(self):\n    x_value = 1\n    assert x_value\n"


@pytest.mark.parametrize(
    "base, change",
    [
        pytest.param(_THREE_TESTS, {"tests/unit/three_test.py": _join_tests(_MIDDLE, _LAST)}, id="first-test"),
        pytest.param(_THREE_TESTS, {"tests/unit/three_test.py": _join_tests(_FIRST, _LAST)}, id="middle-test"),
        pytest.param(_THREE_TESTS, {"tests/unit/three_test.py": _join_tests(_FIRST, _MIDDLE)}, id="last-test"),
        pytest.param(
            {"tests/unit/three_test.py": _join_tests(*_KEPT, _FIRST, _MIDDLE, _LAST)},
            {"tests/unit/three_test.py": None, "tests/unit/moved_test.py": _join_tests(*_KEPT, _FIRST, _LAST)},
            id="middle-test-of-a-renamed-file",
        ),
        pytest.param(_THREE_TESTS, {"tests/unit/three_test.py": None}, id="whole-file"),
        pytest.param(
            # Rule (b) matches the class as well as the name: TestY.test_same is untouched.
            {"tests/unit/classes_test.py": _SAME_NAME_IN_TWO_CLASSES},
            {"tests/unit/classes_test.py": _SAME_NAME_IN_TWO_CLASSES.replace(_X_TEST_SAME, "")},
            id="test-whose-name-another-class-also-uses",
        ),
    ],
)
def test_cli_deleting_a_test_flags_neither_neighbour(tmp_path, base, change):
  # A deleted test leaves nothing to run, and its untouched neighbours must not start TPU7X runners.
  repo = _make_pr_repo(tmp_path, base, change)
  assert not _ok(_run_cli(repo, "--base", "main"))


@pytest.mark.parametrize(
    "middle",
    [
        pytest.param(_MIDDLE.replace("@pytest.mark.skip_on_tpu7x\n", ""), id="decorator-right-above-the-def"),
        pytest.param(_MIDDLE.replace("@pytest.mark.tpu_only\n", ""), id="first-decorator"),
        pytest.param(_MIDDLE.replace("  assert value\n  assert value > 1\n", ""), id="trailing-body-lines"),
    ],
)
def test_cli_deleting_lines_of_a_test_flags_that_test(tmp_path, middle):
  path = "tests/unit/three_test.py"
  repo = _make_pr_repo(tmp_path, _THREE_TESTS, {path: _join_tests(_FIRST, middle, _LAST)})
  assert _ok(_run_cli(repo, "--base", "main")) == [f"{path}::test_middle"]


def test_cli_deletions_fall_back_to_the_lines_around_the_cut_when_the_base_does_not_parse(tmp_path):
  # The PR deletes the line that broke the file. With no parseable base version, the deletion is attributed to the
  # new-file lines on both sides of it, which may flag a neighbour but never misses the edited test.
  path = "tests/unit/fixed_test.py"
  broken = "import pytest\n\n\n@pytest.mark.tpu_only\ndef test_fixed():\n  x = (\n  assert True\n"
  repo = _make_pr_repo(tmp_path, {path: broken}, {path: broken.replace("  x = (\n", "")})
  assert _ok(_run_cli(repo, "--base", "main")) == [f"{path}::test_fixed"]


_NOT_TESTS_SOURCE = textwrap.dedent(
    """\
    import functools

    import pytest
    from pytest import fixture

    pytestmark = pytest.mark.tpu_only


    @pytest.fixture
    def test_data():
      return 1


    @pytest.fixture(scope="module")
    def test_module_data():
      return 2


    @fixture
    def test_imported_fixture():
      return 3


    class TestValues:

      @property
      def test_value(self):
        return 4

      @functools.cached_property
      def test_cached(self):
        return 5

      def test_real(self, test_data):
        assert test_data == 1
    """
)


def test_cli_and_pytest_agree_that_fixtures_and_properties_named_test_are_not_tests(tmp_path):
  # pytest collects no fixture or property, whatever its name. If the detector listed one, its TPU7X flavor would
  # start, select nothing and fail on pytest's exit code 5.
  path = "tests/unit/not_tests_test.py"
  change = _NOT_TESTS_SOURCE
  for value in range(1, 6):
    change = change.replace(f"return {value}\n", f"return {value * 10}\n")
  repo = _make_pr_repo(tmp_path, {path: _NOT_TESTS_SOURCE}, {path: change})
  assert not _ok(_run_cli(repo, "--base", "main"))
  assert _eligible(repo) == []
  # pytest itself collects only test_real from this module.
  env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
  env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
  result = subprocess.run(
      [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", path],
      cwd=repo,
      env=env,
      capture_output=True,
      text=True,
      check=False,
  )
  assert [line for line in result.stdout.splitlines() if "::" in line] == [f"{path}::TestValues::test_real"], (
      result.stdout + result.stderr
  )


# --- Encodings, user git config, paths ----------------------------------------


def test_cli_ignores_non_utf8_files_outside_tests(tmp_path):
  # A latin-1 file anywhere in the PR must not make the diff undecodable (which would exit 2).
  repo = _make_pr_repo(tmp_path, {"tests/unit/sample_test.py": _src()}, {"src/notes.txt": b"cr\xe8me br\xfbl\xe9e\n"})
  assert not _ok(_run_cli(repo, "--base", "main"))
  assert not _eligible(repo)


def test_cli_reads_non_utf8_test_sources_and_diff_lines(tmp_path):
  latin1_test = (
      b"# -*- coding: latin-1 -*-\n"
      b"import pytest\n\n\n"
      b"@pytest.mark.tpu_only\n"
      b"def test_latin1_source():\n"
      b"  assert '\xe9t\xe9'\n"
  )
  repo = _make_pr_repo(
      tmp_path,
      {"tests/unit/sample_test.py": _src()},
      {
          "tests/assets/latin1.txt": b"\xe9t\xe9\n",
          "tests/unit/latin1_test.py": latin1_test,
          "tests/unit/clean_test.py": _src("tpu_only", name="test_clean"),
      },
  )
  assert _ok(_run_cli(repo, "--base", "main")) == [
      "tests/unit/clean_test.py::test_clean",
      "tests/unit/latin1_test.py::test_latin1_source",
  ]
  assert _eligible(repo) == ["tpu7x-unit"]


def test_cli_reports_test_files_with_non_ascii_names(tmp_path):
  # By default git prints such a path C-quoted: "b/tests/unit/r\303\251sum\303\251_test.py".
  path = "tests/unit/r\u00e9sum\u00e9_test.py"
  repo = _make_pr_repo(tmp_path, {}, {path: _src("tpu_only")})
  assert _ok(_run_cli(repo, "--base", "main")) == [f"{path}::test_case"]


_HOSTILE_OLD = _join_tests("def test_a():\n  assert 1\n", "def test_b():\n  assert 2\n", "def test_c():\n  assert 3\n")
# A rename, plus edits to test_a and test_c that are 8 lines apart, with test_b untouched between them.
_HOSTILE_PR = {
    "tests/unit/old_name_test.py": None,
    "tests/unit/new_name_test.py": _HOSTILE_OLD.replace("assert 1", "assert 10").replace("assert 3", "assert 30"),
    "tests/unit/added_test.py": _src(name="test_added"),
}
_HOSTILE_EXPECTED = [
    "tests/unit/added_test.py::test_added",
    "tests/unit/new_name_test.py::test_a",
    "tests/unit/new_name_test.py::test_c",
]
# Setting name -> (git config entries, environment variables); "{home}" is the temporary HOME.
_HOSTILE_SETTINGS = {
    "none": ({}, {}),
    "color": ({"color.ui": "always"}, {}),
    "noprefix": ({"diff.noprefix": "true"}, {}),
    "mnemonicprefix": ({"diff.mnemonicPrefix": "true"}, {}),
    "custom-prefixes": ({"diff.srcPrefix": "SRC/", "diff.dstPrefix": "DST/"}, {}),
    "external-diff": ({"diff.external": "{home}/external-diff.sh"}, {}),
    "textconv": ({"core.attributesFile": "{home}/attributes", "diff.hostile.textconv": "{home}/textconv.sh"}, {}),
    "no-renames": ({"diff.renames": "false"}, {}),
    "inter-hunk-context": ({"diff.interHunkContext": "10"}, {}),
    "GIT_DIFF_OPTS": ({}, {"GIT_DIFF_OPTS": "--unified=3"}),
    "GIT_EXTERNAL_DIFF": ({}, {"GIT_EXTERNAL_DIFF": "{home}/external-diff.sh"}),
}


@pytest.fixture(name="rename_repo", scope="module")
def rename_repo_fixture(tmp_path_factory):
  """A PR that renames a test file, edits two of its three tests, and adds a test file."""
  return _make_pr_repo(tmp_path_factory.mktemp("rename"), {"tests/unit/old_name_test.py": _HOSTILE_OLD}, _HOSTILE_PR)


@pytest.mark.parametrize("setting", [*_HOSTILE_SETTINGS, "all"])
def test_cli_output_does_not_depend_on_user_git_config(rename_repo, tmp_path, setting):
  # Each of these settings could otherwise change what the detector reports, often to "nothing" with exit 0.
  home = tmp_path / "home"
  home.mkdir()
  scripts = {"external-diff.sh": 'echo "external diff: $*"\n', "textconv.sh": 'echo injected\necho injected\ncat "$1"\n'}
  for name, body in scripts.items():
    (home / name).write_text("#!/bin/sh\n" + body, encoding="utf-8")
    (home / name).chmod(0o755)
  (home / "attributes").write_text("*.py diff=hostile\n", encoding="utf-8")
  if setting == "all":
    config = {key: value for entries, _ in _HOSTILE_SETTINGS.values() for key, value in entries.items()}
    variables = {key: value for _, entries in _HOSTILE_SETTINGS.values() for key, value in entries.items()}
  else:
    config, variables = _HOSTILE_SETTINGS[setting]
  gitconfig = home / ".gitconfig"
  gitconfig.touch()
  for key, value in config.items():
    _git(home, "config", "--file", str(gitconfig), key, value.format(home=home))
  # Only this HOME's config applies: no system config, and no GIT_* variables from the outer environment.
  env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
  env.update({key: value.format(home=home) for key, value in variables.items()})
  env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), GIT_CONFIG_NOSYSTEM="1")
  assert _ok(_run_cli(rename_repo, "--base", "main", env=env)) == _HOSTILE_EXPECTED


def test_cli_fetches_the_base_from_origin_under_github_actions(tmp_path):
  path = "tests/unit/sample_test.py"
  repo = _make_pr_repo(tmp_path, {path: _src()}, {path: _src() + "\n\ndef test_new():\n  assert True\n"})
  # Forget origin/main, as in a checkout that fetched only the PR head: now no ref names the base.
  _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
  env = {key: value for key, value in os.environ.items() if key != "GITHUB_ACTIONS"}
  assert _run_cli(repo, "--base", "main", env=env).returncode == 2
  assert _ok(_run_cli(repo, "--base", "main", env={**env, "GITHUB_ACTIONS": "true"})) == [f"{path}::test_new"]


def test_cli_reports_nothing_for_an_empty_net_diff(tmp_path):
  # An empty "retrigger" commit, or commits that cancel out: exit 0 and [], so TPU7X is skipped, not failed.
  repo = _make_pr_repo(tmp_path, {"tests/unit/sample_test.py": _src("tpu_only")}, {})
  assert not _ok(_run_cli(repo, "--base", "main"))
  assert not _eligible(repo)


# --- --all-tests --------------------------------------------------------------


def test_cli_all_tests_reads_every_test_file_without_git(tmp_path):
  project = tmp_path / "project"
  _write_files(
      project,
      {
          "tests/unit/a_test.py": _src("tpu_only", name="test_tpu") + "\n\ndef test_cpu():\n  assert True\n",
          "tests/post_training/unit/b_test.py": _src("post_training", "tpu_only", name="test_post_training"),
          "tests/unit/helpers.py": "def test_lookalike():\n  pass\n",
          "src/pkg/c_test.py": "def test_outside_tests():\n  pass\n",
      },
  )
  # A git that records every call: --all-tests must not run git at all, so it works without a repository.
  fake_bin = tmp_path / "bin"
  fake_bin.mkdir()
  calls = tmp_path / "git-calls"
  (fake_bin / "git").write_text(f"#!/bin/sh\necho \"$@\" >> '{calls}'\nexit 1\n", encoding="utf-8")
  (fake_bin / "git").chmod(0o755)
  env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}", "GITHUB_ACTIONS": "true"}
  assert _ok(_run_cli(project, "--all-tests", env=env)) == [
      "tests/post_training/unit/b_test.py::test_post_training",
      "tests/unit/a_test.py::test_cpu",
      "tests/unit/a_test.py::test_tpu",
  ]
  assert _ok(_run_cli(project, "--all-tests", "--flavor", "tpu7x-unit", env=env)) == ["tests/unit/a_test.py::test_tpu"]
  assert _ok(_run_cli(project, "--all-tests", "--eligible-flavors", env=env)) == [
      '["tpu7x-unit","tpu7x-post-training-unit"]'
  ]
  assert not calls.exists()


# --- --collection-skips (the TPU7X pull-request exit-5 check) --------------------

_SKIPPING_MODULE = (
    "import pytest\n\npytest.skip('JetStream is not installed', allow_module_level=True)\n\n\n"
    "@pytest.mark.tpu_only\ndef test_decode():\n  pass\n"
)
_PLAIN_MODULE = "import pytest\n\n\n@pytest.mark.tpu_only\ndef test_plain():\n  pass\n"
# pytest.ini makes the project the rootdir, as the repository root is in CI, so modules get dotted names from it.
_COLLECTION_SKIP_FILES = {
    "pytest.ini": "[pytest]\nmarkers =\n    tpu_only: TPU tests\n    newly_added: touched by the PR\n",
    "tests/unit/decode_test.py": _SKIPPING_MODULE,
    "tests/unit/plain_test.py": _PLAIN_MODULE,
}


def _junit_of_real_pytest_run(project, files, *pytest_args):
  """Write ``files`` under ``project``, run a plain pytest there with ``--junitxml``, and return the XML's path."""
  _write_files(project, files)
  env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
  env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
  subprocess.run(
      [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--junitxml=results.xml", *pytest_args],
      cwd=project,
      env=env,
      capture_output=True,
      text=True,
      check=False,
  )
  return project / "results.xml"


def test_collection_skips_accepts_listed_modules_that_skipped_themselves_at_collection(tmp_path):
  # A TPU7X PR run selects nothing when every listed test's module skips itself during collection, e.g. when JetStream
  # is not installed. pytest's JUnit file records such a module as a testcase with an empty classname and <skipped>.
  project = tmp_path / "project"
  junit = _junit_of_real_pytest_run(project, _COLLECTION_SKIP_FILES, "-m", "tpu_only and newly_added", "tests")
  result = _run_cli(project, "--collection-skips", str(junit), stdin_text="tests/unit/decode_test.py::test_decode\n")
  assert result.returncode == 0, result.stdout + result.stderr
  assert "tests.unit.decode_test: " in result.stdout
  assert "JetStream is not installed" in result.stdout


def test_collection_skips_rejects_a_listed_module_that_did_not_skip(tmp_path):
  # A listed test that pytest collected but did not select means the newly_added marking disagrees with the detector.
  project = tmp_path / "project"
  junit = _junit_of_real_pytest_run(project, _COLLECTION_SKIP_FILES, "-m", "tpu_only and newly_added", "tests")
  listed = "tests/unit/decode_test.py::test_decode\ntests/unit/plain_test.py::test_plain\n"
  result = _run_cli(project, "--collection-skips", str(junit), stdin_text=listed)
  assert result.returncode == 1, result.stdout + result.stderr
  assert "tests.unit.plain_test" in result.stderr


def test_collection_skips_ignores_skips_of_single_tests(tmp_path):
  # A skipped test (not a skipped module) has a classname: it is not a module that skipped itself during collection.
  junit = tmp_path / "results.xml"
  junit.write_text(
      '<testsuites><testsuite name="pytest"><testcase classname="tests.unit.decode_test" name="test_decode">'
      '<skipped message="skip" /></testcase></testsuite></testsuites>',
      encoding="utf-8",
  )
  result = _run_cli(tmp_path, "--collection-skips", str(junit), stdin_text="tests/unit/decode_test.py::test_decode\n")
  assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize(
    "xml, listed, message",
    [
        pytest.param(None, "tests/unit/decode_test.py::test_decode\n", "cannot read the JUnit file", id="missing-file"),
        pytest.param(
            "<testsuites><testcase",
            "tests/unit/decode_test.py::test_decode\n",
            "cannot read the JUnit file",
            id="malformed",
        ),
        pytest.param(
            '<testsuites><testcase classname="" name="tests.unit.decode_test"><skipped /></testcase></testsuites>',
            "\n",
            "no path::test_name line on stdin",
            id="nothing-listed",
        ),
    ],
)
def test_collection_skips_fails_when_it_cannot_tell(tmp_path, xml, listed, message):
  junit = tmp_path / "results.xml"
  if xml is not None:
    junit.write_text(xml, encoding="utf-8")
  result = _run_cli(tmp_path, "--collection-skips", str(junit), stdin_text=listed)
  assert result.returncode == 1, result.stdout + result.stderr
  assert message in result.stderr
  assert not result.stdout


# --- item_test_names / is_changed_item (the tests/conftest.py match) ----------


def test_is_changed_item_matches_absl_parameterized_methods_by_their_def_name():
  # pytest collects an absl parameterized def only under generated names, so matching on item.name or
  # item.originalname alone never marks it newly_added, and its TPU7X job would select nothing.
  parameterized = pytest.importorskip("absl.testing.parameterized")

  # Built here, not at module level, where pytest would collect it as real tests.
  class AbslCase(parameterized.TestCase):
    """absl replaces each parameterized def with generated methods: test_named_bf16, test_plain0, ..."""

    @parameterized.named_parameters(("bf16", 1), ("f32", 2))
    def test_named(self, x):
      del x

    @parameterized.parameters(1, 2)
    def test_plain(self, x):
      del x

    @parameterized.product(a=[1, 2], b=[3])
    def test_product(self, a, b):
      del a, b

    def test_regular(self):
      pass

  path = "tests/unit/absl_case_test.py"
  defs = ("test_named", "test_plain", "test_product", "test_regular")
  generated = sorted(name for name in vars(AbslCase) if name.startswith("test"))
  assert len(generated) == 7 and not set(defs[:3]) & set(generated), generated
  for name in generated:
    def_name = max((candidate for candidate in defs if name.startswith(candidate)), key=len)
    # pytest collects a generated method under its generated name, as both name and originalname.
    item = types.SimpleNamespace(
        nodeid=f"{path}::AbslCase::{name}", name=name, originalname=name, function=getattr(AbslCase, name)
    )
    assert def_name in item_test_names(item), name
    assert is_changed_item(item, {(path, def_name)}), name
    assert not is_changed_item(item, {(path, other) for other in defs if other != def_name}), name
    assert not is_changed_item(item, {("tests/unit/other_test.py", def_name)}), name


def test_is_changed_item_matches_pytest_parametrize_items_by_originalname():
  path = "tests/unit/param_test.py"
  item = types.SimpleNamespace(nodeid=f"{path}::test_param[1-a]", name="test_param[1-a]", originalname="test_param")
  assert item_test_names(item) == {"test_param", "test_param[1-a]"}
  assert is_changed_item(item, {(path, "test_param")})
  assert not is_changed_item(item, {(path, "test_other"), ("tests/unit/other_test.py", "test_param")})


def test_item_test_names_tolerates_missing_attributes_and_wrapper_loops():
  bare = types.SimpleNamespace(nodeid="tests/unit/doc_test.py::doc", name="doc")
  assert item_test_names(bare) == {"doc"}

  def looping():
    pass

  looping.__wrapped__ = looping  # inspect.unwrap raises ValueError on a cycle.
  item = types.SimpleNamespace(
      nodeid="tests/unit/loop_test.py::test_loop", name="test_loop", originalname="test_loop", function=looping
  )
  assert "test_loop" in item_test_names(item)
  assert is_changed_item(item, {("tests/unit/loop_test.py", "test_loop")})
  assert not is_changed_item(item, set())


_ABSL_TEST_MODULE = textwrap.dedent(
    """\
    import pytest
    from absl.testing import parameterized


    class AbslCase(parameterized.TestCase):

      @parameterized.named_parameters(("bf16", 1), ("f32", 2))
      def test_named(self, x):
        del x

      @parameterized.parameters(1, 2)
      def test_plain(self, x):
        del x

      @parameterized.product(a=[1, 2], b=[3])
      def test_product(self, a, b):
        del a, b

      def test_regular(self):
        pass


    @pytest.mark.parametrize("x", [1, 2])
    def test_param(x):
      del x


    def test_other():
      pass
    """
)
# Marks items newly_added the way tests/conftest.py does, with the detector imported from DETECTOR_DIR.
_MARKING_CONFTEST = textwrap.dedent(
    """\
    import json
    import os
    import sys

    sys.path.insert(0, os.environ["DETECTOR_DIR"])
    import newly_added_detection

    CHANGED = {tuple(pair) for pair in json.loads(os.environ["CHANGED_TESTS"])}


    def pytest_collection_modifyitems(items):
      for item in items:
        if newly_added_detection.is_changed_item(item, CHANGED):
          item.add_marker("newly_added")
    """
)


def test_is_changed_item_marks_every_generated_item_in_a_real_pytest_session(tmp_path):
  pytest.importorskip("absl.testing.parameterized")
  project = tmp_path / "project"
  path = "tests/unit/absl_case_test.py"
  _write_files(
      project,
      {
          "pytest.ini": "[pytest]\npython_files = *_test.py\nmarkers =\n    newly_added: touched by the PR\n",
          "conftest.py": _MARKING_CONFTEST,
          path: _ABSL_TEST_MODULE,
      },
  )
  changed = [[path, name] for name in ("test_named", "test_plain", "test_product", "test_param")]
  # A plain pytest: no plugins or options from the outer session.
  env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
  env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", DETECTOR_DIR=os.path.dirname(_SCRIPT), CHANGED_TESTS=json.dumps(changed))
  result = subprocess.run(
      [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", "newly_added"],
      cwd=project,
      env=env,
      capture_output=True,
      text=True,
      check=False,
  )
  expected = [
      f"{path}::AbslCase::test_named_bf16",
      f"{path}::AbslCase::test_named_f32",
      f"{path}::AbslCase::test_plain0",
      f"{path}::AbslCase::test_plain1",
      f"{path}::AbslCase::test_product0",
      f"{path}::AbslCase::test_product1",
      f"{path}::test_param[1]",
      f"{path}::test_param[2]",
  ]
  assert sorted(line for line in result.stdout.splitlines() if "::" in line) == expected, result.stdout + result.stderr
