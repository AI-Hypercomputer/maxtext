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

import textwrap

import pytest

from tests.utils.newly_added_detection import _build_diff_commands
from tests.utils.newly_added_detection import _is_test_file
from tests.utils.newly_added_detection import parse_changed_line_map
from tests.utils.newly_added_detection import touched_test_names

pytestmark = pytest.mark.cpu_only


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
