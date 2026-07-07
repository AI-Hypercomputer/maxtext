# Copyright 2023-2026 Google LLC
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

"""Unit tests for utils_rl.extract_answer (CPU-only).

Covers the two-part contract of the boxed-extraction change:
  1. `\\boxed{N}` is extracted (with/without <answer> tags, nested LaTeX,
     multiple boxed, whitespace, negatives, and answer-tag scoping).
  2. Legacy plain-text answers inside the solution tags still work, so
     existing recipes that do not emit `\\boxed` are unaffected.
"""

import unittest
from types import SimpleNamespace

import pytest

from maxtext.trainers.post_train.rl import utils_rl

pytestmark = [pytest.mark.post_training]


def _make_config():
  """Minimal config carrying the solution/reasoning tokens extract_answer reads."""
  return SimpleNamespace(
      reasoning_start_token="<reasoning>",
      reasoning_end_token="</reasoning>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
  )


class ExtractAnswerTest(unittest.TestCase):
  """Verify boxed extraction and legacy-fallback behavior of extract_answer."""

  def setUp(self):
    super().setUp()
    self.config = _make_config()

  # ---- boxed extraction ----

  @pytest.mark.cpu_only
  def test_boxed_inside_answer_tags(self):
    got = utils_rl.extract_answer("<reasoning>2+2</reasoning><answer>\\boxed{4}</answer>", self.config)
    self.assertEqual(got, "4")

  @pytest.mark.cpu_only
  def test_boxed_without_answer_tags(self):
    got = utils_rl.extract_answer("the result is \\boxed{42}", self.config)
    self.assertEqual(got, "42")

  @pytest.mark.cpu_only
  def test_boxed_nested_latex(self):
    """Brace-balanced scan keeps the full nested LaTeX content."""
    got = utils_rl.extract_answer("<answer>\\boxed{\\frac{1}{2}}</answer>", self.config)
    self.assertEqual(got, "\\frac{1}{2}")

  @pytest.mark.cpu_only
  def test_multiple_boxed_returns_last(self):
    got = utils_rl.extract_answer("first \\boxed{1} then \\boxed{99}", self.config)
    self.assertEqual(got, "99")

  @pytest.mark.cpu_only
  def test_boxed_strips_whitespace(self):
    got = utils_rl.extract_answer("<answer>\\boxed{ 7 }</answer>", self.config)
    self.assertEqual(got, "7")

  @pytest.mark.cpu_only
  def test_boxed_negative(self):
    got = utils_rl.extract_answer("answer: \\boxed{-3}", self.config)
    self.assertEqual(got, "-3")

  @pytest.mark.cpu_only
  def test_answer_tag_scopes_over_reasoning_boxed(self):
    """A boxed value in <reasoning> must not win over the one in <answer>."""
    resp = "<reasoning>maybe \\boxed{1}</reasoning><answer>\\boxed{8}</answer>"
    self.assertEqual(utils_rl.extract_answer(resp, self.config), "8")

  @pytest.mark.cpu_only
  def test_scoping_follows_configured_solution_tokens(self):
    """Scoping uses solution_start/end_token, not a hardcoded <answer> tag."""
    config = SimpleNamespace(
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<sol>",
        solution_end_token="</sol>",
    )
    resp = "<reasoning>maybe \\boxed{1}</reasoning><sol>\\boxed{8}</sol>"
    self.assertEqual(utils_rl.extract_answer(resp, config), "8")

  # ---- legacy fallback (no boxed) ----

  @pytest.mark.cpu_only
  def test_legacy_plain_answer_in_tags(self):
    """A plain-text answer inside <answer> tags (no boxed) still extracts."""
    got = utils_rl.extract_answer("<reasoning>work</reasoning><answer>42</answer>", self.config)
    self.assertEqual(got, "42")

  @pytest.mark.cpu_only
  def test_legacy_last_answer_wins(self):
    got = utils_rl.extract_answer("<answer>1</answer> ... <answer>5</answer>", self.config)
    self.assertEqual(got, "5")

  # ---- no answer ----

  @pytest.mark.cpu_only
  def test_no_answer_returns_fallback_constant(self):
    got = utils_rl.extract_answer("I have no idea", self.config)
    self.assertEqual(got, utils_rl.FALLBACK_ANSWER)


if __name__ == "__main__":
  unittest.main()
