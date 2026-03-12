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

"""Unit tests for RL result parsing and reward scoring (CPU-only)."""

import unittest
import pytest
from types import SimpleNamespace

pytestmark = [pytest.mark.post_training]

evaluate_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.evaluate_rl",
    reason="tunix (required by evaluate_rl) is not installed GPU",
)

utils_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.utils_rl",
    reason="tunix (required by utils_rl) is not installed GPU",
)


def _make_config():
  """Create a minimal config object with the parameters required by score_responses."""
  return SimpleNamespace(
      reasoning_start_token="<reasoning>",
      reasoning_end_token="</reasoning>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
      reward_exact_format_match=2.0,
      reward_partial_format_match=0.5,
      reward_white_space_format_match=1.5,
      reward_ratio_guess_to_answer_high=1.0,
      reward_ratio_guess_to_answer_low=0.5,
      penalty_incorrect_format=-0.5,
      penalty_incorrect_answer=-0.5,
      dataset_name="test",
      debug=SimpleNamespace(rl=False),
  )


class TestScoreResponses(unittest.TestCase):
  """Tests for evaluate_rl.score_responses parsing and correctness logic."""

  def setUp(self):
    self.config = _make_config()

  @pytest.mark.cpu_only
  def test_nested_tags(self):
    """Response with nested reasoning tags still extracts the correct answer."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question="What is 11/3?",
        responses=[
            "<reasoning>Need to use <reasoning> and </reasoning>, "
            "<answer> and </answer></reasoning><answer>11/3</answer>"
        ],
        answer="11/3",
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  @pytest.mark.cpu_only
  def test_with_extra_ending_tags(self):
    """Answer with extra ending tags such as <end_of_turn>."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question=(
            "James buys a new wardrobe.  He buys 10 suits and 10 dress pants.  "
            "He also buys 3 dress shirts per suit.  The suits cost $750 each and "
            "the dress pants cost 1/5 that cost.  The dress shirts were $60 each.  "
            "How much did everything cost?"
        ),
        responses=[
            "<reasoning>This is the sum of the cost of the suits, the pants, and the "
            "shirts: $7500 + $1500 + $1800 = $10800.\n\n</reasoning>\n"
            "<answer>10800</answer><end_of_turn>"
        ],
        answer="10,800",
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  @pytest.mark.cpu_only
  def test_with_incomplete_reasoning_tags(self):
    """(1) Incomplete reasoning tags still extracts the correct answer."""
    """(2) Currency symbols works with math_verify."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question="What is the price of the item?",
        responses=["<reasoning>The item costs $16.<answer>$16</answer>"],
        answer="16",
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)


class TestNormalizeFinalAnswer(unittest.TestCase):
  """Tests for utils_rl.normalize_final_answer."""

  @pytest.mark.cpu_only
  def test_comma_boxed_and_currency(self):
    # Comma-separated numbers, \\boxed{}, and leading $ are all normalized to plain integers
    self.assertEqual(utils_rl.normalize_final_answer("1,000"), "1000")
    self.assertEqual(utils_rl.normalize_final_answer("$1,000"), "1000")
    self.assertEqual(utils_rl.normalize_final_answer("\\boxed{1,000}"), "1000")

  @pytest.mark.cpu_only
  def test_equation_splitting_and_unit_removal(self):
    # Expressions with '=' are split on '='; trailing unit words are stripped
    self.assertEqual(utils_rl.normalize_final_answer("x = 10"), "10")
    self.assertEqual(utils_rl.normalize_final_answer("total = 100 meters"), "100")
    self.assertEqual(utils_rl.normalize_final_answer("42 mph"), "42")

  @pytest.mark.cpu_only
  def test_latex_wrappers(self):
    # \\text{}, \\textbf{}, and \\overline{} wrappers are removed, leaving inner content
    self.assertEqual(utils_rl.normalize_final_answer("\\text{hello}"), "hello")
    self.assertEqual(utils_rl.normalize_final_answer("\\textbf{42}"), "42")
    self.assertEqual(utils_rl.normalize_final_answer("\\overline{AB}"), "AB")

  @pytest.mark.cpu_only
  def test_dollar_math_extraction(self):
    # Content inside $...$ is extracted
    self.assertEqual(utils_rl.normalize_final_answer("The answer is $\\frac{1}{2}$"), "\\frac{1}{2}")

  @pytest.mark.cpu_only
  def test_shorthand_frac_and_sqrt(self):
    # Shorthand \\fracab and \\sqrta are expanded to their full LaTeX forms
    self.assertEqual(utils_rl.normalize_final_answer("\\fracab"), "\\frac{a}{b}")
    self.assertEqual(utils_rl.normalize_final_answer("\\sqrta"), "\\sqrt{a}")


class TestMatchFormatApproximatelyScores(unittest.TestCase):
  """Tests for utils_rl.match_format_approximately.

  Each tag that appears exactly once contributes reward_partial_format_match (0.5).
  Each tag that is absent or appears more than once contributes penalty_incorrect_format (-0.5).
  With 4 tags the score ranges from -2.0 (all wrong) to 2.0 (all correct).
  """

  def setUp(self):
    self.config = _make_config()

  def _score(self, completion):
    return utils_rl.match_format_approximately(None, completion, self.config)

  @pytest.mark.cpu_only
  def test_score_all_tags_present_exactly_once(self):
    # All four tags present exactly once -> 4 * 0.5 = 2.0
    self.assertEqual(self._score(["<reasoning>think</reasoning><answer>42</answer>"])[0], 2.0)

  @pytest.mark.cpu_only
  def test_score_no_tags_present(self):
    # No tags at all -> 4 * -0.5 = -2.0
    self.assertEqual(self._score(["The answer is 42."])[0], -2.0)

  @pytest.mark.cpu_only
  def test_score_only_answer_tags_present(self):
    # Only <answer>...</answer> present -> 2 * 0.5 + 2 * -0.5 = 0.0
    self.assertEqual(self._score(["<answer>42</answer>"])[0], 0.0)

  @pytest.mark.cpu_only
  def test_score_duplicate_reasoning_start_tag(self):
    # Duplicate <reasoning> tag -> 3 * 0.5 + 1 * -0.5 = 1.0
    self.assertEqual(self._score(["<reasoning><reasoning>think</reasoning><answer>42</answer>"])[0], 1.0)

  @pytest.mark.cpu_only
  def test_score_multiple_completions(self):
    # Multiple completions at once -> one score per entry
    multi_completions = [
        "<reasoning>think</reasoning><answer>42</answer>",  # 2.0
        "no tags here",  # -2.0
    ]
    scores = self._score(multi_completions)
    self.assertEqual(len(scores), 2)
    self.assertEqual(scores[0], 2.0)
    self.assertEqual(scores[1], -2.0)


class TestExtractHashAnswer(unittest.TestCase):
  """Tests for utils_rl.extract_hash_answer."""

  @pytest.mark.cpu_only
  def test_with_hash(self):
    """Test extraction when #### is present."""
    self.assertEqual(utils_rl.extract_hash_answer("The answer is #### 42"), "42")
    self.assertEqual(utils_rl.extract_hash_answer("Some reasoning ####   123.45  "), "123.45")
    self.assertEqual(utils_rl.extract_hash_answer("####"), "")

  @pytest.mark.cpu_only
  def test_without_hash(self):
    """Test extraction when #### is not present."""
    self.assertIsNone(utils_rl.extract_hash_answer("The answer is 42"))
    self.assertIsNone(utils_rl.extract_hash_answer(""))


if __name__ == "__main__":
  unittest.main()
