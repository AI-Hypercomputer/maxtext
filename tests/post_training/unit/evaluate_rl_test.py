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

"""Unit tests for evaluate_rl.py (CPU-only)."""

import unittest
import pytest
from types import SimpleNamespace

from maxtext.trainers.post_train.rl import evaluate_rl

pytestmark = [pytest.mark.post_training]


def _make_config(eval_mode="pass"):
  """Create a minimal config object with the parameters required by score_responses."""
  return SimpleNamespace(
      reasoning_start_token="<reasoning>",
      reasoning_end_token="</reasoning>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
      reward_exact_answer=3.0,
      reward_exact_format_match=2.0,
      reward_partial_format_match=0.5,
      reward_white_space_format_match=1.5,
      reward_ratio_guess_to_answer_high=1.0,
      reward_ratio_guess_to_answer_low=0.5,
      penalty_incorrect_format=-0.5,
      penalty_incorrect_answer=-0.5,
      dataset_name="test",
      debug=SimpleNamespace(rl=False),
      eval_mode=eval_mode,
  )


class TestScoreResponses(unittest.TestCase):
  """Tests for evaluate_rl.score_responses parsing and correctness logic."""

  def setUp(self):
    self.config = _make_config(eval_mode="pass")
    self.maj_config = _make_config(eval_mode="maj")

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
        answers=["11/3"],
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
        answers=["10,800"],
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
        answers=["16"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)

  @pytest.mark.cpu_only
  def test_for_mcq_value(self):
    """Test for MCQ, where model responds with a math value."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question=(
            r"What is the quantity of the item? "
            r"(A) $2 \frac{1}{3}$ (B) $3 \frac{1}{3}$ "
            r"(C) $1 \frac{2}{3}$ (D) $1 \frac{1}{3}$ (E) 2"
        ),
        responses=["<reasoning>The answer is 3\frac{1}{3}.<answer>3\frac{1}{3}</answer>"],
        answers=["3\frac{1}{3}", "B"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)

  @pytest.mark.cpu_only
  def test_for_mcq_option(self):
    """Test for MCQ, where model responds with an option."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question=(
            r"What is the quantity of the item? "
            r"(A) $2 \frac{1}{3}$ (B) $3 \frac{1}{3}$ "
            r"(C) $1 \frac{2}{3}$ (D) $1 \frac{1}{3}$ (E) 2"
        ),
        responses=["<reasoning>The answer is B.<answer>B</answer>"],
        answers=["3\frac{1}{3}", "B"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)

  @pytest.mark.cpu_only
  def test_majority_eval_mode(self):
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.maj_config,
        question="What is the quantity of the item?",
        responses=[
            r"<reasoning>The item is 3\frac{1}{3}</reasoning><answer>3\frac{1}{3}</answer>",
            r"<reasoning>It is 3\frac{1}{3}</reasoning><answer>3\frac{1}{3}</answer>",
            r"<reasoning>The item is 3\frac{1}{3}</reasoning><answer>\frac{1}{3}</answer>",
        ],
        answers=[r"3\frac{1}{3}"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  @pytest.mark.cpu_only
  def test_pass_at_1_eval_mode(self):
    """pass@1 returns fraction of correct samples, not a boolean."""
    config = _make_config(eval_mode="pass_at_1")
    # 3 out of 4 samples are correct → expect 0.75
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=config,
        question="What is 2+2?",
        responses=[
            "<reasoning>2+2=4</reasoning><answer>4</answer>",
            "<reasoning>2+2=4</reasoning><answer>4</answer>",
            "<reasoning>2+2=4</reasoning><answer>4</answer>",
            "<reasoning>2+2=5</reasoning><answer>5</answer>",
        ],
        answers=["4"],
    )
    self.assertAlmostEqual(is_correct, 0.75)
    self.assertAlmostEqual(is_partially_correct, 0.75)
    self.assertAlmostEqual(has_correct_format, 1.0)

  @pytest.mark.cpu_only
  def test_pass_at_1_all_wrong(self):
    """pass@1 with all wrong samples returns 0.0."""
    config = _make_config(eval_mode="pass_at_1")
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=config,
        question="What is 2+2?",
        responses=[
            "<reasoning>2+2=5</reasoning><answer>5</answer>",
            "<reasoning>2+2=6</reasoning><answer>6</answer>",
        ],
        answers=["4"],
    )
    self.assertAlmostEqual(is_correct, 0.0)
    self.assertAlmostEqual(is_partially_correct, 0.0)
    self.assertAlmostEqual(has_correct_format, 1.0)

  @pytest.mark.cpu_only
  def test_pass_at_1_all_correct(self):
    """pass@1 with all correct samples returns 1.0."""
    config = _make_config(eval_mode="pass_at_1")
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=config,
        question="What is 2+2?",
        responses=[
            "<reasoning>2+2=4</reasoning><answer>4</answer>",
            "<reasoning>Simple: 4</reasoning><answer>4</answer>",
        ],
        answers=["4"],
    )
    self.assertAlmostEqual(is_correct, 1.0)
    self.assertAlmostEqual(is_partially_correct, 1.0)
    self.assertAlmostEqual(has_correct_format, 1.0)


if __name__ == "__main__":
  unittest.main()
