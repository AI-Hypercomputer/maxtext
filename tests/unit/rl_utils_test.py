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

evaluate_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.evaluate_rl",
    reason="tunix (required by evaluate_rl) is not installed GPU",
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


if __name__ == "__main__":
  unittest.main()
