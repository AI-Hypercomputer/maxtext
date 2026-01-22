# Copyright 2023–2025 Google LLC
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

"""Unit tests for RL utility functions."""

# import pytest
import unittest
from unittest.mock import MagicMock

from MaxText.rl import utils_rl


class MockDebugConfig:
  """Mock debug configuration."""

  def __init__(self, rl=False):
    self.rl = rl


class MockTMVPConfig:
  """Mock TMVP configuration for testing reward functions."""

  def __init__(
      self,
      reasoning_start_token="<think>",
      reasoning_end_token="</think>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
      reward_exact_format_match=3.0,  # aligned with rl.yml
      reward_white_space_format_match=1.5,
      reward_partial_format_match=0.5,
      reward_ratio_guess_to_answer_high=0.5,
      reward_ratio_guess_to_answer_low=0.25,
      penalty_incorrect_format=-0.5,
      penalty_incorrect_answer=-1.0,
      dataset_name="gsm8k",  # TODO support DAPO
      debug_rl=False,
  ):
    self.reasoning_start_token = reasoning_start_token
    self.reasoning_end_token = reasoning_end_token
    self.solution_start_token = solution_start_token
    self.solution_end_token = solution_end_token
    self.reward_exact_format_match = reward_exact_format_match
    self.reward_partial_format_match = reward_partial_format_match
    self.penalty_incorrect_format = penalty_incorrect_format
    self.reward_white_space_format_match = reward_white_space_format_match
    self.reward_ratio_guess_to_answer_high = reward_ratio_guess_to_answer_high
    self.reward_ratio_guess_to_answer_low = reward_ratio_guess_to_answer_low
    self.penalty_incorrect_answer = penalty_incorrect_answer
    self.dataset_name = dataset_name
    self.debug = MockDebugConfig(debug_rl)


class TestExtractHashAnswer(unittest.TestCase):
  """Tests for extract_hash_answer function."""

  def test_extract_hash_answer_with_valid_hash(self):
    """Test extraction with valid #### delimiter."""
    text = "The calculation is 2+2=4. #### 4"
    result = utils_rl.extract_hash_answer(text)
    self.assertEqual(result, "4")

  def test_extract_hash_answer_with_spaces(self):
    """Test extraction handles trailing/leading spaces."""
    text = "Some reasoning here. ####   42  "
    result = utils_rl.extract_hash_answer(text)
    self.assertEqual(result, "42")

  def test_extract_hash_answer_without_hash(self):
    """Test extraction returns None when no #### delimiter."""
    text = "No hash delimiter in this text"
    result = utils_rl.extract_hash_answer(text)
    self.assertIsNone(result)

  def test_extract_hash_answer_with_multiple_hashes(self):
    """Test extraction with multiple #### delimiters takes first split."""
    text = "Step 1 #### answer1 #### answer2"
    result = utils_rl.extract_hash_answer(text)
    self.assertEqual(result, "answer1")

  def test_extract_hash_answer_with_negative_number(self):
    """Test extraction with negative number."""
    text = "The answer is #### -5"
    result = utils_rl.extract_hash_answer(text)
    self.assertEqual(result, "-5")


class TestGetMatchFormatRegex(unittest.TestCase):
  """Tests for get_match_format_regex function."""

  def test_regex_matches_correct_format(self):
    """Test regex matches properly formatted completion."""
    config = MockTMVPConfig()
    regex = utils_rl.get_match_format_regex(config)

    completion = "<think>Let me think about this</think><answer>42</answer>"
    match = regex.search(completion)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "42")

  def test_regex_matches_with_whitespace(self):
    """Test regex handles leading/trailing whitespace."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_format_regex(config)

    completion = "  <think>Reasoning</think><answer>100</answer>  "
    match = regex.search(completion)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "100")

  def test_regex_matches_multiline_reasoning(self):
    """Test regex handles multiline reasoning."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_format_regex(config)

    completion = "<think>Line 1\nLine 2\nLine 3</think><answer>yes</answer>"
    match = regex.search(completion)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "yes")

  def test_regex_does_not_match_missing_reasoning(self):
    """Test regex doesn't match when reasoning section is missing."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_format_regex(config)

    completion = "<answer>42</answer>"
    match = regex.search(completion)
    self.assertIsNone(match)

  def test_regex_does_not_match_missing_answer(self):
    """Test regex doesn't match when answer section is missing."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_format_regex(config)

    completion = "<think>Some reasoning</think>"
    match = regex.search(completion)
    self.assertIsNone(match)

  def test_regex_with_custom_tokens(self):
    """Test regex works with custom tokens."""

    config = MockTMVPConfig(
        reasoning_start_token="<REASON>",
        reasoning_end_token="</REASON>",
        solution_start_token="<SOLUTION>",
        solution_end_token="</SOLUTION>",
    )
    regex = utils_rl.get_match_format_regex(config)

    completion = "<REASON>Thinking...</REASON><SOLUTION>answer</SOLUTION>"
    match = regex.search(completion)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "answer")


class TestMatchFormatExactly(unittest.TestCase):
  """Tests for match_format_exactly function."""

  def test_exact_format_match_gives_reward(self):
    """Test exact format match returns reward."""

    config = MockTMVPConfig(reward_exact_format_match=2.0)
    prompts = ["What is 2+2?"]
    completions = ["<think>2+2=4</think><answer>4</answer>"]

    scores = utils_rl.match_format_exactly(prompts, completions, config)
    self.assertEqual(scores, [2.0])

  def test_incorrect_format_gives_zero(self):
    """Test incorrect format returns zero."""

    config = MockTMVPConfig(reward_exact_format_match=2.0)
    prompts = ["What is 2+2?"]
    completions = ["The answer is 4"]

    scores = utils_rl.match_format_exactly(prompts, completions, config)
    self.assertEqual(scores, [0])

  def test_multiple_completions(self):
    """Test scoring multiple completions."""

    config = MockTMVPConfig(reward_exact_format_match=2.0)
    prompts = ["Q1", "Q2", "Q3"]
    completions = [
        "<think>Thinking</think><answer>1</answer>",  # Matches
        "Just an answer: 2",  # Doesn't match
        "<think>More thinking</think><answer>3</answer>",  # Matches
    ]

    scores = utils_rl.match_format_exactly(prompts, completions, config)
    self.assertEqual(scores, [2.0, 0, 2.0])


class TestMatchFormatApproximately(unittest.TestCase):
  """Tests for match_format_approximately function."""

  def test_all_tokens_present_once(self):
    """Test reward when all tokens appear exactly once."""

    config = MockTMVPConfig(
        reward_partial_format_match=0.5,
        penalty_incorrect_format=-0.5,
    )
    completions = ["<think>Thinking</think><answer>42</answer>"]

    scores = utils_rl.match_format_approximately([], completions, config)
    # 4 tokens * 0.5 = 2.0
    self.assertEqual(scores, [2.0])

  def test_missing_tokens_penalized(self):
    """Test penalty when tokens are missing."""

    config = MockTMVPConfig(
        reward_partial_format_match=0.5,
        penalty_incorrect_format=-0.5,
    )
    completions = ["Just plain text without any tokens"]

    scores = utils_rl.match_format_approximately([], completions, config)
    # 4 tokens * -0.5 = -2.0
    self.assertEqual(scores, [-2.0])

  def test_duplicate_tokens_penalized(self):
    """Test penalty when tokens appear multiple times."""

    config = MockTMVPConfig(
        reward_partial_format_match=0.5,
        penalty_incorrect_format=-0.5,
    )
    completions = ["<think>First</think><think>Second</think><answer>42</answer>"]

    scores = utils_rl.match_format_approximately([], completions, config)
    # <think>: 2 times -> -0.5, </think>: 2 times -> -0.5
    # <answer>: 1 time -> 0.5, </answer>: 1 time -> 0.5
    self.assertEqual(scores, [0.0])

  def test_partial_tokens_mixed_rewards(self):
    """Test mixed rewards for partial token presence."""

    config = MockTMVPConfig(
        reward_partial_format_match=0.5,
        penalty_incorrect_format=-0.5,
    )
    completions = ["<think>Thinking</think>"]  # Missing answer tokens

    scores = utils_rl.match_format_approximately([], completions, config)
    # <think>: 1 -> 0.5, </think>: 1 -> 0.5
    # <answer>: 0 -> -0.5, </answer>: 0 -> -0.5
    self.assertEqual(scores, [0.0])


class TestCheckAnswer(unittest.TestCase):
  """Tests for check_answer function."""

  def test_exact_answer_match(self):
    """Test exact answer match gets full reward."""

    config = MockTMVPConfig(reward_exact_format_match=2.0)
    completions = ["<think>2+2=4</think><answer>4</answer>"]
    answers = ["4"]

    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [2.0])

  def test_whitespace_answer_match(self):
    """Test answer match with whitespace differences."""

    config = MockTMVPConfig(
        reward_exact_format_match=2.0,
        reward_white_space_format_match=1.5,
    )
    completions = ["<think>Thinking</think><answer> 42 </answer>"]
    answers = ["42"]

    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [1.5])

  def test_ratio_answer_high_reward(self):
    """Test answer within 10% gets high ratio reward."""

    config = MockTMVPConfig(
        reward_ratio_guess_to_answer_high=1.0,
        reward_ratio_guess_to_answer_low=0.5,
    )
    completions = ["<think>Approx</think><answer>95</answer>"]
    answers = ["100"]  # 95/100 = 0.95, within 0.9-1.1

    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [1.0])

  def test_ratio_answer_low_reward(self):
    """Test answer within 20% but not 10% gets low ratio reward."""

    config = MockTMVPConfig(
        reward_ratio_guess_to_answer_high=1.0,
        reward_ratio_guess_to_answer_low=0.5,
    )
    completions = ["<think>Approx</think><answer>85</answer>"]
    answers = ["100"]  # 85/100 = 0.85, within 0.8-1.2 but not 0.9-1.1

    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [0.5])

  def test_wrong_answer_penalized(self):
    """Test wrong answer gets penalty."""

    config = MockTMVPConfig(penalty_incorrect_answer=-1.0)
    completions = ["<think>Wrong</think><answer>50</answer>"]
    answers = ["100"]  # 50/100 = 0.5, outside all ranges

    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [-1.0])

  def test_no_answer_extracted(self):
    """Test zero score when answer cannot be extracted."""

    config = MockTMVPConfig()
    completions = ["No proper format here"]
    answers = ["42"]
    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [0])

  def test_non_numeric_answer_penalized(self):
    """Test non-numeric answer gets format penalty."""

    config = MockTMVPConfig(penalty_incorrect_format=-0.5)
    completions = ["<think>Thinking</think><answer>not a number</answer>"]
    answers = ["42"]

    scores = utils_rl.check_answer([], completions, answers, config)
    self.assertEqual(scores, [-0.5])


class TestGetMatchNumbersRegex(unittest.TestCase):
  """Tests for get_match_numbers_regex function."""

  def test_extracts_integer(self):
    """Test extraction of integer from answer."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_numbers_regex(config)
    text = "<answer>42</answer>"
    match = regex.search(text)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "42")

  def test_extracts_integer_with_comma(self):
    """Test extraction of negative integer from answer."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_numbers_regex(config)
    text = "<answer>1,234</answer>"
    match = regex.search(text)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "1,234")

  def test_extracts_negative_integer(self):
    """Test extraction of negative integer from answer."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_numbers_regex(config)
    text = "<answer>-42</answer>"
    match = regex.search(text)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "-42")

  def test_extracts_decimal(self):
    """Test extraction of decimal from answer."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_numbers_regex(config)
    text = "<answer>3.14159</answer>"
    match = regex.search(text)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "3.14159")

  def test_extracts_number_with_text(self):
    """Test extraction of number from text with surrounding words."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_numbers_regex(config)

    text = "<answer>The answer is 123 dollars</answer>"
    match = regex.search(text)
    self.assertIsNotNone(match)
    self.assertEqual(match.group(1), "123")

  def test_no_number_in_answer(self):
    """Test no match when answer contains no number."""

    config = MockTMVPConfig()
    regex = utils_rl.get_match_numbers_regex(config)
    text = "<answer>no numbers here</answer>"
    match = regex.search(text)
    self.assertIsNone(match)


class TestCheckNumbers(unittest.TestCase):
  """Tests for check_numbers function."""

  def test_exact_number_match(self):
    """Test exact number match gets reward."""

    config = MockTMVPConfig()
    completions = ["<answer>42</answer>"]
    answers = ["42"]

    scores = utils_rl.check_numbers([], completions, answers, config, question=["Q"])
    self.assertEqual(scores, [1.5])

  def test_number_mismatch(self):
    """Test number mismatch gets zero."""

    config = MockTMVPConfig()
    completions = ["<answer>10</answer>"]
    answers = ["42"]

    scores = utils_rl.check_numbers([], completions, answers, config, question=["Q"])
    self.assertEqual(scores, [0.0])

  def test_no_number_extracted(self):
    """Test zero when no number can be extracted."""

    config = MockTMVPConfig()
    completions = ["No answer tag here"]
    answers = ["42"]

    scores = utils_rl.check_numbers([], completions, answers, config, question=["Q"])
    self.assertEqual(scores, [0])

  def test_float_comparison(self):
    """Test float number comparison."""

    config = MockTMVPConfig()
    completions = ["<answer>3.5</answer>"]
    answers = ["3.5"]

    scores = utils_rl.check_numbers([], completions, answers, config, question=["Q"])
    self.assertEqual(scores, [1.5])

  def test_invalid_answer_format(self):
    """Test zero when answer is not a valid number."""

    config = MockTMVPConfig()
    completions = ["<answer>42</answer>"]
    answers = ["not a number"]

    scores = utils_rl.check_numbers([], completions, answers, config, question=["Q"])
    self.assertEqual(scores, [0])


class TestGetOptimizer(unittest.TestCase):
  """Tests for get_optimizer function."""

  def test_get_optimizer_without_gradient_clipping(self):
    """Test optimizer creation without gradient clipping."""

    config = MagicMock()
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.1
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_weight_decay = 0.01
    config.gradient_clipping_threshold = 0  # No clipping

    optimizer = utils_rl.get_optimizer(config, max_train_steps=1000)
    self.assertIsNotNone(optimizer)

  def test_get_optimizer_with_gradient_clipping(self):
    """Test optimizer creation with gradient clipping."""

    config = MagicMock()
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.1
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_weight_decay = 0.01
    config.gradient_clipping_threshold = 1.0  # With clipping

    optimizer = utils_rl.get_optimizer(config, max_train_steps=1000)
    self.assertIsNotNone(optimizer)

  def test_get_optimizer_warmup_steps(self):
    """Test optimizer warmup steps calculation."""

    config = MagicMock()
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.2
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_weight_decay = 0.01
    config.gradient_clipping_threshold = 0

    max_train_steps = 1000
    optimizer = utils_rl.get_optimizer(config, max_train_steps=max_train_steps)
    self.assertIsNotNone(optimizer)
    # Warmup steps should be 0.2 * 1000 = 200


if __name__ == "__main__":
  unittest.main()
