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

"""Unit tests for RL evaluation functions."""

# import pytest
import unittest
from unittest.mock import MagicMock, patch

from MaxText.rl.evaluate_rl import score_responses
from MaxText.rl.evaluate_rl import generate_responses
from MaxText.rl.evaluate_rl import evaluate


class MockDebugConfig:
  """Mock debug configuration."""

  def __init__(self, rl=False):
    self.rl = rl


class MockTMVPConfig:
  """Mock TMVP configuration for testing evaluation functions."""

  def __init__(
      self,
      reasoning_start_token="<think>",
      reasoning_end_token="</think>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
      debug_rl=False,
  ):
    self.reasoning_start_token = reasoning_start_token
    self.reasoning_end_token = reasoning_end_token
    self.solution_start_token = solution_start_token
    self.solution_end_token = solution_end_token
    self.debug = MockDebugConfig(debug_rl)


class TestScoreResponses(unittest.TestCase):
  """Tests for score_responses function."""

  def test_exact_correct_answer(self):
    """Test scoring when answer is exactly correct."""

    config = MockTMVPConfig()
    question = "What is 2+2?"
    responses = ["<think>2+2=4</think><answer>4</answer>"]
    answer = "4"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_partially_correct_answer(self):
    """Test scoring when answer is within 10% of correct."""

    config = MockTMVPConfig()
    question = "What is 100+0?"
    responses = ["<think>Approximately 100</think><answer>95</answer>"]
    answer = "100"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertFalse(is_correct)
    self.assertTrue(is_partially_correct)  # 95/100 = 0.95, within 0.9-1.1
    self.assertTrue(has_correct_format)

  def test_incorrect_answer(self):
    """Test scoring when answer is incorrect."""

    config = MockTMVPConfig()
    question = "What is 50+50?"
    responses = ["<think>Let me calculate</think><answer>50</answer>"]
    answer = "100"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertFalse(is_correct)
    self.assertFalse(is_partially_correct)  # 50/100 = 0.5, outside range
    self.assertTrue(has_correct_format)

  def test_correct_format_wrong_answer(self):
    """Test scoring with correct format but wrong answer."""

    config = MockTMVPConfig()
    question = "What is 10+10?"
    responses = ["<think>Thinking...</think><answer>5</answer>"]
    answer = "20"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertFalse(is_correct)
    self.assertFalse(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_incorrect_format(self):
    """Test scoring when format is incorrect."""

    config = MockTMVPConfig()
    question = "What is 2+2?"
    responses = ["The answer is 4"]  # Missing format tokens
    answer = "4"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertFalse(is_correct)
    self.assertFalse(is_partially_correct)
    self.assertFalse(has_correct_format)

  def test_multiple_responses_finds_correct(self):
    """Test that multiple responses are checked and correct one found."""

    config = MockTMVPConfig()
    question = "What is 5+5?"
    responses = [
        "Wrong answer",  # No format, wrong
        "<think>Thinking</think><answer>8</answer>",  # Format but wrong
        "<think>5+5=10</think><answer>10</answer>",  # Correct!
    ]
    answer = "10"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_answer_with_currency_symbol(self):
    """Test scoring handles currency symbols in answer."""

    config = MockTMVPConfig()
    question = "What is the price?"
    responses = ["<think>Calculating</think><answer>$100</answer>"]
    answer = "$100"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_answer_with_comma(self):
    """Test scoring handles commas in numbers."""

    config = MockTMVPConfig()
    question = "What is 1000+234?"
    responses = ["<think>Adding</think><answer>1,234</answer>"]
    answer = "1,234"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_non_numeric_answer_handling(self):
    """Test scoring handles non-numeric extracted values gracefully."""

    config = MockTMVPConfig()
    question = "What is the result?"
    responses = ["<think>Thinking</think><answer>hello</answer>"]
    answer = "42"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertFalse(is_correct)
    self.assertFalse(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_empty_responses_list(self):
    """Test scoring with empty responses list."""

    config = MockTMVPConfig()
    question = "What is 2+2?"
    responses = []
    answer = "4"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertFalse(is_correct)
    self.assertFalse(is_partially_correct)
    self.assertFalse(has_correct_format)

  def test_custom_tokens(self):
    """Test scoring with custom reasoning/answer tokens."""

    config = MockTMVPConfig(
        reasoning_start_token="<THINK>",
        reasoning_end_token="</THINK>",
        solution_start_token="<AND>",
        solution_end_token="</AND>",
    )
    question = "What is 3+3?"
    responses = ["<THINK>3+3=6</THINK><AND>6</AND>"]
    answer = "6"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_negative_numbers(self):
    """Test scoring with negative numbers."""

    config = MockTMVPConfig()
    question = "What is 5-10?"
    responses = ["<think>5-10=-5</think><answer>-5</answer>"]
    answer = "-5"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  def test_decimal_numbers(self):
    """Test scoring with decimal numbers."""

    config = MockTMVPConfig()
    question = "What is 1/4?"
    responses = ["<think>1/4=0.25</think><answer>0.25</answer>"]
    answer = "0.25"

    is_correct, is_partially_correct, has_correct_format = score_responses(config, question, responses, answer)

    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)


class TestGenerateResponses(unittest.TestCase):
  """Tests for generate_responses function."""

  def test_generate_responses_single_pass(self):
    """Test generating responses with single pass."""

    config = MagicMock()
    config.max_target_length = 512
    config.max_prefill_predict_length = 128
    config.eval_sampling_strategy = "greedy"  ## can be "greedy", "standard", or "liberal"
    config.generation_configs = {
        "greedy": {
            "eval_temperature": 0.01,
            "eval_top_k": 1,
            "eval_top_p": 1.0,
        }
    }
    config.debug = MockDebugConfig(rl=False)

    mock_rl_cluster = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ["Response 1", "Response 2"]
    mock_rl_cluster.rollout.generate.return_value = mock_response

    prompts = ["Prompt 1", "Prompt 2"]

    responses = generate_responses(
        tmvp_config=config,
        prompts=prompts,
        rl_cluster=mock_rl_cluster,
        num_passes=1,
    )

    self.assertEqual(len(responses), 2)
    self.assertEqual(responses[0], ["Response 1"])
    self.assertEqual(responses[1], ["Response 2"])

  def test_generate_responses_multiple_passes(self):
    """Test generating responses with multiple passes."""

    config = MagicMock()
    config.max_target_length = 512
    config.max_prefill_predict_length = 128
    config.eval_sampling_strategy = "default"
    config.generation_configs = {
        "default": {
            "eval_temperature": 0.7,
            "eval_top_k": 50,
            "eval_top_p": 0.9,
        }
    }
    config.debug = MockDebugConfig(rl=False)

    mock_rl_cluster = MagicMock()
    mock_response_1 = MagicMock()
    mock_response_1.text = ["Pass1-R1", "Pass1-R2"]
    mock_response_2 = MagicMock()
    mock_response_2.text = ["Pass2-R1", "Pass2-R2"]
    mock_rl_cluster.rollout.generate.side_effect = [
        mock_response_1,
        mock_response_2,
    ]

    prompts = ["Prompt 1", "Prompt 2"]

    responses = generate_responses(
        tmvp_config=config,
        prompts=prompts,
        rl_cluster=mock_rl_cluster,
        num_passes=2,
    )

    self.assertEqual(len(responses), 2)
    self.assertEqual(responses[0], ["Pass1-R1", "Pass2-R1"])
    self.assertEqual(responses[1], ["Pass1-R2", "Pass2-R2"])


class TestEvaluate(unittest.TestCase):
  """Tests for evaluate function."""

  @patch("MaxText.rl.evaluate_rl.generate_responses")
  def test_evaluate_perfect_accuracy(self, mock_generate):
    """Test evaluate with perfect accuracy."""

    config = MockTMVPConfig()

    # Mock dataset with one batch
    mock_batch = {
        "answer": ["4", "10"],
        "question": ["2+2?", "5+5?"],
        "prompts": ["Prompt1", "Prompt2"],
    }
    mock_dataset = [mock_batch]

    # Mock generate_responses to return correct formatted answers
    mock_generate.return_value = [
        ["<think>2+2=4</think><answer>4</answer>"],
        ["<think>5+5=10</think><answer>10</answer>"],
    ]

    mock_rl_cluster = MagicMock()

    (corr, total, accuracy, partial_accuracy, format_accuracy), _ = evaluate(
        tmvp_config=config,
        dataset=mock_dataset,
        rl_cluster=mock_rl_cluster,
        num_passes=1,
    )

    self.assertEqual(corr, 2)
    self.assertEqual(total, 2)
    self.assertEqual(accuracy, 100.0)
    self.assertEqual(partial_accuracy, 100.0)
    self.assertEqual(format_accuracy, 100.0)

  @patch("MaxText.rl.evaluate_rl.generate_responses")
  def test_evaluate_partial_accuracy(self, mock_generate):
    """Test evaluate with partial accuracy."""

    config = MockTMVPConfig()

    mock_batch = {
        "answer": ["4", "10"],
        "question": ["2+2?", "5+5?"],
        "prompts": ["Prompt1", "Prompt2"],
    }
    mock_dataset = [mock_batch]

    # First is correct, second is wrong
    mock_generate.return_value = [
        ["<think>Correct</think><answer>4</answer>"],
        ["<think>Wrong</think><answer>5</answer>"],  # Wrong answer
    ]

    mock_rl_cluster = MagicMock()

    (corr, total, accuracy, partial_accuracy, format_accuracy), _ = evaluate(
        tmvp_config=config,
        dataset=mock_dataset,
        rl_cluster=mock_rl_cluster,
        num_passes=1,
    )

    self.assertEqual(corr, 1)
    self.assertEqual(total, 2)
    self.assertEqual(accuracy, 50.0)
    self.assertEqual(partial_accuracy, 50.0)
    self.assertEqual(format_accuracy, 100.0)

  @patch("MaxText.rl.evaluate_rl.generate_responses")
  def test_evaluate_with_format_errors(self, mock_generate):
    """Test evaluate with format errors."""

    config = MockTMVPConfig()

    mock_batch = {
        "answer": ["4"],
        "question": ["2+2?"],
        "prompts": ["Prompt1"],
    }
    mock_dataset = [mock_batch]

    # Missing format tokens
    mock_generate.return_value = [
        ["The answer is 4"],  # No format tokens
    ]

    mock_rl_cluster = MagicMock()

    (corr, total, accuracy, partial_accuracy, format_accuracy), _ = evaluate(
        tmvp_config=config,
        dataset=mock_dataset,
        rl_cluster=mock_rl_cluster,
        num_passes=1,
    )

    self.assertEqual(corr, 0)
    self.assertEqual(total, 1)
    self.assertEqual(accuracy, 0.0)
    self.assertEqual(partial_accuracy, 0.0)
    self.assertEqual(format_accuracy, 0.0)

  @patch("MaxText.rl.evaluate_rl.generate_responses")
  def test_evaluate_make_lst_correct(self, mock_generate):
    """Test evaluate returns list of correct responses when corr_lst=True."""

    config = MockTMVPConfig()

    mock_batch = {
        "answer": ["4", "10"],
        "question": ["2+2?", "5+5?"],
        "prompts": ["Prompt1", "Prompt2"],
    }
    mock_dataset = [mock_batch]

    mock_generate.return_value = [
        ["<think>Correct</think><answer>4</answer>"],
        ["<think>Wrong</think><answer>5</answer>"],
    ]

    mock_rl_cluster = MagicMock()

    _, response_lst = evaluate(
        tmvp_config=config,
        dataset=mock_dataset,
        rl_cluster=mock_rl_cluster,
        num_passes=1,
        corr_lst=True,
        make_lst=True,
    )

    # Should only contain correct response
    self.assertEqual(len(response_lst), 1)
    self.assertEqual(response_lst[0][0], "2+2?")  # question
    self.assertEqual(response_lst[0][1], "4")  # answer

  @patch("MaxText.rl.evaluate_rl.generate_responses")
  def test_evaluate_make_lst_incorrect(self, mock_generate):
    """Test evaluate returns list of incorrect responses when corr_lst=False."""

    config = MockTMVPConfig()

    mock_batch = {
        "answer": ["4", "10"],
        "question": ["2+2?", "5+5?"],
        "prompts": ["Prompt1", "Prompt2"],
    }
    mock_dataset = [mock_batch]

    mock_generate.return_value = [
        ["<think>Correct</think><answer>4</answer>"],
        ["<think>Wrong</think><answer>5</answer>"],
    ]

    mock_rl_cluster = MagicMock()

    _, response_lst = evaluate(
        tmvp_config=config,
        dataset=mock_dataset,
        rl_cluster=mock_rl_cluster,
        num_passes=1,
        corr_lst=False,
        make_lst=True,
    )

    # Should only contain incorrect response
    self.assertEqual(len(response_lst), 1)
    self.assertEqual(response_lst[0][0], "5+5?")  # question
    self.assertEqual(response_lst[0][1], "10")  # answer


if __name__ == "__main__":
  unittest.main()
