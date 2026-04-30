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

"""Instruction data processing test."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import datasets

from maxtext.input_pipeline import instruction_data_processing


class InstructionDataProcessingTest(unittest.TestCase):
  """Test instruction data processing."""

  def _run_math_qa_test(self, template_config, example_input, expected_user, expected_assistant):
    """Helper to run math_qa_formatting tests."""
    result = instruction_data_processing.math_qa_formatting(example_input, template_config=template_config)

    messages = {msg["role"]: msg["content"] for msg in result["messages"]}
    self.assertEqual(messages.get("user"), expected_user)
    self.assertEqual(messages.get("assistant"), expected_assistant)

  def test_load_data_template_from_file(self):
    template_config = instruction_data_processing.load_data_template_from_file(
        "maxtext/examples/chat_templates/gsm8k_rl.json"
    )
    self.assertEqual(
        template_config,
        {
            "SYSTEM_PROMPT": (
                "You are given a problem. Think about the problem and provide"
                " your reasoning. Place it between {reasoning_start_token} and"
                " {reasoning_end_token}. Then, provide the final answer (i.e.,"
                " just one numerical value) between {solution_start_token} and"
                " {solution_end_token}."
            ),
            "TEMPLATE": ("<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n<start_of_turn>model"),
        },
    )

  def test_math_qa_formatting_with_prompt_completion_template(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={"question": "What is 2 + 2?", "answer": "Because 2 and 2 make 4.\n ## 4"},
        expected_user="This is a question: What is 2 + 2?",
        expected_assistant="<reasoning>\nBecause 2 and 2 make 4.\n</reasoning>\n<answer>\n4\n</answer>",
    )

  def test_math_qa_formatting_with_no_reasoning_template(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "The answer is: {answer}",
        },
        example_input={"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
        expected_user="This is a question: What is the capital of France?",
        expected_assistant="The answer is: The capital of France is Paris.",
    )

  def test_math_qa_formatting_with_missing_reasoning_placeholder(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "The answer is: {answer}",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={"question": "What is 2 + 2?", "answer": "Because 2 and 2 make 4.\n ## The answer is: 4"},
        expected_user="This is a question: What is 2 + 2?",
        expected_assistant="Because 2 and 2 make 4.\n ## The answer is: 4",
    )

  def test_math_qa_formatting_with_missing_answer_placeholder(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {question}",
            "COMPLETION_TEMPLATE": "The answer is: {reply}",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={"question": "What is 2 + 2?", "answer": "Because 2 and 2 make 4.\n ## The answer is: 4"},
        expected_user="This is a question: What is 2 + 2?",
        expected_assistant="Because 2 and 2 make 4.\n ## The answer is: 4",
    )

  def test_math_qa_formatting_with_missing_question_placeholder(self):
    self._run_math_qa_test(
        template_config={
            "PROMPT_TEMPLATE": "This is a question: {user_question}",
            "COMPLETION_TEMPLATE": "The answer is: {reply}",
            "REASONING_ANSWER_SEPARATOR": "##",
        },
        example_input={
            "question": "What is 2 + 2?",
            "answer": "The answer is: Because 2 and 2 make 4.\n ## The answer is: 4",
        },
        expected_user="What is 2 + 2?",
        expected_assistant="The answer is: Because 2 and 2 make 4.\n ## The answer is: 4",
    )

  def test_math_qa_formatting_with_no_templates(self):
    self._run_math_qa_test(
        template_config=None,
        example_input={"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
        expected_user="What is the capital of Germany?",
        expected_assistant="The capital of Germany is Berlin.",
    )

  def test_load_chat_template_from_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Test .jinja file
      jinja_path = os.path.join(tmpdir, "test.jinja")
      with open(jinja_path, "w", encoding="utf-8") as f:
        f.write("test jinja template")
      self.assertEqual(
          instruction_data_processing.load_chat_template_from_file(jinja_path),
          "test jinja template",
      )

      # Test .json file with chat_template
      json_path = os.path.join(tmpdir, "test.json")
      with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"chat_template": "test json template"}, f)
      self.assertEqual(
          instruction_data_processing.load_chat_template_from_file(json_path),
          "test json template",
      )

      # Test .json file without chat_template
      json_no_key_path = os.path.join(tmpdir, "no_key.json")
      with open(json_no_key_path, "w", encoding="utf-8") as f:
        json.dump({"other_key": "other_value"}, f)
      self.assertIsNone(instruction_data_processing.load_chat_template_from_file(json_no_key_path))

      # Test non-existent file
      self.assertIsNone(instruction_data_processing.load_chat_template_from_file("non_existent.jinja"))


class TestCustomDataFormatting(unittest.TestCase):
  """Test custom data formatting."""

  def setUp(self):
    super().setUp()
    self.columns = ["question", "answer"]
    self.dataset = datasets.Dataset.from_dict({col: [f"val_{col}_{i}" for i in range(3)] for col in self.columns})
    self.dataset_features = datasets.Features(
        {"messages": [{"content": datasets.Value("string"), "role": datasets.Value("string")}]}
    )

  def test_data_formatter_without_formatting_func_path(self):
    returned_dataset, returned_columns = instruction_data_processing.convert_to_conversational_format(
        self.dataset,
        self.columns,
    )

    expected_dataset = datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": f"val_question_{i}"}, {"role": "assistant", "content": f"val_answer_{i}"}]
                for i in range(3)
            ]
        },
        features=self.dataset_features,
    )

    self.assertEqual(returned_columns, ["messages"])
    assert list(returned_dataset) == list(expected_dataset)
    assert returned_dataset["messages"] == expected_dataset["messages"]

  def test_data_formatter_with_formatting_func_path_and_kwargs(self):
    expected_dataset = datasets.Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] for _ in range(3)]},
        features=self.dataset_features,
    )
    mock_formatter = MagicMock(return_value=expected_dataset)

    with patch.object(instruction_data_processing, "load_formatter", return_value=mock_formatter) as mock_load:
      returned_dataset, returned_columns = instruction_data_processing.convert_to_conversational_format(
          self.dataset,
          self.columns,
          formatting_func_path="some.module.my_func",
          formatting_func_kwargs={"template_path": "/tmp/tmpl.json"},
      )

    args, kwargs = mock_load.call_args
    self.assertEqual(args[0], "some.module.my_func")
    self.assertEqual(kwargs["remove_columns"], self.columns)
    self.assertTrue("template_config" in kwargs)
    mock_formatter.assert_called_once_with(self.dataset, self.dataset_features)
    self.assertIs(returned_dataset, expected_dataset)
    self.assertEqual(returned_columns, ["messages"])

  def test_data_formatter_with_formatting_func_path_without_kwargs(self):
    expected_dataset = datasets.Dataset.from_dict(
        {"messages": [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] for _ in range(3)]},
        features=self.dataset_features,
    )
    mock_formatter = MagicMock(return_value=expected_dataset)

    with patch.object(instruction_data_processing, "load_formatter", return_value=mock_formatter) as mock_load:
      returned_dataset, returned_columns = instruction_data_processing.convert_to_conversational_format(
          self.dataset,
          self.columns,
          formatting_func_path="some.module.my_func",
          formatting_func_kwargs={},
      )

    args, kwargs = mock_load.call_args
    self.assertEqual(args[0], "some.module.my_func")
    self.assertEqual(kwargs["remove_columns"], self.columns)
    mock_formatter.assert_called_once_with(self.dataset, self.dataset_features)
    self.assertIs(returned_dataset, expected_dataset)
    self.assertEqual(returned_columns, ["messages"])


if __name__ == "__main__":
  unittest.main()
