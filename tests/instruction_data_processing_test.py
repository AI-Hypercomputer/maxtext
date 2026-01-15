# Copyright 2023–2026 Google LLC
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

import unittest

from MaxText.input_pipeline import instruction_data_processing


class InstructionDataProcessingTest(unittest.TestCase):

  def test_map_qa_data_to_conversation_with_prompt_completion_template(self):
    template_config = {
        "PROMPT_TEMPLATE": "This is a question: {question}",
        "COMPLETION_TEMPLATE": "<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>",
        "REASONING_ANSWER_SEPARATOR": "##",
    }
    example = {
        "question": "What is 2 + 2?",
        "answer": "Because 2 and 2 make 4.\n ## 4",
    }
    example = instruction_data_processing.map_qa_data_to_conversation(example, template_config)
    self.assertTrue("messages" in example)
    for data in example["messages"]:
      if data["role"] == "user":
        self.assertEqual(data["content"], "This is a question: What is 2 + 2?")
      if data["role"] == "assistant":
        self.assertEqual(data["content"], "<reasoning>\nBecause 2 and 2 make 4.\n</reasoning>\n<answer>\n4\n</answer>")

  def test_map_qa_data_to_conversation_with_no_reasoning_template(self):
    template_config = {
        "PROMPT_TEMPLATE": "This is a question: {question}",
        "COMPLETION_TEMPLATE": "The answer is: {answer}",
    }
    example = {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
    }
    example = instruction_data_processing.map_qa_data_to_conversation(example, template_config)
    self.assertTrue("messages" in example)
    for data in example["messages"]:
      if data["role"] == "user":
        self.assertEqual(data["content"], "This is a question: What is the capital of France?")
      if data["role"] == "assistant":
        self.assertEqual(data["content"], "The answer is: The capital of France is Paris.")

  def test_map_qa_data_to_conversation_with_missing_reasoning_placeholder(self):
    template_config = {
        "PROMPT_TEMPLATE": "This is a question: {question}",
        "COMPLETION_TEMPLATE": "The answer is: {answer}",
        "REASONING_ANSWER_SEPARATOR": "##",
    }
    example = {
        "question": "What is 2 + 2?",
        "answer": "Because 2 and 2 make 4.\n ## The answer is: 4",
    }
    example = instruction_data_processing.map_qa_data_to_conversation(example, template_config)
    self.assertTrue("messages" in example)
    for data in example["messages"]:
      if data["role"] == "user":
        self.assertEqual(data["content"], "This is a question: What is 2 + 2?")
      if data["role"] == "assistant":
        self.assertEqual(data["content"], "Because 2 and 2 make 4.\n ## The answer is: 4")

  def test_map_qa_data_to_conversation_with_missing_answer_placeholder(self):
    template_config = {
        "PROMPT_TEMPLATE": "This is a question: {question}",
        "COMPLETION_TEMPLATE": "The answer is: {reply}",
        "REASONING_ANSWER_SEPARATOR": "##",
    }
    example = {
        "question": "What is 2 + 2?",
        "answer": "Because 2 and 2 make 4.\n ## The answer is: 4",
    }
    example = instruction_data_processing.map_qa_data_to_conversation(example, template_config)
    self.assertTrue("messages" in example)
    for data in example["messages"]:
      if data["role"] == "user":
        self.assertEqual(data["content"], "This is a question: What is 2 + 2?")
      if data["role"] == "assistant":
        self.assertEqual(data["content"], "Because 2 and 2 make 4.\n ## The answer is: 4")

  def test_map_qa_data_to_conversation_with_missing_question_placeholder(self):
    template_config = {
        "PROMPT_TEMPLATE": "This is a question: {user_question}",
        "COMPLETION_TEMPLATE": "The answer is: {answer}",
    }
    example = {
        "question": "What is 2 + 2?",
        "answer": "Because 2 and 2 make 4.\n ## The answer is: 4",
    }
    example = instruction_data_processing.map_qa_data_to_conversation(example, template_config)
    self.assertTrue("messages" in example)
    for data in example["messages"]:
      if data["role"] == "user":
        self.assertEqual(data["content"], "What is 2 + 2?")
      if data["role"] == "assistant":
        self.assertEqual(data["content"], "The answer is: Because 2 and 2 make 4.\n ## The answer is: 4")

  def test_map_qa_data_to_conversation_with_no_templates(self):
    template_config = {}
    example = {
        "question": "What is the capital of Germany?",
        "answer": "The capital of Germany is Berlin.",
    }
    example = instruction_data_processing.map_qa_data_to_conversation(example, template_config)
    self.assertTrue("messages" in example)
    for data in example["messages"]:
      if data["role"] == "user":
        self.assertEqual(data["content"], "What is the capital of Germany?")
      if data["role"] == "assistant":
        self.assertEqual(data["content"], "The capital of Germany is Berlin.")


if __name__ == "__main__":
  unittest.main()
