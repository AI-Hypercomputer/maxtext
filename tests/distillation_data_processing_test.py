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

"""Data processing tests for distillation."""

import argparse
import os
import subprocess
import unittest

import transformers

from datasets import Dataset

from maxtext.src.maxtext.globals import MAXTEXT_ASSETS_ROOT
from maxtext.src.maxtext.input_pipeline import _distillation_data_processing

PROMPT_DATA = [
    [
        {"content": "What color is the sky?", "role": "user"},
        {"content": "Why is the sky blue?", "role": "user"},
    ],
    [
        {"content": "Can you tell me how many days are in a week?", "role": "user"},
    ],
]

COMPLETION_DATA = [
    [
        {"content": "The sky is blue.", "role": "assistant"},
        {"content": "The sky appears blue due a phenomemon called Rayleigh scattering.", "role": "assistant"},
    ],
    [
        {"content": "There are 7 days in a week.", "role": "assistant"},
    ],
]

MESSAGES_DATA = [
    [
        {"content": "What color is the sky?", "role": "user"},
        {"content": "The sky is blue.", "role": "assistant"},
        {"content": "Why is the sky blue?", "role": "user"},
        {"content": "The sky appears blue due a phenomemon called Rayleigh scattering.", "role": "assistant"},
    ],
    [
        {"content": "Can you tell me how many days are in a week?", "role": "user"},
        {"content": "There are 7 days in a week.", "role": "assistant"},
    ],
]


def add_arguments_to_parser(parser):
  parser.add_argument("--data-columns", nargs="+", required=True, help="Columns names that contain relevant data.")
  parser.add_argument("--use-chat-template", action="store_true", help="Enable tokenizer to apply a chat template.")
  parser.add_argument("--max-prefill-length", type=int, default=16, help="The maximum length for prompt tokens.")
  parser.add_argument(
      "--max-target-length", type=int, default=32, help="The maximum prompt length plus the output completion length."
  )
  return parser


class DistillationDataProcessingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    exit_code = subprocess.call(
        [
            "gsutil",
            "cp",
            "-r",
            "gs://maxtext-dataset/hf/llama2-chat-tokenizer",
            os.path.join(MAXTEXT_ASSETS_ROOT, ""),
        ]
    )
    if exit_code != 0:
      raise ValueError(f"Download tokenizer with gsutil cp failed with exit code: {exit_code}")

  def setUp(self):
    super().setUp()
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(MAXTEXT_ASSETS_ROOT, "llama2-chat-tokenizer"),
    )
    self.parser = argparse.ArgumentParser()
    self.parser = add_arguments_to_parser(self.parser)

  def test_data_processing_with_messages(self):
    config = self.parser.parse_args(["--data-columns", "messages"])
    dataset = Dataset.from_dict({"messages": MESSAGES_DATA})

    processed_dataset = _distillation_data_processing.process_dataset(config, dataset)

    expected_prompts = [
        ["What color is the sky?", "Why is the sky blue?"],
        ["Can you tell me how many days are in a week?"],
    ]
    expected_completions = [
        ["The sky is blue.", "The sky appears blue due a phenomemon called Rayleigh scattering."],
        ["There are 7 days in a week."],
    ]

    self.assertEqual(len(processed_dataset), len(expected_prompts))
    for idx, data in enumerate(processed_dataset):
      self.assertEqual(len(data["prompt"]), len(expected_prompts[idx]))
      for p_idx, prompt in enumerate(expected_prompts[idx]):
        self.assertEqual(data["prompt"][p_idx], prompt)

      self.assertEqual(len(data["completion"]), len(expected_completions[idx]))
      for c_idx, completion in enumerate(expected_completions[idx]):
        self.assertEqual(data["completion"][c_idx], completion)

  def test_data_filtering_with_messages(self):
    config = self.parser.parse_args(["--data-columns", "messages", "--use-chat-template"])
    dataset = Dataset.from_dict({"messages": MESSAGES_DATA})

    processed_dataset = _distillation_data_processing.process_dataset(config, dataset)
    filtered_dataset = _distillation_data_processing.filter_dataset(config, processed_dataset, self.tokenizer)

    self.assertEqual(len(filtered_dataset), 1)
    self.assertEqual(filtered_dataset[0].prompt, "What color is the sky?")
    self.assertEqual(filtered_dataset[0].actual_completion, "The sky is blue.")

  def test_data_processing_with_prompt_completion(self):
    config = self.parser.parse_args(["--data-columns", "prompt", "completion"])
    dataset = Dataset.from_dict({"prompt": PROMPT_DATA, "completion": COMPLETION_DATA})

    processed_dataset = _distillation_data_processing.process_dataset(config, dataset)

    expected_prompts = [
        ["What color is the sky?", "Why is the sky blue?"],
        ["Can you tell me how many days are in a week?"],
    ]
    expected_completions = [
        ["The sky is blue.", "The sky appears blue due a phenomemon called Rayleigh scattering."],
        ["There are 7 days in a week."],
    ]

    self.assertEqual(len(processed_dataset), len(expected_prompts))
    for idx, data in enumerate(processed_dataset):
      self.assertEqual(len(data["prompt"]), len(expected_prompts[idx]))
      for p_idx, prompt in enumerate(expected_prompts[idx]):
        self.assertEqual(data["prompt"][p_idx], prompt)

      self.assertEqual(len(data["completion"]), len(expected_completions[idx]))
      for c_idx, completion in enumerate(expected_completions[idx]):
        self.assertEqual(data["completion"][c_idx], completion)

  def test_data_filtering_with_prompt_completion(self):
    config = self.parser.parse_args(["--data-columns", "prompt", "completion", "--use-chat-template"])
    dataset = Dataset.from_dict({"prompt": PROMPT_DATA, "completion": COMPLETION_DATA})

    processed_dataset = _distillation_data_processing.process_dataset(config, dataset)
    filtered_dataset = _distillation_data_processing.filter_dataset(config, processed_dataset, self.tokenizer)

    self.assertEqual(len(filtered_dataset), 1)
    self.assertEqual(filtered_dataset[0].prompt, "What color is the sky?")
    self.assertEqual(filtered_dataset[0].actual_completion, "The sky is blue.")


if __name__ == "__main__":
  unittest.main()
