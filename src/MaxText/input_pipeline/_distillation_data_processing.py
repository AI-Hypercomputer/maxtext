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

"""
Input pipeline to generate knowledge distillation dataset from conversational dataset.
The conversational dataset should conform to one of the two schemas:
1. Contains a `messages` column: Typically holding a list of message
   (e.g., [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...]).
2. Contains `prompt` and `completion` columns: Separating the input
    query [{'role': 'user', 'content': '...'}] from the target output [{'role': 'assistant', 'content': '...'}].
"""

from dataclasses import dataclass, field
from typing import List

import datasets

from MaxText import max_logging
from MaxText.input_pipeline import _input_pipeline_utils


@dataclass
class InputRequest:
  prompt: str = ""
  prompt_token_ids: List[int] = field(default_factory=list)
  actual_completion: str = ""
  max_output_tokens: int = 0


def map_to_prompt_completion(example):
  """
  example = {
    "messages": [
      {"role": "user", "content": "prompt_1"},
      {"role": "assistant", "content": "completion_1"},
      {"role": "user", "content": "prompt_2"},
      {"role": "assistant", "content": "completion_2"}
    ]
  }
  map_to_prompt_completion(example) returns:
    {
      "prompt": [{"role": "user", "content": "prompt_1"}, {"role": "user", "content": "prompt_2"}],
      "completion": [{"role": "assistant", "content": "completion_1"}, {"role": "assistant", "content": "completion_2"}]
    }
  """
  messages = example["messages"]
  example["prompt"] = [message for message in messages if message["role"] == "user"]
  example["completion"] = [message for message in messages if message["role"] == "assistant"]
  return example


def extract_content(example, data_column_names):
  """
  example = {
    "prompt": [{"role": "user", "content": "prompt_1"}, {"role": "user", "content": "prompt_2"}],
    "completion": [{"role": "assistant", "content": "completion_1"}, {"role": "assistant", "content": "completion_2"}]
  }
  extract_content(example, ["prompt", "completion"]) returns:
    {
      "prompt": ["prompt_1", "prompt_2"],
      "completion": ["completion_1", "completion_2"]
    }
  """
  for column in data_column_names:
    example[column] = [data["content"] for data in example[column]]
  return example


def process_dataset(config, dataset):  # pylint: disable=redefined-outer-name
  """Pipeline for preprocessing dataset."""
  data_column_names = config.data_columns
  dataset = dataset.select_columns(data_column_names)

  supported_columns = [["prompt", "completion"], ["messages"]]
  assert any(
      set(data_column_names) == set(supported) for supported in supported_columns
  ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_column_names}"
  assert _input_pipeline_utils.is_conversational(
      dataset.features, data_column_names
  ), "Dataset is not in conversational format."

  # maps "messages" to "prompt" and "completion" pairs
  if "messages" in data_column_names:
    dataset = dataset.map(map_to_prompt_completion, remove_columns=data_column_names)

  data_column_names = ["prompt", "completion"]
  dataset = dataset.map(extract_content, fn_kwargs={"data_column_names": data_column_names})

  return dataset


def load_dataset(config):  # pylint: disable=redefined-outer-name
  """Loads dataset from Hugging Face."""
  assert config.dataset_type == "huggingface", "Only dataset from Hugging Face is supported."

  return datasets.load_dataset(
      config.dataset_path,
      split=config.data_split,
      token=config.hf_access_token,
  )


def filter_dataset(config, dataset, tokenizer):
  "Filter out samples from the dataset."
  filtered_dataset = []
  for data in dataset:
    prompt = data["prompt"][0]
    actual_completion = data["completion"][0]

    max_output_length = config.max_target_length - config.max_prefill_length
    max_output_tokens = min(max_output_length, len(tokenizer.encode(actual_completion)))
    if config.use_chat_template:
      message = [{"role": "user", "content": prompt}]
      prompt_token_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True)
    else:
      prompt_token_ids = tokenizer.encode(prompt)

    # Filter out prompt sequences that are longer than max_prefill_length
    if len(prompt_token_ids) > config.max_prefill_length:
      continue

    request = InputRequest(prompt, prompt_token_ids, actual_completion, max_output_tokens)
    filtered_dataset.append(request)
  if len(filtered_dataset) < len(dataset):
    max_logging.log("Some prompts are longer than `max-prefill-length` and will be filtered out.")
    max_logging.log(f"Filtering reduced dataset batch from {len(dataset)} to {len(filtered_dataset)} samples.")
  return filtered_dataset
