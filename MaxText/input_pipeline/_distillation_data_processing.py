#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Input pipeline to generate knowledge distillation dataset from conversational dataset.
The conversational dataset should conform to one of the two schemas:
1. Contains a `messages` column: Typically holding a list of message
   (e.g., [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...]).
2. Contains `prompt` and `completion` columns: Separating the input
    query [{'role': 'user', 'content': '...'}] from the target output [{'role': 'assistant', 'content': '...'}].
"""

import datasets
import jax
import grain.python as grain
import numpy as np
import transformers

from jax.sharding import Mesh

from MaxText import max_utils
from MaxText import multihost_dataloading

from MaxText.input_pipeline import input_pipeline_interface, _input_pipeline_utils


def pad_to_max_length(example, column_names, max_length, pad_id):
  """Pad data to max_length."""
  for column in column_names:
    data = example[column]
    data_true_length = []
    for idx, _ in enumerate(data):
      true_length = len(data[idx])
      data_true_length.append(true_length)
      pad_amount = max(max_length - true_length, 0)
      if pad_amount:
        data[idx] += [pad_id] * pad_amount
    example[column] = np.array(data)
    example[f"{column}_true_length"] = np.array(data_true_length)
  return example


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
  prompt = []
  completion = []
  for message in messages:
    if message["role"] == "user":
      prompt.append(message)
    elif message["role"] == "assistant":
      completion.append(message)
  example["prompt"] = prompt
  example["completion"] = completion
  return example


def extract_content(example, data_column_names):
  """
  example = {
    "prompt": [{"role": "user", "content": "prompt_1"}, {"role": "user", "content": "prompt_2"}],
    "completion": [{"role": "assistant", "content": "completion_1"}, {"role": "assistant", "content": "completion_2"}]
  }
  extract_content(example, ["prompt", "completion"]) returns:
    {
      "prompt": [["prompt_1"], ["prompt_2"]],
      "completion": [["completion_1"], ["completion_2"]]
    }
  """
  for column in data_column_names:
    data_list = example[column]
    content_list = []
    for data in data_list:
      content_list.append([data["content"]])
    example[column] = content_list
  return example


def flatten_dataset(batch):
  """
  example = {
    "prompt": [["prompt_1"], ["prompt_2"]],
    "completion": [["completion_1"], ["completion_2"]]
  }
  flatten_dataset(example) returns:
    {
      "prompt": ["prompt_1"],
      "completion": ["completion_1"]
    }
  """
  return {
      "prompt": [prompt[0] for prompt in batch["prompt"]],
      "completion": [completion[0] for completion in batch["completion"]],
  }


def get_pad_id(tokenizer):
  """Get padding token id for the tokenizer."""
  if getattr(tokenizer, "pad_token_id", None):
    return tokenizer.pad_token_id
  if getattr(tokenizer, "pad_token", None):
    return tokenizer.encode(tokenizer.pad_token)[0]
  if getattr(tokenizer, "unk_token_id", None):
    return tokenizer.unk_token_id
  if getattr(tokenizer, "unk_token", None):
    return tokenizer.encode(tokenizer.unk_token)[0]
  return -1


def process_dataset(config, dataset):  # pylint: disable=redefined-outer-name
  """Pipeline for preprocessing dataset."""
  data_column_names = config.train_data_columns

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

  dataset = dataset.map(flatten_dataset, batched=True)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      add_bos_token=False,
      add_eos_token=False,
      legacy=False,
      token=config.hf_access_token,
  )
  dataset = dataset.map(
      _input_pipeline_utils.tokenization,
      batched=True,
      fn_kwargs={
          "hf_tokenizer": tokenizer,
          "truncation": True,
          "max_length": config.max_prefill_predict_length,
          "column_names": data_column_names,
      },
  )

  pad_id = get_pad_id(tokenizer)
  dataset = dataset.map(
      pad_to_max_length,
      fn_kwargs={"column_names": data_column_names, "max_length": config.max_prefill_predict_length, "pad_id": pad_id},
  )

  return dataset, data_column_names


def get_data_iterator(config):  # pylint: disable=redefined-outer-name
  """Load, preprocess dataset and return iterators."""
  dataset = datasets.load_dataset(
      config.hf_path,
      data_dir=config.hf_data_dir,
      data_files=config.hf_train_files,
      split=config.train_split,
      streaming=True,
      token=config.hf_access_token,
  )

  dataset, data_column_names = process_dataset(config, dataset)

  devices_array = max_utils.create_device_mesh(config=config, devices=jax.devices())
  mesh = Mesh(devices_array, config.mesh_axes)
  process_indices = input_pipeline_interface.get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )

  operations = []
  operations.append(grain.Batch(batch_size=config.global_batch_size_to_load // jax.process_count(), drop_remainder=False))
  dataloading_host_index = process_indices.index(jax.process_index())
  dataloading_host_count = len(process_indices)
  dataset = _input_pipeline_utils.HFDataSource(
      dataset,
      dataloading_host_index,
      dataloading_host_count,
      1,
      False,
      config.max_prefill_predict_length,
      data_column_names,
  )

  dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=config.enable_data_shuffling,
      seed=config.data_shuffle_seed,
  )
  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=1,
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
  )
  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
  return multihost_gen
