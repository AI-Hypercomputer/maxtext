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

"""Input pipeline for GRPO training using Hugging Face datasets.

This module provides functions to create a data loading and preprocessing pipeline
for Group Relative Policy Optimization (GRPO). It leverages the `datasets` library
to stream data from Hugging Face and `grain` for efficient processing and
batching. The pipeline tokenizes, pads/trims, and batches text data to be
used as prompts for the GRPO generation and training loop.
"""

from collections.abc import Iterable

import numpy as np

import jax
from jax.sharding import Mesh

import transformers

import grain.python as grain

from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import input_pipeline_utils


class SingleHostDataLoader:
  """A data loader for a single host that wraps a grain.DataLoader.

  This class provides a standard Python iterator interface over a `grain.DataLoader`.
  It is designed to be used on a single host and ensures that the iterator can be
  reset.

  Attributes:
    global_mesh: The JAX device mesh.
    dataloader: The underlying `grain.DataLoader` instance.
    local_iterator: The Python iterator created from the dataloader.
  """

  def __init__(self, dataloader: grain.DataLoader, global_mesh: Mesh):
    """Initializes the SingleHostDataLoader.

    Args:
      dataloader: A `grain.DataLoader` to be wrapped.
      global_mesh: The JAX device mesh.
    """
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    if not isinstance(self.dataloader, Iterable):
      raise ValueError("Type error: dataloader should be an Iterable.")
    self.local_iterator = iter(self.dataloader)

  def reset(self):
    """Resets the internal iterator."""
    if not isinstance(self.dataloader, Iterable):
      raise ValueError("Type error: dataloader should be a grain.DataLoader.")
    self.local_iterator = iter(self.dataloader)

  def __iter__(self):
    """Returns the iterator object itself."""
    self.reset()
    return self

  def __next__(self):
    """Returns the next batch of data from the iterator."""
    local_data = next(self.local_iterator)
    return local_data


def preprocessing_pipeline(
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    data_column_names,
    tokenize,
    tokenizer_path,
    hf_access_token,
    global_batch_size,
    max_target_length,
    shuffle: bool = False,
    data_shuffle_seed=0,
    add_bos=True,
    add_eos=True,
    num_threads=1,
    drop_remainder=False,
):
  """Creates a preprocessing pipeline for a Hugging Face dataset.

  This function sets up a series of operations to tokenize, pad, and batch
  data from a streaming Hugging Face dataset. It is designed to be used
  within a multi-host data loading setup.

  Args:
    dataloading_host_index: The index of the current host in the data loading
      process.
    dataloading_host_count: The total number of hosts involved in data loading.
    global_mesh: The JAX device mesh.
    dataset: The Hugging Face `IterableDataset` to preprocess.
    data_column_names: The list of column names in the dataset to use.
    tokenize: A boolean indicating whether to tokenize the data.
    tokenizer_path: The path to the tokenizer model.
    hf_access_token: The Hugging Face access token.
    global_batch_size: The total batch size across all devices.
    max_target_length: The maximum sequence length for padding or trimming.
    shuffle: Whether to shuffle the dataset.
    data_shuffle_seed: The seed for shuffling the data.
    add_bos: Whether to add a beginning-of-sequence token.
    add_eos: Whether to add an end-of-sequence token.
    num_threads: The number of threads to use for data processing.
    drop_remainder: Whether to drop the last batch if it's smaller than the
      batch size.

  Returns:
    An iterator that yields preprocessed data batches for the local host.
  """

  if global_batch_size % global_mesh.size != 0:
    raise ValueError("Batch size should be divisible number of global devices.")

  if tokenize:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        model_max_length=max_target_length,
        legacy=False,
        token=hf_access_token,
    )

    dataset = dataset.map(
        input_pipeline_utils.tokenization,
        batched=True,
        fn_kwargs={
            "hf_tokenizer": tokenizer,
            "truncation": True,
            "max_length": max_target_length - 1,
            "column_names": data_column_names,
        },
    )
  dataset = dataset.select_columns(data_column_names)
  dataset = input_pipeline_utils.HFDataSource(
      dataset,
      dataloading_host_index,
      dataloading_host_count,
      num_threads,
      max_target_length,
      data_column_names,
  )

  def lists2array(x):
    """Convert lists/tuples to array"""
    return jax.tree.map(np.asarray, x, is_leaf=lambda y: isinstance(y, (list, tuple)))

  operations = [
      grain.MapOperation(lists2array),
      input_pipeline_utils.PadOrTrimToMaxLength(max_target_length, add_true_length=True),
      grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder),
  ]

  # Since HuggingFace IterableDataset does not support access through index
  # Indexes generated by dummy_index_sampler is not used.
  # dummy_index_sampler is used as an input placeholder for grain.Dataloader
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=shuffle,
      seed=data_shuffle_seed,
  )

  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=1,  # only supports one worker for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=128),
  )

  # single_host_gen = SingleHostDataLoader(dataloader, global_mesh)
  return iter(dataloader)


def make_hf_train_iterator(
    config,
    global_mesh,
    process_indices_train,
):
  """Loads a Hugging Face dataset and creates a local preprocessed iterator.

  This function loads a streaming dataset from the Hugging Face Hub, then
  applies the `preprocessing_pipeline` to create an iterator that yields
  batches of data suitable for training on the current host.

  Args:
    config: The configuration object with dataset and model parameters.
    global_mesh: The JAX device mesh.
    process_indices_train: A list of process indices that are loading data.

  Returns:
    A local data iterator for the training set.
  """
  import datasets  # pylint: disable=import-outside-toplevel

  train_ds = datasets.load_dataset(
      config.hf_path,
      data_dir=config.hf_data_dir,
      data_files=config.hf_train_files,
      split="train",
      streaming=True,
      token=config.hf_access_token,
  )
  local_iter = preprocessing_pipeline(
      dataloading_host_index=process_indices_train.index(jax.process_index()),
      dataloading_host_count=len(process_indices_train),
      global_mesh=global_mesh,
      dataset=train_ds,
      data_column_names=[config.train_data_columns],
      tokenize=config.tokenize_train_data,
      tokenizer_path=config.tokenizer_path,
      hf_access_token=config.hf_access_token,
      global_batch_size=config.global_batch_size_to_load,
      max_target_length=config.max_prefill_predict_length,
  )
  return local_iter


def create_data_iterator(config, mesh):
  """Creates a data iterator for GRPO training.

  This function determines which processes should load data and then creates a
  process-specific data iterator for the training prompts. It currently does
  not support evaluation data.

  Args:
    config: The configuration object containing data loading settings.
    mesh: The JAX device mesh.

  Returns:
    A data iterator that yields batches of training prompts.

  Raises:
    ValueError: If evaluation is configured (`eval_interval > 0`), as the
      GRPO input pipeline does not support it.
  """
  process_indices_train = input_pipeline_interface.get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )
  if config.eval_interval > 0:
    raise ValueError("GRPO input pipeline is not supported for eval data")
  train_iterator = input_pipeline_interface.create_process_specific_iterator(
      config, mesh, process_indices_train, make_hf_train_iterator
  )
  return train_iterator
