"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Input pipeline for a LM1B dataset."""

from typing import Optional
import warnings

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax

import multihost_dataloading
import tokenizer
import sequence_packing
from input_pipeline import _input_pipeline_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_datasets(
    dataset_name,
    data_split,
    shuffle_files,
    shuffle_seed,
    dataloading_host_index,
    dataloading_host_count,
):
  """Load a TFDS dataset."""
  ds_builder = tfds.builder(dataset_name)

  if shuffle_files:
    read_config = tfds.ReadConfig(shuffle_seed=shuffle_seed)
  else:
    read_config = tfds.ReadConfig()

  if ds_builder.info.splits[data_split].num_shards >= dataloading_host_count:
    read_config.input_context = tf.distribute.InputContext(
        input_pipeline_id=dataloading_host_index,
        num_input_pipelines=dataloading_host_count,
    )
    ds = ds_builder.as_dataset(split=data_split, read_config=read_config, shuffle_files=shuffle_files)
  else:
    warnings.warn(
        f"WARNING: Inefficient dataloading. Your {dataset_name} contains {ds_builder.info.splits[data_split].num_shards} shards, "
        f"smaller than {dataloading_host_count=}. This is known to lead to inefficient dataloading."
        "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
    )
    ds = ds_builder.as_dataset(split=data_split, read_config=read_config, shuffle_files=shuffle_files)
    ds = ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)

  return ds


def preprocessing_pipeline(
    dataset,
    tokenizer_path,
    global_batch_size: int,
    global_mesh,
    max_target_length: int,
    data_column_name,
    shuffle: bool = False,
    data_shuffle_seed=0,
    tokenize: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
    num_epochs: Optional[int] = 1,
    pack_examples: bool = True,
    shuffle_buffer_size: int = 1024,
    shift: bool = True,
    drop_remainder: bool = True,
    prefetch_size=tf.data.experimental.AUTOTUNE,
):
  """pipeline for preprocessing TFDS dataset."""
  dataset = dataset.map(lambda x: _input_pipeline_utils.normalize_features(x, data_column_name), num_parallel_calls=AUTOTUNE)

  if tokenize:
    tokenizer_model = _input_pipeline_utils.get_tokenizer(tokenizer_path, add_bos, add_eos)
    dataset = dataset.map(lambda x: tokenizer.TokenizeOp(tokenizer=tokenizer_model, features=x), num_parallel_calls=AUTOTUNE)

  if max_target_length > 0:
    # We can take upto max_length+1 because there would be truncation by 1 token
    # for both inputs and targets
    dataset = dataset.map(
        lambda x: _input_pipeline_utils.truncate_to_max_allowable_length(x, max_target_length + 1),
        num_parallel_calls=AUTOTUNE,
    )

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # Shift inputs for teacher-forced training
  if shift:
    dataset = dataset.map(
        _input_pipeline_utils.shift_data_by_truncation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

  # Perform greedy sequence packing and batching
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."
  if pack_examples:
    dataset = sequence_packing.pack_dataset(dataset, max_target_length)
    dataset = dataset.batch(global_batch_size // jax.process_count(), drop_remainder=drop_remainder)
  else:
    # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        global_batch_size // jax.process_count(),
        padded_shapes={"inputs": max_target_length, "targets": max_target_length},
        padding_values={"inputs": 0, "targets": 0},
        drop_remainder=drop_remainder,
    )

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


def make_tfds_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_train,
):
  """load dataset, preprocess and return iterators"""
  train_ds = get_datasets(
      dataset_name=config.dataset_name,
      data_split="train",
      shuffle_files=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      dataloading_host_index=process_indices_train.index(jax.process_index()),
      dataloading_host_count=len(process_indices_train),
  )
  train_iter = preprocessing_pipeline(
      dataset=train_ds,
      tokenizer_path=config.tokenizer_path,
      global_batch_size=config.global_batch_size_to_load,
      global_mesh=global_mesh,
      max_target_length=config.max_target_length,
      data_column_name=config.train_data_column,
      shuffle=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed,
      tokenize=config.tokenize_train_data,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
  )
  return train_iter


def make_tfds_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_eval,
):
  eval_ds = get_datasets(
      dataset_name=config.eval_dataset_name,
      data_split=config.eval_split,
      shuffle_files=False,
      shuffle_seed=config.data_shuffle_seed,
      dataloading_host_index=process_indices_eval.index(jax.process_index()),
      dataloading_host_count=len(process_indices_eval),
  )

  eval_iter = preprocessing_pipeline(
      dataset=eval_ds,
      tokenizer_path=config.tokenizer_path,
      global_batch_size=config.global_batch_size_to_load_eval,
      global_mesh=global_mesh,
      max_target_length=config.max_target_length,
      data_column_name=config.eval_data_column,
      shuffle=False,
      data_shuffle_seed=config.data_shuffle_seed,
      tokenize=config.tokenize_eval_data,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
  )

  return eval_iter
