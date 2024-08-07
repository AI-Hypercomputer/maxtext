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
    read_config=None,
):
  """Load a TFDS dataset."""
  ds_builder = tfds.builder(dataset_name)
  ds = ds_builder.as_dataset(split=data_split, read_config=read_config, shuffle_files=shuffle_files)

  return ds


def preprocessing_pipeline(
    dataset,
    tokenizer_path,
    global_batch_size: int,
    global_mesh,
    max_target_length: int,
    dataloading_host_index,
    dataloading_host_count,
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
  dataset = dataset.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
  dataset = dataset.map(lambda x: _input_pipeline_utils.normalize_features(x, data_column_name), num_parallel_calls=AUTOTUNE)

  if tokenize:
    tokenizer_model = _input_pipeline_utils.get_tokenizer(tokenizer_path, add_bos, add_eos)
    dataset = dataset.map(lambda x: tokenizer.TokenizeOp(tokenizer=tokenizer_model, features=x), num_parallel_calls=AUTOTUNE)

  if max_target_length > 0:
    # We can take upto max_length+1 because there would be truncation by 1 token
    # for both inputs and targets
    dataset = dataset.map(lambda x: _input_pipeline_utils.truncate_to_max_allowable_length(x, max_target_length + 1), num_parallel_calls=AUTOTUNE)

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
  
  # repeat forever
  dataset = dataset.repeat()

  # Shift inputs for teacher-forced training
  if shift:
    dataset = dataset.map(_input_pipeline_utils.shift_data_by_truncation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

  # Perform greedy sequence packing
  if pack_examples:
    dataset = sequence_packing.pack_dataset(dataset, max_target_length)

  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."
  # Batch examples.
  if pack_examples:
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


def make_tfds_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """load dataset, preprocess and return iterators"""
  read_config = tfds.ReadConfig(
    shuffle_seed=config.data_shuffle_seed,
  )
  train_ds = get_datasets(
    dataset_name=config.dataset_name,
    data_split='train',
    shuffle_files=config.enable_data_shuffling,
    read_config=read_config,
  )
  train_iter = preprocessing_pipeline(
    dataset=train_ds,
    tokenizer_path=config.tokenizer_path,
    global_batch_size=config.global_batch_size_to_load,
    global_mesh=global_mesh,
    max_target_length=config.max_target_length,
    dataloading_host_index=process_indices.index(jax.process_index()),
    dataloading_host_count=len(process_indices),
    data_column_name=config.train_data_column,
    shuffle=config.enable_data_shuffling,
    data_shuffle_seed=config.data_shuffle_seed,
    tokenize=config.tokenize_train_data,
    add_bos=config.add_bos,
    add_eos=config.add_eos,
  )

  if config.eval_interval > 0:
    eval_ds = get_datasets(
      dataset_name=config.eval_dataset_name,
      data_split=config.eval_split,
      shuffle_files=False,
    )

    if config.eval_per_device_batch_size > 0:
      eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
      eval_batch_size = config.global_batch_size_to_load

    eval_iter = preprocessing_pipeline(
        dataset=eval_ds,
        tokenizer_path=config.tokenizer_path,
        global_batch_size=eval_batch_size,
        global_mesh=global_mesh,
        max_target_length=config.max_target_length,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        data_column_name=config.eval_data_column,
        shuffle=False,
        data_shuffle_seed=config.data_shuffle_seed,
        tokenize=config.tokenize_eval_data,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
    )
  else:
    eval_iter = None

  return train_iter, eval_iter
