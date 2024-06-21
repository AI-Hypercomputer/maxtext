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

import os
from typing import Optional

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax

import multihost_dataloading
import tokenizer
import sequence_packing

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Right-shifting token inputs for teacher-forced training.
# -----------------------------------------------------------------------------


def shift_right_tf(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = tf.pad(x, tf.constant(pad_widths), mode="constant", constant_values=tf.constant(0, x.dtype))
  return padded[tuple(slices)]


def shift_inputs_tf(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right_tf(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= tf.cast(segment_ids == shift_right_tf(segment_ids, axis=axis), x.dtype)
  return shifted


def shift_data(x, axis=0, segmented=True):
  segment_ids = x["inputs_segmentation"] if segmented else None
  x["inputs"] = shift_inputs_tf(x["inputs"], segment_ids=segment_ids, axis=axis)
  return x


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def normalize_features(ds):
  """Normalize text feature keys."""

  def _normalize_features(features):
    features["inputs"] = features.pop("text")
    features["targets"] = features["inputs"]
    return features

  return ds.map(_normalize_features, num_parallel_calls=AUTOTUNE)


def length_trim(ds, max_len):
  """ "Trim to Max length"""

  def _trim_fn(features):
    if tf.shape(features["inputs"])[0] > max_len:
      features["inputs"] = features["inputs"][:max_len]
    if tf.shape(features["targets"])[0] > max_len:
      features["targets"] = features["targets"][:max_len]
    return features

  return ds.map(_trim_fn, num_parallel_calls=AUTOTUNE)


# -----------------------------------------------------------------------------
# Main dataset preparation.
# -----------------------------------------------------------------------------


def preprocessing_pipeline(
    dataset,
    batch_size: int,
    global_mesh,
    shuffle: bool,
    num_epochs: Optional[int] = 1,
    pack_examples: bool = True,
    shuffle_buffer_size: int = 1024,
    max_length: int = 512,
    shift: bool = True,
    drop_remainder: bool = True,
    prefetch_size=tf.data.experimental.AUTOTUNE,
    data_shuffle_seed=0,
):
  """Shuffle and batch/pack the given dataset."""

  def truncate_to_max_allowable_length(x, max_length):
    x["inputs"] = x["inputs"][:max_length]
    x["targets"] = x["targets"][:max_length]
    return x

  if max_length > 0:
    # We can take upto max_length+1 because there would be truncation by 1 token
    # for both inputs and targets
    dataset = dataset.map(lambda x: truncate_to_max_allowable_length(x, max_length + 1))

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # Shift inputs for teacher-forced training
  if shift:
    dataset = dataset.map(shift_data_by_truncation, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

  # Perform greedy sequence packing
  if pack_examples:
    dataset = sequence_packing.pack_dataset(dataset, max_length)
  assert batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  # Batch examples.
  if pack_examples:
    dataset = dataset.batch(batch_size // jax.process_count(), drop_remainder=drop_remainder)
  else:
    # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        batch_size // jax.process_count(),
        padded_shapes={"inputs": max_length, "targets": max_length},
        padding_values={"inputs": 0, "targets": 0},
        drop_remainder=drop_remainder,
    )

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


def get_datasets(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
    read_config=None,
):
  """Load and return dataset of batched examples for use during training."""
  # Training dataset.
  train_ds_builder = tfds.builder(config.dataset_name)
  # train_data = get_raw_dataset(train_ds_builder, 'train')
  train_ds = train_ds_builder.as_dataset(split="train", read_config=read_config, shuffle_files=config.enable_data_shuffling)
  # shard the dataset as soon as it is loaded
  train_ds = train_ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
  train_ds = normalize_features(train_ds)

  # Evaluation dataset.
  if config.eval_dataset_name:
    eval_ds_builder = tfds.builder(config.eval_dataset_name)
  else:
    eval_ds_builder = train_ds_builder
  # eval_data = get_raw_dataset(eval_ds_builder, config.eval_split)
  eval_ds = eval_ds_builder.as_dataset(split=config.eval_split, read_config=read_config, shuffle_files=False)
  eval_ds = eval_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
  eval_ds = normalize_features(eval_ds)

  return train_ds, eval_ds


def preprocess_dataset(
    config: ml_collections.ConfigDict,
    global_mesh,
    train_ds,
    eval_ds,
    sp_tokenizer,
    data_shuffle_seed=0,
):
  """Pre-process the dataset and return iterators"""
  # Tokenize data.
  train_ds = train_ds.map(lambda x: tokenizer.TokenizeOp(tokenizer=sp_tokenizer, features=x), num_parallel_calls=AUTOTUNE)
  eval_ds = eval_ds.map(lambda x: tokenizer.TokenizeOp(tokenizer=sp_tokenizer, features=x), num_parallel_calls=AUTOTUNE)

  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
  else:
    eval_batch_size = global_batch_size_to_load

  def filter_keys(record):
    return {"inputs": record["inputs"], "targets": record["targets"]}

  train_ds = train_ds.map(filter_keys, num_parallel_calls=tf.data.AUTOTUNE)
  eval_ds = eval_ds.map(filter_keys, num_parallel_calls=tf.data.AUTOTUNE)

  train_iter = preprocessing_pipeline(
      train_ds,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=None,
      pack_examples=True,
      max_length=config.max_target_length,
      shift=True,
      data_shuffle_seed=data_shuffle_seed,
  )

  eval_iter = preprocessing_pipeline(
      eval_ds,
      eval_batch_size,
      global_mesh,
      shuffle=False,
      pack_examples=False,
      max_length=config.max_target_length,
      shift=False,
      drop_remainder=False,
      data_shuffle_seed=data_shuffle_seed,
  )

  predict_iter = preprocessing_pipeline(
      eval_ds,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=False,
      max_length=config.max_target_length,
      shift=False,
      drop_remainder=False,
      data_shuffle_seed=data_shuffle_seed,
  )

  return train_iter, eval_iter, predict_iter
