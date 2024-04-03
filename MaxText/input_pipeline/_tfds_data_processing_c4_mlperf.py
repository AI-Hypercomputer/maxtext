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

"""Input pipeline for gpt3 c4 mlperf dataset."""

from typing import Optional

import functools

import numpy as np

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

import tokenizer
import multihost_dataloading
import sequence_packing

AUTOTUNE = tf.data.experimental.AUTOTUNE

# data processing functions:
#   _shift_left_and_pad, rekey, reduce_concat_tokens and split_tokens_to_targets_length
# Adapted from:
#   https://github.com/google-research/text-to-text-transfer-transformer/blob/ba171b6f94eafcee60d0714fd6d60749b572d1f2/t5/data/preprocessors.py
# -----------------------------------------------------------------------------
def _shift_left_and_pad(tensor, pad_val):
  """Shift the input to the left with pad_val"""
  # Expand dims here so that the below code can work with 1-d tensors.
  v = tf.expand_dims(tensor, 0)
  # Make sure we keep tensor as ragged to allow for uneven concat.
  if isinstance(v, tf.Tensor):
    v = tf.RaggedTensor.from_tensor(v)

  # Append padding to the last item of every sequence.
  pad_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
  pad_tensor = tf.broadcast_to(pad_val, pad_shape)
  last_in_sequence = tf.concat([v[..., -1:, 1:], pad_tensor], axis=-1)
  # Concat back the newly modified final sequence item.
  v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
  # Un-expand outer dimension.
  v = v[0]
  return v

def rekey(ds, key_map=None):
  """normalization with key mapping"""
  def _rekey(x, key_map=None):
    """Replace the feature keys according to the mapping in `key_map`.
    For example, if the dataset returns examples of the format:
    {'foo': 'something', 'bar': 'something else', 'zoo': 'others'}
    and key_map = {'boo': 'foo', 'spar': 'bar', 'zoo': None} then this function will return
    examples with the format
    {'boo': 'something', 'spar': 'something else'}
    If a mapping is to None, then the key will be dropped.
    Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys
    Returns:
      A preprocessed example with the format listed above.
    """
    if key_map:
      return {
          new_key: x[old_key]
          for new_key, old_key in key_map.items() if old_key
      }
    return x

  return ds.map(
      functools.partial(_rekey, key_map=key_map),
      num_parallel_calls=AUTOTUNE)

def reduce_concat_tokens(dataset,
                         feature_key='targets',
                         batch_size=128,
                         ):
  """Token-preprocessor to concatenate multiple unrelated documents.
  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, folowed by
  split_tokens.
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one
  Returns:
    a dataset
  """
  dataset = dataset.map(
      lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})
  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)

def split_tokens(dataset,
                 max_tokens_per_segment=128,
                 feature_key='targets',
                 ):
  """Split examples into multiple examples each.
  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.
  This function is generally preceded by select_random_chunk.
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
  Returns:
    a dataset
  """
  def _split_tokens(x):
    """Split one token sequence into multiple multiple."""
    tokens = x[feature_key]
    n_tokens = tf.size(tokens)
    length = max_tokens_per_segment

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
        tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)),
        tf.int32)
    padding = num_segments * length - tf.size(tokens)
    tokens = tf.pad(tokens, [[0, padding]])
    return tf.reshape(tokens, [-1, length])

  def _strip_padding(x):
    return {feature_key: tf.boolean_mask(x, tf.cast(x, tf.bool))}

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  dataset = dataset.map(_split_tokens, num_parallel_calls=AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset.map(
      _strip_padding, num_parallel_calls=AUTOTUNE)

def split_tokens_to_targets_length(dataset, sequence_length):
  return split_tokens(dataset, max_tokens_per_segment=sequence_length)

def _pad_to_batch_size(ds: tf.data.Dataset,  batch_size: int, num_examples: Optional[int] = None,) -> tf.data.Dataset:
  """Pad unevenly distributed eval data in each shard with new entries to multiples of batch size."""

  # local_num represents the total number of examples in eval dataset,
  if num_examples:
    local_num = num_examples
  else:
    def _get_num_examples(ds: tf.data.Dataset) -> int:
      # Iterate one-by-one instead of len(list(...)) to reduce peak memory.
      num_examples = 0
      for _ in ds:
        num_examples += 1

      return num_examples

    local_num = _get_num_examples(ds)
  local_num_batches = (local_num + batch_size - 1) // batch_size
  # Find the max number of batches required across all Jax processes.
  num_batches_all = multihost_utils.process_allgather(
      jnp.array([local_num_batches]), tiled=False)
  num_batches = np.max(num_batches_all)

  pad_num = num_batches * batch_size - local_num
  assert pad_num >= 0
  print(
      f'Eval data has {local_num} local entries, padding now with '
      f'{pad_num} extra entries to get {num_batches} batches.')
  # Repeat a random example to make the last batch full.
  def _add_pad(x):
    x['targets_segmentation'] *= 0
    return x
  pad_ds = ds.take(1).map(_add_pad).repeat(pad_num)
  return ds.concatenate(pad_ds)

def get_datasets(
  config: ml_collections.ConfigDict,
  dataloading_host_index,
  dataloading_host_count,
):
  """Load and return dataset of batched examples for use during training."""
  # Training dataset.
  read_config = tfds.ReadConfig(
     shuffle_seed = config.data_shuffle_seed,
     )
  train_ds_builder = tfds.builder(config.dataset_name)
  train_ds = train_ds_builder.as_dataset(split='train2', read_config=read_config, shuffle_files=config.enable_data_shuffling)

  eval_ds_builder = tfds.builder(config.eval_dataset_name)
  eval_ds = eval_ds_builder.as_dataset(split='validation_tokenized_5662seqs', read_config=read_config, shuffle_files=False)

  # shard the dataset as soon as it is loaded
  train_ds = train_ds.shard(num_shards = dataloading_host_count, index = dataloading_host_index)
  train_ds = rekey(train_ds, {'inputs': None, 'targets': 'text'})

  eval_ds = eval_ds.shard(num_shards = dataloading_host_count, index = dataloading_host_index)
  # note validation_tokenized_5662seqs split is pre tokenized, reduce_concated and splitted to target_length
  #   mainly to avoid eval sequences change depending on the number of hosts
  eval_ds = rekey(eval_ds, {'inputs': None, 'targets': 'ids'})

  return train_ds, eval_ds

def preprocess_dataset(config: ml_collections.ConfigDict,
                       global_mesh,
                       train_ds, eval_ds, sp_tokenizer,
                       data_shuffle_seed: int = 0,
                       shuffle_buffer_size: int = 128,
                       ):
  """Pre-process the dataset and return iterators for mlperf training."""
  # tokenize
  train_ds = train_ds.map(
      tokenizer.TokenizeOp(sp_tokenizer, data_keys=('targets',)), num_parallel_calls=AUTOTUNE)

  train_ds = reduce_concat_tokens(train_ds, feature_key='targets', batch_size=4096)
  train_ds = split_tokens_to_targets_length(train_ds, config.max_target_length)
  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

  # note eval_ds is pre tokenized, reduce_concated and splitted to target_length
  #   mainly to avoid eval sequences change depending on the number of hosts
  train_ds = sequence_packing.pack_dataset(train_ds, config.max_target_length)
  eval_ds = sequence_packing.pack_dataset(eval_ds, config.max_target_length)

  def format_fn(x, eos_id: int = 1, pad_id: int = 0):
    x["inputs"] = x["targets"]
    x["inputs_position"] = x["targets_position"]
    x["targets"] = _shift_left_and_pad(x["targets"], eos_id)
    x["inputs_segmentation"] = tf.where(
      tf.logical_and(x["targets"] != eos_id, x["targets"] != pad_id),
      x["targets_segmentation"], 0)
    x["targets_segmentation"] = x["inputs_segmentation"]
    return x

  train_ds = train_ds.map(format_fn, num_parallel_calls=AUTOTUNE)
  eval_ds = eval_ds.map(format_fn, num_parallel_calls=AUTOTUNE)

  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
  else:
    eval_batch_size = global_batch_size_to_load

  train_ds = train_ds.batch(global_batch_size_to_load // jax.process_count(), drop_remainder=True)

  # ensure array split in an equal division for each device
  # pad zeros up to the same batch_size among all processes
  eval_ds = _pad_to_batch_size(eval_ds, eval_batch_size // jax.process_count())
  eval_ds = eval_ds.batch(eval_batch_size // jax.process_count(), drop_remainder=False)
  # We are running eval over exactly one epoch.
  # We explicitly cache the entire epoch (in memory) to ensure that it is the
  # same across different iterations.
  eval_ds = eval_ds.cache()

  train_ds = train_ds.prefetch(AUTOTUNE)
  eval_ds = eval_ds.prefetch(AUTOTUNE)

  train_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(train_ds, global_mesh)
  eval_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(eval_ds, global_mesh)

  # Return multi-host jax.Array prep iterator
  return train_multihost_gen, eval_multihost_gen
