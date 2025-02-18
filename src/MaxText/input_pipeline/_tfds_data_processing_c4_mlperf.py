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

"""Input pipeline for gpt3 c4 mlperf dataset."""

import functools

import numpy as np

import ml_collections

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from MaxText import tokenizer
from MaxText import multihost_dataloading
from MaxText import sequence_packing
from MaxText import max_logging
from MaxText.input_pipeline._input_pipeline_utils import get_tokenizer

AUTOTUNE = tf.data.experimental.AUTOTUNE


# data processing functions:
#   _shift_left_and_pad, rekey, reduce_concat_tokens and split_tokens_to_targets_length
# Adapted from:
#   https://github.com/google-research/text-to-text-transfer-transformer/blob/ba171b6/t5/data/preprocessors.py
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
      return {new_key: x[old_key] for new_key, old_key in key_map.items() if old_key}
    return x

  return ds.map(functools.partial(_rekey, key_map=key_map), num_parallel_calls=AUTOTUNE)


def reduce_concat_tokens(
    dataset,
    feature_key="targets",
    batch_size=128,
):
  """Token-preprocessor to concatenate multiple unrelated documents.
  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, followed by
  split_tokens.
  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one
  Returns:
    a dataset
  """
  dataset = dataset.map(lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})

  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


def split_tokens(
    dataset,
    max_tokens_per_segment=128,
    feature_key="targets",
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
    num_segments = tf.cast(tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)), tf.int32)
    padding = num_segments * length - tf.size(tokens)
    tokens = tf.pad(tokens, [[0, padding]])
    return tf.reshape(tokens, [-1, length])

  def _strip_padding(x):
    return {feature_key: tf.boolean_mask(x, tf.cast(x, tf.bool))}

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  dataset = dataset.map(_split_tokens, num_parallel_calls=AUTOTUNE)
  dataset = dataset.unbatch()
  return dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)


def split_tokens_to_targets_length(dataset, sequence_length):
  return split_tokens(dataset, max_tokens_per_segment=sequence_length)


def _pad_to_batch_size(
    ds: tf.data.Dataset,
    batch_size: int,
    num_examples: None | int = None,
) -> tf.data.Dataset:
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
  num_batches_all = multihost_utils.process_allgather(jnp.array([local_num_batches]), tiled=False)
  num_batches = np.max(num_batches_all)

  pad_num = num_batches * batch_size - local_num
  assert pad_num >= 0
  max_logging.log(
      f"Eval data has {local_num} local entries, padding now with "
      f"{pad_num} extra entries to get {num_batches} batches."
  )

  # Repeat a random example to make the last batch full.
  def _add_pad(x):
    x["targets_segmentation"] *= 0
    return x

  pad_ds = ds.take(1).map(_add_pad).repeat(pad_num)
  return ds.concatenate(pad_ds)


def get_dataset(
    dataset_name: str,
    split: str,
    dataloading_host_index: int,
    dataloading_host_count: int,
    enable_data_shuffling: bool = False,
    data_shuffle_seed: int = 0,
    shard_in_read: bool = False,
) -> tf.data.Dataset:
  """Load and return a dataset of examples."""
  # enable shard_in_read in training dataset
  if split == "train2":
    shard_in_read = True
    max_logging.log(
        f"overwriting {shard_in_read=} with {dataloading_host_count=}"
    )
  if shard_in_read:
    # shard dataset in reading
    read_config = tfds.ReadConfig(
        shuffle_seed=data_shuffle_seed,
        input_context=tf.distribute.InputContext(
            input_pipeline_id=dataloading_host_index,
            num_input_pipelines=dataloading_host_count,
        ),
    )
    ds_builder = tfds.builder(dataset_name)
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split=split, read_config=read_config, shuffle_files=enable_data_shuffling)
  else:
    # shard dataset after reading
    read_config = tfds.ReadConfig(shuffle_seed=data_shuffle_seed)
    ds_builder = tfds.builder(dataset_name)
    ds = ds_builder.as_dataset(split=split, read_config=read_config, shuffle_files=enable_data_shuffling)
    ds = ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
  return ds


def format_fn(x, eos_id: int = 1, pad_id: int = 0):
  """Format function for c4_mlperf."""
  x["inputs"] = x["targets"]
  x["inputs_position"] = x["targets_position"]
  x["targets"] = _shift_left_and_pad(x["targets"], eos_id)
  x["inputs_segmentation"] = tf.where(
      tf.logical_and(x["targets"] != eos_id, x["targets"] != pad_id), x["targets_segmentation"], 0
  )
  x["targets_segmentation"] = x["inputs_segmentation"]
  return x


def preprocess_train_dataset(
    train_ds: tf.data.Dataset,
    sp_tokenizer,
    train_global_batch_size_to_load: int,
    max_target_length: int,
    shuffle_buffer_size: int,
    data_shuffle_seed: int,
) -> tf.data.Dataset:
  """Preprocess the training dataset."""
  if sp_tokenizer.pad_id is not None:
    pad_id = sp_tokenizer.pad_id
  elif sp_tokenizer.unk_id is not None:
    pad_id = sp_tokenizer.unk_id
  else:
    pad_id = -1
  train_ds = train_ds.map(
      lambda x: tokenizer.TokenizeOp(tokenizer=sp_tokenizer, features=x, data_keys=("targets",)),
      num_parallel_calls=AUTOTUNE,
  )
  train_ds = reduce_concat_tokens(train_ds, feature_key="targets", batch_size=4096)
  train_ds = split_tokens_to_targets_length(train_ds, max_target_length)
  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
  train_ds = sequence_packing.pack_dataset(train_ds, max_target_length, pad_id=pad_id)
  train_ds = train_ds.map(lambda x: format_fn(x, pad_id=pad_id), num_parallel_calls=AUTOTUNE)
  train_ds = train_ds.batch(train_global_batch_size_to_load // jax.process_count(), drop_remainder=True)
  train_ds = train_ds.prefetch(AUTOTUNE)
  return train_ds


def preprocess_eval_dataset(
    eval_ds: tf.data.Dataset,
    sp_tokenizer,
    eval_global_batch_size_to_load: int,
    max_target_length: int,
    num_examples: None | int = None,
    is_tokenized_dataset: bool = True,
) -> tf.data.Dataset:
  """Preprocess the evaluation dataset."""
  # group text up to max_target_length if the dataset is not pre-tokenized/pre-processed
  if not is_tokenized_dataset:
    eval_ds = eval_ds.map(
        lambda x: tokenizer.TokenizeOp(tokenizer=sp_tokenizer, features=x, data_keys=("targets",)),
        num_parallel_calls=AUTOTUNE,
    )
    # hardcode batch_sizes 24567 i.e. the exp size in split validation_24567exp
    #   to avoid padding tokens inserted in group text
    eval_ds = reduce_concat_tokens(eval_ds, feature_key="targets", batch_size=24567)
    eval_ds = split_tokens_to_targets_length(eval_ds, max_target_length)

  if sp_tokenizer.pad_id is not None:
    pad_id = sp_tokenizer.pad_id
  elif sp_tokenizer.unk_id is not None:
    pad_id = sp_tokenizer.unk_id
  else:
    pad_id = -1
  eval_ds = sequence_packing.pack_dataset(eval_ds, max_target_length, pad_id=pad_id)

  eval_ds = eval_ds.map(lambda x: format_fn(x, pad_id=pad_id), num_parallel_calls=AUTOTUNE)

  # ensure array split in an equal division for each device
  # pad zeros up to the same batch_size among all processes
  eval_ds = _pad_to_batch_size(eval_ds, eval_global_batch_size_to_load // jax.process_count(), num_examples)
  eval_ds = eval_ds.batch(eval_global_batch_size_to_load // jax.process_count(), drop_remainder=False)

  # We are running eval over exactly one epoch.
  # We explicitly cache the entire epoch (in memory) to ensure that it is the
  # same across different iterations.
  eval_ds = eval_ds.cache()
  eval_ds = eval_ds.prefetch(AUTOTUNE)

  return eval_ds


def make_c4_mlperf_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Make train iterator of customized C4 dataset for mlperf gpt3 training."""
  train_ds = get_dataset(
      dataset_name=config.dataset_name,
      split="train2",
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      enable_data_shuffling=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed,
  )
  train_ds = rekey(train_ds, {"inputs": None, "targets": "text"})

  sp_tokenizer = get_tokenizer(
      config.tokenizer_path, config.tokenizer_type, config.add_bos, config.add_eos, config.hf_access_token
  )
  train_ds = preprocess_train_dataset(
      train_ds,
      sp_tokenizer=sp_tokenizer,
      train_global_batch_size_to_load=config.global_batch_size_to_load,
      max_target_length=config.max_target_length,
      shuffle_buffer_size=128,
      data_shuffle_seed=config.data_shuffle_seed,
  )
  train_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(train_ds, global_mesh)
  return train_multihost_gen


def make_c4_mlperf_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Make eval iterator of customized C4 dataset for mlperf gpt3 training."""
  eval_slit = "None"
  if config.eval_dataset_name == "c4/en:3.0.5":
    is_tokenized_dataset = True
  elif config.eval_dataset_name == "c4/en:3.0.4":
    is_tokenized_dataset = False
    eval_slit = "validation_24567exp"
  elif config.eval_dataset_name in ["c4/en:3.0.1", "c4/en:3.0.8", "c4/en:3.0.9"]:
    is_tokenized_dataset = False
    eval_slit = "validation"
  else:
    raise ValueError(f"{config.eval_dataset_name=} should be one of ('c4/en:3.0.1', 'c4/en:3.0.4', 'c4/en:3.0.5')")

  if is_tokenized_dataset:
    eval_ds = get_dataset(
        dataset_name=config.eval_dataset_name,
        split="validation_tokenized_5662seqs",
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        enable_data_shuffling=False,
    )
    # note validation_tokenized_5662seqs split is pre tokenized, reduce_concated and split to target_length
    #   mainly to avoid eval sequences change depending on the number of hosts
    eval_ds = rekey(eval_ds, {"inputs": None, "targets": "ids"})
  else:
    eval_ds = get_dataset(
        dataset_name=config.eval_dataset_name,
        split=eval_slit,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        enable_data_shuffling=False,
    )

    eval_ds = rekey(eval_ds, {"inputs": None, "targets": "text"})

  sp_tokenizer = get_tokenizer(
      config.tokenizer_path, config.tokenizer_type, config.add_bos, config.add_eos, config.hf_access_token
  )
  eval_ds = preprocess_eval_dataset(
      eval_ds,
      sp_tokenizer=sp_tokenizer,
      eval_global_batch_size_to_load=config.global_batch_size_to_load_eval,
      max_target_length=config.max_target_length,
      is_tokenized_dataset=is_tokenized_dataset,
  )

  eval_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(eval_ds, global_mesh)

  # Return multi-host jax.Array prep iterator
  return eval_multihost_gen
