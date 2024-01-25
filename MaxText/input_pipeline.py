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
import functools

import numpy as np

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental import multihost_utils

import tokenizer
import multihost_dataloading
import sequence_packing

AUTOTUNE = tf.data.experimental.AUTOTUNE

# left-shifting token inputs for teacher-forced training.
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

# Right-shifting token inputs for teacher-forced training.
# -----------------------------------------------------------------------------

def shift_right_tf(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [slice(None),] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = tf.pad(
      x,
      tf.constant(pad_widths),
      mode='constant',
      constant_values=tf.constant(0, x.dtype))
  return padded[tuple(slices)]


def shift_inputs_tf(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right_tf(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= tf.cast(
        segment_ids == shift_right_tf(segment_ids, axis=axis), x.dtype
    )
  return shifted

def shift_data(x, axis=0, segmented=True):
  segment_ids = x['inputs_segmentation'] if segmented else None
  x['inputs'] = shift_inputs_tf(x['inputs'], segment_ids=segment_ids, axis=axis)
  return x

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
  prefetch_size = tf.data.experimental.AUTOTUNE,
  data_shuffle_seed = 0,
):
  """Shuffle and batch/pack the given dataset."""

  # Max length filter.
  def length_filter(max_len):
    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)
    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed = data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # Perform greedy sequence packing
  if pack_examples:
    dataset = sequence_packing.pack_dataset(dataset, max_length)

  # Shift inputs for teacher-forced training
  if shift:
    dataset = dataset.map(
      functools.partial(shift_data, axis=0, segmented=pack_examples),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True)

  assert (
        batch_size % global_mesh.size == 0
    ), 'Batch size should be divisible number of global devices.'

  # Batch examples.
  if pack_examples:
    dataset = dataset.batch(batch_size // jax.process_count(), drop_remainder=drop_remainder)
  else:
    # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        batch_size // jax.process_count(),
        padded_shapes={'inputs': max_length, 'targets': max_length},
        padding_values={'inputs': 0, 'targets': 0},
        drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen

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

def _pad_to_batch_size(ds: tf.data.Dataset,  batch_size: int, num_examples: int = None,) -> tf.data.Dataset:
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
  read_config = None,
):
  """Load and return dataset of batched examples for use during training."""
  # Training dataset.
  train_ds_builder = tfds.builder(config.dataset_name)
  # train_data = get_raw_dataset(train_ds_builder, 'train')
  train_ds = train_ds_builder.as_dataset(split='train',
                                           read_config = read_config,
                                           shuffle_files=config.enable_data_shuffling)
  # shard the dataset as soon as it is loaded
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())
  train_ds = rekey(train_ds, {'inputs': 'text', 'targets': 'text', 'text': None})

  # Evaluation dataset.
  if config.eval_dataset_name:
    eval_ds_builder = tfds.builder(config.eval_dataset_name)
  else:
    eval_ds_builder = train_ds_builder
  # eval_data = get_raw_dataset(eval_ds_builder, config.eval_split)
  eval_ds = eval_ds_builder.as_dataset(split=config.eval_split,
                                          read_config = read_config,
                                          shuffle_files=config.enable_data_shuffling)
  eval_ds = eval_ds.shard(num_shards = jax.process_count(), index = jax.process_index())
  eval_ds = rekey(eval_ds, {'inputs': 'text', 'targets': 'text', 'text': None})

  return train_ds, eval_ds

def preprocess_dataset(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds, eval_ds,
                        vocab_path: Optional[str] = None,
                        data_shuffle_seed = 0,):
  """Pre-process the dataset and return iterators"""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

  # Load tokenizer
  sp_tokenizer = tokenizer.load_tokenizer(vocab_path=vocab_path,
                                          add_bos=config.add_bos,
                                          add_eos=config.add_eos)

  # Tokenize data.
  train_ds = train_ds.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  eval_ds = eval_ds.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
  else:
    eval_batch_size = global_batch_size_to_load

  def filter_keys(record):
    return {'inputs': record['inputs'], 'targets': record['targets']}
  train_ds = train_ds.map(filter_keys,num_parallel_calls=tf.data.AUTOTUNE)
  eval_ds = eval_ds.map(filter_keys,num_parallel_calls=tf.data.AUTOTUNE)

  train_iter = preprocessing_pipeline(
      train_ds,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=None,
      pack_examples=True,
      max_length=config.max_target_length,
      shift=True,
      data_shuffle_seed = data_shuffle_seed,)

  eval_iter = preprocessing_pipeline(
      eval_ds,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=False,
      max_length=config.max_target_length,
      shift=False,
      data_shuffle_seed = data_shuffle_seed,)

  predict_iter = preprocessing_pipeline(
      eval_ds,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=False,
      max_length=config.max_target_length,
      shift=False,
      drop_remainder=False,
      data_shuffle_seed = data_shuffle_seed,)

  return train_iter, eval_iter, predict_iter, sp_tokenizer


def make_c4_train_iterator_and_tokenizer(config, mesh):
  """ Make train iterator and tokenizer for C4 dataset"""
  read_config = tfds.ReadConfig(
    shuffle_seed = config.data_shuffle_seed,
  )
  train_ds, eval_ds = get_datasets(
    config=config,
    read_config = read_config,
  )
  train_iter, _, _, sp_tokenizer = preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds,
    vocab_path=os.path.join(config.assets_path, config.vocab_relative_path),
    data_shuffle_seed = config.data_shuffle_seed,
  )
  return train_iter, None, sp_tokenizer

class SyntheticDataIterator():
  """Creates a synthetic data iterator for performance testing work"""
  def __init__(self, config, mesh):
    self.mesh = mesh
    self.config = config
    data_pspec = P(*config.data_sharding)
    data_pspec_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
    self.data_generator = jax.jit(SyntheticDataIterator.raw_generate_synthetic_data,
        out_shardings=data_pspec_shardings,
        static_argnums=0)

  def __iter__(self):
    return self

  def __next__(self):
    with self.mesh:
      return self.data_generator(self.config)

  @staticmethod
  def raw_generate_synthetic_data(config):
    """Generates a single batch of syntehtic data"""
    output = {}
    output['inputs'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                       dtype=jax.numpy.int32)
    output['inputs_position'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                                dtype=jax.numpy.int32)
    output['inputs_segmentation'] = jax.numpy.ones( (config.global_batch_size_to_load, config.max_target_length),
                                                   dtype=jax.numpy.int32)
    output['targets'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                        dtype=jax.numpy.int32)
    output['targets_position'] = jax.numpy.zeros( (config.global_batch_size_to_load, config.max_target_length),
                                                 dtype=jax.numpy.int32)
    output['targets_segmentation'] = jax.numpy.ones( (config.global_batch_size_to_load, config.max_target_length),
                                                    dtype=jax.numpy.int32)
    return output

def create_data_iterator_with_tokenizer(config, mesh):
  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None, None
  elif config.dataset_type == "c4":
    return make_c4_train_iterator_and_tokenizer(config, mesh)
  elif config.dataset_type == "c4_mlperf":
    return make_c4_mlperf_train_iterator_and_tokenizer(config, mesh)
  else:
    assert False, "dataset type not implemented"

def get_shaped_batch(config):
  """ Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""
  batch_shape = (config.global_batch_size_to_load, config.max_target_length)
  shaped_batch = {}
  shaped_batch['inputs'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  return shaped_batch

def make_c4_mlperf_train_iterator_and_tokenizer(config, mesh, shuffle_buffer_size=128):
  """ Make train iterator and tokenizer for C4 mlperf dataset"""

  print(
    "Given customized setup in c4_mlperf dataset, the following data related keys will be reset:\n",
    f"reset dataset_path from {config.dataset_path} to gs://mlperf-llm-public2\n",
    f"reset train dataset_name from {config.dataset_name} to c4/en:3.0.4\n",
    f"reset eval_dataset_name from {config.eval_dataset_name} to c4/en:3.0.5\n",
    f"reset vocat_path from {os.path.join(config.assets_path, config.vocab_relative_path)} to"
     "gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model \n",
    )

  os.environ["TFDS_DATA_DIR"] = "gs://mlperf-llm-public2"
  read_config = tfds.ReadConfig(
    shuffle_seed = config.data_shuffle_seed,
    skip_prefetch = True,
  )
  train_ds_builder = tfds.builder('c4/en:3.0.4')
  train_ds = train_ds_builder.as_dataset(split='train2', read_config = read_config, shuffle_files=True)

  eval_ds_builder = tfds.builder('c4/en:3.0.5', try_gcs=True)
  eval_ds = eval_ds_builder.as_dataset(split='validation_tokenized_5662seqs', read_config = read_config, shuffle_files=False)

  # shard the dataset as soon as it is loaded
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())
  train_ds = rekey(train_ds, {'inputs': None, 'targets': 'text'})
  sp_tokenizer = tokenizer.load_tokenizer(
    vocab_path='gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model',
    add_bos=config.add_bos,
    add_eos=config.add_eos,
    )

  # tokenize
  train_ds = train_ds.map(
      tokenizer.TokenizeOp(sp_tokenizer, data_keys=('targets',)), num_parallel_calls=AUTOTUNE)

  train_ds = reduce_concat_tokens(train_ds, feature_key='targets', batch_size=4096)
  train_ds = split_tokens_to_targets_length(train_ds, config.max_target_length)

  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=config.data_shuffle_seed)

  eval_ds = eval_ds.shard(num_shards = jax.process_count(), index = jax.process_index())
  # note validation_tokenized_5662seqs split is pre tokenized, reduce_concated and splitted to target_length
  #   mainly to avoid eval sequences change depending on the number of hosts
  eval_ds = rekey(eval_ds, {'inputs': None, 'targets': 'ids'})

  train_ds = sequence_packing.pack_dataset(train_ds, config.max_target_length)
  eval_ds = sequence_packing.pack_dataset(eval_ds, config.max_target_length)

  def map_fn(x, eos_id=1, pad_id=0):
    x["inputs"] = x["targets"]
    x["inputs_position"] = x["targets_position"]
    x["inputs_segmentation"] = x["targets_segmentation"]
    x["targets"] = _shift_left_and_pad(x["targets"], eos_id)
    x["targets_segmentation"] = tf.where(
      tf.logical_and(x["targets"] != eos_id, x["targets"] != pad_id),
      x["targets_segmentation"], 0)
    return x

  train_ds = train_ds.map(map_fn, num_parallel_calls=AUTOTUNE)
  eval_ds = eval_ds.map(map_fn, num_parallel_calls=AUTOTUNE)

  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * mesh.size
  else:
    eval_batch_size = global_batch_size_to_load

  train_ds = train_ds.batch(global_batch_size_to_load // jax.process_count(), drop_remainder=True)

  # ensure array split in an equal division for each device
  eval_ds = _pad_to_batch_size(eval_ds, eval_batch_size // jax.process_count())
  eval_ds = eval_ds.batch(eval_batch_size // jax.process_count(), drop_remainder=False)

  # We are running eval over exactly one epoch.
  # We explicitly cache the entire epoch (in memory) to ensure that it is the
  # same across different iterations.
  eval_ds = eval_ds.cache()
  train_ds = train_ds.prefetch(AUTOTUNE)
  eval_ds = eval_ds.prefetch(AUTOTUNE)

  train_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(train_ds, mesh)
  eval_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(eval_ds, mesh)

  # Return multi-host jax.Array prep iterator
  return train_multihost_gen, eval_multihost_gen, sp_tokenizer
