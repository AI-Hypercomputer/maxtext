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

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P

import tokenizer
import multihost_dataloading
import sequence_packing

AUTOTUNE = tf.data.experimental.AUTOTUNE


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


def normalize_features(ds):
  """Normalize text feature keys."""
  def _normalize_features(features):
    features['inputs'] = features.pop('text')
    features['targets'] = features['inputs']
    return features

  return ds.map(
      _normalize_features,
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
  data_sharding = None,
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

  # Multihost dataloading: sharding and jax.Array prep function
  dataset_structure = tf.data.experimental.get_structure(dataset)
  global_data_shape = jax.tree_map(
      lambda x: P(batch_size, max_length), dataset_structure
  )
  data_axes = jax.tree_map(lambda x: P(*data_sharding), dataset_structure)


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

  multihost_gen = (
      multihost_dataloading.get_batch_sharded_data_pipeline(
          dataset, data_sharding, global_data_shape, global_mesh, data_axes
      )
  )
  # Return multi-host jax.Array prep iterator
  return multihost_gen


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
  train_ds = normalize_features(train_ds)

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
  eval_ds = normalize_features(eval_ds)

  return train_ds, eval_ds

def preprocess_dataset(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds, eval_ds,
                        vocab_path: Optional[str] = None,
                        data_shuffle_seed = 0,):
  """Pre-process the dataset and return iterators"""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

  # Train or load tokenizer
  sp_tokenizer = tokenizer.load_or_train_tokenizer(
      train_ds,
      vocab_path=vocab_path,
      vocab_size=config.vocab_size,
      max_corpus_chars=config.max_corpus_chars)

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
      data_sharding = config.data_sharding,
      data_shuffle_seed = data_shuffle_seed,)

  eval_iter = preprocessing_pipeline(
      eval_ds,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=False,
      max_length=config.max_eval_target_length,
      shift=False,
      data_sharding = config.data_sharding,
      data_shuffle_seed = data_shuffle_seed,)

  predict_iter = preprocessing_pipeline(
      eval_ds,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=False,
      max_length=config.max_predict_length,
      shift=False,
      drop_remainder=False,
      data_sharding = config.data_sharding,
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
    vocab_path=os.path.join(config.base_output_directory, config.vocab_relative_path),
    data_shuffle_seed = config.data_shuffle_seed,
  )
  return train_iter, sp_tokenizer

class SyntheticDataIterator():
  """Creates a synthetic data iterator for performance testing work"""
  def __init__(self, config, mesh):
    self.mesh = mesh
    self.config = config
    data_pspec = P(*config.data_sharding)
    with self.mesh:
      self.data_generator = pjit(SyntheticDataIterator.raw_generate_synthetic_data,
        in_shardings=None,
        out_shardings=data_pspec,
        static_argnums=0)
  def __call__(self):
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
    return SyntheticDataIterator(config, mesh), None
  elif config.dataset_type == "c4":
    return make_c4_train_iterator_and_tokenizer(config, mesh)
  else:
    assert False, "dataset type not implemented"
