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
import re
from typing import Optional
import functools

# from array_record.python import array_record_data_source
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
import grain.python as pygrain
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tokenizer
import multihost_dataloading
import sequence_packing
import pygrain_operations
import pygrain_tokenizer

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

# Max length filter.
def length_filter(max_len):
  def filter_fn(x):
    source, target = x['inputs'], x['targets']
    l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
    return tf.less(l, max_len + 1)
  return filter_fn

def add_annotations(ds):
  def _add_annotations(features):
    features['inputs_segmentation'] = tf.ones_like(features['inputs'], dtype = tf.int32)
    features['inputs_position'] = tf.range(tf.size(features['inputs']), dtype = tf.int32)
    features['targets_segmentation'] = tf.ones_like(features['targets'], dtype = tf.int32)
    features['targets_position'] = tf.range(tf.size(features['targets']), dtype = tf.int32)
    # wherever target element is 0, set corresponding segmentation to 0
    nonzero_mask = tf.not_equal(features['targets'], 0)
    nonzero_mask = tf.cast(nonzero_mask, tf.int32)
    features['inputs_segmentation'] *= nonzero_mask
    features['targets_segmentation'] *= nonzero_mask
    return features

  return ds.map(
      _add_annotations,
      num_parallel_calls=AUTOTUNE)
# -----------------------------------------------------------------------------
# Main dataset preparation.
# -----------------------------------------------------------------------------


def preprocessing_pipeline(
  dataset,
  dataset_type,
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

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  # Shuffle and repeat.
  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size, seed = data_shuffle_seed)

  dataset = dataset.repeat(num_epochs)

  # Perform greedy sequence packing
  if pack_examples:
    dataset = sequence_packing.pack_dataset(dataset, max_length)
  else:
    # dataset = add_annotations(dataset)
    dataset.apply(add_annotations)

  # Shift inputs for teacher-forced training
  if shift:
    dataset = dataset.map(
      functools.partial(shift_data, axis=0, segmented=1),
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
        padded_shapes={
                        'inputs': max_length, 
                        'targets': max_length,
                        'inputs_position': max_length,
                        'targets_position': max_length,
                        'inputs_segmentation': max_length,
                        'targets_segmentation': max_length,
                      },
        padding_values={'inputs': 0, 'targets': 0,
                        'inputs_position': 0,
                        'targets_position': 0,
                        'inputs_segmentation': 0,
                        'targets_segmentation': 0,                       
                        },
        drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  multihost_gen = multihost_dataloading.get_batch_sharded_data_pipeline(dataset_type, dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


def preprocessing_pipeline_pygrain(
  dataset,
  dataset_type,
  grain_worker_count,
  vocab_path,
  batch_size: int,
  global_mesh,
  shuffle: bool,
  num_epochs: Optional[int] = 1,
  pack_examples: bool = True,
  shuffle_buffer_size: int = 1024,
  max_length: int = 512,
  shift: bool = True,
  drop_remainder: bool = True,
  data_sharding = None,
  data_shuffle_seed = 0,
):
  """Apply pygrain operations to preprocess the given dataset."""
  operations = []
  operations.append(pygrain_operations.ParseFeatures())
  operations.append(pygrain_operations.NormalizeFeatures())
  operations.append(pygrain_tokenizer.Tokenize(["inputs","targets"], max_length, vocab_path))
  operations.append(pygrain_operations.LengthFilter(max_length))

  # Pack and Batch examples.
  if pack_examples:
    operations.append(pygrain.experimental.PackAndBatchOperation(
                        batch_size=batch_size // jax.process_count(),
                        length_struct={'inputs':max_length,'targets':max_length}))
    operations.append(pygrain_operations.ReformatPacking())
  else:
    operations.append(pygrain_operations.PadToMaxLength(max_length))
    # operations.append(pygrain.MapOperation(map_function=pygrain_operations.PadToMaxLength(max_length)))
    operations.append(pygrain.Batch(batch_size=batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  if shift:
    operations.append(pygrain_operations.ShiftData(axis=1))

  index_sampler = pygrain.IndexSampler(
    num_records=len(dataset),
    num_epochs = num_epochs,
    shard_options=pygrain.ShardOptions(
      shard_index = jax.process_index(), shard_count = jax.process_count(), drop_remainder = True
    ),
    shuffle = shuffle,
    seed = data_shuffle_seed
  )

  dataloader = pygrain.DataLoader(
      data_source = dataset,
      operations = operations,
      sampler = index_sampler,
      worker_count=grain_worker_count,
  )

  data_iter = iter(dataloader)

  return data_iter


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

def get_datasets_pygrain(
  config: ml_collections.ConfigDict,
  read_config = None,
):
  """Load dataset from array_record files for using with pygrain"""
  data_dir = os.path.join(config.dataset_path, config.dataset_name)
  train_files = [data_dir + '/' + f for f in os.listdir(data_dir) if re.match(r'.*train.*', f)]
  train_ds = pygrain.ArrayRecordDataSource(train_files)
  if config.eval_dataset_name:
    eval_files = [data_dir + '/' + f for f in os.listdir(data_dir) if re.match(rf'.*{config.eval_split}.*', f)]
    eval_ds = pygrain.ArrayRecordDataSource(eval_files)
  else:
    eval_ds = train_ds


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
                                          vocab_size=config.vocab_size)

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
      config.dataset_type,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=None,
      pack_examples=config.pack_examples,
      max_length=config.max_target_length,
      shift=True,
      data_shuffle_seed = data_shuffle_seed,)

  eval_iter = preprocessing_pipeline(
      eval_ds,
      config.dataset_type,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=config.pack_examples,
      max_length=config.max_eval_target_length,
      shift=False,
      data_shuffle_seed = data_shuffle_seed,)

  predict_iter = preprocessing_pipeline(
      eval_ds,
      config.dataset_type,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=config.pack_examples,
      max_length=config.max_predict_length,
      shift=False,
      drop_remainder=False,
      data_shuffle_seed = data_shuffle_seed,)

  return train_iter, eval_iter, predict_iter, sp_tokenizer

def preprocess_dataset_pygrain(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds, eval_ds,
                        vocab_path: Optional[str] = None,
                        data_shuffle_seed = 0,):
  """PyGrain: Pre-process the dataset and return iterators"""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

  # Load tokenizer
  sp_tokenizer = tokenizer.load_tokenizer(vocab_path=vocab_path,
                                          vocab_size=config.vocab_size)
                                          
  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
  else:
    eval_batch_size = global_batch_size_to_load

  train_iter = preprocessing_pipeline_pygrain(
      train_ds,
      config.dataset_type,
      config.grain_worker_count,
      vocab_path,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=1,
      pack_examples=config.pack_examples,
      max_length=config.max_target_length,
      shift=True,
      data_sharding=config.data_sharding,
      data_shuffle_seed = data_shuffle_seed,)

  eval_iter = preprocessing_pipeline_pygrain(
      eval_ds,
      config.dataset_type,
      config.grain_worker_count,
      vocab_path,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=config.pack_examples,
      max_length=config.max_eval_target_length,
      shift=True,
      data_sharding=config.data_sharding,
      data_shuffle_seed = data_shuffle_seed,)

  predict_iter = preprocessing_pipeline_pygrain(
      eval_ds,
      config.dataset_type,
      config.grain_worker_count,
      vocab_path,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=config.pack_examples,
      max_length=config.max_eval_target_length,
      shift=True,
      data_sharding=config.data_sharding,
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
  return train_iter, sp_tokenizer

def make_pygrain_train_iterator_and_tokenizer(config, mesh):
  """ Make train iterator and tokenizer for C4 dataset"""
  read_config = tfds.ReadConfig(
    shuffle_seed = config.data_shuffle_seed,
  )
  train_ds, eval_ds = get_datasets_pygrain(
    config=config,
    read_config = read_config,
  )
  train_iter, _, _, sp_tokenizer = preprocess_dataset_pygrain(
    config,
    mesh,
    train_ds, eval_ds,
    vocab_path=os.path.join(config.assets_path, config.vocab_relative_path),
    data_shuffle_seed = config.data_shuffle_seed,
  )
  return train_iter, sp_tokenizer

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
  elif config.dataset_type == "array_record":
    return make_pygrain_train_iterator_and_tokenizer(config, mesh)
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
