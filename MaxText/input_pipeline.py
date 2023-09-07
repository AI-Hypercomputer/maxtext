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
from typing import Optional, Dict

import ml_collections

import tensorflow as tf
import tensorflow_datasets as tfds
from array_record.python import array_record_data_source
import jax

import tokenizer
import multihost_dataloading
import numpy as np

import grain.python as pygrain

AUTOTUNE = tf.data.experimental.AUTOTUNE
Features = Dict[str, tf.Tensor]

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

class normalize_features():
  """Normalize text feature keys."""

  def __call__(self, features):
    def _normalize_features(features):
      features['inputs'] = features.pop('text')
      features['targets'] = features['inputs']
      return features
    return _normalize_features(features)


# -----------------------------------------------------------------------------
# Main dataset preparation.
# -----------------------------------------------------------------------------


def preprocessing_pipeline(
  dataset,
  operations,
  batch_size: int,
  shuffle: bool,
  num_epochs: Optional[int] = 1,
  pack_examples: bool = True,
  max_length: int = 512,
  shift: bool = True,
  drop_remainder: bool = True,
  data_shuffle_seed = 0,
):
  """Shuffle and batch/pack the given dataset."""

  # Max length filter.
  class length_filter():
    def __init__(self,max_length):
      self.max_len = max_length
    def __call__(self, x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, self.max_len + 1)
  operations.append(pygrain.FilterOperation(condition_function = length_filter(max_length)))



  # Padd examples.
  class PadToMaxLength():
    """Pads each input to the specified length.
    """

    def __init__(self, feature_lengths):
      self.feature_lengths = feature_lengths

    def __call__(self, data):
      def pad(x, max_length):
        pad_amount = max(max_length - x.shape[0], 0)
        pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
        return np.pad(x, pad_amount)
      data['inputs_segmentation'] = np.ones(data['inputs'].shape)
      data['inputs_position'] = np.ones(data['inputs'].shape, dtype = np.int32)
      for key, _ in data.items():
        data[key] = pad(data[key], self.feature_lengths)
      return data

  class CombineKeys():
    """ Combine tuples of sequence packing output in different keys
    """
    def __call__(self, data):
      combined_data = data[0]
      segments = data[1]
      segments['inputs_segmentation'] = segments.pop('inputs')
      segments['targets_segmentation'] = segments.pop('targets')
      positions = data[2]
      positions['inputs_position'] = positions.pop('inputs')
      positions['targets_position'] = positions.pop('targets')
      combined_data.update(segments)
      combined_data.update(positions)
      return combined_data

  # Pack and Batch examples.
  if pack_examples:
    operations.append(pygrain.experimental.PackAndBatchOperation(
                        batch_size=batch_size // jax.process_count(),
                        length_struct={'inputs':max_length,'targets':max_length}))
    operations.append(pygrain.MapOperation(map_function=CombineKeys()))
  else:
    operations.append(pygrain.MapOperation(map_function=PadToMaxLength(max_length)))
    operations.append(pygrain.BatchOperation(batch_size=batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  class ShiftData():
    def __init__(self, axis = 0, segmented=True):
      self.axis = axis
      self.segmented = segmented

    def __call__(self, x):
      segment_ids = x['inputs_segmentation'] if self.segmented else None
      x['inputs'] = shift_inputs_tf(x['inputs'], segment_ids=segment_ids, axis=self.axis)
      return x

  if shift:
    operations.append(pygrain.MapOperation(map_function=ShiftData(axis=0,segmented=pack_examples)))


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
    worker_count=10,
  )
  data_iter = iter(dataloader)
  # Return PyGrainIterator
  return data_iter


def get_next_sharded_batch(config: ml_collections.ConfigDict, data_iter, global_mesh):
  """ returns next sharded batch
  """
  # Set global batch size.
  batch_size = config.per_device_batch_size * global_mesh.size
  assert (
        batch_size % global_mesh.size == 0
    ), 'Batch size should be divisible number of global devices.'
  data_sharding = config.data_sharding

  multihost_gen = (
      multihost_dataloading.get_batch_sharded_data_pipeline(
          data_iter, data_sharding, global_mesh, global_shape = (batch_size, config.max_target_length),
      )
  )
  return multihost_gen

def get_datasets(
  config: ml_collections.ConfigDict,
):
  """Load and return dataset of batched examples for use during training."""
  train_ds = tfds.data_source(
    config.dataset_name, split='train'
  )

  # list files
  train_files = [train_ds.dataset_info.data_dir + '/' + f for f in os.listdir(train_ds.dataset_info.data_dir) if re.match(r'.*train*', f)]
  array_train_ds = array_record_data_source.ArrayRecordDataSource(
    train_files
  )
  if config.eval_dataset_name:
    eval_ds = tfds.data_source(config.eval_dataset_name, split=config.eval_split)
    eval_files = [eval_ds.dataset_info.data_dir + '/' + f for f in os.listdir(eval_ds.dataset_info.data_dir) if re.match(r'.*test*', f)]
    array_eval_ds = array_record_data_source.ArrayRecordDataSource(
      eval_files
    )
  else:
    array_eval_ds = array_train_ds

  return array_train_ds, array_eval_ds

def preprocess_dataset(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds,
                        vocab_path: Optional[str] = None,
                        data_shuffle_seed = 0,):
  """Pre-process the dataset and return iterators"""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

  # Train or load tokenizer
  # Use tf.data.Dataset for training the tokenizer

  sp_tokenizer = tokenizer.load_or_train_tokenizer(
      train_ds,
      vocab_path=vocab_path,
      vocab_size=config.vocab_size,
      max_corpus_chars=config.max_corpus_chars)

  # Tokenize data.
  class TokenizeOperation():
    """ TokenizeOp
    """
    def __init__(self, sp_tokenizer):
      self.sp_tokenizer = sp_tokenizer

    def __call__(self, features: Features) -> Features:
      data_keys = ('inputs', 'targets')
      for k in data_keys:
        features[k] = sp_tokenizer.tokenize(features[k]).numpy()
      return features

  operations = [pygrain.MapOperation(map_function=normalize_features())]
  operations.append(pygrain.MapOperation(map_function=TokenizeOperation(sp_tokenizer)))


  def filter_keys(record):
    return {'inputs': record['inputs'], 'targets': record['targets']}
  operations.append(pygrain.MapOperation(map_function=filter_keys))

  batch_size = config.per_device_batch_size * global_mesh.size

  #TODO: change pack_examples and shift to True once figured out sequence_packing.py
  train_iter = preprocessing_pipeline(
      train_ds,
      operations,
      batch_size,
      shuffle=config.enable_data_shuffling,
      pack_examples=True,
      max_length=config.max_target_length,
      shift=True,
      data_shuffle_seed = data_shuffle_seed)

  return train_iter, sp_tokenizer
