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

"""Input pipeline"""

import os

import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from input_pipeline import _tfds_data_processing
from input_pipeline import _grain_data_processing
from input_pipeline import _tfds_data_processing_c4_mlperf
from input_pipeline import _hf_data_processing
import tokenizer

def get_tokenizer(tokenizer_path, add_bos=True, add_eos=True):
  # Load tokenizer
  sp_tokenizer = tokenizer.load_tokenizer(tokenizer_path=tokenizer_path,
                                          add_bos=add_bos,
                                          add_eos=add_eos)
  return sp_tokenizer

def make_c4_mlperf_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  """ Make train iterator and tokenizer for customized C4 dataset for mlperf gpt3 training."""
  train_ds, eval_ds = _tfds_data_processing_c4_mlperf.get_datasets(
    config=config,
  )
  sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, eval_iter = _tfds_data_processing_c4_mlperf.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds, sp_tokenizer,
    data_shuffle_seed=config.data_shuffle_seed
  )
  return train_iter, eval_iter, sp_tokenizer

def make_c4_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  """ Make train iterator and tokenizer for C4 dataset"""
  read_config = tfds.ReadConfig(
    shuffle_seed = config.data_shuffle_seed,
  )
  train_ds, eval_ds = _tfds_data_processing.get_datasets(
    config=config,
    read_config = read_config,
  )
  sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, _, _ = _tfds_data_processing.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds, sp_tokenizer,
    data_shuffle_seed = config.data_shuffle_seed,
  )
  return train_iter, None, sp_tokenizer

def make_hf_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  train_ds, eval_ds = _hf_data_processing.get_datasets(
    config=config
  )
  #sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, _, _ = _hf_data_processing.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds,
    tokenizer_path=config.tokenizer_path,
    data_shuffle_seed = config.data_shuffle_seed,
    add_bos = add_bos,
    add_eos = add_eos
  )
  return train_iter, None, None

def make_grain_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos):
  """ Make train iterator and tokenizer for C4 dataset"""
  train_ds, eval_ds = _grain_data_processing.get_datasets(
    config=config
  )
  sp_tokenizer = get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  train_iter, _, _ = _grain_data_processing.preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds,
    vocab_path=config.tokenizer_path,
    data_shuffle_seed = config.data_shuffle_seed,
    add_bos = add_bos,
    add_eos = add_eos
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

def create_data_iterator_with_tokenizer(config, mesh, add_bos = True, add_eos = True):
  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None, get_tokenizer(config.tokenizer_path, add_bos, add_eos)
  elif config.dataset_type == "c4":
    return make_c4_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos)
  elif config.dataset_type == "c4-array_record":
    return make_grain_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos)
  elif config.dataset_type == "c4_mlperf":
    print("Overwrite both add_bos and add_eos to False")
    return make_c4_mlperf_train_iterator_and_tokenizer(config, mesh, add_bos=False, add_eos=False)
  elif config.dataset_type == "hf":
    return make_hf_train_iterator_and_tokenizer(config, mesh, add_bos, add_eos)
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
