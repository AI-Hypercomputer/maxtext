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

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from input_pipeline._tfds_data_processing import make_tfds_iterator
from input_pipeline._grain_data_processing import make_grain_iterator
from input_pipeline._tfds_data_processing_c4_mlperf import make_c4_mlperf_iterator
from input_pipeline._hf_data_processing import make_hf_iterator
from input_pipeline._input_pipeline_utils import get_tokenizer
import multihost_dataloading


class SyntheticDataIterator:
  """Creates a synthetic data iterator for performance testing work"""

  def __init__(self, config, mesh):
    self.mesh = mesh
    self.config = config
    data_pspec = P(*config.data_sharding)
    data_pspec_shardings = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
    self.data_generator = jax.jit(
        SyntheticDataIterator.raw_generate_synthetic_data, out_shardings=data_pspec_shardings, static_argnums=0
    )

  def __iter__(self):
    return self

  def __next__(self):
    with self.mesh:
      return self.data_generator(self.config)

  @staticmethod
  def raw_generate_synthetic_data(config):
    """Generates a single batch of synthetic data"""
    output = {}
    output["inputs"] = jax.numpy.zeros((config.global_batch_size_to_load, config.max_target_length), dtype=jax.numpy.int32)
    output["inputs_position"] = jax.numpy.zeros(
        (config.global_batch_size_to_load, config.max_target_length), dtype=jax.numpy.int32
    )
    output["inputs_segmentation"] = jax.numpy.ones(
        (config.global_batch_size_to_load, config.max_target_length), dtype=jax.numpy.int32
    )
    output["targets"] = jax.numpy.zeros((config.global_batch_size_to_load, config.max_target_length), dtype=jax.numpy.int32)
    output["targets_position"] = jax.numpy.zeros(
        (config.global_batch_size_to_load, config.max_target_length), dtype=jax.numpy.int32
    )
    output["targets_segmentation"] = jax.numpy.ones(
        (config.global_batch_size_to_load, config.max_target_length), dtype=jax.numpy.int32
    )
    return output


class BadSyntheticDataIterator:
  """Creates a Bad synthetic data iterator for loading on subset of hosts"""

  def __init__(self, config, mesh):
    self.mesh = mesh
    dataset = BadSyntheticDataIterator.get_bad_synthetic_data(config)
    self.data_generator = multihost_dataloading.MultiHostDataLoadIterator(dataset, self.mesh)

  def __iter__(self):
    return self.data_generator

  def __next__(self):
    return next(self.data_generator)

  @staticmethod
  def get_bad_synthetic_data(config):
    """fill negative value in synthetic data"""
    output = {}
    output["inputs"] = tf.data.Dataset.from_tensor_slices(np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32))
    output["inputs_position"] = tf.data.Dataset.from_tensor_slices(
        np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32)
    )
    output["inputs_segmentation"] = tf.data.Dataset.from_tensor_slices(
        np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32)
    )
    output["targets"] = tf.data.Dataset.from_tensor_slices(np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32))
    output["targets_position"] = tf.data.Dataset.from_tensor_slices(
        np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32)
    )
    output["targets_segmentation"] = tf.data.Dataset.from_tensor_slices(
        np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32)
    )
    dataset = tf.data.Dataset.zip((output))  # pytype: disable=wrong-arg-types
    dataset = dataset.repeat()
    dataset = dataset.batch(config.global_batch_size_to_load // jax.process_count())
    return dataset


def get_process_loading_real_data(config, mesh):
  """Get list of processes loading data from GCS when expansion_factor_real_data != -1"""
  sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))
  devices_indices_map = sharding.devices_indices_map((config.global_batch_size_to_load, config.max_target_length))
  batch_cutoff = config.global_batch_size_to_train_on
  process_loading_real_data = set()
  for p, indices in devices_indices_map.items():
    if indices[0].stop <= batch_cutoff:
      process_loading_real_data.add(p.process_index)
  return list(process_loading_real_data)


def make_mixed_train_iterator(config, mesh, add_bos, add_eos):
  """Return iterators according to dataset_type"""
  process_indices = get_process_loading_real_data(config, mesh)
  if config.expansion_factor_real_data != -1:  # assert number of hosts loading real data
    assert len(process_indices) == jax.process_count() // config.expansion_factor_real_data
  if jax.process_index() in process_indices:
    if config.dataset_type == "tfds":
      return make_tfds_iterator(config, mesh, add_bos, add_eos, process_indices)
    elif config.dataset_type == "grain":
      return make_grain_iterator(config, mesh, add_bos, add_eos, process_indices)
    elif config.dataset_type == "c4_mlperf":
      print("Overwrite both add_bos and add_eos to False")
      return make_c4_mlperf_iterator(config, mesh, add_bos=False, add_eos=False, process_indices=process_indices)
    elif config.dataset_type == "hf":
      return make_hf_iterator(config, mesh, add_bos, add_eos, process_indices)
  else:
    return BadSyntheticDataIterator(config, mesh), None


def create_data_iterator(config, mesh, add_bos=True, add_eos=True):
  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None
  elif config.dataset_type in ("tfds", "grain", "c4_mlperf", "hf"):
    return make_mixed_train_iterator(config, mesh, add_bos, add_eos)
  else:
    assert False, f"Unknown dataset_type {config.dataset_type}, dataset_type must be synthetic, tfds, grain, hf or c4_mlperf"


def get_shaped_batch(config):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator, but eval_shape doesn't work, see b/306901078."""
  batch_shape = (config.global_batch_size_to_load, config.max_target_length)
  shaped_batch = {}
  shaped_batch["inputs"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  return shaped_batch


def get_shaped_batch_eval(config, mesh):
  """ Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078."""
  global_batch_size_to_load = config.global_batch_size_to_load
  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * mesh.size
  else:
    eval_batch_size = global_batch_size_to_load
  batch_shape = (eval_batch_size, config.max_target_length)

  shaped_batch = {}
  shaped_batch['inputs'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['inputs_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_position'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch['targets_segmentation'] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  return shaped_batch
