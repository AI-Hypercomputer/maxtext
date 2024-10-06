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
import functools
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from input_pipeline._tfds_data_processing import make_tfds_train_iterator, make_tfds_eval_iterator
from input_pipeline._grain_data_processing import make_grain_train_iterator, make_grain_eval_iterator
from input_pipeline._tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator
from input_pipeline._hf_data_processing import make_hf_train_iterator, make_hf_eval_iterator
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


def get_process_loading_real_data(
    data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh
):
  """Get list of processes loading data from GCS when expansion_factor_real_data != -1"""
  sharding = jax.sharding.NamedSharding(mesh, P(*data_sharding))
  devices_indices_map = sharding.devices_indices_map((global_batch_size_to_load, max_target_length))
  batch_cutoff = global_batch_size_to_train_on
  process_loading_real_data = set()
  for p, indices in devices_indices_map.items():
    if indices[0].stop <= batch_cutoff:
      process_loading_real_data.add(p.process_index)
  return list(process_loading_real_data)


def make_mixed_iterator(config, mesh, process_indices_train, process_indices_eval, train_iterator_fn, eval_iterator_fn):
  """Return iterators according to dataset_type"""
  # train_iterator, eval_iterator = make_tfds_iterator(config, mesh, process_indices_train, process_indices_eval)
  if jax.process_index() in process_indices_train:
    train_iterator = train_iterator_fn()
  else:
    train_iterator = BadSyntheticDataIterator(config, mesh)

  if config.eval_interval <= 0:
    eval_iterator = None
  else:
    if jax.process_index() in process_indices_eval:
      eval_iterator = eval_iterator_fn()
    else:
      eval_iterator = BadSyntheticDataIterator(config, mesh)
  return train_iterator, eval_iterator


def create_data_iterator(config, mesh):
  if config.dataset_type == "synthetic":
    return SyntheticDataIterator(config, mesh), None

  process_indices_train = get_process_loading_real_data(
      config.data_sharding,
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.max_target_length,
      mesh,
  )
  if config.eval_interval > 0:
    process_indices_eval = get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load_eval,
        config.global_batch_size_to_eval_on,
        config.max_target_length,
        mesh,
    )
  else:
    process_indices_eval = []

  if config.expansion_factor_real_data != -1:  # assert number of hosts loading real data
    assert len(process_indices_train) == jax.process_count() // config.expansion_factor_real_data
    if config.eval_interval > 0:
      assert len(process_indices_eval) == jax.process_count() // config.expansion_factor_real_data
  if config.dataset_type == "tfds":
    train_iterator_fn = functools.partial(make_tfds_train_iterator, config, mesh, process_indices_train)
    eval_iterator_fn = functools.partial(make_tfds_eval_iterator, config, mesh, process_indices_eval)
  elif config.dataset_type == "grain":
    train_iterator_fn = functools.partial(make_grain_train_iterator, config, mesh, process_indices_train)
    eval_iterator_fn = functools.partial(make_grain_eval_iterator, config, mesh, process_indices_eval)
  elif config.dataset_type == "hf":
    train_iterator_fn = functools.partial(make_hf_train_iterator, config, mesh, process_indices_train)
    eval_iterator_fn = functools.partial(make_hf_eval_iterator, config, mesh, process_indices_eval)
  elif config.dataset_type == "c4_mlperf":
    train_iterator_fn = functools.partial(make_c4_mlperf_train_iterator, config, mesh, process_indices_train)
    eval_iterator_fn = functools.partial(make_c4_mlperf_eval_iterator, config, mesh, process_indices_eval)
  else:
    assert False, f"Unknown dataset_type {config.dataset_type}, dataset_type must be synthetic, tfds, grain, hf or c4_mlperf"
  return make_mixed_iterator(config, mesh, process_indices_train, process_indices_eval, train_iterator_fn, eval_iterator_fn)


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
