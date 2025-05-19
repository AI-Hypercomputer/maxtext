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
from collections.abc import Callable
from typing import Any
import functools

import numpy as np

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from MaxText import multihost_dataloading
from MaxText import pyconfig
from MaxText.input_pipeline._grain_data_processing import make_grain_train_iterator, make_grain_eval_iterator
from MaxText.input_pipeline._hf_data_processing import make_hf_train_iterator, make_hf_eval_iterator
from MaxText.input_pipeline._tfds_data_processing import make_tfds_train_iterator, make_tfds_eval_iterator
from MaxText.input_pipeline._tfds_data_processing_c4_mlperf import make_c4_mlperf_train_iterator, make_c4_mlperf_eval_iterator


class SyntheticDataIterator:
  """Creates a synthetic data iterator for performance testing work"""

  data_generator: Callable[[pyconfig.HyperParameters, tuple[Any, ...]], dict]

  def __init__(self, config, mesh):
    self.mesh = mesh
    self.config = config
    data_pspec = P(*config.data_sharding)
    data_pspec_shardings = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
    self.data_generator = jax.jit(
        SyntheticDataIterator.raw_generate_synthetic_data, out_shardings=data_pspec_shardings, static_argnums=0
    )

    tokens = jax.random.randint(
        jax.random.PRNGKey(0),
        (config.global_batch_size_to_load, config.max_target_length + 1),
        0,
        config.vocab_size,
        dtype=jnp.int32,
    )

    sequence_positions = jnp.arange(0, config.max_target_length + 1, dtype=jnp.int32).reshape(1, -1)
    batch_positions = jnp.broadcast_to(sequence_positions, (config.global_batch_size_to_load, config.max_target_length + 1))
    segmentation = jnp.ones((config.global_batch_size_to_load, config.max_target_length), dtype=jnp.int32)
    self.data = (tokens, batch_positions, segmentation)

  def __iter__(self):
    return self

  def __next__(self):
    with self.mesh:
      return self.data_generator(self.config, self.data)  # pylint: disable=not-callable

  @staticmethod
  def raw_generate_synthetic_data(config: pyconfig.HyperParameters, data):
    """Generates a single batch of synthetic data"""
    tokens, positions, segmentation = data

    output = {}
    output["inputs"] = tokens[:, :-1]
    output["inputs_position"] = positions[:, :-1]
    output["inputs_segmentation"] = segmentation
    output["targets"] = tokens[:, 1:]
    output["targets_position"] = positions[:, 1:]
    output["targets_segmentation"] = segmentation
    return output


class BadSyntheticDataIterator:
  """Creates a Bad synthetic data iterator for loading on subset of hosts"""

  def __init__(self, config: pyconfig.HyperParameters, mesh):
    self.mesh = mesh
    dataset = BadSyntheticDataIterator.get_bad_synthetic_data(config)
    self.data_generator = multihost_dataloading.MultiHostDataLoadIterator(dataset, self.mesh)

  def __iter__(self):
    return self.data_generator

  def __next__(self):
    return next(self.data_generator)

  @staticmethod
  def get_bad_synthetic_data(config: pyconfig.HyperParameters):
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
    if not indices[0].stop or indices[0].stop <= batch_cutoff:
      process_loading_real_data.add(p.process_index)
  return list(process_loading_real_data)


def make_mixed_iterator(
    config: pyconfig.HyperParameters, mesh, process_indices_train, process_indices_eval, train_iterator_fn, eval_iterator_fn
):
  """Return iterators according to dataset_type"""
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


def create_data_iterator(config: pyconfig.HyperParameters, mesh):
  """create data iterator"""
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
    assert config.packing, "c4_mlperf dataloader only works with packing. For padded version, use tfds dataloader"
    train_iterator_fn = functools.partial(make_c4_mlperf_train_iterator, config, mesh, process_indices_train)
    eval_iterator_fn = functools.partial(make_c4_mlperf_eval_iterator, config, mesh, process_indices_eval)
  else:
    assert False, f"Unknown dataset_type {config.dataset_type}, dataset_type must be synthetic, tfds, grain, hf or c4_mlperf"
  return make_mixed_iterator(config, mesh, process_indices_train, process_indices_eval, train_iterator_fn, eval_iterator_fn)
