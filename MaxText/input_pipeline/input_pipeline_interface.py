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
from MaxText.input_pipeline._syn_data_processing import SyntheticDataIterator, PlaceHolderDataIterator

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
    train_iterator = PlaceHolderDataIterator(config, mesh)

  if config.eval_interval <= 0:
    eval_iterator = None
  else:
    if jax.process_index() in process_indices_eval:
      eval_iterator = eval_iterator_fn()
    else:
      eval_iterator = PlaceHolderDataIterator(config, mesh)
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
