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

# pylint: disable=unused-import
"""SPMD Multihost Dataloading Utilities.

Adapted from Sholto's:
https://github.com/sholtodouglas/multihost_dataloading
"""
from collections import defaultdict  # pylint: disable=g-importing-member
from dataclasses import dataclass  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
import os
from typing import Callable, Any, Dict, List, Tuple, Optional
import tensorflow as tf  # pylint: disable=g-import-not-at-top
import time
import numpy as np

import jax
from jax.experimental import global_device_array as gda_lib
from jax.experimental import PartitionSpec
from jax.experimental.maps import Mesh

import max_logging


Pytree = Any
Device = Any


DATA_DIM = 0  # assume data dimension is the first


def check_inputs(dataset, global_data_shape, data_axes):
  # pylint: disable=missing-function-docstring
  # dataset_structure = jax.tree_util.tree_structure(iter(dataset).next())
  dataset_structure = jax.tree_util.tree_structure(
      tf.data.experimental.get_structure(dataset)
  )
  global_data_shape_structure = jax.tree_util.tree_structure(global_data_shape)
  data_axes_structure = jax.tree_util.tree_structure(data_axes)
  try:
    assert (
        dataset_structure == global_data_shape_structure == data_axes_structure
    ), 'All inputs should have the same pytree structure.'
  except AssertionError as msg:
    (max_logging.log(
       f"""{msg} - The most likely reason for this is that global shapes should
       be array or classes not tuples, otherwise tree map enumerates indiviudal
       dimensions as leaves. Dataset: {dataset_structure}, \n Shapes:
       {global_data_shape_structure}, \n Axes: {data_axes_structure}"""))
  shapes, _ = jax.tree_util.tree_flatten(global_data_shape)
  batch_dims = [s[0] for s in shapes]
  assert all(
      b == batch_dims[0] for b in batch_dims
  ), 'All batch axis should be equal for gdas'
  assert all(
      b[0] == shapes[0][0] for b in shapes
  ), 'All dataset elements should be sharded along the data axis identically'
  batch_dim = batch_dims[0]
  return batch_dim

################################################################################
### Shard data parallelism over devices #####
################################################################################


def get_batch_sharded_data_pipeline(
    dataset: tf.data.Dataset, data_sharding, global_data_shape: np.ndarray, global_mesh: Mesh,
    data_axes: PartitionSpec) -> Callable[[], jax.Array]:
  """ Each device loads batch_size/num_devices,
  To do this, each host first loads batch_size/num_hosts, then shards that
  equally across it's devices.
  Args:
    dataset: tf dataset over all files
    data_sharding: data sharding axes
    global_data_shape: what the size of the GDA should be
    global_mesh: global devices mesh
    data_axes: axes along which data is partitioned
  Returns:
    sharded_dataset: per_host dataset
  """
  _ = check_inputs(dataset, global_data_shape, data_axes)

  dataset = iter(dataset.as_numpy_iterator())

  multihost_generator = partial(get_next_batch_sharded, dataset,
                   data_sharding, global_data_shape, global_mesh)

  return multihost_generator

def get_next_batch_sharded(local_dataset: tf.data.Dataset,
                           data_sharding,
                           global_data_shape: Pytree,
                           global_mesh: Mesh) -> jax.Array:
  """Splits the host loaded data equally over all devices."""


  try:
    local_data = local_dataset.next()
  except:
    max_logging.log("Failed to get next data batch, retrying")
    time.sleep(10)
    local_data = local_dataset.next()

  # local_devices = jax.local_devices()
  local_devices = global_mesh.local_devices
  local_device_count = jax.local_device_count()

  def _put_to_devices(x):
    try:
      per_device_arrays = np.split(x, local_device_count, axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f'Unable to put to devices shape {x.shape} with '
          f'local device count {local_device_count}') from array_split_error
    device_buffers = [
        jax.device_put(arr, d)
        for arr, d in zip(per_device_arrays, local_devices)
    ]
    return device_buffers
  # 'fully shard' the data (first) axis across both axes
  # of the hardware mesh. This is layout matches the
  # manual device placing we just did.
  input_sharding_constraint = PartitionSpec(*data_sharding, None)

  def form_gda(local_data, shape):
    device_buffers = _put_to_devices(local_data)
    #  Wrap device buffers as GDA
    shape = tuple(shape)
    input_gda = jax.make_array_from_single_device_arrays(shape,
        jax.sharding.NamedSharding(global_mesh, input_sharding_constraint), device_buffers)
    return input_gda

  input_gdas = jax.tree_map(form_gda, local_data, global_data_shape)

  return input_gdas

