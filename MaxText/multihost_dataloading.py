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
from functools import lru_cache, partial  # pylint: disable=g-importing-member
from typing import Callable, Any
import tensorflow as tf  # pylint: disable=g-import-not-at-top
import time
import numpy as np

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.sharding import Mesh

import max_logging

def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
  sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

  return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """ Put local sharded array into local devices
  """
  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)

  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
  except ValueError as array_split_error:
    raise ValueError(
      f"Unable to put to devices shape {array.shape} with "
      f"local device count {len(global_mesh.local_devices)} "
      f"at {jtu.keystr(path)}"
    ) from array_split_error

  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


def get_batch_sharded_data_pipeline(
  dataset_type, dataset: tf.data.Dataset, global_mesh: Mesh
) -> Callable[[], jax.Array]:
  """Each device loads batch_size/num_devices,
  To do this, each host first loads batch_size/num_hosts, then shards that
  equally across it's devices.
  Args:
    dataset: tf dataset over all files
  Returns:
    sharded_dataset: per_host dataset
  """
  if dataset_type == 'c4':
    dataset = iter(dataset.as_numpy_iterator())
  elif dataset_type == 'array_record':
    dataset = iter(dataset)
  else:
    raise ValueError('Unknow dataset_type, must be c4, c4-array_record or synthetic')

  multihost_generator = partial(get_next_batch_sharded, dataset, global_mesh)

  return multihost_generator


def get_next_batch_sharded(
  local_dataset: tf.data.Dataset, global_mesh: Mesh
) -> jax.Array:
  """Splits the host loaded data equally over all devices."""

  SLEEP_TIME = 10
  MAX_DATA_LOAD_ATTEMPTS = 30

  data_load_attempts = 0
  loaded_data_success = False
  while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
    data_load_attempts += 1
    try:
      local_data = next(local_dataset)
      loaded_data_success = True
    except tf.errors.FailedPreconditionError:
      max_logging.log("Failed to get next data batch, retrying")
      time.sleep(SLEEP_TIME)

  # Try one last time, if this fails we will see the full stack trace.
  if not loaded_data_success:
    local_data = next(local_dataset)

  input_gdas = jtu.tree_map_with_path(partial(_form_global_array, global_mesh = global_mesh), local_data)

  return input_gdas

# def get_next_batch_sharded_pygrain(data_iter,
#                            data_sharding,
#                            global_shape: Pytree,
#                            global_mesh: Mesh) -> jax.Array:
#   """Splits the host loaded data equally over all devices."""

#   SLEEP_TIME = 10
#   MAX_DATA_LOAD_ATTEMPTS = 30
#   data_load_attempts = 0
#   loaded_data_success = False
#   while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
#     data_load_attempts += 1
#     try:
#       local_data = next(data_iter)
#       loaded_data_success = True
#     except tf.errors.FailedPreconditionError:
#       max_logging.log("Failed to get next data batch, retrying")
#       time.sleep(SLEEP_TIME)
#   # Try one last time, if this fails we will see the full stack trace.
#   if not loaded_data_success:
#     local_data = next(data_iter)

#   global_data_shape = jax.tree_map(
#       lambda x: PartitionSpec(*global_shape), local_data
#   )
#   data_axes = jax.tree_map(lambda x: PartitionSpec(*data_sharding), local_data)
#   _ = check_inputs("array_record", local_data, global_data_shape, data_axes)

#   # local_devices = jax.local_devices()
#   local_devices = global_mesh.local_devices
#   local_device_count = jax.local_device_count()
#   print(f"local_device: {local_devices}")
#   print(f"local_device_count: {local_device_count}")

#   def _put_to_devices(x):
#     try:
#       per_device_arrays = np.split(x, local_device_count, axis=0)
#     except ValueError as array_split_error:
#       raise ValueError(
#           f'Unable to put to devices shape {x.shape} with '
#           f'local device count {local_device_count}') from array_split_error
#     device_buffers = [
#         jax.device_put(arr, d)
#         for arr, d in zip(per_device_arrays, local_devices)
#     ]
#     return device_buffers
#   # 'fully shard' the data (first) axis across both axes
#   # of the hardware mesh. This is layout matches the
#   # manual device placing we just did.
#   input_sharding_constraint = PartitionSpec(*data_sharding, None)

#   def form_gda(local_data, shape):
#     device_buffers = _put_to_devices(local_data)
#     #  Wrap device buffers as GDA
#     shape = tuple(shape)
#     print("####################### Debug")
#     print(f"shape: {shape}; global_mesh: {global_mesh}; ")
#     print(f"input_sharding_constraint: {input_sharding_constraint};")
#     print(f"jax.sharding.NamedSharding: {jax.sharding.NamedSharding(global_mesh, input_sharding_constraint)};")
#     input_gda = jax.make_array_from_single_device_arrays(shape,
#         jax.sharding.NamedSharding(global_mesh, input_sharding_constraint), device_buffers)
#     return input_gda

#   input_gdas = jax.tree_map(form_gda, local_data, global_data_shape)

#   return input_gdas