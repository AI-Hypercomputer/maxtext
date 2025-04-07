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
from typing import Callable, Any, Union, Sequence
from collections.abc import Iterator, Iterable
import tensorflow as tf  # pylint: disable=g-import-not-at-top
import time
import numpy as np

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jax.experimental import colocated_python
import jax.numpy as jnp
import grain.python as grain

from MaxText import maxtext_utils
from MaxText import max_logging


def _build_global_shape(local_shape: tuple[int, ...]) -> tuple[tuple[int, ...], NamedSharding]:
  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape


def _calculate_input_data_sharding_prod(mesh: Mesh, input_data_shardings: NamedSharding) -> int:
  """Calculate the product of the mesh sizes for the batch dimension of input data sharding."""
  try:
    physical_axes = input_data_shardings.spec[0]
  except AttributeError:
    raise ValueError(
        f"input_data_shardings does not have a .spec attribute (expected a PartitionSpec), got: {type(input_data_shardings)}"
    )
  except (IndexError, TypeError):
    raise ValueError(f"input_data_shardings.spec is not indexable: {input_data_shardings.spec}")

  # If it's a string (single axis), convert to tuple for consistent handling
  if isinstance(physical_axes, str):
    physical_axes = (physical_axes,)

  # Calculate product of mesh sizes for these dimensions
  product = 1
  for dim_name in physical_axes:
    if dim_name in mesh.shape:
      product *= mesh.shape[dim_name]

  return product


def _form_global_array(
    path, array: np.ndarray, global_mesh: Mesh, input_data_shardings: NamedSharding, input_data_sharding_factor: int
) -> jax.Array:
  """Put local sharded array into local devices"""
  global_shape = _build_global_shape(np.shape(array))

  try:
    local_device_arrays = np.split(array, input_data_sharding_factor, axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
        f"at {jtu.keystr(path)}"
    ) from array_split_error

  # The data sharding factor is only part of the data loading axes, when enable
  # e.g. TP, device per shard is no longer 1
  devices_per_shard = len(global_mesh.local_devices) // input_data_sharding_factor
  local_device_buffers = []
  for i, shard in enumerate(local_device_arrays):
    # Calculate which devices should get this shard
    shard_devices = global_mesh.local_devices[i * devices_per_shard : (i + 1) * devices_per_shard]

    # Put the same shard data on each device in this group
    for device in shard_devices:
      local_device_buffers.append(jax.device_put(shard, device))

  return jax.make_array_from_single_device_arrays(global_shape, input_data_shardings, local_device_buffers)


def get_next_batch_sharded(
    local_iterator: Iterator, global_mesh: Mesh, input_data_shardings: NamedSharding, input_data_sharding_factor: int
) -> jax.Array:
  """Splits the host loaded data equally over all devices."""

  SLEEP_TIME = 10
  MAX_DATA_LOAD_ATTEMPTS = 30

  data_load_attempts = 0
  loaded_data_success = False
  while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
    data_load_attempts += 1
    try:
      local_data = next(local_iterator)
      loaded_data_success = True
    except tf.errors.FailedPreconditionError:
      max_logging.log("Failed to get next data batch, retrying")
      time.sleep(SLEEP_TIME)

  # Try one last time, if this fails we will see the full stack trace.
  if not loaded_data_success:
    local_data = next(local_iterator)
  input_gdas = jtu.tree_map_with_path(
      partial(
          _form_global_array,
          global_mesh=global_mesh,
          input_data_shardings=input_data_shardings,
          input_data_sharding_factor=input_data_sharding_factor,
      ),
      local_data,
  )

  return input_gdas


class MultiHostDataLoadIterator:
  """fold get_next_batch_sharded into a iterator class"""

  def __init__(self, dataloader: Union[tf.data.Dataset, Iterable], global_mesh: Mesh, logical_axis_rules):
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    self.input_data_shardings = maxtext_utils.get_input_data_sharding(global_mesh, logical_axis_rules)
    self.input_data_sharding_factor = _calculate_input_data_sharding_prod(global_mesh, self.input_data_shardings)
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")

  def reset(self):
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or grain.DataLoader.")

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    return get_next_batch_sharded(
        self.local_iterator, self.global_mesh, self.input_data_shardings, self.input_data_sharding_factor
    )


@colocated_python.colocated_python
def _get_next(dummy_array):
  """get next batch from the iterator stored in the state of colocated python"""
  if "iterator" not in colocated_python.__dict__:
    raise ValueError("iterator not found in colocated_python.__dict__")
  if "global_shape" not in colocated_python.__dict__:
    raise ValueError("_global_shape not found in colocated_python.__dict__")
  local_data = next(colocated_python.__dict__["iterator"])
  global_shape = colocated_python.__dict__["global_shape"]
  for k, v in local_data.items():
    local_data[k] = jnp.asarray(v)

  def form_global_array_colocated_python(path, array, devices, global_shape, sharding):
    try:
      device_arrays = np.split(array, len(devices), axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f"Unable to put to devices shape {array.shape} with "
          f"local device count {len(devices)} "
          f"at {jtu.keystr(path)}"
      ) from array_split_error
    device_arrays = jax.device_put(device_arrays, devices)
    return jax.make_array_from_single_device_arrays(shape=global_shape, sharding=sharding, arrays=device_arrays)

  return jtu.tree_map_with_path(
      partial(
          form_global_array_colocated_python,
          devices=list(dummy_array.sharding.addressable_devices),
          global_shape=global_shape,
          sharding=dummy_array.sharding,
      ),
      local_data,
  )


def _colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  """Returns CPU devices colocated with the given devices."""
  return colocated_python.colocated_cpu_devices(devices)


def _get_cpu_mesh(mesh: Mesh):
  flat_devices = tuple(mesh.devices.flat)
  flat_cpu_devices = _colocated_cpu_devices(flat_devices)
  cpu_mesh = jax.sharding.Mesh(
      np.array(flat_cpu_devices).reshape(mesh.devices.shape), mesh.axis_names, axis_types=mesh.axis_types
  )
  return cpu_mesh


class RemoteIterator:
  "iterator class for using colocated python, iterator is initiated remotely and stored in the state of colocated python"

  def __init__(self, get_ds_fn, preprocessing_fn, global_mesh, global_shape):
    self.cpu_devices = _colocated_cpu_devices(jax.local_devices())
    self.tpu_devices = jax.local_devices()
    self.cpu_mesh = _get_cpu_mesh(global_mesh)
    self.tpu_sharding = jax.sharding.NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))
    self.cpu_sharding = jax.sharding.NamedSharding(self.cpu_mesh, PartitionSpec(self.cpu_mesh.axis_names))
    self.dummy_array = jnp.zeros((len(self.cpu_devices)))
    self.dummy_array = jax.device_put(self.dummy_array, self.cpu_sharding)

    @colocated_python.colocated_python
    def init(dummy_array):
      colocated_python.global_shape = global_shape
      ds = get_ds_fn(dataloading_host_index=jax.process_index(), dataloading_host_count=jax.process_count())
      dataloader = preprocessing_fn(dataset=ds)
      if isinstance(dataloader, tf.data.Dataset):
        colocated_python.iterator = dataloader.as_numpy_iterator()
      elif isinstance(dataloader, Iterable):
        colocated_python.iterator = iter(dataloader)
      else:
        raise ValueError("Type error: dataloader should be either tf.data.Dataset or grain.DataLoader.")
      return dummy_array

    out = jax.device_get(init(self.dummy_array))
    if out is not None:
      max_logging.log("RemoteIterator initiated.")

  def __iter__(self):
    return self

  def __next__(self):
    out = _get_next(self.dummy_array)

    def put_to_tpu_devices(path, array, sharding):
      try:
        jax.device_put(array, sharding)
      except Exception as e:  # pylint: disable=broad-exception-caught
        max_logging.log(f"Error putting data to TPU device path{path}, exception={e}")

    input_gdas = jtu.tree_map_with_path(partial(put_to_tpu_devices, sharding=self.tpu_sharding), out)

    return input_gdas
