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

from flax import linen as nn

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jax.experimental import colocated_python
import jax.numpy as jnp

from MaxText import max_logging
from MaxText import max_utils
import ml_collections


def _build_global_shape(local_shape: tuple[int, ...]) -> tuple[int, ...]:
  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape


def _calculate_input_data_sharding_prod(mesh: Mesh, input_data_shardings: NamedSharding) -> tuple[int, ...]:
  """Calculate the product of the mesh sizes for input data sharding."""
  try:
    spec = input_data_shardings.spec
  except AttributeError:
    raise ValueError(
        f"input_data_shardings does not have a .spec attribute (expected a PartitionSpec), got: {type(input_data_shardings)}"
    )

  batch_product = max_utils.compute_axis_product(spec[0], mesh.shape)
  sequence_product = max_utils.compute_axis_product(spec[1], mesh.shape) if len(spec) > 1 else 1

  return (batch_product, sequence_product)


def _form_global_array(
    path,
    array: np.ndarray,
    global_mesh: Mesh,
    input_data_shardings: NamedSharding,
    input_data_sharding_prods: tuple[int, ...],  # e.g., (4, 1)
) -> jax.Array:
  """Put local sharded array into devices using ND sharding, with replication on leftover devices."""

  global_shape = _build_global_shape(array.shape)

  if len(input_data_sharding_prods) != array.ndim:
    raise ValueError(
        f"Sharding factor {input_data_sharding_prods} must match array rank {array.ndim} "
        f"for array of shape {array.shape}"
    )

  # Split the array into subarrays based on the 2D (B, S) input data sharding
  def recursive_split(arr, factors, axis=0):
    if axis >= len(factors):
      return [arr]
    if factors[axis] == 1:
      return recursive_split(arr, factors, axis + 1)
    split_chunks = np.array_split(arr, factors[axis], axis=axis)
    return [s for chunk in split_chunks for s in recursive_split(chunk, factors, axis + 1)]

  # Shard the array into subarrays
  local_shards = recursive_split(array, input_data_sharding_prods)

  num_shards = np.prod(input_data_sharding_prods)
  num_devices = len(global_mesh.local_devices)

  if num_devices % num_shards != 0:
    raise ValueError(f"Cannot evenly replicate {num_shards} shards across {num_devices} devices. " f"at {jtu.keystr(path)}")

  replication_factor = num_devices // num_shards
  local_device_buffers = []

  for i, shard in enumerate(local_shards):
    shard_devices = global_mesh.local_devices[i * replication_factor : (i + 1) * replication_factor]
    for device in shard_devices:
      local_device_buffers.append(jax.device_put(shard, device))

  return jax.make_array_from_single_device_arrays(global_shape, input_data_shardings, local_device_buffers)


def get_next_batch_sharded(
    local_iterator: Iterator,
    global_mesh: Mesh,
    input_data_shardings: NamedSharding,
    input_data_sharding_prods: tuple[int, ...],
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
          input_data_sharding_prods=input_data_sharding_prods,
      ),
      local_data,
  )

  return input_gdas


class MultiHostDataLoadIterator:
  """fold get_next_batch_sharded into a iterator class"""

  def __init__(self, dataloader: Union[tf.data.Dataset, Iterable], global_mesh: Mesh, config: ml_collections.ConfigDict):
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    data_pspec = PartitionSpec(*config.input_data_sharding_logical_axes)
    self.input_data_shardings = nn.logical_to_mesh_sharding(data_pspec, global_mesh, config.logical_axis_rules)
    self.input_data_sharding_prods = _calculate_input_data_sharding_prod(global_mesh, self.input_data_shardings)
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
        self.local_iterator, self.global_mesh, self.input_data_shardings, self.input_data_sharding_prods
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
