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
from typing import Callable, Any, Union
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

from typing import Sequence, Optional

import max_logging
import max_utils
import ml_collections


def _get_mesh_axes_prod(config, default_num: int, prefix: str, mesh_axes: list[str]) -> int:
  """Get the mesh axes parallelism product from the config.
  E.g. for mesh_axes = ["data", "fsdp"], and *_data_parallelism = 2 and *_fsdp_parallelism = 3 it will return 6
  """
  mesh_prod = 1
  for mesh_axis in mesh_axes:
    parallelism_name = max_utils.construct_parallelism_name(mesh_axis, prefix)
    try:
      scale_val = getattr(config, parallelism_name)
    except AttributeError:
      raise ValueError(f"Config does not have {parallelism_name}, " f"but it is needed for logical axes {mesh_axes}")
    if scale_val == -1:
      scale_val = default_num
    assert scale_val > 0, f"Scale value for {parallelism_name} must be positive, got {scale_val}"
    mesh_prod *= scale_val
  return mesh_prod


def _get_input_data_parallelisms(
    config: ml_collections.ConfigDict, prefix: str, default_mesh_parallelism: int
) -> tuple[int, ...]:
  """Get the global input data scale from the config.
  prefix: either "dcn" or "ici"
  default_mesh_parallelism: fills the -1 in the parallelism config
  Returns a tuple of integers representing the parallelism for each logical axis.

  E.g. for input_data_sharding_logical_axes = ["batch", "length"]
  logical_axes_rules = [["batch", ["data"]], ["length", ["sequence"]]]
  ici_data_parallelism = 2
  ici_sequence_parallelism = 3, it will return (2, 3)
  """
  input_data_prod = []
  for i, logical_axis in enumerate(config.input_data_sharding_logical_axes):
    # Find matching rule from the list
    mesh_axes = []
    for rule in config.logical_axis_rules:
      if rule[0] == logical_axis:
        mesh_axes = [rule[1]] if isinstance(rule[1], str) else rule[1]
        break

    assert len(mesh_axes) > 0, f"No matching rule found for logical axis {logical_axis}"
    mesh_prod = _get_mesh_axes_prod(config, default_mesh_parallelism, prefix, mesh_axes)

    input_data_prod.append(mesh_prod)
  if len(input_data_prod) == 1:
    input_data_prod.append(1)
  return tuple(input_data_prod)


def _build_global_shape(local_shape: tuple[int, ...], input_data_dcn_parallelisms: tuple[int, ...]) -> tuple[int, ...]:
  """Builds the global shape from the local shape."""
  assert len(local_shape) == len(input_data_dcn_parallelisms), (
      f"Shape mismatch: local_shape has {len(local_shape)} dims, "
      f"but input_data_dcn_parallelisms has {len(input_data_dcn_parallelisms)} axes"
  )
  global_shape = []
  for local_dim, global_scale in zip(local_shape, input_data_dcn_parallelisms):
    global_shape.append(local_dim * global_scale)
  return tuple(global_shape)


def _form_global_array(
    path,
    array: np.ndarray,
    global_mesh: Mesh,
    input_data_shardings: NamedSharding,
    input_data_dcn_parallelisms: tuple[int, ...],
    input_data_ici_parallelisms: tuple[int, ...],
) -> jax.Array:
  """Put local sharded array into devices sharding, and return global array."""
  global_shape = _build_global_shape(array.shape, input_data_dcn_parallelisms)

  if len(global_shape) != array.ndim:
    raise ValueError(
        f"Sharding factor {global_shape} must match array rank {array.ndim} " f"for array of shape {array.shape}"
    )

  # Split the array into subarrays based on the 2D (B, S) ici data sharding
  def recursive_split(arr, factors, axis=0):
    if axis >= len(factors):
      return [arr]
    if factors[axis] == 1:
      return recursive_split(arr, factors, axis + 1)
    split_chunks = np.array_split(arr, factors[axis], axis=axis)
    return [s for chunk in split_chunks for s in recursive_split(chunk, factors, axis + 1)]

  # Shard the array into subarrays
  local_shards = recursive_split(array, input_data_ici_parallelisms)

  num_shards = np.prod(input_data_ici_parallelisms)
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
    input_data_dcn_parallelisms: tuple[int, ...],
    input_data_ici_parallelisms: tuple[int, ...],
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
          input_data_dcn_parallelisms=input_data_dcn_parallelisms,
          input_data_ici_parallelisms=input_data_ici_parallelisms,
      ),
      local_data,
  )

  return input_gdas


class MultiHostDataLoadIterator:
  """fold get_next_batch_sharded into a iterator class"""

  def __init__(self, dataloader: Union[tf.data.Dataset, Iterable], global_mesh: Mesh, config: ml_collections.ConfigDict):
    self.global_mesh = global_mesh
    self.dataloader = dataloader

    num_devices = jax.device_count()
    num_slices = 1 if config.inference_benchmark_test else config.num_slices
    num_devices_per_slice = int(num_devices // num_slices)

    default_dcn_parallelism = (
        max_utils.get_unspecified_mesh_axes_value(config.dcn_parallelism, num_slices, "DCN")
        if config.dcn_parallelism.count(-1) == 1
        else 0
    )
    default_ici_parallelism = (
        max_utils.get_unspecified_mesh_axes_value(config.ici_parallelism, num_devices_per_slice, "ICI")
        if config.ici_parallelism.count(-1) == 1
        else 0
    )

    self.input_data_dcn_parallelisms = _get_input_data_parallelisms(
        config, prefix="dcn", default_mesh_parallelism=default_dcn_parallelism
    )
    self.input_data_ici_parallelisms = _get_input_data_parallelisms(
        config, prefix="ici", default_mesh_parallelism=default_ici_parallelism
    )

    max_logging.log(
        f"Input data dcn parallelisms are {self.input_data_dcn_parallelisms} ici parallelisms are {self.input_data_ici_parallelisms}"
    )
    data_pspec = PartitionSpec(*config.input_data_sharding_logical_axes)
    self.input_data_shardings = nn.logical_to_mesh_sharding(data_pspec, global_mesh, config.logical_axis_rules)

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
        self.local_iterator,
        self.global_mesh,
        self.input_data_shardings,
        self.input_data_dcn_parallelisms,
        self.input_data_ici_parallelisms,
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
