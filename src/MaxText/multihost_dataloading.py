# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=unused-import
"""SPMD Multihost Dataloading Utilities.

Adapted from Sholto's:
https://github.com/sholtodouglas/multihost_dataloading
"""
import itertools
from functools import partial
from typing import Union, Sequence, Optional
from collections.abc import Iterator, Iterable
import time

import tensorflow as tf  # pylint: disable=g-import-not-at-top

import numpy as np

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jax.experimental import colocated_python
import jax.numpy as jnp

from MaxText import max_logging


def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
  sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

  return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """Put local sharded array into local devices"""
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


class MultiHostDataLoadIterator:
  """fold get_next_batch_sharded into a iterator class"""

  def __init__(self, dataloader: tf.data.Dataset | Iterable, global_mesh: Mesh, generate_padding_batch: bool = False, microbatch_size_to_run: Optional[int]=None, input_data_sharding=None):
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    self.generate_padding_batch = generate_padding_batch
    self.microbatch_size_to_run = microbatch_size_to_run
    self.input_data_sharding = input_data_sharding
    self.reset()

  def reset(self):
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")
    self.out_of_data = False
    self.last_local_data = None

    sharded_iter = self._base_iter()
    if self.microbatch_size_to_run:
       self.local_iterator = itertools.chain.from_iterable(
         self.explode_to_micro(b) for b in sharded_iter
       )
    else:
       self.local_iterator = sharded_iter

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    return next(self.local_iterator)

  def _base_iter(self):
    while True:
      yield self._get_next_batch_sharded()

  def explode_to_micro(self, batch):
    """Splits larger batch into smaller equally sized batches"""
    mb = self.microbatch_size_to_run
    # `batch` is a dict-like PyTree of jax.Arrays
    k0 = next(iter(batch))
    B = batch[k0].shape[0]
    assert B % mb == 0, f"global batch {B} not divisible by microbatch {mb}"
    M = B // mb
    reshaped = {k: v.reshape((M, mb) + v.shape[1:]) for k, v in batch.items()}
    for i in range(M):
      microbatch = {k: reshaped[k][i] for k in reshaped}
      if self.input_data_sharding is not None:
        microbatch = jax.lax.with_sharding_constraint(microbatch, self.input_data_sharding)
      yield microbatch

  def _get_next_batch_sharded(self) -> jax.Array:
    """Splits the host loaded data equally over all devices."""
    if self.out_of_data and self.generate_padding_batch:
      local_data = self._make_padding_batch()

    else:
      SLEEP_TIME = 10
      MAX_DATA_LOAD_ATTEMPTS = 30

      for _ in range(MAX_DATA_LOAD_ATTEMPTS):
        try:
          local_data = next(self.local_iterator)
          break  # exit the loop on success
        except tf.errors.FailedPreconditionError as e:
          max_logging.log(f"Failed to get next data batch due to {e}, retrying")
          time.sleep(SLEEP_TIME)
        except Exception as e:
          if isinstance(e, StopIteration) or "StopIteration" in str(e):
            if self.generate_padding_batch:
              max_logging.log(
                  f"MultiHostDataLoadIterator: host {jax.process_index()} failed to load data with {type(e)} error: ({e}). It may reach the end of data, generating padding batch as generate_padding_batch=True."
              )
              self.out_of_data = True
              local_data = self._make_padding_batch()
              break
          else:
            raise e
      else:
        raise TimeoutError(
            f"Failed to load data after {MAX_DATA_LOAD_ATTEMPTS} retry attempts."
        )

      self.last_local_data = local_data
    input_gdas = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=self.global_mesh), local_data)

    return input_gdas

  def _make_padding_batch(self):
    if self.last_local_data is None:
      raise ValueError("last_local_data is None, cannot make padding batch.")
    return jtu.tree_map(lambda x: jnp.full_like(x, 0), self.last_local_data)


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


def _colocated_cpu_mesh(mesh: Mesh) -> Mesh:
  """Returns a CPU mesh that has colocated CPU devices."""
  return colocated_python.colocated_cpu_devices(mesh)


class RemoteIterator:
  "iterator class for using colocated python, iterator is initiated remotely and stored in the state of colocated python"

  def __init__(self, get_ds_fn, preprocessing_fn, global_mesh, global_shape):
    self.cpu_devices = _colocated_cpu_devices(jax.local_devices())
    self.tpu_devices = jax.local_devices()
    self.cpu_mesh = _colocated_cpu_mesh(global_mesh)
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
      max_logging.log(f"RemoteIterator initiated. Test output: {out}")

  def __iter__(self):
    return self

  def __next__(self):
    out = _get_next(self.dummy_array)

    def put_to_tpu_devices(path, array, sharding):
      try:
        return jax.device_put(array, sharding)
      except Exception as e:  # pylint: disable=broad-exception-caught
        max_logging.log(f"Error putting data to TPU device path{path}, exception={e}")
        raise

    input_gdas = jtu.tree_map_with_path(partial(put_to_tpu_devices, sharding=self.tpu_sharding), out)

    return input_gdas
