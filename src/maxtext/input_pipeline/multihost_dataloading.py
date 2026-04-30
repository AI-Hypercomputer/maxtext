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
from functools import partial
from typing import Union, Sequence
from collections.abc import Iterator, Iterable
import time
import json

from etils import epath
import tensorflow as tf  # pylint: disable=g-import-not-at-top

import numpy as np

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jax.experimental import colocated_python
import jax.numpy as jnp

from maxtext.utils import max_logging


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
  """fold get_next_batch_sharded into a iterator class.
  expansion_factor_for_grain is only used for grain pipeline when having a subset of hosts loading real data.
  """

  def __init__(
      self,
      dataloader: tf.data.Dataset | Iterable,
      global_mesh: Mesh,
      generate_padding_batch: bool = False,
      expansion_loading_factor_for_grain: int = -1,
  ):
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")
    self.out_of_data = False
    self.last_local_data = None
    self.generate_padding_batch = generate_padding_batch
    self.expansion_loading_factor_for_grain = expansion_loading_factor_for_grain

  def reset(self):
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError("Type error: dataloader should be either tf.data.Dataset or Iterable.")
    self.out_of_data = False
    self.last_local_data = None

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    return self._get_next_batch_sharded()

  def _get_next_batch_sharded(self) -> jax.Array:
    """Splits the host loaded data equally over all devices."""
    if self.out_of_data and self.generate_padding_batch:
      local_data = self._make_padding_batch()

    else:
      SLEEP_TIME = 10
      MAX_DATA_LOAD_ATTEMPTS = 30

      local_data = None
      for _ in range(MAX_DATA_LOAD_ATTEMPTS):
        try:
          local_data = next(self.local_iterator)
          if self.expansion_loading_factor_for_grain > 1:
            # Since grain checkpoint requires fixed batch_size, we run the dataIterator for
            # expansion_loading_factor_for_grain times to get the
            # right batch_size for the host that is loading real data.
            local_data_list = [local_data]
            for _ in range(1, int(self.expansion_loading_factor_for_grain)):
              next_batch = next(self.local_iterator)
              local_data_list.append(next_batch)
            local_data = jtu.tree_map(lambda *xs: np.concatenate(xs, axis=0), *local_data_list)
          break  # exit the loop on success
        except tf.errors.FailedPreconditionError as e:
          max_logging.log(f"Failed to get next data batch due to {e}, retrying")
          time.sleep(SLEEP_TIME)
        except StopIteration as e:
          if self.generate_padding_batch:
            max_logging.log(
                f"MultiHostDataLoadIterator: host {jax.process_index()} failed to load data with {type(e)} error: ({e}). "
                "It may have reached the end of the data. Generating a padding batch as generate_padding_batch=True."
            )
            self.out_of_data = True
            local_data = self._make_padding_batch()
            break
          else:
            raise e
      else:
        raise TimeoutError(f"Failed to load data after {MAX_DATA_LOAD_ATTEMPTS} retry attempts.")

      self.last_local_data = local_data
    input_gdas = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=self.global_mesh), local_data)

    return input_gdas

  def _make_padding_batch(self):
    if self.last_local_data is None:
      raise ValueError("last_local_data is None, cannot make padding batch.")
    return jtu.tree_map(lambda x: jnp.full_like(x, 0), self.last_local_data)


def _colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  """Returns CPU devices colocated with the given devices."""
  return colocated_python.colocated_cpu_devices(devices)


def _colocated_cpu_mesh(mesh: Mesh) -> Mesh:
  """Returns a CPU mesh that has colocated CPU devices."""
  return colocated_python.colocated_cpu_devices(mesh)


@colocated_python.colocated_python_class
class RemoteIterator:
  "iterator class for using colocated python class"

  def __init__(self, get_ds_fn, preprocessing_fn, global_shape, checkpoint_path, elastic=False):
    self.get_ds_fn = get_ds_fn
    self.preprocessing_fn = preprocessing_fn
    self.global_shape = global_shape
    self.checkpoint_path = checkpoint_path
    self.elastic = elastic
    self.reset()
    max_logging.log("RemoteIterator initiated")

  def reset(self):
    ds = self.get_ds_fn(dataloading_host_index=jax.process_index(), dataloading_host_count=jax.process_count())
    dataloader = self.preprocessing_fn(dataset=ds)
    if isinstance(dataloader, tf.data.Dataset):
      self.iterator = dataloader.as_numpy_iterator()
    elif isinstance(dataloader, Iterable):
      self.iterator = iter(dataloader)
    else:
      raise ValueError("Type error: dataloader should be Iterable.")

  def get_next(self, dummy_array):
    """Gets the next batch of data and forms a global array."""
    local_data = next(self.iterator)

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
            global_shape=self.global_shape,
            sharding=dummy_array.sharding,
        ),
        local_data,
    )

  def save_state(self, step_array):
    """Saves the iterator state to a file."""
    step = step_array.addressable_data(0).item()
    directory = epath.Path(self.checkpoint_path) / str(step) / "iter"
    if self.elastic:
      # ElasticIterator state is a single global scalar shared by all shards,
      # so we write one fixed file (from process 0 only) and every process
      # reads the same file on restore — this survives elastic resizes that
      # change `jax.process_count()`.
      if jax.process_index() == 0:
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "process_0.json"
        filename.write_text(json.dumps(self.iterator.get_state(), indent=4))
      return step_array
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    state = json.dumps(self.iterator.get_state(), indent=4)
    filename.write_text(state)
    return step_array

  def restore_state(self, step_array):
    step = step_array.addressable_data(0).item()
    directory = epath.Path(self.checkpoint_path) / str(step) / "iter"
    if self.elastic:
      filename = directory / "process_0.json"
    else:
      filename = directory / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    state = json.loads(filename.read_text())
    self.iterator.set_state(state)
    return step_array


class RemoteIteratorWrapper:
  """Wrapper for RemoteIterator that handles device placement."""

  def __init__(self, get_ds_fn, preprocessing_fn, global_mesh, global_shape, checkpoint_path="", elastic=False):
    self.cpu_devices = _colocated_cpu_devices(jax.local_devices())
    self.tpu_devices = jax.local_devices()
    self.cpu_mesh = _colocated_cpu_mesh(global_mesh)
    self.tpu_sharding = jax.sharding.NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))
    self.cpu_sharding = jax.sharding.NamedSharding(self.cpu_mesh, PartitionSpec(self.cpu_mesh.axis_names))
    self.dummy_array = jnp.zeros((len(self.cpu_devices)))
    self.dummy_array = jax.device_put(self.dummy_array, self.cpu_sharding)
    # This is a proxy to a RemoteIterator running in a colocated process,
    # named "local_iterator" to match MultiHostDataLoadIterator's interface.
    self.local_iterator = RemoteIterator(get_ds_fn, preprocessing_fn, global_shape, checkpoint_path, elastic=elastic)
    max_logging.log("RemoteIteratorWrapper initiated")

  def __iter__(self):
    return self

  def reset(self):
    self.local_iterator.reset()

  def __next__(self):
    out = self.local_iterator.get_next(self.dummy_array)
    # use tree_map is out is a dict
    return jax.device_put(out, self.tpu_sharding)

  def save_state(self, step):
    step_array = jnp.full(self.dummy_array.shape, step, dtype=jnp.int32)
    step_array = jax.device_put(step_array, self.cpu_sharding)
    self.local_iterator.save_state(step_array)

  def restore_state(self, step):
    step_array = jnp.full(self.dummy_array.shape, step, dtype=jnp.int32)
    step_array = jax.device_put(step_array, self.cpu_sharding)
    self.local_iterator.restore_state(step_array)
