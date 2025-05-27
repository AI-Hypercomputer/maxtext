"""Copyright 2023 Google LLC

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
from functools import partial
from typing import Union, Sequence
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
  global_shape, sharding = _build_global_shape_and_sharding(
      np.shape(array), global_mesh
  )

  try:
    local_device_arrays = np.split(
        array, len(global_mesh.local_devices), axis=0
    )
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
        f"at {jtu.keystr(path)}"
    ) from array_split_error

  local_device_buffers = jax.device_put(
      local_device_arrays, global_mesh.local_devices
  )
  return jax.make_array_from_single_device_arrays(
      global_shape, sharding, local_device_buffers
  )


def get_next_batch_sharded(
    local_iterator: Iterator, global_mesh: Mesh
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
      partial(_form_global_array, global_mesh=global_mesh), local_data
  )

  return input_gdas


class MultiHostDataLoadIterator:
  """fold get_next_batch_sharded into a iterator class"""

  def __init__(
      self, dataloader: Union[tf.data.Dataset, Iterable], global_mesh: Mesh
  ):
    self.global_mesh = global_mesh
    self.dataloader = dataloader
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError(
          "Type error: dataloader should be either tf.data.Dataset or Iterable."
      )

  def reset(self):
    if isinstance(self.dataloader, tf.data.Dataset):
      self.local_iterator = self.dataloader.as_numpy_iterator()
    elif isinstance(self.dataloader, Iterable):
      self.local_iterator = iter(self.dataloader)
    else:
      raise ValueError(
          "Type error: dataloader should be either tf.data.Dataset or"
          " grain.DataLoader."
      )

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    return get_next_batch_sharded(self.local_iterator, self.global_mesh)


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

  def form_global_array_colocated_python(
      path, array, devices, global_shape, sharding
  ):
    try:
      device_arrays = np.split(array, len(devices), axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f"Unable to put to devices shape {array.shape} with "
          f"local device count {len(devices)} "
          f"at {jtu.keystr(path)}"
      ) from array_split_error
    device_arrays = jax.device_put(device_arrays, devices)
    return jax.make_array_from_single_device_arrays(
        shape=global_shape, sharding=sharding, arrays=device_arrays
    )

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
  print(f"*** [ksadi] mesh: {mesh}")
  flat_devices = tuple(mesh.devices.flat)
  flat_cpu_devices = _colocated_cpu_devices(flat_devices)
  cpu_mesh = jax.sharding.Mesh(
      np.array(flat_cpu_devices).reshape(mesh.devices.shape),
      mesh.axis_names,
      axis_types=mesh.axis_types,
  )
  return cpu_mesh


class RemoteIterator:
  "iterator class for using colocated python, iterator is initiated remotely and stored in the state of colocated python"

  def __init__(self, get_ds_fn, preprocessing_fn, global_mesh, global_shape):
    self.cpu_devices = _colocated_cpu_devices(jax.local_devices())
    self.tpu_devices = jax.local_devices()
    self.cpu_mesh = _get_cpu_mesh(global_mesh)
    self.tpu_sharding = jax.sharding.NamedSharding(
        global_mesh, PartitionSpec(global_mesh.axis_names)
    )
    self.cpu_sharding = jax.sharding.NamedSharding(
        self.cpu_mesh, PartitionSpec(self.cpu_mesh.axis_names)
    )
    self.dummy_array = jnp.zeros((len(self.cpu_devices)))
    self.dummy_array = jax.device_put(self.dummy_array, self.cpu_sharding)

    @colocated_python.colocated_python
    def init(dummy_array):
      colocated_python.global_shape = global_shape
      ds = get_ds_fn(
          dataloading_host_index=jax.process_index(),
          dataloading_host_count=jax.process_count(),
      )
      dataloader = preprocessing_fn(dataset=ds)
      if isinstance(dataloader, tf.data.Dataset):
        colocated_python.iterator = dataloader.as_numpy_iterator()
      elif isinstance(dataloader, Iterable):
        colocated_python.iterator = iter(dataloader)
      else:
        raise ValueError(
            "Type error: dataloader should be either tf.data.Dataset or"
            " grain.DataLoader."
        )
      return dummy_array

    print(f"*** patching jax.jaxlib path")
    ##### PATCH START #####
    # patch sys.modules so that jax.jaxlib points to jaxlib
    import sys
    import importlib  # Not strictly needed for this sys.modules manipulation but good practice for imports

    # --- Configuration ---
    # The Fully Qualified Name (FQN) that the JAX package will expect for its jaxlib components.
    TARGET_JAX_NAMESPACE_PREFIX = "jax.jaxlib"

    # --- Main Patch Logic ---
    print(
        "PATCH SCRIPT: Starting attempt to make 'jax.jaxlib' use the standalone"
        " 'jaxlib'."
    )
    try:
      # Step 1: Import the standalone 'jaxlib'. This is your source of truth.
      # This action loads the module and initializes its C++ components (1st Abseil init).
      print(f"PATCH INFO: Importing standalone 'jaxlib'...")
      # Ensure 'jaxlib' is truly imported at the top level.
      # The module object will be stored in sys.modules['jaxlib'].
      import jaxlib

      # Get a reference to the imported standalone jaxlib module object.
      standalone_jaxlib_module = sys.modules["jaxlib"]
      standalone_jaxlib_base_name = (
          standalone_jaxlib_module.__name__
      )  # Should be 'jaxlib'

      print(
          f"PATCH INFO: Standalone '{standalone_jaxlib_base_name}' imported"
          " successfully."
      )
      print(f"  - Module object: {standalone_jaxlib_module}")
      print(
          "  - Loaded from:"
          f" {getattr(standalone_jaxlib_module, '__file__', 'N/A')}"
      )

      # Step 2: Alias the main standalone 'jaxlib' module object to 'TARGET_JAX_NAMESPACE_PREFIX'
      # (e.g., make sys.modules['jax.jaxlib'] point to the standalone_jaxlib_module).
      if TARGET_JAX_NAMESPACE_PREFIX not in sys.modules:
        sys.modules[TARGET_JAX_NAMESPACE_PREFIX] = standalone_jaxlib_module
        print(
            f"PATCH INFO: Aliased '{standalone_jaxlib_base_name}' to"
            f" '{TARGET_JAX_NAMESPACE_PREFIX}'."
        )
      elif (
          sys.modules[TARGET_JAX_NAMESPACE_PREFIX]
          is not standalone_jaxlib_module
      ):
        print(
            "PATCH WARNING: Overwriting existing"
            f" '{TARGET_JAX_NAMESPACE_PREFIX}' in sys.modules with standalone"
            f" '{standalone_jaxlib_base_name}'."
        )
        sys.modules[TARGET_JAX_NAMESPACE_PREFIX] = standalone_jaxlib_module
      else:
        print(
            f"PATCH INFO: '{TARGET_JAX_NAMESPACE_PREFIX}' already correctly"
            f" points to standalone '{standalone_jaxlib_base_name}'."
        )

      # Step 3: Scan sys.modules for all currently loaded submodules of the standalone 'jaxlib'
      # (e.g., 'jaxlib.xla_client') and create corresponding aliases under the
      # 'TARGET_JAX_NAMESPACE_PREFIX' (e.g., 'jax.jaxlib.xla_client').

      # Take a snapshot of keys as sys.modules might be read by other threads,
      # though direct modification within this loop is avoided by collecting first or careful updates.
      original_module_keys_snapshot = list(sys.modules.keys())

      newly_aliased_sub_count = 0
      overwritten_sub_count = 0
      already_correct_sub_count = 0

      print(
          "PATCH INFO: Scanning for submodules of"
          f" '{standalone_jaxlib_base_name}' to alias under"
          f" '{TARGET_JAX_NAMESPACE_PREFIX}'."
      )
      for module_name_key in original_module_keys_snapshot:
        # We are interested in modules that are submodules of the standalone 'jaxlib'.
        # Example: 'jaxlib.xla_client' is a submodule of 'jaxlib'.
        if (
            module_name_key == standalone_jaxlib_base_name
        ):  # Skip the base module itself
          continue

        if module_name_key.startswith(standalone_jaxlib_base_name + "."):
          # This is a submodule.
          # suffix will be like '.xla_client' or '._src.pytree'
          suffix = module_name_key[len(standalone_jaxlib_base_name) :]

          # Construct the new FQN under the target JAX namespace.
          # e.g., 'jax.jaxlib' + '.xla_client' -> 'jax.jaxlib.xla_client'
          aliased_target_fqn = TARGET_JAX_NAMESPACE_PREFIX + suffix

          original_submodule_obj = sys.modules[module_name_key]

          if aliased_target_fqn not in sys.modules:
            sys.modules[aliased_target_fqn] = original_submodule_obj
            newly_aliased_sub_count += 1
          elif sys.modules[aliased_target_fqn] is not original_submodule_obj:
            # print(f"PATCH DEBUG: Overwriting submodule alias '{aliased_target_fqn}'.")
            sys.modules[aliased_target_fqn] = original_submodule_obj
            overwritten_sub_count += 1
          else:
            already_correct_sub_count += 1

      print(
          "PATCH INFO: Submodule aliasing for"
          f" '{TARGET_JAX_NAMESPACE_PREFIX}.*' complete:"
      )
      print(f"  - Newly aliased: {newly_aliased_sub_count}")
      print(
          "  - Overwritten (potentially corrected an old alias):"
          f" {overwritten_sub_count}"
      )
      print(
          f"  - Already correct (no action needed): {already_correct_sub_count}"
      )

    except ImportError as e:
      # Check if the very first 'import jaxlib' failed
      if "standalone_jaxlib_module" not in locals() and (
          "jaxlib" in str(e) or "No module named" in str(e)
      ):
        print(
            "PATCH CRITICAL ERROR: Failed to import the standalone 'jaxlib'. "
            f"This module is the prerequisite for the patch. Error: {e}"
        )
      else:
        print(
            "PATCH ERROR: An ImportError occurred. This might be during"
            f" 'import jax' or a related JAX component. Error: {e}"
        )
      raise
    except Exception as e:
      print(
          "PATCH UNEXPECTED ERROR: An error occurred during the patching"
          f" process: {e}"
      )
      raise

    ##### PATCH END #####

    out = jax.device_get(init(self.dummy_array))
    if out is not None:
      max_logging.log(f"RemoteIterator initiated. out: {out}")
    else:
      max_logging.log(f"RemoteIterator failed to initialize. out: {out}")
      raise ValueError(f"RemoteIterator failed to initialize. out: {out}")

  def __iter__(self):
    return self

  def __next__(self):
    out = _get_next(self.dummy_array)
    # max_logging.log(f"RemoteIterator next. out: {out}")

    def put_to_tpu_devices(path, array, sharding):
      try:
        return jax.device_put(array, sharding)
      except Exception as e:  # pylint: disable=broad-exception-caught
        max_logging.log(
            f"Error putting data to TPU device path{path}, exception={e}"
        )

    input_gdas = jtu.tree_map_with_path(
        partial(put_to_tpu_devices, sharding=self.tpu_sharding), out
    )
    max_logging.log(f"RemoteIterator next. input_gdas: {input_gdas}")

    return input_gdas
