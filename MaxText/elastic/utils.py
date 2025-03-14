# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for elastic training."""

import collections
from collections.abc import Callable, Mapping, Sequence
import itertools
import logging
import operator
import traceback
from typing import Any, TypeAlias

import jax
import numpy as np
from elastic import common_utils
from elastic import reshard

jax._src.array.ArrayImpl._check_if_deleted = lambda _: False  # pylint: disable=protected-access

PyTree: TypeAlias = Any

logger = logging.getLogger(__name__)


class ElasticUtils:
  """Utility class for elastic training."""
  _devices: Sequence[jax.Device]
  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  total_slice_count: int
  snapshot_period: int
  reshard_check_period: int
  max_failure_count: int | None
  max_reshard_retry_count: int | None
  failure_count: int
  reshard_retry_count: int
  good_slice_indices: set[int]
  _snapshots: collections.deque[Mapping[str, int | PyTree]]

  TEST_VALUE = 100

  def __init__(
      self,
      devices: Sequence[jax.Device],
      total_slice_count: int,
      snapshot_period: int = 1,
      snapshot_buffer_size: int = 1,
      reshard_check_period: int = 1,
      max_failure_count: int | None = None,
      max_reshard_retry_count: int | None = None,
  ) -> None:
    self.devices = devices
    self.total_slice_count = total_slice_count
    self.snapshot_period = snapshot_period
    self.reshard_check_period = reshard_check_period
    self.max_failure_count = max_failure_count
    self.max_reshard_retry_count = max_reshard_retry_count

    self.failure_count = 0
    self.reshard_retry_count = 0
    self.good_slice_indices = self.get_slice_availability()

    self._snapshots = collections.deque(maxlen=snapshot_buffer_size)

  @property
  def devices(self) -> Sequence[jax.Device]:
    """Returns the devices."""
    return self._devices

  @devices.setter
  def devices(self, devices: Sequence[jax.Device]) -> None:
    """Sets the devices."""
    self._devices = devices

    self.slice_to_devices = collections.defaultdict(list)
    for d in self._devices:
      self.slice_to_devices[d.slice_index].append(d)
    self.slice_to_devices = dict(self.slice_to_devices)

  @property
  def good_slice_to_devices(self) -> dict[int, Sequence[jax.Device]]:
    """Returns the good slice to devices map."""
    return {
        slice_index: self.slice_to_devices[slice_index]
        for slice_index in self.good_slice_indices
    }

  @property
  def good_devices(self) -> Sequence[jax.Device]:
    """Returns the good data slice indices."""
    return list(
        itertools.chain.from_iterable(self.good_slice_to_devices.values())
    )

  @property
  def default_device(self) -> jax.Device:
    """Returns the device that should be set to the default device."""
    try:
      return self.slice_to_devices[next(iter(self.good_slice_indices))][0]
    except StopIteration as error:
      raise ValueError("No good slices") from error

  @property
  def good_slice_count(self) -> int:
    """Returns the number of slices."""
    return len(self.good_slice_indices)

  def slice_device_count(self, slice_index: int) -> int:
    """Returns the number of devices in a slice."""
    try:
      return len(self.slice_to_devices[slice_index])
    except KeyError as error:
      raise ValueError(
          f"Slice {slice_index=} not found in {self.slice_to_devices=}"
      ) from error

  def scale_by_good_slices(self, x: int | float) -> int | float:
    """Scale x by the number of good slices."""
    if isinstance(x, int):
      ret, remainder = divmod(x * self.good_slice_count, self.total_slice_count)
      if remainder:
        raise ValueError(
            f"Cannot scale {x=} by good slices because it will result in a "
            f"remainder of {remainder=}."
        )
      return ret
    elif isinstance(x, float):
      return x * self.good_slice_count / self.total_slice_count
    else:
      raise ValueError(f"Unsupported type: {type(x)=}")

  @common_utils.timeit
  def initialize_snapshot(
      self,
      step: int,
      snapshot: Mapping[str, int | PyTree],
  ) -> None:
    """Initializes the snapshot.

    Args:
      step: The current step.
      snapshot: The snapshot to initialize.
    """
    self._snapshots.clear()
    self.maybe_snapshot(
        step=step,
        snapshot=snapshot,
        blocking=True,
        force=True,
    )
    self._extend_snapshots()

  def _extend_snapshots(self) -> None:
    fill_count = self._snapshots.maxlen - len(self._snapshots)
    try:
      self._snapshots.extend([self._snapshots[-1]] * fill_count)
    except IndexError as error:
      raise ValueError("No snapshots available") from error

  def pop_snapshot(self) -> tuple[int, Mapping[str, int | PyTree]]:
    """Pop next snapshot."""
    try:
      snapshot_dict = self._snapshots.popleft()
    except IndexError as error:
      raise ValueError("No snapshots available") from error

    step = snapshot_dict.pop("step")
    snapshot = snapshot_dict.pop("snapshot")

    return step, snapshot

  def _slice_down(self, reshard_retry: bool = False) -> None:
    """Slice down."""
    logger.debug("Slice down")
    self.good_slice_indices = self.get_slice_availability()
    self.failure_count += 1
    if reshard_retry:
      self.reshard_retry_count += 1
    else:
      self.reshard_retry_count = 0

    logger.debug(
        "self.failure_count=%s self.max_failure_count=%s",
        self.failure_count,
        self.max_failure_count,
    )
    if (
        self.max_failure_count is not None
        and self.failure_count >= self.max_failure_count
    ):
      raise RuntimeError(f"Max failure count reached {self.max_failure_count=}")

    logger.debug(
        "self.reshard_retry_count=%s self.max_reshard_retry_count=%s",
        self.reshard_retry_count,
        self.max_reshard_retry_count,
    )
    if (
        self.max_reshard_retry_count is not None
        and self.reshard_retry_count > self.max_reshard_retry_count
    ):
      raise RuntimeError(
          f"Max reshard retry count reached {self.max_reshard_retry_count=}"
      )

  @staticmethod
  def _is_error_due_to_slice_down(error: Exception) -> bool:
    """Check if the error is due to slice down."""
    if "DATA_LOSS" in str(error):
      logger.debug("Caught JaxRuntimeError DATA_LOSS exception")
    elif "NOT_FOUND" in str(error):
      logger.debug("Caught JaxRuntimeError NOT_FOUND exception")
    elif "INTERNAL" in str(error):
      logger.debug("Caught JaxRuntimeError INTERNAL exception")

    else:
      logger.debug("Unknown JaxRuntimeError")
      return False

    logger.debug("\n".join(traceback.format_exception(error)))
    return True

  def _is_ready_to_reshard(self, step: int) -> bool:
    """Indicates if it is time to reshard.

    May update `good_slice_indices`.

    Args:
      step: The current step.

    Returns:
      True if it is time to reshard, False otherwise.
    """
    if step % self.reshard_check_period:
      return False
    if self.good_slice_count >= self.total_slice_count:
      return False

    good_slice_indices = self.get_slice_availability()

    if not good_slice_indices & self.good_slice_indices:
      raise ValueError("All copies of the snapshot have been lost")

    if len(good_slice_indices) <= self.good_slice_count:
      return False

    logger.debug("New slice available.")
    logger.debug(
        "Previous good slice indices: self.good_slice_indices=%s",
        self.good_slice_indices,
    )
    logger.debug(
        "Current good slice indices: good_slice_indices=%s", good_slice_indices
    )

    self.good_slice_indices = good_slice_indices

    return True

  def _simple_execution(
      self, devices: Sequence[jax.Device], block: bool = False
  ) -> jax.Array:
    """Simple execution to test if a slice is available."""
    x = np.zeros(len(devices), dtype=float) + (self.TEST_VALUE - 1)
    y = jax.pmap(lambda x: x + 1, devices=devices)(x)
    if block:
      jax.block_until_ready(y)
    return y

  @common_utils.timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = set()

    results = {
        slice_index: self._simple_execution(devices)
        for slice_index, devices in self.slice_to_devices.items()
    }

    for slice_index, x in results.items():
      logger.debug("Checking slice_index=%s", slice_index)
      expected = (
          np.zeros(self.slice_device_count(slice_index), dtype=float)
          + self.TEST_VALUE
      )
      try:
        with common_utils.Timer(f"Checking {slice_index=}"):
          if np.allclose(x, expected):
            good_slice_indices.add(slice_index)
            logger.debug("slice_index=%s good", slice_index)
          else:
            logger.error(  # pylint: disable=logging-fstring-interpolation
                "Error with _simple_execution for slice_index=%s. "
                "This should not happen.",
                slice_index,
            )
      except jax.errors.JaxRuntimeError as error:
        if "DATA_LOSS" in str(error):
          logger.debug(  # pylint: disable=logging-fstring-interpolation
              "Caught JaxRuntimeError DATA_LOSS exception for slice_index=%s",
              slice_index,
          )
          logger.debug("error=%s", error)
        else:
          logger.exception(
              "Unknown JaxRuntimeError for slice_index=%s", slice_index
          )
        logger.debug("slice_index=%s bad", slice_index)

    logger.debug("good_slice_indices=%s", good_slice_indices)

    return good_slice_indices

  @staticmethod
  def _get_snapshot_size(snapshot: Mapping[str, int | PyTree]) -> int:
    """Returns the size of a snapshot."""
    total_nbytes = 0
    for v in snapshot.values():
      nbytes = jax.tree.map(lambda x: x.nbytes, v)
      total_nbytes += jax.tree.reduce(operator.add, nbytes)
    return total_nbytes

  @staticmethod
  def _put_snapshot_on_host(
      snapshot: Mapping[str, int | PyTree],
  ) -> Mapping[str, int | PyTree]:
    """Returns the size of a snapshot."""
    return {
        k: jax.device_put(
            v,
            jax.tree.map(
                lambda x: x.sharding.with_memory_kind(kind="pinned_host"), v
            ),
        )
        for k, v in snapshot.items()
    }

  @common_utils.timeit
  def maybe_snapshot(
      self,
      step: int,
      snapshot: Mapping[str, int | PyTree],
      blocking: bool = False,
      force: bool = False,
  ) -> None:
    """Save step and state."""
    if not force and step % self.snapshot_period:
      logger.debug("Not saving a snapshot")
      return

    total_nbytes = self._get_snapshot_size(snapshot)

    logger.debug("Saving a snapshot of %s bytes", total_nbytes)

    snapshot_host = self._put_snapshot_on_host(snapshot)

    logger.debug("Snapshot dispatched")
    if blocking:
      jax.block_until_ready(snapshot_host)
      logger.debug("Snapshot completed")

    self._snapshots.appendleft({"step": step, "snapshot": snapshot_host})

  @common_utils.timeit
  def get_resharded_snapshot(
      self, mesh: jax.sharding.Mesh
  ) -> tuple[int, Mapping[str, int | PyTree]]:
    """Get the resharded snapshot.

    Args:
      mesh: The mesh.

    Returns:
      The next snapshot.
    """
    while True:
      step, snapshot = self.pop_snapshot()

      resharded_snapshot_host = {}
      resharded_snapshot_device = {}
      for k, v in snapshot.items():
        resharded_host, resharded_device = self._reshard_snapshot(v, mesh)
        resharded_snapshot_host[k] = resharded_host
        resharded_snapshot_device[k] = resharded_device

      try:
        jax.block_until_ready(resharded_snapshot_host)
        jax.block_until_ready(resharded_snapshot_device)
        break
      except Exception as error:  # pylint: disable=broad-except
        if not self._is_error_due_to_slice_down(error):
          raise
        logger.debug("Retrying with the next snapshot")

    self.initialize_snapshot(
        step=step,
        snapshot=resharded_snapshot_host,
    )

    return step, resharded_snapshot_device

  @common_utils.timeit
  def maybe_reshard_down(
      self,
      error: Exception,
      elastic_handler: Callable[..., Any],
      handler_args: tuple[Any, ...] | None = None,
      reshard_retry: bool = False,
  ) -> Any:
    """Reshards down if the error is due to slice down."""
    if handler_args is None:
      handler_args = ()

    while True:
      if not self._is_error_due_to_slice_down(error):
        logger.debug("Not resharding down")
        raise error from error.__cause__

      logger.debug("Resharding down")
      self._slice_down(reshard_retry)

      try:
        ret = elastic_handler(*handler_args)
        break
      except jax.errors.JaxRuntimeError as e:
        logger.debug("Elastic handler raised an error.")
        error = e
        reshard_retry = True

    logger.debug("Successfully resharded down")
    return ret

  @common_utils.timeit
  def maybe_reshard_up(
      self,
      step: int,
      snapshot: Mapping[str, int | PyTree],
      elastic_handler: Callable[..., Any],
      handler_args: tuple[Any, ...] | None = None,
  ) -> Any:
    """Reshards up if it is time to reshard."""
    if handler_args is None:
      handler_args = ()

    if not self._is_ready_to_reshard(step):
      logger.debug("Not resharding up")
      return

    self.maybe_snapshot(
        step=step,
        snapshot=snapshot,
        force=True,
    )

    try:
      ret = elastic_handler(*handler_args)
    except jax.errors.JaxRuntimeError as error:
      logger.debug("Elastic handler failed. Trying again")
      ret = self.maybe_reshard_down(
          error=error,
          elastic_handler=elastic_handler,
          handler_args=handler_args,
          reshard_retry=True,
      )

    logger.debug("Finished resharding up")
    return ret

  @common_utils.timeit
  def _reshard_snapshot(
      self,
      x: PyTree,
      mesh: jax.sharding.Mesh,
  ) -> tuple[PyTree, PyTree]:
    """Reshards `x` to the specified `sharding`.

    Donates `x`
    Args:
        x: An array, scalar, or a nested Python container thereof.
        mesh: The mesh.

    Returns:
        A copy of `x` on host memory.
        A copy of `x` on device memory.
    """
    sharding_pinned_host = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(
            mesh, x.sharding.spec, memory_kind="pinned_host"
        ),
        x,
    )
    # Don't donate in case the snapshot is duplicated in _snapshots
    resharded_pinned_host = reshard.reshard(
        x,
        sharding_pinned_host,
    )

    sharding_device = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(
            mesh, x.sharding.spec, memory_kind="device"
        ),
        resharded_pinned_host,
    )

    resharded_device = reshard.reshard(
        resharded_pinned_host,
        sharding_device,
        put_array=self.put_array_device_put0,
        donate_input=False,
    )

    return resharded_pinned_host, resharded_device

  @staticmethod
  def put_array_device_put0(
      arr: jax.Array | PyTree,
      dst_sharding: jax.sharding.Sharding | PyTree,
      donate_input: bool,
  ) -> PyTree:
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")
    return jax.device_put(arr, dst_sharding, donate=donate_input)

  default_put_array = put_array_device_put0

  def put_array_device_put1(
      self,
      arr: jax.Array | PyTree,
      dst_sharding: jax.sharding.Sharding | PyTree,
      donate_input: bool,  # pylint: disable=unused-argument
  ) -> PyTree:
    """Reshards `arr` to the specified `dst_sharding`.

    Args:
        arr: An array, scalar, or a nested Python container thereof.
        dst_sharding: A `Sharding` or a nested `Sharding` in a Python container
          (must match the structure of `x`), specifying the target sharding.
        donate_input: If `True`, donates the input arrays to reduce memory
          needed for resharding. Donated buffers should not be reused.

    Returns:
        A copy of `x` with the specified `sharding`.
    """
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")

    if dst_sharding.num_devices <= arr.sharding.num_devices:
      # Reshard down
      arrays = [
          x.data
          for x in arr.addressable_shards
          if x.device.slice_index in self.good_slice_indices
      ]
    else:
      # Reshard up
      slice_to_arrays = collections.defaultdict(list)
      for x in arr.addressable_shards:
        slice_to_arrays[x.data.device.slice_index].append(x.data)
      slice_to_arrays = dict(slice_to_arrays)

      good_data_slice_indices = {d.slice_index for d in arr.sharding.device_set}
      new_slice_indices = self.good_slice_indices - good_data_slice_indices

      new_arrays = []

      good_slice_index = good_data_slice_indices.pop()
      arrays_to_put = slice_to_arrays[good_slice_index]

      for new_slice_index in new_slice_indices:
        devices = self.slice_to_devices[new_slice_index]
        shardings = [
            jax.sharding.SingleDeviceSharding(
                device,
                memory_kind=array.sharding.memory_kind,
            )
            for array, device in zip(arrays_to_put, devices)
        ]

        new_arrays += [
            jax.device_put(array, sharding)
            for array, sharding in zip(arrays_to_put, shardings)
        ]

      arrays = sum(slice_to_arrays.values(), []) + new_arrays

    return jax.make_array_from_single_device_arrays(
        arr.shape, dst_sharding, arrays
    )

  def put_array_device_put2(
      self,
      arr: jax.Array | PyTree,
      dst_sharding: jax.sharding.Sharding | PyTree,
      donate_input: bool,  # pylint: disable=unused-argument
  ) -> PyTree:
    """Reshards `arr` to the specified `dst_sharding`.

    Args:
        arr: An array, scalar, or a nested Python container thereof.
        dst_sharding: A `Sharding` or a nested `Sharding` in a Python container
          (must match the structure of `x`), specifying the target sharding.
        donate_input: If `True`, donates the input arrays to reduce memory
          needed for resharding. Donated buffers should not be reused.

    Returns:
        A copy of `x` with the specified `sharding`.
    """
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")

    if dst_sharding.num_devices <= arr.sharding.num_devices:
      # Reshard down
      arrays = [
          x.data
          for x in arr.addressable_shards
          if x.device.slice_index in self.good_slice_indices
      ]
    else:
      # Reshard up
      slice_to_arrays = collections.defaultdict(list)
      for x in arr.addressable_shards:
        slice_to_arrays[x.data.device.slice_index].append(x.data)
      slice_to_arrays = dict(slice_to_arrays)

      good_data_slice_indices = {d.slice_index for d in arr.sharding.device_set}
      new_slice_indices = self.good_slice_indices - good_data_slice_indices

      new_arrays = []

      good_slice_index = good_data_slice_indices.pop()
      arrays_to_put = slice_to_arrays[good_slice_index]

      for new_slice_index in new_slice_indices:
        devices = self.slice_to_devices[new_slice_index]
        shardings = [
            jax.sharding.SingleDeviceSharding(
                device,
                memory_kind=array.sharding.memory_kind,
            )
            for array, device in zip(arrays_to_put, devices)
        ]
        new_arrays += jax.device_put(arrays_to_put, shardings)

      arrays = sum(slice_to_arrays.values(), []) + new_arrays

    return jax.make_array_from_single_device_arrays(
        arr.shape, dst_sharding, arrays
    )

  # Not working yet
  def put_array_device_put3(
      self,
      arr: jax.Array | PyTree,
      dst_sharding: jax.sharding.Sharding | PyTree,
      donate_input: bool,  # pylint: disable=unused-argument
  ) -> PyTree:
    """Reshards `arr` to the specified `dst_sharding`.

    Args:
        arr: An array, scalar, or a nested Python container thereof.
        dst_sharding: A `Sharding` or a nested `Sharding` in a Python container
          (must match the structure of `x`), specifying the target sharding.
        donate_input: If `True`, donates the input arrays to reduce memory
          needed for resharding. Donated buffers should not be reused.

    Returns:
        A copy of `x` with the specified `sharding`.
    """
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")

    if dst_sharding.num_devices <= arr.sharding.num_devices:
      # Reshard down
      arrays = [
          x.data
          for x in arr.addressable_shards
          if x.device.slice_index in self.good_slice_indices
      ]
    else:
      # Reshard up
      slice_to_arrays = collections.defaultdict(list)
      for x in arr.addressable_shards:
        slice_to_arrays[x.data.device.slice_index].append(x.data)
      slice_to_arrays = dict(slice_to_arrays)

      good_data_slice_indices = {d.slice_index for d in arr.sharding.device_set}
      new_slice_indices = self.good_slice_indices - good_data_slice_indices

      new_arrays = []
      for i, slice_index in enumerate(good_data_slice_indices):
        slice_arrays = slice_to_arrays[slice_index]
        start_index = len(slice_arrays) * i // len(good_data_slice_indices)
        end_index = len(slice_arrays) * (i + 1) // len(good_data_slice_indices)

        arrays_to_put = slice_arrays[start_index:end_index]

        for new_slice_index in new_slice_indices:
          devices = self.slice_to_devices[new_slice_index][
              start_index:end_index
          ]
          shardings = [
              jax.sharding.SingleDeviceSharding(
                  device,
                  memory_kind=array.sharding.memory_kind,
              )
              for array, device in zip(arrays_to_put, devices)
          ]
          new_arrays += jax.device_put(arrays_to_put, shardings)

      arrays = sum(slice_to_arrays.values(), []) + new_arrays

    return jax.make_array_from_single_device_arrays(
        arr.shape, dst_sharding, arrays
    )

