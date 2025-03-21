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
"""Elasticity manager.

This class is responsible for managing the elastic training.

It is responsible for:
- Tracking the availability of slices.
- Tracking the number of failures and reshard retries.
- Tracking the snapshots.
- Resharding the snapshots.
- Resharding down if the error is due to slice down.
- Resharding up if it is time to reshard.
- Resharding the snapshot.
"""

import collections
from collections.abc import Callable, Mapping, Sequence
import itertools
import logging
import operator
import traceback
from typing import Any, TypeAlias

import jax
import numpy as np
from elastic.debug import timing
from elastic import reshard

# TODO: b/393445969 - Remove this when the bug is fixed.
jax._src.array.ArrayImpl._check_if_deleted = lambda _: False  # pylint: disable=protected-access

PyTree: TypeAlias = Any

_logger = logging.getLogger(__name__)


class Manager:
  """Utility class for elastic training."""
  _devices: Sequence[jax.Device]
  _total_slice_count: int | None = None
  slice_to_devices: Mapping[int, Sequence[jax.Device]]
  snapshot_period: int
  reshard_check_period: int
  max_failure_count: int | None
  max_reshard_retry_count: int | None
  failure_count: int
  reshard_retry_count: int
  good_slice_indices: set[int]
  _snapshot_0: Mapping[str, int | jax.Array | PyTree]
  _snapshot_1: Mapping[str, int | jax.Array | PyTree]
  _current_snapshot: int | None

  _SIMPLE_EXECUTION_TEST_VALUE = 100

  def __init__(
      self,
      devices: Sequence[jax.Device] | None = None,
      reshard_check_period: int = 1,
      snapshot_period: int = 1,
      snapshot_buffer_size: int = 1,
      max_failure_count: int | None = None,
      max_reshard_retry_count: int | None = None,
  ) -> None:
    if devices is None:
      devices = jax.devices()
    self.devices = devices
    self.reshard_check_period = reshard_check_period
    self.snapshot_period = snapshot_period
    self.max_failure_count = max_failure_count
    self.max_reshard_retry_count = max_reshard_retry_count

    self.failure_count = 0
    self.reshard_retry_count = 0

    self.good_slice_indices = self.get_slice_availability()
    self._snapshot_0 = None
    self._snapshot_1 = None
    self._current_snapshot = None

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
  def total_slice_count(self) -> int:
    """Returns the total number of slices."""
    if self._total_slice_count is None:
      self._total_slice_count = len(self.slice_to_devices)
    return self._total_slice_count

  def slice_device_count(self, slice_index: int) -> int:
    """Returns the number of devices in a slice."""
    try:
      return len(self.slice_to_devices[slice_index])
    except KeyError as error:
      raise ValueError(
          f"Slice {slice_index=} not found in {self.slice_to_devices=}"
      ) from error

  @staticmethod
  def _is_error_due_to_slice_down(error: Exception) -> bool:
    """Check if the error is due to slice down."""
    if "DATA_LOSS" in str(error):
      _logger.debug("Caught JaxRuntimeError DATA_LOSS error")
    elif "NOT_FOUND" in str(error):
      _logger.debug("Caught JaxRuntimeError NOT_FOUND error")
    elif "INTERNAL" in str(error):
      _logger.debug("Caught JaxRuntimeError INTERNAL error")

    else:
      _logger.debug("Caught unknown JaxRuntimeError")
      for line in traceback.format_exception(error):
        _logger.debug(line)
      return False

    for line in traceback.format_exception(error):
      _logger.debug(line)
    return True

  @classmethod
  def _simple_execution(cls, devices: Sequence[jax.Device]) -> jax.Array:
    """Simple execution to test if a slice is available.

    This function is used to test if a slice is available. It executes a simple
    computation on the devices and returns the result. If any of the devices are
    not available, the returned array will fail with a JaxRuntimeError used.

    Simply executing this function is not enough to determine if the slice is
    available. We also need to check the value of the returned array.

    Args:
      devices: The devices to execute on.

    Returns:
      The result of the execution.
    """
    if not devices:
      raise ValueError("No devices")

    test_input = np.zeros(len(devices), dtype=float) + (
        cls._SIMPLE_EXECUTION_TEST_VALUE - 1
    )

    return jax.pmap(lambda x: x + 1, devices=devices)(test_input)

  @timing.timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = set()

    results = {
        slice_index: self._simple_execution(devices)
        for slice_index, devices in self.slice_to_devices.items()
    }

    for slice_index, x in results.items():
      _logger.debug("Checking slice_index=%s", slice_index)
      expected = (
          np.zeros(self.slice_device_count(slice_index), dtype=float)
          + self._SIMPLE_EXECUTION_TEST_VALUE
      )
      try:
        with timing.Timer(f"Checking {slice_index=}"):
          jax.block_until_ready(x)
          if np.allclose(x, expected):
            good_slice_indices.add(slice_index)
            _logger.debug("slice_index=%s good", slice_index)
          else:
            _logger.error(
                "Error with _simple_execution for slice_index=%s. "
                "This should never happen. Expected: %s, Actual: %s",
                slice_index,
                expected,
                x,
            )
            raise ValueError(
                f"Error with _simple_execution for slice_index={slice_index}."
            )
      except jax.errors.JaxRuntimeError as error:
        if not self._is_error_due_to_slice_down(error):
          raise
        _logger.debug("slice_index=%s bad", slice_index)

    _logger.debug("good_slice_indices=%s", good_slice_indices)

    return good_slice_indices

  def _is_ready_to_reshard(self, step: int) -> bool:
    """Returns if it is time to reshard.

    May update `good_slice_indices`.

    Args:
      step: The current step.
    """
    if step % self.reshard_check_period:
      return False
    if self.good_slice_count >= self.total_slice_count:
      return False

    good_slice_indices = self.get_slice_availability()

    # If any of the existing good slices are no longer good, we cannot reshard.
    if self.good_slice_indices - good_slice_indices:
      return False

    if len(good_slice_indices) == len(self.good_slice_indices):
      return False

    _logger.debug("New slice available.")
    _logger.debug(
        "Previous good slice indices: self.good_slice_indices=%s",
        self.good_slice_indices,
    )
    _logger.debug(
        "Current good slice indices: good_slice_indices=%s", good_slice_indices
    )

    self.good_slice_indices = good_slice_indices

    return True

  @property
  def good_slice_to_devices(self) -> dict[int, Sequence[jax.Device]]:
    """The mapping from a good slice to its devices."""
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

  def _slice_down(self, reshard_retry: bool = False) -> None:
    """Slice down."""
    _logger.debug("Slice down")
    self.good_slice_indices = self.get_slice_availability()
    self.failure_count += 1
    if reshard_retry:
      self.reshard_retry_count += 1
    else:
      self.reshard_retry_count = 0

    _logger.debug(
        "self.failure_count=%s self.max_failure_count=%s",
        self.failure_count,
        self.max_failure_count,
    )
    if (
        self.max_failure_count is not None
        and self.failure_count >= self.max_failure_count
    ):
      raise ValueError(f"Max failure count reached {self.max_failure_count=}")

    _logger.debug(
        "self.reshard_retry_count=%s self.max_reshard_retry_count=%s",
        self.reshard_retry_count,
        self.max_reshard_retry_count,
    )
    if (
        self.max_reshard_retry_count is not None
        and self.reshard_retry_count > self.max_reshard_retry_count
    ):
      raise ValueError(
          f"Max reshard retry count reached {self.max_reshard_retry_count=}"
      )

  @timing.timeit
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
    self.maybe_snapshot(
        step=step,
        snapshot=snapshot,
        block=True,
        force=True,
    )

  def pop_snapshot(self) -> tuple[int, Mapping[str, int | PyTree]]:
    """Pop next snapshot."""

    if self._current_snapshot == 0:
      snapshot_dict = self._snapshot_0
      self._snapshot_0 = None
      self._current_snapshot = 1
    elif self._current_snapshot == 1:
      snapshot_dict = self._snapshot_1
      self._snapshot_1 = None
      self._current_snapshot = 0
    else:
      raise IndexError("No snapshots left")

    if snapshot_dict is None:
      raise IndexError("No snapshots left")

    step = snapshot_dict.pop("step")
    snapshot = snapshot_dict.pop("snapshot")

    return step, snapshot

  @staticmethod
  def _get_snapshot_size(snapshot: Mapping[str, int | PyTree]) -> int:
    """Returns the size of a snapshot.

    Ingores leaves that do not have a `nbytes` attribute.

    Args:
      snapshot: The snapshot to get the size of.
    """

    def size_in_bytes(x: Any) -> int:
      try:
        return x.nbytes
      except AttributeError:
        return 0

    nbytes_tree = jax.tree.map(size_in_bytes, snapshot)
    return jax.tree.reduce(operator.add, nbytes_tree, initializer=0)

  @staticmethod
  def _put_snapshot_on_host(
      snapshot: Mapping[str, int | PyTree],
  ) -> Mapping[str, int | PyTree]:
    """Puts a copy of the snapshot on the host.

    JAX arrays are copied to the host. Other leaves are copied if they have a
    `copy` method. Otherwise, they are copied by reference (unless literals).

    Args:
      snapshot: The snapshot to move to the host.

    Returns:
      A copy of the snapshot on the host.
    """
    slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, snapshot)
    slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
    _logger.debug("_put_snapshot_on_host: snapshot is on slices: %s", slice_indices)

    jax.block_until_ready(snapshot)
    shardings = jax.tree.map(lambda x: x.sharding.with_memory_kind("pinned_host"), snapshot)

    slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.device_set}, shardings)
    slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
    _logger.debug("_put_snapshot_on_host: shardings are on slices: %s", slice_indices)

    ret = jax.device_put(snapshot, shardings)
    jax.block_until_ready(ret)
    return ret

    # def copy(x: Any):
    #   try:
    #     # return jax.device_put(x, x.sharding)
    #     return jax.device_put(x, x.sharding.with_memory_kind("pinned_host"))
    #     return jax.device_put(x, x.sharding.with_memory_kind("pinned_host"))
    #   except AttributeError:
    #     pass
    #   raise RuntimeError(f"Snapshot leaf is not a JAX array {type(x)=}")

    #   try:
    #     return x.copy()
    #   except AttributeError:
    #     pass

    #   return x

    # return jax.tree.map(copy, snapshot)

  @timing.timeit
  def maybe_snapshot(
      self,
      step: int,
      snapshot: Mapping[str, int | PyTree],
      block: bool = False,
      force: bool = False,
  ) -> None:
    """Save step and state."""
    if not force and step % self.snapshot_period:
      _logger.debug("Not saving a snapshot")
      return

    total_nbytes = self._get_snapshot_size(snapshot)

    _logger.debug("Saving a snapshot of %s bytes", total_nbytes)

    slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, snapshot)
    slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
    _logger.debug("snapshot is on slices: %s", slice_indices)

    snapshot_host = {}
    for k, v in snapshot.items():
      snapshot_host[k] = self._put_snapshot_on_host(v)
    _logger.debug("Snapshot dispatched")

    slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, snapshot_host)
    slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
    _logger.debug("snapshot_host is on slices: %s", slice_indices)

    snapshot_dict = {"step": step, "snapshot": snapshot_host}
    if True or block:
      jax.block_until_ready(snapshot_dict)
      _logger.debug("Snapshot completed")

    if self._current_snapshot == 0:
      self._snapshot_1 = snapshot_dict
      self._current_snapshot = 1
    elif self._current_snapshot == 1:
      self._snapshot_0 = snapshot_dict
      self._current_snapshot = 0
    else:
      self._snapshot_0 = snapshot_dict
      self._current_snapshot = 0

  @timing.timeit
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

      slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, snapshot)
      slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
      _logger.debug("snapshot is on slices: %s", slice_indices)

      snapshot_host = {}
      snapshot_device = {}
      for k, v in snapshot.items():
        host, device = self._reshard_snapshot(v, mesh)
        snapshot_host[k] = host
        snapshot_device[k] = device

      slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, snapshot_host)
      slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
      _logger.debug("snapshot_host is on slices: %s", slice_indices)

      slice_index_tree = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, snapshot_device)
      slice_indices = jax.tree.reduce(operator.or_, slice_index_tree, set())
      _logger.debug("snapshot_device is on slices: %s", slice_indices)

      try:
        jax.block_until_ready(snapshot_device)
        break
      except Exception as error:  # pylint: disable=broad-except
        self._is_error_due_to_slice_down(error)
        _logger.debug("Retrying with the next snapshot")

    self.initialize_snapshot(
        step=step,
        snapshot=snapshot_device,
    )

    return step, snapshot_device

  @timing.timeit
  def maybe_reshard_down(
      self,
      error: Exception,
      elastic_handler: Callable[..., Any],
      handler_args: tuple[Any, ...] | None = None,
      handler_kwargs: Mapping[str, Any] | None = None,
      reshard_retry: bool = False,
  ) -> Any:
    """Reshards down if the error is due to slice down."""
    if handler_args is None:
      handler_args = ()

    if handler_kwargs is None:
      handler_kwargs = {}

    while True:
      if not self._is_error_due_to_slice_down(error):
        _logger.debug("Not resharding down")
        raise error from error.__cause__

      _logger.debug("Resharding down")
      self._slice_down(reshard_retry)

      try:
        ret = elastic_handler(*handler_args, **handler_kwargs)
        break
      except jax.errors.JaxRuntimeError as e:
        _logger.debug("Elastic handler raised an error.")
        error = e
        reshard_retry = True

    _logger.debug("Successfully resharded down")
    return ret

  @timing.timeit
  def maybe_reshard_up(
      self,
      step: int,
      snapshot: Mapping[str, int | PyTree],
      elastic_handler: Callable[..., Any],
      handler_args: tuple[Any, ...] | None = None,
      handler_kwargs: Mapping[str, Any] | None = None,
  ) -> Any:
    """Reshards up if it is time to reshard."""
    if handler_args is None:
      handler_args = ()

    if handler_kwargs is None:
      handler_kwargs = {}

    if not self._is_ready_to_reshard(step):
      _logger.debug("Not resharding up")
      return

    self.maybe_snapshot(
        step=step,
        snapshot=snapshot,
        force=True,
        block=True,
    )

    try:
      ret = elastic_handler(*handler_args, **handler_kwargs)
    except jax.errors.JaxRuntimeError as error:
      _logger.debug("Elastic handler failed. Trying again")
      ret = self.maybe_reshard_down(
          error=error,
          elastic_handler=elastic_handler,
          handler_args=handler_args,
          handler_kwargs=handler_kwargs,
          reshard_retry=True,
      )

    _logger.debug("Finished resharding up")
    return ret

  @timing.timeit
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
    # Don't donate in case the snapshot is duplicated in _snapshot
    resharded_pinned_host = reshard.reshard(
        x,
        sharding_pinned_host,
        put_array=self.device_put_per_shard0,
        donate_input=False,
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
        put_array=self.device_put_per_shard0,
        donate_input=False,
    )

    return resharded_pinned_host, resharded_device

  def device_put_per_shard0(
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
      _logger.debug("Getting shards from good_slice_indices=%s", self.good_slice_indices)
      arrays = []
      for x in arr.addressable_shards:
        try:
          jax.block_until_ready(x.data)
          if x.device.slice_index not in self.good_slice_indices:
            _logger.debug("shard ready on slice: %s which is supposed to be bad", x.device.slice_index)
        except:
          if x.device.slice_index in self.good_slice_indices:
            _logger.debug("shard not ready on slice: %s which is supposed to be good", x.device.slice_index)

        if x.device.slice_index in self.good_slice_indices:
          arrays.append(x.data)

      # arrays = [
      #     x.data
      #     for x in arr.addressable_shards
      #     if x.device.slice_index in self.good_slice_indices
      # ]
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

  def device_put_per_shard1(
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
  def device_put_per_shard2(
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

