"""Utilities for elastic training."""

import collections
import contextlib
import functools
import itertools
import logging
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional, Sequence

import jax
import numpy as np

jax._src.array.ArrayImpl._check_if_deleted = lambda _: False  # pylint: disable=protected-access

PyTree = Any

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

#  pylint: disable=logging-fstring-interpolation


class Profile:
  """Profile context manager."""

  def __init__(self, gcs_path: Optional[str] = None):
    self.gcs_path = gcs_path

  def __enter__(self):
    if self.gcs_path:
      jax.profiler.start_trace(self.gcs_path)

  def __exit__(self, exc_type, exc_value, tb):
    if self.gcs_path:
      jax.profiler.stop_trace()


class Timer:
  """Timer context manager."""

  def __init__(self, name):
    self.name = name

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.stop = time.time()
    self.time = self.stop - self.start
    logger.info(str(self))

  def __str__(self):
    return f"{self.name} elaspsed {self.time}." 


def timeit(
    func: Callable[..., Any], name: Optional[str] = None
) -> Callable[..., Any]:
  if name is None:
    name = getattr(func, "__name__", "Unknown")

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with Timer(name):
      return func(*args, **kwargs)
  return wrapper


class ElasticUtils:
  """Utility class for elastic training."""

  TEST_VALUE = 100

  def __init__(
      self,
      devices: Sequence[jax.Device],
      total_slice_count: int,
      save_period: Optional[int] = None,
      reshard_check_period: Optional[int] = None,
      max_failures: Optional[int] = None,
  ):
    self.devices = devices
    self.total_slice_count = total_slice_count

    if save_period is None:
      save_period = 1
    self.save_period = save_period

    if reshard_check_period is None:
      reshard_check_period = 1
    self.reshard_check_period = reshard_check_period

    if max_failures is None:
      max_failures = float("inf")
    self.max_failures = max_failures

    self.failure_count = 0
    self.good_slice_indices = self.get_slice_availability()
    self.data = {}

  def slice_down(self):
    """Slice down."""
    logger.info("Slice down")
    self.good_slice_indices = self.get_slice_availability()
    self.failure_count += 1

    logger.info(
        f"Failure count: {self.failure_count} with max {self.max_failures}"
    )
    if self.failure_count >= self.max_failures:
      logger.fatal(f"Max failures reached {self.max_failures}")

  @timeit
  def save(self, save_step: int, **kwargs):
    """Save step and state."""
    # In case DATA_LOSS occurs during jax.block_until_ready, overwrite self.data
    # at the end
    data = {
        k: jax.device_put(
            v,
            jax.tree.map(lambda x: x.sharding.with_memory_kind(kind="pinned_host"), v),
        )
        for k, v in kwargs.items()
    }
    for v in data.values():
      jax.block_until_ready(v)
    data["save_step"] = save_step

    self.data = data

  def is_ready_to_reshard(self, step: int):
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

    if len(good_slice_indices) <= self.good_slice_count:
      return False

    logger.info("New slice available.")
    logger.info(f"Previous good slice indices: {self.good_slice_indices}")
    logger.info(f"Current good slice indices: {good_slice_indices}")

    if not good_slice_indices & self.good_slice_indices:
      raise ValueError("All copies of the data have been lost")

    self.good_slice_indices = good_slice_indices

    return True

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
    return self.slice_to_devices[next(iter(self.good_slice_indices))][0]

  @property
  def good_slice_count(self) -> int:
    """Returns the number of slices."""
    return len(self.good_slice_indices)

  def slice_device_count(self, slice_index: int) -> int:
    """Returns the number of devices in a slice."""
    return len(self.slice_to_devices[slice_index])

  def _simple_execution(
      self, devices: Sequence[jax.Device], block: bool = True
  ) -> jax.Array:
    """Simple execution to test if a slice is available."""
    x = np.zeros(len(devices), dtype=float) + (self.TEST_VALUE - 1)
    y = jax.pmap(lambda x: x + 1, devices=devices)(x)
    if block:
      y.block_until_ready()
    return y

  @timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = set()

    results = {
        slice_index: self._simple_execution(devices, block=False)
        for slice_index, devices in self.slice_to_devices.items()
    }

    for slice_index, x in results.items():
      logger.info(f"checking {slice_index=}")  # pylint: disable=logging-fstring-interpolation
      expected = (
          np.zeros(self.slice_device_count(slice_index), dtype=float)
          + self.TEST_VALUE
      )
      try:
        with Timer(f"checking {slice_index=}"):
          if np.allclose(x, expected):
            good_slice_indices.add(slice_index)
            logger.info(f"{slice_index=} good")
          else:
            logger.error(  # pylint: disable=logging-fstring-interpolation
                f"Error with _simple_execution for {slice_index=}. "
                "This should not happen."
            )
      except jax.errors.JaxRuntimeError as e:
        if "DATA_LOSS" in str(e):
          logger.info(  # pylint: disable=logging-fstring-interpolation
              f"Caught JaxRuntimeError DATA_LOSS exception for {slice_index=}"
          )
          logger.info(f"{e}")
        else:
          logger.exception(f"Unknown JaxRuntimeError for {slice_index=}")  # pylint: disable=logging-fstring-interpolation
        logger.info(f"{slice_index=} bad")

    logger.info(f"{good_slice_indices=}")

    return good_slice_indices

  @classmethod
  @timeit
  def reshard(
      cls,
      x: Any,
      sharding: jax.sharding.Sharding | Any,
      *,
      donate_input: bool = True,
      put_array: Optional[
          Callable[
              [jax.Array, Sequence[jax.sharding.Sharding], bool], jax.Array
          ]
      ] = None,
  ) -> Any:
    """Reshards `x` to the specified `sharding`.

    Args:
        x: An array, scalar, or a nested Python container thereof.
        sharding: A `Sharding` or a nested `Sharding` in a Python container
          (must match the structure of `x`), specifying the target sharding.
        donate_input: If `True`, donates the input arrays to reduce memory
          needed for resharding. Donated buffers should not be reused.
        put_array: A function that takes an array, a sharding, and a boolean
          indicating whether to donate the input, and returns a copy of the
          array with the specified sharding.

    Returns:
        A copy of `x` with the specified `sharding`.
    """
    if put_array is None:
      put_array = cls.default_put_array

    flat_x, tree_def = jax.tree_util.tree_flatten(x)
    flat_sharding = jax.api_util.flatten_axes(
        "reshard sharding", tree_def, sharding
    )

    if len(flat_x) != len(flat_sharding):
      raise ValueError("Mismatched length between `x` and `sharding`.")

    arrays = [
        put_array(arr, dst_sharding, donate_input)
        for arr, dst_sharding in zip(flat_x, flat_sharding)
    ]
    return jax.tree_util.tree_unflatten(tree_def, arrays)

  @staticmethod
  def put_array_device_put0(
      arr: jax.Array,
      dst_sharding: jax.sharding.Sharding,
      donate_input: bool,
  ):
    if not isinstance(dst_sharding, jax.sharding.Sharding):
      raise ValueError("`sharding` must contain only `Sharding` instances.")
    return jax.device_put(arr, dst_sharding, donate=donate_input)

  default_put_array = put_array_device_put0

  def put_array_device_put1(
      self,
      arr: jax.Array,
      dst_sharding: jax.sharding.Sharding,
      donate_input: bool,  # pylint: disable=unused-argument
  ):
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
      arrays = [x.data for x in arr.addressable_shards]

      good_reference_slice = arr.addressable_shards[0].device.slice_index
      good_reference_arrays = [
          array
          for array in arrays
          if array.device.slice_index == good_reference_slice
      ]

      new_slice_index = (
          self.good_slice_indices
          - {d.slice_index for d in arr.sharding.device_set}
      ).pop()

      for device, array in zip(
          self.slice_to_devices[new_slice_index], good_reference_arrays
      ):
        arrays.append(jax.device_put(array, device))

    return jax.make_array_from_single_device_arrays(
        arr.shape, dst_sharding, arrays
    )

  def put_array_device_put2(
      self,
      arr: jax.Array,
      dst_sharding: jax.sharding.Sharding,
      donate_input: bool,  # pylint: disable=unused-argument
  ):
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
      arrays = [x.data for x in arr.addressable_shards]

      good_reference_slice = arr.addressable_shards[0].device.slice_index
      good_reference_arrays = [
          array
          for array in arrays
          if array.device.slice_index == good_reference_slice
      ]

      new_slice_index = (
          self.good_slice_indices
          - {d.slice_index for d in arr.sharding.device_set}
      ).pop()

      new_arrays = jax.device_put(
          good_reference_arrays, self.slice_to_devices[new_slice_index]
      )

      arrays += new_arrays

    return jax.make_array_from_single_device_arrays(
        arr.shape, dst_sharding, arrays
    )

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
      raise ValueError(f"Unsupported type: {type(x)}")


@contextlib.contextmanager
def watchdog(timeout: float):
  """Watchdog context manager.

  Prints the stack trace of all threads every `timeout` seconds.

  Args:
    timeout: The timeout in seconds.

  Yields:
    None
  """
  event = threading.Event()

  def handler():
    count = 0
    while not event.wait(timeout):
      logger.info(f"Watchdog thread dump every {timeout=} seconds. {count=}")
      try:
        for thread in threading.enumerate():
          try:
            logger.info(f"Thread: {thread.ident}")
            logger.info(
                "".join(
                    traceback.format_stack(
                        sys._current_frames()  # pylint: disable=protected-access
                        .get(thread.ident, [])
                    )
                )
            )
          except Exception:  # pylint: disable=broad-exception-caught
            logger.info(f"Error print traceback for {thread.ident=}")
            pass
      finally:
        # logger.fatal("Timeout from timebomb!")
        # os.abort()
        pass

      count += 1

  logger.debug("Registering watchdog")
  watchdog_thread = threading.Thread(target=handler, name="watchdog")
  watchdog_thread.start()
  try:
    yield
  finally:
    event.set()
    watchdog_thread.join()
    logger.debug("Deregistering watchdog")

