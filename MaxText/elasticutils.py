"""Utilities for elastic training."""

import collections
import contextlib
import itertools
import logging
import time
from typing import Sequence, Any, Optional
import jax
import numpy as np

PyTree = Any

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def timer(name: str):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    logger.info("%s elaspsed %.2fs.", name, end - start)


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
    self.good_data_slice_indices = set()

    self.data = {}

  def slice_down(self):
    """Slice down."""
    logger.info("Slice down")
    self.good_slice_indices = self.get_slice_availability()
    self.failure_count += 1

  def save(self, data: Any, save_based_on_step: Optional[int] = None):
    """Maybe save data."""
    if save_based_on_step is None or save_based_on_step % self.save_period == 0:
      self.data = data

  def is_ready_to_reshard(self, step: int):
    """Indicates if it is time to reshard."""
    if step % self.reshard_check_period:
      return False
    if self.good_slice_count >= self.total_slice_count:
      return False

    good_slice_indices = self.get_slice_availability()

    if len(good_slice_indices) <= self.good_slice_count:
      return False

    logger.info("New slice available. %s", good_slice_indices)

    self.good_data_slice_indices = (
        good_slice_indices & self.good_slice_indices
    )

    if not self.good_data_slice_indices:
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

  def _get_slice_availability(self) -> set[int]:
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
        with timer(f"checking {slice_index=}"):
          if np.allclose(x, expected):
            good_slice_indices.add(slice_index)
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
        else:
          logger.exception(f"Unknown JaxRuntimeError for {slice_index=}")  # pylint: disable=logging-fstring-interpolation

    return good_slice_indices

  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    with timer("get_slice_availability"):
      return self._get_slice_availability()

  @staticmethod
  def _reshard(
      x: PyTree,
      dst_sharding: jax.sharding.Sharding,
      *,
      donate: bool = True,
  ) -> PyTree:
    """Reshard a PyTree."""
    flat_x, tree_def = jax.tree.flatten(x)
    flat_sharding = jax.api_util.flatten_axes(
        "reshard sharding", tree_def, dst_sharding
    )

    if len(flat_x) != len(flat_sharding):
      raise ValueError("Mismatched length between `x` and `sharding`.")

    arrays = []
    for arr, dst_arr_sharding in zip(flat_x, flat_sharding):
      if not isinstance(dst_arr_sharding, jax.sharding.Sharding):
        raise ValueError(
            "`dst_sharding` must contain only `jax.sharding.Sharding`"
        )
      arrays.append(jax.device_put(arr, dst_arr_sharding, donate=donate))

    return jax.tree.unflatten(tree_def, arrays)

  @staticmethod
  def reshard(
      x: PyTree,
      dst_sharding: jax.sharding.Sharding,
      *,
      donate: bool = True,
  ) -> PyTree:
    """Reshard a PyTree."""
    with timer("reshard"):
      return ElasticUtils._reshard(x, dst_sharding, donate=donate)

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

