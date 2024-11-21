"""Utilities for elastic training."""

import collections
import contextlib
import itertools
import functools
import logging
import os
import sys
import time
import threading
import traceback
from typing import Sequence, Any, Optional, Callable
import jax
import numpy as np

PyTree = Any

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


@contextlib.contextmanager
def timer(name: str):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    logger.info("%s elaspsed %.2fs.", name, end - start)

def timeit(func: Callable):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with timer(func.__name__):
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

    logger.info(f"Failure count: {self.failure_count} with max {self.max_failures}")
    if self.failure_count >= self.max_failures:
      logger.fatal(f"Max failures reached {self.max_failures}")

  @timeit
  def save(self, save_step: int, **kwargs):
    """Save step and state."""
    # In case DATA_LOSS occurs during jax.block_until_ready, overwrite self.data
    # at the end
    data = {k: jax.tree.map(lambda x: x.copy(), v) for k, v in kwargs.items()}
    for v in data.values():
      jax.block_until_ready(v)
    data["save_step"] = save_step

    self.data = data

  def is_ready_to_reshard(self, step: int):
    """
    Indicates if it is time to reshard.

    May update `good_slice_indices`.
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
        with timer(f"checking {slice_index=}"):
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

  @staticmethod
  @timeit
  def reshard(
      tree: PyTree,
      mesh: jax.sharding.Mesh,
      *,
      donate: bool = True,
  ) -> PyTree:
    """Reshard a PyTree."""
    def func(leaf):
        return jax.device_put(
            leaf,
            jax.sharding.NamedSharding(mesh, leaf.sharding.spec),
            donate=donate,
        )

    return jax.tree.map(func, tree)

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
def watchdog(timeout):
  event = threading.Event()

  def handler():
    count = 0
    while not event.wait(timeout):
      logger.info(f"Watchdog thread dump every {timeout=} seconds. {count=}")
      try:
        for thread in threading.enumerate():
          try:
            logger.info(f"Thread: {thread.ident}")
            logger.info("".join(traceback.format_stack(sys._current_frames().get(thread.ident, []))))
          except:
            logger.info(f"Error print traceback for {thread.ident=}")
            pass
      finally:
        # logger.fatal("Timeout from timebomb!")
        # os.abort()
        pass

      count += 1

  logger.debug("Registering watchdog")
  watchdog = threading.Thread(target=handler, name="watchdog")
  watchdog.start()
  try:
    yield
  finally:
    event.set()
    watchdog.join()
    logger.debug("Degistering watchdog")
