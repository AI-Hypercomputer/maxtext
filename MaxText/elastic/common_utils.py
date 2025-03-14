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
"""Common utilities."""

import contextlib
import functools
import logging
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable

import jax


logger = logging.getLogger(__name__)


class Profile:
  """Profile context manager.

  Attributes:
    gcs_path: The GCS path to save the profile. If None, the profile will not be
      saved.
  """

  def __init__(self, gcs_path: str | None = None):
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
    logger.debug(str(self))

  def __str__(self):
    return f"{self.name} elapsed {self.time}."


def timeit(
    func: Callable[..., Any], name: str | None = None
) -> Callable[..., Any]:
  if name is None:
    name = getattr(func, "__name__", "Unknown")

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with Timer(name):
      return func(*args, **kwargs)

  return wrapper


@contextlib.contextmanager
def watchdog(timeout: float, repeat: bool = True):
  """Watchdog context manager.

  Prints the stack trace of all threads after `timeout` seconds.

  Args:
    timeout: The timeout in seconds. If the timeout is reached, the stack trace
      of all threads will be printed.
    repeat: Whether to repeat the watchdog after the timeout. If False, the
      process will be aborted after the first timeout.

  Yields:
    None
  """
  event = threading.Event()

  def handler():
    count = 0
    while not event.wait(timeout):
      logger.debug(
          "Watchdog thread dump every %s seconds. Count: %s", timeout, count
      )
      try:
        for thread in threading.enumerate():
          try:
            logger.debug("Thread: %s", thread.ident)
            logger.debug(
                "".join(
                    traceback.format_stack(
                        sys._current_frames().get(  # pylint: disable=protected-access
                            thread.ident, []
                        )
                    )
                )
            )
          except Exception:  # pylint: disable=broad-exception-caught
            logger.debug("Error print traceback for thread: %s", thread.ident)
            pass
      finally:
        if not repeat:
          logger.critical("Timeout from watchdog!")
          os.abort()

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

