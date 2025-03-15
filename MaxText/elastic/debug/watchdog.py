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
"""Watchdog context manager.

This context manager is used to monitor the progress of a long running process.
If the process takes longer than the specified timeout, it will print the stack
trace of all threads.
"""

import contextlib
import logging
import os
import sys
import threading
import traceback


_logger = logging.getLogger(__name__)


def _log_thread_stack(thread: threading.Thread):
  _logger.debug("Thread: %s", thread.ident)
  _logger.debug(
      "".join(
          traceback.format_stack(
              sys._current_frames().get(  # pylint: disable=protected-access
                  thread.ident, []
              )
          )
      )
  )


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
      _logger.debug(
          "Watchdog thread dump every %s seconds. Count: %s", timeout, count
      )
      try:
        for thread in threading.enumerate():
          try:
            _log_thread_stack(thread)
          except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("Error print traceback for thread: %s", thread.ident)
      finally:
        if not repeat:
          _logger.critical("Timeout from watchdog!")
          os.abort()

      count += 1

  _logger.debug("Registering watchdog")
  watchdog_thread = threading.Thread(target=handler, name="watchdog")
  watchdog_thread.start()
  try:
    yield
  finally:
    event.set()
    watchdog_thread.join()
    _logger.debug("Deregistering watchdog")
