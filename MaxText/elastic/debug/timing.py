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
"""Timing utilities.

This module provides utilities for timing code with decorators and context
managers.
"""

import functools
import logging
import time
from typing import Any, Callable


_logger = logging.getLogger(__name__)


class Timer:
  """Timer context manager.

  Attributes:
    name: The name of the timer.
    start: The start time of the timer.
    stop: The stop time of the timer.
    duration: The elapsed time of the timer.
  """

  def __init__(self, name: str):
    self.name = name

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.stop = time.time()
    self.duration = self.stop - self.start
    _logger.debug(str(self))

  def __str__(self):
    return f"{self.name} elapsed {self.duration:.4f} seconds."


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
  """Decorator to time a function.

  Args:
    func: The function to time.

  Returns:
    The decorated function.
  """

  @functools.wraps(func)
  def wrapper(*args: Any, **kwargs: Any):
    with Timer(getattr(func, "__name__", "Unknown")):
      return func(*args, **kwargs)

  return wrapper
