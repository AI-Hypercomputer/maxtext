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

"""Logging utilities."""
from absl import logging


def log(user_str):
  """Logs a message at the INFO level."""
  # Note, stacklevel=2 makes the log show the caller of this function.
  logging.info(user_str, stacklevel=2)


def debug(user_str):
  """Logs a message at the DEBUG level."""
  logging.debug(user_str, stacklevel=2)


def info(user_str):
  """Logs a message at the INFO level."""
  logging.info(user_str, stacklevel=2)


def warning(user_str):
  """Logs a message at the WARNING level."""
  logging.warning(user_str, stacklevel=2)


def error(user_str):
  """Logs a message at the ERROR level."""
  logging.error(user_str, stacklevel=2)
