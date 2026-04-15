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

"""Utility functions for Elastic Training."""

import functools
import jax
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
import pathwaysutils
from pathwaysutils.elastic import manager


elastic_manager: manager.Manager | None = None


def elastic_enabled(config) -> bool:
  """Returns whether elastic mode is enabled."""
  return pathwaysutils.is_pathways_backend_used() and config.elastic_enabled


def clean_up_checkpoints(checkpoint_dir: str):
  """Cleans up incomplete checkpoints after an elastic event."""
  max_logging.log("Elastic utils: Checking for incomplete checkpoint after an elastic event...")
  checkpoint_dir = gcs_utils.add_trailing_slash(checkpoint_dir)

  # 1. List the "directories" (steps)
  checkpoints = gcs_utils.gcs_list_directories(checkpoint_dir)

  # 2. Filter for directories that are numbers
  checkpoints = [cp for cp in checkpoints if cp.isdigit()]

  if not checkpoints:
    max_logging.log("Found no existing checkpoints. Continuing")
    return

  # Sort naturally (numerical sort) and get the last one
  checkpoints.sort(key=int)
  latest_checkpoint_name = checkpoints[-1]
  latest_checkpoint_path = f"{checkpoint_dir}{latest_checkpoint_name}/"

  max_logging.log(f"Checking latest checkpoint: {latest_checkpoint_path}")

  # 3. Check for commit_success file
  success_markers = gcs_utils.gcs_glob_pattern(f"{latest_checkpoint_path}commit_success*")

  if not success_markers:
    max_logging.log(f"No commit_success file found. Deleting {latest_checkpoint_path}...")
    # TODO: Use Orbax 'Cancel Ongoing Checkpointing' API when available to
    # prevent deleting a checkpoint that is currently being written.
    gcs_utils.gcs_delete_directory(latest_checkpoint_path)
  else:
    max_logging.log(f"Found commit_success file. Keeping {latest_checkpoint_path}.")


def live_devices():
  """Returns the list of live devices."""
  global elastic_manager
  # If pathways is not used or elastic_manager is not initialized, return all devices
  if pathwaysutils.is_pathways_backend_used():
    if elastic_manager is None:
      elastic_manager = manager.Manager()
    # Filter devices that are in active slices
    return [d for d in jax.devices() if d.slice_index in elastic_manager.active_slice_indices]
  return jax.devices()


def chain_callbacks(*funcs):
  """Helper function to chain callbacks."""

  def wrapper():
    for func in funcs:
      func()

  return wrapper


def elastic_retry(config, callback_fn=None):
  """Decorator for elastic retry.

  If an elastic event occurs, the decorator will retry the decorated function
  up to `config.elastic_max_retries` times.
  Before each retry, it cleans up partial checkpoints by calling
  `clean_up_checkpoints`. If `callback_fn` is provided, it is
  called after `clean_up_checkpoints`.

  Args:
    config: Config object.
    callback_fn: Optional callback function to be called after
      `clean_up_checkpoints` on an elastic event.

  Returns:
    A decorator for elastic retry.
  """
  global elastic_manager
  if not elastic_enabled(config):
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled:"
        f" {config.elastic_enabled}, pathways backend used:"
        f" {pathwaysutils.is_pathways_backend_used()}"
    )
    raise ValueError(msg)

  max_logging.log("Elastic Retry Enabled")
  if elastic_manager is None:
    elastic_manager = manager.Manager()

  cleanup_partial = functools.partial(clean_up_checkpoints, config.checkpoint_dir)

  if callback_fn is None:
    effective_callback = cleanup_partial
  else:
    effective_callback = chain_callbacks(cleanup_partial, callback_fn)

  return elastic_manager.elastic_retry(
      max_retries=config.elastic_max_retries,
      timeout=config.elastic_timeout_seconds,
      on_elastic_event_callback=effective_callback,
  )
