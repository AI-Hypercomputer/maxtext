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
from collections import Counter

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


def ensure_elastic_manager_initialized(config):
  """Initializes elastic manager if it's not initialized and pathways is used."""
  global elastic_manager
  if pathwaysutils.is_pathways_backend_used() and config.elastic_enabled and elastic_manager is None:
    elastic_manager = manager.Manager()


def get_local_batch_size(config) -> int:
  """Returns the local batch size based on the config."""
  return config.per_device_batch_size * get_devices_per_host(config)


def live_devices(config=None):
  """Returns the list of live devices."""
  # If pathways is not used or elastic_manager is not initialized, return all devices
  if pathwaysutils.is_pathways_backend_used() and config is not None:
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    # Filter devices that are in active slices
    return [
        d for d in jax.devices() if d is not None and getattr(d, "slice_index", 0) in elastic_manager.active_slice_indices
    ]
  return jax.devices()


def live_slice_indices(config) -> set[int]:
  """Returns the set of live slice indices."""
  return {getattr(d, "slice_index", 0) for d in live_devices(config) if d is not None}


def get_devices_per_host(config):
  """Dynamically calculates the number of chips per physical worker VM."""
  devices = Counter(d.task_id for d in live_devices(config))

  max_logging.log(f"elastic_utils: Device counts per task: {devices}")
  if not devices:
    raise ValueError("elastic_utils: get_devices_per_host: No devices found.")

  devices_per_host = next(iter(devices.values()))
  if devices_per_host == 0:
    raise ValueError("elastic_utils: get_devices_per_host: Devices per host is 0.")
  max_logging.log(f"elastic_utils: Devices per host: {devices_per_host}")

  return devices_per_host


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
  if not elastic_enabled(config):
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled:"
        f" {config.elastic_enabled}, pathways backend used:"
        f" {pathwaysutils.is_pathways_backend_used()}"
    )
    raise ValueError(msg)

  max_logging.log("Elastic Retry Enabled")

  ensure_elastic_manager_initialized(config)
  assert elastic_manager is not None

  cleanup_partial = functools.partial(clean_up_checkpoints, config.checkpoint_dir)

  if callback_fn is None:
    effective_callback = cleanup_partial
  else:
    effective_callback = chain_callbacks(cleanup_partial, callback_fn)

  return elastic_manager.elastic_retry(
      max_retries=config.elastic_max_retries,
      timeout=config.elastic_timeout_seconds,
      minimum_slice_count=None if config.elastic_min_slice_count == -1 else config.elastic_min_slice_count,
      on_elastic_event_callback=effective_callback,
  )


def is_scale_up_event(config) -> bool:
  """Returns whether a scale up event is detected."""
  if elastic_enabled(config):
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    return elastic_manager.new_slice_event.is_set()

  return False


def maybe_elastic_scale_up(config, checkpoint_manager):
  """Waits for a checkpoint to finish before interrupting for scale up."""
  if is_scale_up_event(config):
    max_logging.log(
        "Started a checkpoint and a new slice is available. Waiting for current"
        " checkpoint to finish before interrupting."
    )
    if checkpoint_manager is not None:
      checkpoint_manager.wait_until_finished()
    max_logging.log("Checkpoint save completed. Interrupting")
    raise manager.ScaleUpSignalError()
