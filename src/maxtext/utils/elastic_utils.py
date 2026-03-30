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
import re
import subprocess
import jax

import maxtext.utils.max_logging as max_logging
import pathwaysutils
from pathwaysutils.elastic import manager


elastic_manager: manager.Manager | None = None


def elastic_mode_enabled(config) -> bool:
  """Returns whether elastic mode is enabled."""
  return (pathwaysutils.is_pathways_backend_used() and
          config.elastic_pause_resume)


def clean_up_checkpoints(checkpoint_dir: str):
  """Cleans up incomplete checkpoints after an elastic event."""
  max_logging.log(f"Elastic utils: Checking for incomplete checkpoint after an elastic event...")
  checkpoint_dir = f"{checkpoint_dir}"

  # 1. List the directory
  result = subprocess.run(['gsutil', 'ls', checkpoint_dir], capture_output=True, text=True)

  if result.returncode != 0:
    max_logging.log("Failed to inspect checkpoint dir. Continuing")
    return

  # 2. Filter for directories ending in numbers/ (equivalent to your grep and sort)
  checkpoints = [line for line in result.stdout.splitlines() if re.search(r'/\d+/$', line)]

  if not checkpoints:
    max_logging.log("Found no existing checkpoints. Continuing")
    return

  # Sort naturally (Version sort) and get the last one
  checkpoints.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
  latest_checkpoint = checkpoints[-1]

  max_logging.log(f"Checking latest checkpoint: {latest_checkpoint}")

  # 3. Check for commit_success file
  # gsutil -q stat returns 0 if found, non-zero if not
  stat_check = subprocess.run(['gsutil', '-q', 'stat', f"{latest_checkpoint}commit_success*"])

  if stat_check.returncode != 0:
    max_logging.log(f"No commit_success file found. Deleting {latest_checkpoint}...")
    subprocess.run(['gsutil', '-m', 'rm', '-rf', latest_checkpoint])
  else:
    max_logging.log(f"Found commit_success file. Keeping {latest_checkpoint}.")


def live_devices():
  device_list = jax.devices()

  if pathwaysutils.is_pathways_backend_used() and elastic_manager is not None:
      return [
          d for d in device_list
          if d.slice_index in elastic_manager.active_slice_indices
      ]
  else:
    return device_list


def elastic_pause_resume(config, callback_fn=None):
  """Pauses and resumes elastic training."""
  cleanup_partial = functools.partial(
      clean_up_checkpoints, config.checkpoint_dir
  )
  callback_fn = cleanup_partial if callback_fn is None else callback_fn
  return elastic_manager.elastic_retry(
      max_retries=10,
      poll_interval=10,
      timeout=config.elastic_timeout,
      on_elastic_event_callback=callback_fn,
  )


