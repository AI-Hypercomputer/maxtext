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

"""
This module provides helper functions for parsing command-line arguments
in benchmark recipes.

It primarily offers a standardized way to handle a `--delete` flag, which can
be used to clean up existing XPK workloads before starting a new run.
"""

import argparse
import os

from benchmarks.benchmark_utils import get_xpk_path
from benchmarks.xpk_configs import XpkClusterConfig

# Constants for defining supported actions
DELETE = "delete"


def _handle_delete(cluster_config: XpkClusterConfig, user: str, **kwargs) -> int:
  """Handles the deletion of workloads.

  Args:
      cluster_config: XpkClusterConfig object
      user: User string
      **kwargs: Optional keyword arguments, such as xpk_path
  """
  xpk_path = kwargs.get("xpk_path", get_xpk_path())  # Default to "xpk" if not provided
  first_three_chars = user[:3]
  delete_command = (
      f"python3 {xpk_path}/xpk.py workload delete "
      f"--project={cluster_config.project} --cluster={cluster_config.cluster_name}"
      f" --filter-by-job={first_three_chars} --zone={cluster_config.zone}"
  )
  print(f"Deleting workloads starting with: {first_three_chars} using command:" f" {delete_command}")
  os.system(delete_command)


def handle_delete_specific_workload(cluster_config: XpkClusterConfig, workload_name: str, **kwargs) -> int:
  """Handles the deletion of workloads with a specific name.

  Args:
      cluster_config: XpkClusterConfig object
      workload_name: workload name
      **kwargs: Optional keyword arguments, such as xpk_path
  """
  xpk_path = kwargs.get("xpk_path", get_xpk_path())
  delete_command = (
      f"python3 {xpk_path}/xpk.py workload delete "
      f"--project={cluster_config.project} --cluster={cluster_config.cluster_name}"
      f" --filter-by-job={workload_name} --zone={cluster_config.zone}"
  )
  print(f"Deleting workload: {workload_name} using command:" f" {delete_command}")
  os.system(f"yes | {delete_command}")


def handle_cmd_args(cluster_config: XpkClusterConfig, *actions: str, **kwargs) -> bool:
  """Parses command-line arguments and executes the specified actions.

  Args:
      cluster_config: Contains Cluster configuration information that's helpful
        for running the actions.
      *actions: Variable number of string arguments representing the actions to
        be performed.
      **kwargs: Optional keyword arguments to be passed to action handlers.

  Raises:
    ValueError: If an unsupported action is provided or if unknown arguments are
    passed.
  """

  parser = argparse.ArgumentParser()

  if DELETE in actions:
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete workloads starting with the user's first five characters.",
    )

  known_args, unknown_args = parser.parse_known_args()

  if unknown_args:
    raise ValueError(f"Unrecognized arguments: {unknown_args}")

  # Get user
  user = os.environ["USER"]

  # Handle actions
  should_continue = True
  if DELETE in actions and known_args.delete:
    _handle_delete(cluster_config, user, **kwargs)
    should_continue = False

  return should_continue
