"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

# Needed to import files from the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import maxtext_xpk_runner as mxr

# Constants for defining supported actions
DELETE = "delete"


def _handle_delete(
    cluster_config: mxr.XpkClusterConfig, user: str, **kwargs
) -> int:
  """Handles the deletion of workloads.

  Args:
      cluster_config: mxr.XpkClusterConfig object
      user: User string
      **kwargs: Optional keyword arguments, such as xpk_path
  """
  xpk_path = kwargs.get("xpk_path", "xpk")  # Default to "xpk" if not provided
  first_five_chars = user[:5]
  delete_command = (
      f"python3 {xpk_path}/xpk.py workload delete "
      f"--project={cluster_config.project} --cluster={cluster_config.cluster_name}"
      f" --filter-by-job={first_five_chars} --zone={cluster_config.zone}"
  )
  print(
      f"Deleting workloads starting with: {first_five_chars} using command:"
      f" {delete_command}"
  )
  os.system(delete_command)


def handle_cmd_args(
    cluster_config: mxr.XpkClusterConfig, *actions: str, **kwargs
) -> bool:
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
