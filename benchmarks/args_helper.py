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

import maxtext_xpk_runner as mxr


# Constants for defining supported actions
# Add more actions here as needed, e.g.,
DELETE = "delete"


def _handle_delete(cluster_config: mxr.XpkClusterConfig, user: str) -> None:
  """Handles the deletion of workloads."""
  first_five_chars = user[:5]
  delete_command = (
      "python3 xpk/xpk.py workload delete "
      f"--project={cluster_config.project} --cluster={cluster_config.cluster_name}"
      f" --filter-by-job={first_five_chars} --zone={cluster_config.zone}"
  )
  print(
      f"Deleting workloads starting with: {first_five_chars} using command:"
      f" {delete_command}"
  )
  os.system(delete_command)


def handle_cmd_args(
    cluster_config: mxr.XpkClusterConfig, *actions: str
) -> None:
  """Parses command-line arguments and executes the specified actions.

  Args:
      cluster_config: Contains Cluster configuration information that's helpful
        for running the actions.
      *actions: Variable number of string arguments representing the actions to
        be performed.

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
  if DELETE in actions and known_args.delete:
    _handle_delete(cluster_config, user)
