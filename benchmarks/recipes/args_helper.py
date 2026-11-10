# Copyright 2023–2026 Google LLC
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
be used to clean up existing Cluster Toolkit (gcluster) workloads before
starting a new run.
"""

import os
import subprocess

from benchmarks.ctk_configs import ClusterConfig
from benchmarks.recipes.runner_utils import isolated_kubeconfig


def _handle_delete(cluster_config: ClusterConfig, user: str, **kwargs) -> int:
  """Handles the deletion of workloads starting with the user's prefix.

  Args:
      cluster_config: ClusterConfig object
      user: User string
      **kwargs: Optional keyword arguments (retained for backward compatibility)
  """
  del kwargs
  first_three_chars = user[:3]
  location = getattr(cluster_config, "location", None) or cluster_config.zone
  get_jobs_cmd = "kubectl get jobset -o custom-columns=NAME:.metadata.name --no-headers"
  try:
    with isolated_kubeconfig(cluster_config.cluster_name, cluster_config.project, location):
      output = subprocess.check_output(get_jobs_cmd, shell=True, text=True)
  except subprocess.CalledProcessError as e:
    print(f"Failed to fetch jobs from cluster {cluster_config.cluster_name}: {e}")
    return 1

  jobs_to_cancel = [job.strip() for job in output.splitlines() if job.strip().startswith(first_three_chars)]

  final_return_code = 0
  for job in jobs_to_cancel:
    cancel_cmd = (
        f"gcluster job cancel {job} "
        f"--project={cluster_config.project} --cluster={cluster_config.cluster_name}"
        f" --location={location}"
    )
    print(f"Deleting workload: {job} using command: {cancel_cmd}")
    ret_code = os.system(cancel_cmd)
    if ret_code != 0:
      final_return_code = ret_code

  return final_return_code


def handle_delete_specific_workload(cluster_config: ClusterConfig, workload_name: str, **kwargs) -> int:
  """Handles the deletion of workloads with a specific name.

  Args:
      cluster_config: ClusterConfig object
      workload_name: workload name
      **kwargs: Optional keyword arguments (retained for backward compatibility)
  """
  del kwargs
  location = getattr(cluster_config, "location", None) or cluster_config.zone
  delete_command = (
      f"gcluster job cancel {workload_name} "
      f"--project={cluster_config.project} --cluster={cluster_config.cluster_name}"
      f" --location={location}"
  )
  print(f"Deleting workload: {workload_name} using command:" f" {delete_command}")
  return os.system(delete_command)


def handle_cmd_args(cluster_config: ClusterConfig, is_delete: bool = False, user: str | None = None, **kwargs) -> bool:
  """Parses command-line arguments and executes the specified actions.

  Args:
      cluster_config: Contains Cluster configuration information that's helpful
        for running the actions.
      is_delete: A boolean indicating whether the delete action should be
        performed.
      user: Optional user string whose 3-character prefix filters jobs to delete.
        Defaults to `os.environ.get("USER", "user")`.
      **kwargs: Optional keyword arguments to be passed to action handlers.
  """
  # Handle actions
  should_continue = True
  if is_delete:
    resolved_user = user or os.environ.get("USER", "user")
    _handle_delete(cluster_config, resolved_user, **kwargs)
    should_continue = False

  return should_continue
