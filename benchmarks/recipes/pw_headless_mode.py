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
This recipe demonstrates how to launch a Pathways workload in headless mode.

In headless mode, the Cluster Toolkit (gcluster) workload starts the Pathways
server and proxy components but does not run a user command. This is useful for
setting up a persistent training environment that can be connected to later by a
separate runner process.
"""

import dataclasses
import sys

import benchmarks.recipes.args_helper as helper
from benchmarks import maxtext_ctk_runner as mcr
from benchmarks.recipes.user_configs import USER_CONFIG


def main() -> int:
  # Handle command line arguments using args_helper
  is_delete = USER_CONFIG.delete or ("--delete" in sys.argv)
  should_continue = helper.handle_cmd_args(
      USER_CONFIG.cluster_config,
      is_delete,
      user=USER_CONFIG.user,
  )

  if not should_continue:
    return 0

  num_slices = 2

  # Run workloads in the following slice configurations
  wl_config = mcr.WorkloadConfig(
      model=None,
      num_slices=num_slices,
      device_type=USER_CONFIG.cluster_config.device_type,
      base_output_directory=USER_CONFIG.base_output_directory,
      max_restarts=0,
      libtpu_type=None,
      libtpu_nightly_version="",
      base_docker_image=None,
      pathways_config=dataclasses.replace(USER_CONFIG.pathways_config, headless=True),
  )
  command, name = mcr.generate_workload_cmd(
      cluster_config=USER_CONFIG.cluster_config,
      wl_config=wl_config,
      workload_name=USER_CONFIG.headless_workload_name,
  )

  print(f"Name of the workload is: {name} \n")
  print(f"gcluster command to be used is: {command} \n")

  return_code = mcr.run_command_with_updates(command, name)
  if return_code != 0:
    print(f"Unable to run gcluster workload: {name}")

  return return_code


if __name__ == "__main__":
  main()
