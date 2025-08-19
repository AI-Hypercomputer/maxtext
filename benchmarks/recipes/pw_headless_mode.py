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

import os
import sys

import benchmarks.recipes.args_helper as helper
import maxtext_xpk_runner as mxr
from recipes.user_configs import cluster_config, xpk_path, pathways_config, base_output_directory, headless_workload_name


def main() -> int:
  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(
      cluster_config, helper.DELETE, xpk_path=xpk_path
  )

  if not should_continue:
    return 0

  num_slices = 2

  # Run workloads in the following slice configurations
  wl_config = mxr.WorkloadConfig(
      model=None,
      num_slices=num_slices,
      device_type=cluster_config.device_type,
      base_output_directory=base_output_directory,
      max_restarts=0,
      libtpu_type=None,
      libtpu_nightly_version="",
      base_docker_image="",
      pathways_config=pathways_config,
      xpk_path=xpk_path,
  )
  command, name = mxr.generate_xpk_workload_cmd(
      cluster_config=cluster_config,
      wl_config=wl_config,
      workload_name=headless_workload_name,
  )

  print(f"Name of the workload is: {name} \n")
  print(f"XPK command to be used is: {command} \n")

  return_code = mxr.run_command_with_updates(command, name)
  if return_code != 0:
    print(f"Unable to run xpk workload: {name}")


if __name__ == "__main__":
  main()
