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

import os
import sys
import benchmarks.recipes.args_helper as helper
import benchmarks.recipes.user_configs as ucfg

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import maxtext_xpk_runner as mxr


def main() -> int:
  # Handle command line arguments using configurations from user_configs.py
  should_continue = helper.handle_cmd_args(
      ucfg.cluster_config, helper.DELETE, xpk_path=ucfg.xpk_path
  )

  if not should_continue:
    return 0

  xpk_workload_cmds = []
  xpk_workload_names = []

  # Add checkpointing every 5 steps for all models
  for model in ucfg.list_of_models:
    model.tuning_params['checkpoint_period'] = 5

  # Iterate through models, and slice configurations defined in user_configs.py
  for model in ucfg.list_of_models:
    # The cluster is also sourced from the config file.
    # The list structure is kept to allow for easy expansion if you add more clusters.
    for cluster_config in [ucfg.cluster_config]:
      for num_slices in ucfg.num_slices_list:
        # Create workload config by merging imported configurations and defaults
        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=ucfg.base_output_directory,
            pathways_config=ucfg.pathways_config,
            xpk_path=ucfg.xpk_path,
            **ucfg.workload_config_defaults,  # Unpack default workload settings
        )
        command, name = mxr.generate_xpk_workload_cmd(
            cluster_config=cluster_config, wl_config=wl_config
        )

        print(f"Name of the workload is: {name} \n")
        xpk_workload_names.append(name)

        print(f"XPK command to be used is: {command} \n")
        xpk_workload_cmds.append(command)

  # Run all the generated workloads in sequence
  for xpk_workload_name, xpk_workload_cmd in zip(
      xpk_workload_names, xpk_workload_cmds
  ):
    return_code = mxr.run_command_with_updates(
        xpk_workload_cmd, xpk_workload_name
    )
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")

  return 0  # Explicitly return 0 on successful completion of the script


if __name__ == "__main__":
  main()
