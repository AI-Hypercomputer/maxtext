"""
 Copyright 2025 Google LLC

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

import datetime
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import recipes.args_helper as helper
import maxtext_v5e_model_configs as model_configs
import maxtext_xpk_runner as mxr
import recipes.user_configs as ucfg

MAX_RESTARTS = 1
BENCHMARK_STEPS = 20


def main() -> int:
  # DELETED - Cluster config is now loaded from ucfg
  # cluster_config = XpkClusterConfig(...)

  # Handle command line arguments using args_helper
  # CHANGED - Use configs from ucfg
  should_continue = helper.handle_cmd_args(
      ucfg.cluster_config, helper.DELETE, xpk_path=ucfg.xpk_path
  )

  if not should_continue:
    return 0

  model_list = [
      # model_configs.llama3_1_70b_8192_pw_lr_real_data,
      # model_configs.llama3_1_8b_8192,
      model_configs.llama3_1_8b_8192_v5e_256_real_data,
      # model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
      # model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds,
      # model_configs.llama3_1_70b_8192_iter_synthetic,
  ]

  num_slices_list = [
      2
  ]

  xpk_workload_cmds = []
  xpk_workload_names = []

  for model in model_list:
    # Run workloads on the below clusters
    # CHANGED - Loop now uses the imported cluster_config
    for cluster_config in [
        ucfg.cluster_config,
    ]:

      # Make modifications to the model config here to add in any additional
      # flags or changes to the model config.
      model.tuning_params["use_vertex_tensorboard"] = True
      # CHANGED - Use project from ucfg
      model.tuning_params["vertex_tensorboard_project"] = ucfg.project
      # The region is derived from the zone in ucfg, so we calculate it here
      region = "-".join(ucfg.cluster_config.zone.split("-")[:-1])
      model.tuning_params["vertex_tensorboard_region"] = region
      model.tuning_params["profiler"] = "xplane"

      # Run workloads in the following slice configurations
      for num_slices in num_slices_list:
        # CHANGED - Use configs from ucfg for WorkloadConfig
        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=ucfg.base_output_directory,
            max_restarts=MAX_RESTARTS,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image=None,
            pathways_config=ucfg.pathways_config,
            xpk_path=ucfg.xpk_path,
            num_steps=BENCHMARK_STEPS,
            priority="medium",
        )
        command, name = mxr.generate_xpk_workload_cmd(
            cluster_config=cluster_config, wl_config=wl_config
        )

        print(f"Name of the workload is: {name} \n")
        xpk_workload_names.append(name)

        print(f"XPK command to be used is: {command} \n")
        xpk_workload_cmds.append(command)

  for xpk_workload_name, xpk_workload_cmd in zip(
      xpk_workload_names, xpk_workload_cmds
  ):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] Running workload: {xpk_workload_name} with command:"
        f" {xpk_workload_cmd}"
    )
    return_code = mxr.run_command_with_updates(
        xpk_workload_cmd, xpk_workload_name
    )
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")


if __name__ == "__main__":
  main()
