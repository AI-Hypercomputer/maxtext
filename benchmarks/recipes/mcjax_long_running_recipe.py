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

"""A recipe for running a long-running MaxText benchmark using McJAX.

This script is designed for stability and long-duration runs. It configures
and launches a workload on a GKE cluster using Cluster Toolkit (gcluster), with
a high number of restarts enabled. It defines the cluster, Docker image, and
model configurations for a McJAX-based run.
"""

import datetime
import os
import sys

import benchmarks.recipes.args_helper as helper
import benchmarks.maxtext_trillium_model_configs as model_configs
import benchmarks.maxtext_ctk_runner as mcr
from benchmarks.ctk_configs import ClusterConfig
from benchmarks.recipes import user_configs

# Cluster Params
CLUSTER = "v6e-256-cluster"
PROJECT = "tpu-prod-env-cluster"
ZONE = "us-east5-b"
REGION = "us-east5"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

# Other parameters (MUST BE SET BY USER)
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = f"gs://{USER}-{PROJECT}-{COUNTRY}/mcjax_long_run/"
# Generate your own runner image from MaxText repo.
RUNNER = f"gcr.io/{PROJECT}/{USER}_latest"

MAX_RESTARTS = 10_000
BENCHMARK_STEPS = 10_000_000


def main() -> None:
  # V6e cluster config
  cluster_config = ClusterConfig(
      cluster_name=CLUSTER,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  # Handle command line arguments using args_helper
  is_delete = user_configs.USER_CONFIG.delete or ("--delete" in sys.argv)
  should_continue = helper.handle_cmd_args(cluster_config, is_delete, USER)

  if not should_continue:
    return

  model_list = [
      # model_configs.llama3_1_70b_8192_pw_lr_real_data,
      # model_configs.llama3_1_8b_8192,
      model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
      # model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds,
  ]
  num_slices_list = [2]

  workload_cmds = []
  workload_names = []

  for model in model_list:
    # Run workloads on the below clusters
    for cluster_config in [
        cluster_config,
    ]:

      # Make modifications to the model config here to add in any additional
      # flags or changes to the model config.
      model.tuning_params["use_vertex_tensorboard"] = True
      model.tuning_params["vertex_tensorboard_project"] = PROJECT
      model.tuning_params["vertex_tensorboard_region"] = REGION

      # Run workloads in the following slice configurations
      for num_slices in num_slices_list:
        wl_config = mcr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=BASE_OUTPUT_DIRECTORY,
            max_restarts=MAX_RESTARTS,
            libtpu_type=mcr.LibTpuType.MAXTEXT,
            libtpu_nightly_version="",
            base_docker_image=RUNNER,
            num_steps=BENCHMARK_STEPS,
            priority="medium",
        )
        command, name = mcr.generate_workload_cmd(cluster_config=cluster_config, wl_config=wl_config)

        print(f"Name of the workload is: {name} \n")
        workload_names.append(name)

        print(f"gcluster command to be used is: {command} \n")
        workload_cmds.append(command)

  for workload_name, workload_cmd in zip(workload_names, workload_cmds):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Running workload: {workload_name} with command: {workload_cmd}")
    return_code = mcr.run_command_with_updates(workload_cmd, workload_name)
    if return_code != 0:
      print(f"Unable to run gcluster workload: {workload_name}")


if __name__ == "__main__":
  main()
