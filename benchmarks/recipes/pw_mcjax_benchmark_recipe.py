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

"""Used to perf benchmarks between Pathways and McJax."""

import datetime
import os
import args_helper as helper

import benchmarks.src/MaxText_trillium_model_configs as model_configs
import benchmarks.src/MaxText_xpk_runner as mxr
from benchmarks.xpk_configs import XpkClusterConfig


PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server"
SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"
RUNNER = "us-docker.pkg.dev/path/to/src/MaxText_runner"

# Cluster Params
CLUSTER = "v6e-256-cluster"
PROJECT = "tpu-prod-env-cluster"
ZONE = "us-east5-b"
REGION = "us-east5"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

# Other parameters (MUST BE SET BY USER)
XPK_PATH = os.path.join("~", "xpk")  # We're running this script from the src/MaxText directory
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = (
    f"gs://{USER}-{PROJECT}-{COUNTRY}/pw_mcjax_benchmarking/"
)

BENCHMARK_STEPS = 20


def main() -> int:
  # V6e cluster config
  cluster_config = XpkClusterConfig(
      cluster_name=CLUSTER,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(
      cluster_config, helper.DELETE, xpk_path=XPK_PATH
  )

  if not should_continue:
    return 0

  models = {
      "mcjax": [
          # model_configs.llama3_1_8b_8192,
          # model_configs.llama3_1_70b_8192,
          # model_configs.llama3_1_405b_8192_fsdp_dcn,
          # model_configs.llama2_70b_4096_real_data_long_run,
      ],
      "pathways": [
          model_configs.llama3_1_8b_8192,
          # model_configs.llama3_1_70b_8192,
          # model_configs.llama3_1_405b_8192_fsdp_dcn,
          # model_configs.llama2_70b_4096_real_data_long_run,
      ]
  }
  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_server_image=PROXY_IMAGE,
      runner_image=RUNNER,

      # User can add additional flags here.
      server_flags="",
      proxy_flags="",
      worker_flags="",
  )
  num_slices_list = [
      2
  ]

  xpk_workload_cmds = []
  xpk_workload_names = []

  for infra, model_list in models.items():
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
          wl_config = mxr.WorkloadConfig(
              model=model,
              num_slices=num_slices,
              device_type=cluster_config.device_type,
              base_output_directory=BASE_OUTPUT_DIRECTORY
              + f"{infra}_{num_slices}_slice_{DEVICE_TYPE}_{model.model_name}/",
              max_restarts=0,
              libtpu_type=None,
              libtpu_nightly_version="",
              base_docker_image=RUNNER if infra == "mcjax" else None,
              pathways_config=pathways_config if infra == "pathways" else None,
              xpk_path=XPK_PATH,
              num_steps=BENCHMARK_STEPS,
              priority="low",
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
    print(f"[{timestamp}] Running workload: {xpk_workload_name} with command: {xpk_workload_cmd}")
    return_code = mxr.run_command_with_updates(
        xpk_workload_cmd, xpk_workload_name
    )
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")


if __name__ == "__main__":
  main()
