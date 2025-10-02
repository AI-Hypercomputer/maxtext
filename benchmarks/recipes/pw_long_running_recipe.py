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

"""A recipe for running a long-running MaxText benchmark using Pathways.

This script is designed for stability and long-duration runs. It configures
and launches a workload on a GKE cluster using XPK, with a high number of
restarts enabled. It defines the cluster, Docker images, and model
configurations for a Pathways-based run.
"""

import datetime
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import recipes.args_helper as helper

import maxtext_trillium_model_configs as model_configs

import maxtext_xpk_runner as mxr

from xpk_configs import XpkClusterConfig

PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server"
SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"
RUNNER = "us-docker.pkg.dev/path/to/maxtext_runner"

# Cluster Params
CLUSTER = "v6e-256-cluster"
PROJECT = "tpu-prod-env-cluster"
ZONE = "us-east5-b"
REGION = "us-east5"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

# Other parameters (MUST BE SET BY USER)
XPK_PATH = os.path.join("~", "xpk")  # We're running this script from the maxtext directory
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = f"gs://{USER}-{PROJECT}-{COUNTRY}/pw_long_run/"

MAX_RESTARTS = 10_000
BENCHMARK_STEPS = 10_000_000


def main():
  # V6e cluster config
  cluster_config = XpkClusterConfig(
      cluster_name=CLUSTER,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(cluster_config, helper.DELETE, xpk_path=XPK_PATH)

  if not should_continue:
    return

  model_list = [
      # model_configs.llama3_1_70b_8192_pw_lr_real_data,
      # model_configs.llama3_1_8b_8192,
      # model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
      # model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds,
      model_configs.llama3_1_70b_8192_iter_synthetic,
  ]

  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_server_image=PROXY_IMAGE,
      runner_image=RUNNER,
      # User can add additional flags here.
      server_flags="--enable_metrics_collection=true",
      proxy_flags="--enable_metrics_collection=true",
      worker_flags="--enable_metrics_collection=true",
      # server_flags="--enable_metrics_collection=false",
      # proxy_flags="--enable_metrics_collection=false",
      # worker_flags="--enable_metrics_collection=false",
  )
  num_slices_list = [2]

  xpk_workload_cmds = []
  xpk_workload_names = []

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
      model.tuning_params["profiler"] = "xplane"

      # Run workloads in the following slice configurations
      for num_slices in num_slices_list:
        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=BASE_OUTPUT_DIRECTORY,
            max_restarts=MAX_RESTARTS,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image=None,
            pathways_config=pathways_config,
            xpk_path=XPK_PATH,
            num_steps=BENCHMARK_STEPS,
            priority="medium",
        )
        command, name = mxr.generate_xpk_workload_cmd(cluster_config=cluster_config, wl_config=wl_config)

        print(f"Name of the workload is: {name} \n")
        xpk_workload_names.append(name)

        print(f"XPK command to be used is: {command} \n")
        xpk_workload_cmds.append(command)

  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Running workload: {xpk_workload_name} with command:" f" {xpk_workload_cmd}")
    return_code = mxr.run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")


if __name__ == "__main__":
  main()
