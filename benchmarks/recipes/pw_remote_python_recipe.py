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

"""A recipe for running a MaxText benchmark using Pathways with a remote Python sidecar.

This script configures and launches a workload on a GKE cluster using Cluster
Toolkit (gcluster). It defines the cluster, Docker images for the server, proxy,
and runner, and sets up the model configuration for a Pathways-based run.
"""

import os
import sys

import benchmarks.recipes.args_helper as helper

from benchmarks import maxtext_trillium_model_configs as model_configs
from benchmarks import maxtext_ctk_runner as mcr
from benchmarks.ctk_configs import ClusterConfig
from benchmarks.recipes import user_configs


def main():
  # V6e cluster config
  cluster_config = ClusterConfig(
      cluster_name="v6e-256-cluster",
      project="tpu-project",
      zone="us-east5-b",
      device_type="v6e-256",
  )

  # Handle command line arguments using args_helper
  is_delete = user_configs.USER_CONFIG.delete or ("--delete" in sys.argv)
  should_continue = helper.handle_cmd_args(cluster_config, is_delete, os.environ["USER"])

  if not should_continue:
    return

  # Configure test images
  user = os.environ["USER"]
  loc = cluster_config.location or cluster_config.zone
  loc_parts = loc.split("-")
  region = "-".join(loc_parts[:-1]) if len(loc_parts) >= 3 else loc
  proxy_image = f"us-docker.pkg.dev/cloud-tpu-v2-images/pathways/gke/{user}/" "proxy_server:latest"
  server_image = f"us-docker.pkg.dev/cloud-tpu-v2-images/pathways/gke/{user}/" "server:latest"
  colocated_python_image = f"gcr.io/{cluster_config.project}/{user}/colocated_python_sidecar_latest:latest"
  runner = f"gcr.io/{cluster_config.project}/{user}_latest:latest"
  base_output_directory = f"gs://{user}-{region}/{user}/"

  list_of_models = [
      model_configs.default_basic_1,
  ]
  pathways_config = mcr.PathwaysConfig(
      server_image=server_image,
      proxy_server_image=proxy_image,
      runner_image=runner,
      colocated_python_sidecar_image=colocated_python_image,
  )
  num_slices_list = [1]

  workload_cmds = []
  workload_names = []

  for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
        cluster_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in num_slices_list:
        wl_config = mcr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=base_output_directory,
            max_restarts=0,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image=None,
            pathways_config=pathways_config,
            num_steps=1000000,
        )
        command, name = mcr.generate_workload_cmd(cluster_config=cluster_config, wl_config=wl_config)

        print(f"Name of the workload is: {name} \n")
        workload_names.append(name)

        print(f"gcluster command to be used is: {command} \n")
        workload_cmds.append(command)

  for workload_name, workload_cmd in zip(workload_names, workload_cmds):
    return_code = mcr.run_command_with_updates(workload_cmd, workload_name)
    if return_code != 0:
      print(f"Unable to run gcluster workload: {workload_name}")


if __name__ == "__main__":
  main()
