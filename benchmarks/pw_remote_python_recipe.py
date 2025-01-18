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

import argparse
import os
import sys

import maxtext_trillium_model_configs as model_configs
import maxtext_xpk_runner as mxr


def main() -> int:
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--delete",
      action="store_true",
      help=(
          "Only delete workloads starting with the user's first five"
          " characters."
      ),
  )
  args = parser.parse_args()
  user = os.environ["USER"]

  # V6e cluster config
  # cluster_config = mxr.XpkClusterConfig(
  #     cluster_name="bodaborg-v6e-256-dnd-yucmhab",
  #     project="tpu-prod-env-one-vm",
  #     zone="us-east5-b",
  #     device_type="v6e-256",
  # )

  # V5e cluster config
  cluster_config = mxr.XpkClusterConfig(
      cluster_name="sujinesh-in-memory-test-cluster",
      project="cloud-tpu-multipod-dev",
      zone="us-west1-c",
      device_type="v5litepod-16",
  )

  region = "-".join(cluster_config.zone.split("-")[:-1])

  if args.delete:
    # Delete workloads starting with the first 5 characters of the user's name.
    first_five_chars = user[:5]
    delete_command = (
        f"python3 xpk/xpk.py workload delete "
        f" --project={cluster_config.project} --cluster={cluster_config.cluster_name}"
        f" --filter-by-job={first_five_chars}"
    )

    print(f"Deleting workloads starting with: {first_five_chars}")
    os.system(delete_command)
    print("Deletion initiated. Exiting.")
    return 0

  # If --delete is not passed, proceed with creating and running workloads:
  proxy_image = (
      f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/{user}/"
      "proxy_server:latest"
  )
  server_image = (
      f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/{user}/"
      "server:latest"
  )
  runner = f"gcr.io/{cluster_config.project}/{user}_latest:latest"
  remote_python_image = (
      f"gcr.io/{cluster_config.project}/{user}/remote_python_sidecar_latest:latest"
  )

  pathways_config = mxr.PathwaysConfig(
      server_image=server_image,
      proxy_image=proxy_image,
      runner_image=runner,
      remote_python_sidecar_image=remote_python_image,
  )

  base_output_directory = f"gs://{user}-{region}/{user}"

  list_of_models = [model_configs.llama2_70b_4096_pw_long_run]

  xpk_workload_cmds = []
  xpk_workload_names = []

  for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
        cluster_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in [
          1,
      ]:
        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=base_output_directory,
            max_restarts=0,
            libtpu_type=None,  # Pathways doesn't use libtpu.
            libtpu_nightly_version="",
            base_docker_image="",  # Pathways doesn't use base docker image
            pathways_config=pathways_config,
            xpk_path="xpk",
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
    return_code = mxr.run_command_with_updates(
        xpk_workload_cmd, xpk_workload_name
    )
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")


if __name__ == "__main__":
  main()
