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
import random
import string
import sys
import args_helper as helper

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"parent_dir: {parent_dir}")
parent_parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
print(f"parent_parent_dir: {parent_parent_dir}")

sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

import maxtext_trillium_model_configs as model_configs
import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig


def main() -> int:
  # V6e cluster config
  cluster_config = XpkClusterConfig(
      cluster_name="bodaborg-v6e-256-tt-c",
      project="tpu-prod-env-multipod",
      zone="us-west1-c",
      device_type="v6e-256",
      # cluster_name="pw-scale-test-v5e-32",
      # zone="us-south1-a",
      # project="cloud-tpu-multipod-dev",
      # device_type="v5litepod-32",
  )

  xpk_path = "/google/src/cloud/ksadi/remote-python-par-approach/google3/xpk"

  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(
      cluster_config, helper.DELETE, xpk_path=xpk_path
  )

  if not should_continue:
    return 0

  # Configure test images
  user = os.environ["USER"]
  region = "-".join(cluster_config.zone.split("-")[:-1])
  proxy_image = f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server:latest"
  server_image = f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_server:latest"
  colocated_python_image = f"us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/remote_python_sidecar_server:latest"
  runner = f"gcr.io/cloud-tpu-multipod-dev/{user}_runner:latest"
  # base_output_directory = f"gs://{user}-{region}/{user}"
  base_output_directory = "gs://trillium-scale-tests-q1-25-west/ksadi/"

  list_of_models = [
      # model_configs.default_basic_1,
      # model_configs.default_basic_1_colocated_python,
      # model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds_remote_python,
      model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds_colocated_python_elastic_train,
      # model_configs.default_basic_1_colocated_python_elastic_train,
  ]
  pathways_config = mxr.PathwaysConfig(
      server_image=server_image,
      proxy_server_image=proxy_image,
      runner_image=runner,
      colocated_python_sidecar_image=colocated_python_image,
  )
  num_slices_list = [1]
  # num_slices_list = [1, 2, 4, 8, 16, 32]

  xpk_workload_cmds = []
  xpk_workload_names = []

  for model in list_of_models:
    # Run workloads on the below clusters
    for cluster_config in [
        cluster_config,
    ]:
      # Run workloads in the following slice configurations
      for num_slices in num_slices_list:
        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=cluster_config.device_type,
            base_output_directory=base_output_directory,
            max_restarts=0,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image="",
            pathways_config=pathways_config,
            xpk_path=xpk_path,
            num_steps=1000000,
        )
        random_4_digit_suffix = "".join(
            random.choice(string.ascii_lowercase + string.digits)
            for _ in range(4)
        )
        command, name = mxr.generate_xpk_workload_cmd(
            cluster_config=cluster_config,
            wl_config=wl_config,
            workload_name=f"ksadi-col-py-{random_4_digit_suffix}",
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
