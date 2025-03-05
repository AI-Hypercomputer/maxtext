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
import args_helper as helper

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import maxtext_trillium_model_configs as model_configs
import maxtext_xpk_runner as mxr

PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest"
SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest"
# RUNNER = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest"
# RUNNER = "gcr.io/cloud-tpu-multipod-dev/lukebaumann_runner:latest"
RUNNER = "gcr.io/cloud-tpu-v2-images-dev/shauryag_latest:elastic"

# Cluster Params
# CLUSTER = "v6e-256-cluster"
# PROJECT = "tpu-prod-env-cluster"
# ZONE = "us-east5-b"
# COUNTRY = "us"
# DEVICE_TYPE = "v6e-256"

# # 4 slices
# CLUSTER = "bodaborg-v6e-16-debug"
# PROJECT = "tpu-prod-env-one-vm"
# ZONE = "us-east5-b"
# COUNTRY = "us"
# DEVICE_TYPE = "v6e-16"

# 3 slices
# CLUSTER = "bodaborg-v6e-256-dnd-yucmhab"
# PROJECT = "tpu-prod-env-one-vm"
# ZONE = "us-east5-b"
# COUNTRY = "us"
# DEVICE_TYPE = "v6e-256"

# # Many slices
# CLUSTER = "bodaborg-v6e-256-ts"
# PROJECT = "tpu-prod-env-multipod"
# ZONE = "us-west1-c"
# COUNTRY = "us"
# DEVICE_TYPE = "v6e-256"
CLUSTER = "bodaborg-v6e-256-ts"
PROJECT = "tpu-prod-env-multipod"
ZONE = "us-west1-c"
REGION = "us-west1"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

# Other parameters (MUST BE SET BY USER)
XPK_PATH = "../xpk"  # We're running this script from the maxtext directory
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = (
#    f"gs://akshu-v6e-m/shauryag/pw_mcjax_benchmarking/"
    f"/tmp/gcsfuse/shauryag/pw_mcjax_benchmarking/"
)

BENCHMARK_STEPS=2000


def main() -> int:
  # V6e cluster config
  cluster_config = mxr.XpkClusterConfig(
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

  model_list = [
      # model_configs.llama3_1_70b_8192_lr_real_data,
      # model_configs.llama3_1_8b_8192,
      # model_configs.llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds
      model_configs.llama3_1_70b_8192_iter_real_data_and_sync_checkpointing_tfds
      # model_configs.llama3_1_70b_8192_iter_real_data_tfds
  ]
  num_slices_list = [
      48
  ]
  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_server_image=PROXY_IMAGE,
      runner_image=RUNNER,

      # User can add additional flags here.
      server_flags="--temporary_flags_for_debugging=temporary_flag_for_debugging_enable_late_binding=false",
      proxy_flags=f"--temporary_flags_for_debugging=temporary_flag_for_debugging_experimental_elastic_slices=true;;;temporary_flag_for_debugging_experimental_num_ok_missing_slices={num_slices_list[0]}",
      worker_flags="",
  )

  xpk_workload_cmds = []
  xpk_workload_names = []

  for model in model_list:
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
            base_output_directory=BASE_OUTPUT_DIRECTORY,
            executable="MaxText/elastic_train.py",
            max_restarts=10_000,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image=None,
            pathways_config=pathways_config,
            xpk_path=XPK_PATH,
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
    print(f"[{timestamp}] Running workload: {xpk_workload_name} with command: {xpk_workload_cmd}")
    return_code = mxr.run_command_with_updates(
        xpk_workload_cmd, xpk_workload_name
    )
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")


if __name__ == "__main__":
  main()
