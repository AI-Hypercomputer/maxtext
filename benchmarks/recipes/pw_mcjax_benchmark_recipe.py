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

import sys
import os
import args_helper as helper

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import maxtext_trillium_model_configs as model_configs
import maxtext_xpk_runner as mxr

PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest"
SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest"
RUNNER = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest"

CLUSTER = "v6e-256-cluster"
PROJECT = "tpu-prod-env-cluster"
ZONE = "us-east5-b"
REGION = "us-east5"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

BENCHMARK_STEPS = 20


def main() -> int:
  # V6e cluster config
  cluster_config = mxr.XpkClusterConfig(
      cluster_name=CLUSTER,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )
  xpk_path = "../xpk"  # We're running this script from the maxtext directory

  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(
      cluster_config, helper.DELETE, xpk_path=xpk_path
  )

  if not should_continue:
    return 0

  user = os.environ["USER"]
  base_output_directory = f"gs://{user}-{PROJECT}-{COUNTRY}/pw_mcjax_benchmarking/"

  models = {
      # "mcjax": [
          # model_configs.llama3_1_70b_8192,  # works
      #     # model_configs.llama3_1_405b_8192_fsdp_dcn,  # works
      #     model_configs.llama2_70b_4096_real_data_long_run,  # works
      # ],
      "pathways": [
          # model_configs.llama3_1_70b_8192_pw,  # fails with a hang
          # model_configs.llama3_1_405b_8192_fsdp_dcn_pw,  # fails with a data loss error
          # model_configs.llama2_70b_4096_real_data_pw_long_run,  # Has a data loss error
          model_configs.llama3_1_8b_8192_pw,
      ]
  }
  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_image=PROXY_IMAGE,
      runner_image=RUNNER,
      server_flags="--temporary_flags_for_debugging=temporary_flag_for_debugging_worker_expected_tpu_chip_config=megachip_tccontrol --xla_tpu_use_enhanced_launch_barrier",
      # proxy_flags="--megascale_grpc_enable_xor_tracer=false --xla_tpu_use_enhanced_launch_barrier",
      # worker_flags="--megascale_grpc_enable_xor_tracer=false --xla_tpu_use_enhanced_launch_barrier"
      proxy_flags="--xla_tpu_use_enhanced_launch_barrier",
      worker_flags="--xla_tpu_use_enhanced_launch_barrier"
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
              base_docker_image=RUNNER if infra == "mcjax" else None,
              pathways_config=pathways_config if infra == "pathways" else None,
              xpk_path=xpk_path,
              num_steps=BENCHMARK_STEPS,
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
    print(f"Running workload: {xpk_workload_name} with command: {xpk_workload_cmd}")
    return_code = mxr.run_command_with_updates(
        xpk_workload_cmd, xpk_workload_name
    )
    if return_code != 0:
      print(f"Unable to run xpk workload: {xpk_workload_name}")


if __name__ == "__main__":
  main()
