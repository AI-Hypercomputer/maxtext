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
import dataclasses
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

# Cluster Params
CLUSTER = "v6e-256-cluster"
PROJECT = "tpu-prod-env-cluster"
ZONE = "us-east5-b"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

################################################################################

PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/unsanitized_proxy_server:latest"
SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/sujinesh/unsanitized_server:latest"
RUNNER = "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest"

CLUSTER = "bodaborg-v6e-16-debug"
PROJECT = "tpu-prod-env-one-vm"
ZONE = "us-east5-b"
COUNTRY = "us"
DEVICE_TYPE = "v6e-16"

CLUSTER = "bodaborg-v6e-256-ts"
PROJECT = "tpu-prod-env-multipod"
ZONE = "us-west1-c"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

################################################################################

# Other parameters (MUST BE SET BY USER)
XPK_PATH = "../xpk"  # We're running this script from the maxtext directory
USER = os.environ["USER"]
# BASE_OUTPUT_DIRECTORY = (
#     f"gs://{USER}-{PROJECT}-{COUNTRY}/pw_mcjax_benchmarking/"
# )
BASE_OUTPUT_DIRECTORY = (
    f"gs://trillium-scale-tests-q1-25-west/pw_mcjax_benchmarking/{USER}/"
)
TEST_RESTORE = True
MAX_RESTARTS = 10

BENCHMARK_STEPS = 101
RESTORE_BENCHMARK_STEPS = BENCHMARK_STEPS + 20  # Define steps for restore run


def _get_xpk_commands(
    models,
    cluster_config,
    num_slices_list,
    pathways_config,
    num_steps=BENCHMARK_STEPS,
):
  """Generates xpk commands for the given models and configurations.

  Args:
    models: A dictionary of model lists, keyed by infrastructure type.
    cluster_config: The cluster configuration.
    num_slices_list: A list of the number of slices to use for each workload.
    pathways_config: The pathways configuration.
    num_steps: Number of steps for the workload.

  Returns:
    A list of tuples, where each tuple contains the xpk workload name, the
    xpk command, and the workload config.
  """
  xpk_workloads = []

  current_time = datetime.datetime.now()
  timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")

  for infra, model_list in models.items():
    for model in model_list:
      # Run workloads on the below clusters
      for config in [
          cluster_config,
      ]:
        # Run workloads in the following slice configurations
        for num_slices in num_slices_list:
          # Ensure that we're checkpointing for the models.
          if model.pathways_tuning_params is None:
            tuning_params = (
                model_configs.PATHWAYS_SHORT_RUN_CHECKPOINTING_TUNING_PARAMS
            )
            # tuning_params["checkpoint_period"] = 5
            model.pathways_tuning_params = tuning_params

          wl_config = mxr.WorkloadConfig(
              model=model,
              num_slices=num_slices,
              device_type=config.device_type,
              base_output_directory=BASE_OUTPUT_DIRECTORY
              + f"{timestamp_str}_{infra}_{num_slices}_slice_{DEVICE_TYPE}_{model.model_name}/",
              max_restarts=MAX_RESTARTS,
              libtpu_type=None,
              libtpu_nightly_version="",
              base_docker_image=RUNNER if infra == "mcjax" else None,
              pathways_config=pathways_config if infra == "pathways" else None,
              xpk_path=XPK_PATH,
              num_steps=num_steps,
              # priority="low",
          )
          command, name = mxr.generate_xpk_workload_cmd(
              cluster_config=config, wl_config=wl_config
          )

          print(f"Name of the workload is: {name} \n")
          print(f"XPK command to be used is: {command} \n")
          xpk_workloads.append(
              (name, command, wl_config)
          )

  return xpk_workloads


def _run_workloads_async(xpk_workloads, cluster_config, run_type="Initial"):
  """Runs the given xpk workloads asynchronously and yields workload names as they complete.

  Args:
    xpk_workloads: A list of tuples, each containing workload name, command and
      wl_config.
    cluster_config: The XPK cluster configuration.
    run_type: String to indicate if it's "Initial" or "Restore" run for logging.

  Yields:
    Tuple[workload_name, return_code, wl_config]: The name of the workload, its
      return code, and its workload config as it completes.
  """
  print(f"\nStarting {run_type} Workloads (Asynchronously)...\n")

  workload_configs_dict = {}  # Store wl_config by workload_name
  workload_names = []
  for workload_name, workload_cmd, wl_config in xpk_workloads:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] Launching {run_type} workload: {workload_name} with"
        f" command: {workload_cmd}"
    )
    return_code = mxr.run_command_with_updates(
        workload_cmd, workload_name
    )

    if return_code != 0:
      print(
          f"Warning: Unable to start {run_type} xpk workload:"
          f" {workload_name}. Creation command failed, but continuing to"
          " launch others."
      )
      continue
    workload_names.append(workload_name)
    workload_configs_dict[workload_name] = wl_config # Store config

  # Wait for completion asynchronously and yield names and return codes as they
  # complete
  completed_workloads = mxr.wait_for_xpk_workloads_completion_async(
      cluster_config, workload_names, xpk_path=XPK_PATH
  )

  for completed_workload_name, return_code in completed_workloads:
    yield completed_workload_name, return_code, workload_configs_dict[
        completed_workload_name
    ]


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

  models = {
      "pathways": [
          model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
          # synthetic data
          # model_configs.llama3_1_70b_8192,

          # model_configs.llama3_1_8b_8192,
          # model_configs.llama3_1_70b_8192,
          # model_configs.llama3_1_405b_8192_fsdp_dcn,
          # model_configs.llama2_70b_4096_real_data_long_run,
      ],
      "mcjax": [
          model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
          # model_configs.llama3_1_8b_8192,
          # model_configs.llama3_1_70b_8192,
          # model_configs.llama3_1_405b_8192_fsdp_dcn,
          # model_configs.llama2_70b_4096_real_data_long_run,
      ],
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
      10, 16, 24, 32, 40, 46
  ]

  # --- Initial Run for Benchmark Steps ---
  print(
      "\n--- Starting Initial Benchmark Run ---"
  )  # Added clear log separation
  xpk_workloads_initial = _get_xpk_commands(
      models,
      cluster_config,
      num_slices_list,
      pathways_config,
      num_steps=BENCHMARK_STEPS,
  )

  completed_initial_workloads = {}
  initial_workload_configs = {}

  for completed_workload_name, return_code, wl_config in _run_workloads_async(
      xpk_workloads_initial, cluster_config, run_type="Initial"
  ):  # Iterate through completed workloads as they yield
    print(
        f"\n--- Initial Workload '{completed_workload_name}' COMPLETED with"
        f" code: {return_code}. ---\n"
    )
    if return_code == 0:
      completed_initial_workloads[completed_workload_name] = (
          return_code  # Track completed workload names
      )
      initial_workload_configs[completed_workload_name] = (
          wl_config  # Store the config
      )
    else:
      print(
          f"--- Workload '{completed_workload_name}' FAILED, NOT restoring. ---"
      )

    if TEST_RESTORE and return_code == 0:
      # --- Restore Run for Additional Steps ---
      print(f"\n--- Starting Restore Run for '{completed_workload_name}' ---")

      # First delete the workload so we can restore it from scratch.
      helper.handle_delete_specific_workload(
          cluster_config, completed_workload_name, xpk_path=XPK_PATH
      )

      original_wl_config = initial_workload_configs[completed_workload_name]

      # Create new config for restore with the number of steps increased
      restore_wl_config = dataclasses.replace(
          original_wl_config, num_steps=RESTORE_BENCHMARK_STEPS)

      restore_command, _ = mxr.generate_xpk_workload_cmd(
          cluster_config=cluster_config,
          wl_config=restore_wl_config,
          workload_name=completed_workload_name,
      )
      print(
          f"Restore command for '{completed_workload_name}': {restore_command}"
      )

      print(f"Launching restore workload: {completed_workload_name}")
      restore_return_code = mxr.run_command_with_updates(
          restore_command, f"Restore {completed_workload_name}"
      )
      if restore_return_code == 0:
        print(
            f"\n--- Restore Workload for '{completed_workload_name}' launched"
            " successfully. ---"
        )
      else:
        print(
            f"\n--- Restore Workload for '{completed_workload_name}' FAILED to"
            " launch. ---"
        )

  return 0


if __name__ == "__main__":
  main()
