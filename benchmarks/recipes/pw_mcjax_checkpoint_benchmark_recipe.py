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

"""Recipe to test checkpointing and restore for Pathways and McJax.
 The recipe will run synchronously and will run the benchmark steps for each
 model and infrastructure combination. If TEST_RESTORE is set to True, it will
 restore from the initial run for the given number of steps and then continue
 the run.
"""

import datetime
import dataclasses
import os
import args_helper as helper
from benchmarks.benchmark_utils import get_xpk_path

from benchmarks import maxtext_trillium_model_configs as model_configs
import benchmarks.maxtext_xpk_runner as mxr
from benchmarks.xpk_configs import XpkClusterConfig

PROXY_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server"
SERVER_IMAGE = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server"
RUNNER = "us-docker.pkg.dev/path/to/maxtext_runner"

# Cluster Params
CLUSTER = "v6e-256-cluster"
PROJECT = "tpu-prod-env-cluster"
ZONE = "us-east5-b"
COUNTRY = "us"
DEVICE_TYPE = "v6e-256"

# Other parameters (MUST BE SET BY USER)
XPK_PATH = get_xpk_path()
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = f"gs://{USER}-{PROJECT}-{COUNTRY}/pw_mcjax_benchmarking/"
# This needs to be set to True to test restore and if you want to restore from
# a previous run, you'll need to set RESUME_CHECKPOINT_NAMES below.
TEST_RESTORE = False
MAX_RESTARTS = 100

BENCHMARK_STEPS = 41
RESTORE_BENCHMARK_STEPS = 20  # Define steps for restore run

RESUME_CHECKPOINT_NAMES = {
    "pathways": {
        # Key is number of slices, value is a dictionary of run_name,
        # base_output_directory, and num_steps.
        32: {
            "run_name": "restoring_run_name",
            "base_output_directory": f"gs://{BASE_OUTPUT_DIRECTORY}/...",
            "num_steps": BENCHMARK_STEPS + RESTORE_BENCHMARK_STEPS,
        }
    },
    # "mcjax": {
    # 32: {}
    # },
}


def _get_xpk_commands(
    models,
    cluster_config,
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

  for infra, model_config in models.items():
    for model in model_config["models"]:
      # Run workloads
      config = cluster_config
      # Run workloads in the following slice configurations
      for num_slices in model_config["num_slices_list"]:
        # Ensure that we're checkpointing for the models.
        if model.pathways_tuning_params is None:
          model.pathways_tuning_params = model_configs.PATHWAYS_SHORT_RUN_CHECKPOINTING_TUNING_PARAMS

        run_name = None
        base_output_directory = (
            BASE_OUTPUT_DIRECTORY + f"{timestamp_str}_{infra}_{num_slices}_slice_{DEVICE_TYPE}_{model.model_name}/"
        )
        if infra in RESUME_CHECKPOINT_NAMES:
          if num_slices in RESUME_CHECKPOINT_NAMES[infra]:
            run_name = RESUME_CHECKPOINT_NAMES[infra][num_slices]["run_name"]
            base_output_directory = RESUME_CHECKPOINT_NAMES[infra][num_slices]["base_output_directory"]
            num_steps = RESUME_CHECKPOINT_NAMES[infra][num_slices]["num_steps"]

        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=config.device_type,
            base_output_directory=base_output_directory,
            max_restarts=MAX_RESTARTS,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image=RUNNER if infra == "mcjax" else None,
            pathways_config=pathways_config if infra == "pathways" else None,
            xpk_path=XPK_PATH,
            num_steps=num_steps,
            priority="medium",
            run_name=run_name,
        )
        command, name = mxr.generate_xpk_workload_cmd(cluster_config=config, wl_config=wl_config, workload_name=run_name)

        print(f"Name of the workload is: {name} \n")
        print(f"XPK command to be used is: {command} \n")
        xpk_workloads.append((name, command, wl_config))

  return xpk_workloads


def _run_workloads_async(xpk_workloads, cluster_config, run_type="Initial"):
  """Runs the given xpk workloads asynchronously and yields workload names as they complete.

  Args:
    xpk_workloads: A list of tuples, each containing workload name, command and
      wl_config.
    cluster_config: The XPK cluster configuration.
    run_type: String to indicate if it's "Initial" or "Restore" run for logging.

  Yields:
    tuple[workload_name, return_code, wl_config]: The name of the workload, its
      return code, and its workload config as it completes.
  """
  print(f"\nStarting {run_type} Workloads (Asynchronously)...\n")

  workload_configs_dict = {}  # Store wl_config by workload_name
  workload_names = []
  for workload_name, workload_cmd, wl_config in xpk_workloads:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Launching {run_type} workload: {workload_name} with" f" command: {workload_cmd}")
    return_code = mxr.run_command_with_updates(workload_cmd, workload_name)

    if return_code != 0:
      print(
          f"Warning: Unable to start {run_type} xpk workload:"
          f" {workload_name}. Creation command failed, but continuing to"
          " launch others."
      )
      continue
    workload_names.append(workload_name)
    workload_configs_dict[workload_name] = wl_config  # Store config

  # Wait for completion asynchronously and yield names and return codes as they
  # complete
  completed_workloads = mxr.wait_for_xpk_workloads_completion_async(cluster_config, workload_names, xpk_path=XPK_PATH)

  for completed_workload_name, return_code in completed_workloads:
    yield completed_workload_name, return_code, workload_configs_dict[completed_workload_name]


def main() -> int:
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
    return 0

  models = {
      "pathways": {
          "models": [
              model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
          ],
          "num_slices_list": [
              2,
          ],
      },
      "mcjax": {
          "models": [
              model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
          ],
          "num_slices_list": [
              2,
          ],
      },
  }
  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_server_image=PROXY_IMAGE,
      runner_image=RUNNER,
      # User can add additional flags here. We're adding StreamZ flags here.
      server_flags="--enable_metrics_collection=true",
      proxy_flags="--enable_metrics_collection=true",
      worker_flags="--enable_metrics_collection=true",
  )

  # --- Initial Run for Benchmark Steps ---
  print("\n--- Starting Initial Benchmark Run ---")
  xpk_workloads_initial = _get_xpk_commands(
      models,
      cluster_config,
      pathways_config,
      num_steps=BENCHMARK_STEPS,
  )

  completed_initial_workloads = {}
  initial_workload_configs = {}

  # Iterate through completed workloads as they yield
  for completed_workload_name, return_code, wl_config in _run_workloads_async(
      xpk_workloads_initial, cluster_config, run_type="Initial"
  ):
    print(f"\n--- Initial Workload '{completed_workload_name}' COMPLETED with" f" code: {return_code}. ---\n")
    if return_code == 0:
      completed_initial_workloads[completed_workload_name] = return_code  # Track completed workload names
      initial_workload_configs[completed_workload_name] = wl_config  # Store the config
    else:
      print(f"--- Workload '{completed_workload_name}' FAILED, NOT restoring. ---")

    if TEST_RESTORE and return_code == 0:
      # --- Restore Run for Additional Steps ---
      print(f"\n--- Starting Restore Run for '{completed_workload_name}' ---")

      # First delete the workload so we can restore it from scratch.
      helper.handle_delete_specific_workload(cluster_config, completed_workload_name, xpk_path=XPK_PATH)

      original_wl_config = initial_workload_configs[completed_workload_name]

      # Create new config for restore with the number of steps increased
      restore_wl_config = dataclasses.replace(
          original_wl_config,
          num_steps=original_wl_config.num_steps + RESTORE_BENCHMARK_STEPS,
      )

      restore_command, _ = mxr.generate_xpk_workload_cmd(
          cluster_config=cluster_config,
          wl_config=restore_wl_config,
          workload_name=completed_workload_name,
      )
      print(f"Restore command for '{completed_workload_name}': {restore_command}")

      print(f"Launching restore workload: {completed_workload_name}")
      restore_return_code = mxr.run_command_with_updates(restore_command, f"Restore {completed_workload_name}")
      if restore_return_code == 0:
        print(f"\n--- Restore Workload for '{completed_workload_name}' launched" " successfully. ---")
      else:
        print(f"\n--- Restore Workload for '{completed_workload_name}' FAILED to" " launch. ---")

  return 0


if __name__ == "__main__":
  main()
