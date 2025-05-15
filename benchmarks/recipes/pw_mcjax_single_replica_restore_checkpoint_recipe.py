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

"""Recipe to test checkpointing and restore for Pathways and McJax.
 The recipe will run synchronously and will run the benchmark steps for each
 model and infrastructure combination. If TEST_RESTORE is set to True, it will
 restore from the initial run for the given number of steps and then continue
 the run.
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
from user_configs import cluster_config, xpk_path, BASE_OUTPUT_DIRECTORY, server_image, proxy_image, runner, user, pathways_config, headless_workload_name, MAX_RESTARTS, BENCHMARK_STEPS, RESTORE_BENCHMARK_STEPS, RESUME_CHECKPOINT_NAMES, TEST_RESTORE


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
      # Run workloads on the below clusters
      for config in [
          cluster_config,
      ]:
        # Run workloads in the following slice configurations
        for num_slices in model_config["num_slices_list"]:
          # Ensure that we're checkpointing for the models.
          if model.pathways_tuning_params is None:
            model.pathways_tuning_params = model_configs.PATHWAYS_SHORT_RUN_CHECKPOINTING_TUNING_PARAMS

          # Enable single replica checkpointing for restore.
          model.tuning_params["enable_single_replica_ckpt_restoring"] = True

          run_name = None
          base_output_directory = (
              BASE_OUTPUT_DIRECTORY
              + f"{timestamp_str}_{infra}_{num_slices}_slice_{config.device_type}_{model.model_name}/"
          )
          if infra in RESUME_CHECKPOINT_NAMES:
            if num_slices in RESUME_CHECKPOINT_NAMES[infra]:
              run_name = RESUME_CHECKPOINT_NAMES[infra][num_slices][
                  "run_name"
              ]
              base_output_directory = RESUME_CHECKPOINT_NAMES[infra][
                  num_slices
              ]["base_output_directory"]
              num_steps = RESUME_CHECKPOINT_NAMES[infra][num_slices][
                  "num_steps"
              ]

          wl_config = mxr.WorkloadConfig(
              model=model,
              num_slices=num_slices,
              device_type=config.device_type,
              base_output_directory=base_output_directory,
              max_restarts=MAX_RESTARTS,
              libtpu_type=None,
              libtpu_nightly_version="",
              base_docker_image=runner if infra == "mcjax" else None,
              pathways_config=pathways_config if infra == "pathways" else None,
              xpk_path=xpk_path,
              num_steps=num_steps,
              priority="medium",
              run_name=run_name,
          )
          command, name = mxr.generate_xpk_workload_cmd(
              cluster_config=config, wl_config=wl_config, workload_name=run_name
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
      cluster_config, workload_names, xpk_path=xpk_path
  )

  for completed_workload_name, return_code in completed_workloads:
    yield completed_workload_name, return_code, workload_configs_dict[
        completed_workload_name
    ]


def main() -> int:
  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(
      cluster_config, helper.DELETE, xpk_path=xpk_path
  )

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
      # "mcjax": {
      #     "models": [
      #         model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing,
      #     ],
      #     "num_slices_list": [
      #         2,
      #     ],
      # },
  }

  # --- Initial Run for Benchmark Steps ---
  print(
      "\n--- Starting Initial Benchmark Run ---"
  )
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
          cluster_config, completed_workload_name, xpk_path=xpk_path
      )

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
