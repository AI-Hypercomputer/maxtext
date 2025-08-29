"""Utilities for generating and running XPK workloads."""

import datetime
import logging
import maxtext_xpk_runner as mxr
import maxtext_trillium_model_configs as model_configs


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_and_run_workloads(user_config, num_slices_list, num_steps, priority="medium"):
  """
  Generates and executes XPK workloads based on the given configuration.

  Args:
    user_config: A UserConfig object containing all necessary configurations.
    models: A dictionary where keys are "mcjax" or "pathways" and values are lists of model settings.
    num_slices_list: A list of the number of slices to be executed.
    num_steps: The number of steps for each workload.
  """
  xpk_workload_cmds = []
  xpk_workload_names = []

  for infra, model_list in user_config.models.items():
    if not model_list:
      logging.info(f"Skipping empty model list for infrastructure: {infra}")
      continue

    for model in model_list:
      for num_slices in num_slices_list:
        # Create a WorkloadConfig object
        wl_config = mxr.WorkloadConfig(
            model=model,
            num_slices=num_slices,
            device_type=user_config.cluster_config.device_type,
            base_output_directory=(
                f"{user_config.base_output_directory}{infra}_{num_slices}_slice_"
                f"{user_config.device_type}_{model.model_name}/"
            ),                    
            max_restarts=0,
            libtpu_type=None,
            libtpu_nightly_version="",
            base_docker_image=user_config.runner if infra == "mcjax" else None,
            pathways_config=user_config.pathways_config if infra == "pathways" else None,
            xpk_path=user_config.xpk_path,
            num_steps=num_steps,
            priority=priority,
        )

        # Generate XPK command
        command, name = mxr.generate_xpk_workload_cmd(
            cluster_config=user_config.cluster_config, wl_config=wl_config
        )

        logging.info(f"Generated workload: {name}")
        logging.info(f"Generated command: {command}")
        xpk_workload_names.append(name)
        xpk_workload_cmds.append(command)

  # Execute all generated workloads
  for xpk_workload_name, xpk_workload_cmd in zip(xpk_workload_names, xpk_workload_cmds):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[{timestamp}] Running workload: {xpk_workload_name}")
    return_code = mxr.run_command_with_updates(xpk_workload_cmd, xpk_workload_name)
    if return_code != 0:
      logging.error(f"Failed to run xpk workload: {xpk_workload_name}. Return code: {return_code}")