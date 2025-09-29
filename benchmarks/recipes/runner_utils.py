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

"""Utilities for generating and running XPK workloads."""

import logging

from .. import maxtext_xpk_runner as mxr
from benchmarks.benchmark_utils import Framework
from benchmarks.disruption_management.disruption_manager import construct_disruption_configs


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _create_workload_config(
    framework: str, model, num_slices: int, user_config, num_steps: int, priority: str, **kwargs
) -> mxr.WorkloadConfig:
  """Creates a single, unified WorkloadConfig object."""
  config_args = {
      "model": model,
      "num_slices": num_slices,
      "device_type": user_config.cluster_config.device_type,
      "base_output_directory": (
          f"{user_config.base_output_directory}{framework}_{num_slices}_slice_"
          f"{user_config.device_type}_{model.model_name}/"
      ),
      "max_restarts": user_config.max_restarts,
      "libtpu_type": None,
      "libtpu_nightly_version": "",
      "base_docker_image": (user_config.runner if Framework(framework) == Framework.MCJAX else None),
      "pathways_config": (user_config.pathways_config if Framework(framework) == Framework.PATHWAYS else None),
      "xpk_path": user_config.xpk_path,
      "num_steps": num_steps,
      "priority": priority,
  }
  # Add any extra arguments, like disruption_configs, if they exist
  config_args.update(kwargs)
  return mxr.WorkloadConfig(**config_args)


def _generate_workloads(
    user_config,
    num_slices_list: list,
    num_steps: int,
    priority: str,
    disruption_method: str = "",
    disruptions: dict = None,
):
  """A unified generator that yields WorkloadConfig objects."""
  for framework, model_list in user_config.models.items():
    if not model_list:
      logging.warning("Skipping empty model list for infrastructure: %s", framework)
      continue

    extra_config_kwargs = {}
    if disruptions:
      extra_config_kwargs["disruption_configs"] = construct_disruption_configs(framework, disruption_method, disruptions)

    for model in model_list:
      for num_slices in num_slices_list:
        yield _create_workload_config(
            framework, model, num_slices, user_config, num_steps, priority, **extra_config_kwargs
        )


def generate_and_run_workloads(
    user_config, num_slices_list, num_steps, priority="medium", disruption_method="", disruptions=None
):
  """
  Generates and executes XPK workloads, with or without disruptions.
  """
  workload_configs = list(
      _generate_workloads(
          user_config,
          num_slices_list,
          num_steps,
          priority,
          disruption_method=disruption_method,
          disruptions=disruptions,
      )
  )

  if not workload_configs:
    logging.warning("No workloads were generated. Exiting.")
    return 0

  disruption_manager = mxr.xpk_benchmark_runner(
      cluster_config=user_config.cluster_config,
      workload_configs=workload_configs,
  )

  if disruptions:
    disruption_manager.start_disruptions_and_wait_for_completion()
    print("Benchmark recipe disruptions completed. Please check logs for results.")

  else:
    print("Benchmark recipe completed. Please check logs for results.")
  return 0
