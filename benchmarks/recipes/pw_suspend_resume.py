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

"""
This recipe tests the suspend and resume functionality in Pathways.

It launches a MaxText workload and then uses the DisruptionManager to send a
SIGTERM signal at a specific training step. This simulates a planned preemption
event, allowing validation of the framework's ability to gracefully suspend,
checkpoint, and later resume training.
"""
import os

import args_helper as helper

from benchmarks.disruption_management.disruption_handler import DisruptionConfig
from benchmarks.disruption_management.disruption_handler import DisruptionMethod
from benchmarks.disruption_management.disruption_handler import PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX
from benchmarks.disruption_management.disruption_handler import PATHWAYS_WORKER_CONTAINER_NAME
from benchmarks.disruption_management.disruption_handler import PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX
from benchmarks.disruption_management.disruption_handler import TriggerType
from benchmarks.maxtext_trillium_model_configs import MaxTextModel
from benchmarks import maxtext_v5e_model_configs as v5e_model_configs
from benchmarks import maxtext_xpk_runner as mxr
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
XPK_PATH = "../xpk"  # We're running this script from the maxtext directory
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = f"gs://{USER}-{PROJECT}-{COUNTRY}/disruption_management/"
MAX_RESTARTS = 0
NUM_SLICES = 2
BENCHMARK_STEPS = 101


def construct_disruption_configs() -> list[DisruptionConfig]:
  """Constructs the disruption configs for the benchmark."""
  return [
      DisruptionConfig(
          name="sigill_5step",
          trigger_type=TriggerType.STEP,
          trigger_value=3,  # 3rd step trigger
          disruption_method=DisruptionMethod.SIGTERM,
          target_pod_regex=PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX,
          step_pod_regex=PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX,
          worker_container_name=PATHWAYS_WORKER_CONTAINER_NAME,
      ),
  ]


def construct_workload_config_with_disruptions(
    cluster_config: XpkClusterConfig,
    model: MaxTextModel,
    pathways_config: mxr.PathwaysConfig,
) -> list[mxr.WorkloadConfig]:
  """Constructs the workload configs for the benchmark."""
  return mxr.WorkloadConfig(
      model=model,
      num_slices=NUM_SLICES,
      device_type=cluster_config.device_type,
      base_output_directory=BASE_OUTPUT_DIRECTORY,
      max_restarts=MAX_RESTARTS,
      libtpu_type=None,
      libtpu_nightly_version="",
      base_docker_image=RUNNER,
      pathways_config=pathways_config,
      xpk_path=XPK_PATH,
      num_steps=BENCHMARK_STEPS,
      disruption_configs=construct_disruption_configs(),
  )


def main():
  """Main function to run the suspend/resume disruption test."""

  # Cluster Configuration
  cluster_config = XpkClusterConfig(
      cluster_name=CLUSTER,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  # Handle command line arguments using args_helper
  should_continue = helper.handle_cmd_args(cluster_config, helper.DELETE, xpk_path=XPK_PATH)

  if not should_continue:
    return

  # Model Configuration - Using a simple default model for testing
  model = v5e_model_configs.llama3_1_8b_8192

  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_server_image=PROXY_IMAGE,
      runner_image=RUNNER,
      # User can add additional flags here.
      server_flags="--enable_metrics_collection=false",
      proxy_flags="--enable_metrics_collection=false",
      worker_flags="--enable_metrics_collection=false",
  )

  # Pathways Workload Configuration with Disruption
  workload_configs = []
  pathways_workload_config = construct_workload_config_with_disruptions(cluster_config, model, pathways_config)
  workload_configs.append(pathways_workload_config)

  # Run the benchmark and use the returned disruption manager.
  disruption_manager = mxr.xpk_benchmark_runner(
      cluster_config=cluster_config,
      workload_configs=workload_configs,
  )

  # Wait for disruptions to complete
  disruption_manager.start_disruptions_and_wait_for_completion()

  print("Suspend/Resume disruptions completed. Please check logs for results.")


if __name__ == "__main__":
  main()
