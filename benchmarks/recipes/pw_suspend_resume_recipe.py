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

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from disruption_management.disruption_manager import DisruptionConfig
from disruption_management.disruption_manager import RecoverMethod
from disruption_management.disruption_manager import DisruptionMethod
from disruption_management.disruption_manager import PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX
from disruption_management.disruption_manager import TriggerType
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

# CLUSTER = "bodaborg-v6e-256-ts"
# PROJECT = "tpu-prod-env-multipod"
# ZONE = "us-west1-c"
# COUNTRY = "us"
# DEVICE_TYPE = "v6e-256"

################################################################################

# Other parameters (MUST BE SET BY USER)
XPK_PATH = "../xpk"  # We're running this script from the maxtext directory
USER = os.environ["USER"]
BASE_OUTPUT_DIRECTORY = (
    f"gs://{USER}-{PROJECT}-{COUNTRY}/pw_mcjax_benchmarking/"
)
# BASE_OUTPUT_DIRECTORY = (
#     f"gs://trillium-scale-tests-q1-25-west/pw_mcjax_benchmarking/{USER}/"
# )
MAX_RESTARTS = 10
NUM_SLICES = 2
BENCHMARK_STEPS = 101


def main() -> None:
  """Main function to run the Suspend/Resume disruption test."""

  # Cluster Configuration
  cluster_config = mxr.XpkClusterConfig(
      cluster_name=CLUSTER,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  # Disruption Manager
  disruption_manager = DisruptionManager()

  # Model Configuration - Using a simple default model for testing
  model = model_configs.llama3_1_70b_8192_iter_synth_data_and_checkpointing

  pathways_config = mxr.PathwaysConfig(
      server_image=SERVER_IMAGE,
      proxy_server_image=PROXY_IMAGE,
      runner_image=RUNNER,

      # User can add additional flags here.
      server_flags="",
      proxy_flags="",
      worker_flags="",
  )

  # Do 2 total disruptions, once at the 20th step and once at the 60th step
  disruption_configs = [
      DisruptionConfig(
          trigger_type=TriggerType.STEP,
          trigger_value=20,
          method=DisruptionMethod.SIGTERM,
          recover_method=RecoverMethod.SIGTERM,
          target_pod_regex=PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX,
      ),
      DisruptionConfig(
          disruption_trigger_type=TriggerType.STEP,
          disruption_trigger_value=60,
          disruption_method=DisruptionMethod.SIGTERM,
          recover_trigger_type=TriggerType.TIME_SECONDS,
          recover_trigger_value=60 * 5,
          recover_method=RecoverMethod.SIGTERM,
          target_pod_regex=PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX,
      ),
  ]

  # Workload Configuration with Disruption
  workload_config = mxr.WorkloadConfig(
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
      num_steps=100,
      disruption_configs=disruption_configs
  )

  # Run the benchmark with disruption manager
  mxr.xpk_benchmark_runner(
      cluster_config=cluster_config,
      workload_configs=[workload_config],
      disruption_manager=disruption_manager
  )

  # Wait for disruptions to complete
  disruption_manager.wait_for_disruptions_completed()

  print("SIGTERM disruption test completed")

if __name__ == "__main__":
  main()
