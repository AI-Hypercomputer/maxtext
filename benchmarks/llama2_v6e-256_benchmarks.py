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
"""Defines and runs Llama2 benchmarks on a v6e-256 cluster.

This script configures benchmark runs for Llama2-7B and Llama2-70B models
on a specific v6e-256 hardware setup using the XPK runner.
"""

import os

from benchmarks import maxtext_trillium_model_configs as model_configs
from benchmarks.maxtext_xpk_runner import WorkloadConfig
from benchmarks.maxtext_xpk_runner import xpk_benchmark_runner
from benchmarks.maxtext_xpk_runner import XpkClusterConfig


DATE = "20241009"
BASE_DOCKER_IMAGE = "maxtext_base_image"

ZONE = "europe-west4"
PROJECT = "tpu-prod-env-multipod"
CLUSTER_NAME = "mlperf-v6e-256"
DEVICE_TYPE = "v6e-256"
NUM_SLICES = 1
BASE_OUTPUT_DIR = "gs://maxtext-experiments-tpem/"
XPK_PATH = os.path.join("~", "xpk")
BENCHMARK_STEPS = 20


def main() -> None:
  cluster_config = XpkClusterConfig(
      cluster_name=CLUSTER_NAME,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  workload_configs = []
  for model in [model_configs.llama2_7b_4096, model_configs.llama2_70b_4096]:
    workload_configs.append(
        WorkloadConfig(
            model=model,
            num_slices=NUM_SLICES,
            device_type=DEVICE_TYPE,
            base_output_directory=BASE_OUTPUT_DIR,
            base_docker_image=BASE_DOCKER_IMAGE,
            libtpu_type=None,
            libtpu_nightly_version=DATE,
            pathways_config=None,
            xpk_path=XPK_PATH,
            num_steps=BENCHMARK_STEPS,
            priority="medium",
        )
    )

  xpk_benchmark_runner(cluster_config, workload_configs)


if __name__ == "__main__":
  main()
