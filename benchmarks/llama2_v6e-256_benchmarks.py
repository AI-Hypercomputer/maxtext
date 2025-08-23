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
import maxtext_trillium_model_configs as model_configs
from maxtext_xpk_runner import BenchmarkRunner
from maxtext_xpk_runner import HWConfig
from maxtext_xpk_runner import SWconfig
from maxtext_xpk_runner import xpk_benchmark_runner
from maxtext_xpk_runner import XpkConfig


DATE = '20241009'
BASE_DOCKER_IMAGE = 'maxtext_base_image'

ZONE = 'europe-west4'
PROJECT = 'tpu-prod-env-multipod'
CLUSTER_NAME = 'mlperf-v6e-256'
DEVICE_TYPE = 'v6e-256'
NUM_SLICES = 1
BASE_OUTPUT_DIR = 'gs://maxtext-experiments-tpem/'

v6e_env_configs = SWconfig(
    base_docker_image=BASE_DOCKER_IMAGE, libtpu_version=DATE
)
v6e_256_configs = HWConfig(num_slices=NUM_SLICES, device_type=DEVICE_TYPE)

llama2_70b_4096 = BenchmarkRunner(
    model_name=model_configs.llama2_70b_4096,
    software_config=v6e_env_configs,
    hardware_config=v6e_256_configs,
)

llama2_7b_4096 = BenchmarkRunner(
    model_name=model_configs.llama2_7b_4096,
    software_config=v6e_env_configs,
    hardware_config=v6e_256_configs,
)


def main() -> None:
  cluster_config = XpkConfig(
      cluster_name=CLUSTER_NAME,
      project=PROJECT,
      zone=ZONE,
      num_slices=NUM_SLICES,
      device_type=DEVICE_TYPE,
      base_output_directory=BASE_OUTPUT_DIR,
  )

  xpk_benchmark_runner(cluster_config, [llama2_7b_4096, llama2_70b_4096])


if __name__ == '__main__':
  main()
