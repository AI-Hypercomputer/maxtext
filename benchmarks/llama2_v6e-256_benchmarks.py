from maxstar_model_configs import llama2_70b_4096
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

v6e_env_configs = SWconfig(
    base_docker_image=BASE_DOCKER_IMAGE, libtpu_version=DATE
)
v6e_256_configs = HWConfig(num_slices=NUM_SLICES, device_type=DEVICE_TYPE)

llama2_70b_4096 = BenchmarkRunner(
    model_name=llama2_70b_4096,
    software_config=v6e_env_configs,
    hardware_config=v6e_256_configs,
)

llama2_7b_4096 = BenchmarkRunner(
    model_name=llama2_7b_4096,
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
  )

  xpk_benchmark_runner(cluster_config, [llama2_7b_4096, llama2_70b_4096])


if __name__ == '__main__':
  main()
