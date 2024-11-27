from e2e.c4_exp import llama3_1_405b_8192_fsdp_dcn_c4_c4multien, setupConvHParams
from maxtext_viperfish_model_configs import ConvHParams, llama3_1_405b_8192_fsdp_dcn
from maxtext_xpk_runner import BenchmarkRunner
from maxtext_xpk_runner import HWConfig
from maxtext_xpk_runner import SWconfig
from maxtext_xpk_runner import xpk_benchmark_runner
from maxtext_xpk_runner import XpkConfig


DATE = '20241028'
BASE_DOCKER_IMAGE = 'maxtext_base_image'

# ZONE = 'europe-west4'
# #PROJECT = 'tpu-prod-env-one-vm'
# PROJECT = 'cloud-tpu-multipod-dev'
# CLUSTER_NAME = 'lizhiyu-moe-v5p-512'

ZONE = 'europe-west1'
#PROJECT = 'tpu-prod-env-one-vm'
PROJECT = 'cloud-tpu-best-effort-colo'
CLUSTER_NAME = 'perf-v5p-4096'

NUM_SLICES = 4
NUM_DEVICES = 2048
DEVICE_TYPE = 'v5p-'+ str(NUM_DEVICES*2)

v5p_env_configs = SWconfig(
    base_docker_image=BASE_DOCKER_IMAGE, libtpu_version=DATE,
)
v5p_512_configs = HWConfig(num_slices=NUM_SLICES, device_type=DEVICE_TYPE)

llama31_405b_c4_benchmark_v5p = BenchmarkRunner(
    model_name=llama3_1_405b_8192_fsdp_dcn_c4_c4multien,
    software_config=v5p_env_configs,
    hardware_config=v5p_512_configs,
)

import copy
import math

#warm_up_samples = 600 * 512
warm_up_samples = 8000 * 512
decay_samples = 360000 * 512
total_samples = 360000 * 512

c4_gbs2048 = ConvHParams(
        global_batch_size=2048,
        learning_rate=8e-5,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=total_samples,
        eval_interval=24567*20)

c4_gbs4096 = ConvHParams(
        global_batch_size=4096,
        learning_rate=10e-5,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=total_samples,
        eval_interval=24567*20)

c4_gbs8192 = ConvHParams(
        global_batch_size=8192,
        learning_rate=12e-5,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=total_samples,
        eval_interval=24567*20)

def main() -> None:
  cluster_config = XpkConfig(
      cluster_name=CLUSTER_NAME,
      project=PROJECT,
      zone=ZONE,
      num_slices=NUM_SLICES,
      device_type=DEVICE_TYPE,
      base_output_directory="gs://maxtext-experiments-tpem/llama-conv/"
  )

  model_name = llama3_1_405b_8192_fsdp_dcn_c4_c4multien
  exp_name = "llama405b-perf"
  benchmark_lists=[]
  for config in [c4_gbs2048]:
    model = copy.deepcopy(model_name)
    setupConvHParams(model, config, NUM_DEVICES*NUM_SLICES)
    benchmark_model = BenchmarkRunner(
      model_name=model,
      software_config=v5p_env_configs,
      hardware_config=v5p_512_configs,
    )
    benchmark_lists.append(benchmark_model)

  xpk_benchmark_runner(cluster_config, benchmark_lists, exp_name)
if __name__ == '__main__':
  main()
