from convergence.c4_exp import llama3_1_8b_8192_c4en, llama3_1_8b_8192_c4multien, setupConvHParams
from maxtext_viperfish_model_configs import ConvHParams, llama2_70b_4096_real_data, llama2_70b_4096_real_data_int8, llama2_70b_4096_int8, llama2_70b_4096_int8_ckp
from maxtext_xpk_runner import BenchmarkRunner
from maxtext_xpk_runner import HWConfig
from maxtext_xpk_runner import SWconfig
from maxtext_xpk_runner import xpk_benchmark_runner
from maxtext_xpk_runner import XpkConfig


DATE = '20241028'
BASE_DOCKER_IMAGE = 'maxtext_base_image'

ZONE = 'europe-west1'
#PROJECT = 'tpu-prod-env-one-vm'
PROJECT = 'cloud-tpu-best-effort-colo'
CLUSTER_NAME = 'perf-v5p-4096'
NUM_SLICES = 2
NUM_DEVICES = 2048
DEVICE_TYPE = 'v5p-'+ str(NUM_DEVICES*2)

v5p_env_configs = SWconfig(
    base_docker_image=BASE_DOCKER_IMAGE, libtpu_version=DATE,
)
v5p_512_configs = HWConfig(num_slices=NUM_SLICES, device_type=DEVICE_TYPE)

llama2_70b_4096_real_data_v5p = BenchmarkRunner(
    model_name=llama2_70b_4096_int8_ckp,
    software_config=v5p_env_configs,
    hardware_config=v5p_512_configs,
)

import copy
import math

#warm_up_samples = 600 * 512
warm_up_samples = 600 * 512
decay_samples = 14000 * 512
total_samples = 14000 * 512
eval_samples = 36 * 512

c4_gbs256 = ConvHParams(
        global_batch_size=256,
        learning_rate=2e-4,
        warmup_samples=warm_up_samples,
        decay_end_samples=total_samples,
        total_tokens_to_train=decay_samples,
        eval_interval=24567)

c4_gbs512 = ConvHParams(
        global_batch_size=512,
        learning_rate=2e-4,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=total_samples,
        eval_interval=24567*10)

c4_gbs1024 = ConvHParams(
        global_batch_size=1024,
        learning_rate=2e-4,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=total_samples,
        eval_interval=24567)

c4_gbs2048 = ConvHParams(
        global_batch_size=2048,
        learning_rate=3e-4,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=int(total_samples * 1.2),
        eval_interval=24567)

c4_gbs4096 = ConvHParams(
        global_batch_size=4096,
        learning_rate=4e-4,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=int(total_samples * 1.5),
        eval_interval=24567)

c4_gbs8192 = ConvHParams(
        global_batch_size=8192,
        learning_rate=5e-4,
        warmup_samples=warm_up_samples,
        decay_end_samples=decay_samples,
        total_tokens_to_train=int(total_samples * 2),
        eval_interval=24567)

def main() -> None:
  cluster_config = XpkConfig(
      cluster_name=CLUSTER_NAME,
      project=PROJECT,
      zone=ZONE,
      num_slices=NUM_SLICES,
      device_type=DEVICE_TYPE,
      base_output_directory="gs://maxtext-experiments-tpem/llama-perf/"
  )

  model_name = llama2_70b_4096_real_data_v5p
  exp_name = "llama70b-perf"

  xpk_benchmark_runner(cluster_config, [model_name], exp_name)  

if __name__ == '__main__':
  main()
