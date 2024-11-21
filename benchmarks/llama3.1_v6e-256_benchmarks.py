from e2e.c4_exp import llama3_1_8b_8192_c4en, llama3_1_8b_8192_c4multien, setupConvHParams
from maxtext_trillium_model_configs import ConvHParams
from maxtext_xpk_runner import BenchmarkRunner
from maxtext_xpk_runner import HWConfig
from maxtext_xpk_runner import SWconfig
from maxtext_xpk_runner import xpk_benchmark_runner
from maxtext_xpk_runner import XpkConfig


DATE = '20241028'
BASE_DOCKER_IMAGE = 'maxtext_base_image'

ZONE = 'us-east5'
#PROJECT = 'tpu-prod-env-one-vm'
PROJECT = 'tpu-prod-env-automated'
CLUSTER_NAME = 'bodaborg-v6e-256'
DEVICE_TYPE = 'v6e-256'
NUM_SLICES = 1
NUM_DEVICES = 256
DEVICE_TYPE = 'v6e-'+ str(NUM_DEVICES)

v6e_env_configs = SWconfig(
    base_docker_image=BASE_DOCKER_IMAGE, libtpu_version=DATE,
)
v6e_256_configs = HWConfig(num_slices=NUM_SLICES, device_type=DEVICE_TYPE)

llama31_8b_c4_benchmark_v6e = BenchmarkRunner(
    model_name=llama3_1_8b_8192_c4en,
    software_config=v6e_env_configs,
    hardware_config=v6e_256_configs,
)

import copy
import math

#warm_up_samples = 600 * 512
warm_up_samples = 800 * 512
decay_samples = 10000 * 512
total_samples = 10000 * 512
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
        eval_interval=24567)

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
      base_output_directory="gs://maxtext-experiments-tpem/llama-conv/"
  )

  # warmup_steps_fractions = range(1000, 3000, 1000)
  # learning_rates = map(lambda x: x/1e5, range(2, 30, 4))
  # total_steps = 3000
  # benchmark_lists = []
  # for learning_rate in learning_rates:
  #   #for warmup_steps in warmup_steps_fractions:
  #     model = copy.deepcopy(llama31_8b_c4_benchmark_v6e)
  #     #model.model_name.tuning_params["warmup_steps_fraction"] = float(warmup_steps) / total_steps
  #     model.model_name.tuning_params["learning_rate"] = learning_rate
  #     benchmark_lists.append(model)
  # #xpk_benchmark_runner(cluster_config, benchmark_lists)

  benchmark_lists = [] #c4_gbs1024, c4_gbs2048, c4_gbs4096, c4_gbs8192
  for config in [c4_gbs512]:
    model = copy.deepcopy(llama3_1_8b_8192_c4en)
    setupConvHParams(model, config, NUM_DEVICES*NUM_SLICES)
  
    benchmark_model = BenchmarkRunner(
      model_name=model,
      software_config=v6e_env_configs,
      hardware_config=v6e_256_configs,
    )
    benchmark_lists.append(benchmark_model)
  
  #xpk_benchmark_runner(cluster_config, benchmark_lists, "llama8b-c4multi-exp")  
  xpk_benchmark_runner(cluster_config, benchmark_lists, "llama8b-c4-exp")  
if __name__ == '__main__':
  main()
