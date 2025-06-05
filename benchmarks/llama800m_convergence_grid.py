from convergence.c4_exp import c4_en_hp
from benchmarks.convergence.convergence_utils import ConvHParams, _setup_model_convergence_
from maxtext_v5p_model_configs import llama2_800m_4096
from benchmarks.maxtext_xpk_runner import PathwaysConfig
from benchmarks.maxtext_xpk_runner import WorkloadConfig
from benchmarks.maxtext_xpk_runner import xpk_benchmark_runner
from benchmarks.maxtext_xpk_runner import on_device_benchmark_runner
from benchmarks.maxtext_xpk_runner import LibTpuType
from benchmarks.xpk_configs import XpkClusterConfig

BASE_DOCKER_IMAGE = 'maxtext_base_image'

ZONE = 'europe-west4'
PROJECT = 'cloud-tpu-multipod-dev'
CLUSTER_NAME = 'mlperf-v5p-32-1'
NUM_SLICES = 1
NUM_DEVICES = 16

DEVICE_TYPE = 'v5p-'+ str(NUM_DEVICES*2)

OUT_DIR="gs://maxtext-experiments-tpem/llama-conv/c4en/2x"

import copy
import math
def main() -> None:
  cluster_config = XpkClusterConfig(
      cluster_name=CLUSTER_NAME,
      project=PROJECT,
      zone=ZONE,
      device_type=DEVICE_TYPE,
  )

  benchmark_lists = [] 

  model_name = llama2_800m_4096
  path = None

  total_samples = 2536282
  decay_samples = total_samples
  warm_up_samples = int(total_samples/128*0.4)
  eval_interval = 256 * 100 * 4096

  for batch_size in [192,]:
        for base_factor in range(-39, -29, 1):
                base_lr = pow(2, base_factor*0.25)
                benchmark_model = model_name

                llama_conv = ConvHParams(
                    global_batch_size = batch_size,
                    warmup_samples = warm_up_samples*batch_size, 
                    decay_end_samples = decay_samples,
                    total_tokens_to_train = total_samples * 4096,
                    training_scaleing_factor = 1.0,
                    learning_rate = base_lr / batch_size,
                    eval_interval = eval_interval,
                    eval_tokens = -1)

                benchmark_model = _setup_model_convergence_(
                  model_name,
                  c4_en_hp,
                  llama_conv,
                  num_devices = NUM_DEVICES,
                  global_batch_size = batch_size,

                )
                workload_config = WorkloadConfig(
                model=benchmark_model,
                num_slices=NUM_SLICES,
                num_steps=-1,
                device_type=DEVICE_TYPE,
                base_output_directory=OUT_DIR,
                base_docker_image=BASE_DOCKER_IMAGE,
                libtpu_type=LibTpuType.MAXTEXT,
                pathways_config=None,
                # Internal only support, not for customers
                generate_metrics_and_upload_to_big_query=False,
                )

                
                benchmark_lists.append(workload_config, )

  xpk_benchmark_runner(cluster_config, benchmark_lists)

if __name__ == '__main__':
  main()
