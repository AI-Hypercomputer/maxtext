import functools
import gc
import logging
import os
import sys
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from flax import nnx
from jax import config as jax_config
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
import torch
from tunix.models.qwen3 import model as qwen3_lib
from vllm import LLM, SamplingParams

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# JAX compilation cache settings - adjust as needed for your environment
_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"

# Flags
FLAGS = flags.FLAGS
_XPROF = flags.DEFINE_bool('xprof', False, 'xprof')
_RAND_INIT = flags.DEFINE_bool('rand_init', True, 'Whether to use random initialization instead of loading from checkpoint, for faster testing.')  

def _setup_jax_compilation_cache():
  jax_config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR)
  jax_config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax_config.update("jax_enable_compilation_cache", True)


def _setup_vllm():
  # for vLLM we can skip JAX precompilation with this flag, it makes startup faster
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["JAX_RANDOM_WEIGHTS"] = "False"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def _clean_device_memory():
  """Forces Python garbage collection and waits for JAX devices to idle."""
  logging.info("Cleaning JAX device memory...")
  # Run Python's garbage collector to free Python-level references
  gc.collect()
  # Wait for all devices to finish pending operations.
  # This allows JAX to reclaim memory associated with arrays
  # that are no longer referenced.
  for x in jax.live_arrays():
      x.delete()      
  logging.info("Device memory cleanup complete.")


@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.4f} seconds")  


def save_dict_to_file(dict, filename):
    with open(filename, 'w') as f:
        for key in sorted(dict.keys()):
            f.write(f"{key}: {dict[key].shape}\n")


def main():
  print(f"JAX devices: {jax.devices()}")  
  _setup_jax_compilation_cache()
  _setup_vllm()
  _clean_device_memory()

  FLAGS(sys.argv)

  llm = LLM(
    "Qwen/Qwen3-30B-A3B",
    max_model_len=16,
    # tensor_parallel_size=4,
    tensor_parallel_size=2,
    data_parallel_size=2,
    gpu_memory_utilization=0.65,
    # load_format="dummy",
    async_scheduling=False,
  )
  print("\n" + "="*80)
  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
  # save_dict_to_file(llm_state, "vllm_model_state_shapes.txt")
  # print("\n" + "="*80)

  # # [TP, DP, EP]
  # weight_name = "vllm_model.model.layers.0.mlp.experts.w13_weight"
  # target_weight = llm_state[weight_name][0, :, :]
  # np.save("vllm_expert_weight_4_1_1.npy", target_weight)
  # # np.save("vllm_expert_weight_2_2_1.npy", target_weight)
  # print(f"Saved weight {weight_name} with shape {target_weight.shape} to vllm_expert_weight_2_2_1.npy")

  # [TP, DP, EP]
  weight_name = "vllm_model.model.layers.0.self_attn.qkv_proj.weight"
  target_weight = llm_state[weight_name]
  # np.save("vllm_qkv_proj_4_1_1.npy", target_weight)
  # print(f"Saved weight {weight_name} with shape {target_weight.shape} to vllm_qkv_proj_4_1_1.npy")
  np.save("vllm_qkv_proj_2_2_1.npy", target_weight)
  print(f"Saved weight {weight_name} with shape {target_weight.shape} to vllm_qkv_proj_2_2_1.npy")

  # sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
  # print("\n" + "="*80)
  # print("Generation test after weight transfer:")
  # with timer("Generation"):
  #   print(llm.generate("Paris is", sampling_params=sampling_params))


if __name__ == "__main__":
  main()
