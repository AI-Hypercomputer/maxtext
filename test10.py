import os
# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "False"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
print(os.getcwd())

from vllm import LLM

import functools
import jax
import humanize
import gc
from tunix.generate import utils
from tunix.rl import reshard
from MaxText.integration.tunix.weight_mapping.qwen3 import QWEN3_VLLM_MAPPING


MODEL = "unsloth/gpt-oss-20b-BF16"


def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


def clean_device_memory():
  """
  Forces Python garbage collection and waits for JAX devices to idle.

  This helps ensure device memory is freed after Python references
  to arrays (e.g., DeviceArrays) are deleted.
  """
  print("Cleaning JAX device memory...")

  # Run Python's garbage collector to free Python-level references
  gc.collect()

  # Wait for all devices to finish pending operations.
  # This allows JAX to reclaim memory associated with arrays
  # that are no longer referenced.
  for x in jax.live_arrays():
    x.delete()
  print("Device memory cleanup complete.")


show_hbm_usage()
clean_device_memory()
show_hbm_usage()


import gc
import pathwaysutils
import MaxText.utils.ckpt_conversion.utils.param_mapping as param_mapping
import MaxText.utils.ckpt_conversion.utils.hf_model_configs as hf_model_configs

pathwaysutils.initialize()


import sys
from flax import nnx
import jax
from tunix.models.qwen3 import model as qwen3_lib
from MaxText.globals import MAXTEXT_ASSETS_ROOT


# ~/HOME/maxtext/MaxText/examples

# Get the directory of the current script
script_dir = os.getcwd()

# Go up two levels to get the project root
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter


print(f"JAX devices: {jax.devices()}")

DEBUG = True  # set to True to run in debug mode, for more print statements
HOME = os.path.expanduser("~") + "/"
print(f"Home directory (from Python): {HOME}")

# Look for base.yml in two possible locations.
print(os.path.join(HOME))
path1 = "/deps/src/MaxText/configs/base.yml"
path2 = os.path.join(HOME, "maxtext/src/MaxText/configs/base.yml")
if os.path.exists(path1):
  BASE_YAML_PATH = path1
elif os.path.exists(path2):
  BASE_YAML_PATH = path2
else:
  raise FileNotFoundError("Could not find base.yml in the expected locations: " f"{path1} or {path2}")


def get_ref_maxtext_model(config):

  model, mesh = model_creation_utils.create_nnx_model(config)
  with mesh:

    # tunix_model = TunixMaxTextAdapter(base_model=model, hf_model_config=hf_model_configs.qwen3_8b_config)
    tunix_model = TunixMaxTextAdapter(base_model=model, use_standalone_mappings=True)

    # model_config = qwen3_lib.ModelConfig.qwen3_8b()

    # tunix_model.config = model_config

  return tunix_model, mesh


# model_config = qwen3_lib.ModelConfig.qwen3_8b()

# Load the reference model
# Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
# To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
config_ref = pyconfig.initialize(
    [
        "",
        BASE_YAML_PATH,
    ],
    base_output_directory="gs://runner-maxtext-logs",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-gpt-oss",
    tokenizer_type="huggingface",
    tokenizer_path=MODEL,
    load_parameters_path="gs://shuningjin-multipod-dev/gpt-oss-20b/scan-flags-false-2025-11-11-01-42-40/0/items",
    per_device_batch_size=1,
    max_prefill_predict_length=32,
    max_target_length=64,
    steps=100,
    async_checkpointing="false",
    model_name="gpt-oss-20b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
)

gpt_oss, mesh = get_ref_maxtext_model(config_ref)
# qwen3_8b.config = model_config
print("Maxtext model loaded successfully")
src_state = nnx.state(gpt_oss)
for k, v in src_state.flat_state():
  print("-".join(k), "|", v.value.shape)
# sys.exit()


MODEL = "unsloth/gpt-oss-20b-BF16"
golden_llm = LLM(
    MODEL,
    max_model_len=64,
    # max_model_len=128,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
)

print(golden_llm.generate("what is the capital of France?"))
print("vLLM model loaded successfully")

dst_golden_state = golden_llm.llm_engine.model_executor.driver_worker.model_runner.state
reshard_fn = reshard.reshard_pytree

tgt_flat_list = dst_golden_state.flat_state()
tgt_flat_dict = {".".join(str(k) for k in keys): v for keys, v in tgt_flat_list}


# mapping = QWEN3_VLLM_MAPPING.to_hf_mapping()
# hooks = QWEN3_VLLM_MAPPING.to_hf_hook_fns()
# transpose = QWEN3_VLLM_MAPPING.to_hf_transpose_keys()


def transfer():
  pass


result = transfer(src_state)


# result = utils.transfer_state_with_mappings(src_state=nnx.state(qwen3_8b),dst_state=golden_llm.llm_engine.model_executor.driver_worker.model_runner.state,key_mappings=qwen3_8b.to_hf_mappings(),key_mapping_hook_fns=qwen3_8b.to_hf_hook_fns(),transpose_keys=qwen3_8b.to_hf_transpose_keys(), reshard_fn=reshard.reshard_pytree,)

matched = utils.verify_state_closeness(
    state=result, golden_state=golden_llm.llm_engine.model_executor.driver_worker.model_runner.state
)
print(matched)
