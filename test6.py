import os
import functools
import jax
import humanize
import gc
from tunix.generate import utils
from tunix.rl import reshard
# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "False"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
print(os.getcwd())

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

from vllm import LLM

MODEL = "Qwen/Qwen3-8B"
golden_llm = LLM(
    MODEL,
    max_model_len=128,
    tensor_parallel_size=8
    )

print("vLLM model loaded successfully")

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
  raise FileNotFoundError(
      "Could not find base.yml in the expected locations: "
      f"{path1} or {path2}"
  )


os.environ["JAX_RANDOM_WEIGHTS"] = "False" 

def get_ref_maxtext_model(config):

  model, mesh = model_creation_utils.create_nnx_model(config)
  with mesh:
 
    tunix_model = TunixMaxTextAdapter(base_model=model, hf_model_config=hf_model_configs.qwen3_8b_config)
    model_config = qwen3_lib.ModelConfig.qwen3_8b()

    tunix_model.config = model_config

  return tunix_model, mesh

model_config = qwen3_lib.ModelConfig.qwen3_8b()

# Load the reference model
# Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
# To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
config_ref = pyconfig.initialize(
    [
        "",
        BASE_YAML_PATH,
    ],
    base_output_directory="gs://abhinavsing_bucket/",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-qwen3-8b",
    tokenizer_type="huggingface",
    tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
    load_parameters_path="gs://abhinavsing_bucket/qwen3_scanned/0/items",
    per_device_batch_size=1,
    max_prefill_predict_length=64,
    max_target_length=1024,
    steps=100,
    async_checkpointing="false",
    model_name="qwen3-8b",
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

qwen3_8b, mesh = get_ref_maxtext_model(config_ref)
qwen3_8b.config = model_config
print("Maxtext model loaded successfully")



os.environ["JAX_RANDOM_WEIGHTS"] = "False" 

src_state = nnx.state(qwen3_8b)
dst_golden_state = golden_llm.llm_engine.model_executor.driver_worker.model_runner.state
reshard_fn = reshard.reshard_pytree

tgt_flat_list = dst_golden_state.flat_state()
tgt_flat_dict = {
      '.'.join(str(k) for k in keys): v for keys, v in tgt_flat_list
  }


print("Maxtext model loaded successfully")
print(golden_llm.generate("What is the capital of France?"))

mapping = QWEN3_VLLM_MAPPING.to_hf_mapping()
hooks = QWEN3_VLLM_MAPPING.to_hf_hook_fns()
transpose = QWEN3_VLLM_MAPPING.to_hf_transpose_keys()


result = utils.transfer_state_with_mappings(src_state=nnx.state(qwen3_8b),dst_state=golden_llm.llm_engine.model_executor.driver_worker.model_runner.state,key_mappings=qwen3_8b.to_hf_mappings(),key_mapping_hook_fns=qwen3_8b.to_hf_hook_fns(),transpose_keys=qwen3_8b.to_hf_transpose_keys(), reshard_fn=reshard.reshard_pytree,)
