import os
import functools
import gc
import sys

# Fix for temp directory error in absl.testing (imported by chex -> optax -> flax)
from absl import flags
os.environ.setdefault('TMPDIR', '/tmp')
os.environ.setdefault('TEST_TMPDIR', '/tmp')

from jax import config as jax_config
import jax.sharding as jax_sharding
import humanize
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import time

import maxtext.checkpoint_conversion.utils.hf_model_configs as hf_model_configs
import maxtext.checkpoint_conversion.utils.param_mapping as param_mapping
import pathwaysutils
from maxtext.utils import model_creation_utils
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from tunix.generate import utils
from tunix.rl import reshard
from tunix.models.qwen3 import model as qwen3_lib
from vllm import LLM

import maxtext.checkpoint_conversion.to_huggingface as to_hf_utils
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.utils import process_maxtext_param

_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"
_XPROF_PATH="/home/wyzhang_google_com/mnt/xprof"

jax_config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR)
jax_config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax_config.update("jax_enable_compilation_cache", True)

FLAGS = flags.FLAGS
flags.DEFINE_bool('xprof', False, 'Enable xprof profiling.')

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["JAX_RANDOM_WEIGHTS"] = "False"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# ==============================================================================
# Helper Functions
# ==============================================================================
def _normalize_vllm_state(vllm_state):
  """Normalize vLLM state to provide consistent interface.
  
  Args:
    vllm_state: Either an NNX State object (JAX backend) or a dictionary (Torchax backend)
    
  Returns:
    State object with flat_state() method
  """
  # Check if it's already an NNX state with flat_state method
  if hasattr(vllm_state, 'flat_state') and callable(vllm_state.flat_state):
    return vllm_state
  # Otherwise, assume it's a dictionary from Torchax backend
  elif isinstance(vllm_state, dict):
    return DictStateAdapter(vllm_state)
  else:
    raise TypeError(
      f"Unsupported vLLM state type: {type(vllm_state)}. "
      "Expected NNX State (JAX backend) or dict (Torchax backend)."
    )

def _show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)
  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")

def _clean_device_memory():
  """Forces Python garbage collection and waits for JAX devices to idle."""
  print("Cleaning JAX device memory...")
  # Run Python's garbage collector to free Python-level references
  gc.collect()
  # Wait for all devices to finish pending operations.
  # This allows JAX to reclaim memory associated with arrays
  # that are no longer referenced.
  for x in jax.live_arrays():
      x.delete()      
  print("Device memory cleanup complete.")

def _get_ref_maxtext_model(config):
  """Creates and returns a Tunix-adapted MaxText model and mesh."""
  model, mesh = model_creation_utils.create_nnx_model(config)
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model)
    # Use the appropriate model config based on the model name
    if config.model_name == "qwen3-30b-a3b":
      model_config = qwen3_lib.ModelConfig.qwen3_30b()
    elif config.model_name == "qwen3-0.6b":
      model_config = qwen3_lib.ModelConfig.qwen3_0_6b()
    else:
      raise ValueError(f"Unsupported model: {config.model_name}")
    tunix_model.config = model_config
  return tunix_model, mesh

def get_nested_attr(obj, attr_path: str):
  """Access nested attributes/keys from a dot-separated string path.
  
  Handles both object attributes and dictionary keys automatically.
  
  Args:
      obj: The root object (e.g., qwen3_model)
      attr_path: Dot-separated path like 'base.decoder.decoder_norm.scale.value'
  
  Returns:
      The value at that path
  """
  parts = attr_path.split('.')
  current = obj
  
  for part in parts:
    # Try attribute access first
    if hasattr(current, part):
      current = getattr(current, part)
    # Fall back to dictionary access
    elif isinstance(current, dict) and part in current:
      current = current[part]
    else:
      raise AttributeError(
          f"Cannot access '{part}' in path '{attr_path}'. "
          f"Current object type: {type(current)}"
      )
  
  return current

# ==============================================================================
# Main Execution
# ==============================================================================

FLAGS(sys.argv)
print("Starting script...")
print(f"Current working directory: {os.getcwd()}")
print(f"JAX devices: {jax.devices()}")

# --- Initial Setup ---
pathwaysutils.initialize()
_clean_device_memory()

# --- Load vLLM Model (Golden Reference) ---
print("\n" + "="*80)
print("Loading vLLM model...")
print("="*80)

# Path to your locally downloaded and modified model directory.
# Make sure you have edited the config.json in this directory.
# VLLM_MODEL_PATH = "Qwen/Qwen3-0.6B"
# VLLM_MODEL_PATH = "Qwen/Qwen3-8B"
VLLM_MODEL_PATH = "Qwen/Qwen3-30B-A3B"
# VLLM_MODEL_PATH = "Qwen/Qwen3-235B-A22B"
golden_llm = LLM(
  VLLM_MODEL_PATH,
  max_model_len=16,
  tensor_parallel_size=4,
  gpu_memory_utilization=0.25,  # Limit to 40% memory per device
  download_dir="/home/wyzhang_google_com/ckpt/hf",
)
dst_golden_state = golden_llm.llm_engine.model_executor.driver_worker.model_runner.state

# Use deterministic sampling for consistent outputs
from vllm import SamplingParams
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

# --- Load MaxText Model ---
print("\n" + "="*80)
print("Loading MaxText model...")
print("="*80)

# Find base.yml config path
HOME = os.path.expanduser("~")
path1 = "/home/wyzhang_google_com/mnt/rl/maxtext/src/maxtext/configs/base.yml"
path2 = os.path.join(HOME, "mnt/rl/maxtext/src/maxtext/configs/base.yml")
if os.path.exists(path1):
  BASE_YAML_PATH = path1
elif os.path.exists(path2):
  BASE_YAML_PATH = path2
else:
  raise FileNotFoundError(
      f"Could not find base.yml in the expected locations: {path1} or {path2}"
  )

# Initialize MaxText config
config_ref = pyconfig.initialize(
    [ "", BASE_YAML_PATH, ],
    base_output_directory="gs://wyzhang-dev/tmp",  # Not used in Tunix.
    run_name="test-tunix-maxtext-qwen3-8b",
    tokenizer_type="huggingface",
    tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
    # model_name="qwen3-0.6b",
    # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-0.6b/scanned/2026-01-21-11-35/0/items",
    model_name="qwen3-30b-a3b",
    load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items",
    # load_parameters_path="/dev/shm/hengtaoguo/0/items/0/items",
    # model_name="qwen3-235b-a22b",
    # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-235b-a22b/scanned/001/0/items",
    # scan_layers="true",
    per_device_batch_size=1,
    max_prefill_predict_length=8,
    max_target_length=16,
    steps=100,
    async_checkpointing="false",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
    ici_fsdp_parallelism=2,
    ici_tensor_parallelism=4,
    override_model_config="true",
    # base_num_decoder_layers=2,
)

# Create the MaxText model
qwen3_model, mesh = _get_ref_maxtext_model(config_ref)
print("MaxText model loaded successfully")
print(f"Model: {config_ref.model_name}")


hf_config_obj = HF_MODEL_CONFIGS[config_ref.model_name]
mappings = to_hf_utils._get_model_mappings(config_ref.model_name, 
                                           config_ref.scan_layers, 
                                           hf_config_obj.to_dict(), 
                                           config_ref)
param_map = mappings["param_mapping"]
shape_map = mappings["shape_mapping"]
hook_fn_map = mappings["hook_fn_mapping"]

print("\n" + "="*80)
print("Starting BATCHED weight transfer (3-phase approach)...")
print("="*80)
start_time = time.time()

# ==============================================================================
# PHASE 1: Extract ALL weights from MaxText at once (~2 seconds)
# ==============================================================================
print("\n[Phase 1/3] Extracting all MaxText weights...")
phase1_start = time.time()

all_maxtext_weights = {}
for maxtext_key in param_map.keys():
  tunix_path = maxtext_key.replace('-', '.').replace('params.', 'base.') + '.value'
  all_maxtext_weights[maxtext_key] = get_nested_attr(qwen3_model, tunix_path)

print(f"✅ Phase 1 complete: Extracted {len(all_maxtext_weights)} weight tensors in {time.time() - phase1_start:.2f}s")

# ==============================================================================
# PHASE 2: Batch process ALL transformations (~5 seconds)
# ==============================================================================
print("\n[Phase 2/3] Processing all weight transformations...")
phase2_start = time.time()


all_processed_weights = []
for maxtext_key, weight in all_maxtext_weights.items():
  xprof = FLAGS.xprof and (maxtext_key == "params-decoder-layers-moe_block-wi_0")

  if xprof:
    jax.profiler.start_trace(_XPROF_PATH)
  jax_to_numpy = False
  processed_params = process_maxtext_param(maxtext_key, weight, param_map, hook_fn_map, shape_map, config_ref, jax_to_numpy=jax_to_numpy)
  if xprof:
    jax.profiler.stop_trace()

  # NOTE(wyzhang): verify all processed weights are JAX arrays
  if jax_to_numpy:
    for _, w in processed_params:
      assert isinstance(w, jax.Array), f"Expected JAX array, got {type(w)}"
  all_processed_weights.extend(processed_params)

print(f"✅ Phase 2 complete: Processed {len(all_processed_weights)} HF weights in {time.time() - phase2_start:.2f}s")

# Free MaxText weights and model immediately
del all_maxtext_weights, qwen3_model
gc.collect()

# ==============================================================================
# PHASE 3: Batch assign with pre-allocated fusion buffers (~3 seconds)
# ==============================================================================
print("\n[Phase 3/3] Batched assignment with fusion...")
phase3_start = time.time()

# Pre-allocate fusion buffers for all layers
num_layers = config_ref.base_num_decoder_layers
num_experts = 128  # Adjust based on your model

layer_qkv_buffers = {}  # {layer_idx: {'q': array, 'k': array, 'v': array}}
layer_expert_buffers = {}  # {layer_idx: {'gate': [...], 'up': [...], 'down': [...]}}

# Initialize buffers
for layer_idx in range(num_layers):
  layer_qkv_buffers[layer_idx] = {}
  layer_expert_buffers[layer_idx] = {
    'gate': [None] * num_experts,
    'up': [None] * num_experts,
    'down': [None] * num_experts
  }

# Fill buffers and assign direct mappings
direct_assignments = {}
for hf_key, hf_weight in all_processed_weights:
  # Detect QKV projections
  qkv_match = None
  for proj_type in ['q_proj', 'k_proj', 'v_proj']:
    if f'.self_attn.{proj_type}.weight' in hf_key:
      qkv_match = proj_type[0]  # 'q', 'k', or 'v'
      layer_idx = int(hf_key.split('.layers.')[1].split('.')[0])
      layer_qkv_buffers[layer_idx][qkv_match] = hf_weight
      break
  
  # Detect MoE expert projections
  if qkv_match is None and 'experts.' in hf_key:
    layer_idx = int(hf_key.split('.layers.')[1].split('.')[0])
    expert_idx = int(hf_key.split('.experts.')[1].split('.')[0])
    
    if 'gate_proj.weight' in hf_key:
      layer_expert_buffers[layer_idx]['gate'][expert_idx] = hf_weight
    elif 'up_proj.weight' in hf_key:
      layer_expert_buffers[layer_idx]['up'][expert_idx] = hf_weight
    elif 'down_proj.weight' in hf_key:
      layer_expert_buffers[layer_idx]['down'][expert_idx] = hf_weight
  
  # Standard 1:1 mappings (non-QKV, non-expert)
  elif qkv_match is None:
    vllm_key = f"vllm_model.{hf_key}"
    direct_assignments[vllm_key] = hf_weight

# Batch fuse all QKV layers
print(f"Fusing QKV for {len(layer_qkv_buffers)} layers...")
qkv_fuse_start = time.time()
for layer_idx, qkv_dict in layer_qkv_buffers.items():
  if len(qkv_dict) == 3:  # All Q, K, V present
    q_proj = qkv_dict['q']
    k_proj = qkv_dict['k']
    v_proj = qkv_dict['v']
    
    # Calculate dimensions
    num_q_heads = 32  # Adjust based on model
    num_kv_heads = 4  # Adjust based on model
    
    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), static_argnames=['num_q_heads', 'num_kv_heads'])
    def fuse_qkv(q, k, v, num_q_heads, num_kv_heads):
      head_dim = q.shape[0] // num_q_heads
      hidden_size = q.shape[1]
      heads_per_group = num_q_heads // num_kv_heads
      q_r = q.reshape(num_kv_heads, heads_per_group, head_dim, hidden_size)
      k_r = k.reshape(num_kv_heads, 1, head_dim, hidden_size)
      v_r = v.reshape(num_kv_heads, 1, head_dim, hidden_size)
      fused = jnp.concatenate([q_r, k_r, v_r], axis=1)
      return fused.reshape(-1, hidden_size)

    qkv_fused = fuse_qkv(q_proj, k_proj, v_proj, num_q_heads, num_kv_heads)
    
    vllm_qkv_key = f"vllm_model.model.layers.{layer_idx}.self_attn.qkv_proj.weight"
    direct_assignments[vllm_qkv_key] = qkv_fused

print(f"  QKV fusion complete in {time.time() - qkv_fuse_start:.2f}s")

# Batch fuse all expert layers
print(f"Fusing experts for {len(layer_expert_buffers)} layers...")
expert_fuse_start = time.time()
while layer_expert_buffers:
  layer_idx, expert_dict = layer_expert_buffers.popitem()
  gate_list = expert_dict['gate']
  up_list = expert_dict['up']
  down_list = expert_dict['down']
  
  # Check if all experts are present
  all_present = all(
    gate_list[i] is not None and up_list[i] is not None and down_list[i] is not None
    for i in range(num_experts)
  )
  
  if all_present:
    # Fuse w13 (gate + up interleaved)
    tensor_parallel_size = 4
    num_chunks = tensor_parallel_size
    
    @functools.partial(jax.jit, donate_argnums=(0, 1), static_argnames=['num_chunks'])
    def fuse_w13_experts_optimized(gates, ups, num_chunks):
        n_experts, inter_dim, hidden = gates.shape
        chunk_size = inter_dim // num_chunks
        g = gates.reshape(n_experts, num_chunks, chunk_size, hidden)
        u = ups.reshape(n_experts, num_chunks, chunk_size, hidden)
        # Interleave: stack on a new axis then flatten the chunk+interleave axes
        # This is more efficient than manual concatenation in a loop
        fused = jnp.stack([g, u], axis=2) # (n_experts, num_chunks, 2, chunk_size, hidden)
        return fused.reshape(n_experts, -1, hidden)

    w13_fused = fuse_w13_experts_optimized( jnp.array(gate_list), jnp.array(up_list), num_chunks)
    w2_fused = jnp.stack(down_list, axis=0)
    
    w13_key = f"vllm_model.model.layers.{layer_idx}.mlp.experts.w13_weight"
    w2_key = f"vllm_model.model.layers.{layer_idx}.mlp.experts.w2_weight"
    direct_assignments[w13_key] = w13_fused
    direct_assignments[w2_key] = w2_fused
print(f"  Expert fusion complete in {time.time() - expert_fuse_start:.2f}s")

# Batch assign all weights to vLLM state
print(f"Assigning {len(direct_assignments)} weights to vLLM state...")
for vllm_key, weight in direct_assignments.items():
  assert vllm_key in dst_golden_state, f"Key not found: {vllm_key}"
  target_shape = dst_golden_state[vllm_key].shape
  # NOTE(wyzhang): Hack to workaround incompatible shape between downloaded weights and this benchmark code.
  if weight.shape == target_shape:
    pass
  elif len(weight.shape) == 3 and weight.shape == (target_shape[0], target_shape[2], target_shape[1]):
    @jax.jit(donate_argnums=(0,))
    def _transpose_weight(w):
      return w.transpose(0, 2, 1)
    weight = _transpose_weight(weight)
  else:
    assert weight.shape == target_shape, f"Shape mismatch for {vllm_key}: {weight.shape} vs {target_shape}"
  assert(isinstance(weight, jax.Array)), f"Expected JAX array for {vllm_key}, got {type(weight)}"
  dst_golden_state[vllm_key] = weight

print(f"✅ Phase 3 complete: All weights assigned in {time.time() - phase3_start:.2f}s")

# Single cleanup at end
del layer_qkv_buffers, layer_expert_buffers, direct_assignments
gc.collect()  

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n{'='*80}")
print(f"⏱️  TOTAL weight transfer completed in {elapsed_time:.2f} seconds")
print(f"{'='*80}")

_show_hbm_usage()
print("\n" + "="*80)
print("Generation test after weight transfer:")
# NOTE(wyzhang): Remain the same behavior to assign numpy to vllm state.
# TODO(wyzhang): When passing jax array directly here, vllm seems to not recognize and handle it properly.
for key, jax_array in dst_golden_state.items():
    # This blocks until the array is ready and copies it to Host RAM
    dst_golden_state[key] = np.array(jax_array)
print(golden_llm.generate("Paris is", sampling_params=sampling_params))
