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
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# JAX compilation cache settings - adjust as needed for your environment
_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"

# Flags
FLAGS = flags.FLAGS
_XPROF = flags.DEFINE_bool('xprof', False, 'xprof')
_RAND_INIT = flags.DEFINE_bool('rand_init', False, 'Whether to use random initialization instead of loading from checkpoint, for faster testing.')  

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

def _get_maxtext_model(config):
  """Creates and returns a Tunix-adapted MaxText model and mesh."""
  logging.info(f'Creating model with config: {config}')
  model, mesh = model_creation_utils.create_nnx_model(
    # config, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    config, model_mode=MODEL_MODE_AUTOREGRESSIVE, use_rand_init=_RAND_INIT.value)    
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model)
    # Use the appropriate model config based on the model name
    if config.model_name == "qwen3-30b-a3b":
      model_config = qwen3_lib.ModelConfig.qwen3_30b_a3b()
    elif config.model_name == "qwen3-0.6b":
      model_config = qwen3_lib.ModelConfig.qwen3_0_6b()
    elif config.model_name == "qwen3-235b-a22b":
      model_config = qwen3_lib.ModelConfig(
        num_layers=94,
        vocab_size=151936,
        embed_dim=4096,
        hidden_dim=1536,
        num_heads=64,
        head_dim=128,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=5_000_000,
        num_experts=128,
        num_experts_per_tok=8,
    )
    else:
      raise ValueError(f"Unsupported model: {config.model_name}")
    tunix_model.config = model_config
  return tunix_model, mesh

def _load_maxtext_model(base_yaml_path):
  # Initialize MaxText config
  # config_ref = pyconfig.initialize(
  #     [ "", BASE_YAML_PATH, ],
  #     base_output_directory="gs://wyzhang-dev/tmp",  # Not used in Tunix.
  #     run_name="test-tunix-maxtext-qwen3-8b",
  #     tokenizer_type="huggingface",
  #     tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
  #     # model_name="qwen3-0.6b",
  #     # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-0.6b/scanned/2026-01-21-11-35/0/items",
  #     model_name="qwen3-30b-a3b",
  #     load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items",
  #     # load_parameters_path="/dev/shm/hengtaoguo/0/items/0/items",
  #     # model_name="qwen3-235b-a22b",
  #     # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-235b-a22b/scanned/001/0/items",
  #     # scan_layers="true",
  #     per_device_batch_size=1,
  #     max_prefill_predict_length=8,
  #     max_target_length=16,
  #     steps=100,
  #     async_checkpointing="false",
  #     checkpoint_period=5,
  #     skip_jax_distributed_system="true",
  #     weight_dtype="bfloat16",
  #     attention="dot_product",
  #     remat_policy="custom",
  #     decoder_layer_input="offload",
  #     query_proj="offload",
  #     key_proj="offload",
  #     value_proj="offload",
  #     ici_fsdp_parallelism=2,
  #     ici_tensor_parallelism=4,
  #     override_model_config="true",
  #     # base_num_decoder_layers=2,
  # )  
  config_ref = pyconfig.initialize(
      [ "", base_yaml_path, ],
      base_output_directory="gs://wyzhang-dev/tmp",  # Not used in Tunix.
      run_name="test-tunix-maxtext-qwen3",
      tokenizer_type="huggingface",
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
      # model_name="qwen3-235b-a22b",
      # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-235b-a22b/scanned/Qwen3-235B-A22B/0/items",
      model_name="qwen3-30b-a3b",
      load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items",
      scan_layers="true",
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
      ici_expert_parallelism=4,
      # ici_fsdp_parallelism=,
      # ici_tensor_parallelism=,
      override_model_config="true",
      # debug_sharding="true",
      checkpoint_storage_concurrent_gb=80,
  )
  model, mesh = _get_maxtext_model(config_ref)
  return config_ref, model, mesh

class MaxTextToVLLMConverter:
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.num_layers = config.base_num_decoder_layers
        self.vllm_state = {}

    # --- 1. Top-Level Entry Point ---
    def convert(self, model_state: dict):
        """Main entry point to convert all weights."""
        logging.info(f"\n{GREEN}Starting Conversion...{RESET}")
        start_time = time.time()

        with timer("Convert Global Weights"):
          self._convert_global(model_state)
        with timer("Convert Attention Weights"):
          self._convert_attn(model_state)
        with timer("Convert MoE Weights"):
          self._convert_moe(model_state)
        
        return self.vllm_state

    def _convert_global(self, params):
        self._to_embed_tokens(params)
        self._to_final_norm(params)
        self._to_lm_head(params)
                
    def _convert_attn(self, params):
      @jax.jit(donate_argnums=(0,))
      def _transpose_unstack(x):
        return jnp.unstack(jnp.transpose(x, (1, 0)))
    
      pre_ln = params['base']['decoder']['layers']['pre_self_attention_layer_norm']['scale']
      convert_pre_ln = _transpose_unstack(pre_ln)
      assert len(convert_pre_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(convert_pre_ln)}"
      for i, layer in enumerate(convert_pre_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.input_layernorm.weight': layer})
      del convert_pre_ln
      
      post_ln = params['base']['decoder']['layers']['post_self_attention_layer_norm']['scale']
      converted_post_ln = _transpose_unstack(post_ln)
      assert len(converted_post_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(converted_post_ln)}"
      for i, layer in enumerate(converted_post_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.post_attention_layernorm.weight': layer})      
      del post_ln      
      
      attn = params['base']['decoder']['layers']['self_attention']
      self_attn = self._to_attn(attn)
      for key, layers in self_attn.items():
        self.vllm_state.update({f'vllm_model.model.layers.{i}.{key}': layer for i, layer in enumerate(layers)})
      del attn
      
      gc.collect()
    
    def _convert_moe(self, params):
      moe = params['base']['decoder']['layers']['moe_block'].to_pure_dict()
      prefix = "vllm_model.model.layers"
      
      self.vllm_state.update({
          f"{prefix}.{i}.mlp.gate.weight": w 
          for i, w in enumerate(self._to_mlp_gate(moe['gate']['kernel']))
      })
      del moe['gate']['kernel']
      gc.collect()

      self.vllm_state.update({
          f"{prefix}.{i}.mlp.experts.w2_weight": w 
          for i, w in enumerate(self._to_mlp_expert_down(moe['wo']))
      })
      del moe['wo']
      gc.collect()

      self._to_mlp_expert_gate_up(
          moe['wi_0'], moe['wi_1'], 
          self.num_layers, prefix, 'mlp.experts.w13_weight'
      )
      del moe['wi_0'], moe['wi_1']
      
      gc.collect()
      
    def _to_final_norm(self, params):
      self.vllm_state["vllm_model.model.norm.weight"] = params['base']['decoder']['decoder_norm']['scale']

    def _to_embed_tokens(self, params):
      self.vllm_state["vllm_model.model.embed_tokens.weight"] = params['base']['token_embedder']['embedding']

    def _to_lm_head(self, params):
      @jax.jit(donate_argnums=(0,))
      def _transpose(x):
        return jnp.transpose(x, (1, 0))
      self.vllm_state["vllm_model.lm_head.weight"] = _transpose(
          params['base']['decoder']['logits_dense']['kernel']
      )
      
    @staticmethod
    @functools.partial(jax.jit, donate_argnums=(0,))
    def _to_attn(attn: PyTree) -> dict[str, jax.Array]:
      # (d_model, l, h, d) -> (l, d_model, h, d)
      q = jnp.transpose(attn['query']['kernel'], (1, 0, 2, 3))
      k = jnp.transpose(attn['key']['kernel'], (1, 0, 2, 3))
      v = jnp.transpose(attn['value']['kernel'], (1, 0, 2, 3))
      # GQA interleaving: [Q_group0, K0, V0, Q_group1, K1, V1, ...]
      # q: (l, d_model, num_q_heads, head_dim)
      # k: (l, d_model, num_kv_heads, head_dim)
      num_q_heads = q.shape[2]
      num_kv_heads = k.shape[2]
      heads_per_group = num_q_heads // num_kv_heads
      l, d_model, _, head_dim = q.shape
      # Group q: (l, d_model, num_kv_heads, heads_per_group, head_dim)
      q_grouped = q.reshape(l, d_model, num_kv_heads, heads_per_group, head_dim)
      # Group k, v: (l, d_model, num_kv_heads, 1, head_dim)
      k_grouped = k.reshape(l, d_model, num_kv_heads, 1, head_dim)
      v_grouped = v.reshape(l, d_model, num_kv_heads, 1, head_dim)
      # Concat within each group: (l, d_model, num_kv_heads, heads_per_group+2, head_dim)
      group = jnp.concatenate([q_grouped, k_grouped, v_grouped], axis=3)
      # Flatten and transpose: (l, num_kv_heads*(heads_per_group+2)*head_dim, d_model)
      qkv_proj = jnp.transpose(group.reshape(l, d_model, -1), (0, 2, 1))
      
      # (h, l, d, d_model) -> (l, d_model, Heads, d)
      o = jnp.transpose(attn['out']['kernel'], (1, 3, 0, 2))
      # (h, l, d, d_model) -> (l, d_model, Total_Dim)
      o_proj = o.reshape(o.shape[0], o.shape[1], -1)
    
      # (d_model, l) -> (l, d_model)
      q_norm = jnp.transpose(attn['query_norm']['scale'], (1, 0))
      # (d_model, l) -> (l, d_model)
      k_norm = jnp.transpose(attn['key_norm']['scale'], (1, 0))

      return {
          "self_attn.qkv_proj.weight": jnp.unstack(qkv_proj),
          "self_attn.o_proj.weight": jnp.unstack(o_proj),
          "self_attn.q_norm.weight": jnp.unstack(q_norm),
          "self_attn.k_norm.weight": jnp.unstack(k_norm)
      }
    
    def _to_mlp_gate(self, param):
      @functools.partial(jax.jit, donate_argnums=(0,))
      @jax.shard_map(
        in_specs=P(None, None, 'expert'), # [d_model, l, e]
        out_specs=P(None, 'expert', None), # [l, e, d_model]
        mesh=self.mesh)
      def _transpose(param):
        def _single_layer(moe_slice):
          # Transpose [d_model, e] -> [e, d_model]
          return jnp.transpose(moe_slice, (1, 0))
        return jax.vmap(_single_layer, 
                        in_axes=(1,))(param)
      param = _transpose(param)
      return self._unstack_layer(param)

    def _to_mlp_expert_down(self, param):
      @functools.partial(jax.jit, donate_argnums=(0,))
      @jax.shard_map(
        in_specs=P('expert', None, None, None), # [Expert, Layer, Hidden, Inter]
        out_specs=P(None, 'expert', None, None), # [Layer, Expert, Hidden, Inter]
        mesh=self.mesh)
      def _transpose(param):
        def _single_layer(moe_slice):
          # vLLM >= 0.17.1: w2_weight shape is [Expert, Hidden, Inter] (no additional transpose)
          return moe_slice
        return jax.vmap(_single_layer, 
                        in_axes=(1,))(param)  
      param = _transpose(param)
      return self._unstack_layer(param)        
    
    def _to_mlp_expert_gate_up(self, wi_0, wi_1, num_layers, layer_key_prefix, layer_key_suffix):
      @functools.partial(jax.jit, donate_argnums=(0,))
      @jax.shard_map(
        in_specs=(P('expert', None, None, None),),
        out_specs=P(None, 'expert', None, None),
        mesh=self.mesh)
      def _transpose(x):
        # (e, l, d_model, d_inner) -> (l, e, d_model, d_inner)
        return jnp.transpose(x, (1, 0, 2, 3))
      
      wi_0_layers = list(jnp.unstack(_transpose(wi_0)))
      del wi_0
      gc.collect()

      wi_1_layers = list(jnp.unstack(_transpose(wi_1)))
      del wi_1
      gc.collect()

      wi_01 = {}
      for i in range(num_layers - 1, -1, -1):
        wi_0_layer_i = wi_0_layers.pop()
        wi_1_layer_i = wi_1_layers.pop()
        
        @functools.partial(jax.jit, donate_argnums=(0, 1))
        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(
                P('expert', None, None), 
                P('expert', None, None), 
            ),
            out_specs=P('expert', None, None), 
        )
        def _fuse(wi_0, wi_1):
          # [e_shard, d_model, d_inner]
          e, d_model, d_inner = wi_0.shape
          # Chunk-level interleave to match vLLM TP sharding:
          # layout: [gate_chunk0, up_chunk0, gate_chunk1, up_chunk1, ...]
          num_chunks = 4  # tensor_parallel_size for vLLM
          chunk_size = d_inner // num_chunks
          gate_chunks = wi_0.reshape(e, d_model, num_chunks, chunk_size)
          up_chunks = wi_1.reshape(e, d_model, num_chunks, chunk_size)
          # [e, d_model, num_chunks, 2, chunk_size] -> [e, d_model, 2*d_inner]
          # vLLM >= 0.17.1: w13_weight shape is [Expert, d_model, 2*d_inner]
          combined = jnp.stack([gate_chunks, up_chunks], axis=3)
          return combined.reshape(e, d_model, 2 * d_inner)

        layer_i = _fuse(wi_0_layer_i, wi_1_layer_i)
        self.vllm_state.update({f"{layer_key_prefix}.{i}.{layer_key_suffix}": layer_i})
        del wi_0_layer_i, wi_1_layer_i
        # Force GC collection once every 8 layers to reduce the overhead of freuqent GC,
        # while still ensuring we don't hold onto too many large arrays at once.
        if i % 8 == 0:
          gc.collect()

      gc.collect()
      return wi_01
      
    @staticmethod
    @functools.partial(jax.jit, donate_argnums=(0,))
    def _unstack_layer(param):
        return jnp.unstack(param, axis=0)


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

  # Load maxtext model
  print("="*80)
  print("Loading MaxText model...")
  print("="*80)
  # path1 = "/home/wyzhang_google_com/mnt/rl/maxtext/src/maxtext/configs/base.yml"
  path1 = "/home/hengtaoguo_google_com/projects/maxtext/src/maxtext/configs/base.yml"
  path2 = os.path.join(os.path.expanduser("~"), "mnt/rl/maxtext/src/maxtext/configs/base.yml")
  if os.path.exists(path1):
    base_yaml_path = path1
  elif os.path.exists(path2):
    base_yaml_path = path2
  else:
    raise FileNotFoundError(
        f"Could not find base.yml in the expected locations: {path1} or {path2}"
    )
  config, model, mesh = _load_maxtext_model(base_yaml_path)
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {config.model_name}")
  print(f"Mesh: {mesh}")

  # Convert weights to VLLM format
  print("="*80)
  print("Converting weights to VLLM format")
  print("="*80)
  model_state = nnx.state(model)
  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, 'shape') and hasattr(leaf, 'sharding'):
      path_str = jax.tree_util.keystr(path)
      logging.info(f"Name: {path_str}, shape: {leaf.shape}")
      logging.info(f"\tSharding: {leaf.sharding}")
  
  converter = MaxTextToVLLMConverter(config, mesh)
  with timer("Overall Conversion "):
    vllm_state = converter.convert(model_state)
  del model_state
  gc.collect()
  # save_dict_to_file(vllm_state, "vllm_state_shapes.txt")

  # Load vLLM model and run generation test
  print("="*80)
  print("Loading vLLM model for generation test...")
  print("="*80)
  llm = LLM(
    "Qwen/Qwen3-30B-A3B",
    max_model_len=16,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.5,
  )
  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
  # save_dict_to_file(llm_state, "llm_state_shapes.txt")

  with timer(f"Assigning {len(vllm_state)} weights to vLLM model"):
    for key, weight in vllm_state.items():
      target_sharding = llm_state[key].sharding
      llm_state[key] = jax.device_put(np.asarray(weight), target_sharding)

  sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
  print("\n" + "="*80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate("Paris is", sampling_params=sampling_params))


if __name__ == "__main__":
  main()
