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
from maxtext.utils.globals import HF_IDS, MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR

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
_RAND_INIT = flags.DEFINE_bool('rand_init', False, 'Whether to use random initialization instead of loading from checkpoint, for faster testing.')
_MODEL_NAME = flags.DEFINE_string('model_name', 'qwen3-30b-a3b', 'MaxText model name (e.g. qwen3-30b-a3b, gemma4-26b, gemma4-31b)')
_CHECKPOINT = flags.DEFINE_string('checkpoint_path', 'gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items', 'GCS or local path to the scanned checkpoint')
_VLLM_MODEL_ID = flags.DEFINE_string('vllm_model_id', 'Qwen/Qwen3-30B-A3B', 'HuggingFace model ID passed to vLLM (e.g. google/gemma-4-26b-it)')
_FSDP_TP = flags.DEFINE_integer('ici_fsdp_parallelism', -1, 'ICI FSDP parallelism (-1 = auto-shard to fill remaining devices)')
_ICI_TP = flags.DEFINE_integer('ici_tensor_parallelism', 2, 'ICI tensor parallelism')
_ROLLOUT_TP = flags.DEFINE_integer('rollout_tensor_parallelism', 2, 'Rollout tensor parallelism')

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
    elif config.model_name == "gemma4-26b":
      # Tunix has no Gemma4 model class yet; config is unused for weight sync.
      model_config = None
    elif config.model_name.startswith("deepseek3"):
      # Tunix has no DeepSeekV3 model class yet; config is unused for weight sync.
      model_config = None
    elif config.model_name == "gemma4-31b":
      # Tunix has no Gemma4 model class yet; config is unused for weight sync.
      model_config = None
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
  model_name = _MODEL_NAME.value
  tokenizer_dir = "gemma4-tokenizer" if model_name.startswith("gemma4") else "qwen3-tokenizer"
  
  # Determine tokenizer path: prefer local assets if they exist, otherwise use HF ID.
  local_tokenizer_dir = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", tokenizer_dir)
  if os.path.isdir(local_tokenizer_dir):
    tokenizer_path = local_tokenizer_dir
  else:
    tokenizer_path = HF_IDS.get(model_name, os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers/tokenizer.llama2"))

  logging.info("Loading model_name=%s checkpoint=%s fsdp=%d ici_tp=%d rollout_tp=%d tokenizer=%s",
               model_name, _CHECKPOINT.value, _FSDP_TP.value, _ICI_TP.value, _ROLLOUT_TP.value, tokenizer_path)
               
  argv = ["", base_yaml_path] + [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("--")]
  config_ref = pyconfig.initialize(
      argv,
      base_output_directory="gs://tmp",  # Not used in Tunix.
      run_name=f"bench-weight-sync-{model_name}",
      tokenizer_type="huggingface",
      tokenizer_path=tokenizer_path,
      model_name=model_name,
      load_parameters_path=_CHECKPOINT.value,
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
      ici_fsdp_parallelism=_FSDP_TP.value,
      ici_tensor_parallelism=_ICI_TP.value,
      rollout_tensor_parallelism=_ROLLOUT_TP.value,
      override_model_config="true",
      checkpoint_storage_concurrent_gb=80,
      async_scheduling="false",
  )
  model, mesh = _get_maxtext_model(config_ref)
  return config_ref, model, mesh

class MaxTextToVLLMConverter:
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.num_layers = config.base_num_decoder_layers
        self.vllm_state = {}
        self.vllm_tp = self.config.rollout_tensor_parallelism

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
        logging.info("_convert_global: embed_tokens...")
        self._to_embed_tokens(params)
        logging.info("_convert_global: final_norm...")
        self._to_final_norm(params)
        logging.info("_convert_global: lm_head...")
        self._to_lm_head(params)
        logging.info("_convert_global: done")
                
    def _convert_attn(self, params):
      @jax.jit
      def _transpose_unstack(x):
        return jnp.unstack(jnp.transpose(x, (1, 0)))

      logging.info("_convert_attn: pre_self_attention_layer_norm...")
      pre_ln = params['base']['decoder']['layers']['pre_self_attention_layer_norm']['scale']
      convert_pre_ln = _transpose_unstack(pre_ln)
      assert len(convert_pre_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(convert_pre_ln)}"
      for i, layer in enumerate(convert_pre_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.input_layernorm.weight': layer})
      del convert_pre_ln

      logging.info("_convert_attn: post_self_attention_layer_norm...")
      post_ln = params['base']['decoder']['layers']['post_self_attention_layer_norm']['scale']
      converted_post_ln = _transpose_unstack(post_ln)
      assert len(converted_post_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(converted_post_ln)}"
      for i, layer in enumerate(converted_post_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.post_attention_layernorm.weight': layer})      
      del post_ln

      logging.info("_convert_attn: self_attention (qkv/o/norms)...")
      attn = params['base']['decoder']['layers']['self_attention']
      self_attn = self._to_attn(attn)
      for key, layers in self_attn.items():
        self.vllm_state.update({f'vllm_model.model.layers.{i}.{key}': layer for i, layer in enumerate(layers)})
      del attn
      logging.info("_convert_attn: done")
      gc.collect()

    def _convert_moe(self, params):
      logging.info("_convert_moe: extracting moe_block...")
      moe = params['base']['decoder']['layers']['moe_block'].to_pure_dict()
      prefix = "vllm_model.model.layers"

      logging.info("_convert_moe: gate weights...")
      self.vllm_state.update({
          f"{prefix}.{i}.mlp.gate.weight": w 
          for i, w in enumerate(self._to_mlp_gate(moe['gate']['kernel']))
      })
      del moe['gate']['kernel']
      gc.collect()

      logging.info("_convert_moe: expert down (w2) weights...")
      self.vllm_state.update({
          f"{prefix}.{i}.mlp.experts.w2_weight": w 
          for i, w in enumerate(self._to_mlp_expert_down(moe['wo']))
      })
      del moe['wo']
      gc.collect()

      logging.info("_convert_moe: expert gate+up (w13) weights (fuse_all jit+vmap)...")
      self._to_mlp_expert_gate_up(
          moe['wi_0'], moe['wi_1'], 
          self.num_layers, prefix, 'mlp.experts.w13_weight'
      )
      del moe['wi_0'], moe['wi_1']
      logging.info("_convert_moe: done")
      gc.collect()
      
    def _to_final_norm(self, params):
      self.vllm_state["vllm_model.model.norm.weight"] = params['base']['decoder']['decoder_norm']['scale']

    def _to_embed_tokens(self, params):
      self.vllm_state["vllm_model.model.embed_tokens.weight"] = params['base']['token_embedder']['embedding']

    def _to_lm_head(self, params):
      @jax.jit
      def _transpose(x):
        return jnp.transpose(x, (1, 0))
      self.vllm_state["vllm_model.lm_head.weight"] = _transpose(
          params['base']['decoder']['logits_dense']['kernel']
      )
      
    def _to_attn(self, attn: PyTree) -> dict[str, jax.Array]:
      tp = min(self.vllm_tp, self.config.base_num_kv_heads)  # Don't TP-shard more heads than exist in the model.

      @jax.jit
      def _compute(attn):
        # (d_model, l, h, d) -> (l, d_model, h, d)
        q = jnp.transpose(attn['query']['kernel'], (1, 0, 2, 3))
        k = jnp.transpose(attn['key']['kernel'], (1, 0, 2, 3))
        v = jnp.transpose(attn['value']['kernel'], (1, 0, 2, 3))

        num_q_heads = q.shape[2]
        num_kv_heads = k.shape[2]
        head_dim = q.shape[3]
        l, d_model = q.shape[0], q.shape[1]

        kv_per_tp = num_kv_heads // tp
        q_per_tp = num_q_heads // tp

        q_by_tp = q.reshape(l, d_model, tp, q_per_tp, head_dim)
        k_by_tp = k.reshape(l, d_model, tp, kv_per_tp, head_dim)
        v_by_tp = v.reshape(l, d_model, tp, kv_per_tp, head_dim)

        qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)
        qkv_flat = qkv_by_tp.reshape(l, d_model, -1)
        qkv_proj = jnp.transpose(qkv_flat, (0, 2, 1))

        o = jnp.transpose(attn['out']['kernel'], (1, 3, 0, 2))
        o_proj = o.reshape(o.shape[0], o.shape[1], -1)

        q_norm = jnp.transpose(attn['query_norm']['scale'], (1, 0))
        k_norm = jnp.transpose(attn['key_norm']['scale'], (1, 0))

        return {
            "self_attn.qkv_proj.weight": jnp.unstack(qkv_proj),
            "self_attn.o_proj.weight": jnp.unstack(o_proj),
            "self_attn.q_norm.weight": jnp.unstack(q_norm),
            "self_attn.k_norm.weight": jnp.unstack(k_norm)
        }

      return _compute(attn)
    
    def _to_mlp_gate(self, param):
      # param: [d_model, l, total_e] -> [l, total_e, d_model]
      # shard_map removed: plain transpose lets GSPMD propagate sharding
      # without requiring param and mesh to be on the same device set.
      @jax.jit
      def _transpose(param):
        return jnp.transpose(param, (1, 2, 0))
      param = _transpose(param)
      return self._unstack_layer(param)

    def _to_mlp_expert_down(self, param):
      # param: [E, L, Hidden, Inter] -> [L, E, Inter, Hidden]
      @jax.jit
      def _transpose(param):
        return jnp.transpose(param, (1, 0, 3, 2))
      param = _transpose(param)
      # vLLM 0.17+ expects (E, Hidden, Inter) -> (E, Inter, Hidden)
      # So for each layer, do param[i]: (E, Inter, Hidden) -> (E, Hidden, Inter)
      param = jnp.transpose(param, (0, 1, 3, 2))
      return self._unstack_layer(param)

    def _to_mlp_expert_gate_up(self, wi_0, wi_1, num_layers, layer_key_prefix, layer_key_suffix):
      # Process all layers in one JIT call using vmap to avoid per-layer dispatch
      # overhead (which was ~50 separate device syncs on multi-host v5p-64).
      tp = self.vllm_tp
      
      @jax.jit
      def _fuse_all(wi_0, wi_1):
        # wi_0, wi_1: (e, l, d_model, d_inner) -> (l, e, d_model, d_inner)
        wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))  # -> (l, e, d_model, d_inner)
        wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

        def _fuse_single(w0, w1):
          # [e, d_model, d_inner] -> [e, 2*d_inner, d_model]
          w0 = jnp.transpose(w0, (0, 2, 1))  # gate: [e, d_inner, d_model]
          w1 = jnp.transpose(w1, (0, 2, 1))  # up:   [e, d_inner, d_model]
          e, d_inner, d_model = w0.shape
          # Chunk-level interleave to match vLLM TP sharding:
          # layout: [gate_chunk0, up_chunk0, gate_chunk1, up_chunk1, ...]
          chunk_size = d_inner // tp
          gate_chunks = w0.reshape(e, tp, chunk_size, d_model)
          up_chunks = w1.reshape(e, tp, chunk_size, d_model)
          combined = jnp.stack([gate_chunks, up_chunks], axis=2)
          return combined.reshape(e, 2 * d_inner, d_model)

        return jax.vmap(_fuse_single)(wi_0, wi_1)  # -> (l, e, 2*d_inner, d_model)

      logging.info("_to_mlp_expert_gate_up: dispatching _fuse_all (single JIT+vmap)...")
      fused = _fuse_all(wi_0, wi_1)
      logging.info("_to_mlp_expert_gate_up: _fuse_all complete, shape=%s, unstacking layers...", fused.shape)
      del wi_0, wi_1
      gc.collect()

      # vLLM 0.17+ expects (e, 2*d_inner, d_model) -> (e, d_model, 2*d_inner)
      for i, layer_i in enumerate(jnp.unstack(fused, axis=0)):
        layer_i = jnp.transpose(layer_i, (0, 2, 1))  # (e, 2*d_inner, d_model) -> (e, d_model, 2*d_inner)
        self.vllm_state[f"{layer_key_prefix}.{i}.{layer_key_suffix}"] = layer_i
        if i % 8 == 7:
          gc.collect()
      del fused
      gc.collect()
      
    @staticmethod
    @jax.jit
    def _unstack_layer(param):
        return jnp.unstack(param, axis=0)


class Gemma4ToVLLMConverter:
    """Converts MaxText Gemma4 weights to the layout expected by a vLLM Gemma4 model.

    Supports both gemma4-26b (MoE: 128 routed + 1 shared expert) and
    gemma4-31b (Dense).

    MaxText Gemma4 stores layers in a scanned-block structure:
      state['base']['decoder']['scanned_blocks']['layers_{slot}']
    where slot ∈ [0..5].  Slots 0–4 are local-sliding-window attention layers
    and slot 5 is a global attention layer.  The 'L' dimension (axis 1 of each
    weight tensor) holds 'num_reps = num_layers // 6' repetitions of each slot.
    Final vLLM layer index = rep * 6 + slot.

    Global attention (slot 5) uses a shared KV projection — 'key' serves as
    both K and V; there is no separate 'value' tensor.

    Key names and tensor transformations are derived from the MaxText HF param mapping
    at src/maxtext/checkpoint_conversion/utils/param_mapping.py.

    Attention: Gemma4 uses SEPARATE q/k/v proj weights (not fused QKV).
    MoE (26B): gate+up proj are fused into experts.gate_up_proj (E, 2*d_inner, d_model).
    Embedding: MaxText stores embedding * sqrt(d_model); divide out before writing to vLLM.
    """

    NUM_SLOTS = 6  # 5 local + 1 global

    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.num_layers = config.base_num_decoder_layers
        assert self.num_layers % self.NUM_SLOTS == 0, (
            f"num_layers {self.num_layers} must be divisible by {self.NUM_SLOTS}"
        )
        self.num_reps = self.num_layers // self.NUM_SLOTS
        self.vllm_state = {}
        self.is_moe = (config.model_name == "gemma4-26b")
        self.d_model = config.base_emb_dim
        self.vllm_tp = self.config.rollout_tensor_parallelism

    # --- 1. Top-Level Entry Point ---

    def convert(self, model_state: dict):
        logging.info(f"\n{GREEN}Starting Gemma4 Conversion (is_moe={self.is_moe}, "
                     f"num_layers={self.num_layers}, num_reps={self.num_reps})...{RESET}")
        blocks = model_state['base']['decoder']['scanned_blocks']
        prefix = "vllm_model.model.layers"

        with timer("Convert Global Weights"):
            self._convert_global(model_state)
        with timer("Convert Layer Norms"):
            self._convert_norms(blocks, prefix)
        with timer("Convert Attention Weights"):
            self._convert_attn_weights(blocks, prefix)
        if self.is_moe:
            with timer("Convert MoE Weights"):
                self._convert_moe_weights(blocks, prefix)
        else:
            with timer("Convert Dense MLP Weights"):
                self._convert_dense_mlp_weights(blocks, prefix)

        return self.vllm_state

    @staticmethod
    @jax.jit
    def _pack_attn(q, k, v, o, qnorm, knorm):
        """Prepares separate q/k/v, o, and norms for all layers in a slot.
        Input shapes (MaxText scanned):
          q/k/v: (d_model, L, nH, D)
          o:     (nH, D, L, d_model)
          norms: (D, L)
        Returns: L × (nH*D, d_model) for q/k/v, etc.
        """
        # q/k/v: (d_model, L, nH, D) -> (L, nH, D, d_model) -> (L, nH*D, d_model)
        q = jnp.transpose(q, (1, 2, 3, 0)).reshape(q.shape[1], -1, q.shape[0])
        k = jnp.transpose(k, (1, 2, 3, 0)).reshape(k.shape[1], -1, k.shape[0])
        v = jnp.transpose(v, (1, 2, 3, 0)).reshape(v.shape[1], -1, v.shape[0])
        
        # o: (nH, D, L, d_model) -> (L, d_model, nH, D) -> (L, d_model, nH*D)
        o = jnp.transpose(o, (2, 3, 0, 1)).reshape(o.shape[2], o.shape[3], -1)
        
        # norms: (D, L) -> (L, D)
        qnorm = jnp.transpose(qnorm, (1, 0))
        knorm = jnp.transpose(knorm, (1, 0))
        
        return (jnp.unstack(q), jnp.unstack(k), jnp.unstack(v), 
                jnp.unstack(o), jnp.unstack(qnorm), jnp.unstack(knorm))

    # --- 2. Global (non-per-layer) weights ---

    def _convert_global(self, params):
        # Gemma4 uses tied embeddings: no logits_dense; lm_head.weight = embed_tokens.weight.
        # MaxText stores embedding pre-multiplied by sqrt(d_model) (applied at runtime in HF/vLLM).
        # Divide it out so vLLM gets the raw embedding and can apply its own normalizer.
        logging.info("_convert_global: embed_tokens (de-normalize) + lm_head (tied) + final_norm...")
        normalizer = self.d_model ** 0.5

        @jax.jit
        def _denorm_embed(x):
            return (x / normalizer).astype(x.dtype)

        raw_embedding = _denorm_embed(params['base']['token_embedder']['embedding'])
        self.vllm_state["vllm_model.model.embed_tokens.weight"] = raw_embedding
        self.vllm_state["vllm_model.lm_head.weight"] = raw_embedding  # tied
        self.vllm_state["vllm_model.model.norm.weight"] = (
            params['base']['decoder']['decoder_norm']['scale']
        )
        logging.info("_convert_global: done")

    # --- 3. Per-layer norms ---

    def _convert_norms(self, blocks, prefix):
        """Converts all 4 per-layer norm vectors across all layers."""
        @jax.jit
        def _unstack_norm(x):
            # x: (d_model, L) -> L tensors of (d_model,)
            return jnp.unstack(x, axis=1)

        for slot in range(self.NUM_SLOTS):
            slot_data = blocks[f'layers_{slot}']
            pre_attn  = _unstack_norm(slot_data['pre_self_attention_norm']['scale'])
            post_attn = _unstack_norm(slot_data['post_self_attention_norm']['scale'])
            pre_ffw   = _unstack_norm(slot_data['pre_ffw_norm']['scale'])
            post_ffw  = _unstack_norm(slot_data['post_ffw_norm']['scale'])
            for rep in range(self.num_reps):
                i = rep * self.NUM_SLOTS + slot
                self.vllm_state[f"{prefix}.{i}.input_layernorm.weight"]            = pre_attn[rep]
                self.vllm_state[f"{prefix}.{i}.post_attention_layernorm.weight"]   = post_attn[rep]
                self.vllm_state[f"{prefix}.{i}.pre_feedforward_layernorm.weight"]  = pre_ffw[rep]
                self.vllm_state[f"{prefix}.{i}.post_feedforward_layernorm.weight"] = post_ffw[rep]
            del pre_attn, post_attn, pre_ffw, post_ffw
        gc.collect()

    # --- 4. Per-layer attention weights ---

    def _convert_attn_weights(self, blocks, prefix):
        """Converts separate q/k/v proj, o proj, q-norm, k-norm for all layers.

        HF/vLLM Gemma4 uses separate projections (not fused QKV).  Global attention
        layers (slot 5) have no 'value' tensor; vLLM sets v_proj = k_proj.

        Tensor transformations (MaxText → HF, i.e. saving_to_hf=True in reshape_kernel):
          q/k/v kernel: (d_model, nH, D) → (nH*D, d_model)  [reshape then transpose]
          out kernel:   (nH, D, d_model) → (d_model, nH*D)   [reshape then transpose]
          norms:        (D,)             → (D,)               [identity]
        """
        @jax.jit
        def _pack_local(attn):
            # q/k/v: (d_model, L, nH, D)
            q = attn['query']['kernel']
            k = attn['key']['kernel']
            v = attn['value']['kernel']
            return Gemma4ToVLLMConverter._pack_attn(
                q, k, v, attn['out']['kernel'],
                attn['query_norm']['scale'], attn['key_norm']['scale']
            )

        @jax.jit
        def _pack_global(attn):
            # Global: no 'value'; key used as both K and V (shared KV projection).
            q = attn['query']['kernel']
            k = attn['key']['kernel']
            return Gemma4ToVLLMConverter._pack_attn(
                q, k, k, attn['out']['kernel'],
                attn['query_norm']['scale'], attn['key_norm']['scale']
            )

        for slot in range(self.NUM_SLOTS):
            is_global = (slot == self.NUM_SLOTS - 1)
            attn = blocks[f'layers_{slot}']['self_attention']
            pack_fn = _pack_global if is_global else _pack_local
            q_layers, k_layers, v_layers, o_layers, qnorm_layers, knorm_layers = pack_fn(attn)
            for rep in range(self.num_reps):
                i = rep * self.NUM_SLOTS + slot
                self.vllm_state[f"{prefix}.{i}.self_attn.q_proj.weight"] = q_layers[rep]
                self.vllm_state[f"{prefix}.{i}.self_attn.k_proj.weight"] = k_layers[rep]
                self.vllm_state[f"{prefix}.{i}.self_attn.v_proj.weight"] = v_layers[rep]
                self.vllm_state[f"{prefix}.{i}.self_attn.o_proj.weight"] = o_layers[rep]
                self.vllm_state[f"{prefix}.{i}.self_attn.q_norm.weight"] = qnorm_layers[rep]
                self.vllm_state[f"{prefix}.{i}.self_attn.k_norm.weight"] = knorm_layers[rep]
            del q_layers, k_layers, v_layers, o_layers, qnorm_layers, knorm_layers
        gc.collect()

    # --- 5a. MoE weights (gemma4-26b only) ---

    def _convert_moe_weights(self, blocks, prefix):
        """Converts router, routed experts (fused gate_up_proj), shared expert, MoE norms (26B).

        Tensor transformations:
          router.proj.weight:       gate.kernel (d_model, L, E) → (E, d_model)
          router.scale:             pre_forward_scale_2 (d_model, L) → (d_model,)  [identity]
          router.per_expert_scale:  per_expert_scale (E, L) → (E,)  [identity]
          experts.gate_up_proj:     fuse wi_0+wi_1 (E, L, d_model, d_inner) → (E, 2*d_inner, d_model)
          experts.down_proj:        wo (E, L, d_inner, d_model) → (E, d_model, d_inner)
          shared mlp.*:             (d_model, L, d_sh) or (d_sh, L, d_model) → HF convention
          extra norms:              (d_model, L) → (d_model,)  [identity per layer]
        """
        @functools.partial(jax.jit, static_argnames=['vllm_tp'])
        def _pack_moe(routed, shared, extra, vllm_tp):
            L = routed['wi_0'].shape[1]
            E = routed['wi_0'].shape[0]
            d_inner = routed['wi_0'].shape[3]
            d_model = routed['wi_0'].shape[2]

            # Router proj: (d_model, L, E) -> L × (E, d_model)
            router_proj = jnp.unstack(
                jnp.transpose(routed['gate']['kernel'], (1, 2, 0)), axis=0
            )
            # Router scale: (d_model, L) -> L × (d_model,)  [identity — no reshape_kernel]
            router_scale = jnp.unstack(extra['pre_forward_scale_2'], axis=1)
            # Per-expert scale: (E, L) -> L × (E,)
            per_expert_scale = jnp.unstack(routed['per_expert_scale'], axis=1)

            # Fused gate+up proj for routed experts (TP-interleaved):
            #   wi_0 (gate): (E, L, d_model, d_inner) -> (L, E, TP, d_inner//TP, d_model)
            #   wi_1 (up):   (E, L, d_model, d_inner) -> (L, E, TP, d_inner//TP, d_model)
            #   stack along new axis 3: (L, E, TP, 2, d_inner//TP, d_model)
            #   reshape: (L, E, 2*d_inner, d_model) = gate_up_proj
            w0 = jnp.transpose(routed['wi_0'], (1, 0, 3, 2)).reshape(L, E, vllm_tp, -1, d_model)
            w1 = jnp.transpose(routed['wi_1'], (1, 0, 3, 2)).reshape(L, E, vllm_tp, -1, d_model)
            combined = jnp.stack([w0, w1], axis=3)
            gate_up = combined.reshape(L, E, 2 * d_inner, d_model)
            gate_up_proj = jnp.unstack(gate_up, axis=0)

            # Down proj: (E, L, d_inner, d_model) -> L × (E, d_model, d_inner)
            down_proj = jnp.unstack(jnp.transpose(routed['wo'], (1, 0, 3, 2)), axis=0)

            # Shared expert:
            #   wi_0/wi_1: (d_model, L, d_sh) -> L × (d_sh, d_model)
            #   wo:        (d_sh, L, d_model)  -> L × (d_model, d_sh)
            sh_gate = jnp.unstack(jnp.transpose(shared['wi_0']['kernel'], (1, 2, 0)), axis=0)
            sh_up   = jnp.unstack(jnp.transpose(shared['wi_1']['kernel'], (1, 2, 0)), axis=0)
            sh_down = jnp.unstack(jnp.transpose(shared['wo']['kernel'],   (1, 2, 0)), axis=0)

            # Extra MoE norms: (d_model, L) -> L × (d_model,)
            pre_ln_2  = jnp.unstack(extra['pre_feedforward_layernorm_2']['scale'], axis=1)
            post_ln_1 = jnp.unstack(extra['post_feedforward_layernorm_1']['scale'], axis=1)
            post_ln_2 = jnp.unstack(extra['post_feedforward_layernorm_2']['scale'], axis=1)

            return (router_proj, router_scale, per_expert_scale,
                    gate_up_proj, down_proj,
                    sh_gate, sh_up, sh_down,
                    pre_ln_2, post_ln_1, post_ln_2)

        for slot in range(self.NUM_SLOTS):
            moe_block = blocks[f'layers_{slot}']['mlp']['moe_block']
            routed = moe_block['MoeBlock_0']
            shared = moe_block['shared_experts']
            extra  = blocks[f'layers_{slot}']['mlp']
            (router_proj, router_scale, per_expert_scale,
             gate_up_proj, down_proj,
             sh_gate, sh_up, sh_down,
             pre_ln_2, post_ln_1, post_ln_2) = _pack_moe(routed, shared, extra, self.vllm_tp)

            for rep in range(self.num_reps):
                i = rep * self.NUM_SLOTS + slot
                p = f"{prefix}.{i}"
                # Router
                self.vllm_state[f"{p}.router.proj.weight"]        = router_proj[rep]
                self.vllm_state[f"{p}.router.scale"]              = router_scale[rep]
                self.vllm_state[f"{p}.router.per_expert_scale"]   = per_expert_scale[rep]
                # Routed experts (fused gate+up, separate down)
                self.vllm_state[f"{p}.experts.gate_up_proj"] = gate_up_proj[rep]
                self.vllm_state[f"{p}.experts.down_proj"]    = down_proj[rep]
                # Shared expert (uses mlp.* keys — same as Dense MLP naming)
                self.vllm_state[f"{p}.mlp.gate_proj.weight"] = sh_gate[rep]
                self.vllm_state[f"{p}.mlp.up_proj.weight"]   = sh_up[rep]
                self.vllm_state[f"{p}.mlp.down_proj.weight"] = sh_down[rep]
                # Extra MoE norms
                self.vllm_state[f"{p}.pre_feedforward_layernorm_2.weight"]  = pre_ln_2[rep]
                self.vllm_state[f"{p}.post_feedforward_layernorm_1.weight"] = post_ln_1[rep]
                self.vllm_state[f"{p}.post_feedforward_layernorm_2.weight"] = post_ln_2[rep]

            del router_proj, router_scale, per_expert_scale, gate_up_proj, down_proj
            del sh_gate, sh_up, sh_down, pre_ln_2, post_ln_1, post_ln_2
            gc.collect()

    # --- 5b. Dense MLP weights (gemma4-31b only) ---

    def _convert_dense_mlp_weights(self, blocks, prefix):
        """Converts gate/up/down projections for all layers (31B only).

        Tensor transformations:
          wi_0 (gate): (d_model, L, d_mlp) → L × (d_mlp, d_model)
          wi_1 (up):   (d_model, L, d_mlp) → L × (d_mlp, d_model)
          wo  (down):  (d_mlp,  L, d_model) → L × (d_model, d_mlp)
        """
        @jax.jit
        def _pack_mlp(mlp):
            # wi_0 (gate): (d_model, L, d_mlp) -> L × (d_mlp, d_model)
            gate = jnp.unstack(jnp.transpose(mlp['wi_0']['kernel'], (1, 2, 0)), axis=0)
            # wi_1 (up):   (d_model, L, d_mlp) -> L × (d_mlp, d_model)
            up   = jnp.unstack(jnp.transpose(mlp['wi_1']['kernel'], (1, 2, 0)), axis=0)
            # wo  (down):  (d_mlp,  L, d_model) -> L × (d_model, d_mlp)
            down = jnp.unstack(jnp.transpose(mlp['wo']['kernel'], (1, 2, 0)), axis=0)
            return gate, up, down

        for slot in range(self.NUM_SLOTS):
            mlp = blocks[f'layers_{slot}']['mlp']
            gate_layers, up_layers, down_layers = _pack_mlp(mlp)
            for rep in range(self.num_reps):
                i = rep * self.NUM_SLOTS + slot
                p = f"{prefix}.{i}"
                # TODO: vLLM Gemma4 Dense may fuse gate+up into gate_up_proj (like Gemma3).
                # Verify naming once vLLM Gemma4 is available.
                self.vllm_state[f"{p}.mlp.gate_proj.weight"] = gate_layers[rep]
                self.vllm_state[f"{p}.mlp.up_proj.weight"]   = up_layers[rep]
                self.vllm_state[f"{p}.mlp.down_proj.weight"] = down_layers[rep]
            del gate_layers, up_layers, down_layers
            gc.collect()


def save_dict_to_file(dict, filename):
    with open(filename, 'w') as f:
        for key in sorted(dict.keys()):
            f.write(f"{key}: {dict[key].shape}\n")


class DeepSeekV3ToVLLMConverter:
    """Converts MaxText DeepSeekV3 weights to the layout expected by vLLM."""

    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.num_layers = config.base_num_decoder_layers
        self.vllm_state = {}
        self.vllm_tp = self.config.rollout_tensor_parallelism

    def convert(self, model_state: dict):
        logging.info(f"\n{GREEN}Starting DeepSeekV3 Conversion...{RESET}")
        
        with timer("Convert Global Weights"):
            self._convert_global(model_state)
        
        with timer("Convert Layer Weights"):
            self._convert_layers(model_state)
            
        return self.vllm_state

    def _convert_global(self, params):
        logging.info("_convert_global: embed_tokens...")
        self.vllm_state["vllm_model.model.embed_tokens.weight"] = params['base']['token_embedder']['embedding']
        
        logging.info("_convert_global: final_norm...")
        self.vllm_state["vllm_model.model.norm.weight"] = params['base']['decoder']['decoder_norm']['scale']
        
        logging.info("_convert_global: lm_head...")
        @jax.jit
        def _transpose(x):
            return jnp.transpose(x, (1, 0))
        self.vllm_state["vllm_model.lm_head.weight"] = _transpose(params['base']['decoder']['logits_dense']['kernel'])

    def _convert_layers(self, params):
        layers = params['base']['decoder']['layers']
        prefix = "vllm_model.model.layers"

        # 1. Layer Norms
        logging.info("_convert_layers: layer norms...")
        @jax.jit
        def _unstack_norm(x):
            return jnp.unstack(jnp.transpose(x, (1, 0)))
        
        input_norms = _unstack_norm(layers['pre_self_attention_layer_norm']['scale'])
        post_attn_norms = _unstack_norm(layers['post_self_attention_layer_norm']['scale'])
        
        for i in range(self.num_layers):
            self.vllm_state[f"{prefix}.{i}.input_layernorm.weight"] = input_norms[i]
            self.vllm_state[f"{prefix}.{i}.post_attention_layernorm.weight"] = post_attn_norms[i]
        
        # 2. MLA Attention
        logging.info("_convert_layers: MLA attention...")
        self._convert_mla(layers['self_attention'], prefix)

        # 3. MoE / MLP
        logging.info("_convert_layers: MoE / MLP...")
        if 'moe_block' in layers:
            self._convert_moe(layers['moe_block'], prefix)
        elif 'mlp' in layers:
            # Handle dense MLP if any (though DSV3 is mostly MoE)
            self._convert_dense_mlp(layers['mlp'], prefix)
        
    def _convert_mla(self, attn, prefix):
        tp = self.vllm_tp

        @jax.jit
        def _process_mla(attn):
            # wq_a: (d_model, L, Rank) -> (L, Rank, d_model)
            wq_a = jnp.transpose(attn['wq_a']['kernel'], (1, 2, 0))
            
            # q_norm: (Rank, L) -> (L, Rank)
            q_norm = jnp.transpose(attn['q_norm']['scale'], (1, 0))

            # wq_b: (Rank, L, Heads, HeadDim) -> (L, Heads * HeadDim, Rank)
            wq_b = jnp.transpose(attn['wq_b']['kernel'], (1, 2, 3, 0))
            l, nh, dh, r = wq_b.shape
            wq_b = wq_b.reshape(l, nh * dh, r)
            
            # wkv_a: (d_model, L, Rank + 2 * qk_head_dim) -> (L, Rank + 2 * qk_head_dim, d_model)
            wkv_a = jnp.transpose(attn['wkv_a']['kernel'], (1, 2, 0))
            
            # kv_norm: (Rank, L) -> (L, Rank)
            kv_norm = jnp.transpose(attn['kv_norm']['scale'], (1, 0))

            # wkv_b: (Rank, L, Heads, HeadDim + qk_head_dim) -> (L, Heads * (HeadDim + qk_head_dim), Rank)
            wkv_b = jnp.transpose(attn['wkv_b']['kernel'], (1, 2, 3, 0))
            l, nh, dh_total, r = wkv_b.shape
            wkv_b = wkv_b.reshape(l, nh * dh_total, r)
            
            # out: (Heads, HeadDim, L, d_model) -> (L, d_model, Heads * HeadDim)
            wo = jnp.transpose(attn['out']['kernel'], (2, 3, 0, 1))
            l, dm, nh, dh = wo.shape
            wo = wo.reshape(l, dm, nh * dh)
            # Standard is (out_features, in_features), so (d_model, Heads * HeadDim).
            # vLLM expects o_proj.weight as (d_model, Heads * HeadDim)
            
            return {
                "q_a_proj.weight": jnp.unstack(wq_a),
                "q_b_proj.weight": jnp.unstack(wq_b),
                "kv_a_proj_with_mqa.weight": jnp.unstack(wkv_a),
                "kv_b_proj.weight": jnp.unstack(wkv_b),
                "o_proj.weight": jnp.unstack(wo),
                "q_a_layernorm.weight": jnp.unstack(q_norm),
                "kv_a_layernorm.weight": jnp.unstack(kv_norm),
            }

        mla_weights = _process_mla(attn)
        for key, layers in mla_weights.items():
            for i, weight in enumerate(layers):
                self.vllm_state[f"{prefix}.{i}.self_attn.{key}"] = weight

    def _convert_moe(self, moe, prefix):
        tp = self.vllm_tp

        logging.info("_convert_moe: gate...")
        @jax.jit
        def _process_gate(gate_kernel):
            # (emb, L, E) -> (L, E, emb)
            return jnp.unstack(jnp.transpose(gate_kernel, (1, 2, 0)))
        
        gate_layers = _process_gate(moe['MoeBlock_0']['gate']['kernel'])
        for i, w in enumerate(gate_layers):
            self.vllm_state[f"{prefix}.{i}.mlp.gate.weight"] = w

        logging.info("_convert_moe: routed experts...")
        @jax.jit
        def _process_routed(wi_0, wi_1, wo):
            # wi_0: (E, L, d_model, d_inner) -> (L, E, d_inner, d_model)
            l = wi_0.shape[1]
            e = wi_0.shape[0]
            dm = wi_0.shape[2]
            di = wi_0.shape[3]
            
            w0 = jnp.transpose(wi_0, (1, 0, 3, 2)) # (L, E, d_inner, d_model)
            w1 = jnp.transpose(wi_1, (1, 0, 3, 2)) # (L, E, d_inner, d_model)
            
            # Interleave for TP
            chunk_size = di // tp
            w0_chunks = w0.reshape(l, e, tp, chunk_size, dm)
            w1_chunks = w1.reshape(l, e, tp, chunk_size, dm)
            gate_up = jnp.stack([w0_chunks, w1_chunks], axis=3).reshape(l, e, 2 * di, dm)
            # vLLM expects (E, d_model, 2*d_inner)
            gate_up = jnp.transpose(gate_up, (0, 1, 3, 2))
            
            # wo: (E, L, d_inner, d_model) -> (L, E, d_model, d_inner)
            down = jnp.transpose(wo, (1, 0, 3, 2))
            
            return jnp.unstack(gate_up), jnp.unstack(down)

        gate_up_layers, down_layers = _process_routed(moe['MoeBlock_0']['wi_0'], moe['MoeBlock_0']['wi_1'], moe['MoeBlock_0']['wo'])
        
        for i in range(self.num_layers):
            self.vllm_state[f"{prefix}.{i}.mlp.experts.gate_up_proj"] = gate_up_layers[i]
            self.vllm_state[f"{prefix}.{i}.mlp.experts.down_proj"] = down_layers[i]

        logging.info("_convert_moe: shared experts...")
        @jax.jit
        def _process_shared(wi_0, wi_1, wo):
            # (emb, L, d_shared) -> (L, d_shared, emb)
            l = wi_0.shape[1]
            dm = wi_0.shape[0]
            ds = wi_0.shape[2]

            w0 = jnp.transpose(wi_0, (1, 2, 0))
            w1 = jnp.transpose(wi_1, (1, 2, 0))
            
            # Interleave for TP? Shared experts might also be TPed in vLLM.
            # If so, they are usually treated as a single dense MLP.
            chunk_size = ds // tp
            w0_chunks = w0.reshape(l, tp, chunk_size, dm)
            w1_chunks = w1.reshape(l, tp, chunk_size, dm)
            gate_up = jnp.stack([w0_chunks, w1_chunks], axis=2).reshape(l, 2 * ds, dm)
            # vLLM expects (2*d_shared, d_model) -> (d_model, 2*d_shared)
            gate_up = jnp.transpose(gate_up, (0, 2, 1))

            # wo: (d_shared, L, d_model) -> (L, d_model, d_shared)
            down = jnp.transpose(wo, (1, 2, 0))
            
            return jnp.unstack(gate_up), jnp.unstack(down)

        sh_gate_up, sh_down = _process_shared(
            moe['shared_experts']['wi_0']['kernel'],
            moe['shared_experts']['wi_1']['kernel'],
            moe['shared_experts']['wo']['kernel']
        )
        
        for i in range(self.num_layers):
            self.vllm_state[f"{prefix}.{i}.mlp.shared_experts.gate_up_proj.weight"] = sh_gate_up[i]
            self.vllm_state[f"{prefix}.{i}.mlp.shared_experts.down_proj.weight"] = sh_down[i]

    def _convert_dense_mlp(self, mlp, prefix):
        # Implementation for dense MLP layers if needed
        pass



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
  base_yaml_path = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  config, model, mesh = _load_maxtext_model(base_yaml_path)
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {config.model_name}")
  print(f"Mesh: {mesh}")

  # Convert weights to VLLM format
  print("="*80)
  print("Converting weights to VLLM format")
  print("="*80)
  model_state = nnx.state(model, nnx.Not(nnx.RngState))
  
  if 'to_nnx__rngs' in model_state:
    del model_state['to_nnx__rngs']

  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, 'shape') and hasattr(leaf, 'sharding'):
      path_str = jax.tree_util.keystr(path)
      logging.info(f"Name: {path_str}, shape: {leaf.shape}")
      logging.info(f"\tSharding: {leaf.sharding}")
  
  if config.model_name.startswith("gemma4"):
    converter = Gemma4ToVLLMConverter(config, mesh)
  elif config.model_name.startswith("deepseek3"):
    converter = DeepSeekV3ToVLLMConverter(config, mesh)
  else:
    converter = MaxTextToVLLMConverter(config, mesh)


  with timer("Overall Conversion "):
    vllm_state = converter.convert(model_state)
  del model_state
  del model
  del mesh
  gc.collect()
  # save_dict_to_file(vllm_state, "vllm_state_shapes.txt")

  # Load vLLM model and run generation test
  print("="*80)
  print("Loading vLLM model for generation test...")
  print("="*80)
  llm = LLM(
    _VLLM_MODEL_ID.value,
    max_model_len=16,
    tensor_parallel_size=_ROLLOUT_TP.value,
    data_parallel_size=_FSDP_TP.value,
    gpu_memory_utilization=0.35,
    async_scheduling=False,
    enforce_eager=True,
  )
  print("\n" + "="*80)
  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
  # save_dict_to_file(llm_state, "llm_state_shapes.txt")

  # Detect once whether MaxText and vLLM meshes share the same physical
  # devices (single-VM) or are disjoint (multi-host cluster partition).
  # Same devices → jax.jit with out_shardings lets XLA emit on-device
  # collectives (fast, ~0.16 s for 30 B).  Disjoint devices → jax.jit
  # raises "incompatible devices"; fall back to jax.device_put which uses
  # ICI/DCN for the cross-host transfer without a CPU roundtrip.
  _any_src = next(iter(vllm_state.values()))
  _any_src_arr = _any_src.value if hasattr(_any_src, 'value') else _any_src
  _any_dst = next(iter(llm_state.values()))
  _same_devices = (
      frozenset(d.id for d in _any_src_arr.sharding.mesh.devices.flat) ==
      frozenset(d.id for d in _any_dst.sharding.mesh.devices.flat)
  )
  logging.info("Weight sync: same_devices=%s (jit=%s, device_put=%s)",
               _same_devices, _same_devices, not _same_devices)

  @functools.lru_cache(maxsize=None)
  def _get_reshard_fn(dst_sharding):
    if _same_devices:
      return jax.jit(lambda x: x, out_shardings=dst_sharding)
    else:
      return functools.partial(jax.device_put, device=dst_sharding)

  with timer(f"Assigning {len(vllm_state)} weights to vLLM model"):
    for key, weight in vllm_state.items():
      # Unwrap NNX Param → plain jax.Array; plain arrays pass through unchanged.
      weight_array = weight.value if hasattr(weight, 'value') else weight
      dst_sharding = llm_state[key].sharding
      # print(f"Assigning {key}: src shape={weight_array.shape}, src sharding={weight_array.sharding}; dst shape={llm_state[key].shape}, dst sharding={dst_sharding}")
      llm_state[key] = _get_reshard_fn(dst_sharding)(weight_array)
      # array_allclose = np.allclose(
      #     np.asarray(llm_state[key]), np.asarray(weight_array), rtol=1e-2, atol=1e-2
      # )
      # print(f"{key}: {array_allclose}")
      # llm_state[key] = jax.device_put(np.asarray(weight_array), dst_sharding)
    jax.effects_barrier()  # wait for all on-device reshards to finish


  sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
  print("\n" + "="*80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate("Paris is", sampling_params=sampling_params))


if __name__ == "__main__":
  main()
