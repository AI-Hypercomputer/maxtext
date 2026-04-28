import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from vllm import LLM, SamplingParams
import functools
import gc
import pathwaysutils
import logging
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

from tunix.models.qwen3 import model as qwen3_lib

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
_MODEL_NAME = flags.DEFINE_string('model_name', 'qwen3-30b-a3b', 'Model name')
_TOKENIZER_PATH = flags.DEFINE_string('tokenizer_path', 'Qwen/Qwen3-30B-A3B', 'Tokenizer path')
_TP_SIZE = flags.DEFINE_integer('tp_size', 8, 'tensor parallelism size')
_RAND_INIT = False  # Whether to use random initialization instead of loading from checkpoint, for faster testing
_LOAD_PARAMETERS_PATH = flags.DEFINE_string(
    'load_parameters_path',
    'gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items',
    'Path to load parameters from'
)

_SAVE_WEIGHTS = flags.DEFINE_bool('save_weights', False, 'Save Layer 0 weights for debugging')
_TARGET_LAYER = flags.DEFINE_integer('target_layer', 0, 'Target layer for comparison and saving')
_GCS_BUCKET = flags.DEFINE_string('gcs_bucket', '', 'GCS bucket for uploading saved weights')

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


def _log_mem_stats(tag: str) -> None:
  """Log JAX live-array count/bytes and process RSS.

  Filter in cloud logging with:  textPayload =~ "\\[ROLLOUT_MEM\\]"
  """
  live = jax.live_arrays()
  num_arrays = len(live)
  total_bytes = sum(a.nbytes for a in live if hasattr(a, "nbytes"))
  rss_gb = 0.0
  try:
    with open("/proc/self/status") as _f:
      for _line in _f:
        if _line.startswith("VmRSS:"):
          rss_gb = int(_line.split()[1]) / 1e6
          break
  except OSError:
    pass
  logging.info(
      "[ROLLOUT_MEM] %s | live_arrays=%d jax_bytes=%.3f GB rss=%.3f GB",
      tag, num_arrays, total_bytes / 1e9, rss_gb,
  )


def _get_maxtext_model(config, devices=None):
  """Creates and returns a Tunix-adapted MaxText model and mesh."""
  logging.info(f'Creating model with config: {config}')
  model, mesh = model_creation_utils.create_nnx_model(
    config, model_mode=MODEL_MODE_AUTOREGRESSIVE, devices=devices)
    # config, model_mode=MODEL_MODE_AUTOREGRESSIVE, use_rand_init=_RAND_INIT.value)    
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

def _load_maxtext_model(base_yaml_path, devices=None):
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
  rollout_dp = int(os.environ.get("rollout_data_parallelism", "16"))
  rollout_tp = int(os.environ.get("rollout_tensor_parallelism", str(_TP_SIZE.value)))
  rollout_ep = int(os.environ.get("rollout_expert_parallelism", "1"))

  config_ref = pyconfig.initialize(
      [ "", base_yaml_path, ],
      base_output_directory="gs://wyzhang-dev/tmp",  # Not used in Tunix.
      run_name="test-tunix-maxtext-qwen3",
      tokenizer_type="huggingface",
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
      model_name=_MODEL_NAME.value,
      load_parameters_path=_LOAD_PARAMETERS_PATH.value,
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
      rollout_data_parallelism=rollout_dp,
      rollout_tensor_parallelism=rollout_tp,
      rollout_expert_parallelism=rollout_ep,
      ici_fsdp_parallelism=-1,
      num_samplers_slices=1,
      num_trainer_slices=1,
      num_slices=1,
      # ici_tensor_parallelism=8,
      override_model_config="true",
      checkpoint_storage_concurrent_gb=80,
      async_scheduling="false",
      colocated_python_checkpointing="true",
      enable_single_controller="true",
  )
  model, mesh = _get_maxtext_model(config_ref, devices=devices)
  return config_ref, model, mesh

class MaxTextToVLLMConverter:
    def __init__(self, config, mesh, use_ep: bool = False):
        self.config = config
        self.mesh = mesh
        self.num_layers = config.base_num_decoder_layers
        self.vllm_tp = self.config.rollout_tensor_parallelism
        self.use_ep = use_ep

    # --- 1. Top-Level Entry Point ---
    def convert(self, model_state: dict):
        """Main entry point to convert all weights."""
        logging.info(f"\n{GREEN}Starting Conversion...{RESET}")
        start_time = time.time()
        self.vllm_state = {}
        _log_mem_stats("converter:start")

        with timer("Convert Global Weights"):
          self._convert_global(model_state)
        _log_mem_stats("converter:post_global")
        with timer("Convert Attention Weights"):
          self._convert_attn(model_state)
        _log_mem_stats("converter:post_attn")
        with timer("Convert MoE Weights"):
          self._convert_moe(model_state)
        _log_mem_stats("converter:post_moe")
        
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
      logging.info("_convert_attn: pre_self_attention_layer_norm...")
      pre_ln = params['base']['decoder']['layers']['pre_self_attention_layer_norm']['scale']
      convert_pre_ln = self._transpose_unstack(pre_ln)
      assert len(convert_pre_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(convert_pre_ln)}"
      for i, layer in enumerate(convert_pre_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.input_layernorm.weight': layer})
      del convert_pre_ln

      logging.info("_convert_attn: post_self_attention_layer_norm...")
      post_ln = params['base']['decoder']['layers']['post_self_attention_layer_norm']['scale']
      converted_post_ln = self._transpose_unstack(post_ln)
      assert len(converted_post_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(converted_post_ln)}"
      for i, layer in enumerate(converted_post_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.post_attention_layernorm.weight': layer})      
      del post_ln, converted_post_ln

      logging.info("_convert_attn: self_attention (qkv/o/norms)...")
      attn = params['base']['decoder']['layers']['self_attention']
      self_attn = self._to_attn(attn)
      for key, layers in self_attn.items():
        self.vllm_state.update({f'vllm_model.model.layers.{i}.{key}': layer for i, layer in enumerate(layers)})
      del attn, self_attn
      logging.info("_convert_attn: done")
      gc.collect()

    def _convert_moe(self, params):
      logging.info("_convert_moe: extracting moe_block...")
      _log_mem_stats("converter:moe_start")
      moe = params['base']['decoder']['layers']['moe_block'].to_pure_dict()
      prefix = "vllm_model.model.layers"

      if self.use_ep:
        logging.info("_convert_moe: generating unfused expert weights for EP...")
        wi_0 = moe['wi_0']
        wi_1 = moe['wi_1']
        wo = moe['wo']
        for i in range(self.num_layers):
          self.vllm_state.update({
            f"{prefix}.{i}.mlp.experts.kernel_gating_EDF": jnp.transpose(wi_0[:, i, :, :], (0, 1, 2)),
            f"{prefix}.{i}.mlp.experts.kernel_up_proj_EDF": jnp.transpose(wi_1[:, i, :, :], (0, 1, 2)),
            f"{prefix}.{i}.mlp.experts.kernel_down_proj_EFD": wo[:, i, :, :],
          })
        del wi_0, wi_1, wo
      else:
        logging.info("_convert_moe: gate weights...")
        self.vllm_state.update({
            f"{prefix}.{i}.mlp.gate.weight": w 
            for i, w in enumerate(self._to_mlp_gate(moe['gate']['kernel']))
        })
        del moe['gate']
        gc.collect()
        _log_mem_stats("converter:moe_post_gate")

        logging.info("_convert_moe: expert down (w2) weights...")
        self.vllm_state.update({
            f"{prefix}.{i}.mlp.experts.w2_weight": w 
            for i, w in enumerate(self._to_mlp_expert_down(moe['wo']))
        })
        del moe['wo']
        gc.collect()
        _log_mem_stats("converter:moe_post_w2")

        logging.info("_convert_moe: expert gate+up (w13) weights (fuse_all jit+vmap)...")
        self._to_mlp_expert_gate_up(
            moe['wi_0'], moe['wi_1'], 
            self.num_layers, prefix, 'mlp.experts.w13_weight'
        )
        del moe['wi_0'], moe['wi_1']

      del moe
      logging.info("_convert_moe: done")
      gc.collect()
      _log_mem_stats("converter:moe_post_w13")
      
    def _to_final_norm(self, params):
      # Explicit copy: avoids aliasing the actor model's buffer into vllm_state,
      # which would cause 'Array has been deleted' if the source is ever donated.
      self.vllm_state["vllm_model.model.norm.weight"] = jnp.array(
          params['base']['decoder']['decoder_norm']['scale']
      )

    def _to_embed_tokens(self, params):
      # Explicit copy: avoids aliasing the actor model's buffer into vllm_state.
      self.vllm_state["vllm_model.model.embed_tokens.weight"] = jnp.array(
          params['base']['token_embedder']['embedding']
      )

    def _to_lm_head(self, params):
      self.vllm_state["vllm_model.lm_head.weight"] = self._transpose_2d(
          params['base']['decoder']['logits_dense']['kernel']
      )
      
    def _to_attn(self, attn: PyTree) -> dict[str, jax.Array]:
      tp = min(self.vllm_tp, self.config.base_num_kv_heads)  # Don't TP-shard more heads than exist in the model.
      _compute = self._make_attn_compute(tp)
      return _compute(attn)

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _make_attn_compute(tp: int):
      """Return a JIT-compiled attn converter for a given TP degree.

      Cached by tp so the same XLA executable is reused across all steps.
      """
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

        # vLLM QKVParallelLinear logical layout: [Q_all, K_all, V_all].
        # TP sharding is applied externally via _get_reshard_fn — don't pre-interleave.
        q_flat = q.reshape(l, d_model, num_q_heads * head_dim)
        k_flat = k.reshape(l, d_model, num_kv_heads * head_dim)
        v_flat = v.reshape(l, d_model, num_kv_heads * head_dim)
        qkv_flat = jnp.concatenate([q_flat, k_flat, v_flat], axis=2)
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

      return _compute
    
    def _to_mlp_gate(self, param):
      # param: [d_model, l, total_e] -> [l, total_e, d_model]
      # shard_map removed: plain transpose lets GSPMD propagate sharding
      # without requiring param and mesh to be on the same device set.
      param = self._transpose_gate(param)
      return self._unstack_layer(param)

    def _to_mlp_expert_down(self, param):
      # param: [E, L, Hidden, Inter] -> [L, E, Inter, Hidden]
      param = self._transpose_expert_down(param)
      # vLLM 0.17+ expects (E, Hidden, Inter) -> (E, Inter, Hidden)
      # So for each layer, do param[i]: (E, Inter, Hidden) -> (E, Hidden, Inter)
      param = jnp.transpose(param, (0, 1, 3, 2))
      return self._unstack_layer(param)

    # --- 2. Streaming API (memory-efficient path for production) ---

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _make_attn_compute_single(tp: int):
      """Return a JIT-compiled attn converter for a given TP degree."""
      @jax.jit
      def _compute(q, k, v, o, q_norm, k_norm):
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        d_model = q.shape[0]

        # vLLM QKVParallelLinear logical layout: [Q_all, K_all, V_all].
        # TP sharding is applied externally via _get_reshard_fn — don't pre-interleave.
        q_flat = q.reshape(d_model, num_q_heads * head_dim)
        k_flat = k.reshape(d_model, num_kv_heads * head_dim)
        v_flat = v.reshape(d_model, num_kv_heads * head_dim)
        qkv_flat = jnp.concatenate([q_flat, k_flat, v_flat], axis=1)
        qkv_proj = jnp.transpose(qkv_flat)

        if o.ndim == 2:
          o_proj = jnp.transpose(o)
        else:
          o_proj = jnp.transpose(o, (2, 0, 1)).reshape(d_model, -1)

        return {
            "self_attn.qkv_proj.weight": qkv_proj,
            "self_attn.o_proj.weight": o_proj,
            "self_attn.q_norm.weight": q_norm,
            "self_attn.k_norm.weight": k_norm
        }

      return _compute

    def convert_streaming(self, model_state: dict):
      """Yield (key, array) pairs without ever accumulating the full vllm_state."""
      yield from self._stream_global(model_state)
      yield from self._stream_attn(model_state)
      yield from self._stream_moe(model_state)

    def convert_streaming_per_layer(self, params: dict):
      """Yield dictionaries of weights (per layer) to minimize program loading OOM."""
      # 1. Global weights
      global_dict = {}
      for key, arr in self._stream_global(params):
        global_dict[key] = arr
      yield -1, global_dict
      
      # 2. Per-layer weights
      prefix = "vllm_model.model.layers"
      moe = params['base']['decoder']['layers']['moe_block']
      attn = params['base']['decoder']['layers']['self_attention']
      pre_ln = params['base']['decoder']['layers']['pre_self_attention_layer_norm']['scale']
      post_ln = params['base']['decoder']['layers']['post_self_attention_layer_norm']['scale']
      
      tp = min(self.vllm_tp, self.config.base_num_kv_heads)
      _attn_compute = self._make_attn_compute_single(tp)
      _fuse_chunk = self._make_fuse_chunk(self.vllm_tp, 1)

      # Unstack continuous weight arrays to free original large buffers
      q_list = jnp.unstack(attn['query']['kernel'][...], axis=1)
      k_list = jnp.unstack(attn['key']['kernel'][...], axis=1)
      v_list = jnp.unstack(attn['value']['kernel'][...], axis=1)
      o_list = jnp.unstack(attn['out']['kernel'][...], axis=1)
      q_norm_list = jnp.unstack(attn['query_norm']['scale'][...], axis=1)
      k_norm_list = jnp.unstack(attn['key_norm']['scale'][...], axis=1)

      pre_ln_list = jnp.unstack(pre_ln[...], axis=1)
      post_ln_list = jnp.unstack(post_ln[...], axis=1)

      if self.use_ep:
        wi_0_list = jnp.unstack(moe['wi_0'][...], axis=1)
        wi_1_list = jnp.unstack(moe['wi_1'][...], axis=1)
        wo_list = jnp.unstack(moe['wo'][...], axis=1)
      else:
        gate_list = jnp.unstack(moe['gate']['kernel'][...], axis=1)
        wo_list = jnp.unstack(moe['wo'][...], axis=1)
        wi_0_list = jnp.unstack(moe['wi_0'][...], axis=1)
        wi_1_list = jnp.unstack(moe['wi_1'][...], axis=1)

      gc.collect()

      for i in range(self.num_layers):
        logging.info(f"[STREAMING DEBUG] Slicing and converting layer {i}/{self.num_layers}")
        # 1. Attention weights
        attn_dict = {}
        q_i = q_list[i]
        k_i = k_list[i]
        v_i = v_list[i]
        o_i = o_list[i]
        q_norm_i = q_norm_list[i]
        k_norm_i = k_norm_list[i]
        
        converted_attn = _attn_compute(q_i, k_i, v_i, o_i, q_norm_i, k_norm_i)
        
        pre_ln_i = pre_ln_list[i]
        post_ln_i = post_ln_list[i]
        
        attn_dict[f'{prefix}.{i}.input_layernorm.weight'] = pre_ln_i
        attn_dict[f'{prefix}.{i}.post_attention_layernorm.weight'] = post_ln_i
        
        for suffix, arr in converted_attn.items():
          attn_dict[f'{prefix}.{i}.{suffix}'] = arr
          
        yield i, attn_dict

        # Cleanup attention arrays
        del attn_dict, converted_attn, q_i, k_i, v_i, o_i, q_norm_i, k_norm_i, pre_ln_i, post_ln_i
        gc.collect()
        
        # 2. MoE weights
        moe_dict = {}
        if self.use_ep:
          wi_0_i = wi_0_list[i]
          wi_1_i = wi_1_list[i]
          wo_i = wo_list[i]
          moe_dict[f"{prefix}.{i}.mlp.experts.kernel_gating_EDF"] = jnp.transpose(wi_0_i, (0, 1, 2))
          moe_dict[f"{prefix}.{i}.mlp.experts.kernel_up_proj_EDF"] = jnp.transpose(wi_1_i, (0, 1, 2))
          moe_dict[f"{prefix}.{i}.mlp.experts.kernel_down_proj_EFD"] = wo_i
          yield i, moe_dict

          del wi_0_i, wi_1_i, wo_i
        else:
          gate_i = gate_list[i]
          moe_dict[f"{prefix}.{i}.mlp.gate.weight"] = jnp.transpose(gate_i, (1, 0))

          w2_i = wo_list[i]
          moe_dict[f"{prefix}.{i}.mlp.experts.w2_weight"] = w2_i

          wi_0_i = wi_0_list[i]
          wi_1_i = wi_1_list[i]
          fused = _fuse_chunk(jnp.expand_dims(wi_0_i, axis=1), jnp.expand_dims(wi_1_i, axis=1))
          moe_dict[f"{prefix}.{i}.mlp.experts.w13_weight"] = jnp.transpose(jnp.unstack(fused, axis=0)[0], (0, 2, 1))
          yield i, moe_dict

          del gate_i, w2_i, wi_0_i, wi_1_i, fused

        del moe_dict
        gc.collect()


    def _stream_global(self, params):
      yield "vllm_model.model.embed_tokens.weight", jnp.array(
          params['base']['token_embedder']['embedding'][...])
      yield "vllm_model.model.norm.weight", jnp.array(
          params['base']['decoder']['decoder_norm']['scale'][...])
      yield "vllm_model.lm_head.weight", self._transpose_2d(
          params['base']['decoder']['logits_dense']['kernel'][...])

    def _stream_attn(self, params):
      pre_ln = self._transpose_unstack(
          params['base']['decoder']['layers']['pre_self_attention_layer_norm']['scale'][...])
      post_ln = self._transpose_unstack(
          params['base']['decoder']['layers']['post_self_attention_layer_norm']['scale'][...])
      attn = params['base']['decoder']['layers']['self_attention']
      self_attn = self._to_attn(attn)
      for i in range(self.num_layers):
        yield f'vllm_model.model.layers.{i}.input_layernorm.weight', pre_ln[i]
        yield f'vllm_model.model.layers.{i}.post_attention_layernorm.weight', post_ln[i]
        for suffix, layers in self_attn.items():
          yield f'vllm_model.model.layers.{i}.{suffix}', layers[i]
      del pre_ln, post_ln, self_attn
      gc.collect()

    def _stream_moe(self, params, chunk_size: int = 1):
      prefix = "vllm_model.model.layers"
      moe = params['base']['decoder']['layers']['moe_block']

      for i in range(self.num_layers):
        if self.use_ep:
          wi_0_i = moe['wi_0'][...][:, i, :, :]
          wi_1_i = moe['wi_1'][...][:, i, :, :]
          wo_i = moe['wo'][...][:, i, :, :]
          yield f"{prefix}.{i}.mlp.experts.kernel_gating_EDF", jnp.transpose(wi_0_i, (0, 1, 2))
          yield f"{prefix}.{i}.mlp.experts.kernel_up_proj_EDF", jnp.transpose(wi_1_i, (0, 1, 2))
          yield f"{prefix}.{i}.mlp.experts.kernel_down_proj_EFD", wo_i
          del wi_0_i, wi_1_i, wo_i
        else:
          gate_i = moe['gate']['kernel'][...][:, i, :]
          yield f"{prefix}.{i}.mlp.gate.weight", jnp.transpose(gate_i, (1, 0))

          w2_i = moe['wo'][...][:, i, :, :]
          yield f"{prefix}.{i}.mlp.experts.w2_weight", w2_i

          wi_0 = jax.lax.slice_in_dim(moe['wi_0'][...], i, i + 1, axis=1)
          wi_1 = jax.lax.slice_in_dim(moe['wi_1'][...], i, i + 1, axis=1)
          fused = self._make_fuse_chunk(self.vllm_tp, 1)(wi_0, wi_1)
          yield f"{prefix}.{i}.mlp.experts.w13_weight", jnp.transpose(jnp.unstack(fused, axis=0)[0], (0, 2, 1))
          del gate_i, w2_i, wi_0, wi_1, fused
        gc.collect()

    @staticmethod
    @functools.lru_cache(maxsize=16)
    def _make_fuse_chunk(tp: int, chunk_L: int):
      """JIT-compiled w13 fuser keyed by (tp, chunk_L) for per-chunk caching."""
      @jax.jit
      def _fuse(wi_0, wi_1):
        wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))
        wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

        def _fuse_single(w0, w1):
          w0 = jnp.transpose(w0, (0, 2, 1))
          w1 = jnp.transpose(w1, (0, 2, 1))
          e, d_inner, d_model = w0.shape
          c = d_inner // tp
          gate_c = w0.reshape(e, tp, c, d_model)
          up_c = w1.reshape(e, tp, c, d_model)
                    
          # Pad local chunk dimension to multiple of 128 to match tpu-inference expectation
          padded_c = (c + 127) // 128 * 128
          pad_amount = padded_c - c
          gate_c = jnp.pad(gate_c, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          up_c = jnp.pad(up_c, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          
          combined = jnp.stack([gate_c, up_c], axis=2)
          return combined.reshape(e, 2 * tp * padded_c, d_model)

        return jax.vmap(_fuse_single)(wi_0, wi_1)

      return _fuse

    # --- 3. Original bulk-convert helpers (kept for bench_weight_sync main()) ---

    def _to_mlp_expert_gate_up(self, wi_0, wi_1, num_layers, layer_key_prefix, layer_key_suffix):
      # Process all layers in one JIT call using vmap to avoid per-layer dispatch
      # overhead (which was ~50 separate device syncs on multi-host v5p-64).
      _fuse_all = self._make_fuse_all(self.vllm_tp)

      logging.info("_to_mlp_expert_gate_up: dispatching _fuse_all (single JIT+vmap)...")
      fused = _fuse_all(wi_0, wi_1)
      logging.info("_to_mlp_expert_gate_up: _fuse_all complete, shape=%s, unstacking layers...", fused.shape)
      del wi_0, wi_1
      gc.collect()
      _log_mem_stats("converter:moe_w13_post_fuse")

      # vLLM 0.17+ expects (e, 2*d_inner, d_model) -> (e, d_model, 2*d_inner)
      for i, layer_i in enumerate(jnp.unstack(fused, axis=0)):
        layer_i = jnp.transpose(layer_i, (0, 2, 1))  # (e, 2*d_inner, d_model) -> (e, d_model, 2*d_inner)
        self.vllm_state[f"{layer_key_prefix}.{i}.{layer_key_suffix}"] = layer_i
        if i % 8 == 7:
          gc.collect()
      del fused, layer_i
      gc.collect()
      _log_mem_stats("converter:moe_w13_post_unstack")

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _make_fuse_all(tp: int):
      """Return a JIT-compiled w13 fuser for a given TP degree.

      Cached by tp so the same XLA executable is reused across all steps.
      """
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

      return _fuse_all
      
    @staticmethod
    @jax.jit
    def _unstack_layer(param):
        return jnp.unstack(param, axis=0)

    @staticmethod
    @jax.jit
    def _transpose_unstack(x):
        return jnp.unstack(jnp.transpose(x, (1, 0)))

    @staticmethod
    @jax.jit
    def _transpose_2d(x):
        return jnp.transpose(x, (1, 0))

    @staticmethod
    @jax.jit
    def _transpose_gate(param):
        return jnp.transpose(param, (1, 2, 0))

    @staticmethod
    @jax.jit
    def _transpose_expert_down(param):
        return jnp.transpose(param, (1, 0, 3, 2))


def save_dict_to_file(dict, filename):
    with open(filename, 'w') as f:
        for key in sorted(dict.keys()):
            f.write(f"{key}: {dict[key].shape}\n")


def main():
  pathwaysutils.initialize()
  all_devices = jax.devices()
  print(f"JAX devices: {all_devices}")  
  _setup_jax_compilation_cache()
  _setup_vllm()
  _clean_device_memory()

  FLAGS(sys.argv)

  if _SAVE_WEIGHTS.value and _GCS_BUCKET.value:
    logging.info("Testing GCS write permission early...")
    dummy_file = "/tmp/permission_test.txt"
    try:
      with open(dummy_file, "w") as f:
        f.write("test")
      gcs_path = os.path.join(_GCS_BUCKET.value, "permission_test.txt")
      cmd = f"gsutil cp {dummy_file} {gcs_path}"
      ret = os.system(cmd)
      if ret != 0:
        logging.error(f"GCS write permission check failed for bucket {_GCS_BUCKET.value}. Exit code: {ret}")
        sys.exit(1)
      else:
        logging.info("GCS write permission check passed. Cleaning up test file.")
        os.system(f"gsutil rm {gcs_path}")
    except Exception as e:
      logging.error(f"Error checking GCS permission: {e}")
      sys.exit(1)
    finally:
      if os.path.exists(dummy_file):
        os.remove(dummy_file)

  # Split devices for MaxText and vLLM by slice_index
  devices_by_slice = {}
  for d in all_devices:
      idx = getattr(d, 'slice_index', 0)
      if idx not in devices_by_slice:
          devices_by_slice[idx] = []
      devices_by_slice[idx].append(d)
  
  slice_indices = sorted(devices_by_slice.keys())
  print(f"Found slices: {slice_indices}")
  
  if len(slice_indices) >= 2:
      maxtext_devices = devices_by_slice[slice_indices[0]]
      vllm_devices = devices_by_slice[slice_indices[1]]
  else:
      # Fallback if less than 2 slices, split in half
      half = len(all_devices) // 2
      maxtext_devices = all_devices[:half]
      vllm_devices = all_devices[half:]
      
  print(f"MaxText devices: {len(maxtext_devices)}")
  print(f"vLLM devices: {len(vllm_devices)}")

  # Load maxtext model
  print("="*80)
  print("Loading MaxText model...")
  print("="*80)
  base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
  base_yaml_path = os.path.join(base_dir, "configs", "post_train", "rl.yml")
  if not os.path.exists(base_yaml_path):
    raise FileNotFoundError(f"Could not find base.yml at expected location: {base_yaml_path}")
  config, model, mesh = _load_maxtext_model(base_yaml_path, devices=maxtext_devices)
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {config.model_name}")
  print(f"Mesh: {mesh}")

  # Convert weights to VLLM format
  print("="*80)
  print("Converting weights to VLLM format")
  print("="*80)
  model_state = nnx.state(model)
  del model
  gc.collect()
  
  print("="*80)
  print("Loading vLLM model for generation test...")
  print("="*80)
  
  vllm_device_indexes = [d.id for d in vllm_devices]
  rollout_ep = int(os.environ.get("rollout_expert_parallelism", "1"))
  
  llm = LLM(
    _TOKENIZER_PATH.value,
    max_model_len=16,
    data_parallel_size=16,
    tensor_parallel_size=_TP_SIZE.value,
    gpu_memory_utilization=0.9,
    additional_config={
        "sharding": {
            "sharding_strategy": {
                "device_indexes": vllm_device_indexes,
                "expert_parallelism": rollout_ep,
            }
        }
    }
  )
  
  print("\n" + "="*80)
  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state

  converter = MaxTextToVLLMConverter(config, mesh)
  mismatches = 0

  _same_devices = False
  @functools.lru_cache(maxsize=None)
  def _get_reshard_fn(dst_sharding):
    if _same_devices:
      return jax.jit(lambda x: x, out_shardings=dst_sharding)
    else:
      return functools.partial(jax.device_put, device=dst_sharding)

  print("="*80)
  print("Comparing and transferring weights layer by layer...")
  for i, layer_dict in converter.convert_streaming_per_layer(model_state):
    for key, stream_w in layer_dict.items():
      if key not in llm_state:
        print(f"Missing key in real vLLM: {key}")
        mismatches += 1
        continue

      llm_w = llm_state[key]
      llm_w = llm_w.value if hasattr(llm_w, 'value') else llm_w
      stream_w = stream_w.value if hasattr(stream_w, 'value') else stream_w

      if llm_w.shape != stream_w.shape:
        print(f"Shape mismatch for {key}: vLLM={llm_w.shape}, streaming={stream_w.shape}")
        mismatches += 1
        continue

      dst_sharding = llm_state[key].sharding
      stream_w_resharded = _get_reshard_fn(dst_sharding)(stream_w)

      is_target = (i == _TARGET_LAYER.value) or (i == -1 and _TARGET_LAYER.value < 0)
      if jax.process_index() == 0 and _SAVE_WEIGHTS.value and is_target:
        local_dir = "/tmp/saved_weights"
        os.makedirs(local_dir, exist_ok=True)

        key_clean = key.replace(".", "_")
        try:
          llm_w_np = jax.device_get(llm_w)
          stream_w_np = jax.device_get(stream_w)

          np.save(os.path.join(local_dir, f"vllm_{key_clean}.npy"), llm_w_np)
          np.save(os.path.join(local_dir, f"maxtext_{key_clean}.npy"), stream_w_np)
          logging.info(f"Saved weights for {key} to {local_dir}")
        except Exception as e:
          logging.error(f"Error saving weights for {key}: {e}")

      if is_target:
        try:
          max_diff = jnp.max(jnp.abs(llm_w - stream_w_resharded))
          is_close = bool(max_diff < 1e-5)
          if not is_close:
            print(f"Value mismatch for {key} (max diff: {max_diff})")
            mismatches += 1
        except Exception as e:
          print(f"Error comparing {key}: {e}")
          mismatches += 1

      llm_state[key] = stream_w_resharded
    
    del layer_dict
    gc.collect()

  print(f"Total mismatches found: {mismatches}")
  print("="*80)

  if jax.process_index() == 0 and _SAVE_WEIGHTS.value and _GCS_BUCKET.value:
    local_dir = "/tmp/saved_weights"
    gcs_path = os.path.join(_GCS_BUCKET.value, "saved_weights")
    logging.info(f"Uploading saved weights from {local_dir} to {gcs_path}...")
    cmd = f"gsutil -m cp -r {local_dir}/* {gcs_path}/"
    ret = os.system(cmd)
    if ret == 0:
      logging.info("Upload successful.")
    else:
      logging.error(f"Upload failed with exit code {ret}.")


      # llm_state[key] = jax.device_put(np.asarray(weight_array), dst_sharding)  # when mesh mismatch, fallback to CPU

  sampling_params = SamplingParams(temperature=0.0, max_tokens=30)
  print("\n" + "="*80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate("Paris is", sampling_params=sampling_params))


if __name__ == "__main__":
  main()
