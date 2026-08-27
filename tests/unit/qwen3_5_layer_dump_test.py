# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for extracting and dumping intermediate tensors (activations/logits)

from 1 decoder layer of Qwen3.5 MoE between MaxText Training
(attention="flash", sparse_matmul=True) and vLLM Inference
(attention="vllm_rpa", fused_moe_matmul=True).
"""

import os
import sys
import tempfile
from typing import Any
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, Array
from maxtext.configs import pyconfig
from maxtext.models import qwen3_5
from maxtext.utils import max_logging, maxtext_utils
import numpy as np
import pytest
from tests.utils.test_helpers import get_test_config_path

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"


def compute_drift_metrics(
    t_ref: Array | np.ndarray,
    t_tgt: Array | np.ndarray,
) -> dict[str, Any]:
  """Computes numerical drift metrics between reference (training) and target (inference) tensors."""
  if tuple(t_ref.shape) != tuple(t_tgt.shape):
    return {
        "shape_match": False,
        "ref_shape": list(t_ref.shape),
        "tgt_shape": list(t_tgt.shape),
        "max_abs_err": float("nan"),
        "mae": float("nan"),
        "cos_sim": float("nan"),
        "rel_err": float("nan"),
        "rms_err": float("nan"),
    }

  a = np.array(jax.device_get(t_ref), dtype=np.float32)
  b = np.array(jax.device_get(t_tgt), dtype=np.float32)

  abs_diff = np.abs(a - b)
  max_abs_err = float(np.max(abs_diff))
  mae = float(np.mean(abs_diff))
  rms_err = float(np.sqrt(np.mean(abs_diff**2)))

  a_flat = a.reshape(-1)
  b_flat = b.reshape(-1)
  norm_a = float(np.linalg.norm(a_flat))
  norm_b = float(np.linalg.norm(b_flat))
  rel_err = float(np.linalg.norm(a_flat - b_flat) / (norm_a + 1e-12)) if norm_a > 0 else 0.0

  denom = (norm_a * norm_b) + 1e-12
  dot_prod = float(np.dot(a_flat, b_flat))
  cos_sim = float(dot_prod / denom) if denom > 0 else 1.0

  return {
      "shape_match": True,
      "shape": list(t_ref.shape),
      "ref_dtype": str(t_ref.dtype),
      "tgt_dtype": str(t_tgt.dtype),
      "max_abs_err": max_abs_err,
      "mae": mae,
      "cos_sim": cos_sim,
      "rel_err": abs(rel_err),
      "rms_err": rms_err,
  }


def generate_comparison_markdown_table(metrics_dict: dict[str, dict[str, Any]]) -> str:
  """Formats computed tensor metrics into a clean markdown table for analysis."""
  header = (
      "| Tensor Name | Shape | Max Abs Err ($L\_\\infty$) | MAE | Cosine"
      " Sim | Rel Err |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
  )
  rows = []
  for tensor_name, m in metrics_dict.items():
    if not m.get("shape_match", True):
      rows.append(
          f"| `{tensor_name}` | Shape Mismatch: {m.get('ref_shape')} vs"
          f" {m.get('tgt_shape')} | N/A | N/A | N/A | N/A |"
      )
    else:
      shape_str = "x".join(map(str, m.get("shape", [])))
      max_err = f"{m.get('max_abs_err', 0.0):.6e}"
      mae = f"{m.get('mae', 0.0):.6e}"
      cos_sim = f"{m.get('cos_sim', 1.0):.6f}"
      rel_err = f"{m.get('rel_err', 0.0):.6e}"
      rows.append(
          f"| `{tensor_name}` | `{shape_str}` | `{max_err}` | `{mae}` |"
          f" `{cos_sim}` | `{rel_err}` |"
      )
  return header + "\n".join(rows)


def dump_tensors_to_npz(tensors_dict: dict[str, Any], file_path: str):
  """Dumps a dictionary of named tensors to a compressed .npz archive."""
  np_dict = {
      k: np.array(v)
      for k, v in tensors_dict.items()
      if v is not None and not isinstance(v, (dict, list, tuple))
  }
  np.savez_compressed(file_path, **np_dict)


def sync_qwen3_5_layer_weights(
    src_layer: qwen3_5.Qwen3_5DecoderLayer,
    dst_layer: qwen3_5.Qwen3_5DecoderLayer,
    dst_mesh: Mesh | None = None,
):
  """Synchronizes all learnable parameters from src_layer to dst_layer with full bit-parity.

  Updates parameter .value without replacing module instances, preserving each
  layer's own mesh and submodules.
  """
  target_mesh = dst_mesh if dst_mesh is not None else dst_layer.mesh
  sharding = NamedSharding(target_mesh, P())

  def copy_param(src_param, dst_param):
    if src_param is not None and dst_param is not None:
      src_val = (
          src_param.value
          if isinstance(src_param, nnx.Variable)
          else src_param
      )
      val = np.array(jax.device_get(src_val))
      dst_param.value = jax.device_put(
          jnp.array(val, dtype=dst_param.dtype), sharding
      )

  def copy_linear(src_mod, dst_mod):
    if src_mod is None or dst_mod is None:
      return
    for attr in ["kernel", "bias", "scale"]:
      if hasattr(src_mod, attr) and hasattr(dst_mod, attr):
        copy_param(getattr(src_mod, attr), getattr(dst_mod, attr))

  # 1. LayerNorm parameters
  if hasattr(src_layer, "input_layernorm") and hasattr(
      dst_layer, "input_layernorm"
  ):
    copy_linear(src_layer.input_layernorm, dst_layer.input_layernorm)
  if hasattr(src_layer, "post_attention_layernorm") and hasattr(
      dst_layer, "post_attention_layernorm"
  ):
    copy_linear(
        src_layer.post_attention_layernorm, dst_layer.post_attention_layernorm
    )

  # 2. Attention block parameters
  src_attn = getattr(src_layer.attention, "attention", src_layer.attention)
  dst_attn = getattr(dst_layer.attention, "attention", dst_layer.attention)
  for proj_name in ["query", "key", "value", "out", "query_norm", "key_norm"]:
    if hasattr(src_attn, proj_name) and hasattr(dst_attn, proj_name):
      copy_linear(getattr(src_attn, proj_name), getattr(dst_attn, proj_name))

  # 3. MoE Shared Expert and Gate
  if hasattr(src_layer.mlp, "shared_expert_gate") and hasattr(
      dst_layer.mlp, "shared_expert_gate"
  ):
    copy_linear(
        src_layer.mlp.shared_expert_gate, dst_layer.mlp.shared_expert_gate
    )

  if hasattr(src_layer.mlp, "shared_expert") and hasattr(
      dst_layer.mlp, "shared_expert"
  ):
    src_se = src_layer.mlp.shared_expert
    dst_se = dst_layer.mlp.shared_expert
    for dense_name in ["wi_0", "wi_1", "wi", "wo"]:
      if hasattr(src_se, dense_name) and hasattr(dst_se, dense_name):
        copy_linear(getattr(src_se, dense_name), getattr(dst_se, dense_name))

  # 4. MoE Routed Experts
  if hasattr(src_layer.mlp, "routed_experts") and hasattr(
      dst_layer.mlp, "routed_experts"
  ):
    src_moe = src_layer.mlp.routed_experts
    dst_moe = dst_layer.mlp.routed_experts

    if hasattr(src_moe, "gate") and hasattr(dst_moe, "gate"):
      copy_linear(src_moe.gate, dst_moe.gate)

    if hasattr(src_moe, "wo") and hasattr(dst_moe, "wo"):
      copy_linear(src_moe.wo, dst_moe.wo)

    # Handle prefused vs separate wi expert weights
    if (
        hasattr(dst_moe, "wi")
        and hasattr(src_moe, "wi_0")
        and hasattr(src_moe, "wi_1")
    ):
      w0 = np.array(
          jax.device_get(
              src_moe.wi_0.value
              if isinstance(src_moe.wi_0, nnx.Variable)
              else src_moe.wi_0
          )
      )
      w1 = np.array(
          jax.device_get(
              src_moe.wi_1.value
              if isinstance(src_moe.wi_1, nnx.Variable)
              else src_moe.wi_1
          )
      )
      wi_fused = np.concatenate([w0, w1], axis=-1)
      dst_moe.wi.value = jax.device_put(
          jnp.array(wi_fused, dtype=dst_moe.wi.dtype), sharding
      )
    elif hasattr(dst_moe, "wi_0") and hasattr(src_moe, "wi_0"):
      copy_param(src_moe.wi_0, dst_moe.wi_0)
      if hasattr(dst_moe, "wi_1") and hasattr(src_moe, "wi_1"):
        copy_param(src_moe.wi_1, dst_moe.wi_1)
    elif hasattr(dst_moe, "wi") and hasattr(src_moe, "wi"):
      copy_param(src_moe.wi, dst_moe.wi)


# pylint: disable=too-many-positional-arguments
def capture_qwen3_5_layer_intermediates(
    layer: qwen3_5.Qwen3_5DecoderLayer,
    inputs: Array,
    decoder_segment_ids: Array | None,
    decoder_positions: Array | None,
    model_mode: str,
    deterministic: bool = True,
    attention_metadata: Any | None = None,
    kv_cache: Any | None = None,
) -> tuple[Array, dict[str, Array]]:
  """Sequentially executes all submodules of Qwen3_5DecoderLayer and captures all intermediate tensors."""
  import jax.lax

  # Eager-mode safe wrapper: in eager mode, with_sharding_constraint is just a compiler hint;
  # resharding across differently-sized train (4-device) vs infer (2-device) meshes fails in eager dispatch.
  orig_wsc = jax.lax.with_sharding_constraint

  def safe_wsc(x, shardings):
    try:
      return orig_wsc(x, shardings)
    except Exception:
      return x

  jax.lax.with_sharding_constraint = safe_wsc

  try:
    with jax.set_mesh(layer.mesh):
      tensors: dict[str, Array] = {}
      tensors["T01_layer_input"] = inputs

      # Step 1: Pre-Attention LayerNorm
      norm1_out = layer.input_layernorm(inputs)
      tensors["T02_input_layernorm_out"] = norm1_out

      # Step 2: Attention Block
      if isinstance(layer.attention, qwen3_5.Qwen3_5FullAttention):
        attn_module = layer.attention.attention
        batch_size, seq_len, _ = inputs.shape

        # Projections
        qkv_sharding = None
        q_proj = attn_module.query_projection(
            norm1_out, out_sharding=qkv_sharding
        )
        k_proj = attn_module.kv_projection(
            norm1_out, proj_name="key", out_sharding=qkv_sharding
        )
        v_proj = attn_module.kv_projection(
            norm1_out, proj_name="value", out_sharding=qkv_sharding
        )

        # Query and Gate Split (Qwen3 hybrid attention)
        if attn_module.is_qwen3_hybrid:
          q_split, gate = jnp.split(q_proj, 2, axis=-1)
          gate_flat = gate.reshape(
              batch_size,
              seq_len,
              attn_module.config.num_query_heads * attn_module.config.head_dim,
          )
          tensors["T03_q_proj_raw"] = q_proj
          tensors["T04_q_proj_heads"] = q_split
          tensors["T05_query_gate"] = gate_flat
          q_to_norm = q_split
        else:
          tensors["T03_q_proj_raw"] = q_proj
          q_to_norm = q_proj
          gate_flat = None
          tensors["T04_q_proj_heads"] = q_proj
          tensors["T05_query_gate"] = jnp.zeros(
              (batch_size, seq_len, 1), dtype=inputs.dtype
          )

        tensors["T06_k_proj_heads"] = k_proj
        tensors["T07_v_proj_heads"] = v_proj

        # QK Normalization
        q_norm = (
            attn_module.query_norm(q_to_norm)
            if (attn_module.use_qk_norm or attn_module.is_qwen3_hybrid)
            else q_to_norm
        )
        k_norm = (
            attn_module.key_norm(k_proj)
            if (attn_module.use_qk_norm or attn_module.is_qwen3_hybrid)
            else k_proj
        )
        tensors["T08_q_norm_out"] = q_norm
        tensors["T09_k_norm_out"] = k_norm

        # Rotary Position Embedding (RoPE)
        if not attn_module.is_nope_layer:
          q_rope = attn_module.apply_rotary_embedding(
              q_norm, inputs_positions=decoder_positions
          )
          k_rope = attn_module.apply_rotary_embedding(
              k_norm, inputs_positions=decoder_positions
          )
        else:
          q_rope = q_norm
          k_rope = k_norm

        tensors["T10_q_rope_out"] = q_rope
        tensors["T11_k_rope_out"] = k_rope

        if (
            attn_module.query_pre_attn_scalar
            and attn_module.query_pre_attn_scalar != 1.0
        ):
          q_rope = q_rope * attn_module.query_pre_attn_scalar

        # Attention Core (Flash Attention vs vLLM RPA)
        if (
            layer.config.attention in ("vllm_rpa", "vllm_batched_rpa")
            and model_mode != MODEL_MODE_TRAIN
        ):
          if attention_metadata is None or kv_cache is None:
            block_size = 128
            num_blocks_per_seq = (seq_len + block_size - 1) // block_size
            total_pages = batch_size * num_blocks_per_seq
            block_tables = jnp.arange(total_pages, dtype=jnp.int32)
            seq_lens = jnp.array([seq_len] * batch_size, dtype=jnp.int32)
            query_start_loc = jnp.arange(
                0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32
            )
            request_distribution = jnp.array(
                [0, 0, batch_size], dtype=jnp.int32
            )
            input_positions = decoder_positions.reshape(-1)

            class SimpleAttentionMetadata:

              def __init__(
                  self,
                  input_positions,
                  block_tables,
                  seq_lens,
                  query_start_loc,
                  request_distribution,
                  mamba_state_indices=None,
              ):
                self.input_positions = input_positions
                self.block_tables = block_tables
                self.seq_lens = seq_lens
                self.query_start_loc = query_start_loc
                self.request_distribution = request_distribution
                self.mamba_state_indices = mamba_state_indices

            attention_metadata = SimpleAttentionMetadata(
                input_positions=input_positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )
            num_kv_heads = attn_module.config.num_kv_heads
            head_dim = attn_module.config.head_dim
            try:
              from tpu_inference.layers.common.attention_interface import (
                  get_kv_cache_shape,
              )

              kv_shape = get_kv_cache_shape(
                  total_pages,
                  block_size,
                  num_kv_heads,
                  head_dim,
                  inputs.dtype,
              )
              kv_cache = jnp.zeros(kv_shape, dtype=inputs.dtype)
            except Exception:
              kv_cache = jnp.zeros(
                  (total_pages, block_size, num_kv_heads, 2, head_dim),
                  dtype=inputs.dtype,
              )

          attn_core_raw, _ = attn_module.forward_serve_vllm(
              q_rope,
              k_rope,
              v_proj,
              rpa_kv_cache=kv_cache,
              rpa_metadata=attention_metadata,
          )
          attn_core = attn_core_raw.reshape(
              batch_size,
              seq_len,
              attn_module.config.num_query_heads,
              attn_module.config.head_dim,
          )
        else:
          cached_values = [None, None]
          attn_core = attn_module.attention_op(
              q_rope,
              k_rope,
              v_proj,
              decoder_segment_ids,
              decoder_positions,
              model_mode,
              cached_values,
          )

        tensors["T12_attn_core_out"] = attn_core

        # Gated Attention & Out Projection
        if attn_module.is_qwen3_hybrid and gate_flat is not None:
          attn_flat = attn_core.reshape(
              batch_size,
              seq_len,
              attn_module.config.num_query_heads * attn_module.config.head_dim,
          )
          gated_attn = attn_flat * jax.nn.sigmoid(gate_flat)
          tensors["T13_attn_gated_out"] = gated_attn
          attn_out = attn_module.out_projection(gated_attn)
        else:
          tensors["T13_attn_gated_out"] = attn_core
          attn_out = attn_module.out_projection(attn_core)

        tensors["T14_attn_out_proj"] = attn_out
      else:
        # Linear Attention (GDN) fallback
        attn_out, _ = layer.attention(
            norm1_out,
            model_mode=model_mode,
            decoder_segment_ids=decoder_segment_ids,
        )
        tensors["T14_attn_out_proj"] = attn_out

      # Step 3: Post-Attention Residual
      post_attn_residual = inputs + attn_out
      tensors["T15_post_attn_residual"] = post_attn_residual

      # Step 4: Pre-MoE LayerNorm
      norm2_out = layer.post_attention_layernorm(post_attn_residual)
      tensors["T16_post_attn_layernorm_out"] = norm2_out

      # Step 5: MoE Block (Shared Expert + Routed Experts)
      moe_block = layer.mlp
      shared_gate_logits = moe_block.shared_expert_gate(norm2_out)
      shared_gate_prob = jax.nn.sigmoid(
          shared_gate_logits.astype(jnp.float32)
      ).astype(inputs.dtype)
      shared_mlp_out = moe_block.shared_expert(
          norm2_out, deterministic=deterministic
      )

      tensors["T17_shared_expert_gate_logits"] = shared_gate_logits
      tensors["T18_shared_expert_gate_prob"] = shared_gate_prob
      tensors["T19_shared_expert_mlp_out"] = shared_mlp_out

      # Router Gate Logits
      router_gate_logits, _ = moe_block.routed_experts.gate(norm2_out)
      tensors["T20_router_gate_logits"] = router_gate_logits

      # Routed MoE Computation
      routed_out, _, _ = moe_block.routed_experts(norm2_out)
      tensors["T23_routed_moe_out"] = routed_out

      # Combined MoE Output
      moe_combined_out = routed_out + shared_gate_prob * shared_mlp_out
      tensors["T24_moe_combined_out"] = moe_combined_out

      # Step 6: Final Layer Residual Output
      layer_output = post_attn_residual + moe_combined_out
      tensors["T25_layer_output"] = layer_output

      return layer_output, tensors
  finally:
    jax.lax.with_sharding_constraint = orig_wsc


@pytest.mark.tpu_only
class Qwen3_5LayerDumpTest(unittest.TestCase):
  """Unit and regression tests for 1-layer Qwen3.5 MoE tensor dumping and comparison."""

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest(
          "Qwen3.5 layer dump tests require TPU hardware to execute Flash"
          " Attention and vLLM RPA kernels."
      )
    self.batch_size = max(len(jax.devices()), 4)
    self.seq_len = 16
    self.emb_dim = 256
    self.mlp_dim = 256
    self.moe_mlp_dim = 256
    self.num_experts = 8
    self.num_experts_per_tok = 8
    self.num_query_heads = 4
    self.num_kv_heads = 2
    self.head_dim = 64
    self.vocab_size = 1000

    self.base_kwargs = {
        "override_model_config": True,
        "num_decoder_layers": 1,
        "model_name": "qwen3.5-35b-a3b",
        "base_emb_dim": self.emb_dim,
        "base_mlp_dim": self.mlp_dim,
        "base_moe_mlp_dim": self.moe_mlp_dim,
        "num_experts": self.num_experts,
        "num_experts_per_tok": self.num_experts_per_tok,
        "base_num_query_heads": self.num_query_heads,
        "base_num_kv_heads": self.num_kv_heads,
        "head_dim": self.head_dim,
        "vocab_size": self.vocab_size,
        "max_target_length": self.seq_len,
        "max_prefill_predict_length": self.seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "inhomogeneous_layer_cycle_interval": (
            1  # Ensure layer 0 is full attention
        ),
        "norm_topk_prob": True,
        "float32_logits": True,
        "float32_gate_logits": True,
        "float32_weight_sum": True,
    }

  def _create_configs_and_layers(
      self, dtype_str: str = "bfloat16"
  ) -> tuple[qwen3_5.Qwen3_5DecoderLayer, qwen3_5.Qwen3_5DecoderLayer, Mesh]:
    """Instantiates and synchronizes Training and Inference Qwen3.5 decoder layers."""
    train_kwargs = dict(self.base_kwargs)
    train_kwargs.update({
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "sparse_matmul": True,
        "wi_tile_fwd_batch_seq": 256,
        "wi_tile_fwd_embed_dim": 128,
        "wi_tile_fwd_mlp_dim": 128,
        "use_tokamax_splash": True,
        "sa_use_base2_exp": False,
        "sa_fuse_reciprocal": True,
    })
    cfg_train = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path(),
            "attention=flash",
            "use_tokamax_splash=True",
            "sa_use_base2_exp=False",
            "sa_fuse_reciprocal=True",
            "sparse_matmul=True",
            "megablox=True",
            "use_tokamax_gmm=True",
            "use_gmm_v2=True",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **train_kwargs,
    )
    cfg_infer = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path("inference/vllm.yml"),
            "attention=vllm_rpa",
            "prefuse_moe_weights=True",
            "model_call_mode=inference",
            "ici_data_parallelism=-1",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **self.base_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(cfg_train)
    train_mesh = Mesh(train_devices, cfg_train.mesh_axes)

    infer_devices = maxtext_utils.create_device_mesh(cfg_infer)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    rng = nnx.Rngs(params=42)
    train_layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_train,
        mesh=train_mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=0,
        rngs=rng,
    )

    infer_layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_infer,
        mesh=infer_mesh,
        model_mode=MODEL_MODE_PREFILL,
        layer_idx=0,
        rngs=rng,
    )

    sync_qwen3_5_layer_weights(train_layer, infer_layer, dst_mesh=infer_mesh)

    return train_layer, infer_layer, train_mesh

  def test_qwen3_5_layer_tensor_dump_and_metrics_bf16(self):
    """Verifies that all 25 intermediate tensors are dumped and compared in bfloat16."""
    train_layer, infer_layer, mesh = self._create_configs_and_layers("bfloat16")

    key = jax.random.PRNGKey(101)
    k1, _ = jax.random.split(key)
    inputs = jax.random.normal(
        k1, (self.batch_size, self.seq_len, self.emb_dim), dtype=jnp.bfloat16
    )
    decoder_positions = jnp.broadcast_to(
        jnp.arange(self.seq_len, dtype=jnp.int32),
        (self.batch_size, self.seq_len),
    )
    decoder_segment_ids = jnp.ones(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )

    is_tpu = jax.devices()[0].platform == "tpu"
    if is_tpu:
      inputs = jax.device_put(
          inputs, NamedSharding(mesh, P(("data", "fsdp"), None, None))
      )
      decoder_positions = jax.device_put(
          decoder_positions, NamedSharding(mesh, P(("data", "fsdp"), None))
      )
      decoder_segment_ids = jax.device_put(
          decoder_segment_ids, NamedSharding(mesh, P(("data", "fsdp"), None))
      )

    _, train_tensors = capture_qwen3_5_layer_intermediates(
        train_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_TRAIN,
    )

    _, infer_tensors = capture_qwen3_5_layer_intermediates(
        infer_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_PREFILL,
    )

    metrics: dict[str, dict[str, Any]] = {}
    for name, t_train in train_tensors.items():
      self.assertIn(
          name, infer_tensors, f"Tensor {name} missing in inference dump."
      )
      m = compute_drift_metrics(t_train, infer_tensors[name])
      metrics[name] = m

    table_md = generate_comparison_markdown_table(metrics)
    max_logging.log(
        "\n================ QWEN3.5 1-LAYER DUMP METRICS (BF16)"
        " ================\n"
        + table_md
    )

    self.assertGreaterEqual(
        len(train_tensors),
        15,
        "Expected at least 15 intermediate tensors captured.",
    )
    self.assertTrue(metrics["T25_layer_output"]["shape_match"])
    self.assertGreater(metrics["T25_layer_output"]["cos_sim"], 0.95)

  def test_qwen3_5_layer_tensor_dump_and_metrics_fp32(self):
    """Verifies that running in float32 achieves near-perfect cosine similarity (~1.0)."""
    train_layer, infer_layer, mesh = self._create_configs_and_layers("float32")

    key = jax.random.PRNGKey(202)
    inputs = jax.random.normal(
        key, (self.batch_size, self.seq_len, self.emb_dim), dtype=jnp.float32
    )
    decoder_positions = jnp.broadcast_to(
        jnp.arange(self.seq_len, dtype=jnp.int32),
        (self.batch_size, self.seq_len),
    )
    decoder_segment_ids = jnp.ones(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )

    is_tpu = jax.devices()[0].platform == "tpu"
    if is_tpu:
      inputs = jax.device_put(
          inputs, NamedSharding(mesh, P(("data", "fsdp"), None, None))
      )
      decoder_positions = jax.device_put(
          decoder_positions, NamedSharding(mesh, P(("data", "fsdp"), None))
      )
      decoder_segment_ids = jax.device_put(
          decoder_segment_ids, NamedSharding(mesh, P(("data", "fsdp"), None))
      )

    _, train_tensors = capture_qwen3_5_layer_intermediates(
        train_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_TRAIN,
    )
    _, infer_tensors = capture_qwen3_5_layer_intermediates(
        infer_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_PREFILL,
    )

    metrics = {
        k: compute_drift_metrics(v, infer_tensors[k])
        for k, v in train_tensors.items()
    }
    table_md = generate_comparison_markdown_table(metrics)
    max_logging.log(
        "\n================ QWEN3.5 1-LAYER DUMP METRICS (FP32)"
        " ================\n"
        + table_md
    )

    self.assertGreater(metrics["T25_layer_output"]["cos_sim"], 0.999)

  def test_export_npz_archive(self):
    """Verifies exporting and reloading intermediate tensor archives from disk."""
    train_layer, infer_layer, mesh = self._create_configs_and_layers("bfloat16")

    inputs = jnp.ones(
        (self.batch_size, self.seq_len, self.emb_dim), dtype=jnp.bfloat16
    )
    decoder_positions = jnp.broadcast_to(
        jnp.arange(self.seq_len, dtype=jnp.int32),
        (self.batch_size, self.seq_len),
    )
    decoder_segment_ids = jnp.ones(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )

    is_tpu = jax.devices()[0].platform == "tpu"
    if is_tpu:
      inputs = jax.device_put(
          inputs, NamedSharding(mesh, P(("data", "fsdp"), None, None))
      )
      decoder_positions = jax.device_put(
          decoder_positions, NamedSharding(mesh, P(("data", "fsdp"), None))
      )
      decoder_segment_ids = jax.device_put(
          decoder_segment_ids, NamedSharding(mesh, P(("data", "fsdp"), None))
      )

    _, train_tensors = capture_qwen3_5_layer_intermediates(
        train_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_TRAIN,
    )
    _, infer_tensors = capture_qwen3_5_layer_intermediates(
        infer_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_PREFILL,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      train_path = os.path.join(tmp_dir, "qwen3_5_train_tensors.npz")
      infer_path = os.path.join(tmp_dir, "qwen3_5_infer_tensors.npz")

      dump_tensors_to_npz(train_tensors, train_path)
      dump_tensors_to_npz(infer_tensors, infer_path)

      self.assertTrue(os.path.exists(train_path))
      self.assertTrue(os.path.exists(infer_path))

      loaded_train = np.load(train_path)
      loaded_infer = np.load(infer_path)

      self.assertIn("T01_layer_input", loaded_train.files)
      self.assertIn("T25_layer_output", loaded_infer.files)
      self.assertEqual(loaded_train["T01_layer_input"].shape, inputs.shape)


if __name__ == "__main__":
  unittest.main()
