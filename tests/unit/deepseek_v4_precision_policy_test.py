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

"""Numeric precision-policy tests for DeepSeek-V4 in production bfloat16 mode.

Verifies that every operation listed in deepseek4.precision.yml (router gate logits,
RMSNorm variance, attention softmax with sinks, mHC Sinkhorn normalization, and CSA
indexer score accumulation) matches a forced-float32 variant within the float32 noise
floor and differs from a forced-bfloat16 variant by more than that floor.
"""

from __future__ import annotations

import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_compressed
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import normalizations

pytestmark = [pytest.mark.cpu_only]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
  da = np.asarray(a, dtype=np.float64).ravel()
  db = np.asarray(b, dtype=np.float64).ravel()
  denom = float(np.linalg.norm(db))
  if denom == 0.0:
    return 0.0 if float(np.linalg.norm(da)) == 0.0 else float("inf")
  return float(np.linalg.norm(da - db) / denom)


def _make_bf16_config(**overrides):
  base_yml = os.path.join(os.path.dirname(pyconfig.__file__), "base.yml")
  kwargs = {
      "model_name": "deepseek4-tiny",
      "override_model_config": True,
      "run_name": "ds4_precision_policy",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "float32_gate_logits": True,
      "matmul_precision": "default",
      "attention": "dot_product",
      "attention_type": "compressed",
      "use_indexer": True,
      "indexer_sparse_training": True,
      "indexer_loss_scaling_factor": 0.0,
      "sparse_matmul": True,
      "megablox": False,
      "per_device_batch_size": 1,
      "max_target_length": 64,
      "scan_layers": False,
      "base_num_decoder_layers": 5,
      "first_num_hash_layers": 3,
      "compress_ratios": [0, 0, 4, 128, 4],
      "base_emb_dim": 64,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 1,
      "head_dim": 32,
      "qk_rope_head_dim": 16,
      "q_lora_rank": 32,
      "o_lora_rank": 32,
      "o_groups": 2,
      "num_experts": 16,
      "num_experts_per_tok": 4,
      "shared_experts": 1,
      "base_mlp_dim": 64,
      "base_moe_mlp_dim": 64,
      "indexer_n_heads": 4,
      "indexer_head_dim": 32,
      "indexer_topk": 8,
  }
  kwargs.update(overrides)
  return pyconfig.initialize(["", base_yml], **kwargs)


class DeepSeekV4PrecisionPolicyTest(unittest.TestCase):
  """Tests numeric adherence to deepseek4.precision.yml in production bfloat16 mode."""

  def setUp(self):
    super().setUp()
    self.mesh = jax.make_mesh(
        (1,) * 12,
        (
            "diloco",
            "data",
            "stage",
            "fsdp",
            "fsdp_transpose",
            "context",
            "context_usp_ulysses",
            "context_autoregressive",
            "tensor",
            "tensor_sequence",
            "expert",
            "autoregressive",
        ),
    )

  def test_precision_spec_yaml_schema(self):
    """Spec-derived: verifies deepseek4.precision.yml lists all 5 fp32-promoted operations with official citations."""
    spec_path = os.path.join(
        os.path.dirname(pyconfig.__file__),
        "models",
        "deepseek4.precision.yml",
    )
    self.assertTrue(os.path.exists(spec_path), f"Missing precision spec at {spec_path}")
    with open(spec_path, "r", encoding="utf-8") as f:
      spec = yaml.safe_load(f)
    self.assertEqual(spec["model_family"], "deepseek4")
    self.assertEqual(spec["production_dtype"], "bfloat16")
    op_names = [op["name"] for op in spec["ops"]]
    self.assertEqual(
        op_names,
        [
            "router_gate_logits",
            "rms_norm_variance",
            "attention_softmax_with_sinks",
            "mhc_sinkhorn_normalization",
            "csa_indexer_scores",
        ],
    )
    for op in spec["ops"]:
      self.assertEqual(op["required_compute_dtype"], "float32")
      self.assertIn("inference/model.py:", op["official_citation"])

  def test_router_gate_logits_fp32_policy(self):
    """Spec-derived (inference/model.py:Gate.forward, b/566474598): GateLogit in bf16 mode matches fp32 and differs from bf16."""
    cfg_prod = _make_bf16_config(float32_gate_logits=True)
    cfg_bf16 = _make_bf16_config(float32_gate_logits=False)
    rng = np.random.default_rng(42)
    x_bf16 = jnp.asarray(rng.standard_normal((2, 64, 64)), dtype=jnp.bfloat16)
    w_bf16 = jnp.asarray(rng.standard_normal((64, 16)) * 0.25, dtype=jnp.bfloat16)
    b_bf16 = jnp.asarray(rng.standard_normal((16,)) * 0.05, dtype=jnp.bfloat16)

    with jax.set_mesh(self.mesh):
      gate_prod = moe.GateLogit(
          in_features_shape=64,
          out_features_shape=16,
          model_name="deepseek4-284b",
          mesh=self.mesh,
          rngs=nnx.Rngs(params=0),
          dtype=jnp.float32 if cfg_prod.float32_gate_logits else jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          use_bias=True,
          score_func="sqrtsoftplus",
          matmul_precision=cfg_prod.matmul_precision,
      )
      gate_bf16 = moe.GateLogit(
          in_features_shape=64,
          out_features_shape=16,
          model_name="deepseek4-284b",
          mesh=self.mesh,
          rngs=nnx.Rngs(params=0),
          dtype=jnp.float32 if cfg_bf16.float32_gate_logits else jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          use_bias=True,
          score_func="sqrtsoftplus",
          matmul_precision=cfg_bf16.matmul_precision,
      )
      gate_prod.kernel.set_value(w_bf16)
      gate_prod.bias.set_value(b_bf16)
      gate_bf16.kernel.set_value(w_bf16)
      gate_bf16.bias.set_value(b_bf16)

      out_prod, pre_prod = gate_prod(x_bf16)
      out_bf16, pre_bf16 = gate_bf16(x_bf16)

    # Explicit fp32 reference matching inference/model.py:Gate.forward
    raw_fp32 = jnp.dot(
        x_bf16.astype(jnp.float32),
        w_bf16.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    pre_fp32 = jnp.sqrt(jax.nn.softplus(raw_fp32))
    out_fp32 = pre_fp32 + b_bf16.astype(jnp.float32)

    rel_vs_fp32 = _rel_l2(np.asarray(out_prod), np.asarray(out_fp32))
    rel_vs_bf16 = _rel_l2(np.asarray(out_prod), np.asarray(out_bf16))
    pre_rel_vs_fp32 = _rel_l2(np.asarray(pre_prod), np.asarray(pre_fp32))
    pre_rel_vs_bf16 = _rel_l2(np.asarray(pre_prod), np.asarray(pre_bf16))

    _, idx_prod = jax.lax.top_k(out_prod, 4)
    _, idx_bf16 = jax.lax.top_k(out_bf16, 4)
    slot_disagree = float(np.mean(np.asarray(idx_prod) != np.asarray(idx_bf16)))

    print(
        f"\n[precision:router_gate_logits] rel_vs_fp32={rel_vs_fp32:.3e} rel_vs_bf16={rel_vs_bf16:.3e} "
        f"pre_rel_vs_fp32={pre_rel_vs_fp32:.3e} pre_rel_vs_bf16={pre_rel_vs_bf16:.3e} "
        f"topk_slot_disagree={slot_disagree:.4f}"
    )
    self.assertLess(rel_vs_fp32, 1e-6)
    self.assertLess(pre_rel_vs_fp32, 1e-6)
    self.assertGreater(rel_vs_bf16, 1e-3)
    self.assertGreater(pre_rel_vs_bf16, 1e-3)

  def test_rmsnorm_variance_fp32_policy(self):
    """Spec-derived (inference/model.py:RMSNorm.forward): RMSNorm in bf16 mode accumulates square mean in fp32."""
    rng = np.random.default_rng(43)
    x_bf16 = jnp.asarray(rng.standard_normal((2, 64, 64)) * 3.0, dtype=jnp.bfloat16)
    scale_bf16 = jnp.asarray(rng.standard_normal((64,)) * 0.5 + 1.0, dtype=jnp.bfloat16)
    eps = 1e-6

    with jax.set_mesh(self.mesh):
      norm = normalizations.RMSNorm(
          num_features=64,
          epsilon=eps,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          rngs=nnx.Rngs(params=0),
      )
      norm.scale.set_value(scale_bf16)
      out_prod = norm(x_bf16)

    # Forced fp32 variance reduction then cast to bf16 before scale multiply
    x_f32 = x_bf16.astype(jnp.float32)
    mean2_fp32 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
    out_fp32 = jnp.asarray(x_f32 * jax.lax.rsqrt(mean2_fp32 + eps), jnp.bfloat16) * scale_bf16

    # Forced bf16 variance reduction (buggy if fp32 cast omitted)
    mean2_bf16 = jnp.mean(jax.lax.square(x_bf16), axis=-1, keepdims=True)
    out_bf16 = (x_bf16 * jax.lax.rsqrt(mean2_bf16 + jnp.bfloat16(eps))) * scale_bf16

    rel_vs_fp32 = _rel_l2(np.asarray(out_prod, dtype=np.float32), np.asarray(out_fp32, dtype=np.float32))
    rel_vs_bf16 = _rel_l2(np.asarray(out_prod, dtype=np.float32), np.asarray(out_bf16, dtype=np.float32))
    print(f"[precision:rms_norm_variance] rel_vs_fp32={rel_vs_fp32:.3e} rel_vs_bf16={rel_vs_bf16:.3e}")
    self.assertEqual(rel_vs_fp32, 0.0)
    self.assertGreater(rel_vs_bf16, 1e-4)

  def test_attention_softmax_with_sinks_fp32_policy(self):
    """Spec-derived (inference/model.py:MLA.forward): Attention softmax with sinks computes softmax in fp32."""
    cfg_prod = _make_bf16_config(float32_logits=True)
    cfg_bf16 = _make_bf16_config(float32_logits=False)
    rng = np.random.default_rng(44)
    q = jnp.asarray(rng.standard_normal((1, 16, 4, 32)), dtype=jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((1, 16, 1, 32)), dtype=jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((1, 16, 1, 32)), dtype=jnp.bfloat16)
    sinks = jnp.asarray(rng.standard_normal((4,)), dtype=jnp.bfloat16)

    with jax.set_mesh(self.mesh):
      attn_prod = attention_compressed.CompressedAttention(
          config=cfg_prod,
          num_query_heads=4,
          num_kv_heads=1,
          head_dim=32,
          max_target_length=64,
          mesh=self.mesh,
          attention_kernel="dot_product",
          inputs_q_shape=(1, 16, cfg_prod.emb_dim),
          inputs_kv_shape=(1, 16, cfg_prod.emb_dim),
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          float32_logits=cfg_prod.float32_logits,
          sliding_window_size=64,
          compress_ratio=0,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      attn_bf16 = attention_compressed.CompressedAttention(
          config=cfg_bf16,
          num_query_heads=4,
          num_kv_heads=1,
          head_dim=32,
          max_target_length=64,
          mesh=self.mesh,
          attention_kernel="dot_product",
          inputs_q_shape=(1, 16, cfg_bf16.emb_dim),
          inputs_kv_shape=(1, 16, cfg_bf16.emb_dim),
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          float32_logits=cfg_bf16.float32_logits,
          sliding_window_size=64,
          compress_ratio=0,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      attn_fp32 = attention_compressed.CompressedAttention(
          config=cfg_prod,
          num_query_heads=4,
          num_kv_heads=1,
          head_dim=32,
          max_target_length=64,
          mesh=self.mesh,
          attention_kernel="dot_product",
          inputs_q_shape=(1, 16, cfg_prod.emb_dim),
          inputs_kv_shape=(1, 16, cfg_prod.emb_dim),
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          float32_logits=True,
          sliding_window_size=64,
          compress_ratio=0,
          rngs=nnx.Rngs(params=0, dropout=0),
      )
      attn_prod.sinks.set_value(sinks)
      attn_bf16.sinks.set_value(sinks)
      attn_fp32.sinks.set_value(sinks.astype(jnp.float32))
      out_prod = attn_prod.attention_op(q, k, v, None, None, MODEL_MODE_TRAIN, sinks=attn_prod.sinks.get_value())
      out_bf16 = attn_bf16.attention_op(q, k, v, None, None, MODEL_MODE_TRAIN, sinks=attn_bf16.sinks.get_value())
      out_fp32 = attn_fp32.attention_op(
          q.astype(jnp.float32),
          k.astype(jnp.float32),
          v.astype(jnp.float32),
          None,
          None,
          MODEL_MODE_TRAIN,
          sinks=attn_fp32.sinks.get_value(),
      )

    rel_vs_fp32 = _rel_l2(np.asarray(out_prod, dtype=np.float32), np.asarray(out_fp32, dtype=np.float32))
    rel_vs_bf16 = _rel_l2(np.asarray(out_prod, dtype=np.float32), np.asarray(out_bf16, dtype=np.float32))
    print(f"[precision:attention_softmax_with_sinks] rel_vs_fp32={rel_vs_fp32:.3e} rel_vs_bf16={rel_vs_bf16:.3e}")
    self.assertLess(rel_vs_fp32, 1e-6)
    self.assertGreater(rel_vs_bf16, 1e-4)

  def test_mhc_sinkhorn_fp32_policy(self):
    """Spec-derived (inference/model.py:Block.hc_pre): mhc.sinkhorn runs 20 Sinkhorn iterations in fp32."""
    rng = np.random.default_rng(45)
    t_bf16 = jnp.asarray(rng.standard_normal((2, 16, 4, 4)) * 2.0, dtype=jnp.bfloat16)

    out_prod = mhc.sinkhorn(t_bf16, iters=20)
    out_fp32 = mhc.sinkhorn(t_bf16.astype(jnp.float32), iters=20).astype(jnp.bfloat16)

    # Pure bf16 Sinkhorn without fp32 promotion
    t_b = t_bf16
    eps_b = jnp.bfloat16(1e-6)
    t_b = jax.nn.softmax(t_b, axis=-1) + eps_b
    t_b = t_b / (jnp.sum(t_b, axis=-2, keepdims=True) + eps_b)
    for _ in range(19):
      t_b = t_b / (jnp.sum(t_b, axis=-1, keepdims=True) + eps_b)
      t_b = t_b / (jnp.sum(t_b, axis=-2, keepdims=True) + eps_b)
    out_bf16 = t_b

    rel_vs_fp32 = _rel_l2(np.asarray(out_prod, dtype=np.float32), np.asarray(out_fp32, dtype=np.float32))
    rel_vs_bf16 = _rel_l2(np.asarray(out_prod, dtype=np.float32), np.asarray(out_bf16, dtype=np.float32))
    print(f"[precision:mhc_sinkhorn] rel_vs_fp32={rel_vs_fp32:.3e} rel_vs_bf16={rel_vs_bf16:.3e}")
    self.assertEqual(rel_vs_fp32, 0.0)
    self.assertGreater(rel_vs_bf16, 1e-4)

  def test_csa_indexer_scores_fp32_policy(self):
    """Spec-derived (inference/model.py:Indexer.forward): DeepseekV4Indexer accumulates scores in fp32."""
    cfg = _make_bf16_config()
    rng = np.random.default_rng(46)
    hidden = jnp.asarray(rng.standard_normal((1, 32, 64)), dtype=jnp.bfloat16)
    q_latent = jnp.asarray(rng.standard_normal((1, 32, 32)), dtype=jnp.bfloat16)
    pos = jnp.broadcast_to(jnp.arange(32, dtype=jnp.int32)[None, :], (1, 32))

    with jax.set_mesh(self.mesh):
      rotary = attention_compressed.DeepSeekV4RotaryEmbedding(
          head_dim=cfg.head_dim,
          partial_rotary_factor=cfg.qk_rope_head_dim / cfg.head_dim,
          rope_theta=cfg.compressed_rope_max_timescale,
          rope_type=cfg.rope_type,
          rope_factor=cfg.rope_factor,
          beta_fast=cfg.beta_fast,
          beta_slow=cfg.beta_slow,
          original_max_position_embeddings=cfg.original_max_position_embeddings,
          truncate=cfg.rope_truncate,
          fprop_dtype=jnp.bfloat16,
      )
      indexer = attention_compressed.DeepseekV4Indexer(
          config=cfg,
          compress_ratio=4,
          rotary_embedding=rotary,
          rngs=nnx.Rngs(params=0),
          mesh=self.mesh,
      )
      # Randomize indexer weights so projections are non-degenerate
      for _, idx_var in nnx.to_flat_state(nnx.state(indexer, nnx.Param)):
        arr = rng.standard_normal(idx_var.get_value().shape).astype(np.float32) * 0.2
        idx_var.set_value(jnp.asarray(arr, dtype=jnp.bfloat16))

      _, scores_prod = indexer(hidden, q_latent, pos, model_mode=MODEL_MODE_TRAIN, return_scores=True)

    self.assertEqual(scores_prod.dtype, jnp.float32)
    scores_np = np.asarray(scores_prod)
    finite_mask = np.isfinite(scores_np)
    scores_finite_fp32 = scores_np[finite_mask]
    scores_finite_bf16 = np.asarray(jnp.asarray(scores_finite_fp32, dtype=jnp.bfloat16), dtype=np.float32)
    rel_fp32_vs_bf16 = _rel_l2(scores_finite_fp32, scores_finite_bf16)
    print(f"[precision:csa_indexer_scores] scores.dtype={scores_prod.dtype} rel_fp32_vs_bf16={rel_fp32_vs_bf16:.3e}")
    self.assertGreater(rel_fp32_vs_bf16, 1e-4)


if __name__ == "__main__":
  unittest.main()
