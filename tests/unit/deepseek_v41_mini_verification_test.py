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

# pylint: disable=protected-access
"""Mini-verification suite for DeepSeek-V4.1-Flash LLM backbone on CPU.

Covers:
1. Out-of-the-box config initialization for deepseek4.1-flash and deepseek4.1-flash-mini.
2. Activation quantization (ds41_activation_quant) FP8/FP4 forward values, STE gradients, and MoE path guard.
3. Non-multiple sequence length boundary handling (S % compress_ratio != 0) against the PyTorch reference.
4. Right-padded sequence isolation (decoder_segment_ids == 0 pad tokens do not affect valid token logits).
5. Packed multi-document isolation (decoder_segment_ids == 1, 2 match standalone document runs across all layer types).
6. Individual loss terms (lm_loss, moe_lb_loss), MoEBiasVar updates, scan_layers={False,True} train_step equivalence,
   and multi-step monotonic loss decrease via real train.py functions.
"""

import math
import shutil
import sys
import tempfile
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import attention_deepseek41
from maxtext.layers import deepseek41_quant
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.models import deepseek41
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
import ml_dtypes
import numpy as np
import optax
import pytest
from typing import Any
import json
import os
from safetensors.numpy import save_file as save_numpy_safetensors
from maxtext.layers import engram_deepseek41

TOKENIZER = os.environ.get("DEEPSEEK_V41_TOKENIZER", "deepseek-ai/DeepSeek-V4.1-Flash")
OFFLINE_TOKENIZER = "src/maxtext/assets/tokenizers/qwen3-tokenizer"
MINI_VOCAB_SIZE = 4096

SEQ = 40
COMPRESS_RATIOS = [0, 0, 2, 2, 1, 1, 1, 1]
KV_SOURCES = [2, 4]
INDEX_SOURCES = [2, 4, 6]
CANDIDATE_SOURCE = 4
ENGRAM_LAYERS = [1, 2]

REF_ARGS = {
    "dtype": "fp32",
    "max_batch_size": 2,
    "max_seq_len": 64,
    "vocab_size": 129280,
    "dim": 64,
    "moe_inter_dim": 32,
    "n_layers": len(COMPRESS_RATIOS),
    "n_mtp_layers": 0,
    "n_heads": 4,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
    "score_func": "sqrtsoftplus",
    "route_scale": 1.5,
    "swiglu_limit": 10.0,
    "q_lora_rank": 16,
    "head_dim": 32,
    "rope_head_dim": 8,
    "norm_eps": 1e-20,
    "o_groups": 2,
    "o_lora_rank": 8,
    "window_size": 8,
    "compress_ratios": tuple(COMPRESS_RATIOS),
    "kv_source_layers": tuple(KV_SOURCES),
    "index_source_layers": tuple(INDEX_SOURCES),
    "compress_rope_theta": 160000.0,
    "original_seq_len": 65536,
    "rope_theta": 10000.0,
    "rope_factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "index_n_heads": 16,
    "index_head_dim": 16,
    "index_topk": 3,
    "candidate_source_layer": CANDIDATE_SOURCE,
    "candidate_topk_blocks": 2,
    "candidate_block_size": 4,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": tuple(ENGRAM_LAYERS),
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 50,
    "engram_n_heads": 2,
    "engram_head_dim": 8,
    "engram_pad_id": 2,
    "engram_compressed_vocab_size": 99092,
    "dspark_block_size": 0,
}
REF_ARGS["engram_num_embeddings"] = tuple(
    sum(sum(per_ngram) for per_ngram in layer)
    for layer in engram_deepseek41.engram_primes(
        ENGRAM_LAYERS, REF_ARGS["engram_max_ngram_size"], REF_ARGS["engram_n_heads"], REF_ARGS["engram_vocab_size"]
    )
)

MINI_COMPRESS_RATIOS = [0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
MINI_LAYOUT = {
    "n_layers": len(MINI_COMPRESS_RATIOS),
    "compress_ratios": tuple(MINI_COMPRESS_RATIOS),
    "kv_source_layers": (2, 4, 6, 8),
    "index_source_layers": (2, 4, 6, 8, 10, 12),
    "candidate_source_layer": 8,
    "engram_layer_ids": (1, 6),
}


def _t2j(t: Any) -> jnp.ndarray:
  return jnp.asarray(t.detach().float().numpy())


def _j2np(x: Any) -> np.ndarray:
  return np.asarray(jax.device_get(x), dtype=np.float32)


def _maxtext_config(**overrides: Any) -> Any:
  """Builds a test Config instance with default mini-verification overrides."""
  args = {
      "run_name": "deepseek41_test",
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "deepseek4.1-flash-mini",
      "override_model_config": True,
      "per_device_batch_size": 1.0,
      "max_target_length": SEQ,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "base_emb_dim": REF_ARGS["dim"],
      "base_num_query_heads": REF_ARGS["n_heads"],
      "head_dim": REF_ARGS["head_dim"],
      "qk_rope_head_dim": REF_ARGS["rope_head_dim"],
      "q_lora_rank": REF_ARGS["q_lora_rank"],
      "o_groups": REF_ARGS["o_groups"],
      "o_lora_rank": REF_ARGS["o_lora_rank"],
      "base_mlp_dim": REF_ARGS["moe_inter_dim"],
      "base_moe_mlp_dim": REF_ARGS["moe_inter_dim"],
      "num_experts": REF_ARGS["n_routed_experts"],
      "num_experts_per_tok": REF_ARGS["n_activated_experts"],
      "sparse_matmul": False,
      "capacity_factor": -1,
      "sliding_window_size": REF_ARGS["window_size"],
      "base_num_decoder_layers": len(COMPRESS_RATIOS),
      "compress_ratios": COMPRESS_RATIOS,
      "kv_source_layers": KV_SOURCES,
      "index_source_layers": INDEX_SOURCES,
      "indexer_n_heads": REF_ARGS["index_n_heads"],
      "indexer_head_dim": REF_ARGS["index_head_dim"],
      "indexer_topk": REF_ARGS["index_topk"],
      "candidate_source_layer": CANDIDATE_SOURCE,
      "candidate_topk_blocks": REF_ARGS["candidate_topk_blocks"],
      "candidate_block_size": REF_ARGS["candidate_block_size"],
      "ds41_engram_layers": ENGRAM_LAYERS,
      "ds41_engram_vocab_size": REF_ARGS["engram_vocab_size"],
      "ds41_engram_max_ngram_size": REF_ARGS["engram_max_ngram_size"],
      "ds41_engram_n_heads": REF_ARGS["engram_n_heads"],
      "ds41_engram_head_dim": REF_ARGS["engram_head_dim"],
      "tokenizer_path": TOKENIZER,
  }
  args.update(overrides)
  return pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml"], **args)


def write_numpy_mini_checkpoint(ckpt_dir: str, seed: int = 1) -> tuple[Any, list[np.ndarray]]:
  """Writes a pure-NumPy 14-layer MINI_LAYOUT safetensors checkpoint without PyTorch."""
  cfg = _maxtext_config(
      vocab_size=MINI_VOCAB_SIZE,
      base_num_decoder_layers=MINI_LAYOUT["n_layers"],
      compress_ratios=MINI_COMPRESS_RATIOS,
      kv_source_layers=list(MINI_LAYOUT["kv_source_layers"]),
      index_source_layers=list(MINI_LAYOUT["index_source_layers"]),
      candidate_source_layer=MINI_LAYOUT["candidate_source_layer"],
      ds41_engram_layers=list(MINI_LAYOUT["engram_layer_ids"]),
      ds41_engram_compressed_vocab_size=0,
      tokenizer_path=OFFLINE_TOKENIZER,
      scan_layers=False,
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  rng = np.random.default_rng(seed)

  engram_unpadded = {
      lid: sum(sum(per_ngram) for per_ngram in layer)
      for lid, layer in zip(
          MINI_LAYOUT["engram_layer_ids"],
          engram_deepseek41.engram_primes(
              MINI_LAYOUT["engram_layer_ids"],
              REF_ARGS["engram_max_ngram_size"],
              REF_ARGS["engram_n_heads"],
              REF_ARGS["engram_vocab_size"],
          ),
      )
  }

  init_gate_biases: list[np.ndarray] = []
  hf_tensors: dict[str, np.ndarray] = {
      "embed.weight": np.asarray(model.token_embedder.embedding[...], np.float32)[:MINI_VOCAB_SIZE],
      "norm.weight": np.asarray(model.decoder.decoder_norm.scale[...], np.float32),
      "head.weight": np.asarray(model.decoder.logits_dense.kernel[...], np.float32).T[:MINI_VOCAB_SIZE],
  }

  specs = deepseek41.build_layer_specs(cfg)
  for i, spec in enumerate(specs):
    layer = getattr(model.decoder, f"layers_{i}")
    hf_tensors[f"layers.{i}.attn_norm.weight"] = np.asarray(layer.pre_self_attention_layer_norm.scale[...], np.float32)
    hf_tensors[f"layers.{i}.ffn_norm.weight"] = np.asarray(layer.post_self_attention_layer_norm.scale[...], np.float32)
    for name in ("hc_attn", "hc_ffn"):
      fn_val = np.asarray(getattr(layer, f"{name}_fn")[...], np.float32)
      hf_tensors[f"layers.{i}.{name}_fn"] = fn_val.reshape(fn_val.shape[0], -1).T.copy()
      hf_tensors[f"layers.{i}.{name}_base"] = np.asarray(getattr(layer, f"{name}_base")[...], np.float32)
      hf_tensors[f"layers.{i}.{name}_scale"] = np.asarray(getattr(layer, f"{name}_scale")[...], np.float32)

    attn = layer.self_attention
    for mt_attr, hf_sub in (
        ("wq_a", "wq_a.weight"),
        ("wq_b", "wq_b.weight"),
        ("wkv", "wkv.weight"),
        ("o_b_proj", "wo_b.weight"),
    ):
      k_val = np.asarray(getattr(attn, mt_attr).kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.{hf_sub}"] = k_val.reshape(k_val.shape[0], -1).T.copy()
    oa_val = np.asarray(attn.o_a_proj.kernel[...], np.float32)
    hf_tensors[f"layers.{i}.attn.wo_a.weight"] = oa_val.transpose(0, 2, 1).reshape(-1, oa_val.shape[1]).copy()
    hf_tensors[f"layers.{i}.attn.q_norm.weight"] = np.asarray(attn.q_norm.scale[...], np.float32)
    hf_tensors[f"layers.{i}.attn.kv_norm.weight"] = np.asarray(attn.kv_norm.scale[...], np.float32)
    hf_tensors[f"layers.{i}.attn.attn_sink"] = rng.normal(0.0, 0.2, size=attn.sinks[...].shape).astype(np.float32)

    if spec.is_kv_source:
      cw = np.asarray(attn.compressor.wkv.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.compressor.wkv.weight"] = cw.reshape(cw.shape[0], -1).T.copy()
      if spec.compress_ratio > 1:
        cg = np.asarray(attn.compressor.wgate.kernel[...], np.float32)
        hf_tensors[f"layers.{i}.attn.compressor.wgate.weight"] = cg.reshape(cg.shape[0], -1).T.copy()
      hf_tensors[f"layers.{i}.attn.compressor.norm.weight"] = np.asarray(attn.compressor.norm.scale[...], np.float32)

    if spec.is_index_source:
      iwq = np.asarray(attn.indexer.wq_b.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.indexer.wq_b.weight"] = iwq.reshape(iwq.shape[0], -1).T.copy()
      iwp = np.asarray(attn.indexer.weights_proj.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.attn.indexer.weights_proj.weight"] = iwp.reshape(iwp.shape[0], -1).T.copy()
      if spec.is_kv_source:
        iwk = np.asarray(attn.indexer.wk.kernel[...], np.float32)
        hf_tensors[f"layers.{i}.attn.indexer.wk.weight"] = iwk.reshape(iwk.shape[0], -1).T.copy()
        hf_tensors[f"layers.{i}.attn.indexer.k_norm.weight"] = np.asarray(attn.indexer.k_norm.scale[...], np.float32)

    moe_block = layer.mlp.MoeBlock_0
    gk = np.asarray(moe_block.gate.kernel[...], np.float32)
    hf_tensors[f"layers.{i}.ffn.gate.weight"] = gk.reshape(gk.shape[0], -1).T.copy()
    gb = rng.normal(0.0, 0.05, size=moe_block.gate.bias[...].shape).astype(np.float32)
    hf_tensors[f"layers.{i}.ffn.gate.bias"] = gb
    init_gate_biases.append(gb.copy())
    wi0 = np.asarray(moe_block.wi_0[...], np.float32)
    wi1 = np.asarray(moe_block.wi_1[...], np.float32)
    wo = np.asarray(moe_block.wo[...], np.float32)
    for e in range(cfg.num_experts):
      hf_tensors[f"layers.{i}.ffn.experts.{e}.w1.weight"] = wi0[e].T.copy()
      hf_tensors[f"layers.{i}.ffn.experts.{e}.w3.weight"] = wi1[e].T.copy()
      hf_tensors[f"layers.{i}.ffn.experts.{e}.w2.weight"] = wo[e].T.copy()
    for mt_attr, hf_sub in (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2")):
      sk = np.asarray(getattr(layer.mlp.shared_experts, mt_attr).kernel[...], np.float32)
      hf_tensors[f"layers.{i}.ffn.shared_experts.{hf_sub}.weight"] = sk.reshape(sk.shape[0], -1).T.copy()

    if spec.has_engram:
      rows = engram_unpadded[i]
      hf_tensors[f"layers.{i}.engram.embed.weight"] = rng.normal(
          0.0, 0.2, size=(rows, REF_ARGS["engram_head_dim"])
      ).astype(np.float32)
      ew = np.asarray(layer.engram.wkv.kernel[...], np.float32)
      hf_tensors[f"layers.{i}.engram.wkv.weight"] = ew.reshape(ew.shape[0], -1).T.copy()
      hf_tensors[f"layers.{i}.engram.q_weight"] = np.asarray(layer.engram.q_weight[...], np.float32)
      hf_tensors[f"layers.{i}.engram.k_weight"] = np.asarray(layer.engram.k_weight[...], np.float32)

  save_numpy_safetensors(hf_tensors, os.path.join(ckpt_dir, "model-00001-of-00001.safetensors"))
  with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({"quantization_config": None}, f)
  return cfg, init_gate_biases


torch: Any = None
ref_model: Any = None
ref_runner: Any = None
copy_layer: Any = None
_init_reference: Any = None

try:
  import torch  # pylint: disable=g-import-not-at-top
  from tests.unit import deepseek_v41_vs_reference_test as base_test  # pylint: disable=g-import-not-at-top

  HAS_TORCH = True
  ref_model = getattr(base_test, "ref_model", None)
  ref_runner = getattr(base_test, "ref_runner", None)
  copy_layer = getattr(base_test, "copy_layer", None)
  _init_reference = getattr(base_test, "_init_reference", None)
except (ImportError, ModuleNotFoundError):
  HAS_TORCH = False


_ORIG_MATMUL_PRECISION = None


def setUpModule():
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")


def tearDownModule():
  jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


@pytest.mark.cpu_only
class DeepSeek41ConfigAndQuantTest(unittest.TestCase):
  """Tests default YAML initialization and ds41_activation_quant STE/guards."""

  def test_default_yaml_initialization_without_overrides(self):
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      cfg = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              f"model_name={model_name}",
              "run_name=cfg_init_test",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertEqual(cfg.decoder_block.value, "deepseek41")
      if model_name == "deepseek4.1-flash":
        self.assertTrue(cfg.sparse_matmul)
        self.assertFalse(cfg.ds41_activation_quant)
        self.assertEqual(cfg.capacity_factor, -1.0)
      else:
        self.assertFalse(cfg.sparse_matmul)
        self.assertFalse(cfg.ds41_activation_quant)
        self.assertEqual(cfg.capacity_factor, -1.0)
      self.assertTrue(cfg.float32_gate_logits)

    # Confirm pyconfig.initialize loads deepseek4.1-flash both with and without sparse_matmul on CLI without ValueError
    for sparse_flag in ("sparse_matmul=true", "sparse_matmul=false"):
      cfg_cli = pyconfig.initialize(
          [
              sys.argv[0],
              "src/maxtext/configs/base.yml",
              "model_name=deepseek4.1-flash",
              sparse_flag,
              "run_name=cfg_cli_sparse_test",
              "enable_checkpointing=false",
              "skip_jax_distributed_system=true",
          ]
      )
      self.assertEqual(cfg_cli.decoder_block.value, "deepseek41")

  def test_router_float32_gate_logits_computation(self):
    """Verifies RoutedMoE gate computes router scores in float32 and preserves float64."""
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]), ("fsdp",))
    batch_size = 2
    seq_len = 16
    cfg = _maxtext_config(
        dtype="bfloat16",
        weight_dtype="bfloat16",
        float32_gate_logits=True,
        matmul_precision="default",
        sparse_matmul=False,
        capacity_factor=-1,
    )
    k_init = initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")
    with jax.set_mesh(mesh):
      moe_layer = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=k_init,
          kernel_axes=("embed_router", None),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=nnx.Rngs(params=0),
      )
      self.assertEqual(moe_layer.gate.dtype, jnp.float32)
      self.assertEqual(moe_layer.gate.matmul_precision, "highest")

      key_x = jax.random.PRNGKey(42)
      x_bf16 = jax.random.normal(key_x, (batch_size, seq_len, cfg.emb_dim), dtype=jnp.bfloat16)
      gate_logits, pre_bias_logits = moe_layer.gate(x_bf16)
      self.assertEqual(gate_logits.dtype, jnp.float32)
      self.assertIsNotNone(pre_bias_logits)
      assert pre_bias_logits is not None
      self.assertEqual(pre_bias_logits.dtype, jnp.float32)

      kernel_val = moe_layer.gate.kernel[...]
      ref_raw_fp32 = jnp.einsum(
          "bsd,de->bse",
          x_bf16.astype(jnp.float32),
          kernel_val.astype(jnp.float32),
          precision=jax.lax.Precision.HIGHEST,
      )
      ref_scores_fp32 = jnp.sqrt(jax.nn.softplus(ref_raw_fp32))
      np.testing.assert_allclose(pre_bias_logits, ref_scores_fp32, rtol=1e-5, atol=1e-5)

      # Negative control: float32_gate_logits=False computes gate logits in bfloat16 and diverges.
      cfg_nof32 = _maxtext_config(
          dtype="bfloat16",
          weight_dtype="bfloat16",
          float32_gate_logits=False,
          matmul_precision="default",
          sparse_matmul=False,
          capacity_factor=-1,
      )
      moe_nof32 = moe.RoutedMoE(
          config=cfg_nof32,
          num_experts=cfg_nof32.num_experts,
          num_experts_per_tok=cfg_nof32.num_experts_per_tok,
          mesh=mesh,
          kernel_init=k_init,
          kernel_axes=("embed_router", None),
          intermediate_dim=cfg_nof32.moe_mlp_dim,
          dtype=cfg_nof32.dtype,
          weight_dtype=cfg_nof32.weight_dtype,
          rngs=nnx.Rngs(params=0),
      )
      moe_nof32.gate.kernel[...] = kernel_val
      _, pre_bias_bf16 = moe_nof32.gate(x_bf16)
      assert pre_bias_bf16 is not None
      diff_neg = float(jnp.max(jnp.abs(pre_bias_bf16.astype(jnp.float32) - ref_scores_fp32)))
      self.assertGreater(diff_neg, 1e-4)

      # fp64 parity and upcast safety: float64 must remain float64 when float32_gate_logits is True.
      jax.config.update("jax_enable_x64", True)
      try:
        moe_f64 = moe.RoutedMoE(
            config=cfg,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            mesh=mesh,
            kernel_init=k_init,
            kernel_axes=("embed_router", None),
            intermediate_dim=cfg.moe_mlp_dim,
            dtype=jnp.float64,
            weight_dtype=jnp.float64,
            rngs=nnx.Rngs(params=0),
        )
        self.assertEqual(moe_f64.gate.dtype, jnp.float64)
        x_f64 = x_bf16.astype(jnp.float64)
        _, pre_bias_f64 = moe_f64.gate(x_f64)
        assert pre_bias_f64 is not None
        self.assertEqual(pre_bias_f64.dtype, jnp.float64)
        ref_raw_f64 = jnp.einsum(
            "bsd,de->bse",
            x_f64,
            moe_f64.gate.kernel[...],
            precision=jax.lax.Precision.HIGHEST,
        )
        ref_scores_f64 = jnp.sqrt(jax.nn.softplus(ref_raw_f64))
        frob_rel = float(jnp.linalg.norm(pre_bias_f64 - ref_scores_f64) / jnp.linalg.norm(ref_scores_f64))
        self.assertLessEqual(frob_rel, 1e-12)
      finally:
        jax.config.update("jax_enable_x64", False)

  def test_activation_quant_ste_and_guard(self):
    x = jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).reshape(2, 32)
    q8 = deepseek41_quant.fp8_qdq(x, block_size=32)
    q4 = deepseek41_quant.fp4_qdq(x, block_size=32, e4m3_scale=False)
    q4_e4m3 = deepseek41_quant.fp4_qdq(x, block_size=16, e4m3_scale=True)
    self.assertEqual(q8.shape, x.shape)
    self.assertEqual(q4.shape, x.shape)
    self.assertEqual(q4_e4m3.shape, x.shape)

    # Parity with reference activation quantization and negative control.
    x_np = np.asarray(x, dtype=np.float32)
    g32 = x_np.reshape(2, 1, 32)
    s8_ref = np.exp2(np.ceil(np.log2(np.maximum(np.max(np.abs(g32), axis=-1, keepdims=True), 1e-4) / 448.0))).astype(
        np.float32
    )
    q8_ref = (np.clip(g32 / s8_ref, -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * s8_ref).reshape(
        2, 32
    )
    np.testing.assert_allclose(np.asarray(q8), q8_ref, rtol=0.0, atol=1e-6)

    s4_ref = np.exp2(
        np.ceil(np.log2(np.maximum(np.max(np.abs(g32), axis=-1, keepdims=True), 6.0 * 2.0**-126) / 6.0))
    ).astype(np.float32)
    q4_ref = (np.clip(g32 / s4_ref, -6.0, 6.0).astype(ml_dtypes.float4_e2m1fn).astype(np.float32) * s4_ref).reshape(2, 32)
    np.testing.assert_allclose(np.asarray(q4), q4_ref, rtol=0.0, atol=1e-6)

    g16 = x_np.reshape(2, 2, 16)
    s4_e4m3_ref = (
        (np.maximum(np.max(np.abs(g16), axis=-1, keepdims=True), 6.0 * 2.0**-9) / 6.0)
        .astype(ml_dtypes.float8_e4m3fn)
        .astype(np.float32)
    )
    q4_e4m3_ref = (
        np.clip(g16 / s4_e4m3_ref, -6.0, 6.0).astype(ml_dtypes.float4_e2m1fn).astype(np.float32) * s4_e4m3_ref
    ).reshape(2, 32)
    np.testing.assert_allclose(np.asarray(q4_e4m3), q4_e4m3_ref, rtol=0.0, atol=1e-6)
    self.assertFalse(np.allclose(q8, x))
    self.assertFalse(np.allclose(q4, x))
    self.assertFalse(np.allclose(q4_e4m3, x))

    # Straight-through estimator passes exact unit gradients.
    g8 = jax.grad(lambda t: jnp.sum(deepseek41_quant.fp8_qdq(t, block_size=32)))(x)
    g4 = jax.grad(lambda t: jnp.sum(deepseek41_quant.fp4_qdq(t, block_size=32)))(x)
    np.testing.assert_allclose(np.asarray(g8), np.ones_like(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(g4), np.ones_like(x), rtol=1e-6, atol=1e-6)

    # Forward and backward execution of DeepSeek41DecoderLayer under quant vs no-quant.
    cfg_quant = _maxtext_config(
        dtype="float32",
        weight_dtype="float32",
        ds41_activation_quant=True,
        sparse_matmul=False,
        capacity_factor=-1,
        q_lora_rank=64,
        o_lora_rank=64,
    )
    cfg_noquant = _maxtext_config(
        dtype="float32",
        weight_dtype="float32",
        ds41_activation_quant=False,
        sparse_matmul=False,
        capacity_factor=-1,
        q_lora_rank=64,
        o_lora_rank=64,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    specs = deepseek41.build_layer_specs(cfg_quant)
    with jax.set_mesh(mesh):
      layer_q = deepseek41.DeepSeek41DecoderLayer(
          config=cfg_quant, mesh=mesh, model_mode="train", spec=specs[0], rngs=nnx.Rngs(params=0)
      )
      layer_nq = deepseek41.DeepSeek41DecoderLayer(
          config=cfg_noquant, mesh=mesh, model_mode="train", spec=specs[0], rngs=nnx.Rngs(params=0)
      )

    gdef_q, params_q, other_q = nnx.split(layer_q, nnx.Param, ...)
    gdef_nq, _, other_nq = nnx.split(layer_nq, nnx.Param, ...)
    params_synced = jax.tree.map(lambda t: t, params_q)

    x_in = jax.random.normal(jax.random.PRNGKey(42), (2, 16, cfg_quant.emb_dim), dtype=jnp.float32)
    carry_q = deepseek41.initial_carry(x_in, cfg_quant.mhc_expansion_rate)
    carry_nq = deepseek41.initial_carry(x_in, cfg_noquant.mhc_expansion_rate)
    pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
    tokens = jnp.ones((2, 16), dtype=jnp.int32)
    seg_ids = jnp.ones((2, 16), dtype=jnp.int32)

    def f_q(p, o):
      m = nnx.merge(gdef_q, p, o, copy=True)
      out = m(carry_q, pos, tokens, decoder_segment_ids=seg_ids)
      return jnp.sum(out["h"])

    def f_nq(p, o):
      m = nnx.merge(gdef_nq, p, o, copy=True)
      out = m(carry_nq, pos, tokens, decoder_segment_ids=seg_ids)
      return jnp.sum(out["h"])

    loss_q = f_q(params_q, other_q)
    loss_nq = f_nq(params_synced, other_nq)
    self.assertTrue(jnp.isfinite(loss_q))
    self.assertTrue(jnp.isfinite(loss_nq))
    self.assertGreater(float(jnp.abs(loss_q - loss_nq)), 1e-4)

    grad_q = jax.grad(f_q)(params_q, other_q)
    grad_nq = jax.grad(f_nq)(params_synced, other_nq)
    max_grad_diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(grad_q), jax.tree.leaves(grad_nq)))
    self.assertGreater(max_grad_diff, 1e-4)

    with self.assertRaisesRegex(ValueError, "ds41_activation_quant"):
      _maxtext_config(ds41_activation_quant=True, sparse_matmul=True)


@pytest.mark.cpu_only
@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class DeepSeek41OddSeqLenReferenceTest(unittest.TestCase):
  """Tests non-multiple sequence length boundary handling against the PyTorch reference."""

  @classmethod
  def setUpClass(cls):
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    torch.manual_seed(7)
    cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    cls.ref = ref_runner.build_model(ref_model.ModelArgs(**REF_ARGS), cls.tokenizer)
    _init_reference(cls.ref, seed=7)

  def test_non_multiple_sequence_length_matches_reference(self):
    # S = 35 is odd, so ratio-2 compressor has a trailing 1-token partial group that must be dropped.
    seq_odd = 35
    config = _maxtext_config(max_target_length=seq_odd)
    ids = torch.randint(0, REF_ARGS["vocab_size"], (2, seq_odd))
    with torch.inference_mode():
      hashes = self.ref.engram_hash(ids, 0)
      h = self.ref.embed(ids).unsqueeze(2).repeat(1, 1, REF_ARGS["hc_mult"], 1)
      pre_mix = ref_model.make_identity_pre_mix(h, REF_ARGS["hc_mult"])
      for layer in self.ref.layers:
        if layer.engram is not None:
          h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
        h, pre_mix = layer(h, 0, pre_mix, None)
      ref_collapsed = self.ref.layers[-1].hc_pre(h, pre_mix).numpy()

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    with mesh:
      specs = deepseek41.build_layer_specs(config)
      layers = []
      for i, spec in enumerate(specs):
        layer = deepseek41.DeepSeek41DecoderLayer(config, mesh, "train", spec, rngs=nnx.Rngs(i))
        copy_layer(layer, self.ref.layers[i])
        layers.append(layer)
      tokens = jnp.asarray(ids.numpy(), jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_odd, dtype=jnp.int32), (2, seq_odd))
      seg_ids = jnp.ones((2, seq_odd), dtype=jnp.int32)
      carry = deepseek41.initial_carry(_t2j(self.ref.embed(ids)), REF_ARGS["hc_mult"])
      for layer in layers:
        carry = layer(carry, positions, tokens, decoder_segment_ids=seg_ids)
      mt_collapsed = _j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry["h"], carry["pre_mix"]))
    np.testing.assert_allclose(mt_collapsed, ref_collapsed, rtol=2e-3, atol=2e-3)


@pytest.mark.cpu_only
class DeepSeek41ChallengingInputsTest(unittest.TestCase):
  """Torch-free tests for odd sequence lengths, right-padding isolation, and packed document isolation."""

  @classmethod
  def setUpClass(cls):
    rng = np.random.default_rng(7)
    cls.embed_table = rng.standard_normal((MINI_VOCAB_SIZE, REF_ARGS["dim"])).astype(np.float32)

  def _cfg(self, max_target_length: int, **overrides):
    return _maxtext_config(
        vocab_size=MINI_VOCAB_SIZE,
        max_target_length=max_target_length,
        tokenizer_path=OFFLINE_TOKENIZER,
        ds41_engram_compressed_vocab_size=0,
        **overrides,
    )

  def _build_unscanned_layers(self, config, mesh):
    """Builds unscanned DeepSeek-V4.1 decoder layers with non-zero Engram embeddings."""
    specs = deepseek41.build_layer_specs(config)
    layers = []
    for i, spec in enumerate(specs):
      layer = deepseek41.DeepSeek41DecoderLayer(config, mesh, "train", spec, rngs=nnx.Rngs(100 + i))
      if spec.has_engram:
        engram_rng = np.random.default_rng(200 + i)
        layer.engram.embed[...] = jnp.asarray(
            engram_rng.normal(0.0, 0.2, size=layer.engram.embed[...].shape).astype(np.float32)
        )
      layers.append(layer)
    return layers

  def _embed(self, ids_np: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(self.embed_table[ids_np])

  def test_non_multiple_sequence_length_scanned_vs_unscanned(self):
    ckpt_dir = tempfile.mkdtemp(prefix="dsv41_odd_seq_")
    self.addCleanup(shutil.rmtree, ckpt_dir, True)
    write_numpy_mini_checkpoint(ckpt_dir, seed=5)

    seq_odd = 35
    rng = np.random.default_rng(17)
    tokens = jnp.asarray(rng.integers(0, MINI_VOCAB_SIZE, size=(2, seq_odd), dtype=np.int32))
    positions = jnp.broadcast_to(jnp.arange(seq_odd, dtype=jnp.int32), tokens.shape)
    seg_ids = jnp.ones_like(tokens)
    outs = []
    for scan_layers in (False, True):
      cfg = self._cfg(
          seq_odd,
          base_num_decoder_layers=MINI_LAYOUT["n_layers"],
          compress_ratios=MINI_COMPRESS_RATIOS,
          kv_source_layers=list(MINI_LAYOUT["kv_source_layers"]),
          index_source_layers=list(MINI_LAYOUT["index_source_layers"]),
          candidate_source_layer=MINI_LAYOUT["candidate_source_layer"],
          ds41_engram_layers=list(MINI_LAYOUT["engram_layer_ids"]),
          scan_layers=scan_layers,
          load_parameters_path=ckpt_dir,
          source_checkpoint_layout="safetensors_dynamic",
      )
      mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = model_creation_utils.from_pretrained(cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
      logits = model(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=seg_ids,
          enable_dropout=False,
      )
      outs.append(_j2np(logits))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-5, atol=1e-5)

  def test_right_padding_isolation(self):
    valid_len = 24
    padded_len = 40
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    cfg_valid = self._cfg(valid_len)
    cfg_padded = self._cfg(padded_len)

    rng = np.random.default_rng(21)
    ids_valid = rng.integers(0, MINI_VOCAB_SIZE, size=(1, valid_len), dtype=np.int32)
    pad_tokens_a = rng.integers(0, MINI_VOCAB_SIZE, size=(1, padded_len - valid_len), dtype=np.int32)
    pad_tokens_b = rng.integers(0, MINI_VOCAB_SIZE, size=(1, padded_len - valid_len), dtype=np.int32)
    ids_pad_a = np.concatenate([ids_valid, pad_tokens_a], axis=1)
    ids_pad_b = np.concatenate([ids_valid, pad_tokens_b], axis=1)

    with mesh:
      layers_valid = self._build_unscanned_layers(cfg_valid, mesh)
      layers_padded = self._build_unscanned_layers(cfg_padded, mesh)

      # 1. Unpadded forward
      tok_v = jnp.asarray(ids_valid, jnp.int32)
      pos_v = jnp.arange(valid_len, dtype=jnp.int32)[None, :]
      seg_v = jnp.ones((1, valid_len), dtype=jnp.int32)
      carry_v = deepseek41.initial_carry(self._embed(ids_valid), REF_ARGS["hc_mult"])
      for layer in layers_valid:
        carry_v = layer(carry_v, pos_v, tok_v, decoder_segment_ids=seg_v)
      out_v = _j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry_v["h"], carry_v["pre_mix"]))

      # 2. Padded forward with two different random pad tails
      pos_p = jnp.concatenate(
          [jnp.arange(valid_len, dtype=jnp.int32), jnp.zeros(padded_len - valid_len, dtype=jnp.int32)]
      )[None, :]
      seg_p = jnp.concatenate([jnp.ones(valid_len, dtype=jnp.int32), jnp.zeros(padded_len - valid_len, dtype=jnp.int32)])[
          None, :
      ]
      outs_p = []
      for ids_p in (ids_pad_a, ids_pad_b):
        tok_p = jnp.asarray(ids_p, jnp.int32)
        carry_p = deepseek41.initial_carry(self._embed(ids_p), REF_ARGS["hc_mult"])
        for layer in layers_padded:
          carry_p = layer(carry_p, pos_p, tok_p, decoder_segment_ids=seg_p)
        outs_p.append(_j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry_p["h"], carry_p["pre_mix"])))

    np.testing.assert_allclose(outs_p[0][:, :valid_len], out_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outs_p[0][:, :valid_len], outs_p[1][:, :valid_len], rtol=1e-6, atol=1e-6)

  def test_packed_multi_document_isolation(self):
    """Verifies packed multi-document isolation via decoder_segment_ids.

    Note on sequence packing: packed DS-V4.1 segments must be padded to a multiple of
    lcm(2, candidate_block_size) (8 on full model / 4 on mini) for Doc 2 to match
    standalone execution without boundary-straddling compression artifacts.
    """
    # Doc 1: length 16 (aligned to compress_ratio=2 and candidate_block_size*ratio = 4*2 = 8).
    # Doc 2: length 24. Packed length = 40.
    len1, len2 = 16, 24
    packed_len = len1 + len2
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    cfg1 = self._cfg(len1)
    cfg2 = self._cfg(len2)
    cfg_packed = self._cfg(packed_len)

    rng = np.random.default_rng(29)
    ids1 = rng.integers(0, MINI_VOCAB_SIZE, size=(1, len1), dtype=np.int32)
    ids1_alt = rng.integers(0, MINI_VOCAB_SIZE, size=(1, len1), dtype=np.int32)
    ids2 = rng.integers(0, MINI_VOCAB_SIZE, size=(1, len2), dtype=np.int32)
    ids_packed = np.concatenate([ids1, ids2], axis=1)
    ids_packed_alt = np.concatenate([ids1_alt, ids2], axis=1)

    with mesh:
      layers1 = self._build_unscanned_layers(cfg1, mesh)
      layers2 = self._build_unscanned_layers(cfg2, mesh)
      layers_packed = self._build_unscanned_layers(cfg_packed, mesh)

      def run_standalone(layers, ids_np, length):
        tok = jnp.asarray(ids_np, jnp.int32)
        pos = jnp.arange(length, dtype=jnp.int32)[None, :]
        seg = jnp.ones((1, length), dtype=jnp.int32)
        carry = deepseek41.initial_carry(self._embed(ids_np), REF_ARGS["hc_mult"])
        for layer in layers:
          carry = layer(carry, pos, tok, decoder_segment_ids=seg)
        return _j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry["h"], carry["pre_mix"]))

      out1 = run_standalone(layers1, ids1, len1)
      out2 = run_standalone(layers2, ids2, len2)

      pos_packed = jnp.concatenate([jnp.arange(len1, dtype=jnp.int32), jnp.arange(len2, dtype=jnp.int32)])[None, :]
      seg_packed = jnp.concatenate([jnp.ones(len1, dtype=jnp.int32), jnp.full(len2, 2, dtype=jnp.int32)])[None, :]

      def run_packed(ids_p):
        tok = jnp.asarray(ids_p, jnp.int32)
        carry = deepseek41.initial_carry(self._embed(ids_p), REF_ARGS["hc_mult"])
        for layer in layers_packed:
          carry = layer(carry, pos_packed, tok, decoder_segment_ids=seg_packed)
        return _j2np(deepseek41.DeepSeek41DecoderLayer.hc_pre(carry["h"], carry["pre_mix"]))

      out_packed = run_packed(ids_packed)
      out_packed_alt = run_packed(ids_packed_alt)

    np.testing.assert_allclose(out_packed[:, :len1], out1, rtol=1e-4, atol=5e-5)
    np.testing.assert_allclose(out_packed[:, len1:], out2, rtol=1e-4, atol=5e-5)
    # Changing Doc 1 tokens does not perturb Doc 2 outputs at all.
    np.testing.assert_allclose(out_packed[:, len1:], out_packed_alt[:, len1:], rtol=1e-6, atol=1e-6)


@pytest.mark.cpu_only
class DeepSeek41TrainingAndMiniVerificationTest(unittest.TestCase):
  """Tests real train.py loss_fn, moe_lb_loss, MoEBiasVar updates, scan_layers parity, and multi-step training."""

  @classmethod
  def setUpClass(cls):
    cls.ckpt_dir = tempfile.mkdtemp(prefix="dsv41_train_verify_")
    _, cls.init_gate_biases = write_numpy_mini_checkpoint(cls.ckpt_dir, seed=11)

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.ckpt_dir, ignore_errors=True)
    super().tearDownClass()

  def _make_config(self, scan_layers: bool, **overrides):
    """Constructs a MaxTextConfig for DeepSeek-V4.1 mini training."""
    cfg = _maxtext_config(
        vocab_size=MINI_VOCAB_SIZE,
        base_num_decoder_layers=MINI_LAYOUT["n_layers"],
        compress_ratios=MINI_COMPRESS_RATIOS,
        kv_source_layers=list(MINI_LAYOUT["kv_source_layers"]),
        index_source_layers=list(MINI_LAYOUT["index_source_layers"]),
        candidate_source_layer=MINI_LAYOUT["candidate_source_layer"],
        ds41_engram_layers=list(MINI_LAYOUT["engram_layer_ids"]),
        ds41_engram_compressed_vocab_size=0,
        tokenizer_path=OFFLINE_TOKENIZER,
        scan_layers=scan_layers,
        load_parameters_path=self.ckpt_dir,
        source_checkpoint_layout="safetensors_dynamic",
        load_balance_loss_weight=0.0001,
        routed_bias=True,
        routed_bias_update_rate=0.001,
        learning_rate=1e-3,
        gradient_clipping_threshold=1.0,
        **overrides,
    )
    object.__setattr__(cfg, "model_name", "deepseek4.1-flash-mini")
    return cfg

  def _build_train_state(self, config):
    """Builds TrainStateNNX and shardings on the target mesh."""
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    model = model_creation_utils.from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    tx = optax.adamw(learning_rate=config.learning_rate)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)
    graphdef, pure_state = nnx.split(state)
    state_shardings = jax.tree.map(lambda x: None, pure_state)
    params_shardings = jax.tree.map(lambda x: None, nnx.state(model, nnx.Param))
    return graphdef, pure_state, state_shardings, params_shardings, mesh

  def test_scan_vs_unscanned_train_step_and_aux_losses(self):
    rng = np.random.default_rng(42)
    tokens = jnp.asarray(rng.integers(0, MINI_VOCAB_SIZE, size=(2, SEQ), dtype=np.int32))
    targets = jnp.roll(tokens, -1, axis=1)
    positions = jnp.broadcast_to(jnp.arange(SEQ, dtype=jnp.int32), tokens.shape)
    seg_ids = jnp.ones_like(tokens)
    batch = {
        "inputs": tokens,
        "inputs_position": positions,
        "inputs_segmentation": seg_ids,
        "targets": targets,
        "targets_position": positions,
        "targets_segmentation": seg_ids,
    }

    cfg_unscanned = self._make_config(scan_layers=False)
    cfg_scanned = self._make_config(scan_layers=True)

    g_un, s_un, ss_un, ps_un, mesh_un = self._build_train_state(cfg_unscanned)
    g_sc, s_sc, ss_sc, ps_sc, mesh_sc = self._build_train_state(cfg_scanned)

    with mesh_un:
      s_un_next, metrics_un = train.train_step(g_un, cfg_unscanned, ss_un, ps_un, s_un, batch)
    with mesh_sc:
      s_sc_next, metrics_sc = train.train_step(g_sc, cfg_scanned, ss_sc, ps_sc, s_sc, batch)

    loss_un = float(metrics_un["scalar"]["learning/loss"])
    loss_sc = float(metrics_sc["scalar"]["learning/loss"])
    lb_un = float(metrics_un["scalar"]["learning/moe_lb_loss"])
    lb_sc = float(metrics_sc["scalar"]["learning/moe_lb_loss"])
    gnorm_un = float(metrics_un["scalar"]["learning/grad_norm"])
    gnorm_sc = float(metrics_sc["scalar"]["learning/grad_norm"])

    # 1. Individual loss terms must be finite and non-zero.
    self.assertTrue(np.isfinite(loss_un) and loss_un > 0.0)
    self.assertTrue(np.isfinite(lb_un) and lb_un > 0.0, f"Expected non-zero moe_lb_loss, got {lb_un}")
    self.assertTrue(np.isfinite(gnorm_un) and gnorm_un > 0.0)

    # 2. Scanned and unscanned loss, moe_lb_loss, and grad_norm must match closely.
    np.testing.assert_allclose(loss_sc, loss_un, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lb_sc, lb_un, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gnorm_sc, gnorm_un, rtol=1e-4, atol=1e-4)

    # 3. Verify MoEBiasVar was actually updated and matches layer-by-layer between unscanned and scanned.
    m_un = nnx.merge(g_un, s_un_next).model
    m_sc = nnx.merge(g_sc, s_sc_next).model
    unscanned_biases = [
        np.asarray(getattr(m_un.decoder, f"layers_{i}").mlp.MoeBlock_0.gate.bias.value)
        for i in range(cfg_unscanned.num_decoder_layers)
    ]
    scanned_biases = []
    for k, seg in enumerate(deepseek41.build_segments(cfg_scanned)):
      block = getattr(m_sc.decoder, f"segments_{k}")
      for r in range(seg.repeats):
        for j in range(seg.block_size):
          b_val = np.asarray(getattr(block, f"layers_{j}").mlp.MoeBlock_0.gate.bias.value)
          scanned_biases.append(b_val[r])

    self.assertEqual(len(unscanned_biases), len(scanned_biases))
    for i, (b_u, b_s) in enumerate(zip(unscanned_biases, scanned_biases)):
      np.testing.assert_allclose(b_s, b_u, rtol=1e-6, atol=1e-6, err_msg=f"MoEBiasVar mismatch at layer {i}")
      orig_b = self.init_gate_biases[i]
      self.assertGreater(np.max(np.abs(b_u - orig_b)), 0.0, f"MoEBiasVar did not update at layer {i}")

    # 4. Run 2 more steps on the scanned model and verify monotonic loss reduction on the fixed batch.
    with mesh_sc:
      s_sc_2, m2 = train.train_step(g_sc, cfg_scanned, ss_sc, ps_sc, s_sc_next, batch)
      _, m3 = train.train_step(g_sc, cfg_scanned, ss_sc, ps_sc, s_sc_2, batch)
    loss_2 = float(m2["scalar"]["learning/loss"])
    loss_3 = float(m3["scalar"]["learning/loss"])
    self.assertLess(loss_2, loss_sc)
    self.assertLess(loss_3, loss_2)

  def test_deepseek41_rotary_inv_freq_fp64_to_fp32_bit_exact(self):
    """Verifies DeepSeek41RotaryEmbedding.inv_freq matches host float64->float32 rounding bit-for-bit."""

    def _ref_inv_freq_f64_to_f32(
        *,
        dim: int,
        rope_theta: float,
        rope_factor: float,
        original_seq_len: int,
        beta_fast: int,
        beta_slow: int,
    ) -> np.ndarray:
      base = float(rope_theta)
      freqs = 1.0 / (np.float64(base) ** (np.arange(0, dim, 2, dtype=np.float64) / np.float64(dim)))
      if original_seq_len > 0:

        def corrected_dim(rotations: int) -> float:
          return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / max(high - low, 1e-3), 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / np.float64(rope_factor) * (1.0 - smooth) + freqs * smooth
      return freqs.astype(np.float32)

    configs = [
        ("mla_swa", 64, 160000.0, 40.0, 4096, 32, 1),
        ("indexer", 64, 160000.0, 16.0, 4096, 32, 1),
        ("non_yarn", 64, 160000.0, 40.0, 0, 32, 1),
        ("ds41_yarn", 64, 160000.0, 16.0, 65536, 32, 1),
    ]
    for label, dim, theta, factor, orig_seq, b_fast, b_slow in configs:
      with self.subTest(config=label):
        rot = attention_deepseek41.DeepSeek41RotaryEmbedding(
            head_dim=dim,
            partial_rotary_factor=1.0,
            rope_theta=theta,
            rope_factor=factor,
            original_seq_len=orig_seq,
            beta_fast=b_fast,
            beta_slow=b_slow,
            rngs=nnx.Rngs(0),
        )
        actual = np.asarray(rot.inv_freq, dtype=np.float32)
        expected = _ref_inv_freq_f64_to_f32(
            dim=dim,
            rope_theta=theta,
            rope_factor=factor,
            original_seq_len=orig_seq,
            beta_fast=b_fast,
            beta_slow=b_slow,
        )
        self.assertEqual(actual.shape, (dim // 2,))
        actual_u32 = actual.view(np.uint32)
        expected_u32 = expected.view(np.uint32)
        self.assertTrue(
            np.array_equal(actual_u32, expected_u32),
            msg=f"{label}: uint32 mismatch actual={actual_u32.tolist()} vs expected={expected_u32.tolist()}",
        )

    # Verify bit-exact equality against float64-computed reference cast once to float32.
    ref_inv_freq = np.asarray([1.0 / (float(160000.0) ** (i / 64.0)) for i in range(0, 64, 2)], dtype=np.float64).astype(
        np.float32
    )
    rot_base = attention_deepseek41.DeepSeek41RotaryEmbedding(
        head_dim=64,
        partial_rotary_factor=1.0,
        rope_theta=160000.0,
        rope_factor=40.0,
        original_seq_len=0,
        beta_fast=32,
        beta_slow=1,
        rngs=nnx.Rngs(0),
    )
    actual_base = np.asarray(rot_base.inv_freq, dtype=np.float32)
    self.assertTrue(np.array_equal(actual_base.view(np.uint32), ref_inv_freq.view(np.uint32)))

    # Negative control: perturbed uint32 fails bit-exact comparison.
    perturbed_u32 = ref_inv_freq.view(np.uint32).copy()
    perturbed_u32[3] ^= 1
    self.assertFalse(np.array_equal(actual_base.view(np.uint32), perturbed_u32))

  def test_deepseek41_block_remat_stablehlo_metadata_and_gradient_parity(self):
    """Verifies _run_deepseek41_layer_remat preserves StableHLO loop metadata and matches unscanned gradients."""
    cfg_sc = _maxtext_config(
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        sparse_matmul=False,
        capacity_factor=-1,
        q_lora_rank=64,
        o_lora_rank=64,
        scan_layers=True,
        remat_policy="custom",
    )
    cfg_un = _maxtext_config(
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        sparse_matmul=False,
        capacity_factor=-1,
        q_lora_rank=64,
        o_lora_rank=64,
        scan_layers=False,
        remat_policy="none",
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    specs = deepseek41.build_layer_specs(cfg_sc)
    blk_specs = tuple(specs[i] for i in range(2))
    with jax.set_mesh(mesh):
      remat_policy_fn = jax.checkpoint_policies.save_only_these_names("query_proj")
      blk_sc = deepseek41.DeepSeek41Block(
          config=cfg_sc,
          mesh=mesh,
          model_mode="train",
          specs=blk_specs,
          remat_policy_fn=remat_policy_fn,
          prevent_cse_in_remat=True,
          rngs=nnx.Rngs(params=0),
      )
      blk_un = deepseek41.DeepSeek41Block(
          config=cfg_un,
          mesh=mesh,
          model_mode="train",
          specs=blk_specs,
          remat_policy_fn=None,
          prevent_cse_in_remat=False,
          rngs=nnx.Rngs(params=0),
      )

    embed_shape = blk_sc.layers_1.engram.embed[...].shape
    engram_init = jax.random.normal(jax.random.PRNGKey(123), embed_shape, dtype=jnp.float32) * 0.1
    blk_sc.layers_1.engram.embed[...] = engram_init
    blk_un.layers_1.engram.embed[...] = engram_init

    gdef_sc, params_sc, other_sc = nnx.split(blk_sc, nnx.Param, ...)
    gdef_un, _, other_un = nnx.split(blk_un, nnx.Param, ...)
    params_un = jax.tree.map(lambda x: x, params_sc)

    rng = np.random.default_rng(42)
    tokens_np = rng.integers(1, 100, size=(2, 16), dtype=np.int32)
    tokens = jnp.asarray(tokens_np)
    positions = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32)[None, :], (2, 16))
    x_np = rng.standard_normal((2, 16, cfg_sc.emb_dim)).astype(np.float32)
    x = jnp.asarray(x_np)
    c_sc = deepseek41.initial_carry(x, cfg_sc.mhc_expansion_rate)
    c_un = deepseek41.initial_carry(x, cfg_un.mhc_expansion_rate)

    def f_sc(p, o):
      m = nnx.merge(gdef_sc, p, o, copy=True)
      out, _ = m(c_sc, positions, tokens)
      loss = jnp.sum(out["h"])
      _, _, cur_other = nnx.split(m, nnx.Param, ...)
      return loss, cur_other

    def f_un(p, o):
      m = nnx.merge(gdef_un, p, o, copy=True)
      out, _ = m(c_un, positions, tokens)
      loss = jnp.sum(out["h"])
      _, _, cur_other = nnx.split(m, nnx.Param, ...)
      return loss, cur_other

    (loss_s, out_s), grad_s = jax.value_and_grad(f_sc, has_aux=True)(params_sc, other_sc)
    (loss_u, out_u), grad_u = jax.value_and_grad(f_un, has_aux=True)(params_un, other_un)

    self.assertAlmostEqual(float(loss_s), float(loss_u), delta=1e-4)

    # Check non-zero leaf gradient norms and fp32 parity.
    grad_s_leaves = dict(nnx.to_flat_state(grad_s))
    grad_u_leaves = dict(nnx.to_flat_state(grad_u))
    self.assertEqual(set(grad_s_leaves.keys()), set(grad_u_leaves.keys()))

    for path, gs in grad_s_leaves.items():
      name = "/".join(str(k) for k in path)
      gu = grad_u_leaves[path]
      norm_s = float(jnp.linalg.norm(gs))
      self.assertGreater(norm_s, 1e-7, msg=f"Gradient norm too small for {name}: {norm_s}")
      diff = jnp.abs(gs - gu)
      frob_diff = float(jnp.linalg.norm(diff))
      frob_ref = float(jnp.linalg.norm(gu))
      frob_tol = 1e-5 * max(1.0, frob_ref)
      self.assertLessEqual(frob_diff, frob_tol, msg=f"Frobenius diff {frob_diff} > tol {frob_tol} for {name}")
      atol_leaf = 1e-5 * max(1.0, float(jnp.max(jnp.abs(gu))))
      np.testing.assert_allclose(gs, gu, rtol=1e-5, atol=atol_leaf)

    # Verify sown moe_lb_loss and moe_bias_updates match between blk_sc and blk_un.
    out_s_flat = dict(nnx.to_flat_state(out_s))
    out_u_flat = dict(nnx.to_flat_state(out_u))
    for k in ["layers_0/moe_lb_loss", "layers_0/moe_bias_updates", "layers_1/moe_lb_loss", "layers_1/moe_bias_updates"]:
      found = False
      for p, v_s in out_s_flat.items():
        name = "/".join(str(x) for x in p)
        if k in name:
          found = True
          v_u = out_u_flat[p]
          np.testing.assert_allclose(v_s, v_u, rtol=1e-5, atol=1e-5)
      self.assertTrue(found, f"Intermediate {k} not found in state!")

    # StableHLO metadata assertion.
    lowered_sc = jax.jit(f_sc).lower(params_sc, other_sc).as_text()
    lowered_un = jax.jit(f_un).lower(params_un, other_un).as_text()
    self.assertIn("skip-simplify-while-loops_trip-count-one", lowered_sc)
    self.assertNotIn("skip-simplify-while-loops_trip-count-one", lowered_un)

    # Negative control: perturbing one parameter leaf must break parity.
    params_bad = jax.tree.map(lambda x: x, params_sc)
    val = getattr(params_bad, "layers_1").engram.q_weight[...]
    getattr(params_bad, "layers_1").engram.q_weight[...] = val + 0.1
    (loss_bad, _), _ = jax.value_and_grad(f_sc, has_aux=True)(params_bad, other_sc)
    bad_diff = float(jnp.abs(loss_s - loss_bad))
    self.assertGreater(bad_diff, 1e-4)


if __name__ == "__main__":
  unittest.main()
