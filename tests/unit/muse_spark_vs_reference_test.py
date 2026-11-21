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

"""Unit tests for MaxText Muse-Spark vs HuggingFace MuseSparkForConditionalGeneration."""

from __future__ import annotations

import copy
import os
import tempfile
import types
import unittest

import pytest

try:
  import torch
  from transformers.models.muse_spark import configuration_muse_spark, modeling_muse_spark

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers.models.muse_spark not available"),
]

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_shape, param_mapping
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.models import muse_spark
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from tests.utils import zero_gen_logits_checker

BASE_CONFIG = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

# Minimal dimensions for fast CPU CI execution while exercising every architectural path:
# - GQA (4 query heads, 2 kv heads, head_dim=16)
# - Sliding window (size=4 < seq=6) on sliding layers vs full causal attention on every 4th layer from the end
# - Width-bottlenecked Sparse MoE (EMB=32 -> MOE_IN=16 -> MOE_MLP=24 -> MOE_IN=16 -> EMB=32, 4 experts, top-2)
EMB, Q_HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 16
MLP, MOE_IN, MOE_MLP, EXPERTS, TOPK = 64, 16, 24, 4, 2
SLIDING_WINDOW = 4
VOCAB = 64
ATOL, RTOL = 1e-4, 1e-4


def t2j(tensor: torch.Tensor) -> jnp.ndarray:
  return jnp.asarray(tensor.detach().cpu().float().numpy())


def assert_close(actual, expected, atol=ATOL, rtol=RTOL, msg=""):
  actual_np = np.asarray(actual, dtype=np.float32)
  expected_np = (
      expected.detach().cpu().float().numpy()
      if isinstance(expected, torch.Tensor)
      else np.asarray(expected, dtype=np.float32)
  )
  np.testing.assert_allclose(actual_np, expected_np, atol=atol, rtol=rtol, err_msg=msg)


def randomize_(module: torch.nn.Module, seed: int = 0, std: float = 0.08):
  """Fills a torch module with seeded random weights (including zero-initialized centered norms and router bias)."""
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in module.named_parameters():
      noise = std * torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype)
      if name.endswith("norm.weight") and not any(
          k in name for k in ("input_layernorm", "pre_feedforward_layernorm", "pre_expert_norm", "post_expert_norm")
      ):
        p.copy_(1.0 + noise)
      else:
        p.copy_(noise)
    for name, b in module.named_buffers():
      if "e_score_correction_bias" in name:
        b.copy_(std * torch.randn(b.shape, generator=gen, dtype=torch.float32).to(b.dtype))


def make_config(**kwargs):
  """Builds a small MaxText muse-spark-816b-a42b config for tests."""
  overrides = dict(  # pylint: disable=use-dict-literal
      model_name="muse-spark-816b-a42b",
      override_model_config=True,
      base_emb_dim=EMB,
      base_num_query_heads=Q_HEADS,
      base_num_kv_heads=KV_HEADS,
      head_dim=HEAD_DIM,
      base_mlp_dim=MLP,
      moe_expert_input_dim=MOE_IN,
      base_moe_mlp_dim=MOE_MLP,
      num_experts=EXPERTS,
      num_experts_per_tok=TOPK,
      shared_experts=0,
      routed_scaling_factor=1.0,
      routed_score_func="sigmoid",
      routed_bias=True,
      norm_topk_prob=True,
      first_num_dense_layers=0,
      inhomogeneous_layer_cycle_interval=4,
      sliding_window_size=SLIDING_WINDOW,
      vocab_size=VOCAB,
      base_num_decoder_layers=6,
      per_device_batch_size=1,
      max_target_length=8,
      max_prefill_predict_length=4,
      max_position_embeddings=64,
      original_max_position_embeddings=64,
      scan_layers=False,
      attention="dot_product",
      sparse_matmul=False,
      capacity_factor=-1,
      matmul_precision="highest",
      dtype="float32",
      weight_dtype="float32",
      float32_logits=True,
      float32_qk_product=True,
      dropout_rate=0.0,
      skip_jax_distributed_system=True,
      hardware="cpu",
  )
  overrides.update(kwargs)
  return pyconfig.initialize(["", BASE_CONFIG], **overrides)


def make_hf_text_config(num_layers=6):
  # pylint: disable=unexpected-keyword-arg
  return configuration_muse_spark.MuseSparkTextConfig(
      vocab_size=VOCAB,
      hidden_size=EMB,
      intermediate_size=MLP,
      moe_hidden_size=MOE_IN,
      moe_intermediate_size=MOE_MLP,
      num_hidden_layers=num_layers,
      num_attention_heads=Q_HEADS,
      num_key_value_heads=KV_HEADS,
      head_dim=HEAD_DIM,
      num_local_experts=EXPERTS,
      num_experts_per_tok=TOPK,
      num_shared_experts=0,
      routed_scaling_factor=1.0,
      sliding_window=SLIDING_WINDOW,
      rms_norm_eps=1e-5,
      post_norm_eps=1e-8,
      qk_scale_factor=5.4730064863875,
      output_multiplier=0.17677669529663687,
      residual_gate_temperature=0.3,
      final_logit_softcapping=20.0,
      rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
      max_position_embeddings=64,
      attention_dropout=0.0,
      tie_word_embeddings=False,
      pad_token_id=0,
      bos_token_id=1,
      eos_token_id=2,
      attn_implementation="eager",
  )


def make_hf_full_config(num_layers=6):
  """Builds a small MuseSparkConfig for unit testing."""
  # pylint: disable=unexpected-keyword-arg
  text_cfg = make_hf_text_config(num_layers=num_layers)
  vision_cfg = configuration_muse_spark.MuseSparkVisionConfig(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=2,
      num_attention_heads=2,
      pos_emb_height=4,
      pos_emb_width=4,
  )
  audio_cfg = configuration_muse_spark.MuseSparkAudioConfig(
      depths=[1, 1],
      hidden_sizes=[8, 16],
      patch_sizes=[2, 2],
      num_hidden_layers=2,
      out_hidden_size=EMB,
  )
  cfg = configuration_muse_spark.MuseSparkConfig(
      text_config=text_cfg.to_dict(),
      vision_config=vision_cfg.to_dict(),
      audio_config=audio_cfg.to_dict(),
      projector_hidden_size=16,
      out_hidden_size=24,
  )
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  cfg.text_config._attn_implementation = "eager"  # pylint: disable=protected-access
  return cfg


def set_dense(mt_dense, hf_linear, out_shape=None):
  w = t2j(hf_linear.weight).T
  if out_shape is not None:
    w = w.reshape(w.shape[0], *out_shape)
  mt_dense.kernel[...] = w


def copy_attention(hf_attn, mt_attn):
  set_dense(mt_attn.query, hf_attn.q_proj, out_shape=(Q_HEADS, HEAD_DIM))
  set_dense(mt_attn.key, hf_attn.k_proj, out_shape=(KV_HEADS, HEAD_DIM))
  set_dense(mt_attn.value, hf_attn.v_proj, out_shape=(KV_HEADS, HEAD_DIM))
  set_dense(mt_attn.gate_proj, hf_attn.gate_proj, out_shape=(Q_HEADS, HEAD_DIM))
  mt_attn.out.kernel[...] = t2j(hf_attn.o_proj.weight).T.reshape(Q_HEADS, HEAD_DIM, EMB)


def copy_moe(hf_moe, mt_moe):
  """Copies HuggingFace MuseSparkSparseMoeBlock weights into MaxText MuseSparkMoE."""
  mt_moe.gate.kernel[...] = t2j(hf_moe.gate.weight).T
  mt_moe.gate.bias[...] = t2j(hf_moe.gate.e_score_correction_bias)
  set_dense(mt_moe.pre_expert_proj, hf_moe.pre_expert_proj)
  mt_moe.pre_expert_norm.scale[...] = t2j(hf_moe.pre_expert_norm.weight)
  gate_up = t2j(hf_moe.experts.gate_up_proj)
  gate, up = jnp.split(gate_up, 2, axis=1)
  mt_moe.wi_0[...] = jnp.swapaxes(gate, -1, -2)
  mt_moe.wi_1[...] = jnp.swapaxes(up, -1, -2)
  mt_moe.wo[...] = jnp.swapaxes(t2j(hf_moe.experts.down_proj), -1, -2)
  mt_moe.post_expert_norm.scale[...] = t2j(hf_moe.experts.post_expert_norm.weight)
  set_dense(mt_moe.post_expert_proj, hf_moe.post_expert_proj)


def copy_decoder_layer(hf_layer, mt_layer):
  """Copies HuggingFace MuseSparkDecoderLayer weights into MaxText MuseSparkDecoderLayer."""
  mt_layer.input_layernorm.scale[...] = t2j(hf_layer.input_layernorm.weight)
  copy_attention(hf_layer.self_attn, mt_layer.self_attention)
  mt_layer.post_attention_residual_gate.gate[...] = t2j(hf_layer.post_attention_residual_gate.gate)
  mt_layer.pre_feedforward_layernorm.scale[...] = t2j(hf_layer.pre_feedforward_layernorm.weight)
  if mt_layer.is_moe:
    copy_moe(hf_layer.mlp, mt_layer.moe_block)
  else:
    set_dense(mt_layer.mlp.wi_0, hf_layer.mlp.gate_proj)
    set_dense(mt_layer.mlp.wi_1, hf_layer.mlp.up_proj)
    set_dense(mt_layer.mlp.wo, hf_layer.mlp.down_proj)
  mt_layer.post_feedforward_layernorm.scale[...] = t2j(hf_layer.post_feedforward_layernorm.weight)
  mt_layer.post_feedforward_residual_gate.gate[...] = t2j(hf_layer.post_feedforward_residual_gate.gate)


def copy_full_model(hf_model, mt_model):
  mt_model.token_embedder.embedding[...] = t2j(hf_model.model.language_model.embed_tokens.weight)
  mt_model.decoder.decoder_norm.scale[...] = t2j(hf_model.model.language_model.norm.weight)
  mt_model.decoder.logits_dense.kernel[...] = t2j(hf_model.lm_head.weight).T
  for i, hf_layer in enumerate(hf_model.model.language_model.layers):
    copy_decoder_layer(hf_layer, getattr(mt_model.decoder, f"layers_{i}"))


def convert_hf_state_to_maxtext(hf_state, hf_cfg_dict, cfg_m, scan, save_dtype="float32"):
  """Converts a {hf_key: array_or_tensor} dict into the MaxText checkpoint tree through the `to_maxtext` code path.

  Every mapping/hook is linear (transpose, reshape, split, stack), so the same conversion maps HF gradients
  onto the MaxText parameter layout.
  """
  abstract_dict, treedef = to_maxtext.get_maxtext_model_info(cfg_m)
  pmap = param_mapping.MUSE_SPARK_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, cfg_m, scan_layers=scan)
  hmap = param_mapping.MUSE_SPARK_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, cfg_m, scan_layers=scan, saving_to_hf=False)
  final_mt_weights = [None] * len(abstract_dict)
  for mt_key, hf_targets in pmap.items():
    idx_or_indices, shape_or_shapes = to_maxtext._get_maxtext_indices_and_shapes(  # pylint: disable=protected-access
        mt_key, abstract_dict
    )
    param_dtype = to_maxtext._param_save_dtype(mt_key, save_dtype, cfg_m.model_name)  # pylint: disable=protected-access
    load_fn = to_maxtext._get_hf_loading_function(  # pylint: disable=protected-access
        hf_targets,
        lambda k, dt=param_dtype: to_maxtext._convert_tensor_to_numpy(  # pylint: disable=protected-access
            hf_state[k], dt
        ),
        hmap.get(mt_key),
        shape_or_shapes,
        cfg_m,
        mt_key,
    )
    to_maxtext._get_maxtext_weight(  # pylint: disable=protected-access
        load_fn,
        idx_or_indices,
        shape_or_shapes,
        mt_key,
        final_mt_weights,
        param_dtype,
        use_lazy_load=False,
    )
  return jax.tree_util.tree_unflatten(treedef, final_mt_weights), pmap, abstract_dict


def lookup_converted(converted_tree, col, path):
  """Returns the converted array for an NNX variable path in collection `col` ("params" or "MoEBiasVar")."""
  node = converted_tree[col]
  for p in to_maxtext.param_key_parts_from_path(path):
    node = node[p]
  return jnp.asarray(node)


def load_converted_tree(model, converted_tree):
  """Loads a converted MaxText checkpoint tree (params + MoEBiasVar) into an NNX model."""
  dst_state = nnx.state(model, nnx.Any(nnx.Param, moe.MoEBiasVar))
  new_flat = []
  for path, var in nnx.to_flat_state(dst_state):
    col = "MoEBiasVar" if isinstance(var, moe.MoEBiasVar) else "params"
    value = lookup_converted(converted_tree, col, path)
    if value.shape != var.get_value().shape:
      raise ValueError(f"Shape mismatch for {path}: {value.shape} vs {var.get_value().shape}")
    new_flat.append((path, var.replace(value)))
  nnx.update(model, nnx.from_flat_state(new_flat))


class MuseSparkLayersTest(unittest.TestCase):
  """CPU unit tests comparing MaxText Muse-Spark modules against transformers.models.muse_spark."""

  @classmethod
  def setUpClass(cls):
    torch.set_grad_enabled(False)
    cls.hf_text_cfg = make_hf_text_config()
    cls.hf_full_cfg = make_hf_full_config()
    cls.config = make_config()
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.config), cls.config.mesh_axes)

  def _inputs_3d(self, batch=1, seq=6, dim=EMB, seed=0):
    x = np.random.RandomState(seed).randn(batch, seq, dim).astype(np.float32)
    return torch.tensor(x), jnp.asarray(x)

  def _make_attn(self, is_full_attention: bool, model_mode=MODEL_MODE_TRAIN, q_len=6):
    return muse_spark.MuseSparkAttention(
        config=self.config,
        num_query_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_target_length=self.config.max_target_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, q_len, EMB),
        inputs_kv_shape=(1, q_len, EMB),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        is_full_attention=is_full_attention,
        model_mode=model_mode,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

  def test_layer_pattern_and_scan_layout_match_hf(self):
    full_hf = make_hf_text_config(num_layers=62)

    def stub(n, first_dense=0):
      return types.SimpleNamespace(
          first_num_dense_layers=first_dense,
          inhomogeneous_layer_cycle_interval=4,
          num_decoder_layers=n,
          num_experts=EXPERTS,
      )

    cfg62 = stub(62)
    for i in range(62):
      self.assertEqual(
          muse_spark.is_full_attention_layer(i, cfg62), full_hf.layer_types[i] == "full_attention", f"layer {i}"
      )
      self.assertEqual(muse_spark.is_full_attention_layer(i, cfg62), full_hf.layer_rope_theta[i] == 0, f"layer {i} rope")
      self.assertEqual(muse_spark.is_moe_layer(i, cfg62), full_hf.mlp_layer_types[i] == "sparse", f"layer {i} moe")
    self.assertEqual(muse_spark.scan_layout(cfg62), (2, 15, 62))
    self.assertEqual(muse_spark.scan_layout(stub(6)), (2, 1, 6))
    self.assertEqual(muse_spark.scan_layout(stub(4)), (0, 1, 4))
    self.assertEqual(muse_spark.scan_layout(stub(3)), (3, 0, 3))

  def test_norms_and_residual_gate_match_hf(self):
    xt, xj = self._inputs_3d(seed=1)

    # 1. MuseSparkRMSNorm (with_scale=True and with_scale=False)
    hf_rms = modeling_muse_spark.MuseSparkRMSNorm(EMB, eps=1e-5, with_scale=True).eval()
    randomize_(hf_rms, seed=2)
    mt_rms = muse_spark.MuseSparkRMSNorm(EMB, epsilon=1e-5, with_scale=True, rngs=nnx.Rngs(0))
    mt_rms.scale[...] = t2j(hf_rms.weight)
    assert_close(mt_rms(xj), hf_rms(xt), msg="MuseSparkRMSNorm with_scale=True")

    hf_rms_noscale = modeling_muse_spark.MuseSparkRMSNorm(EMB, eps=1e-8, with_scale=False).eval()
    mt_rms_noscale = muse_spark.MuseSparkRMSNorm(EMB, epsilon=1e-8, with_scale=False, rngs=nnx.Rngs(0))
    assert_close(mt_rms_noscale(xj), hf_rms_noscale(xt), msg="MuseSparkRMSNorm with_scale=False")

    # 2. MuseSparkCenteredRMSNorm (fp32 and bf16 `(1.0 + weight).float()` parity)
    hf_crms = modeling_muse_spark.MuseSparkTextCenteredRMSNorm(EMB, eps=1e-5).eval()
    randomize_(hf_crms, seed=3)
    mt_crms = muse_spark.MuseSparkCenteredRMSNorm(EMB, epsilon=1e-5, rngs=nnx.Rngs(0))
    mt_crms.scale[...] = t2j(hf_crms.weight)
    assert_close(mt_crms(xj), hf_crms(xt), msg="MuseSparkCenteredRMSNorm fp32")

    hf_crms_bf16 = copy.deepcopy(hf_crms).to(torch.bfloat16)
    mt_crms_bf16 = muse_spark.MuseSparkCenteredRMSNorm(
        EMB, epsilon=1e-5, dtype=jnp.bfloat16, weight_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)
    )
    mt_crms_bf16.scale[...] = t2j(hf_crms_bf16.weight).astype(jnp.bfloat16)
    assert_close(
        mt_crms_bf16(xj.astype(jnp.bfloat16)),
        hf_crms_bf16(xt.to(torch.bfloat16)),
        atol=1e-6,
        rtol=1e-6,
        msg="MuseSparkCenteredRMSNorm bf16 weight promotion",
    )

    # 3. MuseSparkInputScaledRMSNorm (test both fp32 and bf16 input/weight rounding)
    hf_irms = modeling_muse_spark.MuseSparkTextInputScaledRMSNorm(EMB, eps=1e-8).eval()
    randomize_(hf_irms, seed=4)
    mt_irms = muse_spark.MuseSparkInputScaledRMSNorm(EMB, epsilon=1e-8, rngs=nnx.Rngs(0))
    mt_irms.scale[...] = t2j(hf_irms.weight)
    assert_close(mt_irms(xj), hf_irms(xt), msg="MuseSparkInputScaledRMSNorm fp32")
    assert_close(
        mt_irms(xj.astype(jnp.bfloat16)),
        hf_irms(xt.to(torch.bfloat16)),
        atol=1e-2,
        rtol=1e-2,
        msg="MuseSparkInputScaledRMSNorm bf16",
    )

    hf_irms_bf16 = copy.deepcopy(hf_irms).to(torch.bfloat16)
    mt_irms_bf16 = muse_spark.MuseSparkInputScaledRMSNorm(
        EMB, epsilon=1e-8, dtype=jnp.bfloat16, weight_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)
    )
    mt_irms_bf16.scale[...] = t2j(hf_irms_bf16.weight).astype(jnp.bfloat16)
    assert_close(
        mt_irms_bf16(xj.astype(jnp.bfloat16)),
        hf_irms_bf16(xt.to(torch.bfloat16)),
        atol=1e-6,
        rtol=1e-6,
        msg="MuseSparkInputScaledRMSNorm bf16 weight promotion",
    )

    # 4. MuseSparkResidualGate (always stores `gate` in float32 even when weight_dtype=bfloat16)
    hf_gate = modeling_muse_spark.MuseSparkResidualGate(self.hf_text_cfg).eval()
    randomize_(hf_gate, seed=5)
    mt_gate = muse_spark.MuseSparkResidualGate(
        EMB,
        temperature=self.hf_text_cfg.residual_gate_temperature,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )
    self.assertEqual(mt_gate.gate.get_value().dtype, jnp.float32)
    mt_gate.gate[...] = t2j(hf_gate.gate)
    yt, yj = self._inputs_3d(seed=6)
    assert_close(mt_gate(xj, yj), hf_gate(xt, yt), msg="MuseSparkResidualGate")

  def test_sliding_rope_and_full_nope_attention_match_hf(self):
    seq = 6  # seq=6 > SLIDING_WINDOW=4 actively tests sliding window mask on layer 0 vs full mask on layer 1
    xt, xj = self._inputs_3d(batch=1, seq=seq, seed=7)
    pos_t = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    pos_j = jnp.arange(seq, dtype=jnp.int32)[None, :]
    seg_j = jnp.ones((1, seq), dtype=jnp.int32)
    rotary = modeling_muse_spark.MuseSparkTextRotaryEmbedding(self.hf_text_cfg)
    pos_emb = rotary(xt, pos_t)

    mask_kwargs = {
        "config": self.hf_text_cfg,
        "inputs_embeds": xt,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": pos_t,
    }
    sliding_mask = modeling_muse_spark.create_sliding_window_causal_mask(**mask_kwargs)
    full_mask = modeling_muse_spark.create_causal_mask(**mask_kwargs)

    # Layer 0: sliding_attention + RoPE
    hf_attn_sliding = modeling_muse_spark.MuseSparkTextAttention(self.hf_text_cfg, layer_idx=0).eval()
    randomize_(hf_attn_sliding, seed=8)
    mt_attn_sliding = self._make_attn(is_full_attention=False, q_len=seq)
    self.assertTrue(mt_attn_sliding.attention_op.float32_qk_product)
    self.assertTrue(mt_attn_sliding.attention_op.float32_logits)
    copy_attention(hf_attn_sliding, mt_attn_sliding)
    ref_sliding, _ = hf_attn_sliding(xt, position_embeddings=pos_emb, attention_mask=sliding_mask, position_ids=pos_t)
    out_sliding, _ = mt_attn_sliding(
        xj, xj, pos_j, decoder_segment_ids=seg_j, model_mode=MODEL_MODE_TRAIN, deterministic=True
    )
    assert_close(out_sliding, ref_sliding, msg="Sliding-window RoPE attention")

    # Layer 1: full_attention + NoPE
    hf_attn_full = modeling_muse_spark.MuseSparkTextAttention(self.hf_text_cfg, layer_idx=1).eval()
    randomize_(hf_attn_full, seed=9)
    mt_attn_full = self._make_attn(is_full_attention=True, q_len=seq)
    copy_attention(hf_attn_full, mt_attn_full)
    ref_full, _ = hf_attn_full(xt, position_embeddings=None, attention_mask=full_mask, position_ids=pos_t)
    out_full, _ = mt_attn_full(xj, xj, pos_j, decoder_segment_ids=seg_j, model_mode=MODEL_MODE_TRAIN, deterministic=True)
    assert_close(out_full, ref_full, msg="Full NoPE attention")

  def test_cached_prefill_and_ar_decode_match_full_forward(self):
    total, prompt, pad = 6, 3, self.config.max_prefill_predict_length
    _, xj = self._inputs_3d(1, total, seed=10)
    pos_full = jnp.arange(total, dtype=jnp.int32)[None, :]
    seg_full = jnp.ones((1, total), dtype=jnp.int32)

    for is_full, layer_idx in ((False, 0), (True, 1)):
      hf_attn = modeling_muse_spark.MuseSparkTextAttention(self.hf_text_cfg, layer_idx=layer_idx).eval()
      randomize_(hf_attn, seed=11 + layer_idx)
      mt_attn = self._make_attn(is_full_attention=is_full, model_mode=MODEL_MODE_PREFILL, q_len=pad)
      copy_attention(hf_attn, mt_attn)

      ref_out, _ = mt_attn(
          xj, xj, pos_full, decoder_segment_ids=seg_full, model_mode=MODEL_MODE_TRAIN, deterministic=True
      )

      xp = jnp.concatenate([xj[:, :prompt], jnp.zeros((1, pad - prompt, EMB), dtype=jnp.float32)], axis=1)
      pos_p = jnp.arange(pad, dtype=jnp.int32)[None, :]
      seg_p = (jnp.arange(pad) < prompt).astype(jnp.int32)[None, :]
      out_p, _ = mt_attn(xp, xp, pos_p, decoder_segment_ids=seg_p, model_mode=MODEL_MODE_PREFILL, deterministic=True)
      assert_close(out_p[:, :prompt], ref_out[:, :prompt], msg=f"prefill is_full={is_full}")

      for t in range(prompt, total):
        xt_step = xj[:, t : t + 1]
        pos_step = jnp.array([[t]], dtype=jnp.int32)
        seg_step = jnp.ones((1, 1), dtype=jnp.int32)
        out_t, _ = mt_attn(
            xt_step,
            xt_step,
            pos_step,
            decoder_segment_ids=seg_step,
            model_mode=MODEL_MODE_AUTOREGRESSIVE,
            deterministic=True,
        )
        assert_close(out_t[:, 0], ref_out[:, t], msg=f"AR step {t} is_full={is_full}")

  def test_moe_block_matches_hf(self):
    hf_moe = modeling_muse_spark.MuseSparkMoE(self.hf_text_cfg).eval()
    randomize_(hf_moe, seed=14)
    mt_moe = muse_spark.MuseSparkMoE(self.config, self.mesh, rngs=nnx.Rngs(params=0))
    copy_moe(hf_moe, mt_moe)

    xt, xj = self._inputs_3d(batch=2, seq=5, seed=15)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mt_out, lb_loss = mt_moe(xj)
    assert_close(mt_out, hf_moe(xt), msg="MuseSparkMoE")
    self.assertIsNone(lb_loss)

    # Verify bfloat16 MuseSparkMoE parity (fp32 router kernel/bias + top_k_weights, bf16 experts/projections).
    hf_moe_bf16 = copy.deepcopy(hf_moe).to(torch.bfloat16)
    hf_moe_bf16.gate.weight.data = hf_moe.gate.weight.data.float()
    hf_moe_bf16.gate.e_score_correction_bias.data = hf_moe.gate.e_score_correction_bias.data.float()
    cfg_bf16 = make_config(dtype="bfloat16", weight_dtype="bfloat16", matmul_precision="default")
    mt_moe_bf16 = muse_spark.MuseSparkMoE(cfg_bf16, self.mesh, rngs=nnx.Rngs(params=0))
    mt_moe_bf16.gate.kernel[...] = t2j(hf_moe_bf16.gate.weight).T
    mt_moe_bf16.gate.bias[...] = t2j(hf_moe_bf16.gate.e_score_correction_bias)
    mt_moe_bf16.pre_expert_proj.kernel[...] = t2j(hf_moe_bf16.pre_expert_proj.weight).T.astype(jnp.bfloat16)
    mt_moe_bf16.pre_expert_norm.scale[...] = t2j(hf_moe_bf16.pre_expert_norm.weight).astype(jnp.bfloat16)
    gate_up_bf16 = t2j(hf_moe_bf16.experts.gate_up_proj).astype(jnp.bfloat16)
    g_bf16, u_bf16 = jnp.split(gate_up_bf16, 2, axis=1)
    mt_moe_bf16.wi_0[...] = jnp.swapaxes(g_bf16, -1, -2)
    mt_moe_bf16.wi_1[...] = jnp.swapaxes(u_bf16, -1, -2)
    mt_moe_bf16.wo[...] = jnp.swapaxes(t2j(hf_moe_bf16.experts.down_proj).astype(jnp.bfloat16), -1, -2)
    mt_moe_bf16.post_expert_norm.scale[...] = t2j(hf_moe_bf16.experts.post_expert_norm.weight).astype(jnp.bfloat16)
    mt_moe_bf16.post_expert_proj.kernel[...] = t2j(hf_moe_bf16.post_expert_proj.weight).T.astype(jnp.bfloat16)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_bf16.logical_axis_rules):
      mt_out_bf16, _ = mt_moe_bf16(xj.astype(jnp.bfloat16))
    assert_close(mt_out_bf16, hf_moe_bf16(xt.to(torch.bfloat16)), atol=2e-2, rtol=2e-2, msg="MuseSparkMoE bf16")

    # Also verify indivisible batch sizes (e.g. batch=1 and batch=3 on 4 devices) forward & backward parity.
    mesh_overrides = [{}]
    if jax.device_count() >= 2 and jax.device_count() % 2 == 0:
      mesh_overrides.append({"ici_fsdp_parallelism": jax.device_count() // 2, "ici_tensor_parallelism": 2})

    for batch in (1, 3):
      xt_b, xj_b = self._inputs_3d(batch=batch, seq=5, seed=90 + batch)
      probe_np = np.random.RandomState(100 + batch).randn(batch, 5, EMB).astype(np.float32)
      probe_j = jnp.asarray(probe_np)

      hf_moe_grad = copy.deepcopy(hf_moe)
      xt_req = xt_b.clone().detach().requires_grad_(True)
      with torch.enable_grad():
        hf_out_b = hf_moe_grad(xt_req)
        (hf_out_b.float() * torch.tensor(probe_np)).sum().backward()

      gate_up_grad = hf_moe_grad.experts.gate_up_proj.grad.detach().cpu().float().numpy()
      half = gate_up_grad.shape[1] // 2
      ref_wi_0_grad = np.swapaxes(gate_up_grad[:, :half, :], -1, -2)
      ref_wi_1_grad = np.swapaxes(gate_up_grad[:, half:, :], -1, -2)
      ref_wo_grad = np.swapaxes(hf_moe_grad.experts.down_proj.grad.detach().cpu().float().numpy(), -1, -2)
      ref_post_norm_grad = hf_moe_grad.experts.post_expert_norm.weight.grad.detach().cpu().float().numpy()

      for mesh_kw in mesh_overrides:
        cfg_m = make_config(**mesh_kw)
        mesh_m = Mesh(maxtext_utils.create_device_mesh(cfg_m), cfg_m.mesh_axes)
        mt_moe_m = muse_spark.MuseSparkMoE(cfg_m, mesh_m, rngs=nnx.Rngs(params=0))
        copy_moe(hf_moe, mt_moe_m)

        def moe_loss(m, x_in, pr=probe_j):
          y, _ = m(x_in)
          return jnp.sum(y.astype(jnp.float32) * pr), y

        with jax.set_mesh(mesh_m), nn_partitioning.axis_rules(cfg_m.logical_axis_rules):
          (_, y_m), (grads_m, grad_x) = nnx.value_and_grad(moe_loss, argnums=(0, 1), has_aux=True)(mt_moe_m, xj_b)

        tag = f"MuseSparkMoE batch={batch} mesh={mesh_kw}"
        assert_close(y_m, hf_out_b, msg=f"{tag} forward")
        assert_close(grad_x, xt_req.grad, msg=f"{tag} dx")
        assert_close(grads_m.wi_0.get_value(), ref_wi_0_grad, msg=f"{tag} dwi_0")
        assert_close(grads_m.wi_1.get_value(), ref_wi_1_grad, msg=f"{tag} dwi_1")
        assert_close(grads_m.wo.get_value(), ref_wo_grad, msg=f"{tag} dwo")
        assert_close(grads_m.post_expert_norm.scale.get_value(), ref_post_norm_grad, msg=f"{tag} dpost_norm")

  def test_decoder_layer_matches_hf(self):
    seq = 6
    xt, xj = self._inputs_3d(batch=1, seq=seq, seed=16)
    pos_t = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    pos_j = jnp.arange(seq, dtype=jnp.int32)[None, :]
    seg_j = jnp.ones((1, seq), dtype=jnp.int32)
    rotary = modeling_muse_spark.MuseSparkTextRotaryEmbedding(self.hf_text_cfg)
    pos_emb = rotary(xt, pos_t)

    mask_kwargs = {
        "config": self.hf_text_cfg,
        "inputs_embeds": xt,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": pos_t,
    }
    sliding_mask = modeling_muse_spark.create_sliding_window_causal_mask(**mask_kwargs)
    full_mask = modeling_muse_spark.create_causal_mask(**mask_kwargs)

    for layer_idx, mask, pe in ((0, sliding_mask, pos_emb), (1, full_mask, None)):
      hf_layer = modeling_muse_spark.MuseSparkTextDecoderLayer(self.hf_text_cfg, layer_idx=layer_idx).eval()
      randomize_(hf_layer, seed=17 + layer_idx)
      mt_layer = muse_spark.MuseSparkDecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          layer_idx=layer_idx,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      copy_decoder_layer(hf_layer, mt_layer)
      ref_out = hf_layer(xt, position_embeddings=pe, attention_mask=mask, position_ids=pos_t)
      mt_out, _ = mt_layer(xj, seg_j, pos_j, deterministic=True, model_mode=MODEL_MODE_TRAIN)
      assert_close(mt_out, ref_out, msg=f"MuseSparkDecoderLayer layer_idx={layer_idx}")

  def test_full_model_unscanned_and_scanned_and_param_mapping_match_hf(self):
    # 6 layers with cycle=4: 2 unrolled prefix layers (0..1: [sliding, full]) +
    # 1 scanned 4-layer cycle (2..5: [sliding, sliding, sliding, full])
    num_layers = 6
    hf_model = modeling_muse_spark.MuseSparkForConditionalGeneration(self.hf_full_cfg).eval()
    randomize_(hf_model, seed=25)

    cfg_u = self.config
    cfg_s = make_config(base_num_decoder_layers=num_layers, scan_layers=True)

    mu = model_creation_utils.from_config(cfg_u, mesh=self.mesh, rngs=nnx.Rngs(params=0, dropout=1))
    ms = model_creation_utils.from_config(cfg_s, mesh=self.mesh, rngs=nnx.Rngs(params=1, dropout=1))
    copy_full_model(hf_model, mu)

    # Verify parameter mapping + hook functions on both unscanned and scanned models
    hf_cfg_dict = self.hf_full_cfg.to_dict()
    expected_shapes = hf_shape.MUSE_SPARK_HF_WEIGHTS_TO_SHAPE(hf_cfg_dict)
    hf_state = {k: v.detach().cpu().float().numpy() for k, v in hf_model.state_dict().items()}
    for k, shp in expected_shapes.items():
      self.assertIn(k, hf_state, f"Missing HF key {k}")
      self.assertEqual(list(hf_state[k].shape), shp, f"Shape mismatch for {k}")

    for scan, cfg_m in ((False, cfg_u), (True, cfg_s)):
      converted_tree, pmap, abstract_dict = convert_hf_state_to_maxtext(hf_state, hf_cfg_dict, cfg_m, scan)
      flat_mapped = set()
      for k in pmap:
        if isinstance(k, tuple):
          flat_mapped.update(k)
        else:
          flat_mapped.add(k)
      self.assertEqual(flat_mapped, set(abstract_dict.keys()), f"Param mapping coverage mismatch for scan={scan}")
      if scan:
        # Populate the scanned model from the `to_maxtext` conversion path to verify checkpoint conversion E2E.
        load_converted_tree(ms, converted_tree)

    # The default test mesh puts every device on `fsdp`, so the batch must be divisible by the device count.
    batch, seq = jax.device_count(), 6
    ids_np = np.random.RandomState(42).randint(0, VOCAB, (batch, seq))
    ref_logits = hf_model(input_ids=torch.tensor(ids_np, dtype=torch.long)).logits

    ids_j = jnp.asarray(ids_np, jnp.int32)
    pos_j = jnp.arange(seq, dtype=jnp.int32)[None, :]
    seg_j = jnp.ones((batch, seq), jnp.int32)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      out_u = mu(decoder_input_tokens=ids_j, decoder_positions=pos_j, decoder_segment_ids=seg_j, enable_dropout=False)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      out_s = ms(decoder_input_tokens=ids_j, decoder_positions=pos_j, decoder_segment_ids=seg_j, enable_dropout=False)

    assert_close(out_u, ref_logits, msg="Full 6-layer unscanned MaxText vs HF")
    assert_close(out_s, out_u, msg="Full 6-layer scanned (via to_maxtext param_mapping) vs unscanned MaxText")
    assert_close(out_s, ref_logits, msg="Full 6-layer scanned MaxText vs HF")

    # Also verify bfloat16 checkpoint conversion (`save_dtype="bfloat16"`) and end-to-end forward parity:
    # `_keep_in_fp32_modules_strict` keeps residual gates and router kernel/bias in float32 while all other weights
    # are bfloat16.
    hf_model_bf16 = copy.deepcopy(hf_model).to(torch.bfloat16)
    fp32_patterns = tuple(hf_model._keep_in_fp32_modules_strict)  # pylint: disable=protected-access
    orig_state = hf_model.state_dict()
    with torch.no_grad():
      for name, param in hf_model_bf16.named_parameters():
        if any(pat in name for pat in fp32_patterns):
          param.data = orig_state[name].float()
      for name, buf in hf_model_bf16.named_buffers():
        if any(pat in name for pat in fp32_patterns):
          buf.data = orig_state[name].float()

    hf_state_bf16 = dict(hf_model_bf16.state_dict())
    cfg_s_bf16 = make_config(
        base_num_decoder_layers=num_layers,
        scan_layers=True,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        float32_logits=True,
        float32_qk_product=False,
        matmul_precision="highest",
    )
    ms_bf16 = model_creation_utils.from_config(cfg_s_bf16, mesh=self.mesh, rngs=nnx.Rngs(params=2, dropout=1))
    tree_bf16, _, _ = convert_hf_state_to_maxtext(hf_state_bf16, hf_cfg_dict, cfg_s_bf16, True, save_dtype="bfloat16")
    self.assertEqual(tree_bf16["params"]["decoder"]["layers_0"]["post_attention_residual_gate"]["gate"].dtype, np.float32)
    self.assertEqual(tree_bf16["params"]["decoder"]["layers_0"]["moe_block"]["gate"]["kernel"].dtype, np.float32)
    self.assertEqual(tree_bf16["MoEBiasVar"]["decoder"]["layers_0"]["moe_block"]["gate"]["bias"].dtype, np.float32)
    self.assertEqual(tree_bf16["params"]["decoder"]["layers_0"]["self_attention"]["query"]["kernel"].dtype, jnp.bfloat16)
    load_converted_tree(ms_bf16, tree_bf16)
    hf_bf16_routed = []
    bf16_hooks = [
        hf_model_bf16.model.language_model.layers[i].mlp.gate.register_forward_hook(
            lambda m, inp, out: hf_bf16_routed.append(out[2].detach().cpu().numpy().reshape(batch, seq, TOPK))
        )
        for i in range(num_layers)
    ]
    try:
      ref_logits_bf16 = hf_model_bf16(input_ids=torch.tensor(ids_np, dtype=torch.long)).logits
    finally:
      for h in bf16_hooks:
        h.remove()
    fre_bf16 = jnp.transpose(jnp.asarray(np.stack(hf_bf16_routed, axis=0), jnp.int32), (1, 2, 0, 3))
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s_bf16.logical_axis_rules):
      out_s_bf16 = ms_bf16(
          decoder_input_tokens=ids_j,
          decoder_positions=pos_j,
          decoder_segment_ids=seg_j,
          enable_dropout=False,
          forced_routed_experts=fre_bf16,
      )
    assert_close(out_s_bf16, ref_logits_bf16, atol=6e-2, rtol=6e-2, msg="Full 6-layer scanned bf16 MaxText vs HF")

  def test_full_model_gradients_match_hf(self):
    """Training-path parity: d(loss)/d(param) for every MaxText parameter (incl. routed experts) vs HF autograd.

    Runs unscanned and scanned models on the default all-`fsdp` mesh and, with an even device count, on a
    `fsdp x tensor` mesh so the expert all-gather and tensor-parallel psum backward passes are exercised too.
    """
    hf_model = modeling_muse_spark.MuseSparkForConditionalGeneration(self.hf_full_cfg).eval()
    randomize_(hf_model, seed=40)
    hf_cfg_dict = self.hf_full_cfg.to_dict()

    batch, seq = jax.device_count(), 6
    # Token ids start at 1: HF builds the embedding with `padding_idx=pad_token_id` (0 here), and PyTorch never
    # back-propagates into that row. In MaxText the pad token only appears at masked (segment 0) positions.
    ids_np = np.random.RandomState(41).randint(1, VOCAB, (batch, seq))
    probe_np = np.random.RandomState(42).randn(batch, seq, VOCAB).astype(np.float32)

    with torch.enable_grad():
      hf_logits = hf_model(input_ids=torch.tensor(ids_np, dtype=torch.long)).logits
      (hf_logits.float() * torch.tensor(probe_np)).sum().backward()
    hf_params = dict(hf_model.named_parameters())
    hf_weights = {k: v.detach().cpu().float().numpy() for k, v in hf_model.state_dict().items()}
    hf_grads = {
        k: (
            hf_params[k].grad.detach().cpu().float().numpy()
            if k in hf_params and hf_params[k].grad is not None
            else np.zeros_like(v)
        )
        for k, v in hf_weights.items()
    }

    ids_j = jnp.asarray(ids_np, jnp.int32)
    pos_j = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], ids_j.shape)
    seg_j = jnp.ones((batch, seq), jnp.int32)
    probe_j = jnp.asarray(probe_np)

    def loss_fn(model):
      logits = model(decoder_input_tokens=ids_j, decoder_positions=pos_j, decoder_segment_ids=seg_j, enable_dropout=False)
      return jnp.sum(logits.astype(jnp.float32) * probe_j)

    mesh_overrides = [{}]
    if jax.device_count() >= 2 and jax.device_count() % 2 == 0:
      mesh_overrides.append({"ici_fsdp_parallelism": jax.device_count() // 2, "ici_tensor_parallelism": 2})

    for mesh_kw in mesh_overrides:
      for scan in (False, True):
        cfg_m = make_config(scan_layers=scan, **mesh_kw)
        mesh = Mesh(maxtext_utils.create_device_mesh(cfg_m), cfg_m.mesh_axes)
        model = model_creation_utils.from_config(cfg_m, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
        load_converted_tree(model, convert_hf_state_to_maxtext(hf_weights, hf_cfg_dict, cfg_m, scan)[0])
        expected_tree = convert_hf_state_to_maxtext(hf_grads, hf_cfg_dict, cfg_m, scan)[0]

        with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_m.logical_axis_rules):
          grads = nnx.grad(loss_fn)(model)

        checked = []
        for path, grad_var in nnx.to_flat_state(grads):
          name = "/".join(map(str, path))
          msg = f"grad {name} scan={scan} mesh={mesh_kw}"
          actual = np.asarray(grad_var.get_value(), dtype=np.float32)
          expected = np.asarray(lookup_converted(expected_tree, "params", path), dtype=np.float32)
          scale = float(np.max(np.abs(expected)))
          self.assertGreater(scale, 0.0, f"HF gradient is zero for {name}")
          # Whole-tensor relative error catches systematic bugs (wrong VJP, missing/extra collective scaling);
          # the element-wise check uses a tolerance relative to the tensor scale to absorb fp32 summation order.
          self.assertLess(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)), 1e-4, msg)
          np.testing.assert_allclose(actual, expected, atol=1e-4 * scale, rtol=1e-3, err_msg=msg)
          checked.append(name)
        # Every parameter, including the routed experts, must receive a gradient.
        self.assertEqual(len(checked), len(nnx.to_flat_state(nnx.state(model, nnx.Param))))
        self.assertTrue(any(n.endswith("wi_0") for n in checked) and any(n.endswith("wo") for n in checked))

  def test_zero_gen_masked_prefill_and_golden_npz_parity(self):
    """Verifies Zero-Generation masked prefill, routing parity, and golden .npz runners against HF."""
    num_layers = 6
    hf_model = modeling_muse_spark.MuseSparkForConditionalGeneration(self.hf_full_cfg).eval()
    randomize_(hf_model, seed=30)

    cfg_u = self.config
    mu = model_creation_utils.from_config(cfg_u, mesh=self.mesh, rngs=nnx.Rngs(params=0, dropout=1))
    copy_full_model(hf_model, mu)

    texts = [
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["prose"],
        zero_gen_logits_checker.GOLDEN_BATCH_SECTIONS["code_json_math"],
    ]
    seq_len = 8
    batch = max(2, jax.device_count())
    tokens_np, seg_ids_np, mask_np = zero_gen_logits_checker.synthetic_golden_batch(
        texts,
        vocab_size=VOCAB,
        seq_len=seq_len,
        pad_token_id=0,
        batch_size=batch,
        target_start_fraction=0.25,
    )

    hf_expert_indices = []

    def _gate_hook(module, inputs, output):
      del module, inputs
      hf_expert_indices.append(output[2].detach().cpu().numpy().reshape(batch, seq_len, TOPK))

    hooks = [
        hf_model.model.language_model.layers[i].mlp.gate.register_forward_hook(_gate_hook) for i in range(num_layers)
    ]
    try:
      ref_logits = hf_model(input_ids=torch.tensor(tokens_np, dtype=torch.long)).logits.detach().cpu().float().numpy()
    finally:
      for h in hooks:
        h.remove()

    ids_j = jnp.asarray(tokens_np, jnp.int32)
    pos_j = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], ids_j.shape)
    seg_j = jnp.asarray(seg_ids_np, jnp.int32)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      jax_logits = np.asarray(
          mu(decoder_input_tokens=ids_j, decoder_positions=pos_j, decoder_segment_ids=seg_j, enable_dropout=False),
          dtype=np.float32,
      )

    metrics = zero_gen_logits_checker.compute_masked_prefill_metrics(jax_logits, ref_logits, mask_np)
    self.assertGreater(metrics.cos_sim, 0.99999)
    self.assertLess(metrics.l_inf, 1e-4)
    self.assertLess(metrics.max_kl_div, 1e-5)
    self.assertEqual(metrics.zero_norm_token_count, 0)

    routing_stack = np.stack(hf_expert_indices, axis=0)
    self.assertTrue(zero_gen_logits_checker.check_routing_parity(routing_stack, routing_stack)["passed"])
    self.assertTrue(zero_gen_logits_checker.check_logits_parity(jax_logits, ref_logits, mask=mask_np)["passed"])

    with tempfile.TemporaryDirectory() as tmpdir:
      npz_path = os.path.join(tmpdir, "muse_spark_golden.npz")
      zero_gen_logits_checker.save_golden_npz(
          npz_path,
          tokens=tokens_np,
          logits=ref_logits,
          routing_indices=routing_stack,
          target_mask=mask_np,
          seq_len=seq_len,
          dtype_str="float32",
      )

      def _jax_runner(toks, return_aux=False):
        del return_aux
        t_j = jnp.asarray(toks, jnp.int32)
        p_j = jnp.broadcast_to(jnp.arange(t_j.shape[1], dtype=jnp.int32)[None, :], t_j.shape)
        s_j = jnp.asarray(seg_ids_np, dtype=jnp.int32)
        with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
          return mu(decoder_input_tokens=t_j, decoder_positions=p_j, decoder_segment_ids=s_j, enable_dropout=False)

      prefill_res = zero_gen_logits_checker.run_prefill_logits_check(
          _jax_runner, npz_path, mask=mask_np, require_aux=False
      )
      self.assertTrue(prefill_res["passed"], f"run_prefill_logits_check failed: {prefill_res}")
      tinymmlu_res = zero_gen_logits_checker.run_tinymmlu_check(_jax_runner, npz_path)
      self.assertTrue(tinymmlu_res["passed"], f"run_tinymmlu_check failed: {tinymmlu_res}")


if __name__ == "__main__":
  unittest.main()
