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

"""Unit tests for MaxText Hy4-preview vs HuggingFace HYV4ForCausalLM."""

from __future__ import annotations

import os
import types
import unittest

import pytest

try:
  import torch
  from transformers.models.hy_v4 import configuration_hy_v4, modeling_hy_v4

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import AttentionType, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.models import hy4
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

BASE_CONFIG = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")

# Minimal dimensions for fast CI execution while preserving all architectural paths.
# SWIGLU_LIMIT=0.05 ensures routed-expert SwiGLU clamping is actively exercised under std=0.08 weights.
EMB, HEADS, HEAD_DIM = 32, 2, 16
Q_LORA, KV_LORA, NOPE, ROPE, V_DIM = 16, 8, 8, 8, 16
IDX_HEADS, IDX_DIM, IDX_TOPK = 4, 16, 3
MLP, MOE_MLP, EXPERTS, TOPK = 32, 16, 4, 2
HC_MULT = 4
VOCAB = 64
SWIGLU_LIMIT = 0.05
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
  """Fills a torch module with seeded random weights (norm weights stay near 1)."""
  gen = torch.Generator().manual_seed(seed)
  norm_suffixes = (
      "input_layernorm.weight",
      "post_attention_layernorm.weight",
      "q_a_layernorm.weight",
      "kv_a_layernorm.weight",
      "k_norm.weight",
      "norm.weight",
  )
  with torch.no_grad():
    for name, p in module.named_parameters():
      noise = std * torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.dtype)
      p.copy_((1.0 + noise) if name.endswith(norm_suffixes) else noise)
    for name, b in module.named_buffers():
      if "e_score_correction_bias" in name:
        b.copy_(std * torch.randn(b.shape, generator=gen, dtype=torch.float32).to(b.dtype))


def make_config(**kwargs):
  """Builds a small MaxText hy4-preview config for tests."""
  overrides = dict(  # pylint: disable=use-dict-literal
      model_name="hy4-preview",
      override_model_config=True,
      base_emb_dim=EMB,
      base_num_query_heads=HEADS,
      base_num_kv_heads=HEADS,
      head_dim=HEAD_DIM,
      q_lora_rank=Q_LORA,
      kv_lora_rank=KV_LORA,
      qk_nope_head_dim=NOPE,
      qk_rope_head_dim=ROPE,
      v_head_dim=V_DIM,
      indexer_n_heads=IDX_HEADS,
      indexer_head_dim=IDX_DIM,
      indexer_topk=IDX_TOPK,
      use_indexer=True,
      mhc_expansion_rate=HC_MULT,
      base_mlp_dim=MLP,
      base_moe_mlp_dim=MOE_MLP,
      num_experts=EXPERTS,
      num_experts_per_tok=TOPK,
      shared_experts=1,
      routed_scaling_factor=2.827,
      routed_score_func="sigmoid",
      routed_bias=True,
      norm_topk_prob=True,
      n_routing_groups=-1,
      topk_routing_group=-1,
      mlp_activations_limit=SWIGLU_LIMIT,
      first_num_dense_layers=1,
      inhomogeneous_layer_cycle_interval=2,
      vocab_size=VOCAB,
      base_num_decoder_layers=3,
      per_device_batch_size=1,
      max_target_length=8,
      max_prefill_predict_length=4,
      max_position_embeddings=64,
      original_max_position_embeddings=64,
      scan_layers=False,
      attention="dot_product",
      attention_type="mla",
      attention_sink=True,
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


def make_hf_config(num_layers=3, index_topk=IDX_TOPK):
  return configuration_hy_v4.HYV4Config(
      vocab_size=VOCAB,
      hidden_size=EMB,
      intermediate_size=MLP,
      moe_intermediate_size=MOE_MLP,
      num_hidden_layers=num_layers,
      num_attention_heads=HEADS,
      num_key_value_heads=HEADS,
      n_routed_experts=EXPERTS,
      num_experts_per_tok=TOPK,
      n_shared_experts=1,
      routed_scaling_factor=2.827,
      norm_topk_prob=True,
      n_group=1,
      topk_group=1,
      q_lora_rank=Q_LORA,
      kv_lora_rank=KV_LORA,
      qk_nope_head_dim=NOPE,
      qk_rope_head_dim=ROPE,
      v_head_dim=V_DIM,
      index_n_heads=IDX_HEADS,
      index_head_dim=IDX_DIM,
      index_topk=index_topk,
      hc_mult=HC_MULT,
      hc_eps=1e-6,
      hc_magnitude=2.0,
      rms_norm_eps=1e-5,
      rope_parameters={"rope_theta": 10_000_000.0, "rope_type": "default"},
      max_position_embeddings=64,
      swiglu_limit=SWIGLU_LIMIT,
      attention_dropout=0.0,
      tie_word_embeddings=False,
      pad_token_id=0,
      bos_token_id=1,
      eos_token_id=2,
      attn_implementation="eager",
  )


def set_dense(mt_dense, hf_linear, out_shape=None):
  w = t2j(hf_linear.weight).T
  if out_shape is not None:
    w = w.reshape(w.shape[0], *out_shape)
  mt_dense.kernel[...] = w


def copy_hc(hf_hc, mt_hc):
  mt_hc.fn[...] = t2j(hf_hc.fn).T
  mt_hc.base[...] = t2j(hf_hc.base)
  mt_hc.scale[...] = t2j(hf_hc.scale)


def copy_hc_head(hf_head, mt_head):
  mt_head.hc_fn[...] = t2j(hf_head.hc_fn).T
  mt_head.hc_base[...] = t2j(hf_head.hc_base)
  mt_head.hc_scale[...] = t2j(hf_head.hc_scale)


def copy_indexer(hf_idx, mt_idx):
  set_dense(mt_idx.wq_b, hf_idx.wq_b, out_shape=(IDX_HEADS, IDX_DIM))
  set_dense(mt_idx.wk, hf_idx.wk)
  mt_idx.k_norm.scale[...] = t2j(hf_idx.k_norm.weight)
  mt_idx.k_norm.bias[...] = t2j(hf_idx.k_norm.bias)
  set_dense(mt_idx.weights_proj, hf_idx.weights_proj)


def copy_attention(hf_attn, mt_attn):
  """Copies HF attention weights into the MaxText attention module."""
  set_dense(mt_attn.wq_a, hf_attn.q_a_proj)
  mt_attn.q_norm.scale[...] = t2j(hf_attn.q_a_layernorm.weight)
  set_dense(mt_attn.wq_b, hf_attn.q_b_proj, out_shape=(HEADS, NOPE + ROPE))
  set_dense(mt_attn.wkv_a, hf_attn.kv_a_proj_with_mqa)
  mt_attn.kv_norm.scale[...] = t2j(hf_attn.kv_a_layernorm.weight)
  set_dense(mt_attn.wkv_b, hf_attn.kv_b_proj, out_shape=(HEADS, NOPE + V_DIM))
  set_dense(mt_attn.gate_proj, hf_attn.gate_proj, out_shape=(HEADS, V_DIM))
  mt_attn.out.kernel[...] = t2j(hf_attn.o_proj.weight).T.reshape(HEADS, V_DIM, EMB)
  mt_attn.sinks[...] = t2j(hf_attn.sinks)
  if mt_attn.indexer is not None and hf_attn.indexer is not None:
    copy_indexer(hf_attn.indexer, mt_attn.indexer)


def copy_mlp(hf_mlp, mt_mlp):
  set_dense(mt_mlp.wi_0, hf_mlp.gate_proj)
  set_dense(mt_mlp.wi_1, hf_mlp.up_proj)
  set_dense(mt_mlp.wo, hf_mlp.down_proj)


def copy_moe(hf_moe, mt_moe):
  routed = mt_moe.MoeBlock_0
  routed.gate.kernel[...] = t2j(hf_moe.gate.weight).T
  routed.gate.bias[...] = t2j(hf_moe.gate.e_score_correction_bias)
  gate_up = t2j(hf_moe.experts.gate_up_proj)
  gate, up = jnp.split(gate_up, 2, axis=1)
  routed.wi_0[...] = jnp.swapaxes(gate, -1, -2)
  routed.wi_1[...] = jnp.swapaxes(up, -1, -2)
  routed.wo[...] = jnp.swapaxes(t2j(hf_moe.experts.down_proj), -1, -2)
  copy_mlp(hf_moe.shared_experts, mt_moe.shared_experts)


def copy_decoder_layer(hf_layer, mt_layer):
  mt_layer.pre_self_attention_layer_norm.scale[...] = t2j(hf_layer.input_layernorm.weight)
  mt_layer.post_self_attention_layer_norm.scale[...] = t2j(hf_layer.post_attention_layernorm.weight)
  copy_hc(hf_layer.attn_hc, mt_layer.attn_hc)
  copy_hc(hf_layer.ffn_hc, mt_layer.ffn_hc)
  copy_attention(hf_layer.self_attn, mt_layer.self_attention)
  if mt_layer.is_moe:
    copy_moe(hf_layer.mlp, mt_layer.moe_block)
  else:
    copy_mlp(hf_layer.mlp, mt_layer.mlp)


def copy_full_model(hf_model, mt_model):
  mt_model.token_embedder.embedding[...] = t2j(hf_model.model.embed_tokens.weight)
  copy_hc_head(hf_model.model.hc_head, mt_model.decoder.hc_head)
  mt_model.decoder.decoder_norm.scale[...] = t2j(hf_model.model.norm.weight)
  mt_model.decoder.logits_dense.kernel[...] = t2j(hf_model.lm_head.weight).T
  for i, hf_layer in enumerate(hf_model.model.layers):
    copy_decoder_layer(hf_layer, getattr(mt_model.decoder, f"layers_{i}"))


class Hy4LayersTest(unittest.TestCase):
  """CPU unit tests comparing MaxText Hy4 modules against transformers.models.hy_v4."""

  @classmethod
  def setUpClass(cls):
    torch.set_grad_enabled(False)
    cls.hf_cfg = make_hf_config()
    cls.config = make_config()
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cls.config), cls.config.mesh_axes)

  def _inputs_3d(self, batch=1, seq=6, seed=0):
    x = np.random.RandomState(seed).randn(batch, seq, EMB).astype(np.float32)
    return torch.tensor(x), jnp.asarray(x)

  def _inputs_4d(self, batch=1, seq=6, seed=0):
    x = np.random.RandomState(seed).randn(batch, seq, HC_MULT, EMB).astype(np.float32)
    return torch.tensor(x), jnp.asarray(x)

  def _make_attn(self, model_mode=MODEL_MODE_TRAIN, q_len=6):
    return hy4.Hy4Attention(
        config=self.config,
        num_query_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        max_target_length=self.config.max_target_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, q_len, EMB),
        inputs_kv_shape=(1, q_len, EMB),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        is_nope_layer=False,
        is_vision=False,
        model_mode=model_mode,
        attention_type=AttentionType.MLA,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        max_position_embeddings=self.config.max_position_embeddings,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        mscale=self.config.mscale,
        rope_factor=self.config.rope_factor,
        rngs=nnx.Rngs(params=0, dropout=1),
        is_full_indexer=True,
    )

  def test_layer_pattern_and_scan_layout_match_hf(self):
    full_hf = make_hf_config(num_layers=78)

    def stub(n):
      return types.SimpleNamespace(
          first_num_dense_layers=1,
          inhomogeneous_layer_cycle_interval=4,
          num_decoder_layers=n,
          num_experts=EXPERTS,
      )

    cfg78 = stub(78)
    for i in range(78):
      self.assertEqual(hy4.is_full_indexer_layer(i, cfg78), full_hf.indexer_types[i] == "full", f"layer {i}")
      self.assertEqual(hy4.is_moe_layer(i, cfg78), full_hf.mlp_layer_types[i] == "sparse", f"layer {i}")
    self.assertEqual(hy4.scan_layout(cfg78), (1, 19, 77))
    self.assertEqual(hy4.scan_layout(stub(3)), (1, 0, 1))
    self.assertEqual(hy4.scan_layout(stub(6)), (1, 1, 5))

  def test_hyper_connection_and_head_match_hf(self):
    hf_hc = modeling_hy_v4.HYV4HyperConnection(self.hf_cfg).eval()
    randomize_(hf_hc, seed=1)
    mt_hc = hy4.Hy4HyperConnection(self.config, self.mesh, rngs=nnx.Rngs(0))
    copy_hc(hf_hc, mt_hc)

    xt, xj = self._inputs_4d(seed=10)
    bt, bj = self._inputs_3d(seed=11)
    ref_post, ref_in = hf_hc(xt)
    ref_res = (ref_post.float().unsqueeze(-1) * bt.float().unsqueeze(-2) + xt.float()).to(xt.dtype)

    mt_in, mt_post = mt_hc.compute_pre_post(xj)
    mt_res = mt_hc.apply_residual(xj, bj, mt_post)
    assert_close(mt_in, ref_in, msg="Hy4HyperConnection branch_in")
    assert_close(mt_post, ref_post, msg="Hy4HyperConnection post")
    assert_close(mt_res, ref_res, msg="Hy4HyperConnection residual")

    hf_head = modeling_hy_v4.HYV4HyperHead(self.hf_cfg).eval()
    randomize_(hf_head, seed=2)
    mt_head = hy4.Hy4HyperHead(self.config, self.mesh, rngs=nnx.Rngs(0))
    copy_hc_head(hf_head, mt_head)
    assert_close(mt_head(xj), hf_head(xt), msg="Hy4HyperHead")

  def test_cached_prefill_and_ar_decode_match_full_forward(self):
    """Prefill (pad=4 > topk=3) + AR decode (total=8 > topk=3) with MlaKVCache_0 and IndexerKVCache_0."""
    hf_attn = modeling_hy_v4.HYV4Attention(self.hf_cfg, layer_idx=0).eval()
    randomize_(hf_attn, seed=9)

    total, prompt, pad = 5, 3, self.config.max_prefill_predict_length
    mt_attn = self._make_attn(MODEL_MODE_PREFILL, q_len=pad)
    copy_attention(hf_attn, mt_attn)

    _, xj = self._inputs_3d(1, total, seed=16)
    pos_full = jnp.arange(total, dtype=jnp.int32)[None]
    seg_full = jnp.ones((1, total), dtype=jnp.int32)
    ref_out, _, _ = mt_attn(
        xj, xj, pos_full, decoder_segment_ids=seg_full, model_mode=MODEL_MODE_TRAIN, deterministic=True
    )

    xp = jnp.concatenate([xj[:, :prompt], jnp.zeros((1, pad - prompt, EMB), dtype=jnp.float32)], axis=1)
    pos_p = jnp.arange(pad, dtype=jnp.int32)[None]
    seg_p = (jnp.arange(pad) < prompt).astype(jnp.int32)[None]
    out_p, _, _ = mt_attn(xp, xp, pos_p, decoder_segment_ids=seg_p, model_mode=MODEL_MODE_PREFILL, deterministic=True)
    assert_close(out_p[:, :prompt], ref_out[:, :prompt], msg="prefill")

    for t in range(prompt, total):
      xt_step = xj[:, t : t + 1]
      pos_step = jnp.array([[t]], dtype=jnp.int32)
      seg_step = jnp.ones((1, 1), dtype=jnp.int32)
      out_t, _, _ = mt_attn(
          xt_step,
          xt_step,
          pos_step,
          decoder_segment_ids=seg_step,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          deterministic=True,
      )
      assert_close(out_t[:, 0], ref_out[:, t], msg=f"AR step {t}")

  def test_full_model_unscanned_and_scanned_match_hf(self):
    # 3 layers with cycle=2: 1 prefix dense+full layer (0) + 1 scanned 2-layer MoE cycle (1..2, [full, shared])
    num_layers = 3
    hf_model = modeling_hy_v4.HYV4ForCausalLM(self.hf_cfg).eval()
    randomize_(hf_model, seed=20)

    cfg_u = self.config
    cfg_s = make_config(base_num_decoder_layers=num_layers, scan_layers=True)

    mu = model_creation_utils.from_config(cfg_u, mesh=self.mesh, rngs=nnx.Rngs(params=0, dropout=1))
    ms = model_creation_utils.from_config(cfg_s, mesh=self.mesh, rngs=nnx.Rngs(params=1, dropout=1))
    copy_full_model(hf_model, mu)

    # Copy unscanned mu state into scanned ms state
    num_prefix, num_blocks, _ = hy4.scan_layout(cfg_s)
    cycle = cfg_s.inhomogeneous_layer_cycle_interval
    var_filter = nnx.Any(nnx.Param, moe.MoEBiasVar)
    src = {tuple(map(str, k)): v for k, v in nnx.to_flat_state(nnx.state(mu, var_filter))}
    dst_state = nnx.state(ms, var_filter)
    new_flat = []
    for path, var in nnx.to_flat_state(dst_state):
      key = tuple(map(str, path))
      if "scanned_blocks" in key:
        i = key.index("scanned_blocks")
        j = int(key[i + 1].split("_")[1])
        slices = [
            src[key[:i] + (f"layers_{num_prefix + b * cycle + j}",) + key[i + 2 :]].get_value() for b in range(num_blocks)
        ]
        value = jnp.stack(slices, axis=0)
        if isinstance(var, nnx.Param) and value.ndim > cfg_s.param_scan_axis:
          value = jnp.moveaxis(value, 0, cfg_s.param_scan_axis)
      else:
        value = src[key].get_value()
      self.assertEqual(value.shape, var.get_value().shape, key)
      new_flat.append((path, var.replace(value)))
    nnx.update(ms, nnx.from_flat_state(new_flat))

    batch, seq = 1, 6  # seq=6 > IDX_TOPK=3 exercises active top-k indexer + cross-layer sharing + clamped SwiGLU MoE
    ids_np = np.random.RandomState(42).randint(0, VOCAB, (batch, seq))
    ref_logits = hf_model(torch.tensor(ids_np, dtype=torch.long)).logits

    ids_j = jnp.asarray(ids_np, jnp.int32)
    pos_j = jnp.arange(seq, dtype=jnp.int32)[None]
    seg_j = jnp.ones((batch, seq), jnp.int32)

    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_u.logical_axis_rules):
      out_u = mu(decoder_input_tokens=ids_j, decoder_positions=pos_j, decoder_segment_ids=seg_j, enable_dropout=False)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(cfg_s.logical_axis_rules):
      out_s = ms(decoder_input_tokens=ids_j, decoder_positions=pos_j, decoder_segment_ids=seg_j, enable_dropout=False)

    assert_close(out_u, ref_logits, msg="Full 3-layer unscanned MaxText vs HF")
    assert_close(out_s, out_u, msg="Full 3-layer scanned vs unscanned MaxText")
    assert_close(out_s, ref_logits, msg="Full 3-layer scanned MaxText vs HF")


if __name__ == "__main__":
  unittest.main()
