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

"""GLM-5.3-Flash (glm5_next) layer tests against the HuggingFace reference.

Requires a transformers build containing `transformers.models.glm5_next` (transformers main).
"""

import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest
import torch

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import kimi_delta_attention as kda
from maxtext.layers.attention_glm5 import Glm5SparseMLA, KPoolIndexer
from maxtext.models.glm5_next import Glm5NextDecoderLayer
from maxtext.utils.globals import MAXTEXT_REPO_ROOT

glm5 = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # pylint: disable=wrong-import-position

torch.set_grad_enabled(False)

EMB, HEADS, HEAD_DIM = 64, 4, 16
Q_LORA, KV_LORA = 32, 16
# 16 indexer heads (real model: 32): an index score is exactly 0 only when every head's ReLU is 0
# (prob ~2^-heads per pool), and torch/jax break exact top-k ties differently.
IDX_HEADS, IDX_DIM, KPOOL, TOPK = 16, 16, 4, 8
N_EXPERTS, TOPK_EXPERTS, MOE_DIM, MLP_DIM = 8, 2, 32, 96


def hf_config(**overrides) -> Glm5NextTextConfig:
  kwargs = dict(
      vocab_size=128,
      hidden_size=EMB,
      intermediate_size=MLP_DIM,
      moe_intermediate_size=MOE_DIM,
      num_hidden_layers=8,
      num_attention_heads=HEADS,
      num_key_value_heads=HEADS,
      n_routed_experts=N_EXPERTS,
      num_experts_per_tok=TOPK_EXPERTS,
      n_shared_experts=1,
      routed_scaling_factor=2.5,
      kv_lora_rank=KV_LORA,
      q_lora_rank=Q_LORA,
      qk_rope_head_dim=0,
      qk_nope_head_dim=HEAD_DIM,
      v_head_dim=HEAD_DIM,
      index_n_heads=IDX_HEADS,
      index_head_dim=IDX_DIM,
      index_topk=TOPK,
      index_kpool=KPOOL,
      rms_norm_eps=1e-5,
      linear_attn_config={"num_heads": HEADS, "head_dim": HEAD_DIM, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
      hc_mult=4,
      swiglu_limit=10.0,
  )
  kwargs.update(overrides)
  cfg = Glm5NextTextConfig(**kwargs)
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  return cfg


def mt_config(**overrides):
  kwargs = dict(
      run_name="glm5_next_layers_test",
      model_name="glm5.3-flash-mini",
      override_model_config=True,
      enable_checkpointing=False,
      decoder_block="glm5_next",
      base_emb_dim=EMB,
      base_num_query_heads=HEADS,
      base_num_kv_heads=HEADS,
      base_mlp_dim=MLP_DIM,
      base_moe_mlp_dim=MOE_DIM,
      base_num_decoder_layers=8,
      first_num_dense_layers=3,
      inhomogeneous_layer_cycle_interval=4,
      head_dim=HEAD_DIM,
      q_lora_rank=Q_LORA,
      kv_lora_rank=KV_LORA,
      qk_nope_head_dim=HEAD_DIM,
      qk_rope_head_dim=0,
      v_head_dim=HEAD_DIM,
      attention_type="mla",
      use_indexer=True,
      indexer_n_heads=IDX_HEADS,
      indexer_head_dim=IDX_DIM,
      indexer_topk=TOPK,
      indexer_kpool=KPOOL,
      kda_num_heads=HEADS,
      kda_head_dim=HEAD_DIM,
      kda_chunk_size=16,
      num_experts=N_EXPERTS,
      num_experts_per_tok=TOPK_EXPERTS,
      shared_experts=1,
      routed_score_func="sigmoid",
      routed_bias=True,
      routed_scaling_factor=2.5,
      norm_topk_prob=True,
      mlp_activations=["silu", "linear"],
      mlp_activations_limit=10.0,
      mhc_expansion_rate=4,
      mhc_final_reduce="mean",
      sinkhorn_iterations=20,
      normalization_layer_epsilon=1e-5,
      sparse_matmul=False,
      capacity_factor=-1.0,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
      float32_logits=True,
      float32_qk_product=True,
      per_device_batch_size=1,
      max_target_length=64,
      max_prefill_predict_length=64,
      enable_dropout=False,
      vocab_size=128,
      scan_layers=False,
      attention="dot_product",
  )
  kwargs.update(overrides)
  return pyconfig.initialize([None, os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")], **kwargs)


def t2j(t: torch.Tensor):
  return jnp.asarray(t.detach().float().numpy())


def set_param(p, value):
  p.value = jnp.asarray(value, dtype=p.value.dtype).reshape(p.value.shape)


def lin_t(linear: torch.nn.Linear):
  """torch Linear weight [out, in] -> numpy kernel [in, out]."""
  return linear.weight.detach().float().numpy().T


def copy_kda(hf_attn, mt_attn):
  hd = HEADS * HEAD_DIM
  set_param(mt_attn.q_proj.kernel, lin_t(hf_attn.q_proj))
  set_param(mt_attn.k_proj.kernel, lin_t(hf_attn.k_proj))
  set_param(mt_attn.v_proj.kernel, lin_t(hf_attn.v_proj))
  conv = hf_attn.conv1d.weight.detach().float().numpy()[:, 0, :]  # [3*HD, W]
  set_param(mt_attn.q_conv1d, conv[:hd].T)
  set_param(mt_attn.k_conv1d, conv[hd : 2 * hd].T)
  set_param(mt_attn.v_conv1d, conv[2 * hd :].T)
  fg = hf_attn.forget_gate
  set_param(mt_attn.f_a_proj.kernel, lin_t(fg.f_a_proj))
  set_param(mt_attn.f_b_proj.kernel, lin_t(fg.f_b_proj))
  set_param(mt_attn.dt_bias, fg.dt_bias.detach().float().numpy())
  set_param(mt_attn.A_log, fg.A_log.detach().float().numpy())
  set_param(mt_attn.b_proj.kernel, lin_t(hf_attn.b_proj))
  set_param(mt_attn.g_a_proj.kernel, lin_t(hf_attn.g_a_proj))
  set_param(mt_attn.g_b_proj.kernel, lin_t(hf_attn.g_b_proj))
  set_param(mt_attn.o_norm, hf_attn.o_norm.weight.detach().float().numpy())
  set_param(mt_attn.o_proj.kernel, lin_t(hf_attn.o_proj))


def copy_indexer(hf_idx, mt_idx):
  set_param(mt_idx.wq_b.kernel, lin_t(hf_idx.wq_b))
  set_param(mt_idx.wk.kernel, lin_t(hf_idx.wk))
  set_param(mt_idx.k_norm.scale, hf_idx.k_norm.weight.detach().float().numpy())
  set_param(mt_idx.k_norm.bias, hf_idx.k_norm.bias.detach().float().numpy())
  set_param(mt_idx.weights_proj.kernel, lin_t(hf_idx.weights_proj))
  set_param(mt_idx.compress_gate.kernel, hf_idx.index_kpool_compress_gate.detach().float().numpy().T)
  set_param(mt_idx.compress_ape, hf_idx.index_kpool_compress_ape.detach().float().numpy())


def copy_mla(hf_attn, mt_attn):
  set_param(mt_attn.wq_a.kernel, lin_t(hf_attn.q_a_proj))
  set_param(mt_attn.q_norm.scale, hf_attn.q_a_layernorm.weight.detach().float().numpy())
  set_param(mt_attn.wq_b.kernel, lin_t(hf_attn.q_b_proj))
  set_param(mt_attn.wkv_a.kernel, lin_t(hf_attn.kv_a_proj_with_mqa))
  set_param(mt_attn.kv_norm.scale, hf_attn.kv_a_layernorm.weight.detach().float().numpy())
  set_param(mt_attn.wkv_b.kernel, lin_t(hf_attn.kv_b_proj))
  set_param(mt_attn.out.kernel, lin_t(hf_attn.o_proj))
  copy_indexer(hf_attn.indexer, mt_attn.indexer)


def copy_mhc(hf_hc, mt_hc, k=4):
  fn = hf_hc.fn.detach().float().numpy()  # [(2+k)*k, k*E]
  base = hf_hc.base.detach().float().numpy()
  scale = hf_hc.scale.detach().float().numpy()
  set_param(mt_hc.pre_alpha, fn[:k].T)
  set_param(mt_hc.post_alpha, fn[k : 2 * k].T)
  set_param(mt_hc.res_alpha, fn[2 * k :].T)
  set_param(mt_hc.pre_beta, base[:k])
  set_param(mt_hc.post_beta, base[k : 2 * k])
  set_param(mt_hc.res_beta, base[2 * k :].reshape(k, k))
  set_param(mt_hc.pre_alpha_scale, scale[0:1])
  set_param(mt_hc.post_alpha_scale, scale[1:2])
  set_param(mt_hc.res_alpha_scale, scale[2:3])
  set_param(mt_hc.mhc_norm.scale, np.ones(mt_hc.mhc_norm.scale.value.shape))


def copy_mlp(hf_mlp, mt_mlp):
  set_param(mt_mlp.wi_0.kernel, lin_t(hf_mlp.gate_proj))
  set_param(mt_mlp.wi_1.kernel, lin_t(hf_mlp.up_proj))
  set_param(mt_mlp.wo.kernel, lin_t(hf_mlp.down_proj))


def copy_moe(hf_moe, mt_moe):
  routed = mt_moe.MoeBlock_0
  set_param(routed.gate.kernel, lin_t(hf_moe.gate))
  set_param(routed.gate.bias, hf_moe.gate.e_score_correction_bias.detach().float().numpy())
  gate_up = hf_moe.experts.gate_up_proj.detach().float().numpy()  # [N, 2I, E]
  set_param(routed.wi_0, gate_up[:, :MOE_DIM, :].transpose(0, 2, 1))
  set_param(routed.wi_1, gate_up[:, MOE_DIM:, :].transpose(0, 2, 1))
  set_param(routed.wo, hf_moe.experts.down_proj.detach().float().numpy().transpose(0, 2, 1))
  copy_mlp(hf_moe.shared_experts, mt_moe.shared_experts)


@torch.no_grad()
def randomize(module: torch.nn.Module, seed: int = 0):
  """Random values for every parameter/buffer (including the constant-initialized ones)."""
  gen = torch.Generator().manual_seed(seed)
  for name, p in list(module.named_parameters()) + list(module.named_buffers()):
    if not p.dtype.is_floating_point:
      continue
    if name.endswith(("norm.weight", "layernorm.weight", "scale")):
      p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
    elif name.endswith("A_log"):
      p.copy_(0.5 * torch.randn(p.shape, generator=gen))
    elif name.endswith(("index_kpool_compress_ape", "dt_bias")):
      p.copy_(0.5 * torch.randn(p.shape, generator=gen))
    elif "conv1d" in name:
      p.copy_(0.3 * torch.randn(p.shape, generator=gen))
    else:
      p.copy_(0.1 * torch.randn(p.shape, generator=gen))


class Glm5NextTestBase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    torch.manual_seed(0)
    self.mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))

  def assert_close(self, mt_out, hf_out, atol=1e-4, rtol=1e-4):
    np.testing.assert_allclose(np.asarray(mt_out, np.float32), hf_out.detach().float().numpy(), atol=atol, rtol=rtol)


class KDAKernelTest(Glm5NextTestBase):
  """chunk_kda / recurrent_kda vs the HF torch reference kernels."""

  def _inputs(self, b=2, s=37, h=3, d=8):
    q, k, v = (torch.randn(b, s, h, d) for _ in range(3))
    g = -5.0 * torch.sigmoid(torch.randn(b, s, h, d))
    beta = torch.sigmoid(torch.randn(b, s, h))
    return q, k, v, g, beta

  def test_chunk_matches_hf_chunk(self):
    for chunk in (8, 16, 64):
      q, k, v, g, beta = self._inputs()
      hf_o, hf_state = glm5.chunk_kimi_delta_attention(
          q, k, v, g, beta, chunk_size=chunk, output_final_state=True, use_qk_l2norm_in_kernel=True
      )
      mt_o, mt_state = kda.chunk_kda(t2j(q), t2j(k), t2j(v), t2j(g), t2j(beta), chunk_size=chunk)
      self.assert_close(mt_o, hf_o, atol=2e-4)
      self.assert_close(mt_state, hf_state, atol=2e-4)

  def test_chunk_matches_hf_recurrent(self):
    q, k, v, g, beta = self._inputs()
    hf_o, _ = glm5.recurrent_kimi_delta_attention(
        q, k, v, g, beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
    )
    mt_o, _ = kda.chunk_kda(t2j(q), t2j(k), t2j(v), t2j(g), t2j(beta), chunk_size=16)
    mt_rec, _ = kda.recurrent_kda(t2j(q), t2j(k), t2j(v), t2j(g), t2j(beta))
    self.assert_close(mt_o, hf_o, atol=2e-4)
    self.assert_close(mt_rec, hf_o, atol=2e-4)


class KDABlockTest(Glm5NextTestBase):

  def test_kda_block_matches_hf(self):
    cfg = hf_config()
    hf_attn = glm5.Glm5NextTextLinearAttention(cfg, layer_idx=0).eval()
    randomize(hf_attn)
    mt_attn = kda.KimiDeltaAttention(mt_config(), self.mesh, jnp.float32, rngs=nnx.Rngs(0))
    copy_kda(hf_attn, mt_attn)
    x = torch.randn(2, 45, EMB)
    hf_out = hf_attn(x)
    mt_out, _ = mt_attn(t2j(x), model_mode=MODEL_MODE_TRAIN)
    self.assert_close(mt_out, hf_out, atol=2e-4)


class SparseMLATest(Glm5NextTestBase):

  def _build(self, seq):
    cfg = hf_config()
    hf_attn = glm5.Glm5NextTextAttention(cfg, layer_idx=3).eval()
    randomize(hf_attn, seed=1)
    mt_attn = Glm5SparseMLA(mt_config(), self.mesh, jnp.float32, rngs=nnx.Rngs(0))
    copy_mla(hf_attn, mt_attn)
    x = torch.randn(2, seq, EMB)
    return hf_attn, mt_attn, x

  def test_indexer_selection_matches_hf(self):
    for seq in (3, 8, 29, 64):
      hf_attn, mt_attn, x = self._build(seq)
      q_resid = hf_attn.q_a_layernorm(hf_attn.q_a_proj(x))
      topk = hf_attn.indexer(x, q_resid, torch.ones(2, seq, dtype=torch.bool), None)
      hf_mask = hf_attn.build_attention_mask_from_topk(topk, torch.zeros(1), seq)[:, 0]
      hf_mask = hf_mask == 0 if hf_mask.dtype != torch.bool else hf_mask
      mt_mask = mt_attn.indexer(t2j(x), t2j(q_resid))
      np.testing.assert_array_equal(np.asarray(mt_mask), hf_mask.numpy(), err_msg=f"seq={seq}")
      if seq > TOPK + KPOOL:
        # Sparse selection must actually drop tokens for long sequences.
        self.assertLess(int(np.asarray(mt_mask)[:, -1].sum(-1).max()), seq)

  def test_sparse_mla_matches_hf(self):
    for seq in (8, 37):
      hf_attn, mt_attn, x = self._build(seq)
      hf_out, _, _ = hf_attn(x, attention_mask=torch.ones(2, seq, dtype=torch.bool))
      mt_out, _ = mt_attn(t2j(x), model_mode=MODEL_MODE_TRAIN)
      self.assert_close(mt_out, hf_out, atol=2e-4)


class KPoolIndexerTest(Glm5NextTestBase):

  def test_kpool1_is_plain_topk(self):
    """With kpool=1 the indexer reduces to per-token causal top-k."""
    idx = KPoolIndexer(mt_config(indexer_kpool=1, indexer_topk=4), jnp.float32, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 12, EMB))
    q = jax.random.normal(jax.random.key(1), (1, 12, Q_LORA))
    mask = np.asarray(idx(x, q))
    self.assertTrue((mask.sum(-1) == np.minimum(np.arange(12) + 1, 4)).all())
    self.assertFalse(np.triu(mask[0], k=1).any())


class DecoderLayerTest(Glm5NextTestBase):

  def _run(self, layer_idx, seq=29):
    cfg = hf_config()
    hf_layer = glm5.Glm5NextTextDecoderLayer(cfg, layer_idx=layer_idx).eval()
    randomize(hf_layer, seed=layer_idx)
    mt_layer = Glm5NextDecoderLayer(
        mt_config(), MODEL_MODE_TRAIN, self.mesh, nnx.Rngs(0), layer_idx=layer_idx
    )
    if mt_layer.is_dsa:
      copy_mla(hf_layer.self_attn, mt_layer.self_attention)
    else:
      copy_kda(hf_layer.self_attn, mt_layer.self_attention)
    if mt_layer.is_dense:
      copy_mlp(hf_layer.mlp, mt_layer.mlp)
    else:
      copy_moe(hf_layer.mlp, mt_layer.mlp)
    set_param(mt_layer.pre_self_attention_layer_norm.scale, hf_layer.input_layernorm.weight.detach().numpy())
    set_param(mt_layer.post_self_attention_layer_norm.scale, hf_layer.post_attention_layernorm.weight.detach().numpy())
    copy_mhc(hf_layer.attn_hc, mt_layer.mhc_attention)
    copy_mhc(hf_layer.ffn_hc, mt_layer.mhc_mlp)

    x = torch.randn(1, seq, 4, EMB)
    mask = torch.ones(1, seq, dtype=torch.bool)
    hf_out, _ = hf_layer(x, attention_mask=mask, prev_topk_indices=None)
    seg = jnp.ones((1, seq), jnp.int32)
    pos = jnp.arange(seq)[None]
    mt_out, _ = mt_layer(t2j(x), seg, pos, True, MODEL_MODE_TRAIN)
    self.assert_close(mt_out, hf_out, atol=5e-4, rtol=5e-4)

  def test_kda_dense_layer(self):
    self._run(layer_idx=0)

  def test_dsa_moe_layer(self):
    self._run(layer_idx=3)

  def test_kda_moe_layer(self):
    self._run(layer_idx=5)


if __name__ == "__main__":
  unittest.main()
