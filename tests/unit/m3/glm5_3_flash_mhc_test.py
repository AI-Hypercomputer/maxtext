# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and numerical parity tests for GLM-5.3-Flash mHC and Decoder Layer."""

from __future__ import annotations

import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import numpy as np
import pytest

try:
  import torch
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]


def _to_np(x):
  return x.detach().cpu().numpy()


def _w(x):
  return jnp.array(x.detach().numpy())


def _t(x):
  return _w(x).T


def _make_cfg() -> tuple[glm5_config.Glm5Config, hf_cfg.Glm5NextTextConfig]:
  """Constructs matching mini Glm5Config and HF Glm5NextTextConfig."""
  f = (
      "hidden_size:64 intermediate_size:128 moe_intermediate_size:32"
      " n_routed_experts:8 num_experts_per_tok:2 q_lora_rank:32 kv_lora_rank:16"
      " num_attention_heads:4 num_key_value_heads:4 qk_nope_head_dim:16"
      " v_head_dim:16 index_n_heads:2 index_head_dim:16 index_topk:8"
      " linear_num_key_heads:4 linear_num_value_heads:4"
      " linear_key_head_dim:16 linear_value_head_dim:16"
  )
  k = {x.split(":")[0]: int(x.split(":")[1]) for x in f.split()}
  c = glm5_config.Glm5Config.mini_config(kda_implementation="xla", dtype="float32", **k)
  k_hf = {x: k[x] for x in k if not x.startswith("linear_") and x != "qk_head_dim"}
  k_hf.update(
      num_hidden_layers=5,
      mlp_layer_types=["dense"] * 3 + ["sparse"] * 2,
      layer_types=list(c.layer_types),
      indexer_types=list(c.indexer_types),
      _attn_implementation="eager",
      index_kpool=4,
      linear_num_heads=4,
      linear_head_dim=16,
      linear_conv_kernel_dim=4,
  )
  return c, hf_cfg.Glm5NextTextConfig(**k_hf)


def _set_proj(dst, src, names):
  for p in names:
    w = getattr(src, f"{p}_proj").weight
    getattr(dst, f"{p}_proj").kernel.value = _t(w)


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5MhcTest(parameterized.TestCase):
  """Numerical parity and invariant tests for mHC and Glm5DecoderLayer."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)
    self.cfg, self.hf_cfg = _make_cfg()

  def _assert_allclose(self, a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)

  def _copy_mhc(self, hf_hc, jax_hc):
    jax_hc.fn.value = _t(hf_hc.fn)
    jax_hc.base.value, jax_hc.scale.value = _w(hf_hc.base), _w(hf_hc.scale)

  def _copy_mlp(self, hf_m, jax_m):
    _set_proj(jax_m, hf_m, ("gate", "up", "down"))

  def _copy_moe(self, hf_m, jax_m):
    self._copy_mlp(hf_m.shared_experts, jax_m.shared_experts)
    g, hg = jax_m.gate, hf_m.gate
    g.weight.value = _t(hg.weight)
    g.e_score_correction_bias.value = _w(hg.e_score_correction_bias)
    gu = _w(hf_m.experts.gate_up_proj).swapaxes(1, 2)
    e, h, two_i = gu.shape
    jax_m.experts.gate_up_proj.value = gu.reshape(e, h, 2, two_i // 2)
    jax_m.experts.down_proj.value = _w(hf_m.experts.down_proj).swapaxes(1, 2)

  def _copy_decoder(self, hf_l, jax_l):
    """Copies weights from HF decoder layer into JAX Glm5DecoderLayer."""
    self._copy_mhc(hf_l.attn_hc, jax_l.attn_hc)
    self._copy_mhc(hf_l.ffn_hc, jax_l.ffn_hc)
    for n in ("input_layernorm", "post_attention_layernorm"):
      getattr(jax_l, n).weight.value = _w(getattr(hf_l, n).weight)
    (self._copy_moe if jax_l.is_moe else self._copy_mlp)(hf_l.mlp, jax_l.mlp)

    hf_a, ja = hf_l.self_attn, jax_l.self_attn
    if jax_l.block_type == "linear_attention":
      fg = hf_a.forget_gate
      _set_proj(ja, hf_a, ("q", "k", "v", "b", "o", "g_a", "g_b"))
      _set_proj(ja, fg, ("f_a", "f_b"))
      cw = _w(hf_a.conv1d.weight)
      for i, c in enumerate(("q", "k", "v")):
        getattr(ja, f"{c}_conv1d").weight.value = cw[i * 64 : (i + 1) * 64]
      ja.A_log.value, ja.dt_bias.value = _w(fg.A_log), _w(fg.dt_bias)
      ja.o_norm.weight.value = _w(hf_a.o_norm.weight)
    else:
      _set_proj(ja, hf_a, ("q_a", "q_b", "kv_b", "o"))
      ja.kv_a_proj_with_mqa.kernel.value = _t(hf_a.kv_a_proj_with_mqa.weight)
      for n in ("q_a_layernorm", "kv_a_layernorm"):
        getattr(ja, n).weight.value = _w(getattr(hf_a, n).weight)
      hf_i, ji = hf_a.indexer, ja.indexer
      for p in ("wq_b", "wk", "weights_proj"):
        getattr(ji, p).kernel.value = _t(getattr(hf_i, p).weight)
      for p in ("weight", "bias"):
        getattr(ji.k_norm, p).value = _w(getattr(hf_i.k_norm, p))
      for n in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
        getattr(ji, n).value = _w(getattr(hf_i, n))

  def test_mhc_forward_backward_sinkhorn(self):
    """Verifies forward/backward parity through 20 Sinkhorn iterations."""
    cfg, hf_c = self.cfg, self.hf_cfg
    hf_hc = hf_model.Glm5NextTextHyperConnection(hf_c).to(torch.float32)
    for p in (hf_hc.fn, hf_hc.base, hf_hc.scale):
      torch.nn.init.normal_(p, 0, 0.1)

    jax_hc = modeling_glm5_3_flash.Glm5HyperConnection(cfg, rngs=nnx.Rngs(0))
    self._copy_mhc(hf_hc, jax_hc)

    b, s, h, d = 2, 4, cfg.mhc_expansion_rate, cfg.hidden_size
    x_np = np.random.randn(b, s, h, d).astype(np.float32)

    res_pt = hf_hc(torch.from_numpy(x_np))
    post_jax, comb_jax, col_jax = jax_hc(jnp.array(x_np))
    for a, b_pt in zip((post_jax, comb_jax, col_jax), res_pt):
      self._assert_allclose(a, _to_np(b_pt))

    ones = jnp.ones((b, s, h))
    for ax in (-1, -2):
      self._assert_allclose(jnp.sum(comb_jax, axis=ax), ones, atol=1e-4)
    self.assertTrue(bool(jnp.all(comb_jax >= 0)))

    rnd = np.random.randn
    c_p, c_c, c_cl = rnd(b, s, h), rnd(b, s, h, h), rnd(b, s, d)
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    p_pt, c_pt, cl_pt = hf_hc(x_pt)
    loss_pt = (
        (p_pt * torch.from_numpy(c_p)).sum()
        + (c_pt * torch.from_numpy(c_c)).sum()
        + (cl_pt * torch.from_numpy(c_cl)).sum()
    )
    loss_pt.backward()

    def loss(m, x):
      p, c, cl = m(x)
      return jnp.sum(p * c_p) + jnp.sum(c * c_c) + jnp.sum(cl * c_cl)

    _, (grads, grad_x) = nnx.value_and_grad(loss, argnums=(0, 1))(jax_hc, jnp.array(x_np))
    self._assert_allclose(grad_x, _to_np(x_pt.grad))
    for g, w in (
        (grads.fn.value, _t(hf_hc.fn.grad)),
        (grads.base.value, _w(hf_hc.base.grad)),
        (grads.scale.value, _w(hf_hc.scale.grad)),
    ):
      self._assert_allclose(g, w)

  def test_hyper_head_and_unweighted_rmsnorm(self):
    """Verifies parity of Glm5HyperHead and Glm5UnweightedRMSNorm."""

    def _test(m_pt, m_jx, sh):
      x_np, cot_np = np.random.randn(*sh), np.random.randn(2, 4, 64)
      self._assert_allclose(m_jx(jnp.array(x_np)), _to_np(m_pt(torch.from_numpy(x_np))))
      x_pt = torch.from_numpy(x_np).requires_grad_(True)
      (m_pt(x_pt) * torch.from_numpy(cot_np)).sum().backward()
      g = jax.grad(lambda a: jnp.sum(m_jx(a) * cot_np))(jnp.array(x_np))
      self._assert_allclose(g, _to_np(x_pt.grad))

    _test(
        hf_model.Glm5NextTextHyperHead(),
        modeling_glm5_3_flash.Glm5HyperHead(),
        (2, 4, 4, 64),
    )
    _test(
        hf_model.Glm5NextTextUnweightedRMSNorm(1e-5),
        modeling_glm5_3_flash.Glm5UnweightedRMSNorm(1e-5),
        (2, 4, 64),
    )

  @parameterized.named_parameters(
      ("L0_kda_dense", 0),
      ("L3_mla_moe", 3),
      ("L4_kda_moe", 4),
  )
  def test_decoder_layer_permutations(self, layer_idx: int):
    """Verifies parity across all 3 block permutations: L0, L3, L4."""
    hf_l = hf_model.Glm5NextTextDecoderLayer(self.hf_cfg, layer_idx=layer_idx).to(torch.float32)
    with torch.no_grad():
      for name, p in hf_l.named_parameters():
        if "layernorm" in name or "norm" in name:
          if "bias" in name:
            p.zero_()
          else:
            p.fill_(1.0)
        elif "A_log" in name:
          p.uniform_(0.1, 1.0).log_()
        else:
          p.normal_(0.0, 0.02)
      if hasattr(hf_l.mlp, "gate"):
        hf_l.mlp.gate.e_score_correction_bias.normal_(0.0, 0.01)
      if hasattr(hf_l.self_attn, "indexer"):
        for n in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
          getattr(hf_l.self_attn.indexer, n).normal_(0.0, 0.02)

    jax_l = modeling_glm5_3_flash.Glm5DecoderLayer(self.cfg, layer_idx=layer_idx, rngs=nnx.Rngs(0))
    self._copy_decoder(hf_l, jax_l)

    b, s, h, d = 2, 4, self.cfg.mhc_expansion_rate, self.cfg.hidden_size
    x_np = np.random.randn(b, s, h, d).astype(np.float32)
    mask_pt = torch.ones((b, s), dtype=torch.bool)
    out_pt, _ = hf_l(torch.from_numpy(x_np), attention_mask=mask_pt)
    out_jax, router_logits, router_topk, _, _, _ = jax_l(jnp.array(x_np), return_aux=False)
    self._assert_allclose(out_jax, _to_np(out_pt), atol=1e-4, rtol=1e-4)

    if jax_l.is_moe:
      self.assertEqual(router_logits.shape, (b * s, self.cfg.n_routed_experts))
      self.assertEqual(router_topk.shape, (b, s, self.cfg.num_experts_per_tok))
    else:
      self.assertIsNone(router_logits)
      self.assertIsNone(router_topk)

    # Right-padded tail parity across all 3 decoder layer types
    orig_fn = hf_model.chunk_kimi_delta_attention
    hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
    try:
      mask_pad_pt = torch.tensor(
          [[True, True, True, False], [True, True, False, False]],
          dtype=torch.bool,
      )
      seg_jax = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32)
      out_pt_pad, _ = hf_l(torch.from_numpy(x_np), attention_mask=mask_pad_pt)
      out_jax_pad, _, _, _, _, _ = jax_l(jnp.array(x_np), decoder_segment_ids=seg_jax, return_aux=False)
      self._assert_allclose(out_jax_pad[0, :3], _to_np(out_pt_pad)[0, :3], atol=1e-4, rtol=1e-4)
      self._assert_allclose(out_jax_pad[1, :2], _to_np(out_pt_pad)[1, :2], atol=1e-4, rtol=1e-4)
    finally:
      hf_model.chunk_kimi_delta_attention = orig_fn

    cot_np = np.random.randn(b, s, h, d).astype(np.float32)
    orig_fn = hf_model.chunk_kimi_delta_attention
    hf_model.chunk_kimi_delta_attention = hf_model.recurrent_kimi_delta_attention
    try:
      x_pt = torch.from_numpy(x_np).requires_grad_(True)
      (hf_l(x_pt, attention_mask=mask_pt)[0] * torch.from_numpy(cot_np)).sum().backward()
    finally:
      hf_model.chunk_kimi_delta_attention = orig_fn

    grads_jax, grad_x = nnx.grad(lambda m, x: jnp.sum(m(x, return_aux=False)[0] * cot_np), argnums=(0, 1))(
        jax_l, jnp.array(x_np)
    )
    self._assert_allclose(grad_x, _to_np(x_pt.grad), atol=1e-4, rtol=1e-4)
    self._assert_allclose(
        grads_jax.attn_hc.fn.value,
        _to_np(hf_l.attn_hc.fn.grad).T,
        atol=1e-4,
        rtol=1e-4,
    )
    self._assert_allclose(
        grads_jax.attn_hc.scale.value,
        _to_np(hf_l.attn_hc.scale.grad),
        atol=1e-4,
        rtol=1e-4,
    )
    self._assert_allclose(
        grads_jax.ffn_hc.fn.value,
        _to_np(hf_l.ffn_hc.fn.grad).T,
        atol=1e-4,
        rtol=1e-4,
    )
    self._assert_allclose(
        grads_jax.self_attn.o_proj.kernel.value,
        _to_np(hf_l.self_attn.o_proj.weight.grad).T,
        atol=1e-4,
        rtol=1e-4,
    )
    if not jax_l.is_moe:
      self._assert_allclose(
          grads_jax.mlp.gate_proj.kernel.value,
          _to_np(hf_l.mlp.gate_proj.weight.grad).T,
          atol=1e-4,
          rtol=1e-4,
      )
    else:
      self._assert_allclose(
          grads_jax.mlp.shared_experts.gate_proj.kernel.value,
          _to_np(hf_l.mlp.shared_experts.gate_proj.weight.grad).T,
          atol=1e-4,
          rtol=1e-4,
      )

  def test_mhc_sharding_and_guards(self):
    """Verifies sharding specs under mesh and negative out-of-bounds guards."""
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    sh = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      hc = modeling_glm5_3_flash.Glm5HyperConnection(self.cfg, rngs=nnx.Rngs(0), sharding=sh)
    self.assertIsNotNone(hc.fn.sharding)

    # Positive test paired with negative tests (Negative Test Assertion Liveness)
    layer = modeling_glm5_3_flash.Glm5DecoderLayer(self.cfg, layer_idx=0, rngs=nnx.Rngs(0))
    self.assertEqual(layer.layer_idx, 0)
    for bad_idx in (-1, self.cfg.num_hidden_layers):
      with self.assertRaisesRegex(ValueError, "out of range"):
        modeling_glm5_3_flash.Glm5DecoderLayer(self.cfg, layer_idx=bad_idx, rngs=nnx.Rngs(0))

  def test_mhc_fresh_init_and_gradients(self):
    """Verifies fresh init invariants (fn non-zero, base 0, scale 1) and gradient flow."""
    hc = modeling_glm5_3_flash.Glm5HyperConnection(self.cfg, rngs=nnx.Rngs(0))
    self.assertGreater(float(jnp.max(jnp.abs(hc.fn.value))), 1e-4)
    self._assert_allclose(hc.base.value, jnp.zeros_like(hc.base.value))
    self._assert_allclose(hc.scale.value, jnp.ones_like(hc.scale.value))

    b, s, h, d = 2, 4, self.cfg.mhc_expansion_rate, self.cfg.hidden_size
    x = jax.random.normal(jax.random.PRNGKey(7), (b, s, h, d), dtype=jnp.float32)

    def loss(m, x_in):
      p, c, col = m(x_in)
      return jnp.sum(p) + jnp.sum(c) + jnp.sum(col)

    _, grads = nnx.value_and_grad(loss)(hc, x)
    self.assertGreater(float(jnp.max(jnp.abs(grads.fn.value))), 1e-4)
    self.assertGreater(float(jnp.max(jnp.abs(grads.base.value))), 1e-4)
    self.assertGreater(float(jnp.max(jnp.abs(grads.scale.value))), 1e-4)

    # Test constructor guard
    with self.assertRaisesRegex(ValueError, "rngs must be provided"):
      modeling_glm5_3_flash.Glm5HyperConnection(self.cfg, rngs=None)


if __name__ == "__main__":
  absltest.main()
