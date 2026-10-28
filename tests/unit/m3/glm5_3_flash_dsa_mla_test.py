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

"""Unit and numerical parity tests for NoPE-DSA-MLA and DSA Indexer."""

# pylint: disable=protected-access

from __future__ import annotations

import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.m3.configs.glm5_3_flash_config import Glm5Config
from maxtext.m3.configs.glm5_3_flash_config import Glm5Sharding
from maxtext.m3.configs.glm5_3_flash_config import validate_no_packing
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5DsaMlaAttention
import numpy as np
import pytest

try:
  import torch
  from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
  from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5DsaMlaTest(parameterized.TestCase):
  """Tests forward/backward numerical parity and invariants of NoPE-DSA-MLA."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def _assert_allclose(self, a, b, atol=1e-5, rtol=1e-5):
    a_np, b_np = np.array(a), np.array(b)
    self.assertTrue(np.all(np.isfinite(a_np)), "Non-finite values in actual")
    self.assertTrue(np.all(np.isfinite(b_np)), "Non-finite values in reference")
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol)

  def _mini_cfg(self, **kwargs) -> Glm5Config:
    """Constructs a mini Glm5Config for NoPE-DSA-MLA tests."""
    d = {
        "hidden_size": 64,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "index_n_heads": 8,
        "index_head_dim": 16,
        "index_topk": 16,
        "dtype": "float32",
    }
    d.update(kwargs)
    return Glm5Config.mini_config(**d)

  def _hf_layer(self, c: Glm5Config) -> Glm5NextTextAttention:
    """Constructs a matching HF Glm5NextTextAttention reference layer."""
    cfg = Glm5NextTextConfig(
        hidden_size=c.hidden_size,
        q_lora_rank=c.q_lora_rank,
        kv_lora_rank=c.kv_lora_rank,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        qk_nope_head_dim=c.qk_nope_head_dim,
        qk_rope_head_dim=0,
        v_head_dim=c.v_head_dim,
        index_n_heads=c.index_n_heads,
        index_head_dim=c.index_head_dim,
        index_topk=c.index_topk,
        index_kpool=c.index_kpool,
        index_kpool_always_select_tail=c.index_kpool_always_select_tail,
        rms_norm_eps=c.rms_norm_eps,
        indexer_types=["full"],
        num_hidden_layers=1,
        layer_types=["full_attention"],
        _attn_implementation="eager",
    )
    layer = Glm5NextTextAttention(cfg, layer_idx=0).to(torch.float32)
    with torch.no_grad():
      torch.nn.init.normal_(layer.indexer.index_kpool_compress_ape, std=0.1)
      torch.nn.init.normal_(layer.indexer.index_kpool_compress_gate, std=0.1)
    return layer

  def _copy_weights(self, hf_l, jax_l):
    """Copies weights from HF Glm5NextTextAttention into JAX Glm5DsaMlaAttention."""

    def _val(x):
      return jnp.array(x.detach().numpy())

    for p in ("q_a", "q_b", "kv_b", "o"):
      w_t = getattr(hf_l, f"{p}_proj").weight.T
      getattr(jax_l, f"{p}_proj").kernel.value = _val(w_t)
    w_mqa = hf_l.kv_a_proj_with_mqa.weight.T
    jax_l.kv_a_proj_with_mqa.kernel.value = _val(w_mqa)
    for n in ("q_a_layernorm", "kv_a_layernorm"):
      getattr(jax_l, n).weight.value = _val(getattr(hf_l, n).weight)

    if hf_l.indexer is not None and jax_l.indexer is not None:
      hf_i, jax_i = hf_l.indexer, jax_l.indexer
      for p in ("wq_b", "wk", "weights_proj"):
        getattr(jax_i, p).kernel.value = _val(getattr(hf_i, p).weight.T)
      jax_i.k_norm.weight.value = _val(hf_i.k_norm.weight)
      jax_i.k_norm.bias.value = _val(hf_i.k_norm.bias)
      jax_i.index_kpool_compress_ape.value = _val(hf_i.index_kpool_compress_ape)
      jax_i.index_kpool_compress_gate.value = _val(hf_i.index_kpool_compress_gate)

  @parameterized.named_parameters(
      ("dense_seq10", 10, 16, 4),
      ("sparse_seq22_with_tail", 22, 8, 4),
  )
  def test_indexer_parity(self, seq_len: int, index_topk: int, kpool: int):
    """Verifies indexer top-k selection and attention mask parity vs HF."""
    cfg = self._mini_cfg(index_topk=index_topk, index_kpool=kpool)
    hf_l = self._hf_layer(cfg)
    jax_l = Glm5DsaMlaAttention(cfg, layer_idx=0, indexer_type="full", rngs=nnx.Rngs(0))
    self._copy_weights(hf_l, jax_l)

    b = 2
    x_np = np.random.randn(b, seq_len, cfg.hidden_size).astype(np.float32)
    q_resid_np = np.random.randn(b, seq_len, cfg.q_lora_rank).astype(np.float32)
    mask_np = np.ones((b, seq_len), dtype=bool)
    mask_np[1, seq_len - 3 :] = False

    x_pt, q_pt, m_pt = map(torch.from_numpy, (x_np, q_resid_np, mask_np))
    hf_topk = hf_l.indexer(x_pt, q_pt, m_pt, past_key_values=None)
    x_j, q_j, m_j = map(jnp.array, (x_np, q_resid_np, mask_np))
    assert jax_l.indexer is not None
    jax_topk = jax_l.indexer(x_j, q_j, attention_mask=m_j)

    h, d = cfg.num_attention_heads, cfg.qk_nope_head_dim
    q_dummy = torch.zeros(b, h, seq_len, d)
    hf_mask = hf_l.build_attention_mask_from_topk(hf_topk, q_dummy, seq_len)
    q_arr = jnp.array(q_dummy.numpy())
    jax_mask = jax_l.build_attention_mask_from_topk(jax_topk, q_arr, seq_len)
    mask_equal = (hf_mask == 0.0).numpy()
    np.testing.assert_array_equal(np.array(jax_mask == 0.0), mask_equal)

  @parameterized.named_parameters(
      ("dense_seq10_unpadded", 10, 16, "none"),
      ("sparse_seq22_unpadded", 22, 8, "none"),
      ("dense_seq10_right_padded", 10, 16, "right"),
      ("sparse_seq22_right_padded", 22, 8, "right"),
      ("dense_seq10_left_padded", 10, 16, "left"),
      ("sparse_seq22_left_padded", 22, 8, "left"),
  )
  def test_dsa_mla_forward_backward_parity(
      self,
      seq_len: int,
      index_topk: int,
      pad_mode: str = "none",
  ):
    """Verifies forward and backward gradients against HF eager attention."""
    cfg = self._mini_cfg(index_topk=index_topk)
    hf_l = self._hf_layer(cfg)
    jax_l = Glm5DsaMlaAttention(cfg, layer_idx=0, indexer_type="full", rngs=nnx.Rngs(0))
    self._copy_weights(hf_l, jax_l)

    b = 2
    x_np = np.random.randn(b, seq_len, cfg.hidden_size).astype(np.float32)

    if pad_mode == "none":
      decoder_segment_ids = None
      mask_pt = torch.ones((b, seq_len), dtype=torch.bool)
      valid_mask = np.ones((b, seq_len), dtype=bool)
    elif pad_mode == "right":
      seg_ids_np = np.ones((b, seq_len), dtype=np.int32)
      seg_ids_np[0, seq_len - 2 :] = 0
      seg_ids_np[1, seq_len - 4 :] = 0
      decoder_segment_ids = jnp.array(seg_ids_np)
      mask_pt = torch.from_numpy(seg_ids_np != 0)
      valid_mask = seg_ids_np != 0
    elif pad_mode == "left":
      seg_ids_np = np.ones((b, seq_len), dtype=np.int32)
      seg_ids_np[0, :2] = 0
      seg_ids_np[1, :4] = 0
      decoder_segment_ids = jnp.array(seg_ids_np)
      mask_pt = torch.from_numpy(seg_ids_np != 0)
      valid_mask = seg_ids_np != 0
    else:
      raise ValueError(f"Unknown pad_mode: {pad_mode}")

    valid_expanded = valid_mask[:, :, None]

    out_pt = hf_l(torch.from_numpy(x_np), attention_mask=mask_pt)[0]
    out_jax = jax_l(jnp.array(x_np), decoder_segment_ids=decoder_segment_ids)
    self._assert_allclose(out_jax * valid_expanded, out_pt.detach().numpy() * valid_expanded)

    cot_np = np.random.randn(b, seq_len, cfg.hidden_size).astype(np.float32)
    cot_np = cot_np * valid_expanded
    x_pt = torch.from_numpy(x_np).requires_grad_(True)
    cot_pt = torch.from_numpy(cot_np)
    loss_pt = (hf_l(x_pt, attention_mask=mask_pt)[0] * cot_pt).sum()
    loss_pt.backward()

    def loss(m, x):
      return jnp.sum(m(x, decoder_segment_ids=decoder_segment_ids) * jnp.array(cot_np))

    vg = nnx.value_and_grad(loss, argnums=(0, 1))
    _, (grads, grad_x) = vg(jax_l, jnp.array(x_np))

    self._assert_allclose(grad_x * valid_expanded, x_pt.grad.numpy() * valid_expanded)
    for p in ("q_a", "q_b", "kv_b", "o"):
      gw = getattr(hf_l, f"{p}_proj").weight.grad.numpy().T
      self._assert_allclose(getattr(grads, f"{p}_proj").kernel.value, gw)
    gw_mqa = hf_l.kv_a_proj_with_mqa.weight.grad.numpy().T
    self._assert_allclose(grads.kv_a_proj_with_mqa.kernel.value, gw_mqa)
    for n in ("q_a_layernorm", "kv_a_layernorm"):
      gw = getattr(hf_l, n).weight.grad.numpy()
      self._assert_allclose(getattr(grads, n).weight.value, gw)

    if grads.indexer is not None:
      for p in jax.tree_util.tree_leaves(nnx.state(grads.indexer)):
        self.assertTrue(jnp.all(p == 0.0))

    if jax_l.indexer is not None:
      # D3: Verify indexer inputs (hidden_states, q_resid) are stop_gradient'ed
      # inside Glm5Indexer.__call__ even if continuous index_scores are summed.
      captured = {}
      orig_top_k = jax.lax.top_k

      def _capture_top_k(operand, k):
        captured["scores"] = operand
        return orig_top_k(operand, k)

      jax.lax.top_k = _capture_top_k
      try:
        q_res = jnp.ones((b, seq_len, cfg.q_lora_rank), dtype=jnp.float32)

        def _score_sum(x_in, q_in):
          assert jax_l.indexer is not None
          jax_l.indexer(x_in, q_in, attention_mask=jnp.array(valid_mask))
          return jnp.sum(captured["scores"])

        x_in = jnp.array(x_np)
        gx_idx, gq_idx = jax.grad(_score_sum, argnums=(0, 1))(x_in, q_res)
        self.assertTrue(jnp.all(gx_idx == 0.0))
        self.assertTrue(jnp.all(gq_idx == 0.0))
      finally:
        jax.lax.top_k = orig_top_k

  def test_index_kpool_always_select_tail_false_parity(self):
    """Verifies index_kpool_always_select_tail=False parity with HF."""
    cfg = self._mini_cfg(
        index_topk=8,
        index_kpool=4,
        index_kpool_always_select_tail=False,
    )
    hf_l = self._hf_layer(cfg)
    jax_l = Glm5DsaMlaAttention(cfg, layer_idx=0, indexer_type="full", rngs=nnx.Rngs(0))
    self._copy_weights(hf_l, jax_l)

    b, seq_len = 2, 22
    x_np = np.random.randn(b, seq_len, cfg.hidden_size).astype(np.float32)
    q_resid_np = np.random.randn(b, seq_len, cfg.q_lora_rank).astype(np.float32)
    mask_np = np.ones((b, seq_len), dtype=bool)

    x_pt, q_pt, m_pt = map(torch.from_numpy, (x_np, q_resid_np, mask_np))
    hf_topk = hf_l.indexer(x_pt, q_pt, m_pt, past_key_values=None)
    x_j, q_j, m_j = map(jnp.array, (x_np, q_resid_np, mask_np))
    assert jax_l.indexer is not None
    jax_topk = jax_l.indexer(x_j, q_j, attention_mask=m_j)

    self.assertEqual(jax_topk.shape[-1], 8)
    np.testing.assert_array_equal(np.array(jax_topk), hf_topk.numpy())

    out_pt = hf_l(torch.from_numpy(x_np), attention_mask=m_pt)[0]
    out_jax = jax_l(jnp.array(x_np))
    self._assert_allclose(out_jax, out_pt.detach().numpy())

  @parameterized.named_parameters(
      ("unpadded", "none"),
      ("right_padded", "right"),
      ("left_padded", "left"),
  )
  def test_static_bypass_bitwise_equals_indexer_path(self, pad_mode: str):
    """Proves static bypass mask and forward/backward outputs equal indexer path bitwise."""
    b, s = 2, 10
    cfg = self._mini_cfg(index_topk=16, index_kpool=4, index_kpool_always_select_tail=True)
    layer = Glm5DsaMlaAttention(cfg, layer_idx=0, indexer_type="full", rngs=nnx.Rngs(0))

    if pad_mode == "none":
      seg_ids = None
      attn_mask = None
    elif pad_mode == "right":
      seg_ids = jnp.array(
          [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
          dtype=jnp.int32,
      )
      attn_mask = seg_ids != 0
    elif pad_mode == "left":
      seg_ids = jnp.array(
          [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
          dtype=jnp.int32,
      )
      attn_mask = seg_ids != 0
    else:
      raise ValueError(f"Unknown pad_mode: {pad_mode}")

    x = jax.random.normal(jax.random.PRNGKey(1), (b, s, cfg.hidden_size), dtype=jnp.float32)
    q_resid = layer.q_a_layernorm(layer.q_a_proj(x))
    q = layer.q_b_proj(q_resid).reshape(b, s, cfg.num_attention_heads, cfg.qk_nope_head_dim).transpose(0, 2, 1, 3)

    # 1. Mask bitwise equality
    mask_bypass = layer._build_causal_padding_mask(attn_mask, b, s, q.dtype)
    assert layer.indexer is not None
    topk_idx = layer.indexer(x, q_resid, attention_mask=attn_mask)
    mask_indexer = layer.build_attention_mask_from_topk(topk_idx, q, s)
    np.testing.assert_array_equal(np.array(mask_bypass), np.array(mask_indexer))

    # 2. Forward output bitwise equality
    out_bypass = layer(x, decoder_segment_ids=seg_ids, _force_indexer=False)
    out_indexer = layer(x, decoder_segment_ids=seg_ids, _force_indexer=True)
    np.testing.assert_array_equal(np.array(out_bypass), np.array(out_indexer))

    # 3. Backward gradients bitwise equality
    cot = jax.random.normal(jax.random.PRNGKey(2), (b, s, cfg.hidden_size), dtype=jnp.float32)

    def loss_bypass(m, x_in):
      return jnp.sum(m(x_in, decoder_segment_ids=seg_ids, _force_indexer=False) * cot)

    def loss_indexer(m, x_in):
      return jnp.sum(m(x_in, decoder_segment_ids=seg_ids, _force_indexer=True) * cot)

    _, (g_bypass, gx_bypass) = nnx.value_and_grad(loss_bypass, argnums=(0, 1))(layer, x)
    _, (g_indexer, gx_indexer) = nnx.value_and_grad(loss_indexer, argnums=(0, 1))(layer, x)

    np.testing.assert_array_equal(np.array(gx_bypass), np.array(gx_indexer))
    for p in ("q_a", "q_b", "kv_b", "o"):
      np.testing.assert_array_equal(
          np.array(getattr(g_bypass, f"{p}_proj").kernel.value),
          np.array(getattr(g_indexer, f"{p}_proj").kernel.value),
      )
    np.testing.assert_array_equal(
        np.array(g_bypass.kv_a_proj_with_mqa.kernel.value),
        np.array(g_indexer.kv_a_proj_with_mqa.kernel.value),
    )
    for n in ("q_a_layernorm", "kv_a_layernorm"):
      np.testing.assert_array_equal(
          np.array(getattr(g_bypass, n).weight.value),
          np.array(getattr(g_indexer, n).weight.value),
      )

  def test_use_indexer_false(self):
    """Verifies use_indexer=False bypasses indexer creation and runs dense attention cleanly."""
    cfg = self._mini_cfg(use_indexer=False, index_topk=8)
    layer = Glm5DsaMlaAttention(cfg, layer_idx=0, indexer_type="full", rngs=nnx.Rngs(0))
    self.assertIsNone(layer.indexer)

    b, s = 2, 22
    x = jax.random.normal(jax.random.PRNGKey(0), (b, s, cfg.hidden_size), dtype=jnp.float32)
    out = layer(x)
    self.assertEqual(out.shape, (b, s, cfg.hidden_size))
    self.assertTrue(np.all(np.isfinite(np.array(out))))

  def test_dsa_mla_sharding_and_guards(self):
    """Verifies parameter sharding under use_tp=False and use_tp=True, jit padding, and guards."""
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    cfg = self._mini_cfg()

    def _chk(items, spec):
      for x in items:
        p = x.kernel if hasattr(x, "kernel") else x
        self.assertEqual(p.sharding, jax.sharding.NamedSharding(mesh, spec))

    # Verify use_tp=False sharding
    sharding_notp = Glm5Sharding(use_tp=False)
    with jax.sharding.set_mesh(mesh):
      layer_notp = Glm5DsaMlaAttention(cfg, 0, indexer_type="full", sharding=sharding_notp, rngs=nnx.Rngs(0))
    idx_notp = layer_notp.indexer
    assert idx_notp is not None

    p_fsdp = P("fsdp", None)
    _chk(
        [
            layer_notp.q_a_proj,
            layer_notp.kv_a_proj_with_mqa,
            layer_notp.q_b_proj,
            layer_notp.kv_b_proj,
            idx_notp.wq_b,
            idx_notp.wk,
            idx_notp.weights_proj,
        ],
        p_fsdp,
    )
    _chk([layer_notp.o_proj], P(None, "fsdp"))
    _chk([idx_notp.k_norm.weight, idx_notp.k_norm.bias], P(None))
    _chk([idx_notp.index_kpool_compress_ape], P(None, None))
    _chk([idx_notp.index_kpool_compress_gate], P(None, "fsdp"))
    self.assertTrue(bool(jnp.all(idx_notp.index_kpool_compress_gate.value == 1.0)))

    # Verify use_tp=True sharding
    sharding_tp = Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      layer_tp = Glm5DsaMlaAttention(cfg, 0, indexer_type="full", sharding=sharding_tp, rngs=nnx.Rngs(0))
    idx_tp = layer_tp.indexer
    assert idx_tp is not None

    _chk(
        [
            layer_tp.q_a_proj,
            layer_tp.kv_a_proj_with_mqa,
            idx_tp.wk,
            idx_tp.weights_proj,
        ],
        p_fsdp,
    )
    _chk(
        [layer_tp.q_b_proj, layer_tp.kv_b_proj, idx_tp.wq_b],
        P("fsdp", "tensor"),
    )
    _chk([layer_tp.o_proj], P("tensor", "fsdp"))
    _chk([idx_tp.k_norm.weight, idx_tp.k_norm.bias], P(None))
    _chk([idx_tp.index_kpool_compress_ape], P(None, None))
    _chk([idx_tp.index_kpool_compress_gate], P(None, "fsdp"))

    # Verify ValueError guards
    for attr, val, pat in (
        ("indexer_loss_scaling_factor", 0.1, "indexer_loss_scaling_factor"),
        ("qk_rope_head_dim", 64, "NoPE"),
        ("indexer_types", ("shared",) * 5, "Shared indexer"),
    ):
      bad = self._mini_cfg()
      object.__setattr__(bad, attr, val)
      assert bad.indexer_types is not None
      with self.assertRaisesRegex(ValueError, pat):
        Glm5DsaMlaAttention(
            bad,
            layer_idx=0,
            indexer_type=bad.indexer_types[0],
            rngs=nnx.Rngs(0),
        )

    # Verify non-vacuous random-input padding and jit execution
    x_rand = jax.random.normal(jax.random.PRNGKey(7), (1, 8, cfg.hidden_size), dtype=jnp.float32)
    seg_ids = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    with jax.sharding.set_mesh(mesh):
      out_jit = jax.jit(layer_tp)(x_rand, decoder_segment_ids=seg_ids)
      out_val = layer_tp(x_rand[:, :6, :])
    self._assert_allclose(out_jit[:, :6, :], out_val)
    self.assertGreater(float(jnp.max(jnp.abs(out_val))), 1e-4)
    self.assertTrue(jnp.all(out_jit[:, 6:, :] == 0.0))

    # Packing rejection via decoder_segment_ids
    packed_ids = jnp.array([[1, 2, 1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "FU-D5"):
      validate_no_packing(packed_ids)
    with self.assertRaisesRegex(ValueError, "packing=True"):
      self._mini_cfg(packing=True)

  def test_bfloat16_mixed_precision(self):
    """Verifies bfloat16 mixed precision execution, indexer int32 mask, and float32 gradients."""
    cfg = self._mini_cfg(dtype="bfloat16", weight_dtype="float32", index_topk=4)
    layer = Glm5DsaMlaAttention(
        cfg,
        layer_idx=0,
        indexer_type=cfg.indexer_types[0],
        rngs=nnx.Rngs(0),
    )
    b, s, h = 2, 8, cfg.hidden_size
    # When index_kpool_always_select_tail=True, Glm5Indexer returns
    # index_topk + index_kpool - 1 indices per token.
    expected_topk_w = cfg.index_topk + cfg.index_kpool - 1
    x_bf16 = jax.random.normal(jax.random.key(0), (b, s, h), dtype=jnp.bfloat16)
    q_bf16 = jax.random.normal(jax.random.key(1), (b, s, cfg.q_lora_rank), dtype=jnp.bfloat16)
    mask_int = jnp.ones((b, s), dtype=jnp.int32)
    topk_idx = layer.indexer(x_bf16, q_bf16, attention_mask=mask_int)
    self.assertEqual(topk_idx.shape, (b, s, expected_topk_w))

    seg_ids = jnp.ones((b, s), dtype=jnp.int32)
    out, topk_ret = layer(x_bf16, seg_ids, return_topk_indices=True)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(topk_ret.shape, (b, s, expected_topk_w))
    self.assertTrue(np.all(np.isfinite(np.asarray(out, dtype=np.float32))))

    def loss_fn(m, x):
      return jnp.sum(m(x, seg_ids).astype(jnp.float32))

    grads_l, grads_x = nnx.grad(loss_fn, argnums=(0, 1))(layer, x_bf16)
    self.assertEqual(grads_x.dtype, jnp.bfloat16)
    self.assertTrue(np.all(np.isfinite(np.asarray(grads_x, dtype=np.float32))))
    flat_params = dict(nnx.state(grads_l, nnx.Param).flat_state())
    self.assertNotEmpty(flat_params)
    for path, p in flat_params.items():
      self.assertEqual(p[...].dtype, jnp.float32, f"Expected float32 for {path}")
      self.assertTrue(
          np.all(np.isfinite(np.asarray(p[...]))),
          f"Expected finite grads for {path}",
      )

  def test_dsa_attention_implementation_dispatch_and_guards(self):
    """Verifies dsa_attention_implementation dispatches to dense_mask and raises on unsupported modes."""
    cfg = self._mini_cfg()
    layer = Glm5DsaMlaAttention(cfg, 0, indexer_type="full", rngs=nnx.Rngs(0))
    self.assertEqual(layer.config.dsa_attention_implementation, "dense_mask")

    x_dummy = jnp.ones((1, 8, cfg.hidden_size), dtype=jnp.float32)
    out = layer(x_dummy)
    self.assertEqual(out.shape, (1, 8, cfg.hidden_size))

    for bad_mode in (
        "tokamax_splash",
        "gather_scan",
        "flash",
        "invalid",
        123,
        None,
    ):
      object.__setattr__(layer.config, "dsa_attention_implementation", bad_mode)
      with self.assertRaisesRegex(ValueError, "Unsupported Glm5Config.dsa_attention_implementation"):
        layer(x_dummy)
    object.__setattr__(layer.config, "dsa_attention_implementation", "dense_mask")


if __name__ == "__main__":
  absltest.main()
