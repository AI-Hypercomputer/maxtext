# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Qwen Sparse Attention (QSA), Indexer, and mRoPE."""

from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen38_flash_next import qsa_attention  # pylint: disable=no-name-in-module
import numpy as np
import pytest

torch: Any
configuration_qwen4_exp: Any
modeling_qwen4_exp: Any

try:
  import torch  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import configuration_qwen4_exp  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top

  _HAS_QWEN4_EXP = True
except ImportError:
  torch = None
  configuration_qwen4_exp = None
  modeling_qwen4_exp = None
  _HAS_QWEN4_EXP = False


def _make_hf_config(**kwargs) -> Any:
  """Constructs a test Qwen4ExpTextConfig."""
  if configuration_qwen4_exp is None:
    raise RuntimeError("transformers.models.qwen4_exp is not available")
  rope = {
      "rope_type": "default",
      "rope_theta": 10000.0,
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": True,
      "mrope_section": [2, 1, 1],
  }
  cfg = {
      "hidden_size": 64,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 32,
      "indexer_n_heads": 2,
      "indexer_kv_heads": 1,
      "indexer_head_dim": 32,
      "indexer_budget": 16,
      "indexer_compress_ratio": 4,
      "rms_norm_eps": 1e-6,
      "attention_bias": False,
      "attention_dropout": 0.0,
      "rope_parameters": rope,
  }
  cfg.update(kwargs)
  return configuration_qwen4_exp.Qwen4ExpTextConfig(**cfg)


def _copy_weights(pt: Any, jx: qsa_attention.Qwen38SparseAttention):
  """Copies PyTorch attention weights to JAX QSA module."""

  def cv(t: Any) -> jax.Array:
    return jnp.array(t.detach().cpu().numpy())

  jx.q_proj.kernel[...] = cv(pt.q_proj.weight.T)
  jx.k_proj.kernel[...] = cv(pt.k_proj.weight.T)
  jx.v_proj.kernel[...] = cv(pt.v_proj.weight.T)
  jx.o_proj.kernel[...] = cv(pt.o_proj.weight.T)
  jx.q_norm.scale[...] = cv(pt.q_norm.weight)
  jx.k_norm.scale[...] = cv(pt.k_norm.weight)
  jx.indexer.index_qk_proj.kernel[...] = cv(pt.indexer.index_qk_proj.weight.T)
  jx.indexer.q_layernorm.scale[...] = cv(pt.indexer.q_layernorm.weight)
  jx.indexer.k_layernorm.scale[...] = cv(pt.indexer.k_layernorm.weight)


@pytest.mark.cpu_only
class Qwen38QSAAttentionTest(absltest.TestCase):
  """Tests QSA attention, indexer, and mRoPE against PyTorch reference."""

  def _make_jax_qsa(
      self, emb_dim: int = 64, rng_seed: int = 0, token_budget: int = 16
  ) -> qsa_attention.Qwen38SparseAttention:
    """Constructs a test JAX QSA module."""
    args = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, token_budget, 4, 1e-6, jnp.float32, jnp.float32)
    return qsa_attention.Qwen38SparseAttention(*args, rngs=nnx.Rngs(rng_seed))

  def test_qsa_parameter_tree_contract(self):
    """Verifies shapes and dtypes of all QSA parameters."""
    m = self._make_jax_qsa()
    self.assertEqual((m.q_proj.kernel[...].shape, m.k_proj.kernel[...].shape), ((64, 256), (64, 64)))
    self.assertEqual((m.v_proj.kernel[...].shape, m.o_proj.kernel[...].shape), ((64, 64), (128, 64)))
    self.assertEqual((m.q_norm.scale[...].shape, m.k_norm.scale[...].shape), ((32,), (32,)))
    self.assertEqual(
        (m.indexer.index_qk_proj.kernel[...].shape, m.indexer.q_layernorm.scale[...].shape), ((64, 96), (32,))
    )
    self.assertEqual((m.indexer.k_layernorm.scale[...].shape, m.q_proj.kernel[...].dtype), ((32,), jnp.float32))

  def _make_pt_and_jax(
      self, seed: int = 0, token_budget: int = 16
  ) -> tuple[Any, Any, qsa_attention.Qwen38SparseAttention]:
    """Constructs parity-initialized PyTorch and JAX attention modules."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      raise RuntimeError("PyTorch and transformers.models.qwen4_exp must be available")
    torch.manual_seed(seed)
    cfg = _make_hf_config(indexer_budget=token_budget)
    pt_r = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(cfg)
    pt_a = modeling_qwen4_exp.Qwen4ExpTextAttention(cfg, layer_idx=3)
    jax_a = self._make_jax_qsa(rng_seed=seed, token_budget=token_budget)
    _copy_weights(pt_a, jax_a)
    return pt_r, pt_a, jax_a

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_rotary_embedding_and_partial_mrope_pytorch_parity(self):
    """Verifies partial interleaved mRoPE matches PyTorch reference."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      self.fail("transformers qwen4_exp reference modules are missing")
    hf_cfg = _make_hf_config()
    pt_rope = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(hf_cfg)
    jax_rope = qsa_attention.Qwen38RotaryEmbedding(32, 0.25, 10000.0, (2, 1, 1))
    b, s = 2, 16
    pos_2d = np.tile(np.arange(s), (b, 1))
    pos_3d = np.stack([pos_2d, pos_2d + 7, pos_2d * 2], axis=0)
    x_np = np.ones((b, s, 32), dtype=np.float32)

    for case_name, pos in (("2d", pos_2d), ("3d", pos_3d)):
      with self.subTest(case_name=case_name):
        pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos))
        jax_cos, jax_sin = jax_rope(jnp.array(x_np), jnp.array(pos))
        np.testing.assert_allclose(np.array(jax_cos), pt_cos.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.array(jax_sin), pt_sin.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_qsa_attention_pytorch_weight_copy_parity_at_32(self):
    """Verifies output and selected-block parity vs HF attention at seq_len=32."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      self.fail("transformers qwen4_exp reference modules are missing")
    b, s, emb_dim = 1, 32, 64
    pt_rope, pt_attn, jax_attn = self._make_pt_and_jax(seed=0)

    x = np.random.default_rng(42).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    pt_cos, pt_sin = pt_rope(torch.from_numpy(x), torch.from_numpy(pos))
    mask = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool))
    pt_float_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)

    pt_out, _ = pt_attn(torch.from_numpy(x), (pt_cos, pt_sin), attention_mask=pt_float_mask)
    jax_out, jax_blocks = jax_attn(jnp.array(x), position_ids=jnp.array(pos), return_selected_blocks=True)

    pt_tok_mask = pt_attn.indexer(torch.from_numpy(x), (pt_cos, pt_sin), pt_float_mask, None)
    pt_tok_mask_np = pt_tok_mask.detach().cpu().numpy() == 0.0
    for q_idx in range(s):
      n_complete = (q_idx + 1) // 4
      if n_complete > 0:
        hf_sel = {m for m in range(n_complete) if np.all(pt_tok_mask_np[0, 0, q_idx, m * 4 : (m + 1) * 4])}
        jax_sel = {int(idx) for idx in np.array(jax_blocks[0, q_idx]) if idx >= 0}
        self.assertEqual(hf_sel, jax_sel)
    np.testing.assert_allclose(np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

  def test_indexer_packed_sequence_and_partial_block_handling(self):
    """Verifies packed-sequence isolation and partial block selection (block_topk < n_complete)."""
    b, s, emb_dim = 1, 28, 64
    # Document A = 1 on [0..7] (blocks 0, 1), Document B = 2 on [8..25], Padding = 0 on [26..27]
    seg_ids = np.zeros((b, s), dtype=np.int32)
    seg_ids[0, :8], seg_ids[0, 8:25] = 1, 2

    x = np.random.default_rng(123).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    # indexer_budget=8 gives block_topk = 8 // 4 = 2.
    # At query t=20, Document B has 3 candidate blocks: [8..11] (2), [12..15] (3), [16..19] (4).
    # Since block_topk = 2 < 3, top-k selection picks exactly 2 blocks from {2, 3, 4}.
    jax_attn = self._make_jax_qsa(emb_dim=emb_dim, rng_seed=1, token_budget=8)

    pt_out: Any = None
    pt_tok_mask_np: Any = None
    if _HAS_QWEN4_EXP:
      if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
        self.fail("transformers qwen4_exp reference modules are missing")
      pt_rope, pt_attn, _ = self._make_pt_and_jax(seed=1, token_budget=8)
      _copy_weights(pt_attn, jax_attn)

      pt_cos, pt_sin = pt_rope(torch.from_numpy(x), torch.from_numpy(pos))
      same = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] != 0)
      pt_valid = same & np.tril(np.ones((s, s), dtype=bool))[None, :, :]
      pt_float_mask = torch.where(
          torch.from_numpy(pt_valid[:, None, :, :]), torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min
      )

      pt_out, _ = pt_attn(torch.from_numpy(x), (pt_cos, pt_sin), attention_mask=pt_float_mask)
      pt_tok_mask = pt_attn.indexer(torch.from_numpy(x), (pt_cos, pt_sin), pt_float_mask, None)
      pt_tok_mask_np = pt_tok_mask.detach().cpu().numpy() == 0.0

    jax_out, jax_blocks = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), segment_ids=jnp.array(seg_ids), return_selected_blocks=True
    )
    cos, sin = jax_attn.rotary_emb(jnp.array(x), jnp.array(pos))
    tok_mask, _ = jax_attn.indexer(jnp.array(x), (cos, sin), segment_ids=jnp.array(seg_ids))
    tok_mask_np = np.array(tok_mask)[0, 0, 20]

    num_blocks = s // 4
    jax_active_blocks = {m for m in range(num_blocks) if np.all(tok_mask_np[m * 4 : (m + 1) * 4])}

    # Pure-JAX invariants:
    self.assertLen(jax_active_blocks, 2)
    self.assertTrue(jax_active_blocks.issubset({2, 3, 4}))
    self.assertNotIn(0, jax_active_blocks)
    self.assertNotIn(1, jax_active_blocks)
    self.assertFalse(np.any(tok_mask_np[:8]))  # Document A tokens NEVER visible
    self.assertTrue(tok_mask_np[20])  # Tail token visible
    self.assertFalse(np.any(tok_mask_np[21:]))  # Future tokens not visible
    selected_jax_blocks = {int(idx) + 2 for idx in np.array(jax_blocks[0, 20]) if idx >= 0}
    self.assertEqual(selected_jax_blocks, jax_active_blocks)

    if _HAS_QWEN4_EXP:
      hf_active_blocks = {m for m in range(num_blocks) if np.all(pt_tok_mask_np[0, 0, 20, m * 4 : (m + 1) * 4])}
      self.assertEqual(jax_active_blocks, hf_active_blocks)
      np.testing.assert_allclose(np.array(jax_out)[:, :25], pt_out[:, :25].detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

  def test_attention_mask_folding_and_init_validations(self):
    """Verifies 2D mask folding equivalence, 4D mask rejection, and init validation raises."""
    b, s, emb_dim = 2, 16, 64
    x = np.random.default_rng(0).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    # 1. 2D mask folding equivalence
    mask_2d = np.ones((b, s), dtype=np.int32)
    mask_2d[0, 12:] = 0
    mask_2d[1, 14:] = 0

    jax_attn = self._make_jax_qsa(emb_dim=emb_dim, rng_seed=42)
    out_mask, blocks_mask = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), attention_mask=jnp.array(mask_2d), return_selected_blocks=True
    )
    folded_seg = (mask_2d != 0).astype(np.int32)
    out_seg, blocks_seg = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), segment_ids=jnp.array(folded_seg), return_selected_blocks=True
    )
    np.testing.assert_allclose(np.array(out_mask), np.array(out_seg), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.array(blocks_mask), np.array(blocks_seg))

    # 2. 4D mask rejection
    mask_4d = jnp.zeros((b, 1, s, s), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      jax_attn(jnp.array(x), attention_mask=mask_4d)

    cos, sin = jax_attn.rotary_emb(jnp.array(x), jnp.array(pos))
    with self.assertRaises(ValueError):
      jax_attn.indexer(jnp.array(x), (cos, sin), attention_mask=mask_4d)

    # 3. Init validation raises
    # Qwen38RotaryEmbedding: length != 3 or sum != dim // 2
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38RotaryEmbedding(head_dim=32, mrope_section=(2, 2))
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38RotaryEmbedding(head_dim=32, mrope_section=(2, 2, 1))
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38RotaryEmbedding(head_dim=64, mrope_section=(11, 11, 10))

    # Qwen38QSAIndexer: index_kv_heads != 1, compress_ratio <= 0, token_budget < compress_ratio
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38QSAIndexer(index_kv_heads=2, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38QSAIndexer(compress_ratio=0, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38QSAIndexer(token_budget=2, compress_ratio=4, rngs=nnx.Rngs(0))

    # Qwen38SparseAttention: num_query_heads % num_kv_heads != 0
    with self.assertRaises(ValueError):
      qsa_attention.Qwen38SparseAttention(num_query_heads=5, num_kv_heads=2, rngs=nnx.Rngs(0))


if __name__ == "__main__":
  absltest.main()
