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

"""Unit tests for Cosmos 3 Core Attention Backbone (Step 2)."""

import os
import subprocess
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.layers import cosmos_attention
import numpy as np

# Module-level aliases for cosmos_attention symbols.
CosmosAttention = cosmos_attention.CosmosAttention
CosmosDualAttention = cosmos_attention.CosmosDualAttention
apply_rotary_pos_emb = cosmos_attention.apply_rotary_pos_emb
build_causal_understanding_mask = cosmos_attention.build_causal_understanding_mask
build_causal_understanding_splash_mask = cosmos_attention.build_causal_understanding_splash_mask
build_cosmos_packing_metadata = cosmos_attention.build_cosmos_packing_metadata
build_full_generative_mask = cosmos_attention.build_full_generative_mask
build_full_generative_splash_mask = cosmos_attention.build_full_generative_splash_mask
causal_understanding_attention = cosmos_attention.causal_understanding_attention
compile_cosmos_splash_mask = cosmos_attention.compile_cosmos_splash_mask
compute_3d_mrope_cos_sin = cosmos_attention.compute_3d_mrope_cos_sin
full_generative_attention = cosmos_attention.full_generative_attention
reinterleave_streams = cosmos_attention.reinterleave_streams
unpack_streams = cosmos_attention.unpack_streams

_GOLDEN_DATA_FILENAME = "cosmos_attention_golden_data.npz"


def _get_golden_data_path() -> str | None:
  """Attempts to download golden test asset from GCS bucket.

  Returns:
    Local file path if available, or None if unavailable or download failed.
  """
  tmp_path = f"/tmp/{_GOLDEN_DATA_FILENAME}"
  if os.path.exists(tmp_path):
    return tmp_path

  gcs_uri = f"gs://maxtext-test-assets/{_GOLDEN_DATA_FILENAME}"
  try:
    ret = subprocess.call(
        ["gcloud", "storage", "cp", gcs_uri, tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
    )
    if ret == 0 and os.path.exists(tmp_path):
      return tmp_path
  except (subprocess.SubprocessError, OSError):
    pass

  return None


class CosmosPackingLogicTest(parameterized.TestCase):
  """Tests for packing logic, metadata, unpacking, and re-interleaving."""

  def test_metadata_construction(self):
    """Verifies boundary metadata, offsets, and max lengths."""
    und_lens = [4, 6, 2]
    gen_lens = [8, 4, 10]
    metadata = build_cosmos_packing_metadata(und_lens, gen_lens)

    self.assertEqual(metadata.num_samples, 3)
    self.assertEqual(metadata.total_tokens, 4 + 6 + 2 + 8 + 4 + 10)  # 34
    self.assertEqual(metadata.num_und_tokens, 4 + 6 + 2)  # 12
    self.assertEqual(metadata.num_gen_tokens, 8 + 4 + 10)  # 22
    self.assertEqual(metadata.max_causal_len, 6)
    self.assertEqual(metadata.max_full_len, 10)
    self.assertEqual(metadata.max_sample_len, 12)  # sample 0: 4+8=12, sample 2: 2+10=12

    # Verify cumulative offsets
    np.testing.assert_array_equal(metadata.causal_q_offsets, [0, 4, 10, 12])
    np.testing.assert_array_equal(metadata.full_q_offsets, [0, 8, 12, 22])
    np.testing.assert_array_equal(metadata.sample_kv_offsets, [0, 12, 22, 34])

  def test_unpack_and_reinterleave_roundtrip(self):
    """Verifies stream unpacking and re-interleaving form an exact roundtrip."""
    und_lens = [5, 3]
    gen_lens = [7, 4]
    metadata = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 64
    rng = jax.random.PRNGKey(42)
    packed_tokens = jax.random.normal(rng, (metadata.total_tokens, dim))

    und_tokens, gen_tokens = unpack_streams(
        packed_tokens,
        metadata.packed_und_token_indexes,
        metadata.packed_gen_token_indexes,
    )
    self.assertEqual(und_tokens.shape, (8, dim))
    self.assertEqual(gen_tokens.shape, (11, dim))

    reconstructed = reinterleave_streams(
        und_tokens,
        gen_tokens,
        metadata.packed_und_token_indexes,
        metadata.packed_gen_token_indexes,
        metadata.total_tokens,
    )
    np.testing.assert_allclose(packed_tokens, reconstructed, rtol=1e-6, atol=1e-6)


class CosmosMaskingTest(parameterized.TestCase):
  """Tests for sample isolation and attention masks."""

  def test_causal_understanding_mask(self):
    """Verifies Kernel 1 mask enforces causality and cross-sample fences."""
    # 2 samples: sample 0 has 3 und tokens, sample 1 has 2 und tokens
    causal_q_offsets = jnp.array([0, 3, 5], dtype=jnp.int32)
    mask = build_causal_understanding_mask(causal_q_offsets, 5)

    self.assertEqual(mask.shape, (5, 5))

    # Sample 0 (indices 0, 1, 2): lower-triangular causal
    for i in range(3):
      for j in range(3):
        expected = j <= i
        self.assertEqual(
            bool(mask[i, j]),
            expected,
            f"Failed at sample 0 position ({i}, {j})",
        )

    # Sample 1 (indices 3, 4): lower-triangular causal
    for i in range(3, 5):
      for j in range(3, 5):
        expected = j <= i
        self.assertEqual(
            bool(mask[i, j]),
            expected,
            f"Failed at sample 1 position ({i}, {j})",
        )

    # Cross-sample positions MUST BE False
    for i in range(3):
      for j in range(3, 5):
        self.assertFalse(bool(mask[i, j]), f"Cross-sample leak at ({i}, {j})")
        self.assertFalse(bool(mask[j, i]), f"Cross-sample leak at ({j}, {i})")

  def test_full_generative_mask(self):
    """Verifies Kernel 2 mask allows full attention within sample."""
    # Sample 0: 2 und, 3 gen -> 5 total (gen indices in full_q: 0, 1, 2;
    # kv indices: 0..4)
    # Sample 1: 1 und, 2 gen -> 3 total (gen indices in full_q: 3, 4;
    # kv indices: 5..7)
    full_q_offsets = jnp.array([0, 3, 5], dtype=jnp.int32)
    sample_kv_offsets = jnp.array([0, 5, 8], dtype=jnp.int32)

    mask = build_full_generative_mask(full_q_offsets, sample_kv_offsets, 5, 8)
    self.assertEqual(mask.shape, (5, 8))

    # Gen queries of sample 0 (0, 1, 2) must attend to all KV of sample 0 (0..4)
    for q in range(3):
      for kv in range(5):
        self.assertTrue(bool(mask[q, kv]), f"Expected True at ({q}, {kv})")
      for kv in range(5, 8):
        self.assertFalse(bool(mask[q, kv]), f"Expected False at ({q}, {kv})")

    # Gen queries of sample 1 (3, 4) must attend to all KV of sample 1 (5..7)
    for q in range(3, 5):
      for kv in range(5):
        self.assertFalse(bool(mask[q, kv]), f"Expected False at ({q}, {kv})")
      for kv in range(5, 8):
        self.assertTrue(bool(mask[q, kv]), f"Expected True at ({q}, {kv})")


class CosmosDualAttentionKernelTest(parameterized.TestCase):
  """Tests for the functional attention kernels."""

  def test_kernel_1_cross_sample_isolation(self):
    """Modifying sample 1 inputs must not alter sample 0 outputs."""
    und_lens = [3, 4]
    gen_lens = [2, 2]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    num_heads = 4
    head_dim = 16
    rng = jax.random.PRNGKey(123)

    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (meta.num_und_tokens, num_heads, head_dim))
    k = jax.random.normal(k2, (meta.num_und_tokens, num_heads, head_dim))
    v = jax.random.normal(k3, (meta.num_und_tokens, num_heads, head_dim))

    out_base = causal_understanding_attention(q, k, v, meta.causal_q_offsets)

    # Modify sample 1 (indices 3..6)
    q_perturbed = q.at[3:].add(10.0)
    k_perturbed = k.at[3:].add(5.0)
    v_perturbed = v.at[3:].add(-7.0)

    out_perturbed = causal_understanding_attention(q_perturbed, k_perturbed, v_perturbed, meta.causal_q_offsets)

    # Sample 0 (indices 0..2) must remain bitwise identical
    np.testing.assert_allclose(out_base[:3], out_perturbed[:3], rtol=1e-6, atol=1e-6)

  def test_kernel_1_causality(self):
    """Modifying future tokens must NOT affect past token outputs."""
    causal_q_offsets = jnp.array([0, 4], dtype=jnp.int32)
    num_heads = 2
    head_dim = 8
    rng = jax.random.PRNGKey(456)

    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (4, num_heads, head_dim))
    k = jax.random.normal(k2, (4, num_heads, head_dim))
    v = jax.random.normal(k3, (4, num_heads, head_dim))

    out_base = causal_understanding_attention(q, k, v, causal_q_offsets)

    # Modify token at index 3 (last token)
    q_mod = q.at[3].add(2.0)
    k_mod = k.at[3].add(3.0)
    v_mod = v.at[3].add(4.0)

    out_mod = causal_understanding_attention(q_mod, k_mod, v_mod, causal_q_offsets)

    # Tokens 0, 1, 2 must not be affected by token 3
    np.testing.assert_allclose(out_base[:3], out_mod[:3], rtol=1e-6, atol=1e-6)

  def test_kernel_2_cross_sample_isolation(self):
    """Modifying sample 1 inputs must not alter sample 0 outputs in Kernel 2."""
    und_lens = [2, 3]
    gen_lens = [3, 2]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    num_heads = 4
    head_dim = 16
    rng = jax.random.PRNGKey(789)

    k1, k2, k3 = jax.random.split(rng, 3)
    q_gen = jax.random.normal(k1, (meta.num_gen_tokens, num_heads, head_dim))
    k_all = jax.random.normal(k2, (meta.total_tokens, num_heads, head_dim))
    v_all = jax.random.normal(k3, (meta.total_tokens, num_heads, head_dim))

    out_base = full_generative_attention(q_gen, k_all, v_all, meta.full_q_offsets, meta.sample_kv_offsets)

    # Sample 0 has gen indices 0, 1, 2 and kv indices 0..4
    # Modify sample 1 (gen query indices 3, 4; kv indices 5..9)
    q_gen_perturbed = q_gen.at[3:].add(5.0)
    k_all_perturbed = k_all.at[5:].add(3.0)
    v_all_perturbed = v_all.at[5:].add(-4.0)

    out_perturbed = full_generative_attention(
        q_gen_perturbed,
        k_all_perturbed,
        v_all_perturbed,
        meta.full_q_offsets,
        meta.sample_kv_offsets,
    )

    # Sample 0's GEN queries (0, 1, 2) must remain unchanged
    np.testing.assert_allclose(out_base[:3], out_perturbed[:3], rtol=1e-6, atol=1e-6)


class CosmosMRoPETest(parameterized.TestCase):
  """Tests for 3D M-RoPE position encoding."""

  def test_3d_mrope_shape_and_norm(self):
    """Verifies 3D M-RoPE produces expected shapes and preserves L2 norm."""
    num_tokens = 10
    head_dim = 64
    positions_3d = jnp.zeros((num_tokens, 3), dtype=jnp.float32)
    # Assign varying (t, h, w) positions
    positions_3d = positions_3d.at[:, 0].set(jnp.arange(num_tokens))
    positions_3d = positions_3d.at[:, 1].set(jnp.arange(num_tokens) % 3)
    positions_3d = positions_3d.at[:, 2].set(jnp.arange(num_tokens) // 3)

    cos, sin = compute_3d_mrope_cos_sin(positions_3d, head_dim=head_dim, mrope_section=(12, 10, 10))
    self.assertEqual(cos.shape, (num_tokens, head_dim))
    self.assertEqual(sin.shape, (num_tokens, head_dim))

    # Test rotary rotation norm preservation: x and rotated(x) should have
    # same norm.
    rng = jax.random.PRNGKey(101)
    x = jax.random.normal(rng, (num_tokens, 4, head_dim))
    x_rot = apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1)
    self.assertEqual(x_rot.shape, x.shape)

    norm_x = jnp.linalg.norm(x, axis=-1)
    norm_x_rot = jnp.linalg.norm(x_rot, axis=-1)
    np.testing.assert_allclose(norm_x, norm_x_rot, rtol=1e-5, atol=1e-5)


class CosmosDualAttentionModuleTest(parameterized.TestCase):
  """Tests for the CosmosDualAttention NNX module."""

  def test_dual_attention_forward_packed_sequence(self):
    """Verifies that packed 2D tokens produce a re-interleaved packed tensor."""
    und_lens = [4, 2]
    gen_lens = [3, 5]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 64
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 16

    rngs = nnx.Rngs(42)
    dual_attn = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
        rngs=rngs,
    )

    rng = jax.random.PRNGKey(1)
    tokens = jax.random.normal(rng, (meta.total_tokens, dim))

    # Generate dummy cos/sin
    pos_3d = jnp.zeros((meta.total_tokens, 3), dtype=jnp.float32)
    cos, sin = compute_3d_mrope_cos_sin(pos_3d, head_dim=head_dim, mrope_section=(4, 2, 2))

    # Default forward returns re-interleaved packed tensor
    output_tokens = dual_attn(tokens, meta, cos=cos, sin=sin)
    assert isinstance(output_tokens, jax.Array)

    self.assertEqual(output_tokens.shape, (meta.total_tokens, dim))
    self.assertTrue(jnp.all(jnp.isfinite(output_tokens)))

  def test_dual_attention_forward_unpacked_tuple(self):
    """Verifies separate output shapes when an unpacked tuple is passed."""
    und_lens = [4, 2]
    gen_lens = [3, 5]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 64
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 16

    rngs = nnx.Rngs(42)
    dual_attn = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
        rngs=rngs,
    )

    rng = jax.random.PRNGKey(1)
    k_und, k_gen = jax.random.split(rng, 2)
    und_tokens = jax.random.normal(k_und, (meta.num_und_tokens, dim))
    gen_tokens = jax.random.normal(k_gen, (meta.num_gen_tokens, dim))

    pos_3d = jnp.zeros((meta.total_tokens, 3), dtype=jnp.float32)
    cos, sin = compute_3d_mrope_cos_sin(pos_3d, head_dim=head_dim, mrope_section=(4, 2, 2))

    und_out, gen_out = dual_attn((und_tokens, gen_tokens), meta, cos=cos, sin=sin, reinterleave=False)

    self.assertEqual(und_out.shape, (meta.num_und_tokens, dim))
    self.assertEqual(gen_out.shape, (meta.num_gen_tokens, dim))
    self.assertTrue(jnp.all(jnp.isfinite(und_out)))
    self.assertTrue(jnp.all(jnp.isfinite(gen_out)))

  def test_cross_sample_isolation_end_to_end(self):
    """Verifies modifying sample 1 tokens does NOT alter sample 0 output."""
    und_lens = [4, 3]
    gen_lens = [2, 4]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8

    rngs = nnx.Rngs(77)
    dual_attn = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        rngs=rngs,
    )

    rng = jax.random.PRNGKey(55)
    tokens = jax.random.normal(rng, (meta.total_tokens, dim))

    pos_3d = jnp.zeros((meta.total_tokens, 3), dtype=jnp.float32)
    cos, sin = compute_3d_mrope_cos_sin(pos_3d, head_dim=head_dim, mrope_section=(2, 1, 1))

    out_base = dual_attn(tokens, meta, cos=cos, sin=sin)
    assert isinstance(out_base, jax.Array)

    # Sample 0 occupies tokens 0..5 (4 und + 2 gen)
    # Sample 1 occupies tokens 6..12 (3 und + 4 gen)
    tokens_perturbed = tokens.at[6:].add(10.0)

    out_perturbed = dual_attn(tokens_perturbed, meta, cos=cos, sin=sin)
    assert isinstance(out_perturbed, jax.Array)

    # Sample 0 outputs (tokens 0..5) must not be affected by changes to
    # sample 1.
    np.testing.assert_allclose(out_base[:6], out_perturbed[:6], rtol=1e-5, atol=1e-5)

  def test_jit_compilation_and_gradients(self):
    """Verifies CosmosDualAttention compiles with JIT and differentiates."""
    und_lens = [2, 2]
    gen_lens = [2, 2]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 16
    head_dim = 8
    num_heads = 2
    num_kv_heads = 2

    rngs = nnx.Rngs(12)
    dual_attn = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rngs=rngs,
    )

    rng = jax.random.PRNGKey(88)
    tokens = jax.random.normal(rng, (meta.total_tokens, dim))

    def loss_fn(attn_module, x):
      out = attn_module(x, meta)
      assert isinstance(out, jax.Array)
      return jnp.sum(out**2)

    grad_fn = nnx.jit(nnx.grad(loss_fn, argnums=1))
    grad_tokens = grad_fn(dual_attn, tokens)

    self.assertEqual(grad_tokens.shape, tokens.shape)
    self.assertTrue(jnp.all(jnp.isfinite(grad_tokens)))

  def test_alias_and_gqa_qk_norm_variations(self):
    """Verifies CosmosAttention alias and asymmetric QK-norm configurations."""
    self.assertIs(CosmosAttention, CosmosDualAttention)

    und_lens = [3, 2]
    gen_lens = [4, 1]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    rngs = nnx.Rngs(33)
    # Nemotron-style: no QK norm for text, but QK norm for diffusion +
    # cross-attn UND K norm.
    dual_attn = CosmosAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
        rngs=rngs,
    )

    rng = jax.random.PRNGKey(3)
    tokens = jax.random.normal(rng, (meta.total_tokens, dim))
    out = dual_attn(tokens, meta)
    assert isinstance(out, jax.Array)

    self.assertEqual(out.shape, (meta.total_tokens, dim))
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_splash_attention_mask_builders(self):
    """Verifies Splash Attention masks for Kernel 1 and Kernel 2 build."""
    und_lens = [4, 2]
    gen_lens = [3, 5]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    # Splash Mask for Kernel 1
    splash_und_mask = build_causal_understanding_splash_mask(meta.causal_q_offsets, meta.num_und_tokens)
    self.assertEqual(splash_und_mask.shape, (meta.num_und_tokens, meta.num_und_tokens))
    dense_und_mask = build_causal_understanding_mask(meta.causal_q_offsets, meta.num_und_tokens)
    np.testing.assert_array_equal(
        np.array(splash_und_mask[slice(None), slice(None)]),
        np.array(dense_und_mask),
    )

    # Splash Mask for Kernel 2
    splash_gen_mask = build_full_generative_splash_mask(
        meta.full_q_offsets,
        meta.sample_kv_offsets,
        meta.num_gen_tokens,
        meta.total_tokens,
    )
    self.assertEqual(splash_gen_mask.shape, (meta.num_gen_tokens, meta.total_tokens))
    dense_gen_mask = build_full_generative_mask(
        meta.full_q_offsets,
        meta.sample_kv_offsets,
        meta.num_gen_tokens,
        meta.total_tokens,
    )
    np.testing.assert_array_equal(
        np.array(splash_gen_mask[slice(None), slice(None)]),
        np.array(dense_gen_mask),
    )

  def test_compile_cosmos_splash_mask(self):
    """Verifies compiling a 128-aligned mask into Splash Attention MaskInfo."""
    causal_q_offsets = jnp.array([0, 128], dtype=jnp.int32)
    dense_mask = build_causal_understanding_mask(causal_q_offsets, 128)
    mask_info = compile_cosmos_splash_mask(dense_mask, block_q=128, block_kv=128)
    self.assertIsNotNone(mask_info)
    self.assertIsNotNone(mask_info.block_mask)
    self.assertIsNotNone(mask_info.active_rows)
    self.assertIsNotNone(mask_info.active_cols)

    # Test error when not divisible by block size
    with self.assertRaises(ValueError):
      compile_cosmos_splash_mask(dense_mask[:100, :100], block_q=128, block_kv=128)

  def test_attention_kernel_dispatch_and_validation(self):
    """Verifies kernel parameter validation and dot_product execution."""
    rngs = nnx.Rngs(10)
    dual_attn = CosmosDualAttention(
        dim=32,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        attention_kernel="dot_product",
        rngs=rngs,
    )
    self.assertEqual(dual_attn.attention_kernel, "dot_product")

    # Invalid kernel name raises ValueError
    with self.assertRaises(ValueError):
      CosmosDualAttention(
          dim=32,
          num_heads=4,
          num_kv_heads=2,
          head_dim=8,
          attention_kernel="invalid_kernel",
          rngs=rngs,
      )

  def test_ragged_ops_parity(self):
    """Verifies unpack and reinterleave with use_ragged_ops=True match."""
    und_lens = [3, 2]
    gen_lens = [4, 1]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)
    tokens = jax.random.normal(jax.random.PRNGKey(42), (meta.total_tokens, 16))

    und_def, gen_def = unpack_streams(
        tokens,
        meta.packed_und_token_indexes,
        meta.packed_gen_token_indexes,
        use_ragged_ops=False,
    )
    und_rag, gen_rag = unpack_streams(
        tokens,
        meta.packed_und_token_indexes,
        meta.packed_gen_token_indexes,
        use_ragged_ops=True,
    )
    np.testing.assert_array_equal(np.array(und_def), np.array(und_rag))
    np.testing.assert_array_equal(np.array(gen_def), np.array(gen_rag))

    reint_def = reinterleave_streams(
        und_def,
        gen_def,
        meta.packed_und_token_indexes,
        meta.packed_gen_token_indexes,
        meta.total_tokens,
        use_ragged_ops=False,
    )
    reint_rag = reinterleave_streams(
        und_def,
        gen_def,
        meta.packed_und_token_indexes,
        meta.packed_gen_token_indexes,
        meta.total_tokens,
        use_ragged_ops=True,
    )
    np.testing.assert_array_equal(np.array(reint_def), np.array(reint_rag))


class CosmosDualAttentionGoldenParityTest(parameterized.TestCase):
  """Tests verifying exact numerical parity against golden outputs.

  Outputs were generated directly from the reference NVIDIA cosmos-framework
  repository (PackedAttentionMoT / two_way_attention) on an A100 GPU.
  """

  golden_data: Any = None

  def setUp(self):
    super().setUp()
    golden_file = _get_golden_data_path()
    if golden_file is None:
      self.skipTest(
          f"Golden test asset {_GOLDEN_DATA_FILENAME} not found locally under "
          "MAXTEXT_TEST_ASSETS_ROOT, and could not be fetched from GCS."
      )
    if self.golden_data is None:
      type(self).golden_data = np.load(golden_file)

  def _run_golden_parity_check(self, case_name: str, kernel: str = "dot_product"):
    """Verifies numerical parity of MaxText outputs against golden outputs."""
    assert self.golden_data is not None
    data = dict(self.golden_data)

    # Load parameters and metadata
    dim = int(data[f"{case_name}_dim"])
    num_heads = int(data[f"{case_name}_num_heads"])
    num_kv_heads = int(data[f"{case_name}_num_kv_heads"])
    head_dim = int(data[f"{case_name}_head_dim"])
    qk_norm_for_text = bool(data[f"{case_name}_qk_norm_for_text"])
    qk_norm_for_diffusion = bool(data[f"{case_name}_qk_norm_for_diffusion"])
    use_und_k_norm_for_gen = bool(data[f"{case_name}_use_und_k_norm_for_gen"])
    apply_rope = bool(data[f"{case_name}_apply_rope"])

    und_lens = data[f"{case_name}_und_lens"].tolist()
    gen_lens = data[f"{case_name}_gen_lens"].tolist()
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    tokens = jnp.array(data[f"{case_name}_tokens"])
    golden_out = data[f"{case_name}_golden_output"]

    cos_jax = jnp.array(data[f"{case_name}_cos"]) if apply_rope else None
    sin_jax = jnp.array(data[f"{case_name}_sin"]) if apply_rope else None

    # Instantiate JAX CosmosDualAttention
    rngs = nnx.Rngs(0)
    jax_module = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=qk_norm_for_text,
        qk_norm_for_diffusion=qk_norm_for_diffusion,
        use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        attention_kernel=kernel,
        rngs=rngs,
    )

    # Load PyTorch weights (transpose for Flax NNX linear kernel: (in, out))
    jax_module.q_proj.kernel.value = jnp.array(data[f"{case_name}_q_proj_weight"].T)
    jax_module.k_proj.kernel.value = jnp.array(data[f"{case_name}_k_proj_weight"].T)
    jax_module.v_proj.kernel.value = jnp.array(data[f"{case_name}_v_proj_weight"].T)
    jax_module.o_proj.kernel.value = jnp.array(data[f"{case_name}_o_proj_weight"].T)

    jax_module.q_proj_gen.kernel.value = jnp.array(data[f"{case_name}_q_proj_moe_gen_weight"].T)
    jax_module.k_proj_gen.kernel.value = jnp.array(data[f"{case_name}_k_proj_moe_gen_weight"].T)
    jax_module.v_proj_gen.kernel.value = jnp.array(data[f"{case_name}_v_proj_moe_gen_weight"].T)
    jax_module.o_proj_gen.kernel.value = jnp.array(data[f"{case_name}_o_proj_moe_gen_weight"].T)

    if qk_norm_for_text:
      assert jax_module.q_norm is not None
      assert jax_module.k_norm is not None
      q_scale = jax_module.q_norm.scale
      k_scale = jax_module.k_norm.scale
      assert q_scale is not None
      assert k_scale is not None
      q_scale.value = jnp.array(data[f"{case_name}_q_norm_weight"])
      k_scale.value = jnp.array(data[f"{case_name}_k_norm_weight"])

    if qk_norm_for_diffusion:
      assert jax_module.q_norm_gen is not None
      assert jax_module.k_norm_gen is not None
      q_scale_gen = jax_module.q_norm_gen.scale
      k_scale_gen = jax_module.k_norm_gen.scale
      assert q_scale_gen is not None
      assert k_scale_gen is not None
      q_scale_gen.value = jnp.array(data[f"{case_name}_q_norm_moe_gen_weight"])
      k_scale_gen.value = jnp.array(data[f"{case_name}_k_norm_moe_gen_weight"])

    if use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
      assert jax_module.k_norm_und_for_gen is not None
      k_scale_und_gen = jax_module.k_norm_und_for_gen.scale
      assert k_scale_und_gen is not None
      k_scale_und_gen.value = jnp.array(data[f"{case_name}_k_norm_und_for_gen_weight"])

    # Run forward pass
    jax_out = jax_module(
        tokens=tokens,
        metadata=meta,
        cos=cos_jax,
        sin=sin_jax,
    )
    assert isinstance(jax_out, jax.Array)

    jax_arr = np.array(jax_out)
    abs_diff = np.abs(jax_arr - golden_out)
    rel_diff = abs_diff / (np.abs(golden_out) + 1e-7)
    print(
        f"\n[Golden Parity: {case_name} | kernel={kernel}]"
        f" max_abs={np.max(abs_diff):.5e}, avg_abs={np.mean(abs_diff):.5e},"
        f" max_rel={np.max(rel_diff):.5e}, min_rel={np.min(rel_diff):.5e},"
        f" avg_rel={np.mean(rel_diff):.5e}"
    )

    # Assert exact numerical equality against golden reference output
    # (bfloat16 precision).
    np.testing.assert_allclose(
        jax_arr,
        golden_out,
        rtol=2e-2,
        atol=1.5e-2,
        err_msg=(
            "MaxText output differs from reference cosmos-framework golden" f" output in {case_name} (kernel={kernel})"
        ),
    )

  def test_case_standard_dual_qk_norm(self):
    self._run_golden_parity_check("case_standard_dual_qk_norm")

  def test_case_nemotron_asymmetric_gqa(self):
    self._run_golden_parity_check("case_nemotron_asymmetric_gqa")

  def test_case_standard_gqa(self):
    self._run_golden_parity_check("case_standard_gqa")

  def test_case_no_rope(self):
    self._run_golden_parity_check("case_no_rope")

  def test_case_block_aligned_standard_mha_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_standard_mha", kernel="dot_product")

  def test_case_block_aligned_standard_mha_flash(self):
    self._run_golden_parity_check("case_block_aligned_standard_mha", kernel="flash")

  def test_case_block_aligned_nemotron_gqa_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_nemotron_gqa", kernel="dot_product")

  def test_case_block_aligned_nemotron_gqa_flash(self):
    self._run_golden_parity_check("case_block_aligned_nemotron_gqa", kernel="flash")

  def test_case_block_aligned_standard_gqa_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_standard_gqa", kernel="dot_product")

  def test_case_block_aligned_standard_gqa_flash(self):
    self._run_golden_parity_check("case_block_aligned_standard_gqa", kernel="flash")

  def test_case_block_aligned_no_rope_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_no_rope", kernel="dot_product")

  def test_case_block_aligned_no_rope_flash(self):
    self._run_golden_parity_check("case_block_aligned_no_rope", kernel="flash")


if __name__ == "__main__":
  absltest.main()
