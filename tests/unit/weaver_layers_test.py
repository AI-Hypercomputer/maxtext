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

"""Unit and golden parity tests for Weaver layers.

Consolidates tests for attention, multi-layer perceptron (MLP), and
Mixture-of-Transformers decoder layer modules.
"""

import os
import subprocess
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.models import weaver
import numpy as np
import pytest

pytestmark = [
    pytest.mark.cpu_only,
]

# Module-level aliases for weaver attention and layer symbols.
WeaverConfig = weaver.WeaverConfig
WeaverMLP = weaver.WeaverMLP
WeaverMoTDecoderLayer = weaver.WeaverMoTDecoderLayer
WeaverAttention = weaver.WeaverAttention
WeaverDualAttention = weaver.WeaverDualAttention
apply_rotary_pos_emb = weaver.apply_rotary_pos_emb
build_causal_understanding_mask = weaver.build_causal_understanding_mask
build_causal_understanding_splash_mask = weaver.build_causal_understanding_splash_mask
build_weaver_packing_metadata = weaver.build_weaver_packing_metadata
build_full_generative_mask = weaver.build_full_generative_mask
build_full_generative_splash_mask = weaver.build_full_generative_splash_mask
causal_understanding_attention = weaver.causal_understanding_attention
compile_weaver_splash_mask = weaver.compile_weaver_splash_mask
compute_3d_mrope_cos_sin = weaver.compute_3d_mrope_cos_sin
full_generative_attention = weaver.full_generative_attention
reinterleave_streams = weaver.reinterleave_streams
unpack_streams = weaver.unpack_streams

_ATTENTION_GOLDEN_FILENAME = "weaver_attention_golden_data.npz"
_MOT_GOLDEN_FILENAME = "weaver_mot_golden_data.npz"

_TOLERANCES = {
    "float32": {"rtol": 1e-4, "atol": 1e-4},
    "bfloat16": {"rtol": 2e-2, "atol": 3.2e-2},
}

_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


def _get_golden_data_path(filename: str) -> str | None:
  """Attempts to download golden test asset from GCS bucket.

  Args:
    filename: Name of the golden npz file.

  Returns:
    Local file path if available, or None if unavailable or download failed.
  """
  tmp_path = f"/tmp/{filename}"
  if os.path.exists(tmp_path):
    return tmp_path

  gcs_uri = f"gs://maxtext-test-assets/{filename}"
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


class WeaverPackingLogicTest(parameterized.TestCase):
  """Tests for packing logic, metadata, unpacking, and re-interleaving."""

  def test_metadata_construction(self):
    """Verifies boundary metadata, offsets, and max lengths."""
    und_lens = [4, 6, 2]
    gen_lens = [8, 4, 10]
    metadata = build_weaver_packing_metadata(und_lens, gen_lens)

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
    metadata = build_weaver_packing_metadata(und_lens, gen_lens)

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


class WeaverMaskingTest(parameterized.TestCase):
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


class WeaverDualAttentionKernelTest(parameterized.TestCase):
  """Tests for the functional attention kernels."""

  def test_kernel_1_cross_sample_isolation(self):
    """Modifying sample 1 inputs must not alter sample 0 outputs."""
    und_lens = [3, 4]
    gen_lens = [2, 2]
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

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
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

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


class WeaverMRoPETest(parameterized.TestCase):
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


class WeaverDualAttentionModuleTest(parameterized.TestCase):
  """Tests for the WeaverDualAttention NNX module."""

  def test_dual_attention_forward_packed_sequence(self):
    """Verifies that packed 2D tokens produce a re-interleaved packed tensor."""
    und_lens = [4, 2]
    gen_lens = [3, 5]
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

    dim = 64
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 16

    rngs = nnx.Rngs(42)
    dual_attn = WeaverDualAttention(
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
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

    dim = 64
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 16

    rngs = nnx.Rngs(42)
    dual_attn = WeaverDualAttention(
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
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

    dim = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8

    rngs = nnx.Rngs(77)
    dual_attn = WeaverDualAttention(
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

  @pytest.mark.scheduled_only
  def test_jit_compilation_and_gradients(self):
    """Verifies WeaverDualAttention compiles with JIT and differentiates."""
    und_lens = [2, 2]
    gen_lens = [2, 2]
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

    dim = 16
    head_dim = 8
    num_heads = 2
    num_kv_heads = 2

    rngs = nnx.Rngs(12)
    dual_attn = WeaverDualAttention(
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
    """Verifies WeaverAttention alias and asymmetric QK-norm configurations."""
    self.assertIs(WeaverAttention, WeaverDualAttention)

    und_lens = [3, 2]
    gen_lens = [4, 1]
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

    dim = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    rngs = nnx.Rngs(33)
    # weaver-max style: no QK norm for text, but QK norm for diffusion +
    # cross-attn UND K norm.
    dual_attn = WeaverAttention(
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
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

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

  def test_compile_weaver_splash_mask(self):
    """Verifies compiling a 128-aligned mask into Splash Attention MaskInfo."""
    causal_q_offsets = jnp.array([0, 128], dtype=jnp.int32)
    dense_mask = build_causal_understanding_mask(causal_q_offsets, 128)
    mask_info = compile_weaver_splash_mask(dense_mask, block_q=128, block_kv=128)
    self.assertIsNotNone(mask_info)
    self.assertIsNotNone(mask_info.block_mask)
    self.assertIsNotNone(mask_info.active_rows)
    self.assertIsNotNone(mask_info.active_cols)

    # Test error when not divisible by block size
    with self.assertRaises(ValueError):
      compile_weaver_splash_mask(dense_mask[:100, :100], block_q=128, block_kv=128)

  def test_attention_kernel_dispatch_and_validation(self):
    """Verifies kernel parameter validation and dot_product execution."""
    rngs = nnx.Rngs(10)
    dual_attn = WeaverDualAttention(
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
      WeaverDualAttention(
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
    meta = build_weaver_packing_metadata(und_lens, gen_lens)
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


class WeaverMLPTest(parameterized.TestCase):
  """Tests for the per-tower feed-forward block."""

  @parameterized.named_parameters(("silu", "silu"), ("relu2", "relu2"))
  def test_output_shape(self, hidden_act):
    """Verifies the MLP preserves the hidden dimension."""
    mlp = WeaverMLP(
        hidden_size=32,
        intermediate_size=64,
        hidden_act=hidden_act,
        rngs=nnx.Rngs(0),
    )
    out = mlp(jax.random.normal(jax.random.PRNGKey(0), (5, 32)))
    self.assertEqual(out.shape, (5, 32))

  def test_relu2_has_no_gate_projection(self):
    """Verifies the squared-ReLU variant is ungated."""
    mlp = WeaverMLP(hidden_size=32, intermediate_size=64, hidden_act="relu2", rngs=nnx.Rngs(0))
    self.assertIsNone(mlp.gate_proj)

  def test_silu_has_gate_projection(self):
    """Verifies the SwiGLU variant is gated."""
    mlp = WeaverMLP(hidden_size=32, intermediate_size=64, hidden_act="silu", rngs=nnx.Rngs(0))
    self.assertIsNotNone(mlp.gate_proj)

  def test_rejects_unknown_activation(self):
    """Verifies unsupported activations are rejected at construction."""
    with self.assertRaises(ValueError):
      WeaverMLP(
          hidden_size=32,
          intermediate_size=64,
          hidden_act="gelu",
          rngs=nnx.Rngs(0),
      )


class WeaverMoTDecoderLayerStructureTest(parameterized.TestCase):
  """Tests for layer composition, aliasing, and shape contracts."""

  def _build(self, **overrides):
    """Builds a small test layer with optional overrides."""
    kwargs = {
        "hidden_size": 64,
        "head_dim": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "rngs": nnx.Rngs(0),
    }
    kwargs.update(overrides)
    return WeaverMoTDecoderLayer(**kwargs)

  def test_canonical_aliases_share_attention_weights(self):
    """Verifies the canonical names alias the attention submodule."""
    layer = self._build()
    self.assertIs(layer.to_q, layer.self_attn.q_proj)
    self.assertIs(layer.to_k, layer.self_attn.k_proj)
    self.assertIs(layer.to_v, layer.self_attn.v_proj)
    self.assertIs(layer.to_out, layer.self_attn.o_proj)
    self.assertIs(layer.norm_q, layer.self_attn.q_norm)
    self.assertIs(layer.norm_k, layer.self_attn.k_norm)
    self.assertIs(layer.add_q_proj, layer.self_attn.q_proj_gen)
    self.assertIs(layer.add_k_proj, layer.self_attn.k_proj_gen)
    self.assertIs(layer.add_v_proj, layer.self_attn.v_proj_gen)
    self.assertIs(layer.to_add_out, layer.self_attn.o_proj_gen)
    self.assertIs(layer.norm_added_q, layer.self_attn.q_norm_gen)
    self.assertIs(layer.norm_added_k, layer.self_attn.k_norm_gen)

  def test_both_towers_are_present(self):
    """Verifies the layer holds independent norms and MLPs per tower."""
    layer = self._build()
    self.assertIsNot(layer.input_layernorm, layer.input_layernorm_moe_gen)
    self.assertIsNot(layer.post_attention_layernorm, layer.post_attention_layernorm_moe_gen)
    self.assertIsNot(layer.mlp, layer.mlp_moe_gen)

  def test_custom_kernel_axes_propagation(self):
    """Verifies custom kernel_axes are forwarded to projections and norms."""
    layer = self._build(
        q_kernel_axes=("custom_embed", "custom_q"),
        kv_kernel_axes=("custom_embed", "custom_kv"),
        o_kernel_axes=("custom_o", "custom_embed"),
        qk_norm_kernel_axes=("custom_qk_norm",),
        mlp_up_kernel_axes=("custom_embed", "custom_mlp"),
        mlp_down_kernel_axes=("custom_mlp", "custom_embed"),
        norm_kernel_axes=("custom_norm",),
    )
    self.assertEqual(layer.to_q.kernel_axes, ("custom_embed", "custom_q"))
    self.assertEqual(layer.add_q_proj.kernel_axes, ("custom_embed", "custom_q"))
    self.assertEqual(layer.to_k.kernel_axes, ("custom_embed", "custom_kv"))
    self.assertEqual(layer.add_v_proj.kernel_axes, ("custom_embed", "custom_kv"))
    self.assertEqual(layer.to_out.kernel_axes, ("custom_o", "custom_embed"))
    self.assertEqual(layer.to_add_out.kernel_axes, ("custom_o", "custom_embed"))
    self.assertEqual(layer.norm_q.kernel_axes, ("custom_qk_norm",))
    self.assertEqual(layer.norm_added_k.kernel_axes, ("custom_qk_norm",))
    self.assertEqual(layer.mlp.up_proj.kernel_axes, ("custom_embed", "custom_mlp"))
    self.assertEqual(layer.mlp_moe_gen.down_proj.kernel_axes, ("custom_mlp", "custom_embed"))
    self.assertEqual(layer.input_layernorm.kernel_axes, ("custom_norm",))
    self.assertEqual(layer.post_attention_layernorm_moe_gen.kernel_axes, ("custom_norm",))

  def test_output_shapes(self):
    """Verifies both pathways preserve their input shapes."""
    layer = self._build()
    meta = build_weaver_packing_metadata([8], [12])
    und = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (12, 64))
    und_out, gen_out = layer(und, gen, meta)
    self.assertEqual(und_out.shape, (8, 64))
    self.assertEqual(gen_out.shape, (12, 64))

  def test_understanding_pathway_is_causal(self):
    """Verifies a late understanding token cannot affect an earlier one."""
    layer = self._build()
    meta = build_weaver_packing_metadata([8], [12])
    und = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (12, 64))

    und_out_base, _ = layer(und, gen, meta)
    und_perturbed = und.at[6].set(und[6] + 10.0)
    und_out_perturbed, _ = layer(und_perturbed, gen, meta)

    np.testing.assert_allclose(
        np.array(und_out_base[:6]),
        np.array(und_out_perturbed[:6]),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_generation_pathway_does_not_leak_into_understanding(self):
    """Verifies generation tokens never influence understanding outputs."""
    layer = self._build()
    meta = build_weaver_packing_metadata([8], [12])
    und = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (12, 64))

    und_out_base, _ = layer(und, gen, meta)
    und_out_mod, _ = layer(und, gen * 5.0, meta)

    np.testing.assert_allclose(np.array(und_out_base), np.array(und_out_mod), rtol=1e-5, atol=1e-5)

  def test_residual_paths_are_connected(self):
    """Verifies outputs retain the residual stream rather than replacing it."""
    layer = self._build()
    meta = build_weaver_packing_metadata([8], [12])
    und = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (12, 64))

    und_out, gen_out = layer(und, gen, meta)
    # With random init the block output is not the identity, but it must also
    # not be independent of the input: shifting the input must shift the output.
    und_out_shifted, gen_out_shifted = layer(und + 1.0, gen + 1.0, meta)
    self.assertGreater(float(jnp.mean(jnp.abs(und_out_shifted - und_out))), 1e-3)
    self.assertGreater(float(jnp.mean(jnp.abs(gen_out_shifted - gen_out))), 1e-3)

  @pytest.mark.scheduled_only
  def test_jit_compilation(self):
    """Verifies the layer is jit-compatible via nnx.split/merge."""
    layer = self._build()
    meta = build_weaver_packing_metadata([8], [12])
    graphdef, state = nnx.split(layer)

    @jax.jit
    def forward(state, und, gen):
      merged = nnx.merge(graphdef, state)
      return merged(und, gen, meta)

    und = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (12, 64))
    und_out, gen_out = forward(state, und, gen)
    self.assertEqual(und_out.shape, (8, 64))
    self.assertEqual(gen_out.shape, (12, 64))

  @pytest.mark.scheduled_only
  def test_gradients_flow_to_both_towers(self):
    """Verifies both towers receive gradient signal."""
    layer = self._build()
    meta = build_weaver_packing_metadata([8], [12])
    und = jax.random.normal(jax.random.PRNGKey(0), (8, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (12, 64))

    def loss_fn(model):
      und_out, gen_out = model(und, gen, meta)
      return jnp.sum(und_out**2) + jnp.sum(gen_out**2)

    grads = nnx.grad(loss_fn)(layer)
    flat = jax.tree_util.tree_leaves(grads)
    self.assertNotEmpty(flat)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in flat))
    self.assertGreater(sum(float(jnp.sum(jnp.abs(g))) for g in flat), 0.0)

  def test_pure_nnx_leaf_layers(self):
    """Verifies all leaf projections and norms are pure flax.nnx built-ins."""
    layer = self._build()
    for proj in (
        layer.to_q,
        layer.to_k,
        layer.to_v,
        layer.to_out,
        layer.add_q_proj,
        layer.add_k_proj,
        layer.add_v_proj,
        layer.to_add_out,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
        layer.mlp_moe_gen.gate_proj,
        layer.mlp_moe_gen.up_proj,
        layer.mlp_moe_gen.down_proj,
    ):
      self.assertIsInstance(proj, nnx.Linear)

    for norm in (
        layer.norm_q,
        layer.norm_k,
        layer.norm_added_q,
        layer.norm_added_k,
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.input_layernorm_moe_gen,
        layer.post_attention_layernorm_moe_gen,
    ):
      self.assertIsInstance(norm, nnx.RMSNorm)

  def test_config_construction(self):
    """Verifies config-driven construction from WeaverConfig."""
    cfg = WeaverConfig(
        hidden_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    layer = WeaverMoTDecoderLayer.from_config(cfg, rngs=nnx.Rngs(0))
    meta = build_weaver_packing_metadata([4], [6])
    und = jax.random.normal(jax.random.PRNGKey(0), (4, 64))
    gen = jax.random.normal(jax.random.PRNGKey(1), (6, 64))
    und_out, gen_out = layer(und, gen, meta)
    self.assertEqual(und_out.shape, (4, 64))
    self.assertEqual(gen_out.shape, (6, 64))


@pytest.mark.scheduled_only
class WeaverDualAttentionGoldenParityTest(parameterized.TestCase):
  """Tests verifying exact numerical parity against golden outputs."""

  golden_data: Any = None

  def setUp(self):
    super().setUp()
    golden_file = _get_golden_data_path(_ATTENTION_GOLDEN_FILENAME)
    if golden_file is None:
      self.skipTest(
          f"Golden test asset {_ATTENTION_GOLDEN_FILENAME} not found locally under "
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
    meta = build_weaver_packing_metadata(und_lens, gen_lens)

    tokens = jnp.array(data[f"{case_name}_tokens"])
    golden_out = data[f"{case_name}_golden_output"]

    cos_jax = jnp.array(data[f"{case_name}_cos"]) if apply_rope else None
    sin_jax = jnp.array(data[f"{case_name}_sin"]) if apply_rope else None

    # Instantiate JAX WeaverDualAttention
    rngs = nnx.Rngs(0)
    jax_module = WeaverDualAttention(
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
        err_msg=f"MaxText output differs from reference golden output in {case_name} (kernel={kernel})",
    )

  def test_case_standard_dual_qk_norm(self):
    self._run_golden_parity_check("case_standard_dual_qk_norm")

  def test_case_weaver_max_asymmetric_gqa(self):
    self._run_golden_parity_check("case_weaver_max_asymmetric_gqa")

  def test_case_standard_gqa(self):
    self._run_golden_parity_check("case_standard_gqa")

  def test_case_no_rope(self):
    self._run_golden_parity_check("case_no_rope")

  def test_case_block_aligned_standard_mha_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_standard_mha", kernel="dot_product")

  def test_case_block_aligned_standard_mha_flash(self):
    self._run_golden_parity_check("case_block_aligned_standard_mha", kernel="flash")

  def test_case_block_aligned_weaver_max_gqa_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_weaver_max_gqa", kernel="dot_product")

  def test_case_block_aligned_weaver_max_gqa_flash(self):
    self._run_golden_parity_check("case_block_aligned_weaver_max_gqa", kernel="flash")

  def test_case_block_aligned_standard_gqa_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_standard_gqa", kernel="dot_product")

  def test_case_block_aligned_standard_gqa_flash(self):
    self._run_golden_parity_check("case_block_aligned_standard_gqa", kernel="flash")

  def test_case_block_aligned_no_rope_dot_product(self):
    self._run_golden_parity_check("case_block_aligned_no_rope", kernel="dot_product")

  def test_case_block_aligned_no_rope_flash(self):
    self._run_golden_parity_check("case_block_aligned_no_rope", kernel="flash")


@pytest.mark.scheduled_only
class WeaverMoTDecoderLayerGoldenParityTest(parameterized.TestCase):
  """Numerical parity against the reference Mixture-of-Transformers decoder layer."""

  golden_data = None
  golden_path = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.golden_path = _get_golden_data_path(_MOT_GOLDEN_FILENAME)
    if cls.golden_path is not None:
      cls.golden_data = np.load(cls.golden_path)

  def setUp(self):
    super().setUp()
    if self.golden_data is None:
      self.skipTest(
          f"Golden test asset {_MOT_GOLDEN_FILENAME} not found locally under /tmp and could not be downloaded "
          "from gs://maxtext-test-assets/."
      )

  def _load_weights(self, layer, data, case_name, dtype):
    """Copies reference parameters into the MaxText layer.

    PyTorch ``nn.Linear`` stores ``(out_features, in_features)`` while the Flax
    ``nnx.Linear`` kernel is ``(in_features, out_features)``, so every
    projection weight is transposed on the way in.
    """

    def param(name):
      return jnp.array(data[f"{case_name}_param_{name}"], dtype=dtype)

    def set_kernel(module, name):
      module.kernel.value = param(name).T

    def set_scale(module, name):
      scale = module.scale
      assert scale is not None
      scale.value = param(name)

    # Attention projections, addressed through the canonical aliases.
    set_kernel(layer.to_q, "self_attn.to_q.weight")
    set_kernel(layer.to_k, "self_attn.to_k.weight")
    set_kernel(layer.to_v, "self_attn.to_v.weight")
    set_kernel(layer.to_out, "self_attn.to_out.weight")
    set_kernel(layer.add_q_proj, "self_attn.add_q_proj.weight")
    set_kernel(layer.add_k_proj, "self_attn.add_k_proj.weight")
    set_kernel(layer.add_v_proj, "self_attn.add_v_proj.weight")
    set_kernel(layer.to_add_out, "self_attn.to_add_out.weight")

    # QK normalizations.
    if layer.norm_q is not None:
      set_scale(layer.norm_q, "self_attn.norm_q.weight")
      set_scale(layer.norm_k, "self_attn.norm_k.weight")
    set_scale(layer.norm_added_q, "self_attn.norm_added_q.weight")
    set_scale(layer.norm_added_k, "self_attn.norm_added_k.weight")
    if layer.k_norm_und_for_gen is not None:
      set_scale(layer.k_norm_und_for_gen, "self_attn.k_norm_und_for_gen.weight")

    # Block normalizations.
    set_scale(layer.input_layernorm, "input_layernorm.weight")
    set_scale(layer.input_layernorm_moe_gen, "input_layernorm_moe_gen.weight")
    set_scale(layer.post_attention_layernorm, "post_attention_layernorm.weight")
    set_scale(
        layer.post_attention_layernorm_moe_gen,
        "post_attention_layernorm_moe_gen.weight",
    )

    # Feed-forward networks.
    for mlp, prefix in ((layer.mlp, "mlp"), (layer.mlp_moe_gen, "mlp_moe_gen")):
      if mlp.gate_proj is not None:
        set_kernel(mlp.gate_proj, f"{prefix}.gate_proj.weight")
      set_kernel(mlp.up_proj, f"{prefix}.up_proj.weight")
      set_kernel(mlp.down_proj, f"{prefix}.down_proj.weight")

  def _run_parity_check(self, case_name: str, precision: str):
    """Compares MaxText outputs against the reference golden activations."""
    assert self.golden_data is not None
    data = dict(self.golden_data)
    dtype = _DTYPES[precision]

    hidden_size = int(data[f"{case_name}_hidden_size"])
    head_dim = int(data[f"{case_name}_head_dim"])
    num_attention_heads = int(data[f"{case_name}_num_attention_heads"])
    num_key_value_heads = int(data[f"{case_name}_num_key_value_heads"])
    intermediate_size = int(data[f"{case_name}_intermediate_size"])
    rms_norm_eps = float(data[f"{case_name}_rms_norm_eps"])
    hidden_act = str(data[f"{case_name}_hidden_act"])
    attention_bias = bool(data[f"{case_name}_attention_bias"])
    qk_norm_for_text = bool(data[f"{case_name}_qk_norm_for_text"])
    use_und_k_norm_for_gen = bool(data[f"{case_name}_use_und_k_norm_for_gen"])
    und_len = int(data[f"{case_name}_und_len"])
    gen_len = int(data[f"{case_name}_gen_len"])

    layer = WeaverMoTDecoderLayer(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        attention_bias=attention_bias,
        rms_norm_eps=rms_norm_eps,
        hidden_act=hidden_act,
        qk_norm_for_text=qk_norm_for_text,
        # The reference always normalizes the generation pathway.
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        dtype=dtype,
        weight_dtype=dtype,
        rngs=nnx.Rngs(0),
    )
    self._load_weights(layer, data, case_name, dtype)

    # Single packed sample: understanding tokens followed by generation tokens.
    meta = build_weaver_packing_metadata([und_len], [gen_len])

    # Scatter the per-pathway rotary tables into global packed order.
    cos = jnp.zeros((meta.total_tokens, head_dim), dtype=dtype)
    sin = jnp.zeros((meta.total_tokens, head_dim), dtype=dtype)
    cos = cos.at[meta.packed_und_token_indexes].set(jnp.array(data[f"{case_name}_cos_und"], dtype=dtype))
    sin = sin.at[meta.packed_und_token_indexes].set(jnp.array(data[f"{case_name}_sin_und"], dtype=dtype))
    cos = cos.at[meta.packed_gen_token_indexes].set(jnp.array(data[f"{case_name}_cos_gen"], dtype=dtype))
    sin = sin.at[meta.packed_gen_token_indexes].set(jnp.array(data[f"{case_name}_sin_gen"], dtype=dtype))

    und_seq = jnp.array(data[f"{case_name}_und_seq"], dtype=dtype)
    gen_seq = jnp.array(data[f"{case_name}_gen_seq"], dtype=dtype)

    und_out, gen_out = layer(und_seq, gen_seq, meta, cos, sin)

    golden_und = data[f"{case_name}_und_out_{precision}"]
    golden_gen = data[f"{case_name}_gen_out_{precision}"]
    und_arr = np.array(und_out, dtype=np.float32)
    gen_arr = np.array(gen_out, dtype=np.float32)

    print(
        f"\n[WeaverMoT Parity: {case_name} | {precision}]"
        f" und_max_abs={np.max(np.abs(und_arr - golden_und)):.3e},"
        f" und_avg_abs={np.mean(np.abs(und_arr - golden_und)):.3e},"
        f" gen_max_abs={np.max(np.abs(gen_arr - golden_gen)):.3e},"
        f" gen_avg_abs={np.mean(np.abs(gen_arr - golden_gen)):.3e}"
    )

    tol = _TOLERANCES[precision]
    np.testing.assert_allclose(
        und_arr,
        golden_und,
        err_msg=f"und_out differs from reference in {case_name} ({precision})",
        **tol,
    )
    np.testing.assert_allclose(
        gen_arr,
        golden_gen,
        err_msg=f"gen_out differs from reference in {case_name} ({precision})",
        **tol,
    )

  @parameterized.named_parameters(
      ("mini_silu_mha_float32", "mini_silu_mha", "float32"),
      ("mini_silu_mha_bfloat16", "mini_silu_mha", "bfloat16"),
      ("mini_silu_gqa_float32", "mini_silu_gqa", "float32"),
      ("mini_silu_gqa_bfloat16", "mini_silu_gqa", "bfloat16"),
      ("mini_relu2_asym_float32", "mini_relu2_asym", "float32"),
      ("mini_relu2_asym_bfloat16", "mini_relu2_asym", "bfloat16"),
  )
  def test_golden_parity(self, case_name, precision):
    """Checks numerical parity against the golden reference."""
    self._run_parity_check(case_name, precision)


if __name__ == "__main__":
  absltest.main()
