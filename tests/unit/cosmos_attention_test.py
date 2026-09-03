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

# pylint: disable=too-many-positional-arguments
from collections.abc import Sequence
import math
from absl.testing import absltest
from absl.testing import parameterized

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

try:
  import torch
  from torch import nn

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False
  torch = None
  nn = None

from maxtext.layers.cosmos_attention import (
    apply_rotary_pos_emb,
    build_causal_understanding_mask,
    build_cosmos_packing_metadata,
    build_full_generative_mask,
    causal_understanding_attention,
    compute_3d_mrope_cos_sin,
    CosmosAttention,
    CosmosDualAttention,
    full_generative_attention,
    reinterleave_streams,
    unpack_streams,
)


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
    """Verifies unpack_streams and reinterleave_streams form an exact roundtrip."""
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
    """Verifies Kernel 2 mask allows full attention within sample, blocks cross-sample."""
    # Sample 0: 2 und, 3 gen -> 5 total (gen indices in full_q: 0, 1, 2; kv indices: 0..4)
    # Sample 1: 1 und, 2 gen -> 3 total (gen indices in full_q: 3, 4; kv indices: 5..7)
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
    """Modifying sample 1's inputs must strictly NOT alter sample 0's outputs."""
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
    """Modifying future tokens must NOT affect past token outputs within sample."""
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
    """Modifying sample 1's inputs must strictly NOT alter sample 0's outputs in Kernel 2."""
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
    """Verifies 3D M-RoPE produces expected shapes and preserves head L2 norm."""
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

    # Test rotary rotation norm preservation: x and rotated(x) should have same norm
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
    """Verifies projections, dual kernels, and separate output shapes when tuple is passed."""
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
    """Verifies that modifying sample 1's tokens does NOT alter sample 0's output in the attention block."""
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

    # Sample 0's outputs (tokens 0..5) must not be affected by changes to sample 1
    np.testing.assert_allclose(out_base[:6], out_perturbed[:6], rtol=1e-5, atol=1e-5)

  def test_jit_compilation_and_gradients(self):
    """Verifies that CosmosDualAttention compiles with JIT and is fully differentiable."""
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
    # Nemotron-style: no QK norm for text, but QK norm for diffusion + cross-attn UND K norm
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


# -----------------------------------------------------------------------------
# PyTorch Reference Implementation of Cosmos 3 PackedAttentionMoT
# Reference: NVIDIA cosmos-framework
# (cosmos_framework/model/generator/mot/unified_mot.py and attention.py)
# -----------------------------------------------------------------------------

if HAS_TORCH:

  class RMSNormPT(nn.Module):
    """RMS normalization in PyTorch matching Qwen3VLTextRMSNorm / MaxText RMSNorm."""

    def __init__(self, num_features: int, eps: float = 1e-6):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(num_features))
      self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      variance = x.pow(2).mean(-1, keepdim=True)
      return x * torch.rsqrt(variance + self.eps) * self.weight

  def rotate_half_PT(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)

  def apply_rotary_pos_emb_PT(
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      unsqueeze_dim: int = 1,
  ) -> torch.Tensor:
    """PyTorch implementation of rotary position embedding (3D M-RoPE)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half_PT(x) * sin)

  def causal_understanding_attention_PT(
      q_und: torch.Tensor,
      k_und: torch.Tensor,
      v_und: torch.Tensor,
      causal_q_offsets: Sequence[int],
      scale: float | None = None,
  ) -> torch.Tensor:
    """PyTorch reference Kernel 1: causal self-attention per sample on UND tokens."""
    n_und, num_heads, head_dim = q_und.shape
    if n_und == 0:
      return torch.zeros_like(q_und)

    num_kv_heads = k_und.shape[1]
    if num_heads != num_kv_heads:
      repeat_factor = num_heads // num_kv_heads
      k_und = k_und.repeat_interleave(repeat_factor, dim=1)
      v_und = v_und.repeat_interleave(repeat_factor, dim=1)

    if scale is None:
      scale = 1.0 / math.sqrt(head_dim)

    # [N_und, num_heads, head_dim] -> [num_heads, N_und, head_dim]
    q_t = q_und.transpose(0, 1)
    k_t = k_und.transpose(0, 1)
    v_t = v_und.transpose(0, 1)

    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale  # [num_heads, N_und, N_und]

    mask = torch.zeros((n_und, n_und), dtype=torch.bool, device=q_und.device)
    for i in range(len(causal_q_offsets) - 1):
      start = causal_q_offsets[i]
      end = causal_q_offsets[i + 1]
      sample_len = end - start
      if sample_len > 0:
        mask[start:end, start:end] = torch.tril(
            torch.ones((sample_len, sample_len), dtype=torch.bool, device=q_und.device)
        )

    scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
    attn_weights = torch.softmax(scores.to(torch.float32), dim=-1).to(q_und.dtype)
    out = torch.matmul(attn_weights, v_t)  # [num_heads, N_und, head_dim]
    return out.transpose(0, 1)  # [N_und, num_heads, head_dim]

  def full_generative_attention_PT(
      q_gen: torch.Tensor,
      k_all: torch.Tensor,
      v_all: torch.Tensor,
      full_q_offsets: Sequence[int],
      sample_kv_offsets: Sequence[int],
      scale: float | None = None,
  ) -> torch.Tensor:
    """PyTorch reference Kernel 2: full cross+self attention per sample on GEN tokens."""
    n_gen, num_heads, head_dim = q_gen.shape
    total_tokens = k_all.shape[0]
    if n_gen == 0 or total_tokens == 0:
      return torch.zeros_like(q_gen)

    num_kv_heads = k_all.shape[1]
    if num_heads != num_kv_heads:
      repeat_factor = num_heads // num_kv_heads
      k_all = k_all.repeat_interleave(repeat_factor, dim=1)
      v_all = v_all.repeat_interleave(repeat_factor, dim=1)

    if scale is None:
      scale = 1.0 / math.sqrt(head_dim)

    q_t = q_gen.transpose(0, 1)
    k_t = k_all.transpose(0, 1)
    v_t = v_all.transpose(0, 1)

    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale  # [num_heads, N_gen, N_total]

    mask = torch.zeros((n_gen, total_tokens), dtype=torch.bool, device=q_gen.device)
    for i in range(len(full_q_offsets) - 1):
      q_start = full_q_offsets[i]
      q_end = full_q_offsets[i + 1]
      kv_start = sample_kv_offsets[i]
      kv_end = sample_kv_offsets[i + 1]
      if (q_end > q_start) and (kv_end > kv_start):
        mask[q_start:q_end, kv_start:kv_end] = True

    scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
    attn_weights = torch.softmax(scores.to(torch.float32), dim=-1).to(q_gen.dtype)
    out = torch.matmul(attn_weights, v_t)  # [num_heads, N_gen, head_dim]
    return out.transpose(0, 1)  # [N_gen, num_heads, head_dim]

  class CosmosDualAttentionReferencePT(nn.Module):
    """Exact PyTorch reference implementation of Cosmos 3 PackedAttentionMoT."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        use_und_k_norm_for_gen: bool = False,
        use_bias: bool = False,
        eps: float = 1e-6,
    ):
      super().__init__()
      self.dim = dim
      self.num_heads = num_heads
      self.num_kv_heads = num_kv_heads
      self.head_dim = head_dim
      self.qk_norm_for_text = qk_norm_for_text
      self.qk_norm_for_diffusion = qk_norm_for_diffusion
      self.use_und_k_norm_for_gen = use_und_k_norm_for_gen
      self.scaling = 1.0 / math.sqrt(head_dim)

      # Understanding projections
      self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=use_bias)
      self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
      self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
      self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=use_bias)

      # Understanding QK norm
      if qk_norm_for_text:
        self.q_norm = RMSNormPT(head_dim, eps=eps)
        self.k_norm = RMSNormPT(head_dim, eps=eps)
      else:
        self.q_norm = None
        self.k_norm = None

      # Generation projections
      self.q_proj_gen = nn.Linear(dim, num_heads * head_dim, bias=use_bias)
      self.k_proj_gen = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
      self.v_proj_gen = nn.Linear(dim, num_kv_heads * head_dim, bias=use_bias)
      self.o_proj_gen = nn.Linear(num_heads * head_dim, dim, bias=use_bias)

      # Generation QK norm
      if qk_norm_for_diffusion:
        self.q_norm_gen = RMSNormPT(head_dim, eps=eps)
        self.k_norm_gen = RMSNormPT(head_dim, eps=eps)
      else:
        self.q_norm_gen = None
        self.k_norm_gen = None

      # Cross-attention K norm
      if use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
        self.k_norm_und_for_gen = RMSNormPT(head_dim, eps=eps)
      else:
        self.k_norm_und_for_gen = None

    def forward(
        self,
        tokens: torch.Tensor,
        causal_q_offsets: Sequence[int],
        full_q_offsets: Sequence[int],
        sample_kv_offsets: Sequence[int],
        packed_und_token_indexes: Sequence[int],
        packed_gen_token_indexes: Sequence[int],
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
      """Executes PyTorch reference forward pass for Cosmos 3 PackedAttentionMoT."""
      n_total = tokens.shape[0]
      und_idx = torch.tensor(packed_und_token_indexes, dtype=torch.long)
      gen_idx = torch.tensor(packed_gen_token_indexes, dtype=torch.long)

      und_tokens = tokens[und_idx] if len(packed_und_token_indexes) > 0 else tokens.new_zeros((0, self.dim))
      gen_tokens = tokens[gen_idx] if len(packed_gen_token_indexes) > 0 else tokens.new_zeros((0, self.dim))

      n_und = und_tokens.shape[0]
      n_gen = gen_tokens.shape[0]

      # 1. Projections
      if n_und > 0:
        q_und = self.q_proj(und_tokens).view(n_und, self.num_heads, self.head_dim)
        k_und = self.k_proj(und_tokens).view(n_und, self.num_kv_heads, self.head_dim)
        v_und = self.v_proj(und_tokens).view(n_und, self.num_kv_heads, self.head_dim)
      else:
        q_und = tokens.new_zeros((0, self.num_heads, self.head_dim))
        k_und = tokens.new_zeros((0, self.num_kv_heads, self.head_dim))
        v_und = tokens.new_zeros((0, self.num_kv_heads, self.head_dim))

      if n_gen > 0:
        q_gen = self.q_proj_gen(gen_tokens).view(n_gen, self.num_heads, self.head_dim)
        k_gen = self.k_proj_gen(gen_tokens).view(n_gen, self.num_kv_heads, self.head_dim)
        v_gen = self.v_proj_gen(gen_tokens).view(n_gen, self.num_kv_heads, self.head_dim)
      else:
        q_gen = tokens.new_zeros((0, self.num_heads, self.head_dim))
        k_gen = tokens.new_zeros((0, self.num_kv_heads, self.head_dim))
        v_gen = tokens.new_zeros((0, self.num_kv_heads, self.head_dim))

      # 2. QK Norm
      if self.q_norm is not None and n_und > 0:
        q_und = self.q_norm(q_und)
      if self.k_norm is not None and n_und > 0:
        k_und = self.k_norm(k_und)

      if self.q_norm_gen is not None and n_gen > 0:
        q_gen = self.q_norm_gen(q_gen)
      if self.k_norm_gen is not None and n_gen > 0:
        k_gen = self.k_norm_gen(k_gen)

      # 3. 3D M-RoPE
      k_und_pre_rope = k_und
      cos_und = None
      sin_und = None
      if cos is not None and sin is not None:
        if n_und > 0:
          cos_und = cos[und_idx]
          sin_und = sin[und_idx]
          q_und = apply_rotary_pos_emb_PT(q_und, cos_und, sin_und, unsqueeze_dim=1)
          k_und = apply_rotary_pos_emb_PT(k_und, cos_und, sin_und, unsqueeze_dim=1)
        if n_gen > 0:
          cos_gen = cos[gen_idx]
          sin_gen = sin[gen_idx]
          q_gen = apply_rotary_pos_emb_PT(q_gen, cos_gen, sin_gen, unsqueeze_dim=1)
          k_gen = apply_rotary_pos_emb_PT(k_gen, cos_gen, sin_gen, unsqueeze_dim=1)

      # 4. Cross-attention UND K Norm
      if self.k_norm_und_for_gen is not None and n_und > 0:
        k_und_norm_for_gen = self.k_norm_und_for_gen(k_und_pre_rope)
        if cos_und is not None and sin_und is not None:
          k_und_for_gen = apply_rotary_pos_emb_PT(k_und_norm_for_gen, cos_und, sin_und, unsqueeze_dim=1)
        else:
          k_und_for_gen = k_und_norm_for_gen
      else:
        k_und_for_gen = k_und

      # 5. Assemble K_all and V_all
      k_all = tokens.new_zeros((n_total, self.num_kv_heads, self.head_dim))
      v_all = tokens.new_zeros((n_total, self.num_kv_heads, self.head_dim))
      if n_und > 0:
        k_all[und_idx] = k_und_for_gen
        v_all[und_idx] = v_und
      if n_gen > 0:
        k_all[gen_idx] = k_gen
        v_all[gen_idx] = v_gen

      # 6. Kernel 1
      if n_und > 0:
        und_attn = causal_understanding_attention_PT(q_und, k_und, v_und, causal_q_offsets, scale=self.scaling)
        und_attn_flat = und_attn.reshape(n_und, self.num_heads * self.head_dim)
        und_out = self.o_proj(und_attn_flat)
      else:
        und_out = tokens.new_zeros((0, self.dim))

      # 7. Kernel 2
      if n_gen > 0:
        gen_attn = full_generative_attention_PT(
            q_gen, k_all, v_all, full_q_offsets, sample_kv_offsets, scale=self.scaling
        )
        gen_attn_flat = gen_attn.reshape(n_gen, self.num_heads * self.head_dim)
        gen_out = self.o_proj_gen(gen_attn_flat)
      else:
        gen_out = tokens.new_zeros((0, self.dim))

      # 8. Reinterleave
      out = tokens.new_zeros((n_total, self.dim))
      if n_und > 0:
        out[und_idx] = und_out
      if n_gen > 0:
        out[gen_idx] = gen_out
      return out

  def copy_pytorch_weights_to_jax(
      pt_module: CosmosDualAttentionReferencePT,
      jax_module: CosmosDualAttention,
  ):
    """Copies weights from PyTorch reference module to Flax NNX CosmosDualAttention."""
    # Linear projections: weights are transposed (out_features, in_features) -> (in_features, out_features)
    assert isinstance(jax_module.q_proj.kernel, nnx.Param)
    assert isinstance(jax_module.k_proj.kernel, nnx.Param)
    assert isinstance(jax_module.v_proj.kernel, nnx.Param)
    assert isinstance(jax_module.o_proj.kernel, nnx.Param)
    assert isinstance(jax_module.q_proj_gen.kernel, nnx.Param)
    assert isinstance(jax_module.k_proj_gen.kernel, nnx.Param)
    assert isinstance(jax_module.v_proj_gen.kernel, nnx.Param)
    assert isinstance(jax_module.o_proj_gen.kernel, nnx.Param)

    jax_module.q_proj.kernel[...] = jnp.array(pt_module.q_proj.weight.detach().numpy().T)
    jax_module.k_proj.kernel[...] = jnp.array(pt_module.k_proj.weight.detach().numpy().T)
    jax_module.v_proj.kernel[...] = jnp.array(pt_module.v_proj.weight.detach().numpy().T)
    jax_module.o_proj.kernel[...] = jnp.array(pt_module.o_proj.weight.detach().numpy().T)

    jax_module.q_proj_gen.kernel[...] = jnp.array(pt_module.q_proj_gen.weight.detach().numpy().T)
    jax_module.k_proj_gen.kernel[...] = jnp.array(pt_module.k_proj_gen.weight.detach().numpy().T)
    jax_module.v_proj_gen.kernel[...] = jnp.array(pt_module.v_proj_gen.weight.detach().numpy().T)
    jax_module.o_proj_gen.kernel[...] = jnp.array(pt_module.o_proj_gen.weight.detach().numpy().T)

    # QK Norms
    if (
        jax_module.q_norm is not None
        and pt_module.q_norm is not None
        and jax_module.k_norm is not None
        and pt_module.k_norm is not None
    ):
      assert isinstance(jax_module.q_norm.scale, nnx.Param)
      assert isinstance(jax_module.k_norm.scale, nnx.Param)
      jax_module.q_norm.scale[...] = jnp.array(pt_module.q_norm.weight.detach().numpy())
      jax_module.k_norm.scale[...] = jnp.array(pt_module.k_norm.weight.detach().numpy())

    if (
        jax_module.q_norm_gen is not None
        and pt_module.q_norm_gen is not None
        and jax_module.k_norm_gen is not None
        and pt_module.k_norm_gen is not None
    ):
      assert isinstance(jax_module.q_norm_gen.scale, nnx.Param)
      assert isinstance(jax_module.k_norm_gen.scale, nnx.Param)
      jax_module.q_norm_gen.scale[...] = jnp.array(pt_module.q_norm_gen.weight.detach().numpy())
      jax_module.k_norm_gen.scale[...] = jnp.array(pt_module.k_norm_gen.weight.detach().numpy())

    if jax_module.k_norm_und_for_gen is not None and pt_module.k_norm_und_for_gen is not None:
      assert isinstance(jax_module.k_norm_und_for_gen.scale, nnx.Param)
      jax_module.k_norm_und_for_gen.scale[...] = jnp.array(pt_module.k_norm_und_for_gen.weight.detach().numpy())

else:
  CosmosDualAttentionReferencePT = None
  copy_pytorch_weights_to_jax = None


class CosmosDualAttentionVsPyTorchReferenceTest(parameterized.TestCase):
  """Direct numerical parity tests between MaxText CosmosDualAttention and PyTorch reference."""

  def setUp(self):
    super().setUp()
    if not HAS_TORCH:
      self.skipTest("PyTorch is not available; skipping PyTorch reference parity tests.")
    torch.manual_seed(42)
    np.random.seed(42)

  def test_numerical_parity_standard_dual_norm(self):
    """Verifies exact output equality with QK norm on both pathways and 3D M-RoPE."""
    und_lens = [3, 2]
    gen_lens = [4, 3]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    # 1. Initialize PyTorch reference module
    pt_module = CosmosDualAttentionReferencePT(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=False,
    )
    # Randomize norm weights around 1.0
    with torch.no_grad():
      assert pt_module.q_norm is not None
      assert pt_module.k_norm is not None
      assert pt_module.q_norm_gen is not None
      assert pt_module.k_norm_gen is not None
      pt_module.q_norm.weight.uniform_(0.8, 1.2)
      pt_module.k_norm.weight.uniform_(0.8, 1.2)
      pt_module.q_norm_gen.weight.uniform_(0.8, 1.2)
      pt_module.k_norm_gen.weight.uniform_(0.8, 1.2)

    # 2. Initialize Flax NNX MaxText module
    rngs = nnx.Rngs(42)
    jax_module = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=False,
        rngs=rngs,
    )

    # 3. Copy weights from PyTorch to JAX
    copy_pytorch_weights_to_jax(pt_module, jax_module)

    # 4. Generate identical input tokens and RoPE frequencies
    tokens_np = np.random.randn(meta.total_tokens, dim).astype(np.float32)
    pos_3d_np = np.zeros((meta.total_tokens, 3), dtype=np.float32)
    pos_3d_np[:, 0] = np.arange(meta.total_tokens)
    pos_3d_np[:, 1] = np.arange(meta.total_tokens) % 4
    pos_3d_np[:, 2] = np.arange(meta.total_tokens) // 4

    cos_jax, sin_jax = compute_3d_mrope_cos_sin(jnp.array(pos_3d_np), head_dim=head_dim, mrope_section=(2, 1, 1))
    cos_np = np.array(cos_jax)
    sin_np = np.array(sin_jax)

    # 5. Forward passes
    tokens_pt = torch.from_numpy(tokens_np)
    cos_pt = torch.from_numpy(cos_np)
    sin_pt = torch.from_numpy(sin_np)

    pt_out = pt_module(
        tokens=tokens_pt,
        causal_q_offsets=np.array(meta.causal_q_offsets).tolist(),
        full_q_offsets=np.array(meta.full_q_offsets).tolist(),
        sample_kv_offsets=np.array(meta.sample_kv_offsets).tolist(),
        packed_und_token_indexes=np.array(meta.packed_und_token_indexes).tolist(),
        packed_gen_token_indexes=np.array(meta.packed_gen_token_indexes).tolist(),
        cos=cos_pt,
        sin=sin_pt,
    )

    jax_out = jax_module(
        tokens=jnp.array(tokens_np),
        metadata=meta,
        cos=cos_jax,
        sin=sin_jax,
    )
    assert isinstance(jax_out, jax.Array)

    # 6. Verify exact numerical equality
    np.testing.assert_allclose(
        pt_out.detach().numpy(),
        np.array(jax_out),
        rtol=1e-5,
        atol=1e-5,
        err_msg="MaxText output differs from PyTorch reference output in standard dual-norm config",
    )

  def test_numerical_parity_nemotron_style_cross_attn_k_norm(self):
    """Verifies numerical parity with Nemotron asymmetric QK-norm and cross-attention UND K-norm."""
    und_lens = [4, 3]
    gen_lens = [2, 5]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8

    pt_module = CosmosDualAttentionReferencePT(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
    )
    with torch.no_grad():
      assert pt_module.q_norm_gen is not None
      assert pt_module.k_norm_gen is not None
      assert pt_module.k_norm_und_for_gen is not None
      pt_module.q_norm_gen.weight.uniform_(0.8, 1.2)
      pt_module.k_norm_gen.weight.uniform_(0.8, 1.2)
      pt_module.k_norm_und_for_gen.weight.uniform_(0.8, 1.2)

    rngs = nnx.Rngs(101)
    jax_module = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=False,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=True,
        rngs=rngs,
    )

    copy_pytorch_weights_to_jax(pt_module, jax_module)

    tokens_np = np.random.randn(meta.total_tokens, dim).astype(np.float32)
    pos_3d_np = np.zeros((meta.total_tokens, 3), dtype=np.float32)
    pos_3d_np[:, 0] = np.arange(meta.total_tokens)
    pos_3d_np[:, 1] = np.arange(meta.total_tokens) % 3
    pos_3d_np[:, 2] = np.arange(meta.total_tokens) // 3

    cos_jax, sin_jax = compute_3d_mrope_cos_sin(jnp.array(pos_3d_np), head_dim=head_dim, mrope_section=(2, 1, 1))

    pt_out = pt_module(
        tokens=torch.from_numpy(tokens_np),
        causal_q_offsets=np.array(meta.causal_q_offsets).tolist(),
        full_q_offsets=np.array(meta.full_q_offsets).tolist(),
        sample_kv_offsets=np.array(meta.sample_kv_offsets).tolist(),
        packed_und_token_indexes=np.array(meta.packed_und_token_indexes).tolist(),
        packed_gen_token_indexes=np.array(meta.packed_gen_token_indexes).tolist(),
        cos=torch.from_numpy(np.array(cos_jax)),
        sin=torch.from_numpy(np.array(sin_jax)),
    )

    jax_out = jax_module(
        tokens=jnp.array(tokens_np),
        metadata=meta,
        cos=cos_jax,
        sin=sin_jax,
    )
    assert isinstance(jax_out, jax.Array)

    np.testing.assert_allclose(
        pt_out.detach().numpy(),
        np.array(jax_out),
        rtol=1e-5,
        atol=1e-5,
        err_msg="MaxText output differs from PyTorch reference in Nemotron asymmetric norm config",
    )

  def test_numerical_parity_no_rope(self):
    """Verifies numerical parity when RoPE is not applied (cos=None, sin=None)."""
    und_lens = [2, 3, 2]
    gen_lens = [3, 2, 4]
    meta = build_cosmos_packing_metadata(und_lens, gen_lens)

    dim = 24
    num_heads = 3
    num_kv_heads = 3
    head_dim = 8

    pt_module = CosmosDualAttentionReferencePT(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
    )
    rngs = nnx.Rngs(202)
    jax_module = CosmosDualAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        rngs=rngs,
    )

    copy_pytorch_weights_to_jax(pt_module, jax_module)

    tokens_np = np.random.randn(meta.total_tokens, dim).astype(np.float32)

    pt_out = pt_module(
        tokens=torch.from_numpy(tokens_np),
        causal_q_offsets=np.array(meta.causal_q_offsets).tolist(),
        full_q_offsets=np.array(meta.full_q_offsets).tolist(),
        sample_kv_offsets=np.array(meta.sample_kv_offsets).tolist(),
        packed_und_token_indexes=np.array(meta.packed_und_token_indexes).tolist(),
        packed_gen_token_indexes=np.array(meta.packed_gen_token_indexes).tolist(),
        cos=None,
        sin=None,
    )

    jax_out = jax_module(
        tokens=jnp.array(tokens_np),
        metadata=meta,
        cos=None,
        sin=None,
    )
    assert isinstance(jax_out, jax.Array)

    np.testing.assert_allclose(
        pt_out.detach().numpy(),
        np.array(jax_out),
        rtol=1e-5,
        atol=1e-5,
        err_msg="MaxText output differs from PyTorch reference without RoPE",
    )


if __name__ == "__main__":
  absltest.main()
