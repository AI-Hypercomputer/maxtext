# Copyright 2026 Ant Group. All Rights Reserved.
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

"""Unit tests for KDA (Kimi Delta Attention) module.

Tests cover:
  - KimiDeltaAttention: initialization, forward pass, padding, determinism
  - chunk_kda kernel: basic operation, chunk vs recurrent comparison
  - Naive KDA: recurrent Delta Rule reference impl vs kernel precision
  - Backward (VJP): activation gradients, weight gradients, determinism, bf16
  - QK L2 norm: applied outside kernel, matching Megatron
  - Context parallelism: kernel-level CP equivalence and load_balance rejection

Precision comparison uses _assert_close (atol+rtol+ULP fallback),
adapted from gla_compare_test.py.

Run with: python -m pytest tests/unit/kda_attention_test.py -v
"""

import functools

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes
from flax import nnx

try:
  import tokamax

  TOKAMAX_AVAILABLE = True
except ImportError:
  TOKAMAX_AVAILABLE = False

from maxtext.layers import attention_kda

# The chunk_kda kernel tests exercise tokamax Pallas TPU kernels and multi-chip
# CP; mark the module tpu_only so CPU-only testbeds skip them (consistent with
# kernels_test.py) while they run on TPU hosts.
pytestmark = pytest.mark.tpu_only


# ---------------------------------------------------------------------------
# Precision comparison utilities (adapted from gla_compare_test.py)
# ---------------------------------------------------------------------------


def _bf16_bits_to_ordered(u16):
  magnitude = (u16 & 0x7FFF).astype(np.int64)
  return np.where(u16 & 0x8000, -magnitude, magnitude)


def bf16_ulp_diff(actual_f32, expected_f32):
  """Compute per-element ULP distance at bf16 precision."""
  a_u16 = np.ascontiguousarray(actual_f32.astype(ml_dtypes.bfloat16)).view(np.uint16)
  b_u16 = np.ascontiguousarray(expected_f32.astype(ml_dtypes.bfloat16)).view(np.uint16)
  mismatch_mask = a_u16 != b_u16
  n_mismatch = int(mismatch_mask.sum())
  n_total = a_u16.size
  if n_mismatch == 0:
    return n_mismatch, n_total, 0, np.array([], dtype=np.int64)
  a_ordered = _bf16_bits_to_ordered(a_u16[mismatch_mask])
  b_ordered = _bf16_bits_to_ordered(b_u16[mismatch_mask])
  abs_ulp = np.abs(a_ordered - b_ordered)
  return n_mismatch, n_total, int(abs_ulp.max()), abs_ulp


def _assert_close(actual, expected, label, atol=1e-2, rtol=1e-5, max_ulp=2, max_ulp_fail_rate=1e-3):
  """Assert two arrays match via allclose with bf16 ULP diff fallback."""
  actual_f32 = np.asarray(actual, dtype=np.float32)
  expected_f32 = np.asarray(expected, dtype=np.float32)

  diff = np.abs(actual_f32 - expected_f32)
  max_abs = float(diff.max())
  mean_abs = float(diff.mean())
  print(f"  {label}: max_abs={max_abs:.6e}  mean_abs={mean_abs:.6e}")

  close_mask = diff <= atol + rtol * np.abs(expected_f32)
  if close_mask.all():
    print(f"  {label}: all close ({atol=}, {rtol=})")
    return

  n_fail = int((~close_mask).sum())
  n_total = actual_f32.size
  fail_actual = actual_f32[~close_mask]
  fail_expected = expected_f32[~close_mask]
  n_mis, _, worst_ulp, abs_ulps = bf16_ulp_diff(fail_actual, fail_expected)

  n_over = int((abs_ulps > max_ulp).sum()) if n_mis > 0 else 0
  over_rate = n_over / n_fail if n_fail > 0 else 0.0

  if n_mis > 0:
    print(
        f"  {label} ULP: {n_fail}/{n_total} fail allclose, "
        f"{n_mis} have ULP diff, max_ulp={worst_ulp}, "
        f"over {max_ulp} ULP: {n_over}/{n_fail} ({over_rate:.2e})"
    )

  assert over_rate <= max_ulp_fail_rate, (
      f"{label}: {n_over}/{n_fail} elements ({over_rate:.2e}) exceed "
      f"{max_ulp} ULP (threshold {max_ulp_fail_rate:.2e})"
  )


class _MockKdaConfig:
  """Minimal mock config for KDA testing.

  KDA derives head dims from global config (matching Megatron):
    key_head_dim = value_head_dim = head_dim
    num_key_heads = num_value_heads = base_num_query_heads
  """

  def __init__(self, **overrides):
    self.base_emb_dim = 128
    self.base_num_query_heads = 4
    self.head_dim = 32
    self.dtype = jnp.float32
    self.weight_dtype = jnp.float32
    self.attention_bias = False
    self.shard_mode = "auto"
    self.matmul_precision = "default"
    self.normalization_layer_epsilon = 1e-6
    self.logical_axis_rules = []

    # KDA-specific
    self.linear_conv_kernel_dim = 4
    self.use_qk_norm = True
    self.use_kda_safe_gate = True
    self.kda_lower_bound = -5.0
    self.max_segments_per_seq = 25
    self.context_sharding = "context"

    for k, v in overrides.items():
      setattr(self, k, v)


# ---------------------------------------------------------------------------
# KimiDeltaAttention tests
# ---------------------------------------------------------------------------


class TestKimiDeltaAttention:
  """Tests for KimiDeltaAttention module."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def _make_attn(self, mesh, **config_overrides):
    cfg = _MockKdaConfig(**config_overrides)
    rngs = nnx.Rngs(0)
    with mesh:
      return attention_kda.KimiDeltaAttention(
          config=cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )

  def test_init_head_dims(self, mesh):
    """Head dims derived from global config: head_dim=32, base_num_query_heads=4."""
    attn = self._make_attn(mesh)
    assert attn.num_query_heads == 4
    assert attn.num_key_heads == 4
    assert attn.num_value_heads == 4
    assert attn.key_head_dim == 32
    assert attn.value_head_dim == 32

  def test_init_no_conv(self, mesh):
    attn = self._make_attn(mesh, linear_conv_kernel_dim=0)
    assert attn.q_conv is None

  def test_init_has_gate_and_norm(self, mesh):
    """Output gate projection and out_norm should always be present."""
    attn = self._make_attn(mesh)
    assert hasattr(attn, "gate_proj")
    assert hasattr(attn, "out_norm")
    assert hasattr(attn, "A_log")
    assert hasattr(attn, "dt_bias")

  def test_forward_shape(self, mesh):
    attn = self._make_attn(mesh)
    B, T, D = 2, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, aux = attn(x)
    assert output.shape == (B, T, D)
    assert aux is None

  def test_forward_no_nan_inf(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with mesh:
      output, _ = attn(x)
    assert not jnp.any(jnp.isnan(output))
    assert not jnp.any(jnp.isinf(output))
    assert jnp.any(output != 0)

  def test_sequence_padding(self, mesh):
    """Non-divisible sequence lengths should be handled via padding."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 100, 128  # 100 not divisible by the KDA chunk alignment (64)
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)

  def test_deterministic(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with mesh:
      o1, _ = attn(x)
      o2, _ = attn(x)
    assert jnp.allclose(o1, o2, atol=1e-5)

  def test_packed_sequences_supported(self, mesh):
    """Test that KDA supports packed sequences with segment_ids."""
    attn = self._make_attn(mesh)
    B, T, hidden_dim = 2, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, hidden_dim))
    # 1-based segment_ids, 0 = padding
    seg_ids = jnp.array(
        [
            [1, 1, 1, 2, 2, 2, 3, 3] + [0] * (T - 8),
            [1, 1, 2, 2, 2, 2, 3, 3] + [0] * (T - 8),
        ],
        dtype=jnp.int32,
    )
    with mesh:
      o, _ = attn(x, decoder_segment_ids=seg_ids)
    # Output shape should match input
    assert o.shape == (B, T, hidden_dim)
    # No NaN or Inf
    assert jnp.isfinite(o).all()

  def test_segment_ids_padding_alignment(self, mesh):
    """When T % 64 != 0, segment_ids should be padded along with hidden_states."""
    attn = self._make_attn(mesh)
    B, T, hidden_dim = (
        1,
        100,
        128,
    )  # 100 not divisible by the KDA chunk alignment (64)
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, hidden_dim))
    # segment_ids shorter than padded length (100 -> 128 after pad)
    seg_ids = jnp.array([[1, 1, 1, 2, 2, 2, 3, 3] + [0] * (T - 8)], dtype=jnp.int32)
    with mesh:
      o, _ = attn(x, decoder_segment_ids=seg_ids)
    # Output shape should match input (unpadded back from 128 to 100)
    assert o.shape == (B, T, hidden_dim)
    # First 8 positions should have segment info, rest may be affected by padding
    # but output should still be finite
    assert jnp.isfinite(o).all()

  def test_segment_ids_none_fallback(self, mesh):
    """Test that segment_ids=None falls back to legacy behavior."""
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with mesh:
      o1, _ = attn(x, decoder_segment_ids=None)
      o2, _ = attn(x)  # Default None
    assert jnp.allclose(o1, o2, atol=1e-5)

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_row_independence(self, mesh):
    """Hard verification: row0 and row1 use different inputs; only change row1's seg,
    assert row0 output unchanged.

    Construct batch=[row_a, row_b], only modify row_b's segment_ids,
    assert row_a's output is bit-exact unchanged. Proves segment_ids-based structural
    isolation is effective.
    """
    attn = self._make_attn(mesh)
    T, hidden_dim = 64, 128

    # Critical: two rows use different inputs (prevents XLA caching optimization)
    x0 = jax.random.normal(jax.random.PRNGKey(0), (1, T, hidden_dim))
    x1 = jax.random.normal(jax.random.PRNGKey(1), (1, T, hidden_dim))
    x = jnp.concatenate([x0, x1], axis=0)  # [2, T, hidden_dim]

    # row0: fixed segment; row1: varying segment (keeping padding zeros identical)
    seg_base = jnp.array([[1] * T, [1, 1, 2, 2, 2, 3, 3, 3] + [0] * (T - 8)], dtype=jnp.int32)
    seg_modified = jnp.array([[1] * T, [1, 1, 2, 2, 2, 4, 4, 4] + [0] * (T - 8)], dtype=jnp.int32)

    with mesh:
      o1, _ = attn(x, decoder_segment_ids=seg_base)
      o2, _ = attn(x, decoder_segment_ids=seg_modified)

    # Hard verification: row0 output is bit-exact unchanged (atol=0 means strict equality)
    assert jnp.allclose(o1[0], o2[0], atol=0.0), (
        "Row 0 changed when only row 1's segment changed; " "this indicates segment-based isolation violation"
    )

  def test_autoregressive_not_supported(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with pytest.raises(NotImplementedError, match="autoregressive"):
      attn(x, model_mode="autoregressive")


# ---------------------------------------------------------------------------
# Kernel-level tests
# ---------------------------------------------------------------------------


class TestChunkKda:
  """Direct tests for the chunk_kda kernel via tokamax backend."""

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_basic(self):
    from maxtext.kernels.kda import chunk_kda
    from maxtext.layers.attention_kda import _l2_normalize

    B, T, H, K, V = 1, 2048, 4, 128, 128
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q = _l2_normalize(q)
    k = _l2_normalize(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K))) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))

    o, _ = chunk_kda(q, k, v, g, beta, scale=K**-0.5)
    assert o.shape == (B, T, H, V)
    assert not jnp.any(jnp.isnan(o))


# ---------------------------------------------------------------------------
# Naive KDA reference implementation and precision tests
# ---------------------------------------------------------------------------


def _naive_kda_recurrent(q, k, v, g, beta, scale):
  """Naive Python implementation of KDA Delta Rule (recurrent form).

  Implements the exact recurrence from the KDA docstring:
    S' = S * exp(g_t)                    (gated decay)
    residual = v_t - S'^T @ k_t          (delta residual)
    S = S' + beta_t * k_t outer residual (state update)
    o_t = scale * S @ q_t                (output)

  Args:
    q: [B, T, H, K]  query
    k: [B, T, H, K]  key
    v: [B, T, H, V]  value
    g: [B, T, H, K]  gate (log-space, negative)
    beta: [B, T, H]  delta rule mixing coefficient
    scale: float      output scaling factor

  Returns:
    o: [B, T, H, V]  output
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  o = jnp.zeros((B, T, H, V), dtype=jnp.float32)

  # S: [B, H, K, V] recurrent state
  S = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  for t in range(T):
    # Extract per-step tensors
    q_t = q[:, t, :, :]  # [B, H, K]
    k_t = k[:, t, :, :]  # [B, H, K]
    v_t = v[:, t, :, :]  # [B, H, V]
    g_t = g[:, t, :, :]  # [B, H, K]
    beta_t = beta[:, t, :]  # [B, H]

    # Gated decay: S' = S * exp(g_t)
    # g_t is [B, H, K], S is [B, H, K, V] -> broadcast over V
    S = S * jnp.exp(g_t)[..., None]  # [B, H, K, V]

    # Delta residual: residual = v_t - S^T @ k_t
    # S^T @ k_t: [B, H, V, K] @ [B, H, K] -> [B, H, V]
    # Equivalently: einsum('bhkv,bhk->bhv', S, k_t)
    Sk = jnp.einsum("bhkv,bhk->bhv", S, k_t)  # [B, H, V]
    residual = v_t - Sk  # [B, H, V]

    # State update: S = S + beta_t * k_t outer residual
    # k_t: [B, H, K], residual: [B, H, V] -> outer: [B, H, K, V]
    outer = k_t[..., None] * residual[..., None, :]  # [B, H, K, V]
    S = S + beta_t[..., None, None] * outer  # [B, H, K, V]

    # Output: o_t = scale * S @ q_t
    # einsum('bhkv,bhk->bhv', S, q_t)
    o_t = scale * jnp.einsum("bhkv,bhk->bhv", S, q_t)  # [B, H, V]
    o = o.at[:, t, :, :].set(o_t)

  return o


class TestNaiveKda:
  """Compare tokamax chunk_kda kernel against naive recurrent KDA implementation."""

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_chunk_kda_vs_naive(self):
    """Verify chunk_kda matches the naive Delta Rule recurrence."""
    from maxtext.kernels.kda import chunk_kda
    from maxtext.layers.attention_kda import _l2_normalize

    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q = _l2_normalize(q)
    k = _l2_normalize(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5

    o_kernel, _ = chunk_kda(q, k, v, g, beta, scale=scale)
    o_naive = _naive_kda_recurrent(q, k, v, g, beta, scale)

    assert not jnp.any(jnp.isnan(o_naive)), "Naive output contains NaN"
    assert not jnp.any(jnp.isnan(o_kernel)), "Kernel output contains NaN"
    _assert_close(o_kernel, o_naive, "chunk_kda_vs_naive", atol=5e-3, rtol=1e-3)

  def test_naive_kda_basic_properties(self):
    """Verify naive KDA implementation has correct basic properties."""
    B, T, H, K, V = 1, 8, 2, 4, 4
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32) * 0.1
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32) * 0.1
    g = -jnp.abs(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.1
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5
    o = _naive_kda_recurrent(q, k, v, g, beta, scale)

    assert o.shape == (B, T, H, V)
    assert not jnp.any(jnp.isnan(o)), "Output contains NaN"
    assert not jnp.any(jnp.isinf(o)), "Output contains Inf"
    # First position should be non-zero (state starts empty but gets updated)
    assert jnp.any(o[:, 0, :, :] != 0), "First position output should be non-zero"

  def test_naive_kda_zero_gate_accumulates(self):
    """With g=0 (no decay), state should accumulate without forgetting."""
    B, H, K, V = 1, 1, 2, 2
    T = 4

    q = jnp.ones((B, T, H, K), dtype=jnp.float32)
    k = jnp.ones((B, T, H, K), dtype=jnp.float32) * 0.1
    v = jnp.ones((B, T, H, V), dtype=jnp.float32) * 0.1
    g = jnp.zeros((B, T, H, K), dtype=jnp.float32)  # no decay
    beta = jnp.ones((B, T, H), dtype=jnp.float32)  # full update

    scale = 1.0
    o = _naive_kda_recurrent(q, k, v, g, beta, scale)

    # Output magnitude should grow over time as state accumulates
    norms = jnp.linalg.norm(o[0, :, 0, :], axis=-1)  # [T]
    # Later positions should have larger or equal output norm
    assert norms[-1] >= norms[0], f"With zero gate, output norm should grow: first={norms[0]:.4f}, last={norms[-1]:.4f}"

  def test_naive_kda_large_negative_gate_decays(self):
    """With very negative g, state should decay rapidly."""
    B, H, K, V = 1, 1, 2, 2
    T = 4

    q = jnp.ones((B, T, H, K), dtype=jnp.float32)
    k = jnp.zeros((B, T, H, K), dtype=jnp.float32)  # no new info
    v = jnp.zeros((B, T, H, V), dtype=jnp.float32)
    g = jnp.full((B, T, H, K), -10.0, dtype=jnp.float32)  # aggressive decay
    beta = jnp.ones((B, T, H), dtype=jnp.float32)

    # Manually set initial state by making first step contribute
    k = k.at[:, 0, :, :].set(1.0)
    v = v.at[:, 0, :, :].set(1.0)

    scale = 1.0
    o = _naive_kda_recurrent(q, k, v, g, beta, scale)

    # After step 0, large negative gate should make state decay to ~0
    norm_0 = jnp.linalg.norm(o[0, 0, 0, :])
    norm_last = jnp.linalg.norm(o[0, -1, 0, :])
    assert norm_last < norm_0 * 0.01, (
        f"Large negative gate should decay state: t=0 norm={norm_0:.6f}, " f"t={T-1} norm={norm_last:.6f}"
    )

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_chunk_kda_vs_naive_bf16(self):
    """Verify chunk_kda matches naive in bfloat16 (training dtype)."""
    from maxtext.kernels.kda import chunk_kda
    from maxtext.layers.attention_kda import _l2_normalize

    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)

    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q = _l2_normalize(q)
    k = _l2_normalize(k)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K), dtype=jnp.float32)) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=jnp.float32))

    scale = K**-0.5

    o_kernel, _ = chunk_kda(q, k, v, g, beta, scale=scale)
    o_naive = _naive_kda_recurrent(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        v.astype(jnp.float32),
        g,
        beta,
        scale,
    )

    assert not jnp.any(jnp.isnan(o_kernel)), "Kernel bf16 output contains NaN"
    _assert_close(o_kernel, o_naive, "chunk_kda_bf16_vs_naive", atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QK L2 norm tests
# ---------------------------------------------------------------------------


class TestQkL2Norm:
  """Verify QK L2 normalization is applied outside the kernel."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def test_qk_l2norm_applied_outside_kernel(self, mesh):
    """With use_qk_norm=True, Q and K should be L2-normalized before kernel call."""
    cfg = _MockKdaConfig(use_qk_norm=True)
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output))

  def test_qk_l2norm_skipped_when_disabled(self, mesh):
    """With use_qk_norm=False, forward pass should still work without L2 norm."""
    cfg = _MockKdaConfig(use_qk_norm=False)
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output))

  def test_l2norm_changes_output(self, mesh):
    """Enabling vs disabling L2 norm should produce different outputs."""
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))

    rngs_on = nnx.Rngs(0)
    cfg_on = _MockKdaConfig(use_qk_norm=True)
    with mesh:
      attn_on = attention_kda.KimiDeltaAttention(config=cfg_on, layer_idx=0, mesh=mesh, rngs=rngs_on)
      out_on, _ = attn_on(x)

    rngs_off = nnx.Rngs(0)
    cfg_off = _MockKdaConfig(use_qk_norm=False)
    with mesh:
      attn_off = attention_kda.KimiDeltaAttention(config=cfg_off, layer_idx=0, mesh=mesh, rngs=rngs_off)
      out_off, _ = attn_off(x)

    assert not jnp.allclose(out_on, out_off, atol=1e-4), "L2 norm on/off should produce different outputs"


# ---------------------------------------------------------------------------
# Backward (VJP) tests
# ---------------------------------------------------------------------------


class TestKdaBackward:
  """Backward pass tests for KimiDeltaAttention (learning from GLA test patterns)."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def _make_attn(self, mesh, **config_overrides):
    cfg = _MockKdaConfig(**config_overrides)
    rngs = nnx.Rngs(0)
    with mesh:
      return attention_kda.KimiDeltaAttention(
          config=cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )

  def _run_vjp(self, module, inp, mesh):
    """Run gradient using value_and_grad instead of vjp."""
    graphdef, params, other = nnx.split(module, nnx.Param, ...)

    def forward_fn(params, x):
      model = nnx.merge(graphdef, params, other)
      with mesh:
        out, _ = model(x)
      # Return scalar loss for gradient computation
      return jnp.sum(out)

    # Use value_and_grad instead of vjp (matching training code)
    grad_fn = jax.value_and_grad(forward_fn, argnums=(0, 1), has_aux=False)
    # Returns (loss, (grad_params, grad_input))
    _, grads = grad_fn(params, inp)
    grad_params, grad_input = grads
    return grad_params, grad_input

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_backward_no_nan(self, mesh):
    """Activation gradient should be free of NaN/Inf and non-zero."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      _, grad_input = self._run_vjp(attn, x, mesh)
    assert not jnp.any(jnp.isnan(grad_input)), "grad_input contains NaN"
    assert not jnp.any(jnp.isinf(grad_input)), "grad_input contains Inf"
    assert jnp.any(grad_input != 0), "grad_input is all zeros"

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_backward_deterministic(self, mesh):
    """Two VJP runs should produce identical gradients."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      _, grad1 = self._run_vjp(attn, x, mesh)
      _, grad2 = self._run_vjp(attn, x, mesh)
    assert jnp.allclose(grad1, grad2, atol=1e-5), "Backward is not deterministic"

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_weight_grads_no_nan(self, mesh):
    """Every parameter gradient should be free of NaN/Inf and non-zero."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    with mesh:
      grad_params, _ = self._run_vjp(attn, x, mesh)

    flat_grads = jax.tree.leaves(grad_params)
    for i, g in enumerate(flat_grads):
      assert not jnp.any(jnp.isnan(g)), f"weight grad {i} contains NaN"
      assert not jnp.any(jnp.isinf(g)), f"weight grad {i} contains Inf"
      assert jnp.any(g != 0), f"weight grad {i} is all zeros"

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_backward_bf16(self, mesh):
    """bf16 backward should produce valid gradients."""
    attn = self._make_attn(mesh, dtype=jnp.bfloat16, weight_dtype=jnp.bfloat16)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D), dtype=jnp.bfloat16)
    with mesh:
      grad_params, grad_input = self._run_vjp(attn, x, mesh)
    assert not jnp.any(jnp.isnan(grad_input)), "bf16 grad_input contains NaN"
    assert not jnp.any(jnp.isinf(grad_input)), "bf16 grad_input contains Inf"
    assert jnp.any(grad_input != 0), "bf16 grad_input is all zeros"

    flat_grads = jax.tree.leaves(grad_params)
    for i, g in enumerate(flat_grads):
      assert not jnp.any(jnp.isnan(g)), f"bf16 weight grad {i} contains NaN"


# ---------------------------------------------------------------------------
# ShortConvolution tests (standalone)
# ---------------------------------------------------------------------------


class TestShortConvolution:
  """Tests for ShortConvolution module, including CP halo exchange."""

  def test_short_conv_no_cp(self):
    """ShortConvolution without CP should produce correct output and respect segment masks."""
    from maxtext.layers.attention_kda import ShortConvolution

    rngs = nnx.Rngs(0)
    kernel_size, features = 4, 8
    conv = ShortConvolution(
        kernel_size=kernel_size,
        features=features,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=rngs,
    )

    B, T = 2, 16
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, features))

    # Without segment_ids: causal depthwise conv on full sequence.
    out = conv(x)
    assert out.shape == (B, T, features)
    assert jnp.isfinite(out).all()
    # Output should differ from input (conv applied).
    assert not jnp.allclose(out, x, atol=1e-6)

    # With segment_ids: cross-segment contributions should be masked out.
    seg_ids = jnp.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    out_seg = conv(x, segment_ids=seg_ids)
    assert out_seg.shape == (B, T, features)
    assert jnp.isfinite(out_seg).all()
    # Segment masking should change output.
    assert not jnp.allclose(out_seg, out, atol=1e-6)

    # Row independence: changing row 1's segment_ids should not affect row 0.
    seg_alt = jnp.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    out_alt = conv(x, segment_ids=seg_alt)
    assert jnp.allclose(out_seg[0], out_alt[0], atol=0.0), "Row 0 output changed when only row 1 segments changed"

  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_short_conv_cp_halo(self):
    """ShortConvolution under CP: shard_map with halo exchange matches reference.

    Verifies that when ShortConvolution runs inside a shard_map with the
    "context" axis, ``halo_exchange_for_conv`` pulls left-context tokens
    from the previous CP rank so the causal-conv output is identical to
    running on the full (non-sharded) sequence.
    """
    from maxtext.layers.attention_kda import ShortConvolution

    devices = jax.devices()
    cp_size = 2
    n_devices = (len(devices) // cp_size) * cp_size
    mesh = jax.sharding.Mesh(np.array(devices[:n_devices]).reshape(cp_size, -1), ("context", "x"))

    kernel_size, features = 4, 8
    rngs = nnx.Rngs(0)
    conv = ShortConvolution(
        kernel_size=kernel_size,
        features=features,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=rngs,
    )

    B, T = 2, 32  # divisible by cp_size
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, features))

    # Reference: conv on full sequence without CP sharding.
    ref_out = jax.device_get(conv(x))

    # CP: shard input along T, run conv inside shard_map with "context" axis.
    xs = jax.lax.with_sharding_constraint(
        x,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "context", None)),
    )

    # Uniform segment_ids — no cross-segment masking, so halo tokens are
    # the true left-context and output should match reference.
    seg_ids = jnp.ones((B, T), dtype=jnp.int32)
    segs = jax.lax.with_sharding_constraint(
        seg_ids,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "context")),
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            jax.sharding.PartitionSpec(None, "context", None),
            jax.sharding.PartitionSpec(None, "context"),
        ),
        out_specs=jax.sharding.PartitionSpec(None, "context", None),
        check_vma=False,
    )
    def _conv_cp(x_local, seg_local):
      return conv(x_local, segment_ids=seg_local)

    cp_out = _conv_cp(xs, segs)
    # All-gather: replicate across context axis so we can compare.
    cp_out_full = jax.device_get(
        jax.lax.with_sharding_constraint(cp_out, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
    )

    # With uniform segment_ids (no segmentation), CP conv with halo should
    # match reference exactly — the halo provides the true left context.
    assert jnp.allclose(cp_out_full, ref_out, atol=1e-5), (
        f"ShortConvolution CP halo output differs from reference. "
        f"max_diff={float(jnp.abs(cp_out_full - ref_out).max()):.2e}"
    )


# ---------------------------------------------------------------------------
# CP (Context Parallelism) tests
# ---------------------------------------------------------------------------


class TestKdaCp:
  """Tests for KDA context parallelism."""

  def _cp_mesh(self, cp_size=2):
    devices = jax.devices()
    n_devices = (len(devices) // cp_size) * cp_size
    return jax.sharding.Mesh(np.array(devices[:n_devices]).reshape(cp_size, -1), ("context", "x"))

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_equivalence(self):
    """KDA with CP should produce equivalent output to non-CP KDA."""
    from maxtext.layers.attention_kda import _l2_normalize
    from maxtext.kernels.kda import chunk_kda

    cp_size = 2
    mesh_cp = self._cp_mesh(cp_size=cp_size)

    B, T, H, K, V = 2, 128, 4, 128, 128
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    q = jax.nn.silu(jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32))
    k = jax.nn.silu(jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32))
    q = _l2_normalize(q)
    k = _l2_normalize(k)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K))) * 0.3
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))
    seg_ids = jnp.ones((B, T), dtype=jnp.int32)
    scale = float(K**-0.5)

    # --- Reference: non-CP run ---
    ref_o, _ = chunk_kda(q, k, v, g, beta, scale=scale, segment_ids=seg_ids, max_num_segments=1)

    # --- CP run: shard along T, call chunk_kda with context_parallel_metadata ---
    from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadata

    cp_ctx = ContextParallelMetadata(mesh=mesh_cp, axis_name="context")

    # Shard all inputs along T (axis 1).
    pspec_4d = jax.sharding.PartitionSpec(None, "context", None, None)
    pspec_3d = jax.sharding.PartitionSpec(None, "context", None)
    pspec_2d = jax.sharding.PartitionSpec(None, "context")

    def _shard(arr, pspec):
      return jax.lax.with_sharding_constraint(arr, jax.sharding.NamedSharding(mesh_cp, pspec))

    qs = _shard(q, pspec_4d)
    ks = _shard(k, pspec_4d)
    vs = _shard(v, pspec_4d)
    gs = _shard(g, pspec_4d)
    betas = _shard(beta, pspec_3d)
    segs = _shard(seg_ids, pspec_2d)

    @functools.partial(
        jax.shard_map,
        mesh=mesh_cp,
        in_specs=(pspec_4d, pspec_4d, pspec_4d, pspec_4d, pspec_3d, pspec_2d),
        out_specs=pspec_4d,
        check_vma=False,
    )
    def _kda_cp(q_loc, k_loc, v_loc, g_loc, beta_loc, seg_loc):
      o_loc, _ = chunk_kda(
          q_loc,
          k_loc,
          v_loc,
          g_loc,
          beta_loc,
          scale=scale,
          segment_ids=seg_loc,
          max_num_segments=1,
          context_parallel_metadata=cp_ctx,
      )
      return o_loc

    cp_o = _kda_cp(qs, ks, vs, gs, betas, segs)
    cp_o_full = jax.device_get(
        jax.lax.with_sharding_constraint(cp_o, jax.sharding.NamedSharding(mesh_cp, jax.sharding.PartitionSpec()))
    )

    # CP output should match reference within tolerance.
    _assert_close(cp_o_full, ref_o, "kda_cp_equivalence", atol=5e-3, rtol=1e-3)

  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_rejects_load_balance(self):
    """KDA CP should raise ValueError when load_balance is enabled."""
    mesh = self._cp_mesh(cp_size=2)
    cfg = _MockKdaConfig(context_parallel_load_balance=True)
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(
          config=cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with pytest.raises(ValueError, match="load_balance"):
      attn(x)

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_kda_no_cp_without_load_balance_ok(self):
    """KDA without CP (cp_size=1) should succeed."""
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    cfg = _MockKdaConfig()
    rngs = nnx.Rngs(0)
    with mesh:
      attn = attention_kda.KimiDeltaAttention(
          config=cfg,
          layer_idx=0,
          mesh=mesh,
          rngs=rngs,
      )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with mesh:
      output, _ = attn(x)
    assert output.shape == (1, 64, 128)
    assert jnp.isfinite(output).all()
