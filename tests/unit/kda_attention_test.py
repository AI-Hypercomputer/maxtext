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
from types import SimpleNamespace

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ml_dtypes
from flax import nnx

try:
  from tokamax._src.ops.experimental.kda import api as tokamax_kda_api
  from tokamax._src.ops.experimental.kda.cp_utils import ContextParallelMetadata

  TOKAMAX_AVAILABLE = True
except ImportError:
  tokamax_kda_api = None
  ContextParallelMetadata = None
  TOKAMAX_AVAILABLE = False

from maxtext.configs.types import KdaAttention
from maxtext.kernels.kda import chunk_kda
from maxtext.kernels.kda.tokamax import tokamax_chunk_kda
from maxtext.layers import attention_kda
from maxtext.layers.attention_kda import ShortConvolution, _l2_normalize, halo_exchange_for_conv
from maxtext.layers.normalizations import RMSNorm

# Marker policy: `tpu_only` is applied per test/class — only where a test
# invokes the tokamax Mosaic Pallas kernel or multi-device CP. Pure
# config / pure-op / non-CP tests (init checks, the naive recurrence, the L2
# norm helper, standalone ShortConvolution, config guards) also run in
# regular CPU CI, keeping fast regression coverage outside TPU testbeds.

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
  """Assert two arrays match via allclose with bf16 ULP diff fallback.

  Diagnostics are collected silently and surfaced only in the assertion
  failure message, so passing tests keep normal pytest output quiet.
  """
  actual_f32 = np.asarray(actual, dtype=np.float32)
  expected_f32 = np.asarray(expected, dtype=np.float32)

  diff = np.abs(actual_f32 - expected_f32)
  max_abs = float(diff.max())
  mean_abs = float(diff.mean())

  close_mask = diff <= atol + rtol * np.abs(expected_f32)
  if close_mask.all():
    return

  n_fail = int((~close_mask).sum())
  n_total = actual_f32.size
  fail_actual = actual_f32[~close_mask]
  fail_expected = expected_f32[~close_mask]
  n_mis, _, worst_ulp, abs_ulps = bf16_ulp_diff(fail_actual, fail_expected)

  n_over = int((abs_ulps > max_ulp).sum()) if n_mis > 0 else 0
  over_rate = n_over / n_fail if n_fail > 0 else 0.0

  assert over_rate <= max_ulp_fail_rate, (
      f"{label}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}; "
      f"{n_fail}/{n_total} elements fail allclose (atol={atol}, rtol={rtol}), "
      f"{n_mis} have bf16 ULP diff, worst_ulp={worst_ulp}, "
      f"{n_over}/{n_fail} elements ({over_rate:.2e}) exceed {max_ulp} ULP "
      f"(threshold {max_ulp_fail_rate:.2e})"
  )


def _assert_rel_l2_close(actual, expected, label, tol=2e-2):
  """Assert the relative L2 distance between two arrays is at most ``tol``.

  Suited for gradient comparisons whose per-element absolute tails are
  dominated by accumulation (e.g. weight gradients summed over tokens): the
  norm ratio is scale- and tail-insensitive, while genuine wiring errors
  (relative diff O(1)) still fail hard.
  """
  a = np.asarray(actual, dtype=np.float32).ravel()
  e = np.asarray(expected, dtype=np.float32).ravel()
  rel = float(np.linalg.norm(a - e) / max(float(np.linalg.norm(e)), 1e-8))
  assert rel <= tol, (
      f"{label}: relative L2 diff {rel:.3e} exceeds {tol:.3e} " f"(max_abs={float(np.abs(a - e).max()):.6e})"
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

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_forward_no_conv(self, mesh):
    """Forward with linear_conv_kernel_dim=0 (the conv path is skipped entirely)."""
    attn = self._make_attn(mesh, linear_conv_kernel_dim=0)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    output, _ = attn(x)
    assert output.shape == (1, 64, 128)
    assert jnp.isfinite(output).all()

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_lower_bound_warns_without_safe_gate(self, mesh):
    """use_kda_safe_gate=False with a non-zero kda_lower_bound must warn."""
    attn = self._make_attn(mesh, use_kda_safe_gate=False, kda_lower_bound=-1.0)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with pytest.warns(UserWarning, match="use_kda_safe_gate=False"):
      output, _ = attn(x)
    assert jnp.isfinite(output).all()

  def test_init_has_gate_and_norm(self, mesh):
    """Output gate projection and out_norm should always be present."""
    attn = self._make_attn(mesh)
    assert hasattr(attn, "gate_proj")
    assert hasattr(attn, "out_norm")
    assert hasattr(attn, "A_log")
    assert hasattr(attn, "dt_bias")

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_forward_shape(self, mesh):
    attn = self._make_attn(mesh)
    B, T, D = 2, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    output, aux = attn(x)
    assert output.shape == (B, T, D)
    assert aux is None

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_forward_no_nan_inf(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    output, _ = attn(x)
    assert not jnp.any(jnp.isnan(output))
    assert not jnp.any(jnp.isinf(output))
    assert jnp.any(output != 0)

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_sequence_padding(self, mesh):
    """Non-divisible sequence lengths should be handled via padding."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 100, 128  # 100 not divisible by the KDA chunk alignment (64)
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    output, _ = attn(x)
    assert output.shape == (B, T, D)

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_deterministic(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    o1, _ = attn(x)
    o2, _ = attn(x)
    assert jnp.allclose(o1, o2, atol=1e-5)

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
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
    o, _ = attn(x, decoder_segment_ids=seg_ids)
    # Output shape should match input
    assert o.shape == (B, T, hidden_dim)
    # No NaN or Inf
    assert jnp.isfinite(o).all()

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
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
    o, _ = attn(x, decoder_segment_ids=seg_ids)
    # Output shape should match input (unpadded back from 128 to 100)
    assert o.shape == (B, T, hidden_dim)
    # First 8 positions should have segment info, rest may be affected by padding
    # but output should still be finite
    assert jnp.isfinite(o).all()

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_segment_ids_none_fallback(self, mesh):
    """Test that segment_ids=None falls back to legacy behavior."""
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    o1, _ = attn(x, decoder_segment_ids=None)
    o2, _ = attn(x)  # Default None
    assert jnp.allclose(o1, o2, atol=1e-5)

  @pytest.mark.tpu_only
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

    o1, _ = attn(x, decoder_segment_ids=seg_base)
    o2, _ = attn(x, decoder_segment_ids=seg_modified)

    # Hard verification: row0 output is bit-exact unchanged (atol=0 means strict equality)
    assert jnp.allclose(o1[0], o2[0], atol=0.0), (
        "Row 0 changed when only row 1's segment changed; " "this indicates segment-based isolation violation"
    )

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_packed_segment_no_leak_within_row(self, mesh):
    """Within a single row, changing tokens in segment 2 must not affect segment 1's
    output (and vice versa). Complements test_row_independence (cross-row) by
    proving packed segments inside ONE row are structurally isolated."""
    attn = self._make_attn(mesh)
    T, hidden_dim = 64, 128

    # One row with two packed segments: positions [0,32)=seg1, [32,64)=seg2.
    seg = jnp.array([[1] * 32 + [2] * 32], dtype=jnp.int32)
    x_base = jax.random.normal(jax.random.PRNGKey(2), (1, T, hidden_dim))

    key = jax.random.PRNGKey(3)
    x_mod2 = x_base.at[0, 32:, :].set(jax.random.normal(key, (32, hidden_dim)))  # change seg2 tokens
    x_mod1 = x_base.at[0, :32, :].set(jax.random.normal(key, (32, hidden_dim)))  # change seg1 tokens

    o_base, _ = attn(x_base, decoder_segment_ids=seg)
    o_mod2, _ = attn(x_mod2, decoder_segment_ids=seg)
    o_mod1, _ = attn(x_mod1, decoder_segment_ids=seg)

    # Changing segment 2 must leave segment 1's positions bit-exact unchanged.
    assert jnp.allclose(
        o_base[0, :32], o_mod2[0, :32], atol=0.0
    ), "Segment 1 output changed when only segment 2's tokens changed (same row)."
    # Symmetric: changing segment 1 must leave segment 2 unchanged.
    assert jnp.allclose(
        o_base[0, 32:], o_mod1[0, 32:], atol=0.0
    ), "Segment 2 output changed when only segment 1's tokens changed (same row)."

  def test_autoregressive_not_supported(self, mesh):
    attn = self._make_attn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    with pytest.raises(NotImplementedError, match="autoregressive"):
      attn(x, model_mode="autoregressive")


# ---------------------------------------------------------------------------
# Kernel-level tests
# ---------------------------------------------------------------------------


@pytest.mark.tpu_only
class TestChunkKda:
  """Direct tests for the chunk_kda kernel via tokamax backend."""

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_basic(self):

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

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_chunk_kda_vs_naive(self):
    """Verify chunk_kda matches the naive Delta Rule recurrence."""

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

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_chunk_kda_vs_naive_bf16(self):
    """Verify chunk_kda matches naive in bfloat16 (training dtype)."""

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

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_qk_l2norm_applied_outside_kernel(self, mesh):
    """With use_qk_norm=True, Q and K should be L2-normalized before kernel call."""
    cfg = _MockKdaConfig(use_qk_norm=True)
    rngs = nnx.Rngs(0)
    attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output))

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_qk_l2norm_skipped_when_disabled(self, mesh):
    """With use_qk_norm=False, forward pass should still work without L2 norm."""
    cfg = _MockKdaConfig(use_qk_norm=False)
    rngs = nnx.Rngs(0)
    attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    output, _ = attn(x)
    assert output.shape == (B, T, D)
    assert not jnp.any(jnp.isnan(output))

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_l2norm_changes_output(self, mesh):
    """Enabling vs disabling L2 norm should produce different outputs."""
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))

    rngs_on = nnx.Rngs(0)
    cfg_on = _MockKdaConfig(use_qk_norm=True)
    attn_on = attention_kda.KimiDeltaAttention(config=cfg_on, layer_idx=0, mesh=mesh, rngs=rngs_on)
    out_on, _ = attn_on(x)

    rngs_off = nnx.Rngs(0)
    cfg_off = _MockKdaConfig(use_qk_norm=False)
    attn_off = attention_kda.KimiDeltaAttention(config=cfg_off, layer_idx=0, mesh=mesh, rngs=rngs_off)
    out_off, _ = attn_off(x)

    assert not jnp.allclose(out_on, out_off, atol=1e-4), "L2 norm on/off should produce different outputs"

  def test_l2_normalize_produces_unit_norm(self):
    """Direct check that _l2_normalize yields unit L2 norm along the last axis.

    The layer applies this to Q/K before the kernel; verifying the helper's
    norm (not just output shape/NaN) is the actual correctness property.
    """

    x = jax.random.normal(jax.random.PRNGKey(11), (2, 16, 4, 128))
    normed = _l2_normalize(x)

    norms = jnp.linalg.norm(normed.astype(jnp.float32), axis=-1)
    _assert_close(norms, jnp.ones_like(norms), "l2_unit_norm", atol=1e-4, rtol=1e-4)

    # Direction preserved: normalized vector stays parallel to the input.
    in_norm = jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True)
    expected = x.astype(jnp.float32) / in_norm
    _assert_close(normed.astype(jnp.float32), expected, "l2_direction", atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Backward (VJP) tests
# ---------------------------------------------------------------------------


@pytest.mark.tpu_only
class TestKdaBackward:
  """Backward pass tests for KimiDeltaAttention (learning from GLA test patterns)."""

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def _make_attn(self, mesh, **config_overrides):
    cfg = _MockKdaConfig(**config_overrides)
    rngs = nnx.Rngs(0)
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
    _, grad1 = self._run_vjp(attn, x, mesh)
    _, grad2 = self._run_vjp(attn, x, mesh)
    assert jnp.allclose(grad1, grad2, atol=1e-5), "Backward is not deterministic"

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_weight_grads_no_nan(self, mesh):
    """Every parameter gradient should be free of NaN/Inf and non-zero."""
    attn = self._make_attn(mesh)
    B, T, D = 1, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
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
    grad_params, grad_input = self._run_vjp(attn, x, mesh)
    assert not jnp.any(jnp.isnan(grad_input)), "bf16 grad_input contains NaN"
    assert not jnp.any(jnp.isinf(grad_input)), "bf16 grad_input contains Inf"
    assert jnp.any(grad_input != 0), "bf16 grad_input is all zeros"

    flat_grads = jax.tree.leaves(grad_params)
    for i, g in enumerate(flat_grads):
      assert not jnp.any(jnp.isnan(g)), f"bf16 weight grad {i} contains NaN"


# ---------------------------------------------------------------------------
# Full-layer kernel parity (Mosaic vs tokamax XLA reference)
# ---------------------------------------------------------------------------


@pytest.mark.tpu_only
class TestKdaLayerParity:
  """Full-layer parity between the Mosaic Pallas kernel and the tokamax XLA
  reference implementation.

  Kernel-vs-recurrent-reference parity (TestNaiveKda) validates the kernel
  in isolation, and the CP equivalence tests validate sharding. Neither
  exercises the *composed layer* (QKV projection, ShortConvolution, gate /
  beta transforms, output RMSNorm + gating, output projection) against an
  independent implementation. Running the same layer with identical weights
  on both tokamax implementations catches composition-level argument and
  constraint bugs that kernel-only tests cannot see.
  """

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  def test_full_layer_mosaic_vs_xla_parity(self, monkeypatch):
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    cfg = _MockKdaConfig()
    rngs = nnx.Rngs(0)
    attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)

    B, T, D = 2, 128, 128
    x = jax.random.normal(jax.random.PRNGKey(13), (B, T, D))
    # Two packed segments so varlen handling is part of the parity check.
    seg_ids = jnp.array([[1] * 64 + [2] * 64, [1] * 128], dtype=jnp.int32)

    def _loss(model, x_in):
      o, _ = model(x_in, decoder_segment_ids=seg_ids)
      return o.astype(jnp.float32).sum()

    # Input gradients: differentiate the output sum w.r.t. the input.
    def _loss_x(x_in):
      return _loss(attn, x_in)

    # Weight gradients via the split-params pattern (see TestKdaBackward).
    graphdef, params, other = nnx.split(attn, nnx.Param, ...)

    def _loss_params(params):
      return _loss(nnx.merge(graphdef, params, other), x)

    def _run():
      o, _ = attn(x, decoder_segment_ids=seg_ids)
      return (
          jax.device_get(o),
          jax.device_get(jax.grad(_loss_x)(x)),
          [jax.device_get(g) for g in jax.tree.leaves(jax.grad(_loss_params)(params))],
      )

    # --- Mosaic (production path, the adapter's default) ---
    o_mosaic, grad_x_mosaic, leaves_mosaic = _run()

    # --- XLA reference implementation, same module and weights ---
    orig = tokamax_kda_api.kimi_delta_attention

    def _xla_impl(*args, **kwargs):
      kwargs["implementation"] = "xla"
      return orig(*args, **kwargs)

    # The adapter imports kimi_delta_attention lazily inside the function,
    # so patching the module attribute switches the implementation.
    monkeypatch.setattr(tokamax_kda_api, "kimi_delta_attention", _xla_impl)
    o_xla, grad_x_xla, leaves_xla = _run()

    # The two tokamax implementations are not bitwise identical — tokamax's
    # own CI validates the mosaic kernel against the XLA reference at ~5e-3
    # RMS, and the layer's gated norm/projections propagate that rounding.
    # Use a generous absolute tolerance; gross wiring errors deviate by O(0.1)
    # and still fail hard.
    _assert_close(o_mosaic, o_xla, "full_layer_mosaic_vs_xla_fwd", atol=2e-2, rtol=1e-2)
    # Gradients: relative L2 norm comparison (accumulated weight-gradient
    # tails make absolute tolerances unreliable here; see helper docstring).
    _assert_rel_l2_close(grad_x_mosaic, grad_x_xla, "full_layer_mosaic_vs_xla_dx", tol=2e-2)

    assert len(leaves_mosaic) == len(leaves_xla)
    for i, (gm, gx) in enumerate(zip(leaves_mosaic, leaves_xla)):
      _assert_rel_l2_close(gm, gx, f"full_layer_mosaic_vs_xla_param_{i}", tol=2e-2)


# ---------------------------------------------------------------------------
# ShortConvolution tests (standalone)
# ---------------------------------------------------------------------------


class TestShortConvolution:
  """Tests for ShortConvolution module, including CP halo exchange."""

  def test_short_conv_no_cp(self):
    """ShortConvolution without CP should produce correct output and respect segment masks."""

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

  def test_short_conv_rejects_wrong_features(self):
    """Input with the wrong feature count must fail loudly, not silently."""
    rngs = nnx.Rngs(0)
    conv = ShortConvolution(kernel_size=4, features=8, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=rngs)
    with pytest.raises(ValueError, match="Input features"):
      conv(jnp.zeros((1, 16, 7)))

  def test_short_conv_kernel_size_one(self):
    """kernel_size=1 -> halo_size=0: the exchange returns the input untouched."""
    rngs = nnx.Rngs(0)
    conv = ShortConvolution(kernel_size=1, features=8, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=rngs)
    B, T = 2, 16
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, 8))
    out = conv(x)
    assert out.shape == (B, T, 8)
    assert jnp.isfinite(out).all()

  def test_halo_exchange_single_rank_shard_map(self):
    """Inside a shard_map whose CP axis has size 1 the exchange degrades to zero-pad."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("context",))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 4))

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(None, "context", None),
        out_specs=jax.sharding.PartitionSpec(None, "context", None),
        check_vma=False,
    )
    def _exchange(x_local):
      return halo_exchange_for_conv(x_local, halo_size=2, axis_name="context", seq_axis=1)

    out = _exchange(x)
    assert out.shape == (1, 10, 4)
    assert jnp.allclose(out[:, :2, :], 0.0, atol=0.0), "halo rows must be zeros on the only rank"
    assert jnp.allclose(out[:, 2:, :], x, atol=0.0)

  @pytest.mark.tpu_only
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  @pytest.mark.parametrize("layout", ["uniform", "rank_boundary", "spanning_ranks"])
  def test_short_conv_cp_halo(self, layout):
    """ShortConvolution under CP: shard_map with halo exchange matches reference.

    Verifies that when ShortConvolution runs inside a shard_map with the
    "context" axis, ``halo_exchange_for_conv`` pulls left-context tokens
    from the previous CP rank so the causal-conv output is identical to
    running on the full (non-sharded) sequence.
    """

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

    # Segment layouts exercising different halo/masking interactions.
    # T_local = T // cp_size = 16 per rank.
    if layout == "uniform":
      # No cross-segment masking; halo tokens are the true left context.
      seg_ids = jnp.ones((B, T), dtype=jnp.int32)
    elif layout == "rank_boundary":
      # Segment boundary exactly at the rank boundary: rank 1's first
      # tokens must NOT attend into rank 0 (halo must be masked out).
      seg_ids = jnp.broadcast_to(jnp.array([1] * 16 + [2] * 16, dtype=jnp.int32), (B, T))
    elif layout == "spanning_ranks":
      # Segment 1 spans both ranks: rank 1's leading tokens of segment 1
      # MUST read rank 0's tail through the halo.
      seg_ids = jnp.broadcast_to(jnp.array([1] * 24 + [2] * 8, dtype=jnp.int32), (B, T))
    else:
      raise ValueError(f"unknown layout {layout}")
    seg_ids = seg_ids.astype(jnp.int32)

    # Reference: conv on the full (non-sharded) sequence with the same
    # segment_ids.
    ref_out = jax.device_get(conv(x, segment_ids=seg_ids))

    # CP: shard input along T, run conv inside shard_map with "context" axis.
    xs = jax.lax.with_sharding_constraint(
        x,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "context", None)),
    )

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

    # For every layout, CP conv with halo (and halo'd segment masking) must
    # match the non-sharded reference exactly — halos carry the true left
    # context, and segment masks must suppress cross-segment halo reads.
    assert jnp.allclose(cp_out_full, ref_out, atol=1e-5), (
        f"ShortConvolution CP halo output differs from reference "
        f"(layout={layout}). "
        f"max_diff={float(jnp.abs(cp_out_full - ref_out).max()):.2e}"
    )

  @pytest.mark.tpu_only
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_short_conv_cp_rejects_oversized_halo(self):
    """halo_size > T_local under CP must fail clearly, not silently convolve
    with wrong context.

    The halo exchange only reads from the immediately preceding rank; a
    receptive field (kernel_size - 1) larger than the per-rank sequence
    length would span multiple ranks, which is not implemented.
    """

    devices = jax.devices()
    cp_size = 2
    n_devices = (len(devices) // cp_size) * cp_size
    mesh = jax.sharding.Mesh(np.array(devices[:n_devices]).reshape(cp_size, -1), ("context", "x"))

    kernel_size, features = 8, 8  # halo_size = 7 > T_local = 4 below
    rngs = nnx.Rngs(0)
    conv = ShortConvolution(
        kernel_size=kernel_size,
        features=features,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=rngs,
    )

    B, T = 1, 8  # T_local = T // cp_size = 4 < kernel_size - 1 = 7
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, features))
    xs = jax.lax.with_sharding_constraint(
        x,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "context", None)),
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(None, "context", None),
        out_specs=jax.sharding.PartitionSpec(None, "context", None),
        check_vma=False,
    )
    def _conv_cp(x_local):
      return conv(x_local)

    with pytest.raises(ValueError, match="halo_size"):
      _conv_cp(xs)


# ---------------------------------------------------------------------------
# CP (Context Parallelism) tests
# ---------------------------------------------------------------------------


@pytest.mark.tpu_only
class TestKdaCp:
  """Tests for KDA context parallelism."""

  def _cp_mesh(self, cp_size=2):
    devices = jax.devices()
    n_devices = (len(devices) // cp_size) * cp_size
    return jax.sharding.Mesh(np.array(devices[:n_devices]).reshape(cp_size, -1), ("context", "x"))

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  @pytest.mark.parametrize("cp_size", [2, 4])
  def test_kda_cp_equivalence(self, cp_size):
    """KDA with CP should produce equivalent output to non-CP KDA."""
    if len(jax.devices()) < cp_size:
      pytest.skip(f"need >={cp_size} devices for CP={cp_size}")

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

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_backward(self):
    """CP backward: gradients match the non-CP reference (fwd+bwd through the CP kernels)."""

    cp_size = 2
    mesh_cp = self._cp_mesh(cp_size=cp_size)

    B, T, H, K, V = 2, 128, 4, 128, 128
    key = jax.random.PRNGKey(43)
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
    args = (q, k, v, g, beta)

    # --- Reference: non-CP gradients ---
    def _loss_ref(q, k, v, g, beta):
      o, _ = chunk_kda(q, k, v, g, beta, scale=scale, segment_ids=seg_ids, max_num_segments=1)
      return o.astype(jnp.float32).sum()

    ref_grads = jax.grad(_loss_ref, argnums=(0, 1, 2, 3, 4))(*args)

    # --- CP gradients: shard inputs along T, run the kernel under CP ---
    cp_ctx = ContextParallelMetadata(mesh=mesh_cp, axis_name="context")
    pspec_4d = jax.sharding.PartitionSpec(None, "context", None, None)
    pspec_3d = jax.sharding.PartitionSpec(None, "context", None)
    pspec_2d = jax.sharding.PartitionSpec(None, "context")

    def _shard(arr, pspec):
      return jax.lax.with_sharding_constraint(arr, jax.sharding.NamedSharding(mesh_cp, pspec))

    def _loss_cp(q, k, v, g, beta):
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

      o = _kda_cp(
          _shard(q, pspec_4d),
          _shard(k, pspec_4d),
          _shard(v, pspec_4d),
          _shard(g, pspec_4d),
          _shard(beta, pspec_3d),
          _shard(seg_ids, pspec_2d),
      )
      return o.astype(jnp.float32).sum()

    cp_grads = jax.grad(_loss_cp, argnums=(0, 1, 2, 3, 4))(*args)

    # dq/dk/dv tighter than dg/dbeta (matches tokamax CI tolerances).
    names = ("dq", "dk", "dv", "dg", "dbeta")
    tols = (8e-3, 8e-3, 8e-3, 2e-2, 2e-2)
    for name, tol, cg, rg in zip(names, tols, cp_grads, ref_grads):
      _assert_close(jax.device_get(cg), jax.device_get(rg), f"kda_cp_bwd_{name}", atol=tol, rtol=1e-3)

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_full_layer_dummy_segments(self):
    """Full KimiDeltaAttention under CP without user segment_ids.

    Covers the layer's internal dummy-segment synthesis path (cp_size > 1
    with decoder_segment_ids=None): conv halo exchange, T-axis pspec
    injection and kernel-side CP metadata derivation must combine to
    reproduce the non-CP output, and the CP backward must be finite.
    """
    cp_size = 2
    mesh_cp = self._cp_mesh(cp_size=cp_size)
    B, T, D = 2, 128, 128
    x = jax.random.normal(jax.random.PRNGKey(7), (B, T, D))

    def _build(mesh):
      # Same rng seed on both meshes -> identical weights.
      # head_dim must be a multiple of 128 under CP (mosaic kernel constraint).
      cfg = _MockKdaConfig(head_dim=128)
      rngs = nnx.Rngs(0)
      return attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)

    attn_cp = _build(mesh_cp)
    # Non-CP reference: same weights on a mesh without a CP axis.
    mesh_ref = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    attn_ref = _build(mesh_ref)

    o_cp, _ = attn_cp(x)
    o_full = jax.device_get(
        jax.lax.with_sharding_constraint(o_cp, jax.sharding.NamedSharding(mesh_cp, jax.sharding.PartitionSpec()))
    )
    o_ref, _ = attn_ref(x)

    _assert_close(o_full, jax.device_get(o_ref), "kda_cp_full_layer_dummy_seg", atol=5e-3, rtol=1e-3)
    assert not np.any(np.isnan(o_full)), "NaN in full-layer CP output"

    # CP backward through the whole layer (conv shard_map + kernel shard_map).
    def _sum_cp(x):
      o, _ = attn_cp(x)
      return o.astype(jnp.float32).sum()

    grad_x = jax.device_get(jax.grad(_sum_cp)(x))
    assert np.all(np.isfinite(grad_x)), "non-finite gradient through full-layer CP backward"

  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="tokamax not available")
  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_full_layer_packed_segments(self):
    """Full KimiDeltaAttention under CP with multiple real packed segments.

    Unlike the dummy-segment test, the layouts here stress the composition
    of CP mechanics inside one layer:
      - row 0: segment 2 spans the rank boundary, so recurrent state and
        conv halo must cross ranks within one segment;
      - row 1: a segment boundary exactly at the rank split, so the halo
        tokens pulled across ranks belong to a different segment and must
        be masked out by the conv segment logic, and the kernel must reset
        recurrent state at the rank edge.
    Forward output and input/weight gradients must match the non-CP run.
    """
    cp_size = 2
    mesh_cp = self._cp_mesh(cp_size=cp_size)
    mesh_ref = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    B, T, D = 2, 128, 128
    x = jax.random.normal(jax.random.PRNGKey(11), (B, T, D))

    def _build(mesh):
      # Same rng seed on both meshes -> identical weights.
      # head_dim must be a multiple of 128 under CP (mosaic kernel constraint).
      cfg = _MockKdaConfig(head_dim=128)
      rngs = nnx.Rngs(0)
      return attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)

    attn_cp = _build(mesh_cp)
    attn_ref = _build(mesh_ref)

    # T_local = 64 per rank.
    seg_ids = jnp.array(
        [
            # seg 2 [30, 90) spans the rank boundary at 64.
            [1] * 30 + [2] * 60 + [3] * 38,
            # segment boundary exactly at the rank split: [1]*64 | [2]*64.
            [1] * 64 + [2] * 64,
        ],
        dtype=jnp.int32,
    )

    # --- Forward: CP vs non-CP ---
    o_cp, _ = attn_cp(x, decoder_segment_ids=seg_ids)
    o_cp_full = jax.device_get(
        jax.lax.with_sharding_constraint(o_cp, jax.sharding.NamedSharding(mesh_cp, jax.sharding.PartitionSpec()))
    )
    o_ref, _ = attn_ref(x, decoder_segment_ids=seg_ids)
    # Full-layer tolerance: 2x the kernel-level CP tolerance (5e-3, tokamax CI
    # baseline) since the gated norm / output projection propagate the
    # cross-rank chunk-boundary rounding through the rest of the layer.
    _assert_close(o_cp_full, jax.device_get(o_ref), "kda_cp_full_layer_packed_seg_fwd", atol=1e-2, rtol=1e-3)
    assert not np.any(np.isnan(o_cp_full)), "NaN in full-layer CP packed-segment output"

    # --- Backward: input and weight gradients, CP vs non-CP ---
    def _grads(attn, mesh):
      graphdef, params, other = nnx.split(attn, nnx.Param, ...)

      def loss_fn(params, x):
        model = nnx.merge(graphdef, params, other)
        o, _ = model(x, decoder_segment_ids=seg_ids)
        return o.astype(jnp.float32).sum()

      _, (grad_params, grad_x) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params, x)
      return jax.tree.leaves(grad_params), jax.device_get(grad_x)

    leaves_cp, grad_x_cp = _grads(attn_cp, mesh_cp)
    leaves_ref, grad_x_ref = _grads(attn_ref, mesh_ref)

    _assert_close(grad_x_cp, grad_x_ref, "kda_cp_full_layer_packed_seg_dx", atol=2e-2, rtol=1e-2)

    # Weight gradients are summed over the whole sequence, so their absolute
    # tails are larger than per-token kernel tolerances; compare by
    # relative L2 norm instead.
    assert len(leaves_cp) == len(leaves_ref), "CP and non-CP param gradient trees differ"
    for i, (gc, gr) in enumerate(zip(leaves_cp, leaves_ref)):
      _assert_rel_l2_close(
          jax.device_get(gc),
          jax.device_get(gr),
          f"kda_cp_full_layer_packed_seg_param_{i}",
          tol=2e-2,
      )

  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_requires_tokamax_metadata(self, monkeypatch):
    """If ContextParallelMetadata failed to import, CP must refuse to run.

    CP without cross-rank metadata would silently split the recurrent
    state across ranks, so the layer raises instead of degrading.
    """
    mesh = self._cp_mesh(cp_size=2)
    cfg = _MockKdaConfig(head_dim=128)
    rngs = nnx.Rngs(0)
    attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)

    monkeypatch.setattr(attention_kda, "TokamaxContextParallelMetadata", None)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 128, 128))
    with pytest.raises(ImportError, match="ContextParallelMetadata"):
      attn(x)

  @pytest.mark.skipif(len(jax.devices()) < 2, reason="need >=2 devices for CP test")
  def test_kda_cp_rejects_load_balance(self):
    """KDA CP should raise ValueError when load_balance is enabled."""
    mesh = self._cp_mesh(cp_size=2)
    cfg = _MockKdaConfig(context_parallel_load_balance=True)
    rngs = nnx.Rngs(0)
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
    attn = attention_kda.KimiDeltaAttention(
        config=cfg,
        layer_idx=0,
        mesh=mesh,
        rngs=rngs,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    output, _ = attn(x)
    assert output.shape == (1, 64, 128)
    assert jnp.isfinite(output).all()


# ---------------------------------------------------------------------------
# Config guard tests
# ---------------------------------------------------------------------------


class TestKdaConfigGuards:
  """Config-time guards for invalid KDA combinations."""

  def test_safe_gate_requires_valid_lower_bound(self):
    """use_kda_safe_gate=True with kda_lower_bound outside [-5,0) must be rejected."""

    with pytest.raises(ValueError, match="kda_lower_bound"):
      KdaAttention(use_kda_safe_gate=True, kda_lower_bound=0.0)
    with pytest.raises(ValueError, match="kda_lower_bound"):
      KdaAttention(use_kda_safe_gate=True, kda_lower_bound=-6.0)
    # Valid combinations pass.
    KdaAttention(use_kda_safe_gate=True, kda_lower_bound=-5.0)
    KdaAttention(use_kda_safe_gate=True, kda_lower_bound=-1.0)
    KdaAttention(use_kda_safe_gate=False)

  def test_use_kda_lora_true_rejected(self):
    """use_kda_lora=True is an unimplemented no-op and must be rejected."""

    with pytest.raises(ValueError, match="use_kda_lora"):
      KdaAttention(use_kda_lora=True)
    KdaAttention(use_kda_lora=False)

  @pytest.fixture
  def mesh(self):
    return jax.sharding.Mesh(jax.devices(), ("x",))

  def test_packing_requires_max_segments_per_seq(self, mesh):
    """Layer must fail fast when packed sequences are used without max_segments_per_seq."""
    cfg = _MockKdaConfig(max_segments_per_seq=-1)
    rngs = nnx.Rngs(0)
    attn = attention_kda.KimiDeltaAttention(config=cfg, layer_idx=0, mesh=mesh, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 64, 128))
    seg = jnp.ones((1, 64), dtype=jnp.int32)
    with pytest.raises(ValueError, match="max_segments_per_seq"):
      attn(x, decoder_segment_ids=seg)


# ---------------------------------------------------------------------------
# Kernel input-guard tests (raise before any kernel call; run in CPU CI)
# ---------------------------------------------------------------------------


class TestKdaKernelGuards:
  """``initial_state`` / ``output_final_state`` are rejected before dispatch.

  These guards fire before any tokamax kernel is touched, so they are safe
  to run on CPU (no tpu_only marker).
  """

  @staticmethod
  def _dummy_inputs():
    """Small random q/k/v/g/beta tensors shaped for the kernel interface."""
    B, T, H, K, V = 1, 64, 2, 16, 16
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (B, T, H, K))
    k = jax.random.normal(keys[1], (B, T, H, K))
    v = jax.random.normal(keys[2], (B, T, H, V))
    g = jax.random.normal(keys[3], (B, T, H, K))
    beta = jax.random.normal(keys[4], (B, T, H))
    return q, k, v, g, beta

  def test_chunk_kda_rejects_initial_state(self):
    q, k, v, g, beta = self._dummy_inputs()
    with pytest.raises(NotImplementedError, match="initial_state"):
      chunk_kda(q, k, v, g, beta, scale=0.25, initial_state=jnp.zeros((2, 16, 16)))

  def test_chunk_kda_rejects_output_final_state(self):
    q, k, v, g, beta = self._dummy_inputs()
    with pytest.raises(NotImplementedError, match="output_final_state"):
      chunk_kda(q, k, v, g, beta, scale=0.25, output_final_state=True)

  def test_tokamax_adapter_rejects_initial_state(self):
    q, k, v, g, beta = self._dummy_inputs()
    with pytest.raises(NotImplementedError, match="initial_state"):
      tokamax_chunk_kda(q, k, v, g, beta, scale=0.25, initial_state=jnp.zeros((2, 16, 16)))

  def test_tokamax_adapter_rejects_output_final_state(self):
    q, k, v, g, beta = self._dummy_inputs()
    with pytest.raises(NotImplementedError, match="output_final_state"):
      tokamax_chunk_kda(q, k, v, g, beta, scale=0.25, output_final_state=True)


# ---------------------------------------------------------------------------
# End-to-end training smoke (delayed-copy task; replaces the former
# scripts/dev/kda_e2e_smoke.py)
# ---------------------------------------------------------------------------

_KDA_SMOKE_VOCAB = 128


class _KdaBlock(nnx.Module):
  """Pre-norm transformer block: RMSNorm -> KimiDeltaAttention -> MLP."""

  def __init__(self, cfg, mesh, layer_idx, *, rngs):
    self.attn_norm = RMSNorm(
        num_features=cfg.base_emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.attn = attention_kda.KimiDeltaAttention(cfg, layer_idx=layer_idx, mesh=mesh, rngs=rngs)
    self.mlp_norm = RMSNorm(
        num_features=cfg.base_emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    hidden = 4 * cfg.base_emb_dim
    self.wi = nnx.Linear(cfg.base_emb_dim, hidden, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)
    self.wo = nnx.Linear(hidden, cfg.base_emb_dim, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)

  def __call__(self, x):
    attn_out, _ = self.attn(self.attn_norm(x).astype(self.attn.config.dtype))
    x = x + attn_out.astype(x.dtype)
    h = nnx.gelu(self.wi(self.mlp_norm(x)))
    x = x + self.wo(h).astype(x.dtype)
    return x


class _TinyKdaLM(nnx.Module):
  """Embed -> N x _KdaBlock -> RMSNorm -> lm_head."""

  def __init__(self, cfg, mesh, num_layers, *, rngs):
    self.embed = nnx.Embed(_KDA_SMOKE_VOCAB, cfg.base_emb_dim, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs)
    self.blocks = nnx.List([_KdaBlock(cfg, mesh, i, rngs=rngs) for i in range(num_layers)])
    self.final_norm = RMSNorm(
        num_features=cfg.base_emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.lm_head = nnx.Linear(
        cfg.base_emb_dim, _KDA_SMOKE_VOCAB, dtype=cfg.dtype, param_dtype=cfg.weight_dtype, rngs=rngs
    )

  def __call__(self, tokens):
    x = self.embed(tokens)
    for block in self.blocks:
      x = block(x)
    return self.lm_head(self.final_norm(x))


def _delayed_copy_dataset(seed, num_seqs, seq_len, delay):
  """Delayed-copy sequences over random tokens: t[i] = t[i-delay]."""
  rng = np.random.default_rng(seed)
  total = seq_len + 1
  seqs = rng.integers(0, _KDA_SMOKE_VOCAB, size=(num_seqs, total), dtype=np.int32)
  for i in range(delay, total):
    seqs[:, i] = seqs[:, i - delay]
  return seqs


class TestKdaE2eSmoke:
  """End-to-end training smoke for the KimiDeltaAttention layer.

  The task is delayed copy: i.i.d. tokens with ``t[i] = t[i-delay]`` where
  ``delay`` exceeds the short convolution's receptive field. Neither a
  memoryless model nor the convolution alone can predict the next token, so
  the loss collapses to near zero only if the KDA recurrent state carries
  history — validating the full forward/backward/optimizer chain through the
  real Pallas kernels.
  """

  @pytest.mark.tpu_only
  @pytest.mark.skipif(not TOKAMAX_AVAILABLE, reason="KDA API not available in the installed tokamax")
  def test_delayed_copy_loss_collapses(self):
    seq_len, delay, steps, batch, num_layers = 64, 5, 300, 32, 2
    cfg = SimpleNamespace(
        base_emb_dim=256,
        base_num_query_heads=8,
        head_dim=64,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        attention_bias=False,
        shard_mode="auto",
        matmul_precision="default",
        normalization_layer_epsilon=1e-6,
        logical_axis_rules=[],
        linear_conv_kernel_dim=4,
        use_qk_norm=True,
        use_kda_safe_gate=True,
        kda_lower_bound=-5.0,
        max_segments_per_seq=25,
        context_sharding="context",
    )
    # Delay beyond the conv receptive field, or the task is solvable without
    # any KDA state (see review of the original permutation task).
    assert delay > cfg.linear_conv_kernel_dim

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    rngs = nnx.Rngs(0)
    model = _TinyKdaLM(cfg, mesh, num_layers, rngs=rngs)

    data = _delayed_copy_dataset(seed=42, num_seqs=4096, seq_len=seq_len, delay=delay)
    # Label position j predicts token j+1 = t[j+1-delay]; positions with
    # j+1 < delay have random targets (irreducible), so mask them out.
    loss_mask = jnp.asarray(np.arange(seq_len) >= delay - 1, dtype=jnp.float32)[None, :]
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    def masked_ce(logits, labels):
      ce = optax.softmax_cross_entropy_with_integer_labels(logits=logits.astype(jnp.float32), labels=labels)
      return (ce * loss_mask).sum() / loss_mask.sum() / labels.shape[0]

    @nnx.jit
    def train_step(model, optimizer, tokens):
      def loss_fn(model):
        logits = model(tokens[:, :-1])
        return masked_ce(logits, tokens[:, 1:]), logits

      (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
      optimizer.update(model, grads)
      return loss

    perm_rng = np.random.default_rng(1)
    losses = []
    for step in range(steps):
      idx = perm_rng.integers(0, data.shape[0], size=batch)
      loss_val = float(train_step(model, optimizer, jnp.asarray(data[idx])))
      assert np.isfinite(loss_val), f"non-finite loss {loss_val} at step {step}"
      losses.append(loss_val)

    init_loss, final_loss = losses[0], float(np.mean(losses[-20:]))
    assert final_loss < 0.5 * init_loss and final_loss < 1.0, (
        f"delayed-copy loss did not collapse: init={init_loss:.4f} final={final_loss:.4f} "
        "(the KDA recurrent state is not carrying history through training)"
    )
