# Copyright 2026 Google LLC
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

"""Tests for quantizations.native_fp8_dot_general against real Qwen3.5-397B-A17B-FP8
checkpoint tensors (self_attn, linear_attn, and MoE-expert families), fetched once via
HTTP range requests and cached as .npy fixtures under tests/assets/native_fp8.

Validates two things per tensor family:
  1. Numerical correctness vs. an independent (no qwix, no maxtext) from-scratch numpy
     block-dequant reference.
  2. That qwix genuinely dispatches to its native "fast" (dequantize-on-output) path for
     these exact shapes, not a silent fallback to dequantize-first -- by asserting the
     public qwix.dot_general call is bit-identical to qwix's own _fast_dot_general forced
     directly.
"""

import os
import unittest

import ml_dtypes
import numpy as np
import jax.numpy as jnp
from qwix._src.core import dot_general as qwix_dg

from maxtext.layers import quantizations

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "native_fp8")


def _load_real(key_w, key_s):
  w_u8 = np.load(os.path.join(FIXTURES_DIR, f"real_{key_w}.npy"))
  w = w_u8.view(ml_dtypes.float8_e4m3fn)
  s = np.load(os.path.join(FIXTURES_DIR, f"real_{key_s}.npy")).astype(ml_dtypes.bfloat16)
  return w, s


def _independent_block_dequant(w_fp8: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """From-scratch (no qwix, no maxtext) reference block dequantization: (out, in)."""
  w = w_fp8.astype(np.float32)
  s = np.asarray(scale).astype(np.float32)
  bo = w.shape[0] // s.shape[0]
  bi = w.shape[1] // s.shape[1]
  s_full = np.repeat(np.repeat(s, bo, axis=0), bi, axis=1)
  return w * s_full


class NativeFp8DotGeneralTest(unittest.TestCase):
  """Validates native_fp8_dot_general against real checkpoint tensors."""

  def _run_family(self, key_w, key_s, batch=8, seed=0):
    """Runs test against a checkpoint tensor family."""
    w_np, s_np = _load_real(key_w, key_s)  # w: (out, in), s: (out_blocks, in_blocks)
    in_dim = w_np.shape[1]

    rng = np.random.default_rng(seed)
    act = (rng.normal(size=(batch, in_dim)).astype(np.float32) * 0.02).astype(ml_dtypes.bfloat16)
    act_jax = jnp.asarray(act)
    w_jax = jnp.asarray(w_np)
    s_jax = jnp.asarray(s_np)

    block_size = w_np.shape[1] // s_np.shape[1]  # 128 for every family we test
    axis = (1,)  # inputs: (batch, in) contracts axis 1
    contract_ind = (1,)  # kernel: (out, in) contracts axis 1

    out = quantizations.native_fp8_dot_general(
        act_jax, w_jax, s_jax, block_size, axis, contract_ind, compute_dtype=jnp.float32
    )
    out = np.asarray(out)

    # Independent numpy reference. Note: this must dequantize the SAME quantized
    # activation that native_fp8_dot_general itself produces, not the original
    # unquantized `act` -- fp8 activation quantization has its own, expected,
    # few-percent numerical cost (that's the subject of the end-to-end logit-checker
    # comparison, not this test). This test instead isolates whether the block-scaled
    # *weight* math is correct: given the same quantized activation on both sides, do
    # they agree to near machine precision?
    import qwix  # pylint: disable=import-outside-toplevel

    lhs = qwix.quantize(
        act_jax.astype(jnp.float32),
        jnp.float8_e4m3fn,
        channelwise_axes=[0],
        tiled_axes={1: block_size},
        calibration_method="absmax",
    )
    act_dequant = np.asarray(lhs.qvalue).astype(np.float32) * np.repeat(
        np.asarray(lhs.scale).astype(np.float32), block_size, axis=1
    )
    w_dequant_ref = _independent_block_dequant(w_np, s_np)  # (out, in)
    ref_out = act_dequant @ w_dequant_ref.T  # (batch, out)

    relerr = float(np.max(np.abs(out - ref_out)) / (np.max(np.abs(ref_out)) + 1e-12))
    self.assertLess(relerr, 1e-4, f"{key_w}: native_fp8_dot_general vs independent reference relerr={relerr:.3e}")

    # Confirm qwix genuinely took the native fast path (not a silent fallback), by
    # replicating exactly what native_fp8_dot_general does internally and comparing
    # against the forced _fast_dot_general result -- must be bit-identical.
    rhs = qwix.QArray(qvalue=w_jax, scale=s_jax)
    dnums = (((1,), (1,)), ((), ()))
    fast_out = qwix_dg._fast_dot_general(lhs, rhs, dnums, preferred_element_type=jnp.float32)  # pylint: disable=protected-access
    np.testing.assert_array_equal(
        out, np.asarray(fast_out), err_msg=f"{key_w}: qwix.dot_general did not dispatch to the native fast path"
    )

  def test_self_attn_k_proj(self):
    self._run_family("attn_w", "attn_s")

  def test_linear_attn_in_proj_z(self):
    self._run_family("gdn_w", "gdn_s")

  def test_moe_expert_gate_proj(self):
    self._run_family("moe_w", "moe_s")

  def test_rejects_multi_axis_contraction(self):
    w = jnp.zeros((4, 4), dtype=jnp.float8_e4m3fn)
    s = jnp.ones((1, 1), dtype=jnp.float32)
    x = jnp.zeros((2, 4), dtype=jnp.bfloat16)
    with self.assertRaises(NotImplementedError):
      quantizations.native_fp8_dot_general(x, w, s, 4, (0, 1), (0, 1))


def _requantize_per_tensor(w_fp8: np.ndarray, scale: np.ndarray):
  """Dequantize a real block-quantized weight, then re-quantize with one global scale.

  Mirrors the real checkpoint conversion: the scale is collapsed to its true minimal
  shape (1, 1), not the original block grid uniformly filled -- infer_scale_granularity
  relies on shape alone to detect the scheme, so the shape has to be honest.
  """
  w_dequant = _independent_block_dequant(w_fp8, scale)
  tensor_scale = np.max(np.abs(w_dequant)) / 448.0
  w_q = np.clip(np.round(w_dequant / tensor_scale), -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn)
  scale_minimal = np.full((1, 1), tensor_scale, dtype=np.float32)
  return w_q, scale_minimal, tensor_scale


class InferScaleGranularityTest(unittest.TestCase):
  """Validates the shape-only scheme detector used by native_fp8_dot_general and
  RoutedMoE's _maybe_native_gmm_weight."""

  def test_per_tensor(self):
    self.assertEqual(quantizations.infer_scale_granularity((1, 1)), ("per_tensor", None))

  def test_per_channel(self):
    self.assertEqual(quantizations.infer_scale_granularity((1, 128)), ("per_channel", 1))
    self.assertEqual(quantizations.infer_scale_granularity((128, 1)), ("per_channel", 0))

  def test_block_wise(self):
    self.assertEqual(quantizations.infer_scale_granularity((8, 32)), ("block_wise", None))


class NativeFp8DotGeneralPerTensorTest(unittest.TestCase):
  """Validates native_fp8_dot_general's auto-detected per-tensor path -- both operands
  quantized with a single global scale, no per-token/per-block granularity at all.
  """

  def _run_family(self, key_w, key_s, batch=8, seed=0):
    """Runs per-tensor test against a checkpoint tensor family."""
    w_np, s_np = _load_real(key_w, key_s)  # w: (out, in), s: (out_blocks, in_blocks)
    w_pt, w_scale, tensor_scale = _requantize_per_tensor(w_np, s_np)
    in_dim = w_np.shape[1]

    rng = np.random.default_rng(seed)
    act = (rng.normal(size=(batch, in_dim)).astype(np.float32) * 0.02).astype(ml_dtypes.bfloat16)
    act_jax = jnp.asarray(act)

    out = quantizations.native_fp8_dot_general(
        act_jax,
        jnp.asarray(w_pt),
        jnp.asarray(w_scale),
        in_dim,  # block_size unused in the per-tensor branch; only the block-wise fallback reads it
        (1,),
        (1,),
        compute_dtype=jnp.float32,
    )
    out = np.asarray(out)

    # Reference dequantizes the SAME qwix-quantized activation native_fp8_dot_general
    # produces internally (not an independently-rounded one) -- isolates whether the
    # per-tensor *weight* math is correct, same rationale as NativeFp8DotGeneralTest
    # above: an independent quantization would just measure benign rounding-convention
    # noise between two different absmax implementations, not a real bug.
    import qwix  # pylint: disable=import-outside-toplevel

    lhs = qwix.quantize(
        act_jax.astype(jnp.float32),
        jnp.float8_e4m3fn,
        channelwise_axes=[],
        tiled_axes={},
        calibration_method="absmax",
    )
    act_dequant = np.asarray(lhs.qvalue).astype(np.float32) * np.asarray(lhs.scale).astype(np.float32)
    w_dequant_ref = w_pt.astype(np.float32) * tensor_scale
    ref_out = act_dequant @ w_dequant_ref.T

    relerr = float(np.max(np.abs(out - ref_out)) / (np.max(np.abs(ref_out)) + 1e-12))
    self.assertLess(relerr, 1e-4, f"{key_w}: per-tensor native vs independent reference relerr={relerr:.3e}")

  def test_self_attn_k_proj(self):
    self._run_family("attn_w", "attn_s")

  def test_linear_attn_in_proj_z(self):
    self._run_family("gdn_w", "gdn_s")

  def test_moe_expert_gate_proj(self):
    self._run_family("moe_w", "moe_s")

  def test_accepts_genuine_scalar_scale(self):
    """DenseGeneral's own true per-tensor convention (weight_block_size=None,
    linears.py: `resolved_scale_shape = ()`) stores a real 0-d scalar -- rank-
    mismatched against the 2D kernel, which qwix.QArray rejects unless reshaped
    first. Confirms native_fp8_dot_general's reshape-to-match-rank handles this
    without crashing and without changing the numerical result.
    """
    w_np, s_np = _load_real("attn_w", "attn_s")
    w_pt, w_scale_2d, tensor_scale = _requantize_per_tensor(w_np, s_np)
    in_dim = w_np.shape[1]
    act = jnp.asarray(
        (np.random.default_rng(0).normal(size=(4, in_dim)).astype(np.float32) * 0.02).astype(ml_dtypes.bfloat16)
    )

    out_2d = quantizations.native_fp8_dot_general(
        act, jnp.asarray(w_pt), jnp.asarray(w_scale_2d), in_dim, (1,), (1,), compute_dtype=jnp.float32
    )
    scalar_scale = jnp.asarray(tensor_scale)  # true 0-d, ndim=0
    self.assertEqual(scalar_scale.ndim, 0)
    out_scalar = quantizations.native_fp8_dot_general(
        act, jnp.asarray(w_pt), scalar_scale, in_dim, (1,), (1,), compute_dtype=jnp.float32
    )
    np.testing.assert_array_equal(np.asarray(out_2d), np.asarray(out_scalar))


def _requantize_per_channel(w_fp8: np.ndarray, scale: np.ndarray):
  """Dequantize a real block-quantized weight, then re-quantize with one scale per
  output row (channel) -- scale shape (out, 1), matching infer_scale_granularity's
  per_channel detection on the weight's non-contracted axis.
  """
  w_dequant = _independent_block_dequant(w_fp8, scale)
  channel_scale = np.max(np.abs(w_dequant), axis=1, keepdims=True) / 448.0  # (out, 1)
  channel_scale = np.where(channel_scale == 0, 1.0, channel_scale)
  w_q = np.clip(np.round(w_dequant / channel_scale), -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn)
  return w_q, channel_scale.astype(np.float32)


class NativeFp8DotGeneralPerChannelTest(unittest.TestCase):
  """Validates native_fp8_dot_general's auto-detected per-channel path -- one scale
  per output row, per-token (no block tiling) activation quantization.
  """

  def _run_family(self, key_w, key_s, batch=8, seed=0):
    """Runs per-channel test against a checkpoint tensor family."""
    w_np, s_np = _load_real(key_w, key_s)  # w: (out, in), s: (out_blocks, in_blocks)
    w_pc, w_scale = _requantize_per_channel(w_np, s_np)
    in_dim = w_np.shape[1]

    rng = np.random.default_rng(seed)
    act = (rng.normal(size=(batch, in_dim)).astype(np.float32) * 0.02).astype(ml_dtypes.bfloat16)
    act_jax = jnp.asarray(act)

    out = quantizations.native_fp8_dot_general(
        act_jax,
        jnp.asarray(w_pc),
        jnp.asarray(w_scale),
        in_dim,  # block_size unused in the per-channel branch
        (1,),
        (1,),
        compute_dtype=jnp.float32,
    )
    out = np.asarray(out)

    # Same rationale as the per-tensor test: reference dequantizes the SAME
    # qwix-quantized (per-token) activation native_fp8_dot_general produces
    # internally, isolating whether the per-channel weight math is correct.
    import qwix  # pylint: disable=import-outside-toplevel

    lhs = qwix.quantize(
        act_jax.astype(jnp.float32),
        jnp.float8_e4m3fn,
        channelwise_axes=[0],
        tiled_axes={},
        calibration_method="absmax",
    )
    act_dequant = np.asarray(lhs.qvalue).astype(np.float32) * np.asarray(lhs.scale).astype(np.float32)
    w_dequant_ref = w_pc.astype(np.float32) * w_scale
    ref_out = act_dequant @ w_dequant_ref.T

    relerr = float(np.max(np.abs(out - ref_out)) / (np.max(np.abs(ref_out)) + 1e-12))
    self.assertLess(relerr, 1e-4, f"{key_w}: per-channel native vs independent reference relerr={relerr:.3e}")

  def test_self_attn_k_proj(self):
    self._run_family("attn_w", "attn_s")

  def test_linear_attn_in_proj_z(self):
    self._run_family("gdn_w", "gdn_s")

  def test_moe_expert_gate_proj(self):
    self._run_family("moe_w", "moe_s")


class DenseGeneralNativeFp8Test(unittest.TestCase):
  """Differential test: DenseGeneral wired to native_fp8_compute vs. the existing
  dequantize-then-matmul path, using the same real checkpoint weights on both sides.
  """

  def _make_layer(self, w_np, s_np, quant, rngs):
    """Creates DenseGeneral test layer."""
    from maxtext.layers import linears  # pylint: disable=import-outside-toplevel

    out_dim, in_dim = w_np.shape
    layer = linears.DenseGeneral(
        in_features_shape=in_dim,
        out_features_shape=out_dim,
        weight_dtype=jnp.float8_e4m3fn,
        dtype=jnp.bfloat16,
        block_size=in_dim // s_np.shape[1],
        kernel_axes=("embed", "mlp"),
        quant=quant,
        rngs=rngs,
    )
    layer.kernel[...] = jnp.asarray(w_np.T)  # DenseGeneral kernel is (in, out)
    layer.kernel_scale[...] = jnp.asarray(s_np.T)
    return layer

  def _run_family(self, key_w, key_s, seed=0):
    """Runs parity test comparing native FP8 against dequantize baseline."""
    from flax import nnx  # pylint: disable=import-outside-toplevel

    w_np, s_np = _load_real(key_w, key_s)  # (out, in), (out_blocks, in_blocks)
    in_dim = w_np.shape[1]

    old_layer = self._make_layer(w_np, s_np, None, nnx.Rngs(0))
    new_layer = self._make_layer(w_np, s_np, quantizations.ServeFp8WeightQuantization(), nnx.Rngs(0))

    # New path must not build the stateful Linen quant_dot_general at all.
    self.assertIsNone(new_layer.quant_dot_general)

    rng = np.random.default_rng(seed)
    x = jnp.asarray((rng.normal(size=(4, in_dim)).astype(np.float32) * 0.02).astype(ml_dtypes.bfloat16))

    old_out = np.asarray(old_layer(x)).astype(np.float32)
    new_out = np.asarray(new_layer(x)).astype(np.float32)

    relerr = float(np.max(np.abs(old_out - new_out)) / (np.max(np.abs(old_out)) + 1e-12))
    # Loose tolerance: unlike NativeFp8DotGeneralTest above, this compares against the
    # *existing* dequantize path (exact bf16 activation), so the difference is dominated
    # by native_fp8_compute's activation quantization -- a real, expected few-percent
    # cost (see logit-checker end-to-end validation), not something to drive to zero here.
    self.assertLess(relerr, 0.1, f"{key_w}: DenseGeneral native vs dequantize path relerr={relerr:.3e}")

  def test_self_attn_k_proj(self):
    self._run_family("attn_w", "attn_s")

  def test_linear_attn_in_proj_z(self):
    self._run_family("gdn_w", "gdn_s")

  def test_moe_expert_gate_proj(self):
    self._run_family("moe_w", "moe_s")


if __name__ == "__main__":
  unittest.main()
