# Copyright 2026 Google LLC
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

"""Tests for qk_norm_rope_tc (Pallas interpret mode on CPU, compiled on TPU)."""

# pylint: disable=too-many-positional-arguments,missing-kwoa

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels import qk_norm_rope_tc

_ON_TPU = jax.default_backend() == "tpu"
_EPS = 1e-6


def _problem(seed, b, s, n, h, r, dtype, max_timescale=1e4, gemma4_partial=0.0):
  """Random x/scale/dy plus rotate-half tables for `r` roped dims.

  With `gemma4_partial = p > 0` the tables are built like `Gemma4PartialRotaryEmbedding`: `r == h`
  angles with the last `(1 - p)` fraction set to `inf` (pass-through dims in the middle of each half).
  """
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
  x = jax.random.normal(k1, (b, s, n, h), jnp.float32).astype(dtype)
  scale = 1.0 + 0.1 * jax.random.normal(k2, (h,), jnp.float32)
  positions = jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32)[None], (b, s))
  if gemma4_partial:
    half_rot = int(h * gemma4_partial) // 2
    timescale = 1.0 * (max_timescale ** (2 * jnp.arange(half_rot) / h))
    timescale = jnp.pad(timescale, (0, h // 2 - half_rot), constant_values=jnp.inf)
  else:
    timescale = 1.0 * (max_timescale ** (2 * jnp.arange(r // 2) / r))
  sin_half, cos_half = qk_norm_rope_tc.rope_tables(positions, timescale, dtype)
  dy = jax.random.normal(k3, (b, n, s, h), jnp.float32).astype(dtype)
  return x, scale, sin_half, cos_half, dy


def _fns(sin_half, cos_half, block_s=None, **kw):
  """(fused kernel, pure-JAX reference) closed over the tables and the RMSNorm/RoPE options (`eps`, ...)."""
  fused = functools.partial(
      qk_norm_rope_tc.fused_rmsnorm_rope,
      sin_half=sin_half,
      cos_half=cos_half,
      interpret=not _ON_TPU,
      block_s=block_s,
      **kw,
  )
  ref = functools.partial(qk_norm_rope_tc.reference_rmsnorm_rope, sin_half=sin_half, cos_half=cos_half, **kw)
  return fused, ref


def _loss(fn, dy):
  return lambda x, scale: jnp.sum(fn(x, scale).astype(jnp.float32) * dy.astype(jnp.float32))


def _assert_close_bf16(actual, expected, ulps=1.0, row_relative=False, outlier_ulps=None, outlier_frac=1e-6):
  """Elementwise |a - e| <= ulps * bf16_ulp(e) (plus a tiny absolute slack).

  With `row_relative`, the ulp is that of the largest |e| in the row (last axis): the bf16 XLA
  reference rounds each RoPE term before the add, so its cancellation error scales with the
  operands, not with the result. `outlier_ulps` allows at most an `outlier_frac` fraction of the
  elements to exceed `ulps` (the XLA TPU reference has rare 3-ulp roundings) but never `outlier_ulps`.
  """
  a = np.asarray(actual, np.float32)
  e = np.asarray(expected, np.float32)
  mag = np.max(np.abs(e), axis=-1, keepdims=True) if row_relative else np.abs(e)
  ulp = np.exp2(np.floor(np.log2(np.maximum(mag, 1e-30))) - 7)
  err = np.abs(a - e)
  bad = err > ulps * ulp + 1e-6
  msg = f"{bad.mean():.3e} of elements beyond {ulps} bf16 ulp; max abs err {err.max()}"
  if outlier_ulps is None:
    assert not bad.any(), msg
  else:
    assert bad.mean() <= outlier_frac, msg
    assert not (err > outlier_ulps * ulp + 1e-6).any(), msg


@pytest.mark.parametrize(
    "b,s,n,h,r,block_s,gemma4_partial",
    [
        (1, 128, 16, 512, 512, 64, 0.0),  # full RoPE, R == H
        (1, 128, 16, 512, 256, 64, 0.0),  # PartialRotaryEmbedding, R = H/2 < H
        (1, 128, 2, 512, 128, 64, 0.0),  # R = H/4 (lane-unaligned R/2)
        (2, 1030, 2, 512, 512, 128, 0.25),  # Gemma4 global tables; odd S padded to the block
        (1, 72, 8, 256, 256, 72, 0.0),  # block == S
        (1, 200, 8, 256, 256, 64, 0.0),  # partial tail block (200 = 3 * 64 + 8)
    ],
)
def test_fwd_bwd_f32_matches_reference(b, s, n, h, r, block_s, gemma4_partial):
  """f32 activations: forward and grads (x, scale) match jax.grad of the pure-JAX reference tightly."""
  x, scale, sin_half, cos_half, dy = _problem(0, b, s, n, h, r, jnp.float32, gemma4_partial=gemma4_partial)
  fused, ref = _fns(sin_half, cos_half, block_s=block_s, eps=_EPS)
  y = jax.jit(fused)(x, scale)
  y_ref = jax.jit(ref)(x, scale)
  assert y.shape == (b, n, s, h) and y.dtype == jnp.float32
  np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-5, atol=1e-5)
  gx, gs = jax.jit(jax.grad(_loss(fused, dy), argnums=(0, 1)))(x, scale)
  gx_ref, gs_ref = jax.jit(jax.grad(_loss(ref, dy), argnums=(0, 1)))(x, scale)
  assert gx.dtype == x.dtype and gs.dtype == scale.dtype and gs.shape == (h,)
  np.testing.assert_allclose(np.asarray(gx), np.asarray(gx_ref), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(np.asarray(gs), np.asarray(gs_ref), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "scale_offset,pre_attn_scalar,r",
    [(0.0, 1.0, 512), (1.0, 1.0, 512), (0.0, 0.0625, 256), (1.0, 0.5, 128)],
)
def test_scale_offset_and_pre_attn_scalar(scale_offset, pre_attn_scalar, r):
  """RMSNorm scale_offset path and query_pre_attn_scalar, f32."""
  b, s, n, h = 1, 64, 4, 512
  x, scale, sin_half, cos_half, dy = _problem(1, b, s, n, h, r, jnp.float32)
  fused, ref = _fns(sin_half, cos_half, block_s=32, eps=_EPS, scale_offset=scale_offset, pre_attn_scalar=pre_attn_scalar)
  np.testing.assert_allclose(np.asarray(fused(x, scale)), np.asarray(ref(x, scale)), rtol=1e-5, atol=1e-5)
  gx, gs = jax.grad(_loss(fused, dy), argnums=(0, 1))(x, scale)
  gx_ref, gs_ref = jax.grad(_loss(ref, dy), argnums=(0, 1))(x, scale)
  np.testing.assert_allclose(np.asarray(gx), np.asarray(gx_ref), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(np.asarray(gs), np.asarray(gs_ref), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("r,gemma4_partial", [(512, 0.0), (256, 0.0), (512, 0.25)])
def test_bf16_matches_reference_within_ulp(r, gemma4_partial):
  """bf16 activations (the training dtype).

  The kernel rounds the normed bf16 activations exactly like `RMSNorm` but applies RoPE in f32 with a
  single final rounding, so it is compared two ways:
    * against the reference evaluated with f32 tables (same semantics): within 1 bf16 ulp of the row
      maximum (an rsqrt that differs in the last f32 bit can flip the bf16 rounding of the normed value);
    * against the bf16 XLA reference (rounds each RoPE term): within 2 ulp of the row maximum.
  dscale is compared against an f32-precision reference because the bf16 XLA reference accumulates its
  scale cotangent in bf16 and is itself ~3% off.
  """
  b, s, n, h = 1, 256, 8, 512
  x, scale, sin_half, cos_half, dy = _problem(2, b, s, n, h, r, jnp.bfloat16, gemma4_partial=gemma4_partial)
  fused, ref = _fns(sin_half, cos_half, block_s=64, eps=_EPS)
  ref_f32rope = functools.partial(
      qk_norm_rope_tc.reference_rmsnorm_rope,
      sin_half=sin_half.astype(jnp.float32),
      cos_half=cos_half.astype(jnp.float32),
      eps=_EPS,
  )
  y = jax.jit(fused)(x, scale)
  assert y.dtype == jnp.bfloat16
  _assert_close_bf16(y, jax.jit(ref_f32rope)(x, scale), ulps=1.0, row_relative=True)
  _assert_close_bf16(y, jax.jit(ref)(x, scale), ulps=2.0, row_relative=True)
  gx, gs = jax.jit(jax.grad(_loss(fused, dy), argnums=(0, 1)))(x, scale)
  gx_ref, _ = jax.jit(jax.grad(_loss(ref, dy), argnums=(0, 1)))(x, scale)
  assert gx.dtype == jnp.bfloat16
  _assert_close_bf16(gx, gx_ref, ulps=2.0, row_relative=True)
  _, gs32 = jax.grad(_loss(ref_f32rope, dy), argnums=(0, 1))(x.astype(jnp.float32), scale)
  np.testing.assert_allclose(np.asarray(gs), np.asarray(gs32), rtol=2e-2, atol=1e-2)


def test_out_dtype_and_default_block():
  b, s, n, h = 1, 96, 4, 256
  x, scale, sin_half, cos_half, _ = _problem(3, b, s, n, h, h, jnp.float32)
  fused, ref = _fns(sin_half, cos_half, eps=_EPS, out_dtype=jnp.bfloat16)
  y = fused(x, scale)
  assert y.dtype == jnp.bfloat16 and y.shape == (b, n, s, h)
  _assert_close_bf16(y, ref(x, scale), ulps=1.0, row_relative=True)
  assert qk_norm_rope_tc.default_block_s(16384, 16, 512, 2) == 256
  assert qk_norm_rope_tc.default_block_s(16384, 2, 512, 2) == 2048
  assert qk_norm_rope_tc.default_block_s(16384, 2, 512, 4) == 1024
  assert qk_norm_rope_tc.default_block_s(1030, 16, 512, 2) == 256  # falls back to a padded tail
  assert qk_norm_rope_tc.default_block_s(72, 16, 512, 2) == 72


def test_rejects_bad_shapes():
  x, scale, sin_half, cos_half, _ = _problem(4, 1, 32, 2, 256, 256, jnp.float32)
  with pytest.raises(ValueError):
    qk_norm_rope_tc.fused_rmsnorm_rope(x[0], scale, sin_half, cos_half, eps=_EPS, interpret=True)
  with pytest.raises(ValueError):
    qk_norm_rope_tc.fused_rmsnorm_rope(x, scale[:128], sin_half, cos_half, eps=_EPS, interpret=True)
  with pytest.raises(ValueError):
    qk_norm_rope_tc.fused_rmsnorm_rope(x, scale, sin_half, cos_half, eps=_EPS, interpret=True, block_s=12)


@pytest.mark.tpu_only
@pytest.mark.skipif(not _ON_TPU, reason="TPU only")
@pytest.mark.parametrize("n,h", [(16, 512), (2, 512), (16, 256), (8, 256)])
def test_gemma4_layer_shapes_tpu(n, h):
  """Gemma4-26B layer shapes [2, 16384, N, H] bf16 compiled on TPU: forward and grads vs the XLA reference."""
  b, s = 2, 16384
  gemma4_partial = 0.25 if h == 512 else 0.0
  x, scale, sin_half, cos_half, dy = _problem(5, b, s, n, h, h, jnp.bfloat16, 1e6, gemma4_partial=gemma4_partial)
  fused, ref = _fns(sin_half, cos_half, eps=_EPS)
  _assert_close_bf16(jax.jit(fused)(x, scale), jax.jit(ref)(x, scale), ulps=2.0, row_relative=True, outlier_ulps=4.0)
  gx, gs = jax.jit(jax.grad(_loss(fused, dy), argnums=(0, 1)))(x, scale)
  gx_ref, gs_ref = jax.jit(jax.grad(_loss(ref, dy), argnums=(0, 1)))(x, scale)
  _assert_close_bf16(gx, gx_ref, ulps=2.0, row_relative=True, outlier_ulps=4.0)
  np.testing.assert_allclose(np.asarray(gs), np.asarray(gs_ref), rtol=5e-2, atol=5e-2 * np.abs(np.asarray(gs_ref)).max())


@pytest.mark.tpu_only
@pytest.mark.skipif(not _ON_TPU, reason="TPU only")
def test_benchmark_gemma4_q_tpu(capsys):
  """Prints fwd and fwd+bwd ms of projection + fused kernel vs projection + XLA chain at Gemma4 q size.

  The projection matmul is part of both paths: behind it the kernel's `[B, N, S, H]` operand is a bitcast
  of the dot's natural layout, whereas a row-major jit input would make the transpose a real copy.
  """
  b, s, d, n, h = 2, 16384, 2816, 16, 512
  _, scale, sin_half, cos_half, dy = _problem(6, 1, s, 1, h, h, jnp.bfloat16, 1e6, gemma4_partial=0.25)
  k1, k2 = jax.random.split(jax.random.PRNGKey(7))
  inp = jax.random.normal(k1, (b, s, d), jnp.float32).astype(jnp.bfloat16)
  w = (jax.random.normal(k2, (d, n, h), jnp.float32) * 0.02).astype(jnp.bfloat16)
  sin_half, cos_half = (jnp.broadcast_to(t, (b, s, h // 2)) for t in (sin_half, cos_half))
  dy = jnp.broadcast_to(dy, (b, n, s, h))
  fused, ref = _fns(sin_half, cos_half, eps=_EPS)

  def proj(inp, w):
    return jax.lax.dot_general(inp, w, (((2,), (0,)), ((), ())))

  paths = {
      "proj only": lambda inp, w, scale: jnp.transpose(proj(inp, w), (0, 2, 1, 3)),
      "proj+fused": lambda inp, w, scale: fused(proj(inp, w), scale),
      "proj+xla": lambda inp, w, scale: ref(proj(inp, w), scale),
  }

  def bench(f, *args, iters=20):
    out = f(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
      out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3

  rows = {}
  for name, f_fwd in paths.items():

    def loss(inp, w, scale, f=f_fwd):
      return jnp.sum(f(inp, w, scale).astype(jnp.float32) * dy.astype(jnp.float32))

    rows[name] = (bench(jax.jit(f_fwd), inp, w, scale), bench(jax.jit(jax.grad(loss, argnums=(0, 1, 2))), inp, w, scale))
  with capsys.disabled():
    for name, (t_fwd, t_fb) in rows.items():
      print(f"\n[qk_norm_rope bench {b}x{s}x{n}x{h} bf16] {name}: fwd {t_fwd:.3f} ms, fwd+bwd {t_fb:.3f} ms")
  assert rows["proj+fused"][0] < rows["proj+xla"][0] and rows["proj+fused"][1] < rows["proj+xla"][1]
