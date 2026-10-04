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

"""TPU tests: 3D-layout gmm_v2 / tgmm_v2 match the 2D kernels (bf16 and fp8); gmm_v2 is exact on a partial last K tile."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import qwix.pallas as qpl

from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.kernels.ragged import ragged_sort_tc
import pytest

pytestmark = pytest.mark.tpu_only  # Pallas TPU kernels.

FP8 = jnp.float8_e4m3fn
M, D, F, E = 2048, 7168, 512, 4
GROUP_SIZES = np.array([300, 0, 517, 700], np.int32)  # sum 1517 < M: padding rows.


def _to3d(x):
  return x.reshape(x.shape[0], -1, 128)


def _rand(key, shape, dtype, scale=1.0):
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _f32(x):
  return np.asarray(x.astype(jnp.float32))


def _assert_close_chunked_k(got, want, rtol=2e-2, max_mismatch_frac=None):
  """For outputs of the chunked 3D-lhs path (8-bit operands, 3D lhs K tile 7168).

  The kernel relayouts that K tile in D0 chunks and accumulates the chunk matmuls;
  on some TPU generations this rounds differently from one K-wide matmul (about one
  bf16 ulp on a few elements), so compare with a tolerance instead of bit-exactly.
  """
  g, w = _f32(got), _f32(want)
  np.testing.assert_allclose(g, w, rtol=rtol, atol=rtol * float(np.max(np.abs(w))))
  if max_mismatch_frac is not None:
    assert np.mean(g != w) <= max_mismatch_frac, np.mean(g != w)


class Gmm3dTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")
    self.gs = jnp.asarray(GROUP_SIZES)

  @parameterized.product(lhs_dtype=["bfloat16", "float8_e4m3fn"], tile_k=[1024, 1792, 3584, 7168])
  def test_gmm_3d_lhs(self, lhs_dtype, tile_k):
    """wi fwd (fp8 tokens x fp8 weight) / wo dlhs: 3D lhs, 2D out."""
    lhs_dtype = jnp.dtype(lhs_dtype)
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs = _rand(k0, (M, D), lhs_dtype, 4.0)
    rhs = _rand(k1, (E, D, F), FP8 if lhs_dtype == FP8 else jnp.bfloat16, 4.0)
    rhs_scale = None
    if lhs_dtype == FP8:
      rhs_scale = jnp.abs(_rand(k2, (E, 1, 1, F), jnp.float32)) * 0.01
    tiles = gmm_v2.TileSizes(tile_m=512, tile_k=tile_k, tile_n=512)
    f = functools.partial(gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=False)
    want = jax.jit(f)(lhs, rhs, self.gs, rhs_scale)
    got = jax.jit(f)(_to3d(lhs), rhs, self.gs, rhs_scale)
    if lhs_dtype == FP8 and tile_k == D:
      _assert_close_chunked_k(got, want, max_mismatch_frac=1e-3)
    else:
      np.testing.assert_array_equal(_f32(got), _f32(want))
    self.assertTrue(np.all(_f32(got)[GROUP_SIZES.sum() :] == 0))

  @parameterized.product(
      lhs_dtype=["bfloat16", "float8_e4m3fn"], tile_n=[1024, 1792, 3584, 7168], quantize_lhs=[False, True]
  )
  def test_gmm_3d_out(self, lhs_dtype, tile_n, quantize_lhs):
    """wo fwd (bf16 act, quantized in-kernel) / wi dlhs (fp8 grad): 2D lhs, 3D bf16 out."""
    lhs_dtype = jnp.dtype(lhs_dtype)
    if quantize_lhs and lhs_dtype == FP8:
      self.skipTest("fp8 lhs is already quantized")
    k0, k1, k2 = jax.random.split(jax.random.key(1), 3)
    lhs = _rand(k0, (M, F), lhs_dtype, 4.0)
    rhs_q = lhs_dtype == FP8 or quantize_lhs
    rhs = _rand(k1, (E, F, D), FP8 if rhs_q else jnp.bfloat16, 4.0)
    rhs_scale = jnp.abs(_rand(k2, (E, 1, 1, D), jnp.float32)) * 0.01 if rhs_q else None
    lhs_scale = jnp.full((1, 1), 0.02, jnp.float32) if quantize_lhs else None
    tiles = gmm_v2.TileSizes(tile_m=512, tile_k=F, tile_n=tile_n)
    f = functools.partial(
        gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=quantize_lhs
    )
    want = jax.jit(f)(lhs, rhs, self.gs, rhs_scale, lhs_scale=lhs_scale)
    got = jax.jit(functools.partial(f, out_is_3d=True))(lhs, rhs, self.gs, rhs_scale, lhs_scale=lhs_scale)
    self.assertEqual(got.shape, (M, D // 128, 128))
    np.testing.assert_array_equal(_f32(got.reshape(M, D)), _f32(want))

  def test_gmm_3d_lhs_and_out(self):
    """dx = dgate @ W^T with 3D in and 3D out (both D-sized)."""
    k0, k1 = jax.random.split(jax.random.key(2))
    lhs = _rand(k0, (M, D), jnp.bfloat16)
    rhs = _rand(k1, (E, D, D), jnp.bfloat16, 0.05)
    tiles = gmm_v2.TileSizes(tile_m=256, tile_k=1024, tile_n=1024)
    f = functools.partial(gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16)
    want = jax.jit(f)(lhs, rhs, self.gs)
    got = jax.jit(functools.partial(f, out_is_3d=True))(_to3d(lhs), rhs, self.gs)
    np.testing.assert_array_equal(_f32(got.reshape(M, D)), _f32(want))

  def test_gmm_3d_lhs_full_k_wi_fwd(self):
    """wi fwd tiles 512 / 7168 / 2048 (fp8): the 3D lhs K tile is relayouted in chunks."""
    k0, k1, k2 = jax.random.split(jax.random.key(7), 3)
    f_mlp = 2048
    lhs = _rand(k0, (M, D), FP8, 4.0)
    rhs = _rand(k1, (E, D, f_mlp), FP8, 4.0)
    rhs_scale = jnp.abs(_rand(k2, (E, 1, 1, f_mlp), jnp.float32)) * 0.01
    tiles = gmm_v2.TileSizes(tile_m=512, tile_k=7168, tile_n=2048)
    f = functools.partial(gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=False)
    want = jax.jit(f)(lhs, rhs, self.gs, rhs_scale)
    got = jax.jit(f)(_to3d(lhs), rhs, self.gs, rhs_scale)
    _assert_close_chunked_k(got, want, max_mismatch_frac=1e-3)
    self.assertTrue(np.all(_f32(got)[GROUP_SIZES.sum() :] == 0))

  @parameterized.product(dtype=["bfloat16", "float8_e4m3fn"], which=["lhs", "rhs"], tile=[1024, 1792, 3584, 7168])
  def test_tgmm_3d(self, dtype, which, tile):
    """dWi (3D lhs tokens) / dWo (3D rhs grad)."""
    dtype = jnp.dtype(dtype)
    k0, k1 = jax.random.split(jax.random.key(3))
    x = _rand(k0, (M, D), dtype, 4.0)
    y = _rand(k1, (M, F), dtype, 4.0)
    # Keep the (double-buffered f32) output tile within VMEM for tile=7168.
    tm, tf = (512, F) if tile < 7168 else (256, 256)
    if which == "lhs":
      lhs, rhs = x, y
      tiles = gmm_v2.TileSizes(tile_m=tm, tile_k=tile, tile_n=tf)
      lhs3, rhs3 = _to3d(lhs), rhs
    else:
      lhs, rhs = y, x
      tiles = gmm_v2.TileSizes(tile_m=tm, tile_k=tf, tile_n=tile)
      lhs3, rhs3 = lhs, _to3d(rhs)
    f = functools.partial(tgmm_v2.tgmm_v2, num_actual_groups=E, tile_info=tiles, preferred_element_type=jnp.float32)
    want = jax.jit(f)(lhs, rhs, self.gs)
    got = jax.jit(f)(lhs3, rhs3, self.gs)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

  @parameterized.product(lhs_dtype=["bfloat16", "float8_e4m3fn"], tile_k=[1024, 3584, 7168])
  def test_gmm_3d_lhs_transpose_rhs(self, lhs_dtype, tile_k):
    """wo dlhs with native transpose_rhs: 3D lhs (the 3D grad), rhs [g, n, k] contracted on its last dim."""
    lhs_dtype = jnp.dtype(lhs_dtype)
    k0, k1 = jax.random.split(jax.random.key(8))
    lhs = _rand(k0, (M, D), lhs_dtype, 4.0)
    # wo [g, mlp, emb]: dlhs = grad @ wo^T (fp8 grad x fp8 weight, or bf16 x bf16).
    rhs_t = _rand(k1, (E, F, D), FP8 if lhs_dtype == FP8 else jnp.bfloat16, 0.05 if lhs_dtype != FP8 else 4.0)
    tiles = gmm_v2.TileSizes(tile_m=512, tile_k=tile_k, tile_n=F)
    f = functools.partial(gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=False)
    want = jax.jit(f)(lhs, rhs_t.swapaxes(1, 2), self.gs)
    got = jax.jit(functools.partial(f, transpose_rhs=True))(_to3d(lhs), rhs_t, self.gs)
    if lhs_dtype == FP8 and tile_k == D:
      _assert_close_chunked_k(got, want, max_mismatch_frac=1e-3)
    else:
      np.testing.assert_array_equal(_f32(got), _f32(want))

  @parameterized.product(tile_n=[1024, 3584, 7168], ps_dtype=["bfloat16", "float32"], transpose_rhs=[False, True])
  def test_gmm_3d_out_partial_sum(self, tile_n, ps_dtype, transpose_rhs):
    """wi_0 dlhs accumulated onto the wi_1 dlhs (3D partial_sum), optionally with native transpose_rhs."""
    ps_dtype = jnp.dtype(ps_dtype)
    k0, k1, k2 = jax.random.split(jax.random.key(9), 3)
    lhs = _rand(k0, (M, F), FP8, 4.0)
    rhs_t = _rand(k1, (E, D, F), jnp.bfloat16, 0.05)  # wi [g, emb, mlp]: dlhs = grad @ wi^T.
    ps = _rand(k2, (M, D), ps_dtype)
    lhs_scale = jnp.full((1, 1), 0.02, jnp.float32)
    tiles = gmm_v2.TileSizes(tile_m=256, tile_k=F, tile_n=tile_n)
    f = functools.partial(gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=ps_dtype, maybe_quantize_lhs=False)
    rhs = rhs_t if transpose_rhs else rhs_t.swapaxes(1, 2)
    want = jax.jit(functools.partial(f, transpose_rhs=transpose_rhs))(
        lhs, rhs, self.gs, partial_sum=ps, lhs_scale=lhs_scale
    )
    got = jax.jit(functools.partial(f, transpose_rhs=transpose_rhs, out_is_3d=True))(
        lhs, rhs, self.gs, partial_sum=_to3d(ps), lhs_scale=lhs_scale
    )
    self.assertEqual(got.shape, (M, D // 128, 128))
    np.testing.assert_array_equal(_f32(got.reshape(M, D)), _f32(want))

  @parameterized.product(
      tiles=["1024", "window"], option=["none", "accumulate_wi_dlhs", "dlhs_transpose_rhs", "chunk_wgrad", "all"]
  )
  def test_ops_gmm_3d_vjp(self, tiles, option):
    """ops.gmm custom VJP (fp8 QArray lhs path as in moe) with 3D lhs / out, alone and with the gmm options.

    accumulate_wi_dlhs: the wi_0 gmm returns its lhs for wi_1 and accumulates its dlhs onto wi_1's (partial_sum);
    dlhs_transpose_rhs: native transpose_rhs in the dlhs gmms; chunk_wgrad: two token chunks where the first
    returns the (quantized) weights it consumed and the second reuses them (tgmm partial_sum).
    """
    accum_wi = option in ("accumulate_wi_dlhs", "all")
    dlhs_tr = option in ("dlhs_transpose_rhs", "all")
    chunk_wgrad = option in ("chunk_wgrad", "all")
    rule = qwix.QtRule(
        weight_qtype=FP8,
        act_qtype=FP8,
        bwd_qtype=FP8,
        weight_calibration_method="fixed,-224,224",
        act_calibration_method="fixed,-224,224",
        bwd_calibration_method="fixed,0.00125",
        disable_channelwise_axes=True,
    )
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(4), 5)
    x = _rand(k0, (M, D), jnp.bfloat16)
    w0 = _rand(k1, (E, D, F), jnp.bfloat16, 0.05)
    w1 = _rand(k4, (E, D, F), jnp.bfloat16, 0.05)
    wo = _rand(k2, (E, F, D), jnp.bfloat16, 0.05)
    douts = [_rand(k, (M, D), jnp.bfloat16, 1e-3) for k in jax.random.split(k3, 2)]
    if tiles == "window":
      # Embed tiles 3584 / 1792 (tile_d0 28 / 14): full-D0 window blocks.
      tiling = (512, 7168, 512, 512, 512, 3584, 512, 1792, 512)
      tiling_o = (512, 512, 3584, 512, 1792, 512, 512, 512, 1792)
    else:
      tiling = (512, 7168, 512, 512, 512, 1024, 512, 1024, 512)
      tiling_o = (512, 512, 1024, 512, 1024, 512, 512, 512, 1024)
    xq2 = qpl.quantize(x, FP8, channelwise_axes=[], calibration_method="fixed,-224,224")
    xq3 = qpl.QArray(qvalue=_to3d(xq2.qvalue), scale=xq2.scale, zero_point=None, qtype=xq2.qtype)
    common = {
        "preferred_element_type": jnp.bfloat16,
        "use_qwix_quantization": True,
        "qwix_rule": rule,
        "use_tokamax_backend": True,
        "use_gmm_v2": True,
        "use_dlhs_transpose_rhs": dlhs_tr,
        "fuse_dlhs_scale": accum_wi,
    }

    def chunk(xq, w0, w1, wo, is3d, return_rhs):
      def mm(lhs, w, tiling, **kw):
        out = ops.gmm(lhs, w, self.gs, tiling=tiling, return_rhs=return_rhs, **kw, **common)
        return out if return_rhs else (out, w)

      x1 = xq
      if accum_wi:
        res = ops.gmm(xq, w0, self.gs, tiling=tiling, return_lhs=True, return_rhs=return_rhs, **common)
        h0, x1, w0 = res if return_rhs else (*res, w0)
      else:
        h0, w0 = mm(xq, w0, tiling)
      h1, w1 = mm(x1, w1, tiling)
      h = (jax.nn.silu(h0.astype(jnp.float32)) * h1.astype(jnp.float32)).astype(jnp.bfloat16)
      out, wo = mm(h, wo, tiling_o, out_is_3d=is3d)
      return out, (w0, w1, wo)

    def fwd(xq, w0, w1, wo, is3d):
      if not chunk_wgrad:
        return (chunk(xq, w0, w1, wo, is3d, False)[0],)
      out_a, (w0, w1, wo) = chunk(xq, w0, w1, wo, is3d, True)
      out_b, _ = chunk(xq, w0, w1, wo, is3d, False)
      return out_a, out_b

    def run(xq, is3d):
      def f(xq, w0, w1, wo):
        outs, vjp = jax.vjp(lambda *a: fwd(*a, is3d), xq, w0, w1, wo)
        cts = tuple(_to3d(d) if is3d else d for d in douts[: len(outs)])
        return outs, vjp(cts)

      return jax.jit(f)(xq, w0, w1, wo)

    outs2, (dx2, dw02, dw12, dwo2) = run(xq2, False)
    outs3, (dx3, dw03, dw13, dwo3) = run(xq3, True)
    for out3, out2 in zip(outs3, outs2):
      self.assertEqual(out3.shape, (M, D // 128, 128))
      _assert_close_chunked_k(out3.reshape(M, D), out2)
    # The wi fwd (tile_k 7168, fp8 x fp8) takes the chunked path; its rounding propagates to all outputs.
    dx3_deq = dx3.qvalue.reshape(M, D).astype(jnp.float32) * dx3.scale
    dx2_deq = dx2.qvalue.astype(jnp.float32) * dx2.scale
    _assert_close_chunked_k(dx3_deq, dx2_deq, rtol=0.15)  # fp8 qvalue: ~1 e4m3 ulp.
    _assert_close_chunked_k(dw03, dw02)
    _assert_close_chunked_k(dw13, dw12)
    _assert_close_chunked_k(dwo3, dwo2)
    self.assertGreater(float(jnp.max(jnp.abs(dx2.qvalue.astype(jnp.float32)))), 0.0)

  def test_tc_sort_unsort_3d(self):
    """TC ragged sort -> weights on activation -> unsort: keep_3d matches 2D (values and grads)."""
    n_tok, topk, n_exp, buf = 512, 8, 16, 3072  # buf < n_tok * topk: truncated buffer.
    k0, k1, k2 = jax.random.split(jax.random.key(5), 3)
    h = _rand(k0, (n_tok, D), jnp.bfloat16)
    idx = jax.random.randint(k1, (n_tok, topk), 0, n_exp, jnp.int32)
    w = jax.nn.softmax(jax.random.normal(k2, (n_tok, topk), jnp.float32), axis=-1)

    def f(h, w, keep_3d):
      x, gs, _, routing = ragged_sort_tc.ring_ragged_sort_tc(
          h, idx, n_exp, topk, "ep", 1, buf, flatten_block_size=256, keep_3d=keep_3d
      )
      w_rows = ragged_sort_tc.tc_buffer_row_weights(routing, w.reshape(-1), x.shape[0])
      w_rows = w_rows.reshape(-1, *([1] * (x.ndim - 1)))
      y = (x.astype(jnp.float32) * w_rows).astype(x.dtype)
      out = ragged_sort_tc.ring_ragged_unsort_tc(y, routing, topk, w.reshape(-1), flatten_block_size=256, prescaled=True)
      return out, gs

    def run(keep_3d):
      def g(h, w):
        (out, gs), vjp = jax.vjp(lambda a, b: f(a, b, keep_3d), h, w)
        ct = jnp.ones_like(out) * 0.01
        return out, gs, vjp((ct, np.zeros(gs.shape, jax.dtypes.float0)))

      return jax.jit(g)(h, w)

    out2, gs2, (dh2, dw2) = run(False)
    out3, gs3, (dh3, dw3) = run(True)
    np.testing.assert_array_equal(np.asarray(gs3), np.asarray(gs2))
    np.testing.assert_array_equal(_f32(out3), _f32(out2))
    np.testing.assert_array_equal(_f32(dh3), _f32(dh2))
    # dw reduces ct * x over the hidden dim, which XLA orders differently for [cap, 56, 128] vs
    # [cap, 7168] (in moe the weights multiply the 2D (cap, mlp) activation, so this is test-only).
    np.testing.assert_allclose(np.asarray(dw3), np.asarray(dw2), rtol=1e-5, atol=1e-6)
    self.assertGreater(float(jnp.max(jnp.abs(out2.astype(jnp.float32)))), 0.0)


K_TAIL_G, K_TAIL_M, K_TAIL_N = 8, 512, 256
# sum 211 < K_TAIL_M: padding rows, one empty group.
K_TAIL_GROUP_SIZES = np.array([26, 31, 0, 30, 29, 28, 32, 35], np.int32)


def _poison_hbm():
  """Fills and frees ~2 GiB of HBM with NaN, so stale memory the kernel reads past an operand's end is NaN."""
  xs = [jnp.full((64, 1024, 1024), jnp.nan, jnp.float32) for _ in range(8)]
  jax.block_until_ready(xs)
  del xs


def _k_tail_reference(lhs, rhs, transpose_rhs):
  lhs, rhs = np.asarray(lhs, np.float32), np.asarray(rhs, np.float32)
  if transpose_rhs:
    rhs = rhs.swapaxes(1, 2)
  out = np.zeros((K_TAIL_M, K_TAIL_N), np.float32)
  start = 0
  for g, size in enumerate(K_TAIL_GROUP_SIZES):
    out[start : start + size] = lhs[start : start + size] @ rhs[g]
    start += size
  return out


class GmmV2KTailTest(parameterized.TestCase):
  """tile_k does not divide size_k: the partial last K tile must mask lhs as well as rhs (stale VMEM is not zero)."""

  @parameterized.product(
      size_k_tile_k=((256, 384), (384, 256), (640, 512)),
      transpose_rhs=(False, True),
  )
  def test_partial_k_tile_matches_reference(self, size_k_tile_k, transpose_rhs):
    size_k, tile_k = size_k_tile_k
    lhs = jax.random.normal(jax.random.PRNGKey(0), (K_TAIL_M, size_k), jnp.float32).astype(jnp.bfloat16)
    rhs_shape = (K_TAIL_G, K_TAIL_N, size_k) if transpose_rhs else (K_TAIL_G, size_k, K_TAIL_N)
    rhs = (jax.random.normal(jax.random.PRNGKey(1), rhs_shape, jnp.float32) * 0.05).astype(jnp.bfloat16)
    group_sizes = jnp.asarray(K_TAIL_GROUP_SIZES)
    tiles = gmm_v2.TileSizes(tile_m=128, tile_k=tile_k, tile_n=K_TAIL_N)

    _poison_hbm()
    out = jax.jit(
        lambda lhs, rhs: gmm_v2.gmm_v2(
            lhs, rhs, group_sizes, tile_info=tiles, preferred_element_type=jnp.float32, transpose_rhs=transpose_rhs
        )
    )(lhs, rhs)

    num_valid = int(K_TAIL_GROUP_SIZES.sum())
    np.testing.assert_allclose(
        np.asarray(out)[:num_valid], _k_tail_reference(lhs, rhs, transpose_rhs)[:num_valid], rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
