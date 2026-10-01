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

"""TPU tests: gmm_v2 flat_rhs / in-kernel transpose_rhs match the default kernel exactly."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
import pytest

pytestmark = pytest.mark.tpu_only  # Pallas TPU kernels.

FP8 = jnp.float8_e4m3fn
E5M2 = jnp.float8_e5m2
M, D, F, E = 2048, 7168, 512, 4
GROUP_SIZES = np.array([300, 0, 517, 700], np.int32)  # sum 1517 < M: padding rows.


def _rand(key, shape, dtype, scale=1.0):
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _f32(x):
  return np.asarray(x.astype(jnp.float32))


class GmmWeightLayoutTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")
    self.gs = jnp.asarray(GROUP_SIZES)

  @parameterized.product(
      case=["wi_fwd_fp8", "wi_fwd_fp8_tk1024", "wo_fwd_quant_lhs", "bf16"],
  )
  def test_flat_rhs(self, case):
    """Flat [E*K, N] rhs view == 3D rhs (fp8 weights with scales, bf16)."""
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    lhs_scale = None
    if case.startswith("wi_fwd_fp8"):
      # fp8 tokens x fp8 weight [E, D, F], per-channel scale.
      lhs = _rand(k0, (M, D), FP8, 4.0)
      rhs = _rand(k1, (E, D, F), FP8, 4.0)
      rhs_scale = jnp.abs(_rand(k2, (E, 1, 1, F), jnp.float32)) * 0.01
      tk = 7168 if case == "wi_fwd_fp8" else 1024
      tiles = gmm_v2.TileSizes(tile_m=512, tile_k=tk, tile_n=512)
      quantize_lhs = False
    elif case == "wo_fwd_quant_lhs":
      # bf16 act quantized in-kernel with a fixed scale x fp8 weight [E, F, D].
      lhs = _rand(k0, (M, F), jnp.bfloat16, 4.0)
      rhs = _rand(k1, (E, F, D), FP8, 4.0)
      rhs_scale = jnp.full((E, 1, 1, D), 0.01, jnp.float32)
      lhs_scale = jnp.full((1, 1), 0.02, jnp.float32)
      tiles = gmm_v2.TileSizes(tile_m=512, tile_k=F, tile_n=1792)
      quantize_lhs = True
    else:
      lhs = _rand(k0, (M, D), jnp.bfloat16)
      rhs = _rand(k1, (E, D, F), jnp.bfloat16, 0.05)
      rhs_scale = None
      tiles = gmm_v2.TileSizes(tile_m=512, tile_k=1792, tile_n=512)
      quantize_lhs = False
    f = functools.partial(
        gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=quantize_lhs
    )
    want = jax.jit(f)(lhs, rhs, self.gs, rhs_scale, lhs_scale=lhs_scale)
    got = jax.jit(functools.partial(f, flat_rhs=True))(lhs, rhs, self.gs, rhs_scale, lhs_scale=lhs_scale)
    np.testing.assert_array_equal(_f32(got), _f32(want))
    self.assertGreater(float(np.max(np.abs(_f32(want)))), 0.0)

  @parameterized.product(
      case=["wi_dlhs", "wo_dlhs"],
      dtype=["float8", "bfloat16"],
      flat=[False, True],
  )
  def test_transpose_rhs(self, case, dtype, flat):
    """dlhs = dout @ W^T with W read as [E, K, N] and transposed in VMEM == HBM swapaxes."""
    k0, k1 = jax.random.split(jax.random.key(1))
    if case == "wi_dlhs":
      # W [E, D, F]; dout [M, F] -> dx [M, D].
      w_shape, n_in, tiles = (E, D, F), F, gmm_v2.TileSizes(tile_m=512, tile_k=F, tile_n=3584)
    else:
      # W [E, F, D]; dout [M, D] -> dh [M, F].
      w_shape, n_in, tiles = (E, F, D), D, gmm_v2.TileSizes(tile_m=512, tile_k=1792, tile_n=F)
    if dtype == "float8":
      lhs = _rand(k0, (M, n_in), E5M2, 4.0)
      w = _rand(k1, w_shape, FP8, 4.0)
    else:
      lhs = _rand(k0, (M, n_in), jnp.bfloat16)
      w = _rand(k1, w_shape, jnp.bfloat16, 0.05)
    f = functools.partial(
        gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=False
    )
    want = jax.jit(f)(lhs, w.swapaxes(1, 2), self.gs)
    got = jax.jit(functools.partial(f, transpose_rhs=True, flat_rhs=flat))(lhs, w, self.gs)
    np.testing.assert_array_equal(_f32(got), _f32(want))
    self.assertGreater(float(np.max(np.abs(_f32(want)))), 0.0)

  def test_transpose_rhs_3d_out(self):
    """wi dlhs in the 3D token path: in-kernel transpose + 3D output."""
    k0, k1 = jax.random.split(jax.random.key(2))
    lhs = _rand(k0, (M, F), E5M2, 4.0)
    w = _rand(k1, (E, D, F), FP8, 4.0)
    tiles = gmm_v2.TileSizes(tile_m=512, tile_k=F, tile_n=1024)
    f = functools.partial(
        gmm_v2.gmm_v2, tile_info=tiles, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=False
    )
    want = jax.jit(f)(lhs, w.swapaxes(1, 2), self.gs)
    got = jax.jit(functools.partial(f, transpose_rhs=True, flat_rhs=True, out_is_3d=True))(lhs, w, self.gs)
    np.testing.assert_array_equal(_f32(got.reshape(M, D)), _f32(want))

  @parameterized.product(bwd_qtype=["float8_e4m3fn", "float8_e5m2"])
  def test_ops_gmm_vjp_weight_layout(self, bwd_qtype):
    """ops.gmm fwd + bwd with flat_rhs + kernel_transpose_dlhs == default (fp8 qwix, as in moe)."""
    import qwix  # pylint: disable=g-import-not-at-top
    import qwix.pallas as qpl  # pylint: disable=g-import-not-at-top

    rule = qwix.QtRule(
        weight_qtype=FP8,
        act_qtype=FP8,
        bwd_qtype=jnp.dtype(bwd_qtype),
        weight_calibration_method="fixed,-224,224",
        act_calibration_method="fixed,-224,224",
        bwd_calibration_method="fixed,0.00125",
        disable_channelwise_axes=True,
    )
    k0, k1, k2, k3 = jax.random.split(jax.random.key(4), 4)
    x = _rand(k0, (M, D), jnp.bfloat16)
    w0 = _rand(k1, (E, D, F), jnp.bfloat16, 0.05)
    wo = _rand(k2, (E, F, D), jnp.bfloat16, 0.05)
    dout = _rand(k3, (M, D), jnp.bfloat16, 1e-3)
    # (fwd m,k,n | dlhs m,k,n | drhs m,k,n)
    tiling = (512, 7168, 512, 512, 512, 3584, 512, 1792, 512)
    tiling_o = (512, 512, 3584, 512, 1792, 512, 512, 512, 1792)
    xq = qpl.quantize(x, FP8, channelwise_axes=[], calibration_method="fixed,-224,224")
    common = dict(
        preferred_element_type=jnp.bfloat16,
        use_qwix_quantization=True,
        qwix_rule=rule,
        use_tokamax_backend=True,
        use_gmm_v2=True,
    )

    def run(weight_layout):
      def fwd(xq, w0, wo):
        h = ops.gmm(xq, w0, self.gs, tiling=tiling, weight_layout=weight_layout, **common)
        h = jax.nn.silu(h.astype(jnp.float32)).astype(jnp.bfloat16)
        return ops.gmm(h, wo, self.gs, tiling=tiling_o, weight_layout=weight_layout, **common)

      def f(xq, w0, wo):
        out, vjp = jax.vjp(fwd, xq, w0, wo)
        return out, vjp(dout)

      return jax.jit(f)(xq, w0, wo)

    out_a, (dx_a, dw0_a, dwo_a) = run(None)
    out_b, (dx_b, dw0_b, dwo_b) = run(ops.WeightLayoutOpts(flat_rhs=True, kernel_transpose_dlhs=True))
    np.testing.assert_array_equal(_f32(out_b), _f32(out_a))
    np.testing.assert_array_equal(_f32(dx_b.qvalue), _f32(dx_a.qvalue))
    np.testing.assert_array_equal(_f32(dw0_b), _f32(dw0_a))
    np.testing.assert_array_equal(_f32(dwo_b), _f32(dwo_a))
    self.assertGreater(float(jnp.max(jnp.abs(dx_a.qvalue.astype(jnp.float32)))), 0.0)

  @parameterized.product(sc_core=[-1, 1])
  def test_hoisted_collect_matches_in_gmm_qag(self, sc_core):
    """collect_quantized_weight once + 2 chunked gmms == gmm's in-kernel QAG per chunk (fwd + grads)."""
    import qwix  # pylint: disable=g-import-not-at-top
    import qwix.pallas as qpl  # pylint: disable=g-import-not-at-top
    from jax.sharding import PartitionSpec as P  # pylint: disable=g-import-not-at-top

    n = len(jax.devices())
    mesh = jax.make_mesh((n,), ("fsdp",), axis_types=(jax.sharding.AxisType.Auto,))
    rule = qwix.QtRule(
        weight_qtype=FP8,
        act_qtype=FP8,
        bwd_qtype=jnp.dtype(E5M2),
        weight_calibration_method="fixed,-224,224",
        act_calibration_method="fixed,-224,224",
        bwd_calibration_method="fixed,0.00125",
        disable_channelwise_axes=True,
    )
    k0, k1, k2, k3 = jax.random.split(jax.random.key(5), 4)
    x = _rand(k0, (2, M, D), jnp.bfloat16)
    w0 = _rand(k1, (E, D, F), jnp.bfloat16, 0.05)
    wo = _rand(k2, (E, F, D), jnp.bfloat16, 0.05)
    dout = _rand(k3, (2, M, D), jnp.bfloat16, 1e-3)
    tiling = (512, 7168, 512, 512, 512, 3584, 512, 1792, 512)
    tiling_o = (512, 512, 3584, 512, 1792, 512, 512, 512, 1792)
    wi_axes, wo_axes = [("fsdp", 1)], [("fsdp", 2)]
    layout = ops.WeightLayoutOpts(sc_collect_core=sc_core, flat_rhs=True, kernel_transpose_dlhs=True)
    common = dict(
        preferred_element_type=jnp.bfloat16,
        use_qwix_quantization=True,
        qwix_rule=rule,
        use_tokamax_backend=True,
        use_gmm_v2=True,
        weight_layout=layout,
    )
    gs = self.gs

    def body(hoist, x, w0, wo):
      if hoist:
        w0 = ops.collect_quantized_weight(w0, wi_axes, rule, sc_core)
        wo = ops.collect_quantized_weight(wo, wo_axes, rule, sc_core)
        a0, ao = [], []
      else:
        a0, ao = wi_axes, wo_axes
      outs = []
      for c in range(2):  # two token chunks sharing the weights
        xq = qpl.quantize(x[c], FP8, channelwise_axes=[], calibration_method="fixed,-224,224")
        h = ops.gmm(xq, w0, gs, tiling=tiling, weight_gather_axes=a0, **common)
        h = jax.nn.silu(h.astype(jnp.float32)).astype(jnp.bfloat16)
        outs.append(ops.gmm(h, wo, gs, tiling=tiling_o, weight_gather_axes=ao, **common))
      return jnp.stack(outs)

    def run(hoist):
      f = jax.shard_map(
          functools.partial(body, hoist),
          mesh=mesh,
          in_specs=(P(), P(None, "fsdp", None), P(None, None, "fsdp")),
          out_specs=P(),
          check_vma=False,
      )

      def g(x, w0, wo):
        out, vjp = jax.vjp(f, x, w0, wo)
        return out, vjp(dout)

      ns = lambda *spec: jax.sharding.NamedSharding(mesh, P(*spec))
      args = (
          jax.device_put(x, ns()),
          jax.device_put(w0, ns(None, "fsdp", None)),
          jax.device_put(wo, ns(None, None, "fsdp")),
      )
      return jax.jit(g)(*args)

    out_a, (dx_a, dw0_a, dwo_a) = run(False)
    out_b, (dx_b, dw0_b, dwo_b) = run(True)
    np.testing.assert_array_equal(_f32(out_b), _f32(out_a))
    np.testing.assert_array_equal(_f32(dx_b), _f32(dx_a))
    for got, want in ((dw0_b, dw0_a), (dwo_b, dwo_a)):
      self.assertEqual(got.dtype, want.dtype)
      self.assertEqual(got.shape, want.shape)
      # Hoisting sums the two chunks' bf16 grads before the reduce-scatter (vs. after); allow
      # that reassociation, but report whether it is bit-exact.
      diff = np.max(np.abs(_f32(got) - _f32(want)))
      print(f"sc_core={sc_core} dw max|diff|={diff} max|w|={np.max(np.abs(_f32(want)))}")
      np.testing.assert_allclose(_f32(got), _f32(want), rtol=2e-2, atol=1e-6 * np.max(np.abs(_f32(want))))


if __name__ == "__main__":
  absltest.main()
