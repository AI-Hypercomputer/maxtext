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

"""CPU dispatch and TPU parity tests for routing 3D-lhs GMM through Tokamax gmm_fuser."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as forked_gmm_v2
import numpy as np
import pytest
import qwix
import qwix.pallas as qpl

try:  # decoupled-requirements.txt pins tokamax==0.0.12, before gmm_fuser existed.
  from tokamax._src.ops.gmm_v2 import gmm_fuser  # pylint: disable=g-import-not-at-top
except ImportError:
  gmm_fuser = None


def _to3d(x: jax.Array) -> jax.Array:
  return x.reshape(x.shape[0], -1, 128)


def _rand(key: jax.Array, shape: tuple[int, ...], dtype, scale: float = 0.25) -> jax.Array:
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _q_fp8(x: jax.Array, axes: list[int]):
  return qpl.quantize(x, jnp.float8_e4m3fn, channelwise_axes=axes, calibration_method="fixed,-224,224")


class Dispatch3dLhsFuserTest(parameterized.TestCase):
  """CPU-compatible dispatch and fallback tests for ops._call_gmm_v2."""

  @parameterized.named_parameters(
      {"testcase_name": "3d_lhs_fuser", "lhs_shape": (512, 8, 128), "tile_k": 1024, "route": "fuser"},
      {
          "testcase_name": "3d_lhs_fuser_96pct_vmem",
          "lhs_shape": (512, 8, 128),
          "tile_k": 1024,
          "route": "fuser",
          "vmem_mib": 64,
      },
      {"testcase_name": "out_is_3d", "lhs_shape": (512, 8, 128), "tile_k": 1024, "out_is_3d": True, "route": "fuser"},
      {
          "testcase_name": "partial_sum",
          "lhs_shape": (512, 8, 128),
          "tile_k": 1024,
          "has_partial_sum": True,
          "route": "upstream",
      },
      {
          "testcase_name": "both_3d_out_and_ps",
          "lhs_shape": (512, 8, 128),
          "tile_k": 1024,
          "out_is_3d": True,
          "has_partial_sum": True,
          "route": "upstream",
      },
      {"testcase_name": "2d_lhs", "lhs_shape": (512, 1024), "tile_k": 1024},
      {"testcase_name": "unaligned_window_tile", "lhs_shape": (512, 56, 128), "tile_k": 1792},
      {"testcase_name": "fuser_unavailable", "lhs_shape": (512, 8, 128), "tile_k": 1024, "hide_fuser": True},
      {"testcase_name": "heuristic_tiling", "lhs_shape": (512, 8, 128), "tile_k": 1024, "heuristic": True},
      {"testcase_name": "tail_chunks", "lhs_shape": (512, 8, 128), "tile_k": 1024, "tail_chunks": True},
  )
  def test_call_gmm_v2_dispatch(
      self,
      lhs_shape: tuple[int, ...],
      tile_k: int,
      route: str = "fork",
      out_is_3d: bool = False,
      has_partial_sum: bool = False,
      hide_fuser: bool = False,
      heuristic: bool = False,
      tail_chunks: bool = False,
      vmem_mib: int | None = None,
  ):
    if route != "fork" and gmm_fuser is None:
      self.skipTest("Requires tokamax >= 0.0.18")
    k_dim = lhs_shape[1] * 128 if len(lhs_shape) == 3 else lhs_shape[1]
    out_shape = (512, 2, 128) if out_is_3d else (512, 256)
    lhs, rhs = jnp.ones(lhs_shape, jnp.bfloat16), jnp.ones((2, k_dim, 256), jnp.bfloat16)
    gs = jnp.array([256, 256], jnp.int32)
    ps = jnp.ones(out_shape, jnp.bfloat16) if has_partial_sum else None
    tiles = forked_gmm_v2.calculate_tiling if heuristic else forked_gmm_v2.TileSizes(256, tile_k, 256)
    if vmem_mib is not None:
      fake_tpu = mock.Mock(vmem_capacity_bytes=vmem_mib * 1024 * 1024)
      tpu_cm = mock.patch.object(ops.pltpu, "get_tpu_info", autospec=True, return_value=fake_tpu)
    else:
      tpu_cm = mock.patch.object(ops, "_tpu_vmem_limit_bytes", autospec=True, return_value=None)
    if hide_fuser or gmm_fuser is None:
      fuser_cm = mock.patch.object(ops, "gmm_fuser", None)
    else:
      fuser_cm = mock.patch.object(
          gmm_fuser, "_run_gmm_v2_fusible", autospec=True, return_value=jnp.zeros((512, 256), jnp.bfloat16)
      )
    up_cm = (
        mock.patch.object(ops.tokamax_gmm_v2, "gmm_v2", autospec=True, return_value=jnp.zeros((512, 256), jnp.bfloat16))
        if ops.tokamax_gmm_v2 is not None
        else mock.patch.object(ops, "tokamax_gmm_v2", None)
    )

    with tpu_cm, fuser_cm as f_mock, up_cm as u_mock:
      with mock.patch.object(
          ops.gmm_v2, "gmm_v2", autospec=True, return_value=jnp.ones(out_shape, jnp.bfloat16)
      ) as g_mock:
        out = ops._call_gmm_v2(  # pylint: disable=protected-access
            lhs=lhs,
            rhs=rhs,
            group_sizes=gs,
            partial_sum=ps,
            tile_info=tiles,
            preferred_element_type=jnp.bfloat16,
            out_is_3d=out_is_3d,
            tail_chunks=tail_chunks,
        )
    if route == "fuser":
      self.assertEqual(f_mock.call_count, 2)
      exp_vmem = int(0.96 * vmem_mib * 1024 * 1024) if vmem_mib else None
      self.assertEqual(f_mock.call_args.kwargs["vmem_limit_bytes"], exp_vmem)
      g_mock.assert_not_called()
    elif route == "upstream":
      u_mock.assert_called_once()
      g_mock.assert_not_called()
    else:
      if f_mock is not None:
        f_mock.assert_not_called()
      if u_mock is not None:
        u_mock.assert_not_called()
      g_mock.assert_called_once()
    self.assertEqual(out.shape, out_shape)


@pytest.mark.tpu_only
class Tpu3dLhsFuserParityTest(parameterized.TestCase):
  """TPU parity tests: 3D-lhs gmm_fuser vs forked pallas_mosaic_tpu_v2_gmm_kernel."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU only")

  def test_3d_lhs_bf16_bit_exact_vs_forked(self):
    m, k, n, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2 = jax.random.split(jax.random.key(11), 3)
    lhs_3d = _to3d(_rand(k0, (m, k), jnp.bfloat16))
    w, wt = _rand(k1, (g, k, n), jnp.bfloat16, 0.05), _rand(k2, (g, n, k), jnp.bfloat16, 0.05)
    tiling, fork_tiles = (512, 1024, n) * 3, forked_gmm_v2.TileSizes(512, 1024, n)

    run_ops = jax.jit(lambda a, b: ops.gmm(a, b, gs, jnp.bfloat16, tiling, use_tokamax_backend=True, use_gmm_v2=True))
    y_fork = jax.jit(
        lambda a, b: forked_gmm_v2.gmm_v2(a, b, gs, tile_info=fork_tiles, preferred_element_type=jnp.bfloat16)
    )(lhs_3d, w)
    np.testing.assert_array_equal(run_ops(lhs_3d, w).astype(jnp.float32), y_fork.astype(jnp.float32))

    dlhs_fuser, _ = jax.jit(
        lambda d, r: ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
            d, r, gs, None, jnp.bfloat16, tiling, False, use_dlhs_transpose_rhs=True
        )
    )(lhs_3d, wt)
    dlhs_fork = jax.jit(
        lambda d, r: forked_gmm_v2.gmm_v2(
            d, r, gs, tile_info=fork_tiles, preferred_element_type=jnp.bfloat16, transpose_rhs=True
        )
    )(lhs_3d, wt)
    np.testing.assert_array_equal(dlhs_fuser.astype(jnp.float32), dlhs_fork.astype(jnp.float32))

  def test_3d_lhs_fp8_matches_forked(self):
    high_vmem = pltpu.get_tpu_info().vmem_capacity_bytes >= 64 * 1024 * 1024
    m, k, n, g, tile_k = (2048, 7168, 2048, 16, 7168) if high_vmem else (2048, 2048, 1024, 4, 1024)
    gs = jnp.full((g,), m // g, jnp.int32)
    k0, k1 = jax.random.split(jax.random.key(12), 2)
    xq, wq = _q_fp8(_rand(k0, (m, k), jnp.bfloat16), []), _q_fp8(_rand(k1, (g, k, n), jnp.bfloat16, 0.05), [0, 2])
    lhs_3d = qpl.QArray(qvalue=_to3d(xq.qvalue), scale=xq.scale, zero_point=None, qtype=xq.qtype)
    rhs_in = qpl.QArray(qvalue=wq.qvalue, scale=wq.scale, zero_point=None, qtype=jnp.float8_e4m3fn)
    tiling = (512, tile_k, n) * 3

    run_ops = jax.jit(lambda a, b: ops.gmm(a, b, gs, jnp.bfloat16, tiling, use_tokamax_backend=True, use_gmm_v2=True))
    y_fuser = run_ops(lhs_3d, rhs_in)
    with mock.patch.object(ops, "gmm_fuser", None):
      y_fork = run_ops(lhs_3d, rhs_in)
    got, want = y_fuser.astype(jnp.float32), y_fork.astype(jnp.float32)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(want))))
    self.assertLessEqual(float(np.mean(np.asarray(got) != np.asarray(want))), 1e-3)

  @parameterized.parameters("bf16", "fp8_accum_chunk")
  def test_ops_gmm_3d_lhs_vjp_matches_forked(self, mode: str):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(22), 5)
    x = _rand(k0, (m, d), jnp.bfloat16)
    w0, w1 = _rand(k1, (g, d, f), jnp.bfloat16, 0.05), _rand(k4, (g, d, f), jnp.bfloat16, 0.05)
    wo = _rand(k2, (g, f, d), jnp.bfloat16, 0.05)
    douts = [_rand(k, (m, d), jnp.bfloat16, 1e-2) for k in jax.random.split(k3, 2)]
    tiling, use_qwix = (512, 1024, 1024) * 3, mode == "fp8_accum_chunk"
    rule = None
    if use_qwix:
      rule = qwix.QtRule(
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          weight_calibration_method="fixed,-224,224",
          act_calibration_method="fixed,-224,224",
          bwd_calibration_method="fixed,0.00125",
          disable_channelwise_axes=True,
      )
      xq = _q_fp8(x, [])
      xq3 = qpl.QArray(qvalue=_to3d(xq.qvalue), scale=xq.scale, zero_point=None, qtype=xq.qtype)
    else:
      xq3 = _to3d(x)
    common = {
        "preferred_element_type": jnp.bfloat16,
        "use_qwix_quantization": use_qwix,
        "qwix_rule": rule,
        "use_gmm_v2": True,
        "use_tokamax_backend": True,
        "use_dlhs_transpose_rhs": True,
        "fuse_dlhs_scale": True,
        "fuse_3d_dlhs_ps": True,
    }

    def run_vjp():
      def chunk(x_in, w0_in, w1_in, wo_in, ret_rhs):
        def mm(lhs, w, **kw):
          res = ops.gmm(lhs, w, gs, tiling=tiling, return_rhs=ret_rhs, **kw, **common)
          return res if ret_rhs else (res, w)

        res0 = ops.gmm(x_in, w0_in, gs, tiling=tiling, return_lhs=True, return_rhs=ret_rhs, **common)
        h0, x1, w0_out = res0 if ret_rhs else (*res0, w0_in)
        h1, w1_out = mm(x1, w1_in)
        h = (jax.nn.silu(h0.astype(jnp.float32)) * h1.astype(jnp.float32)).astype(jnp.bfloat16)
        out, wo_out = mm(h, wo_in, out_is_3d=True)
        return out, (w0_out, w1_out, wo_out)

      def f(x_in, w0_in, w1_in, wo_in):
        def fwd(a, b, c, e):
          out_a, (wq0, wq1, wqo) = chunk(a, b, c, e, True)
          return out_a, chunk(a, wq0, wq1, wqo, False)[0]

        outs, vjp = jax.vjp(fwd, x_in, w0_in, w1_in, wo_in)
        return outs, vjp(tuple(_to3d(c) for c in douts))

      return jax.jit(f)(xq3, w0, w1, wo)

    outs_new, grads_new = run_vjp()
    with mock.patch.object(ops, "gmm_fuser", None):
      outs_fork, grads_fork = run_vjp()
    for o_new, o_fork in zip(outs_new, outs_fork):
      np.testing.assert_array_equal(o_new.astype(jnp.float32), o_fork.astype(jnp.float32))
    if use_qwix:
      dx_n = grads_new[0].qvalue.astype(jnp.float32) * grads_new[0].scale
      dx_f = grads_fork[0].qvalue.astype(jnp.float32) * grads_fork[0].scale
      np.testing.assert_allclose(dx_n, dx_f, rtol=1e-2, atol=1e-2)
      for g_n, g_f in zip(grads_new[1:], grads_fork[1:]):
        np.testing.assert_allclose(g_n.astype(jnp.float32), g_f.astype(jnp.float32), rtol=1e-2, atol=1e-2)
    else:
      for g_n, g_f in zip(grads_new, grads_fork):
        np.testing.assert_array_equal(g_n.astype(jnp.float32), g_f.astype(jnp.float32))


if __name__ == "__main__":
  absltest.main()
