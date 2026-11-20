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

"""CPU dispatch and TPU parity tests for routing 3D-out and 3D-partial_sum GMM to Tokamax."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
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


class Dispatch3dReshapeTest(parameterized.TestCase):
  """CPU-compatible dispatch tests for PR2a 3D-out and 3D-partial_sum GMM routing."""

  @parameterized.named_parameters(
      {"testcase_name": "2d_lhs_3d_out", "lhs_shape": (512, 1024), "out_is_3d": True, "route": "up_gmm"},
      {"testcase_name": "3d_lhs_3d_out_fuser", "lhs_shape": (512, 8, 128), "out_is_3d": True, "route": "fuser_gmm"},
      {"testcase_name": "2d_lhs_3d_ps", "lhs_shape": (512, 1024), "out_is_3d": True, "ps_3d": True, "route": "up_gmm"},
      {"testcase_name": "3d_lhs_3d_ps", "lhs_shape": (512, 8, 128), "out_is_3d": True, "ps_3d": True, "route": "up_gmm"},
      {
          "testcase_name": "2d_only_stays_forked",
          "lhs_shape": (512, 1024),
          "out_is_3d": False,
          "ps_3d": False,
          "route": "fork_gmm",
      },
      {
          "testcase_name": "tail_chunks_stays_forked",
          "lhs_shape": (512, 1024),
          "out_is_3d": True,
          "ps_3d": True,
          "tail_chunks": True,
          "route": "fork_gmm",
      },
      {
          "testcase_name": "fallback_when_fuser_none",
          "lhs_shape": (512, 1024),
          "out_is_3d": True,
          "ps_3d": True,
          "hide_fuser": True,
          "route": "fork_gmm",
      },
  )
  def test_gmm_3d_out_and_partial_sum_dispatch(
      self,
      lhs_shape: tuple[int, ...],
      out_is_3d: bool,
      ps_3d: bool = False,
      tail_chunks: bool = False,
      hide_fuser: bool = False,
      route: str = "fork_gmm",
  ):
    if route != "fork_gmm" and gmm_fuser is None:
      self.skipTest("Requires tokamax >= 0.0.18")
    lhs = jnp.ones(lhs_shape, jnp.bfloat16)
    rhs = jnp.ones((2, 1024, 256), jnp.bfloat16)
    gs = jnp.array([256, 256], jnp.int32)
    out_shape = (512, 2, 128) if out_is_3d else (512, 256)
    ps = jnp.ones(out_shape, jnp.bfloat16) if ps_3d else None
    tiles = forked_gmm_v2.TileSizes(256, 1024, 256)
    tpu_cm = mock.patch.object(ops, "_tpu_vmem_limit_bytes", autospec=True, return_value=None)
    if hide_fuser or gmm_fuser is None:
      fuser_cm = mock.patch.object(ops, "gmm_fuser", None)
      up_cm = mock.patch.object(ops, "tokamax_gmm_v2", None)
    else:
      zero_2d = jnp.zeros((512, 256), jnp.bfloat16)
      fuser_cm = mock.patch.object(gmm_fuser, "_run_gmm_v2_fusible", autospec=True, return_value=zero_2d)
      up_cm = mock.patch.object(ops.tokamax_gmm_v2, "gmm_v2", autospec=True, return_value=zero_2d)
    fork_cm = mock.patch.object(ops.gmm_v2, "gmm_v2", autospec=True, return_value=jnp.ones(out_shape, jnp.bfloat16))

    with tpu_cm, fuser_cm as f_mock, up_cm as u_mock, fork_cm as g_mock:
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
    if route == "fuser_gmm":
      self.assertEqual(f_mock.call_count, 2)
      u_mock.assert_not_called()
      g_mock.assert_not_called()
    elif route == "up_gmm":
      u_mock.assert_called_once()
      self.assertEqual(u_mock.call_args.kwargs["lhs"].shape, (512, 1024))
      if ps_3d:
        self.assertEqual(u_mock.call_args.kwargs["partial_sum"].shape, (512, 256))
      g_mock.assert_not_called()
    else:
      g_mock.assert_called_once()
    self.assertEqual(out.shape, out_shape)


@pytest.mark.tpu_only
class Tpu3dReshapeParityTest(parameterized.TestCase):
  """TPU parity tests: PR2a 3D-out and 3D-partial_sum GMM vs forked kernel."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU only")

  def test_3d_out_bf16_bit_exact_vs_forked(self):
    m, k, n, g = 2048, 1024, 2048, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1 = jax.random.split(jax.random.key(31), 2)
    x, w = _rand(k0, (m, k), jnp.bfloat16), _rand(k1, (g, k, n), jnp.bfloat16, 0.05)
    tiling = (512, 1024, 1024) * 3
    run_ops = jax.jit(
        lambda a, b: ops.gmm(a, b, gs, jnp.bfloat16, tiling, use_tokamax_backend=True, use_gmm_v2=True, out_is_3d=True)
    )
    y_new = run_ops(x, w)
    with mock.patch.object(ops, "gmm_fuser", None):
      y_fork = run_ops(x, w)
    self.assertEqual(y_new.shape, (m, n // 128, 128))
    np.testing.assert_array_equal(y_new.astype(jnp.float32), y_fork.astype(jnp.float32))

  def test_3d_out_fp8_matches_forked(self):
    m, k, n, g = 2048, 1024, 2048, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1 = jax.random.split(jax.random.key(32), 2)
    xq, wq = _q_fp8(_rand(k0, (m, k), jnp.bfloat16), []), _q_fp8(_rand(k1, (g, k, n), jnp.bfloat16, 0.05), [0, 2])
    lhs_q = qpl.QArray(qvalue=xq.qvalue, scale=xq.scale, zero_point=None, qtype=xq.qtype)
    rhs_q = qpl.QArray(qvalue=wq.qvalue, scale=wq.scale, zero_point=None, qtype=jnp.float8_e4m3fn)
    tiling = (512, 1024, 1024) * 3
    run_ops = jax.jit(
        lambda a, b: ops.gmm(a, b, gs, jnp.bfloat16, tiling, use_tokamax_backend=True, use_gmm_v2=True, out_is_3d=True)
    )
    y_new = run_ops(lhs_q, rhs_q)
    with mock.patch.object(ops, "gmm_fuser", None):
      y_fork = run_ops(lhs_q, rhs_q)
    got, want = y_new.astype(jnp.float32), y_fork.astype(jnp.float32)
    self.assertEqual(y_new.shape, (m, n // 128, 128))
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(want))))
    self.assertLessEqual(float(np.mean(np.asarray(got) != np.asarray(want))), 1e-3)

  def test_3d_partial_sum_dlhs_bf16_bit_exact_vs_forked(self):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2 = jax.random.split(jax.random.key(33), 3)
    dout = _rand(k0, (m, f), jnp.bfloat16)
    wt, ps_gmm = _rand(k1, (g, d, f), jnp.bfloat16, 0.05), _to3d(_rand(k2, (m, d), jnp.bfloat16, 0.1))
    tiling = (512, 1024, 1024) * 3
    run_dlhs = jax.jit(
        lambda a, b, p: ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
            a,
            b,
            gs,
            None,
            jnp.bfloat16,
            tiling,
            False,
            p,
            use_dlhs_transpose_rhs=True,
            out_is_3d=True,
            fuse_3d_dlhs_ps=True,
        )[0]
    )
    dlhs_new = run_dlhs(dout, wt, ps_gmm)
    with mock.patch.object(ops, "gmm_fuser", None):
      dlhs_fork = run_dlhs(dout, wt, ps_gmm)
    np.testing.assert_array_equal(dlhs_new.astype(jnp.float32), dlhs_fork.astype(jnp.float32))

  def test_3d_partial_sum_dlhs_fp8_matches_forked(self):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2 = jax.random.split(jax.random.key(34), 3)
    dq = _q_fp8(_rand(k0, (m, f), jnp.bfloat16), [])
    dout_q = qpl.QArray(qvalue=dq.qvalue, scale=dq.scale, zero_point=None, qtype=dq.qtype)
    wt, ps_gmm = _rand(k1, (g, d, f), jnp.bfloat16, 0.05), _to3d(_rand(k2, (m, d), jnp.bfloat16, 0.1))
    tiling = (512, 1024, 1024) * 3
    run_dlhs = jax.jit(
        lambda a, b, p: ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
            a,
            b,
            gs,
            None,
            jnp.bfloat16,
            tiling,
            False,
            p,
            use_dlhs_transpose_rhs=True,
            out_is_3d=True,
            fuse_3d_dlhs_ps=True,
        )[0]
    )
    dlhs_new = run_dlhs(dout_q, wt, ps_gmm).astype(jnp.float32)
    with mock.patch.object(ops, "gmm_fuser", None):
      dlhs_fork = run_dlhs(dout_q, wt, ps_gmm).astype(jnp.float32)
    np.testing.assert_allclose(dlhs_new, dlhs_fork, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(dlhs_fork))))

  @parameterized.parameters("bf16", "fp8_accum_chunk")
  def test_ops_gmm_3d_vjp_matches_forked(self, mode: str):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(35), 5)
    x = _rand(k0, (m, d), jnp.bfloat16)
    w0 = _rand(k1, (g, d, f), jnp.bfloat16, 0.05)
    w1, wo = _rand(k4, (g, d, f), jnp.bfloat16, 0.05), _rand(k2, (g, f, d), jnp.bfloat16, 0.05)
    dout = _to3d(_rand(k3, (m, d), jnp.bfloat16, 1e-2))
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
      def f(x_in, w0_in, w1_in, wo_in):
        def fwd(a, b, c, e):
          h0, x1 = ops.gmm(a, b, gs, tiling=tiling, return_lhs=True, **common)
          h1 = ops.gmm(x1, c, gs, tiling=tiling, **common)
          h = (jax.nn.silu(h0.astype(jnp.float32)) * h1.astype(jnp.float32)).astype(jnp.bfloat16)
          return ops.gmm(h, e, gs, tiling=tiling, out_is_3d=True, **common)

        out, vjp = jax.vjp(fwd, x_in, w0_in, w1_in, wo_in)
        return out, vjp(dout)

      return jax.jit(f)(xq3, w0, w1, wo)

    out_new, grads_new = run_vjp()
    with mock.patch.object(ops, "gmm_fuser", None):
      out_fork, grads_fork = run_vjp()
    np.testing.assert_array_equal(out_new.astype(jnp.float32), out_fork.astype(jnp.float32))
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
