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

"""CPU dispatch and TPU parity tests for routing 3D DRHS TGMM through upstream Tokamax."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from maxtext.kernels.megablox import ops
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


class DispatchTgmm3dTest(parameterized.TestCase):
  """CPU-compatible dispatch tests for PR2b ops._call_tgmm_v2."""

  @parameterized.named_parameters(
      {"testcase_name": "3d_lhs_fuser", "lhs_3d": True, "route": "fuser_tgmm"},
      {"testcase_name": "3d_rhs_fuser", "rhs_3d": True, "route": "fuser_tgmm"},
      {"testcase_name": "3d_lhs_partial_sum", "lhs_3d": True, "has_ps": True, "route": "up_tgmm"},
      {"testcase_name": "3d_window_tile", "lhs_3d": True, "tile_k": 1792, "k_dim": 7168, "route": "up_tgmm"},
      {"testcase_name": "heuristic_tiling", "lhs_3d": True, "heuristic": True, "route": "up_tgmm"},
      {"testcase_name": "2d_stays_forked", "route": "fork_tgmm"},
      {"testcase_name": "fallback_when_fuser_none", "lhs_3d": True, "hide_fuser": True, "route": "fork_tgmm"},
  )
  def test_tgmm_3d_dispatch(
      self,
      lhs_3d: bool = False,
      rhs_3d: bool = False,
      has_ps: bool = False,
      tile_k: int = 1024,
      k_dim: int = 1024,
      heuristic: bool = False,
      hide_fuser: bool = False,
      route: str = "fork_tgmm",
  ):
    if route != "fork_tgmm" and gmm_fuser is None:
      self.skipTest("Requires tokamax >= 0.0.18")
    lhs = jnp.ones((512, k_dim // 128, 128) if lhs_3d else (512, k_dim), jnp.bfloat16)
    rhs = jnp.ones((512, 2, 128) if rhs_3d else (512, 256), jnp.bfloat16)
    gs = jnp.array([256, 256], jnp.int32)
    ps = jnp.ones((2, k_dim, 256), jnp.bfloat16) if has_ps else None
    tiling = (256, 1024, 256, 256, 1024, 256, 256, tile_k, 256)
    tpu_cm = mock.patch.object(ops, "_tpu_vmem_limit_bytes", autospec=True, return_value=None)
    if hide_fuser or gmm_fuser is None:
      fuser_cm = mock.patch.object(ops, "gmm_fuser", None)
      up_cm = mock.patch.object(ops, "tokamax_tgmm_v2", None)
    else:
      zero_out = jnp.zeros((2, k_dim, 256), jnp.bfloat16)
      fuser_cm = mock.patch.object(gmm_fuser, "_run_tgmm_v2_fusible", autospec=True, return_value=zero_out)
      up_cm = mock.patch.object(ops.tokamax_tgmm_v2, "tgmm_v2", autospec=True, return_value=zero_out)
    fork_cm = mock.patch.object(
        ops.tgmm_v2, "tgmm_v2", autospec=True, return_value=jnp.ones((2, k_dim, 256), jnp.bfloat16)
    )

    with tpu_cm, fuser_cm as f_mock, up_cm as u_mock, fork_cm as g_mock:
      out = ops._call_tgmm_v2(  # pylint: disable=protected-access
          lhs=lhs,
          rhs=rhs,
          group_sizes=gs,
          num_actual_groups=2,
          rhs_scale=None,
          group_offset=None,
          partial_sum=ps,
          preferred_element_type=jnp.bfloat16,
          tiling=tiling,
          use_gmm_v2_heuristic_tiling=heuristic,
      )
    if route == "fuser_tgmm":
      self.assertEqual(f_mock.call_count, 2)
      u_mock.assert_not_called()
      g_mock.assert_not_called()
    elif route == "up_tgmm":
      u_mock.assert_called_once()
      self.assertEqual(u_mock.call_args.kwargs["lhs"].shape, (512, k_dim))
      g_mock.assert_not_called()
    else:
      g_mock.assert_called_once()
    self.assertEqual(out.shape, (2, k_dim, 256))


@pytest.mark.tpu_only
class TpuTgmm3dParityTest(parameterized.TestCase):
  """TPU parity tests: PR2b 3D DRHS TGMM (fuser + partial_sum) vs forked kernel."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU only")

  def test_tgmm_3d_fuser_bf16_bit_exact_vs_forked(self):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1 = jax.random.split(jax.random.key(41), 2)
    dout = _rand(k0, (m, f), jnp.bfloat16)
    lhs3 = _to3d(_rand(k1, (m, d), jnp.bfloat16))
    tiling = (512, 1024, 1024) * 3
    run_drhs = jax.jit(
        lambda c, a: ops._drhs_run_tokamax_v2(c, a, gs, None, g, jnp.float32, tiling, False)  # pylint: disable=protected-access
    )
    drhs_new = run_drhs(dout, lhs3)
    with mock.patch.object(ops, "tokamax_tgmm_v2", None):
      drhs_fork = run_drhs(dout, lhs3)
    np.testing.assert_array_equal(drhs_new, drhs_fork)

  def test_tgmm_3d_fuser_fp8_matches_forked(self):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1 = jax.random.split(jax.random.key(42), 2)
    dq = _q_fp8(_rand(k0, (m, f), jnp.bfloat16), [])
    dout_q = qpl.QArray(qvalue=dq.qvalue, scale=dq.scale, zero_point=None, qtype=dq.qtype)
    lhs3_q = _to3d(_q_fp8(_rand(k1, (m, d), jnp.bfloat16), []).qvalue)
    tiling = (512, 1024, 1024) * 3
    run_drhs = jax.jit(
        lambda c, a: ops._drhs_run_tokamax_v2(c, a, gs, None, g, jnp.float32, tiling, False)  # pylint: disable=protected-access
    )
    drhs_new = run_drhs(dout_q, lhs3_q)
    with mock.patch.object(ops, "tokamax_tgmm_v2", None):
      drhs_fork = run_drhs(dout_q, lhs3_q)
    np.testing.assert_allclose(drhs_new, drhs_fork, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(drhs_fork))))

  def test_tgmm_3d_partial_sum_bf16_bit_exact_vs_forked(self):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2 = jax.random.split(jax.random.key(43), 3)
    dout = _rand(k0, (m, f), jnp.bfloat16)
    lhs3 = _to3d(_rand(k1, (m, d), jnp.bfloat16))
    ps_tgmm = _rand(k2, (g, d, f), jnp.float32, 0.1)
    tiling = (512, 1024, 1024) * 3
    run_drhs = jax.jit(
        lambda c, a, p: ops._drhs_run_tokamax_v2(  # pylint: disable=protected-access
            c, a, gs, None, g, jnp.float32, tiling, False, partial_sum=p
        )
    )
    drhs_new = run_drhs(dout, lhs3, ps_tgmm)
    with mock.patch.object(ops, "tokamax_tgmm_v2", None):
      drhs_fork = run_drhs(dout, lhs3, ps_tgmm)
    np.testing.assert_array_equal(drhs_new, drhs_fork)

  def test_tgmm_3d_partial_sum_fp8_matches_forked(self):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2 = jax.random.split(jax.random.key(44), 3)
    dq = _q_fp8(_rand(k0, (m, f), jnp.bfloat16), [])
    dout_q = qpl.QArray(qvalue=dq.qvalue, scale=dq.scale, zero_point=None, qtype=dq.qtype)
    lhs3_q = _to3d(_q_fp8(_rand(k1, (m, d), jnp.bfloat16), []).qvalue)
    ps_tgmm = _rand(k2, (g, d, f), jnp.float32, 0.1)
    tiling = (512, 1024, 1024) * 3
    run_drhs = jax.jit(
        lambda c, a, p: ops._drhs_run_tokamax_v2(  # pylint: disable=protected-access
            c, a, gs, None, g, jnp.float32, tiling, False, partial_sum=p
        )
    )
    drhs_new = run_drhs(dout_q, lhs3_q, ps_tgmm)
    with mock.patch.object(ops, "tokamax_tgmm_v2", None):
      drhs_fork = run_drhs(dout_q, lhs3_q, ps_tgmm)
    np.testing.assert_allclose(drhs_new, drhs_fork, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(drhs_fork))))

  @parameterized.parameters("bf16", "fp8_accum_chunk")
  def test_ops_gmm_3d_tgmm_vjp_matches_forked(self, mode: str):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2, k3 = jax.random.split(jax.random.key(45), 4)
    x = _rand(k0, (m, d), jnp.bfloat16)
    wi = _rand(k1, (g, d, f), jnp.bfloat16, 0.05)
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
      def chunk(x_in, wi_in, wo_in, ret_rhs):
        res0 = ops.gmm(x_in, wi_in, gs, tiling=tiling, return_rhs=ret_rhs, **common)
        h0, wi_out = res0 if ret_rhs else (res0, wi_in)
        act = jax.nn.silu(h0.astype(jnp.float32)).astype(jnp.bfloat16)
        res1 = ops.gmm(act, wo_in, gs, tiling=tiling, return_rhs=ret_rhs, out_is_3d=True, **common)
        out, wo_out = res1 if ret_rhs else (res1, wo_in)
        return out, (wi_out, wo_out)

      def f(x_in, wi_in, wo_in):
        def fwd(a, b, c):
          out_a, (wi_q, wo_q) = chunk(a, b, c, True)
          return out_a, chunk(a, wi_q, wo_q, False)[0]

        outs, vjp = jax.vjp(fwd, x_in, wi_in, wo_in)
        return outs, vjp(tuple(_to3d(c) for c in douts))

      return jax.jit(f)(xq3, wi, wo)

    outs_new, grads_new = run_vjp()
    with mock.patch.object(ops, "tokamax_tgmm_v2", None):
      outs_fork, grads_fork = run_vjp()
    for o_n, o_f in zip(outs_new, outs_fork):
      np.testing.assert_array_equal(o_n.astype(jnp.float32), o_f.astype(jnp.float32))
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
