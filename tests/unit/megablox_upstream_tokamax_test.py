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

"""CPU dispatch and TPU parity tests for routing 3D GMM paths through upstream Tokamax."""

import contextlib
import functools
import logging
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

_TILES = forked_gmm_v2.TileSizes(256, 1024, 256)
_W1792 = forked_gmm_v2.TileSizes(256, 1792, 256)
_FULL_3584 = forked_gmm_v2.TileSizes(256, 3584, 256)
_HEUR = forked_gmm_v2.calculate_tiling


def _to3d(x: jax.Array) -> jax.Array:
  return x.reshape(x.shape[0], -1, 128)


def _rand(key: jax.Array, shape: tuple[int, ...], dtype, scale: float = 0.25) -> jax.Array:
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _q_fp8(x: jax.Array, axes: list[int]):
  return qpl.quantize(x, jnp.float8_e4m3fn, channelwise_axes=axes, calibration_method="fixed,-224,224")


def _route_ctx(use_upstream: bool):
  return contextlib.nullcontext() if use_upstream else mock.patch.object(ops, "gmm_fuser", None)


def _assert_distinct_hlo(runner, *args):
  hlo_new = runner.lower(True, *args).compile().as_text()
  hlo_fork = runner.lower(False, *args).compile().as_text()
  cc_new = [line for line in hlo_new.splitlines() if "custom-call" in line]
  cc_fork = [line for line in hlo_fork.splitlines() if "custom-call" in line]
  logging.info("HLO custom-call upstream=%s | fork=%s", cc_new, cc_fork)
  assert (
      cc_new and cc_fork and cc_new != cc_fork
  ), f"Expected upstream and forked routes to lower distinct custom calls: upstream={cc_new} vs fork={cc_fork}"


class UpstreamTokamaxDispatchTest(parameterized.TestCase):
  """CPU-compatible tests for _gmm_route, _tpu_vmem_limit_bytes, and _call_gmm_v2."""

  @parameterized.named_parameters(
      {"testcase_name": "3d_lhs_fuser", "args": (3, False, False, _TILES, False), "want": "fuser"},
      {"testcase_name": "3d_lhs_full_3584", "args": (3, False, False, _FULL_3584, False), "k_dim": 3584, "want": "fuser"},
      {"testcase_name": "3d_lhs_3d_out_fuser", "args": (3, True, False, _TILES, False), "want": "fuser"},
      {"testcase_name": "3d_lhs_partial_sum", "args": (3, False, True, _TILES, False), "want": "upstream"},
      {"testcase_name": "3d_out_2d_lhs", "args": (2, True, False, _TILES, False), "want": "upstream"},
      {"testcase_name": "3d_out_and_ps", "args": (3, True, True, _TILES, False), "want": "upstream"},
      {"testcase_name": "3d_out_heuristic", "args": (2, True, False, _HEUR, False), "want": "fork"},
      {"testcase_name": "3d_lhs_ps_heuristic", "args": (3, False, True, _HEUR, False), "want": "fork"},
      {"testcase_name": "3d_out_window_1792", "args": (3, True, False, _W1792, False), "k_dim": 7168, "want": "upstream"},
      {
          "testcase_name": "3d_fused_window_1792",
          "args": (3, False, False, _W1792, False),
          "k_dim": 7168,
          "want": "upstream",
      },
      {
          "testcase_name": "3d_lhs_nondiv_1536",
          "args": (3, False, False, _TILES, False),
          "k_dim": 1536,
          "want": "upstream",
      },
      {
          "testcase_name": "3d_lhs_nondiv_2048",
          "args": (3, False, False, forked_gmm_v2.TileSizes(256, 2048, 256), False),
          "k_dim": 7168,
          "want": "upstream",
      },
      {
          "testcase_name": "3d_lhs_oversized_512",
          "args": (3, False, False, _TILES, False),
          "k_dim": 512,
          "want": "upstream",
      },
      {
          "testcase_name": "3d_lhs_zero_tile",
          "args": (3, False, False, forked_gmm_v2.TileSizes(256, 0, 256), False),
          "k_dim": 1024,
          "want": "upstream",
      },
      {"testcase_name": "3d_fused_heuristic", "args": (3, False, False, _HEUR, False), "want": "fork"},
      {"testcase_name": "tail_chunks", "args": (3, False, False, _TILES, True), "want": "fork"},
      {"testcase_name": "2d_only", "args": (2, False, False, _TILES, False), "want": "fork"},
      {"testcase_name": "2d_only_2d_ps", "args": (2, False, True, _TILES, False), "want": "fork"},
  )
  def test_gmm_route(self, args, want: str, k_dim: int | None = None):
    self.assertEqual(ops._gmm_route(*args, k_dim=k_dim), want)  # pylint: disable=protected-access

  def test_tpu_vmem_limit_bytes(self):
    if jax.default_backend() == "cpu":
      with self.assertRaises(ValueError):
        pltpu.get_tpu_info()
      self.assertIsNone(ops._tpu_vmem_limit_bytes())  # pylint: disable=protected-access
    fake_tpu = mock.Mock(vmem_capacity_bytes=64 * 1024 * 1024)
    with mock.patch.object(ops.pltpu, "get_tpu_info", autospec=True, return_value=fake_tpu):
      self.assertEqual(ops._tpu_vmem_limit_bytes(), int(0.96 * 64 * 1024 * 1024))  # pylint: disable=protected-access
    with mock.patch.object(ops.pltpu, "get_tpu_info", autospec=True, side_effect=ValueError("cpu")):
      self.assertIsNone(ops._tpu_vmem_limit_bytes())  # pylint: disable=protected-access

  @parameterized.named_parameters(
      {"testcase_name": "fuser", "lhs_shape": (512, 8, 128), "want": "fuser"},
      {"testcase_name": "upstream", "lhs_shape": (512, 8, 128), "out_is_3d": True, "ps_3d": True, "want": "upstream"},
      {"testcase_name": "heuristic", "lhs_shape": (512, 8, 128), "out_is_3d": True, "tile_info": _HEUR, "want": "fork"},
      {"testcase_name": "fork_2d", "lhs_shape": (512, 1024), "want": "fork"},
      {"testcase_name": "hide_fuser", "lhs_shape": (512, 8, 128), "hide_fuser": True, "want": "fork"},
  )
  def test_call_gmm_v2_wiring(
      self,
      lhs_shape,
      want: str,
      out_is_3d: bool = False,
      ps_3d: bool = False,
      hide_fuser: bool = False,
      tile_info=_TILES,
  ):
    out_shape = (512, 2, 128) if out_is_3d else (512, 256)
    lhs, rhs = jnp.ones(lhs_shape, jnp.bfloat16), jnp.ones((2, 1024, 256), jnp.bfloat16)
    gs = jnp.array([256, 256], jnp.int32)
    ps = jnp.ones(out_shape, jnp.bfloat16) if ps_3d else None
    zero_2d = jnp.zeros((512, 256), jnp.bfloat16)
    # Stand-in Tokamax modules: tests the routing and argument wiring on any tokamax version (GitHub CPU CI
    # pins tokamax==0.0.12, which predates gmm_fuser). Kernel numerics are covered by the TPU parity tests.
    fuser = mock.Mock(gmm_v2_fusible=mock.Mock(return_value=zero_2d))
    up_gmm = mock.Mock(gmm_v2=mock.Mock(return_value=zero_2d))
    fusion = mock.Mock(fuse=lambda f: f)
    if hide_fuser:
      fuser = up_gmm = fusion = None
    with (
        mock.patch.object(ops.pltpu, "get_tpu_info", autospec=True, return_value=mock.Mock(vmem_capacity_bytes=64)),
        mock.patch.object(ops, "gmm_fuser", fuser),
        mock.patch.object(ops, "tokamax_gmm_v2", up_gmm),
        mock.patch.object(ops, "op_fusion", fusion),
        mock.patch.object(ops.gmm_v2, "gmm_v2", autospec=True, return_value=jnp.ones(out_shape, jnp.bfloat16)) as fork,
    ):
      out = ops._call_gmm_v2(  # pylint: disable=protected-access
          lhs=lhs,
          rhs=rhs,
          group_sizes=gs,
          partial_sum=ps,
          tile_info=tile_info,
          preferred_element_type=jnp.bfloat16,
          out_is_3d=out_is_3d,
      )
    self.assertEqual(out.shape, out_shape)
    if want == "fuser":
      fuser.gmm_v2_fusible.assert_called_once()
      self.assertEqual(fuser.gmm_v2_fusible.call_args.kwargs["lhs"].shape, (512, 1024))
      self.assertEqual(fuser.gmm_v2_fusible.call_args.kwargs["vmem_limit_bytes"], int(0.96 * 64))
      up_gmm.gmm_v2.assert_not_called()
      fork.assert_not_called()
    elif want == "upstream":
      up_gmm.gmm_v2.assert_called_once()
      self.assertEqual(up_gmm.gmm_v2.call_args.kwargs["lhs"].shape, (512, 1024))
      self.assertEqual(up_gmm.gmm_v2.call_args.kwargs["partial_sum"].shape, (512, 256))
      fork.assert_not_called()
    else:
      fork.assert_called_once()


@pytest.mark.tpu_only
class UpstreamTokamaxTpuParityTest(parameterized.TestCase):
  """TPU parity tests: PR A upstream GMM routes vs forked pallas_mosaic_tpu_v2_gmm_kernel."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("TPU only")
    if ops.gmm_fuser is None or ops.op_fusion is None:
      self.skipTest("needs tokamax >= 0.0.18")

  @parameterized.product(path=("3d_lhs_fuser", "3d_lhs_window", "3d_out", "3d_partial_sum"), dtype=("bf16", "fp8"))
  def test_gmm_parity_vs_forked(self, path: str, dtype: str):
    high_vmem = path == "3d_lhs_fuser" and dtype == "fp8" and pltpu.get_tpu_info().vmem_capacity_bytes >= 64 * 1024 * 1024
    if path == "3d_lhs_window":
      m, k, n, g, tile_k = 2048, 7168, 1024, 4, _W1792.tile_k
    elif high_vmem:
      m, k, n, g, tile_k = 2048, 7168, 2048, 16, 7168
    else:
      m, k, n, g, tile_k = 2048, 2048, 1024, 4, 1024
    gs = jnp.full((g,), m // g, jnp.int32) if high_vmem else jnp.array([300, 0, 517, 700], jnp.int32)
    tiling = (512, tile_k, n) * 3
    k0, k1, k2 = jax.random.split(jax.random.key(11), 3)
    lhs_shape = (m, n) if path == "3d_partial_sum" else (m, k)
    x_raw, w_raw = _rand(k0, lhs_shape, jnp.bfloat16), _rand(k1, (g, k, n), jnp.bfloat16, 0.05)

    is_3d_lhs = path in ("3d_lhs_fuser", "3d_lhs_window")
    if dtype == "fp8":
      xq = _q_fp8(x_raw, [])
      wq = _q_fp8(w_raw, [] if path == "3d_partial_sum" else [0, 2])
      lhs_in = qpl.QArray(
          qvalue=_to3d(xq.qvalue) if is_3d_lhs else xq.qvalue,
          scale=xq.scale,
          zero_point=None,
          qtype=xq.qtype,
      )
      rhs_in = (
          wq.qvalue
          if path == "3d_partial_sum"
          else qpl.QArray(qvalue=wq.qvalue, scale=wq.scale, zero_point=None, qtype=jnp.float8_e4m3fn)
      )
      dlhs_scale = wq.scale
    else:
      lhs_in = _to3d(x_raw) if is_3d_lhs else x_raw
      rhs_in, dlhs_scale = w_raw, None

    if path == "3d_partial_sum":
      ps_in = _to3d(_rand(k2, (m, k), jnp.bfloat16, 0.1))

      @functools.partial(jax.jit, static_argnames=("use_upstream",))
      def runner(use_upstream: bool, a, b, p):
        with _route_ctx(use_upstream):
          return ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
              a,
              b,
              gs,
              None,
              jnp.bfloat16,
              tiling,
              False,
              p,
              lhs_scale=dlhs_scale,
              fuse_dlhs_scale=True,
              use_dlhs_transpose_rhs=True,
              out_is_3d=True,
              fuse_3d_dlhs_ps=True,
          )[0]

      _assert_distinct_hlo(runner, lhs_in, rhs_in, ps_in)
      y_new, y_fork = runner(True, lhs_in, rhs_in, ps_in), runner(False, lhs_in, rhs_in, ps_in)
    else:

      @functools.partial(jax.jit, static_argnames=("use_upstream",))
      def runner(use_upstream: bool, a, b):
        with _route_ctx(use_upstream):
          return ops.gmm(
              a, b, gs, jnp.bfloat16, tiling, use_tokamax_backend=True, use_gmm_v2=True, out_is_3d=(path == "3d_out")
          )

      _assert_distinct_hlo(runner, lhs_in, rhs_in)
      y_new, y_fork = runner(True, lhs_in, rhs_in), runner(False, lhs_in, rhs_in)

    got, want = y_new.astype(jnp.float32), y_fork.astype(jnp.float32)
    self.assertEqual(y_new.shape, y_fork.shape)
    if dtype == "bf16":
      np.testing.assert_array_equal(got, want)
    else:
      np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(want))))
      self.assertLessEqual(float(np.mean(np.asarray(got) != np.asarray(want))), 1e-3)

  @parameterized.parameters("bf16", "fp8_accum_chunk")
  def test_ops_gmm_vjp_matches_forked(self, mode: str):
    m, d, f, g = 2048, 2048, 1024, 4
    gs = jnp.array([300, 0, 517, 700], jnp.int32)
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(35), 5)
    x, w0, w1, wo = (
        _rand(k0, (m, d), jnp.bfloat16),
        _rand(k1, (g, d, f), jnp.bfloat16, 0.05),
        _rand(k4, (g, d, f), jnp.bfloat16, 0.05),
        _rand(k2, (g, f, d), jnp.bfloat16, 0.05),
    )
    dout = _to3d(_rand(k3, (m, d), jnp.bfloat16, 1e-2))
    tiling, use_qwix = (512, 1024, 1024) * 3, mode == "fp8_accum_chunk"
    rule = None
    xq3 = _to3d(x)
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

    @functools.partial(jax.jit, static_argnames=("use_upstream",))
    def runner(use_upstream: bool, x_in, w0_in, w1_in, wo_in):
      def fwd(a, b, c, e):
        h0, x1 = ops.gmm(a, b, gs, tiling=tiling, return_lhs=True, **common)
        h1 = ops.gmm(x1, c, gs, tiling=tiling, **common)
        h = (jax.nn.silu(h0.astype(jnp.float32)) * h1.astype(jnp.float32)).astype(jnp.bfloat16)
        return ops.gmm(h, e, gs, tiling=tiling, out_is_3d=True, **common)

      with _route_ctx(use_upstream):
        out, vjp = jax.vjp(fwd, x_in, w0_in, w1_in, wo_in)
        return out, vjp(dout)

    _assert_distinct_hlo(runner, xq3, w0, w1, wo)
    out_new, grads_new = runner(True, xq3, w0, w1, wo)
    out_fork, grads_fork = runner(False, xq3, w0, w1, wo)
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
