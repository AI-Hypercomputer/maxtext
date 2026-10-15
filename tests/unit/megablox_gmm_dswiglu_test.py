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

"""The SwiGLU backward in the gmm_v2 DLHS epilogue (fuse_dact, megablox gmm_dswiglu, moe_wo_gmm_fused_dswiglu).

On CPU:
- apply_dact_fn, the epilogue itself, is d[gate | up] of silu(gate) * up at the float32 accumulator u (times the
  per-row scale), and its dot is <u, silu(gate) * up>; checked against jax.vjp in float32, two-sided.
- make_gmm_configs, under a TPU v7x target, admits fuse_dact and return_pre_act="concat" with one n tile and rejects
  them otherwise, with a tile narrower or wider than the dim: each n tile's accumulator is stored as one [gate | up]
  block, so with more than one tile the halves of different tiles interleave, and a wider tile reads past [gate | up].
- gmm_v2 with fuse_dact (with and without the row scale and the dot) and gmm_silu's [gate | up] forward lower to one
  Mosaic kernel each for a TPU v7x on this CPU host (a lowering, not a compile).
- What gmm_v2's wrapper hands the fused DLHS kernel, with pallas_call replaced by a recorder: [gate | up] as given and
  the row scale in float32 (no kernel runs).

On TPU only (Pallas TPU kernels): the fused DLHS against the unfused one (gmm_v2 with a float32 output, then the
epilogue), rows outside the groups zero, an expert-parallel shard (group_offset); gmm_silu(fuse_bwd_dswiglu) writes
the unfused output's [gate | up] bit for bit; gmm_dswiglu's VJP against gmm + the SwiGLU VJP, with the DLHS reading wo
transposed (wo^T) and as stored (transpose_rhs, as on the bench).
"""

import functools
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.kernels.megablox import ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2

GROUP_SIZES = np.array([60, 0, 100, 40], np.int32)  # sum 200 < 256: padding rows; boundaries inside sublanes


def _tpu_target(device_kind="TPU7x"):
  """A context in which traces target a TPU of `device_kind`, as in an AOT compile on a CPU host."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device))


def _rand(key, shape, scale=1.0, dtype=jnp.bfloat16):
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(dtype)


def _f32(x):
  return np.asarray(x.astype(jnp.float32))


def _reference_dact(u, gate, up, row_scale):
  """d[gate | up] and <u, act> of act = silu(gate) * up, by autodiff in float32."""
  g, v = gate.astype(jnp.float32), up.astype(jnp.float32)
  act, vjp = jax.vjp(lambda a, b: jax.nn.silu(a) * b, g, v)
  du = u if row_scale is None else u * row_scale
  return jnp.concatenate(vjp(du), axis=-1), jnp.sum(u * act, axis=-1, keepdims=True)


def _close(a, b, rtol=1e-5, atol=1e-6):
  return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol))


@pytest.mark.cpu_only
class ApplyDactFnTest(parameterized.TestCase):
  """The epilogue: float32 SwiGLU backward at the accumulator, with the optional row scale and dot."""

  def _inputs(self, seed):
    k = jax.random.split(jax.random.key(seed), 4)
    u = jax.random.normal(k[0], (64, 256), jnp.float32)
    gate, up = _rand(k[1], (64, 256), 3.0), _rand(k[2], (64, 256))
    scale = jax.random.uniform(k[3], (64, 1), jnp.float32)
    return u, gate, up, scale

  @parameterized.product(with_scale=[False, True], return_dot=[False, True])
  def test_matches_autodiff(self, with_scale, return_dot):
    u, gate, up, scale = self._inputs(0)
    scale = scale if with_scale else None
    d, dot = gmm_v2.apply_dact_fn(u, gate, up, scale, "swiglu", return_dot=return_dot)
    want_d, want_dot = _reference_dact(u, gate, up, scale)
    self.assertEqual(d.dtype, jnp.float32)
    self.assertEqual(d.shape, (64, 512))
    self.assertTrue(_close(d, want_d))
    if return_dot:
      self.assertTrue(_close(dot, want_dot))
    else:
      self.assertIsNone(dot)

  def test_wrong_derivative_is_detected(self):
    """Two-sided: the check above rejects a derivative that drops silu's second term, and swapped halves."""
    u, gate, up, scale = self._inputs(1)
    want_d, _ = _reference_dact(u, gate, up, scale)
    g, v = gate.astype(jnp.float32), up.astype(jnp.float32)
    sig = jax.nn.sigmoid(g)
    wrong = jnp.concatenate([u * scale * v * sig, u * scale * g * sig], axis=-1)  # silu'(g) taken as sigmoid(g)
    self.assertFalse(_close(wrong, want_d))
    swapped, _ = gmm_v2.apply_dact_fn(u, up, gate, scale, "swiglu")
    self.assertFalse(_close(swapped, want_d))

  def test_unknown_activation(self):
    u, gate, up, _ = self._inputs(2)
    with self.assertRaisesRegex(NotImplementedError, "Unsupported backward activation"):
      gmm_v2.apply_dact_fn(u, gate, up, None, "gelu")


def _configs(lhs_shape, rhs_shape, tile_n, **kwargs):
  lhs = jax.ShapeDtypeStruct(lhs_shape, jnp.bfloat16)
  rhs = jax.ShapeDtypeStruct(rhs_shape, jnp.bfloat16)
  gs = jnp.asarray(GROUP_SIZES)
  with _tpu_target():
    return gmm_v2.make_gmm_configs(
        lhs,
        rhs,
        None,
        None,
        None,
        gs,
        jnp.array([0], jnp.int32),
        tile_info=gmm_v2.TileSizes(tile_m=128, tile_k=256, tile_n=tile_n),
        vmem_limit_bytes=None,
        out_dtype=jnp.bfloat16,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        **kwargs,
    )


@pytest.mark.cpu_only
class FusedGmmConfigTest(parameterized.TestCase):
  """make_gmm_configs: fuse_dact and return_pre_act="concat" need the [gate | up] layout, i.e. one n tile."""

  _GATE_UP = jax.ShapeDtypeStruct((256, 512), jnp.bfloat16)

  def test_fuse_dact_single_tile(self):
    cfgs = _configs((256, 256), (4, 256, 256), 256, fuse_dact="swiglu", dact_gate_up=self._GATE_UP)
    self.assertEqual((cfgs.out_size_n, cfgs.out_tile_n), (512, 512))

  @parameterized.named_parameters(
      ("two_tiles", {"tile_n": 128}, NotImplementedError, "single n-tile"),
      # Wider than the MLP dim (e.g. wo_tile_dlhs_mlp_dim 1024 against 512 per shard): the halves would split at
      # tile_n, past the end of gate.
      ("tile_wider_than_n", {"tile_n": 512}, NotImplementedError, "single n-tile"),
      ("no_gate_up", {"dact_gate_up": None}, ValueError, "requires dact_gate_up"),
      ("gate_up_shape", {"dact_gate_up": jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)}, ValueError, "dact_gate_up"),
      ("other_activation", {"fuse_dact": "gelu"}, NotImplementedError, "swiglu"),
  )
  def test_fuse_dact_rejected(self, change, error, message):
    kwargs = {"tile_n": 256, "fuse_dact": "swiglu", "dact_gate_up": self._GATE_UP, **change}
    with self.assertRaisesRegex(error, message):
      _configs((256, 256), (4, 256, 256), **kwargs)

  def test_dact_options_need_fuse_dact(self):
    with self.assertRaisesRegex(ValueError, "require fuse_dact"):
      _configs((256, 256), (4, 256, 256), 256, return_dact_dot=True)

  def test_concat_single_tile(self):
    cfgs = _configs((256, 256), (4, 256, 512), 256, fuse_act="silu", return_pre_act="concat")
    self.assertEqual(cfgs.return_pre_act, "concat")

  def test_concat_two_tiles_rejected(self):
    """Two tiles per half would store [gate_0 | up_0 | gate_1 | up_1]: rejected, not silently misordered."""
    with self.assertRaisesRegex(NotImplementedError, r'return_pre_act="concat" requires a single n tile'):
      _configs((256, 256), (4, 256, 512), 128, fuse_act="silu", return_pre_act="concat")
    # Nor a tile wider than each half.
    with self.assertRaisesRegex(NotImplementedError, r'return_pre_act="concat" requires a single n tile'):
      _configs((256, 256), (4, 256, 512), 512, fuse_act="silu", return_pre_act="concat")
    # The two-output form has no such constraint.
    self.assertTrue(_configs((256, 256), (4, 256, 512), 128, fuse_act="silu", return_pre_act=True).return_pre_act)

  def test_bad_return_pre_act(self):
    with self.assertRaisesRegex(ValueError, "return_pre_act must be"):
      _configs((256, 256), (4, 256, 512), 256, fuse_act="silu", return_pre_act="concatenate")


def _kernel_names(fn, *args):
  with _tpu_target():
    lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
  return re.findall(r'kernel_name = "([^"]+)"', lowered.as_text())


@pytest.mark.cpu_only
class FusedGmmLoweringTest(parameterized.TestCase):
  """The fused kernels lower to one Mosaic kernel each for a TPU v7x (a lowering on this host, not a compile)."""

  @parameterized.product(with_scale=[False, True], return_dot=[False, True], transpose_rhs=[False, True])
  def test_fused_dlhs_lowers(self, with_scale, return_dot, transpose_rhs):
    m, k, n = 256, 256, 256
    args = [
        jax.ShapeDtypeStruct((m, k), jnp.bfloat16),  # dout
        jax.ShapeDtypeStruct((4, n, k) if transpose_rhs else (4, k, n), jnp.bfloat16),  # wo, or wo^T
        jax.ShapeDtypeStruct((4,), jnp.int32),
        jax.ShapeDtypeStruct((m, 2 * n), jnp.bfloat16),  # [gate | up]
    ]
    if with_scale:
      args.append(jax.ShapeDtypeStruct((m,), jnp.float32))

    def fn(dout, w, gs, gate_up, scale=None):
      return gmm_v2.gmm_v2(
          dout,
          w,
          gs,
          tile_info=gmm_v2.TileSizes(128, 256, n),
          preferred_element_type=jnp.bfloat16,
          transpose_rhs=transpose_rhs,
          fuse_dact="swiglu",
          dact_gate_up=gate_up,
          dact_row_scale=scale,
          return_dact_dot=return_dot,
      )

    names = _kernel_names(fn, *args)
    self.assertLen(names, 1)
    self.assertIn("-dact_swiglu", names[0])

  def test_gmm_silu_gate_up_forward_lowers(self):
    lhs = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16)
    tiling = (128, 256, 512, 128, 256, 256, 128, 256, 512)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)

    def fwd(lhs, rhs, gs):
      # The differentiated forward of gmm_silu(fuse_bwd_dswiglu=True): act and [gate | up] from one kernel.
      return jax.vjp(lambda l, r: ops.gmm_silu(l, r, gs, jnp.bfloat16, tiling, fuse_bwd_dswiglu=True), lhs, rhs)[0]

    names = _kernel_names(fwd, lhs, rhs, gs)
    self.assertLen(names, 1)
    self.assertIn("-pre_act_concat", names[0])

  def test_unfused_kernel_name_unchanged(self):
    """The options add nothing to the other gmm_v2 calls' kernel names (their metadata is unchanged too)."""
    lhs = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 256, 256), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    fn = functools.partial(gmm_v2.gmm_v2, tile_info=gmm_v2.TileSizes(128, 256, 256), preferred_element_type=jnp.bfloat16)
    names = _kernel_names(fn, lhs, rhs, gs)
    self.assertLen(names, 1)
    self.assertNotIn("dact", names[0])
    self.assertNotIn("pre_act", names[0])


@pytest.mark.cpu_only
class FusedDlhsOperandsTest(parameterized.TestCase):
  """The operands gmm_v2's wrapper hands the fused DLHS kernel, with pallas_call replaced by a recorder.

  No kernel runs: the recorder returns the [gate | up] operand as the d[gate | up] output and the row-scale operand
  as the dot output, so gmm_v2's own output slicing returns what the kernel would have read. The row scale must
  reach the kernel in float32, one lane row per row, whatever its input shape: a bf16 rounding of it would hide
  inside the bf16 output's rounding in the TPU test.
  """

  M, K, N = 384, 128, 128  # Shapes no other test here traces gmm_v2 at, so its jit cache cannot hold a real trace.

  def _operands(self, scale_in):
    """The fused DLHS wrapper with the recorder: ([gate | up], what the kernel reads as [gate | up] and as row scale)."""
    with _tpu_target():
      tpu_info = pltpu.get_tpu_info()  # v7x, for the wrapper's tiling and layout checks (it runs on this CPU host)
    k = jax.random.split(jax.random.key(4), 3)
    dout, wo = _rand(k[0], (self.M, self.K)), _rand(k[1], (4, self.N, self.K))
    gate_up = _rand(k[2], (self.M, 2 * self.N))

    def recording_pallas_call(kernel, *, out_shape, **kwargs):
      del kernel, kwargs

      def call(*operands):
        (dact,) = [o for o in operands if isinstance(o, gmm_v2.DactInputRef)]
        out, dot = out_shape
        self.assertEqual((dact.gate_up.shape, dact.row_scale.shape), (out.shape, dot.shape))
        return dact.gate_up.astype(out.dtype), dact.row_scale

      return call

    with (
        mock.patch.object(pl, "pallas_call", recording_pallas_call),
        mock.patch.object(pltpu, "get_tpu_info", lambda *args, **kwargs: tpu_info),
    ):
      seen_gate_up, seen_scale = gmm_v2.gmm_v2(
          dout,
          wo,
          jnp.asarray([100, 0, 200, 50], jnp.int32),
          tile_info=gmm_v2.TileSizes(128, 128, self.N),
          preferred_element_type=jnp.bfloat16,
          transpose_rhs=True,
          fuse_dact="swiglu",
          dact_gate_up=gate_up,
          dact_row_scale=scale_in,
          return_dact_dot=True,
      )
    return gate_up, seen_gate_up, seen_scale

  @parameterized.named_parameters(("vector", ()), ("column", (1,)), ("lane_row", (128,)))
  def test_row_scale_reaches_the_kernel_in_float32(self, trailing):
    scale = 0.5 + jax.random.uniform(jax.random.key(5), (self.M,), jnp.float32)  # mostly not bf16-representable
    scale_in = jnp.broadcast_to(scale.reshape(-1, *([1] * len(trailing))), (self.M, *trailing))
    gate_up, seen_gate_up, seen_scale = self._operands(scale_in)
    self.assertEqual(seen_scale.dtype, jnp.float32)
    np.testing.assert_array_equal(np.asarray(seen_scale), np.asarray(scale))
    np.testing.assert_array_equal(_f32(seen_gate_up), _f32(gate_up))
    # Two-sided: the same comparison rejects the scale rounded to bf16.
    rounded = scale.astype(jnp.bfloat16).astype(jnp.float32)
    self.assertFalse(np.array_equal(np.asarray(seen_scale), np.asarray(rounded)))


# ------------------------------------------------------------------------------------------------------------------
# TPU only.
# ------------------------------------------------------------------------------------------------------------------
M, H, N = 1024, 512, 512  # rows, hidden (the wo GMM's output width), MLP (the activation width)
TPU_GROUP_SIZES = np.array([300, 0, 517, 133], np.int32)  # sum 950 < M; group ends inside sublanes
TILING = (256, 512, 512, 256, 512, 512, 256, 512, 512)  # gmm's 9-tuple for the wo GMM (DLHS tile_n = N)


def _computed_rows(group_offset):
  rows = np.zeros(M, bool)
  start = int(np.sum(TPU_GROUP_SIZES[:group_offset]))
  rows[start : start + int(np.sum(TPU_GROUP_SIZES[group_offset:]))] = True
  return rows


@pytest.mark.tpu_only
class FusedDlhsTpuTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")

  @parameterized.product(group_offset=[0, 2], with_scale=[False, True])
  def test_fused_dlhs_is_the_epilogue_of_the_unfused_accumulator(self, group_offset, with_scale):
    k = jax.random.split(jax.random.key(group_offset * 3 + with_scale), 5)
    dout = _rand(k[0], (M, H))
    wo = _rand(k[1], (len(TPU_GROUP_SIZES) - group_offset, N, H), 0.05)
    gate_up = _rand(k[2], (M, 2 * N), 2.0)
    scale = jax.random.uniform(k[3], (M,), jnp.float32) if with_scale else None
    gs, go = jnp.asarray(TPU_GROUP_SIZES), jnp.array([group_offset], jnp.int32)
    common = {"tile_info": gmm_v2.TileSizes(256, 512, N), "group_offset": go, "transpose_rhs": True}
    acc = jax.jit(functools.partial(gmm_v2.gmm_v2, preferred_element_type=jnp.float32, **common))(dout, wo, gs)
    d, dot = jax.jit(
        functools.partial(
            gmm_v2.gmm_v2, preferred_element_type=jnp.bfloat16, fuse_dact="swiglu", return_dact_dot=True, **common
        )
    )(dout, wo, gs, dact_gate_up=gate_up, dact_row_scale=scale)
    rows = _computed_rows(group_offset)
    want_d, want_dot = gmm_v2.apply_dact_fn(
        acc, gate_up[:, :N], gate_up[:, N:], None if scale is None else scale[:, None], "swiglu", return_dot=True
    )
    self.assertTrue(np.all(_f32(d)[~rows] == 0))
    self.assertTrue(np.all(np.asarray(dot)[~rows] == 0))
    np.testing.assert_allclose(_f32(d)[rows], _f32(want_d.astype(jnp.bfloat16))[rows], rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(np.asarray(dot)[rows], np.asarray(want_dot[:, 0])[rows], rtol=1e-4, atol=1e-4)

  @parameterized.parameters(0, 2)
  def test_gmm_silu_writes_the_unfused_gate_up(self, group_offset):
    k0, k1 = jax.random.split(jax.random.key(group_offset))
    lhs = _rand(k0, (M, H))
    rhs = _rand(k1, (len(TPU_GROUP_SIZES) - group_offset, H, 2 * N), 0.05)
    gs, go = jnp.asarray(TPU_GROUP_SIZES), jnp.array([group_offset], jnp.int32)
    tiling = (256, 512, 2 * N, 256, 512, 512, 256, 512, 512)  # one forward n tile per half
    unfused = jax.jit(
        functools.partial(gmm_v2.gmm_v2, tile_info=gmm_v2.TileSizes(256, 512, 512), preferred_element_type=jnp.bfloat16)
    )(lhs, rhs, gs, group_offset=go)
    act_ref = jax.jit(lambda l, r: ops.gmm_silu(l, r, gs, jnp.bfloat16, tiling, group_offset=go))(lhs, rhs)
    act, gate_up = jax.jit(
        lambda l, r: jax.vjp(
            lambda a, b: ops.gmm_silu(a, b, gs, jnp.bfloat16, tiling, group_offset=go, fuse_bwd_dswiglu=True), l, r
        )[0]
    )(lhs, rhs)
    np.testing.assert_array_equal(_f32(gate_up), _f32(unfused))
    np.testing.assert_array_equal(_f32(act), _f32(act_ref))

  @parameterized.product(group_offset=[0, 2], transpose_rhs=[False, True])
  def test_gmm_dswiglu_vjp_matches_the_unfused_path(self, group_offset, transpose_rhs):
    """With the DLHS reading wo^T, or wo as stored with the kernel's transpose_rhs (the bench's layout)."""
    k = jax.random.split(jax.random.key(10 + group_offset), 4)
    gate_up = _rand(k[0], (M, 2 * N), 2.0)
    wo = _rand(k[1], (len(TPU_GROUP_SIZES) - group_offset, N, H), 0.05)
    cot = _rand(k[2], (M, H))
    gs, go = jnp.asarray(TPU_GROUP_SIZES), jnp.array([group_offset], jnp.int32)

    def unfused(gate_up, wo):
      h = (jax.nn.silu(gate_up[:, :N]) * gate_up[:, N:]).astype(jnp.bfloat16)
      kw = {"use_tokamax_backend": True, "use_gmm_v2": True, "group_offset": go, "use_dlhs_transpose_rhs": transpose_rhs}
      return ops.gmm(h, wo, gs, preferred_element_type=jnp.bfloat16, tiling=TILING, **kw), h

    def fused(act, gate_up, wo):
      return ops.gmm_dswiglu(
          act, gate_up, wo, gs, jnp.bfloat16, TILING, group_offset=go, use_dlhs_transpose_rhs=transpose_rhs
      )

    @jax.jit
    def run(gate_up, wo, cot):
      # The fused op's activation is the unfused path's own h: a bf16 SwiGLU computed apart (say eagerly, outside this
      # jit) can round differently, and the forwards would then differ by that, not by the op.
      y_u, vjp_u, h = jax.vjp(unfused, gate_up, wo, has_aux=True)
      y_f, vjp_f = jax.vjp(fused, h, gate_up, wo)
      return y_u, vjp_u(cot), y_f, vjp_f(cot)

    y_u, (dgu_u, dwo_u), y_f, (dact_f, dgu_f, dwo_f) = run(gate_up, wo, cot)
    np.testing.assert_array_equal(_f32(y_f), _f32(y_u))  # The forward is gmm's.
    np.testing.assert_array_equal(_f32(dwo_f), _f32(dwo_u))  # The same TGMM on the same operands.
    self.assertTrue(np.all(_f32(dact_f) == 0))  # The gradient goes through gate_up.
    # d[gate | up] from the float32 accumulator instead of its bf16 rounding: equal to rounding, in relative L2 (4.4e-3
    # with the jnp stand-ins on CPU). Not elementwise: where the bf16 VJP cancels (1 - sigmoid near 1) single elements
    # differ by up to ~1% of the array's max, which failed rtol=2e-2, atol=2e-3 on 0.2% of them.
    a, b = _f32(dgu_f), _f32(dgu_u)
    self.assertLess(np.linalg.norm(a - b) / np.linalg.norm(b), 1e-2)


if __name__ == "__main__":
  absltest.main()
