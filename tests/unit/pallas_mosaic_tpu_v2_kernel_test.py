# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for Pallas Mosaic TPU v2 kernels."""

import collections
import inspect
from unittest import mock

import pytest

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.megablox import common
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend


def poison_tpu_memory():
  """Fills TPU scratchpad memory with NaNs to simulate garbage state."""
  tpu_info = pltpu.get_tpu_info()
  # Security: Use a large but safe portion of VMEM/SMEM to avoid OOM.
  vmem_size = (4 * 1024 * 1024) // 4  # 4MB
  smem_size = (tpu_info.smem_capacity_bytes // 4) - 8192

  def poison_kernel(in_ref, out_ref, v_scratch, s_scratch):
    del in_ref, out_ref
    v_scratch[...] = jnp.full_like(v_scratch, jnp.nan)
    for i in range(s_scratch.shape[0]):
      s_scratch[i] = 0x7FC00000  # IEEE 754 NaN bit pattern

  pl.pallas_call(
      poison_kernel,
      out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
      grid=(1,),
      scratch_shapes=[
          pltpu.VMEM((vmem_size // 128, 128), jnp.float32),
          pltpu.SMEM((smem_size,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
  )(jnp.zeros((1,), dtype=jnp.float32))


_GroupConfig = collections.namedtuple("_GroupConfig", ["num_groups", "group_offset", "num_local_groups"])


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
  distribution = jax.random.uniform(jax.random.key(0), (num_groups - 1,), dtype=jnp.float32)
  distribution = distribution / jnp.sum(distribution)
  group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
  return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256):
  """Quantizes a tensor along a specified axis in blocks."""
  if jnp.issubdtype(dtype, jnp.integer):
    dtype_info = jnp.iinfo(dtype)
    max_val = int(dtype_info.max)
    min_val = int(dtype_info.min)
  else:
    dtype_info = jnp.finfo(dtype)
    max_val = float(dtype_info.max)
    min_val = float(dtype_info.min)

  orig_shape = x.shape
  blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
  x_blocked = x.reshape(blocked_shape)

  x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
  scale = x_blocked_abs_max / max_val
  x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

  x_q = x_blocked_q.reshape(orig_shape)
  x_q = jnp.nan_to_num(x_q)
  scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
  return x_q, scale


def reference_gmm(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [num_groups, k, n]
    group_sizes: jax.Array,  # [num_groups]
    partial_sum: jax.Array | None = None,  # [m, n]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,  # int32[1]
):
  """Computes reference grouped matrix multiplication."""
  num_tokens = lhs.shape[0]
  num_groups, in_size, out_size = rhs.shape
  assert num_groups > 0, f"rhs must have at least 1 group, got {num_groups}"
  assert lhs.shape[1] == in_size

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  if rhs_scale is not None:
    num_blocks = rhs_scale.shape[1]
  else:
    num_blocks = 1
  block_size = in_size // num_blocks

  start = 0
  gmm_out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]

    group = global_group - group_offset[0]
    end = min(start + group_size, num_tokens)
    group_size = end - start
    if 0 <= group < num_groups:
      lhs_slice = lhs[start:end]
      rhs_slice = rhs[group]

      out = jnp.array(0.0, dtype=jnp.float32)
      for block in range(num_blocks):
        block_start = block * block_size
        block_end = block_start + block_size
        lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
        rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

        acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
        if rhs_scale is not None:
          acc *= rhs_scale[group][block]
        out += acc
      if rhs_bias is not None:
        out = out + rhs_bias[group]
      if partial_sum is not None:
        out = out + partial_sum[start:end]
    else:
      out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

    gmm_out.append(out.astype(lhs.dtype))
    start = end

  return jnp.concat(gmm_out, axis=0)


def reference_tgmm(
    lhs,  # [k, m]
    rhs,  # [m, n]
    group_sizes,  # [num_groups]
    # num_actual_groups comes from weights.shape[0]
    num_actual_groups,  # int32
    # group_offset is obtained from
    # jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset=None,
    partial_sum=None,
):  # [num_groups, k, n]
  """Computes reference transposed grouped matrix multiplication."""
  # Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :]
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  start = 0
  out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]
    group = global_group - group_offset[0]
    end = start + group_size
    if 0 <= group < num_actual_groups:
      res = lhs[:, start:end].astype(jnp.float32) @ rhs[start:end, :].astype(jnp.float32)
      if partial_sum is not None:
        res = res + partial_sum[group].astype(jnp.float32)
      out.append(res.astype(lhs.dtype))
    start = end
  return jnp.stack(out)


# Default per-dtype tolerances, mirroring
# jax._src.public_test_util._default_tolerance. Extend this map if a new output
# dtype is introduced into a default-tolerance assertion.
_DTYPE_TOL = {
    jnp.dtype(jnp.bfloat16): 1e-1,
}


def _lookup_tol(dtype):
  key = jnp.dtype(dtype)
  if key not in _DTYPE_TOL:
    raise KeyError(f"No default tolerance for dtype {key!r}. " f"Add it to _DTYPE_TOL or pass explicit atol/rtol.")
  return _DTYPE_TOL[key]


def assert_arrays_all_close(actual, desired, *, atol=None, rtol=None):
  if atol is None:
    atol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  if rtol is None:
    rtol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  chex.assert_trees_all_close(actual, desired, atol=atol, rtol=rtol)


class GmmDispatchTest(parameterized.TestCase):
  """CPU-compatible routing tests; the TPU kernels are mocked."""

  @parameterized.product(
      rhs_dtype=(jnp.int4, jnp.int8, jnp.bfloat16, jnp.float8_e4m3fn),
      use_gmm_v2_heuristic_tiling=(False, True),
      use_dlhs_transpose_rhs=(None, False, True),
  )
  def test_dlhs_rhs_transpose_dispatch_by_dtype(self, rhs_dtype, use_gmm_v2_heuristic_tiling, use_dlhs_transpose_rhs):
    """DLHS materializes sub-byte RHS transposes before calling GMM v2."""
    num_groups, out_size, in_size, batch_size = 2, 16, 32, 8
    dlhs_dout = jnp.ones((batch_size, in_size), dtype=jnp.bfloat16)
    rhs_shape = (num_groups, out_size, in_size)
    rhs = (jnp.arange(num_groups * out_size * in_size).reshape(rhs_shape) % 8).astype(rhs_dtype)
    group_sizes = jnp.array([batch_size // num_groups] * num_groups, dtype=jnp.int32)
    expected = jnp.zeros((batch_size, out_size), dtype=jnp.bfloat16)
    tiling = (256, 512, 256, 512, 1024, 512, 256, 512, 256)

    # None exercises the public helper default for backward compatibility.
    kwargs = {} if use_dlhs_transpose_rhs is None else {"use_dlhs_transpose_rhs": use_dlhs_transpose_rhs}
    with mock.patch.object(megablox_ops.gmm_v2, "gmm_v2", return_value=expected) as gmm_v2_mock:
      actual, lhs_scale_applied = megablox_ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
          dlhs_dout=dlhs_dout,
          rhs=rhs,
          group_sizes=group_sizes,
          group_offset=None,
          lhs_dtype=jnp.bfloat16,
          tiling=tiling,
          use_gmm_v2_heuristic_tiling=use_gmm_v2_heuristic_tiling,
          **kwargs,
      )

    call_kwargs = gmm_v2_mock.call_args.kwargs
    expected_transpose_rhs = bool(use_dlhs_transpose_rhs) and jax.dtypes.itemsize_bits(rhs_dtype) >= 8
    expected_rhs = rhs if expected_transpose_rhs else rhs.swapaxes(1, 2)
    self.assertIs(actual, expected)
    self.assertFalse(lhs_scale_applied)
    self.assertEqual(call_kwargs["transpose_rhs"], expected_transpose_rhs)
    self.assertTrue(bool(jnp.array_equal(call_kwargs["rhs"], expected_rhs)))
    expected_tiling = (
        gmm_backend.calculate_tiling if use_gmm_v2_heuristic_tiling else gmm_backend.TileSizes(512, 1024, 512)
    )
    self.assertEqual(call_kwargs["tile_info"], expected_tiling)
    self.assertTrue(call_kwargs["maybe_quantize_lhs"])

  @parameterized.product(
      use_dlhs_transpose_rhs=(None, False, True),
      use_gmm_v2_heuristic_tiling=(False, True),
      has_partial_sum=(False, True),
  )
  def test_dlhs_transpose_flag_through_custom_vjp(
      self, use_dlhs_transpose_rhs, use_gmm_v2_heuristic_tiling, has_partial_sum
  ):
    """Check flag routing and cotangents through JIT with reference backends."""
    num_groups, batch_size, in_size, out_size = 2, 8, 16, 32
    lhs = jnp.linspace(-1.0, 1.0, batch_size * in_size).reshape(batch_size, in_size)
    rhs = jnp.linspace(-0.5, 0.5, num_groups * in_size * out_size).reshape(num_groups, in_size, out_size)
    partial_sum = jnp.ones((batch_size, out_size)) if has_partial_sum else None
    group_sizes = jnp.array([batch_size // num_groups] * num_groups, dtype=jnp.int32)

    def mock_gmm(lhs, rhs, transpose_rhs=False, partial_sum=None, **kwargs):
      del kwargs
      if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
      out = jnp.einsum("gmk,gkn->gmn", lhs.reshape(num_groups, -1, lhs.shape[-1]), rhs)
      out = out.reshape(batch_size, -1)
      return out if partial_sum is None else out + partial_sum

    def mock_tgmm(lhs, rhs, **kwargs):
      del kwargs
      return jnp.einsum(
          "gmk,gmn->gkn",
          lhs.reshape(num_groups, -1, lhs.shape[-1]),
          rhs.reshape(num_groups, -1, rhs.shape[-1]),
      )

    def loss(lhs, rhs, partial_sum):
      kwargs = {} if use_dlhs_transpose_rhs is None else {"use_dlhs_transpose_rhs": use_dlhs_transpose_rhs}
      out = megablox_ops.gmm(
          lhs,
          rhs,
          group_sizes,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          use_gmm_v2_heuristic_tiling=use_gmm_v2_heuristic_tiling,
          partial_sum=partial_sum,
          **kwargs,
      )
      return jnp.sum(out**2)

    def reference_loss(lhs, rhs, partial_sum):
      return jnp.sum(mock_gmm(lhs, rhs, partial_sum=partial_sum) ** 2)

    with mock.patch.object(megablox_ops.gmm_v2, "gmm_v2", side_effect=mock_gmm) as gmm_mock:
      with mock.patch.object(megablox_ops.tgmm_v2, "tgmm_v2", side_effect=mock_tgmm):
        actual = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(lhs, rhs, partial_sum)
    expected = jax.jit(jax.value_and_grad(reference_loss, argnums=(0, 1, 2)))(lhs, rhs, partial_sum)
    chex.assert_trees_all_close(actual, expected, atol=1e-4, rtol=1e-5)
    self.assertEqual(gmm_mock.call_args.kwargs["transpose_rhs"], bool(use_dlhs_transpose_rhs))

  def test_tgmm_multi_window_steps_defaults_off(self):
    """Every hop of the multi-window flag defaults to the HEAD behavior."""
    flag_by_fn = {
        megablox_ops.gmm: "use_tgmm_v2_multi_window_steps",
        megablox_ops._gmm_fwd: "use_tgmm_v2_multi_window_steps",  # pylint: disable=protected-access
        megablox_ops._compute_drhs: "multi_window_steps",  # pylint: disable=protected-access
        megablox_ops._drhs_run_tokamax_v2: "multi_window_steps",  # pylint: disable=protected-access
        tgmm_backend.tgmm_v2: "multi_window_steps",
    }
    for fn, name in flag_by_fn.items():
      self.assertIs(inspect.signature(fn).parameters[name].default, False, msg=fn.__name__)

  @parameterized.product(
      use_tgmm_v2_multi_window_steps=(None, False, True),
      use_gmm_v2_heuristic_tiling=(False, True),
  )
  def test_tgmm_multi_window_steps_through_custom_vjp(self, use_tgmm_v2_multi_window_steps, use_gmm_v2_heuristic_tiling):
    """The flag reaches tgmm_v2 in the backward pass and is off unless set."""
    num_groups, batch_size, in_size, out_size = 2, 8, 16, 32
    lhs = jnp.linspace(-1.0, 1.0, batch_size * in_size).reshape(batch_size, in_size)
    rhs = jnp.linspace(-0.5, 0.5, num_groups * in_size * out_size).reshape(num_groups, in_size, out_size)
    group_sizes = jnp.array([batch_size // num_groups] * num_groups, dtype=jnp.int32)

    def mock_gmm(lhs, rhs, transpose_rhs=False, partial_sum=None, **kwargs):
      del kwargs, partial_sum
      if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
      out = jnp.einsum("gmk,gkn->gmn", lhs.reshape(num_groups, -1, lhs.shape[-1]), rhs)
      return out.reshape(batch_size, -1)

    def mock_tgmm(lhs, rhs, **kwargs):
      del kwargs
      return jnp.einsum(
          "gmk,gmn->gkn",
          lhs.reshape(num_groups, -1, lhs.shape[-1]),
          rhs.reshape(num_groups, -1, rhs.shape[-1]),
      )

    def loss(lhs, rhs):
      kwargs = {}
      if use_tgmm_v2_multi_window_steps is not None:
        kwargs["use_tgmm_v2_multi_window_steps"] = use_tgmm_v2_multi_window_steps
      out = megablox_ops.gmm(
          lhs,
          rhs,
          group_sizes,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          use_gmm_v2_heuristic_tiling=use_gmm_v2_heuristic_tiling,
          **kwargs,
      )
      return jnp.sum(out**2)

    with mock.patch.object(megablox_ops.gmm_v2, "gmm_v2", side_effect=mock_gmm):
      with mock.patch.object(megablox_ops.tgmm_v2, "tgmm_v2", side_effect=mock_tgmm) as tgmm_mock:
        jax.jit(jax.grad(loss, argnums=(0, 1)))(lhs, rhs)

    self.assertEqual(tgmm_mock.call_count, 1)
    self.assertIs(
        tgmm_mock.call_args.kwargs["multi_window_steps"],
        bool(use_tgmm_v2_multi_window_steps),
    )


def _tgmm_configs(size_m, size_k, size_n, *, tile_m=512, lhs_is_3d=False, rhs_is_3d=False):
  """Builds the GmmConfigs of a bf16 2D tgmm with full-size k/n tiles (no TPU needed)."""
  num_lanes = 128
  tiles = gmm_backend.TileSizes(
      tile_m=tile_m,
      tile_k=gmm_backend.align_to(size_k, num_lanes),
      tile_n=gmm_backend.align_to(size_n, num_lanes),
  )
  dims = gmm_backend.Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=128,
      size_lhs_group=128,
      size_lhs_sublane=16,
  )
  return gmm_backend.GmmConfigs(
      tiles=tiles,
      dims=dims,
      lhs_cfgs=gmm_backend.InputConfigs(
          quant_dtype=None,
          quant_block_size=-1,
          dtype=jnp.bfloat16,
          is_3d=lhs_is_3d,
      ),
      rhs_cfgs=gmm_backend.InputConfigs(
          quant_dtype=None,
          quant_block_size=size_m,
          dtype=jnp.bfloat16,
          is_3d=rhs_is_3d,
      ),
      out_dtype=jnp.dtype(jnp.bfloat16),
      acc_dtype=jnp.dtype(jnp.float32),
      has_partial_sum=False,
      zero_init=False,
      fuse_act=None,
      transpose_rhs=False,
  )


class TgmmStepConfigTest(parameterized.TestCase):
  """CPU-compatible tests of the multi-window step choice; no TPU kernel runs."""

  _MIB = 1024 * 1024
  _ZERO_REF_BYTES = 2 * _MIB

  @parameterized.parameters((2816, 704), (704, 2816))
  def test_gemma_drhs_shape_picks_widest_config_on_v7x_vmem(self, size_k, size_n):
    cfgs = _tgmm_configs(131072, size_k, size_n)
    vmem_limit_bytes = int(64 * self._MIB * 0.9)
    self.assertEqual(
        tgmm_backend._tgmm_step_config(cfgs, vmem_limit_bytes, self._ZERO_REF_BYTES),  # pylint: disable=protected-access
        (3, 3),
    )

  @parameterized.parameters(
      (50, (2, 3)),
      (40, (2, 2)),
      (36, (2, 2)),
      (35, (1, 2)),
      (1, (1, 2)),
  )
  def test_falls_back_as_vmem_shrinks(self, vmem_limit_mib, expected):
    # k=2816, n=704: the candidates need 56.9, 46.4, 42.3, 35.3 and 35.9 MiB.
    cfgs = _tgmm_configs(131072, 2816, 704)
    actual = tgmm_backend._tgmm_step_config(cfgs, int(vmem_limit_mib * self._MIB), self._ZERO_REF_BYTES)  # pylint: disable=protected-access
    self.assertEqual(actual, expected)

  @parameterized.parameters((512, (1, 3)), (1000, (2, 3)), (1025, (3, 3)))
  def test_sub_windows_need_more_rows_than_the_window(self, size_m, expected):
    cfgs = _tgmm_configs(size_m, 2816, 704)
    actual = tgmm_backend._tgmm_step_config(cfgs, 1024 * self._MIB, self._ZERO_REF_BYTES)  # pylint: disable=protected-access
    self.assertEqual(actual, expected)

  @parameterized.parameters((True, False), (False, True))
  def test_3d_operands_keep_single_window_double_buffering(self, lhs_is_3d, rhs_is_3d):
    cfgs = _tgmm_configs(131072, 2816, 768, lhs_is_3d=lhs_is_3d, rhs_is_3d=rhs_is_3d)
    actual = tgmm_backend._tgmm_step_config(cfgs, 1024 * self._MIB, self._ZERO_REF_BYTES)  # pylint: disable=protected-access
    self.assertEqual(actual, (1, 2))

  @parameterized.parameters(None, False, True)
  def test_tgmm_v2_picks_step_config_only_when_flag_is_on(self, multi_window_steps):
    size_m, size_k, size_n, num_groups = 2048, 256, 256, 2
    lhs = jnp.zeros((size_m, size_k), jnp.bfloat16)
    rhs = jnp.zeros((size_m, size_n), jnp.bfloat16)
    group_sizes = jnp.array([size_m // num_groups] * num_groups, dtype=jnp.int32)
    kwargs = {} if multi_window_steps is None else {"multi_window_steps": multi_window_steps}
    # The host has no TPU: stand in for the v7x info and the pallas_call (the
    # kernel never runs).
    tpu_info = mock.Mock(
        num_lanes=128,
        mxu_column_size=256,
        vmem_capacity_bytes=64 * self._MIB,
        smem_capacity_bytes=self._MIB,
    )
    tpu_info.get_sublane_tiling.return_value = 16
    with mock.patch.object(pltpu, "get_tpu_info", return_value=tpu_info):
      with mock.patch.object(tgmm_backend, "_tgmm_step_config", return_value=(2, 3)) as step_config_mock:
        with mock.patch.object(tgmm_backend.pl, "pallas_call") as pallas_call_mock:
          pallas_call_mock.return_value = lambda *args: jnp.zeros((num_groups, size_k, size_n), jnp.bfloat16)
          # __wrapped__ skips the jit, so the mocks always see a fresh trace.
          tgmm_backend.tgmm_v2.__wrapped__(
              lhs,
              rhs,
              group_sizes,
              num_groups,
              tile_info=gmm_backend.TileSizes(tile_m=512, tile_k=size_k, tile_n=size_n),
              preferred_element_type=jnp.bfloat16,
              vmem_limit_bytes=64 * self._MIB,
              **kwargs,
          )
    kernel_kwargs = pallas_call_mock.call_args.args[0].keywords
    if multi_window_steps:
      step_config_mock.assert_called_once()
      self.assertEqual((kernel_kwargs["num_sub"], kernel_kwargs["num_buffers"]), (2, 3))
    else:
      step_config_mock.assert_not_called()
      self.assertEqual((kernel_kwargs["num_sub"], kernel_kwargs["num_buffers"]), (1, 2))


class GmmTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 512],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      has_bias=[True, False],
      has_partial_sum=[True, False],
      group_offset=[0, 2, 3],
  )
  def test_gmm_basic(self, batch_size, in_size, out_size, num_groups, has_bias, has_partial_sum, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(k2, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)
    ps = None
    if has_partial_sum:
      ps = jax.random.normal(k3, (batch_size, out_size), dtype=jnp.bfloat16)

    expected = reference_gmm(lhs, rhs, group_sizes, partial_sum=ps, rhs_bias=rhs_bias, group_offset=group_offset)

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        partial_sum=ps,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      input_dtype=(jnp.bfloat16, jnp.float8_e4m3fn),
      tile_sizes=((128, 128, 128), (512, 1024, 1024)),
  )
  def test_gmm_native_rhs_transpose_matches_swapaxes(self, input_dtype, tile_sizes):
    """Native transpose_rhs must be bit-exact with a materialized transpose."""
    num_groups, out_size, in_size, batch_size = 8, 1280, 2048, 4096
    lhs_key, rhs_key = jax.random.split(jax.random.PRNGKey(2026))
    lhs = jax.random.normal(lhs_key, (batch_size, in_size), dtype=jnp.bfloat16).astype(input_dtype)
    rhs = jax.random.normal(
        rhs_key,
        (num_groups, out_size, in_size),
        dtype=jnp.bfloat16,
    ).astype(input_dtype)
    group_sizes = jnp.array([batch_size // num_groups] * num_groups, dtype=jnp.int32)
    tiles = gmm_backend.TileSizes(*tile_sizes)

    out_swap = gmm_backend.gmm_v2(
        lhs=lhs,
        rhs=rhs.swapaxes(1, 2),
        group_sizes=group_sizes,
        rhs_scale=None,
        tile_info=tiles,
        preferred_element_type=jnp.bfloat16,
        transpose_rhs=False,
    )
    out_native = gmm_backend.gmm_v2(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        rhs_scale=None,
        tile_info=tiles,
        preferred_element_type=jnp.bfloat16,
        transpose_rhs=True,
    )

    self.assertEqual(out_native.shape, (batch_size, out_size))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_swap))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_native))))
    self.assertTrue(bool(jnp.array_equal(out_swap, out_native)))

  @parameterized.product(
      input_dtype=(jnp.bfloat16, jnp.float8_e4m3fn),
      fuse_act=("silu", "gelu", "swigluoai"),
      maybe_quantize_lhs=(False, True),
  )
  def test_gmm_fused_activation_with_native_rhs_transpose(self, input_dtype, fuse_act, maybe_quantize_lhs):
    """Fused activation supports native transposed RHS weights."""
    num_groups, fused_out_size, in_size, batch_size = 8, 2048, 1024, 4096
    lhs_key, rhs_key, bias_key = jax.random.split(jax.random.PRNGKey(2026), 3)
    lhs = jax.random.normal(lhs_key, (batch_size, in_size), dtype=jnp.bfloat16).astype(input_dtype)
    rhs = jax.random.normal(
        rhs_key,
        (num_groups, in_size, fused_out_size),
        dtype=jnp.bfloat16,
    ).astype(input_dtype)
    rhs_scale = jnp.linspace(0.5, 1.5, num_groups * 2 * fused_out_size).reshape(num_groups, 2, 1, fused_out_size)
    rhs_bias = jax.random.normal(
        bias_key,
        (num_groups, 1, fused_out_size),
        dtype=jnp.bfloat16,
    )
    group_sizes = jnp.array([batch_size // num_groups] * num_groups, dtype=jnp.int32)
    tiles = gmm_backend.TileSizes(512, 1024, 512)

    out_standard = gmm_backend.gmm_v2(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        tile_info=tiles,
        preferred_element_type=jnp.bfloat16,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
        transpose_rhs=False,
    )
    out_transposed = gmm_backend.gmm_v2(
        lhs=lhs,
        rhs=rhs.swapaxes(1, 2),
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        tile_info=tiles,
        preferred_element_type=jnp.bfloat16,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
        transpose_rhs=True,
    )

    self.assertEqual(out_transposed.shape, (batch_size, fused_out_size // 2))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_standard))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out_transposed))))
    self.assertTrue(bool(jnp.array_equal(out_standard, out_transposed)))

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
  )
  def test_gmm_partial_sum(self, batch_size, in_size, out_size, num_groups, group_offset):
    """Test GMM with partial sum accumulation and memory aliasing."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)
    ps = jax.random.normal(k2, (batch_size, out_size), dtype=jnp.bfloat16)

    expected = reference_gmm(lhs, rhs, group_sizes, partial_sum=ps, group_offset=group_offset)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        partial_sum=ps,
        group_offset=group_offset,
    )
    assert_arrays_all_close(actual, expected)

  @parameterized.product(
      lhs_dtype=[jnp.bfloat16, jnp.float8_e4m3fn],
      has_partial_sum=[True, False],
      group_offset=[0, 2],
  )
  def test_gmm_post_matmul_lhs_scale(self, lhs_dtype, has_partial_sum, group_offset):
    """With maybe_quantize_lhs=False, lhs_scale multiplies the matmul result before partial_sum is added."""
    batch_size, in_size, out_size, num_groups = 128, 512, 512, 4
    num_local_groups = num_groups - group_offset
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16).astype(lhs_dtype)
    rhs = jax.random.normal(k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    lhs_scale = jnp.full((1, 1), 0.5, dtype=jnp.float32)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)
    ps = jax.random.normal(k2, (batch_size, out_size), dtype=jnp.bfloat16) if has_partial_sum else None

    # Scaling lhs is the same as scaling the matmul result, and reference_gmm adds partial_sum afterwards.
    scaled_lhs = (lhs.astype(jnp.float32) * lhs_scale).astype(jnp.bfloat16)
    expected = reference_gmm(scaled_lhs, rhs, group_sizes, partial_sum=ps, group_offset=group_offset)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        partial_sum=ps,
        group_offset=group_offset,
        lhs_scale=lhs_scale,
        preferred_element_type=jnp.bfloat16,
        maybe_quantize_lhs=False,
    )
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 1024],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[5, 16, 32],
      has_partial_sum=[True, False],
      group_offset=[0, 2, 3],
  )
  def test_tgmm_basic(self, batch_size, in_size, out_size, num_groups, has_partial_sum, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)  # [m, k]
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)  # [m, n]
    group_sizes = get_group_sizes(batch_size, num_groups)
    # if batch_size=128, num_groups=3, an example group_size is
    # group_sizes=Array([14, 14, ..., 7]).
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    ps = None
    if has_partial_sum:
      ps = jax.random.normal(key3, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)

    lhs_t = lhs.swapaxes(0, 1)  # [k, m]
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset, partial_sum=ps)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        partial_sum=ps,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # diff = jnp.abs(expected - actual)
    # max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    # print(f"Output max diff: {jnp.max(diff)} at index {max_diff_idx}")
    # print(f"Output mean diff: {jnp.mean(jnp.abs(expected - actual))}")
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255, 500],
      out_size=[255, 500],
      num_groups=[16],
      group_offset=[0],
  )
  def test_tgmm_implicit_padding(self, batch_size, in_size, out_size, num_groups, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[256, 1024],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16],
      group_offset=[0, 2],
      tile_k=[256, 512],
      tile_n=[256, 512],
  )
  def test_tgmm_with_tile_info(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      tile_k,
      tile_n,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)

    tile_info = gmm_backend.TileSizes(tile_m=256, tile_k=tile_k, tile_n=tile_n)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1, 2, 3],
  )
  def test_tgmm_empty_group(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly zeros output for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    # Redistribute the empty group's tokens to the last group.
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)

    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[256],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1, 2, 3],
  )
  def test_tgmm_empty_group_with_partial_sum(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly preserves partial sum for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    ps = jax.random.normal(key3, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset, partial_sum=ps)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        partial_sum=ps,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  def test_tgmm_explicitly_exercises_all_branches(self):
    # Group 0 (size 4*tile_m, 4 gm tiles): matmul_new_group, matmul, matmul,
    # matmul_group_changing.
    # Group 1 (size 64, 1 gm tile): matmul_new_group_and_changing.

    tile_m = tile_k = tile_n = 256
    in_size = out_size = 256
    num_local_groups = 2
    g0, g1 = 4 * tile_m, 64
    batch_size = g0 + g1

    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([g0, g1], dtype=jnp.int32)
    group_offset = jnp.array(0, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)
    tile_info = gmm_backend.TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      has_bias=[True, False],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[64, 128, 256, 512],
      group_offset=[0, 2, 3],
  )
  def test_gmm_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and common.tpu_generation() < 7:
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  def test_gmm_security_isolation(self):
    """Verifies that sequences (experts) are isolated from each other.

    This test checks that NaNs or extreme values in one expert group do not
    pollute the output of other expert groups, even if they share the same
    sublane tile.
    """
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(42)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_groups, in_size, out_size), dtype=jnp.bfloat16)

    # We use very small group sizes to force expert groups to share tiles.
    # sublane_size is typically 8 or 16.
    group_sizes = jnp.array([4, 4, 4, batch_size - 12], dtype=jnp.int32)

    # 1. Run baseline
    actual_clean = gmm_backend.gmm_v2(lhs, rhs, group_sizes)

    # 2. Inject NaNs into all experts except the first one.
    # If isolation fails, the NaNs will leak into the first expert's output.
    rhs_malicious = rhs.at[1:].set(jnp.nan)
    actual_malicious = gmm_backend.gmm_v2(lhs, rhs_malicious, group_sizes)

    # Verify that the first expert's output is identical and NaN-free.
    first_expert_size = group_sizes[0]
    chex.assert_trees_all_close(
        actual_malicious[:first_expert_size],
        actual_clean[:first_expert_size],
        atol=0.0,
        rtol=0.0,
    )
    self.assertFalse(jnp.any(jnp.isnan(actual_malicious[:first_expert_size])))

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  def test_gmm_uninitialized_memory_robustness(self):
    """Verifies that the kernel is robust against uninitialized scratchpads.

    This test intentionally poisons TPU VMEM/SMEM with NaNs before running the
    GMM kernel. This ensures that  no stale data from previous sessions can leak
    into the output.
    """
    # 1. Poison TPU memory with NaNs
    poison_tpu_memory()

    # 2. Run GMM kernel
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(0)
    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_groups, in_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([batch_size // 4] * 4, dtype=jnp.int32)

    actual = gmm_backend.gmm_v2(lhs, rhs, group_sizes)

    # 3. Verify that the output is NaN-free
    self.assertFalse(jnp.any(jnp.isnan(actual)))

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[1024],
      tile_k=[128, 256, 512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test that quant_block_size > tile_k is handled correctly."""
    if weight_dtype == jnp.float4_e2m1fn and common.tpu_generation() < 7:
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_backend.TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int4, jnp.int8, jnp.float8_e4m3fn],
      block_size=[1024],
      tile_k=[128, 256, 512],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test activation+weight quantized path with quant_block_size > tile_k."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_backend.TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.2, rtol=1.2)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      weight_dtype=[jnp.int4, jnp.uint4, jnp.int8, jnp.float8_e4m3fn],
      block_size=[512, 1024],
      group_offset=[0, 2, 3],
  )
  def test_gmm_activation_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and common.tpu_generation() < 7:
      self.skipTest("Expect TPUv7+")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.1, rtol=1.1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255, 500],
      out_size=[255, 500],
      num_groups=[16],
      has_bias=[True, False],
      group_offset=[0],
  )
  def test_gmm_implicit_padding(self, batch_size, in_size, out_size, num_groups, has_bias, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[500],
      num_groups=[16],
      has_bias=[True, False],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
      block_size=[512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_padding(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertEqual(actual.shape, (batch_size, out_size))
    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      # group_config: (num_groups, group_offset, num_local_groups)
      group_config=[
          # groups 0-1: group<0, groups 2-5: local and active,
          # groups 6-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
          # no negative groups, groups 0-7: local and active,
          # groups 8-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
          # groups 0-3: group<0, groups 4-7: local and active,
          # groups 8-31: group>=num_local_groups
          _GroupConfig(num_groups=32, group_offset=4, num_local_groups=4),
      ],
  )
  def test_gmm_nonlocal_groups_produce_zeros(self, batch_size, in_size, out_size, group_config):
    num_groups, group_offset, num_local_groups = group_config
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True, False],
      use_weight_scale=[True, False],
      maybe_quantize_lhs=[True, False],
      fuse_act=["silu", "swigluoai", "gelu"],
      group_offset=[0, 2],
      block_size=[256, 512],
  )
  def test_gmm_fused_activation(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      use_weight_scale,
      maybe_quantize_lhs,
      fuse_act,
      group_offset,
      block_size,
  ):
    if maybe_quantize_lhs and not use_weight_scale:
      self.skipTest("LHS quantization requires RHS quantization/scale in this config.")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    key = jax.random.key(0)
    final_out_size = out_size // 2
    num_local_groups = num_groups - group_offset

    # 1. Generate Inputs
    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)

    rhs_q = rhs
    rhs_scale = None
    if use_weight_scale:
      rhs_q, rhs_scale = quantize_tensor(rhs, jnp.int8, axis=1, block_size=block_size)
      rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([group_offset], dtype=jnp.int32)

    # 2. Simulate LHS Quantization Noise
    lhs_simulated = lhs
    # because the kernel quantizes LHS in blocks, while reference does it at the
    # whole tensor level, and output is casted down we need to simulate that
    # quantization noise in the reference as well for a fair comparison
    if maybe_quantize_lhs:
      lhs_block_size = min(512, in_size)
      lhs_q, lhs_scale_factor = quantize_tensor(lhs, jnp.int8, axis=1, block_size=lhs_block_size)
      lhs_q_blocked = lhs_q.reshape(batch_size, -1, lhs_block_size).astype(jnp.float32)
      lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
      lhs_simulated = (lhs_q_blocked * lhs_scale_expanded).reshape(lhs.shape).astype(lhs.dtype)

    # 3. Compute Reference Output
    raw_expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    # Slice the reference and apply the activation function
    expected = gmm_backend.apply_act_fn(raw_expected.astype(jnp.float32), fuse_act).astype(lhs.dtype)

    # 4. Compute Actual Kernel Output
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    # 5. Compare Results
    self.assertEqual(actual.shape, (batch_size, final_out_size))

    # tolerances based quantization noise difference between reference and
    # gmm_v2
    if maybe_quantize_lhs:
      atol, rtol = 4.0, 2.0  # Act + Weight Quantization
    elif use_weight_scale:
      atol, rtol = 3e-1, 3e-1  # Weight Quantization Only
    else:
      atol, rtol = 5e-2, 5e-2  # Unquantized Path (bfloat16 precision diffs)

    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
  absltest.main()
