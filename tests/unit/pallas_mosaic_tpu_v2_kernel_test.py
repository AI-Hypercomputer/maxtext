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
import pytest
import re
from typing import Any
from absl import logging
from jax.experimental import layout
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.megablox import common
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend

pytestmark = pytest.mark.tpu_only


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


# ============================================================================
# Spatial-minor TGMM tests
# ============================================================================

Format = layout.Format
Layout = layout.Layout

# Layout constants for spatial-minor TGMM tests.
M2M_012 = (2, 1, 0)  # prints {0,1,2}: k major, n sublane, g minormost (lanes)
M2M_102 = (2, 0, 1)  # prints {1,0,2}: k major, g sublane, n minormost (lanes)

TARGET_N = 2048
TARGET_K = 7168
TARGET_M = 4096


def _fmt(major_to_minor: tuple[int, ...]) -> Any:
  """Builds a concrete single-device Format for the given major_to_minor."""
  return Format(
      Layout(major_to_minor=major_to_minor),
      jax.sharding.SingleDeviceSharding(jax.devices()[0]),
  )


# Target logical shape: [g=16, ?=2048, ?=7168] bf16.
G, D1, D2 = 16, 2048, 7168


def _hlo_text(compiled: Any) -> str:
  text = compiled.as_text()
  assert text is not None
  return text


def _entry_result_layout(compiled: Any) -> str:
  """Extracts the ENTRY computation's result layout string from HLO text."""
  for line in _hlo_text(compiled).splitlines():
    if line.startswith("ENTRY "):
      return line.strip()
  return "<no ENTRY line>"


def _output_bytes(compiled: Any) -> int:
  mem = compiled.memory_analysis()
  assert mem is not None
  return mem.output_size_in_bytes


def _peak_bytes(compiled: Any) -> int:
  """Peak HBM: live temporaries + arguments + output."""
  mem = compiled.memory_analysis()
  assert mem is not None
  return mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes


def entry_computation_layout(compiled: Any) -> str:
  """Extracts the `entry_computation_layout={...}` clause from the HLO header.

  The `ENTRY` line does not carry layouts; the layout string with its tiling
  (e.g. `bf16[16,2048,7168]{0,1,2:T(8,128)(2,1)}`) only appears in the module
  header. This is the authoritative record of what the compiler actually
  materialises, as opposed to what we asked for.

  Args:
    compiled: A compiled JAX executable to read the HLO module header from.

  Returns:
    The full `entry_computation_layout={...}` clause, including the balanced
    braces.
  """
  text = _hlo_text(compiled)
  marker = "entry_computation_layout="
  start = text.find(marker)
  if start < 0:
    raise AssertionError("no entry_computation_layout in HLO module header")
  # The clause runs to the end of the balanced {...} group that follows.
  brace_start = text.index("{", start)
  depth = 0
  for i in range(brace_start, len(text)):
    if text[i] == "{":
      depth += 1
    elif text[i] == "}":
      depth -= 1
      if depth == 0:
        return text[start : i + 1]
  raise AssertionError("unterminated entry_computation_layout clause")


def reference_tgmm_spatial_minor(
    lhs: np.ndarray,  # [m, k]
    rhs: np.ndarray,  # [m, n]
    group_sizes: np.ndarray,  # [size_lhs_group]
    num_groups: int,
    group_offset: int = 0,
) -> np.ndarray:
  """Loop reference producing `[g, n, k]` with `out[g, n, k] == dW_g[k, n]`.

  `group_sizes` partitions the rows of `lhs`/`rhs` in order and is indexed
  globally. `group_offset` selects which window of it is computed: groups
  `group_sizes[group_offset : group_offset + num_groups]`. The groups before
  the offset are not computed but *do* consume rows, so the row cursor starts
  at their cumulative size rather than at zero. The result is written locally,
  i.e. `out[i]` holds the product for global group `group_offset + i`.

  Args:
    lhs: The left-hand side array, `[m, k]`.
    rhs: The right-hand side array, `[m, n]`.
    group_sizes: Per-group row counts, `[size_lhs_group]`.
    num_groups: Number of groups to compute, i.e. `num_actual_groups`.
    group_offset: Index of the first group to compute.

  Returns:
    A `[num_groups, n, k]` float32 array.
  """
  size_k = lhs.shape[1]
  size_n = rhs.shape[1]
  out = np.zeros((num_groups, size_n, size_k), dtype=np.float32)
  # Rows owned by the groups before the offset are skipped, not re-based.
  start = int(np.sum(group_sizes[:group_offset]))
  for i in range(num_groups):
    end = start + int(group_sizes[group_offset + i])
    if end > start:
      # dW_g = lhs[g].T @ rhs[g], shape [k, n]. We store its transpose.
      dw = lhs[start:end].astype(np.float32).T @ rhs[start:end].astype(np.float32)
      out[i] = dw.T
    start = end
  return out


class TgmmSpatialMinorCorrectnessTest(parameterized.TestCase):
  """Numerical correctness of `tgmm_spatial_minor_v2` vs a loop reference."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.named_parameters(
      ("even_2g", 512, 256, 256, [256, 256]),
      ("uneven_2g", 512, 256, 256, [128, 384]),
      ("first_empty", 512, 256, 256, [0, 512]),
      ("last_empty", 512, 256, 256, [512, 0]),
      ("one_row", 512, 256, 256, [1, 511]),
      ("one_row_last", 512, 256, 256, [511, 1]),
      ("all_empty", 512, 256, 256, [0, 0]),
      ("g4_mixed", 512, 256, 384, [0, 200, 0, 312]),
      ("g16_even", 1024, 512, 256, [64] * 16),
      ("g16_sparse", 1024, 512, 256, [0, 0, 512, 0, 0, 0, 512] + [0] * 9),
      ("unaligned_m", 500, 256, 256, [123, 377]),
      # Non-power-of-2 and lane/sublane-unaligned k and n. The kernel pads both
      # up to a whole tile and slices the padding back off, so these exercise
      # that the slice boundaries are right.
      ("unaligned_k", 512, 250, 256, [200, 312]),
      ("unaligned_n", 512, 256, 200, [200, 312]),
      ("odd_k_and_n", 512, 257, 129, [256, 256]),
      ("prime_k_and_n", 512, 199, 61, [100, 412]),
      ("all_unaligned", 500, 250, 200, [123, 377]),
      # Larger g: 64 exceeds the f32 `bg` word alignment of 8 and stays within
      # one 128 lane group chunk; 128 is a full lane's worth of groups.
      ("g64_even", 1024, 256, 256, [16] * 64),
      ("g128_sparse", 1024, 128, 128, [8] * 128),
  )
  def test_matches_reference(self, m, k, n, group_sizes_list):
    num_groups = len(group_sizes_list)
    self.assertLessEqual(sum(group_sizes_list), m)

    rng = np.random.default_rng(0)
    lhs_np = rng.normal(size=(m, k)).astype(np.float32) / 8.0
    rhs_np = rng.normal(size=(m, n)).astype(np.float32) / 8.0
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)

    lhs = jnp.asarray(lhs_np, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rhs_np, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    got = tgmm_backend.tgmm_spatial_minor_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        out_major_to_minor=tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
    )
    self.assertEqual(got.shape, (num_groups, n, k))

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-2, atol=2e-2)

  @parameterized.named_parameters(
      ("even_2g", 512, 256, 256, [256, 256]),
      ("uneven_4g", 512, 256, 384, [0, 200, 0, 312]),
  )
  def test_matches_existing_tgmm_v2(self, m, k, n, group_sizes_list):
    """The new kernel must agree with the shipped `tgmm_v2`, modulo transpose."""
    num_groups = len(group_sizes_list)
    rng = np.random.default_rng(1)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray(group_sizes_list, dtype=np.int32))

    baseline = tgmm_backend.tgmm_v2(  # [g, k, n]
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.float32,
    )
    got = tgmm_backend.tgmm_spatial_minor_v2(  # [g, n, k]
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        out_major_to_minor=tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(jnp.transpose(baseline, (0, 2, 1))),
        rtol=2e-2,
        atol=2e-2,
    )


class TgmmSpatialMinorFallbackEquivalenceTest(parameterized.TestCase):
  """Both dispatch arms must produce the same `[g, n, k]` values.

  `tgmm_spatial_minor_v2` routes to the native spatial-minor Pallas kernel only
  when `out_major_to_minor == SPATIAL_MINOR_MAJOR_TO_MINOR`; every other layout
  falls back to `tgmm_v2` plus a logical transpose. Those are two completely
  different kernels, so the fact that they agree is a property that has to be
  measured on hardware, not asserted by construction.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.named_parameters(
      ("even_2g", 512, 256, 256, [256, 256]),
      ("uneven_2g", 512, 256, 256, [128, 384]),
      ("g4_mixed", 512, 256, 384, [0, 200, 0, 312]),
      ("g16_even", 1024, 512, 256, [64] * 16),
      ("g16_sparse", 1024, 512, 256, [0, 0, 512, 0, 0, 0, 512] + [0] * 9),
      ("unaligned_m", 500, 256, 256, [123, 377]),
      ("unaligned_k", 512, 250, 256, [200, 312]),
      ("unaligned_n", 512, 256, 200, [200, 312]),
      ("all_unaligned", 500, 250, 200, [123, 377]),
      ("g64_even", 1024, 256, 256, [16] * 64),
  )
  def test_fallback_matches_spatial_minor(self, m, k, n, group_sizes_list):
    num_groups = len(group_sizes_list)
    self.assertLessEqual(sum(group_sizes_list), m)

    rng = np.random.default_rng(7)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    def run(out_major_to_minor):
      return tgmm_backend.tgmm_spatial_minor_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=num_groups,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
      )

    spatial = run(tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR)
    # The default, and a third explicit non-spatial-minor permutation, must
    # both take the fallback and land on the same values.
    default = run(tgmm_backend.DEFAULT_MAJOR_TO_MINOR)
    other = run(M2M_102)

    for name, arr in (
        ("spatial", spatial),
        ("default", default),
        ("other", other),
    ):
      self.assertEqual(arr.shape, (num_groups, n, k), msg=name)
      self.assertEqual(arr.dtype, jnp.float32, msg=name)

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    # Both arms must match the NumPy loop reference ...
    np.testing.assert_allclose(np.asarray(spatial), want, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(default), want, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(other), want, rtol=2e-2, atol=2e-2)
    # ... and, more tightly, each other. Both accumulate in f32 on the same
    # MXU, so the only permitted difference is reduction order.
    np.testing.assert_allclose(np.asarray(default), np.asarray(spatial), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.asarray(other), np.asarray(default), rtol=1e-2, atol=1e-2)

  def test_omitting_the_layout_takes_the_fallback(self):
    """`tgmm_gnk_v2` with no layout argument must equal the explicit default.

    This is the behaviour that makes the fast path the one you get by
    accident: `tgmm_gnk_v2` runs `tgmm_v2` plus a transpose unless asked
    otherwise. The values are the same either way, which is what this pins
    down. (`tgmm_spatial_minor_v2` defaults the other way; see
    `EntryPointNamingTest`.)
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [200, 312]
    num_groups = len(group_sizes_list)
    rng = np.random.default_rng(11)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    implicit = tgmm_backend.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.float32,
    )
    explicit = tgmm_backend.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        out_major_to_minor=tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_array_equal(np.asarray(implicit), np.asarray(explicit))

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    np.testing.assert_allclose(np.asarray(implicit), want, rtol=2e-2, atol=2e-2)


class TgmmSpatialMinorGroupOffsetTest(parameterized.TestCase):
  """Non-zero `group_offset` must be honoured identically by both arms.

  `group_offset` selects a window of `group_sizes`: the kernel computes groups
  `group_sizes[q : q + num_actual_groups]` and writes them to `out[0:...]`,
  i.e. the output is indexed *locally* while `group_sizes` and the row cursor
  are indexed *globally*. The groups before `q` are not computed but still own
  rows, so they advance the row cursor.

  Both arms reach this through completely separate group metadata. The
  `tgmm_v2` fallback fills it with `gmm_v2.fill_metadata`. The spatial minor
  wrapper precomputes `m_bounds`, `g_first` and `g_last` from `group_sizes`
  and `group_offset`, and prefetches them into SMEM. Agreement is therefore a
  property that has to be measured, not assumed. Every case below checks both
  arms against the NumPy reference *and* against each other.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.named_parameters(
      # (name, m, k, n, full group_sizes, num_actual_groups, group_offset)
      ("skip_first_of_4", 512, 256, 256, [128, 96, 160, 128], 2, 1),
      ("skip_two_of_4", 512, 256, 256, [128, 96, 160, 128], 2, 2),
      ("last_group_only", 512, 256, 256, [128, 96, 160, 128], 1, 3),
      ("whole_window_offset_0", 512, 256, 256, [128, 96, 160, 128], 4, 0),
      # The skipped prefix is empty, so the row cursor must still start at 0.
      ("skipped_prefix_empty", 512, 256, 256, [0, 200, 0, 312], 2, 1),
      # The computed window contains empty groups.
      ("empty_inside_window", 512, 256, 256, [100, 0, 412, 0], 3, 1),
      # A window that starts unaligned to the sublane tiling of m.
      ("unaligned_row_cursor", 512, 256, 256, [100, 150, 162, 100], 2, 1),
      # m is not a whole number of sublane rows, so the dispatcher pads it.
      ("unaligned_m", 500, 256, 256, [123, 77, 200, 100], 2, 1),
      # Non-tile-aligned k and n on top of an offset.
      ("unaligned_k_and_n", 512, 250, 200, [128, 96, 160, 128], 2, 1),
      # A wider group axis: window of 4 taken from the middle of 16.
      ("g16_window_4", 1024, 256, 256, [64] * 16, 4, 6),
      # Offset past a long run of empty groups.
      (
          "g16_sparse_offset",
          1024,
          256,
          256,
          [0, 0, 512, 0, 0, 0, 512] + [0] * 9,
          8,
          4,
      ),
  )
  def test_group_offset_matches_reference_on_both_arms(self, m, k, n, group_sizes_list, num_groups, offset):
    self.assertLessEqual(sum(group_sizes_list), m)
    self.assertLessEqual(offset + num_groups, len(group_sizes_list))

    rng = np.random.default_rng(23)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)
    group_offset = jnp.asarray(np.asarray([offset], dtype=np.int32))

    def run(out_major_to_minor):
      return tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_groups,
          group_offset,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
      )

    spatial = run(tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR)
    fallback = run(tgmm_backend.DEFAULT_MAJOR_TO_MINOR)

    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
        group_offset=offset,
    )
    self.assertEqual(want.shape, (num_groups, n, k))

    for name, arr in (("spatial", spatial), ("fallback", fallback)):
      self.assertEqual(arr.shape, (num_groups, n, k), msg=name)
      np.testing.assert_allclose(np.asarray(arr), want, rtol=2e-2, atol=2e-2, err_msg=name)
    # The two arms must also agree with each other, more tightly than either
    # agrees with the f32 reference.
    np.testing.assert_allclose(np.asarray(fallback), np.asarray(spatial), rtol=1e-2, atol=1e-2)

  @parameterized.named_parameters(
      ("spatial_minor", tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR),
      ("fallback", tgmm_backend.DEFAULT_MAJOR_TO_MINOR),
  )
  def test_offset_actually_shifts_the_window(self, out_major_to_minor):
    """A different offset must give a different answer.

    Without this, every assertion in
    `test_group_offset_matches_reference_on_both_arms` would still pass if
    `group_offset` were quietly ignored by *both* the kernel and the
    reference, since they would then be consistently wrong together. Here the
    group sizes and contents differ per group, so shifting the window by one
    must change the result, and the result at offset `q` must equal the
    reference's window at offset `q`.

    Args:
      out_major_to_minor: Layout hint selecting the dispatch arm.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = 2
    rng = np.random.default_rng(29)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(group_sizes_np)

    def run(offset):
      return np.asarray(
          tgmm_backend.tgmm_gnk_v2(
              lhs,
              rhs,
              group_sizes,
              num_groups,
              jnp.asarray(np.asarray([offset], dtype=np.int32)),
              out_major_to_minor=out_major_to_minor,
              preferred_element_type=jnp.float32,
          )
      )

    at0, at1, at2 = run(0), run(1), run(2)

    # Sanity: the windows genuinely differ, so the comparisons below have
    # something to catch.
    self.assertFalse(np.allclose(at0, at1, rtol=1e-3, atol=1e-3))
    self.assertFalse(np.allclose(at1, at2, rtol=1e-3, atol=1e-3))

    lhs_np = np.asarray(lhs, dtype=np.float32)
    rhs_np = np.asarray(rhs, dtype=np.float32)
    for offset, got in ((0, at0), (1, at1), (2, at2)):
      want = reference_tgmm_spatial_minor(lhs_np, rhs_np, group_sizes_np, num_groups, group_offset=offset)
      np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2, err_msg=f"offset={offset}")

    # An explicit zero offset must be identical to omitting the argument.
    implicit = np.asarray(
        tgmm_backend.tgmm_gnk_v2(
            lhs,
            rhs,
            group_sizes,
            num_groups,
            out_major_to_minor=out_major_to_minor,
            preferred_element_type=jnp.float32,
        )
    )
    np.testing.assert_array_equal(implicit, at0)

  @parameterized.named_parameters(
      ("spatial_minor", tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR),
      ("fallback", tgmm_backend.DEFAULT_MAJOR_TO_MINOR),
  )
  def test_offset_accepts_scalar_and_rank1_spellings(self, out_major_to_minor):
    """A Python int, a 0-d array and a `(1,)` array must all mean the same.

    Args:
      out_major_to_minor: Layout hint selecting the dispatch arm.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = 2
    rng = np.random.default_rng(31)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray(group_sizes_list, dtype=np.int32))

    def run(offset):
      return np.asarray(
          tgmm_backend.tgmm_gnk_v2(
              lhs,
              rhs,
              group_sizes,
              num_groups,
              offset,
              out_major_to_minor=out_major_to_minor,
              preferred_element_type=jnp.float32,
          )
      )

    rank1 = run(jnp.asarray(np.asarray([1], dtype=np.int32)))
    scalar_0d = run(jnp.asarray(np.int32(1)))
    python_int = run(1)

    np.testing.assert_array_equal(scalar_0d, rank1)
    np.testing.assert_array_equal(python_int, rank1)


class TgmmSpatialMinorInputValidationTest(parameterized.TestCase):
  """Malformed operands must be rejected up front, not deep inside Pallas."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def _call(self, lhs, rhs, group_sizes, num_groups=2, **kwargs):
    return tgmm_backend.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=num_groups,
        preferred_element_type=jnp.float32,
        **kwargs,
    )

  @parameterized.named_parameters(
      ("lhs_3d", (2, 256, 128), (512, 256), "lhs", 3),
      ("lhs_1d", (512,), (512, 256), "lhs", 1),
      ("rhs_3d", (512, 128), (2, 256, 256), "rhs", 3),
      ("rhs_1d", (512, 128), (512,), "rhs", 1),
      ("lhs_4d", (1, 1, 512, 128), (512, 256), "lhs", 4),
  )
  def test_rejects_wrong_rank(self, lhs_shape, rhs_shape, operand, rank):
    lhs = jnp.zeros(lhs_shape, dtype=jnp.bfloat16)
    rhs = jnp.zeros(rhs_shape, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes)

    message = str(ctx.exception)
    self.assertIn(f"tgmm {operand} must be a rank-2", message)
    self.assertIn(f"got rank {rank}", message)
    self.assertIn("Batched (>2D) operands are not supported", message)

  def test_rejects_mismatched_m(self):
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((256, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes)

    message = str(ctx.exception)
    self.assertIn("must agree on the contracted size_m dimension", message)
    self.assertIn("lhs.shape[0]=512", message)
    self.assertIn("rhs.shape[0]=256", message)

  def test_rejects_non_1d_group_sizes(self):
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.zeros((2, 1), dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes)

    message = str(ctx.exception)
    self.assertIn("tgmm group_sizes must be a rank-1", message)
    self.assertIn("got rank 2", message)

  @parameterized.named_parameters(
      ("not_a_permutation", (0, 1, 1)),
      ("wrong_length", (0, 1)),
      ("out_of_range", (0, 1, 3)),
  )
  def test_rejects_bad_layout(self, out_major_to_minor):
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))

    with self.assertRaises(ValueError) as ctx:
      self._call(lhs, rhs, group_sizes, out_major_to_minor=out_major_to_minor)

    message = str(ctx.exception)
    self.assertIn("out_major_to_minor must be a permutation of (0, 1, 2)", message)
    self.assertIn(str(out_major_to_minor), message)

  def test_accepts_valid_shapes(self):
    """The happy path must survive all of the above checks."""
    lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    out = self._call(lhs, rhs, group_sizes)
    self.assertEqual(out.shape, (2, 256, 128))

  def test_validate_tgmm_inputs_without_operands_is_unchanged(self):
    """Existing `tgmm_v2` callers pass no operands and must be unaffected."""
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    tgmm_backend.validate_tgmm_inputs(group_sizes, 2)

    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.validate_tgmm_inputs(group_sizes, 4)
    self.assertIn("must be >= group_offset", str(ctx.exception))

  def test_validate_tgmm_inputs_with_operands(self):
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    good_lhs = jnp.zeros((512, 128), dtype=jnp.bfloat16)
    good_rhs = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    tgmm_backend.validate_tgmm_inputs(group_sizes, 2, lhs=good_lhs, rhs=good_rhs)

    bad_lhs = jnp.zeros((2, 256, 128), dtype=jnp.bfloat16)
    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.validate_tgmm_inputs(group_sizes, 2, lhs=bad_lhs, rhs=good_rhs)
    self.assertIn("tgmm lhs must be a rank-2", str(ctx.exception))

    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.validate_tgmm_inputs(group_sizes, 2, lhs=good_lhs)
    self.assertIn("takes lhs and rhs together or not at all", str(ctx.exception))


class TgmmSpatialMinorDtypeCoverageTest(parameterized.TestCase):
  """Value correctness across input, output and accumulator dtypes.

  Two things vary independently and both matter:

    * The *input* dtype sets the sublane tiling of `m` (8 for f32, 16 for
      bf16), so it changes how the dispatcher pads `m` and how both kernels
      reshape their operands.
    * The *output* dtype is also the stage dtype on the spatial minor path, so
      it selects the epilogue: bf16 packs pairs of `n` rows into uint32 words
      before transposing, while f32 transposes directly.

  `num_groups = 4` pads the group axis to `Gp = 8` on the spatial minor path,
  so every case also covers the zeroed padding groups and the final slice
  that drops them.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.named_parameters(
      # (name, in_dtype, preferred_element_type, acc_dtype, tol)
      ("bf16_in_f32_out", jnp.bfloat16, jnp.float32, None, 2e-2),
      ("bf16_in_bf16_out", jnp.bfloat16, jnp.bfloat16, None, 4e-2),
      ("bf16_in_default_out", jnp.bfloat16, None, None, 4e-2),
      ("f32_in_f32_out", jnp.float32, jnp.float32, None, 1e-2),
      ("f32_in_bf16_out", jnp.float32, jnp.bfloat16, None, 4e-2),
      ("f32_in_default_out", jnp.float32, None, None, 1e-2),
      # Explicit accumulator alongside an explicit output dtype.
      ("f32_in_f32_out_f32_acc", jnp.float32, jnp.float32, jnp.float32, 1e-2),
      (
          "bf16_in_bf16_out_f32_acc",
          jnp.bfloat16,
          jnp.bfloat16,
          jnp.float32,
          4e-2,
      ),
      ("f32_in_bf16_out_f32_acc", jnp.float32, jnp.bfloat16, jnp.float32, 4e-2),
  )
  def test_dtypes_match_reference_on_both_arms(self, in_dtype, out_dtype, acc_dtype, tol):
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = len(group_sizes_list)

    rng = np.random.default_rng(37)
    lhs_np = rng.normal(size=(m, k)).astype(np.float32) / 8.0
    rhs_np = rng.normal(size=(m, n)).astype(np.float32) / 8.0
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)

    lhs = jnp.asarray(lhs_np, dtype=in_dtype)
    rhs = jnp.asarray(rhs_np, dtype=in_dtype)
    group_sizes = jnp.asarray(group_sizes_np)

    # `preferred_element_type=None` means "use lhs.dtype".
    want_dtype = jnp.dtype(out_dtype if out_dtype is not None else in_dtype)

    def run(out_major_to_minor):
      return tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=num_groups,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=out_dtype,
          acc_dtype=acc_dtype,
      )

    spatial = run(tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR)
    fallback = run(tgmm_backend.DEFAULT_MAJOR_TO_MINOR)

    # Round the reference through the input dtype so the comparison measures
    # the kernel, not the operand quantisation.
    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )

    for name, arr in (("spatial", spatial), ("fallback", fallback)):
      self.assertEqual(arr.shape, (num_groups, n, k), msg=name)
      self.assertEqual(arr.dtype, want_dtype, msg=name)
      np.testing.assert_allclose(
          np.asarray(arr, dtype=np.float32),
          want,
          rtol=tol,
          atol=tol,
          err_msg=name,
      )
    np.testing.assert_allclose(
        np.asarray(fallback, dtype=np.float32),
        np.asarray(spatial, dtype=np.float32),
        rtol=tol,
        atol=tol,
    )

  @parameterized.named_parameters(
      ("f32_out_align_g_8", jnp.float32),
      ("bf16_out_align_g_16", jnp.bfloat16),
  )
  def test_explicit_bg_respects_output_dtype_alignment(self, out_dtype):
    """`bg` must be a multiple of `32 // out_dtype.itemsize`.

    `align_g` is 8 for an f32 output and 16 for a bf16 one, so the same `bg`
    is legal for one and rejected for the other. This pins that the rule is
    driven by the output dtype rather than the input dtype.

    Args:
      out_dtype: The `preferred_element_type` under test.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = len(group_sizes_list)
    align_g = 32 // jnp.dtype(out_dtype).itemsize

    rng = np.random.default_rng(41)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes_np = np.asarray(group_sizes_list, dtype=np.int32)
    group_sizes = jnp.asarray(group_sizes_np)

    def run(bg):
      return tgmm_backend.tgmm_spatial_minor_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=num_groups,
          preferred_element_type=out_dtype,
          bg=bg,
      )

    # An aligned bg computes the right answer.
    got = run(align_g)
    want = reference_tgmm_spatial_minor(
        np.asarray(lhs, dtype=np.float32),
        np.asarray(rhs, dtype=np.float32),
        group_sizes_np,
        num_groups,
    )
    np.testing.assert_allclose(np.asarray(got, dtype=np.float32), want, rtol=4e-2, atol=4e-2)

    # A misaligned one is rejected, naming the required multiple.
    with self.assertRaises(ValueError) as ctx:
      run(align_g + 1)
    self.assertIn(f"must be a multiple of {align_g}", str(ctx.exception))


class TgmmSpatialMinorOnlyArgumentTest(parameterized.TestCase):
  """Arguments only the spatial-minor kernel can honour must not be dropped.

  `tile_info` and `bg` belong to the spatial minor Pallas kernel: `tile_info`
  sets its tiling and `bg` is validated against its output alignment. On the
  `tgmm_v2` fallback there is nowhere to forward them, so supplying them
  alongside a non-spatial-minor `out_major_to_minor` is always a caller
  mistake: the request would otherwise vanish with no signal. `None` means
  "not supplied" and must keep working on both arms.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def _operands(self, m=512, k=256, n=256):
    rng = np.random.default_rng(43)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    return lhs, rhs, group_sizes

  @parameterized.named_parameters(
      ("default", tgmm_backend.DEFAULT_MAJOR_TO_MINOR),
      # `M2M_102` is defined further down the module, so spell it out here.
      ("other_permutation", (2, 0, 1)),
  )
  def test_bg_on_fallback_is_rejected(self, out_major_to_minor):
    """An explicit `bg` on a fallback layout is refused, naming the argument.

    Args:
      out_major_to_minor: A non-spatial-minor layout hint.
    """
    lhs, rhs, group_sizes = self._operands()
    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
          bg=16,
      )
    message = str(ctx.exception)
    self.assertIn("bg", message)
    self.assertIn("spatial-minor kernel only", message)
    self.assertIn(str(out_major_to_minor), message)

  def test_tile_info_on_fallback_is_rejected(self):
    lhs, rhs, group_sizes = self._operands()
    tiles = gmm_backend.TileSizes(tile_m=128, tile_k=128, tile_n=128)
    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
          preferred_element_type=jnp.float32,
          tile_info=tiles,
      )
    message = str(ctx.exception)
    self.assertIn("tile_info", message)
    self.assertIn("tgmm_v2 directly", message)

  def test_both_are_named_in_one_message(self):
    lhs, rhs, group_sizes = self._operands()
    tiles = gmm_backend.TileSizes(tile_m=128, tile_k=128, tile_n=128)
    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
          preferred_element_type=jnp.float32,
          tile_info=tiles,
          bg=16,
      )
    message = str(ctx.exception)
    self.assertIn("bg", message)
    self.assertIn("tile_info", message)

  def test_accepted_on_the_spatial_minor_arm(self):
    """The same arguments are accepted when the spatial minor kernel runs."""
    lhs, rhs, group_sizes = self._operands()
    out = tgmm_backend.tgmm_gnk_v2(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups=2,
        out_major_to_minor=tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.float32,
        tile_info=gmm_backend.TileSizes(tile_m=128, tile_k=128, tile_n=128),
        bg=8,
    )
    self.assertEqual(out.shape, (2, 256, 256))

  def test_omitting_them_still_works_on_both_arms(self):
    """`None` must continue to mean "not supplied" on the fallback."""
    lhs, rhs, group_sizes = self._operands()
    for out_major_to_minor in (
        tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
        tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR,
    ):
      out = tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=2,
          out_major_to_minor=out_major_to_minor,
          preferred_element_type=jnp.float32,
          tile_info=None,
          bg=None,
      )
      self.assertEqual(out.shape, (2, 256, 256), msg=str(out_major_to_minor))

  def test_rejects_group_sizes_shorter_than_num_actual_groups(self):
    """The offset-independent half of the `validate_tgmm_inputs` bound."""
    lhs, rhs, _ = self._operands()
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    with self.assertRaises(ValueError) as ctx:
      tgmm_backend.tgmm_gnk_v2(
          lhs,
          rhs,
          group_sizes,
          num_actual_groups=4,
          preferred_element_type=jnp.float32,
      )
    message = str(ctx.exception)
    self.assertIn("group_sizes has 2 entries", message)
    self.assertIn("num_actual_groups=4", message)


class TgmmSpatialMinorEntryPointNamingTest(parameterized.TestCase):
  """Each entry point's *default* must match what its name promises.

  `tgmm_gnk_v2` is named for its result shape, `[g, n, k]`, and says nothing
  about the implementation, so it defaults to the fast `tgmm_v2` fallback.
  `tgmm_spatial_minor_v2` is named for the kernel, so it defaults to running
  it. Passing `out_major_to_minor` explicitly makes the two identical.

  The claim is invisible in the returned values (both arms compute the same
  thing), so it is read out of the Pallas scope name in the compiled HLO.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def _compile(self, fn):
    m, k, n = 512, 256, 256
    rng = np.random.default_rng(47)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray([256, 256], dtype=np.int32))
    return jax.jit(fn).lower(lhs, rhs, group_sizes).compile()

  def test_gnk_default_selects_the_fallback(self):
    compiled = self._compile(
        lambda l, r, g: tgmm_backend.tgmm_gnk_v2(l, r, g, num_actual_groups=2, preferred_element_type=jnp.float32)
    )
    scope = kernel_scope_from_hlo(compiled)
    logging.info("tgmm_gnk_v2 default scope: %s", scope)
    self.assertStartsWith(scope, "tgmm_v2-")
    self.assertNotIn("tgmm_spatial_minor_v2", scope)

  def test_spatial_minor_default_selects_the_spatial_minor_kernel(self):
    compiled = self._compile(
        lambda l, r, g: tgmm_backend.tgmm_spatial_minor_v2(
            l, r, g, num_actual_groups=2, preferred_element_type=jnp.float32
        )
    )
    scope = kernel_scope_from_hlo(compiled)
    logging.info("tgmm_spatial_minor_v2 default scope: %s", scope)
    self.assertStartsWith(scope, "tgmm_spatial_minor_v2-")

  def test_alias_forwards_an_explicit_layout(self):
    """With an explicit layout the alias must behave exactly like `gnk`."""
    compiled = self._compile(
        lambda l, r, g: tgmm_backend.tgmm_spatial_minor_v2(
            l,
            r,
            g,
            num_actual_groups=2,
            out_major_to_minor=tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
            preferred_element_type=jnp.float32,
        )
    )
    scope = kernel_scope_from_hlo(compiled)
    self.assertStartsWith(scope, "tgmm_v2-")

  @parameterized.named_parameters(
      ("spatial_minor", tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR),
      ("fallback", tgmm_backend.DEFAULT_MAJOR_TO_MINOR),
  )
  def test_alias_and_gnk_agree_value_for_value(self, out_major_to_minor):
    """Given the same explicit layout, both names must return the same array.

    Args:
      out_major_to_minor: The layout hint passed explicitly to both.
    """
    m, k, n = 512, 256, 256
    group_sizes_list = [128, 96, 160, 128]
    num_groups = len(group_sizes_list)
    rng = np.random.default_rng(53)
    lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
    group_sizes = jnp.asarray(np.asarray(group_sizes_list, dtype=np.int32))
    group_offset = jnp.asarray(np.asarray([1], dtype=np.int32))

    kwargs = {
        "num_actual_groups": num_groups - 1,
        "out_major_to_minor": out_major_to_minor,
        "preferred_element_type": jnp.float32,
    }
    via_gnk = tgmm_backend.tgmm_gnk_v2(lhs, rhs, group_sizes, group_offset=group_offset, **kwargs)
    via_alias = tgmm_backend.tgmm_spatial_minor_v2(lhs, rhs, group_sizes, group_offset=group_offset, **kwargs)
    np.testing.assert_array_equal(np.asarray(via_gnk), np.asarray(via_alias))


# ---------------------------------------------------------------------------
# Target configuration, benchmark arms.
# ---------------------------------------------------------------------------

# The deliverable tensor is bf16[16, 2048, 7168] with layout {0,1,2}. The
# kernel's logical result is [g, n, k], so n = 2048 and k = 7168.

# `major_to_minor` tuples, and the layout string each one prints as. XLA's
# layout string is `minor_to_major`, i.e. the reverse of `major_to_minor`
# (xla_data.proto LayoutProto: "from minor (fastest varying index) to major").

_MIB = 1024 * 1024


def even_group_sizes(m: int, g: int) -> np.ndarray:
  sizes = np.full((g,), m // g, dtype=np.int32)
  sizes[: m % g] += 1
  return sizes


def make_inputs(m: int, k: int, n: int, g: int, seed: int = 0):
  rng = np.random.default_rng(seed)
  lhs = jnp.asarray(rng.normal(size=(m, k)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
  rhs = jnp.asarray(rng.normal(size=(m, n)).astype(np.float32) / 8.0, dtype=jnp.bfloat16)
  return lhs, rhs, jnp.asarray(even_group_sizes(m, g))


def arm_tgmm_v2_plus_relayout(g: int):
  """Shipped `tgmm_v2` -> `[g, k, n]`, then a logical transpose to `[g, n, k]`.

  Whether that transpose is a real relayout copy or a free bitcast depends
  entirely on the output layout it is pinned to, which is what arms 1 and 3
  vary.

  Args:
    g: Number of groups (experts) to pass as `num_actual_groups`.

  Returns:
    A callable `(lhs, rhs, group_sizes) -> [g, n, k]` array.
  """

  def f(lhs, rhs, gs):
    out = tgmm_backend.tgmm_v2(lhs, rhs, gs, num_actual_groups=g, preferred_element_type=jnp.bfloat16)
    return jnp.transpose(out, (0, 2, 1))

  return f


def arm_spatial_minor(g: int, vmem_limit_bytes: int | None = None):
  """The new kernel, which emits the `[k, n, g]` byte order natively.

  Args:
    g: Number of groups (experts) to pass as `num_actual_groups`.
    vmem_limit_bytes: Optional scoped-VMEM reservation override.

  Returns:
    A callable `(lhs, rhs, group_sizes) -> [g, n, k]` array.
  """

  def f(lhs, rhs, gs):
    return tgmm_backend.tgmm_spatial_minor_v2(
        lhs,
        rhs,
        gs,
        num_actual_groups=g,
        out_major_to_minor=tgmm_backend.SPATIAL_MINOR_MAJOR_TO_MINOR,
        preferred_element_type=jnp.bfloat16,
        vmem_limit_bytes=vmem_limit_bytes,
    )

  return f


def arm_dispatch(g: int, out_major_to_minor: tuple[int, ...]):
  """`tgmm_spatial_minor_v2` invoked through its layout dispatch.

  Args:
    g: Number of groups (experts) to pass as `num_actual_groups`.
    out_major_to_minor: The layout hint that selects the implementation.

  Returns:
    A callable `(lhs, rhs, group_sizes) -> [g, n, k]` array.
  """

  def f(lhs, rhs, gs):
    return tgmm_backend.tgmm_spatial_minor_v2(
        lhs,
        rhs,
        gs,
        num_actual_groups=g,
        out_major_to_minor=out_major_to_minor,
        preferred_element_type=jnp.bfloat16,
    )

  return f


def jit_to_layout(f, major_to_minor):
  return jax.jit(f, out_shardings=_fmt(major_to_minor))


def kernel_scope_from_hlo(compiled: Any) -> str:
  """Recovers the tiling the kernel actually selected, from the HLO.

  The two kernels name their scopes differently: `tgmm_spatial_minor_v2`
  encodes `tm/tk/tn/bg`, while the shipped `tgmm_v2` encodes `act` and no tile
  sizes. Match both.

  Args:
    compiled: A compiled JAX executable to search for a Pallas scope name.

  Returns:
    The lexicographically first matching scope name, or a placeholder string
    if the HLO contains no Pallas scope.
  """
  pattern = r"tgmm(?:_spatial_minor)?_v2-g_\d+-m_\d+-k_\d+-[A-Za-z0-9_\-]+"
  found = re.findall(pattern, _hlo_text(compiled))
  return sorted(set(found))[0] if found else "<no pallas scope in HLO>"


ARMS = (
    ("arm1_tgmm_v2_plus_copy", arm_tgmm_v2_plus_relayout, M2M_012, "{0,1,2}"),
    ("arm2_spatial_minor", arm_spatial_minor, M2M_012, "{0,1,2}"),
    ("arm3_tgmm_v2_to_102", arm_tgmm_v2_plus_relayout, M2M_102, "{1,0,2}"),
)


class TgmmSpatialMinorLayoutAssertionTest(parameterized.TestCase):
  """The new kernel must materialise exactly the requested layout."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  def test_spatial_minor_produces_target_layout(self):
    g = 16
    lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
    f = jit_to_layout(arm_spatial_minor(g), M2M_012)
    compiled = f.lower(lhs, rhs, gs).compile()

    clause = entry_computation_layout(compiled)
    logging.info("arm2 g=16 entry_computation_layout:\n%s", clause)
    logging.info("arm2 g=16 kernel scope: %s", kernel_scope_from_hlo(compiled))
    logging.info("arm2 g=16 output bytes: %d", _output_bytes(compiled))

    self.assertIn("bf16[16,2048,7168]{0,1,2:T(8,128)(2,1)}", clause)

  def test_all_arms_report_their_layouts(self):
    """Records what each arm actually materialises, for the report."""
    for g in (16, 128):
      lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
      for name, arm, m2m, expect in ARMS:
        compiled = jit_to_layout(arm(g), m2m).lower(lhs, rhs, gs).compile()
        clause = entry_computation_layout(compiled)
        logging.info(
            "g=%d %s expect=%s\n  scope=%s\n  out_bytes=%d peak_bytes=%d\n  %s",
            g,
            name,
            expect,
            kernel_scope_from_hlo(compiled),
            _output_bytes(compiled),
            0,
            clause,
        )
        # The layout string always carries its tiling, e.g.
        # `bf16[16,2048,7168]{0,1,2:T(8,128)(2,1)}`, so match up to the colon.
        self.assertIn(f"bf16[{g},2048,7168]{expect[:-1]}:", clause.replace(" ", ""))


class TgmmSpatialMinorFallbackLayoutTest(parameterized.TestCase):
  """The layout dispatch must pick the right kernel and honour the layout.

  Two independent claims are checked from the compiled HLO, at the real target
  shape:

    1. The requested layout genuinely materialises (`entry_computation_layout`).
    2. The non-spatial-minor requests genuinely avoid the slow kernel. That is
       the whole point of the fallback, and it is invisible in the values, so
       it has to be read out of the Pallas scope name in the HLO.
  """

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")

  @parameterized.named_parameters(
      # (name, out_major_to_minor hint, layout to pin, expected layout string,
      #  expected Pallas scope prefix)
      (
          "spatial_minor_012",
          M2M_012,
          M2M_012,
          "{0,1,2}",
          "tgmm_spatial_minor_v2-",
      ),
      ("fallback_102", M2M_102, M2M_102, "{1,0,2}", "tgmm_v2-"),
      (
          "fallback_default_210",
          tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
          tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
          "{2,1,0}",
          "tgmm_v2-",
      ),
      # The honest worst case: the caller wants `{0,1,2}` but did not ask for
      # the spatial-minor kernel, so they get `tgmm_v2` plus a real relayout
      # copy. At this shape that is still faster than the kernel it declines
      # to use; see `BenchmarkTest`.
      (
          "fallback_hint_mismatch_012",
          tgmm_backend.DEFAULT_MAJOR_TO_MINOR,
          M2M_012,
          "{0,1,2}",
          "tgmm_v2-",
      ),
  )
  def test_dispatch_selects_kernel_and_layout(self, hint, pinned, expect_layout, expect_scope_prefix):
    g = 16
    lhs, rhs, gs = make_inputs(TARGET_M, TARGET_K, TARGET_N, g)
    compiled = jit_to_layout(arm_dispatch(g, hint), pinned).lower(lhs, rhs, gs).compile()

    clause = entry_computation_layout(compiled)
    scope = kernel_scope_from_hlo(compiled)
    logging.info("hint=%s pinned=%s -> scope=%s\n  %s", hint, pinned, scope, clause)

    self.assertIn(
        f"bf16[{g},{TARGET_N},{TARGET_K}]{expect_layout[:-1]}:",
        clause.replace(" ", ""),
    )
    self.assertStartsWith(scope, expect_scope_prefix)
    if expect_scope_prefix == "tgmm_v2-":
      self.assertNotIn("tgmm_spatial_minor_v2", scope)


if __name__ == "__main__":
  absltest.main()
