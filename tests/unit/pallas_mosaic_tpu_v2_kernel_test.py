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
import contextlib
import functools
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
import numpy as np


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


# gmm_v2's tail chunks (`gmm_v2(tail_chunks=True)`; MaxText's moe_gmm_v2_tail_chunks). On TPU7x, a gmm_v2 tile whose
# valid rows fit in half the tile runs the unchanged matmul once per 128-row chunk (256-row with transposed rhs), and
# the lhs and out blocks are triple-buffered. make_gmm_configs allows this only for configs where it was checked
# bit-identical; every other call, and every call without the flag, traces the kernel without it. The kernel then
# counts the call's short tiles and runs the chunks only if at least 1/16 of its tiles are short, else the kernel
# without them. The CPU tests below check that the flag reaches gmm_v2, pin the gate, check that count and lower the
# kernel outside the gate; they mock the kernels, target an abstract TPU device or run in Pallas interpret mode, so
# they need no TPU. GmmV2TailChunksTpuTest checks on TPU7x that the chunks change no output bit.

# (k, n, (tile_m, tile_k, tile_n)) of gmm_v2 calls inside the gate. The gemma4-26b calls (embed 2816, expert mlp 704)
# use its v7x recipe tiles for the forward and input gradient of wi and wo; the input gradients read the weights with
# transposed rhs, so they run 256-row chunks.
_TAIL_CHUNK_CALLS = {
    "gemma4_wi_fwd": {"k": 2816, "n": 704, "tiles": (1024, 2816, 768)},
    "gemma4_wi_dlhs": {"k": 704, "n": 2816, "tiles": (1024, 768, 2816), "transpose_rhs": True},
    "gemma4_wo_fwd": {"k": 704, "n": 2816, "tiles": (1024, 768, 2816)},
    "gemma4_wo_dlhs": {"k": 2816, "n": 704, "tiles": (1024, 2816, 768), "transpose_rhs": True},
    "tile_m_512": {"k": 704, "n": 2816, "tiles": (512, 768, 2816)},
    "tile_m_256": {"k": 2048, "n": 1024, "tiles": (256, 2048, 1024)},
    "several_n_tiles": {"k": 1024, "n": 1408, "tiles": (1024, 1024, 768)},  # Two N tiles; the second is partial.
    "tile_k_of_11_lanes": {"k": 1408, "n": 768, "tiles": (1024, 1408, 768)},
    # base.yml's default input-gradient tiles (512/1024/1024) for a Qwen3-30B-A3B-like wi: one 256-row chunk.
    "transposed_tile_m_512": {"k": 768, "n": 2048, "tiles": (512, 1024, 1024), "transpose_rhs": True},
    # The narrowest tile_n the gate allows, with an odd number of 256-wide K slices (3).
    "tile_n_512": {"k": 768, "n": 512, "tiles": (512, 768, 512)},
}

# Group sizes at and around sublane (16), chunk (128, 256), half-tile and tile boundaries, and empty groups. They
# also move later groups' first rows off the sublane grid.
_EDGE_SIZES = (0, 1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 1000, 1023, 1024, 1025)
_EDGE_SIZES += (1536, 2048, 2049, 3, 700, 64)


def _short_tiles(sizes, group_offset, num_groups, tile_m, short_rows):
  """(tiles, short tiles) of a routing, where a short tile has at most `short_rows` rows.

  Each group is tiled from its first sublane row, so only its last tile can be short.
  """
  start, tiles, short = 0, 0, 0
  for group, size in enumerate(sizes):
    size = int(size)
    if size > 0 and group_offset <= group < group_offset + num_groups:
      rows = start % 16 + size
      group_tiles = -(-rows // tile_m)
      tiles += group_tiles
      short += rows - (group_tiles - 1) * tile_m <= short_rows
    start += size
  return tiles, short


@contextlib.contextmanager
def _on_chip(device_kind: str):
  """Makes `pltpu.get_tpu_info()` describe `device_kind`, an abstract device (no TPU needed)."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device)):
    yield


def _operand_shapes(*, k, n, groups, rows_per_group, local_groups=None, transpose_rhs=False, dtype=jnp.bfloat16):
  """lhs, rhs, group_sizes and group_offset of a gmm_v2 call, as shapes."""
  local_groups = local_groups or groups
  lhs = jax.ShapeDtypeStruct((groups * rows_per_group, k), dtype)
  rhs = jax.ShapeDtypeStruct((local_groups, n, k) if transpose_rhs else (local_groups, k, n), dtype)
  return lhs, rhs, jax.ShapeDtypeStruct((groups,), jnp.int32), jax.ShapeDtypeStruct((1,), jnp.int32)


def _gmm_configs(
    *,
    k,
    n,
    tiles,
    transpose_rhs=False,
    groups=128,
    rows_per_group=1024,
    local_groups=None,
    dtype=jnp.bfloat16,
    out_dtype=None,
    rhs_dtype=None,
    rhs_scale=False,
    rhs_bias=False,
    partial_sum=False,
    acc_dtype=None,
    fuse_act=None,
    device_kind="TPU7x",
    vmem_limit_bytes=None,
    tail_chunks=True,
) -> gmm_backend.GmmConfigs:
  """make_gmm_configs for one call, with tail_chunks on and gmm_v2's default VMEM limit unless given."""
  lhs, rhs, group_sizes, group_offset = _operand_shapes(
      k=k,
      n=n,
      groups=groups,
      rows_per_group=rows_per_group,
      local_groups=local_groups,
      transpose_rhs=transpose_rhs,
      dtype=dtype,
  )
  if rhs_dtype is not None:
    rhs = jax.ShapeDtypeStruct(rhs.shape, rhs_dtype)
  scale = jax.ShapeDtypeStruct((rhs.shape[0], 1, 1, n), jnp.float32) if rhs_scale else None
  bias = jax.ShapeDtypeStruct((rhs.shape[0], 1, n), jnp.float32) if rhs_bias else None
  ps = jax.ShapeDtypeStruct((lhs.shape[0], n), dtype) if partial_sum else None
  with _on_chip(device_kind):
    if vmem_limit_bytes is None:
      vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    return gmm_backend.make_gmm_configs(
        lhs,
        rhs,
        scale,
        bias,
        ps,
        group_sizes,
        group_offset,
        tile_info=gmm_backend.TileSizes(*tiles),
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        fuse_act=fuse_act,
        transpose_rhs=transpose_rhs,
        tail_chunks=tail_chunks,
    )


class _AbstractTpuTestCase(parameterized.TestCase):
  """Base of the tail-chunks tests that target an abstract TPU device from the CPU."""

  def setUp(self):
    super().setUp()
    if not hasattr(jax.sharding, "AbstractDevice"):
      self.skipTest("Needs jax.sharding.AbstractDevice to target TPU7x without a TPU.")


class _TailChunksTestCase(_AbstractTpuTestCase):
  """Base of the tests that need tail chunks, and so a JAX whose Pallas pipelines take more than two out buffers."""

  def setUp(self):
    super().setUp()
    if not gmm_backend._DEEP_OUT_BUFFERS:  # pylint: disable=protected-access
      self.skipTest("Tail chunks need Pallas pipelines that take more than two output buffers (not JAX 0.11.x).")


class TailChunksGateTest(_TailChunksTestCase):
  """Which gmm_v2 calls make_gmm_configs gives tail chunks to."""

  @parameterized.named_parameters(
      *_TAIL_CHUNK_CALLS.items(),
      ("gemma4_short_groups", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "rows_per_group": 128}),
      ("gemma4_local_groups", {**_TAIL_CHUNK_CALLS["gemma4_wo_fwd"], "groups": 40, "local_groups": 36}),
      # Long mean groups too (gemma4-26b at 12288 tokens per device): the kernel counts the actual short tiles.
      ("long_tails", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "rows_per_group": 768}),
  )
  def test_on(self, call):
    self.assertTrue(_gmm_configs(**call).tail_chunks)

  @parameterized.named_parameters(
      # Without the flag (MaxText's default).
      ("flag_off", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "tail_chunks": False}),
      # Not checked bit-identical: other chips, other dtypes, and the paths below.
      ("tpu7", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "device_kind": "TPU7"}),
      ("tpu_v6e", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "device_kind": "TPU v6 lite"}),
      ("tpu_v5p", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "device_kind": "TPU v5"}),
      ("fp8_inputs", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "dtype": jnp.float8_e4m3fn}),
      ("f32_inputs", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "dtype": jnp.float32}),
      ("f32_output", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "out_dtype": jnp.float32}),
      ("int8_weights", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "rhs_dtype": jnp.int8, "rhs_scale": True}),
      ("partial_sum", {**_TAIL_CHUNK_CALLS["gemma4_wo_fwd"], "partial_sum": True}),
      ("rhs_bias", {**_TAIL_CHUNK_CALLS["gemma4_wo_fwd"], "rhs_bias": True}),
      ("bf16_accumulator", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "acc_dtype": jnp.bfloat16}),
      ("fused_activation", {"k": 2816, "n": 1536, "tiles": (1024, 2816, 768), "fuse_act": "silu"}),
      ("two_k_steps", {"k": 2816, "n": 704, "tiles": (1024, 1408, 768)}),
      # Outputs differed in the last bit with these tile sizes. At tile_n 256 (one MXU column) they differ when the
      # number of 256-wide K slices is odd.
      ("tile_n_256", {"k": 768, "n": 512, "tiles": (512, 768, 256)}),
      ("transposed_tile_n_256", {"k": 2816, "n": 512, "tiles": (1024, 2816, 256), "transpose_rhs": True}),
      ("tile_n_896", {"k": 4096, "n": 896, "tiles": (1024, 4096, 896)}),
      ("tile_m_2048", {"k": 704, "n": 2816, "tiles": (2048, 768, 1536)}),
      # Too short a tile for two chunks.
      ("tile_m_128", {"k": 2816, "n": 704, "tiles": (128, 2816, 768)}),
      ("transposed_tile_m_256", {"k": 2048, "n": 1024, "tiles": (256, 2048, 1024), "transpose_rhs": True}),
      # The triple-buffered pipeline (49.2 MB here) does not fit the VMEM limit.
      ("vmem_limit", {**_TAIL_CHUNK_CALLS["gemma4_wo_fwd"], "vmem_limit_bytes": 32 * 1024 * 1024}),
      # Its buffers (48.6 MB) fit, but not with the partial out and the headroom for Mosaic's internal scratch.
      ("vmem_headroom", {**_TAIL_CHUNK_CALLS["gemma4_wo_fwd"], "vmem_limit_bytes": 49_000_000}),
  )
  def test_off(self, call):
    self.assertFalse(_gmm_configs(**call).tail_chunks)

  def test_off_without_deep_output_buffers(self):
    """JAX 0.11.x rejects more than two output buffers, so make_gmm_configs leaves tail chunks off there."""
    with mock.patch.object(gmm_backend, "_DEEP_OUT_BUFFERS", False):
      self.assertFalse(_gmm_configs(**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"]).tail_chunks)

  @parameterized.named_parameters(
      ("plain_tile_m_1024", _TAIL_CHUNK_CALLS["gemma4_wi_fwd"], 128, 4),
      ("transposed_tile_m_1024", _TAIL_CHUNK_CALLS["gemma4_wo_dlhs"], 256, 2),
      ("plain_tile_m_512", _TAIL_CHUNK_CALLS["tile_m_512"], 128, 2),
      ("transposed_tile_m_512", _TAIL_CHUNK_CALLS["transposed_tile_m_512"], 256, 1),
      ("plain_tile_m_256", _TAIL_CHUNK_CALLS["tile_m_256"], 128, 1),
  )
  def test_chunk_sizes(self, call, chunk_m, max_chunks):
    cfgs = _gmm_configs(**call)
    self.assertEqual((cfgs.tail_chunk_m, cfgs.max_tail_chunks), (chunk_m, max_chunks))


class TailChunksShortTileCountTest(_TailChunksTestCase):
  """The kernel's count of the tiles that run as chunks (kernel_main runs the chunks if at least 1/16 are short)."""

  @parameterized.named_parameters(
      ("edge_sizes", _EDGE_SIZES, 0, 0, _TAIL_CHUNK_CALLS["gemma4_wo_fwd"]),
      ("local_groups", _EDGE_SIZES, 2, 4, _TAIL_CHUNK_CALLS["gemma4_wo_fwd"]),
      ("tile_m_512", _EDGE_SIZES, 0, 0, _TAIL_CHUNK_CALLS["tile_m_512"]),
      ("transposed", _EDGE_SIZES, 0, 0, _TAIL_CHUNK_CALLS["gemma4_wo_dlhs"]),
      ("transposed_tile_m_512", _EDGE_SIZES, 0, 0, _TAIL_CHUNK_CALLS["transposed_tile_m_512"]),
      ("long_groups", (768,) * 32, 0, 0, _TAIL_CHUNK_CALLS["gemma4_wo_fwd"]),
  )
  def test_counts_short_tiles(self, sizes, group_offset, other_shards_groups, call):
    groups = len(sizes)
    rows_per_group = -(-sum(sizes) // (16 * groups)) * 16
    cfgs = _gmm_configs(**call, groups=groups, rows_per_group=rows_per_group, local_groups=groups - other_shards_groups)
    self.assertTrue(cfgs.tail_chunks)
    short_rows = cfgs.max_tail_chunks * cfgs.tail_chunk_m
    max_num_gm = cfgs.dims.size_group + pl.cdiv(cfgs.dims.size_m, cfgs.tiles.tile_m) - 1

    def kernel(group_sizes_ref, group_offset_ref, out_ref, gm_id_to_group_id, gm_id_to_m_offset):
      metadata = gmm_backend.MetadataRef(gm_id_to_group_id=gm_id_to_group_id, gm_id_to_m_offset=gm_id_to_m_offset)
      out_ref[0], out_ref[1] = gmm_backend._fill_metadata(  # pylint: disable=protected-access
          group_sizes_ref, group_offset_ref, metadata, cfgs=cfgs, short_tile_rows=short_rows
      )

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    counts = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.int32),
        in_specs=[smem, smem],
        out_specs=smem,
        scratch_shapes=[pltpu.SMEM((max_num_gm,), jnp.int32), pltpu.SMEM((max_num_gm + 1,), jnp.int32)],
        interpret=True,
    )(jnp.asarray(sizes, jnp.int32), jnp.asarray([group_offset], jnp.int32))
    expected = _short_tiles(sizes, group_offset, cfgs.dims.size_group, cfgs.tiles.tile_m, short_rows)
    self.assertEqual(tuple(np.asarray(counts)), expected)


class TailChunksLoweringTest(_AbstractTpuTestCase):
  """Calls outside the gate trace and lower to the kernel without tail chunks, also with the flag on."""

  @parameterized.named_parameters(
      # Outside the gate kernel_main builds only the pipeline without tail chunks. With a fused activation, that
      # pipeline pairs the gate and up weights' block specs.
      ("fused_activation", {"k": 2816, "n": 1536, "tiles": (1024, 2816, 768), "fuse_act": "silu"}),
      ("tpu_v6e", {**_TAIL_CHUNK_CALLS["gemma4_wi_fwd"], "device_kind": "TPU v6 lite"}),
  )
  def test_lowers_without_tail_chunks(self, call):
    call = {**call, "groups": 8}
    self.assertFalse(_gmm_configs(**call).tail_chunks)
    lhs, rhs, group_sizes, _ = _operand_shapes(k=call["k"], n=call["n"], groups=8, rows_per_group=1024)
    with _on_chip(call.get("device_kind", "TPU7x")):
      lowered = gmm_backend.gmm_v2.trace(
          lhs,
          rhs,
          group_sizes,
          tile_info=gmm_backend.TileSizes(*call["tiles"]),
          preferred_element_type=jnp.bfloat16,
          fuse_act=call.get("fuse_act"),
          tail_chunks=True,
      ).lower(lowering_platforms=("tpu",))
    self.assertIn("tpu_custom_call", lowered.as_text())


class TailChunksDispatchTest(parameterized.TestCase):
  """MaxText's gmm passes gmm_v2_tail_chunks to the gmm_v2 forward and input-gradient calls; the kernels are mocked."""

  @parameterized.named_parameters(("default", None), ("off", False), ("on", True))
  def test_flag_through_custom_vjp(self, tail_chunks):
    num_groups, batch_size, in_size, out_size = 2, 8, 16, 32
    lhs = jnp.linspace(-1.0, 1.0, batch_size * in_size).reshape(batch_size, in_size)
    rhs = jnp.linspace(-0.5, 0.5, num_groups * in_size * out_size).reshape(num_groups, in_size, out_size)
    group_sizes = jnp.array([batch_size // num_groups] * num_groups, dtype=jnp.int32)

    def mock_gmm(lhs, rhs, transpose_rhs=False, **kwargs):
      del kwargs
      if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
      return jnp.einsum("gmk,gkn->gmn", lhs.reshape(num_groups, -1, lhs.shape[-1]), rhs).reshape(batch_size, -1)

    def mock_tgmm(lhs, rhs, **kwargs):
      del kwargs
      lhs, rhs = lhs.reshape(num_groups, -1, lhs.shape[-1]), rhs.reshape(num_groups, -1, rhs.shape[-1])
      return jnp.einsum("gmk,gmn->gkn", lhs, rhs)

    def loss(lhs, rhs):
      kwargs = {} if tail_chunks is None else {"gmm_v2_tail_chunks": tail_chunks}
      out = megablox_ops.gmm(lhs, rhs, group_sizes, use_tokamax_backend=True, use_gmm_v2=True, **kwargs)
      return jnp.sum(out**2)

    with mock.patch.object(megablox_ops.gmm_v2, "gmm_v2", side_effect=mock_gmm) as gmm_mock:
      with mock.patch.object(megablox_ops.tgmm_v2, "tgmm_v2", side_effect=mock_tgmm):
        jax.jit(jax.grad(loss, argnums=(0, 1)))(lhs, rhs)
    # The forward and the input gradient.
    calls = gmm_mock.call_args_list
    self.assertGreaterEqual(len(calls), 2)
    self.assertEqual({c.kwargs["tail_chunks"] for c in calls}, {bool(tail_chunks)})


# MaxText's gmm tiling (forward, input gradient, weight gradient; each tile_m, tile_k, tile_n) at gemma4-26b's v7x
# recipe tiles.
_TAIL_CHUNK_MAXTEXT_GMMS = {
    "wi": (2816, 704, (1024, 2816, 768, 1024, 768, 2816, 512, 2816, 768)),
    "wo": (704, 2816, (1024, 768, 2816, 1024, 2816, 768, 512, 704, 2816)),
}

_ROUTING_PATTERNS = ("uniform", "short", "skewed", "edge", "padded", "local_groups", "long")
_BF16 = jnp.dtype(jnp.bfloat16)


def _routing(pattern: str) -> tuple[np.ndarray, int, int, int]:
  """Returns (group_sizes, rows, group_offset, local groups) of a routing pattern.

  The mean group sizes keep every call of GmmV2TailChunksTpuTest inside the gate.

  Args:
    pattern: One of _ROUTING_PATTERNS.
  """
  rng = np.random.default_rng(0)
  groups, offset, extra_rows = 128, 0, 0
  if pattern == "uniform":  # ~1024 rows per group: gemma4-26b at 16384 tokens per device, 32 of its experts.
    groups = 32
    sizes = np.bincount(rng.integers(0, groups, groups * 1024), minlength=groups)
  elif pattern == "short":  # ~120 rows per group: gemma4-26b at about 2048 tokens per device.
    sizes = np.bincount(rng.integers(0, groups, groups * 120), minlength=groups)
  elif pattern == "skewed":  # Zipf-like sizes off the tile grid; six groups empty.
    weights = 1.0 / np.arange(1, groups + 1) ** 0.8
    weights[rng.choice(groups, 6, replace=False)] = 0.0
    sizes = np.floor(weights / weights.sum() * groups * 256).astype(np.int64)
    sizes[np.argmax(sizes)] += groups * 256 - sizes.sum()
    sizes = rng.permutation(sizes)
  elif pattern in ("edge", "padded"):  # The boundary sizes, the other groups empty; "padded" adds unrouted rows.
    sizes = rng.permutation(np.pad(np.array(_EDGE_SIZES), (0, groups - len(_EDGE_SIZES))))
    extra_rows = 333 if pattern == "padded" else 0
  elif pattern == "local_groups":  # Expert sharding: groups 0-1 and 38-39 are another shard's.
    groups, offset = 40, 2
    sizes = np.bincount(rng.integers(0, groups, groups * 1024), minlength=groups)
  elif pattern == "long":  # 768 rows per group: no tile of 1024 rows is short, so the kernel runs without the chunks.
    groups = 32
    sizes = np.full(groups, 768)
  else:
    raise ValueError(f"Unknown pattern {pattern!r}.")
  local_groups = groups - 4 if pattern == "local_groups" else groups
  # gmm_v2 zero-fills unrouted rows in whole 16-row sublanes (MaxText pads the rows to a tile multiple).
  rows = -(-(int(sizes.sum()) + extra_rows) // 16) * 16
  return sizes.astype(np.int32), rows, offset, local_groups


def _runs_tail_chunks(sizes, group_offset, local_groups, tile_m, transpose_rhs) -> bool:
  """Whether the kernel runs tail chunks for this routing: at least 1/16 of its tiles fit in half a tile of chunks."""
  chunk_m = 256 if transpose_rhs else 128
  tiles, short = _short_tiles(sizes, group_offset, local_groups, tile_m, tile_m // (2 * chunk_m) * chunk_m)
  return 16 * short >= tiles


def _random_operands(m: int, k: int, n: int, groups: int, transpose_rhs: bool) -> tuple[jax.Array, jax.Array]:
  """bf16 lhs of shape (m, k) and rhs of shape (groups, k, n), or (groups, n, k) with transpose_rhs."""
  key_lhs, key_rhs = jax.random.split(jax.random.key(0))
  lhs = jax.random.normal(key_lhs, (m, k), jnp.float32).astype(jnp.bfloat16)
  shape = (groups, n, k) if transpose_rhs else (groups, k, n)
  rhs = (jax.random.normal(key_rhs, shape, jnp.float32) * k**-0.5).astype(jnp.bfloat16)
  return lhs, rhs


def _run_recording_configs(fn, *args):
  """Runs jit(fn)(*args). Returns the outputs as numpy arrays and the GmmConfigs of each gmm_v2 call."""
  jax.clear_caches()  # gmm_v2 is jitted: retrace it, so that make_gmm_configs runs again.
  configs = []
  make_gmm_configs = gmm_backend.make_gmm_configs

  def recording_make_gmm_configs(*args, **kwargs):
    configs.append(make_gmm_configs(*args, **kwargs))  # pylint: disable=missing-kwoa
    return configs[-1]

  with mock.patch.object(gmm_backend, "make_gmm_configs", side_effect=recording_make_gmm_configs):
    out = jax.block_until_ready(jax.jit(fn)(*args))
  return jax.tree.map(np.asarray, out), configs


# Every call with every pattern for one call per chunk size; the other calls with two patterns.
_GMM_V2_TAIL_CHUNK_CASES = [
    (f"{call}_{pattern}", call, pattern)
    for call in _TAIL_CHUNK_CALLS
    for pattern in (_ROUTING_PATTERNS if call in ("gemma4_wi_fwd", "gemma4_wo_dlhs") else ("uniform", "edge"))
]


@pytest.mark.tpu_only
class GmmV2TailChunksTpuTest(parameterized.TestCase):
  """On TPU7x, tail chunks must not change any output bit.

  Each test runs a call twice: with tail_chunks on, checking that every gmm_v2 call passed the gate and that the
  routing makes the kernel run the chunks (except for the "long" routing, which has no short tile), and with it off
  (the default). It then requires the same bits.
  """

  def setUp(self):
    super().setUp()
    # Tail chunks run only on TPU7x. On another chip both runs would be the kernel without them.
    if jax.devices()[0].device_kind != "TPU7x":
      self.skipTest("Tail chunks run only on TPU7x.")
    if not gmm_backend._DEEP_OUT_BUFFERS:  # pylint: disable=protected-access
      self.skipTest("Tail chunks need Pallas pipelines that take more than two output buffers (not JAX 0.11.x).")

  def _assert_same_bits(self, actual: np.ndarray, expected: np.ndarray, name: str):
    self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype), name)
    reference = expected.astype(np.float32)
    self.assertTrue(np.isfinite(reference).all(), name)
    self.assertGreater(np.abs(reference).max(), 0.0, name)
    bits = np.dtype(f"uint{8 * actual.dtype.itemsize}")
    differ = actual.view(bits) != expected.view(bits)
    if differ.any():
      self.fail(f"{name}: {differ.sum()} of {differ.size} elements differ, the first at {np.argwhere(differ)[0]}.")

  @parameterized.named_parameters(*_GMM_V2_TAIL_CHUNK_CASES)
  def test_gmm_v2_bits_match_without_tail_chunks(self, call, pattern):
    k, n, tiles = (_TAIL_CHUNK_CALLS[call][key] for key in ("k", "n", "tiles"))
    transpose_rhs = _TAIL_CHUNK_CALLS[call].get("transpose_rhs", False)
    sizes, m, offset, local_groups = _routing(pattern)
    lhs, rhs = _random_operands(m, k, n, local_groups, transpose_rhs)

    def gmm(lhs, rhs, group_sizes, group_offset, tail_chunks):
      return gmm_backend.gmm_v2(
          lhs,
          rhs,
          group_sizes,
          group_offset=group_offset,
          tile_info=gmm_backend.TileSizes(*tiles),
          preferred_element_type=_BF16,
          transpose_rhs=transpose_rhs,
          tail_chunks=tail_chunks,
      )

    args = (lhs, rhs, jnp.asarray(sizes), jnp.asarray([offset], jnp.int32))
    on, on_configs = _run_recording_configs(functools.partial(gmm, tail_chunks=True), *args)
    off, off_configs = _run_recording_configs(functools.partial(gmm, tail_chunks=False), *args)
    self.assertEqual([c.tail_chunks for c in on_configs], [True])
    self.assertEqual([c.tail_chunks for c in off_configs], [False])
    self.assertEqual(_runs_tail_chunks(sizes, offset, local_groups, tiles[0], transpose_rhs), pattern != "long")
    self._assert_same_bits(on, off, f"{call} {pattern}")

  @parameterized.product(which=("wi", "wo"), pattern=("uniform", "skewed"))
  def test_maxtext_gmm_fwd_bwd_bits_match_without_tail_chunks(self, which, pattern):
    """MaxText's entry point (megablox gmm on the tokamax gmm_v2 backend), forward and backward."""
    k, n, tiling = _TAIL_CHUNK_MAXTEXT_GMMS[which]
    sizes, m, _, groups = _routing(pattern)
    lhs, rhs = _random_operands(m, k, n, groups, transpose_rhs=False)
    dout = jax.random.normal(jax.random.key(1), (m, n), jnp.float32).astype(jnp.bfloat16)

    def fwd_bwd(lhs, rhs, dout, group_sizes, tail_chunks):
      def gmm(lhs, rhs):
        return megablox_ops.gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=_BF16,
            tiling=tiling,
            use_tokamax_backend=True,
            use_gmm_v2=True,
            use_dlhs_transpose_rhs=True,
            interpret=False,
            gmm_v2_tail_chunks=tail_chunks,
        )

      out, vjp = jax.vjp(gmm, lhs, rhs)
      return (out, *vjp(dout))

    args = (lhs, rhs, dout, jnp.asarray(sizes))
    on, on_configs = _run_recording_configs(functools.partial(fwd_bwd, tail_chunks=True), *args)
    off, off_configs = _run_recording_configs(functools.partial(fwd_bwd, tail_chunks=False), *args)
    # gmm_v2 runs the forward and the input gradient. tgmm_v2 runs the weight gradient; tail chunks leave it alone.
    self.assertEqual([c.tail_chunks for c in on_configs], [True, True])
    self.assertEqual([c.tail_chunks for c in off_configs], [False, False])
    self.assertTrue(all(_runs_tail_chunks(sizes, 0, groups, 1024, transpose) for transpose in (False, True)))
    for name, actual, expected in zip(("out", "dlhs", "drhs"), on, off):
      self._assert_same_bits(actual, expected, f"{which} {pattern} {name}")


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
