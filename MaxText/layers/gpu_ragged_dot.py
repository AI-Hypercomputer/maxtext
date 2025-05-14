# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from functools import partial, lru_cache
from typing import Optional
from collections import namedtuple

import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

__all__ = ["ragged_dot", "ragged_dot_ref", "trans_ragged_dot", "trans_ragged_dot_ref"]

# kernel ###############################################################################################################

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64
DEFAULT_BLOCK_C = 32

_cdiv = lambda a, b: pl.cdiv(a, jnp.array(b, jnp.int32))


def _gpu_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    A_ref,  # [k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    A_scale_ref,  # [n]
    # outputs
    y_ref,  # [k, n]
    # static problem shapes
    m: int,
    k: int,
    n: int,
    g: int,
    # hyperparameters
    block_c: int,
    block_k: int,
    block_n: int,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
  pid = namedtuple("pid", ["i", "j"])(pl.program_id(0), pl.program_id(1))
  size = namedtuple("size", ["m", "k", "n"])(m, k, n)  # pack into named tuple to not lose indices later
  group_sz = group_sizes_ref[pid.i]
  compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

  dim_nums = (((1,), (0,)), ((), ()))
  _dot_fn = partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type=acc_dtype)

  @pl.when(group_sz > 0)
  def _():
    # row index into lhs and output
    start_ridx = jnp.where(pid.i == 0, 0, group_offset_ref[jnp.maximum(pid.i - 1, 0)])

    def outer_compute(r_offset, _):
      ridx = start_ridx + r_offset * block_c  # r_offset is 0,1,2,... need to map it to actual row indices
      lhs_rows_mask = (r_offset * block_c + jnp.arange(block_c)) < group_sz
      lhs_rows_idx = pl.ds(ridx, block_c)
      rhs_cols_idx = pl.ds(0, block_n)
      rhs_cols_mask = (block_n * pid.j + jnp.arange(block_n)) < size.n

      def inner_compute(k, acc):
        inner_idx = pl.ds(k * block_k, block_k)
        inner_mask = (k * block_k + jnp.arange(block_k)) < size.k
        x = pl.load(x_ref, (lhs_rows_idx, inner_idx), mask=lhs_rows_mask[:, None] & inner_mask[None, :], other=0)
        A = pl.load(A_ref, (inner_idx, rhs_cols_idx), mask=inner_mask[:, None] & rhs_cols_mask[None, :], other=0)
        return acc + _dot_fn(x.astype(compute_dtype), A.astype(compute_dtype)).astype(acc.dtype)

      acc = jnp.zeros((block_c, block_n), dtype=acc_dtype)
      acc = jax.lax.fori_loop(0, _cdiv(size.k, block_k), inner_compute, acc)
      if A_scale_ref is not None:
        acc = acc * pl.load(A_scale_ref, rhs_cols_idx, mask=rhs_cols_mask, other=0).astype(acc.dtype)
      acc = acc.astype(y_ref.dtype)
      pl.store(y_ref, (lhs_rows_idx, rhs_cols_idx), acc, mask=lhs_rows_mask[:, None] & rhs_cols_mask[None, :])
      return None

    jax.lax.fori_loop(0, _cdiv(group_sz, block_c), outer_compute, None)

  @pl.when(pid.i == g - 1)
  def _():
    assert group_offset_ref.size == g
    last_offset = group_offset_ref[g - 1]
    col_mask = (block_n * pid.j + jnp.arange(block_n)) < size.n

    def set_zero(i, _):
      row_mask = (last_offset + i * block_c + jnp.arange(block_c)) < size.m
      idx = (pl.ds(last_offset + i * block_c, block_c), pl.ds(0, block_n))
      mask = row_mask[:, None] & col_mask[None, :]
      pl.store(y_ref, idx, jnp.zeros((block_c, block_n), dtype=y_ref.dtype), mask=mask)

    jax.lax.fori_loop(0, _cdiv(size.m - last_offset, block_c), set_zero, None)


# main routine #########################################################################################################


@partial(jax.jit, static_argnums=list(range(4, 13)))
def _gpu_ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    block_c: int = DEFAULT_BLOCK_C,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  """Compute grouped matmul on GPU via a Pallas lowering."""
  del block_m  # not used in ragged_dot (only trans_ragged_dot)
  assert A.ndim == 3 and x.ndim == 2
  assert A.shape[:1] == group_sizes.shape
  if A_scale is not None:
    assert A_scale.shape == (A.shape[0], A.shape[-1])  # one scale per A column
  size = namedtuple("size", ["m", "k", "n", "g"])(x.shape[0], x.shape[-1], A.shape[-1], A.shape[0])

  # normalize the block sizes for GPU
  block_n, block_k, block_c = [
      pl.next_power_of_2(min(b, s)) for b, s in zip([block_n, block_k, block_c], [size.n, size.k, size.m])
  ]
  block_k, block_n = max(block_k, 16), max(block_n, 16)

  group_offsets = jnp.cumsum(group_sizes, -1)  # we'll read 1 down always
  in_specs = [
      pl.BlockSpec((size.m, size.k), lambda i, j: (0, 0)),
      pl.BlockSpec((None, size.k, block_n), lambda i, j: (i, 0, j)),
      pl.BlockSpec((group_sizes.size,), lambda i, j: (0,)),
      pl.BlockSpec((group_offsets.size,), lambda i, j: (0,)),
      pl.BlockSpec((None, block_n), lambda i, j: (i, j)) if A_scale is not None else None,
  ]

  out_shape = jax.ShapeDtypeStruct((size.m, size.n), dtype=x.dtype)
  out_specs = pl.BlockSpec((size.m, block_n), lambda i, j: (0, j))
  grid = (size.g, pl.cdiv(size.n, block_n))
  block_sizes = dict(block_c=block_c, block_k=block_k, block_n=block_n)
  dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
  y = pl.pallas_call(
      partial(_gpu_ragged_dot_kernel, **size._asdict(), **block_sizes, **dtype_spec),
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      interpret=interpret,
      compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps, num_stages=num_stages),
  )(x, A, group_sizes, group_offsets, A_scale)
  return y


# reference implementation #############################################################################################


@partial(jax.jit, static_argnums=list(range(4, 13)))
def _gpu_ragged_dot_ref(
    x: jax.Array,
    A: jax.Array,
    group_sizes: jax.Array,
    A_scale: jax.Array | None = None,
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    block_c: int = DEFAULT_BLOCK_C,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  del block_m, block_n, block_k, block_c, interpret, compute_dtype, acc_dtype, num_warps, num_stages
  ret = jax.lax.ragged_dot(x, A, group_sizes)
  if A_scale is not None:
    indices = jnp.repeat(jnp.arange(A.shape[0]), group_sizes, total_repeat_length=x.shape[0])
    A_scale = jnp.take_along_axis(A_scale, indices[:, None], 0)
    ret = ret * A_scale
  return ret


# tests ################################################################################################################


def _gpu_trans_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    y_ref,  # [k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    # outputs
    A_bar_ref,  # [g, k, n]
    # static problem shapes
    m: int,
    k: int,
    n: int,
    g: int,
    # hyperparameters
    block_m: int,
    block_n: int,
    block_k: int,
    block_c: int,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
  assert A_bar_ref.shape == (block_m, block_n)
  del g
  pid = namedtuple("pid", ["i", "r", "c"])(pl.program_id(0), pl.program_id(1), pl.program_id(2))
  size = namedtuple("size", ["m", "k", "n"])(m, k, n)  # pack into named tuple to not lose indices later
  group_sz = group_sizes_ref[pid.i]
  compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

  dim_nums = (((0,), (0,)), ((), ()))
  _dot_fn = partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type=acc_dtype)

  @pl.when(group_sz > 0)
  def _():
    # row index into lhs and output
    start_ridx = jnp.where(pid.i == 0, 0, group_offset_ref[jnp.maximum(pid.i - 1, 0)])

    def outer_compute(k, _):
      k_idx = pl.ds(k * block_k, block_k)
      k_mask = (pid.r * block_m + k * block_k + jnp.arange(block_k)) < size.k
      cols_idx = pl.ds(0, block_n)
      cols_mask = (block_n * pid.c + jnp.arange(block_n)) < size.n

      def inner_compute(r_offset, acc):
        ridx = start_ridx + r_offset * block_c  # r_offset is 0,1,2,... need to map it to actual row indices
        xy_rows_mask = (r_offset * block_c + jnp.arange(block_c)) < group_sz
        xy_rows_idx = pl.ds(ridx, block_c)

        x = pl.load(x_ref, (xy_rows_idx, k_idx), mask=xy_rows_mask[:, None] & k_mask[None, :], other=0)
        y = pl.load(y_ref, (xy_rows_idx, cols_idx), mask=xy_rows_mask[:, None] & cols_mask[None, :], other=0)
        return acc + _dot_fn(x.astype(compute_dtype), y.astype(compute_dtype)).astype(acc.dtype)

      acc = jnp.zeros((block_k, block_n), dtype=acc_dtype)
      acc = jax.lax.fori_loop(0, _cdiv(group_sz, block_c), inner_compute, acc)
      acc = acc.astype(y_ref.dtype)
      pl.store(A_bar_ref, (k_idx, cols_idx), acc, mask=k_mask[:, None] & cols_mask[None, :])
      return None

    jax.lax.fori_loop(0, _cdiv(block_m, block_k), outer_compute, None)

  @pl.when(group_sz == 0)
  def _():
    rmask = (pid.r * block_m + jnp.arange(block_m)) < size.k
    cmask = (pid.c * block_n + jnp.arange(block_n)) < size.n
    pl.store(A_bar_ref, (pl.ds(None),) * 2, jnp.zeros_like(A_bar_ref), mask=rmask[:, None] & cmask[None, :])


# main routine #########################################################################################################


@partial(jax.jit, static_argnums=list(range(3, 12)))
def _gpu_trans_ragged_dot(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    block_c: int = DEFAULT_BLOCK_C,  # compute window for rows in x, y
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  """Compute grouped matmul on GPU via a Pallas lowering."""
  assert y.ndim == 2 and x.ndim == 2 and x.shape[0] == y.shape[0]
  size = namedtuple("size", ["m", "k", "n", "g"])(x.shape[0], x.shape[-1], y.shape[-1], group_sizes.size)

  block_m, block_n = min(block_m, x.shape[-1]), min(block_n, y.shape[-1])

  # normalize the block sizes for GPU
  block_m, block_n, block_k, block_c = [
      pl.next_power_of_2(min(b, s)) for b, s in zip([block_m, block_n, block_k, block_c], [size.m, size.n, size.k, size.m])
  ]
  block_k, block_n, block_c = min(block_m, block_k), max(block_n, 16), max(block_c, 16)

  group_offsets = jnp.cumsum(group_sizes, -1)  # we'll read 1 down always
  in_specs = [
      pl.BlockSpec((size.m, block_m), lambda i, r, c: (0, r)),
      pl.BlockSpec((size.m, block_n), lambda i, r, c: (0, c)),
      pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
      pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
  ]

  out_shape = jax.ShapeDtypeStruct((size.g, size.k, size.n), dtype=x.dtype)
  out_specs = pl.BlockSpec((None, block_m, block_n), lambda i, r, c: (i, r, c))
  grid = (size.g, pl.cdiv(size.k, block_m), pl.cdiv(size.n, block_n))

  block_sizes = dict(block_m=block_m, block_n=block_n, block_k=block_k, block_c=block_c)
  dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
  y = pl.pallas_call(
      partial(_gpu_trans_ragged_dot_kernel, **size._asdict(), **block_sizes, **dtype_spec),
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      interpret=interpret,
      compiler_params=plgpu.TritonCompilerParams(num_warps=num_warps, num_stages=num_stages),
  )(x, y, group_sizes, group_offsets)
  return y


# reference implementation #############################################################################################


@partial(jax.jit, static_argnums=list(range(3, 12)))
def _gpu_trans_ragged_dot_ref(
    x: jax.Array,
    y: jax.Array,
    group_sizes: jax.Array,
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    block_c: int = DEFAULT_BLOCK_C,  # compute window for rows in x, y
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  del block_m, block_n, block_k, block_c, interpret, compute_dtype, acc_dtype, num_warps, num_stages

  def scan_fn(i_offset, _):
    i, offset = i_offset
    accumulate = lambda j, acc: acc + x[j, :][:, None] @ y[j, :][None, :]
    zero = jnp.zeros((x.shape[-1], y.shape[-1]), dtype=x.dtype)
    Ai = jax.lax.fori_loop(offset, offset + group_sizes[i], accumulate, zero)
    return (i + 1, offset + group_sizes[i]), Ai

  return jax.lax.scan(scan_fn, (0, 0), None, length=group_sizes.shape[0])[1]


# autodiff rules #######################################################################################################


@lru_cache
def _get_ragged_dot(ref: bool = False, **kw):
  @jax.custom_vjp
  def ragged_dot(x, A, group_sizes, A_scale=None):
    return (_gpu_ragged_dot_ref if ref else _gpu_ragged_dot)(x, A, group_sizes, A_scale, **kw)

  def ragged_dot_fwd(x, A, group_sizes, A_scale):
    m = x.shape[0]
    k = x.shape[1]
    n = A.shape[2]
    g = A.shape[0]
    print(f"forward pass: m={m}, k={k}, n={n}, g={g}")
    return ragged_dot(x, A, group_sizes, A_scale), (x, A, group_sizes, A_scale)

  def ragged_dot_bwd(res, g):
    (x, A, group_sizes, A_scale), dy = res, g
    assert A_scale is None, "Differentiating ragged_dot with A_scale is not supported"
    dx = ragged_dot(dy, A.swapaxes(-1, -2), group_sizes, None)
    dA = trans_ragged_dot(x, dy, group_sizes)
    m = x.shape[0]
    k = x.shape[1]
    n = A.shape[2]
    g = A.shape[0]
    print(f"backward pass: m={m}, k={k}, n={n}, g={g}")
    return dx, dA, None, None

  ragged_dot.defvjp(ragged_dot_fwd, ragged_dot_bwd)

  # --------------------------------------------------------------------- #

  @jax.custom_vjp
  def trans_ragged_dot(x, A, group_sizes):
    return (_gpu_trans_ragged_dot_ref if ref else _gpu_trans_ragged_dot)(x, A, group_sizes, **kw)

  def trans_ragged_dot_fwd(x, y, group_sizes):
    m = x.shape[0]
    k = x.shape[1]
    n = y.shape[2]
    g = y.shape[0]
    print(f"trans forward pass: m={m}, k={k}, n={n}, g={g}")
    return trans_ragged_dot(x, y, group_sizes), (x, y, group_sizes)

  def trans_ragged_dot_bwd(res, g):
    (x, y, group_sizes), dA = res, g
    dy = ragged_dot(x, dA, group_sizes)
    dx = ragged_dot(y, dA.swapaxes(-1, -2), group_sizes)
    m = x.shape[0]
    k = x.shape[1]
    n = y.shape[2]
    g = y.shape[0]
    print(f"trans backward pass: m={m}, k={k}, n={n}, g={g}")
    return dx, dy, None

  trans_ragged_dot.defvjp(trans_ragged_dot_fwd, trans_ragged_dot_bwd)
  return ragged_dot, trans_ragged_dot


# exported methods #####################################################################################################


@partial(jax.jit, static_argnums=list(range(4, 13)))
def ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    block_c: int = DEFAULT_BLOCK_C,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  """Ragged dot corresponding to jax.lax.ragged_dot (m, k) x (g, k, n) -> (m, n)"""
  kw = dict(block_m=block_m, block_n=block_n, block_k=block_k, block_c=block_c, interpret=interpret)
  kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
  _ragged_dot = _get_ragged_dot(ref=False, **kw)[0]
  return _ragged_dot(x, A, group_sizes, A_scale)


@partial(jax.jit, static_argnums=list(range(3, 12)))
def trans_ragged_dot(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    block_c: int = DEFAULT_BLOCK_C,  # compute window for rows in x, y
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  """Tranposed ragged dot corresponding to transpose of ragged dot wrt A argument (m, k) x (m, n) -> (g, k, n)"""
  kw = dict(block_m=block_m, block_n=block_n, block_k=block_k, block_c=block_c, interpret=interpret)
  kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
  _trans_ragged_dot = _get_ragged_dot(ref=False, **kw)[1]
  return _trans_ragged_dot(x, y, group_sizes)


@partial(jax.jit, static_argnums=list(range(4, 13)))
def ragged_dot_ref(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    A_scale: jax.Array | None = None,  # [g, n] or None
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    block_c: int = DEFAULT_BLOCK_C,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  kw = dict(block_m=block_m, block_n=block_n, block_k=block_k, block_c=block_c, interpret=interpret)
  kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
  _ragged_dot = _get_ragged_dot(ref=True, **kw)[0]
  return _ragged_dot(x, A, group_sizes, A_scale)


@partial(jax.jit, static_argnums=list(range(3, 12)))
def trans_ragged_dot_ref(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    block_c: int = DEFAULT_BLOCK_C,  # compute window for rows in x, y
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
  kw = dict(block_m=block_m, block_n=block_n, block_k=block_k, block_c=block_c, interpret=interpret)
  kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
  _trans_ragged_dot = _get_ragged_dot(ref=True, **kw)[1]
  return _trans_ragged_dot(x, y, group_sizes)


# a simple tuninig example #############################################################################################

if __name__ == "__main__":
  from tune_jax import tune, tune_logger

  tune_logger.setLevel("DEBUG")

  keys = iter(random.split(random.key(time.time_ns() % 2**31), 1024))
  m, k, n = (1024, 7168, 256)
  g = 128
  dtype = jnp.bfloat16
  x = random.normal(next(keys), (m, k), dtype=dtype)
  A = random.normal(next(keys), (g, k, n), dtype=dtype)

  group_sizes = jnp.round((m - 2) * jax.nn.softmax(1e0 * random.normal(next(keys), g), -1)).astype(jnp.int32)
  assert jnp.sum(group_sizes) <= m

  hyperparams = dict(
      block_m=[32, 64, 128, 256],
      block_n=[16, 64, 128, 256],
      block_k=[16, 64, 128, 256],
      block_c=[4, 8, 16, 32],
  )

  def combined_fn(x, A, group_sizes, **kw):
    y = ragged_dot(x, A, group_sizes, **kw)
    dx, dA = jax.grad(lambda x_, A_, gs_: jnp.sum(ragged_dot(x_, A_, gs_, **kw)), argnums=(0, 1))(x, A, group_sizes)
    return jnp.sum(dA) + jnp.sum(dx) + jnp.sum(y)

  _ = jax.jit(combined_fn)(x, A, group_sizes)
  fn = jax.jit(tune(combined_fn, hyperparams=hyperparams, example_args=(x, A, group_sizes)))
  o = fn(x, A, group_sizes)
  o.block_until_ready()

  # forward pass
  def forward_fn(x, A, group_sizes, **kw):
    y = ragged_dot(x, A, group_sizes, **kw)
    return jnp.sum(y)

  _ = jax.jit(forward_fn)(x, A, group_sizes)
  fn = jax.jit(tune(forward_fn, hyperparams=hyperparams, example_args=(x, A, group_sizes)))
  o = fn(x, A, group_sizes)
  o.block_until_ready()

  # backward pass
  def backward_fn(x, A, group_sizes, **kw):
    dx, dA = jax.grad(lambda x_, A_, gs_: jnp.sum(ragged_dot(x_, A_, gs_, **kw)), argnums=(0, 1))(x, A, group_sizes)
    return jnp.sum(dA) + jnp.sum(dx)

  _ = jax.jit(backward_fn)(x, A, group_sizes)
  fn = jax.jit(tune(backward_fn, hyperparams=hyperparams, example_args=(x, A, group_sizes)))
  o = fn(x, A, group_sizes)
  o.block_until_ready()
