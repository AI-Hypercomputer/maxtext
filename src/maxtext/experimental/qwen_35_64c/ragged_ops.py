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

"""Pallas ragged SwiGLU + expert weighting + dynamic FP8 quantization kernels.

Adapted from `maxtext/experimental/lineage/ops.py` (`ragged_silu_mul_fwd`,
`ragged_silu_mul_bwd`, `ragged_quantize`, `ragged_absmax_quantize`) without
`jaxtyping`/`typeguard` dependencies and with pure-JAX CPU fallbacks.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_VMEM_LIMIT_BYTES = int(0.9 * 64 * 1024 * 1024)


def absmax_to_scale(amax: jax.Array, qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn) -> jax.Array:
  """Converts `[1, 1]` or scalar `absmax` to `[1, 1]` float32 FP8 scale."""
  fmax = float(jnp.finfo(jnp.dtype(qtype)).max)
  amax_f32 = amax.astype(jnp.float32).reshape(1, 1)
  return jnp.where(amax_f32 > 0.0, amax_f32 / fmax, jnp.ones_like(amax_f32))


# ---------------------------------------------------------------------------
# Ragged SwiGLU + Per-Row Expert Routing Weight (`ragged_silu_mul_fwd` / `bwd`)
# ---------------------------------------------------------------------------


def _silu_mul_tile(x: jax.Array, y: jax.Array, c: jax.Array, out_dtype: jax.typing.DTypeLike) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  y_f32 = y.astype(jnp.float32)
  c_f32 = c.astype(jnp.float32)
  return (jax.nn.silu(x_f32) * y_f32 * c_f32).astype(out_dtype)


def _quantize_tile(x: jax.Array, inv_scale: jax.Array, qtype: jax.typing.DTypeLike) -> jax.Array:
  q_max = float(jnp.finfo(jnp.dtype(qtype)).max)
  return jnp.clip(x.astype(jnp.float32) * inv_scale, -q_max, q_max).astype(qtype)


def _ragged_silu_mul_kernel(
    x_ref: Any,
    y_ref: Any,
    c_ref: Any,
    out_ref: Any,
    *,
    out_scale_ref: Any = None,
) -> None:
  act = _silu_mul_tile(x_ref[...], y_ref[...], c_ref[...], x_ref.dtype)
  if out_scale_ref is not None:
    inv_scale = 1.0 / out_scale_ref[0].astype(jnp.float32)
    out_ref[...] = _quantize_tile(act, inv_scale, out_ref.dtype)
  else:
    out_ref[...] = act.astype(out_ref.dtype)


def _ragged_silu_mul_x_specs(
    x_offset: jax.Array,
    x_dtype: jnp.dtype,
    block_size_tokens: int,
    block_size_hidden: int,
    num_hidden_blocks: int,
) -> tuple[pl.BlockSpec, pl.BlockSpec]:
  sublanes = 8 * 4 // jnp.dtype(x_dtype).itemsize
  shape = (pl.Element(block_size_tokens), block_size_hidden)
  row = lambda i: pl.multiple_of(x_offset + i * block_size_tokens, sublanes)
  return (
      pl.BlockSpec(shape, lambda i, j: (row(i), j)),
      pl.BlockSpec(shape, lambda i, j: (row(i), j + num_hidden_blocks)),
  )


def _ragged_silu_mul_fwd_pipeline(
    x_offset_ref: Any,
    num_tokens_ref: Any,
    x_ref: Any,
    c_ref: Any,
    out_scale_ref: Any,
    out_ref: Any,
    *,
    block_size_tokens: int,
    block_size_hidden: int,
    num_hidden_blocks: int,
) -> None:
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    g0_spec, g1_spec = _ragged_silu_mul_x_specs(
        x_offset_ref[0],
        x_ref.dtype,
        block_size_tokens,
        block_size_hidden,
        num_hidden_blocks,
    )
    pltpu.emit_pipeline(
        functools.partial(_ragged_silu_mul_kernel, out_scale_ref=out_scale_ref),
        grid=(pl.cdiv(num_tokens, block_size_tokens), num_hidden_blocks),
        in_specs=[
            g0_spec,
            g1_spec,
            pl.BlockSpec((block_size_tokens, 1), lambda i, j: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_size_tokens, block_size_hidden), lambda i, j: (i, j)),
    )(x_ref, x_ref, c_ref, out_ref)


def ragged_silu_mul_fwd(
    x: jax.Array,
    x_offset: jax.Array | int,
    coeffs: jax.Array,
    num_tokens: jax.Array | int,
    *,
    block_size_tokens: int = 1024,
    block_size_hidden: int = 2048,
    out_dtype: jax.typing.DTypeLike | None = None,
    out_scale: jax.Array | None = None,
    use_pallas: bool = True,
) -> jax.Array:
  """Computes `silu(gate) * up * coeffs` over the first `num_tokens` rows."""
  max_num_tokens = coeffs.shape[0]
  hidden_dim = x.shape[1] // 2
  target_dtype = jnp.dtype(out_dtype) if out_dtype is not None else x.dtype

  if not use_pallas:
    x_slice = jax.lax.dynamic_slice_in_dim(x, jnp.asarray(x_offset, jnp.int32), max_num_tokens, axis=0)
    g0, g1 = x_slice[:, :hidden_dim], x_slice[:, hidden_dim:]
    valid = (jnp.arange(max_num_tokens, dtype=jnp.int32) < jnp.asarray(num_tokens, jnp.int32))[:, None]
    act = _silu_mul_tile(g0, g1, coeffs, x.dtype)
    act = jnp.where(valid, act, jnp.zeros_like(act))
    if out_scale is not None:
      inv_s = 1.0 / out_scale.astype(jnp.float32).reshape(1, 1)
      q = _quantize_tile(act, inv_s, target_dtype)
      return jnp.where(valid, q, jnp.zeros(act.shape, dtype=target_dtype))
    return act.astype(target_dtype)

  block_size_tokens = math.gcd(block_size_tokens, max_num_tokens)
  block_size_hidden = math.gcd(min(block_size_hidden, hidden_dim), hidden_dim)
  num_hidden_blocks = hidden_dim // block_size_hidden
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  out_scale_smem = out_scale.astype(jnp.float32).reshape(1) if out_scale is not None else None
  out_scale_spec = smem if out_scale_smem is not None else None
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  kname = "ragged_silu_mul_fwd-fp8" if (out_dtype is not None or out_scale is not None) else "ragged_silu_mul_fwd"
  return pl.pallas_call(
      functools.partial(
          _ragged_silu_mul_fwd_pipeline,
          block_size_tokens=block_size_tokens,
          block_size_hidden=block_size_hidden,
          num_hidden_blocks=num_hidden_blocks,
      ),
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, hidden_dim),
          target_dtype,
          manual_axis_type=manual_axis_type,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm, hbm, out_scale_spec],
          out_specs=hbm,
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
      name=kname,
  )(
      jnp.asarray(x_offset, jnp.int32)[None],
      jnp.asarray(num_tokens, jnp.int32)[None],
      x,
      coeffs,
      out_scale_smem,
  )


def _ragged_silu_mul_bwd_kernel(
    g_ref: Any,
    x_ref: Any,
    y_ref: Any,
    c_ref: Any,
    dx_ref: Any,
    dy_ref: Any,
    dc_ref: Any,
    *,
    num_tokens_ref: Any = None,
    amax_vmem_ref: Any = None,
) -> None:
  x_f32 = x_ref[...].astype(jnp.float32)
  y_f32 = y_ref[...].astype(jnp.float32)
  g_f32 = g_ref[...].astype(jnp.float32)
  c_f32 = c_ref[...].astype(jnp.float32)

  sig_x = jax.nn.sigmoid(x_f32)
  silu_x = x_f32 * sig_x
  dsilu_x = sig_x * (1.0 + x_f32 * (1.0 - sig_x))

  gc_val = g_f32 * c_f32
  dx = (gc_val * y_f32 * dsilu_x).astype(dx_ref.dtype)
  dy = (gc_val * silu_x).astype(dy_ref.dtype)
  dx_ref[...] = dx
  dy_ref[...] = dy

  if amax_vmem_ref is not None and num_tokens_ref is not None:
    row_idx = pl.program_id(0) * dx.shape[0] + jax.lax.broadcasted_iota(jnp.int32, (dx.shape[0], 1), 0)
    valid = row_idx < num_tokens_ref[0]
    tile_abs = jnp.where(
        valid,
        jnp.maximum(jnp.abs(dx.astype(jnp.float32)), jnp.abs(dy.astype(jnp.float32))),
        0.0,
    )
    tile_max = jnp.max(tile_abs, axis=0).astype(jnp.float32)
    amax_vmem_ref[0] = jnp.maximum(amax_vmem_ref[0], tile_max)

  @pl.when(pl.program_id(1) == 0)
  def _init():
    dc_ref[...] = jnp.zeros_like(dc_ref)

  dc_ref[...] += jnp.sum(
      g_f32 * silu_x * y_f32,
      axis=-1,
      keepdims=True,
  ).astype(dc_ref.dtype)


def _ragged_silu_mul_bwd_pipeline(
    x_offset_ref: Any,
    num_tokens_ref: Any,
    g_ref: Any,
    x_ref: Any,
    c_ref: Any,
    dxy_ref: Any,
    dc_ref: Any,
    amax_smem_ref: Any = None,
    amax_vmem_ref: Any = None,
    *,
    block_size_tokens: int,
    block_size_hidden: int,
    num_hidden_blocks: int,
) -> None:
  num_tokens = num_tokens_ref[0]
  if amax_vmem_ref is not None:
    amax_vmem_ref[...] = jnp.zeros_like(amax_vmem_ref)

  @pl.when(num_tokens > 0)
  def _():
    g0_spec, g1_spec = _ragged_silu_mul_x_specs(
        x_offset_ref[0],
        x_ref.dtype,
        block_size_tokens,
        block_size_hidden,
        num_hidden_blocks,
    )
    hidden_shape = (block_size_tokens, block_size_hidden)
    hidden_spec = pl.BlockSpec(hidden_shape, lambda i, j: (i, j))
    dy_spec = pl.BlockSpec(hidden_shape, lambda i, j: (i, j + num_hidden_blocks))
    coeffs_spec = pl.BlockSpec((block_size_tokens, 1), lambda i, j: (i, 0))
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_silu_mul_bwd_kernel,
            num_tokens_ref=num_tokens_ref,
            amax_vmem_ref=amax_vmem_ref,
        ),
        grid=(pl.cdiv(num_tokens, block_size_tokens), num_hidden_blocks),
        in_specs=[hidden_spec, g0_spec, g1_spec, coeffs_spec],
        out_specs=[hidden_spec, dy_spec, coeffs_spec],
    )(g_ref, x_ref, x_ref, c_ref, dxy_ref, dxy_ref, dc_ref)

  if amax_smem_ref is not None and amax_vmem_ref is not None:
    amax_smem_ref[0, 0] = jnp.max(amax_vmem_ref[0])


def ragged_silu_mul_bwd(
    g: jax.Array,
    x: jax.Array,
    x_offset: jax.Array | int,
    coeffs: jax.Array,
    num_tokens: jax.Array | int,
    *,
    block_size_tokens: int = 512,
    block_size_hidden: int = 1024,
    with_absmax: bool = False,
    use_pallas: bool = True,
) -> Any:
  """Backward pass of `ragged_silu_mul_fwd`, optionally returning `dx_absmax`."""
  max_num_tokens, hidden_dim = g.shape

  if not use_pallas:
    x_slice = jax.lax.dynamic_slice_in_dim(x, jnp.asarray(x_offset, jnp.int32), max_num_tokens, axis=0)
    x_f32 = x_slice[:, :hidden_dim].astype(jnp.float32)
    y_f32 = x_slice[:, hidden_dim:].astype(jnp.float32)
    g_f32 = g.astype(jnp.float32)
    c_f32 = coeffs.astype(jnp.float32)
    valid = (jnp.arange(max_num_tokens, dtype=jnp.int32) < jnp.asarray(num_tokens, jnp.int32))[:, None]
    sig_x = jax.nn.sigmoid(x_f32)
    silu_x = x_f32 * sig_x
    dsilu_x = sig_x * (1.0 + x_f32 * (1.0 - sig_x))
    gc_val = g_f32 * c_f32
    dx = jnp.where(valid, (gc_val * y_f32 * dsilu_x).astype(x.dtype), 0)
    dy = jnp.where(valid, (gc_val * silu_x).astype(x.dtype), 0)
    dxy = jnp.concatenate([dx, dy], axis=-1)
    dc = jnp.where(
        valid,
        jnp.sum(g_f32 * silu_x * y_f32, axis=-1, keepdims=True),
        0.0,
    ).astype(coeffs.dtype)
    if with_absmax:
      amax = jnp.max(jnp.abs(dxy.astype(jnp.float32))).reshape(1, 1)
      return dxy, dc, amax
    return dxy, dc

  if with_absmax and block_size_tokens * block_size_hidden > 512 * 1024:
    block_size_tokens = min(block_size_tokens, 512)
  block_size_tokens = math.gcd(block_size_tokens, max_num_tokens)
  block_size_hidden = math.gcd(min(block_size_hidden, hidden_dim), hidden_dim)
  num_hidden_blocks = hidden_dim // block_size_hidden
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  dxy_struct = jax.ShapeDtypeStruct(
      (max_num_tokens, 2 * hidden_dim),
      x.dtype,
      manual_axis_type=manual_axis_type,
  )
  dc_struct = jax.ShapeDtypeStruct(
      coeffs.shape,
      jnp.float32,
      manual_axis_type=manual_axis_type,
  )
  if with_absmax:
    amax_struct = jax.ShapeDtypeStruct(
        (1, 1),
        jnp.float32,
        manual_axis_type=manual_axis_type,
    )
    out_shape = (dxy_struct, dc_struct, amax_struct)
    out_specs = [hbm, hbm, smem]
    scratch_shapes = [pltpu.VMEM((1, block_size_hidden), jnp.float32)]
    kname = "ragged_silu_mul_bwd-absmax"
  else:
    out_shape = (dxy_struct, dc_struct)
    out_specs = [hbm, hbm]
    scratch_shapes = []
    kname = "ragged_silu_mul_bwd"

  res = pl.pallas_call(
      functools.partial(
          _ragged_silu_mul_bwd_pipeline,
          block_size_tokens=block_size_tokens,
          block_size_hidden=block_size_hidden,
          num_hidden_blocks=num_hidden_blocks,
      ),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm, hbm, hbm],
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
      name=kname,
  )(
      jnp.asarray(x_offset, jnp.int32)[None],
      jnp.asarray(num_tokens, jnp.int32)[None],
      g,
      x,
      coeffs,
  )
  if with_absmax:
    dxy, dc, amax = res
    return dxy, dc.astype(coeffs.dtype), amax
  dxy, dc = res
  return dxy, dc.astype(coeffs.dtype)


# ---------------------------------------------------------------------------
# Ragged Dynamic FP8 Quantization (`ragged_quantize` & `ragged_absmax_quantize`)
# ---------------------------------------------------------------------------


def _ragged_absmax_kernel(
    x_ref: Any,
    *,
    num_tokens_ref: Any,
    amax_ref: Any,
) -> None:
  x_val = x_ref[...]
  abs_x = jnp.abs(x_val.astype(jnp.float32))
  block_tokens = x_val.shape[0]
  rem = num_tokens_ref[0] - pl.program_id(0) * block_tokens
  abs_x = jnp.where(
      (rem >= block_tokens) | (jax.lax.broadcasted_iota(jnp.int32, x_val.shape, 0) < rem),
      abs_x,
      0.0,
  )
  if x_val.ndim == 2:
    block_max = jnp.max(abs_x.reshape(-1, 8, x_val.shape[1]), axis=0)
  else:
    block_max = jnp.max(abs_x, axis=0)
  amax_ref[...] = jnp.maximum(amax_ref[...], block_max)


def _ragged_quantize_kernel(
    x_ref: Any,
    out_ref: Any,
    *,
    num_tokens_ref: Any,
    scale_ref: Any,
    fmax: float,
) -> None:
  x_val = x_ref[...]
  inv_scale = 1.0 / scale_ref[...].astype(jnp.float32)
  if x_val.ndim == 3:
    inv_scale = inv_scale[None]
  q = jnp.clip(x_val.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(out_ref.dtype)
  block_tokens = x_val.shape[0]
  rem = num_tokens_ref[0] - pl.program_id(0) * block_tokens
  q = jnp.where(
      (rem >= block_tokens) | (jax.lax.broadcasted_iota(jnp.int32, x_val.shape, 0) < rem),
      q,
      0,
  )
  out_ref[...] = q


def _ragged_quantize_pipeline(
    num_tokens_ref: Any,
    x_ref: Any,
    scale_ref: Any,
    out_ref: Any,
    *,
    block_size_tokens: int,
    fmax: float,
) -> None:
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    x_spec = pl.BlockSpec(
        (block_size_tokens, *x_ref.shape[1:]),
        lambda i: (i,) + (0,) * (x_ref.ndim - 1),
    )
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_quantize_kernel,
            num_tokens_ref=num_tokens_ref,
            scale_ref=scale_ref,
            fmax=fmax,
        ),
        grid=(pl.cdiv(num_tokens, block_size_tokens),),
        in_specs=[x_spec],
        out_specs=x_spec,
    )(x_ref, out_ref)


def _ragged_absmax_quantize_pipeline(
    num_tokens_ref: Any,
    x_ref: Any,
    out_ref: Any,
    scale_out_ref: Any,
    amax_ref: Any,
    *,
    block_size_tokens: int,
    fmax: float,
) -> None:
  scale_out_ref[...] = jnp.ones((1, 1), jnp.float32)
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    amax_ref[...] = jnp.zeros_like(amax_ref)
    num_blocks = pl.cdiv(num_tokens, block_size_tokens)
    x_spec = pl.BlockSpec(
        (block_size_tokens, *x_ref.shape[1:]),
        lambda i: (i,) + (0,) * (x_ref.ndim - 1),
    )
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_absmax_kernel,
            num_tokens_ref=num_tokens_ref,
            amax_ref=amax_ref,
        ),
        grid=(num_blocks,),
        in_specs=[x_spec],
    )(x_ref)
    amax = jnp.max(amax_ref[...], keepdims=True)
    scale_out_ref[...] = jnp.where(amax > 0, amax / fmax, 1.0)
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_quantize_kernel,
            num_tokens_ref=num_tokens_ref,
            scale_ref=scale_out_ref,
            fmax=fmax,
        ),
        grid=(num_blocks,),
        in_specs=[x_spec],
        out_specs=x_spec,
    )(x_ref, out_ref)


def ragged_quantize(
    x: jax.Array,
    scale: jax.Array,
    qtype: jax.typing.DTypeLike,
    num_tokens: jax.Array | int,
    *,
    block_size_tokens: int | None = None,
    use_pallas: bool = True,
) -> jax.Array:
  """Quantizes the first `num_tokens` rows of 2D or 3D `x` to `qtype` using `[1, 1]` `scale`."""
  qtype = jnp.dtype(qtype)
  fmax = float(jnp.finfo(qtype).max)
  scale_2d = scale.astype(jnp.float32).reshape(1, 1)

  if not use_pallas:
    valid = (jnp.arange(x.shape[0], dtype=jnp.int32) < jnp.asarray(num_tokens, jnp.int32)).reshape(
        -1, *([1] * (x.ndim - 1))
    )
    inv_s = (1.0 / scale_2d).reshape(1, *([1] * (x.ndim - 1)))
    q = jnp.clip(x.astype(jnp.float32) * inv_s, -fmax, fmax).astype(qtype)
    return jnp.where(valid, q, jnp.zeros_like(q))

  if block_size_tokens is None:
    default_block = 512 if x.ndim == 2 else 256
    block_size_tokens = math.gcd(default_block, x.shape[0])
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(
          _ragged_quantize_pipeline,
          block_size_tokens=block_size_tokens,
          fmax=fmax,
      ),
      out_shape=jax.ShapeDtypeStruct(x.shape, qtype, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[hbm, pl.BlockSpec((1, 1), lambda *_: (0, 0))],
          out_specs=hbm,
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
      name="ragged_quantize",
  )(
      jnp.asarray(num_tokens, jnp.int32)[None],
      x,
      scale_2d,
  )


def ragged_absmax_quantize(
    x: jax.Array,
    qtype: jax.typing.DTypeLike,
    num_tokens: jax.Array | int,
    *,
    block_size_tokens: int | None = None,
    use_pallas: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Dynamically computes `[1, 1]` scale over `[0, num_tokens)` and quantizes `x` to `qtype`."""
  qtype = jnp.dtype(qtype)
  fmax = float(jnp.finfo(qtype).max)

  if not use_pallas:
    valid = (jnp.arange(x.shape[0], dtype=jnp.int32) < jnp.asarray(num_tokens, jnp.int32)).reshape(
        -1, *([1] * (x.ndim - 1))
    )
    abs_x = jnp.where(valid, jnp.abs(x.astype(jnp.float32)), 0.0)
    amax = jnp.max(abs_x).reshape(1, 1)
    scale_2d = jnp.where(amax > 0.0, amax / fmax, jnp.ones_like(amax))
    q = ragged_quantize(x, scale_2d, qtype, num_tokens, use_pallas=False)
    return q, scale_2d

  if block_size_tokens is None:
    default_block = 512 if x.ndim == 2 else 256
    block_size_tokens = math.gcd(default_block, x.shape[0])
  amax_shape = (8, x.shape[1]) if x.ndim == 2 else x.shape[1:]
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(
          _ragged_absmax_quantize_pipeline,
          block_size_tokens=block_size_tokens,
          fmax=fmax,
      ),
      out_shape=(
          jax.ShapeDtypeStruct(x.shape, qtype, manual_axis_type=manual_axis_type),
          jax.ShapeDtypeStruct((1, 1), jnp.float32, manual_axis_type=manual_axis_type),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[hbm],
          out_specs=[hbm, pl.BlockSpec((1, 1), lambda *_: (0, 0))],
          scratch_shapes=[pltpu.VMEM(amax_shape, jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
      name="ragged_absmax_quantize",
  )(
      jnp.asarray(num_tokens, jnp.int32)[None],
      x,
  )
