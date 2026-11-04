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

"""Pallas TensorCore VMEM kernels for RMSNorm, Gated RMSNorm, and MoE dW scatter."""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_SCATTER_DW_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def _manual_axis_type(x: jax.Array) -> Any:
  return getattr(jax.typeof(x), "manual_axis_type", None)


def _pure_jax_rms_norm_zc(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
  orig_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
  normed = x_f32 * inv_rms * (1.0 + scale.astype(jnp.float32))
  return normed.astype(orig_dtype)


def _pure_jax_rms_norm_gated(x: jax.Array, gate_z: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
  orig_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
  normed = x_f32 * inv_rms * scale.astype(jnp.float32)
  gated = normed * jax.nn.silu(gate_z.astype(jnp.float32))
  return gated.astype(orig_dtype)


def _choose_norm_block_m(m: int, d: int) -> int:
  target_m = 2048 if d <= 128 else (1024 if d <= 256 else 128)
  return math.gcd(m, target_m)


def rms_norm_zero_centered_pallas(
    x: jax.Array,
    scale: jax.Array,
    eps: float = 1e-6,
    *,
    interpret: bool = False,
) -> jax.Array:
  """VMEM-fused Qwen3Next zero-centered RMSNorm `(x * rsqrt(mean(x^2)+eps)) * (1+scale)`."""
  if x.ndim < 1 or scale.ndim != 1:
    return _pure_jax_rms_norm_zc(x, scale, eps)
  d = x.shape[-1]
  if d < 128 or d % 128 != 0 or scale.shape != (d,):
    return _pure_jax_rms_norm_zc(x, scale, eps)
  m = x.size // d
  if m < 8 or m % 8 != 0:
    return _pure_jax_rms_norm_zc(x, scale, eps)
  block_m = _choose_norm_block_m(m, d)
  if block_m < 8 or block_m % 8 != 0:
    return _pure_jax_rms_norm_zc(x, scale, eps)

  orig_shape = x.shape
  x_2d = x.reshape(m, d)

  @functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
  def _op(x_in: jax.Array, s_in: jax.Array, eps_val: float, interp: bool) -> jax.Array:
    out, _ = _op_fwd(x_in, s_in, eps_val, interp)
    return out

  def _op_fwd(x_in: jax.Array, s_in: jax.Array, eps_val: float, interp: bool):
    mat = _manual_axis_type(x_in)
    s_2d = jnp.broadcast_to(s_in.astype(jnp.float32)[None, :], (8, d))
    num_blocks = m // block_m
    sub_blocks = block_m // 8

    def _fwd_kernel(x_ref, s_ref, o_ref):
      x_f32 = x_ref[...].astype(jnp.float32)
      inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps_val)
      x_hat = x_f32 * inv_rms
      w = jnp.tile(1.0 + s_ref[...], (sub_blocks, 1))
      y = x_hat * w
      o_ref[...] = y.astype(o_ref.dtype)

    y_2d = pl.pallas_call(
        _fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((m, d), x_in.dtype, manual_axis_type=mat),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((8, d), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((block_m, d), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
        interpret=interp,
        name="rmsnorm_zc_fwd",
    )(x_in, s_2d)
    return y_2d, (x_in, s_in)

  def _op_bwd(eps_val: float, interp: bool, res, dy_in: jax.Array):
    x_in, s_in = res
    mat = _manual_axis_type(x_in)
    s_2d = jnp.broadcast_to(s_in.astype(jnp.float32)[None, :], (8, d))
    num_blocks = m // block_m
    sub_blocks = block_m // 8

    def _bwd_kernel(x_ref, dy_ref, s_ref, dx_ref, ds_out_ref):
      x_f32 = x_ref[...].astype(jnp.float32)
      dy_f32 = dy_ref[...].astype(jnp.float32)
      inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps_val)
      x_hat = x_f32 * inv_rms
      ds_blk = jnp.sum((dy_f32 * x_hat).reshape(sub_blocks, 8, d), axis=0)
      ds_out_ref[...] = ds_blk

      w = jnp.tile(1.0 + s_ref[...], (sub_blocks, 1))
      dy_w = dy_f32 * w
      proj = jnp.mean(dy_w * x_hat, axis=-1, keepdims=True)
      dx = inv_rms * (dy_w - x_hat * proj)
      dx_ref[...] = dx.astype(dx_ref.dtype)

    dx_2d, ds_partials = pl.pallas_call(
        _bwd_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, d), x_in.dtype, manual_axis_type=mat),
            jax.ShapeDtypeStruct((num_blocks * 8, d), jnp.float32, manual_axis_type=mat),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((8, d), lambda i: (0, 0)),
            ],
            out_specs=(
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((8, d), lambda i: (i, 0)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
        interpret=interp,
        name="rmsnorm_zc_bwd",
    )(x_in, dy_in, s_2d)
    dscale = jnp.sum(ds_partials, axis=0).astype(s_in.dtype)
    return dx_2d, dscale

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(x_2d, scale, float(eps), bool(interpret)).reshape(orig_shape)


def rms_norm_gated_pallas(
    x: jax.Array,
    gate_z: jax.Array,
    scale: jax.Array,
    eps: float = 1e-6,
    *,
    interpret: bool = False,
) -> jax.Array:
  """VMEM-fused Qwen3Next gated RMSNorm `(x * rsqrt(mean(x^2)+eps) * scale) * silu(gate_z)`."""
  if x.ndim < 1 or x.shape != gate_z.shape or scale.ndim != 1:
    return _pure_jax_rms_norm_gated(x, gate_z, scale, eps)
  d = x.shape[-1]
  if d < 128 or d % 128 != 0 or scale.shape != (d,):
    return _pure_jax_rms_norm_gated(x, gate_z, scale, eps)
  m = x.size // d
  if m < 8 or m % 8 != 0:
    return _pure_jax_rms_norm_gated(x, gate_z, scale, eps)
  block_m = _choose_norm_block_m(m, d)
  if block_m < 8 or block_m % 8 != 0:
    return _pure_jax_rms_norm_gated(x, gate_z, scale, eps)

  orig_shape = x.shape
  x_2d = x.reshape(m, d)
  z_2d = gate_z.reshape(m, d)

  @functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
  def _op(
      x_in: jax.Array,
      z_in: jax.Array,
      s_in: jax.Array,
      eps_val: float,
      interp: bool,
  ) -> jax.Array:
    out, _ = _op_fwd(x_in, z_in, s_in, eps_val, interp)
    return out

  def _op_fwd(
      x_in: jax.Array,
      z_in: jax.Array,
      s_in: jax.Array,
      eps_val: float,
      interp: bool,
  ):
    mat = _manual_axis_type(x_in)
    s_2d = jnp.broadcast_to(s_in.astype(jnp.float32)[None, :], (8, d))
    num_blocks = m // block_m
    sub_blocks = block_m // 8

    def _fwd_kernel(x_ref, z_ref, s_ref, o_ref):
      x_f32 = x_ref[...].astype(jnp.float32)
      z_f32 = z_ref[...].astype(jnp.float32)
      inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps_val)
      x_hat = x_f32 * inv_rms
      normed = x_hat * jnp.tile(s_ref[...], (sub_blocks, 1))
      gated = normed * jax.nn.silu(z_f32)
      o_ref[...] = gated.astype(o_ref.dtype)

    y_2d = pl.pallas_call(
        _fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((m, d), x_in.dtype, manual_axis_type=mat),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((8, d), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((block_m, d), lambda i: (i, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
        interpret=interp,
        name="rmsnorm_gated_fwd",
    )(x_in, z_in, s_2d)
    return y_2d, (x_in, z_in, s_in)

  def _op_bwd(eps_val: float, interp: bool, res, dy_in: jax.Array):
    x_in, z_in, s_in = res
    mat = _manual_axis_type(x_in)
    s_2d = jnp.broadcast_to(s_in.astype(jnp.float32)[None, :], (8, d))
    num_blocks = m // block_m
    sub_blocks = block_m // 8

    def _bwd_kernel(x_ref, z_ref, dy_ref, s_ref, dx_ref, dz_ref, ds_out_ref):
      x_f32 = x_ref[...].astype(jnp.float32)
      z_f32 = z_ref[...].astype(jnp.float32)
      dy_f32 = dy_ref[...].astype(jnp.float32)
      s_rep = jnp.tile(s_ref[...], (sub_blocks, 1))

      inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps_val)
      x_hat = x_f32 * inv_rms
      sig_z = jax.nn.sigmoid(z_f32)
      silu_z = z_f32 * sig_z
      normed = x_hat * s_rep

      d_silu_z = dy_f32 * normed
      dz = d_silu_z * (sig_z + silu_z * (1.0 - sig_z))
      dz_ref[...] = dz.astype(dz_ref.dtype)

      d_normed = dy_f32 * silu_z
      ds_blk = jnp.sum((d_normed * x_hat).reshape(sub_blocks, 8, d), axis=0)
      ds_out_ref[...] = ds_blk

      dx_hat = d_normed * s_rep
      proj = jnp.mean(dx_hat * x_hat, axis=-1, keepdims=True)
      dx = inv_rms * (dx_hat - x_hat * proj)
      dx_ref[...] = dx.astype(dx_ref.dtype)

    dx_2d, dz_2d, ds_partials = pl.pallas_call(
        _bwd_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, d), x_in.dtype, manual_axis_type=mat),
            jax.ShapeDtypeStruct((m, d), z_in.dtype, manual_axis_type=mat),
            jax.ShapeDtypeStruct((num_blocks * 8, d), jnp.float32, manual_axis_type=mat),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((8, d), lambda i: (0, 0)),
            ],
            out_specs=(
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((block_m, d), lambda i: (i, 0)),
                pl.BlockSpec((8, d), lambda i: (i, 0)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
        interpret=interp,
        name="rmsnorm_gated_bwd",
    )(x_in, z_in, dy_in, s_2d)
    dscale = jnp.sum(ds_partials, axis=0).astype(s_in.dtype)
    return dx_2d, dz_2d, dscale

  _op.defvjp(_op_fwd, _op_bwd)
  return _op(x_2d, z_2d, scale, float(eps), bool(interpret)).reshape(orig_shape)


def _scatter_dw_fallback(
    slot_win: jax.Array,
    rows: jax.Array,
    start: jax.Array | int,
    cnt: jax.Array | int,
    *,
    num_slots: int,
) -> jax.Array:
  """Pure-JAX fallback for MoE routing-weight gradient scatter."""
  capacity = slot_win.shape[0]
  iota = jnp.arange(capacity, dtype=jnp.int32)
  idx = jnp.where(
      jnp.asarray(start, jnp.int32) + iota < jnp.asarray(cnt, jnp.int32),
      slot_win,
      num_slots + iota,
  )
  return jnp.zeros((num_slots,), jnp.float32).at[idx].set(rows.astype(jnp.float32), mode="drop", unique_indices=True)


def scatter_dw_vmem_tc(
    slot_win: jax.Array,
    rows: jax.Array,
    start: jax.Array | int,
    cnt: jax.Array | int,
    *,
    num_slots: int,
    interpret: bool = False,
) -> jax.Array:
  """Scatters `rows[:valid]` into `f32[num_slots]` entirely inside 4-bank VMEM.

  Replaces XLA's `scatter_custom_fusion` (`104,448` scalar HBM read-modify-write
  scatters per MoE call = `40.48 ms/step`) with a 4-way unrolled VMEM 1-hot LUT
  FMA (`banks[u, slot >> 10] += val * lut[slot & 1023]`) over only the
  `num_valid = clamp(cnt - start, 0, capacity)` active slots, followed by a
  single contiguous `2.62 MiB` VMEM->HBM DMA write.

  Args:
    slot_win: `int32[capacity]` sorted slot indices for this window.
    rows: `float32[capacity]` routing-weight gradients for this window.
    start: Window offset into global sorted slots.
    cnt: Total active local-expert slots (`<= num_slots`).
    num_slots: Output length (`T_ep * top_k`).
    interpret: Whether to run `pallas_call` in CPU interpret mode.

  Returns:
    `float32[num_slots]` routing-weight gradient contribution for this window.
  """
  capacity = slot_win.shape[0]
  if num_slots < 1024 or num_slots % 1024 != 0 or capacity < 1024 or capacity % 1024 != 0 or rows.shape != (capacity,):
    return _scatter_dw_fallback(slot_win, rows, start, cnt, num_slots=num_slots)
  num_tiles = num_slots // 1024
  # 4 banks * num_tiles * 4 KiB + 1025 * 4 KiB LUT must fit well under 32 MiB.
  if (4 * num_tiles + 1025) * 4096 > 28 * 1024 * 1024:
    return _scatter_dw_fallback(slot_win, rows, start, cnt, num_slots=num_slots)

  meta = jnp.stack(
      [
          jnp.asarray(start, jnp.int32).reshape(()),
          jnp.asarray(cnt, jnp.int32).reshape(()),
      ]
  )
  rows_f32 = rows.astype(jnp.float32)
  slot_i32 = slot_win.astype(jnp.int32)
  lut = (
      (jax.lax.broadcasted_iota(jnp.int32, (1025, 1024), 0) == jax.lax.broadcasted_iota(jnp.int32, (1025, 1024), 1))
      .astype(jnp.float32)
      .reshape(1025, 8, 128)
  )

  def _scatter_dw_kernel(
      meta_smem_ref,
      lut_hbm_ref,
      slot_hbm_ref,
      rows_hbm_ref,
      out_hbm_ref,
      banks_vmem_ref,
      lut_vmem_ref,
      slot_smem_ref,
      rows_smem_ref,
      lut_sem,
      slot_sem,
      rows_sem,
      out_sem,
  ):
    lut_copy = pltpu.make_async_copy(lut_hbm_ref, lut_vmem_ref, lut_sem)
    lut_copy.start()

    start_s = meta_smem_ref[0]
    cnt_s = meta_smem_ref[1]
    num_valid = jnp.minimum(jnp.maximum(cnt_s - start_s, 0), capacity)
    num_chunks = pl.cdiv(num_valid, 1024)

    def _slot_copy(chunk_idx, buf_idx):
      off = pl.multiple_of(chunk_idx * 1024, 1024)
      return pltpu.make_async_copy(
          slot_hbm_ref.at[pl.ds(off, 1024)],
          slot_smem_ref.at[buf_idx],
          slot_sem.at[buf_idx],
      )

    def _rows_copy(chunk_idx, buf_idx):
      off = pl.multiple_of(chunk_idx * 1024, 1024)
      return pltpu.make_async_copy(
          rows_hbm_ref.at[pl.ds(off, 1024)],
          rows_smem_ref.at[buf_idx],
          rows_sem.at[buf_idx],
      )

    @pl.when(num_chunks > 0)
    def _prefetch_first():
      _slot_copy(0, 0).start()
      _rows_copy(0, 0).start()

    z_tile = jnp.zeros((8, 128), jnp.float32)

    @pl.loop(0, num_tiles)
    def _zero_banks(t):
      for u in range(4):
        banks_vmem_ref[u, t, :, :] = z_tile

    lut_copy.wait()

    @pl.loop(0, num_chunks)
    def _process_chunk(c):
      b = c % 2
      nb = 1 - b
      _slot_copy(c, b).wait()
      _rows_copy(c, b).wait()

      @pl.when(c + 1 < num_chunks)
      def _prefetch_next():
        _slot_copy(c + 1, nb).start()
        _rows_copy(c + 1, nb).start()

      chunk_base = c * 1024

      @pl.loop(0, 256)
      def _step4(i):
        off4 = i * 4
        for u in range(4):
          pos = chunk_base + off4 + u
          raw_slot = slot_smem_ref[b, off4 + u]
          in_range = (pos < num_valid) & (raw_slot >= 0) & (raw_slot < num_slots)
          tile_idx = jnp.where(in_range, raw_slot >> 10, 0)
          lut_idx = jnp.where(in_range, raw_slot & 1023, 1024)
          val = jnp.where(in_range, rows_smem_ref[b, off4 + u], 0.0)
          banks_vmem_ref[u, tile_idx, :, :] = banks_vmem_ref[u, tile_idx, :, :] + val * lut_vmem_ref[lut_idx, :, :]

    @pl.loop(0, num_tiles)
    def _reduce_banks(t):
      banks_vmem_ref[0, t, :, :] = (banks_vmem_ref[0, t, :, :] + banks_vmem_ref[1, t, :, :]) + (
          banks_vmem_ref[2, t, :, :] + banks_vmem_ref[3, t, :, :]
      )

    out_copy = pltpu.make_async_copy(banks_vmem_ref.at[0], out_hbm_ref, out_sem)
    out_copy.start()
    out_copy.wait()

  mat = _manual_axis_type(rows_f32)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  out_3d = pl.pallas_call(
      _scatter_dw_kernel,
      out_shape=jax.ShapeDtypeStruct((num_tiles, 8, 128), jnp.float32, manual_axis_type=mat),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          hbm_spec,
          hbm_spec,
          hbm_spec,
      ],
      out_specs=hbm_spec,
      scratch_shapes=[
          pltpu.VMEM((4, num_tiles, 8, 128), jnp.float32),
          pltpu.VMEM((1025, 8, 128), jnp.float32),
          pltpu.SMEM((2, 1024), jnp.int32),
          pltpu.SMEM((2, 1024), jnp.float32),
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA((2,)),
          pltpu.SemaphoreType.DMA((2,)),
          pltpu.SemaphoreType.DMA,
      ],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_SCATTER_DW_VMEM_LIMIT_BYTES,
      ),
      interpret=interpret,
      name="scatter_dw_vmem_tc",
  )(meta, lut, slot_i32, rows_f32)
  return out_3d.reshape(num_slots)
