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

"""Fused spill-free Pallas VMEM kernel for Muse Spark `post_expert_norm`.

Executes `MuseSparkInputScaledRMSNorm` (plus optional per-row routing-weight
scaling) in 1 forward HBM pass and 1 backward HBM pass over ONLY the valid rows
`[0, count)` of a ragged MoE expert buffer (`[cap, D]` or `[cap, D // 128, 128]`),
matching `muse_spark.py` `_input_scaled_rms_norm_save_rstd` on TPU:

  s_f32    = ((1.0 in scale.dtype) + scale).astype(float32)
  sf       = (x.astype(float32) * s_f32).astype(x.dtype).astype(float32)
  var_eps  = mean(square(sf)) + epsilon
  rstd     = rsqrt(var_eps)
  norm     = (sf * rstd).astype(x.dtype)
  out      = (norm.astype(float32) * sorted_weights).astype(x.dtype)  # or norm

In 3D `{2, 1, 0: T(8, 128)}` VMEM layout (`[M, D // 128, 128]`), each `(1, 8, 128)`
tile maps to one native `(8, 128)` vector register. Double-buffered HBM <-> VMEM
DMA transfers `block_m` rows (`256` rows = `2 MiB`) per grid step while an inner
`(8, 8, 128)` (`8` vector registers per chunk) loop accumulates `sum(sf^2)` and
`p = sum(g * sf)` in a single pass with `0` VMEM register spills.
"""

import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_NUM_SUBLANES = 8
_NUM_LANES = 128


def _manual_axis_type(x: jax.Array):
  return getattr(jax.typeof(x), "manual_axis_type", None)


def _centered_scale_fp32(scale: jax.Array) -> jax.Array:
  """Computes `(1.0 + scale).float()` in `scale.dtype` first, matching PyTorch."""
  return jnp.asarray(jnp.asarray(1.0, dtype=scale.dtype) + scale, jnp.float32)


def _jax_reference_fwd(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    sorted_weights: jax.Array | None,
    epsilon: float,
) -> jax.Array:
  """Pure-JAX reference matching `muse_spark.py` over valid rows `[0, count)`."""
  cap = expert_out.shape[0]
  valid_1d = jnp.arange(cap, dtype=jnp.int32) < jnp.asarray(count, jnp.int32)
  valid = valid_1d.reshape(-1, *([1] * (expert_out.ndim - 1)))
  x_safe = jnp.where(valid, expert_out, jnp.zeros((), expert_out.dtype))

  s_f32 = _centered_scale_fp32(scale).reshape(1, *expert_out.shape[1:])
  scaled_f32 = (jnp.asarray(x_safe, jnp.float32) * s_f32).astype(expert_out.dtype).astype(jnp.float32)
  reduce_axes = tuple(range(1, expert_out.ndim))
  dim = float(expert_out.shape[1] * (expert_out.shape[2] if expert_out.ndim == 3 else 1))
  var_eps = (
      lax.div(jnp.sum(lax.square(scaled_f32), axis=reduce_axes, keepdims=True), jnp.asarray(dim, jnp.float32)) + epsilon
  )
  rstd = lax.rsqrt(var_eps)
  norm_out = (scaled_f32 * rstd).astype(expert_out.dtype)
  if sorted_weights is not None:
    w_safe = jnp.where(valid_1d, sorted_weights.astype(jnp.float32), 0.0)
    w_bcast = w_safe.reshape(-1, *([1] * (expert_out.ndim - 1)))
    out = (norm_out.astype(jnp.float32) * w_bcast).astype(expert_out.dtype)
  else:
    out = norm_out
  return jnp.where(valid, out, jnp.zeros((), expert_out.dtype))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _jax_reference_with_vjp(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    epsilon: float,
    sorted_weights: jax.Array | None,
) -> jax.Array:
  return _jax_reference_fwd(expert_out, scale, count, sorted_weights, epsilon)


def _jax_reference_vjp_fwd(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    epsilon: float,
    sorted_weights: jax.Array | None,
):
  out = _jax_reference_fwd(expert_out, scale, count, sorted_weights, epsilon)
  return out, (expert_out, scale, count, sorted_weights)


def _jax_reference_vjp_bwd(epsilon: float, res, g: jax.Array):
  """Backward pass for pure-JAX post-expert norm reference."""
  expert_out, scale, count, sorted_weights = res
  cap = expert_out.shape[0]
  valid_1d = jnp.arange(cap, dtype=jnp.int32) < jnp.asarray(count, jnp.int32)
  valid = valid_1d.reshape(-1, *([1] * (expert_out.ndim - 1)))
  g_safe = jnp.where(valid, g, jnp.zeros((), g.dtype))

  def _loss_fn(x_in, s_in, w_in):
    return _jax_reference_fwd(x_in, s_in, count, w_in, epsilon)

  _, vjp_fn = jax.vjp(_loss_fn, expert_out, scale, sorted_weights)
  dx, dscale, dw = vjp_fn(g_safe)
  return dx, dscale, None, dw


_jax_reference_with_vjp.defvjp(_jax_reference_vjp_fwd, _jax_reference_vjp_bwd)


def _unpack_weights_to_rows(w_sublane: jax.Array, block_m: int) -> jax.Array:
  """Transposes `(8, block_m)` lane-packed weights into `(block_m, 128)` row-replicated VMEM."""
  chunks = []
  for k in range(block_m // _NUM_LANES):
    sub = w_sublane[:, k * _NUM_LANES : (k + 1) * _NUM_LANES]
    chunks.append(jnp.tile(sub, (_NUM_LANES // _NUM_SUBLANES, 1)).T)
  return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)


def _pack_rows_to_weights(row_rep_128: jax.Array, block_m: int) -> jax.Array:
  """Transposes `(block_m, 128)` row-replicated VMEM sums into `(8, block_m)` lane-packed output."""
  chunks = []
  for k in range(block_m // _NUM_LANES):
    sub = row_rep_128[k * _NUM_LANES : (k + 1) * _NUM_LANES, :]
    chunks.append(sub.T[:_NUM_SUBLANES, :])
  return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=1)


def _col_chunk_ranges_3d(num_col_tiles: int) -> tuple[tuple[int, int], ...]:
  """Splits `[0, num_col_tiles)` into 8-sublane-aligned chunks (`<= 8` tiles each) for 0 VMEM spills."""
  return tuple((c0, min(_NUM_SUBLANES, num_col_tiles - c0)) for c0 in range(0, num_col_tiles, _NUM_SUBLANES))


def _sum_2d_col_tiles(val_2d: jax.Array, col_chunk_tiles: int) -> jax.Array:
  """Sums `(8, col_chunk_tiles * 128)` into `(8, 128)` using direct lane slices."""
  acc = val_2d[:, :_NUM_LANES]
  for t in range(1, col_chunk_tiles):
    acc = acc + val_2d[:, t * _NUM_LANES : (t + 1) * _NUM_LANES]
  return acc


def _round_to_dtype_f32(val_f32: jax.Array, target_dtype) -> jax.Array:
  """Rounds a float32 intermediate through `target_dtype`, matching PyTorch `type_as(x)`."""
  if jnp.dtype(target_dtype) == jnp.float32:
    return val_f32
  return val_f32.astype(target_dtype).astype(jnp.float32)


def _x_index_map_3d(i, count_ref):
  del count_ref
  return (i, 0, 0)


def _s_index_map_3d(i, count_ref):
  del i, count_ref
  return (0, 0, 0)


def _x_index_map_2d(i, count_ref):
  del count_ref
  return (i, 0)


def _s_index_map_2d(i, count_ref):
  del i, count_ref
  return (0, 0)


def _w_index_map(i, count_ref):
  del count_ref
  return (0, i)


def _pallas_fwd(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    sorted_weights: jax.Array | None,
    *,
    epsilon: float,
    block_m: int,
) -> jax.Array:
  """1-pass spill-free Pallas VMEM forward over valid rows `[0, count)`."""
  orig_cap = expert_out.shape[0]
  is_3d = expert_out.ndim == 3
  dim = expert_out.shape[1] * (expert_out.shape[2] if is_3d else 1)
  if dim % _NUM_LANES != 0:
    raise ValueError(f"fused_ragged_post_expert_norm requires D % 128 == 0, got D={dim}.")
  if is_3d and expert_out.shape[2] != _NUM_LANES:
    raise ValueError(f"3D expert_out must have trailing dim 128, got {expert_out.shape}.")

  num_col_tiles = dim // _NUM_LANES
  eff_block_m = max(_NUM_LANES, (block_m // _NUM_LANES) * _NUM_LANES)
  aligned_cap = max(eff_block_m, pl.cdiv(max(orig_cap, 1), eff_block_m) * eff_block_m)
  pad_m = aligned_cap - orig_cap
  x_in = jnp.pad(expert_out, ((0, pad_m),) + ((0, 0),) * (expert_out.ndim - 1)) if pad_m > 0 else expert_out

  s_f32 = _centered_scale_fp32(scale)
  has_weights = sorted_weights is not None
  w_in = None
  if sorted_weights is not None:
    w_f32 = sorted_weights.astype(jnp.float32)
    if pad_m > 0:
      w_f32 = jnp.pad(w_f32, (0, pad_m))
    w_in = jnp.broadcast_to(w_f32[None, :], (_NUM_SUBLANES, aligned_cap))

  count_1d = jnp.reshape(jnp.clip(jnp.asarray(count, jnp.int32), 0, orig_cap), (1,))
  num_steps = pl.cdiv(count_1d[0], eff_block_m)
  inv_dim = 1.0 / float(dim)

  if is_3d:
    col_chunks = _col_chunk_ranges_3d(num_col_tiles)
    s_in = s_f32.reshape(1, num_col_tiles, _NUM_LANES)
    x_block_shape = (eff_block_m, num_col_tiles, _NUM_LANES)
    s_block_shape = (1, num_col_tiles, _NUM_LANES)

    def _fwd_kernel_3d(count_ref, *refs):
      if has_weights:
        x_ref, s_ref, w_ref, o_ref, w_scratch_ref = refs
        w_scratch_ref[...] = _unpack_weights_to_rows(w_ref[...], eff_block_m)
      else:
        x_ref, s_ref, o_ref = refs
        w_scratch_ref = None

      pid = pl.program_id(0)
      n_valid = count_ref[0]
      inv_dim_f32 = jnp.asarray(inv_dim, jnp.float32)
      sub_iota = lax.broadcasted_iota(jnp.int32, (_NUM_SUBLANES, _NUM_LANES), 0)

      @pl.loop(0, eff_block_m // _NUM_SUBLANES)
      def _sub_step(sub_idx):
        r0 = pl.multiple_of(sub_idx * _NUM_SUBLANES, _NUM_SUBLANES)
        valid_8 = (pid * eff_block_m + r0 + sub_iota) < n_valid
        valid_3d = jnp.expand_dims(valid_8, axis=1)

        sq_128 = jnp.zeros((_NUM_SUBLANES, _NUM_LANES), jnp.float32)
        for c0, chunk_t in col_chunks:
          if len(col_chunks) == 1:
            x_c = jnp.where(valid_3d, x_ref[pl.ds(r0, _NUM_SUBLANES), :, :], jnp.zeros((), x_ref.dtype))
            s_c = s_ref[:, :, :]
          else:
            x_c = jnp.where(valid_3d, x_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :], jnp.zeros((), x_ref.dtype))
            s_c = s_ref[:, pl.ds(c0, chunk_t), :]
          sf_c = _round_to_dtype_f32(x_c.astype(jnp.float32) * s_c, x_ref.dtype)
          sq_128 = sq_128 + jnp.sum(lax.square(sf_c), axis=1)

        row_sq = jnp.broadcast_to(jnp.sum(sq_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
        rstd_8 = lax.rsqrt(row_sq * inv_dim_f32 + epsilon)
        rstd_3d = jnp.expand_dims(rstd_8, axis=1)
        if w_scratch_ref is not None:
          w_8 = jnp.where(valid_8, w_scratch_ref[pl.ds(r0, _NUM_SUBLANES), :], 0.0)
          w_3d = jnp.expand_dims(w_8, axis=1)
        else:
          w_3d = None

        for c0, chunk_t in col_chunks:
          if len(col_chunks) == 1:
            x_c = jnp.where(valid_3d, x_ref[pl.ds(r0, _NUM_SUBLANES), :, :], jnp.zeros((), x_ref.dtype))
            s_c = s_ref[:, :, :]
          else:
            x_c = jnp.where(valid_3d, x_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :], jnp.zeros((), x_ref.dtype))
            s_c = s_ref[:, pl.ds(c0, chunk_t), :]
          sf_c = _round_to_dtype_f32(x_c.astype(jnp.float32) * s_c, x_ref.dtype)
          norm_c = (sf_c * rstd_3d).astype(x_ref.dtype)
          raw_out_c = (norm_c.astype(jnp.float32) * w_3d).astype(x_ref.dtype) if w_3d is not None else norm_c
          out_c = jnp.where(valid_3d, raw_out_c, jnp.zeros((), x_ref.dtype))
          if len(col_chunks) == 1:
            o_ref[pl.ds(r0, _NUM_SUBLANES), :, :] = out_c
          else:
            o_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :] = out_c

    in_specs = [
        pl.BlockSpec(x_block_shape, _x_index_map_3d),
        pl.BlockSpec(s_block_shape, _s_index_map_3d),
    ]
    inputs = [x_in, s_in]
    scratch_shapes = []
    if w_in is not None:
      in_specs.append(pl.BlockSpec((_NUM_SUBLANES, eff_block_m), _w_index_map))
      inputs.append(w_in)
      scratch_shapes.append(pltpu.VMEM((eff_block_m, _NUM_LANES), jnp.float32))

    out_padded = pl.pallas_call(
        _fwd_kernel_3d,
        out_shape=jax.ShapeDtypeStruct(x_in.shape, x_in.dtype, manual_axis_type=_manual_axis_type(expert_out)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_steps,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(x_block_shape, _x_index_map_3d),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
        name="fused_ragged_post_expert_norm_fwd",
    )(count_1d, *inputs)
    return out_padded[:orig_cap] if pad_m > 0 else out_padded

  # 2D `[cap, dim]` path with `(8, 1024)` register-tiled inner loop (`0` VMEM spills):
  col_chunk_tiles = math.gcd(num_col_tiles, _NUM_SUBLANES)
  col_chunk = col_chunk_tiles * _NUM_LANES
  num_col_chunks = num_col_tiles // col_chunk_tiles

  s_in = jnp.broadcast_to(s_f32.reshape(1, dim), (_NUM_SUBLANES, dim))
  x_block_shape = (eff_block_m, dim)
  s_block_shape = (_NUM_SUBLANES, dim)

  def _fwd_kernel_2d(count_ref, *refs):
    if has_weights:
      x_ref, s_ref, w_ref, o_ref, w_scratch_ref = refs
      w_scratch_ref[...] = _unpack_weights_to_rows(w_ref[...], eff_block_m)
    else:
      x_ref, s_ref, o_ref = refs
      w_scratch_ref = None

    pid = pl.program_id(0)
    n_valid = count_ref[0]
    inv_dim_f32 = jnp.asarray(inv_dim, jnp.float32)
    sub_iota = lax.broadcasted_iota(jnp.int32, (_NUM_SUBLANES, _NUM_LANES), 0)

    @pl.loop(0, eff_block_m // _NUM_SUBLANES)
    def _sub_step(sub_idx):
      r0 = pl.multiple_of(sub_idx * _NUM_SUBLANES, _NUM_SUBLANES)
      valid_8 = (pid * eff_block_m + r0 + sub_iota) < n_valid
      valid_c = valid_8 if col_chunk_tiles == 1 else jnp.tile(valid_8, (1, col_chunk_tiles))

      sq_acc = jnp.zeros((_NUM_SUBLANES, col_chunk), jnp.float32)
      for c in range(num_col_chunks):
        c0 = c * col_chunk
        x_c = jnp.where(valid_c, x_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk], jnp.zeros((), x_ref.dtype))
        sf_c = _round_to_dtype_f32(x_c.astype(jnp.float32) * s_ref[:, c0 : c0 + col_chunk], x_ref.dtype)
        sq_acc = sq_acc + lax.square(sf_c)

      sq_128 = _sum_2d_col_tiles(sq_acc, col_chunk_tiles)
      row_sq = jnp.broadcast_to(jnp.sum(sq_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
      rstd_8 = lax.rsqrt(row_sq * inv_dim_f32 + epsilon)
      rstd_c = rstd_8 if col_chunk_tiles == 1 else jnp.tile(rstd_8, (1, col_chunk_tiles))

      w_c = None
      if w_scratch_ref is not None:
        w_8 = jnp.where(valid_8, w_scratch_ref[pl.ds(r0, _NUM_SUBLANES), :], 0.0)
        w_c = w_8 if col_chunk_tiles == 1 else jnp.tile(w_8, (1, col_chunk_tiles))

      for c in range(num_col_chunks):
        c0 = c * col_chunk
        x_c = jnp.where(valid_c, x_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk], jnp.zeros((), x_ref.dtype))
        sf_c = _round_to_dtype_f32(x_c.astype(jnp.float32) * s_ref[:, c0 : c0 + col_chunk], x_ref.dtype)
        norm_c = (sf_c * rstd_c).astype(x_ref.dtype)
        out_c = (norm_c.astype(jnp.float32) * w_c).astype(x_ref.dtype) if w_c is not None else norm_c
        o_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk] = jnp.where(valid_c, out_c, jnp.zeros((), x_ref.dtype))

  in_specs = [
      pl.BlockSpec(x_block_shape, _x_index_map_2d),
      pl.BlockSpec(s_block_shape, _s_index_map_2d),
  ]
  inputs = [x_in, s_in]
  scratch_shapes = []
  if w_in is not None:
    in_specs.append(pl.BlockSpec((_NUM_SUBLANES, eff_block_m), _w_index_map))
    inputs.append(w_in)
    scratch_shapes.append(pltpu.VMEM((eff_block_m, _NUM_LANES), jnp.float32))

  out_padded = pl.pallas_call(
      _fwd_kernel_2d,
      out_shape=jax.ShapeDtypeStruct(x_in.shape, x_in.dtype, manual_axis_type=_manual_axis_type(expert_out)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(num_steps,),
          in_specs=in_specs,
          out_specs=pl.BlockSpec(x_block_shape, _x_index_map_2d),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
      name="fused_ragged_post_expert_norm_fwd",
  )(count_1d, *inputs)
  return out_padded[:orig_cap] if pad_m > 0 else out_padded


def _pallas_bwd(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    sorted_weights: jax.Array | None,
    g: jax.Array,
    *,
    epsilon: float,
    block_m: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """1-pass HBM spill-free Pallas VMEM backward over valid rows `[0, count)`."""
  orig_cap = expert_out.shape[0]
  is_3d = expert_out.ndim == 3
  dim = expert_out.shape[1] * (expert_out.shape[2] if is_3d else 1)
  num_col_tiles = dim // _NUM_LANES
  eff_block_m = max(_NUM_LANES, (block_m // _NUM_LANES) * _NUM_LANES)
  aligned_cap = max(eff_block_m, pl.cdiv(max(orig_cap, 1), eff_block_m) * eff_block_m)
  pad_m = aligned_cap - orig_cap

  x_in = jnp.pad(expert_out, ((0, pad_m),) + ((0, 0),) * (expert_out.ndim - 1)) if pad_m > 0 else expert_out
  g_in = jnp.pad(g, ((0, pad_m),) + ((0, 0),) * (g.ndim - 1)) if pad_m > 0 else g

  s_f32 = _centered_scale_fp32(scale)
  has_weights = sorted_weights is not None
  w_in = None
  if sorted_weights is not None:
    w_f32 = sorted_weights.astype(jnp.float32)
    if pad_m > 0:
      w_f32 = jnp.pad(w_f32, (0, pad_m))
    w_in = jnp.broadcast_to(w_f32[None, :], (_NUM_SUBLANES, aligned_cap))

  count_1d = jnp.reshape(jnp.clip(jnp.asarray(count, jnp.int32), 0, orig_cap), (1,))
  # Always run >= 1 step so empty shards (`count == 0`) deterministically zero-initialize `d_scale`.
  safe_steps = jnp.maximum(pl.cdiv(count_1d[0], eff_block_m), 1)
  inv_dim = 1.0 / float(dim)

  mat = _manual_axis_type(expert_out)
  s_mat = _manual_axis_type(scale)

  if is_3d:
    col_chunks = _col_chunk_ranges_3d(num_col_tiles)
    s_in = s_f32.reshape(1, num_col_tiles, _NUM_LANES)
    x_block_shape = (eff_block_m, num_col_tiles, _NUM_LANES)
    s_block_shape = (1, num_col_tiles, _NUM_LANES)

    def _bwd_kernel_3d(count_ref, *refs):
      if has_weights:
        x_ref, g_ref, s_ref, w_ref, dx_ref, ds_out_ref, dw_ref, ds_acc_ref, w_scratch_ref, dw_scratch_ref = refs
        w_scratch_ref[...] = _unpack_weights_to_rows(w_ref[...], eff_block_m)
      else:
        x_ref, g_ref, s_ref, dx_ref, ds_out_ref, ds_acc_ref = refs
        dw_ref = w_scratch_ref = dw_scratch_ref = None

      pid = pl.program_id(0)
      n_valid = count_ref[0]
      inv_dim_f32 = jnp.asarray(inv_dim, jnp.float32)
      sub_iota = lax.broadcasted_iota(jnp.int32, (_NUM_SUBLANES, _NUM_LANES), 0)

      @pl.when(pid == 0)
      def _init_acc():
        for c0, chunk_t in col_chunks:
          if len(col_chunks) == 1:
            ds_acc_ref[:, :, :] = jnp.zeros((1, num_col_tiles, _NUM_LANES), jnp.float32)
          else:
            ds_acc_ref[:, pl.ds(c0, chunk_t), :] = jnp.zeros((1, chunk_t, _NUM_LANES), jnp.float32)

      @pl.loop(0, eff_block_m // _NUM_SUBLANES)
      def _sub_step(sub_idx):
        r0 = pl.multiple_of(sub_idx * _NUM_SUBLANES, _NUM_SUBLANES)
        valid_8 = (pid * eff_block_m + r0 + sub_iota) < n_valid
        valid_3d = jnp.expand_dims(valid_8, axis=1)

        if w_scratch_ref is not None:
          w_8 = jnp.where(valid_8, w_scratch_ref[pl.ds(r0, _NUM_SUBLANES), :], 0.0)
          w_3d = jnp.expand_dims(w_8, axis=1)
        else:
          w_3d = None

        # Sweep 1: Accumulate `sq = sum(sf^2)` and `d_rstd = sum(d_norm * sf)` in `(<=8)`-tile registers (`0` spills)!
        sq_128 = jnp.zeros((_NUM_SUBLANES, _NUM_LANES), jnp.float32)
        drstd_128 = jnp.zeros((_NUM_SUBLANES, _NUM_LANES), jnp.float32)
        for c0, chunk_t in col_chunks:
          if len(col_chunks) == 1:
            x_c = jnp.where(valid_3d, x_ref[pl.ds(r0, _NUM_SUBLANES), :, :], jnp.zeros((), x_ref.dtype))
            g_c = jnp.where(valid_3d, g_ref[pl.ds(r0, _NUM_SUBLANES), :, :], jnp.zeros((), g_ref.dtype))
            s_c = s_ref[:, :, :]
          else:
            x_c = jnp.where(
                valid_3d,
                x_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :],
                jnp.zeros((), x_ref.dtype),
            )
            g_c = jnp.where(
                valid_3d,
                g_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :],
                jnp.zeros((), g_ref.dtype),
            )
            s_c = s_ref[:, pl.ds(c0, chunk_t), :]
          sf_c = _round_to_dtype_f32(x_c.astype(jnp.float32) * s_c, x_ref.dtype)
          g_c_f32 = g_c.astype(jnp.float32)
          d_norm_c = _round_to_dtype_f32(g_c_f32 * w_3d, x_ref.dtype) if w_3d is not None else g_c_f32
          sq_128 = sq_128 + jnp.sum(lax.square(sf_c), axis=1)
          drstd_128 = drstd_128 + jnp.sum(d_norm_c * sf_c, axis=1)

        sq_sum = jnp.broadcast_to(jnp.sum(sq_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
        d_rstd_8 = jnp.broadcast_to(jnp.sum(drstd_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
        var_eps_8 = sq_sum * inv_dim_f32 + epsilon
        rstd_8 = lax.rsqrt(var_eps_8)
        rstd_3d = jnp.expand_dims(rstd_8, axis=1)
        d_var_eps_8 = d_rstd_8 * (-0.5 * lax.div(rstd_8, var_eps_8))
        c_mean_3d = jnp.expand_dims(d_var_eps_8 * (2.0 * inv_dim_f32), axis=1)

        # Sweep 2: Compute `dx`, `dw`, and accumulate `(1, <=8, 128)` `ds_acc_ref` using pure VPU `vadd`!
        dw_128 = jnp.zeros((_NUM_SUBLANES, _NUM_LANES), jnp.float32) if dw_scratch_ref is not None else None
        for c0, chunk_t in col_chunks:
          if len(col_chunks) == 1:
            x_c = jnp.where(valid_3d, x_ref[pl.ds(r0, _NUM_SUBLANES), :, :], jnp.zeros((), x_ref.dtype))
            g_c = jnp.where(valid_3d, g_ref[pl.ds(r0, _NUM_SUBLANES), :, :], jnp.zeros((), g_ref.dtype))
            s_c = s_ref[:, :, :]
          else:
            x_c = jnp.where(
                valid_3d,
                x_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :],
                jnp.zeros((), x_ref.dtype),
            )
            g_c = jnp.where(
                valid_3d,
                g_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :],
                jnp.zeros((), g_ref.dtype),
            )
            s_c = s_ref[:, pl.ds(c0, chunk_t), :]
          x_c_f32 = x_c.astype(jnp.float32)
          sf_c = _round_to_dtype_f32(x_c_f32 * s_c, x_ref.dtype)
          g_c_f32 = g_c.astype(jnp.float32)
          if dw_128 is not None:
            norm_f32_c = _round_to_dtype_f32(sf_c * rstd_3d, x_ref.dtype)
            dw_128 = dw_128 + jnp.sum(g_c_f32 * norm_f32_c, axis=1)
          d_norm_c = _round_to_dtype_f32(g_c_f32 * w_3d, x_ref.dtype) if w_3d is not None else g_c_f32
          du_c = jnp.where(valid_3d, _round_to_dtype_f32(d_norm_c * rstd_3d + c_mean_3d * sf_c, x_ref.dtype), 0.0)
          dx_c = jnp.where(valid_3d, (du_c * s_c).astype(x_ref.dtype), jnp.zeros((), x_ref.dtype))
          ds_step_c = jnp.sum(du_c * x_c_f32, axis=0, keepdims=True)
          if len(col_chunks) == 1:
            dx_ref[pl.ds(r0, _NUM_SUBLANES), :, :] = dx_c
            ds_acc_ref[:, :, :] = ds_acc_ref[:, :, :] + ds_step_c
          else:
            dx_ref[pl.ds(r0, _NUM_SUBLANES), pl.ds(c0, chunk_t), :] = dx_c
            ds_acc_ref[:, pl.ds(c0, chunk_t), :] = ds_acc_ref[:, pl.ds(c0, chunk_t), :] + ds_step_c

        if dw_scratch_ref is not None and dw_128 is not None:
          dw_sum = jnp.broadcast_to(jnp.sum(dw_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
          dw_scratch_ref[pl.ds(r0, _NUM_SUBLANES), :] = jnp.where(valid_8, dw_sum, 0.0)

      if dw_ref is not None and dw_scratch_ref is not None:
        dw_ref[...] = _pack_rows_to_weights(dw_scratch_ref[...], eff_block_m)

      for c0, chunk_t in col_chunks:
        if len(col_chunks) == 1:
          ds_out_ref[:, :, :] = ds_acc_ref[:, :, :]
        else:
          ds_out_ref[:, pl.ds(c0, chunk_t), :] = ds_acc_ref[:, pl.ds(c0, chunk_t), :]

    in_specs = [
        pl.BlockSpec(x_block_shape, _x_index_map_3d),
        pl.BlockSpec(x_block_shape, _x_index_map_3d),
        pl.BlockSpec(s_block_shape, _s_index_map_3d),
    ]
    inputs = [x_in, g_in, s_in]
    scratch_shapes = [pltpu.VMEM(s_block_shape, jnp.float32)]
    if w_in is not None:
      in_specs.append(pl.BlockSpec((_NUM_SUBLANES, eff_block_m), _w_index_map))
      inputs.append(w_in)
      scratch_shapes.append(pltpu.VMEM((eff_block_m, _NUM_LANES), jnp.float32))
      scratch_shapes.append(pltpu.VMEM((eff_block_m, _NUM_LANES), jnp.float32))

    out_shapes = [
        jax.ShapeDtypeStruct(x_in.shape, x_in.dtype, manual_axis_type=mat),
        jax.ShapeDtypeStruct(s_block_shape, jnp.float32, manual_axis_type=s_mat),
    ]
    out_specs = [
        pl.BlockSpec(x_block_shape, _x_index_map_3d),
        pl.BlockSpec(s_block_shape, _s_index_map_3d),
    ]
    if sorted_weights is not None:
      w_mat = _manual_axis_type(sorted_weights)
      out_shapes.append(jax.ShapeDtypeStruct((_NUM_SUBLANES, aligned_cap), jnp.float32, manual_axis_type=w_mat))
      out_specs.append(pl.BlockSpec((_NUM_SUBLANES, eff_block_m), _w_index_map))

    outs = pl.pallas_call(
        _bwd_kernel_3d,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(safe_steps,),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
        name="fused_ragged_post_expert_norm_bwd",
    )(count_1d, *inputs)

    if sorted_weights is not None:
      dx_padded, ds_3d, dw_sublanes = outs
      valid_orig = jnp.arange(orig_cap, dtype=jnp.int32) < count_1d[0]
      dw = jnp.where(valid_orig, dw_sublanes[0, :orig_cap], 0.0).astype(sorted_weights.dtype)
    else:
      dx_padded, ds_3d = outs
      dw = None

    dx = dx_padded[:orig_cap] if pad_m > 0 else dx_padded
    dscale = ds_3d.reshape(scale.shape).astype(scale.dtype)
    return dx, dscale, dw

  # 2D `[cap, dim]` backward path with 2-sweep `(8, 1024)` register-tiled loop (`0` VMEM spills):
  col_chunk_tiles = math.gcd(num_col_tiles, _NUM_SUBLANES)
  col_chunk = col_chunk_tiles * _NUM_LANES
  num_col_chunks = num_col_tiles // col_chunk_tiles

  s_in = jnp.broadcast_to(s_f32.reshape(1, dim), (_NUM_SUBLANES, dim))
  x_block_shape = (eff_block_m, dim)
  s_block_shape = (_NUM_SUBLANES, dim)

  def _bwd_kernel_2d(count_ref, *refs):
    if has_weights:
      x_ref, g_ref, s_ref, w_ref, dx_ref, ds_out_ref, dw_ref, ds_acc_ref, w_scratch_ref, dw_scratch_ref = refs
      w_scratch_ref[...] = _unpack_weights_to_rows(w_ref[...], eff_block_m)
    else:
      x_ref, g_ref, s_ref, dx_ref, ds_out_ref, ds_acc_ref = refs
      dw_ref = w_scratch_ref = dw_scratch_ref = None

    pid = pl.program_id(0)
    n_valid = count_ref[0]
    inv_dim_f32 = jnp.asarray(inv_dim, jnp.float32)
    sub_iota = lax.broadcasted_iota(jnp.int32, (_NUM_SUBLANES, _NUM_LANES), 0)

    @pl.when(pid == 0)
    def _init_acc():
      for c in range(num_col_chunks):
        c0 = c * col_chunk
        ds_acc_ref[:, c0 : c0 + col_chunk] = jnp.zeros((_NUM_SUBLANES, col_chunk), jnp.float32)

    @pl.loop(0, eff_block_m // _NUM_SUBLANES)
    def _sub_step(sub_idx):
      r0 = pl.multiple_of(sub_idx * _NUM_SUBLANES, _NUM_SUBLANES)
      valid_8 = (pid * eff_block_m + r0 + sub_iota) < n_valid
      valid_c = valid_8 if col_chunk_tiles == 1 else jnp.tile(valid_8, (1, col_chunk_tiles))

      if w_scratch_ref is not None:
        w_8 = jnp.where(valid_8, w_scratch_ref[pl.ds(r0, _NUM_SUBLANES), :], 0.0)
        w_c = w_8 if col_chunk_tiles == 1 else jnp.tile(w_8, (1, col_chunk_tiles))
      else:
        w_c = None

      sq_acc = jnp.zeros((_NUM_SUBLANES, col_chunk), jnp.float32)
      drstd_acc = jnp.zeros((_NUM_SUBLANES, col_chunk), jnp.float32)
      for c in range(num_col_chunks):
        c0 = c * col_chunk
        x_c = jnp.where(valid_c, x_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk], jnp.zeros((), x_ref.dtype))
        g_c = jnp.where(valid_c, g_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk], jnp.zeros((), g_ref.dtype))
        sf_c = _round_to_dtype_f32(x_c.astype(jnp.float32) * s_ref[:, c0 : c0 + col_chunk], x_ref.dtype)
        g_c_f32 = g_c.astype(jnp.float32)
        d_norm_c = _round_to_dtype_f32(g_c_f32 * w_c, x_ref.dtype) if w_c is not None else g_c_f32
        sq_acc = sq_acc + lax.square(sf_c)
        drstd_acc = drstd_acc + d_norm_c * sf_c

      sq_128 = _sum_2d_col_tiles(sq_acc, col_chunk_tiles)
      drstd_128 = _sum_2d_col_tiles(drstd_acc, col_chunk_tiles)
      sq_sum = jnp.broadcast_to(jnp.sum(sq_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
      d_rstd_8 = jnp.broadcast_to(jnp.sum(drstd_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
      var_eps_8 = sq_sum * inv_dim_f32 + epsilon
      rstd_8 = lax.rsqrt(var_eps_8)
      rstd_c = rstd_8 if col_chunk_tiles == 1 else jnp.tile(rstd_8, (1, col_chunk_tiles))
      d_var_eps_8 = d_rstd_8 * (-0.5 * lax.div(rstd_8, var_eps_8))
      c_mean_8 = d_var_eps_8 * (2.0 * inv_dim_f32)
      c_mean_c = c_mean_8 if col_chunk_tiles == 1 else jnp.tile(c_mean_8, (1, col_chunk_tiles))

      dw_acc = jnp.zeros((_NUM_SUBLANES, col_chunk), jnp.float32) if dw_scratch_ref is not None else None
      for c in range(num_col_chunks):
        c0 = c * col_chunk
        x_c = jnp.where(valid_c, x_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk], jnp.zeros((), x_ref.dtype))
        g_c = jnp.where(valid_c, g_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk], jnp.zeros((), g_ref.dtype))
        x_c_f32 = x_c.astype(jnp.float32)
        s_c = s_ref[:, c0 : c0 + col_chunk]
        sf_c = _round_to_dtype_f32(x_c_f32 * s_c, x_ref.dtype)
        g_c_f32 = g_c.astype(jnp.float32)
        if dw_acc is not None:
          norm_f32_c = _round_to_dtype_f32(sf_c * rstd_c, x_ref.dtype)
          dw_acc = dw_acc + g_c_f32 * norm_f32_c
        d_norm_c = _round_to_dtype_f32(g_c_f32 * w_c, x_ref.dtype) if w_c is not None else g_c_f32
        du_c = jnp.where(valid_c, _round_to_dtype_f32(d_norm_c * rstd_c + c_mean_c * sf_c, x_ref.dtype), 0.0)
        dx_ref[pl.ds(r0, _NUM_SUBLANES), c0 : c0 + col_chunk] = jnp.where(
            valid_c, (du_c * s_c).astype(x_ref.dtype), jnp.zeros((), x_ref.dtype)
        )
        ds_acc_ref[:, c0 : c0 + col_chunk] = ds_acc_ref[:, c0 : c0 + col_chunk] + du_c * x_c_f32

      if dw_scratch_ref is not None and dw_acc is not None:
        dw_128 = _sum_2d_col_tiles(dw_acc, col_chunk_tiles)
        dw_sum = jnp.broadcast_to(jnp.sum(dw_128, axis=-1, keepdims=True), (_NUM_SUBLANES, _NUM_LANES))
        dw_scratch_ref[pl.ds(r0, _NUM_SUBLANES), :] = jnp.where(valid_8, dw_sum, 0.0)

    if dw_ref is not None and dw_scratch_ref is not None:
      dw_ref[...] = _pack_rows_to_weights(dw_scratch_ref[...], eff_block_m)

    for c in range(num_col_chunks):
      c0 = c * col_chunk
      ds_out_ref[:, c0 : c0 + col_chunk] = ds_acc_ref[:, c0 : c0 + col_chunk]

  in_specs = [
      pl.BlockSpec(x_block_shape, _x_index_map_2d),
      pl.BlockSpec(x_block_shape, _x_index_map_2d),
      pl.BlockSpec(s_block_shape, _s_index_map_2d),
  ]
  inputs = [x_in, g_in, s_in]
  scratch_shapes = [pltpu.VMEM(s_block_shape, jnp.float32)]
  if w_in is not None:
    in_specs.append(pl.BlockSpec((_NUM_SUBLANES, eff_block_m), _w_index_map))
    inputs.append(w_in)
    scratch_shapes.append(pltpu.VMEM((eff_block_m, _NUM_LANES), jnp.float32))
    scratch_shapes.append(pltpu.VMEM((eff_block_m, _NUM_LANES), jnp.float32))

  out_shapes = [
      jax.ShapeDtypeStruct(x_in.shape, x_in.dtype, manual_axis_type=mat),
      jax.ShapeDtypeStruct(s_block_shape, jnp.float32, manual_axis_type=s_mat),
  ]
  out_specs = [
      pl.BlockSpec(x_block_shape, _x_index_map_2d),
      pl.BlockSpec(s_block_shape, _s_index_map_2d),
  ]
  if sorted_weights is not None:
    w_mat = _manual_axis_type(sorted_weights)
    out_shapes.append(jax.ShapeDtypeStruct((_NUM_SUBLANES, aligned_cap), jnp.float32, manual_axis_type=w_mat))
    out_specs.append(pl.BlockSpec((_NUM_SUBLANES, eff_block_m), _w_index_map))

  outs = pl.pallas_call(
      _bwd_kernel_2d,
      out_shape=out_shapes,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(safe_steps,),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
      name="fused_ragged_post_expert_norm_bwd",
  )(count_1d, *inputs)

  if sorted_weights is not None:
    dx_padded, ds_sublanes, dw_sublanes = outs
    valid_orig = jnp.arange(orig_cap, dtype=jnp.int32) < count_1d[0]
    dw = jnp.where(valid_orig, dw_sublanes[0, :orig_cap], 0.0).astype(sorted_weights.dtype)
  else:
    dx_padded, ds_sublanes = outs
    dw = None

  dx = dx_padded[:orig_cap] if pad_m > 0 else dx_padded
  dscale = jnp.sum(ds_sublanes, axis=0).reshape(scale.shape).astype(scale.dtype)
  return dx, dscale, dw


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _pallas_norm_with_vjp(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    sorted_weights: jax.Array | None,
    epsilon: float,
    block_m: int,
) -> jax.Array:
  return _pallas_fwd(expert_out, scale, count, sorted_weights, epsilon=epsilon, block_m=block_m)


def _pallas_norm_vjp_fwd(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array,
    sorted_weights: jax.Array | None,
    epsilon: float,
    block_m: int,
):
  """Forward VJP rule for `_pallas_norm_with_vjp`."""
  out = _pallas_fwd(expert_out, scale, count, sorted_weights, epsilon=epsilon, block_m=block_m)
  return out, (expert_out, scale, count, sorted_weights)


def _pallas_norm_vjp_bwd(epsilon: float, block_m: int, res, g: jax.Array):
  """Backward VJP rule for `_pallas_norm_with_vjp`."""
  expert_out, scale, count, sorted_weights = res
  dx, dscale, dw = _pallas_bwd(
      expert_out,
      scale,
      count,
      sorted_weights,
      g,
      epsilon=epsilon,
      block_m=block_m,
  )
  return dx, dscale, None, dw


_pallas_norm_with_vjp.defvjp(_pallas_norm_vjp_fwd, _pallas_norm_vjp_bwd)


def fused_ragged_post_expert_norm(
    expert_out: jax.Array,
    scale: jax.Array,
    count: jax.Array | int,
    *,
    sorted_weights: jax.Array | None = None,
    epsilon: float = 1e-8,
    block_m: int = 256,
    is_tpu: bool = True,
) -> jax.Array:
  """1-pass FWD + 1-pass BWD Pallas VMEM `MuseSparkInputScaledRMSNorm` over `[0, count)`.

  Args:
    expert_out: Ragged expert buffer of shape `[cap, D]` or `[cap, D // 128, 128]`.
    scale: Centered RMSNorm scale parameter of shape `[D]` (computes `(1.0 + scale).f32`).
    count: Scalar `int32` valid row count in `[0, cap]`. Rows `[count, cap)` are not
      visited by the Pallas dynamic grid (and partial tail blocks are masked in VMEM
      so uninitialized `NaN`/`Inf` values in `[count, cap)` never corrupt `rstd` or `d_scale`).
    sorted_weights: Optional `[cap]` `float32` per-row routing weights; when provided,
      multiplies `norm_out` by `sorted_weights` inside the same VMEM pass.
    epsilon: RMSNorm variance epsilon (`1e-8` in Muse Spark).
    block_m: Rows per VMEM tile (aligned to `128`, default `256` = `4 MiB` double-buffered).
    is_tpu: Explicit target-platform flag (`self.mesh.devices.flat[0].platform == "tpu"`).
      Uses pure-JAX reference when `False` without querying `jax.default_backend()` so
      CPU-hosted 256c AOT TPU compilation lowers the real Pallas TPU kernel.

  Returns:
    Normalized (and optionally routing-weight-scaled) buffer of `expert_out.shape`.
  """
  if expert_out.ndim not in (2, 3):
    raise ValueError(f"expert_out must be 2D [cap, D] or 3D [cap, D//128, 128], got {expert_out.shape}.")
  count_arr = jnp.asarray(count, dtype=jnp.int32)
  if not is_tpu:
    return _jax_reference_with_vjp(expert_out, scale, count_arr, epsilon, sorted_weights)
  return _pallas_norm_with_vjp(expert_out, scale, count_arr, sorted_weights, epsilon, block_m)
