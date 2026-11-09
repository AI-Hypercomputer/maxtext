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

"""Fused Pallas VMEM QK prologue and load-balanced CP KV chunk reorder."""

from __future__ import annotations

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


def _make_rot_half_matrix(
    head_dim: int,
    dtype: Any = jnp.bfloat16,
) -> np.ndarray:
  """Builds `[D, D]` matrix mapping `(x1, x2) -> (-x2, x1)` on MXU."""
  half = head_dim // 2
  mat = np.zeros((head_dim, head_dim), dtype=np.float32)
  mat[half:, :half] = -np.eye(half, dtype=np.float32)
  mat[:half, half:] = np.eye(half, dtype=np.float32)
  return mat.astype(dtype)


def select_prologue_block_s(local_seq_len: int, local_num_heads: int) -> int:
  """Selects sequence tile size balancing DMA efficiency and VMEM footprint.

  Args:
    local_seq_len: Sequence length per shard.
    local_num_heads: Number of query or key heads per shard.

  Returns:
    Sequence tile size (`128`, `256`, or `512`).
  """
  if local_num_heads <= 8 and local_seq_len >= 512 and local_seq_len % 512 == 0:
    return 512
  if local_seq_len >= 256 and local_seq_len % 256 == 0:
    return 256
  return 128


def can_use_fused_qk_prologue(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    cos_2d: jax.Array | None = None,
    sin_2d: jax.Array | None = None,
    block_s: int = 0,
) -> bool:
  """Returns True when `x_loc` (`[B, S, H, 128]`) fits Pallas QK prologue.

  Args:
    x_loc: Local input tensor `[B, S, H, 128]` in `bfloat16`.
    effective_scale: RMSNorm scale vector `[128]`.
    cos_2d: Optional RoPE cosine table, or `None` when RoPE is disabled.
    sin_2d: Optional RoPE sine table, or `None` when RoPE is disabled.
    block_s: Sequence tile size, or `0` to auto-select.

  Returns:
    True if the device, dtypes, and shapes support the fused Pallas prologue.
  """
  if not pltpu.is_tpu_device():
    return False
  if x_loc.ndim != 4 or x_loc.dtype != jnp.bfloat16 or x_loc.shape[-1] != 128:
    return False
  if effective_scale.shape != (128,):
    return False
  bsz, s_loc, num_h, _ = x_loc.shape
  if bsz <= 0 or num_h <= 0:
    return False
  bs = block_s if block_s > 0 else select_prologue_block_s(s_loc, num_h)
  if s_loc < bs or s_loc % bs != 0 or (num_h * bs) % 8 != 0:
    return False
  if (cos_2d is None) != (sin_2d is None):
    return False
  if cos_2d is not None:
    assert sin_2d is not None
    valid_shapes = ((bsz, s_loc, 128), (s_loc, 128))
    if cos_2d.shape not in valid_shapes or sin_2d.shape != cos_2d.shape:
      return False
    if cos_2d.dtype != jnp.bfloat16 or sin_2d.dtype != jnp.bfloat16:
      return False
  return True


def _pallas_prologue_fwd_kernel(
    x_ref: Any,
    scale_ref: Any,
    cos_ref: Any,
    sin_ref: Any,
    rot_mat_ref: Any,
    out_ref: Any,
    *,
    num_h: int,
    dim: int,
    epsilon: float,
    pre_attn_scalar: float | None,
    use_rope: bool,
) -> None:
  """Fused VMEM fwd: unpack BSHD -> FP32 RMSNorm -> RoPE -> scalar -> BHSD."""
  x_2d = x_ref[0]
  bs = x_2d.shape[0]
  x_hsd = jnp.stack([x_2d[:, h * dim : (h + 1) * dim] for h in range(num_h)], axis=0)
  x_flat = x_hsd.reshape(num_h * bs, dim).astype(jnp.float32)
  mean2 = jnp.mean(lax.square(x_flat), axis=-1, keepdims=True)
  normed_8 = (x_flat * lax.rsqrt(mean2 + np.float32(epsilon))).reshape((num_h * bs) // 8, 8, dim)
  y_bf16 = (normed_8 * scale_ref[...][None, :, :]).astype(jnp.bfloat16).reshape(num_h, bs, dim)
  if use_rope:
    cos_b = cos_ref[0][None, :, :]
    sin_b = sin_ref[0][None, :, :]
    y_rot = (
        lax.dot_general(
            y_bf16.reshape(num_h * bs, dim),
            rot_mat_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .reshape(num_h, bs, dim)
    )
    y_bf16 = (
        y_bf16.astype(jnp.float32) * cos_b.astype(jnp.float32) + y_rot.astype(jnp.float32) * sin_b.astype(jnp.float32)
    ).astype(jnp.bfloat16)
  if pre_attn_scalar is not None and pre_attn_scalar != 1.0:
    y_bf16 = (y_bf16.astype(jnp.float32) * np.float32(pre_attn_scalar)).astype(jnp.bfloat16)
  out_ref[0] = y_bf16


def _pallas_prologue_bwd_kernel(
    x_ref: Any,
    dy_ref: Any,
    scale_ref: Any,
    cos_ref: Any,
    sin_ref: Any,
    rot_mat_t_ref: Any,
    dx_ref: Any,
    dscale_part_ref: Any,
    *,
    num_h: int,
    dim: int,
    epsilon: float,
    pre_attn_scalar: float | None,
    use_rope: bool,
) -> None:
  """Fused VMEM bwd: BHSD dQ/dK -> scalar -> RoPE^T -> RMSNorm VJP -> BSHD."""
  dy = dy_ref[0]
  bs = dy.shape[1]
  if pre_attn_scalar is not None and pre_attn_scalar != 1.0:
    dy = (dy.astype(jnp.float32) * np.float32(pre_attn_scalar)).astype(jnp.bfloat16)
  if use_rope:
    cos_b = cos_ref[0][None, :, :]
    sin_b = sin_ref[0][None, :, :]
    dy_sin = (dy.astype(jnp.float32) * sin_b.astype(jnp.float32)).astype(jnp.bfloat16).reshape(num_h * bs, dim)
    dy_rot_t = (
        lax.dot_general(
            dy_sin,
            rot_mat_t_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        .astype(jnp.bfloat16)
        .reshape(num_h, bs, dim)
    )
    dy = (dy.astype(jnp.float32) * cos_b.astype(jnp.float32) + dy_rot_t.astype(jnp.float32)).astype(jnp.bfloat16)

  dy_flat = dy.reshape(num_h * bs, dim).astype(jnp.float32)
  x_2d = x_ref[0]
  x_hsd = jnp.stack([x_2d[:, h * dim : (h + 1) * dim] for h in range(num_h)], axis=0)
  x_flat = x_hsd.reshape(num_h * bs, dim).astype(jnp.float32)
  mean2 = jnp.mean(lax.square(x_flat), axis=-1, keepdims=True)
  rstd = lax.rsqrt(mean2 + np.float32(epsilon))
  normed_flat = x_flat * rstd

  dscale_prod = (dy_flat * normed_flat).reshape((num_h * bs) // 8, 8, dim)
  dscale_part_ref[0] = jnp.sum(dscale_prod, axis=0)

  scale_tile = scale_ref[...]
  dy_scaled = (dy_flat.reshape((num_h * bs) // 8, 8, dim) * scale_tile[None, :, :]).reshape(num_h * bs, dim)
  dot_x = jnp.mean(dy_scaled * normed_flat, axis=-1, keepdims=True)
  dx_hsd = (rstd * (dy_scaled - normed_flat * dot_x)).astype(jnp.bfloat16).reshape(num_h, bs, dim)
  dx_ref[0] = jnp.concatenate([dx_hsd[h] for h in range(num_h)], axis=-1)


def _normalize_rope_table(
    table: jax.Array | None,
    bsz: int,
) -> jax.Array | None:
  """Broadcasts 2D `[S_loc, D]` RoPE tables to `[B_loc, S_loc, D]`."""
  if table is None or table.ndim != 2:
    return table
  return jnp.broadcast_to(table[None, :, :], (bsz, table.shape[0], table.shape[1]))


def _run_pallas_fwd_local(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    cos_loc: jax.Array | None,
    sin_loc: jax.Array | None,
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
) -> jax.Array:
  """Runs local Pallas QK prologue forward inside an existing `shard_map`."""
  bsz, seq_len, num_h, dim = x_loc.shape
  cos_loc = _normalize_rope_table(cos_loc, bsz)
  sin_loc = _normalize_rope_table(sin_loc, bsz)
  bs = block_s if block_s > 0 else select_prologue_block_s(seq_len, num_h)
  num_s = seq_len // bs
  x_packed = x_loc.reshape(bsz, seq_len, num_h * dim)
  scale_2d = jnp.broadcast_to(effective_scale.astype(jnp.float32)[None, :], (8, dim))
  use_rope = cos_loc is not None and sin_loc is not None
  if use_rope:
    rot_mat = jnp.asarray(_make_rot_half_matrix(dim, jnp.bfloat16))
    return pl.pallas_call(
        functools.partial(
            _pallas_prologue_fwd_kernel,
            num_h=num_h,
            dim=dim,
            epsilon=epsilon,
            pre_attn_scalar=pre_attn_scalar,
            use_rope=True,
        ),
        out_shape=jax.ShapeDtypeStruct((bsz, num_h, seq_len, dim), jnp.bfloat16),
        grid=(bsz, num_s),
        in_specs=[
            pl.BlockSpec((1, bs, num_h * dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((8, dim), lambda b, s: (0, 0)),
            pl.BlockSpec((1, bs, dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, bs, dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((dim, dim), lambda b, s: (0, 0)),
        ],
        out_specs=pl.BlockSpec((1, num_h, bs, dim), lambda b, s: (b, 0, s, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        name="pallas_qk_prologue_fwd",
    )(x_packed, scale_2d, cos_loc, sin_loc, rot_mat)

  return pl.pallas_call(
      functools.partial(
          _pallas_prologue_fwd_kernel,
          num_h=num_h,
          dim=dim,
          epsilon=epsilon,
          pre_attn_scalar=pre_attn_scalar,
          use_rope=False,
      ),
      out_shape=jax.ShapeDtypeStruct((bsz, num_h, seq_len, dim), jnp.bfloat16),
      grid=(bsz, num_s),
      in_specs=[
          pl.BlockSpec((1, bs, num_h * dim), lambda b, s: (b, s, 0)),
          pl.BlockSpec((8, dim), lambda b, s: (0, 0)),
          None,
          None,
          None,
      ],
      out_specs=pl.BlockSpec((1, num_h, bs, dim), lambda b, s: (b, 0, s, 0)),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      name="pallas_qk_prologue_fwd",
  )(x_packed, scale_2d, None, None, None)


def _run_pallas_bwd_local(
    x_loc: jax.Array,
    dy_loc: jax.Array,
    effective_scale: jax.Array,
    cos_loc: jax.Array | None,
    sin_loc: jax.Array | None,
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
) -> tuple[jax.Array, jax.Array]:
  """Runs local Pallas QK prologue backward (local `dscale`, no `psum`)."""
  bsz, seq_len, num_h, dim = x_loc.shape
  cos_loc = _normalize_rope_table(cos_loc, bsz)
  sin_loc = _normalize_rope_table(sin_loc, bsz)
  bs = block_s if block_s > 0 else select_prologue_block_s(seq_len, num_h)
  num_s = seq_len // bs
  x_packed = x_loc.reshape(bsz, seq_len, num_h * dim)
  scale_2d = jnp.broadcast_to(effective_scale.astype(jnp.float32)[None, :], (8, dim))
  use_rope = cos_loc is not None and sin_loc is not None
  if use_rope:
    rot_mat_t = jnp.asarray(_make_rot_half_matrix(dim, jnp.bfloat16).T)
    dx_packed, dscale_parts = pl.pallas_call(
        functools.partial(
            _pallas_prologue_bwd_kernel,
            num_h=num_h,
            dim=dim,
            epsilon=epsilon,
            pre_attn_scalar=pre_attn_scalar,
            use_rope=True,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((bsz, seq_len, num_h * dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((bsz * num_s, 8, dim), jnp.float32),
        ),
        grid=(bsz, num_s),
        in_specs=[
            pl.BlockSpec((1, bs, num_h * dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, num_h, bs, dim), lambda b, s: (b, 0, s, 0)),
            pl.BlockSpec((8, dim), lambda b, s: (0, 0)),
            pl.BlockSpec((1, bs, dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, bs, dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((dim, dim), lambda b, s: (0, 0)),
        ],
        out_specs=(
            pl.BlockSpec((1, bs, num_h * dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, 8, dim), lambda b, s: (b * num_s + s, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        name="pallas_qk_prologue_bwd",
    )(x_packed, dy_loc, scale_2d, cos_loc, sin_loc, rot_mat_t)
  else:
    dx_packed, dscale_parts = pl.pallas_call(
        functools.partial(
            _pallas_prologue_bwd_kernel,
            num_h=num_h,
            dim=dim,
            epsilon=epsilon,
            pre_attn_scalar=pre_attn_scalar,
            use_rope=False,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((bsz, seq_len, num_h * dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((bsz * num_s, 8, dim), jnp.float32),
        ),
        grid=(bsz, num_s),
        in_specs=[
            pl.BlockSpec((1, bs, num_h * dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, num_h, bs, dim), lambda b, s: (b, 0, s, 0)),
            pl.BlockSpec((8, dim), lambda b, s: (0, 0)),
            None,
            None,
            None,
        ],
        out_specs=(
            pl.BlockSpec((1, bs, num_h * dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, 8, dim), lambda b, s: (b * num_s + s, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        name="pallas_qk_prologue_bwd",
    )(x_packed, dy_loc, scale_2d, None, None, None)

  dscale = jnp.sum(dscale_parts, axis=(0, 1)).astype(effective_scale.dtype)
  return dx_packed.reshape(bsz, seq_len, num_h, dim), dscale


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _fused_qk_prologue_nope_local(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
) -> jax.Array:
  """Runs fused Pallas QK prologue without RoPE."""
  return _run_pallas_fwd_local(x_loc, effective_scale, None, None, epsilon, pre_attn_scalar, block_s)


def _fused_qk_prologue_nope_local_fwd(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Forward rule for `_fused_qk_prologue_nope_local`."""
  y = _run_pallas_fwd_local(x_loc, effective_scale, None, None, epsilon, pre_attn_scalar, block_s)
  return y, (x_loc, effective_scale)


def _fused_qk_prologue_nope_local_bwd(
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
    residuals: tuple[jax.Array, jax.Array],
    dy_bhsd: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Backward rule for `_fused_qk_prologue_nope_local`."""
  x_loc, effective_scale = residuals
  return _run_pallas_bwd_local(
      x_loc,
      dy_bhsd,
      effective_scale,
      None,
      None,
      epsilon,
      pre_attn_scalar,
      block_s,
  )


_fused_qk_prologue_nope_local.defvjp(_fused_qk_prologue_nope_local_fwd, _fused_qk_prologue_nope_local_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _fused_qk_prologue_rope_local(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    cos_2d: jax.Array,
    sin_2d: jax.Array,
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
) -> jax.Array:
  """Runs fused Pallas QK prologue with RoPE."""
  return _run_pallas_fwd_local(
      x_loc,
      effective_scale,
      cos_2d,
      sin_2d,
      epsilon,
      pre_attn_scalar,
      block_s,
  )


def _fused_qk_prologue_rope_local_fwd(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    cos_2d: jax.Array,
    sin_2d: jax.Array,
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
  """Forward rule for `_fused_qk_prologue_rope_local`."""
  y = _run_pallas_fwd_local(
      x_loc,
      effective_scale,
      cos_2d,
      sin_2d,
      epsilon,
      pre_attn_scalar,
      block_s,
  )
  return y, (x_loc, effective_scale, cos_2d, sin_2d)


def _fused_qk_prologue_rope_local_bwd(
    epsilon: float,
    pre_attn_scalar: float | None,
    block_s: int,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    dy_bhsd: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Backward rule for `_fused_qk_prologue_rope_local`."""
  x_loc, effective_scale, cos_2d, sin_2d = residuals
  dx, ds = _run_pallas_bwd_local(
      x_loc,
      dy_bhsd,
      effective_scale,
      cos_2d,
      sin_2d,
      epsilon,
      pre_attn_scalar,
      block_s,
  )
  return dx, ds, jnp.zeros_like(cos_2d), jnp.zeros_like(sin_2d)


_fused_qk_prologue_rope_local.defvjp(_fused_qk_prologue_rope_local_fwd, _fused_qk_prologue_rope_local_bwd)


def _ref_qk_prologue_local(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    *,
    epsilon: float,
    pre_attn_scalar: float | None,
    cos_2d: jax.Array | None,
    sin_2d: jax.Array | None,
) -> jax.Array:
  """Pure-JAX reference fallback matching `fused_qk_prologue_local`."""
  cos_3d = _normalize_rope_table(cos_2d, x_loc.shape[0])
  sin_3d = _normalize_rope_table(sin_2d, x_loc.shape[0])
  x_f32 = x_loc.astype(jnp.float32)
  rstd = lax.rsqrt(jnp.mean(lax.square(x_f32), axis=-1, keepdims=True) + np.float32(epsilon))
  y = (x_f32 * rstd * effective_scale.astype(jnp.float32)).astype(x_loc.dtype)
  if cos_3d is not None and sin_3d is not None:
    half = y.shape[-1] // 2
    y_rot = jnp.concatenate([-y[..., half:], y[..., :half]], axis=-1)
    cos_4d = cos_3d[:, :, None, :].astype(jnp.float32)
    sin_4d = sin_3d[:, :, None, :].astype(jnp.float32)
    y = (y.astype(jnp.float32) * cos_4d + y_rot.astype(jnp.float32) * sin_4d).astype(x_loc.dtype)
  if pre_attn_scalar is not None and pre_attn_scalar != 1.0:
    y = (y.astype(jnp.float32) * np.float32(pre_attn_scalar)).astype(x_loc.dtype)
  return jnp.swapaxes(y, 1, 2)


def fused_qk_prologue_local(
    x_loc: jax.Array,
    effective_scale: jax.Array,
    *,
    epsilon: float,
    pre_attn_scalar: float | None = None,
    cos_2d: jax.Array | None = None,
    sin_2d: jax.Array | None = None,
    block_s: int = 0,
) -> jax.Array:
  """Applies local FP32 RMSNorm (+ RoPE) (+ scalar) + `BSHD -> BHSD`.

  Runs as a single Pallas TPU VMEM kernel when `can_use_fused_qk_prologue` is
  True, and falls back to the pure-JAX reference on CPU or unaligned shapes.

  Args:
    x_loc: Local `[B_loc, S_loc, H, D]` query or key projection tensor.
    effective_scale: RMSNorm scale vector `[D]`.
    epsilon: RMSNorm epsilon.
    pre_attn_scalar: Optional query pre-attention scalar (`head_dim ** -0.5`).
    cos_2d: Optional RoPE cosine table `[B_loc, S_loc, D]` or `[S_loc, D]`.
    sin_2d: Optional RoPE sine table `[B_loc, S_loc, D]` or `[S_loc, D]`.
    block_s: Optional sequence tile size (`0` = auto-select).

  Returns:
    Normalized (and optionally rotated/scaled) `[B_loc, H, S_loc, D]` tensor.
  """
  cos_3d = _normalize_rope_table(cos_2d, x_loc.shape[0])
  sin_3d = _normalize_rope_table(sin_2d, x_loc.shape[0])
  if not can_use_fused_qk_prologue(x_loc, effective_scale, cos_3d, sin_3d, block_s):
    if pltpu.is_tpu_device():
      raise ValueError(
          "use_fused_qk_prologue=True failed TPU preconditions for"
          f" x_loc={x_loc.shape} ({x_loc.dtype}),"
          f" scale={effective_scale.shape},"
          f" cos={None if cos_3d is None else cos_3d.shape}, block_s={block_s}."
      )
    return _ref_qk_prologue_local(
        x_loc,
        effective_scale,
        epsilon=epsilon,
        pre_attn_scalar=pre_attn_scalar,
        cos_2d=cos_3d,
        sin_2d=sin_3d,
    )
  if cos_3d is not None and sin_3d is not None:
    return _fused_qk_prologue_rope_local(
        x_loc,
        effective_scale,
        cos_3d,
        sin_3d,
        epsilon,
        pre_attn_scalar,
        block_s,
    )
  return _fused_qk_prologue_nope_local(x_loc, effective_scale, epsilon, pre_attn_scalar, block_s)


def load_balanced_chunk_order(cp_size: int) -> tuple[int, ...]:
  """Returns chunk IDs in all-gathered LB slots `[0, n-1, ...]`.

  Args:
    cp_size: Size of the context parallelism group.

  Returns:
    Tuple of length `2 * cp_size` with interleaved natural chunk indices.
  """
  num_chunks = 2 * cp_size
  order: list[int] = []
  for r in range(cp_size):
    order += [r, num_chunks - 1 - r]
  return tuple(order)


def _source_slots(cp_size: int, to_contiguous: bool) -> tuple[int, ...]:
  """Returns the input chunk slot copied into each output chunk slot."""
  lb_order = load_balanced_chunk_order(cp_size)
  if to_contiguous:
    return tuple(lb_order.index(c) for c in range(2 * cp_size))
  return lb_order


def _vmem_copy_kernel(x_ref: Any, o_ref: Any) -> None:
  """Copies a VMEM sequence tile directly from input to output."""
  o_ref[...] = x_ref[...]


def _ref_permute_lb_chunks(
    x: jax.Array,
    cp_size: int,
    to_contiguous: bool,
) -> jax.Array:
  """Pure-JAX `[B, H, S, D]` chunk permutation fallback."""
  bsz, heads, seq_len, dim = x.shape
  num_chunks = 2 * cp_size
  group = seq_len // num_chunks
  slots = _source_slots(cp_size, to_contiguous)
  x_chunks = x.reshape(bsz, heads, num_chunks, group, dim)
  parts = [x_chunks[:, :, s : s + 1, :, :] for s in slots]
  return jnp.concatenate(parts, axis=2).reshape(bsz, heads, seq_len, dim)


def _permute_chunks_pallas(
    x: jax.Array,
    cp_size: int,
    to_contiguous: bool,
    tile_s: int,
) -> jax.Array:
  """Permutes the `2 * cp_size` sequence chunks of `[B, H, S, D]` via Pallas."""
  bsz, heads, seq_len, dim = x.shape
  num_chunks = 2 * cp_size
  group = seq_len // num_chunks
  tile = min(tile_s, group)
  tiles_per_chunk = group // tile

  def in_index_map(
      b: jax.Array,
      h: jax.Array,
      out_chunk: jax.Array,
      t: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    if to_contiguous:
      in_chunk = jnp.where(
          out_chunk < cp_size,
          2 * out_chunk,
          2 * (num_chunks - 1 - out_chunk) + 1,
      )
    else:
      pair = lax.div(out_chunk, 2)
      is_odd = lax.rem(out_chunk, 2) != 0
      in_chunk = jnp.where(is_odd, (num_chunks - 1) - pair, pair)
    return (b, h, in_chunk * tiles_per_chunk + t, 0)

  return pl.pallas_call(
      _vmem_copy_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid=(bsz, heads, num_chunks, tiles_per_chunk),
      in_specs=[pl.BlockSpec((1, 1, tile, dim), in_index_map)],
      out_specs=pl.BlockSpec(
          (1, 1, tile, dim),
          lambda b, h, c, t: (b, h, c * tiles_per_chunk + t, 0),
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "parallel")),
      name="lb_kv_vmem_copy",
  )(x)


def supports_pallas_lb_reorder(
    x_shape: tuple[int, ...],
    cp_size: int,
    tile_s: int = 4096,
) -> bool:
  """Returns True if `x_shape` (`[B, H, S, D]`) supports Pallas LB reorder.

  Args:
    x_shape: Input tensor shape `[B, H, S, D]`.
    cp_size: Size of the context parallelism group.
    tile_s: Preferred sequence tile size.

  Returns:
    True if running on TPU and dimensions align with VMEM chunk tiling.
  """
  if not pltpu.is_tpu_device():
    return False
  if len(x_shape) != 4 or cp_size < 2:
    return False
  seq_len = x_shape[2]
  num_chunks = 2 * cp_size
  if seq_len % num_chunks != 0:
    return False
  group = seq_len // num_chunks
  tile = min(tile_s, group)
  return tile >= 16 and tile % 16 == 0 and group % tile == 0 and x_shape[3] % 128 == 0


def _permute_chunks_dispatch(
    x: jax.Array,
    cp_size: int,
    to_contiguous: bool,
    tile_s: int,
) -> jax.Array:
  """Dispatches `[B, H, S, D]` chunk permutation to Pallas VMEM or JAX."""
  if cp_size <= 1:
    return x
  if supports_pallas_lb_reorder(x.shape, cp_size, tile_s):
    return _permute_chunks_pallas(x, cp_size, to_contiguous, tile_s)
  if pltpu.is_tpu_device():
    raise ValueError(
        "use_pallas_lb_kv_reorder=True failed TPU preconditions for" f" x={x.shape}, cp_size={cp_size}, tile_s={tile_s}."
    )
  return _ref_permute_lb_chunks(x, cp_size, to_contiguous)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def reorder_lb_kv_to_contiguous(
    x: jax.Array,
    cp_size: int,
    tile_s: int = 4096,
) -> jax.Array:
  """Permutes all-gathered `[B, H, S, D]` K/V from LB order to natural order.

  Args:
    x: All-gathered `[B, H, S, D]` key or value tensor in LB chunk order.
    cp_size: Size of the context parallelism group.
    tile_s: Preferred sequence tile size for Pallas VMEM copies.

  Returns:
    Reordered `[B, H, S, D]` tensor in contiguous natural sequence order.
  """
  return _permute_chunks_dispatch(x, cp_size, True, tile_s)


def _reorder_fwd(
    x: jax.Array,
    cp_size: int,
    tile_s: int,
) -> tuple[jax.Array, None]:
  """Forward rule for `reorder_lb_kv_to_contiguous`."""
  return _permute_chunks_dispatch(x, cp_size, True, tile_s), None


def _reorder_bwd(
    cp_size: int,
    tile_s: int,
    residuals: None,
    grad_output: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  """Backward rule for `reorder_lb_kv_to_contiguous`."""
  del residuals
  return (_permute_chunks_dispatch(grad_output, cp_size, False, tile_s),)


reorder_lb_kv_to_contiguous.defvjp(_reorder_fwd, _reorder_bwd)
