# Copyright 2026 Google LLC
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

"""Fused TensorCore (Pallas) per-head RMSNorm + RoPE + `[B, S, N, H] -> [B, N, S, H]` relayout.

On TPU the q/k preparation chain in front of splash attention is three memory-bound
XLA passes over the projected activations (`[B, S, N, H]`, bf16): the per-head
`RMSNorm` (`normalizations.RMSNorm`), the rotary embedding (`embeddings.RotaryEmbedding`
and its Gemma 4 / partial variants) and the `jnp.transpose(x, (0, 2, 1, 3))` that
`attention_op.tpu_flash_attention` applies before the splash kernel. Each pass reads and
writes the whole tensor once. This module fuses the three into one Pallas kernel that
reads `x` once and writes the normed + roped result once, directly in `[B, N, S, H]`.

Semantics (per row `x[b, s, n, :]`, `H` lanes; all math in f32 unless noted):

  * RMSNorm (`scale_offset == 0`, the Gemma family):
      `yb = cast(x * rsqrt(mean(x^2) + eps), x.dtype)`; `z = yb * cast(scale, x.dtype)`
    RMSNorm (`scale_offset != 0`):
      `z = cast(x * rsqrt(mean(x^2) + eps) * (scale + scale_offset), x.dtype)`
  * RoPE in rotate-half form on the first `R` lanes (`R <= H`, `R` even), `R/2` angles:
      `y[:R/2] = z[:R/2] * cos - z[R/2:R] * sin`
      `y[R/2:R] = z[R/2:R] * cos + z[:R/2] * sin`
      `y[R:] = z[R:]`
    where `sin, cos = sin/cos(position / timescale)` are precomputed `[B, S, R/2]` tables in
    `x.dtype` (see `rope_tables`), exactly as `RotaryEmbedding.__call__` does. The reference
    multiplies in `x.dtype`; the kernel multiplies in f32 and rounds once, so results can
    differ from the XLA reference by a bf16 ulp.
  * `y = y * pre_attn_scalar` (optional `query_pre_attn_scalar`), cast to `out_dtype`.
  * Output layout `[B, N, S, H]` (the layout splash attention consumes).

Layouts: the kernel consumes `x` as `jnp.transpose(x, (0, 2, 1, 3))`, i.e. `[B, N, S, H]`. On TPU
the q/k projection `dot_general(inputs[B,S,D], w[D,N,H])` is emitted with a layout XLA picks from
the consumer; its unconstrained choice at Gemma 4 dims is `{2,0,1}` = physical `[N][S][H]`, so
the transpose into the kernel is a bitcast and no relayout copy of the activation is needed
(a `[B, S, N, H]` row-major operand, tiled over `(N, H)`, forces one; so does `[B*S, N*H]`, which
makes XLA relayout the weight instead). Inside the kernel every head is a contiguous
`[S_blk, H]` slab of the `[1, N, S_blk, H]` block (lane-sliced `[S_blk, N*H]` loads are 2x
slower in Mosaic), and `dx` is written back in the same `[B, N, S, H]` form.

The custom VJP computes `dx` (`[B, S, N, H]`, `x.dtype`) and `dscale` (`[H]`, f32) in one
Pallas kernel that reads `x` and `dy` (`[B, N, S, H]`) once each: RoPE is orthogonal, so its
adjoint is the same rotate-half with `sin` negated, followed by the standard RMSNorm
backward `dx = r * (dn - n * mean(dn * n))` with `dscale` accumulated in a grid-resident f32
`[1, H]` block.

Only `jax`, `jax.numpy`, `pallas`, `pltpu` are imported so the module is cheap to import
and runs under the Pallas interpreter on CPU.
"""

# pylint: disable=too-many-positional-arguments

import functools
from typing import NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Target bytes of one input block ([N, S_blk, H] in x.dtype). Blocks are double buffered for
# x and the output (and dy in the backward), plus f32 temporaries per head-slice.
_TARGET_BLOCK_BYTES = 4 << 20
_VMEM_LIMIT_BYTES = 96 << 20
# Rows of one head-slice processed per inner iteration ([rows, H] f32 temporaries).
_ROWS_PER_ITER = 128


class _Config(NamedTuple):
  """Static kernel configuration (the non-differentiable custom_vjp argument)."""

  batch: int
  seq: int
  heads: int
  head_dim: int
  rope_dim: int
  block_s: int
  eps: float
  scale_offset: float
  pre_attn_scalar: float
  x_dtype: jnp.dtype
  out_dtype: jnp.dtype
  interpret: bool


# ----------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------


def rope_tables(positions, timescale, dtype):
  """`sin_half, cos_half: [B, S, R/2]` for rotate-half RoPE, matching `RotaryEmbedding.__call__`.

  Args:
    positions: `[B, S]` integer positions.
    timescale: `[R/2]` f32 timescales (`RotaryEmbedding.timescale`; may contain `inf` for the
      Gemma 4 pass-through angles, which yields `sin = 0, cos = 1`).
    dtype: dtype of the activations the tables multiply (the reference casts the tables to
      `inputs.dtype` before the multiply).
  """
  sinusoid = positions[:, :, None] / timescale[None, None, :]
  return jnp.sin(sinusoid).astype(dtype), jnp.cos(sinusoid).astype(dtype)


def _full_tables(sin_half, cos_half, head_dim):
  """Expands `[B, S, R/2]` half tables to `[B, S, H]` `cos` and sign-folded `sin` tables.

  `sin_full = [-sin_half, sin_half, 0...]`, `cos_full = [cos_half, cos_half, 1...]`, so that
  `y = z * cos_full + rot(z) * sin_full` with `rot(z) = [z[R/2:R], z[:R/2], *]`.
  """
  b, s, half = sin_half.shape
  tail = head_dim - 2 * half
  zeros = jnp.zeros((b, s, tail), sin_half.dtype)
  ones = jnp.ones((b, s, tail), cos_half.dtype)
  sin_full = jnp.concatenate([-sin_half, sin_half, zeros], axis=-1)
  cos_full = jnp.concatenate([cos_half, cos_half, ones], axis=-1)
  return sin_full, cos_full


def _effective_scale(scale, cfg: _Config):
  """`[1, H]` f32 multiplier applied to the normalized rows (see module docstring)."""
  scale = scale.astype(jnp.float32) + cfg.scale_offset
  if cfg.scale_offset == 0.0:
    scale = scale.astype(cfg.x_dtype).astype(jnp.float32)
  return scale.reshape(1, cfg.head_dim)


# ----------------------------------------------------------------------------------------
# In-kernel helpers
# ----------------------------------------------------------------------------------------


def _rotate_half(z, cfg: _Config):
  """`rot(z)[i] = z[i + R/2]` for `i < R/2`, `z[i - R/2]` for `R/2 <= i < R`, `0` for lanes `>= R`."""
  h, r = cfg.head_dim, cfg.rope_dim
  if r == h:
    return pltpu.roll(z, h // 2, axis=z.ndim - 1)
  lane = lax.broadcasted_iota(jnp.int32, z.shape, z.ndim - 1)
  lo = pltpu.roll(z, h - r // 2, axis=z.ndim - 1)
  hi = pltpu.roll(z, r // 2, axis=z.ndim - 1)
  return jnp.where(lane < r // 2, lo, jnp.where(lane < r, hi, 0.0))


def _normalize(x, cfg: _Config):
  """Returns `(r, n, yb)`: rsqrt factor `[..., 1]`, `n = x * r` (f32), `yb` = what the scale multiplies."""
  mean2 = jnp.mean(x * x, axis=-1, keepdims=True)
  r = lax.rsqrt(mean2 + cfg.eps)
  n = x * r
  if cfg.scale_offset == 0.0:
    yb = n.astype(cfg.x_dtype).astype(jnp.float32)
  else:
    yb = n
  return r, n, yb


def _fwd_rows(x, g, cos, sin, cfg: _Config):
  """Forward for a `[rows, H]` f32 head-slice (`g: [1, H]`, `cos`/`sin: [rows, H]`); returns `out_dtype` rows."""
  _, _, yb = _normalize(x, cfg)
  z = (yb * g).astype(cfg.x_dtype).astype(jnp.float32)
  y = z * cos + _rotate_half(z, cfg) * sin
  if cfg.pre_attn_scalar != 1.0:
    y = y * cfg.pre_attn_scalar
  return y.astype(cfg.out_dtype)


def _bwd_rows(x, dy, g, cos, sin, cfg: _Config):
  """Backward for a `[rows, H]` f32 head-slice; returns `(dx f32, dz * yb f32)` (the latter sums to dscale)."""
  if cfg.pre_attn_scalar != 1.0:
    dy = dy * cfg.pre_attn_scalar
  dz = dy * cos + _rotate_half(dy * sin, cfg)
  r, n, yb = _normalize(x, cfg)
  dn = dz * g
  dx = r * (dn - n * jnp.mean(dn * n, axis=-1, keepdims=True))
  return dx, dz * yb


def _row_chunks(cfg: _Config):
  rows = min(_ROWS_PER_ITER, cfg.block_s)
  return [(i, rows) for i in range(0, cfg.block_s, rows)]


# ----------------------------------------------------------------------------------------
# Forward kernel
# ----------------------------------------------------------------------------------------


def _fwd_kernel(x_ref, g_ref, cos_ref, sin_ref, o_ref, *, cfg: _Config):
  """One `[1, N, S_blk, H]` tile of the forward, head by head."""
  # x_ref: [1, N, S_blk, H] x.dtype; o_ref: [1, N, S_blk, H] out_dtype; tables [1, S_blk, H].
  g = g_ref[...]  # [1, H]
  for i0, rows in _row_chunks(cfg):
    rs = pl.ds(i0, rows)
    cos = cos_ref[0, rs, :].astype(jnp.float32)
    sin = sin_ref[0, rs, :].astype(jnp.float32)
    for n in range(cfg.heads):
      o_ref[0, n, rs, :] = _fwd_rows(x_ref[0, n, rs, :].astype(jnp.float32), g, cos, sin, cfg)


def _fwd_call(x_bnsh, g, sin_full, cos_full, cfg: _Config):
  b, s, n, h = cfg.batch, cfg.seq, cfg.heads, cfg.head_dim
  grid = (b, pl.cdiv(s, cfg.block_s))
  block = pl.BlockSpec((1, n, cfg.block_s, h), lambda bi, si: (bi, 0, si, 0))
  table = pl.BlockSpec((1, cfg.block_s, h), lambda bi, si: (bi, si, 0))
  return pl.pallas_call(
      functools.partial(_fwd_kernel, cfg=cfg),
      grid=grid,
      in_specs=[block, pl.BlockSpec((1, h), lambda bi, si: (0, 0)), table, table],
      out_specs=block,
      # Under shard_map(check_vma=True) the output varies over the same mesh axes as `x`.
      out_shape=jax.ShapeDtypeStruct((b, n, s, h), cfg.out_dtype, manual_axis_type=jax.typeof(x_bnsh).manual_axis_type),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"), vmem_limit_bytes=_VMEM_LIMIT_BYTES
      ),
      interpret=cfg.interpret,
  )(x_bnsh, g, cos_full, sin_full)


# ----------------------------------------------------------------------------------------
# Backward kernel
# ----------------------------------------------------------------------------------------


def _bwd_kernel(x_ref, dy_ref, g_ref, cos_ref, sin_ref, dx_ref, dg_ref, *, cfg: _Config):
  """One `[1, N, S_blk, H]` tile of the backward; `dg_ref` accumulates dscale over the whole grid."""
  # x_ref / dy_ref / dx_ref: [1, N, S_blk, H]; dg_ref: [1, H] f32, grid-resident.
  bi, si = pl.program_id(0), pl.program_id(1)

  @pl.when((bi == 0) & (si == 0))
  def _init():
    dg_ref[...] = jnp.zeros_like(dg_ref)

  g = g_ref[...]  # [1, H]
  partial_tail = cfg.seq % cfg.block_s != 0
  dg_acc = jnp.zeros((1, cfg.head_dim), jnp.float32)
  for i0, rows in _row_chunks(cfg):
    rs = pl.ds(i0, rows)
    cos = cos_ref[0, rs, :].astype(jnp.float32)
    sin = sin_ref[0, rs, :].astype(jnp.float32)
    if partial_tail:
      # Rows past S of the last block hold garbage: keep them out of the dscale sum.
      row = lax.broadcasted_iota(jnp.int32, (rows, 1), 0) + (i0 + si * cfg.block_s)
      valid = row < cfg.seq
    for n in range(cfg.heads):
      x = x_ref[0, n, rs, :].astype(jnp.float32)
      dy = dy_ref[0, n, rs, :].astype(jnp.float32)
      dx, dzy = _bwd_rows(x, dy, g, cos, sin, cfg)
      if partial_tail:
        dzy = jnp.where(valid, dzy, 0.0)
      dx_ref[0, n, rs, :] = dx.astype(cfg.x_dtype)
      dg_acc = dg_acc + jnp.sum(dzy, axis=0, keepdims=True)
  dg_ref[...] += dg_acc


def _bwd_call(x_bnsh, dy, g, sin_full, cos_full, cfg: _Config):
  b, s, n, h = cfg.batch, cfg.seq, cfg.heads, cfg.head_dim
  grid = (b, pl.cdiv(s, cfg.block_s))
  axis_type = jax.typeof(x_bnsh).manual_axis_type
  block = pl.BlockSpec((1, n, cfg.block_s, h), lambda bi, si: (bi, 0, si, 0))
  table = pl.BlockSpec((1, cfg.block_s, h), lambda bi, si: (bi, si, 0))
  return pl.pallas_call(
      functools.partial(_bwd_kernel, cfg=cfg),
      grid=grid,
      in_specs=[block, block, pl.BlockSpec((1, h), lambda bi, si: (0, 0)), table, table],
      out_specs=[block, pl.BlockSpec((1, h), lambda bi, si: (0, 0))],
      # dscale is a per-shard partial sum, so it varies over the same mesh axes as `x`/`dy`; the
      # enclosing shard_map's transpose of the replicated `scale` turns that into a psum.
      out_shape=[
          jax.ShapeDtypeStruct((b, n, s, h), cfg.x_dtype, manual_axis_type=axis_type),
          jax.ShapeDtypeStruct((1, h), jnp.float32, manual_axis_type=axis_type),
      ],
      # The dscale accumulator is resident across the whole grid: no parallel axes.
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=_VMEM_LIMIT_BYTES
      ),
      interpret=cfg.interpret,
  )(x_bnsh, dy, g, cos_full, sin_full)


# ----------------------------------------------------------------------------------------
# custom_vjp + public API
# ----------------------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _fused_vjp(x, scale, sin_half, cos_half, cfg: _Config):
  sin_full, cos_full = _full_tables(sin_half, cos_half, cfg.head_dim)
  # A bitcast when XLA gives the projection its natural [N][S][H] layout (see module docstring).
  x_bnsh = jnp.transpose(x, (0, 2, 1, 3))
  return _fwd_call(x_bnsh, _effective_scale(scale, cfg), sin_full, cos_full, cfg)


def _fused_fwd(x, scale, sin_half, cos_half, cfg: _Config):
  """custom_vjp forward: the primal plus the (small) residuals; `y` itself is not saved."""
  return _fused_vjp(x, scale, sin_half, cos_half, cfg), (x, scale, sin_half, cos_half)


def _varying_axes(a):
  """Mesh axes `a` varies over inside a `shard_map(check_vma=True)` body (empty elsewhere)."""
  mat = jax.typeof(a).manual_axis_type
  return frozenset(getattr(mat, "varying", ()) or ())


def _fused_bwd(cfg: _Config, residuals, dy):
  """custom_vjp backward: `(dx, dscale, None, None)` from `dy: [B, N, S, H]`."""
  x, scale, sin_half, cos_half = residuals
  sin_full, cos_full = _full_tables(sin_half, cos_half, cfg.head_dim)
  x_bnsh = jnp.transpose(x, (0, 2, 1, 3))
  dx_bnsh, dg = _bwd_call(x_bnsh, dy, _effective_scale(scale, cfg), sin_full, cos_full, cfg)
  dx = jnp.transpose(dx_bnsh, (0, 2, 1, 3))
  dscale = dg.reshape(cfg.head_dim).astype(scale.dtype)
  # Inside shard_map the kernel's dscale is a per-shard partial sum; the cotangent of a `scale`
  # that is replicated over some of those mesh axes must be reduced over them (check_vma=True
  # requires the bwd to return the primal's type; with check_vma=False shard_map's transpose
  # of the replicated input inserts the psum itself).
  reduce_axes = _varying_axes(dscale) - _varying_axes(scale)
  if reduce_axes:
    dscale = lax.psum(dscale, tuple(sorted(reduce_axes)))
  return dx, dscale, None, None


_fused_vjp.defvjp(_fused_fwd, _fused_bwd)


def default_block_s(seq, heads, head_dim, itemsize):
  """Largest row block (multiple of 16) with an `[N, S_blk, H]` input block of about `_TARGET_BLOCK_BYTES`."""
  rows = max(16, _TARGET_BLOCK_BYTES // (heads * head_dim * itemsize))
  rows = min(rows - rows % 16, seq)
  if rows >= seq:
    return seq
  # Prefer a divisor of `seq` so no partial tail block is needed.
  for cand in range(rows, 15, -16):
    if seq % cand == 0:
      return cand
  return rows


def fused_rmsnorm_rope(
    x,
    scale,
    sin_half,
    cos_half,
    *,
    eps,
    scale_offset=0.0,
    out_dtype=None,
    pre_attn_scalar=1.0,
    block_s=None,
    interpret=False,
):
  """Per-head RMSNorm + rotate-half RoPE + relayout, fused.

  Args:
    x: `[B, S, N, H]` activations (bf16 or f32).
    scale: `[H]` RMSNorm scale parameter (any float dtype; used as f32).
    sin_half, cos_half: `[B, S, R/2]` tables from `rope_tables` (`R <= H`, `R` even).
    eps: RMSNorm epsilon.
    scale_offset: RMSNorm `scale_offset` (see `normalizations.RMSNorm`).
    out_dtype: output dtype (defaults to `x.dtype`).
    pre_attn_scalar: optional multiplier applied after RoPE (`query_pre_attn_scalar`).
    block_s: rows of `S` per grid step (multiple of 16 or `== S`); auto if `None`.
    interpret: run under the Pallas interpreter (CPU tests).

  Returns:
    `[B, N, S, H]` array in `out_dtype`.
  """
  if x.ndim != 4:
    raise ValueError(f"x must be [B, S, N, H], got {x.shape}")
  b, s, n, h = x.shape
  if sin_half.shape != cos_half.shape or sin_half.shape[:2] != (b, s):
    raise ValueError(f"tables must be [B, S, R/2]; got {sin_half.shape} / {cos_half.shape} for x {x.shape}")
  r = 2 * sin_half.shape[-1]
  if r > h:
    raise ValueError(f"rope dims {r} exceed head_dim {h}")
  if h % 128 != 0:
    raise ValueError(f"head_dim must be a multiple of 128, got {h}")
  if scale.shape != (h,):
    raise ValueError(f"scale must be [H]={h}, got {scale.shape}")
  out_dtype = jnp.dtype(x.dtype if out_dtype is None else out_dtype)
  if block_s is None:
    block_s = default_block_s(s, n, h, jnp.dtype(x.dtype).itemsize)
  if block_s != s and block_s % 16 != 0:
    raise ValueError(f"block_s must be a multiple of 16 or == S, got {block_s}")
  cfg = _Config(
      batch=b,
      seq=s,
      heads=n,
      head_dim=h,
      rope_dim=r,
      block_s=block_s,
      eps=float(eps),
      scale_offset=float(scale_offset),
      pre_attn_scalar=float(pre_attn_scalar),
      x_dtype=jnp.dtype(x.dtype),
      out_dtype=out_dtype,
      interpret=bool(interpret),
  )
  return _fused_vjp(x, scale, sin_half, cos_half, cfg)


def reference_rmsnorm_rope(x, scale, sin_half, cos_half, *, eps, scale_offset=0.0, out_dtype=None, pre_attn_scalar=1.0):
  """Pure-JAX reference with the exact op order of `RMSNorm` + `RotaryEmbedding`; returns `[B, N, S, H]`."""
  dtype = x.dtype
  xf = x.astype(jnp.float32)
  mean2 = jnp.mean(lax.square(xf), axis=-1, keepdims=True)
  y = (xf * lax.rsqrt(mean2 + eps)).astype(dtype)
  effective_scale = scale.astype(jnp.float32) + scale_offset
  if scale_offset != 0.0:
    y = ((xf * lax.rsqrt(mean2 + eps)) * effective_scale).astype(dtype)
  else:
    y = y * effective_scale.astype(dtype)
  r = 2 * sin_half.shape[-1]
  rot, tail = y[..., :r], y[..., r:]
  sin = jnp.concatenate([sin_half, sin_half], axis=-1)[:, :, None, :]
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)[:, :, None, :]
  x1, x2 = jnp.split(rot, 2, axis=-1)
  rot = rot * cos + jnp.concatenate((-x2, x1), axis=-1) * sin
  y = jnp.concatenate([rot, tail], axis=-1).astype(dtype)
  if pre_attn_scalar != 1.0:
    y = y * pre_attn_scalar
  return jnp.transpose(y, (0, 2, 1, 3)).astype(x.dtype if out_dtype is None else out_dtype)
