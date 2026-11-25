# Copyright 2026 Google LLC
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

"""VMEM-fused GQA-grouped Pallas Mosaic TPU FlashAttention kernel for Qwen Sparse Attention (QSA).

Eliminates multi-TiB HBM score spills on long contexts (`>= 64K`) while fusing all
`g = h_q // h_kv` GQA query heads per KV head (`g = 12` in Qwen3.8-Flash-Next) inside
each VMEM tile (`gbq = g * bq`) and fusing `(b_loc * h_kv)` into the parallel outer grid:
  * Cuts `K`/`V` HBM DMA traffic and 32-bit block-bitmap VPU unpacking by `g` (`12x`).
  * Expands MXU matmul M/K dimensions from `bq` (`128`/`256`) to `g * bq` (`1,536`)
    and widens KV tiles to `bkv = 512` (with `512`-row VMEM compute sub-blocking for
    `bkv_dkv in {1024, 2048}`), eliminating 50% systolic waste on `256x256` MXUs.
  * Uses base-2 hardware EUP `exp2` (`scale * log2(e)`, single `where` mask) and
    stationary-outer 3D grids `(b_loc * h_kv, num_q_tiles, max_act_q)` /
    `(b_loc * h_kv, num_kv_tiles, max_act_kv)` with clamped-tail SMEM index tables.
"""

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

NUM_SUBLANES = 8
NUM_LANES = 128
_BKV = 512
_BKV_DKV_DEFAULT = 512
_BKV_DKV_SUB = 512
_BKV_CANDIDATES = (2048, 1024, 512, 128)
_VMEM_LIMIT_BYTES = 32 * 1024 * 1024
_NT_DIMS = (((1,), (1,)), ((), ()))
_NN_DIMS = (((1,), (0,)), ((), ()))
_TN_DIMS = (((0,), (0,)), ((), ()))
_MASK_NEG_INF = -1e30
_M_INIT_LOG2 = -1e28
_LOG2_E = 1.4426950408889634
_LN_2 = 0.6931471805599453


def _align_up(val: int, align: int) -> int:
  return ((val + align - 1) // align) * align


def choose_bkv(s_kv: int, preferred: int = _BKV) -> int:
  """Selects the largest supported KV tile `<= preferred` (`2048/1024/512/128`) dividing `s_kv`."""
  target = _BKV if preferred <= 0 else preferred
  for cand in _BKV_CANDIDATES:
    if cand <= target and s_kv >= cand and s_kv % cand == 0:
      return cand
  return 128


def _choose_bq(s_q_loc: int, g: int) -> int:
  return 128 if (g >= 2 or s_q_loc % 256 != 0) else 256


def _dot_tile(lhs: jax.Array, rhs: jax.Array, dims: Any) -> jax.Array:
  """Computes MXU `dot_general` accumulating into `float32`."""
  return lax.dot_general(lhs, rhs, dims, preferred_element_type=jnp.float32)


def _group_q_heads_batched(x: jax.Array, *, h_kv: int, bq: int) -> jax.Array:
  """Packs `(b_loc, s_q, h_q, d)` into `(b_loc, h_kv, num_q_tiles, g * bq, d)`."""
  b_loc, s_q, h_q, d = x.shape
  num_q_tiles = s_q // bq
  g = h_q // h_kv
  return (
      x.reshape(b_loc, num_q_tiles, bq, h_kv, g, d)
      .transpose(0, 3, 1, 4, 2, 5)
      .reshape(b_loc, h_kv, num_q_tiles, g * bq, d)
  )


def _ungroup_q_heads_batched(x_grp: jax.Array, *, g: int, bq: int) -> jax.Array:
  """Unpacks `(b_loc, h_kv, num_q_tiles, g * bq, d)` back into `(b_loc, s_q, h_q, d)`."""
  b_loc, h_kv, num_q_tiles, _, d = x_grp.shape
  return (
      x_grp.reshape(b_loc, h_kv, num_q_tiles, g, bq, d)
      .transpose(0, 2, 4, 1, 3, 5)
      .reshape(b_loc, num_q_tiles * bq, h_kv * g, d)
  )


def _active_tile_matrix(
    pmask_b: jax.Array,
    q_token_offset: jax.Array,
    *,
    bq: int,
    bkv: int,
    num_kv_tiles: int,
) -> jax.Array:
  """Computes `(num_q_tiles, num_kv_tiles)` boolean active-tile map for tile width `bkv`."""
  s_q_loc = pmask_b.shape[0]
  num_q_tiles = s_q_loc // bq
  w_per_tile = bkv // 128
  num_words = num_kv_tiles * w_per_tile
  pmask_4d = pmask_b[:, :num_words].reshape(num_q_tiles, bq, num_kv_tiles, w_per_tile)
  tile_or = jnp.bitwise_or.reduce(pmask_4d, axis=(1, 3))

  q_off_arr = jnp.asarray(q_token_offset, dtype=jnp.int32)
  if q_off_arr.ndim == 0:
    q_tok_lo = (q_off_arr + jnp.arange(num_q_tiles, dtype=jnp.int32) * bq)[:, None]
  else:
    q_tok_lo = q_off_arr.reshape(num_q_tiles, 1)
  q_tok_hi = q_tok_lo + (bq - 1)
  q_kv_lo = lax.div(q_tok_lo, jnp.int32(bkv))
  q_kv_hi = lax.div(q_tok_hi, jnp.int32(bkv))
  kv_id = jnp.arange(num_kv_tiles, dtype=jnp.int32)[None, :]
  return (kv_id <= q_kv_hi) & ((kv_id >= q_kv_lo) | (tile_or != jnp.uint32(0)))


def _build_q_schedule(
    pmask_b: jax.Array,
    q_token_offset: jax.Array,
    *,
    bq: int,
    bkv: int,
    num_kv_tiles: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Builds compacted Q-major SMEM schedule `(kv_idx_q, counts_q, max_act_q)`."""
  active = _active_tile_matrix(pmask_b, q_token_offset, bq=bq, bkv=bkv, num_kv_tiles=num_kv_tiles)
  counts_q = jnp.sum(active.astype(jnp.int32), axis=1)
  max_act_q = jnp.maximum(jnp.max(counts_q), jnp.int32(1)).reshape(1)
  kv_ord = jnp.argsort(~active, axis=1, stable=True).astype(jnp.int32)
  j_q = jnp.arange(num_kv_tiles, dtype=jnp.int32)[None, :]
  kv_idx_q = jnp.take_along_axis(kv_ord, jnp.minimum(j_q, jnp.maximum(counts_q[:, None] - 1, 0)), axis=1)
  return kv_idx_q, counts_q, max_act_q


def _build_kv_schedule(
    pmask_b: jax.Array,
    q_token_offset: jax.Array,
    *,
    bq: int,
    bkv: int,
    num_kv_tiles: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Builds compacted KV-major SMEM schedule `(q_idx_kv, counts_kv, max_act_kv)`."""
  active_t = _active_tile_matrix(pmask_b, q_token_offset, bq=bq, bkv=bkv, num_kv_tiles=num_kv_tiles).T
  num_q_tiles = active_t.shape[1]
  counts_kv = jnp.sum(active_t.astype(jnp.int32), axis=1)
  max_act_kv = jnp.maximum(jnp.max(counts_kv), jnp.int32(1)).reshape(1)
  q_ord = jnp.argsort(~active_t, axis=1, stable=True).astype(jnp.int32)
  j_kv = jnp.arange(num_q_tiles, dtype=jnp.int32)[None, :]
  q_idx_kv = jnp.take_along_axis(q_ord, jnp.minimum(j_kv, jnp.maximum(counts_kv[:, None] - 1, 0)), axis=1)
  return q_idx_kv, counts_kv, max_act_kv


def _prepare_q_mask(
    pmask_b: jax.Array,
    *,
    bq: int,
    num_words: int,
    bkv: int,
) -> jax.Array:
  """Formats `(s_q_loc, num_words)` bitmap into `int32` `(num_q_tiles, nw_rows, rows, bq)`."""
  s_q_loc = pmask_b.shape[0]
  num_q_tiles = s_q_loc // bq
  rows = max(NUM_SUBLANES, bkv // 128)
  nw_pad = _align_up(num_words, rows)
  trimmed = lax.bitcast_convert_type(pmask_b[:, :num_words], jnp.int32)
  pad_w = nw_pad - trimmed.shape[1]
  pmask_pad = jnp.pad(trimmed, ((0, 0), (0, pad_w))) if pad_w > 0 else trimmed
  return pmask_pad.reshape(num_q_tiles, bq, nw_pad // rows, rows).transpose(0, 2, 3, 1)


def _prepare_kv_mask(
    pmask_b: jax.Array,
    *,
    bq: int,
    num_words: int,
    bkv_dkv: int,
) -> jax.Array:
  """Formats `(s_q_loc, num_words)` bitmap into `int32` `(nw_rows, num_q_tiles, rows, bq)`."""
  s_q_loc = pmask_b.shape[0]
  num_q_tiles = s_q_loc // bq
  rows = max(NUM_SUBLANES, bkv_dkv // 128)
  nw_pad = _align_up(num_words, rows)
  trimmed = lax.bitcast_convert_type(pmask_b[:, :num_words], jnp.int32)
  pad_w = nw_pad - trimmed.shape[1]
  pmask_pad = jnp.pad(trimmed, ((0, 0), (0, pad_w))) if pad_w > 0 else trimmed
  return pmask_pad.T.reshape(nw_pad // rows, rows, num_q_tiles, bq).transpose(0, 2, 1, 3)


def _unpack_kv_major_subtile_mask(
    mask_kv_ref: Any,
    q_tok_start: jax.Array,
    kv_i: jax.Array,
    *,
    bq: int,
    bkv_dkv: int,
    sub_idx: int = 0,
    bkv_sub: int = 512,
) -> jax.Array:
  """Unpacks `(bkv_sub, bq)` transposed `bool` QSA valid mask once per GQA group inside VMEM."""
  if bkv_dkv == 128:
    row_idx = jnp.broadcast_to(kv_i & jnp.int32(7), (NUM_SUBLANES, bq))
    word_kv_8 = jnp.take_along_axis(mask_kv_ref[...], row_idx, axis=0, mode="promise_in_bounds")
    word_kv = jnp.tile(word_kv_8, (128 // NUM_SUBLANES, 1))
  else:
    m_rows = mask_kv_ref[...]
    if bkv_dkv == 512:
      w4 = jnp.where((kv_i & jnp.int32(1)) == 0, m_rows[:4, :], m_rows[4:, :])
    else:
      w4 = m_rows[sub_idx * 4 : (sub_idx + 1) * 4, :]
    word_kv = jnp.concatenate(
        [lax.broadcast_in_dim(w4[w : w + 1, :], (128, bq), (0, 1)) for w in range(4)],
        axis=0,
    )
  k_local = lax.broadcasted_iota(jnp.int32, (bkv_sub, bq), 0)
  bit_shifts_k = lax.shift_right_logical(k_local, jnp.int32(2)) & jnp.int32(31)
  is_sel_t = (lax.shift_right_logical(word_kv, bit_shifts_k) & jnp.int32(1)) != jnp.int32(0)
  q_idx = q_tok_start + lax.broadcasted_iota(jnp.int32, (bkv_sub, bq), 1)
  k_idx = kv_i * jnp.int32(bkv_dkv) + jnp.int32(sub_idx * bkv_sub) + k_local
  in_tail_t = k_idx >= ((q_idx + jnp.int32(1)) & jnp.int32(-4))
  return (k_idx <= q_idx) & (is_sel_t | in_tail_t)


def _expand_q_tile_offs(q_tok_off: jax.Array, *, b_loc: int, num_q_tiles: int, bq: int) -> jax.Array:
  """Normalizes scalar or 1D query offsets into `(b_loc, num_q_tiles)` `int32` tile token offsets."""
  q_off_arr = jnp.asarray(q_tok_off, dtype=jnp.int32)
  if q_off_arr.size == 1:
    offs_1d = q_off_arr.reshape(()) + jnp.arange(num_q_tiles, dtype=jnp.int32) * jnp.int32(bq)
  else:
    offs_1d = q_off_arr.reshape(num_q_tiles)
  return jnp.broadcast_to(offs_1d[None, :], (b_loc, num_q_tiles))


def _batched_q_schedule_and_mask(
    pmask: jax.Array,
    q_offs_b: jax.Array,
    *,
    bq: int,
    bkv: int,
    s_kv: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds batched Q-major SMEM schedules and VMEM bitmap layouts across `b_loc`."""
  num_kv_tiles = s_kv // bkv
  num_words = s_kv // 128
  kv_idx_b, counts_q_b, max_q_b = jax.vmap(
      lambda pm, qo: _build_q_schedule(pm, qo, bq=bq, bkv=bkv, num_kv_tiles=num_kv_tiles)
  )(pmask, q_offs_b)
  max_act_q = jnp.max(max_q_b).reshape(1)
  pmask_q_b = jax.vmap(functools.partial(_prepare_q_mask, bq=bq, num_words=num_words, bkv=bkv))(pmask)
  return kv_idx_b, counts_q_b, max_act_q, pmask_q_b


def _batched_kv_schedule_and_mask(
    pmask: jax.Array,
    q_offs_b: jax.Array,
    *,
    bq: int,
    bkv_dkv: int,
    s_kv: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds batched KV-major SMEM schedules and VMEM bitmap layouts across `b_loc`."""
  num_kv_tiles = s_kv // bkv_dkv
  num_words = s_kv // 128
  q_idx_b, counts_kv_b, max_kv_b = jax.vmap(
      lambda pm, qo: _build_kv_schedule(pm, qo, bq=bq, bkv=bkv_dkv, num_kv_tiles=num_kv_tiles)
  )(pmask, q_offs_b)
  max_act_kv = jnp.max(max_kv_b).reshape(1)
  pmask_kv_b = jax.vmap(functools.partial(_prepare_kv_mask, bq=bq, num_words=num_words, bkv_dkv=bkv_dkv))(pmask)
  return q_idx_b, counts_kv_b, max_act_kv, pmask_kv_b


def _qsa_fwd_batched(
    q_bgrp: jax.Array,
    k_bhkd: jax.Array,
    v_bhkd: jax.Array,
    pmask_q_b: jax.Array,
    kv_idx_b: jax.Array,
    counts_q_b: jax.Array,
    max_act_q: jax.Array,
    q_offs_b: jax.Array,
    *,
    bq: int,
    bkv: int,
    g: int,
    scale: float,
    interpret: Any,
    disable_bounds_checks: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Runs VMEM-fused GQA-grouped QSA forward across `(b_loc * h_kv)` with `axis=0` VPU online softmax."""
  b_loc, h_kv, num_q_tiles, gbq, d = q_bgrp.shape
  bh = b_loc * h_kv
  s_kv = k_bhkd.shape[2]
  bkv_sub = min(bkv, _BKV_DKV_SUB)
  num_subs = bkv // bkv_sub
  w_per_tile = bkv // 128
  mask_rows = max(NUM_SUBLANES, w_per_tile)
  scale_log2 = scale * _LOG2_E

  q_f = q_bgrp.reshape(bh, num_q_tiles, gbq, d)
  k_f = k_bhkd.reshape(bh, s_kv, d)
  v_f = v_bhkd.reshape(bh, s_kv, d)

  def _fwd_kernel(
      kv_idx_ref,
      counts_q_ref,
      max_act_q_ref,
      q_offs_ref,
      q_ref,
      k_ref,
      v_ref,
      mask_q_ref,
      o_ref,
      lse_ref,
      m_scratch_ref,
      l_scratch_ref,
      acc_scratch_ref,
  ):
    bhi = pl.program_id(0)
    q_i = pl.program_id(1)
    j = pl.program_id(2)
    b_i = lax.div(bhi, jnp.int32(h_kv))

    @pl.when(j == 0)
    def _init():
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, _M_INIT_LOG2)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
      acc_scratch_ref[...] = jnp.zeros_like(acc_scratch_ref)

    @pl.when(j < counts_q_ref[b_i, q_i])
    def _step():
      kv_i = kv_idx_ref[b_i, q_i, j]
      q_tok_start = q_offs_ref[b_i, q_i]
      q_val = q_ref[...]
      m_cur = m_scratch_ref[...]
      l_cur = l_scratch_ref[...]
      acc_cur = acc_scratch_ref[...]
      for s_idx in range(num_subs):
        r0 = s_idx * bkv_sub
        r1 = (s_idx + 1) * bkv_sub
        k_sub = k_ref[...] if num_subs == 1 else k_ref[r0:r1, :]
        v_sub = v_ref[...] if num_subs == 1 else v_ref[r0:r1, :]
        valid_1 = _unpack_kv_major_subtile_mask(
            mask_q_ref,
            q_tok_start,
            kv_i,
            bq=bq,
            bkv_dkv=bkv,
            sub_idx=s_idx,
            bkv_sub=bkv_sub,
        )
        valid_t = valid_1 if g == 1 else jnp.tile(valid_1, (1, g))
        scores_t = _dot_tile(k_sub, q_val, _NT_DIMS) * scale_log2
        scores_t = jnp.where(valid_t, scores_t, _MASK_NEG_INF)

        m_sub_row = jnp.max(scores_t, axis=0, keepdims=True)
        m_next_row = jnp.maximum(m_cur[:1, :], m_sub_row)
        m_next = lax.broadcast_in_dim(m_next_row, (NUM_SUBLANES, gbq), (0, 1))
        alpha = jnp.exp2(m_cur - m_next)
        p_t = jnp.exp2(scores_t - m_next_row)
        l_sub = lax.broadcast_in_dim(jnp.sum(p_t, axis=0, keepdims=True), (NUM_SUBLANES, gbq), (0, 1))
        l_cur = alpha * l_cur + l_sub
        m_cur = m_next

        pv_t = _dot_tile(v_sub, p_t.astype(v_sub.dtype), _TN_DIMS)
        acc_cur = alpha[:1, :] * acc_cur + pv_t
      m_scratch_ref[...] = m_cur
      l_scratch_ref[...] = l_cur
      acc_scratch_ref[...] = acc_cur

    @pl.when(j == max_act_q_ref[0] - 1)
    def _write():
      m_fin = m_scratch_ref[...]
      l_fin = l_scratch_ref[...]
      safe_l = jnp.where(l_fin > 0.0, l_fin, 1.0)
      inv_l_row = jnp.where(l_fin[:1, :] > 0.0, 1.0 / safe_l[:1, :], 0.0)
      o_t = (acc_scratch_ref[...] * inv_l_row).astype(o_ref.dtype)
      o_ref[...] = o_t.T
      lse_ref[...] = jnp.where(l_fin > 0.0, m_fin * _LN_2 + jnp.log(safe_l), 0.0)

  def _q_map(bhi, qi, _j, *_):
    return (bhi, qi, 0, 0)

  def _kv_map(bhi, qi, j, kv_r, *_):
    return (bhi, kv_r[lax.div(bhi, jnp.int32(h_kv)), qi, j], 0)

  def _mq_map(bhi, qi, j, kv_r, *_):
    bi = lax.div(bhi, jnp.int32(h_kv))
    return (bi, qi, lax.div(kv_r[bi, qi, j] * jnp.int32(w_per_tile), mask_rows), 0, 0)

  o_out_spec = pl.BlockSpec((None, None, gbq, d), _q_map, pipeline_mode=pl.Buffered(buffer_count=1))
  lse_out_spec = pl.BlockSpec((None, None, NUM_SUBLANES, gbq), _q_map, pipeline_mode=pl.Buffered(buffer_count=1))
  out_shapes = (
      jax.ShapeDtypeStruct((bh, num_q_tiles, gbq, d), q_f.dtype),
      jax.ShapeDtypeStruct((bh, num_q_tiles, NUM_SUBLANES, gbq), jnp.float32),
  )
  o_f, lse_wide = pl.pallas_call(
      _fwd_kernel,
      out_shape=out_shapes,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          grid=(bh, num_q_tiles, max_act_q[0]),
          in_specs=[
              pl.BlockSpec((None, None, gbq, d), _q_map),
              pl.BlockSpec((None, bkv, d), _kv_map),
              pl.BlockSpec((None, bkv, d), _kv_map),
              pl.BlockSpec((None, None, None, mask_rows, bq), _mq_map),
          ],
          out_specs=(
              o_out_spec,
              lse_out_spec,
          ),
          scratch_shapes=[
              pltpu.VMEM((NUM_SUBLANES, gbq), jnp.float32),
              pltpu.VMEM((NUM_SUBLANES, gbq), jnp.float32),
              pltpu.VMEM((d, gbq), jnp.float32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          disable_bounds_checks=disable_bounds_checks,
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      interpret=interpret,
      name=f"qsa_pallas_fwd_gbq{gbq}_bkv{bkv}",
  )(kv_idx_b, counts_q_b, max_act_q, q_offs_b, q_f, k_f, v_f, pmask_q_b)
  o_bgrp = o_f.reshape(b_loc, h_kv, num_q_tiles, gbq, d)
  lse = (
      lse_wide[:, :, 0, :]
      .reshape(b_loc, h_kv, num_q_tiles, g, bq)
      .transpose(0, 1, 3, 2, 4)
      .reshape(b_loc, h_kv * g, num_q_tiles * bq)
  )
  return o_bgrp, lse


def _qsa_bwd_dkv_batched(
    q_bgrp: jax.Array,
    k_bhkd: jax.Array,
    v_bhkd: jax.Array,
    do_bgrp: jax.Array,
    lse_bgrp: jax.Array,
    delta_bgrp: jax.Array,
    pmask_kv_b: jax.Array,
    q_idx_b: jax.Array,
    counts_kv_b: jax.Array,
    max_act_kv: jax.Array,
    q_offs_b: jax.Array,
    *,
    bq: int,
    bkv_dkv: int,
    g: int,
    scale: float,
    interpret: Any,
    disable_bounds_checks: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs fused 5-matmul Splash-style QSA backward `(dQ, dK, dV)`.

  Uses a 3-slot `dq_alias` ring and `512`-row VMEM sub-blocking.
  """
  b_loc, h_kv, num_q_tiles, gbq, d = q_bgrp.shape
  bh = b_loc * h_kv
  s_kv = k_bhkd.shape[2]
  num_kv_tiles = s_kv // bkv_dkv
  bkv_sub = min(bkv_dkv, _BKV_DKV_SUB)
  num_subs = bkv_dkv // bkv_sub
  w_per_tile = bkv_dkv // 128
  mask_rows = max(NUM_SUBLANES, w_per_tile)
  scale_log2 = scale * _LOG2_E
  dq_slots = 3

  q_f = q_bgrp.reshape(bh, num_q_tiles, gbq, d)
  do_f = do_bgrp.reshape(bh, num_q_tiles, gbq, d)
  k_f = k_bhkd.reshape(bh, s_kv, d)
  v_f = v_bhkd.reshape(bh, s_kv, d)
  lse_log2_f = (lse_bgrp * _LOG2_E).reshape(bh, num_q_tiles, gbq)
  delta_f = delta_bgrp.reshape(bh, num_q_tiles, gbq)
  lse_dkv = jnp.broadcast_to(lse_log2_f[:, :, None, :], (bh, num_q_tiles, NUM_SUBLANES, gbq))
  delta_dkv = jnp.broadcast_to(delta_f[:, :, None, :], (bh, num_q_tiles, NUM_SUBLANES, gbq))
  dq_init = jnp.zeros((dq_slots, bh, num_q_tiles, gbq, d), dtype=jnp.float32)

  def _bwd_dkv_kernel(
      q_idx_ref,
      counts_kv_ref,
      max_act_kv_ref,
      q_offs_ref,
      q_ref,
      k_ref,
      v_ref,
      do_ref,
      lse_ref,
      delta_ref,
      mask_kv_ref,
      dq_alias_ref,
      dq_ref,
      dk_ref,
      dv_ref,
      dk_scratch_ref,
      dv_scratch_ref,
  ):
    bhi = pl.program_id(0)
    kv_i = pl.program_id(1)
    j = pl.program_id(2)
    b_i = lax.div(bhi, jnp.int32(h_kv))
    cnt_kv = counts_kv_ref[b_i, kv_i]

    @pl.when(j == 0)
    def _init():
      dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
      dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

    @pl.when(j < cnt_kv)
    def _step():
      q_i = q_idx_ref[b_i, kv_i, j]
      q_tok_start = q_offs_ref[b_i, q_i]
      q_val = q_ref[...]
      do_val = do_ref[...]
      lse_row = lse_ref[:1, :]
      delta_row = delta_ref[:1, :]
      dq_acc = dq_alias_ref[...]
      for s_idx in range(num_subs):
        r0 = s_idx * bkv_sub
        r1 = (s_idx + 1) * bkv_sub
        k_sub = k_ref[...] if num_subs == 1 else k_ref[r0:r1, :]
        v_sub = v_ref[...] if num_subs == 1 else v_ref[r0:r1, :]
        valid_1 = _unpack_kv_major_subtile_mask(
            mask_kv_ref,
            q_tok_start,
            kv_i,
            bq=bq,
            bkv_dkv=bkv_dkv,
            sub_idx=s_idx,
            bkv_sub=bkv_sub,
        )
        valid_t = valid_1 if g == 1 else jnp.tile(valid_1, (1, g))
        scores_t_log2 = _dot_tile(k_sub, q_val, _NT_DIMS) * scale_log2
        p_t = jnp.exp2(jnp.where(valid_t, scores_t_log2 - lse_row, _MASK_NEG_INF))

        dv_step = _dot_tile(p_t.astype(do_ref.dtype), do_val, _NN_DIMS)
        dp_t = _dot_tile(v_sub, do_val, _NT_DIMS)
        ds_t_f32 = p_t * (dp_t - delta_row) * scale
        dk_step = _dot_tile(ds_t_f32.astype(q_ref.dtype), q_val, _NN_DIMS)
        dq_step = _dot_tile(ds_t_f32.astype(k_sub.dtype), k_sub, _TN_DIMS)
        dq_acc = dq_acc + dq_step
        if num_subs == 1:
          dv_scratch_ref[...] = dv_scratch_ref[...] + dv_step
          dk_scratch_ref[...] = dk_scratch_ref[...] + dk_step
        else:
          dv_scratch_ref[r0:r1, :] = dv_scratch_ref[r0:r1, :] + dv_step
          dk_scratch_ref[r0:r1, :] = dk_scratch_ref[r0:r1, :] + dk_step
      dq_ref[...] = dq_acc

    @pl.when((cnt_kv == 0) & (j == 0))
    def _noop_empty_kv():
      dq_ref[...] = dq_alias_ref[...]

    @pl.when(j == max_act_kv_ref[0] - 1)
    def _write():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  def _kv_map(bhi, kvi, _j, *_):
    return (bhi, kvi, 0)

  def _q_map(bhi, kvi, j, q_r, *_):
    return (bhi, q_r[lax.div(bhi, jnp.int32(h_kv)), kvi, j], 0, 0)

  def _dq_map(bhi, kvi, j, q_r, *_):
    return (lax.rem(kvi, jnp.int32(dq_slots)), bhi, q_r[lax.div(bhi, jnp.int32(h_kv)), kvi, j], 0, 0)

  def _stat_map(bhi, kvi, j, q_r, *_):
    return (bhi, q_r[lax.div(bhi, jnp.int32(h_kv)), kvi, j], 0, 0)

  def _mkv_map(bhi, kvi, j, q_r, *_):
    bi = lax.div(bhi, jnp.int32(h_kv))
    return (bi, lax.div(kvi * jnp.int32(w_per_tile), mask_rows), q_r[bi, kvi, j], 0, 0)

  dq_spec = pl.BlockSpec((None, None, None, gbq, d), _dq_map)
  dk_out_spec = pl.BlockSpec((None, bkv_dkv, d), _kv_map, pipeline_mode=pl.Buffered(buffer_count=1))
  dv_out_spec = pl.BlockSpec((None, bkv_dkv, d), _kv_map, pipeline_mode=pl.Buffered(buffer_count=1))

  dq_unred, dk_f, dv_f = pl.pallas_call(
      _bwd_dkv_kernel,
      out_shape=(
          jax.ShapeDtypeStruct((dq_slots, bh, num_q_tiles, gbq, d), jnp.float32),
          jax.ShapeDtypeStruct((bh, s_kv, d), k_f.dtype),
          jax.ShapeDtypeStruct((bh, s_kv, d), v_f.dtype),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          grid=(bh, num_kv_tiles, max_act_kv[0]),
          in_specs=[
              pl.BlockSpec((None, None, gbq, d), _q_map),
              pl.BlockSpec((None, bkv_dkv, d), _kv_map),
              pl.BlockSpec((None, bkv_dkv, d), _kv_map),
              pl.BlockSpec((None, None, gbq, d), _q_map),
              pl.BlockSpec((None, None, NUM_SUBLANES, gbq), _stat_map),
              pl.BlockSpec((None, None, NUM_SUBLANES, gbq), _stat_map),
              pl.BlockSpec((None, None, None, mask_rows, bq), _mkv_map),
              dq_spec,
          ],
          out_specs=(
              dq_spec,
              dk_out_spec,
              dv_out_spec,
          ),
          scratch_shapes=[
              pltpu.VMEM((bkv_dkv, d), jnp.float32),
              pltpu.VMEM((bkv_dkv, d), jnp.float32),
          ],
      ),
      input_output_aliases={11: 0},
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          disable_bounds_checks=disable_bounds_checks,
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      interpret=interpret,
      name=f"qsa_pallas_bwd_dkv_dq_gbq{gbq}_bkv{bkv_dkv}",
  )(q_idx_b, counts_kv_b, max_act_kv, q_offs_b, q_f, k_f, v_f, do_f, lse_dkv, delta_dkv, pmask_kv_b, dq_init)
  dq_f = jnp.sum(dq_unred, axis=0).astype(q_f.dtype)
  return (
      dq_f.reshape(b_loc, h_kv, num_q_tiles, gbq, d),
      dk_f.reshape(b_loc, h_kv, s_kv, d),
      dv_f.reshape(b_loc, h_kv, s_kv, d),
  )


def qsa_pallas_fwd(
    q_in: jax.Array,
    k_full: jax.Array,
    v_full: jax.Array,
    pmask: jax.Array,
    q_token_offset: jax.Array,
    *,
    scale: float,
    interpret: Any = False,
    cp_axis: str | None = None,
    cp_size: int = 1,
    kv_is_bhkd: bool = False,
    bkv: int = _BKV,
    disable_bounds_checks: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """VMEM-fused QSA forward returning `(out [b_loc, s_q_loc, h_q, d], lse [b_loc, h_q, s_q_loc])`."""
  b_loc, s_q_loc, h_q, _ = q_in.shape
  k_bhkd = k_full if kv_is_bhkd else jnp.transpose(k_full, (0, 2, 1, 3))
  v_bhkd = v_full if kv_is_bhkd else jnp.transpose(v_full, (0, 2, 1, 3))
  _, h_kv, s_kv, _ = k_bhkd.shape
  g = h_q // h_kv
  bq_half = _choose_bq(max(1, s_q_loc // 2), g)
  if cp_axis is not None and cp_size >= 2 and s_q_loc % (2 * bq_half) == 0:
    return qsa_pallas_fwd_balanced_cp(
        q_in,
        k_bhkd,
        v_bhkd,
        pmask,
        cp_axis=cp_axis,
        cp_size=cp_size,
        scale=scale,
        interpret=interpret,
        bkv=bkv,
        disable_bounds_checks=disable_bounds_checks,
    )
  bkv_eff = choose_bkv(s_kv, bkv)
  bq = _choose_bq(s_q_loc, g)
  num_q_tiles = s_q_loc // bq
  q_offs_b = _expand_q_tile_offs(q_token_offset, b_loc=b_loc, num_q_tiles=num_q_tiles, bq=bq)
  q_bgrp = _group_q_heads_batched(q_in, h_kv=h_kv, bq=bq)
  kv_idx_b, counts_q_b, max_act_q, pmask_q_b = _batched_q_schedule_and_mask(
      pmask, q_offs_b, bq=bq, bkv=bkv_eff, s_kv=s_kv
  )
  o_bgrp, lse = _qsa_fwd_batched(
      q_bgrp,
      k_bhkd,
      v_bhkd,
      pmask_q_b,
      kv_idx_b,
      counts_q_b,
      max_act_q,
      q_offs_b,
      bq=bq,
      bkv=bkv_eff,
      g=g,
      scale=scale,
      interpret=interpret,
      disable_bounds_checks=disable_bounds_checks,
  )
  return _ungroup_q_heads_batched(o_bgrp, g=g, bq=bq), lse


def qsa_pallas_bwd(
    q_in: jax.Array,
    k_full: jax.Array,
    v_full: jax.Array,
    out: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    pmask: jax.Array,
    q_token_offset: jax.Array,
    *,
    scale: float,
    interpret: Any = False,
    cp_axis: str | None = None,
    cp_size: int = 1,
    kv_is_bhkd: bool = False,
    bkv: int = _BKV,
    bkv_dkv: int = _BKV_DKV_DEFAULT,
    disable_bounds_checks: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """VMEM-fused QSA backward returning `(dq, dk_full, dv_full)` (`dk/dv` match `kv_is_bhkd` layout)."""
  b_loc, s_q_loc, h_q, _ = q_in.shape
  k_bhkd = k_full if kv_is_bhkd else jnp.transpose(k_full, (0, 2, 1, 3))
  v_bhkd = v_full if kv_is_bhkd else jnp.transpose(v_full, (0, 2, 1, 3))
  _, h_kv, s_kv, _ = k_bhkd.shape
  g = h_q // h_kv
  bq_half = _choose_bq(max(1, s_q_loc // 2), g)
  if cp_axis is not None and cp_size >= 2 and s_q_loc % (2 * bq_half) == 0:
    dq, dk_bhkd, dv_bhkd = qsa_pallas_bwd_balanced_cp(
        q_in,
        k_bhkd,
        v_bhkd,
        out,
        lse,
        do,
        pmask,
        cp_axis=cp_axis,
        cp_size=cp_size,
        scale=scale,
        interpret=interpret,
        bkv=bkv,
        bkv_dkv=bkv_dkv,
        disable_bounds_checks=disable_bounds_checks,
    )
    if kv_is_bhkd:
      return dq, dk_bhkd, dv_bhkd
    return dq, jnp.transpose(dk_bhkd, (0, 2, 1, 3)), jnp.transpose(dv_bhkd, (0, 2, 1, 3))

  del bkv
  bkv_dkv_eff = choose_bkv(s_kv, bkv_dkv)
  bq = _choose_bq(s_q_loc, g)
  num_q_tiles = s_q_loc // bq
  gbq = g * bq
  q_offs_b = _expand_q_tile_offs(q_token_offset, b_loc=b_loc, num_q_tiles=num_q_tiles, bq=bq)

  q_bgrp = _group_q_heads_batched(q_in, h_kv=h_kv, bq=bq)
  do_bgrp = _group_q_heads_batched(do, h_kv=h_kv, bq=bq)
  out_bgrp = _group_q_heads_batched(out, h_kv=h_kv, bq=bq)
  lse_bgrp = lse.reshape(b_loc, h_kv, g, num_q_tiles, bq).transpose(0, 1, 3, 2, 4).reshape(b_loc, h_kv, num_q_tiles, gbq)
  delta_bgrp = jnp.sum(out_bgrp.astype(jnp.float32) * do_bgrp.astype(jnp.float32), axis=-1)

  q_idx_b, counts_kv_b, max_act_kv, pmask_kv_b = _batched_kv_schedule_and_mask(
      pmask, q_offs_b, bq=bq, bkv_dkv=bkv_dkv_eff, s_kv=s_kv
  )
  dq_bgrp, dk_bhkd, dv_bhkd = _qsa_bwd_dkv_batched(
      q_bgrp,
      k_bhkd,
      v_bhkd,
      do_bgrp,
      lse_bgrp,
      delta_bgrp,
      pmask_kv_b,
      q_idx_b,
      counts_kv_b,
      max_act_kv,
      q_offs_b,
      bq=bq,
      bkv_dkv=bkv_dkv_eff,
      g=g,
      scale=scale,
      interpret=interpret,
      disable_bounds_checks=disable_bounds_checks,
  )
  dq = _ungroup_q_heads_batched(dq_bgrp, g=g, bq=bq)
  if kv_is_bhkd:
    return dq, dk_bhkd, dv_bhkd
  return dq, jnp.transpose(dk_bhkd, (0, 2, 1, 3)), jnp.transpose(dv_bhkd, (0, 2, 1, 3))


def qsa_pallas_fwd_balanced_cp(
    q_in: jax.Array,
    k_bhkd: jax.Array,
    v_bhkd: jax.Array,
    pmask: jax.Array,
    *,
    cp_axis: str,
    cp_size: int,
    scale: float,
    interpret: Any = False,
    bkv: int = _BKV,
    disable_bounds_checks: bool = True,
) -> tuple[jax.Array, jax.Array]:
  """Symmetric-pair (`r`, `cp_size - 1 - r`) load-balanced CP QSA forward."""
  b_loc, s_q_loc, h_q, _ = q_in.shape
  _, h_kv, s_kv, _ = k_bhkd.shape
  g = h_q // h_kv
  half_s = s_q_loc // 2
  bq = _choose_bq(half_s, g)
  nq_half = half_s // bq
  bkv_eff = choose_bkv(s_kv, bkv)

  perm = [(i, cp_size - 1 - i) for i in range(cp_size)]
  r = lax.axis_index(cp_axis)
  offs_0_b = _expand_q_tile_offs(r * jnp.int32(s_q_loc), b_loc=b_loc, num_q_tiles=nq_half, bq=bq)
  offs_1_b = _expand_q_tile_offs(
      (jnp.int32(cp_size - 1) - r) * jnp.int32(s_q_loc) + jnp.int32(half_s),
      b_loc=b_loc,
      num_q_tiles=nq_half,
      bq=bq,
  )

  q_0, q_1_loc = q_in[:, :half_s], q_in[:, half_s:]
  m_0, m_1_loc = pmask[:, :half_s], pmask[:, half_s:]
  q_1 = lax.ppermute(q_1_loc, cp_axis, perm)
  m_1 = lax.ppermute(m_1_loc, cp_axis, perm)
  # Complete the sub-millisecond 1-to-1 `ppermute` before launching `q_0` so XLA's
  # live-range minimizer cannot sink `collective-permute-start` (a peer rendezvous)
  # between `q_0` and `q_1`, which would serialize asymmetric `(c_lo, c_hi)` chunks.
  q_0, m_0, q_1, m_1 = lax.optimization_barrier((q_0, m_0, q_1, m_1))

  q0_bgrp = _group_q_heads_batched(q_0, h_kv=h_kv, bq=bq)
  kv_idx_0, cnt_0, max_0, pmq_0 = _batched_q_schedule_and_mask(m_0, offs_0_b, bq=bq, bkv=bkv_eff, s_kv=s_kv)
  o0_bgrp, lse_0 = _qsa_fwd_batched(
      q0_bgrp,
      k_bhkd,
      v_bhkd,
      pmq_0,
      kv_idx_0,
      cnt_0,
      max_0,
      offs_0_b,
      bq=bq,
      bkv=bkv_eff,
      g=g,
      scale=scale,
      interpret=interpret,
      disable_bounds_checks=disable_bounds_checks,
  )

  q1_bgrp = _group_q_heads_batched(q_1, h_kv=h_kv, bq=bq)
  kv_idx_1, cnt_1, max_1, pmq_1 = _batched_q_schedule_and_mask(m_1, offs_1_b, bq=bq, bkv=bkv_eff, s_kv=s_kv)
  o1_bgrp, lse_1_p = _qsa_fwd_batched(
      q1_bgrp,
      k_bhkd,
      v_bhkd,
      pmq_1,
      kv_idx_1,
      cnt_1,
      max_1,
      offs_1_b,
      bq=bq,
      bkv=bkv_eff,
      g=g,
      scale=scale,
      interpret=interpret,
      disable_bounds_checks=disable_bounds_checks,
  )

  out_0 = _ungroup_q_heads_batched(o0_bgrp, g=g, bq=bq)
  out_1_p = _ungroup_q_heads_batched(o1_bgrp, g=g, bq=bq)
  out_0, lse_0, out_1_p, lse_1_p = lax.optimization_barrier((out_0, lse_0, out_1_p, lse_1_p))
  out_1_loc = lax.ppermute(out_1_p, cp_axis, perm)
  lse_1_loc = lax.ppermute(lse_1_p, cp_axis, perm)
  return jnp.concatenate([out_0, out_1_loc], axis=1), jnp.concatenate([lse_0, lse_1_loc], axis=2)


def qsa_pallas_bwd_balanced_cp(
    q_in: jax.Array,
    k_bhkd: jax.Array,
    v_bhkd: jax.Array,
    out: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    pmask: jax.Array,
    *,
    cp_axis: str,
    cp_size: int,
    scale: float,
    interpret: Any = False,
    bkv: int = _BKV,
    bkv_dkv: int = _BKV_DKV_DEFAULT,
    disable_bounds_checks: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Symmetric-pair load-balanced CP QSA backward."""
  del bkv
  b_loc, s_q_loc, h_q, _ = q_in.shape
  _, h_kv, s_kv, _ = k_bhkd.shape
  g = h_q // h_kv
  half_s = s_q_loc // 2
  bq = _choose_bq(half_s, g)
  nq_half = half_s // bq
  nq_total = s_q_loc // bq
  gbq = g * bq
  bkv_dkv_eff = choose_bkv(s_kv, bkv_dkv)

  perm = [(i, cp_size - 1 - i) for i in range(cp_size)]
  r = lax.axis_index(cp_axis)
  offs_0_b = _expand_q_tile_offs(r * jnp.int32(s_q_loc), b_loc=b_loc, num_q_tiles=nq_half, bq=bq)
  offs_1_b = _expand_q_tile_offs(
      (jnp.int32(cp_size - 1) - r) * jnp.int32(s_q_loc) + jnp.int32(half_s),
      b_loc=b_loc,
      num_q_tiles=nq_half,
      bq=bq,
  )
  offs_pair_b = jnp.concatenate([offs_0_b, offs_1_b], axis=1)

  q_0, q_1_loc = q_in[:, :half_s], q_in[:, half_s:]
  out_0, out_1_loc = out[:, :half_s], out[:, half_s:]
  do_0, do_1_loc = do[:, :half_s], do[:, half_s:]
  lse_0, lse_1_loc = lse[:, :, :half_s], lse[:, :, half_s:]
  m_0, m_1_loc = pmask[:, :half_s], pmask[:, half_s:]

  packed_1 = lax.ppermute(jnp.stack([q_1_loc, out_1_loc, do_1_loc], axis=0), cp_axis, perm)
  lse_1 = lax.ppermute(lse_1_loc, cp_axis, perm)
  m_1 = lax.ppermute(m_1_loc, cp_axis, perm)
  q_0, out_0, do_0, lse_0, m_0, packed_1, lse_1, m_1 = lax.optimization_barrier(
      (q_0, out_0, do_0, lse_0, m_0, packed_1, lse_1, m_1)
  )
  q_1, out_1, do_1 = packed_1[0], packed_1[1], packed_1[2]

  q0_bgrp = _group_q_heads_batched(q_0, h_kv=h_kv, bq=bq)
  do0_bgrp = _group_q_heads_batched(do_0, h_kv=h_kv, bq=bq)
  out0_bgrp = _group_q_heads_batched(out_0, h_kv=h_kv, bq=bq)
  lse0_bgrp = lse_0.reshape(b_loc, h_kv, g, nq_half, bq).transpose(0, 1, 3, 2, 4).reshape(b_loc, h_kv, nq_half, gbq)
  del0_bgrp = jnp.sum(out0_bgrp.astype(jnp.float32) * do0_bgrp.astype(jnp.float32), axis=-1)

  q1_bgrp = _group_q_heads_batched(q_1, h_kv=h_kv, bq=bq)
  do1_bgrp = _group_q_heads_batched(do_1, h_kv=h_kv, bq=bq)
  out1_bgrp = _group_q_heads_batched(out_1, h_kv=h_kv, bq=bq)
  lse1_bgrp = lse_1.reshape(b_loc, h_kv, g, nq_half, bq).transpose(0, 1, 3, 2, 4).reshape(b_loc, h_kv, nq_half, gbq)
  del1_bgrp = jnp.sum(out1_bgrp.astype(jnp.float32) * do1_bgrp.astype(jnp.float32), axis=-1)

  m_pair = jnp.concatenate([m_0, m_1], axis=1)
  q_idx_b, cnt_kv_b, max_kv, pmkv_pair_b = _batched_kv_schedule_and_mask(
      m_pair, offs_pair_b, bq=bq, bkv_dkv=bkv_dkv_eff, s_kv=s_kv
  )
  q_pair_bgrp = jnp.concatenate([q0_bgrp, q1_bgrp], axis=2)
  do_pair_bgrp = jnp.concatenate([do0_bgrp, do1_bgrp], axis=2)
  lse_pair_bgrp = jnp.concatenate([lse0_bgrp, lse1_bgrp], axis=2)
  del_pair_bgrp = jnp.concatenate([del0_bgrp, del1_bgrp], axis=2)
  assert q_pair_bgrp.shape[2] == nq_total
  dq_pair_bgrp, dk_bhkd, dv_bhkd = _qsa_bwd_dkv_batched(
      q_pair_bgrp,
      k_bhkd,
      v_bhkd,
      do_pair_bgrp,
      lse_pair_bgrp,
      del_pair_bgrp,
      pmkv_pair_b,
      q_idx_b,
      cnt_kv_b,
      max_kv,
      offs_pair_b,
      bq=bq,
      bkv_dkv=bkv_dkv_eff,
      g=g,
      scale=scale,
      interpret=interpret,
      disable_bounds_checks=disable_bounds_checks,
  )
  dq_0 = _ungroup_q_heads_batched(dq_pair_bgrp[:, :, :nq_half], g=g, bq=bq)
  dq_1_p = _ungroup_q_heads_batched(dq_pair_bgrp[:, :, nq_half:], g=g, bq=bq)
  dq_0, dq_1_p = lax.optimization_barrier((dq_0, dq_1_p))
  dq_1_loc = lax.ppermute(dq_1_p, cp_axis, perm)
  return jnp.concatenate([dq_0, dq_1_loc], axis=1), dk_bhkd, dv_bhkd
