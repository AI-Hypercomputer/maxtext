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

"""Fused FP8 gate backward (dx GMM + dW TGMM) for the DSv3 routed-expert MLP.

In the FP8 routed-expert backward pass (`dsv3_experts.py`), the gate backward
consists of two matmuls sharing the quantized activation gradient
`dgate_q` (`fp8_e5m2 [M, F]`):

  dx3d [M, D0, 128] = (dgate_q @ w_q[e]^T) * dx_scale          (bf16)
  dW   [E, D,  F]   = (x3d^T   @ dgate_q)  * dw_scale (+ acc)  (bf16 / f32)

Running these as separate `gmm_v2` (dx) and `tgmm_v2` (dW) kernels streams
`dgate_q` from HBM 14 times (7 D-tiles in dx + 7 D-tiles in dW). Fusing both
matmuls into a single `(num_d, num_gm)` pipeline streams `dgate_q` only once
per D-tile (7x instead of 14x): each step `(d, gm)` loads one `[tile_rows, F]`
slice of `dgate_q` and one `[tile_rows, tile_d0, 128]` slice of `x3d` into VMEM
and uses the resident `dgate_q` tile for both the `dx` D-slice and the `dW`
f32 accumulator update before advancing to `gm + 1`.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.experimental.lineage.gmmv2 import gmmv2_kernel
from maxtext.experimental.lineage.gmmv2 import relayout

LANES = 128
_SUBLANE = 32  # TPU v7x sublane tiling for 8-bit dtypes.
_DX_COLS = 256  # TPU v7x MXU column size for the dx rhs chunks.
_ZERO_ROWS = 128  # Row chunk for background empty-expert zeroing.
_F32_EXP_BIAS = 127
_F32_MANT_BITS = 23
_MIN_EXP = -126  # Smallest normal f32 exponent.


def _pow2(exp: jax.Array) -> jax.Array:
  """Returns the f32 `2 ** exp` of int32 `exp` in [-126, 127], bit-exact."""
  return lax.bitcast_convert_type((exp + _F32_EXP_BIAS) << _F32_MANT_BITS, jnp.float32)


@dataclasses.dataclass(frozen=True)
class FusedTiles:
  """Tile sizes for `gate_bwd_fused`.

  Attributes:
    tile_rows: Token rows per `gm` step (must be a multiple of `bucket_rows` and
      `_SUBLANE`). Defaults to 1024 (`DSV3_V7X_CONFIG.gate_dx.tile_m`).
    bucket_rows: Dynamic M-bucket granularity for boundary `gm` tiles. Matches
      `DSV3_V7X_CONFIG.gate_dw.bucket_base = 256` by default (TPU v7x MXU
      contracting dimension is 256).
    tile_d: D-dimension tile size (`tile_d0 = tile_d // 128`). Clamped down to
      `D` for small unit-test shapes (`D = 512`).
  """

  tile_rows: int = 1024
  bucket_rows: int = 256
  tile_d: int = 1024


DEFAULT_TILES = FusedTiles()


def is_supported(
    x3d: jax.Array,
    w_q: jax.Array,
    group_sizes: jax.Array,
    dgate_q: jax.Array,
    *,
    dw_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tiles: FusedTiles | None = None,
) -> bool:
  """Returns whether `gate_bwd_fused` supports the given shapes and dtypes."""
  if tiles is None:
    tiles = DEFAULT_TILES
  if x3d.ndim != 3 or x3d.shape[-1] != LANES:
    return False
  if w_q.ndim != 3 or dgate_q.ndim != 2 or group_sizes.ndim != 1:
    return False
  if not (gmmv2_kernel.is_fp8(x3d.dtype) and gmmv2_kernel.is_fp8(w_q.dtype) and gmmv2_kernel.is_fp8(dgate_q.dtype)):
    return False
  size_m, d0, _ = x3d.shape
  size_d = d0 * LANES
  num_experts, w_d, size_f = w_q.shape
  if w_d != size_d or dgate_q.shape != (size_m, size_f):
    return False
  if group_sizes.shape != (num_experts,):
    return False
  if size_m % _SUBLANE != 0 or size_f % LANES != 0:
    return False
  tile_rows = min(tiles.tile_rows, size_m)
  bucket_rows = math.gcd(tile_rows, tiles.bucket_rows)
  if tile_rows % _SUBLANE != 0 or bucket_rows % _SUBLANE != 0 or tile_rows % bucket_rows != 0:
    return False
  tile_d = min(tiles.tile_d, size_d)
  if size_d % tile_d != 0:
    return False
  if tile_d % _DX_COLS != 0 or tile_d % _ZERO_ROWS != 0:
    return False
  tile_d0 = tile_d // LANES
  if tile_d0 % 4 != 0 or (tile_d0 % 8 != 0 and tile_d0 != d0):
    return False
  return jnp.dtype(dw_dtype) in (jnp.bfloat16, jnp.float32)


def _count_gm_tiles(group_sizes: jax.Array, tile_rows: int) -> jax.Array:
  """Computes `num_gm` scalar matching `_fill_metadata` without a scan loop."""
  gs = group_sizes.astype(jnp.int32)
  m_off = jnp.cumsum(gs) - gs
  n_tiles = jnp.where(gs > 0, (gs + (m_off % _SUBLANE) + tile_rows - 1) // tile_rows, 0)
  return jnp.sum(n_tiles)


def _fill_metadata(
    group_sizes_ref: jax.Array,
    num_gm_ref: jax.Array,
    num_wload_ref: jax.Array,
    gm_to_expert_ref: jax.Array,
    gm_to_m_ref: jax.Array,
    gm_to_wload_ref: jax.Array,
    wload_to_expert_ref: jax.Array,
    *,
    num_experts: int,
    tile_rows: int,
    kmax_ref: jax.Array | None = None,
    gm_needs_shift_ref: jax.Array | None = None,
    gm_shifts_ref: jax.Array | None = None,
    dgate_exp_ref: jax.Array | None = None,
    dgrad_tile_rows: int = 256,
) -> tuple[jax.Array, jax.Array]:
  """Populates SMEM tile-to-expert and expert-to-wload metadata tables."""
  gm_to_m_ref[0] = 0
  has_exps = dgate_exp_ref is not None
  ratio = tile_rows // dgrad_tile_rows

  def fill_expert(e, carry):
    gm_idx, wload_idx, m_off, dg_idx = carry
    size = group_sizes_ref[e]
    end_m = m_off + size
    local_off = m_off % _SUBLANE
    n_tiles = jnp.where(size > 0, pl.cdiv(size + local_off, tile_rows), 0)
    next_gm = gm_idx + n_tiles

    @pl.when(size > 0)
    def _record_wload():
      wload_to_expert_ref[wload_idx] = e

    def fill_tile(tm_id, cur_m):
      loc = cur_m % _SUBLANE
      tm = jnp.minimum(tile_rows - loc, end_m - cur_m)
      nxt_m = cur_m + tm
      gm_to_expert_ref[tm_id] = e
      gm_to_wload_ref[tm_id] = wload_idx
      gm_to_m_ref[tm_id] = cur_m
      gm_to_m_ref[tm_id + 1] = nxt_m
      return nxt_m

    lax.fori_loop(gm_idx, next_gm, fill_tile, m_off)
    if has_exps:
      assert kmax_ref is not None and gm_needs_shift_ref is not None and gm_shifts_ref is not None
      d_n_tiles = jnp.where(size > 0, pl.cdiv(size + local_off, dgrad_tile_rows), 0)

      def max_exp_body(j, km):
        return jnp.maximum(km, dgate_exp_ref[dg_idx + j])

      kmax = lax.fori_loop(0, d_n_tiles, max_exp_body, jnp.int32(-(2**30)))
      kmax_ref[e] = kmax

      def fill_shifts(tm_id, _):
        j0 = (tm_id - gm_idx) * ratio
        needs = jnp.int32(0)
        for i in range(ratio):
          j = j0 + i
          k_i = dgate_exp_ref[dg_idx + jnp.minimum(j, d_n_tiles - 1)]
          sh_i = jnp.where(j < d_n_tiles, jnp.maximum(k_i - kmax, _MIN_EXP), 0)
          gm_shifts_ref[tm_id, i] = sh_i
          needs = jnp.bitwise_or(needs, jnp.where(sh_i != 0, 1, 0))
        gm_needs_shift_ref[tm_id] = needs
        return jnp.int32(0)

      lax.fori_loop(gm_idx, next_gm, fill_shifts, jnp.int32(0))
      next_dg = dg_idx + d_n_tiles
    else:
      next_dg = dg_idx
    next_wload = wload_idx + jnp.where(size > 0, 1, 0)
    return next_gm, next_wload, end_m, next_dg

  num_gm, num_wload, _, _ = lax.fori_loop(
      0,
      num_experts,
      fill_expert,
      (jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0)),
  )
  num_gm_ref[0] = num_gm
  num_wload_ref[0] = num_wload
  return num_gm, num_wload


def _kernel_impl(
    group_sizes_ref: jax.Array,
    dx_scale_ref: jax.Array,
    dw_scale_ref: jax.Array,
    dgate_exp_ref: jax.Array,
    x_hbm: jax.Array,
    dgq_hbm: jax.Array,
    w_hbm: jax.Array,
    dx_hbm: jax.Array,
    dw_hbm: jax.Array,
    num_gm_ref: jax.Array,
    num_wload_ref: jax.Array,
    gm_to_expert_ref: jax.Array,
    gm_to_m_ref: jax.Array,
    gm_to_wload_ref: jax.Array,
    wload_to_expert_ref: jax.Array,
    kmax_ref: jax.Array,
    gm_needs_shift_ref: jax.Array,
    gm_shifts_ref: jax.Array,
    dgq_buf: jax.Array,
    x_buf: jax.Array,
    w_buf: jax.Array,
    acc_ref: jax.Array,
    out_buf: jax.Array,
    dx_buf: jax.Array,
    in_sem: jax.Array,
    w_sem: jax.Array,
    out_sem: jax.Array,
    dx_sem: jax.Array,
    *,
    accin_hbm: jax.Array | None,
    accin_sem: jax.Array | None,
    zero_buf: jax.Array | None,
    zero_sem: jax.Array | None,
    num_experts: int,
    size_d: int,
    tile_rows: int,
    bucket_rows: int,
    tile_d: int,
    has_tile_exps: bool = False,
    dgrad_tile_rows: int = 256,
):
  """Inner Pallas kernel body over grid `(num_d, max(num_gm, 1))`."""
  has_acc = accin_hbm is not None
  num_d = size_d // tile_d
  tile_d0 = tile_d // LANES
  num_buckets = tile_rows // bucket_rows
  dw_dtype = dw_hbm.dtype

  d = pl.program_id(0)
  gm = pl.program_id(1)

  def _d_slice(d_idx: jax.Array | int):
    if num_d == 1:
      return slice(None)
    return pl.ds(
        pl.multiple_of(jnp.asarray(d_idx, jnp.int32) * tile_d0, tile_d0),
        tile_d0,
    )

  def _in_dma(d_idx: jax.Array, gm_idx: jax.Array, slot: jax.Array):
    m_s = gm_to_m_ref[gm_idx]
    m_e = gm_to_m_ref[gm_idx + 1]
    a_s = pl.multiple_of((m_s // _SUBLANE) * _SUBLANE, _SUBLANE)
    fetch = pl.multiple_of(pl.cdiv(m_e - a_s, _SUBLANE) * _SUBLANE, _SUBLANE)
    dgq_c = pltpu.make_async_copy(
        dgq_hbm.at[pl.ds(a_s, fetch), :],
        dgq_buf.at[slot, pl.ds(0, fetch), :],
        in_sem.at[slot, 0],
    )
    x_c = pltpu.make_async_copy(
        x_hbm.at[pl.ds(a_s, fetch), _d_slice(d_idx), :],
        x_buf.at[slot, pl.ds(0, fetch), :, :],
        in_sem.at[slot, 1],
    )
    return dgq_c, x_c

  def _w_dma(e_idx: jax.Array, d_idx: jax.Array | int, slot: jax.Array):
    row = pl.multiple_of(e_idx * size_d + jnp.asarray(d_idx, jnp.int32) * tile_d, tile_d)
    return pltpu.make_async_copy(
        w_hbm.at[pl.ds(row, tile_d), :],
        w_buf.at[slot],
        w_sem.at[slot],
    )

  def _dx_dma(d_idx: jax.Array | int, gm_idx: jax.Array, slot: jax.Array):
    m_s = gm_to_m_ref[gm_idx]
    m_e = gm_to_m_ref[gm_idx + 1]
    lo = m_s % _SUBLANE
    n = m_e - m_s
    return pltpu.make_async_copy(
        dx_buf.at[slot, pl.ds(lo, n), :, :],
        dx_hbm.at[pl.ds(m_s, n), _d_slice(d_idx), :],
        dx_sem.at[slot],
    )

  def _accin_dma(e_idx: jax.Array, d_idx: jax.Array | int, slot: jax.Array | int):
    assert accin_hbm is not None and accin_sem is not None
    row = pl.multiple_of(e_idx * size_d + jnp.asarray(d_idx, jnp.int32) * tile_d, tile_d)
    return pltpu.make_async_copy(
        accin_hbm.at[pl.ds(row, tile_d), :],
        out_buf.at[slot],
        accin_sem.at[slot],
    )

  def _out_dma(e_idx: jax.Array, d_idx: jax.Array | int, slot: jax.Array | int):
    row = pl.multiple_of(e_idx * size_d + jnp.asarray(d_idx, jnp.int32) * tile_d, tile_d)
    return pltpu.make_async_copy(
        out_buf.at[slot],
        dw_hbm.at[pl.ds(row, tile_d), :],
        out_sem.at[slot],
    )

  # Prologue: populate metadata, start zero-fills (no-acc) and step-0
  # prefetches.
  @pl.when(jnp.logical_and(d == 0, gm == 0))
  def _prologue():
    num_gm_init, _ = _fill_metadata(
        group_sizes_ref,
        num_gm_ref,
        num_wload_ref,
        gm_to_expert_ref,
        gm_to_m_ref,
        gm_to_wload_ref,
        wload_to_expert_ref,
        num_experts=num_experts,
        tile_rows=tile_rows,
        kmax_ref=kmax_ref if has_tile_exps else None,
        gm_needs_shift_ref=gm_needs_shift_ref if has_tile_exps else None,
        gm_shifts_ref=gm_shifts_ref if has_tile_exps else None,
        dgate_exp_ref=dgate_exp_ref if has_tile_exps else None,
        dgrad_tile_rows=dgrad_tile_rows,
    )
    if not has_acc:
      assert zero_buf is not None and zero_sem is not None
      zero_buf_ref = zero_buf
      zero_sem_ref = zero_sem
      zero_buf_ref[...] = jnp.zeros_like(zero_buf_ref)

      @pl.loop(0, num_experts)
      def _zero_expert(e_idx):
        @pl.when(group_sizes_ref[e_idx] == 0)
        def _do_zero():
          @pl.loop(0, size_d, step=_ZERO_ROWS)
          def _zero_chunk(r):
            row = pl.multiple_of(e_idx * size_d + r, _ZERO_ROWS)
            pltpu.make_async_copy(
                zero_buf_ref,
                dw_hbm.at[pl.ds(row, _ZERO_ROWS), :],
                zero_sem_ref,
            ).start(priority=1)

    @pl.when(num_gm_init > 0)
    def _start_first():
      e0 = wload_to_expert_ref[0]
      _w_dma(e0, 0, jnp.int32(0)).start()
      dgq_c, x_c = _in_dma(jnp.int32(0), jnp.int32(0), jnp.int32(0))
      dgq_c.start()
      x_c.start()

  num_gm = num_gm_ref[0]
  num_wload = num_wload_ref[0]

  @pl.when(gm < num_gm)
  def _step():
    t = d * num_gm + gm
    in_slot = t % 2
    dx_slot = t % 2
    total_steps = num_d * num_gm

    e = gm_to_expert_ref[gm]
    w_idx = gm_to_wload_ref[gm]
    cur_wload = d * num_wload + w_idx
    w_slot = cur_wload % 2
    out_slot = cur_wload % 2
    total_wloads = num_d * num_wload

    prev_gm = jnp.where(gm > 0, gm - 1, 0)
    is_new_expert = jnp.logical_or(gm == 0, e != gm_to_expert_ref[prev_gm])
    is_last_gm = gm == (num_gm - 1)
    next_gm_clamped = jnp.where(is_last_gm, gm, gm + 1)
    is_last_of_expert = jnp.logical_or(is_last_gm, e != gm_to_expert_ref[next_gm_clamped])

    # 1. Wait current W and (dgq, x) at the head of the DMA queue.
    @pl.when(is_new_expert)
    def _wait_cur_w():
      _w_dma(e, d, w_slot).wait()

    dgq_cur, x_cur = _in_dma(d, gm, in_slot)
    dgq_cur.wait()
    x_cur.wait()

    # 2. On new expert: wait out_buf[out_slot] from cur_wload - 2, start
    #    accin(e, d) into out_buf[out_slot] (needed at end of this expert),
    #    then prefetch next expert's W into w_buf[1 - w_slot].
    @pl.when(is_new_expert)
    def _on_new_expert():
      @pl.when(cur_wload >= 2)
      def _wait_prev2_out():
        prev2_w = cur_wload - 2
        prev2_d = prev2_w // num_wload
        prev2_e = wload_to_expert_ref[prev2_w % num_wload]
        _out_dma(prev2_e, prev2_d, out_slot).wait()

      if has_acc:
        _accin_dma(e, d, out_slot).start()

      @pl.when(cur_wload + 1 < total_wloads)
      def _prefetch_w():
        has_more_w = w_idx + 1 < num_wload
        next_w_idx = jnp.where(has_more_w, w_idx + 1, 0)
        next_w_d = jnp.where(has_more_w, d, d + 1)
        next_e = wload_to_expert_ref[next_w_idx]
        _w_dma(next_e, next_w_d, 1 - w_slot).start()

    # 3. Prefetch next step's (dgq, x) into 1 - in_slot.
    @pl.when(t + 1 < total_steps)
    def _prefetch_in():
      has_more_gm = gm + 1 < num_gm
      next_gm = jnp.where(has_more_gm, gm + 1, 0)
      next_d = jnp.where(has_more_gm, d, d + 1)
      dgq_c, x_c = _in_dma(next_d, next_gm, 1 - in_slot)
      dgq_c.start()
      x_c.start()

    # 4. Wait previous dx (step t - 2) before overwriting dx_buf[dx_slot].
    @pl.when(t >= 2)
    def _wait_prev_dx():
      t2 = t - 2
      _dx_dma(t2 // num_gm, t2 % num_gm, dx_slot).wait()

    # 5. Bucketed matmuls for dx and dW.
    m_start = gm_to_m_ref[gm]
    m_end = gm_to_m_ref[gm + 1]
    a_start = (m_start // _SUBLANE) * _SUBLANE
    lo = m_start - a_start
    hi = m_end - a_start

    if has_tile_exps:
      exp_scale = _pow2(kmax_ref[e])
      dx_scale = dx_scale_ref[0] * exp_scale
      dw_scale = dw_scale_ref[0] * exp_scale
    else:
      dx_scale = dx_scale_ref[0]
      dw_scale = dw_scale_ref[0]

    def _maybe_shift_dgq(bucket_m: int):
      if has_tile_exps:
        num_sub = math.ceil(bucket_m / dgrad_tile_rows)

        @pl.when(gm_needs_shift_ref[gm] != 0)
        def _shift_tile():
          for i in range(num_sub):
            r0 = i * dgrad_tile_rows
            r_len = min(dgrad_tile_rows, bucket_m - r0)
            s_i = pl.ds(r0, r_len)
            sub = dgq_buf[in_slot, s_i, :]
            f_i = _pow2(jnp.full((1, 1), gm_shifts_ref[gm, i], jnp.int32)).astype(jnp.bfloat16)
            dgq_buf[in_slot, s_i, :] = (sub.astype(jnp.bfloat16) * f_i).astype(sub.dtype)

    def _dx_matmul_and_store(dgq: jax.Array, bucket_m: int):
      dx_parts = []
      for c0 in range(0, tile_d, _DX_COLS):
        c1 = min(tile_d, c0 + _DX_COLS)
        val = lax.dot_general(
            dgq,
            w_buf[w_slot, c0:c1, :],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        dx_parts.append((val * dx_scale).astype(jnp.bfloat16))
      dx_2d = jnp.concatenate(dx_parts, axis=-1)
      relayout.store_2d_as_3d(dx_buf.at[dx_slot], dx_2d, bucket_m, tile_d0)

    def _matmul_interior():
      _maybe_shift_dgq(tile_rows)
      dgq = dgq_buf[in_slot, :tile_rows, :]
      _dx_matmul_and_store(dgq, tile_rows)
      x_2d = relayout.load_3d_as_2d(x_buf.at[in_slot], tile_rows, tile_d0)
      acc_ref[...] += lax.dot_general(
          x_2d,
          dgq,
          (((0,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    def _matmul_bucket(bucket_m: int):
      _maybe_shift_dgq(bucket_m)
      dgq = dgq_buf[in_slot, :bucket_m, :]
      row_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, 1), 0)
      row_mask = jnp.logical_and(lo <= row_iota, row_iota < hi)
      dg_masked = jnp.where(lax.broadcast_in_dim(row_mask, dgq.shape, (0, 1)), dgq, 0)

      _dx_matmul_and_store(dgq, bucket_m)

      x_2d = relayout.load_3d_as_2d(x_buf.at[in_slot], bucket_m, tile_d0)
      x_masked = jnp.where(lax.broadcast_in_dim(row_mask, x_2d.shape, (0, 1)), x_2d, 0)

      @pl.when(is_new_expert)
      def _dw_new():
        acc_ref[...] = lax.dot_general(
            x_masked,
            dg_masked,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

      @pl.when(jnp.logical_not(is_new_expert))
      def _dw_accum():
        acc_ref[...] += lax.dot_general(
            x_masked,
            dg_masked,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    is_interior = jnp.logical_and(jnp.logical_not(is_new_expert), jnp.logical_not(is_last_of_expert))
    bucket_idx = jnp.maximum(0, (hi - 1) // bucket_rows)
    lax.cond(
        is_interior,
        _matmul_interior,
        lambda: lax.switch(
            bucket_idx,
            [
                functools.partial(
                    _matmul_bucket,
                    bucket_m=bucket_rows * (i + 1),
                )
                for i in range(num_buckets)
            ],
        ),
    )

    # 6. Start dx HBM write for rows [lo, hi) -> [m_start, m_end).
    _dx_dma(d, gm, dx_slot).start()

    # 7. On last gm of expert: finalize acc_ref -> out_buf[out_slot] -> dw_hbm.
    @pl.when(is_last_of_expert)
    def _finalize_dw():
      row_step = min(tile_d, 256)
      if has_acc:
        _accin_dma(e, d, out_slot).wait()

        @pl.loop(0, tile_d, step=row_step)
        def _scale_and_add_accin(r):
          rs = pl.ds(pl.multiple_of(r, row_step), row_step)
          out_buf[out_slot, rs, :] = (acc_ref[rs, :] * dw_scale + out_buf[out_slot, rs, :].astype(jnp.float32)).astype(
              dw_dtype
          )

      else:

        @pl.loop(0, tile_d, step=row_step)
        def _scale_only(r):
          rs = pl.ds(pl.multiple_of(r, row_step), row_step)
          out_buf[out_slot, rs, :] = (acc_ref[rs, :] * dw_scale).astype(dw_dtype)

      _out_dma(e, d, out_slot).start()

  # Epilogue: drain remaining dx, out, and zero DMAs on the last grid step.
  is_last_prog = jnp.logical_and(d == num_d - 1, gm == pl.num_programs(1) - 1)

  @pl.when(is_last_prog)
  def _epilogue():
    @pl.when(num_gm > 0)
    def _drain_active():
      last_t = num_d * num_gm - 1
      _dx_dma(num_d - 1, num_gm - 1, last_t % 2).wait()

      @pl.when(num_d * num_gm >= 2)
      def _drain_second_last_dx():
        prev_t = last_t - 1
        _dx_dma(prev_t // num_gm, prev_t % num_gm, prev_t % 2).wait()

      last_w = num_d * num_wload - 1
      last_e = wload_to_expert_ref[num_wload - 1]
      _out_dma(last_e, num_d - 1, last_w % 2).wait()

      @pl.when(num_d * num_wload >= 2)
      def _drain_second_last_out():
        prev_w = last_w - 1
        _out_dma(
            wload_to_expert_ref[prev_w % num_wload],
            prev_w // num_wload,
            prev_w % 2,
        ).wait()

    if not has_acc:
      assert zero_sem is not None
      zero_sem_ref = zero_sem
      num_empty = num_experts - num_wload

      @pl.when(num_empty > 0)
      def _drain_zeros():
        zero_rows = pl.multiple_of(num_empty * size_d, _ZERO_ROWS)
        pltpu.make_async_copy(
            dw_hbm.at[pl.ds(0, zero_rows), :],
            dw_hbm.at[pl.ds(0, zero_rows), :],
            zero_sem_ref,
        ).wait()


def _kernel_no_acc(
    group_sizes_ref: jax.Array,
    dx_scale_ref: jax.Array,
    dw_scale_ref: jax.Array,
    dgate_exp_ref: jax.Array,
    x_hbm: jax.Array,
    dgq_hbm: jax.Array,
    w_hbm: jax.Array,
    dx_hbm: jax.Array,
    dw_hbm: jax.Array,
    num_gm_ref: jax.Array,
    num_wload_ref: jax.Array,
    gm_to_expert_ref: jax.Array,
    gm_to_m_ref: jax.Array,
    gm_to_wload_ref: jax.Array,
    wload_to_expert_ref: jax.Array,
    kmax_ref: jax.Array,
    gm_needs_shift_ref: jax.Array,
    gm_shifts_ref: jax.Array,
    dgq_buf: jax.Array,
    x_buf: jax.Array,
    w_buf: jax.Array,
    acc_ref: jax.Array,
    out_buf: jax.Array,
    dx_buf: jax.Array,
    in_sem: jax.Array,
    w_sem: jax.Array,
    out_sem: jax.Array,
    dx_sem: jax.Array,
    zero_buf: jax.Array,
    zero_sem: jax.Array,
    *,
    num_experts: int,
    size_d: int,
    tile_rows: int,
    bucket_rows: int,
    tile_d: int,
    has_tile_exps: bool = False,
    dgrad_tile_rows: int = 256,
):
  """Fused gate-backward kernel body when there is no `dw_acc` input."""
  _kernel_impl(
      group_sizes_ref,
      dx_scale_ref,
      dw_scale_ref,
      dgate_exp_ref,
      x_hbm,
      dgq_hbm,
      w_hbm,
      dx_hbm,
      dw_hbm,
      num_gm_ref,
      num_wload_ref,
      gm_to_expert_ref,
      gm_to_m_ref,
      gm_to_wload_ref,
      wload_to_expert_ref,
      kmax_ref,
      gm_needs_shift_ref,
      gm_shifts_ref,
      dgq_buf,
      x_buf,
      w_buf,
      acc_ref,
      out_buf,
      dx_buf,
      in_sem,
      w_sem,
      out_sem,
      dx_sem,
      accin_hbm=None,
      accin_sem=None,
      zero_buf=zero_buf,
      zero_sem=zero_sem,
      num_experts=num_experts,
      size_d=size_d,
      tile_rows=tile_rows,
      bucket_rows=bucket_rows,
      tile_d=tile_d,
      has_tile_exps=has_tile_exps,
      dgrad_tile_rows=dgrad_tile_rows,
  )


def _kernel_with_acc(
    group_sizes_ref: jax.Array,
    dx_scale_ref: jax.Array,
    dw_scale_ref: jax.Array,
    dgate_exp_ref: jax.Array,
    x_hbm: jax.Array,
    dgq_hbm: jax.Array,
    w_hbm: jax.Array,
    accin_hbm: jax.Array,
    dx_hbm: jax.Array,
    dw_hbm: jax.Array,
    num_gm_ref: jax.Array,
    num_wload_ref: jax.Array,
    gm_to_expert_ref: jax.Array,
    gm_to_m_ref: jax.Array,
    gm_to_wload_ref: jax.Array,
    wload_to_expert_ref: jax.Array,
    kmax_ref: jax.Array,
    gm_needs_shift_ref: jax.Array,
    gm_shifts_ref: jax.Array,
    dgq_buf: jax.Array,
    x_buf: jax.Array,
    w_buf: jax.Array,
    acc_ref: jax.Array,
    out_buf: jax.Array,
    dx_buf: jax.Array,
    in_sem: jax.Array,
    w_sem: jax.Array,
    out_sem: jax.Array,
    dx_sem: jax.Array,
    accin_sem: jax.Array,
    *,
    num_experts: int,
    size_d: int,
    tile_rows: int,
    bucket_rows: int,
    tile_d: int,
    has_tile_exps: bool = False,
    dgrad_tile_rows: int = 256,
):
  """Fused gate-backward kernel body that adds `dw_acc` into `dW`."""
  _kernel_impl(
      group_sizes_ref,
      dx_scale_ref,
      dw_scale_ref,
      dgate_exp_ref,
      x_hbm,
      dgq_hbm,
      w_hbm,
      dx_hbm,
      dw_hbm,
      num_gm_ref,
      num_wload_ref,
      gm_to_expert_ref,
      gm_to_m_ref,
      gm_to_wload_ref,
      wload_to_expert_ref,
      kmax_ref,
      gm_needs_shift_ref,
      gm_shifts_ref,
      dgq_buf,
      x_buf,
      w_buf,
      acc_ref,
      out_buf,
      dx_buf,
      in_sem,
      w_sem,
      out_sem,
      dx_sem,
      accin_hbm=accin_hbm,
      accin_sem=accin_sem,
      zero_buf=None,
      zero_sem=None,
      num_experts=num_experts,
      size_d=size_d,
      tile_rows=tile_rows,
      bucket_rows=bucket_rows,
      tile_d=tile_d,
      has_tile_exps=has_tile_exps,
      dgrad_tile_rows=dgrad_tile_rows,
  )


def gate_bwd_fused(
    x3d: jax.Array,
    w_q: jax.Array,
    group_sizes: jax.Array,
    dgate_q: jax.Array,
    *,
    dx_scale: jax.Array,
    dw_scale: jax.Array,
    dgate_tile_exps: jax.Array | None = None,
    dgrad_tile_rows: int = 256,
    dw_acc: jax.Array | None = None,
    dw_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    tiles: FusedTiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Fused FP8 gate backward kernel returning `(dx3d, dw_gate)`.

  Args:
    x3d: Routed FP8 tokens in 3D layout `[M, D // 128, 128]`.
    w_q: Quantized FP8 gate weights `[E, D, F]`.
    group_sizes: Token counts per local expert `int32[E]`.
    dgate_q: Pre-quantized FP8 (`e5m2`) cotangent `[M, F]`.
    dx_scale: Per-tensor f32 scale for `dx` (`dgate` scale * `w_q` scale, or
      `w_q` scale when `dgate_tile_exps` is provided).
    dw_scale: Per-tensor f32 scale for `dW` (`x3d` scale * `dgate` scale, or
      `x3d` scale when `dgate_tile_exps` is provided).
    dgate_tile_exps: Optional `[max_gd]` int32 power-of-two scale exponents of
      `dgate_q` tiles (from `linear_bwd_dgrad`).
    dgrad_tile_rows: Tile row count of `linear_bwd_dgrad` when `dgate_tile_exps`
      is provided.
    dw_acc: Optional `[E, D, F]` weight-gradient accumulator aliased in place.
    dw_dtype: Output dtype of `dw_gate` (`bfloat16` or `float32`).
    tiles: Optional `FusedTiles` override.
    vmem_limit_bytes: Optional Pallas VMEM limit override.

  Returns:
    `(dx3d, dw_gate)` with `dx3d` `bfloat16[M, D // 128, 128]` and `dw_gate`
    `dw_dtype[E, D, F]`.
  """
  if tiles is None:
    tiles = DEFAULT_TILES
  dw_dtype = jnp.dtype(dw_dtype)
  if not is_supported(x3d, w_q, group_sizes, dgate_q, dw_dtype=dw_dtype, tiles=tiles):
    raise ValueError(
        "Unsupported shapes/dtypes for gate_bwd_fused:"
        f" x3d={x3d.shape}/{x3d.dtype}, w_q={w_q.shape}/{w_q.dtype},"
        f" group_sizes={group_sizes.shape}, dgate_q={dgate_q.shape}/"
        f"{dgate_q.dtype}, dw_dtype={dw_dtype}, tiles={tiles}."
    )

  size_m, d0, _ = x3d.shape
  size_d = d0 * LANES
  num_experts, _, size_f = w_q.shape
  tile_rows = min(tiles.tile_rows, size_m)
  has_tile_exps = dgate_tile_exps is not None
  if has_tile_exps:
    dgrad_tile_rows = min(dgrad_tile_rows, size_m)
    tile_rows = max(dgrad_tile_rows, tile_rows // dgrad_tile_rows * dgrad_tile_rows)
    max_gd = num_experts + pl.cdiv(size_m, dgrad_tile_rows)
    if dgate_tile_exps.shape != (max_gd,):
      raise ValueError(f"dgate_tile_exps must have shape ({max_gd},); got" f" {dgate_tile_exps.shape}.")
  bucket_rows = math.gcd(tile_rows, tiles.bucket_rows)
  tile_d = min(tiles.tile_d, size_d)
  tile_d0 = tile_d // LANES
  num_d = size_d // tile_d
  max_gm = num_experts + pl.cdiv(size_m, tile_rows)

  if dw_acc is not None:
    want_shape = (num_experts, size_d, size_f)
    if dw_acc.shape != want_shape or dw_acc.dtype != dw_dtype:
      raise ValueError(
          f"dw_acc must have shape {want_shape} and dtype {dw_dtype}; got" f" {dw_acc.shape}/{dw_acc.dtype}."
      )

  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)

  num_gm = _count_gm_tiles(group_sizes, tile_rows)
  grid_gm = jnp.maximum(num_gm, 1)

  group_sizes_i32 = group_sizes.astype(jnp.int32)
  dx_scale_1d = dx_scale.astype(jnp.float32).reshape(1)
  dw_scale_1d = dw_scale.astype(jnp.float32).reshape(1)
  dgate_exp_1d = dgate_tile_exps.astype(jnp.int32) if has_tile_exps else jnp.zeros((1,), jnp.int32)
  scalars = (group_sizes_i32, dx_scale_1d, dw_scale_1d, dgate_exp_1d)

  w_flat = w_q.reshape(num_experts * size_d, size_f)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)

  dx_out_struct = jax.ShapeDtypeStruct(
      x3d.shape,
      jnp.bfloat16,
      manual_axis_type=jax.typeof(x3d).manual_axis_type,
  )
  dw_mat = jax.typeof(dw_acc).manual_axis_type if dw_acc is not None else jax.typeof(x3d).manual_axis_type
  dw_out_struct = jax.ShapeDtypeStruct(
      (num_experts * size_d, size_f),
      dw_dtype,
      manual_axis_type=dw_mat,
  )

  ratio = max(1, tile_rows // dgrad_tile_rows)
  scratch_shapes = [
      pltpu.SMEM((1,), jnp.int32),  # num_gm_ref
      pltpu.SMEM((1,), jnp.int32),  # num_wload_ref
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_expert_ref
      pltpu.SMEM((max_gm + 1,), jnp.int32),  # gm_to_m_ref
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_wload_ref
      pltpu.SMEM((num_experts,), jnp.int32),  # wload_to_expert_ref
      pltpu.SMEM((num_experts,), jnp.int32),  # kmax_ref
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_needs_shift_ref
      pltpu.SMEM((max_gm, ratio), jnp.int32),  # gm_shifts_ref
      pltpu.VMEM((2, tile_rows, size_f), dgate_q.dtype),  # dgq_buf
      pltpu.VMEM((2, tile_rows, tile_d0, LANES), x3d.dtype),  # x_buf
      pltpu.VMEM((2, tile_d, size_f), w_q.dtype),  # w_buf
      pltpu.VMEM((tile_d, size_f), jnp.float32),  # acc_ref
      pltpu.VMEM((2, tile_d, size_f), dw_dtype),  # out_buf (also holds accin)
      pltpu.VMEM((2, tile_rows, tile_d0, LANES), jnp.bfloat16),  # dx_buf
      pltpu.SemaphoreType.DMA((2, 2)),  # in_sem
      pltpu.SemaphoreType.DMA((2,)),  # w_sem
      pltpu.SemaphoreType.DMA((2,)),  # out_sem
      pltpu.SemaphoreType.DMA((2,)),  # dx_sem
  ]

  flops = 4 * size_m * size_d * size_f
  bytes_accessed = (
      size_m * size_d * x3d.dtype.itemsize
      + num_d * size_m * size_f * dgate_q.dtype.itemsize
      + num_experts * size_d * size_f * w_q.dtype.itemsize
      + size_m * size_d * jnp.dtype(jnp.bfloat16).itemsize
      + num_experts * size_d * size_f * dw_dtype.itemsize * (2 if dw_acc is not None else 1)
  )
  cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)
  exps_tag = "-exps" if has_tile_exps else ""
  acc_tag = "-acc" if dw_acc is not None else ""
  scope_name = (
      f"gate_bwd_fused-g_{num_experts}-m_{size_m}-d_{size_d}-f_{size_f}"
      f"-tr_{tile_rows}-tb_{bucket_rows}-td_{tile_d}{exps_tag}{acc_tag}"
  )
  kernel_kwargs = dict(
      num_experts=num_experts,
      size_d=size_d,
      tile_rows=tile_rows,
      bucket_rows=bucket_rows,
      tile_d=tile_d,
      has_tile_exps=has_tile_exps,
      dgrad_tile_rows=dgrad_tile_rows,
  )
  compiler_params = pltpu.CompilerParams(
      vmem_limit_bytes=vmem_limit_bytes,
      disable_bounds_checks=True,
      dimension_semantics=("arbitrary", "arbitrary"),
  )

  if dw_acc is not None:
    dw_acc_flat = dw_acc.reshape(num_experts * size_d, size_f)
    inputs = (x3d, dgate_q, w_flat, dw_acc_flat)
    scratch_shapes = scratch_shapes + [
        pltpu.SemaphoreType.DMA((2,)),  # accin_sem
    ]
    dx_out, dw_flat_out = pl.pallas_call(
        functools.partial(_kernel_with_acc, **kernel_kwargs),
        out_shape=(dx_out_struct, dw_out_struct),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(num_d, grid_gm),
            in_specs=[hbm_spec, hbm_spec, hbm_spec, hbm_spec],
            out_specs=(hbm_spec, hbm_spec),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        input_output_aliases={len(jax.tree.leaves((scalars, inputs))) - 1: 1},
        name=scope_name,
        cost_estimate=cost_estimate,
    )(*scalars, *inputs)
  else:
    inputs = (x3d, dgate_q, w_flat)
    scratch_shapes = scratch_shapes + [
        pltpu.VMEM((_ZERO_ROWS, size_f), dw_dtype),  # zero_buf
        pltpu.SemaphoreType.DMA(()),  # zero_sem
    ]
    dx_out, dw_flat_out = pl.pallas_call(
        functools.partial(_kernel_no_acc, **kernel_kwargs),
        out_shape=(dx_out_struct, dw_out_struct),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(num_d, grid_gm),
            in_specs=[hbm_spec, hbm_spec, hbm_spec],
            out_specs=(hbm_spec, hbm_spec),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=compiler_params,
        name=scope_name,
        cost_estimate=cost_estimate,
    )(*scalars, *inputs)

  return dx_out, dw_flat_out.reshape(num_experts, size_d, size_f)
