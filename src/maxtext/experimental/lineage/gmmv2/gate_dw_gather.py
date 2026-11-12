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

"""Gathered bf16 gate weight-gradient TGMM of the DSv3 MoE.

Computes ``dW[e] = (dw_acc[e] if given else 0) + sum_{m in e} x[m]^T @ dgate[m]``
where row ``m`` of expert ``e`` is gathered directly from ``x_ag[indices[start
+ m]]`` (`[N, D // 128, 128]` bf16), eliminating both the standalone
``ops.ragged_gather_tc(x_ag)`` copy and the extra passes over ``dgate`` caused
by 3D relayout tiling in generic ``tgmm_v2``.

When ``dw_dtype == jnp.float32``, accumulation happens directly inside
``out_buf`` (no separate ``acc_ref`` or VPU copy/add loop), which halves output
VMEM pressure and allows either single-pass ``(tile_k=D, tile_n=F_gate)`` with
K-strip pipelined output/accin DMAs or full double-buffered ``out_buf``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.experimental.lineage.gmmv2 import linear_bwd_split
from maxtext.experimental.lineage.gmmv2 import relayout

LANES = 128
_SUBLANE = 32  # Row alignment of 2D dgate HBM slices.
_IDX_ALIGN = 128  # Alignment of dynamic 1D int32 HBM slices.
_ISSUE_UNROLL = 16  # Row gather DMAs issued per loop iteration.
_ZERO_ROWS = 32  # Row chunk for zero-filling empty experts.


@dataclasses.dataclass(frozen=True)
class Tiles:
  """Tile sizes of `gate_dw_gather`.

  Attributes:
    tile_rows: Rows per expert-aligned tile (multiple of 32 and `bucket_rows`).
    bucket_rows: Granularity of the matmul rows of partial tiles.
    tile_k: Columns of `x_ag` (out of D) per accumulator tile; defaults to `D`.
      `tile_k // 128` must be a multiple of 8, or `tile_k == D`.
    tile_n: Columns of `dgate` (out of F_gate) per accumulator tile; defaults to
      the largest divisor of F_gate (multiple of 128) that fits in VMEM.
    k_subtile: Output/accin DMA strip size (and default compute K sub-block) of
      `tile_k` in VMEM.
    comp_k: Optional compute K sub-block dividing `k_subtile` (multiple of 256).
      Defaults to a power-of-two factor keeping the `dot_general` result within
      VMEM spill headroom.
    comp_n: Optional compute N sub-block dividing `tile_n` (multiple of 128).
      Defaults to `tile_n`.
  """

  tile_rows: int = 256
  bucket_rows: int = 128
  tile_k: int | None = None
  tile_n: int | None = None
  k_subtile: int = 1024
  comp_k: int | None = None
  comp_n: int | None = None


DEFAULT_TILES = Tiles()


def _cdiv(a, b):
  return (a + b - 1) // b


def _align_to(a: int, b: int) -> int:
  return _cdiv(a, b) * b


def _floor_align(x: jax.Array) -> jax.Array:
  return (x // _SUBLANE) * _SUBLANE


def _load_3d_slice_as_2d(ref, tm: int, tile_d0: int, c_start: int, c_count: int) -> jax.Array:
  """Loads columns `[c_start, c_start + c_count)` of 3D VMEM `ref` as 2D."""
  if c_start == 0 and c_count == tile_d0:
    return relayout.load_3d_as_2d(ref, tm, tile_d0)
  x2d, rows_per_tok, pack = relayout._flat_u32_view(ref, tile_d0)  # pylint: disable=protected-access
  assert tm % pack == 0 and c_start % pack == 0 and c_count % pack == 0
  stride = pack * rows_per_tok
  cols = []
  for c in range(c_start, c_start + c_count, pack):
    words = [x2d[pl.ds(r * rows_per_tok + c // pack, tm // pack, stride=stride), :] for r in range(pack)]
    for out in relayout._transpose_words(words):  # pylint: disable=protected-access
      cols.append(pltpu.bitcast(out, ref.dtype))
  return jnp.concatenate(cols, axis=-1)


def _choose_comp_kn(
    k_subtile: int,
    tile_n: int,
    strip_pipe: bool,
    requested_comp_k: int | None,
    requested_comp_n: int | None,
) -> tuple[int, int]:
  """Selects `(comp_k, comp_n)` keeping `dot_general` result small in VMEM."""
  cn = tile_n if requested_comp_n is None else min(requested_comp_n, tile_n)
  if tile_n % cn or cn % LANES:
    raise ValueError(f"Unsupported comp_n={cn} for tile_n={tile_n}.")
  if requested_comp_k is not None:
    ck = min(requested_comp_k, k_subtile)
    if k_subtile % ck or (ck // LANES) % 2:
      raise ValueError(f"Unsupported comp_k={ck} for k_subtile={k_subtile}.")
    return ck, cn
  max_res_bytes = (5 * 512 * 1024) if strip_pipe else (5 * 1024 * 1024)
  ck = k_subtile
  while ck * cn * 4 > max_res_bytes and ck >= 512 and (ck // 2) % 256 == 0:
    if k_subtile % (ck // 2) == 0:
      ck //= 2
    else:
      break
  if ck * cn * 4 > max_res_bytes and requested_comp_n is None:
    if cn % 2 == 0 and (cn // 2) % LANES == 0:
      cn //= 2
  return ck, cn


def _choose_tile_n_and_bufs(
    size_d: int,
    size_f_gate: int,
    tile_k: int,
    tile_rows: int,
    dw_bytes: int,
    in_place_acc: bool,
    has_acc: bool,
    vmem_limit_bytes: int,
    requested_tile_n: int | None,
) -> tuple[int, int]:
  """Selects `(tile_n, out_bufs)` fitting within `vmem_limit_bytes`."""
  # bf16 3D VMEM buffers align the second-minor dim `tile_k // 128` to 8.
  tk_vmem = _align_to(tile_k // LANES, 8) * LANES
  fixed = 2 * tile_rows * tk_vmem * 2 + 1024 * 1024  # gbuf (bf16, double-buffered)  # compiler scratch headroom

  def _bytes_needed(tn: int, out_bufs: int) -> int:
    acc = 0 if in_place_acc else tile_k * tn * 4
    out = out_bufs * tile_k * tn * dw_bytes
    dgate = 2 * tile_rows * tn * 2
    zero = 0 if has_acc else min(size_d, _ZERO_ROWS) * size_f_gate * dw_bytes
    spill = (5 * 512 * 1024) if (in_place_acc and out_bufs == 1) else (3 * 1024 * 1024)
    return fixed + acc + out + dgate + zero + spill

  if requested_tile_n is not None:
    tn = min(requested_tile_n, size_f_gate)
    if _bytes_needed(tn, 2) <= vmem_limit_bytes:
      return tn, 2
    if _bytes_needed(tn, 1) <= vmem_limit_bytes:
      return tn, 1
    raise ValueError(
        f"Requested tile_n={tn} exceeds VMEM limit ({_bytes_needed(tn, 1)} >"
        f" {vmem_limit_bytes}) for tile_k={tile_k}, tile_rows={tile_rows}."
    )

  # Prefer double-buffered output (hides cross-expert output/accin DMAs) over
  # single-buffered strip pipelining, then largest tile_n (fewest x_ag passes).
  candidates = [
      size_f_gate // d
      for d in range(1, size_f_gate // LANES + 1)
      if size_f_gate % d == 0 and (size_f_gate // d) % LANES == 0
  ]
  for out_bufs in (2, 1):
    for tn in candidates:
      if _bytes_needed(tn, out_bufs) <= vmem_limit_bytes:
        return tn, out_bufs
  raise ValueError(
      f"No tile_n fits in VMEM ({vmem_limit_bytes} B) for D={size_d},"
      f" F_gate={size_f_gate}, tile_k={tile_k}, tile_rows={tile_rows}."
  )


@dataclasses.dataclass(frozen=True)
class _Static:
  """Static parameters of `_kernel`."""

  num_experts: int
  size_d: int
  size_f_gate: int
  tile_rows: int
  bucket_rows: int
  tile_k: int
  tile_n: int
  k_subtile: int
  comp_k: int
  comp_n: int
  num_k: int
  num_n: int
  out_bufs: int
  in_place_acc: bool
  num_strips: int
  idx_window: int
  max_win_start: int
  has_acc: bool
  zero_rows: int


def _kernel(*refs, st: _Static):
  """Kernel body over grid `(num_n, num_k, max(num_tiles, 1))`."""
  if st.has_acc:
    gs_ref, start_ref, idx_hbm, x_hbm, dgate_hbm, accin_hbm, dw_hbm, *scratch = refs
  else:
    gs_ref, start_ref, idx_hbm, x_hbm, dgate_hbm, dw_hbm, *scratch = refs
    accin_hbm = None

  (
      num_ref,
      gm_to_m,
      gm_to_expert,
      gm_to_wload,
      wload_to_expert,
      idx_smem,
      idx_sem,
      gbuf,
      row_sem,
      dgate_buf,
      dgate_sem,
      out_buf,
      out_sem,
      *extra,
  ) = scratch
  acc_ref = None
  if not st.in_place_acc:
    acc_ref, *extra = extra
  if st.has_acc:
    (accin_sem,) = extra
    zero_buf = zero_sem = None
  else:
    zero_buf, zero_sem = extra
    accin_sem = None

  n_idx = pl.program_id(0)
  k_idx = pl.program_id(1)
  gm = pl.program_id(2)

  tile_rows = st.tile_rows
  tile_k = st.tile_k
  tile_n = st.tile_n
  k_subtile = st.k_subtile
  comp_k = st.comp_k
  comp_n = st.comp_n
  tile_d0 = tile_k // LANES
  comp_d0 = comp_k // LANES
  num_strips = st.num_strips
  num_k = st.num_k
  num_n = st.num_n
  size_d = st.size_d
  dw_dtype = dw_hbm.dtype
  strip_pipe = st.in_place_acc and st.out_bufs == 1

  def _n_cols(ni):
    if num_n == 1:
      return slice(None)
    return pl.ds(pl.multiple_of(ni * tile_n, tile_n), tile_n)

  def _dw_block(hbm_ref, e, ki, ni):
    r0 = pl.multiple_of(e * size_d + ki * tile_k, tile_k)
    return hbm_ref.at[pl.ds(r0, tile_k), _n_cols(ni)]

  def _dw_strip(hbm_ref, e, ki, ni, s: int):
    r0 = pl.multiple_of(e * size_d + ki * tile_k + s * k_subtile, k_subtile)
    return hbm_ref.at[pl.ds(r0, k_subtile), _n_cols(ni)]

  def _window(g):
    row = start_ref[0] + gm_to_m[g]
    win = jnp.minimum(row // _IDX_ALIGN * _IDX_ALIGN, st.max_win_start)
    return win, row - win

  def _idx_copy(g, wslot):
    win, _ = _window(g)
    return pltpu.make_async_copy(
        idx_hbm.at[pl.ds(pl.multiple_of(win, _IDX_ALIGN), st.idx_window)],
        idx_smem.at[pl.ds(wslot * st.idx_window, st.idx_window)],
        idx_sem.at[wslot],
    )

  def _start_rows(ki, g, slot):
    m_s = gm_to_m[g]
    n_rows = gm_to_m[g + 1] - m_s
    _, win_off = _window(g)
    base = slot * st.idx_window + win_off
    dst0 = m_s % _SUBLANE

    def _copy(src, dst):
      pltpu.make_async_copy(
          x_hbm.at[src * num_k + ki],
          gbuf.at[slot, pl.ds(dst, 1)],
          row_sem.at[slot],
      ).start()

    unroll = _ISSUE_UNROLL
    num_full = n_rows // unroll

    @pl.loop(0, num_full)
    def _(i):
      b = i * unroll
      srcs = [idx_smem[base + b + u] for u in range(unroll)]
      for u in range(unroll):
        _copy(srcs[u], dst0 + b + u)

    @pl.loop(num_full * unroll, n_rows)
    def _(i):
      _copy(idx_smem[base + i], dst0 + i)

  def _wait_rows(g, slot):
    n_rows = gm_to_m[g + 1] - gm_to_m[g]
    for b in reversed(range(tile_rows.bit_length())):
      size = 1 << b

      @pl.when((n_rows & size) != 0)
      def _(size=size):
        rows = gbuf.at[slot, pl.ds(0, size)]
        pltpu.make_async_copy(rows, rows, row_sem.at[slot]).wait()

  def _dgate_copy(ni, g, slot):
    m_s = gm_to_m[g]
    a_s = pl.multiple_of(_floor_align(m_s), _SUBLANE)
    fetch = pl.multiple_of(_cdiv(gm_to_m[g + 1] - a_s, _SUBLANE) * _SUBLANE, _SUBLANE)
    return pltpu.make_async_copy(
        dgate_hbm.at[pl.ds(a_s, fetch), _n_cols(ni)],
        dgate_buf.at[slot, pl.ds(0, fetch), :],
        dgate_sem.at[slot],
    )

  def _out_copy(e, ki, ni, oslot):
    return pltpu.make_async_copy(out_buf.at[oslot], _dw_block(dw_hbm, e, ki, ni), out_sem.at[oslot])

  def _accin_copy(e, ki, ni, oslot):
    assert accin_hbm is not None and accin_sem is not None
    return pltpu.make_async_copy(_dw_block(accin_hbm, e, ki, ni), out_buf.at[oslot], accin_sem.at[oslot])

  def _out_strip_copy(e, ki, ni, s: int):
    k_sl = pl.ds(s * k_subtile, k_subtile)
    return pltpu.make_async_copy(
        out_buf.at[0, k_sl, :],
        _dw_strip(dw_hbm, e, ki, ni, s),
        out_sem.at[s],
    )

  def _accin_strip_copy(e, ki, ni, s: int):
    assert accin_hbm is not None and accin_sem is not None
    k_sl = pl.ds(s * k_subtile, k_subtile)
    return pltpu.make_async_copy(
        _dw_strip(accin_hbm, e, ki, ni, s),
        out_buf.at[0, k_sl, :],
        accin_sem.at[s],
    )

  def _wload_coords(w, num_wload):
    e = wload_to_expert[w % num_wload]
    nk = w // num_wload
    ki = nk % num_k
    ni = nk // num_k
    return e, ki, ni

  def _next_step(ni, ki, g, num_gm):
    g1 = g + 1
    wrap_g = g1 == num_gm
    g_next = jnp.where(wrap_g, 0, g1)
    k1 = ki + jnp.where(wrap_g, 1, 0)
    wrap_k = k1 == num_k
    k_next = jnp.where(wrap_k, 0, k1)
    n_next = ni + jnp.where(wrap_k, 1, 0)
    return n_next, k_next, g_next

  @pl.when(jnp.logical_and(jnp.logical_and(n_idx == 0, k_idx == 0), gm == 0))
  def _prologue():
    num_gm0, num_wload0 = linear_bwd_split._fill_tiles(  # pylint: disable=protected-access
        gs_ref,
        num_experts=st.num_experts,
        tile_rows=tile_rows,
        gm_to_m=gm_to_m,
        gm_to_expert=gm_to_expert,
        gm_to_wload=gm_to_wload,
        wload_to_expert=wload_to_expert,
    )
    num_ref[0] = num_gm0
    num_ref[1] = num_wload0
    # Zero gbuf once so ungathered prefix/suffix padding rows are finite.
    gbuf[...] = jnp.zeros_like(gbuf)

    if not st.has_acc:
      assert zero_buf is not None and zero_sem is not None
      zero_buf[...] = jnp.zeros_like(zero_buf)

      @pl.loop(0, st.num_experts)
      def _(e):
        @pl.when(gs_ref[e] == 0)
        def _():
          @pl.loop(0, size_d, step=st.zero_rows)
          def _(r):
            row = pl.multiple_of(e * size_d + r, st.zero_rows)
            pltpu.make_async_copy(zero_buf, dw_hbm.at[pl.ds(row, st.zero_rows), :], zero_sem).start(priority=1)

    @pl.when(num_gm0 > 0)
    def _():
      if st.in_place_acc and st.out_bufs == 2 and st.has_acc:
        e0, k0, n0 = _wload_coords(0, num_wload0)
        _accin_copy(e0, k0, n0, 0).start()
      idx_c = _idx_copy(0, 0)
      idx_c.start()
      idx_c.wait()
      _start_rows(0, 0, 0)
      _dgate_copy(0, 0, 0).start()
      total0 = num_n * num_k * num_gm0

      @pl.when(total0 > 1)
      def _():
        _, _, g1 = _next_step(0, 0, 0, num_gm0)
        _idx_copy(g1, 1).start()

  num_gm = num_ref[0]
  num_wload = num_ref[1]

  @pl.when(gm < num_gm)
  def _step():
    nk = n_idx * num_k + k_idx
    t = nk * num_gm + gm
    total = num_n * num_k * num_gm
    slot = t % 2
    e = gm_to_expert[gm]
    cur_wload = nk * num_wload + gm_to_wload[gm]
    oslot = cur_wload % st.out_bufs
    prev_gm = jnp.maximum(gm - 1, 0)
    prev2_gm = jnp.maximum(gm - 2, 0)
    next_gm = jnp.minimum(gm + 1, num_gm - 1)
    is_new_expert = jnp.logical_or(gm == 0, e != gm_to_expert[prev_gm])
    is_prev_new = jnp.logical_or(prev_gm == 0, gm_to_expert[prev_gm] != gm_to_expert[prev2_gm])
    is_second = jnp.logical_and(jnp.logical_not(is_new_expert), is_prev_new)
    is_last_of_expert = jnp.logical_or(gm == num_gm - 1, e != gm_to_expert[next_gm])

    # 0. For strip-pipelined accin, overlap strip 0 accin with input wait and
    #    next-tile gather DMA issue.
    if strip_pipe and st.has_acc:

      @pl.when(is_new_expert)
      def _():
        @pl.when(cur_wload >= 1)
        def _():
          pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
          _out_strip_copy(pe, pk, pn, 0).wait()

        _accin_strip_copy(e, k_idx, n_idx, 0).start()

    # 1. Wait for this tile's gathered x_ag rows and dgate rows.
    _wait_rows(gm, slot)
    _dgate_copy(n_idx, gm, slot).wait()

    # 2. For non-in-place double-buffered output, wait slot from 2 wloads ago
    #    and prefetch accin at expert start.
    if not st.in_place_acc and st.out_bufs == 2:

      @pl.when(is_new_expert)
      def _():
        @pl.when(cur_wload >= 2)
        def _():
          pe, pk, pn = _wload_coords(cur_wload - 2, num_wload)
          _out_copy(pe, pk, pn, oslot).wait()

        if st.has_acc:
          _accin_copy(e, k_idx, n_idx, oslot).start()

    # 3. Prefetch the next tile into (1 - slot) and the indices of t + 2.
    @pl.when(t + 1 < total)
    def _():
      nn, nk1, ng = _next_step(n_idx, k_idx, gm, num_gm)
      _idx_copy(ng, 1 - slot).wait()
      _start_rows(nk1, ng, 1 - slot)
      _dgate_copy(nn, ng, 1 - slot).start()

      @pl.when(t + 2 < total)
      def _():
        _, _, ng2 = _next_step(nn, nk1, ng, num_gm)
        _idx_copy(ng2, slot).start()

    m_s = gm_to_m[gm]
    m_e = gm_to_m[gm + 1]
    a_s = _floor_align(m_s)
    lo = m_s - a_s
    hi = m_e - a_s

    def _run_tile(bucket_m: int, masked: bool, is_new: bool, is_last: bool):
      dg = dgate_buf[slot, :bucket_m, :]
      if masked:
        rows = lax.broadcasted_iota(jnp.int32, (bucket_m, 1), 0)
        valid = jnp.logical_and(rows >= lo, rows < hi)
        dg = jnp.where(valid, dg, jnp.zeros_like(dg))

      def _accum_strip(dst_ref, dst_slot, s: int, overwrite: bool):
        for kk in range(0, k_subtile, comp_k):
          k0 = s * k_subtile + kk
          x_sub = _load_3d_slice_as_2d(gbuf.at[slot], bucket_m, tile_d0, k0 // LANES, comp_d0)
          k_sl = pl.ds(k0, comp_k)
          if comp_n == tile_n:
            res = lax.dot_general(
                x_sub,
                dg,
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            if dst_slot is None:
              if overwrite:
                dst_ref[k_sl, :] = res
              else:
                dst_ref[k_sl, :] += res
            else:
              if overwrite:
                dst_ref[dst_slot, k_sl, :] = res
              else:
                dst_ref[dst_slot, k_sl, :] += res
          else:
            for n0 in range(0, tile_n, comp_n):
              n_sl = pl.ds(n0, comp_n)
              res = lax.dot_general(
                  x_sub,
                  dg[:, n0 : n0 + comp_n],
                  (((0,), (0,)), ((), ())),
                  preferred_element_type=jnp.float32,
              )
              if dst_slot is None:
                if overwrite:
                  dst_ref[k_sl, n_sl] = res
                else:
                  dst_ref[k_sl, n_sl] += res
              else:
                if overwrite:
                  dst_ref[dst_slot, k_sl, n_sl] = res
                else:
                  dst_ref[dst_slot, k_sl, n_sl] += res

      if strip_pipe:
        # In-place f32 accumulator with K-strip pipelined out/accin DMAs.
        for s in range(num_strips):
          if is_new:
            if st.has_acc:
              if s + 1 < num_strips:

                @pl.when(cur_wload >= 1)
                def _(s=s):
                  pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
                  _out_strip_copy(pe, pk, pn, s + 1).wait()

                _accin_strip_copy(e, k_idx, n_idx, s + 1).start()
              _accin_strip_copy(e, k_idx, n_idx, s).wait()
            else:

              @pl.when(cur_wload >= 1)
              def _(s=s):
                pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
                _out_strip_copy(pe, pk, pn, s).wait()

          _accum_strip(out_buf, 0, s, overwrite=is_new and not st.has_acc)
          if is_last:
            _out_strip_copy(e, k_idx, n_idx, s).start()
        return

      if st.in_place_acc:
        # In-place f32 accumulator with double-buffered out_buf (out_bufs == 2).
        if is_new:
          if st.has_acc:
            _accin_copy(e, k_idx, n_idx, oslot).wait()
          else:

            @pl.when(cur_wload >= 2)
            def _():
              pe, pk, pn = _wload_coords(cur_wload - 2, num_wload)
              _out_copy(pe, pk, pn, oslot).wait()

        for s in range(num_strips):
          _accum_strip(out_buf, oslot, s, overwrite=is_new and not st.has_acc)

        if is_last:
          _out_copy(e, k_idx, n_idx, oslot).start()
        return

      # Non-f32 output: accumulate in separate f32 acc_ref, cast on finalize.
      assert acc_ref is not None
      for s in range(num_strips):
        _accum_strip(acc_ref, None, s, overwrite=is_new)

    def _run_full_tile():
      lax.cond(
          is_new_expert,
          lambda: _run_tile(tile_rows, masked=True, is_new=True, is_last=False),
          lambda: _run_tile(tile_rows, masked=False, is_new=False, is_last=False),
      )

    def _run_last_bucket(bucket_m: int):
      lax.cond(
          is_new_expert,
          lambda: _run_tile(bucket_m, masked=True, is_new=True, is_last=True),
          lambda: _run_tile(bucket_m, masked=True, is_new=False, is_last=True),
      )

    num_buckets = tile_rows // st.bucket_rows
    bucket_idx = jnp.maximum(0, (hi - 1) // st.bucket_rows)
    lax.cond(
        is_last_of_expert,
        lambda: lax.switch(
            bucket_idx,
            [functools.partial(_run_last_bucket, bucket_m=st.bucket_rows * (i + 1)) for i in range(num_buckets)],
        ),
        _run_full_tile,
    )

    # For in-place double-buffered accin, wait the previous workload's output
    # store and prefetch the next workload's dw_acc after tile 1 (or tile 0 if
    # the expert has only 1 tile). This gives the previous store ~2 compute
    # tiles to drain and the next load the remaining tiles to arrive.
    if st.in_place_acc and not strip_pipe and st.has_acc:
      prefetch_next = jnp.logical_or(jnp.logical_and(is_new_expert, is_last_of_expert), is_second)
      total_wload = num_n * num_k * num_wload

      @pl.when(jnp.logical_and(prefetch_next, cur_wload + 1 < total_wload))
      def _():
        @pl.when(cur_wload >= 1)
        def _():
          pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
          _out_copy(pe, pk, pn, 1 - oslot).wait()

        ne, nk2, nn2 = _wload_coords(cur_wload + 1, num_wload)
        _accin_copy(ne, nk2, nn2, 1 - oslot).start()

    if not st.in_place_acc:
      assert acc_ref is not None
      if st.out_bufs == 1 and st.has_acc:

        @pl.when(jnp.logical_and(is_new_expert, jnp.logical_not(is_last_of_expert)))
        def _():
          @pl.when(cur_wload >= 1)
          def _():
            pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
            _out_copy(pe, pk, pn, 0).wait()

          _accin_copy(e, k_idx, n_idx, 0).start()

      @pl.when(is_last_of_expert)
      def _finalize():
        if st.out_bufs == 1:
          if st.has_acc:

            @pl.when(is_new_expert)
            def _():
              @pl.when(cur_wload >= 1)
              def _():
                pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
                _out_copy(pe, pk, pn, 0).wait()

              _accin_copy(e, k_idx, n_idx, 0).start()

            _accin_copy(e, k_idx, n_idx, 0).wait()
          else:

            @pl.when(cur_wload >= 1)
            def _():
              pe, pk, pn = _wload_coords(cur_wload - 1, num_wload)
              _out_copy(pe, pk, pn, 0).wait()

        elif st.has_acc:
          _accin_copy(e, k_idx, n_idx, oslot).wait()

        row_step = min(tile_k, 256)

        @pl.loop(0, tile_k, step=row_step)
        def _(r):
          rs = pl.ds(pl.multiple_of(r, row_step), row_step)
          val = acc_ref[rs, :]
          if st.has_acc:
            val = val + out_buf[oslot, rs, :].astype(jnp.float32)
          out_buf[oslot, rs, :] = val.astype(dw_dtype)

        _out_copy(e, k_idx, n_idx, oslot).start()

  is_last_prog = jnp.logical_and(
      jnp.logical_and(n_idx == num_n - 1, k_idx == num_k - 1),
      gm == pl.num_programs(2) - 1,
  )

  @pl.when(is_last_prog)
  def _epilogue():
    @pl.when(num_gm > 0)
    def _():
      last = num_n * num_k * num_wload - 1
      le, lk, ln = _wload_coords(last, num_wload)
      if strip_pipe:
        for s in range(num_strips):
          _out_strip_copy(le, lk, ln, s).wait()
      else:
        _out_copy(le, lk, ln, last % st.out_bufs).wait()
        if st.out_bufs == 2:

          @pl.when(last >= 1)
          def _():
            pe, pk, pn = _wload_coords(last - 1, num_wload)
            _out_copy(pe, pk, pn, (last - 1) % 2).wait()

    if not st.has_acc:
      assert zero_sem is not None
      num_empty = st.num_experts - num_wload

      @pl.when(num_empty > 0)
      def _():
        rows = pl.multiple_of(num_empty * size_d, st.zero_rows)
        pltpu.make_async_copy(dw_hbm.at[pl.ds(0, rows), :], dw_hbm.at[pl.ds(0, rows), :], zero_sem).wait()


def gate_dw_gather(
    x_ag: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    dgate: jax.Array,
    group_sizes: jax.Array,
    *,
    dw_acc: jax.Array | None = None,
    dw_dtype: Any = jnp.float32,
    tiles: Tiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Gathered gate weight gradient `dW[e] = x[e]^T @ dgate[e]`.

  Args:
    x_ag: `[N, D // 128, 128]` bf16 all-gathered input tokens.
    indices: `[num_indices]` int32 row of `x_ag` for each sorted slot.
    start: Sorted slot of chunk row 0.
    dgate: `[M, F_gate]` bf16 cotangent of the gating GMM output.
    group_sizes: `[E]` int32 chunk rows per expert.
    dw_acc: Optional `[E, D, F_gate]` weight gradient to accumulate into, in
      place (`dw_dtype`).
    dw_dtype: Dtype of `dW` (`f32` or `bf16`).
    tiles: Optional tile sizes.
    vmem_limit_bytes: Optional VMEM limit.

  Returns:
    `[E, D, F_gate]` weight gradient in `dw_dtype`.
  """
  if tiles is None:
    tiles = DEFAULT_TILES
  if x_ag.ndim != 3 or x_ag.shape[-1] != LANES or x_ag.dtype != jnp.bfloat16:
    raise ValueError(f"x_ag must be bf16 [N, D // 128, 128]; got {x_ag.shape}/{x_ag.dtype}.")
  if dgate.ndim != 2 or dgate.dtype != jnp.bfloat16:
    raise ValueError(f"dgate must be bf16 [M, F_gate]; got {dgate.shape}/{dgate.dtype}.")
  size_d = x_ag.shape[1] * LANES
  size_m, size_f_gate = dgate.shape
  num_experts = group_sizes.shape[0]
  dw_dtype = jnp.dtype(dw_dtype)
  if size_m % _SUBLANE or size_f_gate % LANES:
    raise ValueError(f"Unsupported dgate shape {dgate.shape}.")
  if dw_acc is not None and (dw_acc.shape != (num_experts, size_d, size_f_gate) or dw_acc.dtype != dw_dtype):
    raise ValueError(
        f"dw_acc must be {(num_experts, size_d, size_f_gate)}/{dw_dtype}; got" f" {dw_acc.shape}/{dw_acc.dtype}."
    )
  tile_rows = min(tiles.tile_rows, size_m)
  bucket_rows = math.gcd(tile_rows, tiles.bucket_rows)
  tile_k = size_d if tiles.tile_k is None else min(tiles.tile_k, size_d)
  tile_d0 = tile_k // LANES
  if size_d % tile_k or tile_d0 % 2 or (tile_d0 % 8 and tile_k != size_d):
    raise ValueError(f"Unsupported tile_k={tile_k} for D={size_d}.")
  k_subtile = min(tiles.k_subtile, tile_k)
  if tile_k % k_subtile or (k_subtile // LANES) % 2:
    k_subtile = tile_k
  if tile_rows % bucket_rows or bucket_rows % _SUBLANE:
    raise ValueError(f"Unsupported tiles {tiles} for M={size_m}.")
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)

  in_place_acc = dw_dtype == jnp.float32
  has_acc = dw_acc is not None
  tile_n, out_bufs = _choose_tile_n_and_bufs(
      size_d=size_d,
      size_f_gate=size_f_gate,
      tile_k=tile_k,
      tile_rows=tile_rows,
      dw_bytes=dw_dtype.itemsize,
      in_place_acc=in_place_acc,
      has_acc=has_acc,
      vmem_limit_bytes=vmem_limit_bytes,
      requested_tile_n=tiles.tile_n,
  )
  if size_f_gate % tile_n or tile_n % LANES:
    raise ValueError(f"Unsupported tile_n={tile_n} for F_gate={size_f_gate}.")

  strip_pipe = in_place_acc and out_bufs == 1
  comp_k, comp_n = _choose_comp_kn(
      k_subtile=k_subtile,
      tile_n=tile_n,
      strip_pipe=strip_pipe,
      requested_comp_k=tiles.comp_k,
      requested_comp_n=tiles.comp_n,
  )
  num_k = size_d // tile_k
  num_n = size_f_gate // tile_n
  num_strips = tile_k // k_subtile
  max_gm = linear_bwd_split.max_tiles(num_experts, size_m, tile_rows)
  idx_window = _cdiv(tile_rows, _IDX_ALIGN) * _IDX_ALIGN + _IDX_ALIGN
  zero_rows = min(size_d, _ZERO_ROWS)

  indices = indices.astype(jnp.int32)
  num_indices = indices.shape[0]
  indices = jnp.pad(
      indices,
      (0, max(-num_indices % _IDX_ALIGN, idx_window - num_indices)),
  )
  max_win_start = indices.shape[0] - idx_window
  x_view = x_ag.reshape(-1, 1, tile_d0, LANES)
  gs = group_sizes.astype(jnp.int32)
  grid_gm = jnp.maximum(linear_bwd_split.num_tiles(gs, tile_rows), 1)
  scalars = (gs, jnp.asarray(start, jnp.int32).reshape(1))

  dw_mat = jax.typeof(dw_acc if has_acc else x_ag).manual_axis_type
  out_shape = jax.ShapeDtypeStruct((num_experts * size_d, size_f_gate), dw_dtype, manual_axis_type=dw_mat)
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  dma = pltpu.SemaphoreType.DMA
  num_out_sems = num_strips if strip_pipe else out_bufs
  scratch_shapes: list[Any] = [
      pltpu.SMEM((2,), jnp.int32),  # num_ref
      pltpu.SMEM((max_gm + 1,), jnp.int32),  # gm_to_m
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_expert
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_wload
      pltpu.SMEM((num_experts,), jnp.int32),  # wload_to_expert
      pltpu.SMEM((2 * idx_window,), jnp.int32),  # idx_smem
      dma((2,)),  # idx_sem
      pltpu.VMEM((2, tile_rows, tile_d0, LANES), x_ag.dtype),  # gbuf
      dma((2,)),  # row_sem
      pltpu.VMEM((2, tile_rows, tile_n), dgate.dtype),  # dgate_buf
      dma((2,)),  # dgate_sem
      pltpu.VMEM((out_bufs, tile_k, tile_n), dw_dtype),  # out_buf
      dma((num_out_sems,)),  # out_sem
  ]
  if not in_place_acc:
    scratch_shapes.append(pltpu.VMEM((tile_k, tile_n), jnp.float32))  # acc_ref
  if has_acc:
    scratch_shapes.append(dma((num_out_sems,)))  # accin_sem
    assert dw_acc is not None
    inputs = (
        indices,
        x_view,
        dgate,
        dw_acc.reshape(num_experts * size_d, size_f_gate),
    )
  else:
    scratch_shapes += [
        pltpu.VMEM((zero_rows, size_f_gate), dw_dtype),  # zero_buf
        dma(()),  # zero_sem
    ]
    inputs = (indices, x_view, dgate)

  st = _Static(
      num_experts=num_experts,
      size_d=size_d,
      size_f_gate=size_f_gate,
      tile_rows=tile_rows,
      bucket_rows=bucket_rows,
      tile_k=tile_k,
      tile_n=tile_n,
      k_subtile=k_subtile,
      comp_k=comp_k,
      comp_n=comp_n,
      num_k=num_k,
      num_n=num_n,
      out_bufs=out_bufs,
      in_place_acc=in_place_acc,
      num_strips=num_strips,
      idx_window=idx_window,
      max_win_start=max_win_start,
      has_acc=has_acc,
      zero_rows=zero_rows,
  )
  flops = 2 * size_m * size_d * size_f_gate
  bytes_accessed = (
      num_n * size_m * size_d * 2
      + num_k * size_m * size_f_gate * 2
      + num_experts * size_d * size_f_gate * dw_dtype.itemsize * (2 if has_acc else 1)
  )
  dw = pl.pallas_call(
      functools.partial(_kernel, st=st),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalars),
          grid=(num_n, num_k, grid_gm),
          in_specs=[hbm] * len(inputs),
          out_specs=hbm,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
      ),
      input_output_aliases={len(jax.tree.leaves((scalars, inputs))) - 1: 0} if has_acc else {},
      cost_estimate=pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0),
      name=(
          f"gate_dw_gather-g_{num_experts}-m_{size_m}-d_{size_d}"
          f"-fg_{size_f_gate}-tr_{tile_rows}-tb_{bucket_rows}"
          f"-tk_{tile_k}-tn_{tile_n}-ob_{out_bufs}" + ("-ip" if in_place_acc else "") + ("-acc" if has_acc else "")
      ),
  )(*scalars, *inputs)
  return dw.reshape(num_experts, size_d, size_f_gate)
