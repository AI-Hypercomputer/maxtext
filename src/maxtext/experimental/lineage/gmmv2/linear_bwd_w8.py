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

"""Weight-only fp8 gathered linear backward + SiLU backward of the DSv3 MoE.

In the backward pass of the weight-only fp8 routed experts (bf16 tokens, fp8
weights dequantized in VMEM, `expert_mlp.WeightOnlyFp8Gmm`), the linear GMM
backward of one chunk is

  grad_routed = ragged_gather(grad_y_ag)                [M, D0, 128] bf16
  act         = silu(g0) * g1 * coeffs  (from the bank)  [M, F]       bf16
  grad_act    = grad_routed @ (w_q[e] * w_scale)^T       [M, F]       bf16
  dW          = act^T @ grad_routed                      [E, F, D]
  grad_gate   = silu_mul_bwd(grad_act, bank, coeffs)     [M, 2F]      bf16

as five kernels with the `grad_act` HBM round trip and two extra passes over
the bank. `linear_bwd_dgrad_w8` fuses everything but the wgrad TGMM into one
kernel (grid over expert-aligned row tiles `gm`, like
`linear_bwd_split.linear_bwd_dgrad`):

* gathers the rows `grad_y_ag[indices[start + m]]` of a tile straight from HBM
  (one DMA per row);
* relayouts them to `[rows, D]` in VMEM (`dy2_buf`), writes them back out in
  2D layout `[M, D]` as `grad_routed` (so the subsequent wgrad TGMM skips all
  3D-to-2D relayouts), and multiplies them with the expert's weight, dequantized
  once per expert to bf16 in VMEM, with an f32 accumulator scaled by `w_scale`
  and rounded to bf16 (like `gmm_v2`);
* runs the SiLU backward on it with the gating activations read from the bank,
  with the same arithmetic as `ops.ragged_silu_mul_bwd`, and rematerializes
  `act` like `ops.ragged_silu_mul_fwd`.

The wgrad `dW = act^T @ grad_routed` is the usual bf16 `tgmm_v2` on the two
2D outputs (see `expert_mlp.WeightOnlyFp8Gmm.linear_bwd_gather_w8`).

Partial row tiles: tiles start at sublane-aligned rows except at expert
boundaries. All four 2D outputs are written in whole 32-row blocks; a tile
leaves its last partial block to the next tile, which merges its leading rows
from the previous tile's VMEM buffer before writing it.
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

from maxtext.experimental.lineage.gmmv2 import gmmv2_kernel
from maxtext.experimental.lineage.gmmv2 import linear_bwd_split
from maxtext.experimental.lineage.gmmv2 import relayout

LANES = 128
_SUBLANE = 32  # Row alignment of the 2D outputs.
_IDX_ALIGN = 128  # Alignment of dynamic 1D int32 HBM slices.
_ISSUE_UNROLL = 16  # Row gather DMAs issued per loop iteration.
_DEQUANT_ROWS = 128  # Weight rows dequantized per loop iteration.
_DEQUANT_COLS = 256


@dataclasses.dataclass(frozen=True)
class Tiles:
  """Tile sizes of `linear_bwd_dgrad_w8`.

  Attributes:
    tile_rows: Rows per tile (a multiple of 32 and of `bucket_rows`).
    bucket_rows: Granularity of the matmul rows of partial tiles.
    tile_n: Output columns of `grad_act` (out of F) per matmul and SiLU epilogue
      step.
    tile_k: Optional gathered columns (out of D) per relayout block; `tile_k //
      128` must be a multiple of 8, or `tile_k` must be `D`. Defaults to `D`
      (single contiguous gather DMA per row, zero-copy view of `dy_ag`).
  """

  tile_rows: int = 256
  bucket_rows: int = 128
  tile_n: int = 256
  tile_k: int | None = None


DEFAULT_TILES = Tiles()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DgradW8Result:
  """Outputs of `linear_bwd_dgrad_w8` (rows past sum(group_sizes) are garbage).

  Attributes:
    grad_gate: `[M, 2F]` bf16 gradient of the gating GMM output.
    grad_coeffs: `[M, 1]` f32 gradient of the routing coefficients.
    act: `[M, F]` bf16 rematerialized activations `silu(g0) * g1 * coeffs`.
    grad_routed: `[M, D]` bf16 gathered and relayouted output gradient rows.
  """

  grad_gate: jax.Array
  grad_coeffs: jax.Array
  act: jax.Array
  grad_routed: jax.Array


def _cdiv(a, b):
  return (a + b - 1) // b


def _floor_align(x: jax.Array) -> jax.Array:
  return (x // _SUBLANE) * _SUBLANE


@dataclasses.dataclass(frozen=True)
class _Static:
  """Static parameters of `_kernel`."""

  num_experts: int
  size_f: int
  size_d: int
  tile_rows: int
  bucket_rows: int
  tile_n: int
  tile_k: int
  num_k: int
  idx_window: int
  max_win_start: int


def _kernel(
    # Scalar prefetch.
    gs_ref,
    start_ref,
    bank_row_ref,
    w_scale_ref,
    # Inputs.
    idx_hbm,
    dy_hbm,
    w_hbm,
    bank_hbm,
    coef_hbm,
    # Outputs.
    ggo_hbm,
    gc_hbm,
    act_hbm,
    dy2_hbm,
    # Scratch.
    num_ref,
    gm_to_m,
    gm_to_expert,
    gm_to_wload,
    wload_to_expert,
    idx_smem,
    idx_sem,
    gbuf,
    row_sem,
    w_buf,
    w_sem,
    wdq_buf,
    bank_buf,
    coef_buf,
    in_sem,
    ggo_buf,
    gc_buf,
    act_buf,
    dy2_buf,
    out_sem,
    *,
    st: _Static,
):
  """Kernel body over grid `(max(num_tiles, 1),)`."""
  gm = pl.program_id(0)
  tile_rows = st.tile_rows
  num_k = st.num_k
  tile_d0 = st.tile_k // LANES
  size_f, size_d = st.size_f, st.size_d

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

  def _start_rows(g, slot):
    """Starts the row DMAs of tile g into gbuf slot, from index window g % 2."""
    m_s = gm_to_m[g]
    n = gm_to_m[g + 1] - m_s
    _, win_off = _window(g)
    base = (g % 2) * st.idx_window + win_off
    dst0 = m_s % _SUBLANE

    def _copy(src, dst):
      pltpu.make_async_copy(
          dy_hbm.at[pl.ds(src * num_k, num_k)],
          gbuf.at[pl.ds(slot * num_k, num_k), pl.ds(dst, 1)],
          row_sem.at[slot],
      ).start()

    unroll = _ISSUE_UNROLL
    num_full = n // unroll

    @pl.loop(0, num_full)
    def _(i):
      b = i * unroll
      srcs = [idx_smem[base + b + u] for u in range(unroll)]
      for u in range(unroll):
        _copy(srcs[u], dst0 + b + u)

    @pl.loop(num_full * unroll, n)
    def _(i):
      _copy(idx_smem[base + i], dst0 + i)

  def _wait_rows(g, slot):
    """Waits for the row DMAs of tile g."""
    n = gm_to_m[g + 1] - gm_to_m[g]
    for b in reversed(range(tile_rows.bit_length())):
      size = 1 << b

      @pl.when((n & size) != 0)
      def _(size=size):
        rows = gbuf.at[pl.ds(slot * num_k, num_k), pl.ds(0, size)]
        pltpu.make_async_copy(rows, rows, row_sem.at[slot]).wait()

  def _in_copies(g, slot):
    m_s = gm_to_m[g]
    a_s = pl.multiple_of(_floor_align(m_s), _SUBLANE)
    fetch = pl.multiple_of(_cdiv(gm_to_m[g + 1] - a_s, _SUBLANE) * _SUBLANE, _SUBLANE)
    bank_c = pltpu.make_async_copy(
        bank_hbm.at[pl.ds(pl.multiple_of(bank_row_ref[0] + a_s, _SUBLANE), fetch)],
        bank_buf.at[slot, pl.ds(0, fetch)],
        in_sem.at[slot, 0],
    )
    coef_c = pltpu.make_async_copy(
        coef_hbm.at[pl.ds(a_s, fetch)],
        coef_buf.at[slot, pl.ds(0, fetch)],
        in_sem.at[slot, 1],
    )
    return bank_c, coef_c

  def _w_copy(e):
    return pltpu.make_async_copy(
        w_hbm.at[pl.ds(pl.multiple_of(e * size_f, size_f), size_f)],
        w_buf.at[0],
        w_sem.at[0],
    )

  out_pairs = (
      (ggo_buf, ggo_hbm),
      (gc_buf, gc_hbm),
      (act_buf, act_hbm),
      (dy2_buf, dy2_hbm),
  )

  def _out_copies(g, slot):
    """2D output DMAs of tile g, and their row count (may be 0)."""
    m_s = gm_to_m[g]
    m_e = gm_to_m[g + 1]
    a_s = pl.multiple_of(_floor_align(m_s), _SUBLANE)
    is_last = g == num_ref[0] - 1
    end = jnp.where(is_last, _cdiv(m_e, _SUBLANE) * _SUBLANE, _floor_align(m_e))
    n = pl.multiple_of(end - a_s, _SUBLANE)
    copies = [
        pltpu.make_async_copy(
            buf.at[slot, pl.ds(0, n)],
            hbm.at[pl.ds(a_s, n)],
            out_sem.at[slot, j],
        )
        for j, (buf, hbm) in enumerate(out_pairs)
    ]
    return copies, n

  def _start_outs(g, slot):
    copies, n = _out_copies(g, slot)

    @pl.when(n > 0)
    def _():
      for c in copies:
        c.start()

  def _wait_outs(g, slot):
    copies, n = _out_copies(g, slot)

    @pl.when(n > 0)
    def _():
      for c in copies:
        c.wait()

  @pl.when(gm == 0)
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

    @pl.when(num_gm0 > 0)
    def _():
      _w_copy(wload_to_expert[0]).start()
      idx_c = _idx_copy(0, 0)
      idx_c.start()
      idx_c.wait()
      _start_rows(0, 0)
      for c in _in_copies(0, 0):
        c.start()

      @pl.when(num_gm0 > 1)
      def _():
        _idx_copy(1, 1).start()

  num_gm = num_ref[0]
  num_wload = num_ref[1]

  @pl.when(gm < num_gm)
  def _step():
    slot = gm % 2
    m_s = gm_to_m[gm]
    m_e = gm_to_m[gm + 1]
    e = gm_to_expert[gm]
    wload = gm_to_wload[gm]
    prev_gm = jnp.maximum(gm - 1, 0)
    is_new_expert = jnp.logical_or(gm == 0, e != gm_to_expert[prev_gm])

    # 1. Wait for this tile's rows, bank rows and coefficients (and weights,
    #    dequantized once per expert; the single fp8 buffer is reused to
    #    prefetch the next expert immediately after dequantization).
    with jax.named_scope("wait_in"):
      _wait_rows(gm, slot)
      for c in _in_copies(gm, slot):
        c.wait()

      @pl.when(is_new_expert)
      def _():
        _w_copy(e).wait()
        with jax.named_scope("dequant"):

          @pl.loop(0, size_f // _DEQUANT_ROWS)
          def _(r):
            r0 = pl.multiple_of(r * _DEQUANT_ROWS, _DEQUANT_ROWS)
            for c in range(0, size_d, _DEQUANT_COLS):
              wdq_buf[pl.ds(r0, _DEQUANT_ROWS), pl.ds(c, _DEQUANT_COLS)] = w_buf[
                  0, pl.ds(r0, _DEQUANT_ROWS), pl.ds(c, _DEQUANT_COLS)
              ].astype(jnp.bfloat16)

        @pl.when(wload + 1 < num_wload)
        def _():
          _w_copy(wload_to_expert[wload + 1]).start()

    # 2. Prefetch the next tile into the other slot (free: tile gm - 1 is
    #    done) and the indices of tile gm + 2 into this tile's index window.
    @pl.when(gm + 1 < num_gm)
    @jax.named_scope("prefetch")
    def _():
      _idx_copy(gm + 1, 1 - slot).wait()
      _start_rows(gm + 1, 1 - slot)
      for c in _in_copies(gm + 1, 1 - slot):
        c.start()

      @pl.when(gm + 2 < num_gm)
      def _():
        _idx_copy(gm + 2, slot).start()

    # 3. The output buffers of this slot were last written by tile gm - 2.
    @pl.when(gm >= 2)
    def _():
      _wait_outs(gm - 2, slot)

    a_s = _floor_align(m_s)
    lo = m_s - a_s
    hi = m_e - a_s

    def _compute(bucket_m: int):
      # 4. Relayout the gathered rows to [rows, D] into dy2_buf[slot].
      with jax.named_scope("relayout"):
        for k in range(num_k):
          dy2_buf[slot, :bucket_m, k * st.tile_k : (k + 1) * st.tile_k] = relayout.load_3d_as_2d(
              gbuf.at[slot * num_k + k], bucket_m, tile_d0
          )
      dy = dy2_buf[slot, :bucket_m, :]
      w_scale = w_scale_ref[0]
      c_val = coef_buf[slot, :bucket_m, :]
      f32 = jnp.float32
      dcoef = jnp.zeros((bucket_m, 1), f32)
      # 5. grad_act = dy @ (w_q[e] * w_scale)^T and the SiLU backward,
      #    tile_n columns of gate_0 / gate_1 at a time.
      for n0 in range(0, size_f, st.tile_n):
        n1 = n0 + st.tile_n
        with jax.named_scope("dact"):
          dact = lax.dot_general(
              dy,
              wdq_buf[n0:n1, :],
              (((1,), (1,)), ((), ())),
              preferred_element_type=f32,
          )
        with jax.named_scope("silu_bwd"):
          # Same arithmetic as `ops._ragged_silu_mul_bwd_kernel` on the
          # bf16-rounded grad_act, and `ops.silu_mul_tile` for act.
          g_val = (dact * w_scale).astype(jnp.bfloat16)
          x_val = bank_buf[slot, :bucket_m, n0:n1]
          y_val = bank_buf[slot, :bucket_m, size_f + n0 : size_f + n1]
          sig_x = jax.nn.sigmoid(x_val)
          silu_x = x_val * sig_x
          dsilu_x = sig_x * (1.0 + x_val * (1.0 - sig_x))
          gc_val = g_val * c_val
          ggo_buf[slot, :bucket_m, n0:n1] = (gc_val * y_val * dsilu_x).astype(ggo_buf.dtype)
          ggo_buf[slot, :bucket_m, size_f + n0 : size_f + n1] = (gc_val * silu_x).astype(ggo_buf.dtype)
          act_buf[slot, :bucket_m, n0:n1] = (silu_x * y_val * c_val).astype(act_buf.dtype)
          dcoef = dcoef + jnp.sum(
              g_val.astype(f32) * silu_x.astype(f32) * y_val.astype(f32),
              axis=-1,
              keepdims=True,
          )
      gc_buf[slot, :bucket_m, :] = dcoef

    num_buckets = tile_rows // st.bucket_rows
    bucket_idx = jnp.maximum(0, (hi - 1) // st.bucket_rows)
    lax.switch(
        bucket_idx,
        [functools.partial(_compute, bucket_m=st.bucket_rows * (i + 1)) for i in range(num_buckets)],
    )

    # 6. Rows [0, lo) belong to the previous tile, whose last partial block it
    #    left to this tile: merge them from its buffers.
    @pl.when(jnp.logical_and(gm > 0, lo > 0))
    @jax.named_scope("merge")
    def _():
      blk = pl.multiple_of(a_s - _floor_align(gm_to_m[prev_gm]), _SUBLANE)
      for buf, _ in out_pairs:
        cur = buf[slot, :_SUBLANE, :]
        prev = buf[1 - slot, pl.ds(blk, _SUBLANE), :]
        r = lax.broadcasted_iota(jnp.int32, cur.shape, 0)
        buf[slot, :_SUBLANE, :] = jnp.where(r < lo, prev, cur)

    # 7. Write the whole 32-row blocks (and, on the last tile, the rest).
    _start_outs(gm, slot)

  @pl.when(gm == pl.num_programs(0) - 1)
  def _epilogue():
    @pl.when(num_gm > 0)
    def _():
      _wait_outs(num_gm - 1, (num_gm - 1) % 2)

      @pl.when(num_gm > 1)
      def _():
        _wait_outs(num_gm - 2, num_gm % 2)


def linear_bwd_dgrad_w8(
    dy_ag: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    w_q: jax.Array,
    group_sizes: jax.Array,
    bank: jax.Array,
    bank_row: jax.Array | int,
    coeffs: jax.Array,
    *,
    w_scale: jax.Array,
    tiles: Tiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> DgradW8Result:
  """Gathered weight-only fp8 linear dgrad fused with the SiLU backward.

  Row m < sum(group_sizes) of the chunk is token `indices[start + m]` of dy_ag,
  routed to the expert of its group. Its gating activations are bank row
  `bank_row + m` and its routing coefficient coeffs[m].

  Args:
    dy_ag: `[N, D // 128, 128]` bf16 cotangent of the top-k reduced outputs.
    indices: `[num_indices]` int32 row of dy_ag of each sorted slot. Requires
      start + sum(group_sizes) <= num_indices.
    start: Sorted slot of chunk row 0.
    w_q: `[E, F, D]` fp8 linear weights.
    group_sizes: `[E]` int32 chunk rows per expert.
    bank: `[R, 2F]` bf16 gating activations (gate_0 | gate_1).
    bank_row: Bank row of chunk row 0, a multiple of 32.
    coeffs: `[M, 1]` routing coefficients of the chunk rows (M the capacity).
    w_scale: `[1, 1]` f32 per-tensor scale of w_q.
    tiles: Optional tile sizes.
    vmem_limit_bytes: Optional VMEM limit.

  Returns:
    The `DgradW8Result`.
  """
  if tiles is None:
    tiles = DEFAULT_TILES
  if dy_ag.ndim != 3 or dy_ag.shape[-1] != LANES:
    raise ValueError(f"dy_ag must be [N, D // 128, 128]; got {dy_ag.shape}.")
  if dy_ag.dtype != jnp.bfloat16 or bank.dtype != jnp.bfloat16:
    raise ValueError(f"dy_ag and bank must be bf16: {dy_ag.dtype}/{bank.dtype}")
  num_experts, size_f, size_d = w_q.shape
  if dy_ag.shape[1] * LANES != size_d or not gmmv2_kernel.is_fp8(w_q.dtype):
    raise ValueError(f"Bad w_q {w_q.shape}/{w_q.dtype} for dy {dy_ag.shape}.")
  if bank.ndim != 2 or bank.shape[1] != 2 * size_f:
    raise ValueError(f"bank must be [R, {2 * size_f}]; got {bank.shape}.")
  size_m = coeffs.shape[0]
  if coeffs.shape != (size_m, 1) or size_m % _SUBLANE:
    raise ValueError(f"coeffs must be [M, 1], M % 32 == 0; got {coeffs.shape}.")
  tile_rows = min(tiles.tile_rows, size_m)
  bucket_rows = math.gcd(tile_rows, tiles.bucket_rows)
  tile_k = size_d if tiles.tile_k is None else min(tiles.tile_k, size_d)
  tile_d0 = tile_k // LANES
  if size_d % tile_k or tile_d0 % 2 or (tile_d0 % 8 and tile_k != size_d):
    raise ValueError(f"Unsupported tile_k {tile_k} for D {size_d}.")
  tile_n = min(tiles.tile_n, size_f)
  if size_f % tile_n or tile_n % LANES or size_f % _DEQUANT_ROWS or size_d % _DEQUANT_COLS:
    raise ValueError(f"Unsupported tile_n {tile_n} for F {size_f}, D {size_d}.")
  if tile_rows % bucket_rows or bucket_rows % _SUBLANE:
    raise ValueError(f"Unsupported tiles {tiles}.")
  num_k = size_d // tile_k
  max_gm = linear_bwd_split.max_tiles(num_experts, size_m, tile_rows)
  idx_window = _cdiv(tile_rows, _IDX_ALIGN) * _IDX_ALIGN + _IDX_ALIGN
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)

  indices = indices.astype(jnp.int32)
  num_indices = indices.shape[0]
  indices = jnp.pad(
      indices,
      (0, max(-num_indices % _IDX_ALIGN, idx_window - num_indices)),
  )
  max_win_start = indices.shape[0] - idx_window
  # [N, D0, 128] -> [N * num_k, 1, tile_d0, 128] (a zero-copy bitcast when
  # tile_k == D), so one DMA gathers all k blocks of a row into k-major VMEM.
  dy_view = dy_ag.reshape(-1, 1, tile_d0, LANES)
  w_flat = w_q.reshape(num_experts * size_f, size_d)
  gs = group_sizes.astype(jnp.int32)
  scalars = (
      gs,
      jnp.asarray(start, jnp.int32).reshape(1),
      jnp.asarray(bank_row, jnp.int32).reshape(1),
      jnp.asarray(w_scale, jnp.float32).reshape(1),
  )
  grid_gm = jnp.maximum(linear_bwd_split.num_tiles(gs, tile_rows), 1)
  mat = jax.typeof(dy_ag).manual_axis_type
  bf16 = jnp.dtype(jnp.bfloat16)
  out_shape = (
      jax.ShapeDtypeStruct((size_m, 2 * size_f), bf16, manual_axis_type=mat),
      jax.ShapeDtypeStruct((size_m, 1), jnp.float32, manual_axis_type=mat),
      jax.ShapeDtypeStruct((size_m, size_f), bf16, manual_axis_type=mat),
      jax.ShapeDtypeStruct((size_m, size_d), bf16, manual_axis_type=mat),
  )
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  dma = pltpu.SemaphoreType.DMA
  scratch_shapes = [
      pltpu.SMEM((2,), jnp.int32),  # num_ref: num tiles, num experts
      pltpu.SMEM((max_gm + 1,), jnp.int32),  # gm_to_m
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_expert
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_wload
      pltpu.SMEM((num_experts,), jnp.int32),  # wload_to_expert
      pltpu.SMEM((2 * idx_window,), jnp.int32),  # idx_smem
      dma((2,)),  # idx_sem
      pltpu.VMEM((2 * num_k, tile_rows, tile_d0, LANES), dy_ag.dtype),  # gbuf
      dma((2,)),  # row_sem
      pltpu.VMEM((1, size_f, size_d), w_q.dtype),  # w_buf
      dma((1,)),  # w_sem
      pltpu.VMEM((size_f, size_d), jnp.bfloat16),  # wdq_buf
      pltpu.VMEM((2, tile_rows, 2 * size_f), bank.dtype),  # bank_buf
      pltpu.VMEM((2, tile_rows, 1), coeffs.dtype),  # coef_buf
      dma((2, 2)),  # in_sem
      pltpu.VMEM((2, tile_rows, 2 * size_f), bf16),  # ggo_buf
      pltpu.VMEM((2, tile_rows, 1), jnp.float32),  # gc_buf
      pltpu.VMEM((2, tile_rows, size_f), bf16),  # act_buf
      pltpu.VMEM((2, tile_rows, size_d), bf16),  # dy2_buf
      dma((2, 4)),  # out_sem
  ]
  st = _Static(
      num_experts=num_experts,
      size_f=size_f,
      size_d=size_d,
      tile_rows=tile_rows,
      bucket_rows=bucket_rows,
      tile_n=tile_n,
      tile_k=tile_k,
      num_k=num_k,
      idx_window=idx_window,
      max_win_start=max_win_start,
  )
  flops = 2 * size_m * size_d * size_f
  bytes_accessed = (
      2 * size_m * size_d * 2  # gather + grad_routed
      + num_experts * size_f * size_d
      + size_m * 2 * size_f * 2 * 2  # bank + grad_gate
      + size_m * size_f * 2  # act
  )
  ggo, gc, act, dy2 = pl.pallas_call(
      functools.partial(_kernel, st=st),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalars),
          grid=(grid_gm,),
          in_specs=[hbm] * 5,
          out_specs=(hbm,) * 4,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          dimension_semantics=("arbitrary",),
      ),
      cost_estimate=pl.CostEstimate(
          flops=flops,
          bytes_accessed=bytes_accessed,
          transcendentals=size_m * size_f,
      ),
      name=(
          f"linear_bwd_dgrad_w8-g_{num_experts}-m_{size_m}-d_{size_d}"
          f"-f_{size_f}-tr_{tile_rows}-tb_{bucket_rows}-tn_{tile_n}"
          f"-tk_{tile_k}"
      ),
  )(*scalars, indices, dy_view, w_flat, bank, coeffs)
  return DgradW8Result(
      grad_gate=ggo,
      grad_coeffs=gc,
      act=act,
      grad_routed=dy2,
  )
