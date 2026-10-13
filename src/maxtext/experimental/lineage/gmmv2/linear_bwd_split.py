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

"""Split FP8 linear backward of the DSv3 routed-expert MLP (gather + SiLU).

In the FP8 routed-expert backward pass (`dsv3_experts.py`), the linear GMM
backward of one chunk used to be

  grad_routed = ragged_gather(grad_y_ag)                 [M, D0, 128] bf16
  act         = silu(g0) * g1 * coeffs  (from the bank)  [M, F]       fp8
  grad_act    = q(grad_routed) @ w_q[e]^T                [M, F]       bf16
  dW          = act^T @ q(grad_routed)                   [E, F, D]
  grad_gate   = silu_mul_bwd(grad_act, bank, coeffs)     [M, 2F]      bf16

as five kernels with the `grad_routed`, `act` and `grad_act` HBM round trips.
This module replaces them with two kernels:

* `linear_bwd_dgrad` (grid over expert-aligned row tiles `gm`): gathers the
  rows `grad_y_ag[indices[start + m]]` of a tile straight from HBM (one DMA
  per row), quantizes them to e5m2 with one dynamic power-of-two scale per tile
  (computed in VMEM), multiplies them with the VMEM-resident `w_q[e]` and runs
  the SiLU backward (including the straight-through mask `|act| <= act_max`)
  in bf16 on the result, like `ops.ragged_silu_mul_bwd`, with `gate_out` read
  from the bank. It writes
  `grad_gate_out` (bf16) and its absmax, `grad_coeffs`, and the two fp8 wgrad
  operands: the quantized `act` and the quantized, relayouted rows `dy_q`
  (`[M, D]` e5m2, half the bytes of `grad_routed`) with the tile exponents.
* `linear_bwd_wgrad` (grid `(D tiles, expert-aligned row tiles)`): accumulates
  `dW[e] = act_q^T @ dy_q` in an f32 VMEM tile per expert, rescaling `dy_q` to
  one power-of-two scale per expert (an exact exponent shift), and adds the
  result into `dw_acc` in place.

Quantization: the e5m2 scale of the gathered rows is `2^ceil(log2(amax /
e5m2_max))` per dgrad tile (rows of one expert), i.e. absmax scaling per tile
rounded up to a power of two. e5m2 rounding is invariant under power-of-two
scales, so the wgrad rescale to the per-expert maximum exponent is exact for
all but values far below the expert's absmax.

Partial row tiles: tiles start at sublane-aligned rows except at expert
boundaries. The 2D fp8 / bf16 outputs are written in whole 32-row blocks; a
tile leaves its last partial block to the next tile, which merges its leading
rows from the previous tile's VMEM buffer before writing it.
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
from maxtext.experimental.lineage.gmmv2 import relayout

LANES = 128
_SUBLANE = 32  # Row alignment of the 2D outputs (8-bit sublane tiling).
_IDX_ALIGN = 128  # Alignment of dynamic 1D int32 HBM slices.
_ISSUE_UNROLL = 16  # Row gather DMAs issued per loop iteration.
_ZERO_ROWS = 128  # Row chunk for zero-filling the dW of empty experts.
_F32_EXP_BIAS = 127
_F32_MANT_BITS = 23
_MIN_EXP = -126  # Smallest normal f32 exponent.


@dataclasses.dataclass(frozen=True)
class DgradTiles:
  """Tile sizes of `linear_bwd_dgrad`.

  Attributes:
    tile_rows: Rows per tile (a multiple of 32 and of `bucket_rows`). Bounded by
      VMEM: two expert weights `[F, D]` stay resident.
    bucket_rows: Granularity of the matmul rows of partial tiles.
    tile_n: Output columns of `grad_act` (out of F) per matmul and SiLU epilogue
      step.
    tile_k: Gathered columns (out of D) per relayout block; `tile_k // 128` must
      be a multiple of 8, or tile_k must be D.
  """

  tile_rows: int = 256
  bucket_rows: int = 128
  tile_n: int = 256
  tile_k: int = 1024


@dataclasses.dataclass(frozen=True)
class WgradTiles:
  """Tile sizes of `linear_bwd_wgrad`.

  Attributes:
    tile_rows: Rows per tile. Must be a multiple of the dgrad `tile_rows`, so
      that every wgrad tile is a union of dgrad tiles.
    bucket_rows: Granularity of the matmul rows of partial tiles.
    tile_d: Columns of dW (out of D) per f32 accumulator.
  """

  tile_rows: int = 1024
  bucket_rows: int = 512
  tile_d: int = 1024


DEFAULT_DGRAD_TILES = DgradTiles()
DEFAULT_WGRAD_TILES = WgradTiles()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DgradResult:
  """Outputs of `linear_bwd_dgrad` (rows past sum(group_sizes) are garbage).

  Attributes:
    grad_gate: `[M, 2F]` bf16 (or `grad_gate_qtype` fp8) gradient of the gating
      GMM output.
    grad_gate_amax: `[1, 1]` f32 absmax of the valid rows of grad_gate (when
      `grad_gate_qtype` is None).
    grad_coeffs: `[M, 1]` f32 gradient of the routing coefficients.
    act_q: `[M, F]` fp8 activations, quantized with the static act scale.
    dy_q: `[M, D]` e5m2 gathered output gradients, quantized per tile (or
      relayouted from pre-quantized `dy_ag`).
    tile_exps: `[max_tiles]` int32 exponent of the scale of each tile of `dy_q`.
    grad_gate_tile_exps: Optional `[max_tiles]` int32 exponent of the scale of
      each tile of `grad_gate` when `grad_gate_qtype` is given.
  """

  grad_gate: jax.Array
  grad_gate_amax: jax.Array
  grad_coeffs: jax.Array
  act_q: jax.Array
  dy_q: jax.Array
  tile_exps: jax.Array
  grad_gate_tile_exps: jax.Array | None = None


def _cdiv(a, b):
  return (a + b - 1) // b


def _floor_align(x: jax.Array) -> jax.Array:
  return (x // _SUBLANE) * _SUBLANE


def max_tiles(num_experts: int, size_m: int, tile_rows: int) -> int:
  """Upper bound of the number of expert-aligned tiles of size_m rows."""
  return num_experts + _cdiv(size_m, tile_rows)


def num_tiles(group_sizes: jax.Array, tile_rows: int) -> jax.Array:
  """Number of expert-aligned tiles, matching `_fill_tiles`."""
  gs = group_sizes.astype(jnp.int32)
  m_off = jnp.cumsum(gs) - gs
  n = jnp.where(gs > 0, (gs + m_off % _SUBLANE + tile_rows - 1) // tile_rows, 0)
  return jnp.sum(n)


def _fill_tiles(
    gs_ref: Any,
    *,
    num_experts: int,
    tile_rows: int,
    gm_to_m: Any,
    gm_to_expert: Any = None,
    gm_to_wload: Any = None,
    wload_to_expert: Any = None,
    gm_to_j: Any = None,
    expert_first: Any = None,
    expert_count: Any = None,
) -> tuple[jax.Array, jax.Array]:
  """Fills SMEM tables of the expert-aligned row tiles.

  Tile j of expert e (rows [o, o + n)) covers rows [max(o, a + j * tile_rows),
  min(o + n, a + (j + 1) * tile_rows)) with a = o rounded down to 32, so tiles
  only start unaligned at expert boundaries.

  Args:
    gs_ref: Group sizes in SMEM.
    num_experts: Number of experts.
    tile_rows: Rows per tile.
    gm_to_m: `[max_tiles + 1]` first row of each tile, and the end.
    gm_to_expert: Optional `[max_tiles]` expert of each tile.
    gm_to_wload: Optional `[max_tiles]` index of the tile's expert among the
      non-empty experts.
    wload_to_expert: Optional `[num_experts]` inverse of gm_to_wload.
    gm_to_j: Optional `[max_tiles]` index of each tile within its expert.
    expert_first: Optional `[num_experts]` first tile of each expert.
    expert_count: Optional `[num_experts]` number of tiles of each expert.

  Returns:
    Tuple of (number of tiles, number of non-empty experts).
  """
  gm_to_m[0] = 0

  def fill_expert(e, carry):
    gm, wload, m_off = carry
    size = gs_ref[e]
    end = m_off + size
    n = jnp.where(size > 0, (size + m_off % _SUBLANE + tile_rows - 1) // tile_rows, 0)
    if expert_first is not None:
      expert_first[e] = gm
      expert_count[e] = n
    if wload_to_expert is not None:

      @pl.when(size > 0)
      def _():
        wload_to_expert[wload] = e

    def fill_tile(i, cur):
      t = gm + i
      nxt = jnp.minimum(_floor_align(cur) + tile_rows, end)
      gm_to_m[t + 1] = nxt
      if gm_to_expert is not None:
        gm_to_expert[t] = e
      if gm_to_wload is not None:
        gm_to_wload[t] = wload
      if gm_to_j is not None:
        gm_to_j[t] = i
      return nxt

    lax.fori_loop(0, n, fill_tile, m_off)
    return gm + n, wload + jnp.where(size > 0, 1, 0), end

  num_gm, num_wload, _ = lax.fori_loop(
      0,
      num_experts,
      fill_expert,
      (jnp.int32(0), jnp.int32(0), jnp.int32(0)),
  )
  return num_gm, num_wload


def _pow2(exp: jax.Array) -> jax.Array:
  """Returns the f32 `2 ** exp` of int32 `exp` in [-126, 127], bit-exact."""
  return lax.bitcast_convert_type((exp + _F32_EXP_BIAS) << _F32_MANT_BITS, jnp.float32)


def _ceil_log2(x: jax.Array) -> jax.Array:
  """Returns int32 `ceil(log2(x))` of f32 x >= 0, clipped to [-126, 126].

  x == 0 gives -126, so that an all-zero tile never sets the largest exponent
  of its expert.

  Args:
    x: Non-negative f32 values.

  Returns:
    The int32 exponents.
  """
  bits = lax.bitcast_convert_type(x, jnp.int32)
  exp = ((bits >> _F32_MANT_BITS) & 0xFF) - _F32_EXP_BIAS
  mant = bits & ((1 << _F32_MANT_BITS) - 1)
  exp = exp + jnp.where(mant != 0, 1, 0)
  return jnp.clip(exp, _MIN_EXP, -_MIN_EXP)


# ------------------------------------------------------------------------------
# dgrad kernel.
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _DgradStatic:
  """Static parameters of `_dgrad_kernel`."""

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
  act_scale: float
  w_scale: float
  act_max: float
  act_qtype: Any
  dy_qtype: Any
  has_dy_scale: bool = False
  ggo_qtype: Any | None = None


def _dgrad_kernel(
    # Scalar prefetch.
    gs_ref,
    start_ref,
    bank_row_ref,
    dy_scale_ref,
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
    dyq_hbm,
    exp_ref,
    amax_ref,
    ggo_exp_ref,
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
    bank_buf,
    coef_buf,
    in_sem,
    ggo_buf,
    gc_buf,
    act_buf,
    dyq_buf,
    out_sem,
    amax_vmem,
    *extra_scratch,
    st: _DgradStatic,
):
  """dgrad kernel body over grid `(max(num_tiles, 1),)`."""
  if st.ggo_qtype is not None:
    (ggo_tmp,) = extra_scratch
  else:
    ggo_tmp = None
  gm = pl.program_id(0)
  tile_rows = st.tile_rows
  num_k = st.num_k
  tile_d0 = st.tile_k // LANES
  size_f = st.size_f

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
    """Waits for the row DMAs of tile g into gbuf slot."""
    n = gm_to_m[g + 1] - gm_to_m[g]
    # A wait only needs the byte count; split the dynamic row count into
    # static power-of-two waits, one per set bit.
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

  def _w_copy(e, wslot):
    return pltpu.make_async_copy(
        w_hbm.at[pl.ds(pl.multiple_of(e * size_f, size_f), size_f)],
        w_buf.at[wslot],
        w_sem.at[wslot],
    )

  out_pairs = (
      (ggo_buf, ggo_hbm),
      (gc_buf, gc_hbm),
      (act_buf, act_hbm),
      (dyq_buf, dyq_hbm),
  )

  def _out_copies(g, slot):
    """Output DMAs of tile g, and their row count (may be 0)."""
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
    num_gm0, num_wload0 = _fill_tiles(
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
    amax_vmem[...] = jnp.zeros_like(amax_vmem)
    if st.has_dy_scale:
      gbuf[...] = jnp.zeros_like(gbuf)

    @pl.when(num_gm0 > 0)
    def _():
      _w_copy(wload_to_expert[0], 0).start()
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
    wslot = wload % 2
    prev_gm = jnp.maximum(gm - 1, 0)
    is_new_expert = jnp.logical_or(gm == 0, e != gm_to_expert[prev_gm])

    # 1. Wait for this tile's rows, bank rows and coefficients (and weights).
    with jax.named_scope("wait_in"):
      _wait_rows(gm, slot)
      for c in _in_copies(gm, slot):
        c.wait()

      @pl.when(is_new_expert)
      def _():
        _w_copy(e, wslot).wait()

        # The other weight slot held the previous expert, which is done.
        @pl.when(wload + 1 < num_wload)
        def _():
          _w_copy(wload_to_expert[wload + 1], 1 - wslot).start()

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
      if st.has_dy_scale:
        exp_ref[gm] = jnp.int32(0)
        dact_scale = dy_scale_ref[0, 0] * st.w_scale
        with jax.named_scope("relayout"):
          for k in range(num_k):
            dyq_buf[slot, :bucket_m, k * st.tile_k : (k + 1) * st.tile_k] = relayout.load_3d_as_2d(
                gbuf.at[slot * num_k + k], bucket_m, tile_d0
            )
      else:
        rpt = tile_d0 // 2  # uint32 rows per token of a bf16 block.
        # 4. Absmax of the valid gathered rows, on the raw bf16 bits: token t
        #    owns uint32 rows [t * rpt, (t + 1) * rpt) of a flat block view.
        with jax.named_scope("absmax"):
          amax_bits = jnp.zeros((1, LANES), jnp.int32)
          for k in range(num_k):
            words = gbuf.at[slot * num_k + k].bitcast(jnp.int32).reshape(-1, LANES)[: bucket_m * rpt, :]
            words = words & 0x7FFF7FFF
            halves = jnp.maximum(words & 0xFFFF, words >> 16)
            r = lax.broadcasted_iota(jnp.int32, halves.shape, 0)
            valid = jnp.logical_and(r >= lo * rpt, r < hi * rpt)
            amax_bits = jnp.maximum(
                amax_bits,
                jnp.max(jnp.where(valid, halves, 0), axis=0, keepdims=True),
            )
          amax_bits = jnp.max(amax_bits, axis=1, keepdims=True)  # [1, 1]
          amax = lax.bitcast_convert_type(amax_bits << 16, jnp.float32)
          dy_qmax = float(jnp.finfo(st.dy_qtype).max)
          exp = _ceil_log2(amax / dy_qmax)
          exp_ref[gm] = jnp.max(exp)
          inv_s = _pow2(-exp)
          dact_scale = _pow2(exp) * st.w_scale

        # 5. Relayout to [rows, D] and quantize with the tile's scale.
        with jax.named_scope("quantize"):
          for k in range(num_k):
            x2d = relayout.load_3d_as_2d(gbuf.at[slot * num_k + k], bucket_m, tile_d0)
            dyq_buf[slot, :bucket_m, k * st.tile_k : (k + 1) * st.tile_k] = (x2d.astype(jnp.float32) * inv_s).astype(
                st.dy_qtype
            )

      # 6. grad_act = dy_q @ w_q[e]^T and the SiLU backward, tile_n columns
      #    of gate_0 / gate_1 at a time.
      dyq = dyq_buf[slot, :bucket_m, :]
      # The SiLU backward runs in bf16 on the bf16-rounded grad_act, like
      # `ops.ragged_silu_mul_bwd`; its coefficient gradient sums in f32.
      ct = jnp.bfloat16
      coef = coef_buf[slot, :bucket_m, :].astype(ct)
      rows = lax.broadcasted_iota(jnp.int32, (bucket_m, 1), 0)
      valid_rows = jnp.logical_and(rows >= lo, rows < hi)
      act_qmax = float(jnp.finfo(st.act_qtype).max)
      dcoef = jnp.zeros((bucket_m, 1), jnp.float32)
      ggo_max = jnp.zeros((1, st.tile_n), jnp.float32)
      for n0 in range(0, size_f, st.tile_n):
        n1 = n0 + st.tile_n
        with jax.named_scope("dact"):
          dact = lax.dot_general(
              dyq,
              w_buf[wslot, n0:n1, :],
              (((1,), (1,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
        with jax.named_scope("silu_bwd"):
          dact = (dact * dact_scale).astype(ct)
          g0 = bank_buf[slot, :bucket_m, n0:n1].astype(ct)
          g1 = bank_buf[slot, :bucket_m, size_f + n0 : size_f + n1].astype(ct)
          sig = jax.nn.sigmoid(g0)
          silu = g0 * sig
          silu_g1 = silu * g1
          act = silu_g1 * coef
          # Straight-through estimator: no gradient through the values that
          # the static act calibration clipped.
          g = jnp.where(jnp.abs(act) <= st.act_max, dact, jnp.zeros_like(dact))
          gc = g * coef
          dg0 = (gc * g1 * (sig * (1.0 + g0 * (1.0 - sig)))).astype(jnp.bfloat16)
          dg1 = (gc * silu).astype(jnp.bfloat16)
          if st.ggo_qtype is not None:
            assert ggo_tmp is not None
            ggo_tmp[:bucket_m, n0:n1] = dg0
            ggo_tmp[:bucket_m, size_f + n0 : size_f + n1] = dg1
          else:
            ggo_buf[slot, :bucket_m, n0:n1] = dg0
            ggo_buf[slot, :bucket_m, size_f + n0 : size_f + n1] = dg1
          # The wgrad kernel only masks dy_q, so act_q must be finite in all
          # the rows it may read: zero outside this tile's rows.
          act_buf[slot, :bucket_m, n0:n1] = jnp.where(
              valid_rows,
              jnp.clip(act * (1.0 / st.act_scale), -act_qmax, act_qmax),
              jnp.zeros_like(act),
          ).astype(st.act_qtype)
          dcoef = dcoef + jnp.sum((g * silu_g1).astype(jnp.float32), axis=1, keepdims=True)
          dg_max = jnp.maximum(jnp.abs(dg0), jnp.abs(dg1)).astype(jnp.float32)
          ggo_max = jnp.maximum(
              ggo_max,
              jnp.max(jnp.where(valid_rows, dg_max, 0.0), axis=0, keepdims=True),
          )
      gc_buf[slot, :bucket_m, :] = dcoef
      if st.ggo_qtype is not None:
        assert ggo_tmp is not None
        with jax.named_scope("quantize_ggo"):
          ggo_amax = jnp.max(ggo_max, axis=1, keepdims=True)  # [1, 1]
          ggo_qmax = float(jnp.finfo(st.ggo_qtype).max)
          ggo_exp = _ceil_log2(ggo_amax / ggo_qmax)
          ggo_exp_ref[gm] = jnp.max(ggo_exp)
          ggo_inv_s = _pow2(-ggo_exp).astype(jnp.bfloat16)
          step_n = min(2 * size_f, max(st.tile_n, 1024))
          for n0 in range(0, 2 * size_f, step_n):
            n1 = min(2 * size_f, n0 + step_n)
            ggo_buf[slot, :bucket_m, n0:n1] = (ggo_tmp[:bucket_m, n0:n1] * ggo_inv_s).astype(st.ggo_qtype)
      else:
        amax_vmem[...] = jnp.maximum(amax_vmem[...], ggo_max)

    num_buckets = tile_rows // st.bucket_rows
    bucket_idx = jnp.maximum(0, (hi - 1) // st.bucket_rows)
    lax.switch(
        bucket_idx,
        [functools.partial(_compute, bucket_m=st.bucket_rows * (i + 1)) for i in range(num_buckets)],
    )

    # 7. Rows [0, lo) belong to the previous tile, whose last partial block it
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

    # 8. Write the whole 32-row blocks (and, on the last tile, the rest).
    _start_outs(gm, slot)

  @pl.when(gm == pl.num_programs(0) - 1)
  def _epilogue():
    @pl.when(num_gm > 0)
    def _():
      _wait_outs(num_gm - 1, (num_gm - 1) % 2)

      @pl.when(num_gm > 1)
      def _():
        _wait_outs(num_gm - 2, num_gm % 2)

    amax_ref[0, 0] = jnp.max(amax_vmem[...])


def _check_dgrad_args(dy_ag, w_q, bank, coeffs, tiles, *, dy_scale, dy_qtype, grad_gate_qtype):
  """Raises if `linear_bwd_dgrad` does not support the arguments."""
  if dy_ag.ndim != 3 or dy_ag.shape[-1] != LANES:
    raise ValueError(f"dy_ag must be [N, D // 128, 128]; got {dy_ag.shape}.")
  if dy_scale is None:
    if dy_ag.dtype != jnp.bfloat16:
      raise ValueError(f"dy_ag must be bfloat16; got {dy_ag.dtype}.")
    dy_pack = 2
  else:
    if dy_ag.dtype != jnp.dtype(dy_qtype):
      raise ValueError(f"dy_ag must have dtype {dy_qtype} when dy_scale is provided; got" f" {dy_ag.dtype}.")
    dy_pack = 4
  if grad_gate_qtype is not None and not gmmv2_kernel.is_fp8(grad_gate_qtype):
    raise ValueError(f"grad_gate_qtype must be fp8; got {grad_gate_qtype}.")
  num_experts, size_f, size_d = w_q.shape
  if dy_ag.shape[1] * LANES != size_d or not gmmv2_kernel.is_fp8(w_q.dtype):
    raise ValueError(f"Bad w_q {w_q.shape}/{w_q.dtype} for dy {dy_ag.shape}.")
  if bank.ndim != 2 or bank.shape[1] != 2 * size_f:
    raise ValueError(f"bank must be [R, {2 * size_f}]; got {bank.shape}.")
  size_m = coeffs.shape[0]
  if coeffs.shape != (size_m, 1) or size_m % _SUBLANE:
    raise ValueError(f"coeffs must be [M, 1], M % 32 == 0; got {coeffs.shape}.")
  tile_k = min(tiles.tile_k, size_d)
  tile_d0 = tile_k // LANES
  if size_d % tile_k or tile_d0 % dy_pack or (tile_d0 % 8 and tile_k != size_d):
    raise ValueError(f"Unsupported tile_k {tile_k} for D {size_d}.")
  tile_n = min(tiles.tile_n, size_f)
  if size_f % tile_n or tile_n % LANES:
    raise ValueError(f"Unsupported tile_n {tile_n} for F {size_f}.")
  if tiles.tile_rows % tiles.bucket_rows or tiles.bucket_rows % _SUBLANE:
    raise ValueError(f"Unsupported dgrad tiles {tiles}.")
  del num_experts


def linear_bwd_dgrad(
    dy_ag: jax.Array,
    indices: jax.Array,
    start: jax.Array | int,
    w_q: jax.Array,
    group_sizes: jax.Array,
    bank: jax.Array,
    bank_row: jax.Array | int,
    coeffs: jax.Array,
    *,
    act_scale: float,
    w_scale: float,
    act_qtype: Any,
    dy_qtype: Any,
    dy_scale: jax.Array | None = None,
    grad_gate_qtype: Any | None = None,
    tiles: DgradTiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> DgradResult:
  """Gathered FP8 linear dgrad fused with the SiLU backward.

  Row m < sum(group_sizes) of the chunk is token `indices[start + m]` of dy_ag,
  routed to the expert of its group. Its gating activations are bank row
  `bank_row + m` and its routing coefficient coeffs[m].

  Args:
    dy_ag: `[N, D // 128, 128]` bf16 (or `dy_qtype` fp8 when `dy_scale` is
      provided) cotangent of the top-k reduced outputs.
    indices: `[num_indices]` int32 row of dy_ag of each sorted slot. Requires
      start + sum(group_sizes) <= num_indices.
    start: Sorted slot of chunk row 0.
    w_q: `[E, F, D]` fp8 linear weights, quantized with `w_scale`.
    group_sizes: `[E]` int32 chunk rows per expert.
    bank: `[R, 2F]` bf16 gating activations (gate_0 | gate_1).
    bank_row: Bank row of chunk row 0, a multiple of 32.
    coeffs: `[M, 1]` routing coefficients of the chunk rows (M the capacity).
    act_scale: Static scale of the activations.
    w_scale: Static scale of w_q.
    act_qtype: fp8 dtype of the activations.
    dy_qtype: fp8 dtype the gathered rows are quantized to (e5m2).
    dy_scale: Optional `[1, 1]` f32 scale of pre-quantized `dy_ag`.
    grad_gate_qtype: Optional fp8 dtype (e.g. e5m2) to quantize `grad_gate` to
      per tile (`2 ** grad_gate_tile_exps[gm]`).
    tiles: Optional tile sizes.
    vmem_limit_bytes: Optional VMEM limit.

  Returns:
    The `DgradResult`.
  """
  if tiles is None:
    tiles = DEFAULT_DGRAD_TILES
  _check_dgrad_args(
      dy_ag,
      w_q,
      bank,
      coeffs,
      tiles,
      dy_scale=dy_scale,
      dy_qtype=dy_qtype,
      grad_gate_qtype=grad_gate_qtype,
  )
  num_experts, size_f, size_d = w_q.shape
  size_m = coeffs.shape[0]
  tile_rows = min(tiles.tile_rows, size_m)
  bucket_rows = math.gcd(tile_rows, tiles.bucket_rows)
  tile_k = min(tiles.tile_k, size_d)
  tile_d0 = tile_k // LANES
  num_k = size_d // tile_k
  tile_n = min(tiles.tile_n, size_f)
  max_gm = max_tiles(num_experts, size_m, tile_rows)
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
  # [N, D0, 128] -> [N * num_k, 1, tile_d0, 128], so that one DMA gathers all
  # k blocks of a row into k-major VMEM blocks. tile_d0 is a multiple of the
  # HBM sublane tiling, so this is a bitcast.
  dy_view = dy_ag.reshape(-1, 1, tile_d0, LANES)
  w_flat = w_q.reshape(num_experts * size_f, size_d)
  gs = group_sizes.astype(jnp.int32)
  has_dy_scale = dy_scale is not None
  dy_scale_scalar = dy_scale.astype(jnp.float32).reshape(1, 1) if has_dy_scale else jnp.zeros((1, 1), jnp.float32)
  scalars = (
      gs,
      jnp.asarray(start, jnp.int32).reshape(1),
      jnp.asarray(bank_row, jnp.int32).reshape(1),
      dy_scale_scalar,
  )
  grid_gm = jnp.maximum(num_tiles(gs, tile_rows), 1)

  ggo_dtype = jnp.dtype(grad_gate_qtype) if grad_gate_qtype is not None else jnp.dtype(jnp.bfloat16)
  mat = jax.typeof(dy_ag).manual_axis_type
  out_shape = (
      jax.ShapeDtypeStruct((size_m, 2 * size_f), ggo_dtype, manual_axis_type=mat),
      jax.ShapeDtypeStruct((size_m, 1), jnp.float32, manual_axis_type=mat),
      jax.ShapeDtypeStruct((size_m, size_f), act_qtype, manual_axis_type=mat),
      jax.ShapeDtypeStruct((size_m, size_d), dy_qtype, manual_axis_type=mat),
      jax.ShapeDtypeStruct((max_gm,), jnp.int32, manual_axis_type=mat),
      jax.ShapeDtypeStruct((1, 1), jnp.float32, manual_axis_type=mat),
      jax.ShapeDtypeStruct((max_gm,), jnp.int32, manual_axis_type=mat),
  )
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  scratch_shapes = [
      pltpu.SMEM((2,), jnp.int32),  # num_ref: num tiles, num experts
      pltpu.SMEM((max_gm + 1,), jnp.int32),  # gm_to_m
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_expert
      pltpu.SMEM((max_gm,), jnp.int32),  # gm_to_wload
      pltpu.SMEM((num_experts,), jnp.int32),  # wload_to_expert
      pltpu.SMEM((2 * idx_window,), jnp.int32),  # idx_smem
      pltpu.SemaphoreType.DMA((2,)),  # idx_sem
      pltpu.VMEM((2 * num_k, tile_rows, tile_d0, LANES), dy_ag.dtype),  # gbuf
      pltpu.SemaphoreType.DMA((2,)),  # row_sem
      pltpu.VMEM((2, size_f, size_d), w_q.dtype),  # w_buf
      pltpu.SemaphoreType.DMA((2,)),  # w_sem
      pltpu.VMEM((2, tile_rows, 2 * size_f), bank.dtype),  # bank_buf
      pltpu.VMEM((2, tile_rows, 1), coeffs.dtype),  # coef_buf
      pltpu.SemaphoreType.DMA((2, 2)),  # in_sem
      pltpu.VMEM((2, tile_rows, 2 * size_f), ggo_dtype),  # ggo_buf
      pltpu.VMEM((2, tile_rows, 1), jnp.float32),  # gc_buf
      pltpu.VMEM((2, tile_rows, size_f), act_qtype),  # act_buf
      pltpu.VMEM((2, tile_rows, size_d), dy_qtype),  # dyq_buf
      pltpu.SemaphoreType.DMA((2, 4)),  # out_sem
      pltpu.VMEM((1, tile_n), jnp.float32),  # amax_vmem
  ]
  if grad_gate_qtype is not None:
    scratch_shapes.append(pltpu.VMEM((tile_rows, 2 * size_f), jnp.bfloat16))  # ggo_tmp
  st = _DgradStatic(
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
      act_scale=act_scale,
      w_scale=w_scale,
      act_max=act_scale * float(jnp.finfo(act_qtype).max),
      act_qtype=jnp.dtype(act_qtype),
      dy_qtype=jnp.dtype(dy_qtype),
      has_dy_scale=has_dy_scale,
      ggo_qtype=jnp.dtype(grad_gate_qtype) if grad_gate_qtype is not None else None,
  )
  flops = 2 * size_m * size_d * size_f
  bytes_accessed = (
      size_m * size_d * (dy_ag.dtype.itemsize + 1)  # gather + dy_q
      + num_experts * size_f * size_d
      + size_m * 2 * size_f * (bank.dtype.itemsize + ggo_dtype.itemsize)
      + size_m * size_f
  )
  ggo, gc, act_q, dy_q, tile_exps, amax, ggo_tile_exps = pl.pallas_call(
      functools.partial(_dgrad_kernel, st=st),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalars),
          grid=(grid_gm,),
          in_specs=[hbm, hbm, hbm, hbm, hbm],
          out_specs=(hbm, hbm, hbm, hbm, smem, smem, smem),
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
          f"linear_bwd_dgrad-g_{num_experts}-m_{size_m}-d_{size_d}-f_{size_f}"
          f"-tr_{tile_rows}-tb_{bucket_rows}-tn_{tile_n}-tk_{tile_k}"
          + ("-dys" if has_dy_scale else "")
          + ("-ggoq" if grad_gate_qtype is not None else "")
      ),
  )(*scalars, indices, dy_view, w_flat, bank, coeffs)
  return DgradResult(
      grad_gate=ggo,
      grad_gate_amax=amax,
      grad_coeffs=gc,
      act_q=act_q,
      dy_q=dy_q,
      tile_exps=tile_exps,
      grad_gate_tile_exps=ggo_tile_exps if grad_gate_qtype is not None else None,
  )


# ------------------------------------------------------------------------------
# wgrad kernel.
# ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _WgradStatic:
  """Static parameters of `_wgrad_kernel`."""

  num_experts: int
  size_f: int
  size_d: int
  tile_rows: int
  bucket_rows: int
  tile_d: int
  dgrad_tile_rows: int
  act_scale: float
  dy_qtype: Any
  has_acc: bool
  has_dy_scale: bool = False


def _wgrad_kernel(*refs, st: _WgradStatic):
  """wgrad kernel body over grid `(num_d, max(num_tiles, 1))`."""
  if st.has_acc:
    (
        gs_ref,
        exp_ref,
        dy_scale_ref,
        act_hbm,
        dyq_hbm,
        accin_hbm,
        dw_hbm,
        *scratch,
    ) = refs
  else:
    gs_ref, exp_ref, dy_scale_ref, act_hbm, dyq_hbm, dw_hbm, *scratch = refs
    accin_hbm = None
  (
      num_ref,
      gw_to_m,
      gw_to_expert,
      gw_to_wload,
      gw_to_j,
      wload_to_expert,
      dg_to_m,
      d_first,
      d_count,
      kmax_ref,
      act_buf,
      dyq_buf,
      acc_ref,
      out_buf,
      in_sem,
      out_sem,
      *extra,
  ) = scratch
  if st.has_acc:
    (accin_sem,) = extra
    zero_buf = zero_sem = None
  else:
    zero_buf, zero_sem = extra
    accin_sem = None

  size_f, size_d, tile_d = st.size_f, st.size_d, st.tile_d
  num_d = size_d // tile_d
  ratio = st.tile_rows // st.dgrad_tile_rows
  dw_dtype = dw_hbm.dtype
  d = pl.program_id(0)
  gw = pl.program_id(1)

  def _d_cols(d_idx):
    if num_d == 1:
      return slice(None)
    return pl.ds(pl.multiple_of(d_idx * tile_d, tile_d), tile_d)

  def _in_copies(d_idx, g, slot):
    m_s = gw_to_m[g]
    a_s = pl.multiple_of(_floor_align(m_s), _SUBLANE)
    fetch = pl.multiple_of(_cdiv(gw_to_m[g + 1] - a_s, _SUBLANE) * _SUBLANE, _SUBLANE)
    act_c = pltpu.make_async_copy(
        act_hbm.at[pl.ds(a_s, fetch), :],
        act_buf.at[slot, pl.ds(0, fetch), :],
        in_sem.at[slot, 0],
    )
    dyq_c = pltpu.make_async_copy(
        dyq_hbm.at[pl.ds(a_s, fetch), _d_cols(d_idx)],
        dyq_buf.at[slot, pl.ds(0, fetch), :],
        in_sem.at[slot, 1],
    )
    return act_c, dyq_c

  def _dw_block(hbm_ref, e, d_idx):
    return hbm_ref.at[pl.ds(pl.multiple_of(e * size_f, size_f), size_f), _d_cols(d_idx)]

  def _out_copy(e, d_idx, oslot):
    return pltpu.make_async_copy(out_buf.at[oslot], _dw_block(dw_hbm, e, d_idx), out_sem.at[oslot])

  def _accin_copy(e, d_idx, oslot):
    assert accin_hbm is not None and accin_sem is not None
    return pltpu.make_async_copy(_dw_block(accin_hbm, e, d_idx), out_buf.at[oslot], accin_sem.at[oslot])

  @pl.when(jnp.logical_and(d == 0, gw == 0))
  def _prologue():
    num_gw0, num_wload0 = _fill_tiles(
        gs_ref,
        num_experts=st.num_experts,
        tile_rows=st.tile_rows,
        gm_to_m=gw_to_m,
        gm_to_expert=gw_to_expert,
        gm_to_wload=gw_to_wload,
        wload_to_expert=wload_to_expert,
        gm_to_j=gw_to_j,
    )
    num_ref[0] = num_gw0
    num_ref[1] = num_wload0
    if not st.has_dy_scale:
      # The dgrad tiles of each expert, and the largest exponent of their
      # scales.
      _fill_tiles(
          gs_ref,
          num_experts=st.num_experts,
          tile_rows=st.dgrad_tile_rows,
          gm_to_m=dg_to_m,
          expert_first=d_first,
          expert_count=d_count,
      )

      @pl.loop(0, st.num_experts)
      def _(e):
        first = d_first[e]

        def body(j, kmax):
          return jnp.maximum(kmax, exp_ref[first + j])

        kmax_ref[e] = lax.fori_loop(0, d_count[e], body, jnp.int32(-(2**30)))

    act_buf[...] = jnp.zeros_like(act_buf)
    if not st.has_acc:
      assert zero_buf is not None and zero_sem is not None
      zero_buf[...] = jnp.zeros_like(zero_buf)

      @pl.loop(0, st.num_experts)
      def _(e):
        @pl.when(gs_ref[e] == 0)
        def _():
          @pl.loop(0, size_f, step=_ZERO_ROWS)
          def _(r):
            row = pl.multiple_of(e * size_f + r, _ZERO_ROWS)
            pltpu.make_async_copy(zero_buf, dw_hbm.at[pl.ds(row, _ZERO_ROWS), :], zero_sem).start(priority=1)

    @pl.when(num_gw0 > 0)
    def _():
      for c in _in_copies(0, 0, 0):
        c.start()

  num_gw = num_ref[0]
  num_wload = num_ref[1]

  @pl.when(gw < num_gw)
  def _step():
    t = d * num_gw + gw
    slot = t % 2
    total = num_d * num_gw
    e = gw_to_expert[gw]
    cur_wload = d * num_wload + gw_to_wload[gw]
    oslot = cur_wload % 2
    prev_gw = jnp.maximum(gw - 1, 0)
    next_gw = jnp.minimum(gw + 1, num_gw - 1)
    is_new_expert = jnp.logical_or(gw == 0, e != gw_to_expert[prev_gw])
    is_last_of_expert = jnp.logical_or(gw == num_gw - 1, e != gw_to_expert[next_gw])

    for c in _in_copies(d, gw, slot):
      c.wait()

    @pl.when(is_new_expert)
    def _():
      @pl.when(cur_wload >= 2)
      def _():
        prev2 = cur_wload - 2
        _out_copy(wload_to_expert[prev2 % num_wload], prev2 // num_wload, oslot).wait()

      if st.has_acc:
        _accin_copy(e, d, oslot).start()

    @pl.when(t + 1 < total)
    def _():
      has_more = gw + 1 < num_gw
      for c in _in_copies(
          jnp.where(has_more, d, d + 1),
          jnp.where(has_more, gw + 1, 0),
          1 - slot,
      ):
        c.start()

    m_s = gw_to_m[gw]
    m_e = gw_to_m[gw + 1]
    a_s = _floor_align(m_s)
    lo = m_s - a_s
    hi = m_e - a_s
    if st.has_dy_scale:
      kmax = jnp.int32(0)
      shifts = [jnp.int32(0)] * ratio
      needs_shift = jnp.bool_(False)
    else:
      # Exponent shifts of the dgrad sub-tiles to the expert's largest one.
      kmax = kmax_ref[e]
      j0 = gw_to_j[gw] * ratio
      first = d_first[e]
      count = d_count[e]
      shifts = []
      for i in range(ratio):
        j = j0 + i
        k_i = exp_ref[first + jnp.minimum(j, count - 1)]
        shifts.append(jnp.where(j < count, jnp.maximum(k_i - kmax, _MIN_EXP), 0))
      needs_shift = functools.reduce(jnp.logical_or, [s != 0 for s in shifts])

    def _dot(lhs, rhs):
      return lax.dot_general(
          lhs,
          rhs,
          (((0,), (0,)), ((), ())),
          preferred_element_type=jnp.float32,
      )

    def _matmul_full():
      # Interior tiles are neither the first nor the last of their expert.
      acc_ref[...] += _dot(act_buf[slot], dyq_buf[slot])

    def _matmul_bucket(bucket_m: int, shift: bool):
      # act_q is finite in all rows (see the dgrad kernel; act_buf starts
      # zeroed), so masking dy_q suffices.
      dyq = dyq_buf[slot, :bucket_m, :]
      rows = lax.broadcasted_iota(jnp.int32, (bucket_m, 1), 0)
      valid = jnp.logical_and(rows >= lo, rows < hi)
      if shift:
        # Exact rescale of each dgrad tile to the expert's scale.
        factor = _pow2(jnp.full((1, 1), shifts[ratio - 1], jnp.int32))
        for i in reversed(range(ratio - 1)):
          f_i = _pow2(jnp.full((1, 1), shifts[i], jnp.int32))
          factor = jnp.where(rows < (i + 1) * st.dgrad_tile_rows, f_i, factor)
        dyq = jnp.where(valid, dyq.astype(jnp.bfloat16) * factor.astype(jnp.bfloat16), 0.0).astype(st.dy_qtype)
      else:
        dyq = jnp.where(valid, dyq, jnp.zeros_like(dyq))
      res = _dot(act_buf[slot, :bucket_m, :], dyq)

      @pl.when(is_new_expert)
      def _():
        acc_ref[...] = res

      @pl.when(jnp.logical_not(is_new_expert))
      def _():
        acc_ref[...] += res

    is_interior = jnp.logical_not(jnp.logical_or(is_new_expert, is_last_of_expert))
    num_buckets = st.tile_rows // st.bucket_rows
    bucket_idx = jnp.maximum(0, (hi - 1) // st.bucket_rows)
    branches = [_matmul_full]
    if st.has_dy_scale:
      for i in range(num_buckets):
        branches.append(functools.partial(_matmul_bucket, bucket_m=st.bucket_rows * (i + 1), shift=False))
      branch = jnp.where(is_interior, 0, 1 + bucket_idx)
    else:
      for i in range(num_buckets):
        for shift in (False, True):
          branches.append(functools.partial(_matmul_bucket, bucket_m=st.bucket_rows * (i + 1), shift=shift))
      branch = jnp.where(
          jnp.logical_and(is_interior, jnp.logical_not(needs_shift)),
          0,
          1 + 2 * bucket_idx + jnp.where(needs_shift, 1, 0),
      )
    lax.switch(branch, branches)

    @pl.when(is_last_of_expert)
    def _finalize():
      if st.has_dy_scale:
        scale = dy_scale_ref[0, 0] * st.act_scale
      else:
        scale = _pow2(jnp.full((1, 1), kmax, jnp.int32)) * st.act_scale
      row_step = min(size_f, 256)
      if st.has_acc:
        _accin_copy(e, d, oslot).wait()

      @pl.loop(0, size_f, step=row_step)
      def _(r):
        rs = pl.ds(pl.multiple_of(r, row_step), row_step)
        val = acc_ref[rs, :] * scale
        if st.has_acc:
          val = val + out_buf[oslot, rs, :].astype(jnp.float32)
        out_buf[oslot, rs, :] = val.astype(dw_dtype)

      _out_copy(e, d, oslot).start()

  is_last_prog = jnp.logical_and(d == num_d - 1, gw == pl.num_programs(1) - 1)

  @pl.when(is_last_prog)
  def _epilogue():
    @pl.when(num_gw > 0)
    def _():
      last = num_d * num_wload - 1
      _out_copy(wload_to_expert[num_wload - 1], num_d - 1, last % 2).wait()

      @pl.when(last >= 1)
      def _():
        prev = last - 1
        _out_copy(wload_to_expert[prev % num_wload], prev // num_wload, prev % 2).wait()

    if not st.has_acc:
      assert zero_sem is not None
      num_empty = st.num_experts - num_wload

      @pl.when(num_empty > 0)
      def _():
        rows = pl.multiple_of(num_empty * size_f, _ZERO_ROWS)
        pltpu.make_async_copy(dw_hbm.at[pl.ds(0, rows), :], dw_hbm.at[pl.ds(0, rows), :], zero_sem).wait()


def linear_bwd_wgrad(
    act_q: jax.Array,
    dy_q: jax.Array,
    tile_exps: jax.Array,
    group_sizes: jax.Array,
    *,
    act_scale: float,
    dgrad_tile_rows: int,
    dy_scale: jax.Array | None = None,
    dw_acc: jax.Array | None = None,
    dw_dtype: Any = jnp.bfloat16,
    tiles: WgradTiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Linear wgrad `dW[e] = act_q^T @ dy_q` from the `linear_bwd_dgrad` operands.

  Args:
    act_q: `[M, F]` fp8 activations (static `act_scale`).
    dy_q: `[M, D]` e5m2 output gradients, quantized per dgrad tile (or with
      `dy_scale`).
    tile_exps: `[max_tiles]` int32 exponents of the dgrad tile scales.
    group_sizes: `[E]` int32 chunk rows per expert.
    act_scale: Static scale of act_q.
    dgrad_tile_rows: The `tile_rows` that `linear_bwd_dgrad` ran with.
    dy_scale: Optional `[1, 1]` f32 scale of `dy_q` when `dy_ag` was quantized
      upstream.
    dw_acc: Optional `[E, F, D]` gradient to accumulate into, in place.
    dw_dtype: Dtype of dW (bf16 or f32); must match dw_acc's.
    tiles: Optional tile sizes.
    vmem_limit_bytes: Optional VMEM limit.

  Returns:
    `[E, F, D]` dW (including dw_acc).
  """
  if tiles is None:
    tiles = DEFAULT_WGRAD_TILES
  size_m, size_f = act_q.shape
  size_d = dy_q.shape[1]
  num_experts = group_sizes.shape[0]
  dw_dtype = jnp.dtype(dw_dtype)
  dgrad_tile_rows = min(dgrad_tile_rows, size_m)
  tile_rows = min(tiles.tile_rows, size_m)
  tile_rows = max(dgrad_tile_rows, tile_rows // dgrad_tile_rows * dgrad_tile_rows)
  bucket_rows = math.gcd(tile_rows, tiles.bucket_rows)
  tile_d = min(tiles.tile_d, size_d)
  if (
      dy_q.shape[0] != size_m
      or size_m % _SUBLANE
      or size_d % tile_d
      or tile_d % LANES
      or size_f % _ZERO_ROWS
      or bucket_rows % _SUBLANE
  ):
    raise ValueError(f"Unsupported linear_bwd_wgrad shapes act_q={act_q.shape}," f" dy_q={dy_q.shape}, tiles={tiles}.")
  if dw_acc is not None and (dw_acc.shape != (num_experts, size_f, size_d) or dw_acc.dtype != dw_dtype):
    raise ValueError(f"dw_acc must be {(num_experts, size_f, size_d)}/{dw_dtype}; got" f" {dw_acc.shape}/{dw_acc.dtype}.")
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
  max_gw = max_tiles(num_experts, size_m, tile_rows)
  max_gd = max_tiles(num_experts, size_m, dgrad_tile_rows)
  if tile_exps.shape != (max_gd,):
    raise ValueError(f"tile_exps must be [{max_gd}]; got {tile_exps.shape}.")
  num_d = size_d // tile_d
  gs = group_sizes.astype(jnp.int32)
  grid_gw = jnp.maximum(num_tiles(gs, tile_rows), 1)
  has_acc = dw_acc is not None
  has_dy_scale = dy_scale is not None
  dy_scale_scalar = dy_scale.astype(jnp.float32).reshape(1, 1) if has_dy_scale else jnp.zeros((1, 1), jnp.float32)
  st = _WgradStatic(
      num_experts=num_experts,
      size_f=size_f,
      size_d=size_d,
      tile_rows=tile_rows,
      bucket_rows=bucket_rows,
      tile_d=tile_d,
      dgrad_tile_rows=dgrad_tile_rows,
      act_scale=act_scale,
      dy_qtype=dy_q.dtype,
      has_acc=has_acc,
      has_dy_scale=has_dy_scale,
  )
  dw_mat = jax.typeof(dw_acc if has_acc else act_q).manual_axis_type
  out_shape = jax.ShapeDtypeStruct((num_experts * size_f, size_d), dw_dtype, manual_axis_type=dw_mat)
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  scratch_shapes = [
      pltpu.SMEM((2,), jnp.int32),  # num_ref
      pltpu.SMEM((max_gw + 1,), jnp.int32),  # gw_to_m
      pltpu.SMEM((max_gw,), jnp.int32),  # gw_to_expert
      pltpu.SMEM((max_gw,), jnp.int32),  # gw_to_wload
      pltpu.SMEM((max_gw,), jnp.int32),  # gw_to_j
      pltpu.SMEM((num_experts,), jnp.int32),  # wload_to_expert
      pltpu.SMEM((max_gd + 1,), jnp.int32),  # dg_to_m
      pltpu.SMEM((num_experts,), jnp.int32),  # d_first
      pltpu.SMEM((num_experts,), jnp.int32),  # d_count
      pltpu.SMEM((num_experts,), jnp.int32),  # kmax_ref
      pltpu.VMEM((2, tile_rows, size_f), act_q.dtype),  # act_buf
      pltpu.VMEM((2, tile_rows, tile_d), dy_q.dtype),  # dyq_buf
      pltpu.VMEM((size_f, tile_d), jnp.float32),  # acc_ref
      pltpu.VMEM((2, size_f, tile_d), dw_dtype),  # out_buf (also accin)
      pltpu.SemaphoreType.DMA((2, 2)),  # in_sem
      pltpu.SemaphoreType.DMA((2,)),  # out_sem
  ]
  if has_acc:
    scratch_shapes.append(pltpu.SemaphoreType.DMA((2,)))  # accin_sem
    inputs = (act_q, dy_q, dw_acc.reshape(num_experts * size_f, size_d))
  else:
    scratch_shapes += [
        pltpu.VMEM((_ZERO_ROWS, size_d), dw_dtype),  # zero_buf
        pltpu.SemaphoreType.DMA(()),  # zero_sem
    ]
    inputs = (act_q, dy_q)
  scalars = (gs, tile_exps.astype(jnp.int32), dy_scale_scalar)
  flops = 2 * size_m * size_d * size_f
  bytes_accessed = (
      num_d * size_m * size_f
      + size_m * size_d
      + num_experts * size_f * size_d * dw_dtype.itemsize * (2 if has_acc else 1)
  )
  dw = pl.pallas_call(
      functools.partial(_wgrad_kernel, st=st),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalars),
          grid=(num_d, grid_gw),
          in_specs=[hbm] * len(inputs),
          out_specs=hbm,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          dimension_semantics=("arbitrary", "arbitrary"),
      ),
      input_output_aliases={len(jax.tree.leaves((scalars, inputs))) - 1: 0} if has_acc else {},
      cost_estimate=pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0),
      name=(
          f"linear_bwd_wgrad-g_{num_experts}-m_{size_m}-d_{size_d}-f_{size_f}"
          f"-tr_{tile_rows}-tb_{bucket_rows}-td_{tile_d}" + ("-dys" if has_dy_scale else "") + ("-acc" if has_acc else "")
      ),
  )(*scalars, *inputs)
  return dw.reshape(num_experts, size_f, size_d)
