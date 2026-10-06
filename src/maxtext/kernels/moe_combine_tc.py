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

"""Fused TensorCore (Pallas) MoE "combine" kernel.

`combine` computes, for expert outputs `x` [R = T*K, E] stored in expert-sorted
order (row r holds flat token-slot `sort_idx[r]`, `sort_idx = argsort(experts)`):

    y[t] = sum_k w[t, k] * x[inv[t*K + k]],     inv = argsort(sort_idx)

with f32 accumulation, plus a custom VJP:

    d_x[inv[t*K+k]] = w[t, k] * dy[t]            (a permutation, no scatter-add)
    d_w[t, k]       = <dy[t], x[inv[t*K+k]]>     (f32)

Why not a per-row DMA gather: on v6e a TensorCore row-gather is bound by
~20 ns per row DMA, i.e. ~5 ms for 262144 rows, no better than SparseCore.

Block-range formulation: process tokens in blocks of C. Because the expert
sort is *stable*, the rows of the tokens in [t0, t0+C) that went to expert g
form ONE contiguous row range [lo[b,g], hi[b,g]) of `x`. Per token block the
kernel DMAs at most G row windows (one DMA each) into a VMEM buffer `buf`
[J, E]; reordering + weighting is a matmul on the MXU:

    fwd:  y_blk[C, E]  = S[C, J] @ buf[J, E]        S[c, j] = w(j) if row j is token c
    bwd:  dw_j         = <dy[tok(j)], x_j>          (masked lane reduction of x_rows @ dy^T)
          dW[C, K]     = exact 0/1 MXU routing of dw_j to (tok(j), k(j))
          dbuf[J, E]   = S^T[J, C] @ dy_blk[C, E]  -> DMA'd back to the same windows

No per-row gathers / scatters / sorts in XLA: a [R, 128] bf16 "row info" array
(token % 256, token // 256, k of every sorted x row; elementwise from sort_idx)
is DMA'd with the same windows as x, so every buffer row carries its (token, k).
S is one compare + select per element, with the row weights selected from the
[C, K] weight block by an exact 0/1 matmul (W @ onehot(k)). The per-block DMA
table needs the (block, expert) row counts: a one-hot histogram matmul.

Tiling / alignment (Mosaic requirement): every DMA and every dynamic VMEM
slice along the row (2nd-minor) dimension uses offsets AND sizes that are
multiples of `align` rows (8 = the bf16 (8,128)(2,1) memref tile; 16 for
(16,128) tiling). So each range is widened to the aligned window
[lo & -A, round_up(hi, A)), in HBM and in VMEM (buffer row = x row + delta,
delta a multiple of A). Windows of the same block that share a granule are
coalesced into one segment (constant delta), so every x row has a single
buffer position per block.

Scatter direction (combine bwd): the output windows are written
whole, so a granule that contains a range boundary (a "partial granule") may
hold rows owned by another token block and is read-modify-written. Each grid
step owns one slot of the output buffer `obuf`, and the partial granules of a
block are staged into its slot one grid step ahead, overlapping the compute:

  * only granules holding rows of *earlier* blocks are staged at all (rows of
    later blocks are overwritten by their owner anyway);
  * a granule that lies in a window of the previous block -- e.g. the head
    granule of (b, g), which is the tail granule of (b-1, g) because the rows
    of an expert are in block order -- is vector-copied from the previous
    step's `obuf` slot, which holds exactly the bytes that step wrote;
  * every other staged granule is DMA'd from HBM; such a read never overlaps
    the previous step's writes.

The compute then writes each owned row and keeps the staged value of every
other row in place (one select per tile), and the windows are written back.
Shared granules are written by consecutive steps, so a step waits for the
previous step's writes before issuing its own. Every x row is owned by exactly
one block, and every non-owned window row lies in a partial granule.

Precision: all products are exact (bf16 x bf16 -> f32) and accumulation is
f32, i.e. the same arithmetic as the `float32_weight_sum=True` einsum. If the
routing weights are not bf16, S is split into bf16 hi + lo parts (2 matmuls).

Requirements for the kernel path (otherwise `combine` uses a pure-jnp
reference): x is 2-D bf16 with
E % 128 == 0; T % block_tokens == 0; block_tokens % 128 == 0; T <= 65280;
T*K % align == 0; `group_sizes` sums to R; `sort_idx` is a *stable* argsort of
the flat expert ids (jnp.argsort default).

Tuned constants (Gemma4-26B-A4B on v6e): 256 tokens per grid step, 512-row
forward contraction buckets, 256-wide output column chunks, 1024-row backward
compute chunks.
"""

import functools
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = ["combine", "combine_reference", "kernel_supported"]

_LANES = 128
_CHUNK_ROWS = 128  # rows per full-size range DMA (static-chunk fallback)
_WAIT_ROWS = 256  # rows per full-size semaphore wait (static-chunk fallback)
_DEFAULT_BLOCK_TOKENS = 256  # tokens per grid step
_COL_CHUNK = 256  # output / dy column chunk per MXU dot
_FWD_BUCKET_ROWS = 512  # forward contraction-size bucket in buffer rows
_SCATTER_ROWS_PER_CHUNK = 1024  # backward: buffer rows per compute chunk
_VMEM_LIMIT = 120 * 1024 * 1024

# Rows of the per-block scalar table (each row has G entries, one per expert).
#   RS    HBM start row of this range's (new part of the) window
#   NL    rows of that window part (multiple of A; 0 if empty / already covered)
#   OFF   buffer row of RS
#   DELTA buffer row - x row, for every row of this range's segment
# Scatter staging (see the module docstring); -1 = nothing to do:
#   HS    HBM start of the head partial granule (lo % A != 0) if it is read from HBM
#   TS    same for the tail partial granule (hi % A != 0)
#   HCR, HCD   head granule carried from the previous step's output buffer:
#         buffer row in the previous slot (HCR) / in this slot (HCD)
#   TCR, TCD   same for the tail granule
# Carries are plain vector loads / stores between the two VMEM slots rather than
# VMEM->VMEM DMAs: Mosaic serializes a whole grid step against any DMA that
# touches a buffer the step also computes on, whereas copying one granule is a
# handful of vector instructions.
_T_RS, _T_NL, _T_OFF, _T_DELTA, _T_HS, _T_TS, _T_HCR, _T_HCD, _T_TCR, _T_TCD = range(10)
_NTAB = 10

# Row-info filler for buffer rows that hold no row of this step: k = 255
# (_INFO_FILL_K > K) marks filler rows so token 65535 (T = 65536) remains valid.
_INFO_FILL = 255.0
_INFO_FILL_K = 255
_MAX_TOKENS = 256 * 256


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compiler_params(vmem_limit_bytes: int):
  params_cls = getattr(pltpu, "CompilerParams", None) or getattr(pltpu, "TPUCompilerParams")
  return params_cls(dimension_semantics=("arbitrary",), vmem_limit_bytes=int(vmem_limit_bytes))


def _vma(x) -> frozenset:
  mat = getattr(jax.typeof(x), "manual_axis_type", None)
  return frozenset(mat.varying) if mat is not None else frozenset()


def _match_vma(*ops):
  """Inside shard_map(check_vma=True): make operands vary over the union of axes."""
  union = frozenset().union(*(_vma(o) for o in ops))
  if not union:
    return ops, None
  out = []
  for o in ops:
    missing = tuple(sorted(union - _vma(o)))
    if missing:
      o = lax.pcast(o, missing, to="varying")
    out.append(o)
  return tuple(out), jax.sharding.ManualAxisType(varying=union)


def _out_struct(shape, dtype, mat):
  if mat is None:
    return jax.ShapeDtypeStruct(shape, dtype)
  return jax.ShapeDtypeStruct(shape, dtype, manual_axis_type=mat)


def _round_up(x, m):
  return (x + m - 1) // m * m


def _al(x, a):
  """Row offset hint: `x` is a multiple of `a` (emits tpu.assume_multiple)."""
  return pl.multiple_of(x, a)


def _tab_stride(g):
  """Per-block SMEM table length: a valid rank-1 Pallas TPU block size >= _NTAB*G."""
  n = _NTAB * g
  if n > 1024:
    return _round_up(n, 1024)
  p = _LANES
  while p < n:
    p *= 2
  return p


def _check_align(align: int) -> int:
  if align < 8 or align & (align - 1):
    raise ValueError(f"align must be a power of two >= 8, got {align}")
  return int(align)


def _buffer_rows(block_tokens, k, g, align=8):
  """Static bound on buffer rows: each non-empty range widens by <= 2*(A-1) rows."""
  ck = block_tokens * k
  n = ck + (2 * align - 2) * min(g, ck)
  return _round_up(n, 2 * _LANES)


def kernel_supported(x_shape, x_dtype, num_tokens, k, block_tokens, align=8) -> bool:
  """True iff the Pallas kernels can handle this problem (see the module docstring)."""
  if len(x_shape) != 2:
    return False
  r, e = x_shape
  return (
      jnp.dtype(x_dtype) == jnp.bfloat16
      and e % _LANES == 0
      and r == num_tokens * k
      and r % align == 0
      and block_tokens % _LANES == 0
      and num_tokens % block_tokens == 0
      and num_tokens <= _MAX_TOKENS
      and k < _INFO_FILL_K
  )


def _default_block_tokens(num_tokens: int, default: int = _DEFAULT_BLOCK_TOKENS) -> int:
  """Largest power-of-two divisor of num_tokens that is <= the tuned default (>= 128)."""
  c = default
  while c > _LANES and num_tokens % c != 0:
    c //= 2
  return c


def _default_interpret():
  return False if jax.default_backend() == "tpu" else pltpu.InterpretParams()


def _kernel_modes(interpret):
  """(upcast, dynamic_dma) for the given interpret setting.

  The legacy HLO interpreter (`interpret=True`) has no bf16 x bf16 -> f32 dot
  and no dynamic-size DMAs, so matmul operands are upcast to f32 (exact) and
  DMAs are issued in static power-of-two row chunks. The Mosaic interpreter
  (`pltpu.InterpretParams`) only needs the upcast.
  """
  upcast = bool(interpret)
  dynamic_dma = interpret is not True
  return upcast, dynamic_dma


def _col_chunk(e):
  for c in (_COL_CHUNK, _LANES):
    if e % c == 0:
      return c
  return _LANES


def _row_chunk(jrows):
  return 256 if jrows % 256 == 0 else _LANES


def _buckets(jrows):
  """Static contraction sizes (multiples of the row chunk) ending at jrows."""
  jq = _row_chunk(jrows)
  step = max(jq, _round_up(_FWD_BUCKET_ROWS, jq))
  sizes = list(range(step, jrows, step)) + [jrows]
  return tuple(sizes)


def _scatter_row_chunk(jrows):
  """Largest multiple of 256 that is <= min(_SCATTER_ROWS_PER_CHUNK, jrows)."""
  return max(256, min(_SCATTER_ROWS_PER_CHUNK, jrows) // 256 * 256)


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------


def combine_reference(x, sort_idx, weights):
  """Pure-jnp reference: sum_k w * x[argsort(sort_idx)] (f32 accumulate)."""
  t, k = weights.shape
  inv = jnp.argsort(sort_idx)
  g = x[inv].reshape(t, k, x.shape[1]).astype(jnp.float32)
  y = jnp.einsum("tke,tk->te", g, weights.astype(jnp.float32), precision=lax.Precision.HIGHEST)
  return y.astype(x.dtype)


# ---------------------------------------------------------------------------
# metadata (plain XLA: cumsums, one small one-hot matmul, elementwise row info)
# ---------------------------------------------------------------------------


def _metadata(sort_idx, group_sizes, num_tokens, k, block_tokens, align=8):
  """Returns the per-block DMA table, flat int32 [nb * _tab_stride(G)].

  No gathers / scatters / sorts: the (block, expert) row counts are a one-hot
  histogram on the MXU (exact: counts < 2^24).
  """
  g = group_sizes.shape[0]
  t, c, a = num_tokens, block_tokens, align
  nb = t // c
  r = t * k
  gs = group_sizes.astype(jnp.int32)
  ends = jnp.cumsum(gs)
  starts = ends - gs
  blk = sort_idx.astype(jnp.int32) // (k * c)  # token block of every sorted row
  rows = jnp.arange(r, dtype=jnp.int32)[:, None]
  in_group = ((rows >= starts[None, :]) & (rows < ends[None, :])).astype(jnp.bfloat16)  # [R, G]
  in_block = (blk[:, None] == jnp.arange(nb, dtype=jnp.int32)[None, :]).astype(jnp.bfloat16)  # [R, nb]
  count = jnp.einsum("rb,rg->bg", in_block, in_group, preferred_element_type=jnp.float32)
  count = count.astype(jnp.int32)  # [nb, g]
  lo = starts[None, :] + jnp.cumsum(count, axis=0) - count
  hi = lo + count
  nonempty = count > 0
  wa = jnp.bitwise_and(lo, -a)  # aligned window [wa, we)
  we = jnp.bitwise_and(hi + (a - 1), -a)
  # Windows are non-decreasing in g within a block; consecutive windows can
  # share (at most) one granule -> coalesce: each range only adds the part of
  # its window not already covered by the previous non-empty range.
  we_ne = jnp.where(nonempty, we, -1)
  prev_we = jnp.concatenate([jnp.full((nb, 1), -1, jnp.int32), lax.cummax(we_ne, axis=1)[:, :-1]], axis=1)
  rs = jnp.maximum(wa, prev_we)
  nl = jnp.where(nonempty, jnp.maximum(we - rs, 0), 0)
  off = jnp.cumsum(nl, axis=1) - nl
  delta = off - rs  # buffer row = x row + delta (constant within a segment)
  hs0 = jnp.where(nonempty & (jnp.bitwise_and(lo, a - 1) != 0), wa, -1)  # head partial granule
  ts0 = jnp.where(nonempty & (jnp.bitwise_and(hi, a - 1) != 0), we - a, -1)  # tail partial granule

  # ---- scatter staging ----
  # A granule only has to be staged (read-modify-write) if it holds rows owned
  # by an earlier block: rows [starts[g'], lo(b, g')) for some expert g'.
  def earlier_owned(x):  # x [nb, g] granule starts (-1: none)
    xl = x[:, :, None]
    ov = jnp.maximum(xl, starts[None, None, :]) < jnp.minimum(xl + a, lo[:, None, :])
    return (x >= 0) & jnp.any(ov, axis=2)

  # A tail granule that is also the head granule of a later expert of this block
  # is covered by that head -- never stage one granule twice.
  heads = jnp.where(nonempty, hs0, -2)
  shared = jnp.any(ts0[:, :, None] == heads[:, None, :], axis=2)
  hs_need = earlier_owned(hs0)
  ts_need = earlier_owned(ts0) & ~shared
  # A granule that lies in a window of the previous block (any expert) is
  # vector-copied from that block's output buffer, which holds the same bytes
  # the block wrote to HBM; everything else is read from HBM, and that read
  # never overlaps the previous block's writes. Coalesced window parts
  # [rs, rs + nl) of a block are disjoint (full windows [wa, we) may share a
  # granule), so a granule lies in at most one of them.
  shift = lambda x, fill: jnp.concatenate([jnp.full((1, g), fill, x.dtype), x[:-1]], axis=0)
  pwa = shift(jnp.where(nl > 0, rs, 1 << 30), 1 << 30)[:, None, :]
  pwe = shift(jnp.where(nl > 0, rs + nl, -1), -1)[:, None, :]
  pdelta = shift(delta, 0)[:, None, :]

  def prev_row(x):  # (in a previous-block window, previous-slot buffer row)
    xl = x[:, :, None]
    hit = (xl >= pwa) & (xl < pwe)
    return jnp.any(hit, axis=2) & (x >= 0), jnp.sum(jnp.where(hit, xl + pdelta, 0), axis=2)

  h_prev, h_row = prev_row(hs0)
  t_prev, t_row = prev_row(ts0)
  hs = jnp.where(hs_need & ~h_prev, hs0, -1)
  ts = jnp.where(ts_need & ~t_prev, ts0, -1)
  hcr, hcd = jnp.where(hs_need & h_prev, h_row, -1), jnp.where(hs_need & h_prev, hs0 + delta, -1)
  tcr, tcd = jnp.where(ts_need & t_prev, t_row, -1), jnp.where(ts_need & t_prev, ts0 + delta, -1)
  cols = [rs, nl, off, delta, hs, ts, hcr, hcd, tcr, tcd]
  tab = jnp.stack(cols, axis=1).astype(jnp.int32).reshape(nb, _NTAB * g)
  tab = jnp.pad(tab, ((0, 0), (0, _tab_stride(g) - _NTAB * g)))
  return tab.reshape(-1)


def _row_info(sort_idx, k):
  """[R, 128] bf16 per sorted x row: lanes (token % 256, token // 256, k), rest 0.

  All values are integers < 256, exact in bf16. DMA'd by the kernels with the
  same row windows as x, so each buffer row carries its own (token, k).
  """
  s = sort_idx.astype(jnp.int32)
  if isinstance(k, int) and k > 0 and (k & (k - 1)) == 0:
    tok = s >> (k.bit_length() - 1)
    rem = jnp.bitwise_and(s, k - 1)
  else:
    tok = s // k
    rem = s - tok * k
  c0 = jnp.bitwise_and(tok, 255).astype(jnp.bfloat16)[:, None]
  c1 = (tok >> 8).astype(jnp.bfloat16)[:, None]
  c2 = rem.astype(jnp.bfloat16)[:, None]
  lane = lax.broadcasted_iota(jnp.int32, (1, _LANES), 1)
  z = jnp.zeros((1, 1), dtype=jnp.bfloat16)
  return jnp.where(
      lane == 0,
      c0,
      jnp.where(lane == 1, c1, jnp.where(lane == 2, c2, z)),
  )


def _padded_weights(w):
  """[T, 128] f32 (weights in lanes [0, K))."""
  return jnp.pad(w.astype(jnp.float32), ((0, 0), (0, _LANES - w.shape[1])))


# ---------------------------------------------------------------------------
# in-kernel building blocks: DMAs
# ---------------------------------------------------------------------------


def _copy_rows(src_fn, dst_fn, nrows, sem, align, dynamic_dma):
  """Starts DMAs copying `nrows` (dynamic, > 0, multiple of align) rows.

  src_fn(o, n) / dst_fn(o, n) return the ref windows for rows [o, o+n) of the
  range; o is a multiple of `align`. With dynamic_dma=True this is a single
  dynamic-size DMA; otherwise static power-of-two chunks (legacy interpreter).
  """
  if dynamic_dma:
    n = _al(nrows, align)
    pltpu.make_async_copy(src_fn(0, n), dst_fn(0, n), sem).start()
    return
  nfull = nrows // _CHUNK_ROWS

  def body(i, carry):
    o = i * _CHUNK_ROWS
    pltpu.make_async_copy(src_fn(o, _CHUNK_ROWS), dst_fn(o, _CHUNK_ROWS), sem).start()
    return carry

  lax.fori_loop(0, nfull, body, 0)
  base = nfull * _CHUNK_ROWS
  rem = nrows - base
  bit = _CHUNK_ROWS // 2
  while bit >= align:

    @pl.when(jnp.bitwise_and(rem, bit) != 0)
    def _(bit=bit):
      o = base + jnp.bitwise_and(rem, ~(2 * bit - 1))
      pltpu.make_async_copy(src_fn(o, bit), dst_fn(o, bit), sem).start()

    bit //= 2


def _wait_rows(desc_fn, sem, nrows, max_rows, align, dynamic_dma):
  """Waits until `nrows` (<= max_rows, multiple of align) rows have landed on `sem`.

  desc_fn(n) returns an n-row ref (offset 0) with the same row size as the DMAs.
  """
  if dynamic_dma:

    @pl.when(nrows > 0)
    def _():
      n = _al(nrows, align)
      pltpu.make_async_copy(desc_fn(n), desc_fn(n), sem).wait()

    return
  chunk = align
  while chunk * 2 <= min(_WAIT_ROWS, max_rows):
    chunk *= 2
  nfull = nrows // chunk

  def body(i, carry):
    pltpu.make_async_copy(desc_fn(chunk), desc_fn(chunk), sem).wait()
    return carry

  lax.fori_loop(0, nfull, body, 0)
  rem = nrows - nfull * chunk
  bit = chunk // 2
  while bit >= align:

    @pl.when(jnp.bitwise_and(rem, bit) != 0)
    def _(bit=bit):
      pltpu.make_async_copy(desc_fn(bit), desc_fn(bit), sem).wait()

    bit //= 2


def _fill_buf(buf, nslots, jrows, value=0):
  """Fills rows [0, jrows) of the first `nslots` slots of `buf` with `value`."""
  fill = jnp.full((_LANES, buf.shape[-1]), value, buf.dtype)
  for s in range(nslots):

    def body(i, carry, s=s):
      buf[s, pl.ds(_al(i * _LANES, _LANES), _LANES), :] = fill
      return carry

    lax.fori_loop(0, jrows // _LANES, body, 0)


def _used_rows(tab, g):
  """Buffer rows filled by the block described by `tab` (end of its last window part)."""
  return tab[_T_OFF * g + g - 1] + tab[_T_NL * g + g - 1]


class _WindowRead(NamedTuple):
  """One HBM array read window-by-window into a double-buffered VMEM scratch."""

  hbm: Any  # [R, cols] source in HBM
  buf: Any  # VMEM [2, J, cols] destination (one slot per grid step)
  sem: Any  # DMA semaphores [2]
  fill: Any  # value both slots are cleared with at step 0, or None (see _read_pipeline)


def _issue_range_reads(tab, reads, dst_slot, g, a, dynamic_dma):
  """DMAs every (aligned) window part of the block described by `tab` into `dst_slot`."""

  def body(gi, carry):
    n = tab[_T_NL * g + gi]

    @pl.when(n > 0)
    def _():
      rs = tab[_T_RS * g + gi]
      off = tab[_T_OFF * g + gi]
      for rd in reads:
        _copy_rows(
            lambda o, m, hbm=rd.hbm: hbm.at[pl.ds(_al(rs + o, a), m)],
            lambda o, m, buf=rd.buf: buf.at[dst_slot, pl.ds(_al(off + o, a), m)],
            n,
            rd.sem.at[dst_slot],
            a,
            dynamic_dma,
        )

    return carry

  lax.fori_loop(0, g, body, 0)


def _read_pipeline(tab_cur, tab_nxt, reads, g, jrows, a, dynamic_dma):
  """Double-buffered window reads: issues the next block's, waits for this block's.

  Returns the slot holding this step's rows. At step 0 every read with a
  `fill` value clears both slots first: rows past the used prefix are either
  multiplied by zeros on the MXU (x: must be finite) or must not look like rows
  of any block (row info: `_INFO_FILL`); afterwards they only hold real rows.
  """
  b = pl.program_id(0)
  nb = pl.num_programs(0)
  slot = b % 2

  @pl.when(b == 0)
  def _():
    for rd in reads:
      if rd.fill is not None:
        _fill_buf(rd.buf, 2, jrows, rd.fill)
    _issue_range_reads(tab_cur, reads, 0, g, a, dynamic_dma)

  @pl.when(b + 1 < nb)
  def _():
    _issue_range_reads(tab_nxt, reads, 1 - slot, g, a, dynamic_dma)

  used = _used_rows(tab_cur, g)
  for rd in reads:
    _wait_rows(lambda n, buf=rd.buf: buf.at[slot, pl.ds(0, n)], rd.sem.at[slot], used, jrows, a, dynamic_dma)
  return slot


# ---------------------------------------------------------------------------
# in-kernel building blocks: compute
# ---------------------------------------------------------------------------


def _chunk_info(ibuf, slot, q0, jq, c, t0, used):
  """Row info of buffer rows [q0, q0+jq) in row (lane = j) orientation.

  t0 = first token of this block, used = number of rows filled this step (rows
  past it are stale rows of an earlier step and may even name tokens of this
  block, from alignment padding of an earlier window).
  Returns (tokrow [1, jq] int32: token-in-block if the row belongs to this
  block else -1, krow [1, jq] int32: top-k slot of the row).
  """
  info = ibuf[slot, pl.ds(q0, jq), :].astype(jnp.float32).T  # [128, jq]
  tok = (info[0:1, :] + 256.0 * info[1:2, :]).astype(jnp.int32)
  cc = tok - t0
  live = lax.broadcasted_iota(jnp.int32, (1, jq), 1) < used - q0
  krow = info[2:3, :].astype(jnp.int32)
  tokrow = jnp.where((cc >= 0) & (cc < c) & live & (krow < _INFO_FILL_K), cc, -1)
  return tokrow, krow


def _chunk_tokens(ibuf, slot, q0, rq, c, t0, used):
  """Row info of buffer rows [q0, q0+rq) in column (sublane = j) orientation.

  Same semantics as `_chunk_info`: returns (tokc [rq, 1] int32: token-in-block
  or -1, kc [rq, 1] int32: top-k slot).
  """
  info = ibuf[slot, pl.ds(q0, rq), :].astype(jnp.float32)  # [rq, 128]
  row = lax.broadcasted_iota(jnp.int32, (rq, 1), 0) + q0
  tok = (info[:, 0:1] + 256.0 * info[:, 1:2]).astype(jnp.int32) - t0
  kc = info[:, 2:3].astype(jnp.int32)
  tokc = jnp.where((tok >= 0) & (tok < c) & (row < used) & (kc < _INFO_FILL_K), tok, -1)
  return tokc, kc


def _chunk_start(q, rq, jrows):
  """Start row of backward compute chunk q, clamped so the last chunk stays inside the buffer."""
  return _al(jnp.minimum(q * rq, jrows - rq), 256)


def _mm(a, b, dims=None, *, upcast=False):
  """f32-accumulating MXU dot; `upcast` is for interpreters without a bf16 x bf16 -> f32 dot (exact)."""
  if upcast:
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
  if dims is None:
    return jnp.dot(a, b, preferred_element_type=jnp.float32)
  return lax.dot_general(a, b, (dims, ((), ())), preferred_element_type=jnp.float32)


def _row_weights(mm, w_hi, w_lo, krow, jq, dtype):
  """[C, jq] f32 weight of token c for top-k slot krow[j] (exact MXU selection)."""
  koh = (lax.broadcasted_iota(jnp.int32, (_LANES, jq), 0) == krow).astype(dtype)  # [128, jq]
  sel_hi = mm(w_hi, koh)
  sel_lo = None if w_lo is None else mm(w_lo, koh)
  return sel_hi, sel_lo


def _split_w(w_ref, split, dtype):
  """Weight block as bf16 hi part (+ exact lo remainder if `split`)."""
  w = w_ref[...]  # [C, 128] f32
  w_hi = w.astype(dtype)
  w_lo = (w - w_hi.astype(jnp.float32)).astype(dtype) if split else None
  return w_hi, w_lo


def _exact_bf16_parts(v, dtype):
  """v (f32) = p1 + p2 + p3 (+ < 2^-24 |v|) with bf16 parts."""
  p1 = v.astype(dtype)
  r1 = v - p1.astype(jnp.float32)
  p2 = r1.astype(dtype)
  p3 = (r1 - p2.astype(jnp.float32)).astype(dtype)
  return p1, p2, p3


def _table_specs(nb, ntab):
  """BlockSpecs of the per-block scalar table for this step and the next (clamped)."""
  return [
      pl.BlockSpec((ntab,), lambda b: (b,), memory_space=pltpu.SMEM),
      pl.BlockSpec((ntab,), lambda b: (jnp.minimum(b + 1, nb - 1),), memory_space=pltpu.SMEM),
  ]


def _common_specs(nb, ntab, c):
  """Table specs + the padded [C, 128] weight block of this step."""
  return _table_specs(nb, ntab) + [pl.BlockSpec((c, _LANES), lambda b: (b, 0))]


# ---------------------------------------------------------------------------
# forward kernel
# ---------------------------------------------------------------------------


def _fwd_kernel(
    tab_cur,
    tab_nxt,
    w_ref,
    x_hbm,
    info_hbm,
    y_ref,
    buf,
    ibuf,
    s_hi_ref,
    s_lo_ref,
    xsem,
    isem,
    *,
    g,
    jrows,
    e,
    split,
    a,
    upcast,
    dynamic_dma,
):
  """y_blk = S @ buf for one token block (S built from the row info)."""
  mm = functools.partial(_mm, upcast=upcast)
  c = y_ref.shape[0]
  reads = (_WindowRead(x_hbm, buf, xsem, fill=0), _WindowRead(info_hbm, ibuf, isem, fill=_INFO_FILL))
  slot = _read_pipeline(tab_cur, tab_nxt, reads, g, jrows, a, dynamic_dma)

  jq = _row_chunk(jrows)
  en = _col_chunk(e)
  buckets = _buckets(jrows)
  used = _used_rows(tab_cur, g)
  jb = jnp.int32(buckets[-1])
  for size in reversed(buckets[:-1]):
    jb = jnp.where(used <= size, jnp.int32(size), jb)
  w_hi, w_lo = _split_w(w_ref, split, s_hi_ref.dtype)
  iota_c = lax.broadcasted_iota(jnp.int32, (c, jq), 0)
  t0 = pl.program_id(0) * c  # (program_id is not allowed inside loops in interpret mode)

  # S [C, jb] (bf16 hi [+ lo]) into VMEM scratch: one compare + select per element.
  def s_body(q, carry):
    q0 = _al(q * jq, jq)
    tokrow, krow = _chunk_info(ibuf, slot, q0, jq, c, t0, used)
    m = iota_c == tokrow
    sel_hi, sel_lo = _row_weights(mm, w_hi, w_lo, krow, jq, s_hi_ref.dtype)
    s_hi_ref[:, pl.ds(q0, jq)] = jnp.where(m, sel_hi, 0.0).astype(s_hi_ref.dtype)
    if split:
      s_lo_ref[:, pl.ds(q0, jq)] = jnp.where(m, sel_lo, 0.0).astype(s_lo_ref.dtype)
    return carry

  lax.fori_loop(0, jb // jq, s_body, 0)

  # y[:, cols] = S[:, :jb] @ buf[:jb, cols]: one MXU-accumulated dot per column chunk.
  for size in buckets:

    @pl.when(jb == size)
    def _(size=size):
      def y_body(n, carry):
        n0 = _al(n * en, en)
        xb = buf[slot, pl.ds(0, size), pl.ds(n0, en)]  # [size, en]
        acc = mm(s_hi_ref[:, pl.ds(0, size)], xb)
        if split:
          acc = acc + mm(s_lo_ref[:, pl.ds(0, size)], xb)
        y_ref[:, pl.ds(n0, en)] = acc.astype(y_ref.dtype)
        return carry

      lax.fori_loop(0, e // en, y_body, 0)


def _fwd_call(x, w_pad, tab, info, *, block_tokens, num_groups, split, interpret, align):
  r, e = x.shape
  t = w_pad.shape[0]
  c, g = block_tokens, num_groups
  nb = t // c
  k = r // t
  jrows = _buffer_rows(c, k, g, align)
  ntab = _tab_stride(g)
  upcast, dynamic_dma = _kernel_modes(interpret)
  in_specs = _common_specs(nb, ntab, c) + [pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)]
  out_spec = pl.BlockSpec((c, e), lambda b: (b, 0))
  itemsize = jnp.dtype(x.dtype).itemsize
  vmem = (
      2 * jrows * (e + _LANES) * itemsize  # buf + row info
      + 2 * c * e * itemsize  # out blocks
      + (2 if split else 1) * c * jrows * itemsize  # S hi / lo
      + 2 * c * _LANES * 4  # weight blocks
      + 16 * 1024 * 1024  # dot operands / temporaries, compiler scratch
  )
  kernel = functools.partial(
      _fwd_kernel, g=g, jrows=jrows, e=e, split=split, a=align, upcast=upcast, dynamic_dma=dynamic_dma
  )
  (tab, w_pad, x, info), mat = _match_vma(tab, w_pad, x, info)
  return pl.pallas_call(
      kernel,
      out_shape=_out_struct((t, e), x.dtype, mat),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(nb,),
          in_specs=in_specs,
          out_specs=out_spec,
          scratch_shapes=[
              pltpu.VMEM((2, jrows, e), x.dtype),  # buf
              pltpu.VMEM((2, jrows, _LANES), info.dtype),  # row info
              pltpu.VMEM((c, jrows), x.dtype),  # S hi
              pltpu.VMEM((c, jrows) if split else (8, _LANES), x.dtype),  # S lo
              pltpu.SemaphoreType.DMA((2,)),
              pltpu.SemaphoreType.DMA((2,)),
          ],
      ),
      compiler_params=_compiler_params(min(vmem, _VMEM_LIMIT)),
      interpret=interpret,
      name="moe_combine_fwd",
  )(tab, tab, w_pad, x, info)


# ---------------------------------------------------------------------------
# scatter direction: output windows written back by DMA (combine bwd)
# ---------------------------------------------------------------------------


class _Scatter(NamedTuple):
  """Output side of the backward (scatter) kernel.

  `obuf[slot]` holds the output rows of the block processed in grid step
  `slot` at the same buffer positions as its x rows (buffer row = x row +
  DELTA); its windows are written back to `out_hbm` by DMA. Partial granules
  are staged into the slot one step ahead (see the module docstring).
  """

  out_hbm: Any  # [R, E] output in HBM
  obuf: Any  # VMEM [2, J, E]
  wsem: Any  # DMA semaphores [2]: window writes issued from each slot
  ssem: Any  # DMA semaphores [2]: granules staged from HBM into each slot
  wcount: Any  # SMEM [2] int32: rows written from each slot (wait size)
  scount: Any  # SMEM [2] int32: rows staged into each slot (wait size)
  g: int  # experts
  jrows: int  # buffer rows J
  a: int  # row alignment A
  dynamic_dma: bool

  def wait_writes(self, slot, nrows):
    _wait_rows(lambda n: self.obuf.at[slot, pl.ds(0, n)], self.wsem.at[slot], nrows, self.jrows, self.a, self.dynamic_dma)

  def wait_staged(self, slot, nrows):
    _wait_rows(lambda n: self.obuf.at[slot, pl.ds(0, n)], self.ssem.at[slot], nrows, self.jrows, self.a, self.dynamic_dma)


def _stage_from_hbm(sc: _Scatter, tab, dst_slot, gi):
  """Reads expert gi's HS / TS granules of the block described by `tab` into obuf[dst_slot].

  Returns the number of rows issued (counted on ssem[dst_slot]).
  """
  g, a = sc.g, sc.a
  delta = tab[_T_DELTA * g + gi]
  nrows = jnp.int32(0)
  for row in (_T_HS, _T_TS):
    st = tab[row * g + gi]

    @pl.when(st >= 0)
    def _(st=st):
      pltpu.make_async_copy(
          sc.out_hbm.at[pl.ds(_al(st, a), a)],
          sc.obuf.at[dst_slot, pl.ds(_al(st + delta, a), a)],
          sc.ssem.at[dst_slot],
      ).start()

    nrows = nrows + a * (st >= 0).astype(jnp.int32)
  return nrows


def _carry_granule(sc: _Scatter, tab, src_slot, dst_slot, row_src, row_dst, gi):
  """Vector-copies one granule of expert gi: obuf[src_slot] row tab[row_src] -> obuf[dst_slot] row tab[row_dst]."""
  g, a = sc.g, sc.a
  src = tab[row_src * g + gi]

  @pl.when(src >= 0)
  def _():
    dst = tab[row_dst * g + gi]
    sc.obuf[dst_slot, pl.ds(_al(dst, a), a), :] = sc.obuf[src_slot, pl.ds(_al(src, a), a), :]


def _scatter_prologue(sc: _Scatter, tab_cur, slot):
  """Makes obuf[slot] hold the staged granules of this block before the compute.

  Step 0 stages its own granules (from HBM only). Every step waits until the
  granules read from HBM into obuf[slot] (issued one step ahead) have landed,
  then carries the granules written by the previous step (HCR -> HCD,
  TCR -> TCD) from its slot.
  """
  b = pl.program_id(0)

  @pl.when(b == 0)
  def _():
    def body(gi, n):
      return n + _stage_from_hbm(sc, tab_cur, 0, gi)

    sc.scount[0] = lax.fori_loop(0, sc.g, body, jnp.int32(0))

  sc.wait_staged(slot, sc.scount[slot])

  @pl.when(b > 0)
  def _():
    def body(gi, carry):
      _carry_granule(sc, tab_cur, 1 - slot, slot, _T_HCR, _T_HCD, gi)
      _carry_granule(sc, tab_cur, 1 - slot, slot, _T_TCR, _T_TCD, gi)
      return carry

    lax.fori_loop(0, sc.g, body, 0)


def _scatter_epilogue(sc: _Scatter, tab_cur, tab_nxt, slot):
  """Writes obuf[slot] windows back and stages the next block into obuf[1 - slot].

  Shared granules are written by consecutive blocks, so the previous block's
  writes must land before this block's are issued. A granule staged from HBM
  for the next block never lies in a window this block writes (such granules
  are carried instead), so the staging reads overlap the writes. Both are
  issued from one scalar loop over the experts.
  """
  b = pl.program_id(0)
  nb = pl.num_programs(0)
  g, a = sc.g, sc.a

  @pl.when(b > 0)
  def _():
    sc.wait_writes(1 - slot, sc.wcount[1 - slot])

  def write(gi):
    n = tab_cur[_T_NL * g + gi]
    rs = tab_cur[_T_RS * g + gi]
    off = tab_cur[_T_OFF * g + gi]

    @pl.when(n > 0)
    def _():
      _copy_rows(
          lambda o, m: sc.obuf.at[slot, pl.ds(_al(off + o, a), m)],
          lambda o, m: sc.out_hbm.at[pl.ds(_al(rs + o, a), m)],
          n,
          sc.wsem.at[slot],
          a,
          sc.dynamic_dma,
      )

    return n

  def write_and_stage(gi, counts):
    nw, ns = counts
    return nw + write(gi), ns + _stage_from_hbm(sc, tab_nxt, 1 - slot, gi)

  def write_only(gi, nw):
    return nw + write(gi)

  @pl.when(b + 1 < nb)
  def _():
    nw, ns = lax.fori_loop(0, g, write_and_stage, (jnp.int32(0), jnp.int32(0)))
    sc.wcount[slot] = nw
    sc.scount[1 - slot] = ns

  @pl.when(b == nb - 1)
  def _():
    nw = lax.fori_loop(0, g, write_only, jnp.int32(0))
    sc.wait_writes(slot, nw)


def _store_owned_rows(obuf, slot, rows, cols, own, values):
  """obuf[slot, rows, cols] = values where `own`; rows of other blocks keep their staged value."""
  obuf[slot, rows, cols] = jnp.where(own, values.astype(obuf.dtype), obuf[slot, rows, cols])


# ---------------------------------------------------------------------------
# backward kernel
# ---------------------------------------------------------------------------


def _bwd_compute(mm, w_ref, dy_ref, dw_ref, buf, ibuf, obuf, slot, used, t0, *, c, e, en, jrows, split, rq):
  """Column-oriented backward compute over RQ-row chunks of the buffer.

  Per chunk (rows j in [q0, q0 + RQ), token-in-block tok_j, top-k slot k_j):
    GT  = x_rows . dy^T               [RQ, C]  (MXU; dy^T is the stationary operand)
    dw_j = GT[j, tok_j]               (masked lane reduction)
    dW[c, k] += sum_j [tok_j = c] [k_j = k] dw_j   (exact 0/1 MXU routing, 3 bf16 parts)
    S[j, c] = [tok_j = c] * W[c, k_j]  [RQ, C]  (no transposes: W^T is built once)
    obuf rows = S @ dy                 (MXU; dy column chunk is the stationary operand),
                                       rows of other blocks keep their staged value
  Large RQ keeps the MXU streaming many rows per stationary tile. The last chunk
  is clamped into the buffer; rows it shares with the previous chunk recompute
  identical obuf values and are excluded from dW.
  """
  dt = dy_ref.dtype
  w = w_ref[...]  # [C, 128] f32
  wt = w.T  # [128, C]
  wt_hi = wt.astype(dt)
  wt_lo = (wt - wt_hi.astype(jnp.float32)).astype(dt) if split else None
  iota_c = lax.broadcasted_iota(jnp.int32, (rq, c), 1)
  iota_l = lax.broadcasted_iota(jnp.int32, (rq, _LANES), 1)
  iota_r = lax.broadcasted_iota(jnp.int32, (rq, 1), 0)
  iota_cr = lax.broadcasted_iota(jnp.int32, (c, rq), 0)
  nq = (used + (rq - 1)) // rq

  def q_body(q, dw):
    qs = q * rq
    q0 = _chunk_start(q, rq, jrows)
    tokc, kc = _chunk_tokens(ibuf, slot, q0, rq, c, t0, used)  # [RQ, 1] each
    m = iota_c == tokc  # [RQ, C]
    # dW
    gt = mm(buf[slot, pl.ds(q0, rq), :], dy_ref[...], ((1,), (1,)))  # [RQ, C]
    dwc = jnp.sum(jnp.where(m, gt, 0.0), axis=1, keepdims=True)  # [RQ, 1]
    a_jk = jnp.where((iota_l == kc) & (iota_r + q0 >= qs), dwc, 0.0)  # [RQ, 128]
    tokrow, _ = _chunk_info(ibuf, slot, q0, rq, c, t0, used)  # [1, RQ]
    mt = (iota_cr == tokrow).astype(dt)  # [C, RQ]
    for part in _exact_bf16_parts(a_jk, dt):
      dw = dw + mm(mt, part)  # [C, 128]
    # obuf rows
    koh = (iota_l == kc).astype(dt)  # [RQ, 128]
    s_hi = jnp.where(m, mm(koh, wt_hi), 0.0).astype(dt)  # [RQ, C]
    s_lo = jnp.where(m, mm(koh, wt_lo), 0.0).astype(dt) if split else None
    own = tokc >= 0  # [RQ, 1]

    def n_body(n, carry):
      n0 = _al(n * en, en)
      dy_cols = dy_ref[:, pl.ds(n0, en)]  # [C, en]
      ob = mm(s_hi, dy_cols)
      if s_lo is not None:
        ob = ob + mm(s_lo, dy_cols)
      _store_owned_rows(obuf, slot, pl.ds(q0, rq), pl.ds(n0, en), own, ob)
      return carry

    lax.fori_loop(0, e // en, n_body, 0)
    return dw

  dw_ref[...] = lax.fori_loop(0, nq, q_body, jnp.zeros((c, _LANES), jnp.float32))


def _bwd_kernel(
    tab_cur,
    tab_nxt,
    w_ref,
    dy_ref,
    x_hbm,
    info_hbm,
    dw_ref,
    dx_hbm,
    buf,
    ibuf,
    obuf,
    xsem,
    isem,
    wsem,
    ssem,
    wcount,
    scount,
    *,
    g,
    jrows,
    e,
    split,
    a,
    upcast,
    dynamic_dma,
):
  """dW block and dx windows for one token block (see _bwd_compute and the module docstring)."""
  mm = functools.partial(_mm, upcast=upcast)
  b = pl.program_id(0)
  reads = (_WindowRead(x_hbm, buf, xsem, fill=None), _WindowRead(info_hbm, ibuf, isem, fill=_INFO_FILL))
  slot = _read_pipeline(tab_cur, tab_nxt, reads, g, jrows, a, dynamic_dma)
  sc = _Scatter(dx_hbm, obuf, wsem, ssem, wcount, scount, g, jrows, a, dynamic_dma)
  c = dy_ref.shape[0]

  _scatter_prologue(sc, tab_cur, slot)
  _bwd_compute(
      mm,
      w_ref,
      dy_ref,
      dw_ref,
      buf,
      ibuf,
      obuf,
      slot,
      _used_rows(tab_cur, g),
      b * c,
      c=c,
      e=e,
      en=_col_chunk(e),
      jrows=jrows,
      split=split,
      rq=_scatter_row_chunk(jrows),
  )
  _scatter_epilogue(sc, tab_cur, tab_nxt, slot)


def _bwd_call(x, w_pad, tab, info, dy, *, block_tokens, num_groups, split, interpret, align):
  r, e = x.shape
  t = w_pad.shape[0]
  c, g = block_tokens, num_groups
  nb = t // c
  k = r // t
  jrows = _buffer_rows(c, k, g, align)
  ntab = _tab_stride(g)
  upcast, dynamic_dma = _kernel_modes(interpret)
  in_specs = _common_specs(nb, ntab, c) + [
      pl.BlockSpec((c, e), lambda b: (b, 0)),
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  # dW [T, 128] (weights in lanes [0, K)) and dx [R, E] written by DMA.
  out_specs = [pl.BlockSpec((c, _LANES), lambda b: (b, 0)), pl.BlockSpec(memory_space=pl.ANY)]
  itemsize = jnp.dtype(x.dtype).itemsize
  vmem = (
      4 * jrows * e * itemsize  # buf + obuf
      + 2 * jrows * _LANES * itemsize  # row info
      + 2 * c * e * itemsize  # dy blocks
      + 4 * c * _LANES * 4  # weight / dW blocks
      + 24 * 1024 * 1024  # chunk temporaries, compiler scratch
  )
  kernel = functools.partial(
      _bwd_kernel, g=g, jrows=jrows, e=e, split=split, a=align, upcast=upcast, dynamic_dma=dynamic_dma
  )
  (tab, w_pad, dy, x, info), mat = _match_vma(tab, w_pad, dy, x, info)
  return pl.pallas_call(
      kernel,
      out_shape=(_out_struct((t, _LANES), jnp.float32, mat), _out_struct((r, e), x.dtype, mat)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(nb,),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=[
              pltpu.VMEM((2, jrows, e), x.dtype),  # buf
              pltpu.VMEM((2, jrows, _LANES), info.dtype),  # row info
              pltpu.VMEM((2, jrows, e), x.dtype),  # obuf (+ partial-granule staging)
              pltpu.SemaphoreType.DMA((2,)),  # x reads
              pltpu.SemaphoreType.DMA((2,)),  # row-info reads
              pltpu.SemaphoreType.DMA((2,)),  # writes
              pltpu.SemaphoreType.DMA((2,)),  # partial-granule staging
              pltpu.SMEM((2,), jnp.int32),  # rows written per slot
              pltpu.SMEM((2,), jnp.int32),  # rows staged per slot
          ],
      ),
      compiler_params=_compiler_params(min(vmem, _VMEM_LIMIT)),
      interpret=interpret,
      name="moe_combine_bwd",
  )(tab, tab, w_pad, dy, x, info)


# ---------------------------------------------------------------------------
# combine custom VJP wrapper
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _combine_vjp(x, w, sort_idx, tab, cfg):
  return _fwd_impl(x, w, sort_idx, tab, cfg)


def _fwd_impl(x, w, sort_idx, tab, cfg):
  _, k = w.shape
  return _fwd_call(x, _padded_weights(w), tab, _row_info(sort_idx, k), **dict(cfg))


def _combine_vjp_fwd(x, w, sort_idx, tab, cfg):
  return _fwd_impl(x, w, sort_idx, tab, cfg), (x, w, sort_idx, tab)


def _combine_vjp_bwd(cfg, res, dy):
  x, w, sort_idx, tab = res
  _, k = w.shape
  dw_pad, dx = _bwd_call(x, _padded_weights(w), tab, _row_info(sort_idx, k), dy.astype(x.dtype), **dict(cfg))
  return dx, dw_pad[:, :k].astype(w.dtype), None, None


_combine_vjp.defvjp(_combine_vjp_fwd, _combine_vjp_bwd)


def combine(
    x,
    sort_idx,
    weights,
    group_sizes,
    *,
    block_tokens=None,
    interpret=None,
    align=8,
):
  """y[t] = sum_k weights[t, k] * x[argsort(sort_idx)[t*K + k]] (f32 accumulate, x.dtype out).

  Args:
    x: [T*K, E] expert outputs in expert-sorted order (bf16 for the kernel path).
    sort_idx: [T*K] int, stable argsort of the flat (token-major) expert ids.
    weights: [T, K] routing weights (any float dtype; non-bf16 weights use an exact hi + lo split).
    group_sizes: [G] int, rows per expert group of `x` (must sum to T*K).
    block_tokens: tokens per grid step (default: 256 or the largest power-of-two divisor of T >= 128).
    interpret: Pallas interpret mode (default: Mosaic interpreter off-TPU).
    align: DMA / VMEM row alignment, a power of two >= 8.

  Returns:
    [T, E] array in x.dtype. Differentiable w.r.t. x and weights. Falls back to
    `combine_reference` when the kernel preconditions (see kernel_supported) do not hold.
  """
  t, k = weights.shape
  align = _check_align(align)
  if block_tokens is None:
    block_tokens = _default_block_tokens(t)
  if interpret is None:
    interpret = _default_interpret()
  if group_sizes is None or not kernel_supported(x.shape, x.dtype, t, k, block_tokens, align):
    return combine_reference(x, sort_idx, weights)
  tab = _metadata(sort_idx, group_sizes, t, k, block_tokens, align)
  cfg = (
      ("block_tokens", block_tokens),
      ("num_groups", int(group_sizes.shape[0])),
      ("split", jnp.dtype(weights.dtype) != jnp.bfloat16),
      ("interpret", interpret),
      ("align", align),
  )
  return _combine_vjp(x, weights, sort_idx, tab, cfg)
