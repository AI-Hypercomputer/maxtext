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

"""Fused FP8 linear GMM + top-k combine (E4) for the DSv3 routed-expert MLP.

In the FP8 routed-expert forward pass (`dsv3_experts.py`), each chunk of sorted
local-expert slots ends with the linear GMM and the top-k combine:

  routed_out [C, D0, 128] = (act[m] @ w_q[expert(m)]) * out_scale   (bf16)
  y [N, D0, 128]         += sum over slots m of token t of routed_out[m]

Run as `gmm_v2` + `ops.ragged_gather_reduce_tc`, `routed_out` is written to and
read back from HBM, and every token row of `y` is written by the reduce. This
kernel computes each `[tile_m, D]` output tile of the GMM in VMEM and DMAs its
rows straight to their tokens' rows of `y`:

* Grid steps are "visits" `(expert, rows)` of at most `tile_m` sorted slots of
  one expert, in slot order. The expert's whole `[F, D]` weight is resident in
  VMEM (double-buffered across experts), so weights are read from HBM once.
  Partially filled visits only compute their rows, rounded up to `bucket_m`.
* A slot row is written to `y` directly if it is its token's first slot of the
  chunk and `y` starts at zero. Otherwise (later slots of a token, or
  accumulating into an existing `y`) the kernel reads the token's row of `y`,
  adds the tile row in f32 and writes it back (read-modify-write), so the
  token's slots are summed in slot order like `ragged_gather_reduce_tc`. A
  bitmap of the tokens seen so far tells first slots apart.
* Rows of one visit are distinct tokens (a token picks an expert at most once),
  and so are the rows of consecutive visits of one expert. Read-modify-writes
  only need to wait for the writes of the previous visit at the first visit of
  each expert, which keeps the row writes asynchronous.
* When `y` starts at zero, the tokens without slots are zero-filled with DMAs
  from a zero VMEM buffer, a window of runs per visit. The runs are the gaps
  between the ascending tokens of the slots (`RaggedGatherReduceMetadata.tokens`
  of the chunk), computed with dense XLA ops.
* The TPU scalar core pays dearly for branches, integer divisions and loop
  iterations: the per-row loops are branch-free and unrolled, and the rows of
  the next visit are classified in straight-line code between the matmuls,
  where it co-issues with the MXU.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.experimental.lineage.gmmv2 import gmmv2_kernel
from maxtext.experimental.lineage.gmmv2 import relayout

LANES = 128
_SUBLANE = 32  # Sublane tiling of the fp8 act rows.
_IDX_ALIGN = 128  # Alignment of the SMEM token windows.
_MXU_COLS = 256  # TPU v7x MXU column size.
_ISSUE_UNROLL = 16  # Row DMAs issued per loop iteration.
_SEEN_BITS = 32  # Tokens per word of the seen-token bitmap.
# SMEM token windows: those of the visit, of the next visit (classified during
# the visit) and of the prefetched one; a power of two for a cheap modulo.
_TOK_SLOTS = 4


@dataclasses.dataclass(frozen=True)
class Tiles:
  """Tile sizes of `linear_combine_fwd`.

  Attributes:
    tile_m: Max sorted slots per visit (grid step). A multiple of 32.
    tile_n: Output columns per iteration of the matmul loop. Divides D and is a
      multiple of 256.
    bucket_m: Row granularity of the GMM of partially filled visits: a power of
      two multiple of 32 dividing tile_m.
    zero_window: Zero runs (one per slot) zero-filled per visit.
    zero_rows: Rows of the zero VMEM buffer, the max rows per zero-fill DMA of
      the fast path.
  """

  tile_m: int = 256
  tile_n: int = 512
  bucket_m: int = 64
  zero_window: int = 256
  zero_rows: int = 32


DEFAULT_TILES = Tiles()


def _count_visits(group_sizes: jax.Array, tile_m: int) -> jax.Array:
  """Returns the number of visits, matching `_fill_metadata`."""
  gs = group_sizes.astype(jnp.int32)
  m_off = jnp.cumsum(gs) - gs
  n_tiles = jnp.where(gs > 0, (gs + (m_off % _SUBLANE) + tile_m - 1) // tile_m, 0)
  return jnp.sum(n_tiles)


def _fill_metadata(
    gs_ref, v2e_ref, v2m_ref, v2w_ref, w2e_ref, *, num_experts: int, tile_m: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Fills the visit tables.

  Visit v covers sorted slots [v2m[v], v2m[v + 1]) of expert v2e[v], whose
  weight is the v2w[v]-th weight load. The 32-aligned rows of a visit's act
  block fit in tile_m rows.

  Args:
    gs_ref: Group sizes (SMEM).
    v2e_ref: Expert of each visit (SMEM).
    v2m_ref: First slot of each visit, followed by the end (SMEM).
    v2w_ref: Weight load of each visit (SMEM).
    w2e_ref: Expert of each weight load (SMEM).
    num_experts: Number of experts.
    tile_m: Max slots per visit.

  Returns:
    (num_visits, num_wloads, num_rows).
  """
  v2m_ref[0] = 0

  def fill_expert(e, carry):
    v_idx, wl, m_off = carry
    size = gs_ref[e]
    end_m = m_off + size
    n_tiles = jnp.where(size > 0, pl.cdiv(size + m_off % _SUBLANE, tile_m), 0)

    @pl.when(size > 0)
    def _record_wload():
      w2e_ref[wl] = e

    def fill_tile(v, cur):
      rows = jnp.minimum(tile_m - cur % _SUBLANE, end_m - cur)
      v2e_ref[v] = e
      v2w_ref[v] = wl
      v2m_ref[v] = cur
      v2m_ref[v + 1] = cur + rows
      return cur + rows

    lax.fori_loop(v_idx, v_idx + n_tiles, fill_tile, m_off)
    return v_idx + n_tiles, wl + jnp.where(size > 0, 1, 0), end_m

  return lax.fori_loop(
      0,
      num_experts,
      fill_expert,
      (jnp.int32(0), jnp.int32(0), jnp.int32(0)),
  )


def _unrolled_loop(n, unroll: int, fn):
  """Calls fn(i) for i in [0, n), unrolling the body `unroll` times."""
  num_full = n // unroll

  @pl.loop(0, num_full)
  def _full(c):
    for u in range(unroll):
      fn(c * unroll + u)

  @pl.loop(num_full * unroll, n)
  def _rest(i):
    fn(i)


def _unrolled_fori(n, unroll: int, body, init):
  """`lax.fori_loop(0, n, body, init)` with the body unrolled `unroll` times."""
  num_full = n // unroll

  def chunk(c, carry):
    for u in range(unroll):
      carry = body(c * unroll + u, carry)
    return carry

  carry = lax.fori_loop(0, num_full, chunk, init)
  return lax.fori_loop(num_full * unroll, n, body, carry)


def _pieces(n: int, parts: int) -> list[tuple[int, int]]:
  """Splits [0, n) into `parts` contiguous static ranges."""
  return [(p * n // parts, (p + 1) * n // parts) for p in range(parts)]


def _wait_rows(sem, ref, num_rows, max_rows: int):
  """Waits on sem for DMAs totalling num_rows leading rows shaped like ref's.

  A single wait of a dynamic-size descriptor: cheaper on the scalar core than
  a static wait per set bit of num_rows.

  Args:
    sem: DMA semaphore.
    ref: Ref whose leading rows have the shape of the DMAs' rows.
    num_rows: Number of rows, at most max_rows.
    max_rows: Bound on num_rows, at most ref.shape[0].
  """
  assert max_rows <= ref.shape[0], (max_rows, ref.shape)

  @pl.when(num_rows > 0)
  def _wait():
    rows = ref.at[pl.ds(0, num_rows)]
    pltpu.make_async_copy(rows, rows, sem).wait()


def _zero_runs(
    sorted_tokens: jax.Array,
    num_rows: jax.Array,
    *,
    num_out_tokens: int,
    num_windows: int,
    window: int,
    max_run: int,
) -> tuple[jax.Array, jax.Array]:
  """Returns the runs of tokens without rows, for the zero-fill.

  Run i is the gap of tokens before sorted token i (or before num_out_tokens,
  for i = num_rows): its first token and length (0 for repeated or consecutive
  tokens). The runs are laid out in windows of `window` runs, each as its starts
  followed by its lengths. Dense elementwise ops only: XLA scatters and sorts
  are slow on TPU.

  Args:
    sorted_tokens: Ascending tokens of the rows (first num_rows entries).
    num_rows: Number of rows.
    num_out_tokens: Number of output tokens.
    num_windows: Number of windows; covers num_rows + 1 runs.
    window: Runs per window.
    max_run: Max run length of the fast zero-fill.

  Returns:
    (is_long, runs): whether each window has a run longer than max_run, and
    the `[num_windows, 2, window]` runs, flattened.
  """
  size = num_windows * window
  tokens = sorted_tokens.astype(jnp.int32)[:size]
  tokens = jnp.pad(tokens, (0, size - tokens.shape[0]))
  idx = jnp.arange(size, dtype=jnp.int32)
  tokens = jnp.where(idx < num_rows, tokens, num_out_tokens)
  prev = jnp.concatenate([jnp.full((1,), -1, jnp.int32), tokens[:-1]])
  length = jnp.maximum(tokens - prev - 1, 0).reshape(num_windows, window)
  start = (prev + 1).reshape(num_windows, window)
  is_long = jnp.any(length > max_run, axis=1).astype(jnp.int32)
  return is_long, jnp.stack([start, length], axis=1).reshape(-1)


@dataclasses.dataclass(frozen=True)
class _Config:
  num_experts: int
  size_f: int
  size_d: int
  num_out_tokens: int
  tile_m: int
  tile_n: int
  tok_window: int
  zero_window: int
  zero_rows: int
  accumulate: bool
  bucket_m: int


def _kernel(
    gs_ref,
    start_ref,
    scale_ref,
    zmeta_ref,
    act_hbm,
    w_hbm,
    tok_hbm,
    *refs,
    cfg: _Config,
):
  """Kernel body over grid `(max(num_visits, 1),)`.

  Visit v reads the y rows of its read-modify-write rows, zero-fills the runs
  of zero window v, computes its GMM tile into obuf[v % 2] while classifying
  the rows of visit v + 1, adds the read rows and writes its rows to y.

  Args:
    gs_ref: Group sizes (SMEM).
    start_ref: Sorted slot of row 0 of act (SMEM).
    scale_ref: GMM output scale (SMEM).
    zmeta_ref: Whether each zero window has a run longer than `zero_rows`
      (SMEM).
    act_hbm: fp8 act `[C, F]`.
    w_hbm: fp8 weights `[E * F, D]`.
    tok_hbm: Output token of each sorted slot.
    *refs: Zero runs (or the aliased y input when accumulating), y, scratch.
    cfg: Static configuration.
  """
  if cfg.accumulate:
    _, y_hbm, *scratch = refs  # The aliased y input is y_hbm.
    zinfo_hbm = None
  else:
    zinfo_hbm, y_hbm, *scratch = refs
  (
      meta_ref,
      v2e_ref,
      v2m_ref,
      v2w_ref,
      w2e_ref,
      seen_ref,
      tok_smem,
      rmw_rows_smem,
      zwin_smem,
      zlist_smem,
      act_buf,
      w_buf,
      obuf,
      rbuf,
      zbuf,
      act_sem,
      tok_sem,
      w_sem,
      wr_sem,
      rd_sem,
      zero_sem,
      zt_sem,
  ) = scratch
  tile_m, size_f, size_d = cfg.tile_m, cfg.size_f, cfg.size_d
  nbt = cfg.num_out_tokens
  tw = cfg.tok_window
  ze = cfg.zero_window
  zrows = cfg.zero_rows
  # meta_ref fields.
  num_visits_i, num_wloads_i, num_rows_i, zcount_i = range(4)
  wpending_i = 4  # + slot: rows of the slot's writes still to be waited for.
  rmw_count_i = 6  # + slot: read-modify-write rows of the slot's visit.

  v = pl.program_id(0)
  last_step = v == pl.num_programs(0) - 1

  def visit(idx):
    m_s = v2m_ref[idx]
    m_e = v2m_ref[idx + 1]
    a_s = m_s // _SUBLANE * _SUBLANE
    return v2e_ref[idx], m_s, m_e, a_s

  def act_dma(idx, slot):
    _, _, m_e, a_s = visit(idx)
    a_s = pl.multiple_of(a_s, _SUBLANE)
    fetch = pl.multiple_of(pl.cdiv(m_e - a_s, _SUBLANE) * _SUBLANE, _SUBLANE)
    return pltpu.make_async_copy(
        act_hbm.at[pl.ds(a_s, fetch)],
        act_buf.at[slot, pl.ds(0, fetch)],
        act_sem.at[slot],
    )

  def tok_win_start(idx):
    row = start_ref[0] + v2m_ref[idx]
    return jnp.minimum(row // _IDX_ALIGN * _IDX_ALIGN, tok_hbm.shape[0] - tw)

  def tok_dma(idx):
    ws = pl.multiple_of(tok_win_start(idx), _IDX_ALIGN)
    slot = jnp.bitwise_and(idx, _TOK_SLOTS - 1)
    return pltpu.make_async_copy(
        tok_hbm.at[pl.ds(ws, tw)],
        tok_smem.at[pl.ds(slot * tw, tw)],
        tok_sem.at[slot],
    )

  def w_dma(wl, slot):
    row = pl.multiple_of(w2e_ref[wl] * size_f, size_f)
    return pltpu.make_async_copy(w_hbm.at[pl.ds(row, size_f)], w_buf.at[slot], w_sem.at[slot])

  def zt_dma(win):
    assert zinfo_hbm is not None
    start = pl.multiple_of(win * 2 * ze, 2 * ze)
    return pltpu.make_async_copy(zinfo_hbm.at[pl.ds(start, 2 * ze)], zwin_smem, zt_sem.at[0])

  def wait_writes(slot):
    _wait_rows(wr_sem.at[slot], y_hbm, meta_ref[wpending_i + slot], tile_m)
    meta_ref[wpending_i + slot] = 0

  def write_row(obuf_row, tok_base, slot, i):
    """Writes row obuf_row + i of obuf[slot] to token tok_smem[tok_base + i]."""
    t = tok_smem[tok_base + i]
    pltpu.make_async_copy(
        obuf.at[slot, pl.ds(obuf_row + i, 1)],
        y_hbm.at[pl.ds(t, 1)],
        wr_sem.at[slot],
    ).start()

  def classify_row(tok_base, rmw_base, num_rows, i, cnt):
    """Classifies row i of a visit.

    Appends the row to the visit's read-modify-write list if its token was
    seen before. Rows past num_rows repeat the last row and are not counted.
    Branch-free and free of divisions: the TPU scalar core pays dearly for
    both.

    Args:
      tok_base: SMEM index of the token of the visit's first row.
      rmw_base: Index of the visit's list in rmw_rows_smem.
      num_rows: Rows of the visit.
      i: Row of the visit (static or traced).
      cnt: Read-modify-write rows so far.

    Returns:
      The new read-modify-write row count.
    """
    t = tok_smem[tok_base + jnp.minimum(i, num_rows - 1)]
    word = jnp.right_shift(t, 5)
    bit = jnp.left_shift(jnp.int32(1), jnp.bitwise_and(t, _SEEN_BITS - 1))
    seen = seen_ref[word]
    seen_ref[word] = seen | bit
    rmw_rows_smem[rmw_base + cnt] = i
    dup = jnp.logical_and((seen & bit) != 0, i < num_rows)
    return cnt + dup.astype(jnp.int32)

  def visit_tok_base(idx):
    """SMEM index of the token of visit idx's first row."""
    return jnp.bitwise_and(idx, _TOK_SLOTS - 1) * tw + start_ref[0] + v2m_ref[idx] - tok_win_start(idx)

  def classify_slow(idx):
    """Classifies the rows of visit idx with a loop (in the prologue)."""
    num_rows = v2m_ref[idx + 1] - v2m_ref[idx]
    meta_ref[rmw_count_i + jnp.bitwise_and(idx, 1)] = _unrolled_fori(
        num_rows,
        _ISSUE_UNROLL,
        functools.partial(
            classify_row,
            visit_tok_base(idx),
            jnp.bitwise_and(idx, 1) * tile_m,
            num_rows,
        ),
        jnp.int32(0),
    )

  def compact_zero_runs():
    """Lists the nonempty runs of the loaded window; returns their count."""

    def compact(j, cnt):
      zlist_smem[cnt] = j
      return cnt + (zwin_smem[ze + j] > 0).astype(jnp.int32)

    return _unrolled_fori(ze, _ISSUE_UNROLL, compact, jnp.int32(0))

  def zero_run(k, count):
    """Zero-fills the k-th listed run (at most zrows long)."""
    j = zlist_smem[k]
    length = zwin_smem[ze + j]
    pltpu.make_async_copy(
        zbuf.at[pl.ds(0, length)],
        y_hbm.at[pl.ds(zwin_smem[j], length)],
        zero_sem.at[0],
    ).start()
    return count + length

  def zero_window_slow():
    """Zero-fills the runs of the loaded window, of any length."""

    def body(j, count):
      start = zwin_smem[j]
      length = zwin_smem[ze + j]
      num_full = length // zrows

      @pl.loop(0, num_full)
      def _full(q):
        pltpu.make_async_copy(zbuf, y_hbm.at[pl.ds(start + q * zrows, zrows)], zero_sem.at[0]).start()

      rem = length - num_full * zrows

      @pl.when(rem > 0)
      def _rest():
        pltpu.make_async_copy(
            zbuf.at[pl.ds(0, rem)],
            y_hbm.at[pl.ds(start + num_full * zrows, rem)],
            zero_sem.at[0],
        ).start()

      return count + length

    meta_ref[zcount_i] = lax.fori_loop(0, ze, body, meta_ref[zcount_i])

  # Prologue.
  @pl.when(v == 0)
  def _prologue():
    num_visits, num_wloads, num_rows = _fill_metadata(
        gs_ref,
        v2e_ref,
        v2m_ref,
        v2w_ref,
        w2e_ref,
        num_experts=cfg.num_experts,
        tile_m=tile_m,
    )
    meta_ref[num_visits_i] = num_visits
    meta_ref[num_wloads_i] = num_wloads
    meta_ref[num_rows_i] = num_rows
    meta_ref[zcount_i] = 0
    for slot in range(2):
      meta_ref[wpending_i + slot] = 0
      meta_ref[rmw_count_i + slot] = 0
    if not cfg.accumulate:
      zbuf[...] = jnp.zeros_like(zbuf)

      @pl.loop(0, seen_ref.shape[0])
      def _clear_seen(i):
        seen_ref[i] = 0

    @pl.when(num_visits > 0)
    def _start_first():
      w_dma(0, 0).start()
      act_dma(0, 0).start()
      tok_dma(0).start()
      if not cfg.accumulate:
        tok_dma(0).wait()
        classify_slow(0)

      @pl.when(num_visits > 1)
      def _second_tokens():
        tok_dma(1).start()

  num_visits = meta_ref[num_visits_i]
  num_wloads = meta_ref[num_wloads_i]
  num_rows = meta_ref[num_rows_i]
  num_windows = pl.cdiv(num_rows + 1, ze)

  @pl.when(v < num_visits)
  def _step():
    slot = jnp.bitwise_and(v, 1)
    e, m_s, m_e, a_s = visit(v)
    nrows = m_e - m_s
    lo = m_s - a_s  # Tile row of slot m_s.
    wl = v2w_ref[v]
    wslot = jnp.bitwise_and(wl, 1)
    is_new_expert = jnp.logical_or(v == 0, e != v2e_ref[jnp.maximum(v - 1, 0)])
    has_next = v + 1 < num_visits

    # obuf[slot] is free once the writes of visit v - 2 are done.
    wait_writes(slot)

    @pl.when(is_new_expert)
    def _wait_w():
      w_dma(wl, wslot).wait()

    act_dma(v, slot).wait()
    if cfg.accumulate:
      tok_dma(v).wait()

    @pl.when(has_next)
    def _prefetch_next():
      act_dma(v + 1, 1 - slot).start()

    @pl.when(v + 2 < num_visits)
    def _prefetch_tokens():
      tok_dma(v + 2).start()

    @pl.when(jnp.logical_and(is_new_expert, wl + 1 < num_wloads))
    def _prefetch_w():
      w_dma(wl + 1, 1 - wslot).start()

    if not cfg.accumulate:

      @pl.when(v < num_windows)
      def _load_zero_window():
        zt_dma(v).start()

    tok_base = visit_tok_base(v)
    rmw_base = slot * tile_m
    # Read-modify-write rows: all of them when accumulating, else the rows
    # classified during the previous visit (or the prologue).
    if cfg.accumulate:
      num_rmw = nrows
      rmw_row = lambda j: j
    else:
      num_rmw = meta_ref[rmw_count_i + slot]
      rmw_row = lambda j: rmw_rows_smem[rmw_base + j]

    # The previous visit may share tokens with this one only if it belongs to
    # another expert: wait for its writes before reading y.
    @pl.when(is_new_expert)
    def _order_writes():
      wait_writes(1 - slot)

    def read_row(j):
      t = tok_smem[tok_base + rmw_row(j)]
      pltpu.make_async_copy(y_hbm.at[pl.ds(t, 1)], rbuf.at[pl.ds(j, 1)], rd_sem.at[0]).start()

    _unrolled_loop(num_rmw, _ISSUE_UNROLL, read_row)

    if not cfg.accumulate:

      @pl.when(v < num_windows)
      def _zero_window():
        zt_dma(v).wait()

        @pl.when(zmeta_ref[v] != 0)
        def _long_runs():
          zero_window_slow()

        @pl.when(zmeta_ref[v] == 0)
        def _short_runs():
          meta_ref[zcount_i] = _unrolled_fori(compact_zero_runs(), _ISSUE_UNROLL, zero_run, meta_ref[zcount_i])

      @pl.when(has_next)
      def _wait_next_tokens():
        tok_dma(v + 1).wait()

    # Classify the next visit's rows during the GMM: straight-line scalar code
    # co-issues with the MXU, while the TPU scalar core pays dearly for loops
    # and branches. Without a next visit, classify this visit again into the
    # unused list: idempotent.
    nxt = jnp.minimum(v + 1, num_visits - 1)
    classify_next = functools.partial(
        classify_row,
        visit_tok_base(nxt),
        (1 - slot) * tile_m,
        v2m_ref[nxt + 1] - v2m_ref[nxt],
    )
    scale = scale_ref[0]
    num_groups = size_d // cfg.tile_n

    def gmm(bucket_m: int):
      """The GMM of the first bucket_m tile rows into obuf[slot]."""
      cnt = jnp.int32(0)
      for g, piece in enumerate(_pieces(tile_m, num_groups)):
        c0 = g * cfg.tile_n
        lhs = act_buf[slot, :bucket_m]
        parts = []
        for sub in range(0, cfg.tile_n, _MXU_COLS):
          acc = lax.dot_general(
              lhs,
              w_buf[wslot, :, pl.ds(c0 + sub, _MXU_COLS)],
              (((1,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
          parts.append((acc * scale).astype(jnp.bfloat16))
        # The strided relayout stores need static offsets: g is static.
        relayout.store_2d_into_3d(
            obuf.at[slot],
            jnp.concatenate(parts, axis=-1),
            bucket_m,
            c0 // LANES,
        )
        if not cfg.accumulate:
          for i in range(*piece):
            cnt = classify_next(i, cnt)
      if not cfg.accumulate:
        meta_ref[rmw_count_i + 1 - slot] = cnt

    # Partially filled visits only compute their rows, rounded up to buckets.
    buckets = list(range(cfg.bucket_m, tile_m + 1, cfg.bucket_m))
    if len(buckets) == 1:
      gmm(tile_m)
    else:
      lax.switch(
          jnp.right_shift(lo + nrows - 1, cfg.bucket_m.bit_length() - 1),
          [functools.partial(gmm, b) for b in buckets],
      )

    # Read-modify-write: add the current y rows in f32.
    _wait_rows(rd_sem.at[0], rbuf, num_rmw, tile_m)

    @pl.loop(0, num_rmw)
    def _add(j):
      r = lo + rmw_row(j)
      obuf[slot, r] = (obuf[slot, r].astype(jnp.float32) + rbuf[j].astype(jnp.float32)).astype(jnp.bfloat16)

    _unrolled_loop(nrows, _ISSUE_UNROLL, functools.partial(write_row, lo, tok_base, slot))
    meta_ref[wpending_i + slot] = nrows

  @pl.when(last_step)
  def _epilogue():
    if not cfg.accumulate:
      # Zero-fill the windows no visit loaded.
      @pl.loop(jnp.minimum(num_visits, num_windows), num_windows)
      def _rest_windows(win):
        zt_dma(win).start()
        zt_dma(win).wait()
        zero_window_slow()

      # Each token is zero-filled at most once.
      _wait_rows(zero_sem.at[0], y_hbm, meta_ref[zcount_i], nbt)
    wait_writes(0)
    wait_writes(1)


def linear_combine_fwd(
    act: jax.Array,
    w_q: jax.Array,
    group_sizes: jax.Array,
    token_indices: jax.Array,
    start: jax.Array | int,
    sorted_tokens: jax.Array | None,
    *,
    out_scale: jax.Array,
    num_out_tokens: int,
    out: jax.Array | None = None,
    tiles: Tiles | None = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Returns `out` plus the top-k combine of the linear GMM of `act`.

  `y[token_indices[start + m]] += (act[m] @ w_q[expert(m)]) * out_scale` for
  the `sum(group_sizes)` rows m of act, sorted by expert; rows of a token are
  summed in f32 in row order. The GMM rows are rounded to bf16 before the sum,
  like `gmm_v2` + `ops.ragged_gather_reduce_tc`.

  Args:
    act: fp8 activations `[C, F]`; rows past sum(group_sizes) are ignored.
    w_q: fp8 weights `[E, F, D]`.
    group_sizes: Rows of act of each expert `int32[E]`; sum <= C.
    token_indices: Output token of each sorted slot, `int32[num_indices]` in [0,
      N) (e.g. `sort_indices // k`). Requires start + sum(group_sizes) <=
      num_indices; the indices past that are never read.
    start: Sorted slot of row 0 of act (the chunk start). May be traced.
    sorted_tokens: Required without `out`: `int32[L]` whose first
      sum(group_sizes) entries are the tokens of the rows in ascending order
      (`ops.RaggedGatherReduceMetadata.tokens` of the chunk). Used to zero-fill
      the tokens without rows.
    out_scale: f32 `[1, 1]` scale of the GMM (act scale * w_q scale).
    num_out_tokens: N, the number of output tokens.
    out: Optional bf16 `[N, D // 128, 128]` to accumulate into, donated. If
      None, the result starts at zero.
    tiles: Optional tile sizes.
    vmem_limit_bytes: Optional Pallas VMEM limit.

  Returns:
    The bf16 `[N, D // 128, 128]` result.
  """
  if tiles is None:
    tiles = DEFAULT_TILES
  size_c, size_f = act.shape
  num_experts, w_f, size_d = w_q.shape
  if not (gmmv2_kernel.is_fp8(act.dtype) and gmmv2_kernel.is_fp8(w_q.dtype)):
    raise ValueError(f"act and w_q must be fp8, got {act.dtype}, {w_q.dtype}.")
  if w_f != size_f or size_d % LANES or group_sizes.shape != (num_experts,):
    raise ValueError(f"Bad shapes: act {act.shape}, w_q {w_q.shape}, group_sizes" f" {group_sizes.shape}.")
  if token_indices.ndim != 1:
    raise ValueError(f"token_indices must be 1D, got {token_indices.shape}.")
  d0 = size_d // LANES
  out_shape = (num_out_tokens, d0, LANES)
  accumulate = out is not None
  if accumulate:
    if out.shape != out_shape or out.dtype != jnp.bfloat16:
      raise ValueError(f"out must be bf16 {out_shape}, got {out.shape}.")
  elif sorted_tokens is None:
    raise ValueError("sorted_tokens is required without out.")
  tile_m = min(tiles.tile_m, pl.cdiv(size_c, _SUBLANE) * _SUBLANE)
  tile_n = min(tiles.tile_n, size_d)
  bucket_m = min(tiles.bucket_m, tile_m)
  if (
      tile_m % _SUBLANE
      or size_d % tile_n
      or tile_n % _MXU_COLS
      or tile_m % bucket_m
      or bucket_m % _SUBLANE
      or bucket_m & (bucket_m - 1)
  ):
    raise ValueError(f"Unsupported tiles {tiles} for D={size_d}.")
  if size_c % _SUBLANE:
    raise ValueError(f"act rows ({size_c}) must be a multiple of {_SUBLANE}.")

  # SMEM token windows of a visit's rows, padded so that no window is clamped
  # past the end.
  tok_window = pl.cdiv(tile_m, _IDX_ALIGN) * _IDX_ALIGN + _IDX_ALIGN
  token_indices = token_indices.astype(jnp.int32)
  num_indices = token_indices.shape[0]
  tok_len = max(pl.cdiv(num_indices, _IDX_ALIGN) * _IDX_ALIGN, tok_window)
  if tok_len != num_indices:
    token_indices = jnp.pad(token_indices, (0, tok_len - num_indices))
  zero_window = tiles.zero_window
  zero_rows = tiles.zero_rows
  num_zero_windows = pl.cdiv(size_c + 1, zero_window)
  inputs = [act, w_q.reshape(num_experts * size_f, size_d), token_indices]
  if accumulate:
    zmeta = jnp.zeros((num_zero_windows,), jnp.int32)
    inputs.append(out)
  else:
    assert sorted_tokens is not None
    zmeta, zinfo = _zero_runs(
        sorted_tokens,
        jnp.sum(group_sizes, dtype=jnp.int32),
        num_out_tokens=num_out_tokens,
        num_windows=num_zero_windows,
        window=zero_window,
        max_run=zero_rows,
    )
    inputs.append(zinfo)

  num_visits = _count_visits(group_sizes, tile_m)
  max_visits = num_experts + pl.cdiv(size_c, tile_m) + 1
  cfg = _Config(
      num_experts=num_experts,
      size_f=size_f,
      size_d=size_d,
      num_out_tokens=num_out_tokens,
      tile_m=tile_m,
      tile_n=tile_n,
      tok_window=tok_window,
      zero_window=zero_window,
      zero_rows=zero_rows,
      accumulate=accumulate,
      bucket_m=bucket_m,
  )
  dma = pltpu.SemaphoreType.DMA
  scratch_shapes = [
      pltpu.SMEM((8,), jnp.int32),  # meta_ref
      pltpu.SMEM((max_visits,), jnp.int32),  # v2e
      pltpu.SMEM((max_visits + 1,), jnp.int32),  # v2m
      pltpu.SMEM((max_visits,), jnp.int32),  # v2w
      pltpu.SMEM((num_experts,), jnp.int32),  # w2e
      pltpu.SMEM((pl.cdiv(num_out_tokens, _SEEN_BITS),), jnp.int32),  # seen
      pltpu.SMEM((_TOK_SLOTS * tok_window,), jnp.int32),  # tok_smem
      pltpu.SMEM((2 * tile_m,), jnp.int32),  # rmw_rows_smem
      pltpu.SMEM((2 * zero_window,), jnp.int32),  # zwin_smem
      pltpu.SMEM((zero_window,), jnp.int32),  # zlist_smem
      pltpu.VMEM((2, tile_m, size_f), act.dtype),  # act_buf
      pltpu.VMEM((2, size_f, size_d), w_q.dtype),  # w_buf
      pltpu.VMEM((2, tile_m, d0, LANES), jnp.bfloat16),  # obuf
      pltpu.VMEM((tile_m, d0, LANES), jnp.bfloat16),  # rbuf
      pltpu.VMEM((zero_rows, d0, LANES), jnp.bfloat16),  # zbuf
      dma((2,)),  # act_sem
      dma((_TOK_SLOTS,)),  # tok_sem
      dma((2,)),  # w_sem
      dma((2,)),  # wr_sem
      dma((1,)),  # rd_sem
      dma((1,)),  # zero_sem
      dma((1,)),  # zt_sem
  ]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  scalars = (
      group_sizes.astype(jnp.int32),
      jnp.asarray(start, jnp.int32).reshape(1),
      out_scale.astype(jnp.float32).reshape(1),
      zmeta,
  )
  manual_axis_type = (jax.typeof(out) if accumulate else jax.typeof(act)).manual_axis_type
  flops = 2 * size_c * size_f * size_d
  bytes_accessed = num_experts * size_f * size_d + size_c * size_f + 2 * num_out_tokens * size_d * 2
  acc_tag = "-acc" if accumulate else ""
  name = f"linear_combine_fwd-g_{num_experts}-m_{size_c}-k_{size_f}-n_{size_d}" f"-tm_{tile_m}-tn_{tile_n}{acc_tag}"
  return pl.pallas_call(
      functools.partial(_kernel, cfg=cfg),
      out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalars),
          grid=(jnp.maximum(num_visits, 1),),
          in_specs=[hbm] * len(inputs),
          out_specs=hbm,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          dimension_semantics=("arbitrary",),
      ),
      input_output_aliases=({len(scalars) + len(inputs) - 1: 0} if accumulate else {}),
      cost_estimate=pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0),
      name=name,
  )(*scalars, *inputs)
