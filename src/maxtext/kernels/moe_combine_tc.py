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

"""Fused TensorCore (Pallas) MoE "combine": unpermute + top-k weighted sum.

Computes, for expert outputs `x` [R = T*K, E] stored in expert-sorted order
(row r holds flat token-slot `sort_idx[r]`, `sort_idx = argsort(experts)`):

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
    bwd:  dwrow[1, J]  = column sums of S01 * (dy_blk @ buf^T)   (<dy[tok(j)], x_j>)
          dW^T[K, C]   = onehot(k)[K, J] * dwrow  @  S01^T[J, C]
          dbuf[J, E]   = S^T[J, C] @ dy_blk[C, E]  -> DMA'd back to the same windows

No per-row gathers / scatters / sorts in XLA: a [R, 128] bf16 "row info" array
(token % 256, token // 256, k of every sorted x row; elementwise from sort_idx)
is DMA'd with the same windows as x, so every buffer row carries its (token, k).
S is one compare + select per element, with the row weights selected from the
[C, K] weight block by an exact 0/1 matmul (W @ onehot(k)). dW is a masked
column sum of dy @ buf^T routed to [K, C] by exact 0/1 matmuls. The per-block
DMA table needs the (block, expert) row counts: a one-hot histogram matmul.

Tiling / alignment (Mosaic requirement): every DMA and every dynamic VMEM
slice along the row (2nd-minor) dimension uses offsets AND sizes that are
multiples of ALIGN rows (default 8 = the bf16 (8,128)(2,1) memref tile; env
MAXTEXT_G4_COMBINE_ALIGN=16 for (16,128) tiling). So each range is widened to
the ALIGN-aligned window [lo & -A, round_up(hi, A)), in HBM and in VMEM (buffer
row = x row + delta, delta a multiple of A). Windows of the same block that
share a granule are coalesced into one segment (constant delta), so every x
row has a single buffer position per block.

Backward writes are whole windows. Granules that contain a range boundary
("partial granules") may hold rows owned by other token blocks: they are
read-modify-written. After the matmuls, the current dx granules are DMA'd from
HBM into `buf` at their buffer positions (x is no longer needed), then every
buffer row not owned by this block takes the staged value (one uniform select),
and the windows are written back. Grid steps run sequentially and each step
waits for the previous step's writes before staging, so the HBM value is final
for already-processed blocks; blocks processed later overwrite their own rows.
Every x row is owned by exactly one block, and every non-owned window row lies
in a staged partial granule.

Precision: all products are exact (bf16 x bf16 -> f32) and accumulation is
f32, i.e. the same arithmetic as the `float32_weight_sum=True` einsum. If the
routing weights are not bf16, S is split into bf16 hi + lo parts (2 matmuls).

Requirements for the kernel path (otherwise a pure-jnp reference is used):
x is 2-D bf16 with E % 128 == 0; T % block_tokens == 0; block_tokens % 128 == 0;
T <= 65280; T*K % ALIGN == 0; `group_sizes` sums to R; `sort_idx` is a
*stable* argsort of the flat expert ids (jnp.argsort default).

Env (read at trace time):
  MAXTEXT_G4_COMBINE_BLOCK    tokens per grid step (default 256).
  MAXTEXT_G4_COMBINE_ALIGN    row alignment of DMAs / VMEM slices (default 8).
  MAXTEXT_G4_COMBINE_DYNDMA   1 (default): one dynamic-size DMA per window and
                              one wait per step; 0: static power-of-2 chunks.
  MAXTEXT_G4_COMBINE_BUCKET   fwd contraction-size bucket in rows (default 512).
  MAXTEXT_G4_COMBINE_EN       output column chunk (default 256).
  MAXTEXT_G4_COMBINE_MERGE    bwd merge select dtype: bf16 (default) | f32.
  MAXTEXT_G4_COMBINE_WSPLIT   1 (default): exact hi+lo split of non-bf16
                              weights; 0: round weights to bf16 (1 matmul).
  MAXTEXT_G4_COMBINE_BWD      backward kernel: v5 (default, column-oriented) | v4.
  MAXTEXT_G4_COMBINE_BWD_RQ   v5 backward rows per chunk (default 1024).
  MAXTEXT_G4_COMBINE_MODE     microbenchmarks only (results are WRONG unless
                              "full"): full | dma | compute | nomerge.
"""

import functools
import os

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

__all__ = ["combine", "combine_reference", "kernel_supported"]

_LANES = 128
_CHUNK_ROWS = 128  # rows per full-size range DMA (static-chunk fallback)
_WAIT_ROWS = 256  # rows per full-size semaphore wait (static-chunk fallback)

# Rows of the per-block scalar table (each row has G entries).
#   RS    HBM start row of this range's (new part of the) window
#   NL    rows of that window part (multiple of A; 0 if empty / already covered)
#   OFF   buffer row of RS
#   DELTA buffer row - x row, for every row of this range's segment
#   HS    HBM start of the head partial granule (lo % A != 0), else -1
#   TS    HBM start of the tail partial granule (hi % A != 0), else -1
_T_RS, _T_NL, _T_OFF, _T_DELTA, _T_HS, _T_TS = range(6)
_NTAB = 6

_MODES = ("full", "dma", "compute", "nomerge")


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


def _env_int(name, default):
  return int(os.environ.get(name, str(default)))


def align_from_env() -> int:
  a = _env_int("MAXTEXT_G4_COMBINE_ALIGN", 8)
  if a < 8 or a & (a - 1):
    raise ValueError(f"MAXTEXT_G4_COMBINE_ALIGN must be a power of two >= 8, got {a}")
  return a


def _kernel_opts():
  """Static kernel options from env (hashable tuple of pairs)."""
  mode = os.environ.get("MAXTEXT_G4_COMBINE_MODE", "full").strip().lower()
  if mode not in _MODES:
    raise ValueError(f"MAXTEXT_G4_COMBINE_MODE must be one of {_MODES}, got {mode!r}")
  merge = os.environ.get("MAXTEXT_G4_COMBINE_MERGE", "bf16").strip().lower()
  if merge not in ("bf16", "f32"):
    raise ValueError(f"MAXTEXT_G4_COMBINE_MERGE must be bf16|f32, got {merge!r}")
  bwd = os.environ.get("MAXTEXT_G4_COMBINE_BWD", "v5").strip().lower()
  if bwd not in ("v4", "v5"):
    raise ValueError(f"MAXTEXT_G4_COMBINE_BWD must be v4|v5, got {bwd!r}")
  return (
      ("mode", mode),
      ("dyn", _env_int("MAXTEXT_G4_COMBINE_DYNDMA", 1) != 0),
      ("bucket", _env_int("MAXTEXT_G4_COMBINE_BUCKET", 512)),
      ("en", _env_int("MAXTEXT_G4_COMBINE_EN", 256)),
      ("merge", merge),
      ("bwd", bwd),
      ("rq", _env_int("MAXTEXT_G4_COMBINE_BWD_RQ", 1024)),
  )


def _buffer_rows(block_tokens, k, g, align=8):
  """Static bound on buffer rows: each non-empty range widens by <= 2*(A-1) rows."""
  ck = block_tokens * k
  n = ck + (2 * align - 2) * min(g, ck)
  return _round_up(n, 2 * _LANES)


def kernel_supported(x_shape, x_dtype, num_tokens, k, block_tokens, align=8) -> bool:
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
      and num_tokens <= 255 * 256
  )


def block_tokens_from_env(num_tokens: int) -> int:
  c = _env_int("MAXTEXT_G4_COMBINE_BLOCK", 256)
  while c > _LANES and num_tokens % c != 0:
    c //= 2
  return c


def _default_interpret():
  return jax.default_backend() != "tpu"


def _col_chunk(e, pref):
  for c in (pref, 256, _LANES):
    if c % _LANES == 0 and e % c == 0:
      return c
  return _LANES


def _row_chunk(jrows):
  return 256 if jrows % 256 == 0 else _LANES


def _buckets(jrows, step):
  """Static contraction sizes (multiples of the row chunk) ending at jrows."""
  jq = _row_chunk(jrows)
  step = max(jq, _round_up(step, jq))
  sizes = list(range(step, jrows, step)) + [jrows]
  return tuple(sizes)


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
  hs = jnp.where(nonempty & (jnp.bitwise_and(lo, a - 1) != 0), wa, -1)
  ts = jnp.where(nonempty & (jnp.bitwise_and(hi, a - 1) != 0), we - a, -1)
  tab = jnp.stack([rs, nl, off, delta, hs, ts], axis=1).astype(jnp.int32).reshape(nb, _NTAB * g)
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
  info = jnp.stack([jnp.bitwise_and(tok, 255), tok >> 8, rem], axis=-1).astype(jnp.bfloat16)
  return jnp.pad(info, ((0, 0), (0, _LANES - info.shape[1])))


def _padded_weights(w):
  """[T, 128] f32 (weights in lanes [0, K))."""
  return jnp.pad(w.astype(jnp.float32), ((0, 0), (0, _LANES - w.shape[1])))


# ---------------------------------------------------------------------------
# in-kernel building blocks
# ---------------------------------------------------------------------------


def _copy_rows(src_fn, dst_fn, nrows, sem, align, dyn):
  """Starts DMAs copying `nrows` (dynamic, > 0, multiple of align) rows.

  src_fn(o, n) / dst_fn(o, n) return the ref windows for rows [o, o+n) of the
  range; o is a multiple of `align`.
  """
  if dyn:
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


def _wait_rows(desc_fn, sem, nrows, max_rows, align, dyn):
  """Waits until `nrows` (<= max_rows, multiple of align) rows have landed on `sem`.

  desc_fn(n) returns an n-row ref (offset 0) with the same row size as the DMAs.
  """
  if dyn:

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
  fill = jnp.full((_LANES, buf.shape[-1]), value, buf.dtype)
  for s in range(nslots):

    def body(i, carry, s=s):
      buf[s, pl.ds(_al(i * _LANES, _LANES), _LANES), :] = fill
      return carry

    lax.fori_loop(0, jrows // _LANES, body, 0)


def _issue_range_reads(tab, pairs, dst_slot, g, a, dyn):
  """DMAs every (aligned) window part of this block: for (hbm, buf, sem) in pairs."""

  def body(gi, carry):
    n = tab[_T_NL * g + gi]

    @pl.when(n > 0)
    def _():
      rs = tab[_T_RS * g + gi]
      off = tab[_T_OFF * g + gi]
      for hbm, buf, sem in pairs:
        _copy_rows(
            lambda o, m, hbm=hbm: hbm.at[pl.ds(_al(rs + o, a), m)],
            lambda o, m, buf=buf: buf.at[dst_slot, pl.ds(_al(off + o, a), m)],
            n,
            sem.at[dst_slot],
            a,
            dyn,
        )

    return carry

  lax.fori_loop(0, g, body, 0)


def _used_rows(tab, g):
  return tab[_T_OFF * g + g - 1] + tab[_T_NL * g + g - 1]


# Row-info filler for buffer rows that hold no row of this step: token 65535
# (never a valid token, T <= _MAX_TOKENS) and k = 255.
_INFO_FILL = 255.0
_MAX_TOKENS = 255 * 256


def _read_pipeline(tab_cur, tab_nxt, pairs, g, jrows, a, zero_x, dyn):
  """Double-buffered window reads of x and row info; returns this step's slot.

  pairs = ((x_hbm, buf, xsem), (info_hbm, ibuf, isem)).
  """
  b = pl.program_id(0)
  nb = pl.num_programs(0)
  slot = b % 2
  (_, buf, xsem), (_, ibuf, isem) = pairs

  @pl.when(b == 0)
  def _():
    if zero_x:
      # Rows past the used prefix are multiplied by zeros in the MXU: they must
      # be finite, so clear both slots once (later they only hold real rows).
      _fill_buf(buf, 2, jrows, 0)
    # Rows past the used prefix must not look like rows of any block.
    _fill_buf(ibuf, 2, jrows, _INFO_FILL)
    _issue_range_reads(tab_cur, pairs, 0, g, a, dyn)

  @pl.when(b + 1 < nb)
  def _():
    _issue_range_reads(tab_nxt, pairs, 1 - slot, g, a, dyn)

  used = _used_rows(tab_cur, g)
  _wait_rows(lambda n: buf.at[slot, pl.ds(0, n)], xsem.at[slot], used, jrows, a, dyn)
  _wait_rows(lambda n: ibuf.at[slot, pl.ds(0, n)], isem.at[slot], used, jrows, a, dyn)
  return slot


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
  tokrow = jnp.where((cc >= 0) & (cc < c) & live, cc, -1)
  krow = info[2:3, :].astype(jnp.int32)
  return tokrow, krow


def _owned_col(ibuf, slot, q0, jq, c, t0, used):
  """[jq, 1] bool: buffer row belongs to this block (column orientation)."""
  info = ibuf[slot, pl.ds(q0, jq), :].astype(jnp.float32)  # [jq, 128]
  tok = info[:, 0:1] + 256.0 * info[:, 1:2]
  t0f = t0.astype(jnp.float32)
  live = lax.broadcasted_iota(jnp.int32, (jq, 1), 0) < used - q0
  return (tok >= t0f) & (tok < t0f + c) & live


def _row_weights(w_hi, w_lo, krow, jq, dtype):
  """[C, jq] f32 weight of token c for top-k slot krow[j] (exact MXU selection)."""
  koh = (lax.broadcasted_iota(jnp.int32, (_LANES, jq), 0) == krow).astype(dtype)  # [128, jq]
  sel_hi = _mm(w_hi, koh)
  sel_lo = None if w_lo is None else _mm(w_lo, koh)
  return sel_hi, sel_lo


_MM_UPCAST = [False]  # legacy CPU interpreter: no bf16 x bf16 -> f32 DotThunk


def _mm(a, b, dims=None):
  if _MM_UPCAST[0]:
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)  # exact: bf16 products fit in f32
  if dims is None:
    return jnp.dot(a, b, preferred_element_type=jnp.float32)
  return lax.dot_general(a, b, (dims, ((), ())), preferred_element_type=jnp.float32)


# ---------------------------------------------------------------------------
# forward kernel
# ---------------------------------------------------------------------------


def _split_w(w_ref, split, dtype):
  w = w_ref[...]  # [C, 128] f32
  w_hi = w.astype(dtype)
  w_lo = (w - w_hi.astype(jnp.float32)).astype(dtype) if split else None
  return w_hi, w_lo


def _fwd_kernel(
    tab_cur, tab_nxt, w_ref, x_hbm, info_hbm, y_ref, buf, ibuf, s_hi_ref, s_lo_ref, xsem, isem, *, g, jrows, e, split, a, opts
):
  opts = dict(opts)
  _MM_UPCAST[0] = opts.get("upcast", False)
  mode, dyn = opts["mode"], opts["dyn"]
  c = y_ref.shape[0]
  pairs = ((x_hbm, buf, xsem), (info_hbm, ibuf, isem))
  if mode == "compute":
    slot = 0

    @pl.when(pl.program_id(0) == 0)
    def _():
      _fill_buf(buf, 2, jrows, 0)
      _fill_buf(ibuf, 2, jrows, _INFO_FILL)

  else:
    slot = _read_pipeline(tab_cur, tab_nxt, pairs, g, jrows, a, zero_x=True, dyn=dyn)
  if mode == "dma":
    y_ref[...] = jnp.zeros(y_ref.shape, y_ref.dtype)
    return

  jq = _row_chunk(jrows)
  en = _col_chunk(e, opts["en"])
  buckets = _buckets(jrows, opts["bucket"])
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
    sel_hi, sel_lo = _row_weights(w_hi, w_lo, krow, jq, s_hi_ref.dtype)
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
        acc = _mm(s_hi_ref[:, pl.ds(0, size)], xb)
        if split:
          acc = acc + _mm(s_lo_ref[:, pl.ds(0, size)], xb)
        y_ref[:, pl.ds(n0, en)] = acc.astype(y_ref.dtype)
        return carry

      lax.fori_loop(0, e // en, y_body, 0)


def _common_specs(nb, ntab, c):
  return [
      pl.BlockSpec((ntab,), lambda b: (b,), memory_space=pltpu.SMEM),
      pl.BlockSpec((ntab,), lambda b: (jnp.minimum(b + 1, nb - 1),), memory_space=pltpu.SMEM),
      pl.BlockSpec((c, _LANES), lambda b: (b, 0)),  # padded weights
  ]


def _fwd_call(x, w_pad, tab, info, *, block_tokens, num_groups, split, interpret, align=8, opts=()):
  r, e = x.shape
  t = w_pad.shape[0]
  c, g = block_tokens, num_groups
  nb = t // c
  k = r // t
  jrows = _buffer_rows(c, k, g, align)
  ntab = _tab_stride(g)
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
  kernel = functools.partial(_fwd_kernel, g=g, jrows=jrows, e=e, split=split, a=align, opts=opts)
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
      compiler_params=_compiler_params(min(vmem, 120 * 1024 * 1024)),
      interpret=interpret,
      name="moe_combine_fwd",
  )(tab, tab, w_pad, x, info)


# ---------------------------------------------------------------------------
# backward kernel
# ---------------------------------------------------------------------------


def _exact_bf16_parts(v, dtype):
  """v (f32) = p1 + p2 + p3 (+ < 2^-24 |v|) with bf16 parts."""
  p1 = v.astype(dtype)
  r1 = v - p1.astype(jnp.float32)
  p2 = r1.astype(dtype)
  p3 = (r1 - p2.astype(jnp.float32)).astype(dtype)
  return p1, p2, p3


def _bwd_row_chunk(jrows, pref):
  """Largest multiple of 256 that is <= min(pref, jrows)."""
  return max(256, min(pref, jrows) // 256 * 256)


def _bwd_compute_v5(w_ref, dy_ref, dw_ref, buf, ibuf, obuf, slot, used, t0, *, c, e, en, jrows, split, rq, do_mm):
  """Column-oriented backward compute over RQ-row chunks of the buffer.

  Per chunk (rows j in [q0, q0 + RQ), token-in-block tok_j, top-k slot k_j):
    GT  = x_rows . dy^T               [RQ, C]  (MXU; dy^T is the stationary operand)
    dw_j = GT[j, tok_j]               (masked lane reduction)
    dW[c, k] += sum_j [tok_j = c] [k_j = k] dw_j   (exact 0/1 MXU routing, 3 bf16 parts)
    S[j, c] = [tok_j = c] * W[c, k_j]  [RQ, C]  (no transposes: W^T is built once)
    obuf rows = S @ dy                 (MXU; dy column chunk is the stationary operand)
  Large RQ keeps the MXU streaming many rows per stationary tile (v4 streamed
  only 256 rows per tile and transposed [C, 256] f32 tiles for S).
  The last chunk is clamped into the buffer (q0 = min(q * RQ, J - RQ)); rows it
  shares with the previous chunk recompute identical obuf values and are
  excluded from dW.
  """
  dt = dy_ref.dtype
  dw = jnp.zeros((c, _LANES), jnp.float32)
  if do_mm:
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
      q0 = _al(jnp.minimum(qs, jrows - rq), 256)
      info = ibuf[slot, pl.ds(q0, rq), :].astype(jnp.float32)  # [RQ, 128]
      row = iota_r + q0
      tok = (info[:, 0:1] + 256.0 * info[:, 1:2]).astype(jnp.int32) - t0
      tokc = jnp.where((tok >= 0) & (tok < c) & (row < used), tok, -1)  # [RQ, 1]
      kc = info[:, 2:3].astype(jnp.int32)
      m = iota_c == tokc  # [RQ, C]
      # dW
      gt = _mm(buf[slot, pl.ds(q0, rq), :], dy_ref[...], ((1,), (1,)))  # [RQ, C]
      dwc = jnp.sum(jnp.where(m, gt, 0.0), axis=1, keepdims=True)  # [RQ, 1]
      a_jk = jnp.where((iota_l == kc) & (row >= qs), dwc, 0.0)  # [RQ, 128]
      tokrow, _ = _chunk_info(ibuf, slot, q0, rq, c, t0, used)  # [1, RQ]
      mt = (iota_cr == tokrow).astype(dt)  # [C, RQ]
      for part in _exact_bf16_parts(a_jk, dt):
        dw = dw + _mm(mt, part)  # [C, 128]
      # obuf rows
      koh = (iota_l == kc).astype(dt)  # [RQ, 128]
      s_hi = jnp.where(m, _mm(koh, wt_hi), 0.0).astype(dt)  # [RQ, C]
      s_lo = jnp.where(m, _mm(koh, wt_lo), 0.0).astype(dt) if split else None

      def n_body(n, carry):
        n0 = _al(n * en, en)
        dyn_ = dy_ref[:, pl.ds(n0, en)]  # [C, en]
        ob = _mm(s_hi, dyn_)
        if s_lo is not None:
          ob = ob + _mm(s_lo, dyn_)
        obuf[slot, pl.ds(q0, rq), pl.ds(n0, en)] = ob.astype(obuf.dtype)
        return carry

      lax.fori_loop(0, e // en, n_body, 0)
      return dw

    dw = lax.fori_loop(0, nq, q_body, dw)
  dw_ref[...] = dw


def _bwd_kernel(
    tab_cur,
    tab_nxt,
    w_ref,
    dy_ref,
    x_hbm,
    info_hbm,
    dwt_ref,
    dx_hbm,
    buf,
    ibuf,
    obuf,
    xsem,
    isem,
    wsem,
    lsem,
    wcount,
    *,
    g,
    jrows,
    e,
    split,
    a,
    opts,
):
  opts = dict(opts)
  _MM_UPCAST[0] = opts.get("upcast", False)
  mode, dyn = opts["mode"], opts["dyn"]
  do_dma = mode != "compute"
  do_mm = mode != "dma"
  do_stage = mode == "full"
  do_merge = mode in ("full", "compute")
  b = pl.program_id(0)
  nb = pl.num_programs(0)
  pairs = ((x_hbm, buf, xsem), (info_hbm, ibuf, isem))
  if do_dma:
    slot = _read_pipeline(tab_cur, tab_nxt, pairs, g, jrows, a, zero_x=False, dyn=dyn)
  else:
    slot = 0

    @pl.when(b == 0)
    def _():
      _fill_buf(buf, 2, jrows, 0)
      _fill_buf(obuf, 2, jrows, 0)
      _fill_buf(ibuf, 2, jrows, _INFO_FILL)

  jq = _row_chunk(jrows)
  en = _col_chunk(e, opts["en"])
  used = _used_rows(tab_cur, g)
  nq = (used + (jq - 1)) // jq
  c = dy_ref.shape[0]
  dt = dy_ref.dtype
  kp = dwt_ref.shape[0]
  merge_f32 = opts["merge"] == "f32"
  t0 = b * c

  if opts["bwd"] == "v5":
    _bwd_compute_v5(w_ref, dy_ref, dwt_ref, buf, ibuf, obuf, slot, used, t0, c=c, e=e, en=en, jrows=jrows, split=split,
                    rq=_bwd_row_chunk(jrows, opts["rq"]), do_mm=do_mm)
  else:
    # ---- compute: dW^T [kp, C], the weighted, permuted dy rows (obuf) ----
    dwt = jnp.zeros((kp, c), jnp.float32)
    if do_mm:
      w_hi, w_lo = _split_w(w_ref, split, dt)
      iota_c = lax.broadcasted_iota(jnp.int32, (c, jq), 0)
      iota_k = lax.broadcasted_iota(jnp.int32, (kp, jq), 0)

      def q_body(q, dwt):
        q0 = _al(q * jq, jq)
        xq = buf[slot, pl.ds(q0, jq), :]  # [jq, E]
        gt = _mm(dy_ref[...], xq, ((1,), (1,)))  # [C, jq] = <dy[c], x row j>
        tokrow, krow = _chunk_info(ibuf, slot, q0, jq, c, t0, used)
        m = iota_c == tokrow
        # dW of row j: its column of gt (at most one owning token); then route it
        # to (k(j), c(j)) with exact 0/1 matmuls (3 bf16 parts of the f32 value).
        dwrow = jnp.sum(jnp.where(m, gt, 0.0), axis=0, keepdims=True)  # [1, jq]
        a_kj = jnp.where(iota_k == krow, dwrow, 0.0)  # [kp, jq]
        m01 = m.astype(dt)
        for part in _exact_bf16_parts(a_kj, dt):
          dwt = dwt + _mm(part, m01, ((1,), (1,)))  # [kp, C]
        sel_hi, sel_lo = _row_weights(w_hi, w_lo, krow, jq, dt)
        st_hi = jnp.where(m, sel_hi, 0.0).T.astype(dt)  # [jq, C]
        st_lo = None
        if split:
          st_lo = jnp.where(m, sel_lo, 0.0).T.astype(dt)

        def n_body(n, carry2):
          n0 = _al(n * en, en)
          dyn_ = dy_ref[:, pl.ds(n0, en)]  # [C, en]
          ob = _mm(st_hi, dyn_)
          if st_lo is not None:
            ob = ob + _mm(st_lo, dyn_)
          obuf[slot, pl.ds(q0, jq), pl.ds(n0, en)] = ob.astype(obuf.dtype)
          return carry2

        lax.fori_loop(0, e // en, n_body, 0)
        return dwt

      dwt = lax.fori_loop(0, nq, q_body, dwt)
    dwt_ref[...] = dwt

  # ---- previous step's writes must land before the partial granules are read ----
  if do_dma:

    @pl.when(b > 0)
    def _():
      _wait_rows(lambda n: obuf.at[1 - slot, pl.ds(0, n)], wsem.at[1 - slot], wcount[1 - slot], jrows, a, dyn)

  # ---- stage partial granules (current dx in HBM) at their buffer rows in buf[slot] ----
  # (x rows are no longer needed. A granule that is the tail of one range and
  # the head of the next is staged twice with identical bytes.)
  if do_stage:

    def stage_body(gi, nrows):
      delta = tab_cur[_T_DELTA * g + gi]
      for row in (_T_HS, _T_TS):
        s = tab_cur[row * g + gi]

        @pl.when(s >= 0)
        def _(s=s):
          pltpu.make_async_copy(
              dx_hbm.at[pl.ds(_al(s, a), a)],
              buf.at[slot, pl.ds(_al(s + delta, a), a)],
              lsem.at[0],
          ).start()

        nrows = nrows + a * (s >= 0).astype(jnp.int32)
      return nrows

    nstage = lax.fori_loop(0, g, stage_body, jnp.int32(0))
    _wait_rows(lambda n: buf.at[slot, pl.ds(0, n)], lsem.at[0], nstage, jrows, a, dyn)

  # ---- merge: rows not owned by this block take the staged (current HBM) value ----
  if do_merge:

    def merge_body(q, carry):
      q0 = _al(q * jq, jq)
      own = _owned_col(ibuf, slot, q0, jq, c, t0, used)  # [jq, 1]
      for n in range(e // en):
        o = obuf[slot, pl.ds(q0, jq), pl.ds(n * en, en)]
        h = buf[slot, pl.ds(q0, jq), pl.ds(n * en, en)]
        if merge_f32:
          o, h = o.astype(jnp.float32), h.astype(jnp.float32)
        obuf[slot, pl.ds(q0, jq), pl.ds(n * en, en)] = jnp.where(own, o, h).astype(obuf.dtype)
      return carry

    lax.fori_loop(0, nq, merge_body, 0)

  # ---- write back every window part ----
  if do_dma:

    def write_body(gi, nrows):
      n = tab_cur[_T_NL * g + gi]
      rs = tab_cur[_T_RS * g + gi]
      off = tab_cur[_T_OFF * g + gi]

      @pl.when(n > 0)
      def _():
        _copy_rows(
            lambda o, m: obuf.at[slot, pl.ds(_al(off + o, a), m)],
            lambda o, m: dx_hbm.at[pl.ds(_al(rs + o, a), m)],
            n,
            wsem.at[slot],
            a,
            dyn,
        )

      return nrows + n

    nw = lax.fori_loop(0, g, write_body, jnp.int32(0))
    wcount[slot] = nw

    @pl.when(b == nb - 1)
    def _():
      _wait_rows(lambda n: obuf.at[slot, pl.ds(0, n)], wsem.at[slot], nw, jrows, a, dyn)


def _bwd_call(x, w_pad, tab, info, dy, *, block_tokens, num_groups, split, interpret, align=8, opts=()):
  r, e = x.shape
  t = w_pad.shape[0]
  c, g = block_tokens, num_groups
  nb = t // c
  k = r // t
  kp = _round_up(k, 16)
  jrows = _buffer_rows(c, k, g, align)
  ntab = _tab_stride(g)
  in_specs = _common_specs(nb, ntab, c) + [
      pl.BlockSpec((c, e), lambda b: (b, 0)),
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  v5 = dict(opts).get("bwd", "v4") == "v5"
  if v5:  # dW [T, 128] (weights in lanes [0, K))
    dw_spec, dw_shape = pl.BlockSpec((c, _LANES), lambda b: (b, 0)), (t, _LANES)
  else:  # dW^T [kp, T]
    dw_spec, dw_shape = pl.BlockSpec((kp, c), lambda b: (0, b)), (kp, t)
  out_specs = [dw_spec, pl.BlockSpec(memory_space=pl.ANY)]
  itemsize = jnp.dtype(x.dtype).itemsize
  vmem = (
      4 * jrows * e * itemsize  # buf + obuf
      + 2 * jrows * _LANES * itemsize  # row info
      + 2 * c * e * itemsize  # dy blocks
      + 2 * c * _LANES * 4 + 2 * kp * c * 4  # weight / dW blocks
      + (24 if v5 else 12) * 1024 * 1024  # chunk temporaries, compiler scratch
  )
  kernel = functools.partial(_bwd_kernel, g=g, jrows=jrows, e=e, split=split, a=align, opts=opts)
  (tab, w_pad, dy, x, info), mat = _match_vma(tab, w_pad, dy, x, info)
  return pl.pallas_call(
      kernel,
      out_shape=(_out_struct(dw_shape, jnp.float32, mat), _out_struct((r, e), x.dtype, mat)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(nb,),
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=[
              pltpu.VMEM((2, jrows, e), x.dtype),  # buf (+ partial-granule staging)
              pltpu.VMEM((2, jrows, _LANES), info.dtype),  # row info
              pltpu.VMEM((2, jrows, e), x.dtype),  # obuf
              pltpu.SemaphoreType.DMA((2,)),  # x reads
              pltpu.SemaphoreType.DMA((2,)),  # row-info reads
              pltpu.SemaphoreType.DMA((2,)),  # writes
              pltpu.SemaphoreType.DMA((1,)),  # partial-granule staging
              pltpu.SMEM((2,), jnp.int32),  # rows written per slot
          ],
      ),
      compiler_params=_compiler_params(min(vmem, 120 * 1024 * 1024)),
      interpret=interpret,
      name="moe_combine_bwd",
  )(tab, tab, w_pad, dy, x, info)


# ---------------------------------------------------------------------------
# custom VJP wrapper
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
  w_pad = _padded_weights(w)
  info = _row_info(sort_idx, k)
  kcfg = dict(cfg)
  dw_out, dx = _bwd_call(x, w_pad, tab, info, dy.astype(x.dtype), **kcfg)
  if dict(kcfg["opts"]).get("bwd", "v4") == "v5":
    dw = dw_out[:, :k]
  else:
    dw = dw_out[:k].T
  return dx, dw.astype(w.dtype), None, None


_combine_vjp.defvjp(_combine_vjp_fwd, _combine_vjp_bwd)


def combine(
    x,
    sort_idx,
    weights,
    group_sizes,
    *,
    block_tokens=None,
    interpret=None,
    align=None,
):
  """y[t] = sum_k weights[t, k] * x[argsort(sort_idx)[t*K + k]] (f32 accumulate, x.dtype out).

  Args:
    x: [T*K, E] expert outputs in expert-sorted order (bf16 for the kernel path).
    sort_idx: [T*K] int, stable argsort of the flat (token-major) expert ids.
    weights: [T, K] routing weights.
    group_sizes: [G] int, rows per expert group of `x` (must sum to T*K).
    block_tokens: tokens per grid step (default from MAXTEXT_G4_COMBINE_BLOCK).
    interpret: Pallas interpret mode (default: True off-TPU).
    align: DMA / VMEM row alignment (default from MAXTEXT_G4_COMBINE_ALIGN, 8).

  Returns:
    [T, E] array in x.dtype. Differentiable w.r.t. x and weights.
  """
  t, k = weights.shape
  if block_tokens is None:
    block_tokens = block_tokens_from_env(t)
  if interpret is None:
    interpret = _default_interpret()
  if align is None:
    align = align_from_env()
  if group_sizes is None or not kernel_supported(x.shape, x.dtype, t, k, block_tokens, align):
    return combine_reference(x, sort_idx, weights)
  tab = _metadata(sort_idx, group_sizes, t, k, block_tokens, align)
  split = jnp.dtype(weights.dtype) != jnp.bfloat16 and _env_int("MAXTEXT_G4_COMBINE_WSPLIT", 1) != 0
  cfg = (
      ("block_tokens", block_tokens),
      ("num_groups", int(group_sizes.shape[0])),
      ("split", bool(split)),
      ("interpret", interpret),
      ("align", int(align)),
      ("opts", _kernel_opts() + (("upcast", interpret is True),)),
  )
  return _combine_vjp(x, weights, sort_idx, tab, cfg)
