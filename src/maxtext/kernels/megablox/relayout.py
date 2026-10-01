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

"""In-VMEM relayout between [tokens, D0, 128] and [tokens, D0 * 128].

Ported from Lineage's gmmv2/relayout.py (bf16 only) and
extended to 8-bit dtypes (fp8 / int8).

gmm_v2 feeds the MXU tiles whose sublanes index tokens and whose lanes index
the contracting dim. Tokens stored as ``[tokens, D0, 128]`` (one contiguous
slab per token) have the sublane axis of each VMEM tile indexing ``D0``
instead. The helpers below convert between the layouts inside VMEM/vregs using
32-bit sublane-strided loads/stores (the only strided access Mosaic supports)
plus a few VALU bit ops to (un)pack the sub-32-bit values.

For a VMEM tile ``[tm, d0, 128]`` of a dtype with ``p`` values per 32-bit word
(``p = 2`` for bf16, ``p = 4`` for 8-bit):

* Viewed as uint32 it is ``[tm, d0 // p, 128]``: rows ``p*i .. p*i + p - 1`` of
  a token share one word (row ``p*i + j`` in bits ``[j*b, (j+1)*b)``).
* Viewed flat it is ``[tm * d0 // p, 128]`` and token ``t`` owns rows
  ``[t * d0 // p, (t + 1) * d0 // p)``.
* A sublane-strided load with stride ``p * d0 // p`` starting at row
  ``j * d0 // p + w`` returns word ``w`` of tokens ``j, j + p, j + 2p, ...``.
  ``p`` such loads followed by a ``p x p`` sub-word transpose give ``p``
  ``[tm, 128]`` columns (``d0`` rows ``p*w .. p*w + p - 1``) with tokens in
  sublanes, i.e. in the native packed layout of a ``[tm, 128]`` value.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

LANES = 128


def packing(dtype) -> int:
  bits = jax.dtypes.itemsize_bits(jnp.dtype(dtype))
  if bits not in (8, 16):
    raise ValueError(f"3D relayout supports 8/16-bit dtypes, got {dtype}.")
  return 32 // bits


def check_tile_d0(tile_d0: int, full_d0: int | None = None, dtype=jnp.bfloat16) -> None:
  """Raises if ``tile_d0`` cannot use the strided-load relayout.

  Args:
    tile_d0: second-minor block dim (``tile_k // 128`` or ``tile_n // 128``).
    full_d0: the full second-minor array dim, if known. A block that covers the
      whole dim is always accepted (Pallas' "block == full dim" rule).
    dtype: element dtype of the 3D operand.
  """
  p = packing(dtype)
  if tile_d0 % p != 0:
    raise ValueError(f"3D relayout packs {p} rows per word; tile_d0 must be a multiple of {p}, got {tile_d0}.")
  if tile_d0 % 8 != 0 and tile_d0 != full_d0:
    raise ValueError(
        "3D relayout requires tile_d0 % 8 == 0 (tile_k / tile_n a multiple"
        f" of {8 * LANES}) or tile_d0 == full D0 ({full_d0}); got"
        f" tile_d0={tile_d0}."
    )


def needs_window(tile_d0: int, full_d0: int) -> bool:
  """Whether a ``tile_d0`` block of a ``full_d0`` 3D operand must use a full-D0 window block.

  Pallas TPU blocks need a second-minor dim that is a multiple of 8 or the full
  dim, and sub-tile DMA windows are not supported. Such tiles (e.g. r19's
  3584 / 1792 of 7168: tile_d0 = 28 / 14) instead DMA the whole ``[.., full_d0,
  128]`` slab and relayout the ``[d0_start, d0_start + tile_d0)`` window from VMEM.
  """
  return tile_d0 != full_d0 and tile_d0 % 8 != 0


def check_window(tile_d0: int, full_d0: int, dtype, *, store: bool) -> None:
  """Raises if a full-D0 window block cannot serve ``tile_d0`` tiles."""
  p = packing(dtype)
  if full_d0 % tile_d0 != 0:
    raise ValueError(f"3D window tiles need tile_d0 ({tile_d0}) to divide D0 ({full_d0}).")
  if full_d0 % p != 0:
    raise ValueError(f"3D D0 ({full_d0}) must be a multiple of the packing {p}.")
  # Loads handle windows that start / end inside a packed word; stores do not.
  if (store or p == 2) and tile_d0 % p != 0:
    raise ValueError(f"3D window {'stores' if store else 'loads'} need tile_d0 % {p} == 0, got {tile_d0}.")


def _flat_u32_view(ref):
  """Returns ``ref`` (``[..., d0, 128]``) as uint32 ``[-1, 128]`` and the u32 rows per token."""
  full_d0 = ref.shape[-2]
  p = packing(ref.dtype)
  assert full_d0 % p == 0, (full_d0, p)
  rows_per_tok = full_d0 // p
  ref_u32 = ref.bitcast(jnp.uint32)  # [..., d0 // p, 128]
  return ref_u32.reshape(-1, LANES), rows_per_tok


def _u32(v):
  return jnp.uint32(v)


def _transpose4x4_bytes(a0, a1, a2, a3):
  """4x4 byte transpose: out_b byte j = a_j byte b."""
  lo16, hi16 = _u32(0x0000FFFF), _u32(0xFFFF0000)
  # Stage 1 (16-bit halves): pair token phases (0, 2) and (1, 3).
  b0 = (a0 & lo16) | (a2 << 16)  # [a0.b0, a0.b1, a2.b0, a2.b1]
  b2 = (a0 >> 16) | (a2 & hi16)  # [a0.b2, a0.b3, a2.b2, a2.b3]
  b1 = (a1 & lo16) | (a3 << 16)  # [a1.b0, a1.b1, a3.b0, a3.b1]
  b3 = (a1 >> 16) | (a3 & hi16)  # [a1.b2, a1.b3, a3.b2, a3.b3]
  # Stage 2 (bytes).
  ev, od = _u32(0x00FF00FF), _u32(0xFF00FF00)
  c0 = (b0 & ev) | ((b1 & ev) << 8)  # [a0.b0, a1.b0, a2.b0, a3.b0]
  c1 = ((b0 >> 8) & ev) | (b1 & od)  # [a0.b1, a1.b1, a2.b1, a3.b1]
  c2 = (b2 & ev) | ((b3 & ev) << 8)
  c3 = ((b2 >> 8) & ev) | (b3 & od)
  return c0, c1, c2, c3


def load_3d_as_2d(ref, tm: int, tile_d0: int, d0_start: int = 0) -> jax.Array:
  """Loads rows ``[d0_start, d0_start + tile_d0)`` of VMEM ref ``[..., D0, 128]`` as ``[tm, tile_d0 * 128]``.

  Args:
    ref: VMEM ref whose trailing two dims are ``(D0, 128)`` (bf16 or 8-bit) and
      whose leading dims multiply to at least ``tm``. ``D0 == tile_d0`` for a
      plain block; a full-D0 window block (``needs_window``) has ``D0 > tile_d0``.
    tm: number of leading rows (tokens) to load. Must be a multiple of the
      packing (2 for bf16, 4 for 8-bit).
    tile_d0: number of D0 rows to load.
    d0_start: static first D0 row. For bf16 it must be even; for 8-bit it may
      start / end inside a packed word.

  Returns:
    Array ``[tm, tile_d0 * 128]`` of ``ref.dtype`` with tokens in rows,
    identical to ``ref[:tm, d0_start:d0_start + tile_d0].reshape(tm, -1)``.
  """
  dtype = ref.dtype
  p = packing(dtype)
  assert tm % p == 0, (tm, p)
  assert 0 <= d0_start and d0_start + tile_d0 <= ref.shape[-2], (d0_start, tile_d0, ref.shape)
  x2d, rows_per_tok = _flat_u32_view(ref)
  stride = p * rows_per_tok  # p tokens, in uint32 rows
  cols = []
  if p == 2:
    assert d0_start % 2 == 0 and tile_d0 % 2 == 0, (d0_start, tile_d0)
    lo_mask, hi_mask = _u32(0xFFFF), _u32(0xFFFF0000)
    for c in range(d0_start, d0_start + tile_d0, 2):
      even = x2d[pl.ds(c // 2, tm // 2, stride=stride), :]
      odd = x2d[pl.ds(rows_per_tok + c // 2, tm // 2, stride=stride), :]
      lo = (even & lo_mask) | (odd << 16)  # d0 == c
      hi = (even >> 16) | (odd & hi_mask)  # d0 == c + 1
      cols.append(pltpu.bitcast(lo, dtype))  # [tm, 128]
      cols.append(pltpu.bitcast(hi, dtype))
  else:
    d0_end = d0_start + tile_d0
    for w in range(d0_start // 4, pl.cdiv(d0_end, 4)):
      a = [x2d[pl.ds(j * rows_per_tok + w, tm // 4, stride=stride), :] for j in range(4)]
      for b, c in enumerate(_transpose4x4_bytes(*a)):
        if d0_start <= 4 * w + b < d0_end:
          cols.append(pltpu.bitcast(c, dtype))  # [tm, 128], d0 == 4 * w + b
  return jnp.concatenate(cols, axis=-1)


def store_2d_as_3d(ref, x2d: jax.Array, tm: int, tile_d0: int, d0_start: int = 0) -> None:
  """Stores ``x2d`` ``[tm, tile_d0 * 128]`` into rows ``[d0_start, d0_start + tile_d0)`` of ``[..., D0, 128]``.

  Inverse of :func:`load_3d_as_2d`; writes the first ``tm`` tokens of ``ref``
  and leaves its other D0 rows untouched. bf16 and 8-bit dtypes are supported
  (``x2d.dtype == ref.dtype``); ``d0_start`` and ``tile_d0`` must be multiples
  of the packing.
  """
  dtype = ref.dtype
  assert x2d.dtype == dtype, (x2d.dtype, dtype)
  assert x2d.shape == (tm, tile_d0 * LANES), (x2d.shape, tm, tile_d0)
  p = packing(dtype)
  assert tm % p == 0, (tm, p)
  assert d0_start % p == 0 and tile_d0 % p == 0, (d0_start, tile_d0, p)
  assert d0_start + tile_d0 <= ref.shape[-2], (d0_start, tile_d0, ref.shape)
  o2d, rows_per_tok = _flat_u32_view(ref)
  stride = p * rows_per_tok
  if p == 2:
    lo_mask, hi_mask = _u32(0xFFFF), _u32(0xFFFF0000)
    for c in range(0, tile_d0, 2):
      lo = pltpu.bitcast(x2d[:, c * LANES : (c + 1) * LANES], jnp.uint32)
      hi = pltpu.bitcast(x2d[:, (c + 1) * LANES : (c + 2) * LANES], jnp.uint32)
      even = (lo & lo_mask) | (hi << 16)  # even tokens: (c, c+1)
      odd = (lo >> 16) | (hi & hi_mask)  # odd tokens
      r = (d0_start + c) // 2
      o2d[pl.ds(r, tm // 2, stride=stride), :] = even
      o2d[pl.ds(rows_per_tok + r, tm // 2, stride=stride), :] = odd
  else:
    for w in range(tile_d0 // 4):
      c = [pltpu.bitcast(x2d[:, (4 * w + b) * LANES : (4 * w + b + 1) * LANES], jnp.uint32) for b in range(4)]
      # The 4x4 byte transpose is an involution: word j byte b <- column b byte j.
      a = _transpose4x4_bytes(*c)
      r = d0_start // 4 + w
      for j in range(4):
        o2d[pl.ds(j * rows_per_tok + r, tm // 4, stride=stride), :] = a[j]


def load_3d_window_as_2d(ref, tm: int, tile_d0: int, window_id) -> jax.Array:
  """``load_3d_as_2d`` of window ``window_id`` (traced) of a full-D0 block: a switch over static starts."""
  num = ref.shape[-2] // tile_d0
  if num == 1:
    return load_3d_as_2d(ref, tm, tile_d0)
  branches = [lambda i=i: load_3d_as_2d(ref, tm, tile_d0, i * tile_d0) for i in range(num)]
  return jax.lax.switch(window_id, branches)


def store_2d_as_3d_window(ref, x2d: jax.Array, tm: int, tile_d0: int, window_id) -> None:
  """``store_2d_as_3d`` into window ``window_id`` (traced) of a full-D0 block."""
  num = ref.shape[-2] // tile_d0
  if num == 1:
    store_2d_as_3d(ref, x2d, tm, tile_d0)
    return
  branches = [lambda i=i: store_2d_as_3d(ref, x2d, tm, tile_d0, i * tile_d0) for i in range(num)]
  jax.lax.switch(window_id, branches)
