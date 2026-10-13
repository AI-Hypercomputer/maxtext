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

gmm_v2 feeds the MXU tiles whose sublanes index tokens and whose lanes index
the contracting dim. Lineage's DSv3 activations are stored as
``[tokens, D0, 128]`` (one contiguous slab per token), where the sublane axis
of each VMEM tile indexes ``D0`` instead of tokens. The two helpers below
convert between the layouts inside VMEM/vregs using 32-bit sublane-strided
loads/stores (the only strided access Mosaic supports) plus a few VALU bit
ops to (un)pack the bf16 pairs (or fp8 quadruples).

For a bf16 VMEM tile ``[tm, d0, 128]``:

* Viewed as uint32 it is ``[tm, d0 // 2, 128]``: rows ``(2i, 2i+1)`` of a
  token share one 32-bit word (low half = row ``2i``, high half = ``2i+1``).
* Viewed flat it is ``[tm * d0 // 2, 128]`` and token ``t`` owns rows
  ``[t * d0 // 2, (t + 1) * d0 // 2)``. Verified on TPU v7x to lower for
  ``d0 % 8 == 0`` (``d0`` in {8, 16, 32}); other values are rejected.
* A sublane-strided load with stride ``d0`` (in uint32 rows) starting at row
  ``c // 2`` therefore returns word ``c // 2`` of every other token in one
  instruction; two such loads (even/odd tokens) plus 3 bit ops give the
  ``[tm, 128]`` bf16 columns ``c`` and ``c + 1`` with tokens in sublanes.

8-bit dtypes pack 4 rows per word instead of 2: four strided loads (tokens
``4j + r``) give a 4x4 block of bytes per lane, which a two-stage transpose
network (16-bit halves, then bytes) turns into columns ``c .. c + 3``. The
network is its own inverse, so stores reuse it.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

LANES = 128
_LO_MASK = 0xFFFF
_HI_MASK = 0xFFFF0000
_EVEN_BYTES = 0x00FF00FF
_ODD_BYTES = 0xFF00FF00


def packing(dtype: jax.typing.DTypeLike) -> int:
  """Rows per 32-bit word for ``dtype`` (2 for bf16, 4 for 8-bit)."""
  bits = jax.dtypes.itemsize_bits(jnp.dtype(dtype))
  if bits not in (8, 16):
    raise ValueError(f"3D relayout supports 16-bit and 8-bit dtypes; got {jnp.dtype(dtype)}.")
  return 32 // bits


def check_tile_d0(tile_d0: int, full_d0: int | None = None, pack: int = 2) -> None:
  """Raises if ``tile_d0`` cannot use the strided-load relayout.

  Args:
    tile_d0: second-minor block dim (``tile_k // 128`` or ``tile_n // 128``).
    full_d0: the full second-minor array dim, if known. A block that covers the
      whole dim is always accepted (Pallas' "block == full dim" rule), which is
      what the small test configs (``D = 512 -> D0 = 4``) rely on.
    pack: rows per 32-bit word (see :func:`packing`).
  """
  # Pallas requires the second-minor block dim to be a multiple of 8 (or the
  # full array dim); tile_d0 = 12 / 28 were probed and rejected by the
  # BlockSpec check, so e.g. tile_n = 3584 is not possible on [M, 56, 128].
  if tile_d0 % pack != 0:
    raise ValueError(
        f"3D relayout packs {pack} rows per 32-bit word; tile_d0 must be a" f" multiple of {pack}, got {tile_d0}."
    )
  if tile_d0 % 8 != 0 and tile_d0 != full_d0:
    raise ValueError(
        "3D relayout requires tile_d0 % 8 == 0 (tile_k / tile_n a multiple"
        f" of {8 * LANES}) or tile_d0 == full D0 ({full_d0}); got"
        f" tile_d0={tile_d0}."
    )


def _flat_u32_view(ref, tile_d0: int):
  """Returns ``ref`` (``[..., tile_d0, 128]``) as uint32 ``[-1, 128]``."""
  # The block's own dim is always tile_d0; the %8-or-full-dim rule is checked
  # by the kernel config (check_tile_d0 with full_d0), only packing here.
  assert ref.shape[-2] == tile_d0, (ref.shape, tile_d0)
  pack = packing(ref.dtype)
  check_tile_d0(tile_d0, full_d0=tile_d0, pack=pack)
  rows_per_tok = tile_d0 // pack
  ref_u32 = ref.bitcast(jnp.uint32)  # [..., tile_d0 // pack, 128]
  return ref_u32.reshape(-1, LANES), rows_per_tok, pack


def _transpose_halves(w0, w1):
  """2x2 transpose of 16-bit halves: returns (low halves, high halves)."""
  lo = (w0 & jnp.uint32(_LO_MASK)) | (w1 << 16)
  hi = (w0 >> 16) | (w1 & jnp.uint32(_HI_MASK))
  return lo, hi


def _transpose_bytes(w0, w1):
  """2x2 transpose of bytes within each 16-bit half: (even, odd) bytes."""
  even = (w0 & jnp.uint32(_EVEN_BYTES)) | ((w1 & jnp.uint32(_EVEN_BYTES)) << 8)
  odd = ((w0 >> 8) & jnp.uint32(_EVEN_BYTES)) | (w1 & jnp.uint32(_ODD_BYTES))
  return even, odd


def _transpose_words(words):
  """Transposes a ``pack x pack`` block of sub-words held in ``pack`` words.

  Output word ``q`` holds sub-word ``q`` of every input word, input ``r`` in
  sub-word position ``r`` (lowest bits first). The network is an involution.

  Args:
    words: ``pack`` uint32 arrays of the same shape.

  Returns:
    The ``pack`` transposed uint32 arrays.
  """
  if len(words) == 2:
    return list(_transpose_halves(*words))
  assert len(words) == 4, len(words)
  w0, w1, w2, w3 = words
  a0, a2 = _transpose_halves(w0, w2)
  a1, a3 = _transpose_halves(w1, w3)
  o0, o1 = _transpose_bytes(a0, a1)
  o2, o3 = _transpose_bytes(a2, a3)
  return [o0, o1, o2, o3]


def load_3d_as_2d(ref, tm: int, tile_d0: int) -> jax.Array:
  """Loads VMEM ref ``[..., tile_d0, 128]`` as ``[tm, tile_d0 * 128]``.

  Args:
    ref: VMEM ref of a 16-bit (bf16) or 8-bit (fp8) dtype whose trailing two
      dims are ``(tile_d0, 128)`` and whose leading dims multiply to at least
      ``tm`` (e.g. ``[tile_m // sublane, sublane, tile_d0, 128]``).
    tm: number of leading rows (tokens) to load. Must be a multiple of the
      packing (2 for bf16, 4 for fp8).
    tile_d0: size of the second-minor dim. Must be a multiple of 8.

  Returns:
    Array ``[tm, tile_d0 * 128]`` of ``ref.dtype`` with tokens in rows,
    identical to ``ref[:tm].reshape(tm, tile_d0 * 128)``.
  """
  x2d, rows_per_tok, pack = _flat_u32_view(ref, tile_d0)
  assert tm % pack == 0, (tm, pack)
  stride = pack * rows_per_tok  # `pack` tokens, in uint32 rows
  cols = []
  for c in range(0, tile_d0, pack):
    # Word c // pack of tokens (pack * j + r): token r's rows c .. c + pack - 1.
    words = [x2d[pl.ds(r * rows_per_tok + c // pack, tm // pack, stride=stride), :] for r in range(pack)]
    # Output word q holds row c + q of tokens (pack * j + r) in sub-word r.
    for out in _transpose_words(words):
      cols.append(pltpu.bitcast(out, ref.dtype))  # [tm, 128]
  return jnp.concatenate(cols, axis=-1)


def store_2d_as_3d(ref, x2d: jax.Array, tm: int, tile_d0: int) -> None:
  """Stores ``x2d`` ``[tm, tile_d0 * 128]`` into VMEM ref ``[..., tile_d0, 128]``.

  Inverse of :func:`load_3d_as_2d`; writes the first ``tm`` tokens of ``ref``.

  Args:
    ref: Destination VMEM ref with trailing dimensions ``(tile_d0, 128)``.
    x2d: 2D tile ``[tm, tile_d0 * 128]`` of ``ref.dtype`` to store.
    tm: Number of tokens in ``x2d`` (a multiple of the packing).
    tile_d0: Second-minor block size in ``ref``.
  """
  assert x2d.dtype == ref.dtype, (x2d.dtype, ref.dtype)
  assert x2d.shape == (tm, tile_d0 * LANES), (x2d.shape, tm, tile_d0)
  o2d, rows_per_tok, pack = _flat_u32_view(ref, tile_d0)
  assert tm % pack == 0, (tm, pack)
  stride = pack * rows_per_tok
  for c in range(0, tile_d0, pack):
    words = [pltpu.bitcast(x2d[:, (c + q) * LANES : (c + q + 1) * LANES], jnp.uint32) for q in range(pack)]
    for r, out in enumerate(_transpose_words(words)):
      o2d[pl.ds(r * rows_per_tok + c // pack, tm // pack, stride=stride), :] = out
