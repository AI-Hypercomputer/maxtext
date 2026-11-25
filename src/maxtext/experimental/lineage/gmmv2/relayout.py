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

import math
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


def store_2d_into_3d(ref, x2d: jax.Array, tm: int, row_start: int) -> None:
  """Stores ``x2d`` ``[tm, n * 128]`` into rows ``[row_start, row_start + n)``.

  Like :func:`store_2d_as_3d`, but writes only ``n`` of the ``d0`` rows of
  each token of ``ref`` (``[..., d0, 128]``), so a kernel can relayout its
  output one column chunk at a time.

  Args:
    ref: Destination VMEM ref with trailing dimensions ``(d0, 128)``.
    x2d: 2D tile ``[tm, n * 128]`` of ``ref.dtype`` to store.
    tm: Number of tokens in ``x2d`` (a multiple of the packing).
    row_start: First of the ``n`` rows written, a multiple of the packing.
  """
  assert x2d.dtype == ref.dtype, (x2d.dtype, ref.dtype)
  assert x2d.shape[0] == tm and x2d.shape[1] % LANES == 0, (x2d.shape, tm)
  o2d, rows_per_tok, pack = _flat_u32_view(ref, ref.shape[-2])
  assert tm % pack == 0, (tm, pack)
  num_rows = x2d.shape[1] // LANES
  assert num_rows % pack == 0, (num_rows, pack)
  stride = pack * rows_per_tok
  word_start = row_start // pack
  for c in range(0, num_rows, pack):
    words = [pltpu.bitcast(x2d[:, (c + q) * LANES : (c + q + 1) * LANES], jnp.uint32) for q in range(pack)]
    for r, out in enumerate(_transpose_words(words)):
      o2d[
          pl.ds(
              r * rows_per_tok + word_start + c // pack,
              tm // pack,
              stride=stride,
          ),
          :,
      ] = out


def _is_tpu_backend() -> bool:
  if jax.default_backend() == "tpu":
    return True
  try:
    pltpu.get_tpu_info()
    return True
  except (ValueError, RuntimeError):
    return False


def relayout_2d_to_3d_pallas(
    x2d: jax.Array,
    *,
    block_tokens: int = 512,
    interpret: bool = False,
) -> jax.Array:
  """Streams ``x2d`` ``[N, D]`` to ``[N, D // 128, 128]`` in VMEM without register spills.

  Replaces XLA's slow cross-sublane HBM reshape before ``gather_gate_gmm`` for
  8-bit and 16-bit activations while remaining bit-exact with ``x2d.reshape(N, D // 128, 128)``.
  """
  if x2d.ndim != 2 or x2d.shape[1] % LANES != 0:
    raise ValueError(f"relayout_2d_to_3d_pallas expects 2D [N, D] with D % {LANES} == 0; got {x2d.shape}.")
  num_tokens, size_d = x2d.shape
  d0 = size_d // LANES
  bits = jax.dtypes.itemsize_bits(x2d.dtype)
  if bits not in (8, 16):
    return x2d.reshape(num_tokens, d0, LANES)
  pack = 32 // bits
  sublane_align = 8 * pack
  tm = math.gcd(block_tokens, num_tokens) if block_tokens > 0 and num_tokens > 0 else 0
  on_tpu = interpret or _is_tpu_backend()
  if not on_tpu or d0 < 8 or d0 % 8 != 0 or d0 % pack != 0 or tm < sublane_align or tm % sublane_align != 0:
    return x2d.reshape(num_tokens, d0, LANES)

  m_sub = 128 if tm >= 128 and tm % 128 == 0 else tm

  def _kernel(in_ref, out_ref):
    o2d, rows_per_tok, p = _flat_u32_view(out_ref, d0)
    stride = p * rows_per_tok
    for m0 in range(0, tm, m_sub):
      for c in range(0, d0, p):
        words = [
            pltpu.bitcast(in_ref[m0 : m0 + m_sub, (c + q) * LANES : (c + q + 1) * LANES], jnp.uint32) for q in range(p)
        ]
        for r, out in enumerate(_transpose_words(words)):
          o2d[pl.ds((m0 + r) * rows_per_tok + c // p, m_sub // p, stride=stride), :] = out

  out_shape = jax.ShapeDtypeStruct(
      (num_tokens, d0, LANES),
      x2d.dtype,
      manual_axis_type=getattr(jax.typeof(x2d), "manual_axis_type", None),
  )
  return pl.pallas_call(
      _kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_tokens // tm,),
          in_specs=[pl.BlockSpec((tm, size_d), lambda i: (i, 0))],
          out_specs=pl.BlockSpec((tm, d0, LANES), lambda i: (i, 0, 0)),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
      interpret=interpret,
      name="relayout_2d_to_3d",
  )(x2d)


def quantize_relayout_2d_to_3d_pallas(
    x2d: jax.Array,
    qtype: jax.typing.DTypeLike,
    scale: float,
    *,
    block_tokens: int = 512,
    interpret: bool = False,
) -> jax.Array:
  """Fuses ``clip(x2d / scale, -fmax, fmax).astype(qtype)`` + ``[N, D // 128, 128]`` relayout in VMEM.

  Eliminates both the standalone HBM quantization pass and XLA's cross-sublane /
  SparseCore 2D->3D data-formatting pass. ``scale`` is a static Python float and the
  kernel multiplies by its reciprocal rounded once to f32, so the result is bit-exact with
  ``moe_fused_gmm3d.static_quantize(x2d, qtype, scale).reshape(N, D // 128, 128)`` for any scale.
  """
  if x2d.ndim != 2 or x2d.shape[1] % LANES != 0:
    raise ValueError(f"quantize_relayout_2d_to_3d_pallas expects 2D [N, D] with D % {LANES} == 0; got {x2d.shape}.")
  num_tokens, size_d = x2d.shape
  d0 = size_d // LANES
  out_dtype = jnp.dtype(qtype)
  bits = jax.dtypes.itemsize_bits(out_dtype)
  fmax = float(jnp.finfo(out_dtype).max)
  in_bits = jax.dtypes.itemsize_bits(x2d.dtype)
  inv_scale = jnp.asarray(1.0 / float(scale), jnp.float32)
  if bits not in (8, 16) or in_bits not in (8, 16, 32):
    xq = jnp.clip(x2d.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(out_dtype)
    return xq.reshape(num_tokens, d0, LANES)
  pack = 32 // bits
  in_sublane_align = 8 * max(1, 32 // in_bits)
  sublane_align = max(8 * pack, in_sublane_align)
  tm = math.gcd(block_tokens, num_tokens) if block_tokens > 0 and num_tokens > 0 else 0
  on_tpu = interpret or _is_tpu_backend()
  if not on_tpu or d0 < 8 or d0 % 8 != 0 or d0 % pack != 0 or tm < sublane_align or tm % sublane_align != 0:
    xq = jnp.clip(x2d.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(out_dtype)
    return xq.reshape(num_tokens, d0, LANES)

  m_sub = 128 if tm >= 128 and tm % 128 == 0 else tm
  out_shape = jax.ShapeDtypeStruct(
      (num_tokens, d0, LANES),
      out_dtype,
      manual_axis_type=getattr(jax.typeof(x2d), "manual_axis_type", None),
  )

  inv_s_float = float(1.0 / float(scale))

  def _static_kernel(in_ref, out_ref):
    o2d, rows_per_tok, p = _flat_u32_view(out_ref, d0)
    stride = p * rows_per_tok
    for m0 in range(0, tm, m_sub):
      for c in range(0, d0, p):
        words = []
        for q in range(p):
          col = in_ref[m0 : m0 + m_sub, (c + q) * LANES : (c + q + 1) * LANES].astype(jnp.float32)
          col_q = jnp.clip(col * inv_s_float, -fmax, fmax).astype(out_dtype)
          words.append(pltpu.bitcast(col_q, jnp.uint32))
        for r, out in enumerate(_transpose_words(words)):
          o2d[pl.ds((m0 + r) * rows_per_tok + c // p, m_sub // p, stride=stride), :] = out

  return pl.pallas_call(
      _static_kernel,
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_tokens // tm,),
          in_specs=[pl.BlockSpec((tm, size_d), lambda i: (i, 0))],
          out_specs=pl.BlockSpec((tm, d0, LANES), lambda i: (i, 0, 0)),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          disable_bounds_checks=True,
      ),
      interpret=interpret,
      name="quantize_relayout_2d_to_3d",
  )(x2d)
