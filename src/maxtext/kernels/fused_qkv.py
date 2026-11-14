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

"""Fused fp8 query / key / value projection forward Pallas TPU kernel.

One `pallas_call` computes, per device, the query, key and value of a
self-attention layer with per-head QK RMSNorm and RoPE (e.g. Qwen3) from the
pre-attention-normed decoder input `x`:

  q = rope(rms_norm(fp8_dot(quant(x), wq) * s, q_norm_scale)) * head_dim**-0.5
  k = rope(rms_norm(fp8_dot(quant(x), wk) * s, k_norm_scale))
  v = fp8_dot(quant(x), wv) * s

with `s = act_scale * weight_scale` the Qwix per-tensor dequantization scale.
The static fp8 activation quantization, the QK RMSNorms, RoPE and the
`query_pre_attn_scalar` all happen in VMEM on the f32 matmul result, so neither
the quantized activations nor the un-normed / un-roped projections round-trip
HBM, and the skinny K / V projections (N = num_kv_heads * head_dim = 512) are
computed by the same wide `[block_m, emb] x [emb, 1024]` fp8 MXU matmuls as the
query heads.

Numerics follow MaxText's `Attention` path (`layers/attentions.py`,
`layers/normalizations.py::RMSNorm`, `layers/embeddings.py::RotaryEmbedding`,
e.g. `models/qwen3.py`) with Qwix `fp8_full` and `fixed,-A,A` calibration
(`layers/quantizations.py`, `qwix/_src/core/{qarray,dot_general_qt}.py`):

  * activations and weights are quantized to e4m3 with the static per-tensor
    scale `A / 448` (`fixed,-224,224` -> 0.5): `clip(x / scale, -448, 448)`;
  * the fp8 x fp8 dot accumulates in f32 and its output is materialized in the
    compute dtype (bf16) before being scaled by `act_scale * weight_scale`
    (for power-of-two scales the order of rounding and scaling is irrelevant);
  * `RMSNorm`: `x * rsqrt(mean(x^2) + eps) * bf16(scale)`, computed in f32
    from the bf16 projection;
  * `RotaryEmbedding`: `x * cos + rotate_half(x) * sin`,
    `rotate_half((x1, x2)) = (-x2, x1)`, with `cos` / `sin` of
    `position / timescale` rounded to the compute dtype, `timescale[i] =
    min_timescale * (max_timescale / min_timescale) ** (2 i / head_dim)`;
  * `query *= bf16(head_dim ** -0.5)` (the weak-typed python scalar is
    rounded to the compute dtype by XLA).

Like XLA's fusions of the reference (which keep excess f32 precision inside
a fusion), the norm, RoPE and query scale are evaluated in f32 and rounded to
the compute dtype once. `fused_qkv_proj_reference` is the pure-JAX
formulation of the same numerics and the parity target of the kernel.

Outputs are written head-major, `q: [Hq, N, D]`, `v: [Hkv, N, D]` (the Splash
`HEAD_DIM_MINOR` layout) and, optionally, `k: [Hkv, D, N]` (`SEQ_MINOR`), so
that transposing them to `[B, T, H, D]` and back to the Splash layouts is a
no-op for XLA.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, Sequence

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

_F32 = jnp.float32
_MIB = 1024 * 1024


def _tpu_vmem_capacity_bytes(default_bytes: int = 64 * _MIB) -> int:
  """Returns active TPU VMEM capacity in bytes, or `default_bytes` when tracing off TPU."""
  try:
    return pltpu.get_tpu_info().vmem_capacity_bytes
  except (ValueError, RuntimeError):
    return default_bytes


def _fused_qkv_vmem_footprint(
    block_m: int,
    *,
    emb: int,
    tn: int,
    kv_cols: int,
    heads_per_tile: int,
    num_kv_heads: int,
    head_dim: int,
    x_itemsize: int,
    itemsize: int,
    q_itemsize: int,
    k_itemsize: int,
    w_itemsize: int,
    need_xq_scratch: bool,
    save_pre_norm: bool,
) -> int:
  """Estimated VMEM working set in bytes of one `fused_qkv_proj` grid step with a `block_m`-row tile."""
  acc_cols = max(tn, kv_cols)
  pre_norm_footprint = 2 * (heads_per_tile + num_kv_heads) * itemsize * block_m * head_dim if save_pre_norm else 0
  return (
      2 * block_m * emb * x_itemsize  # x, double-buffered
      + (block_m * emb if need_xq_scratch else 0)  # xq scratch
      + 2 * emb * tn * w_itemsize  # wq tiles
      + emb * kv_cols * w_itemsize  # wkv, resident
      + block_m * acc_cols * 4  # acc
      + 2 * (heads_per_tile * q_itemsize + num_kv_heads * (k_itemsize + itemsize)) * block_m * head_dim  # outputs
      + pre_norm_footprint
      + 4 * block_m * head_dim * itemsize  # cos / sin
  )


def _shrink_block_m_to_fit(block_m: int, footprint_fn: Any, budget_bytes: int) -> int:
  """Halves `block_m` until `footprint_fn(block_m)` fits `budget_bytes` (or it reaches 128).

  Only halves while `block_m` is a multiple of 256, so the result stays a multiple of 128 that divides the input
  (and therefore still divides `n_tokens` and `seq_len`). `block_m` only tiles the token rows, so the kernel's
  results do not depend on it. Tiles that already fit are returned unchanged.
  """
  while footprint_fn(block_m) > budget_bytes and block_m % 256 == 0:
    block_m //= 2
  return block_m


def static_quantize(x: jax.Array, qtype: Any, scale: float) -> jax.Array:
  """Qwix `fixed,-A,A` per-tensor quantization with static scale `A / fmax`.

  Reproduces `qwix.quantize_with_scale_zero_point` + `numerics.convert_to`:
  `clip(x / scale, -fmax, fmax).astype(qtype)`. The division is written as a
  multiplication by `1 / scale`, which is exact for power-of-two scales (the
  `fixed,-224,224` production scale is 0.5).

  Args:
    x: Array to quantize (any float dtype; evaluated in f32).
    qtype: fp8 dtype, e.g. `jnp.float8_e4m3fn`.
    scale: Static per-tensor scale.

  Returns:
    The quantized array.
  """
  fmax = float(jnp.finfo(qtype).max)
  return jnp.clip(x.astype(_F32) * (1.0 / scale), -fmax, fmax).astype(qtype)


def rope_tables(
    positions: jax.Array,
    head_dim: int,
    *,
    min_timescale: float,
    max_timescale: float,
    dtype: Any,
) -> tuple[jax.Array, jax.Array]:
  """`embeddings.RotaryEmbedding` cos / sin tables of the half head dim.

  Args:
    positions: `[N]` integer positions.
    head_dim: Rotary embedding dim D.
    min_timescale: Start of the geometric timescale index.
    max_timescale: End of the geometric timescale index (`rope_max_timescale`).
    dtype: Compute dtype the tables are rounded to (`inputs.dtype` in MaxText).

  Returns:
    `cos_half`, `sin_half` of shape `[N, D // 2]` in `dtype`.
  """
  if head_dim % 2:
    raise ValueError("head_dim must be even")
  # The timescale is a compile-time constant in the jitted model (XLA folds it
  # with libm's powf); evaluating it in numpy keeps the tables bit-identical
  # between eager and jitted callers.
  fraction = (2 * np.arange(0, head_dim // 2) / head_dim).astype(np.float32)
  timescale = np.float32(min_timescale) * np.float32(max_timescale / min_timescale) ** fraction
  sinusoid_inp = positions[:, None] / timescale
  return jnp.cos(sinusoid_inp).astype(dtype), jnp.sin(sinusoid_inp).astype(dtype)


def _rotate_half(x: jax.Array) -> jax.Array:
  """`RotaryEmbedding._rotate_half`: (x1, x2) -> (-x2, x1)."""
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate((-x2, x1), axis=-1)


def _rounded_scalar(value: float, dtype: Any) -> float:
  """`value` as XLA sees the weak-typed python scalar: rounded to `dtype`."""
  return float(np.asarray(value, dtype=dtype).astype(np.float32))


def _fused_qkv_kernel(
    x_ref,
    wq_ref,
    wkv_ref,
    q_scale_ref,
    k_scale_ref,
    cos_ref,
    sin_ref,
    q_ref,
    k_ref,
    v_ref,
    *extra_refs,
    num_q_tiles: int,
    heads_per_tile: int,
    num_kv_heads: int,
    eps: float,
    act_qtype: Any,
    act_scale: float,
    dequant_scale: float,
    q_scalar: float,
    quant_rows: int,
    q_seq_minor: bool,
    k_seq_minor: bool,
    batched_4d: bool,
    save_pre_norm: bool = False,
    x_is_quantized: bool = False,
    compute_dtype: Any = None,
    precision: lax.Precision | None,
):
  """Computes one `[block_m rows, tn columns]` tile of the fused projection."""
  if save_pre_norm:
    hq_ref, hk_ref, *scratch = extra_refs
  else:
    hq_ref = hk_ref = None
    scratch = extra_refs
  j = pl.program_id(1)
  block_m = x_ref.shape[0]
  head_dim = cos_ref.shape[1]
  compute_dtype = x_ref.dtype if compute_dtype is None else jnp.dtype(compute_dtype)
  q_dtype = q_ref.dtype
  k_dtype = k_ref.dtype
  v_dtype = v_ref.dtype
  acc_ref = scratch[0]
  kv_cols = 2 * num_kv_heads * head_dim

  if act_qtype is not None and not x_is_quantized:
    xq_ref = scratch[1]

    @pl.when(j == 0)
    def _quantize():
      def body(r, carry):
        rows = pl.ds(pl.multiple_of(r * quant_rows, quant_rows), quant_rows)
        xq_ref[rows, :] = static_quantize(x_ref[rows, :], act_qtype, act_scale)
        return carry

      lax.fori_loop(0, block_m // quant_rows, body, None)

    lhs = xq_ref[...]
  else:
    lhs = x_ref[...]

  def dot(w):
    return lax.dot_general(lhs, w, (((1,), (0,)), ((), ())), precision=precision, preferred_element_type=_F32)

  q_tile_cols = heads_per_tile * head_dim

  @pl.when(j < num_q_tiles)
  def _dot_q():
    if acc_ref.shape[1] == q_tile_cols:
      acc_ref[...] = dot(wq_ref[...])
    else:
      acc_ref[:, :q_tile_cols] = dot(wq_ref[...])

  @pl.when(j == num_q_tiles)
  def _dot_kv():
    acc_ref[:, :kv_cols] = dot(wkv_ref[...])

  cos = cos_ref[...].astype(_F32)
  sin = sin_ref[...].astype(_F32)

  def projection(col: int, scale: float) -> jax.Array:
    """Head `col` of the tile times `scale`, rounded to compute_dtype like the reference."""
    h = acc_ref[:, col * head_dim : (col + 1) * head_dim]
    if scale != 1.0:
      h = h * scale
    return h.astype(compute_dtype).astype(_F32)

  normed_eps = eps / (dequant_scale * dequant_scale)

  def norm_rope(h: jax.Array, scale: jax.Array) -> jax.Array:
    mean2 = jnp.mean(h * h, axis=-1, keepdims=True)
    y = h * lax.rsqrt(mean2 + normed_eps) * scale
    # rotate_half(y) * sin == roll(y, D / 2) * [-sin_half, sin_half].
    return y * cos + pltpu.roll(y, head_dim // 2, 1) * sin

  @pl.when(j < num_q_tiles)
  def _q():
    scale = q_scale_ref[...].astype(_F32)
    for c in range(heads_per_tile):
      if hq_ref is not None:
        hq_out = projection(c, dequant_scale).astype(compute_dtype)
        if batched_4d:
          hq_ref[0, c] = hq_out
        else:
          hq_ref[c] = hq_out
      q = norm_rope(projection(c, 1.0), scale) * q_scalar
      if q_seq_minor:
        q = q.T
      q_out = q.astype(q_dtype)
      if batched_4d:
        q_ref[0, c] = q_out
      else:
        q_ref[c] = q_out

  @pl.when(j == num_q_tiles)
  def _kv():
    scale = k_scale_ref[...].astype(_F32)
    for c in range(num_kv_heads):
      if hk_ref is not None:
        hk_out = projection(c, dequant_scale).astype(compute_dtype)
        if batched_4d:
          hk_ref[0, c] = hk_out
        else:
          hk_ref[c] = hk_out
      k = norm_rope(projection(c, 1.0), scale)
      if k_seq_minor:
        k = k.T
      k_out = k.astype(k_dtype)
      v_out = projection(num_kv_heads + c, dequant_scale).astype(v_dtype)
      if batched_4d:
        k_ref[0, c] = k_out
        v_ref[0, c] = v_out
      else:
        k_ref[c] = k_out
        v_ref[c] = v_out


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "eps",
        "rope_max_timescale",
        "rope_min_timescale",
        "query_pre_attn_scalar",
        "act_qtype",
        "act_scale",
        "weight_scale",
        "block_m",
        "heads_per_tile",
        "quant_rows",
        "seq_len",
        "q_seq_minor",
        "k_seq_minor",
        "q_out_dtype",
        "k_out_dtype",
        "save_pre_norm",
        "x_is_quantized",
        "compute_dtype",
        "vmem_limit_bytes",
        "interpret",
    ),
)
def fused_qkv_proj(
    x: jax.Array,
    w_qkv: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    rope_max_timescale: float,
    rope_min_timescale: float = 1.0,
    query_pre_attn_scalar: float | None = None,
    act_qtype: Any = jnp.float8_e4m3fn,
    act_scale: float = 0.5,
    weight_scale: float = 0.5,
    block_m: int = 1024,
    heads_per_tile: int = 8,
    quant_rows: int = 256,
    seq_len: int | None = None,
    q_seq_minor: bool = False,
    k_seq_minor: bool = False,
    q_out_dtype: Any = None,
    k_out_dtype: Any = None,
    save_pre_norm: bool = False,
    x_is_quantized: bool = False,
    compute_dtype: Any = None,
    vmem_limit_bytes: int | None = None,
    interpret: bool = False,
):
  """Fused q / k / v projection with QK-norm, RoPE and query scaling.

  When `seq_len` is provided (`n_tokens = B * seq_len`), writes 4D outputs
  directly in Splash's physical HBM layouts (`[B, H, D, S]` when `*_seq_minor`
  else `[B, H, S, D]`) so downstream `[B, S, H, D] -> Splash` transposes fold to
  exact identity with zero HBM relayout copies. When `save_pre_norm` is True,
  also returns `(hq, hk)` pre-norm projections in `HEAD_DIM_MINOR` for remat bwd.
  """
  n_tokens, emb = x.shape
  if x_is_quantized:
    if act_qtype is None:
      raise ValueError("x_is_quantized=True requires act_qtype to be set")
    if compute_dtype is None:
      raise ValueError("x_is_quantized=True requires compute_dtype to be set")
    if x.dtype != jnp.dtype(act_qtype):
      raise ValueError(f"x must have dtype {act_qtype} when x_is_quantized=True, got {x.dtype}")
    out_dtype = jnp.dtype(compute_dtype)
  else:
    out_dtype = x.dtype if compute_dtype is None else jnp.dtype(compute_dtype)
  q_dtype = out_dtype if q_out_dtype is None else jnp.dtype(q_out_dtype)
  k_dtype = out_dtype if k_out_dtype is None else jnp.dtype(k_out_dtype)
  q_cols = num_q_heads * head_dim
  kv_cols = 2 * num_kv_heads * head_dim
  tn = heads_per_tile * head_dim
  if w_qkv.shape != (emb, q_cols + kv_cols):
    raise ValueError(f"w_qkv has shape {w_qkv.shape}, expected {(emb, q_cols + kv_cols)}")
  if act_qtype is not None and w_qkv.dtype != jnp.dtype(act_qtype):
    raise ValueError(f"w_qkv must be {act_qtype} when activations are quantized, got {w_qkv.dtype}")
  if act_qtype is None and w_qkv.dtype != out_dtype:
    raise ValueError(f"w_qkv must be {out_dtype} in the unquantized mode, got {w_qkv.dtype}")
  if num_q_heads % heads_per_tile:
    raise ValueError(f"{heads_per_tile=} must divide {num_q_heads=}")
  if q_cols % kv_cols:
    raise ValueError(f"The query width {q_cols} must be a multiple of the key | value width {kv_cols}")
  max_tile = n_tokens if seq_len is None else seq_len
  block_m = min(block_m, max_tile)
  while max_tile % block_m != 0 and block_m > 128:
    block_m //= 2
  if block_m <= 0 or n_tokens % block_m or block_m % 128:
    raise ValueError(f"{block_m=} must divide {n_tokens=} and be a multiple of 128")
  if seq_len is not None and (n_tokens % seq_len or seq_len % block_m):
    raise ValueError(f"{seq_len=} must divide {n_tokens=} and be a multiple of {block_m=}")
  x_itemsize = jnp.dtype(x.dtype).itemsize
  itemsize = jnp.dtype(out_dtype).itemsize
  q_itemsize = jnp.dtype(q_dtype).itemsize
  k_itemsize = jnp.dtype(k_dtype).itemsize
  w_itemsize = jnp.dtype(w_qkv.dtype).itemsize
  need_xq_scratch = act_qtype is not None and not x_is_quantized
  footprint_fn = functools.partial(
      _fused_qkv_vmem_footprint,
      emb=emb,
      tn=tn,
      kv_cols=kv_cols,
      heads_per_tile=heads_per_tile,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      x_itemsize=x_itemsize,
      itemsize=itemsize,
      q_itemsize=q_itemsize,
      k_itemsize=k_itemsize,
      w_itemsize=w_itemsize,
      need_xq_scratch=need_xq_scratch,
      save_pre_norm=save_pre_norm,
  )
  if vmem_limit_bytes is None:
    vmem_budget = 60 * _MIB if interpret else max(12 * _MIB, _tpu_vmem_capacity_bytes() - 4 * _MIB)
  else:
    vmem_budget = vmem_limit_bytes
  if not interpret:
    # Shrink the row tile only where its working set would exceed the VMEM budget (e.g. 16 heads per tile with
    # 8 KV heads, or smaller-VMEM TPUs); without this the check below raises. Tiles that fit are unchanged.
    block_m = _shrink_block_m_to_fit(block_m, footprint_fn, vmem_budget)
  quant_rows = min(quant_rows, block_m)
  while block_m % quant_rows != 0 and quant_rows > 32:
    quant_rows //= 2
  if block_m % quant_rows or quant_rows % 32:
    raise ValueError(f"{quant_rows=} must divide {block_m=} and be a multiple of 32")
  if query_pre_attn_scalar is None:
    query_pre_attn_scalar = head_dim**-0.5
  if act_qtype is not None:
    for name, s in (("act_scale", act_scale), ("weight_scale", weight_scale)):
      if not math.log2(s).is_integer():
        raise ValueError(f"{name}={s} must be a power of two for the kernel to match Qwix bit-exactly")

  cos_half, sin_half = rope_tables(
      positions, head_dim, min_timescale=rope_min_timescale, max_timescale=rope_max_timescale, dtype=out_dtype
  )
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)
  sin = jnp.concatenate([-sin_half, sin_half], axis=-1)
  # `RMSNorm` rounds its scale to the compute dtype.
  q_scale = q_norm_scale.astype(out_dtype).reshape(1, head_dim)
  k_scale = k_norm_scale.astype(out_dtype).reshape(1, head_dim)

  num_q_tiles = q_cols // tn
  num_m = n_tokens // block_m
  grid = (num_m, num_q_tiles + 1)
  precision = None if act_qtype is not None else lax.Precision.HIGHEST
  q_scalar_val = (
      _rounded_scalar(query_pre_attn_scalar, out_dtype)
      if jnp.dtype(q_dtype) == jnp.dtype(out_dtype)
      else float(query_pre_attn_scalar)
  )
  batched_4d = seq_len is not None
  kernel = functools.partial(
      _fused_qkv_kernel,
      num_q_tiles=num_q_tiles,
      heads_per_tile=heads_per_tile,
      num_kv_heads=num_kv_heads,
      eps=eps,
      act_qtype=act_qtype,
      act_scale=act_scale,
      dequant_scale=act_scale * weight_scale if act_qtype is not None else 1.0,
      q_scalar=q_scalar_val,
      quant_rows=quant_rows,
      q_seq_minor=q_seq_minor,
      k_seq_minor=k_seq_minor,
      batched_4d=batched_4d,
      save_pre_norm=save_pre_norm,
      x_is_quantized=x_is_quantized,
      compute_dtype=out_dtype,
      precision=precision,
  )

  def resident(i, j):
    del i, j
    return (0, 0)

  def rows(i, j):
    del j
    return (i, 0)

  in_specs = [
      pl.BlockSpec((block_m, emb), rows),
      # Query weight tiles; the key | value step keeps the last tile resident.
      pl.BlockSpec((emb, tn), lambda i, j: (0, jnp.minimum(j, num_q_tiles - 1))),
      # The key | value weights, fetched once (block index in units of kv_cols).
      pl.BlockSpec((emb, kv_cols), lambda i, j: (0, q_cols // kv_cols), pipeline_mode=pl.Buffered(1)),
      pl.BlockSpec((1, head_dim), resident, pipeline_mode=pl.Buffered(1)),
      pl.BlockSpec((1, head_dim), resident, pipeline_mode=pl.Buffered(1)),
      pl.BlockSpec((block_m, head_dim), rows),
      pl.BlockSpec((block_m, head_dim), rows),
  ]
  if batched_4d:
    assert seq_len is not None
    batch_local = n_tokens // seq_len
    tiles_per_seq = seq_len // block_m
    hq_shape = (batch_local, num_q_heads, seq_len, head_dim)
    hq_spec = pl.BlockSpec(
        (1, heads_per_tile, block_m, head_dim),
        lambda i, j: (i // tiles_per_seq, jnp.minimum(j, num_q_tiles - 1), i % tiles_per_seq, 0),
    )
    hk_shape = (batch_local, num_kv_heads, seq_len, head_dim)
    hk_spec = pl.BlockSpec(
        (1, num_kv_heads, block_m, head_dim),
        lambda i, j: (i // tiles_per_seq, 0, i % tiles_per_seq, 0),
    )
    if q_seq_minor:
      q_shape = (batch_local, num_q_heads, head_dim, seq_len)
      q_spec = pl.BlockSpec(
          (1, heads_per_tile, head_dim, block_m),
          lambda i, j: (i // tiles_per_seq, jnp.minimum(j, num_q_tiles - 1), 0, i % tiles_per_seq),
      )
    else:
      q_shape = hq_shape
      q_spec = hq_spec
    if k_seq_minor:
      k_shape = (batch_local, num_kv_heads, head_dim, seq_len)
      k_spec = pl.BlockSpec(
          (1, num_kv_heads, head_dim, block_m),
          lambda i, j: (i // tiles_per_seq, 0, 0, i % tiles_per_seq),
      )
    else:
      k_shape = hk_shape
      k_spec = hk_spec
    v_shape = hk_shape
    v_spec = hk_spec
  else:
    hq_shape = (num_q_heads, n_tokens, head_dim)
    hq_spec = pl.BlockSpec((heads_per_tile, block_m, head_dim), lambda i, j: (jnp.minimum(j, num_q_tiles - 1), i, 0))
    hk_shape = (num_kv_heads, n_tokens, head_dim)
    hk_spec = pl.BlockSpec((num_kv_heads, block_m, head_dim), lambda i, j: (0, i, 0))
    if q_seq_minor:
      q_shape = (num_q_heads, head_dim, n_tokens)
      q_spec = pl.BlockSpec((heads_per_tile, head_dim, block_m), lambda i, j: (jnp.minimum(j, num_q_tiles - 1), 0, i))
    else:
      q_shape = hq_shape
      q_spec = hq_spec
    if k_seq_minor:
      k_shape = (num_kv_heads, head_dim, n_tokens)
      k_spec = pl.BlockSpec((num_kv_heads, head_dim, block_m), lambda i, j: (0, 0, i))
    else:
      k_shape = hk_shape
      k_spec = hk_spec
    v_shape = hk_shape
    v_spec = hk_spec
  out_specs = [q_spec, k_spec, v_spec]
  out_shape = [
      jax.ShapeDtypeStruct(q_shape, q_dtype),
      jax.ShapeDtypeStruct(k_shape, k_dtype),
      jax.ShapeDtypeStruct(v_shape, out_dtype),
  ]
  if save_pre_norm:
    out_specs.extend([hq_spec, hk_spec])
    out_shape.extend(
        [
            jax.ShapeDtypeStruct(hq_shape, out_dtype),
            jax.ShapeDtypeStruct(hk_shape, out_dtype),
        ]
    )
  acc_cols = max(tn, kv_cols)
  scratch_shapes = [pltpu.VMEM((block_m, acc_cols), _F32)]
  if need_xq_scratch:
    scratch_shapes.append(pltpu.VMEM((block_m, emb), act_qtype))

  footprint = footprint_fn(block_m)
  if vmem_limit_bytes is None:
    vmem_limit_bytes = min(footprint + 12 * _MIB, vmem_budget)
  if not interpret and footprint > vmem_limit_bytes:
    raise ValueError(f"VMEM footprint {footprint / _MIB:.1f} MiB exceeds {vmem_limit_bytes / _MIB:.1f} MiB")

  flops = 2 * n_tokens * emb * (q_cols + kv_cols)
  out_bytes = n_tokens * (num_q_heads * q_itemsize + num_kv_heads * (k_itemsize + itemsize)) * head_dim
  if save_pre_norm:
    out_bytes += n_tokens * (num_q_heads + num_kv_heads) * itemsize * head_dim
  bytes_accessed = x.size * x_itemsize + num_m * w_qkv.size * w_itemsize + out_bytes
  return pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shape,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      interpret=interpret,
      name="fused_qkv_remat_fwd" if save_pre_norm else "fused_qkv_fwd",
  )(x, w_qkv, w_qkv, q_scale, k_scale, cos, sin)


def _fused_qkv_bwd_post_kernel(
    hq_ref,
    dq_ref,
    hk_ref,
    dk_ref,
    dv_ref,
    q_scale_ref,
    k_scale_ref,
    cos_ref,
    sin_ref,
    dproj_ref,
    dq_scale_out_ref,
    dk_scale_out_ref,
    amax_out_ref,
    dq_s_scratch_ref,
    amax_scratch_ref,
    *,
    num_q_tiles: int,
    bwd_heads_per_tile: int,
    num_kv_heads: int,
    eps: float,
    q_scalar: float,
    out_qtype: Any,
    out_scale: float | None,
):
  """Fused VMEM backward of query scaling + RoPE + QK RMSNorm + dproj packing."""
  j = pl.program_id(1)
  block_m = cos_ref.shape[0]
  head_dim = cos_ref.shape[1]
  compute_dtype = hq_ref.dtype
  cos = cos_ref[...].astype(_F32)
  sin = sin_ref[...].astype(_F32)

  def bwd_heads(h_3d: jax.Array, dout_3d: jax.Array, scale_2d: jax.Array, scalar: float, n_heads: int):
    rows_total = n_heads * block_m
    h = h_3d.reshape(rows_total, head_dim).astype(_F32)
    drot = dout_3d.reshape(rows_total, head_dim).astype(_F32)
    if scalar != 1.0:
      drot = drot * scalar
    drot_rolled = pltpu.roll(drot, head_dim // 2, 1)
    # Transpose of `y * cos + roll(y, D/2) * sin` where `sin = [-sin_half, sin_half]`,
    # broadcasting `cos`/`sin` across `n_heads` without materializing `[n_heads * block_m, D]` VMEM copies.
    dy = (
        drot.reshape(n_heads, block_m, head_dim) * cos[None, :, :]
        - drot_rolled.reshape(n_heads, block_m, head_dim) * sin[None, :, :]
    ).reshape(rows_total, head_dim)
    rrms = lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
    u = h * rrms
    dscale_8xd = jnp.sum((dy * u).reshape(rows_total // 8, 8, head_dim), axis=0)
    du = dy * scale_2d
    dh = (rrms * (du - u * jnp.mean(du * u, axis=-1, keepdims=True))).astype(compute_dtype)
    amax_8xd = jnp.max(jnp.abs(dh.astype(_F32)).reshape(rows_total // 8, 8, head_dim), axis=0)
    if out_qtype is not None:
      assert out_scale is not None
      dh_out = static_quantize(dh, out_qtype, out_scale)
    else:
      dh_out = dh
    return dh_out.reshape(n_heads, block_m, head_dim), dscale_8xd, amax_8xd

  @pl.when(j == 0)
  def _init_scratch():
    dq_s_scratch_ref[...] = jnp.zeros((8, head_dim), _F32)
    amax_scratch_ref[...] = jnp.zeros((8, head_dim), _F32)

  @pl.when(j < num_q_tiles)
  def _bwd_q():
    q_scale = q_scale_ref[...].astype(_F32)
    dh_3d, ds_q, amax_q = bwd_heads(hq_ref[0], dq_ref[0], q_scale, q_scalar, bwd_heads_per_tile)
    dq_s_scratch_ref[...] = dq_s_scratch_ref[...] + ds_q
    amax_scratch_ref[...] = jnp.maximum(amax_scratch_ref[...], amax_q)
    for c in range(bwd_heads_per_tile):
      dproj_ref[:, pl.ds(c * head_dim, head_dim)] = dh_3d[c]

  @pl.when(j == num_q_tiles)
  def _bwd_kv():
    k_scale = k_scale_ref[...].astype(_F32)
    dh_k_3d, acc_ks, amax_k = bwd_heads(hk_ref[0], dk_ref[0], k_scale, 1.0, num_kv_heads)
    acc_amax = jnp.maximum(amax_scratch_ref[...], amax_k)
    dv_rows = num_kv_heads * block_m
    dv_2d = dv_ref[0].reshape(dv_rows, head_dim).astype(compute_dtype)
    amax_v = jnp.max(jnp.abs(dv_2d.astype(_F32)).reshape(dv_rows // 8, 8, head_dim), axis=0)
    acc_amax = jnp.maximum(acc_amax, amax_v)
    if out_qtype is not None:
      assert out_scale is not None
      dv_2d = static_quantize(dv_2d, out_qtype, out_scale)
    dv_3d = dv_2d.reshape(num_kv_heads, block_m, head_dim)
    for c in range(num_kv_heads):
      dproj_ref[:, pl.ds(c * head_dim, head_dim)] = dh_k_3d[c]
    for c in range(num_kv_heads):
      dproj_ref[:, pl.ds((num_kv_heads + c) * head_dim, head_dim)] = dv_3d[c]
    dq_scale_out_ref[...] = dq_s_scratch_ref[...]
    dk_scale_out_ref[...] = acc_ks
    amax_out_ref[...] = acc_amax


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "rope_max_timescale",
        "rope_min_timescale",
        "query_pre_attn_scalar",
        "q_out_dtype",
        "out_qtype",
        "out_scale",
        "block_m",
        "interpret",
    ),
)
def fused_qkv_bwd_post(
    hq: jax.Array,
    dq_bhsd: jax.Array,
    hk: jax.Array,
    dk_bhsd: jax.Array,
    dv_bhsd: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions: jax.Array,
    *,
    eps: float,
    rope_max_timescale: float,
    rope_min_timescale: float = 1.0,
    query_pre_attn_scalar: float | None = None,
    q_out_dtype: Any = None,
    out_qtype: Any = None,
    out_scale: float | None = None,
    block_m: int = 512,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Fused Pallas backward of QK-norm + RoPE + query scaling directly from HEAD_DIM_MINOR."""
  batch_local, num_q_heads, seq_len, head_dim = hq.shape
  num_kv_heads = hk.shape[1]
  n_tokens = batch_local * seq_len
  out_dtype = hq.dtype
  q_dtype = out_dtype if q_out_dtype is None else jnp.dtype(q_out_dtype)
  bwd_heads_per_tile = 2 * num_kv_heads
  if num_q_heads % bwd_heads_per_tile:
    raise ValueError(f"{bwd_heads_per_tile=} (2 * num_kv_heads) must divide {num_q_heads=}")
  if head_dim % 128:
    raise ValueError(f"{head_dim=} must be a multiple of 128")
  block_m = min(block_m, seq_len)
  while seq_len % block_m != 0 and block_m > 128:
    block_m //= 2
  if seq_len % block_m or block_m % 128:
    raise ValueError(f"{block_m=} must divide {seq_len=} and be a multiple of 128")

  if query_pre_attn_scalar is None:
    query_pre_attn_scalar = head_dim**-0.5
  q_scalar_val = (
      _rounded_scalar(query_pre_attn_scalar, out_dtype)
      if jnp.dtype(q_dtype) == jnp.dtype(out_dtype)
      else float(query_pre_attn_scalar)
  )

  cos_half, sin_half = rope_tables(
      positions, head_dim, min_timescale=rope_min_timescale, max_timescale=rope_max_timescale, dtype=out_dtype
  )
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)
  sin = jnp.concatenate([-sin_half, sin_half], axis=-1)
  q_scale = q_norm_scale.astype(out_dtype).reshape(1, head_dim)
  k_scale = k_norm_scale.astype(out_dtype).reshape(1, head_dim)

  tn = bwd_heads_per_tile * head_dim
  num_q_tiles = num_q_heads // bwd_heads_per_tile
  num_m = n_tokens // block_m
  tiles_per_seq = seq_len // block_m
  grid = (num_m, num_q_tiles + 1)

  kernel = functools.partial(
      _fused_qkv_bwd_post_kernel,
      num_q_tiles=num_q_tiles,
      bwd_heads_per_tile=bwd_heads_per_tile,
      num_kv_heads=num_kv_heads,
      eps=eps,
      q_scalar=q_scalar_val,
      out_qtype=out_qtype,
      out_scale=out_scale,
  )

  def resident(i, j):
    del i, j
    return (0, 0)

  def rows(i, j):
    del j
    return (i, 0)

  def q_idx(i, j):
    return (i // tiles_per_seq, jnp.minimum(j, num_q_tiles - 1), i % tiles_per_seq, 0)

  def kv_idx(i, j):
    del j
    return (i // tiles_per_seq, 0, i % tiles_per_seq, 0)

  in_specs = [
      pl.BlockSpec((1, bwd_heads_per_tile, block_m, head_dim), q_idx),
      pl.BlockSpec((1, bwd_heads_per_tile, block_m, head_dim), q_idx),
      pl.BlockSpec((1, num_kv_heads, block_m, head_dim), kv_idx),
      pl.BlockSpec((1, num_kv_heads, block_m, head_dim), kv_idx),
      pl.BlockSpec((1, num_kv_heads, block_m, head_dim), kv_idx),
      pl.BlockSpec((1, head_dim), resident, pipeline_mode=pl.Buffered(1)),
      pl.BlockSpec((1, head_dim), resident, pipeline_mode=pl.Buffered(1)),
      pl.BlockSpec((block_m, head_dim), rows),
      pl.BlockSpec((block_m, head_dim), rows),
  ]
  dproj_dtype = out_dtype if out_qtype is None else jnp.dtype(out_qtype)
  out_specs = [
      pl.BlockSpec((block_m, tn), lambda i, j: (i, j)),
      pl.BlockSpec((8, head_dim), rows),
      pl.BlockSpec((8, head_dim), rows),
      pl.BlockSpec((8, head_dim), rows),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((n_tokens, (num_q_heads + 2 * num_kv_heads) * head_dim), dproj_dtype),
      jax.ShapeDtypeStruct((num_m * 8, head_dim), _F32),
      jax.ShapeDtypeStruct((num_m * 8, head_dim), _F32),
      jax.ShapeDtypeStruct((num_m * 8, head_dim), _F32),
  ]
  scratch_shapes = [
      pltpu.VMEM((8, head_dim), _F32),
      pltpu.VMEM((8, head_dim), _F32),
  ]
  itemsize = jnp.dtype(out_dtype).itemsize
  dproj_itemsize = jnp.dtype(dproj_dtype).itemsize
  bytes_accessed = (
      n_tokens * (2 * num_q_heads + 3 * num_kv_heads) * head_dim * itemsize
      + n_tokens * (num_q_heads + 2 * num_kv_heads) * head_dim * dproj_itemsize
  )
  flops = 14 * n_tokens * (num_q_heads + num_kv_heads) * head_dim
  dproj, dq_scale_partial, dk_scale_partial, amax_partial = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shape,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary"),
          vmem_limit_bytes=(
              48 * _MIB if interpret else min(48 * _MIB, max(12 * _MIB, _tpu_vmem_capacity_bytes() - 4 * _MIB))
          ),
      ),
      cost_estimate=pl.CostEstimate(
          flops=flops, transcendentals=n_tokens * (num_q_heads + num_kv_heads), bytes_accessed=bytes_accessed
      ),
      interpret=interpret,
      name="fused_qkv_bwd_post",
  )(
      hq,
      dq_bhsd.astype(out_dtype),
      hk,
      dk_bhsd.astype(out_dtype),
      dv_bhsd.astype(out_dtype),
      q_scale,
      k_scale,
      cos,
      sin,
  )
  dq_scale = jnp.sum(dq_scale_partial, axis=0)
  dk_scale = jnp.sum(dk_scale_partial, axis=0)
  dproj_amax = jnp.max(amax_partial)
  return dproj, dq_scale, dk_scale, dproj_amax


def fuse_qkv_weights(
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    *,
    compute_dtype: Any = jnp.bfloat16,
    qtype: Any = jnp.float8_e4m3fn,
    weight_scale: float = 0.5,
) -> jax.Array:
  """Fuses and quantizes MaxText `[emb, H, D]` q / k / v kernels for the kernel."""
  emb = wq.shape[0]
  w = jnp.concatenate([w.reshape(emb, -1).astype(compute_dtype) for w in (wq, wk, wv)], axis=-1)
  if qtype is None:
    return w
  return static_quantize(w, qtype, weight_scale)


def fused_qkv_proj_reference(
    x: jax.Array,
    w_qkv: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    rope_max_timescale: float,
    rope_min_timescale: float = 1.0,
    query_pre_attn_scalar: float | None = None,
    act_qtype: Any = jnp.float8_e4m3fn,
    act_scale: float = 0.5,
    weight_scale: float = 0.5,
    seq_len: int | None = None,
    q_seq_minor: bool = False,
    k_seq_minor: bool = False,
    q_out_dtype: Any = None,
    k_out_dtype: Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Pure-JAX reference of `fused_qkv_proj` with MaxText's `Attention` numerics."""
  n_tokens = x.shape[0]
  out_dtype = x.dtype
  q_dtype = out_dtype if q_out_dtype is None else jnp.dtype(q_out_dtype)
  k_dtype = out_dtype if k_out_dtype is None else jnp.dtype(k_out_dtype)
  if query_pre_attn_scalar is None:
    query_pre_attn_scalar = head_dim**-0.5
  if act_qtype is not None:
    lhs = static_quantize(x, act_qtype, act_scale)
    proj = jnp.dot(lhs, w_qkv, preferred_element_type=_F32) * (act_scale * weight_scale)
  else:
    proj = jnp.dot(x, w_qkv, preferred_element_type=_F32, precision=lax.Precision.HIGHEST)
  # `DenseGeneral` materializes the projections in the compute dtype.
  proj = proj.astype(out_dtype)
  q_cols = num_q_heads * head_dim
  k_cols = num_kv_heads * head_dim
  q = proj[:, :q_cols].reshape(n_tokens, num_q_heads, head_dim)
  k = proj[:, q_cols : q_cols + k_cols].reshape(n_tokens, num_kv_heads, head_dim)
  v = proj[:, q_cols + k_cols :].reshape(n_tokens, num_kv_heads, head_dim)

  cos_half, sin_half = rope_tables(
      positions, head_dim, min_timescale=rope_min_timescale, max_timescale=rope_max_timescale, dtype=out_dtype
  )
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)[:, None, :].astype(_F32)
  sin = jnp.concatenate([sin_half, sin_half], axis=-1)[:, None, :].astype(_F32)

  def norm_rope(h: jax.Array, scale: jax.Array) -> jax.Array:
    """`RMSNorm` then `RotaryEmbedding`, in f32 like XLA's fusion of them."""
    h = h.astype(_F32)
    mean2 = jnp.mean(lax.square(h), axis=-1, keepdims=True)
    y = h * lax.rsqrt(mean2 + eps) * scale.astype(out_dtype).astype(_F32)
    return y * cos + _rotate_half(y) * sin

  q_scalar_val = (
      _rounded_scalar(query_pre_attn_scalar, out_dtype)
      if jnp.dtype(q_dtype) == jnp.dtype(out_dtype)
      else float(query_pre_attn_scalar)
  )
  q = (norm_rope(q, q_norm_scale) * q_scalar_val).astype(q_dtype)
  k = norm_rope(k, k_norm_scale).astype(k_dtype)
  if seq_len is not None:
    batch_local = n_tokens // seq_len
    q = jnp.transpose(
        q.reshape(batch_local, seq_len, num_q_heads, head_dim),
        (0, 2, 3, 1) if q_seq_minor else (0, 2, 1, 3),
    )
    k = jnp.transpose(
        k.reshape(batch_local, seq_len, num_kv_heads, head_dim),
        (0, 2, 3, 1) if k_seq_minor else (0, 2, 1, 3),
    )
    v = jnp.transpose(v.reshape(batch_local, seq_len, num_kv_heads, head_dim), (0, 2, 1, 3))
    return q, k, v
  q = jnp.transpose(q, (1, 2, 0) if q_seq_minor else (1, 0, 2))
  k = jnp.transpose(k, (1, 2, 0) if k_seq_minor else (1, 0, 2))
  v = jnp.transpose(v, (1, 0, 2))
  return q, k, v


@dataclasses.dataclass(frozen=True)
class FusedQkvTrainParams:
  """Static parameters of `fused_qkv_local` (differentiable FSDP QAG + fused QKV)."""

  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  eps: float
  rope_max_timescale: float
  rope_min_timescale: float = 1.0
  query_pre_attn_scalar: float | None = None
  compute_dtype: Any = jnp.bfloat16
  act_qtype: Any = jnp.float8_e4m3fn
  weight_qtype: Any = None
  bwd_qtype: Any = jnp.float8_e5m2
  act_scale: float = 0.5
  weight_scale: float = 0.5
  bwd_scale: float | None = 224.0 / 57344.0
  q_seq_minor: bool = False
  k_seq_minor: bool = False
  q_out_dtype: Any = None
  k_out_dtype: Any = None
  weight_gather_axes: tuple[str, ...] = ()
  extra_w_psum_axes: tuple[str, ...] = ()
  norm_psum_axes: tuple[str, ...] = ()
  defer_norm_grad_psum: bool = False
  x_dtype: Any = jnp.bfloat16
  block_m: int = 1024
  heads_per_tile: int = 8
  quant_rows: int = 256
  use_reference: bool = False
  interpret: bool = False


def _varying_axes(x: jax.Array) -> tuple[str, ...]:
  try:
    return tuple(sorted(jax.typeof(x).manual_axis_type.varying))
  except (AttributeError, TypeError):
    return ()


def _guard_with_tensor(target: jax.Array, dep: jax.Array) -> jax.Array:
  """Zero-FLOP scalar NaN barrier forcing `dep` to execute before `target`'s consumer fusion."""
  dep_vma = set(_varying_axes(dep))
  if dep_vma and not dep_vma.issubset(set(_varying_axes(target))):
    return target
  elem = dep[(0,) * dep.ndim] if dep.ndim > 0 else dep
  cond = jnp.isnan(elem)
  return jnp.where(cond, jnp.zeros_like(target), target)


def _all_gather_rows(x: jax.Array, axes: Sequence[str]) -> jax.Array:
  if not axes:
    return x
  ax = tuple(axes) if len(axes) > 1 else axes[0]
  return lax.all_gather(x, ax, axis=0, tiled=True)


def _reduce_scatter_rows(x: jax.Array, axes: Sequence[str]) -> jax.Array:
  if not axes:
    return x
  ax = tuple(axes) if len(axes) > 1 else axes[0]
  return lax.psum_scatter(x, ax, scatter_dimension=0, tiled=True)


def _all_gather_axis(x: jax.Array, axes: Sequence[str], axis: int) -> jax.Array:
  if not axes:
    return x
  ax = tuple(axes) if len(axes) > 1 else axes[0]
  return lax.all_gather(x, ax, axis=axis, tiled=True)


def _reduce_scatter_axis(x: jax.Array, axes: Sequence[str], axis: int) -> jax.Array:
  if not axes:
    return x
  ax = tuple(axes) if len(axes) > 1 else axes[0]
  return lax.psum_scatter(x, ax, scatter_dimension=axis, tiled=True)


def _pow2_absmax_quantize(
    x: jax.Array,
    qtype: Any,
    *,
    precomputed_amax: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` to `qtype` with a scalar power-of-two absmax scale; returns `(x_q, scale)`."""
  fmax = float(jnp.finfo(qtype).max)
  amax = jnp.max(jnp.abs(x.astype(_F32))) if precomputed_amax is None else precomputed_amax.astype(_F32)
  exp = jnp.ceil(jnp.log2(jnp.maximum(amax / fmax, 1e-30)))
  scale = jnp.where(amax > 0, jnp.exp2(exp), 1.0).astype(_F32)
  xq = jnp.clip(x.astype(_F32) / scale, -fmax, fmax).astype(qtype)
  return xq, scale


def _post_proj_bshd(
    p: FusedQkvTrainParams,
    proj: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    pos_1d: jax.Array,
    batch_local: int,
    seq_len: int,
    *,
    cast_qk_out: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Applies QK RMSNorm + RoPE + q_scalar + fp8/bf16 cast on fused `proj [N, (Hq+2Hkv)*D]`."""
  n_tokens = batch_local * seq_len
  out_dtype = jnp.dtype(p.compute_dtype)
  q_dtype = out_dtype if p.q_out_dtype is None else jnp.dtype(p.q_out_dtype)
  k_dtype = out_dtype if p.k_out_dtype is None else jnp.dtype(p.k_out_dtype)
  q_cols = p.num_q_heads * p.head_dim
  k_cols = p.num_kv_heads * p.head_dim
  q = proj[:, :q_cols].reshape(n_tokens, p.num_q_heads, p.head_dim)
  k = proj[:, q_cols : q_cols + k_cols].reshape(n_tokens, p.num_kv_heads, p.head_dim)
  v = proj[:, q_cols + k_cols :].reshape(batch_local, seq_len, p.num_kv_heads, p.head_dim)

  cos_half, sin_half = rope_tables(
      pos_1d, p.head_dim, min_timescale=p.rope_min_timescale, max_timescale=p.rope_max_timescale, dtype=out_dtype
  )
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)[:, None, :]
  sin = jnp.concatenate([sin_half, sin_half], axis=-1)[:, None, :]

  def norm_rope(h: jax.Array, scale: jax.Array) -> jax.Array:
    h32 = h.astype(_F32)
    mean2 = jnp.mean(lax.square(h32), axis=-1, keepdims=True)
    y = (h32 * lax.rsqrt(mean2 + p.eps)).astype(out_dtype)
    y = y * scale.astype(_F32).astype(out_dtype)
    return ((y * cos) + (_rotate_half(y) * sin)).astype(out_dtype)

  q = norm_rope(q, q_norm_scale)
  k = norm_rope(k, k_norm_scale)
  q_scalar = p.head_dim**-0.5 if p.query_pre_attn_scalar is None else p.query_pre_attn_scalar
  if q_dtype != out_dtype:
    q = (q.astype(_F32) * float(q_scalar)).astype(q_dtype if cast_qk_out else out_dtype)
  else:
    q = (q * _rounded_scalar(q_scalar, out_dtype)).astype(out_dtype)
  if cast_qk_out and k_dtype != out_dtype:
    k = k.astype(k_dtype)
  q = q.reshape(batch_local, seq_len, p.num_q_heads, p.head_dim)
  k = k.reshape(batch_local, seq_len, p.num_kv_heads, p.head_dim)
  return q, k, v.astype(out_dtype)


def _use_pallas_vjp(p: FusedQkvTrainParams, seq_len: int) -> bool:
  return (
      not p.use_reference
      and seq_len % 128 == 0
      and p.head_dim % 128 == 0
      and p.num_q_heads % p.heads_per_tile == 0
      and p.num_q_heads % (2 * p.num_kv_heads) == 0
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def fused_qkv_local(
    p: FusedQkvTrainParams,
    x_bsd: jax.Array,
    wq_local: jax.Array,
    wk_local: jax.Array,
    wv_local: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions_bs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Per-shard fused QKV projection + QK-norm + RoPE + Splash physical layout."""
  batch_local, seq_len, emb = x_bsd.shape
  if not _use_pallas_vjp(p, seq_len):
    return _fused_qkv_local_fwd(p, x_bsd, wq_local, wk_local, wv_local, q_norm_scale, k_norm_scale, positions_bs)[0]
  n_tokens = batch_local * seq_len
  x_2d = x_bsd.reshape(n_tokens, emb).astype(p.compute_dtype)
  pos_1d = jnp.broadcast_to(positions_bs, (batch_local, seq_len)).reshape(n_tokens)
  w_qtype = p.act_qtype if p.weight_qtype is None else p.weight_qtype
  w_qkv_local = fuse_qkv_weights(
      wq_local, wk_local, wv_local, compute_dtype=p.compute_dtype, qtype=w_qtype, weight_scale=p.weight_scale
  )
  w_qkv = _all_gather_rows(w_qkv_local, p.weight_gather_axes)
  q_raw, k_raw, v_raw = fused_qkv_proj(
      x_2d,
      w_qkv,
      q_norm_scale,
      k_norm_scale,
      pos_1d,
      num_q_heads=p.num_q_heads,
      num_kv_heads=p.num_kv_heads,
      head_dim=p.head_dim,
      eps=p.eps,
      rope_max_timescale=p.rope_max_timescale,
      rope_min_timescale=p.rope_min_timescale,
      query_pre_attn_scalar=p.query_pre_attn_scalar,
      act_qtype=p.act_qtype,
      act_scale=p.act_scale,
      weight_scale=p.weight_scale,
      block_m=p.block_m,
      heads_per_tile=p.heads_per_tile,
      quant_rows=p.quant_rows,
      seq_len=seq_len,
      q_seq_minor=p.q_seq_minor,
      k_seq_minor=p.k_seq_minor,
      q_out_dtype=p.q_out_dtype,
      k_out_dtype=p.k_out_dtype,
      x_is_quantized=False,
      compute_dtype=p.compute_dtype,
      interpret=p.interpret,
  )
  q_bshd = jnp.transpose(q_raw, (0, 3, 1, 2) if p.q_seq_minor else (0, 2, 1, 3))
  k_bshd = jnp.transpose(k_raw, (0, 3, 1, 2) if p.k_seq_minor else (0, 2, 1, 3))
  v_bshd = jnp.transpose(v_raw, (0, 2, 1, 3))
  return q_bshd, k_bshd, v_bshd


def _fused_qkv_local_fwd(
    p: FusedQkvTrainParams,
    x_bsd: jax.Array,
    wq_local: jax.Array,
    wk_local: jax.Array,
    wv_local: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions_bs: jax.Array,
):
  """Forward of the per-shard fused QKV projection; returns the outputs and custom-VJP residuals."""
  batch_local, seq_len, emb = x_bsd.shape
  n_tokens = batch_local * seq_len
  x_bsd_c = x_bsd.astype(p.compute_dtype)
  x_2d = x_bsd_c.reshape(n_tokens, emb)
  pos_2d = jnp.broadcast_to(positions_bs, (batch_local, seq_len))
  pos_1d = pos_2d.reshape(n_tokens)
  w_qtype = p.act_qtype if p.weight_qtype is None else p.weight_qtype
  w_qkv_local = fuse_qkv_weights(
      wq_local, wk_local, wv_local, compute_dtype=p.compute_dtype, qtype=w_qtype, weight_scale=p.weight_scale
  )
  w_qkv = _all_gather_rows(w_qkv_local, p.weight_gather_axes)
  if _use_pallas_vjp(p, seq_len):
    q_raw, k_raw, v_raw, hq, hk = fused_qkv_proj(
        x_2d,
        w_qkv,
        q_norm_scale,
        k_norm_scale,
        pos_1d,
        num_q_heads=p.num_q_heads,
        num_kv_heads=p.num_kv_heads,
        head_dim=p.head_dim,
        eps=p.eps,
        rope_max_timescale=p.rope_max_timescale,
        rope_min_timescale=p.rope_min_timescale,
        query_pre_attn_scalar=p.query_pre_attn_scalar,
        act_qtype=p.act_qtype,
        act_scale=p.act_scale,
        weight_scale=p.weight_scale,
        block_m=p.block_m,
        heads_per_tile=p.heads_per_tile,
        quant_rows=p.quant_rows,
        seq_len=seq_len,
        q_seq_minor=p.q_seq_minor,
        k_seq_minor=p.k_seq_minor,
        q_out_dtype=p.q_out_dtype,
        k_out_dtype=p.k_out_dtype,
        save_pre_norm=True,
        x_is_quantized=False,
        compute_dtype=p.compute_dtype,
        interpret=p.interpret,
    )
    q_bshd = jnp.transpose(q_raw, (0, 3, 1, 2) if p.q_seq_minor else (0, 2, 1, 3))
    k_bshd = jnp.transpose(k_raw, (0, 3, 1, 2) if p.k_seq_minor else (0, 2, 1, 3))
    v_bshd = jnp.transpose(v_raw, (0, 2, 1, 3))
    x_dtype_tag = jnp.zeros((0,), dtype=x_bsd.dtype)
    residuals = (x_bsd_c, w_qkv, (hq, hk), q_norm_scale, k_norm_scale, pos_2d, x_dtype_tag, wq_local, wk_local, wv_local)
    return (q_bshd, k_bshd, v_bshd), residuals

  if p.act_qtype is not None:
    x_q = static_quantize(x_2d, p.act_qtype, p.act_scale)
    proj = (jnp.dot(x_q, w_qkv, preferred_element_type=_F32) * (p.act_scale * p.weight_scale)).astype(p.compute_dtype)
  else:
    x_q = x_2d
    proj = jnp.dot(x_2d, w_qkv, preferred_element_type=_F32, precision=lax.Precision.HIGHEST).astype(p.compute_dtype)
  q_bshd, k_bshd, v_bshd = _post_proj_bshd(p, proj, q_norm_scale, k_norm_scale, pos_1d, batch_local, seq_len)
  x_saved_bsd = x_q.reshape(batch_local, seq_len, emb)
  proj_bsd = proj.reshape(batch_local, seq_len, -1)
  x_dtype_tag = jnp.zeros((0,), dtype=x_bsd.dtype)
  residuals = (
      x_saved_bsd,
      w_qkv,
      proj_bsd,
      q_norm_scale,
      k_norm_scale,
      pos_2d,
      x_dtype_tag,
      wq_local,
      wk_local,
      wv_local,
  )
  return (q_bshd, k_bshd, v_bshd), residuals


def _fused_qkv_local_bwd(p: FusedQkvTrainParams, residuals, cts):
  """Backward of the per-shard fused QKV projection (custom-VJP bwd rule)."""
  x_saved_bsd, w_qkv, saved_proj, q_norm_scale, k_norm_scale, pos_2d, x_dtype_tag, wq_local, wk_local, wv_local = (
      residuals
  )
  batch_local, seq_len, emb = x_saved_bsd.shape
  n_tokens = batch_local * seq_len
  x_saved = x_saved_bsd.reshape(n_tokens, emb)
  pos_1d = pos_2d.reshape(n_tokens)
  out_dtype = jnp.dtype(p.compute_dtype)
  dq, dk, dv = cts
  if _use_pallas_vjp(p, seq_len):
    x_q = static_quantize(x_saved, p.act_qtype, p.act_scale) if p.act_qtype is not None else x_saved
    hq, hk = saved_proj
    fuse_static_quant = p.act_qtype is not None and p.bwd_qtype is not None and p.bwd_scale is not None
    # Splash bwd emits dq/dk/dv in HEAD_DIM_MINOR `[B, H, S, D]` before attention_op's
    # `(0, 2, 1, 3)` transpose to `[B, S, H, D]`; transposing `(0, 2, 1, 3)` back is a
    # zero-copy identity composition.
    dproj_or_q, dq_norm, dk_norm, dproj_amax = fused_qkv_bwd_post(
        hq,
        jnp.transpose(dq, (0, 2, 1, 3)),
        hk,
        jnp.transpose(dk, (0, 2, 1, 3)),
        jnp.transpose(dv, (0, 2, 1, 3)),
        q_norm_scale,
        k_norm_scale,
        pos_1d,
        eps=p.eps,
        rope_max_timescale=p.rope_max_timescale,
        rope_min_timescale=p.rope_min_timescale,
        query_pre_attn_scalar=p.query_pre_attn_scalar,
        q_out_dtype=p.q_out_dtype,
        out_qtype=p.bwd_qtype if fuse_static_quant else None,
        out_scale=p.bwd_scale if fuse_static_quant else None,
        block_m=min(1024, p.block_m),
        interpret=p.interpret,
    )
    if p.act_qtype is not None and p.bwd_qtype is not None:
      if fuse_static_quant:
        assert p.bwd_scale is not None
        s_dy: float | jax.Array = p.bwd_scale
        dproj_q = dproj_or_q
      else:
        dproj_q, s_dy = _pow2_absmax_quantize(dproj_or_q, p.bwd_qtype, precomputed_amax=dproj_amax)
      dw_qkv = (jnp.dot(x_q.T, dproj_q, preferred_element_type=_F32) * (p.act_scale * s_dy)).astype(p.compute_dtype)
      dw_local = _reduce_scatter_rows(dw_qkv, p.weight_gather_axes)
      scale_dx = jnp.asarray(s_dy * p.weight_scale, _F32)
      if p.weight_gather_axes:
        scale_dx = _guard_with_tensor(scale_dx, dw_qkv)
      dx_2d = (jnp.dot(dproj_q, w_qkv.T, preferred_element_type=_F32) * scale_dx).astype(p.compute_dtype)
    else:
      dw_qkv = jnp.dot(x_q.T, dproj_or_q, preferred_element_type=_F32, precision=lax.Precision.HIGHEST)
      if p.act_qtype is not None:
        dw_qkv = dw_qkv * p.act_scale
      dw_local = _reduce_scatter_rows(dw_qkv, p.weight_gather_axes)
      dx_2d = jnp.dot(dproj_or_q, w_qkv.T, preferred_element_type=_F32, precision=lax.Precision.HIGHEST)
      if p.act_qtype is not None:
        dx_2d = dx_2d * p.weight_scale
      dx_2d = dx_2d.astype(p.compute_dtype)
      if p.weight_gather_axes:
        dx_2d = _guard_with_tensor(dx_2d, dw_qkv)
  else:
    x_q = x_saved
    proj_2d = saved_proj.reshape(n_tokens, -1)
    # Tokamax Splash's custom_vjp returns bfloat16 dq/dk even when q/k inputs are fp8_e4m3fn;
    # differentiate pre-fp8-cast outputs in out_dtype so dq/dk are never rounded through fp8.
    _, post_vjp = jax.vjp(
        lambda pr, qs, ks: _post_proj_bshd(p, pr, qs, ks, pos_1d, batch_local, seq_len, cast_qk_out=False),
        proj_2d,
        q_norm_scale,
        k_norm_scale,
    )
    dproj, dq_norm, dk_norm = post_vjp((dq.astype(out_dtype), dk.astype(out_dtype), dv.astype(out_dtype)))
    if p.act_qtype is not None and p.bwd_qtype is not None:
      if p.bwd_scale is not None:
        s_dy = p.bwd_scale
        dproj_q = static_quantize(dproj, p.bwd_qtype, p.bwd_scale)
      else:
        dproj_q, s_dy = _pow2_absmax_quantize(dproj, p.bwd_qtype)
      dw_qkv = (jnp.dot(x_q.T, dproj_q, preferred_element_type=_F32) * (p.act_scale * s_dy)).astype(p.compute_dtype)
      dw_local = _reduce_scatter_rows(dw_qkv, p.weight_gather_axes)
      scale_dx = jnp.asarray(s_dy * p.weight_scale, _F32)
      if p.weight_gather_axes:
        scale_dx = _guard_with_tensor(scale_dx, dw_qkv)
      dx_2d = (jnp.dot(dproj_q, w_qkv.T, preferred_element_type=_F32) * scale_dx).astype(p.compute_dtype)
    else:
      dw_qkv = jnp.dot(x_q.T, dproj, preferred_element_type=_F32, precision=lax.Precision.HIGHEST)
      if p.act_qtype is not None:
        dw_qkv = dw_qkv * p.act_scale
      dw_local = _reduce_scatter_rows(dw_qkv, p.weight_gather_axes)
      dx_2d = jnp.dot(dproj, w_qkv.T, preferred_element_type=_F32, precision=lax.Precision.HIGHEST)
      if p.act_qtype is not None:
        dx_2d = dx_2d * p.weight_scale
      dx_2d = dx_2d.astype(p.compute_dtype)
      if p.weight_gather_axes:
        dx_2d = _guard_with_tensor(dx_2d, dw_qkv)

  if p.extra_w_psum_axes:
    dw_local = lax.psum(dw_local, p.extra_w_psum_axes)
  q_cols = p.num_q_heads * p.head_dim
  k_cols = p.num_kv_heads * p.head_dim
  dwq = dw_local[:, :q_cols].reshape(wq_local.shape).astype(wq_local.dtype)
  dwk = dw_local[:, q_cols : q_cols + k_cols].reshape(wk_local.shape).astype(wk_local.dtype)
  dwv = dw_local[:, q_cols + k_cols :].reshape(wv_local.shape).astype(wv_local.dtype)
  if p.norm_psum_axes and not p.defer_norm_grad_psum:
    dq_norm = lax.psum(dq_norm, p.norm_psum_axes)
    dk_norm = lax.psum(dk_norm, p.norm_psum_axes)
  target_x_dtype = p.x_dtype if x_dtype_tag is None else x_dtype_tag.dtype
  return (
      dx_2d.reshape(batch_local, seq_len, emb).astype(target_x_dtype),
      dwq,
      dwk,
      dwv,
      dq_norm.astype(q_norm_scale.dtype),
      dk_norm.astype(k_norm_scale.dtype),
      None,
  )


fused_qkv_local.defvjp(_fused_qkv_local_fwd, _fused_qkv_local_bwd, optimize_remat=True)


@dataclasses.dataclass(frozen=True)
class FusedQkvShardedConfig:
  """Static sharding + kernel config for `fused_qkv_sharded` (custom_vjp outside shard_map)."""

  params: FusedQkvTrainParams
  mesh: Any
  x_pspec: jax.sharding.PartitionSpec
  wq_pspec: jax.sharding.PartitionSpec
  wk_pspec: jax.sharding.PartitionSpec
  wv_pspec: jax.sharding.PartitionSpec
  q_norm_pspec: jax.sharding.PartitionSpec
  k_norm_pspec: jax.sharding.PartitionSpec
  pos_pspec: jax.sharding.PartitionSpec
  q_pspec: jax.sharding.PartitionSpec
  k_pspec: jax.sharding.PartitionSpec
  v_pspec: jax.sharding.PartitionSpec


def _fused_qkv_res_specs(cfg: FusedQkvShardedConfig, local_seq_len: int):
  p_none2 = jax.sharding.PartitionSpec(None, None)
  if _use_pallas_vjp(cfg.params, local_seq_len):
    p_bhsd = jax.sharding.PartitionSpec(cfg.x_pspec[0], None, cfg.x_pspec[1], None)
    saved_proj_spec: Any = (p_bhsd, p_bhsd)
  else:
    saved_proj_spec = cfg.x_pspec
  return (
      cfg.x_pspec,
      p_none2,
      saved_proj_spec,
      cfg.q_norm_pspec,
      cfg.k_norm_pspec,
      cfg.pos_pspec,
      cfg.wq_pspec,
      cfg.wk_pspec,
      cfg.wv_pspec,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def fused_qkv_sharded(
    cfg: FusedQkvShardedConfig,
    x_bsd: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    pos_bs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sharded `fused_qkv_local` with `custom_vjp` outside `shard_map` so `dq_norm`/`dk_norm` skip per-layer `psum`.

  With `cfg.params.defer_norm_grad_psum=True` the backward returns `dq_norm` / `dk_norm` summed over the local
  batch/seq shard only, through replicated out_specs with `check_vma=False`. The caller must then wrap both norm
  scales with `normalizations.post_scan_psum_grad` over the same batch/seq axes outside the layer scan (as
  `NNXDecoder._apply_layers_sequentially` does); without it the QK-norm scale gradients stay per-shard.
  """
  in_specs = (
      cfg.x_pspec,
      cfg.wq_pspec,
      cfg.wk_pspec,
      cfg.wv_pspec,
      cfg.q_norm_pspec,
      cfg.k_norm_pspec,
      cfg.pos_pspec,
  )
  out_specs = (cfg.q_pspec, cfg.k_pspec, cfg.v_pspec)

  @functools.partial(jax.shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
  def _primal(x_l, wq_l, wk_l, wv_l, qn_l, kn_l, pos_l):
    return fused_qkv_local(cfg.params, x_l, wq_l, wk_l, wv_l, qn_l, kn_l, pos_l)

  return _primal(x_bsd, wq, wk, wv, q_norm_scale, k_norm_scale, pos_bs)


def _fused_qkv_sharded_fwd(
    cfg: FusedQkvShardedConfig,
    x_bsd: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    pos_bs: jax.Array,
):
  """Forward of the sharded fused QKV projection under `shard_map`; returns outputs and residuals."""
  seq_ax = cfg.x_pspec[1]
  if seq_ax is None:
    seq_ways = 1
  elif isinstance(seq_ax, str):
    seq_ways = cfg.mesh.shape.get(seq_ax, 1)
  else:
    seq_ways = math.prod(cfg.mesh.shape.get(ax, 1) for ax in seq_ax)
  local_seq_len = x_bsd.shape[1] // seq_ways
  in_specs = (
      cfg.x_pspec,
      cfg.wq_pspec,
      cfg.wk_pspec,
      cfg.wv_pspec,
      cfg.q_norm_pspec,
      cfg.k_norm_pspec,
      cfg.pos_pspec,
  )
  out_specs = (cfg.q_pspec, cfg.k_pspec, cfg.v_pspec)
  res_specs = _fused_qkv_res_specs(cfg, local_seq_len)

  @functools.partial(jax.shard_map, mesh=cfg.mesh, in_specs=in_specs, out_specs=(out_specs, res_specs), check_vma=False)
  def _fwd(x_l, wq_l, wk_l, wv_l, qn_l, kn_l, pos_l):
    qkv_out, res = _fused_qkv_local_fwd(cfg.params, x_l, wq_l, wk_l, wv_l, qn_l, kn_l, pos_l)
    x_s, w_qkv, s_proj, qn_s, kn_s, p_2d, _, wq_s, wk_s, wv_s = res
    return qkv_out, (x_s, w_qkv, s_proj, qn_s, kn_s, p_2d, wq_s, wk_s, wv_s)

  return _fwd(x_bsd, wq, wk, wv, q_norm_scale, k_norm_scale, pos_bs)


def _fused_qkv_sharded_bwd(cfg: FusedQkvShardedConfig, residuals_no_dtype, cts):
  """Backward of the sharded fused QKV projection (custom-VJP bwd rule)."""
  x_s = residuals_no_dtype[0]
  seq_ax = cfg.x_pspec[1]
  if seq_ax is None:
    seq_ways = 1
  elif isinstance(seq_ax, str):
    seq_ways = cfg.mesh.shape.get(seq_ax, 1)
  else:
    seq_ways = math.prod(cfg.mesh.shape.get(ax, 1) for ax in seq_ax)
  local_seq_len = x_s.shape[1] // seq_ways
  res_specs = _fused_qkv_res_specs(cfg, local_seq_len)
  cts_specs = (cfg.q_pspec, cfg.k_pspec, cfg.v_pspec)
  grad_specs = (
      cfg.x_pspec,
      cfg.wq_pspec,
      cfg.wk_pspec,
      cfg.wv_pspec,
      cfg.q_norm_pspec,
      cfg.k_norm_pspec,
  )

  bwd_params = cfg.params
  if not bwd_params.defer_norm_grad_psum and not bwd_params.norm_psum_axes:
    mentioned = set()
    for sp in (cfg.q_norm_pspec, cfg.k_norm_pspec):
      for entry in sp:
        if isinstance(entry, str):
          mentioned.add(entry)
        elif entry is not None:
          mentioned.update(entry)
    x_bs_axes: list[str] = []
    for entry in cfg.x_pspec[:2]:
      if isinstance(entry, str):
        x_bs_axes.append(entry)
      elif entry is not None:
        x_bs_axes.extend(entry)
    auto_norm_axes = tuple(dict.fromkeys(ax for ax in x_bs_axes if cfg.mesh.shape.get(ax, 1) > 1 and ax not in mentioned))
    if auto_norm_axes:
      bwd_params = dataclasses.replace(bwd_params, norm_psum_axes=auto_norm_axes)

  @functools.partial(
      jax.shard_map,
      mesh=cfg.mesh,
      in_specs=(res_specs, cts_specs),
      out_specs=grad_specs,
      check_vma=False,
  )
  def _bwd(res_t, cts_t):
    x_sd, w_qkv, s_proj, qn_s, kn_s, p_2d, wq_s, wk_s, wv_s = res_t
    full_res = (x_sd, w_qkv, s_proj, qn_s, kn_s, p_2d, None, wq_s, wk_s, wv_s)
    dx, dwq, dwk, dwv, dqn, dkn, _ = _fused_qkv_local_bwd(bwd_params, full_res, cts_t)
    return dx, dwq, dwk, dwv, dqn, dkn

  dx, dwq, dwk, dwv, dqn, dkn = _bwd(residuals_no_dtype, cts)
  return dx, dwq, dwk, dwv, dqn, dkn, None


fused_qkv_sharded.defvjp(_fused_qkv_sharded_fwd, _fused_qkv_sharded_bwd, optimize_remat=True)


@dataclasses.dataclass(frozen=True)
class FusedOutTrainParams:
  """Static parameters of `fused_out_local` (differentiable FSDP Attention `out` projection)."""

  compute_dtype: Any = jnp.bfloat16
  act_qtype: Any = jnp.float8_e4m3fn
  weight_qtype: Any = None
  bwd_qtype: Any = jnp.float8_e5m2
  act_scale: float = 0.5
  weight_scale: float = 0.5
  bwd_scale: float | None = None
  weight_gather_axes: tuple[str, ...] = ()
  extra_w_psum_axes: tuple[str, ...] = ()
  ctx_dtype: Any = jnp.bfloat16
  wo_dtype: Any = jnp.float32


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def fused_out_local(
    p: FusedOutTrainParams,
    ctx_bshd: jax.Array,
    wo_local: jax.Array,
) -> jax.Array:
  """Per-shard 4D/3D FSDP Attention `out` projection with local absmax bwd & zero extra VJP residuals."""
  y_bsd, _ = _fused_out_local_fwd(p, ctx_bshd, wo_local)
  return y_bsd


def _fused_out_local_fwd(
    p: FusedOutTrainParams,
    ctx_bshd: jax.Array,
    wo_local: jax.Array,
):
  """Forward of the per-shard fused output projection; returns the output and residuals."""
  ctx_c = ctx_bshd.astype(p.compute_dtype)
  wo_c = wo_local.astype(p.compute_dtype)
  if p.act_qtype is not None:
    w_qtype = p.act_qtype if p.weight_qtype is None else p.weight_qtype
    ctx_q = static_quantize(ctx_c, p.act_qtype, p.act_scale)
    wo_l_q = static_quantize(wo_c, w_qtype, p.weight_scale)
    deq = jnp.asarray(p.act_scale * p.weight_scale, p.compute_dtype)
    wo_q = _all_gather_axis(wo_l_q, p.weight_gather_axes, axis=2)
    y_bsd = (
        lax.dot_general(ctx_q, wo_q, (((2, 3), (0, 1)), ((), ())), preferred_element_type=p.compute_dtype) * deq
    ).astype(p.compute_dtype)
  else:
    ctx_q = ctx_c
    wo_q = _all_gather_axis(wo_c, p.weight_gather_axes, axis=2)
    y_bsd = lax.dot_general(
        ctx_q, wo_q, (((2, 3), (0, 1)), ((), ())), preferred_element_type=_F32, precision=lax.Precision.HIGHEST
    ).astype(p.compute_dtype)
  return y_bsd, (ctx_q, wo_q)


def _fused_out_local_bwd(p: FusedOutTrainParams, residuals, dy_bsd):
  """Backward of the per-shard fused output projection (custom-VJP bwd rule)."""
  ctx_q, wo_q = residuals
  dy_c = dy_bsd.astype(p.compute_dtype)
  if p.act_qtype is not None and p.bwd_qtype is not None:
    if p.bwd_scale is not None:
      s_dy = jnp.asarray(p.bwd_scale, p.compute_dtype)
      dy_q = static_quantize(dy_c, p.bwd_qtype, p.bwd_scale)
    else:
      fmax = float(jnp.finfo(p.bwd_qtype).max)
      amax = jnp.max(jnp.abs(dy_c))
      min_amax = jnp.asarray(float(jnp.finfo(p.compute_dtype).tiny) * fmax * 2.0, p.compute_dtype)
      s_dy = jnp.where(amax < min_amax, jnp.ones_like(amax), amax * jnp.asarray(1.0 / fmax, p.compute_dtype))
      dy_q = jnp.clip(dy_c / s_dy, -fmax, fmax).astype(p.bwd_qtype)
    scale_dw = jnp.asarray(p.act_scale, p.compute_dtype) * s_dy
    scale_dctx = s_dy * jnp.asarray(p.weight_scale, p.compute_dtype)
    dw_full = (
        lax.dot_general(ctx_q, dy_q, (((0, 1), (0, 1)), ((), ())), preferred_element_type=p.compute_dtype) * scale_dw
    ).astype(p.compute_dtype)
    dctx = (
        lax.dot_general(dy_q, wo_q, (((2,), (2,)), ((), ())), preferred_element_type=p.compute_dtype) * scale_dctx
    ).astype(p.ctx_dtype)
    dw_local = _reduce_scatter_axis(dw_full, p.weight_gather_axes, axis=2)
  else:
    dctx = lax.dot_general(
        dy_c, wo_q, (((2,), (2,)), ((), ())), preferred_element_type=_F32, precision=lax.Precision.HIGHEST
    )
    if p.act_qtype is not None:
      dctx = dctx * p.weight_scale
    dctx = dctx.astype(p.ctx_dtype)
    dw_full = lax.dot_general(
        ctx_q, dy_c, (((0, 1), (0, 1)), ((), ())), preferred_element_type=_F32, precision=lax.Precision.HIGHEST
    )
    if p.act_qtype is not None:
      dw_full = dw_full * p.act_scale
    dw_full = dw_full.astype(p.compute_dtype)
    dw_local = _reduce_scatter_axis(dw_full, p.weight_gather_axes, axis=2)

  if p.extra_w_psum_axes:
    dw_local = lax.psum(dw_local, p.extra_w_psum_axes)
  dwo = dw_local.astype(p.wo_dtype)
  return dctx, dwo


fused_out_local.defvjp(_fused_out_local_fwd, _fused_out_local_bwd, optimize_remat=True)


def fused_qkv_proj_maxtext_literal(
    x: jax.Array,
    w_qkv: jax.Array,
    q_norm_scale: jax.Array,
    k_norm_scale: jax.Array,
    positions: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    rope_max_timescale: float,
    rope_min_timescale: float = 1.0,
    query_pre_attn_scalar: float | None = None,
    act_qtype: Any = jnp.float8_e4m3fn,
    act_scale: float = 0.5,
    weight_scale: float = 0.5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """MaxText's `Attention` projection path, op by op, in the compute dtype.

  Unlike `fused_qkv_proj_reference` this rounds to the compute dtype after
  every op exactly as the un-fused JAX code does (how XLA:CPU evaluates it);
  XLA:TPU fuses these ops and may keep f32 in between, so the kernel is
  expected to match this only up to compute-dtype rounding noise.

  Args:
    x: `[N, emb]` input.
    w_qkv: `[emb, (Hq + 2 Hkv) * D]` fused weights.
    q_norm_scale: `[D]` query RMSNorm scale.
    k_norm_scale: `[D]` key RMSNorm scale.
    positions: `[N]` positions.
    num_q_heads: Hq.
    num_kv_heads: Hkv.
    head_dim: D.
    eps: RMSNorm epsilon.
    rope_max_timescale: RoPE max timescale.
    rope_min_timescale: RoPE min timescale.
    query_pre_attn_scalar: Query scale; None means `head_dim ** -0.5`.
    act_qtype: fp8 dtype of the activation quantization, or None.
    act_scale: Static activation quantization scale.
    weight_scale: Static weight quantization scale.

  Returns:
    q, k, v in the `[N, H, D]` layout of `Attention.__call__`.
  """
  n_tokens = x.shape[0]
  out_dtype = x.dtype
  if query_pre_attn_scalar is None:
    query_pre_attn_scalar = head_dim**-0.5
  if act_qtype is not None:
    # `dot_general_qt_fwd` + `_fast_dot_general` with bf16 scales.
    scale = jnp.asarray(act_scale, out_dtype)
    w_scale = jnp.asarray(weight_scale, out_dtype)
    lhs = static_quantize(x, act_qtype, act_scale)
    proj = jnp.dot(lhs, w_qkv, preferred_element_type=out_dtype) * scale * w_scale
  else:
    proj = jnp.dot(x, w_qkv, precision=lax.Precision.HIGHEST)
  proj = proj.astype(out_dtype)
  q_cols = num_q_heads * head_dim
  k_cols = num_kv_heads * head_dim
  q = proj[:, :q_cols].reshape(n_tokens, num_q_heads, head_dim)
  k = proj[:, q_cols : q_cols + k_cols].reshape(n_tokens, num_kv_heads, head_dim)
  v = proj[:, q_cols + k_cols :].reshape(n_tokens, num_kv_heads, head_dim)

  def rms_norm(h: jax.Array, scale: jax.Array) -> jax.Array:
    """`normalizations.RMSNorm.__call__`."""
    h32 = jnp.asarray(h, _F32)
    mean2 = jnp.mean(lax.square(h32), axis=-1, keepdims=True)
    y = jnp.asarray(h32 * lax.rsqrt(mean2 + eps), out_dtype)
    return jnp.einsum("...k,k->...k", y, jnp.asarray(scale, _F32).astype(out_dtype))

  cos_half, sin_half = rope_tables(
      positions, head_dim, min_timescale=rope_min_timescale, max_timescale=rope_max_timescale, dtype=out_dtype
  )
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)[:, None, :]
  sin = jnp.concatenate([sin_half, sin_half], axis=-1)[:, None, :]

  def rope(h: jax.Array) -> jax.Array:
    """`embeddings.RotaryEmbedding.__call__`."""
    return ((h * cos) + (_rotate_half(h) * sin)).astype(out_dtype)

  q = rope(rms_norm(q, q_norm_scale))
  k = rope(rms_norm(k, k_norm_scale))
  q = q * query_pre_attn_scalar
  return q, k, v
