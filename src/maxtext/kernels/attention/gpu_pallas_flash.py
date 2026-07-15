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

"""Pallas (Triton) GPU flash attention with sliding-window, soft-cap and
large-head-dim support.

Forked from jax.experimental.pallas.ops.gpu.attention with extensions needed
to run Gemma-style models efficiently on Ampere (sm80):

  1. ``window``: causal sliding-window attention (query at position i attends
     keys j with ``i - window < j <= i``, matching MaxText's LOCAL_SLIDING
     mask), implemented as FlashAttention-2 style loop-bound narrowing plus an
     in-block mask, so local layers do O(seq * window) work instead of O(seq^2).
  2. ``soft_cap``: gemma2-style attention-logit soft cap
     (``cap * tanh(logits / cap)``), applied in the natural-log domain with the
     matching ``1 - tanh^2`` chain rule in the backward pass.
  3. head_dim > 128 (256, 512): cuDNN fused attention has no sm80 kernel and
     FlashAttention-2 caps at 256. QK^T / PV products are accumulated over
     ``dim_chunk``-sized slices of head_dim so SMEM tile sizes stay bounded
     regardless of head_dim.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp

from maxtext.common.common_types import DEFAULT_MASK_VALUE

_LOG2E = math.log2(math.e)


def _scaled_logits(qk, sm_scale: float, soft_cap: float | None):
  """Applies sm_scale and optional tanh soft-cap (both in the natural-log
  domain), then converts to base-2 domain for exp2-based softmax.

  Returns (base2 logits, tanh value or None) — the tanh is reused by the
  backward pass, whose chain rule needs ``1 - tanh^2``.
  """
  if sm_scale != 1.0:
    qk = qk * sm_scale
  t = None
  if soft_cap is not None:
    t = jnp.tanh(qk / soft_cap)
    qk = soft_cap * t
  return qk * _LOG2E, t


def _pick_dim_chunk(head_dim_padded: int) -> int:
  """head_dim <= 256 keeps the original single-slab layout (tuned); larger
  head dims accumulate over 128-wide chunks to bound SMEM tile sizes."""
  return head_dim_padded if head_dim_padded <= 256 else 128


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """Tile sizes parameterizing the attention kernel."""

  block_q: int
  block_k: int
  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  @classmethod
  def get_default(cls):
    return BlockSizes(
        block_q=128,
        block_k=128,
        block_q_dkv=32,
        block_kv_dkv=32,
        block_q_dq=32,
        block_kv_dq=32,
    )

  @classmethod
  def get_for_head_dim(cls, head_dim: int):
    """Autotuned on A100 (sm80, 164KB SMEM)."""
    if head_dim <= 128:
      return cls.get_default()
    if head_dim <= 256:
      return BlockSizes(
          block_q=64,
          block_k=64,
          block_q_dkv=32,
          block_kv_dkv=32,
          block_q_dq=32,
          block_kv_dq=32,
      )
    # d > 256: chunked accumulation bounds SMEM; smaller q blocks bound the
    # fp32 accumulator register footprint (block_q x head_dim).
    return BlockSizes(
        block_q=32,
        block_k=64,
        block_q_dkv=32,
        block_kv_dkv=32,
        block_q_dq=32,
        block_kv_dq=32,
    )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (self.block_q_dkv, self.block_kv_dkv, self.block_q_dq, self.block_kv_dq)
    return all(b is not None for b in backward_blocks)


def segment_mask(q_segment_ids: jax.Array, kv_segment_ids: jax.Array):
  q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
  if kv_segment_ids.ndim == 1:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
  else:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
  return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


def _combined_mask(span_q, span_k, q_segment_ids, kv_segment_ids, causal, window):
  """Builds the block-local mask combining segments, causality and the sliding window."""
  mask = None
  if q_segment_ids is not None:
    mask = segment_mask(q_segment_ids, kv_segment_ids)
  if causal:
    causal_mask = span_q[:, None] >= span_k[None, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  if window is not None:
    # MaxText LOCAL_SLIDING convention: k > q - window (jointly with k <= q).
    window_mask = span_q[:, None] - span_k[None, :] < window
    mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)
  return mask


def _chunk_slices(head_dim_padded: int, dim_chunk: int):
  assert head_dim_padded % dim_chunk == 0
  return [pl.dslice(c * dim_chunk, dim_chunk) for c in range(head_dim_padded // dim_chunk)]


def _chunk_masks(head_dim: int, head_dim_padded: int, dim_chunk: int):
  return [(jnp.arange(dim_chunk) + c * dim_chunk < head_dim)[None, :] for c in range(head_dim_padded // dim_chunk)]


def _load_chunks(ref, row_slice, chunk_slices, chunk_masks):
  return [plgpu.load(ref.at[row_slice, cs], mask=cm, other=0.0) for cs, cm in zip(chunk_slices, chunk_masks)]


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    o_ref: Any,
    *residual_refs: Any,
    sm_scale: float,
    causal: bool,
    window: int | None,
    soft_cap: float | None,
    block_q: int,
    block_k: int,
    head_dim: int,
    dim_chunk: int,
):
  """Fused forward kernel: online-softmax attention over K/V blocks."""
  seq_len = k_ref.shape[0]
  start_q = pl.program_id(0)
  head_dim_padded = q_ref.shape[-1]
  cslices = _chunk_slices(head_dim_padded, dim_chunk)
  cmasks = _chunk_masks(head_dim, head_dim_padded, dim_chunk)
  num_chunks = len(cslices)

  m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  o_chunks = tuple(jnp.zeros((block_q, dim_chunk), dtype=jnp.float32) for _ in range(num_chunks))

  curr_q_slice = pl.dslice(start_q * block_q, block_q)
  q_chunks = _load_chunks(q_ref, slice(None), cslices, cmasks)
  q_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]

  def body(start_k, carry):
    o_prev, m_prev, l_prev = carry
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    qk = jnp.zeros((block_q, block_k), dtype=jnp.float32)
    for c in range(num_chunks):
      k_c = plgpu.load(k_ref.at[curr_k_slice, cslices[c]], mask=cmasks[c], other=0.0)
      qk += plgpu.dot(q_chunks[c], k_c.T)
    qk, _ = _scaled_logits(qk, sm_scale, soft_cap)

    if causal or window is not None or segment_ids_ref is not None:
      span_q = start_q * block_q + jnp.arange(block_q)
      span_k = start_k * block_k + jnp.arange(block_k)
      kv_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_k_slice]
      mask = _combined_mask(span_q, span_k, q_segment_ids, kv_segment_ids, causal, window)
      assert mask is not None
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp2(m_prev - m_next)
    l_prev_corr = correction * l_prev
    s_curr = jnp.exp2(qk - m_next[:, None])
    l_curr = s_curr.sum(axis=-1)
    l_next = l_prev_corr + l_curr

    o_next = []
    for c in range(num_chunks):
      v_c = plgpu.load(v_ref.at[curr_k_slice, cslices[c]], mask=cmasks[c], other=0.0)
      o_next.append(correction[:, None] * o_prev[c] + plgpu.dot(s_curr.astype(v_c.dtype), v_c))
    return tuple(o_next), m_next, l_next

  if causal:
    upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
  else:
    upper_bound = pl.cdiv(seq_len, block_k)
  if window is not None:
    # First key any query in this block can attend: q_min - window + 1.
    first_k = block_q * start_q - window + 1
    lower_bound = lax.max(0, lax.div(first_k, block_k))
  else:
    lower_bound = 0
  o_chunks, m_i, l_i = lax.fori_loop(lower_bound, upper_bound, body, (o_chunks, m_i, l_i))

  # Fully-masked rows (can happen with packing) produce l_i == 0; avoid NaNs.
  l_safe = jnp.where(l_i == 0.0, 1.0, l_i)[:, None]

  if residual_refs:
    lse_ref = residual_refs[0]
    lse_ref[...] = m_i + jnp.log2(jnp.where(l_i == 0.0, 1.0, l_i))
  for cslice, cmask, o_chunk in zip(cslices, cmasks, o_chunks):
    plgpu.store(o_ref.at[:, cslice], (o_chunk / l_safe).astype(o_ref.dtype), mask=cmask)


@functools.partial(jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "window",
        "soft_cap",
        "block_sizes",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "return_residuals",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    window: int | None = None,
    soft_cap: float | None = None,
    block_sizes: BlockSizes | None = None,
    backward_pass_impl: str = "auto",
    num_warps: int | None = None,
    num_stages: int | None = None,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
):
  """Flash attention over [batch, seq, heads, head_dim] inputs (custom VJP)."""
  del backward_pass_impl
  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len = k.shape[1]
  if block_sizes is None:
    block_sizes = BlockSizes.get_for_head_dim(head_dim)
  block_q = min(block_sizes.block_q, q_seq_len)
  block_k = min(block_sizes.block_k, kv_seq_len)
  head_dim_padded = pl.next_power_of_2(head_dim)
  dim_chunk = _pick_dim_chunk(head_dim_padded)
  if (q.shape[-1] != k.shape[-1]) or (q.shape[-1] != v.shape[-1]):
    raise ValueError(
        f"This kernel expects q, k, and v to have the same head dimension, but" f" found {q.shape=}, {k.shape=}, {v.shape=}."
    )
  if q_seq_len % block_q != 0:
    raise ValueError(f"{q_seq_len=} must be a multiple of {block_q=}")
  if kv_seq_len % block_k != 0:
    raise ValueError(f"{kv_seq_len=} must be a multiple of {block_k=}")
  if window is not None and window <= 0:
    raise ValueError(f"{window=} must be positive when set")

  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    # Autotuned on A100: 4 warps wins at head_dim <= 64 and at 256 (register
    # pressure from the wide fp32 accumulator); 8 warps wins at 128 and 512.
    num_warps_ = 4 if head_dim <= 64 or head_dim_padded == 256 else 8
  num_stages_ = num_stages
  if num_stages_ is None:
    # Chunked loads bound tile sizes, but at d512 double-buffering still sits
    # ~1KB above sm80's 164KB SMEM budget — drop to single-stage there.
    num_stages_ = 1 if head_dim_padded > 256 else 2

  kernel = functools.partial(
      mha_forward_kernel,
      sm_scale=sm_scale,
      block_q=block_q,
      block_k=block_k,
      head_dim=head_dim,
      dim_chunk=dim_chunk,
      causal=causal,
      window=window,
      soft_cap=soft_cap,
  )

  in_specs: list[pl.BlockSpec | None] = [
      pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0)),
      pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda _, j, k: (j, 0, k, 0)),
      pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda _, j, k: (j, 0, k, 0)),
  ]
  in_specs.append(None if segment_ids is None else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0)))
  out_shape = [q]
  out_specs = [pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0))]
  if return_residuals:
    out_shape.append(jax.ShapeDtypeStruct(shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32))
    out_specs.append(pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)))
  out = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=out_specs,
      compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=num_stages_),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward_sw",
  )(q, k, v, segment_ids)
  return out if return_residuals else out[0]


def _mha_forward(
    q,
    k,
    v,
    segment_ids: jax.Array | None,
    sm_scale: float,
    causal: bool,
    window: int | None,
    soft_cap: float | None,
    block_sizes: BlockSizes | None,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int | None,
    grid: Any,
    interpret: bool,
    debug: bool,
    return_residuals: bool,
):
  """Forward-pass wrapper: pads head_dim, launches the kernel, keeps residuals."""
  out, lse = mha(
      q,
      k,
      v,
      segment_ids=segment_ids,
      sm_scale=sm_scale,
      causal=causal,
      window=window,
      soft_cap=soft_cap,
      block_sizes=block_sizes,
      backward_pass_impl=backward_pass_impl,
      num_warps=num_warps,
      num_stages=num_stages,
      grid=grid,
      interpret=interpret,
      debug=debug,
      return_residuals=True,
  )
  residuals = (q, k, v, segment_ids, out, lse)
  ret = (out, lse) if return_residuals else out
  return ret, residuals


def _preprocess_backward_kernel(out_ref, dout_ref, delta_ref, head_dim: int, dim_chunk: int):
  head_dim_padded = out_ref.shape[-1]
  cslices = _chunk_slices(head_dim_padded, dim_chunk)
  cmasks = _chunk_masks(head_dim, head_dim_padded, dim_chunk)
  delta = jnp.zeros(out_ref.shape[0], dtype=jnp.float32)
  for cs, cm in zip(cslices, cmasks):
    o_c = plgpu.load(out_ref.at[:, cs], mask=cm, other=0.0)
    do_c = plgpu.load(dout_ref.at[:, cs], mask=cm, other=0.0)
    delta += jnp.sum(o_c * do_c, axis=1)
  delta_ref[...] = delta.astype(delta_ref.dtype)


@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, lse, block_q: int, debug: bool, interpret: bool):
  """Computes the per-row delta = sum(out * do) residual used by the backward kernels."""
  batch_size, seq_len, num_heads, head_dim = out.shape
  head_dim_padded = pl.next_power_of_2(head_dim)
  dim_chunk = _pick_dim_chunk(head_dim_padded)
  out_shape = jax.ShapeDtypeStruct.like(lse)
  delta = pl.pallas_call(
      functools.partial(_preprocess_backward_kernel, head_dim=head_dim, dim_chunk=dim_chunk),
      grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
      in_specs=[
          pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0)),
          pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0)),
      ],
      out_specs=pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_preprocess_backward",
  )(out, do)
  return delta


def mha_backward_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    window: int | None,
    soft_cap: float | None,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    head_dim: int,
    dim_chunk: int,
):
  """Fused backward kernel computing dQ, dK and dV from forward residuals."""
  del out_ref
  q_seq_len = q_ref.shape[0]
  kv_seq_len = k_ref.shape[0]
  head_dim_padded = q_ref.shape[-1]
  cslices = _chunk_slices(head_dim_padded, dim_chunk)
  cmasks = _chunk_masks(head_dim, head_dim_padded, dim_chunk)
  num_chunks = len(cslices)

  # Scan #1: dK and dV.
  start_k = pl.program_id(2)
  curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)

  dv_chunks = tuple(jnp.zeros([block_kv_dkv, dim_chunk], dtype=jnp.float32) for _ in range(num_chunks))
  dk_chunks = tuple(jnp.zeros([block_kv_dkv, dim_chunk], dtype=jnp.float32) for _ in range(num_chunks))

  k_chunks = _load_chunks(k_ref, curr_k_slice, cslices, cmasks)
  v_chunks = _load_chunks(v_ref, curr_k_slice, cslices, cmasks)
  span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
  kv_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_k_slice]

  def inner_loop_dkdv(start_q, carry):
    dv, dk = carry
    curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)

    q_chunks = _load_chunks(q_ref, curr_q_slice, cslices, cmasks)
    do_chunks = _load_chunks(do_scaled_ref, curr_q_slice, cslices, cmasks)

    qk = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32)
    for c in range(num_chunks):
      qk += plgpu.dot(q_chunks[c], k_chunks[c].T)
    qk, tanh_val = _scaled_logits(qk, sm_scale, soft_cap)

    if causal or window is not None or segment_ids_ref is not None:
      span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
      q_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]
      mask = _combined_mask(span_q, span_k, q_segment_ids, kv_segment_ids, causal, window)
      assert mask is not None
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    lse = lse_ref[curr_q_slice]
    di = delta_ref[curr_q_slice]

    p = jnp.exp2(qk - lse[:, None])
    dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
    for c in range(num_chunks):
      dp += plgpu.dot(do_chunks[c], v_chunks[c].T)
    ds = p * dp
    if soft_cap is not None:
      ds = ds * (1.0 - tanh_val * tanh_val)
    if sm_scale != 1.0:
      ds = ds * sm_scale

    p_t = p.astype(do_chunks[0].dtype).T
    ds_t = ds.astype(q_ref.dtype).T
    dv = tuple(dv[c] + plgpu.dot(p_t, do_chunks[c]) for c in range(num_chunks))
    dk = tuple(dk[c] + plgpu.dot(ds_t, q_chunks[c]) for c in range(num_chunks))
    return dv, dk

  lower_bound_dkdv = lax.div(start_k * block_kv_dkv, block_q_dkv) if causal else 0
  if window is not None:
    # Last query that can attend key k_min is k_min + window - 1; queries in
    # later blocks see none of this KV block.
    last_q = (start_k + 1) * block_kv_dkv + window - 1
    upper_bound_dkdv = lax.min(pl.cdiv(q_seq_len, block_q_dkv), lax.div(last_q + block_q_dkv - 1, block_q_dkv))
  else:
    upper_bound_dkdv = pl.cdiv(q_seq_len, block_q_dkv)
  dv_chunks, dk_chunks = lax.fori_loop(lower_bound_dkdv, upper_bound_dkdv, inner_loop_dkdv, (dv_chunks, dk_chunks))
  for c in range(num_chunks):
    plgpu.store(dv_ref.at[:, cslices[c]], dv_chunks[c].astype(dv_ref.dtype), mask=cmasks[c])
    plgpu.store(dk_ref.at[:, cslices[c]], dk_chunks[c].astype(dk_ref.dtype), mask=cmasks[c])

  # Scan #2: dQ.
  start_q = pl.program_id(2)
  curr_q_slice = pl.ds(start_q * block_q_dq, block_q_dq)
  span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
  dq_chunks = tuple(jnp.zeros([block_q_dq, dim_chunk], dtype=jnp.float32) for _ in range(num_chunks))

  q_chunks = _load_chunks(q_ref, curr_q_slice, cslices, cmasks)
  do_chunks = _load_chunks(do_scaled_ref, curr_q_slice, cslices, cmasks)
  q_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_q_slice]
  lse = lse_ref[curr_q_slice]
  di = delta_ref[curr_q_slice]

  def inner_loop_dq(start_k, dq):
    curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
    k_chunks_dq = _load_chunks(k_ref, curr_k_slice, cslices, cmasks)
    v_chunks_dq = _load_chunks(v_ref, curr_k_slice, cslices, cmasks)

    qk = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32)
    for c in range(num_chunks):
      qk += plgpu.dot(q_chunks[c], k_chunks_dq[c].T)
    qk, tanh_val = _scaled_logits(qk, sm_scale, soft_cap)

    if causal or window is not None or segment_ids_ref is not None:
      span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
      kv_segment_ids = None if segment_ids_ref is None else segment_ids_ref[curr_k_slice]
      mask = _combined_mask(span_q, span_k, q_segment_ids, kv_segment_ids, causal, window)
      assert mask is not None
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    p = jnp.exp2(qk - lse[:, None])
    dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
    for c in range(num_chunks):
      dp += plgpu.dot(do_chunks[c], v_chunks_dq[c].T)
    ds = p * dp
    if soft_cap is not None:
      ds = ds * (1.0 - tanh_val * tanh_val)
    if sm_scale != 1.0:
      ds = ds * sm_scale

    ds_cast = ds.astype(k_ref.dtype)
    return tuple(dq[c] + plgpu.dot(ds_cast, k_chunks_dq[c]).astype(dq[c].dtype) for c in range(num_chunks))

  if causal:
    upper_bound_dq = pl.cdiv((start_q + 1) * block_q_dq, block_kv_dq)
  else:
    upper_bound_dq = pl.cdiv(kv_seq_len, block_kv_dq)
  if window is not None:
    first_k_dq = start_q * block_q_dq - window + 1
    lower_bound_dq = lax.max(0, lax.div(first_k_dq, block_kv_dq))
  else:
    lower_bound_dq = 0

  dq_chunks = lax.fori_loop(lower_bound_dq, upper_bound_dq, inner_loop_dq, dq_chunks)
  for c in range(num_chunks):
    plgpu.store(dq_ref.at[:, cslices[c]], dq_chunks[c].astype(dq_ref.dtype), mask=cmasks[c])


def _xla_backward_from_residuals(q, k, v, segment_ids, out, do, lse, *, sm_scale, causal, window, soft_cap):
  """Unfused XLA backward reusing the kernel's saved residuals (out, lse).

  Unlike ``jax.vjp(mha_reference)`` this never re-runs the forward: softmax
  probabilities are reconstructed from ``lse`` (base-2 domain) and only the
  five essential matmuls (QK^T, dP, dQ, dK, dV) are computed.
  """
  q_seq_len, kv_seq_len = q.shape[1], k.shape[1]
  x = jnp.einsum("bqhc,bkhc->bhqk", q, k, preferred_element_type=jnp.float32)
  if sm_scale != 1.0:
    x = x * sm_scale
  t = None
  if soft_cap is not None:
    t = jnp.tanh(x / soft_cap)
    x = soft_cap * t

  mask = None
  if segment_ids is not None:
    mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
  row_ids = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, kv_seq_len), 0)
  col_ids = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, kv_seq_len), 1)
  if causal:
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  if window is not None:
    window_mask = (row_ids - col_ids < window)[None, None, :, :]
    mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

  x2 = x * _LOG2E
  if mask is not None:
    x2 = jnp.where(mask, x2, DEFAULT_MASK_VALUE)
  p = jnp.exp2(x2 - lse[..., None])  # [b, h, q, kv]

  # Keep matmul operands in bf16 (tensor cores) and accumulate in fp32.
  do_lp = do.astype(q.dtype)
  di = jnp.einsum("bqhc,bqhc->bhq", out, do_lp, preferred_element_type=jnp.float32)
  dp = jnp.einsum("bqhc,bkhc->bhqk", do_lp, v, preferred_element_type=jnp.float32)
  ds = p * (dp - di[..., None])
  if soft_cap is not None:
    ds = ds * (1.0 - t * t)
  if sm_scale != 1.0:
    ds = ds * sm_scale

  ds_lp = ds.astype(q.dtype)
  p_lp = p.astype(q.dtype)
  dq = jnp.einsum("bhqk,bkhc->bqhc", ds_lp, k, preferred_element_type=jnp.float32)
  dk = jnp.einsum("bhqk,bqhc->bkhc", ds_lp, q, preferred_element_type=jnp.float32)
  dv = jnp.einsum("bhqk,bqhc->bkhc", p_lp, do_lp, preferred_element_type=jnp.float32)
  return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), None


def _mha_backward(
    sm_scale: float,
    causal: bool,
    window: int | None,
    soft_cap: float | None,
    block_sizes: BlockSizes | None,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int | None,
    grid: Any,
    interpret: bool,
    debug: bool,
    return_residuals: bool,
    res,
    do,
):
  """Backward-pass wrapper: dispatches to the fused kernel or the XLA fallback."""
  if return_residuals:
    raise ValueError("Kernel differentiation is not supported if return_residuals is True.")
  q, k, v, segment_ids, out, lse = res
  del grid, return_residuals

  if backward_pass_impl == "auto":
    # Triton bwd is register-bound above head_dim 256 (dK/dV accumulators) and
    # loses to XLA's unfused backward there; the fused forward still wins.
    backward_pass_impl = "xla" if pl.next_power_of_2(q.shape[-1]) > 256 else "triton"

  if backward_pass_impl == "xla":
    return _xla_backward_from_residuals(
        q, k, v, segment_ids, out, do, lse, sm_scale=sm_scale, causal=causal, window=window, soft_cap=soft_cap
    )
  elif backward_pass_impl == "triton":
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    if block_sizes is None:
      block_sizes = BlockSizes.get_for_head_dim(head_dim)
    if not block_sizes.has_backward_blocks:
      raise ValueError("Backward block sizes must all be set.")

    block_q = min(block_sizes.block_q, q_seq_len)
    block_q_dkv = min(block_sizes.block_q_dkv, q_seq_len)
    block_kv_dkv = min(block_sizes.block_kv_dkv, kv_seq_len)
    block_q_dq = min(block_sizes.block_q_dq, q_seq_len)
    block_kv_dq = min(block_sizes.block_kv_dq, kv_seq_len)
    head_dim_padded = pl.next_power_of_2(head_dim)
    dim_chunk = _pick_dim_chunk(head_dim_padded)

    if q_seq_len // block_q_dq != kv_seq_len // block_kv_dkv:
      raise ValueError(
          "q_seq_len and kv_seq_len must be divided into the same number of blocks for the fused backward pass."
      )

    delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)
    out_shapes = [
        jax.ShapeDtypeStruct.like(q),
        jax.ShapeDtypeStruct.like(k),
        jax.ShapeDtypeStruct.like(v),
    ]

    in_specs: list[pl.BlockSpec | None] = [
        pl.BlockSpec((None, q_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, q_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, q_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
    ]
    if segment_ids is None:
      in_specs.insert(3, None)
    else:
      in_specs.insert(3, pl.BlockSpec((None, kv_seq_len), lambda i, j, _: (i, 0)))

    grid_bwd = (batch_size, num_heads, pl.cdiv(kv_seq_len, block_kv_dkv))
    num_warps_ = num_warps
    if num_warps_ is None:
      if block_q_dkv * block_kv_dkv < 128 * 128 or block_q_dq * block_kv_dq < 128 * 128:
        num_warps_ = 4
      else:
        num_warps_ = 8
    num_stages_ = num_stages
    if num_stages_ is None:
      num_stages_ = 1 if head_dim_padded > 256 else 2

    dq, dk, dv = pl.pallas_call(
        functools.partial(
            mha_backward_kernel,
            sm_scale=sm_scale,
            causal=causal,
            window=window,
            soft_cap=soft_cap,
            block_q_dkv=block_q_dkv,
            block_kv_dkv=block_kv_dkv,
            block_q_dq=block_q_dq,
            block_kv_dq=block_kv_dq,
            head_dim=head_dim,
            dim_chunk=dim_chunk,
        ),
        out_shape=out_shapes,
        in_specs=in_specs,
        grid=grid_bwd,
        out_specs=[
            pl.BlockSpec((None, block_q_dq, None, head_dim_padded), lambda i, j, k: (i, k, j, 0)),
            pl.BlockSpec((None, block_kv_dkv, None, head_dim_padded), lambda i, j, k: (i, k, j, 0)),
            pl.BlockSpec((None, block_kv_dkv, None, head_dim_padded), lambda i, j, k: (i, k, j, 0)),
        ],
        name="mha_backward_sw",
        debug=debug,
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=num_stages_),
    )(q, k, v, segment_ids, out, do, lse, delta)
  else:
    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
  return dq.astype(q.dtype), dk, dv, None


mha.defvjp(_mha_forward, _mha_backward)


@jax.jit(static_argnames=["sm_scale", "causal", "window", "soft_cap"])
def mha_reference(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale=1.0,
    causal: bool = False,
    window: int | None = None,
    soft_cap: float | None = None,
):
  """Unfused reference implementation (for testing)."""
  q_seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  logits = jnp.einsum("bqhc,bkhc->bhqk", q, k, preferred_element_type=jnp.float32)
  mask = None
  if segment_ids is not None:
    mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
    mask = jnp.broadcast_to(mask, logits.shape)
  row_ids = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, kv_seq_len), 0)
  col_ids = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, kv_seq_len), 1)
  if causal:
    causal_mask = jnp.broadcast_to((col_ids <= row_ids)[None, None, :, :], logits.shape)
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  if window is not None:
    window_mask = jnp.broadcast_to((row_ids - col_ids < window)[None, None, :, :], logits.shape)
    mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)
  logits = logits * sm_scale
  if soft_cap is not None:
    logits = soft_cap * jnp.tanh(logits / soft_cap)
  logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
  weights = jax.nn.softmax(logits)
  return jnp.einsum("bhqk,bkhc->bqhc", weights, v, preferred_element_type=jnp.float32)
