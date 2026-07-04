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

"""Pallas TPU kernels for the Gated Delta Net chunked delta rule.

The chunked gated delta rule splits into a chunk-parallel precomputation
(the WY factors, handled well by XLA) and a strictly sequential inter-chunk
recurrence. The recurrence is the throughput bottleneck when expressed as
`lax.scan`: every step is a separate XLA loop iteration whose recurrent
state round-trips through HBM between many small fusions.

This module fuses the recurrence into a single Pallas kernel. The grid is
(batch, head-tiles, num_chunks) with the chunk dimension marked "arbitrary"
so each program walks its chunks sequentially while the recurrent state
lives in a VMEM scratch buffer for the whole walk. The walk is sequential
per head, but heads are independent, so each grid cell batches a tile of
heads and every matmul is a leading-axis dot_general: Mosaic pipelines the
MXU across heads instead of stalling on the chain of chunk dependencies.

Forward additionally materializes the per-chunk input states (FLA-style
chunkwise checkpointing) so the backward kernel can walk the chunks in
reverse with the same fusion structure, carrying the state cotangent in
VMEM. The initial recurrent state is a first-class input and the final
state a first-class output, so the kernel is differentiable end-to-end
including through the recurrent-state chain.

Matmul operands are cast to `compute_dtype` (bf16 by default) with float32
accumulation via `preferred_element_type` — the MXU fast path. The
recurrent state and the log-decay math stay float32 throughout.

Shapes (per chunk c, head-batched by the grid):
  w, u, q, k: [C, D]   g (cumulative log-decay within chunk): [C]
  recurrent state h: [D_k, D_v]

The module also provides `invert_unit_lower`, a Pallas blockwise inversion
of the unit-lower-triangular UT-transform system that replaces the
row-sequential TPU triangular solve in the chunk-parallel stage.
"""

# Pallas kernel bodies receive their operand refs positionally in the order
# declared by pallas_call, so the many-positional-arguments rule cannot apply.
# pylint: disable=too-many-positional-arguments

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _bdot(a, b, contract, compute_dtype):
  """Batched MXU dot over the leading (heads) axis: contract a-dim vs b-dim."""
  return jax.lax.dot_general(
      a.astype(compute_dtype),
      b.astype(compute_dtype),
      dimension_numbers=(((contract[0],), (contract[1],)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )


def _tril_mask(chunk_size: int, include_diag: bool = True) -> jax.Array:
  rows = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  return rows >= cols if include_diag else rows > cols


def _gdn_scan_fwd_kernel(
    w_ref,
    u_ref,
    q_ref,
    k_ref,
    g_ref,
    h0_ref,
    o_ref,
    h_saved_ref,
    h_final_ref,
    h_scratch,
    *,
    chunk_size: int,
    num_chunks: int,
    compute_dtype: jnp.dtype,
):
  """Forward inter-chunk recurrence for one (batch, head-tile) program."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    h_scratch[...] = h0_ref[0].astype(jnp.float32)

  # [TH, d_k, d_v] — the walk is sequential per head, but the TH heads in
  # this cell are independent, so every dot below is batched over the
  # leading axis and Mosaic can pipeline the MXU across heads.
  h = h_scratch[...]
  # Save the chunk-input state for the backward pass.
  h_saved_ref[0, 0] = h

  # MXU dots: operands in the compute dtype (bf16), float32 accumulation
  # via preferred_element_type. State and decay math stay f32.
  def mxu_dot(a, b, contract=(2, 1)):
    return _bdot(a, b, contract, compute_dtype)

  w = w_ref[0, 0]
  u = u_ref[0, 0].astype(jnp.float32)
  q = q_ref[0, 0]
  k = k_ref[0, 0]
  g = g_ref[0, 0, :, :, 0].astype(jnp.float32)  # [TH, C]

  exp_g = jnp.exp(g)
  q_g = q.astype(jnp.float32) * exp_g[..., None]
  # Merged matmul: [q_g ; w] @ h doubles the LHS rows for the MXU.
  both = mxu_dot(jnp.concatenate([q_g, w.astype(jnp.float32)], axis=1), h)
  attn_inter, v_prime = both[:, :chunk_size], both[:, chunk_size:]
  v_new = u - v_prime

  p = mxu_dot(q, k, contract=(2, 2))
  g_diff = g[:, :, None] - g[:, None, :]
  mask = _tril_mask(chunk_size)
  decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
  s = p * decay

  o_ref[0, 0] = (attn_inter + mxu_dot(s, v_new)).astype(o_ref.dtype)

  # State update.
  g_last = g[:, chunk_size - 1]
  gamma = jnp.exp(g_last)[:, None, None]
  dvec = jnp.exp(g_last[:, None] - g)
  kd = k.astype(jnp.float32) * dvec[..., None]
  h_new = h * gamma + mxu_dot(kd, v_new, contract=(1, 1))
  h_scratch[...] = h_new

  @pl.when(n == num_chunks - 1)
  def _final():
    h_final_ref[0] = h_new


def _gdn_scan_bwd_kernel(
    w_ref,
    u_ref,
    q_ref,
    k_ref,
    g_ref,
    h_saved_ref,
    do_ref,
    dh_final_ref,
    dw_ref,
    du_ref,
    dq_ref,
    dk_ref,
    dg_ref,
    dh0_ref,
    dh_scratch,
    *,
    chunk_size: int,
    num_chunks: int,
    compute_dtype: jnp.dtype,
):
  """Reverse inter-chunk recurrence: walks chunks from last to first."""
  n = pl.program_id(2)

  @pl.when(n == 0)
  def _init():
    # Seed the state cotangent with the final-state cotangent.
    dh_scratch[...] = dh_final_ref[0].astype(jnp.float32)

  dh_next = dh_scratch[...]  # [TH, d_k, d_v]; batched over heads like the fwd

  def mxu_dot(a, b, contract=(2, 1)):
    return _bdot(a, b, contract, compute_dtype)

  w = w_ref[0, 0].astype(jnp.float32)
  u = u_ref[0, 0].astype(jnp.float32)
  q = q_ref[0, 0].astype(jnp.float32)
  k = k_ref[0, 0].astype(jnp.float32)
  g = g_ref[0, 0, :, :, 0].astype(jnp.float32)  # [TH, C]
  h = h_saved_ref[0, 0].astype(jnp.float32)
  do = do_ref[0, 0].astype(jnp.float32)

  # --- Recompute forward intermediates for this chunk ---
  exp_g = jnp.exp(g)
  q_g = q * exp_g[..., None]
  v_prime = mxu_dot(w, h)
  v_new = u - v_prime
  p = mxu_dot(q, k, contract=(2, 2))
  g_diff = g[:, :, None] - g[:, None, :]
  mask = _tril_mask(chunk_size)
  decay = jnp.where(mask, jnp.exp(jnp.where(mask, g_diff, 0.0)), 0.0)
  s = p * decay
  g_last = g[:, chunk_size - 1]
  gamma = jnp.exp(g_last)[:, None, None]
  dvec = jnp.exp(g_last[:, None] - g)
  kd = k * dvec[..., None]

  # --- Backward through the state update: h' = gamma*h + kd^T @ v_new ---
  dgamma = jnp.sum(h * dh_next, axis=(1, 2))
  dkd = mxu_dot(v_new, dh_next, contract=(2, 2))  # [TH, C, D_k]
  dv_new = mxu_dot(kd, dh_next)  # [TH, C, D_v]
  dh = dh_next * gamma

  dk = dkd * dvec[..., None]
  ddvec = jnp.sum(dkd * k, axis=2)

  # --- Backward through the output: o = q_g @ h + s @ v_new ---
  ds = mxu_dot(do, v_new, contract=(2, 2))
  ds = jnp.where(mask, ds, 0.0)
  dv_new = dv_new + mxu_dot(s, do, contract=(1, 1))
  dp = ds * decay
  ddecay = ds * p
  dq = mxu_dot(dp, k)
  dk = dk + mxu_dot(dp, q, contract=(1, 1))
  # decay = exp(g_i - g_j) on the tril: dg_i += sum_j(ddecay*decay); dg_j -= sum_i(...)
  dgd = ddecay * decay
  dg = jnp.sum(dgd, axis=2) - jnp.sum(dgd, axis=1)

  dq_g = mxu_dot(do, h, contract=(2, 2))
  dh = dh + mxu_dot(q_g, do, contract=(1, 1))

  # --- Backward through v_new = u - w @ h ---
  du = dv_new
  dw = -mxu_dot(dv_new, h, contract=(2, 2))
  dh = dh - mxu_dot(w, dv_new, contract=(1, 1))

  # --- Backward through q_g = q * exp(g) ---
  dq = dq + dq_g * exp_g[..., None]
  dg = dg + jnp.sum(dq_g * q, axis=2) * exp_g

  # --- Decay-vector and gamma contributions to g ---
  # dvec = exp(g_last - g); gamma = exp(g_last)
  dg = dg - ddvec * dvec
  dg_last = jnp.sum(ddvec * dvec, axis=1) + dgamma * jnp.exp(g_last)
  one_hot_last = (jax.lax.broadcasted_iota(jnp.int32, (chunk_size,), 0) == chunk_size - 1).astype(jnp.float32)
  dg = dg + dg_last[:, None] * one_hot_last[None, :]

  dw_ref[0, 0] = dw.astype(dw_ref.dtype)
  du_ref[0, 0] = du.astype(du_ref.dtype)
  dq_ref[0, 0] = dq.astype(dq_ref.dtype)
  dk_ref[0, 0] = dk.astype(dk_ref.dtype)
  dg_ref[0, 0, :, :, 0] = dg.astype(dg_ref.dtype)

  dh_scratch[...] = dh

  @pl.when(n == num_chunks - 1)
  def _final():
    # After the reverse walk over chunk 0, dh is dL/dh_0.
    dh0_ref[0] = dh


def _head_tile(num_heads: int, max_tile: int = 8) -> int:
  """Heads per grid cell: independent heads batched to pipeline the MXU.

  The cap is VMEM-driven: the backward kernel keeps 14 head-tiled blocks
  resident (8 inputs + 6 outputs), which exceeds the 16MB scoped VMEM
  limit at 8 heads, so it runs with 4.
  """
  for cand in (max_tile, max_tile // 2, 2):
    if 1 < cand <= num_heads and num_heads % cand == 0:
      return cand
  return 1


def _fwd_pallas(w, u, q, k, g, h0, *, compute_dtype=jnp.bfloat16, interpret=False):
  """Runs the forward kernel. Inputs are [B, N, H, C, D] / g [B, N, H, C] / h0 [B, H, D_k, D_v]."""
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = u.shape[-1]
  th = _head_tile(num_heads)

  grid = (batch, num_heads // th, num_chunks)

  def chunk_spec(d):
    return pl.BlockSpec((1, 1, th, chunk_size, d), lambda b, h, n: (b, n, h, 0, 0))

  # TPU block tiling requires the last two dims to be (8k, 128k) or equal to
  # the array dims; per-chunk vectors ride along with a trailing singleton.
  g_spec = pl.BlockSpec((1, 1, th, chunk_size, 1), lambda b, h, n: (b, n, h, 0, 0))
  state_spec = pl.BlockSpec((1, 1, th, d_k, d_v), lambda b, h, n: (b, n, h, 0, 0))
  bh_state_spec = pl.BlockSpec((1, th, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(
      _gdn_scan_fwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype
  )
  o, h_saved, h_final = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec, bh_state_spec],
      out_specs=[chunk_spec(d_v), state_spec, bh_state_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), compute_dtype),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, d_k, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((th, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_scan_fwd",
  )(w, u, q, k, g, h0)
  return o, h_saved, h_final


def _bwd_pallas(w, u, q, k, g, h_saved, do, dh_final, *, compute_dtype=jnp.bfloat16, interpret=False):
  """Runs the backward kernel (reverse chunk walk via index remapping)."""
  batch, num_chunks, num_heads, chunk_size, d_k = q.shape
  d_v = u.shape[-1]

  th = _head_tile(num_heads, max_tile=4)
  grid = (batch, num_heads // th, num_chunks)

  def rev(b, h, n):
    # Reverse the chunk dimension: grid step n touches chunk (N-1-n).
    return (b, num_chunks - 1 - n, h, 0, 0)

  def chunk_spec(d):
    return pl.BlockSpec((1, 1, th, chunk_size, d), rev)

  g_spec = pl.BlockSpec((1, 1, th, chunk_size, 1), rev)
  state_spec = pl.BlockSpec((1, 1, th, d_k, d_v), rev)
  bh_state_spec = pl.BlockSpec((1, th, d_k, d_v), lambda b, h, n: (b, h, 0, 0))

  g = g[..., None]
  kernel = functools.partial(
      _gdn_scan_bwd_kernel, chunk_size=chunk_size, num_chunks=num_chunks, compute_dtype=compute_dtype
  )
  dw, du, dq, dk, dg, dh0 = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=[
          chunk_spec(d_k),
          chunk_spec(d_v),
          chunk_spec(d_k),
          chunk_spec(d_k),
          g_spec,
          state_spec,
          chunk_spec(d_v),
          bh_state_spec,
      ],
      out_specs=[chunk_spec(d_k), chunk_spec(d_v), chunk_spec(d_k), chunk_spec(d_k), g_spec, bh_state_spec],
      out_shape=[
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_v), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, d_k), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_chunks, num_heads, chunk_size, 1), jnp.float32),
          jax.ShapeDtypeStruct((batch, num_heads, d_k, d_v), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((th, d_k, d_v), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
      ),
      interpret=interpret,
      name="gdn_scan_bwd",
  )(w, u, q, k, g, h_saved, do, dh_final)
  return dw, du, dq, dk, dg[..., 0], dh0


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def gdn_inter_chunk_scan(w, u, q, k, g, h0, interpret=False, compute_dtype=jnp.bfloat16):
  """Fused inter-chunk gated-delta-rule scan.

  Args:
    w, u: WY factors, [B, N, H, C, D_k] / [B, N, H, C, D_v].
    q, k: chunked queries/keys, [B, N, H, C, D_k].
    g: per-chunk cumulative log-decay, [B, N, H, C] (float32).
    h0: initial recurrent state, [B, H, D_k, D_v] (float32).
    interpret: run the Pallas kernels in interpret mode (CPU testing).
    compute_dtype: operand dtype for the MXU matmuls (accumulation is f32).

  Returns:
    (o, h_final): chunk outputs [B, N, H, C, D_v] (float32) and the final
    recurrent state [B, H, D_k, D_v] (float32).
  """
  o, _, h_final = _fwd_pallas(w, u, q, k, g, h0, compute_dtype=compute_dtype, interpret=interpret)
  return o, h_final


def _gdn_scan_vjp_fwd(w, u, q, k, g, h0, interpret, compute_dtype):
  o, h_saved, h_final = _fwd_pallas(w, u, q, k, g, h0, compute_dtype=compute_dtype, interpret=interpret)
  return (o, h_final), (w, u, q, k, g, h_saved)


def _gdn_scan_vjp_bwd(interpret, compute_dtype, residuals, cotangents):
  """Backward rule: runs the reverse-walk kernel on the saved chunk states."""
  w, u, q, k, g, h_saved = residuals
  do, dh_final = cotangents
  # The backward kernel seeds the reverse walk with dh_final and emits dh0,
  # so the state chain is differentiated end-to-end inside the kernel.
  dw, du, dq, dk, dg, dh0 = _bwd_pallas(
      w, u, q, k, g, h_saved, do, dh_final, compute_dtype=compute_dtype, interpret=interpret
  )
  # The kernel accumulates gradients in float32; cotangents must match the
  # primal dtypes (inputs may arrive in bf16).
  return (dw.astype(w.dtype), du.astype(u.dtype), dq.astype(q.dtype), dk.astype(k.dtype), dg.astype(g.dtype), dh0)


gdn_inter_chunk_scan.defvjp(_gdn_scan_vjp_fwd, _gdn_scan_vjp_bwd)


# =============================================================================
# Unit-lower-triangular inversion kernel
# =============================================================================
# Replaces jax.scipy.linalg.solve_triangular for the UT-transform inverse
# A = (I + S)^{-1}. The TPU triangular solve substitutes row by row and
# barely uses the MXU; blockwise inversion is pure matmuls, and running it
# as a Pallas kernel keeps every [C, C] tile in VMEM instead of
# round-tripping the intermediate block products through HBM (which made
# the same algorithm speed-neutral at the XLA level).


def _invert_unit_lower_mxu(s: jax.Array) -> jax.Array:
  """(I + s)^{-1} for strictly lower-triangular s: stable blockwise ladder.

  Level doubling: given X = (I + S_b)^{-1} for the block-diagonal part S_b
  at block size b, the size-2b inverse is exactly X - X @ S_l @ X, where
  S_l holds the entries between sibling b-blocks ([[A,0],[C,B]]^{-1} =
  [[A^{-1},0],[-B^{-1}CA^{-1},B^{-1}]]). Starting from X = I (b=1) and
  doubling to full size costs 2*log2(C) full-width MXU matmuls, and never
  forms powers of s — only realized inverses times raw s — so intermediate
  magnitudes stay at the scale of the true inverse (stable for unbounded
  s, unlike Neumann doubling which overflows once |s| > 1).
  """
  size = s.shape[-1]
  rows = jax.lax.broadcasted_iota(jnp.int32, (size, size), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (size, size), 1)

  def mm(a, b):
    # bf16x3: f32-grade products from bf16 MXU passes (Mosaic's default f32
    # dot truncates operands to bf16; Precision.HIGH is unsupported). The
    # leading tile axis is a batched dot: the per-tile products are
    # independent, so Mosaic can pipeline them through the MXU and hide the
    # fill/drain of the otherwise strictly sequential ladder chain.
    a_hi = a.astype(jnp.bfloat16)
    b_hi = b.astype(jnp.bfloat16)
    a_lo = (a - a_hi.astype(jnp.float32)).astype(jnp.bfloat16)
    b_lo = (b - b_hi.astype(jnp.float32)).astype(jnp.bfloat16)
    dims = (((a.ndim - 1,), (a.ndim - 2,)), (tuple(range(a.ndim - 2)),) * 2)
    dot = functools.partial(jax.lax.dot_general, dimension_numbers=dims, preferred_element_type=jnp.float32)
    return dot(a_hi, b_hi) + dot(a_hi, b_lo) + dot(a_lo, b_hi)

  x = jnp.broadcast_to(jnp.eye(size, dtype=jnp.float32), s.shape).astype(jnp.float32)
  block = 1
  while block < size:
    sibling = (rows // block != cols // block) & (rows // (2 * block) == cols // (2 * block))
    s_level = jnp.where(sibling, s, 0.0)
    x = x - mm(x, mm(s_level, x))
    block *= 2
  return x


def _invert_unit_lower_kernel(s_ref, a_ref, *, chunk_size: int):
  del chunk_size
  a_ref[0, :, 0] = _invert_unit_lower_mxu(s_ref[0, :, 0].astype(jnp.float32))


def _invert_pallas(s, *, interpret=False):
  """Runs the inversion kernel on [B, N, H, C, C] with chunk tiles batched per grid cell."""
  batch, num_chunks, num_heads, c, _ = s.shape
  # Group chunks per grid cell: the ladder's matmuls are sequentially
  # dependent within a tile, so batching independent tiles keeps the MXU
  # pipeline full. VMEM per cell stays ~1MB at tile_n=8, C=128.
  tile_n = 1
  for cand in (8, 4, 2):
    if num_chunks % cand == 0:
      tile_n = cand
      break
  grid = (batch, num_heads, num_chunks // tile_n)
  spec = pl.BlockSpec((1, tile_n, 1, c, c), lambda b, h, n: (b, n, h, 0, 0))
  return pl.pallas_call(
      functools.partial(_invert_unit_lower_kernel, chunk_size=c),
      grid=grid,
      in_specs=[spec],
      out_specs=spec,
      out_shape=jax.ShapeDtypeStruct(s.shape, jnp.float32),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
      ),
      interpret=interpret,
      name="gdn_invert_unit_lower",
  )(s)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def invert_unit_lower(s, interpret=False):
  """A = (I + s)^{-1} for strictly lower-triangular s, [B, N, H, C, C] float32."""
  return _invert_pallas(s, interpret=interpret)


def _invert_vjp_fwd(s, interpret):
  a = _invert_pallas(s, interpret=interpret)
  return a, a


def _invert_vjp_bwd(interpret, a, da):
  """Analytic inversion VJP, masked to the strict lower triangle."""
  del interpret
  # d(L^{-1}) = -L^{-1} dL L^{-1}  =>  dL = -A^T dA A^T; L = I + s with s
  # strictly lower and a unit diagonal, so only the strict lower part flows.
  # The primal s is float32 (like a), so the cotangent dtype already matches.
  # HIGHEST precision matches the forward ladder's bf16x3 dots: A has a large
  # dynamic range, and TPU's default single-pass bf16 matmul would truncate
  # the operands and leak ~2^-8 relative error into the k/beta/g gradients.
  a_t = a.swapaxes(-1, -2)
  hi = jax.lax.Precision.HIGHEST
  ds = -jnp.matmul(a_t, jnp.matmul(da, a_t, precision=hi), precision=hi)
  c = a.shape[-1]
  rows = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
  cols = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
  return (jnp.where(rows > cols, ds, 0.0).astype(a.dtype),)


invert_unit_lower.defvjp(_invert_vjp_fwd, _invert_vjp_bwd)
