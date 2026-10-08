# Copyright 2025-2026 Google LLC
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

"""Fused LM-head projection + cross-entropy Pallas TPU kernels.

Computes, for hidden states `x [T, D]`, an LM-head kernel `w [D, V]` and integer
targets `t [T]`, the per-token log-sum-exp and target logit of
`logits = x @ w` without ever writing the `[T, V]` logits to HBM:

  lse[i]   = log(sum_v exp(logits[i, v]))
  tgt[i]   = logits[i, t[i]]   (0 when t[i] is outside [0, V))

`fused_linear_cross_entropy` builds MaxText's loss from these
(`max_utils.cross_entropy_with_logits` semantics):

  loss[i]   = weights[i] * (lse[i] - tgt[i] + z_loss * lse[i]**2)
  z_term[i] = weights[i] * z_loss * lse[i]**2

The custom VJP takes arbitrary cotangents `(a, b)` for `(lse, tgt)` and returns

  dlogits = a[:, None] * softmax(logits) + b[:, None] * onehot(t)
  dx      = dlogits @ w.T          (f32 accumulation, emitted in x.dtype)
  dw      = x.T @ dlogits          (f32 accumulation, emitted in w.dtype)

so z-loss and token weights are handled by ordinary autodiff of the cheap
`[T]`-sized epilogue. The backward recomputes each logits tile in VMEM, in two
kernels (dx: rows parallel / vocab sequential; dw: vocab parallel / rows
sequential). Matmuls run in `compute_dtype` (bf16) with f32 accumulation; `w` may
be passed as an f32 master weight, it is cast once per call and `dw` is returned
in f32.
"""

import dataclasses
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_MASK_VALUE = -1e30


@dataclasses.dataclass(frozen=True)
class FusedCEConfig:
  """Static tiling configuration for the fused LM-head + CE kernels.

  Attributes:
    block_t_fwd / block_v_fwd: token / vocab tile of the forward kernel.
    block_t_dx / block_v_dx: token / vocab tile of the backward dx kernel.
    block_t_dw / block_v_dw: token / vocab tile of the backward dw kernel.
    compute_dtype: matmul operand dtype (accumulation is always f32).
    vmem_limit_bytes: optional Mosaic scoped-VMEM limit for all three kernels
      (TPU v4 caps scoped VMEM at 16 MiB regardless).
    interpret: run the Pallas kernels in interpret mode (CPU tests).
    single_buffer_resident: single-buffer (pl.Buffered(1)) blocks that only change
      on the outer grid axis (x in fwd/dx, w and dw in dw) to save VMEM.
    dw_transposed: compute dw in transposed space (NT + NN matmuls, one in-VMEM
      transpose per vocab tile) instead of an lhs-transposed matmul per step.
    bwd_fused: use one backward kernel (single logits recompute; dw RMW in HBM)
      instead of separate dx and dw kernels (two recomputes).
    block_t_bwd / block_v_bwd: tiles of the fused backward kernel.
    w_gather_axes: (inside shard_map only) mesh axes over which the local `w`
      shard is all-gathered along dims (0, 1). The gather happens after the cast
      to `compute_dtype` (bf16 all-gather) and its transpose is an f32
      `psum_scatter` of dw, so the head gradient keeps f32 precision.
  """

  block_t_fwd: int = 512
  block_v_fwd: int = 1024
  block_t_dx: int = 256
  block_v_dx: int = 512
  block_t_dw: int = 256
  block_v_dw: int = 512
  compute_dtype: str = "bfloat16"
  vmem_limit_bytes: int | None = None
  interpret: bool = False
  single_buffer_resident: bool = True
  dw_transposed: bool = True
  bwd_fused: bool = True
  block_t_bwd: int = 512
  block_v_bwd: int = 128
  w_gather_axes: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())


def _round_up(x: int, m: int) -> int:
  return (x + m - 1) // m * m


def _compiler_params(cfg: FusedCEConfig, semantics):
  kwargs = {"dimension_semantics": semantics}
  if cfg.vmem_limit_bytes is not None:
    kwargs["vmem_limit_bytes"] = cfg.vmem_limit_bytes
  return pltpu.CompilerParams(**kwargs)


def _tile_logits(x, w, j, block_v, vocab_size, mask_vocab):
  """Returns the f32 logits tile and the in-tile column iota."""
  logits = jnp.dot(x, w, preferred_element_type=jnp.float32)
  cols = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
  if mask_vocab:
    logits = jnp.where(cols < vocab_size - j * block_v, logits, _MASK_VALUE)
  return logits, cols


# ---------------------------------------------------------------------------
# Forward: online log-sum-exp + target logit over vocab tiles.
# ---------------------------------------------------------------------------
def _fwd_kernel(
    x_ref, w_ref, tgt_ref, lse_ref, tl_ref, m_sc, l_sc, t_sc, *, block_v, vocab_size, mask_vocab, num_v
):
  j = pl.program_id(1)

  @pl.when(j == 0)
  def _init():
    m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, jnp.float32)
    l_sc[...] = jnp.zeros(l_sc.shape, jnp.float32)
    t_sc[...] = jnp.zeros(t_sc.shape, jnp.float32)

  logits, cols = _tile_logits(x_ref[...], w_ref[...], j, block_v, vocab_size, mask_vocab)
  m_prev = m_sc[...]
  m_new = jnp.maximum(m_prev, jnp.max(logits, axis=1, keepdims=True))
  l_sc[...] = l_sc[...] * jnp.exp(m_prev - m_new) + jnp.sum(jnp.exp(logits - m_new), axis=1, keepdims=True)
  m_sc[...] = m_new
  local_tgt = tgt_ref[...] - j * block_v  # [bt, 1]
  t_sc[...] += jnp.sum(jnp.where(cols == local_tgt, logits, 0.0), axis=1, keepdims=True)

  @pl.when(j == num_v - 1)
  def _finalize():
    lse_ref[...] = m_sc[...] + jnp.log(l_sc[...])
    tl_ref[...] = t_sc[...]


def _fwd_call(x, w, tgt, cfg: FusedCEConfig, vocab_size: int):
  t_pad, d = x.shape
  v_pad = w.shape[1]
  bt, bv = cfg.block_t_fwd, cfg.block_v_fwd
  num_t, num_v = t_pad // bt, v_pad // bv
  kernel = functools.partial(
      _fwd_kernel, block_v=bv, vocab_size=vocab_size, mask_vocab=v_pad != vocab_size, num_v=num_v
  )
  col = lambda i, j: (i, 0)
  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((t_pad, 1), jnp.float32),
          jax.ShapeDtypeStruct((t_pad, 1), jnp.float32),
      ),
      grid=(num_t, num_v),
      in_specs=[
          pl.BlockSpec((bt, d), col, **_single_buffered(cfg)),
          pl.BlockSpec((d, bv), lambda i, j: (0, j)),
          pl.BlockSpec((bt, 1), col),
      ],
      out_specs=(pl.BlockSpec((bt, 1), col), pl.BlockSpec((bt, 1), col)),
      scratch_shapes=[pltpu.VMEM((bt, 1), jnp.float32) for _ in range(3)],
      compiler_params=_compiler_params(cfg, ("parallel", "arbitrary")),
      interpret=cfg.interpret,
      name="fused_ce_fwd",
  )(x, w, tgt)


# ---------------------------------------------------------------------------
# Backward helpers.
# ---------------------------------------------------------------------------
def _dlogits_tile(x, w, tgt, lse, a, b, j, block_v, vocab_size, mask_vocab, out_dtype):
  """dlogits = a * softmax + b * onehot for one [bt, bv] tile, in out_dtype."""
  logits, cols = _tile_logits(x, w, j, block_v, vocab_size, mask_vocab)
  p = jnp.exp(logits - lse)  # masked columns underflow to exactly 0
  g = a * p
  g = jnp.where(cols == tgt - j * block_v, g + b, g)
  return g.astype(out_dtype)


def _dx_kernel(
    x_ref, w_ref, tgt_ref, lse_ref, a_ref, b_ref, dx_ref, acc_sc, *, block_v, vocab_size, mask_vocab, num_v
):
  j = pl.program_id(1)

  @pl.when(j == 0)
  def _init():
    acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

  w = w_ref[...]
  g = _dlogits_tile(
      x_ref[...],
      w,
      tgt_ref[...],
      lse_ref[...],
      a_ref[...],
      b_ref[...],
      j,
      block_v,
      vocab_size,
      mask_vocab,
      w.dtype,
  )
  # [bt, bv] x [d, bv]^T -> [bt, d]
  acc_sc[...] += jax.lax.dot_general(g, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

  @pl.when(j == num_v - 1)
  def _finalize():
    dx_ref[...] = acc_sc[...].astype(dx_ref.dtype)


def _dw_kernel(x_ref, w_ref, tgt_ref, lse_ref, a_ref, b_ref, dw_ref, *, block_v, vocab_size, mask_vocab):
  """dw tile [d, bv] += x^T @ dlogits (lhs-transposed matmul), rows sequential."""
  j = pl.program_id(0)
  i = pl.program_id(1)
  x = x_ref[...]
  g = _dlogits_tile(
      x,
      w_ref[...],
      tgt_ref[...],
      lse_ref[...],
      a_ref[...],
      b_ref[...],
      j,
      block_v,
      vocab_size,
      mask_vocab,
      x.dtype,
  )
  # [bt, d]^T x [bt, bv] -> [d, bv]
  contrib = jax.lax.dot_general(x, g, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)

  @pl.when(i == 0)
  def _init():
    dw_ref[...] = contrib

  @pl.when(i > 0)
  def _acc():
    dw_ref[...] += contrib


def _dw_t_kernel(
    x_ref, w_ref, tgt_ref, lse_ref, a_ref, b_ref, dw_ref, wt_sc, acc_sc, *, block_v, vocab_size, mask_vocab, num_t
):
  """Transposed-space dw: logits^T = w^T x^T (NT), dw^T += g^T x (NN); no per-step transposes.

  Per-token vectors arrive as lane-dense [1, bt] rows. The w tile is transposed
  once per vocab tile into VMEM, and the f32 dw^T accumulator is transposed once
  into the [d, bv] output block after the last token tile.
  """
  j = pl.program_id(0)
  i = pl.program_id(1)

  @pl.when(i == 0)
  def _init():
    wt_sc[...] = w_ref[...].astype(jnp.float32).T.astype(wt_sc.dtype)
    acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

  x = x_ref[...]
  # [bv, d] x [bt, d]^T -> [bv, bt]
  logits_t = jax.lax.dot_general(wt_sc[...], x, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
  rows = jax.lax.broadcasted_iota(jnp.int32, logits_t.shape, 0)
  if mask_vocab:
    logits_t = jnp.where(rows < vocab_size - j * block_v, logits_t, _MASK_VALUE)
  p = jnp.exp(logits_t - lse_ref[...])
  g = a_ref[...] * p
  g = jnp.where(rows == tgt_ref[...] - j * block_v, g + b_ref[...], g).astype(x.dtype)
  # [bv, bt] x [bt, d] -> [bv, d]
  acc_sc[...] += jnp.dot(g, x, preferred_element_type=jnp.float32)

  @pl.when(i == num_t - 1)
  def _finalize():
    dw_ref[...] = acc_sc[...].T


def _single_buffered(cfg: FusedCEConfig):
  """BlockSpec kwargs for blocks that only change on the outer grid axis."""
  if cfg.single_buffer_resident and not cfg.interpret:
    return {"pipeline_mode": pl.Buffered(1)}
  return {}


def _dx_call(x, w, tgt, lse, a, b, cfg: FusedCEConfig, vocab_size: int, dx_dtype):
  t_pad, d = x.shape
  v_pad = w.shape[1]
  bt, bv = cfg.block_t_dx, cfg.block_v_dx
  num_t, num_v = t_pad // bt, v_pad // bv
  kernel = functools.partial(
      _dx_kernel, block_v=bv, vocab_size=vocab_size, mask_vocab=v_pad != vocab_size, num_v=num_v
  )
  col = lambda i, j: (i, 0)
  sb = _single_buffered(cfg)
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((t_pad, d), dx_dtype),
      grid=(num_t, num_v),
      in_specs=[
          pl.BlockSpec((bt, d), col, **sb),
          pl.BlockSpec((d, bv), lambda i, j: (0, j)),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((bt, 1), col),
      ],
      out_specs=pl.BlockSpec((bt, d), col, **sb),
      scratch_shapes=[pltpu.VMEM((bt, d), jnp.float32)],
      compiler_params=_compiler_params(cfg, ("parallel", "arbitrary")),
      interpret=cfg.interpret,
      name="fused_ce_bwd_dx",
  )(x, w, tgt, lse, a, b)


def _dw_call(x, w, tgt, lse, a, b, cfg: FusedCEConfig, vocab_size: int):
  t_pad, d = x.shape
  v_pad = w.shape[1]
  bt, bv = cfg.block_t_dw, cfg.block_v_dw
  num_t, num_v = t_pad // bt, v_pad // bv
  sb = _single_buffered(cfg)
  row = lambda j, i: (i, 0)
  common = dict(
      out_shape=jax.ShapeDtypeStruct((d, v_pad), jnp.float32),
      grid=(num_v, num_t),
      out_specs=pl.BlockSpec((d, bv), lambda j, i: (0, j), **sb),
      compiler_params=_compiler_params(cfg, ("parallel", "arbitrary")),
      interpret=cfg.interpret,
      name="fused_ce_bwd_dw",
  )
  if cfg.dw_transposed:
    kernel = functools.partial(
        _dw_t_kernel, block_v=bv, vocab_size=vocab_size, mask_vocab=v_pad != vocab_size, num_t=num_t
    )
    vec = lambda j, i: (0, i)
    rows = [r.reshape(1, t_pad) for r in (tgt, lse, a, b)]
    return pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec((bt, d), row),
            pl.BlockSpec((d, bv), lambda j, i: (0, j), **sb),
            pl.BlockSpec((1, bt), vec),
            pl.BlockSpec((1, bt), vec),
            pl.BlockSpec((1, bt), vec),
            pl.BlockSpec((1, bt), vec),
        ],
        scratch_shapes=[pltpu.VMEM((bv, d), w.dtype), pltpu.VMEM((bv, d), jnp.float32)],
        **common,
    )(x, w, *rows)
  kernel = functools.partial(_dw_kernel, block_v=bv, vocab_size=vocab_size, mask_vocab=v_pad != vocab_size)
  return pl.pallas_call(
      kernel,
      in_specs=[
          pl.BlockSpec((bt, d), row),
          pl.BlockSpec((d, bv), lambda j, i: (0, j), **sb),
          pl.BlockSpec((bt, 1), row),
          pl.BlockSpec((bt, 1), row),
          pl.BlockSpec((bt, 1), row),
          pl.BlockSpec((bt, 1), row),
      ],
      **common,
  )(x, w, tgt, lse, a, b)


def _bwd_fused_kernel(
    x_ref,
    w_ref,
    tgt_ref,
    lse_ref,
    a_ref,
    b_ref,
    dwin_ref,
    dx_ref,
    dw_ref,
    *,
    block_v,
    num_vh,
    vocab_size,
    mask_vocab,
):
  """Single-pass backward: one logits recompute feeds both dx and dw.

  Grid (half, token tile, vocab tile within half). dx for the token tile is
  accumulated in the resident f32 output block over the vocab half (one partial
  per half, summed outside); dw is read-modify-written in HBM via an aliased
  input.
  """
  c = pl.program_id(0)
  i = pl.program_id(1)
  j = pl.program_id(2)
  vt = c * num_vh + j

  @pl.when(j == 0)
  def _init():
    dx_ref[...] = jnp.zeros(dx_ref.shape, jnp.float32)

  x = x_ref[...]
  w = w_ref[...]
  g = _dlogits_tile(
      x,
      w,
      tgt_ref[...],
      lse_ref[...],
      a_ref[...],
      b_ref[...],
      vt,
      block_v,
      vocab_size,
      mask_vocab,
      w.dtype,
  )
  dx_ref[0] += jax.lax.dot_general(g, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
  contrib = jax.lax.dot_general(x, g, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)  # [d, bv]

  @pl.when(i == 0)
  def _first():
    dw_ref[...] = contrib

  @pl.when(i > 0)
  def _rest():
    dw_ref[...] = dwin_ref[...] + contrib


def _bwd_fused_call(x, w, tgt, lse, a, b, cfg: FusedCEConfig, vocab_size: int):
  """Returns (dx f32 [T, D], dw f32 [D, V_pad]) from one fused backward kernel."""
  t_pad, d = x.shape
  v_pad = w.shape[1]
  bt, bv = cfg.block_t_bwd, cfg.block_v_bwd
  num_t, num_vh = t_pad // bt, v_pad // (2 * bv)
  kernel = functools.partial(
      _bwd_fused_kernel, block_v=bv, num_vh=num_vh, vocab_size=vocab_size, mask_vocab=v_pad != vocab_size
  )
  sb = _single_buffered(cfg)
  col = lambda c, i, j: (i, 0)
  wcol = lambda c, i, j: (0, c * num_vh + j)
  dw_init = jnp.zeros((d, v_pad), jnp.float32)
  dx_parts, dw = pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((2, t_pad, d), jnp.float32),
          jax.ShapeDtypeStruct((d, v_pad), jnp.float32),
      ),
      grid=(2, num_t, num_vh),
      in_specs=[
          pl.BlockSpec((bt, d), col, **sb),
          pl.BlockSpec((d, bv), wcol),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((bt, 1), col),
          pl.BlockSpec((d, bv), wcol),
      ],
      out_specs=(
          pl.BlockSpec((1, bt, d), lambda c, i, j: (c, i, 0), **sb),
          pl.BlockSpec((d, bv), wcol),
      ),
      scratch_shapes=[],
      input_output_aliases={6: 1},
      compiler_params=_compiler_params(cfg, ("parallel", "arbitrary", "arbitrary")),
      interpret=cfg.interpret,
      name="fused_ce_bwd",
  )(x, w, tgt, lse, a, b, dw_init)
  return dx_parts[0] + dx_parts[1], dw


# ---------------------------------------------------------------------------
# custom_vjp wrapper.
# ---------------------------------------------------------------------------
def _pad_inputs(x, w, targets, cfg: FusedCEConfig):
  """Casts to compute dtype, gathers w (if sharded) and pads T / V to tile multiples.

  Returns:
    `(x_padded, w_padded, targets_padded [T_pad, 1], unpadded_vocab_size)`.
  """
  t, _ = x.shape
  cdt = jnp.dtype(cfg.compute_dtype)
  t_mult = max(cfg.block_t_fwd, cfg.block_t_dx, cfg.block_t_dw, cfg.block_t_bwd)
  v_mult = max(cfg.block_v_fwd, cfg.block_v_dx, cfg.block_v_dw, 2 * cfg.block_v_bwd)
  t_pad = _round_up(t, t_mult)
  xc = x.astype(cdt)
  wc = w.astype(cdt)
  for dim in range(2):
    if cfg.w_gather_axes[dim]:
      wc = jax.lax.all_gather(wc, cfg.w_gather_axes[dim], axis=dim, tiled=True)
  v = wc.shape[1]
  v_pad = _round_up(v, v_mult)
  tgt = targets.astype(jnp.int32).reshape(t, 1)
  if t_pad != t:
    xc = jnp.pad(xc, ((0, t_pad - t), (0, 0)))
    tgt = jnp.pad(tgt, ((0, t_pad - t), (0, 0)), constant_values=-1)
  if v_pad != v:
    wc = jnp.pad(wc, ((0, 0), (0, v_pad - v)))
  return xc, wc, tgt, v


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _linear_lse_and_target_logit(x, w, targets, cfg: FusedCEConfig):
  (lse, tl), _ = _lse_fwd(x, w, targets, cfg)
  return lse, tl


def _lse_fwd(x, w, targets, cfg: FusedCEConfig):
  t, _ = x.shape
  xc, wc, tgt, v = _pad_inputs(x, w, targets, cfg)
  lse_p, tl_p = _fwd_call(xc, wc, tgt, cfg, v)
  # Zero-size prototypes carry the primal dtypes, local shard rows, and unpadded vocab size.
  x_proto = jnp.zeros((0,), x.dtype)
  w_proto = jnp.zeros((0, v), w.dtype)  # v = full (gathered, unpadded) vocab
  d_proto = jnp.zeros((w.shape[0], 0), w.dtype)
  residuals = (xc, wc, tgt, lse_p, x_proto, w_proto, d_proto)
  return (lse_p[:t, 0], tl_p[:t, 0]), residuals


def _lse_bwd(cfg: FusedCEConfig, residuals, cotangents):
  xc, wc, tgt, lse_p, x_proto, w_proto, d_proto = residuals
  d_lse, d_tl = cotangents
  t = d_lse.shape[0]
  t_pad = xc.shape[0]
  vocab_size = w_proto.shape[1]
  shard_d = d_proto.shape[0]
  a = jnp.pad(d_lse.astype(jnp.float32), (0, t_pad - t)).reshape(t_pad, 1)
  b = jnp.pad(d_tl.astype(jnp.float32), (0, t_pad - t)).reshape(t_pad, 1)
  if cfg.bwd_fused:
    dx, dw = _bwd_fused_call(xc, wc, tgt, lse_p, a, b, cfg, vocab_size)
  else:
    dx = _dx_call(xc, wc, tgt, lse_p, a, b, cfg, vocab_size, x_proto.dtype)
    dw = _dw_call(xc, wc, tgt, lse_p, a, b, cfg, vocab_size)
  dw = dw[:, :vocab_size].astype(w_proto.dtype)
  if cfg.w_gather_axes[1]:
    dw = jax.lax.psum_scatter(dw, cfg.w_gather_axes[1], scatter_dimension=1, tiled=True)
  if cfg.w_gather_axes[0]:
    # Pad per-shard rows to an 8-multiple so TPU v7x+ emits a native reduce-scatter
    # instead of decomposing psum_scatter into an unsharded all-reduce + dynamic-slice
    # when D / num_shards is not 8-aligned (e.g. 1792 / 128 = 14).
    # Skip optimization_barrier on TPU v4, where kOptimizationBarrier perturbs CMEM
    # overlay allocation in jit_train_step.
    try:
      is_v4 = "v4" in jax.devices()[0].device_kind.lower()
    except Exception:  # pylint: disable=broad-exception-caught
      is_v4 = False
    pad_d = _round_up(shard_d, 8)
    use_fenced_pad = (not is_v4) and (pad_d != shard_d) and (shard_d > 0)
    if use_fenced_pad:
      num_shards = dw.shape[0] // shard_d
      dw = jnp.pad(
          dw.reshape(num_shards, shard_d, dw.shape[1]),
          ((0, 0), (0, pad_d - shard_d), (0, 0)),
      ).reshape(num_shards * pad_d, dw.shape[1])
      dw = jax.lax.optimization_barrier(dw)
    dw = jax.lax.psum_scatter(dw, cfg.w_gather_axes[0], scatter_dimension=0, tiled=True)
    if use_fenced_pad:
      dw = jax.lax.optimization_barrier(dw)
      dw = dw[:shard_d]
  return dx[:t].astype(x_proto.dtype), dw.astype(w_proto.dtype), None


_linear_lse_and_target_logit.defvjp(_lse_fwd, _lse_bwd)


def linear_lse_and_target_logit(
    x: jax.Array, w: jax.Array, targets: jax.Array, cfg: FusedCEConfig | None = None
) -> tuple[jax.Array, jax.Array]:
  """Returns per-token `(lse, target_logit)` of `x @ w` without materializing logits.

  Args:
    x: `[..., D]` hidden states (any float dtype; cast to `cfg.compute_dtype`).
    w: `[D, V]` LM-head kernel (f32 master or bf16; cast to `cfg.compute_dtype`).
      Its gradient is accumulated in f32 and returned in `w.dtype`.
    targets: `[...]` integer targets. Out-of-range targets give a 0 target logit.
    cfg: tiling configuration.

  Returns:
    `(lse, target_logit)`, both f32 with shape `targets.shape`.
  """
  cfg = cfg or FusedCEConfig()
  lead = targets.shape
  d = x.shape[-1]
  lse, tl = _linear_lse_and_target_logit(x.reshape(-1, d), w, targets.reshape(-1), cfg)
  return lse.reshape(lead), tl.reshape(lead)


def fused_linear_cross_entropy(
    x: jax.Array,
    w: jax.Array,
    targets: jax.Array,
    weights: jax.Array | None = None,
    *,
    z_loss: float = 0.0,
    cfg: FusedCEConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Fused `cross_entropy_with_logits(x @ w, onehot(targets), z_loss)` with masking.

  Matches `max_utils.cross_entropy_with_logits` (forward and gradient) followed by
  multiplication with `weights`; the logits are kept in f32 (never rounded to bf16).

  Args:
    x: `[..., D]` hidden states.
    w: `[D, V]` LM-head kernel.
    targets: `[...]` integer targets.
    weights: optional `[...]` per-token weights (e.g. `segmentation != 0`).
    z_loss: z-loss coefficient.
    cfg: tiling configuration.

  Returns:
    `(loss, z_loss_term)` per token, f32 with shape `targets.shape`; `loss`
    already includes `z_loss_term` (as in `cross_entropy_with_logits`).
  """
  lse, tl = linear_lse_and_target_logit(x, w, targets, cfg)
  z_term = z_loss * jnp.square(lse)
  loss = lse - tl + z_term
  if weights is not None:
    weights = weights.astype(jnp.float32)
    loss = loss * weights
    z_term = z_term * weights
  return loss, z_term


def reference_linear_cross_entropy(x, w, targets, weights=None, *, z_loss=0.0, compute_dtype=jnp.bfloat16):
  """Unfused reference: f32 logits of `x @ w` in compute dtype + log-softmax CE."""
  logits = jnp.einsum(
      "...d,dv->...v", x.astype(compute_dtype), w.astype(compute_dtype), preferred_element_type=jnp.float32
  )
  lse = jax.scipy.special.logsumexp(logits, axis=-1)
  onehot = jax.nn.one_hot(targets, w.shape[1], dtype=jnp.float32)
  tl = jnp.sum(logits * onehot, axis=-1)
  z_term = z_loss * jnp.square(lse)
  loss = lse - tl + z_term
  if weights is not None:
    loss = loss * weights
    z_term = z_term * weights
  return loss, z_term


def _spec_axes(entry) -> tuple[str, ...]:
  if entry is None:
    return ()
  if isinstance(entry, str):
    return (entry,)
  return tuple(a for a in entry if a is not None)


def sharded_fused_linear_cross_entropy(
    x: jax.Array,
    w: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    token_spec: jax.sharding.PartitionSpec,
    w_spec: jax.sharding.PartitionSpec,
    z_loss: float = 0.0,
    cfg: FusedCEConfig | None = None,
) -> tuple[jax.Array, jax.Array]:
  """`fused_linear_cross_entropy` under `shard_map` (data-parallel over tokens).

  Each device runs the kernels on its local tokens against the full LM head. The
  head shard is cast to bf16 and all-gathered inside the custom VJP over the mesh
  axes in `w_spec`; the backward reduce-scatters the f32 head gradient
  (`psum_scatter`) over those axes, and the shard_map transpose psums it over the
  remaining token-sharding axes. Mesh axes not
  used by `token_spec` see replicated tokens and recompute the same result.

  Args:
    x: `[B, S, D]` hidden states.
    w: `[D, V]` LM-head kernel (typically the f32 master weight).
    targets: `[B, S]` integer targets.
    weights: `[B, S]` per-token weights.
    mesh: device mesh.
    token_spec: PartitionSpec of `[B, S]` (D is never sharded inside).
    w_spec: PartitionSpec the head arrives in (e.g. its FSDP sharding).
    z_loss: z-loss coefficient.
    cfg: tiling configuration.

  Returns:
    `(loss, z_loss_term)` per token, `[B, S]` f32, sharded like `token_spec`.
  """
  token_spec = jax.sharding.PartitionSpec(*(tuple(token_spec) + (None,) * (2 - len(tuple(token_spec))))[:2])
  w_spec = jax.sharding.PartitionSpec(*(tuple(w_spec) + (None,) * (2 - len(tuple(w_spec))))[:2])
  x_spec = jax.sharding.PartitionSpec(token_spec[0], token_spec[1], None)

  cfg = dataclasses.replace(cfg or FusedCEConfig(), w_gather_axes=(_spec_axes(w_spec[0]), _spec_axes(w_spec[1])))

  def _local(x_l, w_l, t_l, m_l):
    return fused_linear_cross_entropy(x_l, w_l, t_l, m_l, z_loss=z_loss, cfg=cfg)

  return jax.shard_map(
      _local,
      mesh=mesh,
      in_specs=(x_spec, w_spec, token_spec, token_spec),
      out_specs=(token_spec, token_spec),
      check_vma=False,
  )(x, w, targets, weights)
