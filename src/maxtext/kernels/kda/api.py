# Copyright 2023-2026 Google LLC
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

"""Public entry point of the KDA kernels: `chunk_kda`, the chunked Kimi Delta Attention core (fwd + custom VJP).

See `README.md` in this directory for the math, the layouts and the numerics.
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp

from maxtext.kernels.kda import fused

Layout = Literal["token_major", "chunk_major"]

# Name under which `save_residuals=True` tags the kernel output and the VJP residuals.
RESIDUAL_NAME = "kda_fused"


def segment_resets(segment_ids: jax.Array | None, batch: int, seq_len: int) -> jax.Array:
  """`[B, T]` bool, True where a token starts a new document (its segment id differs from the previous token's)."""
  if segment_ids is None:
    return jnp.zeros((batch, seq_len), dtype=bool)
  prev_ids = jnp.pad(segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
  return segment_ids != prev_ids


def chunk_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    *,
    segment_ids: jax.Array | None = None,
    resets: jax.Array | None = None,
    chunk_size: int = 128,
    state_dtype: Any = jnp.float32,
    layout: Layout = "token_major",
    outer_block: int | None = 32,
    inverse_precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
    output_dtype: Any = jnp.float32,
    save_residuals: bool | Literal["all", "wy"] = False,
    residual_name: str = RESIDUAL_NAME,
    state_stride: int = 1,
    interpret: bool | None = None,
) -> jax.Array:
  """Chunked Kimi Delta Attention (gated delta rule with per-key-channel decay), forward and custom VJP.

  Per head, with state S_t in R^{dk x dv} (zero at the start and at every document start):

      S_t = (I - beta_t k_t k_t^T) Diag(exp(log_decay_t)) S_{t-1} + beta_t k_t v_t^T,    o_t = S_t^T q_t.

  It runs as one Pallas TPU kernel per direction (`fused.kda_fused_token_major` / `fused.kda_fused`). The
  function is per device: under SPMD, call it inside `shard_map` with the batch (and/or head) axes split; the
  sequence must not be split.

  Args:
    q: `[B, T, H, dk]` queries, L2-normalized and scaled (e.g. by dk**-0.5) by the caller.
    k: `[B, T, H, dk]` keys, L2-normalized by the caller.
    v: `[B, T, H, dv]` values.
    log_decay: `[B, T, H, dk]` per-step log-decay (<= 0), float32 recommended.
    beta: `[B, T, H]` write strengths (KDA uses beta in [0, 2)).
    segment_ids: optional `[B, T]` int packing ids; the state resets wherever the id changes. Exclusive with
      `resets`.
    resets: optional `[B, T]` bool, True at document starts (precomputed `segment_resets`).
    chunk_size: tokens per chunk C (128 or 64; T is zero-padded to a multiple of it).
    state_dtype: `jnp.float32` keeps the carried state and the q/k/v operands in f32 between chunks (strict);
      `jnp.bfloat16` rounds the state to bf16 at every chunk boundary and takes bf16 q/k/v.
    layout: kernel HBM layout. 'token_major' reads `[B, T, H * d]` directly and takes the in-chunk cumulative
      decay in-kernel (no transposes; needs dk, dv multiples of 128 on TPU unless H == 1). 'chunk_major'
      transposes to `[B * H, N, C, d]` and takes the cumulative decay in XLA.
    outer_block: rows per outer block of the two-level intra-chunk factorization (divides C; None: one level of
      8-row sub-blocks). 32 is fastest on v4 at C = 128.
    inverse_precision: matmul precision of the (I + M)^-1 block doubling and its VJP (HIGHEST or HIGH).
    output_dtype: dtype the kernel writes the output in (f32 accumulation, one rounding on the store).
    save_residuals: True / 'all': tag the output and all VJP residuals with `checkpoint_name(residual_name)`,
      so a remat policy that saves that name skips this kernel's forward recompute in the bwd. 'wy': tag only
      the intra-chunk inverse X, kk and the scores (the costly part of the forward, 48 MiB per layer at the
      olmoe3-3p5b shape), computed by a separate kernel; the remat recompute then runs only the cheaper
      recurrence + readout kernel.
    residual_name: the checkpoint name used by `save_residuals`.
    state_stride: which chunk states (`dk x dv` per head and chunk, f32 or bf16) the VJP forward keeps as
      residuals. 1: all. 2: every other chunk's (half the bytes); the bwd kernel rebuilds the others from the
      previous chunk's saved state, k, decay and delta (one extra matmul per odd chunk). 0: none; a separate
      light kernel rebuilds them all from the saved deltas before the bwd. All bit-identical.
    interpret: run the Pallas kernels in interpret mode (default: when the backend is not TPU).

  Returns:
    `[B, T, H, dv]` outputs in `output_dtype`.
  """
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  if segment_ids is not None and resets is not None:
    raise ValueError("Pass at most one of segment_ids and resets.")
  if resets is None:
    resets = segment_resets(segment_ids, batch, seq_len)
  if interpret is None:
    interpret = jax.devices()[0].platform != "tpu"
  c = chunk_size
  if outer_block is not None and (outer_block <= 0 or outer_block >= c):
    outer_block = None
  pad = (-seq_len) % c
  if pad:  # trailing zero tokens: causal, so they change nothing before them
    pad_t = lambda t: jnp.pad(t, ((0, 0), (0, pad)) + ((0, 0),) * (t.ndim - 2))
    q, k, v, log_decay, beta, resets = (pad_t(t) for t in (q, k, v, log_decay, beta, resets))
  t_pad = seq_len + pad
  n = t_pad // c
  f32 = jnp.float32
  state_f32 = jnp.dtype(state_dtype) == jnp.dtype(jnp.float32)
  compute_dtype = f32 if state_f32 else jnp.bfloat16
  if save_residuals == "wy":
    name = residual_name + ":wy"
  elif save_residuals in (True, "all"):
    name = residual_name
  elif not save_residuals:
    name = None
  else:
    raise ValueError(f"save_residuals must be a bool, 'all' or 'wy', got {save_residuals!r}")
  common = dict(
      precision=inverse_precision,
      interpret=interpret,
      state_f32=state_f32,
      outer_block=outer_block,
      out_dtype=output_dtype,
      residual_name=name,
      state_stride=state_stride,
  )

  seg = jnp.cumsum(resets.reshape(batch, n, c).astype(jnp.int32), axis=-1)  # [B, N, C]
  any_reset = seg[..., -1] > 0  # [B, N]
  beta_c = beta.reshape(batch, n, c, heads).transpose(0, 3, 1, 2).astype(f32)[..., None]  # [B, H, N, C, 1]
  flat = lambda t: t.reshape(batch * heads, *t.shape[2:])
  q, k, v = (t.astype(compute_dtype) for t in (q, k, v))

  if layout == "token_major":
    log_decay = log_decay.astype(f32)
    g_last = log_decay.reshape(batch, n, c, heads, dk).sum(axis=2).transpose(0, 2, 1, 3)  # [B, H, N, dk]
    decay = jnp.where(any_reset[:, None, :, None], 0.0, jnp.exp(g_last))
    tok = lambda t: t.reshape(batch, t_pad, -1)
    out = fused.kda_fused_token_major(
        tok(q),
        tok(k),
        tok(v),
        tok(log_decay),
        flat(beta_c),
        seg[..., None],
        seg[:, :, None, :],
        flat(decay)[..., None],
        heads,
        **common,
    )
    out = out.reshape(batch, t_pad, heads, dv)
  elif layout == "chunk_major":
    to_chunks = lambda t: t.reshape(batch, n, c, heads, -1).transpose(0, 3, 1, 2, 4)  # [B, H, N, C, d]
    # Cumulative decay as a lower-triangular matmul: jnp.cumsum lowers to an O(C^2) reduce_window on TPU.
    tri_ones = jnp.tril(jnp.ones((c, c), f32))
    g = jnp.einsum("ij,...jd->...id", tri_ones, to_chunks(log_decay).astype(f32), precision=jax.lax.Precision.HIGHEST)
    decay = jnp.where(any_reset[:, None, :, None], 0.0, jnp.exp(g[..., -1, :]))
    out = fused.kda_fused(
        flat(to_chunks(q)),
        flat(to_chunks(k)),
        flat(to_chunks(v)),
        flat(g),
        flat(beta_c),
        seg[..., None],
        seg[:, :, None, :],
        flat(decay)[..., None],
        heads,
        **common,
    )
    out = out.reshape(batch, heads, n, c, dv).transpose(0, 2, 3, 1, 4).reshape(batch, t_pad, heads, dv)
  else:
    raise ValueError(f"Unknown layout {layout!r}; expected 'token_major' or 'chunk_major'.")
  return out[:, :seq_len] if pad else out


def residual_bytes_per_layer(
    batch: int,
    seq_len: int,
    heads: int,
    dk: int,
    dv: int,
    chunk_size: int = 128,
    state_dtype: Any = jnp.float32,
    output_dtype: Any = jnp.float32,
    state_stride: int = 1,
    saved: Literal["all", "wy"] = "all",
) -> dict[str, int]:
  """HBM bytes `save_residuals=saved` keeps per call (per device), by tensor; 'total' sums them."""
  n = -(-seq_len // chunk_size)
  bh = batch * heads
  c = chunk_size
  sizes = {
      "out": batch * n * c * heads * dv * jnp.dtype(output_dtype).itemsize,
      "delta": bh * n * c * dv * 2,
      "x_kk": bh * n * c * c * 4,  # X and kk packed into one f32 matrix
      "scores": bh * n * c * c * 2,
  }
  if saved == "wy":
    sizes = {k: v for k, v in sizes.items() if k in ("x_kk", "scores")}
  elif state_stride:
    sizes["states"] = bh * (-(-n // state_stride)) * dk * dv * jnp.dtype(state_dtype).itemsize
  sizes["total"] = sum(sizes.values())
  return sizes
