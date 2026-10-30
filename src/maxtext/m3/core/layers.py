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

"""Segment-isolated primitives for hybrid attention, causal Conv1d, n-gram hashing, and linear recurrence."""

import jax
import jax.numpy as jnp


def segment_causal_conv1d(
    x: jax.Array,
    weight: jax.Array,
    segment_ids: jax.Array | None = None,
    *,
    bias: jax.Array | None = None,
    dilation: int = 1,
) -> jax.Array:
  """Causal depthwise 1D convolution over `[B, S, D]` with strict packed-segment isolation.

  Zeroes any filter tap `j` whose source index `t - dilation * j` falls before position 0,
  lands in a different document (`segment_ids[:, t] != segment_ids[:, t - dilation * j]`),
  or lies on a padding position (`segment_ids[:, t] == 0`), guaranteeing bit-identical outputs
  between packed multi-document sequences and independent padded single-document runs.

  Args:
    x: Input activations of shape `[B, S, D]`.
    weight: Depthwise convolution kernel of shape `[K, D]` or `[D, K]`.
    segment_ids: Optional packed document IDs of shape `[B, S]` (`0` indicates padding).
    bias: Optional per-channel bias of shape `[D]`.
    dilation: Causal tap spacing (`>= 1`).

  Returns:
    Convolved activations of shape `[B, S, D]` in `x.dtype`.
  """
  if x.ndim != 3:
    raise ValueError(f"`x` must have shape [B, S, D]; got {x.shape}.")
  if dilation < 1:
    raise ValueError(f"`dilation` must be >= 1; got {dilation}.")

  _, seq_len, channels = x.shape
  if weight.ndim != 2:
    raise ValueError(f"`weight` must be rank-2 ([K, D] or [D, K]); got {weight.shape}.")
  if weight.shape[1] == channels:
    kernel = weight  # [K, D]
  elif weight.shape[0] == channels:
    kernel = jnp.swapaxes(weight, 0, 1)  # [D, K] -> [K, D]
  else:
    raise ValueError(f"`weight` shape {weight.shape} is incompatible with channel dim {channels}.")

  kernel_size = kernel.shape[0]
  acc = jnp.zeros_like(x, dtype=jnp.float32)
  x_f32 = x.astype(jnp.float32)
  kernel_f32 = kernel.astype(jnp.float32)

  # Convention: kernel[-1] multiplies x[:, t], kernel[-1 - j] multiplies x[:, t - dilation * j].
  for j in range(kernel_size):
    shift = dilation * j
    tap_weight = kernel_f32[kernel_size - 1 - j]  # [D]
    if shift == 0:
      shifted_x = x_f32
      if segment_ids is not None:
        valid = segment_ids != 0
        shifted_x = jnp.where(valid[:, :, None], shifted_x, 0.0)
    elif shift >= seq_len:
      continue
    else:
      shifted_x = jnp.pad(x_f32[:, : seq_len - shift, :], ((0, 0), (shift, 0), (0, 0)))
      if segment_ids is not None:
        shifted_seg = jnp.pad(segment_ids[:, : seq_len - shift], ((0, 0), (shift, 0)), constant_values=-1)
        same_seg = (segment_ids != 0) & (segment_ids == shifted_seg)
        shifted_x = jnp.where(same_seg[:, :, None], shifted_x, 0.0)
    acc = acc + shifted_x * tap_weight[None, None, :]

  if bias is not None:
    bias_term = bias.astype(jnp.float32)[None, None, :]
    if segment_ids is not None:
      acc = acc + jnp.where((segment_ids != 0)[:, :, None], bias_term, 0.0)
    else:
      acc = acc + bias_term

  return acc.astype(x.dtype)


def segment_boundary_mask(segment_ids: jax.Array) -> jax.Array:
  """Returns boolean `[B, S]` mask that is `True` at every document start (`t == 0` or segment change)."""
  if segment_ids.ndim != 2:
    raise ValueError(f"`segment_ids` must have shape [B, S]; got {segment_ids.shape}.")
  prev_seg = jnp.pad(segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
  return (segment_ids != prev_seg) | (segment_ids == 0)


def reset_tokens_on_segment_boundary(
    tokens: jax.Array,
    segment_ids: jax.Array,
    *,
    reset_token_id: int = 0,
) -> jax.Array:
  """Replaces padding positions (`segment_ids == 0`) with `reset_token_id`."""
  return jnp.where(segment_ids != 0, tokens, jnp.asarray(reset_token_id, dtype=tokens.dtype))


def segment_ngram_hash_ids(
    tokens: jax.Array,
    segment_ids: jax.Array | None = None,
    *,
    ngram_size: int = 2,
    num_buckets: int = 4096,
    vocab_mod: int | None = None,
    reset_token_id: int = 0,
) -> jax.Array:
  """Computes causal n-gram hash IDs `[B, S]` while resetting cross-document history at segment boundaries.

  For each tap `j` in `0 .. ngram_size - 1`, any token at `t - j` belonging to a prior document
  (`segment_ids[:, t] != segment_ids[:, t - j]`) is replaced with `reset_token_id`, matching the
  exact `t < j` boundary behavior of a standalone single-document sequence.
  """
  if ngram_size < 1:
    raise ValueError(f"`ngram_size` must be >= 1; got {ngram_size}.")
  modulus = int(vocab_mod if vocab_mod is not None else num_buckets)
  if modulus < 1:
    raise ValueError(f"`num_buckets` must be >= 1; got {modulus}.")
  _, seq_len = tokens.shape
  primes = (1, 1315423911 % modulus or 31, 2654435761 % modulus or 131, 97531 % modulus or 17)
  acc = jnp.zeros_like(tokens, dtype=jnp.int32)
  for j in range(ngram_size):
    mult = primes[j % len(primes)]
    if j == 0:
      tok_j = tokens.astype(jnp.int32)
      if segment_ids is not None:
        tok_j = jnp.where(segment_ids != 0, tok_j, reset_token_id)
    elif j >= seq_len:
      tok_j = jnp.full_like(tokens, reset_token_id, dtype=jnp.int32)
    else:
      tok_j = jnp.pad(
          tokens[:, : seq_len - j].astype(jnp.int32),
          ((0, 0), (j, 0)),
          constant_values=reset_token_id,
      )
      if segment_ids is not None:
        seg_j = jnp.pad(segment_ids[:, : seq_len - j], ((0, 0), (j, 0)), constant_values=-1)
        same_seg = (segment_ids != 0) & (segment_ids == seg_j)
        tok_j = jnp.where(same_seg, tok_j, reset_token_id)
    acc = (acc + tok_j * mult) % modulus

  if segment_ids is not None:
    acc = jnp.where(segment_ids != 0, acc, 0)
  return acc


def segment_chunk_decay_mask(cum_log_decay: jax.Array, segment_ids: jax.Array) -> jax.Array:
  """Builds causal intra-segment pairwise decay matrix `[B, H, S, S]` from cumulative log-decay `[B, S, H]`.

  Entries `(i, j)` with `i < j`, `segment_ids[:, i] != segment_ids[:, j]`, or `segment_ids[:, i] == 0`
  are masked to `-inf` prior to `exp(...)` so cross-document decay weights are strictly `0.0`.
  """
  if cum_log_decay.ndim != 3:
    raise ValueError(f"`cum_log_decay` must have shape [B, S, H]; got {cum_log_decay.shape}.")
  cum_bhs = jnp.swapaxes(cum_log_decay.astype(jnp.float32), 1, 2)  # [B, H, S]
  seq_len = segment_ids.shape[1]
  diff = cum_bhs[:, :, :, None] - cum_bhs[:, :, None, :]  # [B, H, S, S]
  idx = jnp.arange(seq_len)
  causal = idx[:, None] >= idx[None, :]
  same_seg = (segment_ids[:, :, None] == segment_ids[:, None, :]) & (segment_ids[:, :, None] != 0)
  valid = causal[None, None, :, :] & same_seg[:, None, :, :]
  return jnp.exp(jnp.where(valid, diff, -jnp.inf))


def segment_linear_recurrence(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    log_decay: jax.Array,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
  """Causal gated linear attention / recurrence (`GDN`/`KDA`) with exact segment-boundary state reset.

  Args:
    query: `[B, S, H, D_k]` query vectors.
    key: `[B, S, H, D_k]` key vectors.
    value: `[B, S, H, D_v]` value vectors.
    log_decay: `[B, S, H]` non-positive log-decay gates (`<= 0`).
    segment_ids: Optional `[B, S]` packed document IDs (`0` for padding).

  Returns:
    Output activations of shape `[B, S, H, D_v]` in `value.dtype`, with zero cross-document leakage.
  """
  orig_dtype = value.dtype
  q_f = query.astype(jnp.float32)
  k_f = key.astype(jnp.float32)
  v_f = value.astype(jnp.float32)
  g_f = log_decay.astype(jnp.float32)

  if segment_ids is not None:
    boundary = segment_boundary_mask(segment_ids)
    g_f = jnp.where(boundary[:, :, None], -jnp.inf, g_f)
    valid = (segment_ids != 0)[:, :, None, None]
    q_f = jnp.where(valid, q_f, 0.0)
    k_f = jnp.where(valid, k_f, 0.0)
    v_f = jnp.where(valid, v_f, 0.0)

  decay = jnp.exp(g_f)  # [B, S, H]; 0.0 at every document start and padding token.
  bsz, _, num_heads, d_k = q_f.shape
  d_v = v_f.shape[-1]
  init_state = jnp.zeros((bsz, num_heads, d_k, d_v), dtype=jnp.float32)

  # Scan along sequence axis S.
  xs = (
      jnp.swapaxes(q_f, 0, 1),
      jnp.swapaxes(k_f, 0, 1),
      jnp.swapaxes(v_f, 0, 1),
      jnp.swapaxes(decay, 0, 1),
  )

  def _step(state: jax.Array, step_inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
    q_t, k_t, v_t, decay_t = step_inputs
    kv_t = jnp.einsum("bhk,bhv->bhkv", k_t, v_t)
    new_state = state * decay_t[:, :, None, None] + kv_t
    out_t = jnp.einsum("bhk,bhkv->bhv", q_t, new_state)
    return new_state, out_t

  _, outs = jax.lax.scan(_step, init_state, xs)
  return jnp.swapaxes(outs, 0, 1).astype(orig_dtype)
