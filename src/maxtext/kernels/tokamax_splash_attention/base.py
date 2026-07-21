# pylint: skip-file
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base functionality for Sparse Flash Attention."""

import functools
from typing import Final, NamedTuple, TypeAlias
import jax
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info as mask_info_lib


MaskInfo = mask_info_lib.MaskInfo


DEFAULT_MASK_VALUE: Final[float] = -0.7 * float(np.finfo(np.dtype("float32")).max)


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. For causal self-attention, segment ids form a
  block diagonal matrix so at least one element in each row is set.
  Non-self-attention configurations can have all-masked rows; Splash treats
  those rows as zero-output rows with mask-value logsumexp.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array | jax.sharding.PartitionSpec  # [q_seq_len]
  kv: jax.Array | jax.sharding.PartitionSpec  # [kv_seq_len]


# Return type of SplashAttention function that implements the custom vjp rule.
SplashCustomReturnType: TypeAlias = jax.Array | tuple[jax.Array, dict[str, jax.Array]]

SplashResidualsType = tuple[
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    SegmentIds | None,  # segment_ids
    jax.Array | None,  # sinks
    jax.Array,  # out
    jax.Array,  # logsumexp
    MaskInfo | None,  # dkv_mask_info
]


def _attention_reference_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    save_residuals: bool,
    attn_logits_soft_cap: float | None,
) -> SplashCustomReturnType:
  logits = jnp.einsum("sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32))

  if segment_ids is not None:
    mask = jnp.logical_and(mask, segment_ids.q[:, None] == segment_ids.kv[None, :])

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap

  if sinks is not None:
    assert sinks.shape == ()  # should already be vmapped

  logits = jnp.where(mask, logits, mask_value)
  m = logits.max(axis=-1)
  sinks = None if sinks is None else sinks.astype(logits.dtype)
  m = m if sinks is None else jnp.maximum(m, sinks)
  s = jnp.exp(logits - m[..., None])
  s = jnp.where(mask, s, 0.0)
  l = s.sum(axis=-1) + (0 if sinks is None else jnp.exp(sinks - m))
  safe_l = jnp.where(l == 0.0, 1.0, l)
  p = s / safe_l[..., None]

  o = jnp.einsum("st,td->sd", p, v.astype(jnp.float32))

  if save_residuals:
    logsumexp = jnp.where(l == 0.0, mask_value, m + jnp.log(safe_l))
    return o, {"logsumexp": logsumexp, "max_logits": m}
  return o


def _attention_reference_custom_bwd(
    do,
    q,
    k,
    v,
    mask,
    segment_ids,
    sinks,
    o,
    logsumexp,
    mask_value: float = DEFAULT_MASK_VALUE,
    backward_impl: str = "vanilla",
    attn_logits_soft_cap: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None, jax.Array | None]:
  uncapped_logits = jnp.einsum("qc,kc->qk", q, k, preferred_element_type=jnp.float32)

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap
  else:
    logits = uncapped_logits

  if segment_ids is not None:
    mask = jnp.logical_and(mask, segment_ids.q[:, None] == segment_ids.kv[None, :])
  logits = jnp.where(mask, logits, mask_value)

  p = jnp.exp(logits - logsumexp[..., None])
  p = jnp.where(mask, p, 0.0)
  do = do.astype(jnp.float32)  # pytype: disable=attribute-error
  dv = jnp.einsum("pt,pd->td", p, do).astype(v.dtype)
  dp = jnp.einsum("pd,td->pt", do, v.astype(jnp.float32))

  # These two ways of computing ds are mathematically equivalent. The first
  # involves reducing over the head_dim dimension and the second involves
  # reducing over a sequence dimension. They tend to produce slightly different
  # numerics.
  if backward_impl == "flash":
    di = jnp.sum(o.astype(jnp.float32) * do, axis=-1)[..., None]
  else:
    di = jnp.einsum("st,st->s", dp, p)[:, None]
  ds = (dp - di) * p
  if attn_logits_soft_cap is not None:
    normalized = uncapped_logits / attn_logits_soft_cap
    d = jnp.tanh(normalized)
    g = ds * (1 - d)
    ds = g + g * d
  dk = jnp.einsum("sd,st->td", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("st,td->sd", ds, k.astype(jnp.float32)).astype(q.dtype)
  dsinks = None
  if sinks is not None:
    sinks_exp = -jnp.exp(sinks[..., None, None].astype(jnp.float32) - logsumexp[..., None].astype(jnp.float32))
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))
  return dq, dk, dv, None, None, dsinks


@functools.partial(
    jax.jit,
    static_argnames=[
        "mask_value",
        "save_residuals",
        "attn_logits_soft_cap",
        "is_mqa",
    ],
)
def attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    attn_logits_soft_cap: float | None = None,
):
  """A JIT-compiled reference implementation of attention, handles MQA and MHA."""
  attn_impl = functools.partial(
      _attention_reference_impl,
      mask_value=mask_value,
      save_residuals=save_residuals,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )

  if is_mqa:
    func = jax.vmap(attn_impl, in_axes=(0, None, None, None, None, 0))
  else:
    # In grouped attention (1 < num_kv_heads && num_kv_heads < num_q_heads).
    # We interleave the KV heads across the Q heads.
    # For example: for 8 Q heads and 4 KV heads:
    # Q head [0, 1] see KV head 0
    # Q head [2, 3] see KV head 1
    # Q head [4, 5] see KV head 2
    # Q head [6, 7] see KV head 3

    kv_heads, q_heads = k.shape[0], q.shape[0]
    assert q_heads % kv_heads == 0

    if kv_heads < q_heads:
      # Repeat K and V heads to match the number of Q heads.
      q_heads_per_kv = q_heads // kv_heads
      k = jnp.repeat(k, repeats=q_heads_per_kv, axis=0)
      v = jnp.repeat(v, repeats=q_heads_per_kv, axis=0)

    func = jax.vmap(attn_impl, in_axes=(0, 0, 0, None, None, 0))

  out = func(q, k, v, mask, segment_ids, sinks)
  return out


@functools.partial(jax.jit, static_argnames=["is_mqa", "backward_impl", "attn_logits_soft_cap"])
def attention_reference_vjp(
    do,
    q,
    k,
    v,
    mask,
    segment_ids,
    sinks,
    o,
    logsumexp,
    *,
    is_mqa: bool,
    backward_impl: str = "vanilla",
    attn_logits_soft_cap: float | None = None,
):
  """Wrapper for backward reference that handles GQA/MQA broadcasting and reduction."""
  bwd = functools.partial(
      _attention_reference_custom_bwd,
      backward_impl=backward_impl,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )

  num_q_heads = q.shape[0]
  num_kv_heads = 1 if is_mqa else k.shape[0]

  is_grouped = not is_mqa and num_kv_heads < num_q_heads
  assert num_q_heads % num_kv_heads == 0
  head_multiplier = num_q_heads // num_kv_heads
  if is_mqa:
    bwd = jax.vmap(bwd, in_axes=(0, 0, None, None, None, None, 0, 0, 0))
  else:
    bwd = jax.vmap(bwd, in_axes=(0, 0, 0, 0, None, None, 0, 0, 0))
    # Interleave the KV heads to match the corresponding Q heads.
    if is_grouped:
      k = jnp.repeat(k, head_multiplier, axis=0)
      v = jnp.repeat(v, head_multiplier, axis=0)

  dq, dk, dv, _, _, dsinks = bwd(do, q, k, v, mask, segment_ids, sinks, o, logsumexp)

  if is_mqa:
    dk, dv = dk.sum(axis=0), dv.sum(axis=0)
  elif is_grouped:
    # Perform the sum reduction across the head_multiplier dimension only.
    # So that the output still has KV heads.
    dk = dk.reshape(num_kv_heads, head_multiplier, *dk.shape[1:])
    dv = dv.reshape(num_kv_heads, head_multiplier, *dv.shape[1:])
    dk, dv = dk.sum(axis=1), dv.sum(axis=1)

  return dq, dk, dv, dsinks
