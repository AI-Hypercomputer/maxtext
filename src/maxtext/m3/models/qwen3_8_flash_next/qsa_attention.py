# Copyright 2026 Google LLC
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

"""Qwen Sparse Attention (QSA), Indexer, and mRoPE for Qwen3.8-Flash-Next."""

from collections.abc import Sequence
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
# pylint: disable=no-name-in-module,too-many-positional-arguments
from maxtext.m3.models.qwen3_8_flash_next import hyper_connection


def rotate_half(x: jax.Array) -> jax.Array:
  """Rotates half the hidden dimensions of the input."""
  return jnp.concatenate((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), axis=-1)


def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array | None = None,
    cos: jax.Array | None = None,
    sin: jax.Array | None = None,
    unsqueeze_dim: int = 1,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Applies Rotary Position Embedding to query and optionally key tensors."""
  if cos is None or sin is None:
    raise ValueError("cos and sin must be provided to apply_rotary_pos_emb.")
  cos, sin = jnp.expand_dims(cos, axis=unsqueeze_dim), jnp.expand_dims(sin, axis=unsqueeze_dim)
  d = cos.shape[-1]

  def _rot(x: jax.Array) -> jax.Array:
    return jnp.concatenate([(x[..., :d] * cos) + (rotate_half(x[..., :d]) * sin), x[..., d:]], axis=-1)

  return _rot(q) if k is None else (_rot(q), _rot(k))


class Qwen3_8RotaryEmbedding(nnx.Module):
  """Partial interleaved mRoPE for Qwen3.8-Flash-Next."""

  def __init__(
      self,
      head_dim: int,
      partial_rotary_factor: float = 0.25,
      rope_theta: float = 10000000.0,
      mrope_section: Sequence[int] = (11, 11, 10),
  ):
    self.mrope_section = tuple(mrope_section)
    dim = int(head_dim * partial_rotary_factor)
    if len(self.mrope_section) != 3 or sum(self.mrope_section) != dim // 2:
      raise ValueError(
          f"Expected 3 mrope_section values summing to {dim // 2} (rotary dim {dim} / 2), got {self.mrope_section}."
      )
    self.rotary_dim = dim
    self.rope_theta = float(rope_theta)

  @property
  def inv_freq(self) -> jax.Array:
    """Returns inverse frequencies for rotary position embedding."""
    return 1.0 / (self.rope_theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim))

  def apply_interleaved_mrope(self, freqs: jax.Array, mrope_section: Sequence[int]) -> jax.Array:
    """Interleaves multimodal frequency components."""
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
      idx = slice(offset, mrope_section[dim] * 3, 3)
      freqs_t = freqs_t.at[..., idx].set(freqs[dim, ..., idx])
    return freqs_t

  def __call__(self, x: jax.Array, position_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Generates cos and sin embeddings for input positions."""
    if position_ids.ndim == 2:
      position_ids = jnp.broadcast_to(position_ids[None, ...], (3, *position_ids.shape))
    freqs = position_ids[:, :, :, None].astype(jnp.float32) * self.inv_freq[None, None, None, :]
    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(x.dtype), jnp.sin(emb).astype(x.dtype)


class Qwen3_8QSAIndexer(nnx.Module):
  """MQA micro-block indexer selecting top-k complete blocks for QSA."""

  def __init__(
      self,
      emb_dim: int = 2560,
      index_n_heads: int = 4,
      index_kv_heads: int = 1,
      index_head_dim: int = 128,
      token_budget: int = 2048,
      compress_ratio: int = 4,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    if index_kv_heads != 1:
      raise ValueError(f"Expected index_kv_heads == 1 (MQA indexer keys), got {index_kv_heads}.")
    if compress_ratio <= 0 or token_budget < compress_ratio:
      raise ValueError(
          "Expected compress_ratio > 0 and token_budget >= compress_ratio,"
          f" got compress_ratio={compress_ratio}, token_budget={token_budget}."
      )
    self.index_head_dim, self.index_n_heads = index_head_dim, index_n_heads
    self.compress_ratio, self.block_topk = compress_ratio, token_budget // compress_ratio

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.index_qk_proj = nnx.Linear(emb_dim, (index_n_heads + index_kv_heads) * index_head_dim, **kw)
    self.q_layernorm = hyper_connection.Qwen3_8ZeroCenteredRMSNorm(index_head_dim, eps=eps, rngs=rngs)
    self.k_layernorm = hyper_connection.Qwen3_8ZeroCenteredRMSNorm(index_head_dim, eps=eps, rngs=rngs)

  def __call__(
      self,
      hidden_states: jax.Array,
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Computes sparse token mask (B, 1, S, S) and document-local block indices (B, S, topk).

    Args:
      hidden_states: Input tensor of shape (batch, seq_len, emb_dim).
      position_embeddings: Tuple of (cos, sin) arrays for RoPE.
      attention_mask: Optional (B, S) mask; 0/False = padding, non-zero/True = active (NOT a PyTorch additive mask).
      segment_ids: Optional (B, S) 1-indexed document IDs (0 = padding); when both are given, padding from
        attention_mask is folded into segment_ids.

    Returns:
      Tuple of (token_mask, selected_block_indices).
    """
    if attention_mask is not None and attention_mask.ndim != 2:
      raise ValueError(f"Expected 2D (B, S) attention_mask or None, got shape {attention_mask.shape}.")

    b, s, _ = hidden_states.shape
    c = self.compress_ratio
    full_cos, full_sin = position_embeddings
    curr_cos, curr_sin = full_cos[:, -s:, :], full_sin[:, -s:, :]

    qk = self.index_qk_proj(hidden_states)
    q_dim = self.index_n_heads * self.index_head_dim
    q = self.q_layernorm(qk[..., :q_dim].reshape(b, s, self.index_n_heads, self.index_head_dim))
    q_rot = apply_rotary_pos_emb(q, cos=curr_cos, sin=curr_sin, unsqueeze_dim=2)
    assert isinstance(q_rot, jax.Array)
    raw_keys = qk[..., q_dim:].reshape(b, s, self.index_head_dim)

    if segment_ids is None:
      segment_ids = jnp.ones((b, s), dtype=jnp.int32)
    if attention_mask is not None:
      segment_ids = jnp.where(attention_mask != 0, segment_ids, 0)

    same_doc = (segment_ids[:, :, None] == segment_ids[:, None, :]) & (segment_ids[:, :, None] != 0)
    doc_start = jnp.argmax(same_doc, axis=-1)
    num_blocks = max(1, s // c)
    local_pos, end_t = jnp.arange(s)[None, :] - doc_start, jnp.clip(jnp.arange(s) + c - 1, 0, s - 1)
    is_block_start = (
        (segment_ids != 0)
        & ((local_pos % c) == 0)
        & ((jnp.arange(s) + c - 1 < s)[None, :])
        & (segment_ids[:, end_t] == segment_ids)
    )

    group_starts = jax.vmap(lambda m: jnp.nonzero(m, size=num_blocks, fill_value=0)[0])(is_block_start)
    block_valid = jnp.arange(num_blocks)[None, :] < jnp.sum(is_block_start.astype(jnp.int32), axis=-1, keepdims=True)
    block_seg_ids = jnp.take_along_axis(segment_ids, group_starts, axis=1)

    block_token_idx = jnp.clip(group_starts[:, :, None] + jnp.arange(c)[None, None, :], 0, s - 1)
    block_keys_raw = jax.vmap(lambda k, idx: k[idx])(raw_keys, block_token_idx)
    pooled_keys = self.k_layernorm(jnp.mean(block_keys_raw.astype(jnp.float32), axis=2).astype(raw_keys.dtype))

    cos_blk = jax.vmap(lambda cb, idx: cb[idx])(full_cos, group_starts)
    sin_blk = jax.vmap(lambda sb, idx: sb[idx])(full_sin, group_starts)
    res = apply_rotary_pos_emb(pooled_keys[:, :, None, :], cos=cos_blk, sin=sin_blk, unsqueeze_dim=2)
    assert isinstance(res, jax.Array)
    block_keys = res.squeeze(2)

    raw_scores = jnp.einsum("bthd,bmd->bthm", q_rot.astype(jnp.float32), block_keys.astype(jnp.float32))
    scores = jnp.sum(jax.nn.relu(raw_scores), axis=2) / jnp.sqrt(float(self.index_head_dim))

    valid_q_block = (
        block_valid[:, None, :]
        & (block_seg_ids[:, None, :] == segment_ids[:, :, None])
        & ((group_starts + c - 1)[:, None, :] <= jnp.arange(s)[None, :, None])
        & (segment_ids[:, :, None] != 0)
    )

    k_top = min(self.block_topk, num_blocks)
    top_vals, top_indices = jax.lax.top_k(jnp.where(valid_q_block, scores, -1e9), k_top)
    doc_local_idx = (jnp.take_along_axis(group_starts[:, None, :], top_indices, axis=2) - doc_start[:, :, None]) // c
    selected_block_indices = jnp.where(top_vals > -1e8, doc_local_idx, -1).astype(jnp.int32)
    if self.block_topk > k_top:
      selected_block_indices = jnp.pad(
          selected_block_indices, ((0, 0), (0, 0), (0, self.block_topk - k_top)), constant_values=-1
      )

    m_local = jnp.arange(num_blocks)[None, None, :]
    in_topk = jnp.any(
        (selected_block_indices[:, :, :, None] == m_local) & (selected_block_indices[:, :, :, None] >= 0), axis=2
    )
    ar_k, ar_q = jnp.arange(s)[None, None, :], jnp.arange(s)[None, :, None]
    is_selected = jnp.take_along_axis(in_topk, jnp.clip((ar_k - doc_start[:, :, None]) // c, 0, num_blocks - 1), axis=-1)
    in_tail = ar_k >= (doc_start + ((jnp.arange(s)[None, :] - doc_start + 1) // c) * c)[:, :, None]
    token_mask = (same_doc & (ar_k <= ar_q) & (is_selected | in_tail))[:, None, :, :]
    return token_mask, selected_block_indices


class Qwen3_8SparseAttention(nnx.Module):
  """Qwen Sparse Attention (QSA) with RMSNorm, gated output, and micro-block indexer."""

  def __init__(
      self,
      emb_dim: int = 2560,
      num_query_heads: int = 24,
      num_kv_heads: int = 2,
      head_dim: int = 256,
      partial_rotary_factor: float = 0.25,
      rope_theta: float = 10000000.0,
      mrope_section: Sequence[int] = (11, 11, 10),
      index_n_heads: int = 4,
      index_kv_heads: int = 1,
      index_head_dim: int = 128,
      token_budget: int = 2048,
      compress_ratio: int = 4,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    del sharding
    if num_query_heads % num_kv_heads != 0:
      raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
    self.head_dim, self.num_query_heads, self.num_kv_heads = head_dim, num_query_heads, num_kv_heads

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.q_proj = nnx.Linear(emb_dim, num_query_heads * head_dim * 2, **kw)
    self.k_proj = nnx.Linear(emb_dim, num_kv_heads * head_dim, **kw)
    self.v_proj = nnx.Linear(emb_dim, num_kv_heads * head_dim, **kw)
    self.o_proj = nnx.Linear(num_query_heads * head_dim, emb_dim, **kw)

    self.q_norm = hyper_connection.Qwen3_8ZeroCenteredRMSNorm(head_dim, eps=eps, rngs=rngs)
    self.k_norm = hyper_connection.Qwen3_8ZeroCenteredRMSNorm(head_dim, eps=eps, rngs=rngs)

    self.rotary_emb = Qwen3_8RotaryEmbedding(head_dim, partial_rotary_factor, rope_theta, mrope_section)
    self.indexer = Qwen3_8QSAIndexer(
        emb_dim,
        index_n_heads,
        index_kv_heads,
        index_head_dim,
        token_budget,
        compress_ratio,
        eps,
        dtype,
        weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      position_ids: jax.Array | None = None,
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_selected_blocks: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Forward pass computing sparse attention with document boundary support.

    Args:
      hidden_states: Input tensor of shape (batch, seq_len, emb_dim).
      position_ids: Optional (B, S) or (3, B, S) (T/H/W rows for interleaved mRoPE).
      attention_mask: Optional (B, S) mask; 0/False = padding, non-zero/True = active (NOT a PyTorch additive mask).
      segment_ids: Optional (B, S) 1-indexed document IDs (0 = padding); when both are given, padding from
        attention_mask is folded into segment_ids.
      return_selected_blocks: Whether to return selected block indices.

    Returns:
      Attention output tensor, or tuple of (output, selected_block_indices) if return_selected_blocks is True.
    """
    b, s, _ = hidden_states.shape
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))

    cos, sin = self.rotary_emb(hidden_states, position_ids)
    selected_token_mask, selected_block_indices = self.indexer(
        hidden_states, (cos, sin), attention_mask=attention_mask, segment_ids=segment_ids
    )

    q_and_gate = self.q_proj(hidden_states).reshape(b, s, self.num_query_heads, 2 * self.head_dim)
    query_states = self.q_norm(q_and_gate[..., : self.head_dim])
    gate = q_and_gate[..., self.head_dim :].reshape(b, s, self.num_query_heads * self.head_dim)
    key_states = self.k_norm(self.k_proj(hidden_states).reshape(b, s, self.num_kv_heads, self.head_dim))
    value_states = self.v_proj(hidden_states).reshape(b, s, self.num_kv_heads, self.head_dim)

    res = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin, unsqueeze_dim=2)
    assert isinstance(res, tuple)
    query_states, key_states = res

    if (rep := self.num_query_heads // self.num_kv_heads) > 1:
      key_states, value_states = jnp.repeat(key_states, rep, axis=2), jnp.repeat(value_states, rep, axis=2)

    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query_states, key_states) * (1.0 / jnp.sqrt(float(self.head_dim)))
    min_val = jnp.finfo(jnp.float32).min
    attn_weights = jnp.where(selected_token_mask, attn_weights, min_val)

    probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(value_states.dtype)
    attn_output = jnp.einsum("bhqk,bkhd->bqhd", probs, value_states).reshape(b, s, self.num_query_heads * self.head_dim)
    attn_output = (attn_output * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(attn_output.dtype)

    out = self.o_proj(attn_output)
    return (out, selected_block_indices) if return_selected_blocks else out
