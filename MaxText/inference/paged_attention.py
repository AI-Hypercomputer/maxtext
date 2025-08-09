#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Paged Attention Op

WARNING: THIS FILE IS A WORK IN PROGRESS.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx

from MaxText.inference import page_manager
from MaxText.inference import paged_attention_kernel_v2
from MaxText.common_types import Array, DType, AxisNames, BATCH, LENGTH, HEAD, D_KV, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText.layers.initializers import variable_to_logically_partitioned

_use_kernel_v2 = False


def paged_attention_op_as_linen(
    *,
    mesh: Mesh,
    num_pages: int,
    tokens_per_page: int,
    max_pages_per_slot: int,
    max_pages_per_prefill: int,
    pages_per_compute_block: int,
    num_kv_heads: int,
    kv_head_dim_size: int,
    dtype: DType = jnp.float32,
    attn_logits_soft_cap: float | None = None,
    query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
    kv_pages_axis_names: AxisNames = ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size"),
):
  """A factory function to create a PagedAttentionOp as a Linen module.

  This function serves as a bridge to use the NNX-based `PagedAttentionOp`
  within a Linen model. It wraps the `PagedAttentionOp` module using
  `nnx.bridge.to_linen`, making it compatible with the Linen API. This is
  useful for gradual migration of a codebase from Linen to NNX.

  Args:
    mesh: The device mesh for sharding.
    num_pages: The total number of pages in the KV cache.
    tokens_per_page: The number of tokens each page can hold.
    max_pages_per_slot: The maximum number of pages a single sequence can use.
    max_pages_per_prefill: The maximum number of pages for a prefill sequence.
    pages_per_compute_block: The number of pages processed in one kernel block.
    num_kv_heads: The number of key/value heads.
    kv_head_dim_size: The dimension of each key/value head.
    dtype: The data type for computations.
    attn_logits_soft_cap: The soft cap for attention logits.
    query_axis_names: The logical axis names for the query tensor.
    kv_pages_axis_names: The logical axis names for the KV cache pages.

  Returns:
    A Linen module that wraps the NNX `PagedAttentionOp` module.
  """

  return nnx.bridge.to_linen(
      PagedAttentionOp,
      mesh=mesh,
      num_pages=num_pages,
      tokens_per_page=tokens_per_page,
      max_pages_per_slot=max_pages_per_slot,
      max_pages_per_prefill=max_pages_per_prefill,
      pages_per_compute_block=pages_per_compute_block,
      num_kv_heads=num_kv_heads,
      kv_head_dim_size=kv_head_dim_size,
      dtype=dtype,
      attn_logits_soft_cap=attn_logits_soft_cap,
      query_axis_names=query_axis_names,
      kv_pages_axis_names=kv_pages_axis_names,
      metadata_fn=variable_to_logically_partitioned,
  )


class PagedAttentionOp(nnx.Module):
  """An NNX module for paged attention.

  This module implements the paged attention mechanism, which is an efficient
  method for handling attention in autoregressive models with long sequences.
  It divides the KV cache into fixed-size "pages" to manage memory dynamically.
  """

  def __init__(
      self,
      mesh: Mesh,
      num_pages: int,
      tokens_per_page: int,
      max_pages_per_slot: int,
      max_pages_per_prefill: int,
      pages_per_compute_block: int,
      num_kv_heads: int,
      kv_head_dim_size: int,
      dtype: DType = jnp.float32,
      attn_logits_soft_cap: float | None = None,
      query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
      kv_pages_axis_names: AxisNames = ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size"),
      *,
      # Not used in Embed but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs,
  ):
    """Initializes the PagedAttentionOp module.

    Args:
      mesh: The device mesh for sharding.
      num_pages: The total number of pages in the KV cache.
      tokens_per_page: The number of tokens each page can hold.
      max_pages_per_slot: The maximum number of pages a single sequence can use.
      max_pages_per_prefill: The maximum number of pages for a prefill sequence.
      pages_per_compute_block: The number of pages processed in one kernel block.
      num_kv_heads: The number of key/value heads.
      kv_head_dim_size: The dimension of each key/value head.
      dtype: The data type for computations.
      attn_logits_soft_cap: The soft cap for attention logits.
      query_axis_names: The logical axis names for the query tensor.
      kv_pages_axis_names: The logical axis names for the KV cache pages.
      rngs: The random number generators for initialization (required by NNX).
    """

    self.mesh = mesh
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_pages_per_slot = max_pages_per_slot
    self.max_pages_per_prefill = max_pages_per_prefill
    self.pages_per_compute_block = pages_per_compute_block
    self.num_kv_heads = num_kv_heads
    self.kv_head_dim_size = kv_head_dim_size
    self.dtype = dtype
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.query_axis_names = query_axis_names
    self.kv_pages_axis_names = kv_pages_axis_names

    self.kv_pages_shape = (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size)

    self.key_pages = nnx.Cache(
        jnp.zeros(self.kv_pages_shape, dtype=self.dtype),
        sharding=self.kv_pages_axis_names,
    )
    self.value_pages = nnx.Cache(
        jnp.zeros(self.kv_pages_shape, dtype=self.dtype),
        sharding=self.kv_pages_axis_names,
    )

  def _maybe_materialize_cache(self, cache: nnx.Cache) -> nnx.Cache:
    """Materializes the cache if it's currently a ShapeDtypeStruct."""
    if isinstance(cache.value, jax.ShapeDtypeStruct):
      # This is needed because the Linen bridge lazily creates this state. We
      # need to ensure the cache state is accessible at runtime.
      # TODO: Delete this function when the to_linen bridge is no longer needed.
      return nnx.Cache(
          jnp.zeros(self.kv_pages_shape, dtype=self.dtype),
          sharding=cache.sharding,
      )
    return cache

  def get_kv_pages(self):
    """Retrieves the key and value page caches.

    This method ensures the KV cache pages are materialized (if they are abstract
    ShapeDtypeStructs, a temporary state during Linen bridge initialization) and
    applies the necessary sharding constraints.

    Returns:
      A tuple containing the key pages and value pages caches (`nnx.Cache`).
    """

    # TODO: Remove once to_linen bridge is no longer needed
    self.key_pages = self._maybe_materialize_cache(self.key_pages)
    self.value_pages = self._maybe_materialize_cache(self.value_pages)

    self.key_pages.value = nn.with_logical_constraint(self.key_pages.value, self.kv_pages_axis_names)
    self.value_pages.value = nn.with_logical_constraint(self.value_pages.value, self.kv_pages_axis_names)
    return self.key_pages, self.value_pages

  def pad_qkv(self, *qkv):
    """Pad input to kv_head_dim_size"""

    def pad_to_kv_head_dim_size(x):
      if x.shape[-1] != self.kv_head_dim_size:
        return jnp.pad(
            x,
            ((0, 0), (0, 0), (0, 0), (0, self.kv_head_dim_size - x.shape[-1])),
            mode="constant",
            constant_values=0.0,
        )
      else:
        return x

    # Align Q, K, V to the same head dim. This is required by the kernel.
    return tuple(pad_to_kv_head_dim_size(x) for x in qkv)

  def paged_dot_product_attention_with_max_and_sum(self, query, key, value):
    """paged dot product attention with max & sum"""
    b, t, n, d = query.shape
    _, s, n_kv, _ = key.shape
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))

    attn_weights = jnp.einsum("btkgd,bskd->bkgts", query, key)

    causal_mask = jnp.triu(jnp.ones((t, s)), k=1)
    causal_mask = jnp.reshape(causal_mask, (1, 1, 1, t, s))
    masked_weights = jnp.where(causal_mask, jnp.full_like(attn_weights, -1e10), attn_weights)

    local_max = jnp.max(masked_weights, axis=-1, keepdims=True)
    local_exps = jnp.exp(masked_weights - local_max)
    local_sums = jnp.sum(local_exps, axis=-1, keepdims=True)

    attn = jnp.einsum("bkgts,bskd->btkgd", local_exps, value)
    attn = jnp.reshape(attn, (b, t, n, d))

    local_max = jnp.moveaxis(local_max, -2, 1)
    local_max = jnp.reshape(local_max, (b, t, n, 1))

    local_sums = jnp.moveaxis(local_sums, -2, 1)
    local_sums = jnp.reshape(local_sums, (b, t, n, 1))

    return attn, local_max, local_sums

  # TODO(rupliu): add sharding when SPMD is fully supported
  def paged_attention_v2_prefill(
      self,
      query: Array,
      key_pages_cache: nnx.Cache,
      value_pages_cache: nnx.Cache,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply ragged input Paged Attention in prefill only. The assumption
    is the batch_size is only 1
    """
    assert query.shape[0] == 1  # ensure the batch size is 0
    # shape of key_pages_cache.value is [num_kv_heads, num_pages, tokens_per_page, head_dim]
    k_p = jnp.permute_dims(key_pages_cache.value, (1, 2, 0, 3))
    v_p = jnp.permute_dims(value_pages_cache.value, (1, 2, 0, 3))
    c_q_l = jnp.array([0, page_state.sequence_lengths[0]])  # [0, prefill_true_length]
    num_seqs = jnp.array([1])
    query = query[0]  # [batch_size, max_num_tokens, num_kv_heads, head_dim] to [max_num_tokens, num_kv_heads, head_dim]
    result = paged_attention_kernel_v2.ragged_paged_attention(
        q=query,
        k_pages=k_p,  # [total_num_pages, page_size, num_kv_heads, head_dim]
        v_pages=v_p,
        kv_lens=jnp.array([query.shape[0]]),  # max_prefill_length
        cu_q_lens=c_q_l,  # the accumulative real lengths of requests, starting from 0
        page_indices=page_state.page_map,
        num_seqs=num_seqs,
        # TODO(rupliu) debug: repeated response when enabled below
        # num_kv_pages_per_block=self.pages_per_compute_block,
    )
    return jnp.expand_dims(result, axis=0)  # [batch_size, seq_len, n_kv_head, head_dim] and batch_size is 1 for now

  # TODO(rupliu): add sharding when SPMD is fully supported
  def paged_attention_v2_decode(
      self,
      query: Array,
      key_pages_cache: nnx.Cache,
      value_pages_cache: nnx.Cache,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply ragged input Paged Attention in decode only."""
    batch_size = query.shape[0]
    query = jnp.squeeze(query, axis=1)  # [batch_size, seq_len, n_kv_head, head_dim] to [batch_size, n_kv_head, head_dim]
    k_p = jnp.permute_dims(key_pages_cache.value, (1, 2, 0, 3))
    v_p = jnp.permute_dims(value_pages_cache.value, (1, 2, 0, 3))
    c_q_l = jnp.arange(batch_size + 1)  # one token per sequence
    num_seqs = jnp.array([batch_size])  # real number of requests, set it to batch_size
    result = paged_attention_kernel_v2.ragged_paged_attention(
        q=query,  # [max_batched_num_tokens, num_kv_heads, head_dim]
        k_pages=k_p,  # [total_num_pages, page_size, num_kv_heads, head_dim]
        v_pages=v_p,  # [total_num_pages, page_size, num_kv_heads, head_dim]
        kv_lens=page_state.sequence_lengths,  # [max_num_seqs]
        cu_q_lens=c_q_l,  # [max_num_seqs+1]
        page_indices=page_state.page_map,  # [max_num_seqs, pages_per_seq]
        num_seqs=num_seqs,
        num_kv_pages_per_block=self.pages_per_compute_block,
    )
    return jnp.expand_dims(
        result, axis=1
    )  # [batch_size, n_kv_head, head_dim] to [batch_size, seq_len, n_kv_head, head_dim]

  # v1 kernel has around 20% performance gain than v2 kernel in decode only task
  def paged_attention_v1_decode(
      self,
      query: Array,
      key_pages_cache: nnx.Cache,
      value_pages_cache: nnx.Cache,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply Paged Attention v1 in decode only."""
    kv_pages_pspec = nn.logical_to_mesh_axes(("paged_kv_heads", None, None, None))
    q_pspec = nn.logical_to_mesh_axes((None, None, "paged_kv_heads", None))

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            q_pspec,
            kv_pages_pspec,
            kv_pages_pspec,
            P(None),
            P(None, None),
            None,
        ),
        out_specs=q_pspec,
        check_rep=False,
    )
    def wrap_paged_attention(q, k_pages, v_pages, lengths, page_indices, pages_per_compute_block):
      q = jnp.squeeze(q, axis=1)
      result = paged_attention_kernel.paged_attention(
          q=q,  # [batch_size, num_heads, head_dim]
          k_pages=k_pages,
          v_pages=v_pages,
          lengths=lengths,
          page_indices=page_indices,
          pages_per_compute_block=pages_per_compute_block,
      )
      return jnp.expand_dims(result, axis=1)  # [batch_size, n_kv_head, head_dim] to [batch_size, 1, n_kv_head, head_dim]

    return wrap_paged_attention(
        query,
        key_pages_cache.value,
        value_pages_cache.value,
        page_state.sequence_lengths,
        page_state.page_map,
        self.pages_per_compute_block,
    )

  def __call__(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
  ):
    """Applies the paged attention mechanism.

    This is the main entry point for the module. It takes query, key, and value
    tensors and performs paged attention based on the current model mode
    (prefill or autoregressive).

    Args:
      query: The query tensor.
      key: The key tensor for the current step.
      value: The value tensor for the current step.
      decoder_segment_ids: Segment IDs for the decoder, used for masking.
      model_mode: The current operational mode, either 'prefill' or
        'autoregressive'.
      previous_chunk: Information about previously processed chunks, used for
        chunked prefill.
      slot: The batch slot index for the current request.
      page_state: The current state of the page manager.

    Returns:
        A tuple (output, exponentials_max, exponentials_sum) containing:
        - The attention output tensor.
        - The max of the exponentials (for prefill mode with dot-product attention).
        - The sum of the exponentials (for prefill mode with dot-product attention).
        The latter two are None for autoregressive mode, as this is handled
        internally by the paged attention kernel.
    """

    key_pages_cache, value_pages_cache = self.get_kv_pages()
    query, key, value = self.pad_qkv(query, key, value)

    # update kv pages and call page attention kernel
    if model_mode == MODEL_MODE_PREFILL:
      self.update_prefill_step_pages(key_pages_cache, value_pages_cache, key, value, slot, page_state)
      if _use_kernel_v2:
        return self.paged_attention_v2_prefill(query, key_pages_cache, value_pages_cache, page_state), None, None
      return self.paged_dot_product_attention_with_max_and_sum(query, key, value)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE and page_state is not None:
      self.update_decode_step_pages(key_pages_cache, value_pages_cache, key, value, page_state)
      if _use_kernel_v2:
        return self.paged_attention_v2_decode(query, key_pages_cache, value_pages_cache, page_state), None, None
      return self.paged_attention_v1_decode(query, key_pages_cache, value_pages_cache, page_state), None, None
    else:
      raise NotImplementedError(model_mode)

  def update_prefill_step_pages(
      self,
      key_pages_cache: nnx.Cache,  # [num_kv_heads, num_pages, tokens_per_page, head_dim]
      value_pages_cache: nnx.Cache,
      key: Array,
      value: Array,
      slot: int,
      page_state: page_manager.PageState,
  ) -> None:
    """Update pages for prefill step."""
    assert (
        key.shape == value.shape
    ), f"prefill_step key/value should have the same shape, but getting {key.shape=} and {value.shape=} instead"
    batch_size, seq_len, n_kv_head, head_dim = key.shape
    assert seq_len % self.tokens_per_page == 0, f"seq_length {seq_len} and  tokens_per_page {self.tokens_per_page}"
    assert key_pages_cache.value.shape == value_pages_cache.value.shape, (
        f"prefill_step key/value_pages_cache should have the same shape, but "
        f"getting {key_pages_cache.shape=} and {value_pages_cache.shape=} instead"
    )

    v_n_kv, _, v_p, v_d = key_pages_cache.value.shape
    assert v_n_kv == n_kv_head, f"{v_n_kv=} {n_kv_head=}"
    assert v_p == self.tokens_per_page, f"{v_p=} {self.tokens_per_page=}"
    assert v_d == head_dim, f"{v_d=} {head_dim=}"
    assert page_state.page_map.shape == (page_state.num_pages_used.shape[0], self.max_pages_per_slot)

    # Handle both init (b>1) and runtime (b=1) cases
    if batch_size == 1:
      key = jnp.squeeze(key)  # [batch_size, seq_len, n_kv_head, head_dim] to [seq_len, n_kv_head, head_dim]
      value = jnp.squeeze(value)
    else:
      key = key[0]
      value = value[0]

    key = jnp.transpose(key, axes=(1, 0, 2))
    value = jnp.transpose(value, axes=(1, 0, 2))

    key = jnp.reshape(key, shape=(n_kv_head, max(1, seq_len // self.tokens_per_page), self.tokens_per_page, head_dim))
    value = jnp.reshape(value, shape=(n_kv_head, max(1, seq_len // self.tokens_per_page), self.tokens_per_page, head_dim))

    key_pages_cache.value = nn.with_logical_constraint(key, self.kv_pages_axis_names)
    value_pages_cache.value = nn.with_logical_constraint(value, self.kv_pages_axis_names)

  def update_decode_step_pages(self, key_pages_cache, value_pages_cache, key, value, page_state):
    """Update decode-step pages"""
    key_pages = key_pages_cache.value
    value_pages = value_pages_cache.value

    batch_size, _, kv_heads, head_dim = key.shape
    kv_heads, _, _, head_dim = key_pages.shape

    new_key = key.reshape(batch_size, kv_heads, head_dim)[:, :, :]
    new_key = jnp.transpose(new_key, (1, 0, 2))  # [n_kv_heads, batch_size, head_dim]
    new_value = value.reshape(batch_size, kv_heads, head_dim)[:, :, :]
    new_value = jnp.transpose(new_value, (1, 0, 2))  # [n_kv_heads, batch_size, head_dim]

    broadcast_pages = jnp.tile(page_state.active_page, (kv_heads, 1))  # [n_kv_heads, batch_size]
    broadcast_pos = jnp.tile(page_state.active_page_position, (kv_heads, 1))  # [n_kv_heads, batch_size]

    kv_indices = jnp.arange(kv_heads)[:, None]  # [n_kv_heads, 1]
    kv_indices = jnp.tile(kv_indices, (1, batch_size))  # [n_kv_heads, batch_size]

    # [num_kv_heads, num_pages, tokens_per_page, head_dim]
    key_pages_updated = key_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_key)
    value_pages_updated = value_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_value)

    key_pages_cache.value = key_pages_updated
    value_pages_cache.value = value_pages_updated
    return key_pages_cache, value_pages_cache
