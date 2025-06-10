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
from typing import Optional

import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.inference import page_manager
from MaxText.inference import paged_attention_kernel_v2
from MaxText.common_types import Array, DType, AxisNames, BATCH, LENGTH, HEAD, D_KV, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE

_use_kernel_v2 = False


class PagedAttentionOp(nn.Module):
  """paged-attention op"""

  mesh: Mesh
  num_pages: int
  tokens_per_page: int
  max_pages_per_slot: int
  max_pages_per_prefill: int
  pages_per_compute_block: int

  num_kv_heads: int
  kv_head_dim_size: int
  dtype: DType = jnp.float32
  attn_logits_soft_cap: float | None = None

  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  kv_pages_axis_names: AxisNames = ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size")

  def init_or_get_kv_pages(self, model_mode: str):
    """Get paged attention op."""
    # Get existing variables if they exist
    if self.has_variable("cache", "key_pages"):
      key_pages_var = self.variable("cache", "key_pages")
      value_pages_var = self.variable("cache", "value_pages")

      required_ar_shape = (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size)
      # For AR mode, if shape doesn't match, reinitialize values but not variables
      if key_pages_var.value.shape != required_ar_shape:
        key_pages_var.value = jnp.zeros(required_ar_shape, dtype=self.dtype)
        value_pages_var.value = jnp.zeros(required_ar_shape, dtype=self.dtype)
    else:
      kv_pages_shape = (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size)
      key_pages_var = self.variable(
          "cache",
          "key_pages",
          nn.with_logical_partitioning(jnp.zeros, self.kv_pages_axis_names),
          kv_pages_shape,
          self.dtype,
      )
      value_pages_var = self.variable(
          "cache",
          "value_pages",
          nn.with_logical_partitioning(jnp.zeros, self.kv_pages_axis_names),
          kv_pages_shape,
          self.dtype,
      )
    # Apply logical constraints
    key_pages_var.value = nn.with_logical_constraint(key_pages_var.value, self.kv_pages_axis_names)
    value_pages_var.value = nn.with_logical_constraint(value_pages_var.value, self.kv_pages_axis_names)
    return key_pages_var, value_pages_var

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
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply ragged input Paged Attention in prefill only. The assumption
    is the batch_size is only 1
    """
    assert query.shape[0] == 1  # ensure the batch size is 0
    # shape of key_pages_var.value is [num_kv_heads, num_pages, tokens_per_page, head_dim]
    k_p = jnp.permute_dims(key_pages_var.value, (1, 2, 0, 3))
    v_p = jnp.permute_dims(value_pages_var.value, (1, 2, 0, 3))
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
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply ragged input Paged Attention in decode only."""
    batch_size = query.shape[0]
    query = jnp.squeeze(query, axis=1)  # [batch_size, seq_len, n_kv_head, head_dim] to [batch_size, n_kv_head, head_dim]
    k_p = jnp.permute_dims(key_pages_var.value, (1, 2, 0, 3))
    v_p = jnp.permute_dims(value_pages_var.value, (1, 2, 0, 3))
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
    return jnp.expand_dims(result, axis=1)  # [batch_size, n_kv_head, head_dim] to [batch_size, seq_len, n_kv_head, head_dim]

  # v1 kernel has around 20% performance gain than v2 kernel in decode only task
  def paged_attention_v1_decode(
      self,
      query: Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
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
        key_pages_var.value,
        value_pages_var.value,
        page_state.sequence_lengths,
        page_state.page_map,
        self.pages_per_compute_block,
    )

  @nn.compact
  def __call__(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      previous_chunk=None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ):
    """Apply paged attention mechanism.

    Returns:
        tuple: (output, exponentials_max, exponentials_sum) where the latter two
              are None for autoregressive mode (handled by paged_attention kernel)
    """
    key_pages_var, value_pages_var = self.init_or_get_kv_pages(model_mode)

    # update kv pages and call page attention kernel
    if model_mode == MODEL_MODE_PREFILL:
      self.update_prefill_step_pages(key_pages_var, value_pages_var, key, value, slot, page_state)
      if _use_kernel_v2:
        return self.paged_attention_v2_prefill(query, key_pages_var, value_pages_var, page_state), None, None
      return self.paged_dot_product_attention_with_max_and_sum(query, key, value)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE and page_state is not None:
      self.update_decode_step_pages(key_pages_var, value_pages_var, key, value, page_state)
      if _use_kernel_v2:
        return self.paged_attention_v2_decode(query, key_pages_var, value_pages_var, page_state), None, None
      return self.paged_attention_v1_decode(query, key_pages_var, value_pages_var, page_state), None, None
    else:
      raise NotImplementedError(model_mode)

  def update_prefill_step_pages(
      self,
      key_pages_var: nn.Variable,  # [num_kv_heads, num_pages, tokens_per_page, head_dim]
      value_pages_var: nn.Variable,
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
    assert key_pages_var.value.shape == value_pages_var.value.shape, (
        f"prefill_step key/value_pages_var should have the same shape, but "
        f"getting {key_pages_var.shape=} and {value_pages_var.shape=} instead"
    )

    v_n_kv, _, v_p, v_d = key_pages_var.value.shape
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

    key_pages_var.value = nn.with_logical_constraint(key, self.kv_pages_axis_names)
    value_pages_var.value = nn.with_logical_constraint(value, self.kv_pages_axis_names)

  def update_decode_step_pages(self, key_pages_var, value_pages_var, key, value, page_state):
    """Update decode-step pages"""
    key_pages = key_pages_var.value
    value_pages = value_pages_var.value

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

    key_pages_var.value = key_pages_updated
    value_pages_var.value = value_pages_updated
    return key_pages_var, value_pages_var
