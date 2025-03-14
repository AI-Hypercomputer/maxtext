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

"""Paged Attention Op for efficiently managing KV cache memory."""

import functools
from typing import Optional

from jax import lax
import common_types
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel
from jax.sharding import PartitionSpec as P

from inference import page_manager
from inference import paged_attention_kernel_v2

# pytype: disable=attribute-error

Mesh = common_types.Mesh

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
PRNGKey = common_types.PRNGKey

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV

shard_map = shard_map.shard_map
use_kernel_v2 = False


class PagedAttentionOp(nn.Module):
  """Paged attention operation for handling long sequences efficiently."""

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

        # For AR mode, if shape doesn't match, reinitialize values but not variables
        if model_mode != common_types.MODEL_MODE_PREFILL and key_pages_var.value.shape[1] != self.num_pages:
            kv_pages_shape = (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size)
            key_pages_var.value = jnp.zeros(kv_pages_shape, dtype=self.dtype)
            value_pages_var.value = jnp.zeros(kv_pages_shape, dtype=self.dtype)
    else:
      # Initial creation - choose size based on mode
      num_pages = self.max_pages_per_prefill if model_mode == common_types.MODEL_MODE_PREFILL else self.num_pages
      kv_pages_shape = (self.num_kv_heads, num_pages, self.tokens_per_page, self.kv_head_dim_size)
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
    """Simple implementation of dot product attention for prefill mode."""
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
      layer_idx: int = 0,  # Add layer_idx parameter
  ) -> Array:
    """Apply ragged input Paged Attention in prefill only."""
    k_p = jnp.permute_dims(key_pages_var.value, (1, 2, 0, 3))
    v_p = jnp.permute_dims(value_pages_var.value, (1, 2, 0, 3))
    # Use layer-specific sequence lengths
    c_q_l = jnp.array([0, page_state.sequence_lengths[layer_idx, 0]])
    num_seqs = jnp.array([1])
    query = query[0]  # [batch_size, max_num_tokens, num_kv_heads, head_dim] to [max_num_tokens, num_kv_heads, head_dim]
    result = paged_attention_kernel_v2.ragged_paged_attention(
        q=query,
        k_pages=k_p,  # [total_num_pages, page_size, num_kv_heads, head_dim]
        v_pages=v_p,
        kv_lens=jnp.array([query.shape[0]]),  # max_prefill_length
        cu_q_lens=c_q_l,
        # Use layer-specific page map
        page_indices=page_state.page_map[layer_idx],
        num_seqs=num_seqs,
        # num_kv_pages_per_block=self.pages_per_compute_block, #I saw repeated response when enabled this
    )
    return jnp.expand_dims(result, axis=0)

  # TODO(rupliu): add sharding when SPMD is fully supported
  def paged_attention_v2_decode(
      self,
      query: Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      layer_idx: int = 0,  # Add layer_idx parameter
  ) -> Array:
      """Apply ragged input Paged Attention in decode only."""
      print("paged_attention_v2_decode - START")
      batch_size, seq_len, num_kv_heads, head_dim = query.shape
      print(f"{batch_size=}, {seq_len=}, {num_kv_heads=}, {head_dim=}")

      # Handle both initialization (seq_len > 1) and actual decoding (seq_len = 1)
      if seq_len == 1:
          # Normal decoding case - squeeze the dimension
          query_input = jnp.squeeze(query, axis=1)
      else:
          # Initialization or prefill case - take the last token
          query_input = query[:, -1, :, :]
      print(f"{query_input.shape=}")

      k_p = jnp.permute_dims(key_pages_var.value, (1, 2, 0, 3))
      v_p = jnp.permute_dims(value_pages_var.value, (1, 2, 0, 3))
      c_q_l = jnp.arange(batch_size + 1)  # one token per sequence
      num_seqs = jnp.array([batch_size])  # real number of requests, set it to batch_size
      print(f"{k_p.shape=}, {v_p.shape=}")
      print(f"{page_state.sequence_lengths.shape=}, {page_state.sequence_lengths[layer_idx]=}")
      print(f"{c_q_l=}")
      print(f"{page_state.page_map.shape=}, {page_state.page_map[layer_idx]=}")
      print(f"{num_seqs=}")
      print(f"{self.pages_per_compute_block=}")

      result = paged_attention_kernel_v2.ragged_paged_attention(
          q=query_input,  # [max_batched_num_tokens, num_kv_heads, head_dim]
          k_pages=k_p,  # [total_num_pages, page_size, num_kv_heads, head_dim]
          v_pages=v_p,  # [total_num_pages, page_size, num_kv_heads, head_dim]
          # Use layer-specific sequence lengths
          kv_lens=page_state.sequence_lengths[layer_idx],  # [max_num_seqs]
          cu_q_lens=c_q_l,  # [max_num_seqs+1]
          # Use layer-specific page map
          page_indices=page_state.page_map[layer_idx],  # [max_num_seqs, pages_per_seq]
          num_seqs=num_seqs,
          num_kv_pages_per_block=self.pages_per_compute_block,
      )
      print(f"{result.shape=}")
      print("paged_attention_v2_decode - END")
      return jnp.expand_dims(result, axis=1)

  def paged_attention_v1_decode(
        self,
        query: Array,
        key_pages_var: nn.Variable,
        value_pages_var: nn.Variable,
        page_state: page_manager.PageState,
        layer_idx: int = 0,
  ) -> Array:
        """Simplified implementation for temporary testing."""
        batch_size, seq_len, num_heads, head_dim = query.shape
        q_input = query[:, -1, :, :] if seq_len > 1 else jnp.squeeze(query, axis=1)
        k_pages = key_pages_var.value
        v_pages = value_pages_var.value
        seq_lengths = page_state.sequence_lengths[layer_idx]
        valid_batch_size = min(batch_size, page_state.current_page.shape[1])
        q_valid = q_input[:valid_batch_size]

        all_k = []
        all_v = []

        for batch_idx in range(valid_batch_size):
            page_indices = page_state.page_map[layer_idx, batch_idx]
            seq_len_this_seq = seq_lengths[batch_idx]

            keys_this_seq = []
            values_this_seq = []

            for page_num in page_indices:
                num_tokens_in_page = jnp.minimum(seq_len_this_seq, self.tokens_per_page)

                # Use lax.dynamic_slice, with jnp.stack for the sizes
                k_page = lax.dynamic_slice(
                    k_pages,
                    (0, page_num, 0, 0),  # Start indices: (kv_head, page, token, dim)
                    jnp.stack([k_pages.shape[0], 1, num_tokens_in_page, k_pages.shape[3]]),  # Slice sizes
                )
                v_page = lax.dynamic_slice(
                    v_pages,
                    (0, page_num, 0, 0),  # Start indices
                    jnp.stack([v_pages.shape[0], 1, num_tokens_in_page, v_pages.shape[3]]),  # Slice sizes
                )
                # dynamic_slice returns a view; reshape to collapse the page dimension.
                k_page = k_page.reshape((k_pages.shape[0], num_tokens_in_page, k_pages.shape[3]))
                v_page = v_page.reshape((v_pages.shape[0], num_tokens_in_page, v_pages.shape[3]))

                keys_this_seq.append(k_page)
                values_this_seq.append(v_page)
                seq_len_this_seq = seq_len_this_seq - num_tokens_in_page

            k_seq = jnp.concatenate(keys_this_seq, axis=1)
            v_seq = jnp.concatenate(values_this_seq, axis=1)
            all_k.append(k_seq)
            all_v.append(v_seq)

        stacked_k = jnp.stack(all_k, axis=0)
        stacked_v = jnp.stack(all_v, axis=0)

        stacked_k = jnp.transpose(stacked_k, (0, 2, 1, 3))
        q_valid = jnp.transpose(q_valid, (0, 1, 2))

        attn_weights = jnp.einsum("bnhd,bthd->bnt", q_valid, stacked_k)

        # --- Simplified Causal Masking ---
        mask_value = -1e10
        for i in range(valid_batch_size):
            seq_len_i = seq_lengths[i]
            causal_mask = jnp.tril(jnp.ones((seq_len_i, seq_len_i), dtype=bool), k=0)
            attn_weights = attn_weights.at[i, :, :seq_len_i].set(
                jnp.where(causal_mask, attn_weights[i, :, :seq_len_i], mask_value)
            )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        stacked_v = jnp.transpose(stacked_v, (0, 2, 1, 3))
        attn_output = jnp.einsum("bnt,bthd->bnhd", attn_weights, stacked_v)

        return jnp.expand_dims(attn_output, axis=1)

  @nn.compact
  def __call__(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      previous_chunk=None,
      page_state: Optional[page_manager.PageState] = None,
      layer_idx: Optional[int] = None,
      slot: Optional[int] = None,
  ):
    """Apply paged attention mechanism with layer-specific page state handling."""

    # Only enforce page_state requirement during normal execution, not initialization
    if page_state is None:
      if model_mode != common_types.MODEL_MODE_TRAIN and not self.is_initializing():
        raise ValueError(f"PagedAttentionOp requires page_state in {model_mode} mode")

    key_pages_var, value_pages_var = self.init_or_get_kv_pages(model_mode)

    # Only update pages if not initializing
    if not self.is_initializing():
      self.update(key_pages_var, value_pages_var, key, value, model_mode, page_state, layer_idx)

    # Process attention - only if not initializing
    if not self.is_initializing():
      if model_mode == common_types.MODEL_MODE_PREFILL:
        if use_kernel_v2:
          print(f"Running paged_attention_v2_prefill")
          return self.paged_attention_v2_prefill(query, key_pages_var, value_pages_var, page_state, layer_idx), None, None
        print(f"Running paged_dot_product_attention_with_max_and_sum")
        return self.paged_dot_product_attention_with_max_and_sum(query, key, value)
      elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        if use_kernel_v2:
          print(f"Running paged_attention_v2_decode")
          return self.paged_attention_v2_decode(query, key_pages_var, value_pages_var, page_state, layer_idx), None, None
        print(f"Running paged_attention_v1_decode")
        return self.paged_attention_v1_decode(query, key_pages_var, value_pages_var, page_state, layer_idx), None, None
      else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")
    else:
      # During initialization, return a dummy output with the correct shape.
      batch_size, seq_len, num_heads, head_dim = query.shape
      return jnp.zeros((batch_size, seq_len, num_heads, head_dim), dtype=self.dtype), None, None

  def update(self, key_pages_var, value_pages_var, key, value, model_mode, page_state=None, layer_idx=0):
    """Update KV Pages with layer-specific page state."""
    # Skip updates during initialization
    if self.is_initializing():
        return
        
    if page_state is None:
        if model_mode != common_types.MODEL_MODE_TRAIN:
            raise ValueError(f"page_state must be provided in {model_mode} mode")
        return  # No update needed in training mode
        
    if model_mode == common_types.MODEL_MODE_PREFILL:
        self.update_prefill_step_pages(key_pages_var, value_pages_var, key, value)
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        self.update_decode_step_pages(key_pages_var, value_pages_var, key, value, page_state, layer_idx)

  def update_prefill_step_pages(
      self,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      key: Array,
      value: Array,
  ) -> None:
    """Update pages for prefill step."""

    assert (
        key.shape == value.shape
    ), f"prefill_step key/value should have the same shape, but getting {key.shape=} and {value.shape=} instead"
    b, t, n_kv, d = key.shape
    assert (
        key_pages_var.value.shape == value_pages_var.value.shape
    ), f"prefill_step key/value_pages_var should have the same shape, but getting {key_pages_var.shape=} and {value_pages_var.shape=} instead"

    v_n_kv, v_n_p, v_p, v_d = key_pages_var.value.shape
    assert v_n_kv == n_kv, f"{v_n_kv=} {n_kv=}"
    assert v_p == self.tokens_per_page, f"{v_p=} {self.tokens_per_page=}"
    assert v_d == d, f"{v_d=} {d=}"
    assert v_n_p == self.max_pages_per_prefill, f"{v_n_p=} {self.max_pages_per_prefill=}"

    key_pages_var.value = key
    value_pages_var.value = value

  def update_decode_step_pages(self, key_pages_var, value_pages_var, key, value, page_state, layer_idx=0):
    """Update pages for decode step with layer-specific page state."""
    print("-" * 20)
    print("update_decode_step_pages - START")
    print(f"{layer_idx=}")
    key_pages = key_pages_var.value
    value_pages = value_pages_var.value

    # Get shapes
    batch_size, seq_len, kv_heads, head_dim = key.shape
    kv_heads_pages, num_pages, tokens_per_page, head_dim_pages = key_pages.shape
    print(f"{batch_size=}, {seq_len=}, {kv_heads=}, {head_dim=}")
    print(f"{kv_heads_pages=}, {num_pages=}, {tokens_per_page=}, {head_dim_pages=}")

    # Handle potential shape mismatch - ensure we only use valid batch indices
    max_groups = page_state.page_map.shape[1]
    valid_batch_size = min(batch_size, max_groups)
    print(f"{max_groups=}, {valid_batch_size=}")

    # Always take the last token from each sequence regardless of seq_len
    key_last = key[:valid_batch_size, -1, :, :]  # Shape: [valid_batch, kv_heads, head_dim]
    value_last = value[:valid_batch_size, -1, :, :]
    print(f"{key_last.shape=}, {value_last.shape=}")

    # Transpose to get [kv_heads, valid_batch, head_dim]
    new_key = jnp.transpose(key_last, (1, 0, 2))
    new_value = jnp.transpose(value_last, (1, 0, 2))
    print(f"{new_key.shape=}, {new_value.shape=}")

    # Use layer-specific current page and position
    broadcast_pages = jnp.tile(page_state.current_page[layer_idx, :valid_batch_size], (kv_heads, 1))
    broadcast_pos = jnp.tile(page_state.current_page_position[layer_idx, :valid_batch_size], (kv_heads, 1))
    print(f"{broadcast_pages=}")
    print(f"{broadcast_pos=}")

    kv_indices = jnp.arange(kv_heads)[:, None]
    kv_indices = jnp.tile(kv_indices, (1, valid_batch_size))
    print(f"{kv_indices=}")
    
    # Add assertions to check shapes *before* the update
    assert new_key.shape == (kv_heads, valid_batch_size, head_dim), f"new_key shape mismatch: {new_key.shape}"
    assert new_value.shape == (kv_heads, valid_batch_size, head_dim), f"new_value shape mismatch: {new_value.shape}"
    assert broadcast_pages.shape == (kv_heads, valid_batch_size), f"broadcast_pages shape mismatch: {broadcast_pages.shape}"
    assert broadcast_pos.shape == (kv_heads, valid_batch_size), f"broadcast_pos shape mismatch: {broadcast_pos.shape}"
    assert kv_indices.shape == (kv_heads, valid_batch_size), f"kv_indices shape mismatch: {kv_indices.shape}"


    key_pages_updated = key_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_key)
    value_pages_updated = value_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_value)

    print("key_pages_updated (first 5 elements):\n", key_pages_updated.flatten()[:5])
    print("value_pages_updated (first 5 elements):\n", value_pages_updated.flatten()[:5])

    key_pages_updated = nn.with_logical_constraint(key_pages_updated, self.kv_pages_axis_names)
    value_pages_updated = nn.with_logical_constraint(value_pages_updated, self.kv_pages_axis_names)

    key_pages_var.value = key_pages_updated
    value_pages_var.value = value_pages_updated
    print("update_decode_step_pages - END")
    print("-" * 20)
    return key_pages_var, value_pages_var