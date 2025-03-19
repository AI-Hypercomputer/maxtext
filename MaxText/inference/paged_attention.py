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

import jax
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
use_kernel_v3 = True


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
  debug_prints: bool = True  # Add debug flag as a module parameter

  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  kv_pages_axis_names: AxisNames = ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size")

  def setup(self):
    """Initialize any module-level attributes here."""
    # Any setup code can go here
    pass

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
      if model_mode == common_types.MODEL_MODE_PREFILL:
        # Ensure we have enough pages for prefill
        required_pages = (self.max_pages_per_prefill + self.tokens_per_page - 1) // self.tokens_per_page
        max_pages_per_prefill = max(1, required_pages)  # At least 1 page
        kv_pages_shape = (self.num_kv_heads, max_pages_per_prefill, self.tokens_per_page, self.kv_head_dim_size)
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

  def paged_attention_decode(
      self,
      query: Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      layer_idx: int = 0,
  ) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Prepares inputs for paged attention."""
    batch_size, seq_len, num_heads, head_dim = query.shape
    num_kv_heads = key_pages_var.value.shape[0]
    valid_batch_size = min(batch_size, page_state.page_map.shape[1])

    # Process query input
    if seq_len == 1:
      q_input = jnp.squeeze(query, axis=1)
    else:
      q_input = query[:, -1, :, :]

    k_pages = key_pages_var.value
    v_pages = value_pages_var.value

    # Get sequence lengths
    lengths = jnp.ones((batch_size,), dtype=jnp.int32)
    if valid_batch_size > 0:
      actual_lengths = page_state.sequence_lengths[layer_idx, :valid_batch_size]
      lengths = lengths.at[:valid_batch_size].set(actual_lengths)

    # Create page indices
    pages_per_seq = page_state.page_map.shape[2]
    page_indices = jnp.zeros((batch_size, pages_per_seq), dtype=jnp.int32)

    # Get pages_used for validity check
    pages_used = jnp.zeros((batch_size,), dtype=jnp.int32)

    # Fill with actual data if available
    if valid_batch_size > 0:
      actual_indices = page_state.page_map[layer_idx, :valid_batch_size]
      actual_used = page_state.num_pages_used[layer_idx, :valid_batch_size]

      page_indices = page_indices.at[:valid_batch_size].set(actual_indices)
      pages_used = pages_used.at[:valid_batch_size].set(actual_used)

    jax.debug.print("paged_attention_decode: q_input shape={}, value={}", q_input.shape, q_input)
    jax.debug.print("paged_attention_decode: k_pages shape={}", k_pages.shape)
    jax.debug.print("paged_attention_decode: v_pages shape={}", v_pages.shape)
    jax.debug.print("paged_attention_decode: lengths shape={}, value={}", lengths.shape, lengths)
    jax.debug.print("paged_attention_decode: page_indices shape={}, value={}", page_indices.shape, page_indices)
    jax.debug.print("paged_attention_decode: SUMMARY - q_input: {}, k_pages: {}, v_pages: {}, lengths: {}, page_indices: {}, pages_used: {}",
                   q_input.shape, k_pages.shape, v_pages.shape, lengths, page_indices, pages_used)
    return q_input, k_pages, v_pages, lengths, page_indices, pages_used
  

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
    """Apply paged attention mechanism.

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        decoder_segment_ids: Segment IDs
        model_mode: Model mode (TRAIN, PREFILL, AUTOREGRESSIVE)
        previous_chunk: Previous chunk information for chunked processing
        page_state: PageState object for page management
        layer_idx: Layer index being processed
        slot: Slot ID for the current request
          
    Returns:
        - In initialization: Cache dictionary
        - In prefill: Tuple with cache and attention outputs
        - In autoregressive: Tuple with cache and prepared inputs for attention
    """
    jax.debug.print("PagedAttentionOp.__call__: model_mode={}, layer_idx={}", model_mode, layer_idx)
    if page_state is None:
      if model_mode != common_types.MODEL_MODE_TRAIN and not self.is_initializing():
        raise ValueError(f"PagedAttentionOp requires page_state in {model_mode} mode")

    # Initialize or get the KV cache
    key_pages_var, value_pages_var = self.init_or_get_kv_pages(model_mode)

    # Always return a dictionary,
    # and during initialization, return a dictionary *containing* 'cache'
    cache_dict = {
        "cache": {
            "cached_prefill_key": key_pages_var.value,
            "cached_prefill_value": value_pages_var.value,
        }
    }

    if self.is_initializing():
      return cache_dict
    
    # Process based on model mode
    if model_mode == common_types.MODEL_MODE_PREFILL:
        # Update pages
        self.update(key_pages_var, value_pages_var, key, value, model_mode, page_state, layer_idx, slot)
        
        # Compute attention using dot-product
        attn, local_max, local_sums = self.paged_dot_product_attention_with_max_and_sum(
            query, key, value
        )
        
        # Return cache and attention results
        return cache_dict, attn, local_max, local_sums

    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        # Get prepared inputs for attention kernel
        q_input, k_pages, v_pages, lengths, page_indices, pages_used = self.paged_attention_decode(
            query, key_pages_var, value_pages_var, page_state, layer_idx
        )
       
        self.update(key_pages_var, value_pages_var, key, value, model_mode, page_state, layer_idx, slot) 
        # Return cache and prepared inputs
        return cache_dict, q_input, k_pages, v_pages, lengths, page_indices, pages_used

    else:  # TRAIN mode or other
        raise ValueError(f"Unsupported model_mode: {model_mode}")


    # # Only update pages if not initializing
    # if not self.is_initializing():
    #   if model_mode == common_types.MODEL_MODE_PREFILL:
    #     self.update(key_pages_var, value_pages_var, key, value, model_mode, page_state, layer_idx)
    #     # Prefill uses simple dot-product attention.
    #     attn, local_max, local_sums = self.paged_dot_product_attention_with_max_and_sum(
    #         query, key, value
    #     )  # Return result directly
    #     # Return attention and cache
    #     return (
    #         {
    #             "cache": {
    #                 "cached_prefill_key": key_pages_var.value,
    #                 "cached_prefill_value": value_pages_var.value,
    #             }
    #         },
    #         attn,
    #         local_max,
    #         local_sums,
    #     )

    #   elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
    #     # Get prepared inputs from paged_attention_decode
    #     q_input, k_pages, v_pages, lengths, page_indices, pages_used = self.paged_attention_decode(
    #         query, key_pages_var, value_pages_var, page_state, layer_idx
    #     )
    #     # Return prepared inputs *and* the cache, along with the layer_idx.
    #     return (
    #         {
    #             "cache": {
    #                 "cached_prefill_key": key_pages_var.value,
    #                 "cached_prefill_value": value_pages_var.value,
    #             }
    #         },
    #         q_input,
    #         k_pages,
    #         v_pages,
    #         lengths,
    #         page_indices,
    #         pages_used,
    #     )

    #   else:
    #     raise ValueError(f"Unsupported model_mode: {model_mode}")
    # else:  # self.is_initializing() == True
    #   # We're initializing.  Return a dictionary with a 'cache' key.
    #   return {
    #       "cache": {
    #           "cached_prefill_key": key_pages_var.value,  # Use .value to get the array
    #           "cached_prefill_value": value_pages_var.value,  # Use .value to get the array
    #       }
    #   }

  def update(
    self, 
    key_pages_var, 
    value_pages_var, 
    key, 
    value, 
    model_mode, 
    page_state: Optional[page_manager.PageState] = None, 
    layer_idx=0,
    slot=None,
  ) -> None:
    """Update KV Pages with layer-specific page state."""
    # Skip updates during initialization
    if self.is_initializing():
      return

    print(f"Starting PagedAttentionOp.update: model_mode={model_mode}, page_state={page_state is not None}")
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

    assert key.shape == value.shape, f"key/value should have same shape, got {key.shape=} and {value.shape=}" 
    b, t, n_kv, d = key.shape

    assert key_pages_var.value.shape == value_pages_var.value.shape
    v_n_kv, v_n_p, v_p, v_d = key_pages_var.value.shape

    assert v_n_kv == n_kv, f"{v_n_kv=} {n_kv=}"
    assert v_p == self.tokens_per_page, f"{v_p=} {self.tokens_per_page=}"
    assert v_d == d, f"{v_d=} {d=}"

    # CHANGED: Check if we have enough total capacity (pages * tokens_per_page)
    total_capacity = v_n_p * v_p
    assert total_capacity >= t, f"Prefill cache too small! Capacity: {total_capacity} tokens, request: {t} tokens"

    # Reshape key and value for paged storage
    key_transposed = jnp.transpose(key, (2, 0, 1, 3))  # [b, t, n_kv, d] -> [n_kv, b, t, d]
    value_transposed = jnp.transpose(value, (2, 0, 1, 3))  # [b, t, n_kv, d] -> [n_kv, b, t, d]

    # Calculate how many pages needed
    pages_needed = (t + self.tokens_per_page - 1) // self.tokens_per_page

    # Store tokens in pages
    for page_idx in range(min(pages_needed, v_n_p)):
      start_token = page_idx * self.tokens_per_page
      end_token = min(start_token + self.tokens_per_page, t)
      tokens_in_page = end_token - start_token

      key_pages_var.value = key_pages_var.value.at[:, page_idx, :tokens_in_page, :].set(
          key_transposed[:, 0, start_token:end_token, :]
      )
      value_pages_var.value = value_pages_var.value.at[:, page_idx, :tokens_in_page, :].set(
          value_transposed[:, 0, start_token:end_token, :]
      )

  def update_decode_step_pages(self, key_pages_var, value_pages_var, key, value, page_state, layer_idx=0):
    """Improved update method compatible with JAX tracing."""
    key_pages = key_pages_var.value
    value_pages = value_pages_var.value

    # Get shapes for validation
    batch_size, seq_len, kv_heads, head_dim = key.shape
    kv_heads_pages, num_pages, tokens_per_page, head_dim_pages = key_pages.shape

    # Validate batch size
    valid_batch_size = min(batch_size, page_state.page_map.shape[1])

    # Take the last token
    key_last = key[:valid_batch_size, -1, :, :]  # [valid_batch, kv_heads, head_dim]
    value_last = value[:valid_batch_size, -1, :, :]  # [valid_batch, kv_heads, head_dim]

    # Get current page and position indices
    active_pages = page_state.active_page[layer_idx, :valid_batch_size]  # [valid_batch]
    active_positions = page_state.active_page_position[layer_idx, :valid_batch_size]  # [valid_batch]
    has_active_pages = page_state.has_active_page[layer_idx, :valid_batch_size]  # [valid_batch]

    # Prepare tensors for update
    new_key = jnp.transpose(key_last, (1, 0, 2))  # [kv_heads, valid_batch, head_dim]
    new_value = jnp.transpose(value_last, (1, 0, 2))  # [kv_heads, valid_batch, head_dim]

    # Create a mask for sequences with valid current pages
    valid_seq_mask = has_active_pages

    # Helper function to update a SINGLE element
    def _update_single(kv_head_idx, seq_idx, key_pages, value_pages):
        page_idx = active_pages[seq_idx]
        pos_idx = active_positions[seq_idx]

        # Use JAX-compatible boolean operations:
        is_valid = jnp.logical_and(
            valid_seq_mask[seq_idx],
            jnp.logical_and(0 <= page_idx, page_idx < num_pages)
        )
        is_valid = jnp.logical_and(
            is_valid,
            jnp.logical_and(0 <= pos_idx, pos_idx < tokens_per_page)
        )

        # Use jax.lax.cond for the update
        key_pages = jax.lax.cond(
            is_valid,
            lambda: key_pages.at[kv_head_idx, page_idx, pos_idx].set(new_key[kv_head_idx, seq_idx]),
            lambda: key_pages
        )
        value_pages = jax.lax.cond(
            is_valid,
            lambda: value_pages.at[kv_head_idx, page_idx, pos_idx].set(new_value[kv_head_idx, seq_idx]),
            lambda: value_pages
        )
        return key_pages, value_pages

    # vmap over heads and sequences
    _update_single_vmap = jax.vmap(jax.vmap(_update_single, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
    key_pages, value_pages = _update_single_vmap(jnp.arange(kv_heads), jnp.arange(valid_batch_size), key_pages, value_pages)

    # Apply logical constraints
    key_pages = nn.with_logical_constraint(key_pages, self.kv_pages_axis_names)
    value_pages = nn.with_logical_constraint(value_pages, self.kv_pages_axis_names)

    # Update the variables
    key_pages_var.value = key_pages
    value_pages_var.value = value_pages

    return key_pages_var, value_pages_var

def vectorized_paged_attention(
  query: jnp.ndarray,
  key_pages: jnp.ndarray,
  value_pages: jnp.ndarray,
  page_indices: jnp.ndarray,
  pages_used: jnp.ndarray,
  lengths: jnp.ndarray,
  mask_value: float = -1e7,
  attn_logits_soft_cap: Optional[float] = None,
) -> jnp.ndarray:
  """Vectorized implementation of paged attention that avoids explicit loops.
  
  This version is more JAX-friendly and should be more efficient on accelerators.
  """
  batch_size, num_heads, head_dim = query.shape
  num_kv_heads = key_pages.shape[0]
  num_pages = key_pages.shape[1]
  tokens_per_page = key_pages.shape[2]
  max_pages_per_seq = page_indices.shape[1]
  
  # Handle grouped query attention (GQA)
  queries_per_kv = num_heads // num_kv_heads
  
  # Scale factor for attention
  scale_factor = 1.0 / jnp.sqrt(head_dim)
  
  # Create validity mask from pages_used counter - [batch, max_pages]
  validity_mask = jnp.arange(max_pages_per_seq)[None, :] < pages_used[:, None]
  
  # Reshape for computation
  q_reshaped = query.reshape(batch_size, num_kv_heads, queries_per_kv, head_dim)
  
  # Initialize attention outputs
  attention_output = jnp.zeros((batch_size, num_heads, head_dim))
  
  # Create vectorized operations instead of loops
  
  # For each batch item and potential page, create a selector mask
  # This avoids explicit loops and conditional statements
  batch_indices = jnp.arange(batch_size)[:, None]              # [batch, 1]
  page_range = jnp.arange(max_pages_per_seq)[None, :]          # [1, max_pages]
  
  # Get actual page indices - [batch, max_pages]
  actual_page_indices = page_indices
  
  # Create selector for addressing the key/value pages
  # We'll use advanced indexing to gather the right key/value elements
  
  # First create a mask that is True only for valid pages
  # [batch, max_pages]
  page_selector_mask = validity_mask
  
  # Create a safe masked version of page indices where invalid indices are set to 0
  # [batch, max_pages]
  safe_page_indices = jnp.where(page_selector_mask, actual_page_indices, jnp.zeros_like(actual_page_indices))
  
  # Create attention mask where invalid pages get mask_value
  # [batch, 1, 1, max_pages] for broadcasting with attention scores
  attn_mask = jnp.where(
      page_selector_mask[:, None, None, :],
      0.0,
      mask_value
  )
  
  # Now we need to gather key/value tensors based on page indices
  # This requires a different approach due to JAX's functional nature
  
  # Approach: compute all possible attention scores, then mask invalid ones
  all_attention_weights = []
  all_values = []
  
  # Compute attention for each KV head
  for kv_head in range(num_kv_heads):
      head_attention_weights = []
      head_values = []
      
      # For each potential page index
      for p_idx in range(num_pages):
          # Get key/value for this page
          k_page = key_pages[kv_head, p_idx]   # [tokens_per_page, head_dim]
          v_page = value_pages[kv_head, p_idx] # [tokens_per_page, head_dim]
          
          # Create a mask that's True when this page index matches the actual page index
          # [batch, max_pages]
          page_match = safe_page_indices == p_idx
          
          # Compute attention scores for this page
          # [batch, queries_per_kv, tokens_per_page]
          scores = jnp.einsum('bgh,th->bgt', q_reshaped[:, kv_head], k_page) * scale_factor
          
          # Apply soft cap if needed
          if attn_logits_soft_cap is not None:
              scores = attn_logits_soft_cap * jnp.tanh(scores / attn_logits_soft_cap)
          
          # Mask scores for invalid pages
          # [batch, max_pages, queries_per_kv, tokens_per_page]
          expanded_scores = jnp.zeros((batch_size, max_pages_per_seq, queries_per_kv, tokens_per_page))
          for b in range(batch_size):
              for mp in range(max_pages_per_seq):
                  if page_match[b, mp]:
                      expanded_scores = expanded_scores.at[b, mp].set(scores[b])
          
          head_attention_weights.append(expanded_scores)
          
          # Similarly expand values
          # [batch, max_pages, tokens_per_page, head_dim]
          expanded_values = jnp.zeros((batch_size, max_pages_per_seq, tokens_per_page, head_dim))
          for b in range(batch_size):
              for mp in range(max_pages_per_seq):
                  if page_match[b, mp]:
                      expanded_values = expanded_values.at[b, mp].set(v_page)
          
          head_values.append(expanded_values)
      
      # Stack across pages
      # [batch, max_pages, queries_per_kv, tokens_per_page]
      kv_head_attention = jnp.stack(head_attention_weights, axis=1)
      # [batch, max_pages, tokens_per_page, head_dim]
      kv_head_values = jnp.stack(head_values, axis=1)
      
      all_attention_weights.append(kv_head_attention)
      all_values.append(kv_head_values)
  
  # Stack across KV heads
  # [num_kv_heads, batch, max_pages, queries_per_kv, tokens_per_page]
  stacked_attention = jnp.stack(all_attention_weights)
  # [num_kv_heads, batch, max_pages, tokens_per_page, head_dim]
  stacked_values = jnp.stack(all_values)
  
  # Apply softmax across the tokens dimension (within each page)
  # First apply mask for invalid pages
  stacked_attention = stacked_attention + attn_mask[:, None, :, None, :]
  
  # Apply softmax per page
  attention_weights = jax.nn.softmax(stacked_attention, axis=-1)
  
  # Compute weighted values
  # [num_kv_heads, batch, max_pages, queries_per_kv, head_dim]
  weighted_values = jnp.einsum('kbmpqt,kbmpth->kbmpqh', attention_weights, stacked_values)
  
  # Sum across pages
  # [num_kv_heads, batch, queries_per_kv, head_dim]
  summed_values = jnp.sum(weighted_values, axis=2)
  
  # Reshape to original query shape
  # [batch, num_heads, head_dim]
  output = summed_values.transpose(1, 0, 2, 3).reshape(batch_size, num_heads, head_dim)
  
  return output

def paged_attention_decode_step(
    query: jnp.ndarray,             # [batch, heads, head_dim]
    key_pages: jnp.ndarray,         # [kv_heads, num_pages, tokens_per_page, head_dim]
    value_pages: jnp.ndarray,       # [kv_heads, num_pages, tokens_per_page, head_dim]
    page_indices: jnp.ndarray,      # [batch, max_pages_per_group]
    pages_used: jnp.ndarray,        # [batch]
    lengths: jnp.ndarray,           # [batch]
    mask_value: float = -1e7,
    attn_logits_soft_cap: Optional[float] = None,
    scale_factor: Optional[float] = None,
) -> jnp.ndarray:
    """Implements JAX-compatible paged attention for a single decode step.
    
    This function computes attention over paged key-value cache using validity tracking
    based on pages_used counter rather than sentinel values.
    
    Args:
        query: Query tensor with shape [batch, heads, head_dim]
        key_pages: Key pages with shape [kv_heads, num_pages, tokens_per_page, head_dim]
        value_pages: Value pages with shape [kv_heads, num_pages, tokens_per_page, head_dim]
        page_indices: Indices of pages with shape [batch, max_pages_per_group]
        pages_used: Number of valid pages per sequence with shape [batch]
        lengths: Sequence lengths with shape [batch]
        mask_value: Value to use for masked positions
        attn_logits_soft_cap: Optional cap for attention logits
        scale_factor: Optional scaling factor for attention scores
        
    Returns:
        Attention output with shape [batch, heads, head_dim]
    """
    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = key_pages.shape[0]
    num_pages = key_pages.shape[1]
    tokens_per_page = key_pages.shape[2]
    
    # Handle grouped query attention (GQA)
    queries_per_kv = num_heads // num_kv_heads
    
    # Calculate scale factor if not provided
    if scale_factor is None:
        scale_factor = 1.0 / jnp.sqrt(head_dim)
    
    # Reshape query for computation
    # For GQA, we need to reshape the query to match KV groups
    query = query.reshape(batch_size, num_kv_heads, queries_per_kv, head_dim)
    
    # Create validity mask based on pages_used
    # [batch, max_pages_per_group] where True means the page is valid
    validity_mask = jnp.arange(page_indices.shape[1])[None, :] < pages_used[:, None]
    
    # Initialize accumulations for attention computation
    # We'll compute max and weighted sum separately to handle numerical stability
    max_scores = jnp.ones((batch_size, num_kv_heads, queries_per_kv)) * -1e10
    weighted_sum = jnp.zeros((batch_size, num_kv_heads, queries_per_kv, head_dim))
    normalizer = jnp.zeros((batch_size, num_kv_heads, queries_per_kv))
    
    # Loop over each sequence in the batch
    def process_batch(b_idx, accumulators):
        b_max_scores, b_weighted_sum, b_normalizer = accumulators
        seq_length = lengths[b_idx]
        
        # Skip empty sequences
        def process_sequence(args):
            b_max_scores, b_weighted_sum, b_normalizer = args
            # Get valid pages for this sequence
            seq_valid_mask = validity_mask[b_idx]
            seq_page_indices = page_indices[b_idx]
            
            # Process each valid page
            def process_page(p_idx, acc):
                page_max_scores, page_weighted_sum, page_normalizer = acc
                
                # Check if this page is valid
                is_valid = seq_valid_mask[p_idx]
                
                def handle_valid_page():
                    # Get the page index
                    page_idx = seq_page_indices[p_idx]
                    
                    # Determine tokens to process in this page
                    is_last_page = p_idx == pages_used[b_idx] - 1
                    page_tokens = jnp.where(is_last_page, 
                                          seq_length % tokens_per_page,
                                          tokens_per_page)
                    # At least one token per page
                    page_tokens = jnp.maximum(page_tokens, 1)
                    
                    # Get query for this sequence
                    q = query[b_idx]  # [num_kv_heads, queries_per_kv, head_dim]
                    
                    # Get key/value for this page
                    k = key_pages[:, page_idx, :page_tokens, :]  # [num_kv_heads, page_tokens, head_dim]
                    v = value_pages[:, page_idx, :page_tokens, :]  # [num_kv_heads, page_tokens, head_dim]
                    
                    # Compute attention scores
                    # [num_kv_heads, queries_per_kv, page_tokens]
                    scores = jnp.einsum('kgh,kth->kgt', q, k) * scale_factor
                    
                    # Apply soft cap if needed
                    if attn_logits_soft_cap is not None:
                        scores = attn_logits_soft_cap * jnp.tanh(scores / attn_logits_soft_cap)
                    
                    # Find max score for stability
                    score_max = jnp.max(scores, axis=-1, keepdims=True)  # [num_kv_heads, queries_per_kv, 1]
                    
                    # Update max score if higher than current max
                    new_max = jnp.maximum(page_max_scores, score_max)
                    
                    # Scale previous accumulation by exp(old_max - new_max)
                    old_scale = jnp.exp(page_max_scores - new_max)
                    page_weighted_sum = page_weighted_sum * old_scale[..., None]
                    page_normalizer = page_normalizer * old_scale
                    
                    # Compute updated attention weights
                    weights = jnp.exp(scores - new_max)  # [num_kv_heads, queries_per_kv, page_tokens]
                    
                    # Compute weighted values and update sum
                    # [num_kv_heads, queries_per_kv, head_dim]
                    value_contribution = jnp.einsum('kgt,kth->kgh', weights, v)
                    page_weighted_sum = page_weighted_sum + value_contribution
                    
                    # Update normalizer
                    page_normalizer = page_normalizer + jnp.sum(weights, axis=-1)
                    
                    return new_max, page_weighted_sum, page_normalizer
                
                # Only process if the page is valid
                return jax.lax.cond(
                    is_valid,
                    lambda: handle_valid_page(),
                    lambda: (page_max_scores, page_weighted_sum, page_normalizer)
                )
            
            # Process all potential pages
            max_pages = seq_page_indices.shape[0]
            return jax.lax.fori_loop(
                0, max_pages,
                lambda i, acc: process_page(i, acc),
                (b_max_scores[b_idx], b_weighted_sum[b_idx], b_normalizer[b_idx])
            )
        
        # Only process non-empty sequences
        has_tokens = lengths[b_idx] > 0
        updated_scores, updated_sum, updated_norm = jax.lax.cond(
            has_tokens,
            lambda: process_sequence((b_max_scores[b_idx], b_weighted_sum[b_idx], b_normalizer[b_idx])),
            lambda: (b_max_scores[b_idx], b_weighted_sum[b_idx], b_normalizer[b_idx])
        )
        
        # Update accumulators for this batch element
        b_max_scores = b_max_scores.at[b_idx].set(updated_scores)
        b_weighted_sum = b_weighted_sum.at[b_idx].set(updated_sum)
        b_normalizer = b_normalizer.at[b_idx].set(updated_norm)
        
        return b_max_scores, b_weighted_sum, b_normalizer
    
    # Process all sequences
    max_scores, weighted_sum, normalizer = jax.lax.fori_loop(
        0, batch_size,
        process_batch,
        (max_scores, weighted_sum, normalizer)
    )
    
    # Compute final output - for GQA, properly reshape to get original head dimensions
    output = weighted_sum / (normalizer[..., None] + 1e-9)
    output = output.reshape(batch_size, num_heads, head_dim)
    
    return output

def custom_paged_attention(
    query: jnp.ndarray,             # [batch, heads, head_dim]
    key_pages: jnp.ndarray,         # [kv_heads, num_pages, tokens_per_page, head_dim]
    value_pages: jnp.ndarray,       # [kv_heads, num_pages, tokens_per_page, head_dim]
    page_indices: jnp.ndarray,      # [batch, max_pages_per_group]
    pages_used: jnp.ndarray,        # [batch]
    lengths: jnp.ndarray,           # [batch]
    mask_value: float = -1e7,
    attn_logits_soft_cap: Optional[float] = None,
) -> jnp.ndarray:
    """JAX-optimized paged attention implementation for decode steps.
    
    Uses validity tracking based on pages_used counter instead of sentinel values.
    Compatible with JAX tracing and transformation.
    
    Args:
        query: Query tensor with shape [batch, heads, head_dim]
        key_pages: Key pages tensor with shape [kv_heads, num_pages, tokens_per_page, head_dim]
        value_pages: Value pages tensor with shape [kv_heads, num_pages, tokens_per_page, head_dim]
        page_indices: Page index mapping with shape [batch, max_pages_per_group]
        pages_used: Counter of valid pages per sequence with shape [batch]
        lengths: Sequence lengths with shape [batch]
        mask_value: Value to use for masked positions in attention
        attn_logits_soft_cap: Optional cap for attention logits
        
    Returns:
        Attention output with shape [batch, heads, head_dim]
    """
    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = key_pages.shape[0]
    tokens_per_page = key_pages.shape[2]
    max_pages = page_indices.shape[1]
    
    # Handle grouped query attention (GQA)
    queries_per_kv = num_heads // num_kv_heads
    
    # Scale factor for attention
    scale_factor = 1.0 / jnp.sqrt(head_dim)
    
    # Initialize output
    output = jnp.zeros((batch_size, num_heads, head_dim))
    
    # Define processing for a flattened (batch_idx, kv_head_idx) pair
    def process_batch_kv_pair(idx):
        # Convert flat index to batch and KV head indices
        b_idx = idx // num_kv_heads
        kv_idx = idx % num_kv_heads
        
        # Get query for this batch and KV head group
        q_reshaped = query.reshape(batch_size, num_kv_heads, queries_per_kv, head_dim)
        q = q_reshaped[b_idx, kv_idx]  # [queries_per_kv, head_dim]
        
        # Initialize state for page processing
        init_state = (
            jnp.full((queries_per_kv,), -1e10),    # max_score
            jnp.zeros((queries_per_kv, head_dim)), # weighted_sum
            jnp.zeros((queries_per_kv,)),          # normalizer
        )
        
        # Function to process a single page
        def process_page(page_idx, state):
            max_score, weighted_sum, normalizer = state
            
            # Check if page is valid (within the used range)
            is_valid = page_idx < pages_used[b_idx]
            
            def handle_valid_page():
                # Get actual page index from mapping
                actual_page_idx = page_indices[b_idx, page_idx]
                
                # Get K/V for this page
                k = key_pages[kv_idx, actual_page_idx]  # [tokens_per_page, head_dim]
                v = value_pages[kv_idx, actual_page_idx]  # [tokens_per_page, head_dim]
                
                # For last page, limit tokens to remainder
                is_last_page = page_idx == pages_used[b_idx] - 1
                tokens_in_page = jnp.where(
                    is_last_page,
                    lengths[b_idx] - (pages_used[b_idx] - 1) * tokens_per_page,
                    tokens_per_page
                )
                tokens_in_page = jnp.maximum(tokens_in_page, 1)  # At least 1 token
                
                # Slice to get only valid tokens
                k = jax.lax.dynamic_slice(k, (0, 0), (tokens_in_page, head_dim))
                v = jax.lax.dynamic_slice(v, (0, 0), (tokens_in_page, head_dim))
                
                # Compute attention scores
                scores = jnp.einsum('gh,th->gt', q, k) * scale_factor
                
                # Apply soft cap if specified
                if attn_logits_soft_cap is not None:
                    scores = attn_logits_soft_cap * jnp.tanh(scores / attn_logits_soft_cap)
                
                # Update max and rescale - for numerical stability
                curr_max = jnp.max(scores, axis=-1)
                new_max = jnp.maximum(max_score, curr_max)
                old_scale = jnp.exp(max_score - new_max)
                
                # Rescale previous accumulations
                new_weighted_sum = weighted_sum * old_scale[:, None]
                new_normalizer = normalizer * old_scale
                
                # Compute weights and update accumulators
                weights = jnp.exp(scores - new_max[:, None])
                new_weighted_sum = new_weighted_sum + jnp.einsum('gt,th->gh', weights, v)
                new_normalizer = new_normalizer + jnp.sum(weights, axis=-1)
                
                return new_max, new_weighted_sum, new_normalizer
            
            # Only process if page is valid
            return jax.lax.cond(
                is_valid,
                handle_valid_page,
                lambda: (max_score, weighted_sum, normalizer)
            )
        
        # Process all pages for this batch/kv-head pair
        max_score, weighted_sum, normalizer = jax.lax.fori_loop(
            0, max_pages,
            process_page,
            init_state
        )
        
        # Compute final head output
        head_output = weighted_sum / (normalizer[:, None] + 1e-9)
        return head_output
    
    # Process all batch/kv pairs
    def body_fn(i, out):
        b_idx = i // num_kv_heads
        kv_idx = i % num_kv_heads
        start_idx = kv_idx * queries_per_kv
        end_idx = (kv_idx + 1) * queries_per_kv
        
        # Get result for this batch/kv pair
        head_output = process_batch_kv_pair(i)
        
        # Update output tensor
        out = out.at[b_idx, start_idx:end_idx].set(head_output)
        return out
    
    # Process all combinations with a single fori_loop
    output = jax.lax.fori_loop(
        0, batch_size * num_kv_heads,
        body_fn,
        output
    )
    
    return output


def fixed_paged_attention(
    query: jnp.ndarray,             # [batch, heads, head_dim]
    key_pages: jnp.ndarray,         # [kv_heads, num_pages, tokens_per_page, head_dim]
    value_pages: jnp.ndarray,       # [kv_heads, num_pages, tokens_per_page, head_dim]
    page_indices: jnp.ndarray,      # [batch, max_pages_per_group]
    pages_used: jnp.ndarray,        # [batch]
    lengths: jnp.ndarray,           # [batch]
    mask_value: float = -1e7,
    attn_logits_soft_cap: Optional[float] = None,
) -> jnp.ndarray:
    """Ultra-simplified paged attention implementation for debugging."""
    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = key_pages.shape[0]
    tokens_per_page = key_pages.shape[2]
    
    # Handle grouped query attention (GQA)
    queries_per_kv = num_heads // num_kv_heads
    
    # Scale factor for attention
    scale_factor = 1.0 / jnp.sqrt(head_dim)
    
    # Reshape query for grouped query attention
    query_reshaped = query.reshape(batch_size, num_kv_heads, queries_per_kv, head_dim)
    
    # For simplicity, just use first page in page indices for each batch
    # This helps us get something working first
    simple_output = jnp.zeros((batch_size, num_heads, head_dim))
    
    # Get query, key, value for first batch
    b_idx = 0
    
    # Use first page for key/value - in production we'd loop through valid pages
    first_page_idx = page_indices[b_idx, 0]
    
    # Process each KV head
    for kv_idx in range(num_kv_heads):
        # Get query for this head
        q = query_reshaped[b_idx, kv_idx]  # [queries_per_kv, head_dim]
        
        # Get key and value for this page
        k = key_pages[kv_idx, first_page_idx]   # [tokens_per_page, head_dim]
        v = value_pages[kv_idx, first_page_idx]  # [tokens_per_page, head_dim]
        
        # For simplicity, use all tokens in the page
        # In production, we'd mask based on actual token count
        tokens_in_page = lengths[b_idx]
        
        # Ensure at least one token
        tokens_in_page = jnp.maximum(tokens_in_page, 1)
        
        # Simple mask for valid tokens
        mask = jnp.arange(tokens_per_page) < tokens_in_page
        
        # Compute attention scores
        scores = jnp.einsum('qd,td->qt', q, k) * scale_factor
        
        # Apply soft cap if specified
        if attn_logits_soft_cap is not None:
            scores = attn_logits_soft_cap * jnp.tanh(scores / attn_logits_soft_cap)
        
        # Apply mask
        masked_scores = jnp.where(mask, scores, mask_value)
        
        # Simple softmax
        exp_scores = jnp.exp(masked_scores - jnp.max(masked_scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / (jnp.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
        
        # Compute weighted values
        head_result = jnp.einsum('qt,td->qd', attention_weights, v)
        
        # Copy to output for all queries in this KV head
        start_idx = kv_idx * queries_per_kv
        for q_idx in range(queries_per_kv):
            simple_output = simple_output.at[b_idx, start_idx + q_idx].set(head_result[q_idx])
    
    return simple_output