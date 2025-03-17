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
  ) -> tuple[Array, Array, Array, Array, Array]:  # Return prepared inputs
    """Prepares inputs for paged attention."""

    batch_size, seq_len, num_heads, head_dim = query.shape
    num_kv_heads = key_pages_var.value.shape[0]
    valid_batch_size = min(batch_size, page_state.page_map.shape[1])

    if seq_len == 1:
      q_input = jnp.squeeze(query, axis=1)
    else:
      q_input = query[:, -1, :, :]

    k_pages = key_pages_var.value
    v_pages = value_pages_var.value

    lengths = jnp.ones((batch_size,), dtype=jnp.int32)
    if valid_batch_size > 0:
      actual_lengths = page_state.sequence_lengths[layer_idx, :valid_batch_size]
      actual_lengths = jnp.maximum(actual_lengths, 1)  # Ensure no zeros
      lengths = lengths.at[:valid_batch_size].set(actual_lengths)

    pages_per_seq = page_state.page_map.shape[2] if len(page_state.page_map.shape) > 2 else 32
    page_indices = jnp.zeros((batch_size, pages_per_seq), dtype=jnp.int32)

    if valid_batch_size > 0 and len(page_state.page_map.shape) > 2:
      actual_indices = page_state.page_map[layer_idx, :valid_batch_size]
      actual_indices = jnp.clip(actual_indices, 0, self.num_pages - 1)
      page_indices = page_indices.at[:valid_batch_size].set(actual_indices)

    # Return the prepared inputs
    return q_input, k_pages, v_pages, lengths, page_indices

  # def paged_attention_decode(
  #   self,
  #   query: Array,
  #   key_pages_var: nn.Variable,
  #   value_pages_var: nn.Variable,
  #   page_state: page_manager.PageState,
  #   layer_idx: int = 0,
  # ) -> tuple[Array, None, None]:
  #   """Fully revised implementation of paged attention with JAX integration."""
  #   from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention as jax_paged_attention
  #   from jax.experimental import shard_map
  #   from jax.sharding import PartitionSpec as P
  #   import jax

  #   # Extract shapes
  #   batch_size, seq_len, num_heads, head_dim = query.shape
  #   num_kv_heads = key_pages_var.value.shape[0]
  #   valid_batch_size = min(batch_size, page_state.page_map.shape[1])

  #   # Critical: First prepare all inputs at EXACTLY the shapes the kernel expects
  #   # without any subsequent reshaping

  #   # 1. Query preparation - must be [batch, heads, dim]
  #   if seq_len == 1:
  #       q_input = jnp.squeeze(query, axis=1)
  #   else:
  #       q_input = query[:, -1, :, :]

  #   # 2. Get k/v pages
  #   k_pages = key_pages_var.value
  #   v_pages = value_pages_var.value

  #   # 3. Prepare sequence lengths
  #   lengths = jnp.ones((batch_size,), dtype=jnp.int32)
  #   if valid_batch_size > 0:
  #       actual_lengths = page_state.sequence_lengths[layer_idx, :valid_batch_size]
  #       # Ensure no zeros in lengths
  #       actual_lengths = jnp.maximum(actual_lengths, 1)
  #       lengths = lengths.at[:valid_batch_size].set(actual_lengths)

  #   # 4. Prepare page indices
  #   # Create empty page indices array of correct final shape
  #   pages_per_seq = page_state.page_map.shape[2] if len(page_state.page_map.shape) > 2 else 32
  #   page_indices = jnp.zeros((batch_size, pages_per_seq), dtype=jnp.int32)

  #   # Fill with actual values where available
  #   if valid_batch_size > 0 and len(page_state.page_map.shape) > 2:
  #       actual_indices = page_state.page_map[layer_idx, :valid_batch_size]
  #       # Clip any out-of-bounds indices
  #       actual_indices = jnp.clip(actual_indices, 0, self.num_pages - 1)
  #       page_indices = page_indices.at[:valid_batch_size].set(actual_indices)

  #   # Define mask value - CRITICAL: use a static float value here
  #   DEFAULT_MASK_VALUE = -1e7

  #   # Print diagnostic info
  #   if self.debug_prints:
  #       print(f"q_input.shape: {q_input.shape}, dtype: {q_input.dtype}")
  #       print(f"k_pages.shape: {k_pages.shape}, dtype: {k_pages.dtype}")
  #       print(f"v_pages.shape: {v_pages.shape}, dtype: {v_pages.dtype}")
  #       print(f"lengths.shape: {lengths.shape}, dtype: {lengths.dtype}")
  #       print(f"page_indices.shape: {page_indices.shape}, dtype: {page_indices.dtype}")
  #       print(f"pages_per_compute_block: {self.pages_per_compute_block}")

  #   try:
  #       # Use the absolute minimal PartitionSpecs
  #       # CRITICAL: Don't add partition on batch dim for q here - we handle it in shard_map
  #       q_pspec = P(None, None, None)  # [batch, heads, dim]
  #       k_pspec = P(None, None, None, None)  # [kv_heads, num_pages, page_size, head_dim]
  #       v_pspec = P(None, None, None, None)  # [kv_heads, num_pages, page_size, head_dim]
  #       lengths_pspec = P(None)  # [batch]
  #       page_indices_pspec = P(None, None)  # [batch, pages_per_seq]

  #       # Define simple attention function with minimal operations
  #       def attention_fn(q, k, v, lens, indices):
  #           return jax_paged_attention(
  #               q=q,
  #               k_pages=k,
  #               v_pages=v,
  #               lengths=lens,
  #               page_indices=indices,
  #               mask_value=DEFAULT_MASK_VALUE,
  #               attn_logits_soft_cap=self.attn_logits_soft_cap,
  #               pages_per_compute_block=self.pages_per_compute_block,
  #               inline_seq_dim=True,
  #           )

  #       print(f"  paged_attention_decode - layer_idx: {layer_idx}")
  #       print(f"  paged_attention_decode - lengths: {lengths}, shape: {lengths.shape}")
  #       print(f"  paged_attention_decode - page_indices: {page_indices}, shape: {page_indices.shape}")
  #       print(f"  paged_attention_decode - sequence_lengths: {page_state.sequence_lengths}, shape: {page_state.sequence_lengths.shape}")
  #       print(f"  paged_attention_decode - page_map: {page_state.page_map}, shape: {page_state.page_map.shape}")

  #       # Create sharded function
  #       wrapped_attention = shard_map.shard_map(
  #           attention_fn,
  #           mesh=self.mesh,
  #           in_specs=(q_pspec, k_pspec, v_pspec, lengths_pspec, page_indices_pspec),
  #           out_specs=q_pspec,
  #           check_rep=False,
  #       )

  #       # CRITICAL: Pre-create output tensor with exactly the right shape and dtype
  #       # This prevents allocation conflicts from reshaping after shard_map
  #       output_shape = (batch_size, 1, num_heads, head_dim)
  #       output = jnp.zeros(output_shape, dtype=query.dtype)

  #       # Execute within mesh context
  #       with self.mesh:
  #           # Call wrapped attention function
  #           result = wrapped_attention(q_input, k_pages, v_pages, lengths, page_indices)

  #           # Set result into pre-allocated output tensor
  #           # CRITICAL: Do this INSIDE the mesh context
  #           output = output.at[:, 0, :, :].set(result)

  #       print(f"Finishing paged_attention_v3_decode.")
  #       return output, None, None

  #   except Exception as e:
  #       if self.debug_prints:
  #           print(f"Error in paged attention v3: {str(e)}")
  #           import traceback
  #           traceback.print_exc()

  #       # Return zeros with correct shape on error
  #       print(f"Finishing paged_attention_v3_decode.")
  #       return jnp.zeros_like(query), None, None

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
    """Apply paged attention mechanism."""
    if page_state is None:
      if model_mode != common_types.MODEL_MODE_TRAIN and not self.is_initializing():
        raise ValueError(f"PagedAttentionOp requires page_state in {model_mode} mode")

    # Initialize or get the KV cache
    key_pages_var, value_pages_var = self.init_or_get_kv_pages(model_mode)

    # Always return a dictionary,
    # and during initialization, return a dictionary *containing* 'cache'

    # Only update pages if not initializing
    if not self.is_initializing():
      if model_mode == common_types.MODEL_MODE_PREFILL:
        self.update(key_pages_var, value_pages_var, key, value, model_mode, page_state, layer_idx)
        # Prefill uses simple dot-product attention.
        attn, local_max, local_sums = self.paged_dot_product_attention_with_max_and_sum(
            query, key, value
        )  # Return result directly
        # Return attention and cache
        return (
            {
                "cache": {
                    "cached_prefill_key": key_pages_var.value,
                    "cached_prefill_value": value_pages_var.value,
                }
            },
            attn,
            local_max,
            local_sums,
        )

      elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        # Get prepared inputs from paged_attention_decode
        q_input, k_pages, v_pages, lengths, page_indices = self.paged_attention_decode(
            query, key_pages_var, value_pages_var, page_state, layer_idx
        )
        # Return prepared inputs *and* the cache, along with the layer_idx.
        return (
            {
                "cache": {
                    "cached_prefill_key": key_pages_var.value,
                    "cached_prefill_value": value_pages_var.value,
                }
            },
            q_input,
            k_pages,
            v_pages,
            lengths,
            page_indices,
            layer_idx,
        )

      else:
        raise ValueError(f"Unsupported model_mode: {model_mode}")
    else:  # self.is_initializing() == True
      # We're initializing.  Return a dictionary with a 'cache' key.
      return {
          "cache": {
              "cached_prefill_key": key_pages_var.value,  # Use .value to get the array
              "cached_prefill_value": value_pages_var.value,  # Use .value to get the array
          }
      }

  def update(self, key_pages_var, value_pages_var, key, value, model_mode, page_state=None, layer_idx=0):
    """Update KV Pages with layer-specific page state."""
    # Skip updates during initialization
    print("Starting update")
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
    print("finishing update")

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
      
      # CHANGED: Check if we have enough total capacity (pages * tokens_per_page)
      total_capacity = v_n_p * v_p
      assert total_capacity >= t, f"Prefill cache is too small to accommodate the current request! Cache capacity: {total_capacity} tokens, request has {t} tokens"

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

  # In paged_attention.py, modify update_decode_step_pages
  def update_decode_step_pages(self, key_pages_var, value_pages_var, key, value, page_state, layer_idx=0):
    """Improved update method compatible with JAX tracing."""
    key_pages = key_pages_var.value
    value_pages = value_pages_var.value

    # Get shapes for validation
    batch_size, seq_len, kv_heads, head_dim = key.shape
    kv_heads_pages, num_pages, tokens_per_page, head_dim_pages = key_pages.shape

    # CRITICAL FIX: Don't use jnp.any() in control flow during tracing
    # Instead, just apply the corrections unconditionally

    # Validate batch size
    valid_batch_size = min(batch_size, page_state.page_map.shape[1])

    # Take the last token
    key_last = key[:valid_batch_size, -1, :, :]  # [valid_batch, kv_heads, head_dim]
    value_last = value[:valid_batch_size, -1, :, :]  # [valid_batch, kv_heads, head_dim]

    # Get current page and position indices
    current_pages = page_state.current_page[layer_idx, :valid_batch_size]  # [valid_batch]
    current_positions = page_state.current_page_position[layer_idx, :valid_batch_size]  # [valid_batch]

    # ALWAYS clip indices - don't use conditional logic
    current_pages = jnp.clip(current_pages, 0, num_pages - 1)
    current_positions = jnp.clip(current_positions, 0, tokens_per_page - 1)

    # Prepare tensors for update
    new_key = jnp.transpose(key_last, (1, 0, 2))  # [kv_heads, valid_batch, head_dim]
    new_value = jnp.transpose(value_last, (1, 0, 2))  # [kv_heads, valid_batch, head_dim]

    # Create broadcast arrays
    broadcast_pages = jnp.tile(current_pages[None, :], (kv_heads, 1))  # [kv_heads, valid_batch]
    broadcast_pos = jnp.tile(current_positions[None, :], (kv_heads, 1))  # [kv_heads, valid_batch]

    # Create indices for each KV head
    kv_indices = jnp.arange(kv_heads)[:, None]  # [kv_heads, 1]
    kv_indices = jnp.tile(kv_indices, (1, valid_batch_size))  # [kv_heads, valid_batch]

    # Update the key and value pages
    key_pages_updated = key_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_key)
    value_pages_updated = value_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_value)

    # Apply logical constraints
    key_pages_updated = nn.with_logical_constraint(key_pages_updated, self.kv_pages_axis_names)
    value_pages_updated = nn.with_logical_constraint(value_pages_updated, self.kv_pages_axis_names)

    # Update the variables
    key_pages_var.value = key_pages_updated
    value_pages_var.value = value_pages_updated

    return key_pages_var, value_pages_var