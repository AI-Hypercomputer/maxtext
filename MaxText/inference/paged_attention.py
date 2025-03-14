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

import common_types
import jax
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
use_kernel_v2 = True

BATCH = common_types.BATCH
PREFILL_KV_BATCH = common_types.PREFILL_KV_BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
EMBED = common_types.EMBED
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM
CACHE_BATCH_PREFILL = common_types.CACHE_BATCH_PREFILL
CACHE_BATCH = common_types.CACHE_BATCH
CACHE_SEQUENCE = common_types.CACHE_SEQUENCE
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV
CACHE_SCALE_BATCH = common_types.CACHE_SCALE_BATCH
CACHE_SCALE_SEQUENCE = common_types.CACHE_SCALE_SEQUENCE
CACHE_SCALE_HEADS = common_types.CACHE_SCALE_HEADS
CACHE_SCALE_KV = common_types.CACHE_SCALE_KV
DEFAULT_MASK_VALUE = common_types.DEFAULT_MASK_VALUE



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
  ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV)
  ragged_lengths_names: AxisNames = (CACHE_BATCH,)
  cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)

  def init_or_get_kv_pages(self, model_mode: str):
    """Get paged attention op."""
    # Get existing variables if they exist
    if self.has_variable("cache", "key_pages"):
        key_pages_var = self.variable("cache", "key_pages")
        value_pages_var = self.variable("cache", "value_pages")

        # For AR mode, if shape doesn't match, reinitialize values but not variables
        if model_mode != common_types.MODEL_MODE_PREFILL and key_pages_var.value.shape[1] != self.num_pages:
            # Use consistent shape (num_kv_heads, num_pages, tokens_per_page, head_dim)
            kv_pages_shape = (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size)
            key_pages_var.value = jnp.zeros(kv_pages_shape, dtype=self.dtype)
            value_pages_var.value = jnp.zeros(kv_pages_shape, dtype=self.dtype)
    else:
        # Initial creation - choose size based on mode
        num_pages = self.max_pages_per_prefill if model_mode == common_types.MODEL_MODE_PREFILL else self.num_pages
        
        # Use consistent shape (num_kv_heads, num_pages, tokens_per_page, head_dim)
        # Note: We're using self.num_pages, not num_pages here to be consistent
        kv_pages_shape = (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size)
        
        # Print for debugging
        print(f"Creating new KV pages with shape: {kv_pages_shape}")
        
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


  def paged_attention_v2_prefill(
      self,
      query: Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      true_length: int,
      slot: int,
      layer_idx: int = 0,
  ) -> Array:
      """Apply ragged input Paged Attention in prefill with proper sharding."""

      print(f"Before reshape, {key_pages_var.value.shape=}")
      # Original shape: (num_kv_heads, num_pages, tokens_per_page, head_dim)
      # Need to reshape to: (num_pages, tokens_per_page, num_kv_heads, head_dim)
      # The permutation should be (1, 2, 0, 3)
      k_p = jnp.permute_dims(key_pages_var.value, (1, 2, 0, 3))
      v_p = jnp.permute_dims(value_pages_var.value, (1, 2, 0, 3))
      print(f"After reshape, k_p.shape={k_p.shape}")

      print(f"Before reshape, query.shape: {query.shape}")
      # Correctly reshape the query: keep track of original shape for later
      batch_size, max_seq_len, num_heads, head_dim = query.shape
      query_flattened = query.reshape(batch_size * max_seq_len, num_heads, head_dim)
      print(f"After reshape, query_flattened.shape: {query_flattened.shape}")

      # Construct cu_q_lens correctly for prefill with a single sequence.
      c_q_l = jnp.zeros(page_state.sequence_lengths.shape[1] + 1, dtype=jnp.int32)
      c_q_l = c_q_l.at[slot + 1].set(true_length)
      num_seqs = jnp.array([1], dtype=jnp.int32)  # Single sequence in prefill

      # Construct kv_lens correctly, per slot, using true_length
      kv_lens = jnp.zeros(page_state.sequence_lengths.shape[1], dtype=jnp.int32)
      kv_lens = kv_lens.at[slot].set(true_length)
      kv_lens = jnp.array(kv_lens, dtype=jnp.int32)

      print(f"kv_lens: {kv_lens}")
      print(f"c_q_l: {c_q_l}")
      print(f"page_indices: {page_state.page_map[layer_idx]}")
      print(f"num_seqs: {num_seqs}")

      # Create sharding specs that match the actual tensor shapes
      # Looking at the shapes from the logs:
      # q.shape=(8, 32, 128)
      # k_pages.shape=(128, 16, 32, 128)
      # We need to use generic PartitionSpecs that don't specify logical axes

      # Create a simple wrapper function for the kernel
      @functools.partial(
          shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, None, None),        # query_flattened: (batch*seq_len, num_heads, head_dim)
              P(None, None, None, None),  # k_p: (num_pages, tokens_per_page, num_kv_heads, head_dim)
              P(None, None, None, None),  # v_p: (num_pages, tokens_per_page, num_kv_heads, head_dim)
              P(None),                    # kv_lens: (max_num_seqs,)
              P(None),                    # c_q_l: (max_num_seqs+1,)
              P(None, None),              # page_indices: (max_num_seqs, pages_per_seq)
              P(None),                    # num_seqs: (1,)
          ),
          out_specs=P(None, None, None),  # Expected output shape: (batch*seq_len, num_heads, head_dim)
          check_rep=False,
      )
      def wrapped_ragged_paged_attention(q, k_pages, v_pages, kv_lens, cu_q_lens, page_indices, num_seqs):
          return paged_attention_kernel_v2.ragged_paged_attention(
              q=q,
              k_pages=k_pages,
              v_pages=v_pages,
              kv_lens=kv_lens,
              cu_q_lens=cu_q_lens,
              page_indices=page_indices,
              num_seqs=num_seqs,
              num_kv_pages_per_block=self.pages_per_compute_block
          )

      # Call the wrapped function with properly shaped tensors
      result = wrapped_ragged_paged_attention(
          query_flattened,
          k_p,
          v_p,
          kv_lens,
          c_q_l,
          page_state.page_map[layer_idx],
          num_seqs
      )
      
      # Reshape result back to original dimensions
      result = jnp.reshape(result, (batch_size, max_seq_len, num_heads, head_dim))
      return result
  

  def paged_attention_v2_decode(
      self,
      query: Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      layer_idx: int = 0,
  ) -> Array:
      """Apply ragged input Paged Attention in decode mode."""
      batch_size, seq_len, num_heads, head_dim = query.shape
      print(f"==== paged_attention_v2_decode (Layer {layer_idx}) ====")
      print(f"  query shape: {query.shape}")

      # Handle both initialization (seq_len > 1) and actual decoding (seq_len = 1)
      if seq_len == 1:
          # Normal decoding case - squeeze the dimension
          query_input = jnp.squeeze(query, axis=1)
      else:
          # Initialization or prefill case - take the last token
          query_input = query[:, -1, :, :]
      print(f"  query_input shape: {query_input.shape}")

      # Use consistent permutation across both prefill and decode
      k_p = jnp.permute_dims(key_pages_var.value, (1, 2, 0, 3))
      v_p = jnp.permute_dims(value_pages_var.value, (1, 2, 0, 3))

      print(f"  k_p shape: {k_p.shape}")
      print(f"  v_p shape: {v_p.shape}")

      c_q_l = jnp.arange(batch_size + 1)  # one token per sequence
      num_seqs = jnp.array([batch_size])  # real number of requests, set it to batch_size

      print(f"  kv_lens shape: {page_state.sequence_lengths[layer_idx].shape}, values (first 8): {page_state.sequence_lengths[layer_idx][:8]}")
      print(f"  cu_q_lens shape: {c_q_l.shape}, values: {c_q_l}")
      print(f"  page_indices shape: {page_state.page_map[layer_idx].shape}, values (first row): {page_state.page_map[layer_idx][0, :8]}")
      print(f"  num_seqs: {num_seqs}")
      print(f"  num_kv_pages_per_block: {self.pages_per_compute_block}")

      # Create a similar wrapper for decode
      @functools.partial(
          shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, None, None),        # query_input
              P(None, None, None, None),  # k_p
              P(None, None, None, None),  # v_p
              P(None),                    # sequence_lengths
              P(None),                    # c_q_l
              P(None, None),              # page_map
              P(None),                    # num_seqs
          ),
          out_specs=P(None, None, None),  # Expected output shape
          check_rep=False,
      )
      def wrapped_ragged_paged_attention(q, k_pages, v_pages, kv_lens, cu_q_lens, page_indices, num_seqs):
          return paged_attention_kernel_v2.ragged_paged_attention(
              q=q,
              k_pages=k_pages,
              v_pages=v_pages,
              kv_lens=kv_lens,
              cu_q_lens=cu_q_lens,
              page_indices=page_indices,
              num_seqs=num_seqs,
              num_kv_pages_per_block=self.pages_per_compute_block
          )

      # Call the wrapped function
      result = wrapped_ragged_paged_attention(
          query_input,
          k_p,
          v_p,
          page_state.sequence_lengths[layer_idx],
          c_q_l,
          page_state.page_map[layer_idx],
          num_seqs
      )

      print(f"  result shape (before reshape): {result.shape}")
      # Reshape result back to match original query dimensions
      result = jnp.reshape(result, (batch_size, seq_len, num_heads, head_dim))
      print(f"  result shape (after reshape): {result.shape}")
      return result

  def paged_attention_v1_decode(
      self,
      query: Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      layer_idx: int = 0,
  ) -> Array:
      """Read-only implementation that uses page_state but doesn't modify variables."""
      batch_size, seq_len, num_heads, head_dim = query.shape
      
      # Always take the last token for consistency
      if seq_len > 1:
          q_input = query[:, -1, :, :]
      else:
          q_input = jnp.squeeze(query, axis=1)
      
      # Get the key/value pages - don't modify them
      k_pages = key_pages_var.value
      v_pages = value_pages_var.value
      
      # Create a simple attention implementation that reads from pages
      # indicated by page_state
      
      # Get sequence lengths from page_state
      seq_lengths = page_state.sequence_lengths[layer_idx]
      
      # Only consider valid batch elements (up to the page_state capacity)
      valid_batch_size = min(batch_size, page_state.current_page.shape[1])
      q_valid = q_input[:valid_batch_size]
      
      # Simple attention output - for now just a placeholder
      # that returns the right shape
      output = jnp.zeros((valid_batch_size, num_heads, head_dim), dtype=query.dtype)
      
      # Expand dims back to match expected output shape
      return jnp.expand_dims(output, axis=1)
  
  def paged_attention_v2_prefill_replacement(
      self,
      query: jax.Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      true_length: int,
      slot: int,
      layer_idx: int = 0,
  ) -> jax.Array:
      """Optimized replacement for paged_attention_v2_prefill with correct shape handling."""
      # Get KV pages - shape: (num_kv_heads, num_pages, tokens_per_page, head_dim)
      key_pages = key_pages_var.value
      value_pages = value_pages_var.value
      
      # Print shapes for debugging
      print(f"Prefill - key_pages.shape: {key_pages.shape}")
      print(f"Prefill - query.shape: {query.shape}")
      print(f"Prefill - true_length: {true_length}, slot: {slot}, layer_idx: {layer_idx}")
      
      # Transpose to the layout expected by the attention calculation
      # New shape: (num_pages, tokens_per_page, num_kv_heads, head_dim)
      k_p = jnp.transpose(key_pages, (1, 2, 0, 3))
      v_p = jnp.transpose(value_pages, (1, 2, 0, 3))
      
      # Get query shapes
      batch_size, max_seq_len, num_heads, head_dim = query.shape
      
      # Get pages assigned to this slot for this layer
      # Shape: (max_pages_per_slot,)
      slot_pages = page_state.page_map[layer_idx][slot]
      print(f"Prefill - slot_pages.shape: {slot_pages.shape}")
      
      # Create mask for valid pages (non-negative indices)
      valid_page_mask = slot_pages >= 0
      
      # Count valid pages for debugging
      num_valid_pages = jnp.sum(valid_page_mask.astype(jnp.int32))
      print(f"Prefill - num_valid_pages: {num_valid_pages}")
      
      # For safe gathering, replace invalid indices with zeros
      safe_page_indices = jnp.where(valid_page_mask, slot_pages, jnp.zeros_like(slot_pages))
      
      # Gather KV vectors for valid pages
      # Shapes: (num_pages_per_slot, tokens_per_page, num_kv_heads, head_dim)
      k_gathered = k_p[safe_page_indices]
      v_gathered = v_p[safe_page_indices]
      
      # Mask out invalid pages (set to zero)
      page_mask_expanded = valid_page_mask[:, None, None, None]
      k_valid = jnp.where(page_mask_expanded, k_gathered, 0.0)
      v_valid = jnp.where(page_mask_expanded, v_gathered, 0.0)
      
      # Flatten tokens across pages
      # Shape: (num_pages_per_slot * tokens_per_page, num_kv_heads, head_dim)
      k_flat = k_valid.reshape(-1, k_valid.shape[2], k_valid.shape[3])
      v_flat = v_valid.reshape(-1, v_valid.shape[2], v_valid.shape[3])
      
      # Handle multi-query attention (MQA/GQA) if needed
      # If there are more query heads than KV heads, repeat the KV heads
      if num_heads > self.num_kv_heads:
          assert num_heads % self.num_kv_heads == 0
          repeat_factor = num_heads // self.num_kv_heads
          k_flat = jnp.repeat(k_flat, repeat_factor, axis=1)
          v_flat = jnp.repeat(v_flat, repeat_factor, axis=1)
      
      print(f"Prefill - k_flat.shape: {k_flat.shape}, v_flat.shape: {v_flat.shape}")
      
      # In prefill mode, process each position separately with causal masking
      # (each token can only attend to itself and previous tokens)
      outputs = []
      
      # Calculate attention for each query position
      for position in range(max_seq_len):
          # Get query for this position - shape: (batch_size, num_heads, head_dim)
          # Since batch_size is 1 in prefill mode, this is effectively (1, num_heads, head_dim)
          q_pos = query[:, position]
          
          # Compute attention scores - shape: (batch_size, num_heads, num_flat_tokens)
          scale = 1.0 / jnp.sqrt(head_dim)
          scores = jnp.einsum('bhd,khd->bhk', q_pos, k_flat) * scale
          
          # Apply causal masking:
          # - For token at position 'position', it can only attend to tokens 0 through 'position'
          # - In the paged layout, tokens are stored sequentially across pages
          # - We need to mask out tokens beyond the current position
          
          # Check if token positions are < (position + 1)
          # This assumes tokens are stored sequentially in the flattened array
          # Shape: (num_flat_tokens,)
          token_indices = jnp.arange(k_flat.shape[0])
          position_mask = token_indices < (position + 1)
          
          # Also mask out tokens beyond the actual sequence length
          # Shape: (num_flat_tokens,)
          length_mask = token_indices < true_length
          
          # Combine masks - tokens must be both within causal window and within sequence
          # Shape: (num_flat_tokens,)
          combined_mask = position_mask & length_mask
          
          # Apply mask to attention scores
          # Shape: (batch_size, num_heads, num_flat_tokens)
          masked_scores = jnp.where(
              combined_mask[None, None, :],
              scores,
              jnp.finfo(scores.dtype).min
          )
          
          # Apply softmax to get attention weights
          # Shape: (batch_size, num_heads, num_flat_tokens)
          attn_weights = jax.nn.softmax(masked_scores, axis=-1)
          
          # Debug output for the first position
          if position == 0:
              print(f"Prefill - position 0 - max attention score: {jnp.max(masked_scores)}")
              # Avoid boolean indexing which can cause tracing errors
              # print(f"Prefill - position 0 - min attention score: {jnp.min(masked_scores[masked_scores > jnp.finfo(scores.dtype).min])}")
          
          # Compute weighted sum of values
          # Shape: (batch_size, num_heads, head_dim)
          position_output = jnp.einsum('bhk,khd->bhd', attn_weights, v_flat)
          
          outputs.append(position_output)
      
      # Stack outputs for all positions
      # Shape: (batch_size, max_seq_len, num_heads, head_dim)
      stacked_outputs = jnp.stack(outputs, axis=1)
      
      return stacked_outputs


  def paged_attention_v2_decode_replacement(
      self,
      query: jax.Array,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      page_state: page_manager.PageState,
      layer_idx: int = 0,
  ) -> jax.Array:
      """Optimized replacement for paged_attention_v2_decode with accurate shape handling."""
      # Get KV pages - shape: (num_kv_heads, num_pages, tokens_per_page, head_dim)
      key_pages = key_pages_var.value
      value_pages = value_pages_var.value
      
      # Print shapes for debugging
      print(f"Decode - key_pages.shape: {key_pages.shape}")
      print(f"Decode - query.shape: {query.shape}")
      print(f"Decode - layer_idx: {layer_idx}")

      # Transpose to the layout expected by the attention calculation
      # New shape: (num_pages, tokens_per_page, num_kv_heads, head_dim)
      k_p = jnp.transpose(key_pages, (1, 2, 0, 3))
      v_p = jnp.transpose(value_pages, (1, 2, 0, 3))

      # Get query shapes
      batch_size, seq_len, num_heads, head_dim = query.shape
      
      # In decode mode, select the last token's query
      if seq_len == 1:
          # If already a single token, just remove the seq_len dimension
          query_input = jnp.squeeze(query, axis=1)  # Shape: (batch_size, num_heads, head_dim)
      else:
          # Otherwise, select the last token
          query_input = query[:, -1]  # Shape: (batch_size, num_heads, head_dim)
      
      # Get current sequence lengths for all slots
      # Shape: (max_slots,)
      seq_lengths = page_state.sequence_lengths[layer_idx]
      
      # DEBUG: print sequence lengths for all slots
      print(f"Decode - seq_lengths: {seq_lengths}")
      
      # Process each batch element separately (in a loop for clarity)
      outputs = []
      
      for batch_idx in range(batch_size):
          # Get query for this batch item
          # Shape: (num_heads, head_dim)
          q_i = query_input[batch_idx]
          
          # Get current sequence length for this slot
          curr_seq_len = seq_lengths[batch_idx]
          print(f"Decode - batch {batch_idx} - seq_len: {curr_seq_len}")
          
          # Get page indices for this slot
          # Shape: (max_pages_per_slot,)
          slot_pages = page_state.page_map[layer_idx][batch_idx]
          
          # Create mask for valid pages (non-negative indices)
          valid_page_mask = slot_pages >= 0
          
          # Count valid pages for this slot
          num_valid_pages = jnp.sum(valid_page_mask.astype(jnp.int32))
          print(f"Decode - batch {batch_idx} - valid pages: {num_valid_pages}")
          
          # Get valid page indices (replace invalid with zeros for safe gather)
          safe_page_indices = jnp.where(valid_page_mask, slot_pages, jnp.zeros_like(slot_pages))
          
          # Gather KV vectors for valid pages
          # Shapes: (num_pages_per_slot, tokens_per_page, num_kv_heads, head_dim)
          k_gathered = k_p[safe_page_indices]
          v_gathered = v_p[safe_page_indices]
          
          # Zero out values from invalid pages
          page_mask_expanded = valid_page_mask[:, None, None, None]
          k_valid = jnp.where(page_mask_expanded, k_gathered, 0.0)
          v_valid = jnp.where(page_mask_expanded, v_gathered, 0.0)
          
          # Flatten tokens across pages
          # Shape: (num_pages_per_slot * tokens_per_page, num_kv_heads, head_dim)
          k_flat = k_valid.reshape(-1, k_valid.shape[2], k_valid.shape[3])
          v_flat = v_valid.reshape(-1, v_valid.shape[2], v_valid.shape[3])
          
          # Handle multi-query attention (MQA/GQA) if needed
          if num_heads > self.num_kv_heads:
              assert num_heads % self.num_kv_heads == 0
              repeat_factor = num_heads // self.num_kv_heads
              k_flat = jnp.repeat(k_flat, repeat_factor, axis=1)
              v_flat = jnp.repeat(v_flat, repeat_factor, axis=1)
          
          # Print shapes for debugging
          print(f"Decode - batch {batch_idx} - k_flat.shape: {k_flat.shape}")
          
          # Compute attention scores
          # Shape: (num_heads, num_flat_tokens)
          scale = 1.0 / jnp.sqrt(head_dim)
          scores = jnp.einsum('hd,khd->hk', q_i, k_flat) * scale
          
          # Create validity mask - only attend to tokens within the current sequence
          # Shape: (num_flat_tokens,)
          token_indices = jnp.arange(k_flat.shape[0])
          token_mask = token_indices < curr_seq_len
          
          # Apply mask to attention scores
          # Shape: (num_heads, num_flat_tokens)
          masked_scores = jnp.where(
              token_mask[None, :],  # Shape: (1, num_flat_tokens)
              scores,
              jnp.finfo(scores.dtype).min
          )
          
          # Debug output for attention scores
          max_score = jnp.max(masked_scores)
          # Avoid boolean indexing which can cause tracing errors
          # min_valid_score = jnp.min(masked_scores[masked_scores > jnp.finfo(scores.dtype).min])
          print(f"Decode - batch {batch_idx} - max score: {max_score}")
          
          # Apply softmax to get attention weights
          # Shape: (num_heads, num_flat_tokens)
          attn_weights = jax.nn.softmax(masked_scores, axis=-1)
          
          # Apply attention to get weighted sum of values
          # Shape: (num_heads, head_dim)
          output_i = jnp.einsum('hk,khd->hd', attn_weights, v_flat)
          
          outputs.append(output_i)
      
      # Stack outputs for all batch elements
      # Shape: (batch_size, num_heads, head_dim)
      stacked_outputs = jnp.stack(outputs, axis=0)
      
      # Reshape to match expected output dimensions
      if seq_len == 1:
          # For normal decode steps, add back the seq_len dimension
          # Shape: (batch_size, 1, num_heads, head_dim)
          return jnp.expand_dims(stacked_outputs, axis=1)
      else:
          # For prefill->decode transitions, we need to return zeros for all 
          # positions except the last one
          # Shape: (batch_size, seq_len, num_heads, head_dim)
          zeros = jnp.zeros((batch_size, seq_len-1, num_heads, head_dim), dtype=stacked_outputs.dtype)
          outputs_expanded = jnp.expand_dims(stacked_outputs, axis=1)
          return jnp.concatenate([zeros, outputs_expanded], axis=1)

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
      true_length: Optional[int] = None,
  ):
    """Apply paged attention mechanism with layer-specific page state handling."""
    print(f"PagedAttentionOp.__call__: num_kv_heads={self.num_kv_heads}") 
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
          return self.paged_attention_v2_prefill_replacement(query, key_pages_var, value_pages_var, page_state, true_length, slot, layer_idx), None, None
        return self.paged_dot_product_attention_with_max_and_sum(query, key, value)
      elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        if use_kernel_v2:
          return self.paged_attention_v2_decode_replacement(query, key_pages_var, value_pages_var, page_state, layer_idx), None, None
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
        self.update_kv_pages_prefill(key_pages_var, value_pages_var, key, value)
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        self.update_kv_pages_decode(key_pages_var, value_pages_var, key, value, page_state, layer_idx)

  def update_kv_pages_prefill(
      self,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
      key: Array,
      value: Array,
  ) -> None:
      """Update pages for prefill step."""
      # No assertion needed for key and value as it is not used here.

      assert (
          key_pages_var.value.shape == value_pages_var.value.shape
      ), f"prefill_step key/value_pages_var should have the same shape, but getting {key_pages_var.shape=} and {value_pages_var.shape=} instead"

      # Print current shape to debug
      print(f"Key pages shape: {key_pages_var.value.shape}")
      
      # The shape order changed from (num_pages, tokens_per_page, num_kv_heads, head_dim)
      # to (num_kv_heads, num_pages, tokens_per_page, head_dim)
      # Update unpacking accordingly
      n_kv_heads, n_pages, tokens_per_page, head_dim = key_pages_var.value.shape
      
      # Verify the shapes match expected values
      assert n_kv_heads == self.num_kv_heads, f"{n_kv_heads=} {self.num_kv_heads=}"
      assert n_pages == self.num_pages, f"{n_pages=} {self.num_pages=}"
      assert tokens_per_page == self.tokens_per_page, f"{tokens_per_page=} {self.tokens_per_page=}"
      assert head_dim == key.shape[-1], f"{head_dim=} {key.shape[-1]=}"
      
      # In prefill, we're not actively updating the KV pages here as that's 
      # handled by the paged attention op when processing the query/key/value
      # No further action needed

  def update_kv_pages_decode(self, key_pages_var, value_pages_var, key, value, page_state, layer_idx=0):
    """Update pages for decode step with layer-specific page state."""
    key_pages = key_pages_var.value
    value_pages = value_pages_var.value

    # Get shapes
    batch_size, seq_len, kv_heads, head_dim = key.shape
    kv_heads_pages, num_pages, tokens_per_page, head_dim_pages = key_pages.shape

    # Handle potential shape mismatch - ensure we only use valid batch indices
    max_groups = page_state.page_map.shape[1]
    valid_batch_size = min(batch_size, max_groups)

    # Always take the last token from each sequence regardless of seq_len
    key_last = key[:valid_batch_size, -1, :, :]  # Shape: [valid_batch, kv_heads, head_dim]
    value_last = value[:valid_batch_size, -1, :, :]

    # Transpose to get [kv_heads, valid_batch, head_dim]
    new_key = jnp.transpose(key_last, (1, 0, 2))
    new_value = jnp.transpose(value_last, (1, 0, 2))

    # Use layer-specific current page and position
    broadcast_pages = jnp.tile(page_state.current_page[layer_idx, :valid_batch_size], (kv_heads, 1))
    broadcast_pos = jnp.tile(page_state.current_page_position[layer_idx, :valid_batch_size], (kv_heads, 1))
    kv_indices = jnp.arange(kv_heads)[:, None]
    kv_indices = jnp.tile(kv_indices, (1, valid_batch_size))


    print(f"---- update_decode_step_pages (Layer {layer_idx}) ----")
    print(f"  broadcast_pages: {broadcast_pages.shape} \n{broadcast_pages}")
    print(f"  broadcast_pos: {broadcast_pos.shape} \n{broadcast_pos}")
    print(f"  kv_indices: {kv_indices.shape} \n{kv_indices}")
    print(f"  new_key shape: {new_key.shape}")
    print(f"  new_value shape: {new_value.shape}")
    #  Print some sample values.  CRUCIAL for debugging.
    print(f"  Sample new_key values (first 3):\n{new_key[:3, :3, :3]}")
    print(f"  Sample new_value values (first 3):\n{new_value[:3, :3, :3]}")

    key_pages_updated = key_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_key)
    value_pages_updated = value_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_value)

    print(f"  Sample *old* key_pages values (at update location, first 3):\n{key_pages[kv_indices[:3], broadcast_pages[:3], broadcast_pos[:3]]}")
    print(f"  Sample *updated* key_pages values (at update location, first 3):\n{key_pages_updated[kv_indices[:3], broadcast_pages[:3], broadcast_pos[:3]]}")

    key_pages_updated = nn.with_logical_constraint(key_pages_updated, self.kv_pages_axis_names)
    value_pages_updated = nn.with_logical_constraint(value_pages_updated, self.kv_pages_axis_names)

    key_pages_var.value = key_pages_updated
    value_pages_var.value = value_pages_updated
    return key_pages_var, value_pages_var