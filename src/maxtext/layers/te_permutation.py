# Copyright 2023-2025 Google LLC
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

"""TransformerEngine Permutation Integration for MaxText MoE.

This module provides wrapper functions for TransformerEngine's token dispatch
and combine operations used in Mixture of Experts (MoE) models.

The integration provides:
- te_permute: Route + dispatch tokens (delegates routing to te_router, then dispatches)
- te_unpermute: Combine expert outputs back to original token order
- te_token_dispatch: Low-level scatter tokens to their designated experts
- te_token_combine: Low-level gather tokens back from experts
- sort_chunks_by_index: Reorder chunks for expert parallelism
- Efficient extraction of tokens_per_expert for ragged_all_to_all

Key Design:
- Routing logic (fused top-k, aux loss, bias updates) lives in te_router.py.
- This module handles the permutation layer: dispatching tokens to experts
  and combining them back, using TE's optimized kernels.
- tokens_per_expert is computed efficiently inside the TE kernels,
  avoiding double iteration over the routing map.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from maxtext.layers import te_router

# Import TransformerEngine permutation primitives
try:
  from transformer_engine.jax.permutation import (
      token_dispatch,
      token_combine,
      sort_chunks_by_index,
  )
  TE_PERMUTATION_AVAILABLE = True
except ImportError:
  TE_PERMUTATION_AVAILABLE = False
  token_dispatch = None
  token_combine = None
  sort_chunks_by_index = None


def check_te_permutation_available():
  """Check if TransformerEngine permutation is available."""
  if not TE_PERMUTATION_AVAILABLE:
    raise ImportError(
        "TransformerEngine permutation is not available. "
        "Please install TransformerEngine with JAX support: "
        "pip install transformer-engine[jax]"
    )


def te_token_dispatch(
    inputs: jnp.ndarray,
    routing_map: jnp.ndarray,
    num_out_tokens: int,
    probs: Optional[jnp.ndarray] = None,
    align_size: Optional[int] = None,
) -> Tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    jnp.ndarray,
    Optional[jnp.ndarray],
    jnp.ndarray,
]:
  """Dispatch tokens to experts using TransformerEngine kernels.

  This is a wrapper around TE's token_dispatch that also returns
  tokens_per_expert computed efficiently inside the kernel (no double iteration).

  Args:
    inputs: Input tensor of shape [num_tokens, hidden].
    routing_map: Binary routing mask of shape [num_tokens, num_experts].
    num_out_tokens: Number of output tokens (typically num_tokens * top_k).
    probs: Optional routing probabilities for weighted dispatch.
    align_size: Optional alignment for padding (for efficient GEMM).

  Returns:
    output: Permuted tokens of shape [num_out_tokens, hidden].
    permuted_probs: Permuted probabilities (or None).
    row_id_map: Mapping for token_combine.
    pad_offsets: Padding offsets if align_size provided (or None).
    tokens_per_expert: Token counts per expert [num_experts].
                       Without padding: actual counts.
                       With padding: aligned counts.
  """
  check_te_permutation_available()

  # Call TE token_dispatch
  # tokens_per_expert is always returned (actual counts without padding,
  # aligned counts with padding)
  result = token_dispatch(
      inputs,
      routing_map,
      num_out_tokens=num_out_tokens,
      probs=probs,
      align_size=align_size,
  )

  output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = result

  # Validate tokens_per_expert is not None (should always be returned by TE >= latest)
  if tokens_per_expert is None:
    raise ValueError(
        "TE token_dispatch returned None for tokens_per_expert. "
        "Please ensure you have the latest TransformerEngine with the fix that "
        "always returns tokens_per_expert. Try: pip install -e /path/to/TransformerEngine"
    )

  return output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert


def te_token_combine(
    inputs: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray] = None,
    pad_offsets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Combine tokens from experts back to original positions.

  Args:
    inputs: Expert outputs of shape [num_out_tokens, hidden].
    row_id_map: Row ID map from te_token_dispatch.
    merging_probs: Optional weights for combining expert outputs.
    pad_offsets: Padding offsets from te_token_dispatch (if used).

  Returns:
    output: Combined tensor of shape [num_tokens, hidden].
  """
  check_te_permutation_available()

  return token_combine(
      inputs,
      row_id_map,
      merging_probs=merging_probs,
      pad_offsets=pad_offsets,
  )


def te_sort_chunks_by_expert(
    inputs: jnp.ndarray,
    tokens_per_chunk: jnp.ndarray,
    sorted_chunk_indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Sort chunks of tokens according to expert indices.

  This is used after all-to-all to reorder tokens so that all tokens
  for a given local expert are contiguous.

  Args:
    inputs: Input tensor of shape [num_tokens, hidden].
    tokens_per_chunk: Token counts per chunk [num_chunks].
    sorted_chunk_indices: Permutation of chunk indices [num_chunks].

  Returns:
    output: Sorted tensor.
    row_id_map: Map for reversing the sort.
  """
  check_te_permutation_available()

  return sort_chunks_by_index(inputs, tokens_per_chunk, sorted_chunk_indices)


def compute_ragged_all_to_all_params(
    all_shards_tokens_per_expert: jnp.ndarray,
    shard_id: int,
    num_expert_shards: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute ragged_all_to_all parameters from all shards' token counts.

  This is called after an all_gather of tokens_per_expert, so we have
  visibility into what every shard needs to send/receive.

  Args:
    all_shards_tokens_per_expert: Token counts from all shards.
                                  Shape [num_expert_shards, num_experts].
    shard_id: Current shard index.
    num_expert_shards: Number of expert-parallel shards.

  Returns:
    input_offsets: Shape [num_expert_shards].
    send_sizes: Shape [num_expert_shards].
    output_offsets: Shape [num_expert_shards].
    recv_sizes: Shape [num_expert_shards].
  """
  num_experts = all_shards_tokens_per_expert.shape[1]
  local_expert_size = num_experts // num_expert_shards

  # Get this shard's token counts
  # Use dynamic_slice since shard_id may be a traced value (from jax.lax.axis_index)
  local_tokens_per_expert = jax.lax.dynamic_slice(
      all_shards_tokens_per_expert,
      start_indices=(shard_id, 0),
      slice_sizes=(1, num_experts)
  ).squeeze(0)

  # Reshape to [num_expert_shards, local_expert_size]
  local_reshaped = local_tokens_per_expert.reshape(num_expert_shards, local_expert_size)

  # send_sizes[i] = tokens this shard sends to shard i
  send_sizes = jnp.sum(local_reshaped, axis=1)

  # input_offsets: cumulative send sizes
  input_offsets = jnp.concatenate([
      jnp.array([0], dtype=send_sizes.dtype),
      jnp.cumsum(send_sizes)[:-1]
  ])

  # recv_sizes[i] = tokens shard i sends to this shard
  # We need tokens that shard i has for our local experts
  # Our local experts are [shard_id * local_expert_size : (shard_id+1) * local_expert_size]
  local_expert_start = shard_id * local_expert_size

  # Use dynamic_slice since shard_id may be a traced value (from jax.lax.axis_index)
  # Extract columns [local_expert_start : local_expert_start + local_expert_size]
  local_expert_columns = jax.lax.dynamic_slice(
      all_shards_tokens_per_expert,
      start_indices=(0, local_expert_start),
      slice_sizes=(all_shards_tokens_per_expert.shape[0], local_expert_size)
  )

  # For each source shard, sum tokens destined for our experts
  recv_sizes = jnp.sum(local_expert_columns, axis=1)

  # output_offsets: SENDER-SIDE semantics for jax.lax.ragged_all_to_all.
  # output_offsets[j] = where shard j should place THIS shard's data in shard j's buffer.
  # This equals the cumulative sum of what shards 0..shard_id-1 sent to shard j.
  #
  # Build sends_to_target[i][j] = total tokens shard i sends to shard j
  # by reshaping all_shards_tokens_per_expert from (EP, num_experts) to
  # (EP, EP, local_expert_size) and summing over the local_expert dimension.
  sends_to_target = jnp.sum(
      all_shards_tokens_per_expert.reshape(
          num_expert_shards, num_expert_shards, local_expert_size
      ),
      axis=2,
  )  # shape: (num_expert_shards, num_expert_shards)

  # Prepend a zero row and cumsum along axis 0.
  # cumulated[i] = sum of sends_to_target[0..i-1] per target shard.
  zero_row = jnp.zeros((1, num_expert_shards), dtype=sends_to_target.dtype)
  array_with_zeros = jnp.concatenate([zero_row, sends_to_target], axis=0)
  cumulated = jnp.cumsum(array_with_zeros, axis=0, dtype=sends_to_target.dtype)
  output_offsets = jax.lax.dynamic_slice(
      cumulated,
      start_indices=(shard_id, 0),
      slice_sizes=(1, num_expert_shards),
  ).squeeze(0)

  return input_offsets, send_sizes, output_offsets, recv_sizes


def compute_reverse_ragged_all_to_all_params(
    all_shards_tokens_per_expert: jnp.ndarray,
    shard_id: int,
    num_expert_shards: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Compute reverse ragged_all_to_all parameters (for unpermute).

  In reverse, what we received in the forward pass becomes what we send,
  and vice versa. The sends_to_target matrix is transposed.

  Args:
    all_shards_tokens_per_expert: Token counts from all shards.
                                  Shape [num_expert_shards, num_experts].
    shard_id: Current shard index.
    num_expert_shards: Number of expert-parallel shards.

  Returns:
    input_offsets: Shape [num_expert_shards].
    send_sizes: Shape [num_expert_shards].
    output_offsets: Shape [num_expert_shards].
    recv_sizes: Shape [num_expert_shards].
  """
  num_experts = all_shards_tokens_per_expert.shape[1]
  local_expert_size = num_experts // num_expert_shards
  local_expert_start = shard_id * local_expert_size

  # In reverse: what we received becomes what we send.
  # We received tokens for our local experts from each source shard.
  local_expert_columns = jax.lax.dynamic_slice(
      all_shards_tokens_per_expert,
      start_indices=(0, local_expert_start),
      slice_sizes=(num_expert_shards, local_expert_size)
  )
  send_sizes = jnp.sum(local_expert_columns, axis=1)
  input_offsets = jnp.concatenate([
      jnp.array([0], dtype=send_sizes.dtype),
      jnp.cumsum(send_sizes)[:-1],
  ])

  # What we originally sent (now we receive back).
  local_tokens_per_expert = jax.lax.dynamic_slice(
      all_shards_tokens_per_expert,
      start_indices=(shard_id, 0),
      slice_sizes=(1, num_experts)
  ).squeeze(0)
  local_reshaped = local_tokens_per_expert.reshape(num_expert_shards, local_expert_size)
  recv_sizes = jnp.sum(local_reshaped, axis=1)

  # output_offsets: SENDER-SIDE semantics.
  # In reverse, sends_to_target is the transpose of the forward one.
  fwd_sends_to = jnp.sum(
      all_shards_tokens_per_expert.reshape(
          num_expert_shards, num_expert_shards, local_expert_size
      ),
      axis=2,
  )
  rev_sends_to = jnp.transpose(fwd_sends_to)

  zero_row = jnp.zeros((1, num_expert_shards), dtype=rev_sends_to.dtype)
  rev_cumulated = jnp.cumsum(
      jnp.concatenate([zero_row, rev_sends_to], axis=0),
      axis=0,
      dtype=rev_sends_to.dtype,
  )
  output_offsets = jax.lax.dynamic_slice(
      rev_cumulated,
      start_indices=(shard_id, 0),
      slice_sizes=(1, num_expert_shards),
  ).squeeze(0)

  return input_offsets, send_sizes, output_offsets, recv_sizes


def te_permute(
    inputs: jnp.ndarray,
    gate_logits: jnp.ndarray,
    gate_expert_bias: Optional[jnp.ndarray],
    num_experts: int,
    num_experts_per_tok: int,
    dtype: jnp.dtype,
    logits_dot_in_fp32: bool,
    routed_score_func: str,
    n_routing_groups: int,
    topk_routing_group: int,
    routed_scaling_factor: float,
    load_balance_loss_weight: float,
    should_update_load_balance: bool,
    routed_bias_update_rate: float,
    te_permutation_align_size: int,
    roll_to_expert_id: Optional[int] = None,
    num_experts_per_shard: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
           Optional[jnp.ndarray], jnp.ndarray, Optional[jnp.ndarray]]:
  """TE routing + permutation: fused router then TE token dispatch.

  Delegates routing to te_router.te_route(), then dispatches tokens
  to experts via te_token_dispatch().

  Args:
    inputs: Input tensor of shape [batch, seq, hidden].
    gate_logits: Raw GEMM logits [batch, seq, num_experts] (no score_func/bias).
    gate_expert_bias: Expert bias [num_experts] or empty array [0].
    num_experts: Total number of experts.
    num_experts_per_tok: Number of experts each token is routed to.
    dtype: Compute dtype.
    logits_dot_in_fp32: Whether to cast logits to float32 for routing.
    routed_score_func: Score function name ("softmax", "sigmoid", or "").
    n_routing_groups: Number of groups for grouped top-k routing.
    topk_routing_group: Top-k at group level.
    routed_scaling_factor: Scaling factor for output probabilities.
    load_balance_loss_weight: Weight for load balance loss (0 disables).
    should_update_load_balance: Whether to compute bias updates.
    routed_bias_update_rate: Rate for bias updates.
    te_permutation_align_size: Alignment size for padding (0 disables).
    roll_to_expert_id: Expert ID offset for ring-of-experts.
    num_experts_per_shard: Number of experts per shard in ring-of-experts mode.
      When set, routing decisions are masked to local experts only.

  Returns:
    Tuple of:
      - permuted_outputs: Tokens grouped by expert [num_out_tokens, hidden].
      - row_id_map: Mapping for unpermute phase.
      - tokens_per_expert: Token counts per expert [num_experts].
      - lb_loss: Scalar load balance loss (or None).
      - bias_updates: Bias update direction [num_experts] (or None).
      - sparse_probs: Sparse routing probs [num_tokens, num_experts] for combine.
      - pad_offsets: Padding offsets per expert (or None).
  """
  check_te_permutation_available()

  sparse_probs, routing_map, lb_loss, bias_updates = te_router.te_route(
      gate_logits,
      gate_expert_bias,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      dtype=dtype,
      logits_dot_in_fp32=logits_dot_in_fp32,
      routed_score_func=routed_score_func,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      routed_scaling_factor=routed_scaling_factor,
      load_balance_loss_weight=load_balance_loss_weight,
      should_update_load_balance=should_update_load_balance,
      routed_bias_update_rate=routed_bias_update_rate,
      roll_to_expert_id=roll_to_expert_id,
  )

  # In ring-of-experts mode, mask routing decisions to local experts only.
  # After rolling, experts [0, num_experts_per_shard) are the local ones.
  if num_experts_per_shard is not None:
    local_expert_mask = jnp.arange(num_experts) < num_experts_per_shard
    routing_map = routing_map * local_expert_mask[None, :]
    sparse_probs = sparse_probs * local_expert_mask[None, :].astype(sparse_probs.dtype)

  hidden_size = inputs.shape[-1]
  num_tokens = inputs.shape[0] * inputs.shape[1]
  num_out_tokens = num_tokens * num_experts_per_tok

  align_size = te_permutation_align_size if te_permutation_align_size > 0 else None

  permuted_outputs, _permuted_probs, row_id_map, pad_offsets, tokens_per_expert = (
      te_token_dispatch(
          inputs.reshape(-1, hidden_size),
          routing_map,
          num_out_tokens=num_out_tokens,
          probs=sparse_probs,
          align_size=align_size,
      )
  )

  return (permuted_outputs, row_id_map, tokens_per_expert, lb_loss,
          bias_updates, sparse_probs, pad_offsets)


def te_unpermute(
    expert_outputs: jnp.ndarray,
    row_id_map: jnp.ndarray,
    batch_size: int,
    sequence_length: int,
    dtype: jnp.dtype,
    is_llama4: bool = False,
    dense_probs: Optional[jnp.ndarray] = None,
    pad_offsets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Unpermute expert outputs using TransformerEngine kernels.

  Args:
    expert_outputs: Output from grouped GEMM [num_out_tokens, hidden].
    row_id_map: Row ID map from te_permute.
    batch_size: Original batch size.
    sequence_length: Original sequence length.
    dtype: Output dtype.
    is_llama4: If True, skip weighted combination (implicit weights of 1).
    dense_probs: Dense routing probs [num_tokens, num_experts] from te_permute.
                 Used as merging_probs for weighted combination.
    pad_offsets: Padding offsets per expert from te_permute (for unpadding).

  Returns:
    Combined output tensor [batch, seq, hidden].
  """
  check_te_permutation_available()

  merging_probs = None
  if not is_llama4:
    merging_probs = dense_probs

  output = te_token_combine(
      expert_outputs,
      row_id_map,
      merging_probs=merging_probs,
      pad_offsets=pad_offsets,
  )

  return output.reshape(batch_size, sequence_length, -1).astype(dtype)
