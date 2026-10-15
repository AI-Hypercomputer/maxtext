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

"""Ragged token sorting operations with custom VJP."""

import jax
import jax.numpy as jnp
from maxtext.kernels.ragged.ragged_gather import ragged_gather
from maxtext.kernels.ragged.ragged_gather_reduce_v2 import ragged_gather_reduce
from maxtext.kernels.ragged.ragged_gather_tc import packed_stable_sort

_INDEX_MATH_MODES = ("sort", "count")


def _check_index_math(index_math):
  if index_math not in _INDEX_MATH_MODES:
    raise ValueError(f"index_math must be one of {_INDEX_MATH_MODES}, got {index_math!r}")


def _local_slot_positions(topk_indices_local, num_experts, shard_idx, ep_size, block=128):
  """Position of every flat (token, k) slot routed to this shard, inside this shard's window of the sorted slots.

  Equals ``argsort(argsort(flat_ids)) - shard_output_start`` on the slots of this shard's experts (``argsort``
  stable, by expert id) and is -1 on every other slot. It needs no sort and no scatter: in the stable order by
  expert id, slot (t, k) with local expert e comes after
    - every slot of a lower local expert (an exclusive prefix sum of the local expert counts),
    - every slot of expert e in an earlier token (an exclusive prefix count over tokens), and
    - every earlier slot of token t that also picked e (0 unless a token repeats an expert).
  The prefix count over tokens runs per block of ``block`` tokens as a strictly-lower-triangular matmul (its
  inputs are counts <= topk, exact in bfloat16, and it accumulates in float32, exact below 2**24), plus an int32
  prefix sum over the block totals.

  Args:
    topk_indices_local: ``[num_tokens, topk]`` expert ids. Ids outside this shard's experts (including the
      ``num_experts`` padding id) are never local.
    num_experts: total number of experts.
    shard_idx: this shard's index on the expert-parallel axis (a traced scalar is fine).
    ep_size: number of expert shards.
    block: tokens per block of the prefix count.

  Returns:
    ``[num_tokens * topk]`` int32 positions in ``[0, local slots)``, or -1.
  """
  num_tokens, topk = topk_indices_local.shape
  local_num_experts = num_experts // ep_size
  pad = (-num_tokens) % block
  ids = topk_indices_local.astype(jnp.int32)
  if pad:
    ids = jnp.pad(ids, ((0, pad), (0, 0)), constant_values=num_experts)  # the padding id: never local
  t = ids.shape[0]
  le = ids - shard_idx * local_num_experts  # local expert id, in [0, local_num_experts) iff the slot is local
  is_local = (le >= 0) & (le < local_num_experts)
  hit = le[:, :, None] == jnp.arange(local_num_experts, dtype=jnp.int32)[None, None, :]  # [t, topk, E_local]
  counts_per_token = jnp.sum(hit, axis=1, dtype=jnp.int32)  # [t, E_local]
  kk = jnp.arange(topk, dtype=jnp.int32)
  dup_before = jnp.sum(
      (le[:, :, None] == le[:, None, :]) & (kk[None, None, :] < kk[None, :, None]), axis=-1, dtype=jnp.int32
  )  # [t, topk]: earlier slots of the same token with the same expert
  blocks = counts_per_token.reshape(t // block, block, local_num_experts)
  strict_lower = jnp.tril(jnp.ones((block, block), jnp.bfloat16), k=-1)
  # Slots of each local expert in the earlier tokens of the same block.
  within = jnp.einsum(
      "ij,bjk->bik", strict_lower, blocks.astype(jnp.bfloat16), preferred_element_type=jnp.float32
  ).astype(jnp.int32)
  block_totals = jnp.sum(blocks, axis=1, dtype=jnp.int32)  # [t // block, E_local]
  before_block = jnp.cumsum(block_totals, axis=0) - block_totals
  earlier = (within + before_block[:, None, :]).reshape(t, local_num_experts)  # [t, E_local]
  local_counts = jnp.sum(block_totals, axis=0)
  local_offsets = jnp.cumsum(local_counts) - local_counts
  # One select-reduce over the local experts picks each slot's own expert's terms (no gather).
  pos = jnp.sum(jnp.where(hit, earlier[:, None, :] + local_offsets[None, None, :], 0), axis=-1) + dup_before
  pos = jnp.where(is_local, pos, -1)[:num_tokens]
  return pos.reshape(-1)


def ring_ragged_sort(
    hidden_states_local,
    topk_indices_local,
    num_experts,
    topk,
    ep_name,
    ep_size,
    buffer_size=None,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
    use_single_sparsecore=False,
    tag_routing_fn=None,
    index_math="sort",
):
  """Ragged-gather variant for AG-RS Expert Parallelism token routing.

  Unlike :func:`a2a_ragged_sort`, which operates on a valid prefix within a single shard,
  this function sorts and gathers tokens across the global expert space but extracts
  only the specific range of output rows assigned to the experts residing on this shard.
  The rest of the shard's output buffer is padded with zeros.

  Forward:
    Sorts tokens based on ``topk_indices_local`` and gathers
    ``hidden_states_local[token_indices_sorted[i]]`` into ``out[i]`` for
    ``i`` in ``[shard_output_start, shard_output_end)``; other rows are zeroed.

  Backward (gather-reduce):
    ``g_hidden_states[token_indices_sorted[i]] += g_out[i]`` for
    ``i`` in ``[shard_output_start, shard_output_end)``. Because each input token
    contributes to exactly ``topk`` experts, this maps cleanly to a
    ``ragged_gather_reduce`` with ``reduce_group_size=topk`` along the inverse
    permutation.

  Args:
    hidden_states_local: 2D ``[num_tokens_local, hidden]`` input tensor.
    topk_indices_local: 2D ``[num_tokens_local, topk]`` tensor of target expert indices.
    num_experts: scalar ``int`` representing the total global number of experts.
    topk: scalar ``int`` representing the routing top-k factor.
    ep_name: ``str`` identifying the expert parallel axis name.
    ep_size: scalar ``int`` representing the expert parallel mesh size.
    buffer_size: optional scalar ``int`` representing the size of the local buffer.
    tag_routing_fn: optional function applied to each integer routing array (the sort permutation, its inverse,
      the sorted token indices and the group sizes).
    index_math: how the routing index maps are computed (``moe_routing_index_math``).
      "sort" (default): the permutation and its inverse by two stable argsorts.
      "count": the permutation by one single-operand packed uint32 sort, the sorted token indices by division,
      and the inverse permutation by counting, on this shard's slots only (-1 on every other slot); the
      gather-reduce of the backward packs its validity sort too. Every reader of the inverse masks it to this
      shard's window, so the forward output and the gradients are the same. The unsort of these rows must then
      get ``topk_argsort_indices`` and ``index_math="count"`` (a partial inverse cannot be argsorted back).

  Returns:
    A tuple containing:
      - Processed activations tensor of shape ``[buffer_size, hidden]``
      containing
        only the tokens destined for local experts, padded with zeros elsewhere.
      - 1D tensor ``group_sizes_local`` tracking expert token counts.
      - 1D tensor ``topk_argsort_revert_indices`` for inverse routing.
      - 1D tensor ``topk_argsort_indices``, the inverse of ``topk_argsort_revert_indices``.
  """
  _check_index_math(index_math)
  # The routing index maps depend only on the (integer) expert ids, so they are computed outside the custom_vjp.
  # Only the gather below is differentiated.
  num_tokens_local = hidden_states_local.shape[0]

  topk_indices_flat = topk_indices_local.flatten()  # num_tokens_local x topk
  if index_math == "count":
    # The stable order by expert id as ONE single-operand sort of (expert << pos_bits | slot): the packed values
    # are distinct, so the unstable sort has one result, the stable argsort's permutation. jnp.argsort sorts the
    # (expert, iota) pair stably, which XLA lowers in the forward pass with a duplicated iota operand.
    _, topk_argsort_indices = packed_stable_sort(topk_indices_flat.astype(jnp.int32), num_experts)
    # Flat slot j belongs to token j // topk, so the sorted token indices need no gather of a repeated iota.
    token_indices_sorted = (topk_argsort_indices // topk).astype(jnp.int32)  # num_tokens_local x topk
  else:
    topk_argsort_indices = jnp.argsort(topk_indices_flat)  # num_tokens_local x topk

    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)  # num_tokens_local x topk
    token_indices_sorted = token_indices[topk_argsort_indices]  # num_tokens_local x topk

  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)  # GLOBAL_NUM_EXPERTS

  if index_math == "count":
    # Every reader of the inverse permutation masks it to [shard_output_start, shard_output_end), so it is
    # computed on this shard's slots only, by counting instead of a second argsort, and is -1 elsewhere.
    shard_idx = jax.lax.axis_index(ep_name)
    local_pos = _local_slot_positions(topk_indices_local, num_experts, shard_idx, ep_size)
    shard_start = jnp.sum(jnp.where(jnp.arange(num_experts) < shard_idx * (num_experts // ep_size), group_sizes_local, 0))
    topk_argsort_revert_indices = jnp.where(local_pos >= 0, shard_start + local_pos, -1).astype(jnp.int32)
  else:
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)  # num_tokens_local x topk
  packed_preprocess = index_math == "count"

  if tag_routing_fn is not None:
    topk_argsort_indices, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices = (
        tag_routing_fn(t)
        for t in (topk_argsort_indices, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices)
    )

  @jax.custom_vjp
  def _ring_ragged_sort(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices):
    """Gather activations to different EP shards."""
    return _ring_ragged_sort_fwd(
        hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices
    )[0]

  @jax.named_scope("ragged-sort-fwd")
  def _ring_ragged_sort_fwd(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices):
    """Gather activations forward pass."""

    num_tokens_local = hidden_states_local.shape[0]
    shard_idx = jax.lax.axis_index(ep_name)

    local_num_experts = num_experts // ep_size
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    if buffer_size is None or buffer_size >= num_tokens_local * topk:
      local_buffer_size = num_tokens_local * topk
      x = ragged_gather(
          hidden_states_local,
          token_indices_sorted,
          shard_output_start[None],
          shard_output_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
    else:
      local_buffer_size = buffer_size
      # We only gather up to the available buffer size or the actual number of
      # tokens destined for this shard's experts, whichever is smaller.
      gather_end = jnp.minimum(shard_output_end - shard_output_start, local_buffer_size)
      # Pad the indices to ensure we can safely slice a block of size `local_buffer_size`
      # starting at `shard_output_start` without going out-of-bounds during compilation.
      padded_token_indices_sorted = jnp.pad(token_indices_sorted, (0, local_buffer_size))
      sliced_indices = jax.lax.dynamic_slice_in_dim(
          padded_token_indices_sorted,
          shard_output_start,
          local_buffer_size,
          axis=0,
      )
      x = ragged_gather(
          hidden_states_local,
          sliced_indices,
          jnp.int32(0)[None],
          gather_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )

    res = (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        local_buffer_size,
        hidden_states_local.shape,
    )

    return x, res

  @jax.named_scope("ragged-sort-bwd")
  def _ring_ragged_sort_bwd(res, g_out):
    """Backward pass for the gather: a Pallas SC ragged gather reduce.
    The forward gathers ``hidden_states_local[token_indices_sorted[i]]`` into
    ``x[i]`` for ``i`` in ``[shard_output_start, shard_output_end)``.  The
    gradient w.r.t. ``hidden_states_local`` is therefore::
        g_hidden_states[token_indices_sorted[i]] += g_x[i]    (i in valid range)
    which is exactly a ragged gather reduce.
    """
    (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        local_buffer_size,
        _,
    ) = res
    g_x = g_out
    # Restrict to the [start, end) source range via a validity bitmask. The
    # ragged kernel packs valid rows to the front of each row-partition and
    # only iterates over the populated prefix, so we hand it the mask directly
    # rather than materializing a (mostly-zero) dense buffer ourselves.
    n = topk_argsort_revert_indices.shape[0]

    if local_buffer_size >= n:
      valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
          topk_argsort_revert_indices < shard_output_end
      )
      # The forward scatter-add over `token_indices_sorted` is equivalent to a
      # gather-reduce: each input token has exactly `topk` contributions located
      # at sorted positions `topk_argsort_revert_indices[t*topk:(t+1)*topk]`.
      # `topk_weights` is set to ones because this op has no per-row weighting.
      grad_hidden_states = ragged_gather_reduce(
          g_x,
          topk_argsort_revert_indices,
          topk_weights=jnp.ones((n,), dtype=jnp.float32),
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
          packed_preprocess=packed_preprocess,
      )
    else:
      # Buffering: g_x has size `local_buffer_size` (packed).
      # The revert indices are global [0, n), but they must map to the local
      # packed g_x buffer.
      shifted_indices = topk_argsort_revert_indices - shard_output_start
      local_num_tokens = shard_output_end - shard_output_start
      # We only reduce gradients from the valid portion of the local buffer.
      limit = jnp.minimum(local_num_tokens, local_buffer_size)
      # Mask out tokens that were not gathered (either because they belong to
      # other shards, or they exceeded the local buffer size).
      valid_rows_mask = (shifted_indices >= 0) & (shifted_indices < limit)
      # Clamp invalid indices to 0 to prevent compile-time/run-time out-of-bounds
      # in JAX. These clamped values will be ignored due to `valid_rows_mask`.
      safe_indices = jnp.where(valid_rows_mask, shifted_indices, 0)

      grad_hidden_states = ragged_gather_reduce(
          g_x,
          safe_indices,
          topk_weights=jnp.ones((n,), dtype=jnp.float32),
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
          packed_preprocess=packed_preprocess,
      )
    return grad_hidden_states, None, None, None

  _ring_ragged_sort.defvjp(_ring_ragged_sort_fwd, _ring_ragged_sort_bwd)

  x = _ring_ragged_sort(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices)
  return x, group_sizes_local, topk_argsort_revert_indices, topk_argsort_indices


def ring_ragged_unsort(
    sorted_tokens_local,
    group_sizes_local,
    topk_argsort_revert_indices,
    topk,
    local_num_experts,
    ep_name,
    topk_weights,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
    use_single_sparsecore=False,
    topk_argsort_indices=None,
    index_math="sort",
):
  """Dual of :func:`ring_ragged_sort`.

  Forward:
    ``out[i] = sum_k( w[i*topk+k] * sorted_tokens_local[topk_argsort_revert_indices[i*topk+k]] )``
    for positions where ``topk_argsort_revert_indices`` is in
    ``[shard_output_start, shard_output_end)``; other rows are zeroed.
    This scatters the processed outputs from the experts hosted on this shard
    back to their flat arrival buffer positions, applying routing weights
    during the reduction.

  Backward:
    ``g_sorted_tokens[j] = w[i] * g_out[i // topk]`` and
    ``g_w[i] = <g_out[i // topk], sorted_tokens_local[j]>`` where
    ``j = topk_argsort_revert_indices[i]`` and ``j`` is in
    ``[shard_output_start, shard_output_end)``.

  Args:
    sorted_tokens_local: 2D ``[buffer_size, hidden]`` output tensor from local experts.
    group_sizes_local: 1D tensor tracking the token loads per expert.
    topk_argsort_revert_indices: 1D permutation restoring flat token positions.
    topk: scalar ``int`` representing the routing top-k factor.
    local_num_experts: scalar ``int`` representing the count of experts hosted on this shard.
    ep_name: ``str`` identifying the expert parallel axis name.
    topk_weights: ``[num_tokens_local * topk]`` tensor of per-slot routing weights.
      Differentiated: its gradient is what trains the router.
    topk_argsort_indices: optional inverse of ``topk_argsort_revert_indices`` (as returned by
      :func:`ring_ragged_sort`). The backward needs this permutation; when it is
      None, the backward re-derives it with an argsort. Both give the same permutation.
      Required with ``index_math="count"``, whose inverse covers only this shard's slots.
    index_math: must match the :func:`ring_ragged_sort` call that produced the routing. With "count" the
      gather-reduce packs its validity sort, and in the backward the routing-weight gradient goes back to slot
      order by a masked gather through the inverse permutation instead of a scatter (each valid slot reads its
      own row; other shards' and dropped slots read 0). Same values either way.

  Returns:
    A 2D ``[num_tokens_local, hidden]`` tensor with expert outputs scattered back
    to their original global sequence locations, with unpopulated positions zeroed out.
  """
  _check_index_math(index_math)
  packed_preprocess = index_math == "count"

  @jax.custom_vjp
  def _ring_ragged_unsort(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      topk_argsort_indices,
  ):
    """Unsort and scatter activations."""
    return _ring_ragged_unsort_fwd(
        sorted_tokens_local,
        group_sizes_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        topk_argsort_indices,
    )[0]

  @jax.named_scope("ragged-unsort-fwd")
  def _ring_ragged_unsort_fwd(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      topk_argsort_indices,
  ):
    """Executes unsorting sending tokens back."""
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)

    shard_idx = jax.lax.axis_index(ep_name)
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts

    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    buffer_size = sorted_tokens_local.shape[0]
    num_tokens = topk_argsort_revert_indices.shape[0]

    # We support two buffering modes:
    # 1. buffer_size >= num_tokens: sorted_tokens_local has size >= num_tokens,
    #    and local tokens are at their global positions [start, end).
    # 2. buffer_size < num_tokens: sorted_tokens_local has size < num_tokens,
    #    and local tokens are packed at [0, local_num_tokens).
    if buffer_size >= num_tokens:
      # Express the scatter as a gather-reduce: each output row pulls
      # from sorted_tokens_local at position `topk_argsort_revert_indices[i]` if
      # that position is within this shard's [start, end) range, else zero.
      # The routing weights are applied per-row before the topk reduction.
      valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
          topk_argsort_revert_indices < shard_output_end
      )
      out = ragged_gather_reduce(
          sorted_tokens_local,
          topk_argsort_revert_indices,
          topk_weights=topk_weights_flat,
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
          packed_preprocess=packed_preprocess,
      )
    else:
      # Shift indices so they map to the packed local buffer [0, local_num_tokens).
      shifted_indices = topk_argsort_revert_indices - shard_output_start
      local_num_tokens = shard_output_end - shard_output_start
      limit = jnp.minimum(local_num_tokens, buffer_size)
      valid_rows_mask = (shifted_indices >= 0) & (shifted_indices < limit)
      safe_indices = jnp.where(valid_rows_mask, shifted_indices, 0)

      out = ragged_gather_reduce(
          sorted_tokens_local,
          safe_indices,
          topk_weights=topk_weights_flat,
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
          packed_preprocess=packed_preprocess,
      )

    res = (
        sorted_tokens_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
        topk_argsort_indices,
    )

    return out, res

  @jax.named_scope("ragged-unsort-bwd")
  def _ring_ragged_unsort_bwd(res, g_out):
    """Backward pass for the scatter with routing weights.

    The forward computes (per output token t, with topk slots k=0..topk-1):
      out[t] = sum_k  w[t*topk+k] * sorted_tokens[revert[t*topk+k]]
    masked by validity.

    Gradient w.r.t. sorted_tokens:
      g_sorted_tokens[j] = w[i] * g_out[i // topk]
    where j = revert[i] and j in [start, end).

    Gradient w.r.t. the routing weights:
      g_w[i] = <g_out[i // topk], sorted_tokens[revert[i]]>
    for valid i, zero otherwise.
    """
    (
        sorted_tokens_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
        topk_argsort_indices,
    ) = res
    g_hidden_states_local = g_out

    n = topk_argsort_revert_indices.shape[0]
    # Build the inverse permutation idx_inv such that idx_inv[j] = i
    # where revert[i] = j. revert is itself argsort(topk_argsort_indices), so its inverse is
    # topk_argsort_indices; reuse it when the caller has it rather than sorting again.
    if topk_argsort_indices is not None:
      idx_inv = topk_argsort_indices
    elif index_math == "count":
      # The "count" inverse holds only this shard's slots (-1 elsewhere), so it cannot be argsorted back.
      raise ValueError("index_math='count' needs topk_argsort_indices (the forward permutation of ring_ragged_sort).")
    else:
      idx_inv = jnp.argsort(topk_argsort_revert_indices)

    # Handle the same two buffering modes for backward pass.
    # ragged_gather does the fan-out, by indexing into the un-expanded
    # g_hidden_states_local via idx_inv // topk. It gathers unweighted so the same rows
    # feed both gradients: the activation one after scaling, the weight one after a dot.
    if buffer_size >= n:
      weight_for_sorted = topk_weights_flat[idx_inv]
      gathered = ragged_gather(
          g_hidden_states_local,
          idx_inv // topk,
          shard_output_start[None],
          shard_output_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
      # Mask out gradients that correspond to elements outside the valid shard
      # output range.
      mask = (jnp.arange(n) >= shard_output_start) & (jnp.arange(n) < shard_output_end)
      gathered = jnp.where(mask[:, None], gathered, 0.0)
      grad_sorted_tokens = (gathered * weight_for_sorted[:, None]).astype(gathered.dtype)
      # Row-wise dot in sorted order, then permuted back to flat slot order.
      dot_sorted = jnp.sum(gathered.astype(jnp.float32) * sorted_tokens_local[:n].astype(jnp.float32), axis=-1)
      if index_math == "count":
        # Other shards' slots hold -1, not a row outside the window, so mask them instead of reading row -1.
        slot_valid = (topk_argsort_revert_indices >= shard_output_start) & (
            topk_argsort_revert_indices < shard_output_end
        )
        grad_topk_weights = jnp.where(slot_valid, dot_sorted[jnp.where(slot_valid, topk_argsort_revert_indices, 0)], 0.0)
      else:
        grad_topk_weights = dot_sorted[topk_argsort_revert_indices]
    else:
      # Slice the inverse permutation to match the packed local buffer.
      padded_idx_inv = jnp.pad(idx_inv, (0, buffer_size))
      sliced_idx_inv = jax.lax.dynamic_slice_in_dim(padded_idx_inv, shard_output_start, buffer_size, axis=0)
      gather_end = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
      # Slice the per-slot routing weights to match the packed local buffer.
      padded_weights = jnp.pad(topk_weights_flat[idx_inv], (0, buffer_size))
      sliced_weights = jax.lax.dynamic_slice_in_dim(padded_weights, shard_output_start, buffer_size, axis=0)
      gathered = ragged_gather(
          g_hidden_states_local,
          sliced_idx_inv // topk,
          jnp.int32(0)[None],
          gather_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
      # Mask out gradients for elements beyond the valid limit of the local buffer.
      limit = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
      mask = jnp.arange(buffer_size) < limit
      gathered = jnp.where(mask[:, None], gathered, 0.0)
      grad_sorted_tokens = (gathered * sliced_weights[:, None]).astype(gathered.dtype)
      dot_local = jnp.sum(gathered.astype(jnp.float32) * sorted_tokens_local.astype(jnp.float32), axis=-1)
      if index_math == "count":
        # Route the per-row dot back to flat slot order with a gather through the inverse permutation instead
        # of a TensorCore scatter: each valid slot reads its own buffer row; other shards' slots (-1) and
        # dropped slots read 0, as the scatter leaves them. The table is padded to n entries because XLA only
        # offloads a 1-D gather to the SparseCore for a large table (v7x AOT: 225,280 entries -> TensorCore
        # gather at ~7 ns/element; 655,360 -> SparseCore).
        shifted = topk_argsort_revert_indices - shard_output_start
        slot_valid = (shifted >= 0) & (shifted < limit)
        dot_table = jnp.pad(dot_local, (0, max(0, n - buffer_size)))
        grad_topk_weights = jnp.where(slot_valid, dot_table[jnp.where(slot_valid, shifted, 0)], 0.0)
      else:
        # Scatter the per-slot dot back to flat slot order; dropped slots stay zero.
        slots = jnp.where(mask, sliced_idx_inv, n)
        grad_topk_weights = jnp.zeros((n,), jnp.float32).at[slots].set(dot_local, mode="drop")
    return grad_sorted_tokens, None, None, grad_topk_weights, None

  _ring_ragged_unsort.defvjp(_ring_ragged_unsort_fwd, _ring_ragged_unsort_bwd)

  # Build the flat weights array from the routing weights.
  topk_weights_flat = topk_weights.astype(jnp.float32)

  return _ring_ragged_unsort(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      topk_argsort_indices,
  )


def a2a_ragged_sort(
    inputs,
    sort_indices,
    valid_end,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    use_single_sparsecore=False,
):
  """Ragged-gather variant for ``local_permute``.

  Unlike :func:`ring_ragged_sort`, the rows valid for this shard live in
  the prefix ``[0, valid_end)`` of ``inputs`` (the rest of the buffer is
  padding). This helper sorts ``inputs`` by ``sort_indices`` but only touches
  the valid prefix, making the cost proportional to the *actual* token count
  rather than the padded buffer length.

  Forward:
    ``out[i] = inputs[sort_indices[i]]`` for ``i in [0, valid_end)``;
    other rows are zero.

  Backward (gather-reduce):
    ``g_inputs[sort_indices[i]] += g_out[i]`` for ``i in [0, valid_end)``.
    Because ``sort_indices`` is a permutation, each input row receives exactly
    one contribution; we model this as a ``ragged_gather_reduce`` with
    ``reduce_group_size=1`` along the inverse permutation.

  Args:
    inputs: 2D ``[num_tokens, hidden]`` tensor whose valid rows live in the
      prefix ``[0, valid_end)``.
    sort_indices: 1D permutation of ``[0, num_tokens)`` describing the desired
      ordering. Values at positions ``>= valid_end`` are ignored.
    valid_end: scalar ``int32`` indicating the exclusive end of the valid
      prefix.

  Returns:
    A 2D ``[num_tokens, hidden]`` tensor sorted by ``sort_indices`` over the
    valid prefix, with padded rows zeroed.
  """

  @jax.custom_vjp
  def _a2a_ragged_sort(inputs, sort_indices, valid_end):
    return _a2a_ragged_sort_fwd(inputs, sort_indices, valid_end)[0]

  @jax.named_scope("local-ragged-sort-fwd")
  def _a2a_ragged_sort_fwd(inputs, sort_indices, valid_end):
    start = jnp.int32(0)
    end = valid_end.astype(jnp.int32) if hasattr(valid_end, "astype") else jnp.int32(valid_end)
    out = ragged_gather(
        inputs,
        sort_indices,
        start[None],
        end[None],
        use_single_sparsecore=use_single_sparsecore,
    )
    n = sort_indices.shape[0]
    valid_mask = jnp.arange(n) < end
    out = jnp.where(valid_mask[:, None], out, 0.0)
    res = (sort_indices, end, inputs.shape)
    return out, res

  @jax.named_scope("local-ragged-sort-bwd")
  def _a2a_ragged_sort_bwd(res, g_out):
    sort_indices, end, _ = res
    n = sort_indices.shape[0]
    valid_rows_mask = jnp.arange(n) < end
    # g_inputs[sort_indices[i]] += g_out[i], for i in [0, end). This is a
    # ragged scatter-add, which we express as a gather-reduce along the inverse
    # permutation: each input row j receives exactly one contribution from
    # output row i where sort_indices[i] == j.
    idx_inv = jnp.argsort(sort_indices)
    grad_inputs = ragged_gather_reduce(
        g_out,
        idx_inv,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask[idx_inv],
        reduce_group_size=1,
        enforce_fallback=enforce_gather_reduce_fallback,
        use_single_sparsecore=use_single_sparsecore,
    )
    # custom_vjp must return one gradient per primal arg; valid_end is integer
    # and non-differentiable, so we return None for it.
    return grad_inputs, None, None

  _a2a_ragged_sort.defvjp(_a2a_ragged_sort_fwd, _a2a_ragged_sort_bwd)
  return _a2a_ragged_sort(inputs, sort_indices, valid_end)


def a2a_ragged_unsort(
    sorted_tokens,
    revert_indices,
    valid_end,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    use_single_sparsecore=False,
):
  """Dual of :func:`a2a_ragged_sort`.

  Forward:
    ``out[i] = sorted_tokens[revert_indices[i]]`` for ``i in [0, valid_end)``;
    other rows are zero. This is the unsort step in the local-permute path:
    given a buffer ordered by local expert IDs, restore the original arrival
    order.

  Backward:
    ``g_sorted_tokens[j] = g_out[i]`` where ``j = revert_indices[i]``, for
    ``i in [0, valid_end)``. Since ``revert_indices`` is a permutation, this
    is a simple ragged gather over the inverse permutation.

  Args:
    sorted_tokens: 2D ``[num_tokens, hidden]`` tensor.
    revert_indices: 1D permutation of ``[0, num_tokens)``.
    valid_end: scalar ``int32`` indicating the exclusive end of the valid
      prefix.

  Returns:
    A 2D ``[num_tokens, hidden]`` tensor with rows reordered by
    ``revert_indices`` over the valid prefix and zero elsewhere.
  """

  @jax.custom_vjp
  def _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end):
    return _a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end)[0]

  @jax.named_scope("local-ragged-unsort-fwd")
  def _a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end):
    start = jnp.int32(0)
    end = valid_end.astype(jnp.int32) if hasattr(valid_end, "astype") else jnp.int32(valid_end)
    n = revert_indices.shape[0]
    valid_rows_mask = jnp.arange(n) < end
    out = ragged_gather_reduce(
        sorted_tokens,
        revert_indices,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask,
        reduce_group_size=1,
        enforce_fallback=enforce_gather_reduce_fallback,
        use_single_sparsecore=use_single_sparsecore,
    )
    res = (revert_indices, end, sorted_tokens.shape, start)
    return out, res

  @jax.named_scope("local-ragged-unsort-bwd")
  def _a2a_ragged_unsort_bwd(res, g_out):
    revert_indices, end, sorted_tokens_shape, start = res
    # g_sorted_tokens[revert_indices[i]] = g_out[i] for i in [0, end).
    # Because revert_indices is a permutation, build the inverse and use
    # ragged_gather to pull the per-row gradients to the right positions.
    idx_inv = jnp.argsort(revert_indices)
    grad_sorted = ragged_gather(
        g_out,
        idx_inv,
        start[None],
        end[None],
        use_single_sparsecore=use_single_sparsecore,
    )
    num_rows = sorted_tokens_shape[0]
    pos = jnp.arange(num_rows)
    valid = pos < end
    grad_sorted = jnp.where(valid[:, None], grad_sorted, jnp.zeros_like(grad_sorted))

    return grad_sorted, None, None

  _a2a_ragged_unsort.defvjp(_a2a_ragged_unsort_fwd, _a2a_ragged_unsort_bwd)
  return _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end)
