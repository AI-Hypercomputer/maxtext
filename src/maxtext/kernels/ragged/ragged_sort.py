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
from maxtext.kernels.ragged.ragged_gather_reduce import ragged_gather_reduce
from maxtext.kernels.ragged.ragged_scatter import ragged_scatter


def gather_tokens_locally(hidden_states_local, topk_indices_local, num_experts, topk, ep_name, ep_size):
  """JAX closure of gather tokens implementation."""

  @jax.custom_vjp
  def _gather_tokens_locally(hidden_states_local, topk_indices_local):
    """Sort and gather activations to different EP shards."""
    return _gather_tokens_locally_fwd(hidden_states_local, topk_indices_local)[0]

  @jax.named_scope("ragged-gather-fwd")
  def _gather_tokens_locally_fwd(hidden_states_local, topk_indices_local):
    """Sort and gather activations forward pass."""

    num_tokens_local = hidden_states_local.shape[0]

    topk_indices_flat = topk_indices_local.flatten()  # num_tokens_local x topk
    topk_argsort_indices = jnp.argsort(topk_indices_flat)  # num_tokens_local x topk

    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)  # num_tokens_local x topk
    token_indices_sorted = token_indices[topk_argsort_indices]  # num_tokens_local x topk

    group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)  # GLOBAL_NUM_EXPERTS

    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)  # num_tokens_local x topk
    shard_idx = jax.lax.axis_index(ep_name)

    local_num_experts = num_experts // ep_size
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    x = ragged_gather(
        hidden_states_local,
        token_indices_sorted,
        shard_output_start,
        shard_output_end,
    )

    valid_mask = (jnp.arange(x.shape[0]) >= shard_output_start) & (jnp.arange(x.shape[0]) < shard_output_end)
    x = jnp.where(valid_mask[:, None], x, 0.0)

    out = (x, group_sizes_local, topk_argsort_revert_indices)

    res = (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        hidden_states_local.shape,
    )

    return out, res

  @jax.named_scope("ragged-gather-bwd")
  def _gather_tokens_locally_bwd(res, g_out):
    """Backward pass for the gather: a Pallas SC ragged scatter-add.

    The forward gathers ``hidden_states_local[token_indices_sorted[i]]`` into
    ``x[i]`` for ``i`` in ``[shard_output_start, shard_output_end)``.  The
    gradient w.r.t. ``hidden_states_local`` is therefore::

        g_hidden_states[token_indices_sorted[i]] += g_x[i]    (i in valid range)

    which is exactly a ragged scatter-add.
    """
    topk_argsort_revert_indices, shard_output_start, shard_output_end, _ = res
    g_x, _, _ = g_out
    # Restrict to the [start, end) source range via a validity bitmask. The
    # ragged kernel packs valid rows to the front of each row-partition and
    # only iterates over the populated prefix, so we hand it the mask directly
    # rather than materializing a (mostly-zero) dense buffer ourselves.
    n = topk_argsort_revert_indices.shape[0]
    pos = jnp.arange(n)
    valid_rows_mask = (pos >= shard_output_start) & (pos < shard_output_end)
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
    )
    return grad_hidden_states, None

  _gather_tokens_locally.defvjp(_gather_tokens_locally_fwd, _gather_tokens_locally_bwd)

  return _gather_tokens_locally(hidden_states_local, topk_indices_local)


def scatter_tokens_locally(
    sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, local_num_experts, ep_name
):
  """JAX closure of scatter tokens function."""

  @jax.custom_vjp
  def _scatter_tokens_locally(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices):
    """Unsort and scatter activations."""
    return _scatter_tokens_locally_fwd(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices)[0]

  @jax.named_scope("ragged-scatter-fwd")
  def _scatter_tokens_locally_fwd(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices):
    """Executes unsorting sending tokens back."""
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)

    shard_idx = jax.lax.axis_index(ep_name)
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts

    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    out = ragged_scatter(sorted_tokens_local, topk_argsort_revert_indices, shard_output_start, shard_output_end)

    valid_mask = (topk_argsort_revert_indices >= shard_output_start) & (topk_argsort_revert_indices < shard_output_end)
    out = jnp.where(valid_mask[:, None], out, 0.0)

    res = (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        sorted_tokens_local.shape,
    )

    return out, res

  @jax.named_scope("ragged-scatter-bwd")
  def _scatter_tokens_locally_bwd(res, g_out):
    """Backward pass for the scatter: a Pallas SC ragged gather.

    The forward scatter computes
    ``out[i] = sorted_tokens[topk_argsort_revert_indices[i]]`` masked by
    ``valid_mask[i] = (topk_argsort_revert_indices[i] >= start) and (< end)``.

    Equivalently, defining ``j = topk_argsort_revert_indices[i]``, the forward is
    ``out[i] = sorted_tokens[j]`` whenever ``j in [start, end)``.

    ``topk_argsort_revert_indices`` is a permutation of
    ``[0, sorted_tokens_local_shape[0])``, so the gradient pulls back per-row::

        g_sorted_tokens[j] = g_hidden_states_local[i]   where j = revert[i]
                                                        and j in [start, end).

    This is exactly a ragged gather of ``g_hidden_states_local`` along the
    inverse permutation ``argsort(revert)``, but restricted to the [start,end)
    range of ``j``.  The simpler equivalent: gather of g_hidden_states_local
    using the inverse permutation, masked.
    """
    topk_argsort_revert_indices, shard_output_start, shard_output_end, sorted_tokens_local_shape = res
    g_hidden_states_local = g_out
    num_rows = sorted_tokens_local_shape[0]

    # We want: g_sorted_tokens[j] = g_hidden_states_local[i] where revert[i]=j.
    # Build the inverse permutation idx_inv such that idx_inv[j] = i.
    idx_inv = jnp.argsort(topk_argsort_revert_indices)
    # Because revert is a permutation, gathering with idx_inv reorders correctly.
    grad_sorted_tokens = ragged_gather(
        g_hidden_states_local,
        idx_inv,
        shard_output_start,
        shard_output_end,
    )
    # Outside [start, end), positions must be zero — which the ragged_gather
    # already guarantees because untouched output rows are uninitialized; we
    # explicitly zero them.
    pos = jnp.arange(num_rows)
    valid = (pos >= shard_output_start) & (pos < shard_output_end)
    grad_sorted_tokens = jnp.where(valid[:, None], grad_sorted_tokens, jnp.zeros_like(grad_sorted_tokens))
    return grad_sorted_tokens, None, None

  _scatter_tokens_locally.defvjp(_scatter_tokens_locally_fwd, _scatter_tokens_locally_bwd)

  return _scatter_tokens_locally(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices)
