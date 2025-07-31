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

"""Ring-of-experts is mixture-of-experts for toroidal networks."""

from typing import Any, Callable

import jax
import jax.numpy as jnp


def process_tokens(
    inputs: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    expert_axis: str,
    num_expert_shards: int,
    num_total_experts: int,
    mlp_weights: tuple[Any, Any, Any],
    gmm_fn: Callable[[jax.Array, Any, jax.Array], jax.Array],
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
  """Run ring-of-experts.

  Args:
    inputs: `(..., model_dim)` array of input activations.
    expert_indices: `(..., num_experts_per_tok)` array of integers corresponding
      to the indices of the experts selected for each token.
    expert_weights: `(..., num_experts_per_tok)` array of weights corresponding
      to the linear scaling factor to apply to each selected expert's output.
    expert_axis: Name of the expert axis.
    num_expert_shards: The number of shards along the expert axis.
    num_total_experts: The total number of experts in the MoE layer.
    mlp_weights: `(wi_0, wi_1, wo)` weights for the experts on the this shard.
    gmm_fn: `(inputs, weights, group_sizes) -> outputs` function for grouped
      matrix multiplication.
    activation_fn: `(inputs) -> outputs` activation function to use within the
      mlp layer.

  Returns:
    Output MoE activations of the same shape as `inputs`.
  """
  orig_shape = inputs.shape
  inputs = _flatten_except_last(inputs)
  outputs = jnp.zeros_like(inputs)
  expert_indices = _flatten_except_last(expert_indices)
  expert_weights = _flatten_except_last(expert_weights)

  experts_per_shard = num_total_experts // num_expert_shards

  def inner_loop_body(i, val):
    inputs, outputs, expert_indices, expert_weights = val
    expert_id = experts_per_shard * jax.lax.axis_index("expert") + i
    selection_mask = expert_indices == expert_id
    selection_weights = jnp.sum(expert_weights * selection_mask, axis=-1)
    outputs += _process_tokens_for_one_expert(
        x=inputs,
        selection_weights=selection_weights,
        mlp_weights=tuple(w[i][None, ...] for w in mlp_weights),
        gmm_fn=gmm_fn,
        activation_fn=activation_fn,
    )
    return (inputs, outputs, expert_indices, expert_weights)

  def outer_loop_body(i, val):
    val = jax.lax.fori_loop(
        lower=0,
        upper=experts_per_shard,
        body_fun=inner_loop_body,
        init_val=val,
    )
    return tuple(
        _pass_array_to_neighbor(v, expert_axis, num_expert_shards) for v in val
    )

  _, outputs, _, _ = jax.lax.fori_loop(
      lower=0,
      upper=num_expert_shards,
      body_fun=outer_loop_body,
      init_val=(inputs, outputs, expert_indices, expert_weights),
  )

  return jnp.reshape(outputs, orig_shape)


def _process_tokens_for_one_expert(
    x: jax.Array,
    selection_weights: jax.Array,
    mlp_weights: tuple[Any, Any, Any],
    gmm_fn: Callable[[jax.Array, Any, jax.Array], jax.Array],
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
  """Selectively process tokens assigned to a single expert.

  Args:
    x: `(num_tokens, model_dim)` array of activations.
    selection_weights: `(num_tokens)` array of weights corresponding to the
      linear scaling factor to apply to each selected expert's output. Weights
      of value `0` indicate that the token should not be processed.
    mlp_weights: Weights for a single expert.
    gmm_fn: Function to perform matrix multiplication.
    activation_fn: Activation function to apply to the output of the MLP.

  Returns:
    The MoE activations for the selected tokens.
  """
  # Sort order that brings all tokens assigned to this expert to the front
  # of the array.
  sort_inds = jnp.argsort(selection_weights, descending=True)
  x = jnp.take(x, indices=sort_inds, axis=0)

  # Process tokens.
  num_selections = jnp.sum(selection_weights > 0)[None]
  wi_0, wi_1, wo = mlp_weights
  w0 = activation_fn(gmm_fn(x, wi_0, num_selections))
  w1 = gmm_fn(x, wi_1, num_selections)
  x = jnp.multiply(w0, w1)
  x = gmm_fn(x, wo, num_selections)

  # Unsort.
  x = jnp.take(x, indices=jnp.argsort(sort_inds), axis=0)

  # Scale by selection weights.
  return selection_weights[:, None] * x


def _pass_array_to_neighbor(
    array: jax.Array,
    expert_axis: str,
    num_expert_shards: int,
) -> jax.Array:
  """Pass an array to the next expert shard with wrap-around."""
  return jax.lax.ppermute(
      array,
      axis_name=expert_axis,
      perm=[(i, (i + 1) % num_expert_shards) for i in range(num_expert_shards)],
  )


def _flatten_except_last(x: jax.Array) -> jax.Array:
  """Flattens all dimensions except the last one."""
  return jnp.reshape(x, (-1, x.shape[-1]))
