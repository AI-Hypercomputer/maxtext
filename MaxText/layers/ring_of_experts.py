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

import functools
from typing import Callable

import jax
import jax.numpy as jnp


# Alias for the three weights used in the MLP layer.
MlpWeights = tuple[
    jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike
]


@jax.named_scope("process_tokens")
def process(
    inputs: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    expert_axis: str,
    num_expert_shards: int,
    num_total_experts: int,
    mlp_weights: MlpWeights,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
) -> jax.Array:
  """Run ring-of-experts.

  Runs `inputs` through a moe layer with a ring-of-experts routing scheme. Each
  token in `inputs` has the experts to which it should be matched with stored in
  `expert_indices`, and corresponding scaling factor in `expert_weights`.

  The general strategy for processing tokens in the ring-of-experts style is to
  have an outer and an inner loop where
  - the outer loop iterates over the experts, and
  - the inner loop iterates over the tokens.

  In this context, `process()` defines the outer loop.

  Args:
    inputs: `(..., model_dim)` array of tokens.
    expert_indices: `(..., num_experts_per_tok)` array of integers corresponding
      to the indices of the experts selected for each token.
    expert_weights: `(..., num_experts_per_tok)` array of weights corresponding
      to the linear scaling factor to apply to each selected expert's output.
    expert_axis: Name of the expert axis.
    num_expert_shards: The number of shards along the expert axis.
    num_total_experts: The total number of experts in the MoE layer.
    mlp_weights: `(in_0, in_1, out)` mlp weights for the experts on a single
      expert shard, where `in_0` and `in_1` are of shape `(experts_on_shard,
      model_dim, hidden_dim)` and `out` is of shape `(experts_on_shard,
      hidden_dim, model_dim)`, where `experts_on_shard` is the number of experts
      on the current shard (`num_total_experts // num_expert_shards`).
    activation_fn: Non-linear activation function to use within the mlp layer.
    num_tokens_per_chunk: For performance purposes, at most
      `num_tokens_per_chunk` tokens are processed by a single expert at a time.
      Adjusting this parameter should impact performance but result in only
      minimal numerical differences.

  Returns:
    Output MoE activations of the same shape as `inputs`.
  """
  # All-gather inputs across the expert axis..
  with jax.named_scope("broadcast_tokens"):
    inputs, expert_indices, expert_weights = jax.lax.all_gather(
        (inputs, expert_indices, expert_weights),
        axis_name=expert_axis,
        axis=0,
        tiled=True,
    )

  # Save original shape for reshaping outputs.
  orig_shape = inputs.shape

  # Flatten batch dimension(s) into the sequence/token dimension.
  inputs = jnp.reshape(inputs, (-1, inputs.shape[-1]))
  expert_indices = jnp.reshape(expert_indices, (-1, expert_indices.shape[-1]))
  expert_weights = jnp.reshape(expert_weights, (-1, expert_weights.shape[-1]))

  # Process tokens.
  outputs = outer_experts_loop(
      inputs=inputs,
      expert_indices=expert_indices,
      expert_weights=expert_weights,
      expert_axis=expert_axis,
      mlp_weights=mlp_weights,
      num_expert_shards=num_expert_shards,
      num_total_experts=num_total_experts,
      activation_fn=activation_fn,
      num_tokens_per_chunk=num_tokens_per_chunk,
  )

  # Return outputs to their original shard.
  with jax.named_scope("return_outputs"):
    return jax.lax.psum_scatter(
        jnp.reshape(outputs, orig_shape),
        axis_name=expert_axis,
        scatter_dimension=0,
        tiled=True,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 7, 8))
def outer_experts_loop(
    inputs: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    expert_axis: str,
    num_expert_shards: int,
    num_total_experts: int,
    mlp_weights: MlpWeights,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
) -> jax.Array:
  """Outer loop over experts with custom backprop."""
  return _outer_experts_loop_fwd(
      inputs=inputs,
      expert_indices=expert_indices,
      expert_weights=expert_weights,
      expert_axis=expert_axis,
      num_expert_shards=num_expert_shards,
      num_total_experts=num_total_experts,
      mlp_weights=mlp_weights,
      activation_fn=activation_fn,
      num_tokens_per_chunk=num_tokens_per_chunk,
  )[0]


def _outer_experts_loop_fwd(
    inputs: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    expert_axis: str,
    num_expert_shards: int,
    num_total_experts: int,
    mlp_weights: MlpWeights,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, MlpWeights]]:
  """Forward pass for `outer_experts_loop()`."""

  def scan_fn(
      outputs: jax.Array, expert_id_and_weights: tuple[jax.Array, MlpWeights]
  ) -> tuple[jax.Array, None]:
    """Scan function over experts."""
    return (
        process_expert(
            outputs=outputs,
            expert_id_and_weights=expert_id_and_weights,
            inputs=inputs,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            activation_fn=activation_fn,
            num_tokens_per_chunk=num_tokens_per_chunk,
        ),
        None,
    )

  # Scan over the expert dimension, accumulating outputs.
  experts_per_shard = num_total_experts // num_expert_shards
  expert_ids = experts_per_shard * jax.lax.axis_index(expert_axis) + jnp.arange(
      experts_per_shard
  )
  outputs, _ = jax.lax.scan(
      scan_fn,
      init=jnp.zeros_like(inputs),
      xs=(expert_ids, mlp_weights),
      length=experts_per_shard,
  )

  return outputs, (inputs, expert_indices, expert_weights, mlp_weights)


def _outer_experts_loop_bwd(
    expert_axis: str,
    num_expert_shards: int,
    num_total_experts: int,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
    residuals: tuple[jax.Array, jax.Array, jax.Array, MlpWeights],
    grads: jax.Array,
) -> tuple[jax.Array, None, jax.Array, MlpWeights]:
  """Backward pass for outer_experts_loop."""
  inputs, expert_indices, expert_weights, mlp_weights = residuals

  def scan_fn(
      carry: tuple[jax.Array, jax.Array],
      expert_id_and_weights: tuple[jax.Array, MlpWeights],
  ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, MlpWeights]]:
    """Backward-pass scan function over experts."""
    # Gradients that will be accumulated.
    inputs_grad, expert_weights_grad = carry

    # Computes the vjp "dual" function to the forward pass.
    #
    # This "duality" refers to
    # 1) the linear combination of per-expert outputs in the forward pass,
    #    implemented as the loop-accumulation of the outputs, and
    # 2) the corresponding linear combination of inputs and expert weight
    #    gradients in the backward pass, in an analogous loop-accumulation way.
    #
    _, vjp_fn = jax.vjp(
        functools.partial(
            process_expert,
            outputs=jnp.zeros_like(inputs),
            expert_indices=expert_indices,
            activation_fn=activation_fn,
            num_tokens_per_chunk=num_tokens_per_chunk,
        ),
        inputs,
        expert_weights,
        expert_id_and_weights,
    )
    inputs_grad_new, expert_weights_grad_new, (_, mlp_weights_grad) = vjp_fn(
        grads
    )
    return (
        inputs_grad + inputs_grad_new,
        expert_weights_grad + expert_weights_grad_new,
    ), mlp_weights_grad

  # Scan over the expert dimension, computing weight gradients and accumulating
  # input gradients.
  experts_per_shard = num_total_experts // num_expert_shards
  expert_ids = experts_per_shard * jax.lax.axis_index(expert_axis) + jnp.arange(
      experts_per_shard
  )
  (inputs_grad, expert_weights_grad), mlp_weights_grad = jax.lax.scan(
      scan_fn,
      init=(jnp.zeros_like(inputs), jnp.zeros_like(expert_weights)),
      xs=(expert_ids, mlp_weights),
      length=experts_per_shard,
  )
  return inputs_grad, None, expert_weights_grad, mlp_weights_grad


outer_experts_loop.defvjp(_outer_experts_loop_fwd, _outer_experts_loop_bwd)


@jax.named_scope("process_expert")
def process_expert(
    inputs: jax.Array,
    expert_weights: jax.Array,
    expert_id_and_weights: tuple[jax.Array, MlpWeights],
    outputs: jax.Array,
    expert_indices: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
) -> jax.Array:
  """Inner loop over token chunks for a single expert."""
  expert_id, mlp_weights = expert_id_and_weights

  # Find the tokens and corresponding weights to process for `expert_id`.
  selection_mask = expert_indices == expert_id
  selection_weights = jnp.sum(
      expert_weights * selection_mask, axis=-1, keepdims=True
  )

  # Run the (unscaled) MLP layer for a single expert.
  x = inner_tokens_loop(
      inputs=inputs,
      mask=jnp.any(selection_mask, axis=-1),
      weights=mlp_weights,
      activation_fn=activation_fn,
      num_tokens_per_chunk=num_tokens_per_chunk,
  )

  # Apply scaling factor and accumulate on outputs.
  return outputs + selection_weights * x


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def inner_tokens_loop(
    inputs: jax.Array,
    mask: jax.Array,
    weights: MlpWeights,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
) -> jax.Array:
  """Dynamic loop over token chunks for a single expert."""
  return _inner_tokens_loop_fwd(
      inputs=inputs,
      mask=mask,
      weights=weights,
      activation_fn=activation_fn,
      num_tokens_per_chunk=num_tokens_per_chunk,
  )[0]


def _inner_tokens_loop_fwd(
    inputs: jax.Array,
    mask: jax.Array,
    weights: MlpWeights,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_tokens_per_chunk: int,
) -> tuple[jax.Array, tuple[jax.Array, MlpWeights, jax.Array, jax.Array]]:
  """Forward pass for inner_tokens_loop."""
  # Creates a vector which first contains the indices of the tokens that have
  # `mask=True`, followed by all other indices. Reshape it to be in chunks of
  # `num_tokens_per_chunk`.
  sort_inds = jnp.reshape(
      jnp.argsort(mask, descending=True), (-1, num_tokens_per_chunk)
  )

  # The number of chunks/iterations needed to process all tokens.
  num_iters = jnp.astype(
      jnp.ceil(jnp.sum(mask) / num_tokens_per_chunk), jnp.int32
  )

  def body_fun(i: jax.Array, outputs: jax.Array) -> jax.Array:
    """Process chunk `i`."""
    return process_chunk(
        indices=sort_inds[i],
        inputs=inputs,
        weights=weights,
        outputs=outputs,
        activation_fn=activation_fn,
    )

  # Dynamically process all chunks which have `mask=True` tokens.
  outputs = jax.lax.fori_loop(
      lower=0,
      upper=num_iters,
      body_fun=body_fun,
      init_val=jnp.zeros_like(inputs),
  )

  return outputs, (inputs, weights, sort_inds, num_iters)


def _inner_tokens_loop_bwd(
    activation_fn: Callable[[jax.Array], jax.Array],
    _,
    residuals: tuple[jax.Array, ...],
    grads: jax.Array,
) -> tuple[jax.Array, None, jax.Array]:
  """Backward pass for inner_tokens_loop."""
  inputs, weights, sort_inds, num_iters = residuals

  def body_fun(
      i: jax.Array, val: tuple[jax.Array, MlpWeights]
  ) -> tuple[jax.Array, MlpWeights]:
    """Backward pass for chunk `i`, with gradient accumulation."""
    # Similar to the outer loop, the duality here is that
    # 1) we accumulate outputs in the forward pass, and
    # 2) we accumulate input and weight gradients in the backward pass.
    _, vjp_fn = jax.vjp(
        functools.partial(
            process_chunk,
            indices=sort_inds[i],
            outputs=jnp.zeros_like(inputs),
            activation_fn=activation_fn,
        ),
        inputs,
        weights,
    )
    input_grads_new, weight_grads_new = vjp_fn(grads)
    input_grads, weight_grads = val
    return (
        input_grads + input_grads_new,
        tuple(a + b for a, b in zip(weight_grads, weight_grads_new)),
    )

  # A custom backward pass is needed to do the (dynamic w.r.t. `num_iters`)
  # gradient accumulation.
  input_grads, weight_grads = jax.lax.fori_loop(
      lower=0,
      upper=num_iters,
      body_fun=body_fun,
      init_val=(
          jnp.zeros_like(grads),
          tuple(jnp.zeros_like(w) for w in weights),
      ),
  )

  return input_grads, None, weight_grads


inner_tokens_loop.defvjp(_inner_tokens_loop_fwd, _inner_tokens_loop_bwd)


def process_chunk(
    inputs: jax.Array,
    weights: MlpWeights,
    outputs: jax.Array,
    indices: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
  """Returns unscaled MLP outputs for chunk of tokens."""
  hidden = activation_fn(
      matmul_gather(inputs, weights[0], indices)
  ) * matmul_gather(inputs, weights[1], indices)
  return matmul_scatter(hidden, weights[2], outputs, indices)


@jax.named_scope("matmul_gather")
def matmul_gather(
    lhs: jax.Array,
    rhs: jax.typing.ArrayLike,
    inds: jax.Array,
) -> jax.Array:
  """Gathers `lhs` and performs a matmul with `rhs`."""
  return jnp.matmul(lhs[inds, ...], rhs)


@jax.named_scope("matmul_scatter")
def matmul_scatter(
    lhs: jax.Array,
    rhs: jax.typing.ArrayLike,
    out: jax.Array,
    inds: jax.Array,
) -> jax.Array:
  """Performs `lhs @ rhs` and scatters the result into `out`."""
  return out.at[inds, ...].set(jnp.matmul(lhs, rhs))
