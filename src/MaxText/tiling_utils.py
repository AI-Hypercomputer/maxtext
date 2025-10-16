# Copyright 2025 Google LLC
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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Tiling-related utils functions"""


from flax import linen as nn

import jax
import jax.numpy as jnp
from MaxText import max_utils
from MaxText import maxtext_utils


# Vocab Tiling Helper Functions
def compute_loss_nnx(intermediate_outputs, logits, data, config, model, params, is_train):
  """Computes cross-entropy loss for NNX models.

  Args:
    intermediate_outputs: A dictionary of intermediate model outputs.
    logits: The final model logits.
    data: A dictionary containing the input data, including 'targets' and 'targets_segmentation'.
    config: The model and training configuration.
    model: The NNX model instance.
    params: The model parameters.
    is_train: A boolean indicating if the model is in training mode.

  Returns:
    The total cross-entropy loss.
  """
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  return total_loss


def compute_loss_linen(intermediate_outputs, logits, data, config, model, params, is_train):
  """Computes cross-entropy loss for Linen models, with optional vocab tiling.

  If vocab tiling is enabled (config.num_vocab_tiling > 1), it uses a memory-efficient
  tiled approach. Otherwise, it computes the standard cross-entropy loss.

  Args:
    intermediate_outputs: A dictionary of intermediate model outputs.
    logits: The final model logits.
    data: A dictionary containing the input data, including 'targets' and 'targets_segmentation'.
    config: The model and training configuration.
    model: The Linen model instance.
    params: The model parameters.
    is_train: A boolean indicating if the model is in training mode.

  Returns:
    The total cross-entropy loss.
  """
  if config.num_vocab_tiling > 1:
    hidden_state_key = ("intermediates", "decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
    total_loss = get_vocab_tiling_loss_linen(hidden_states, data, config, model, params, is_train)
  else:
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)
    xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
    # Mask out paddings at the end of each example.
    xent = xent * (data["targets_segmentation"] != 0)
    total_loss = jnp.sum(xent)
  return total_loss


def get_vocab_tiling_loss_linen(
    hidden_states,
    data,
    config,
    model,
    params,
    is_train,
):
  """Calculates cross-entropy loss using vocab tiling for Linen models.

  This function implements a memory-efficient approach for calculating loss when the
  vocabulary is too large to fit in memory. It works by breaking the computation
  into chunks (tiles) and processing them sequentially using `jax.lax.scan`.
  A custom VJP rule is defined to handle the backward pass efficiently.

  Args:
    hidden_states: The final hidden states from the decoder.
    data: A dictionary containing the input data, including 'targets' and 'targets_segmentation'.
    config: The model and training configuration.
    model: The Linen model instance.
    params: The model parameters.
    is_train: A boolean indicating if the model is in training mode.
  Returns:
    The total cross-entropy loss computed via vocab tiling.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  deterministic = not config.enable_dropout if is_train else True

  param_spec = nn.get_partition_spec(params)
  hidden_spec = jax.sharding.NamedSharding(
      model.mesh,
      nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_embed")),
  )
  label_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_length_no_exp"))
  )
  reshaped_hidden_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"))
  )
  reshaped_data_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("num_tile", "activation_embed_and_logits_batch_sequence"))
  )
  chunked_hidden_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch_sequence", "activation_embed"))
  )
  chunked_data_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch_sequence",))
  )
  chunked_logits_spec = jax.sharding.NamedSharding(
      model.mesh, nn.logical_to_mesh_axes(("activation_embed_and_logits_batch_sequence", "activation_vocab"))
  )

  hidden_states = jax.lax.with_sharding_constraint(hidden_states, hidden_spec)
  labels = jax.lax.with_sharding_constraint(labels, label_spec)
  segmentation = jax.lax.with_sharding_constraint(segmentation, label_spec)
  # TODO (chengnuojin) all gather only embedding table instead of all params after NNX module is enabled
  gathered_params = maxtext_utils.all_gather_over_fsdp(params, param_spec, model.mesh, config.logical_axis_rules)

  # Customized forward and backward maps for the embedding tiling
  @jax.custom_vjp
  def chunked_cross_entropy_loss(gathered_params, hidden_states, labels, segmentation):
    """
    Calculates the total cross-entropy loss using vocab tiling.
    """
    total_loss, _ = _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation)
    return total_loss

  def _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

    reshaped_hidden_states = hidden_states.reshape((config.num_vocab_tiling, vocab_tile_size, emb_dim))
    reshaped_hidden_states = jax.lax.with_sharding_constraint(reshaped_hidden_states, reshaped_hidden_spec)
    reshaped_labels = labels.reshape((config.num_vocab_tiling, vocab_tile_size))
    reshaped_labels = jax.lax.with_sharding_constraint(reshaped_labels, reshaped_data_spec)
    reshaped_segmentation = segmentation.reshape((config.num_vocab_tiling, vocab_tile_size))
    reshaped_segmentation = jax.lax.with_sharding_constraint(reshaped_segmentation, reshaped_data_spec)

    # Scan body accumulates loss from each tile given chunked hidden states and labels
    def _fwd_scan_body(loss_accumulator, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data
      hidden_chunk = jax.lax.with_sharding_constraint(hidden_chunk, chunked_hidden_spec)
      label_chunk = jax.lax.with_sharding_constraint(label_chunk, chunked_data_spec)
      segmentation_chunk = jax.lax.with_sharding_constraint(segmentation_chunk, chunked_data_spec)

      # Calculate logits for the current chunk
      chunk_logits = model.apply(
          {"params": gathered_params["params"]},
          hidden_chunk,
          deterministic=deterministic,
          method="logits_from_hidden_states",
      )
      chunk_logits = jax.lax.with_sharding_constraint(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(label_chunk, config.vocab_size)
      chunk_xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk)
      masked_xent = jnp.sum(chunk_xent * (segmentation_chunk != 0))
      loss_accumulator += masked_xent
      return loss_accumulator, None

    initial_loss = 0.0
    total_loss, _ = jax.lax.scan(
        _fwd_scan_body, initial_loss, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    residuals = (
        gathered_params,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    )

    return total_loss, residuals

  def _chunked_cross_entropy_loss_bwd(residuals, loss_cotangent):
    gathered_params, reshaped_hidden_states, reshaped_labels, reshaped_segmentation, batch_size, seq_len, emb_dim = (
        residuals
    )

    def _single_chunk_loss_fn(input_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk):
      chunk_logits = model.apply(
          {"params": input_params["params"]},
          input_hidden_chunk,
          deterministic=deterministic,
          method="logits_from_hidden_states",
      )
      chunk_logits = jax.lax.with_sharding_constraint(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(input_label_chunk, config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk)
      return jnp.sum(xent * (input_segmentation_chunk != 0))

    def _bwd_scan_body(grad_params_acc, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data

      # Apply sharding constraints to the chunk data
      hidden_chunk = jax.lax.with_sharding_constraint(hidden_chunk, chunked_hidden_spec)
      label_chunk = jax.lax.with_sharding_constraint(label_chunk, chunked_data_spec)
      segmentation_chunk = jax.lax.with_sharding_constraint(segmentation_chunk, chunked_data_spec)

      # Create a loss function closure that captures the current chunk's labels and segmentation.
      # This gives `jax.vjp` a function with the required signature: `loss(params, hidden_states)`.
      # pylint: disable=unnecessary-lambda-assignment
      loss_fn_for_vjp = lambda p, h: _single_chunk_loss_fn(p, h, label_chunk, segmentation_chunk)

      # Get the vector-Jacobian product function wrt both params and hidden states
      _, vjp_fn = jax.vjp(loss_fn_for_vjp, gathered_params, hidden_chunk)

      # 1.0 since total_loss is sum of all individual chunked loss
      (grad_params_update, grad_hidden_chunk) = vjp_fn(1.0)
      grad_hidden_chunk = jax.lax.with_sharding_constraint(grad_hidden_chunk, chunked_hidden_spec)

      grad_params_acc = jax.tree_util.tree_map(
          lambda acc, update: acc + update,
          grad_params_acc,
          grad_params_update,
      )
      return grad_params_acc, grad_hidden_chunk

    initial_grad_params_acc = jax.tree_util.tree_map(jnp.zeros_like, gathered_params)

    # The scan now returns the total gradients for the params in the final carry
    grad_params, grad_reshaped_hidden_states = jax.lax.scan(
        _bwd_scan_body, initial_grad_params_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    grad_reshaped_hidden_states = jax.lax.with_sharding_constraint(grad_reshaped_hidden_states, reshaped_hidden_spec)
    # TODO (chengnuojin): we may want to convert grad_params to bf16 to save memory
    # grad_params = jax.tree_util.tree_map(lambda x, y: y.astype(x.dtype), gathered_params, grad_params)
    # Chain-rule to accumulate gradients
    grad_params = jax.tree_util.tree_map(lambda g: g * loss_cotangent, grad_params)
    # Give back sharding constraint
    grad_reshaped_hidden_states = grad_reshaped_hidden_states.reshape((batch_size, seq_len, emb_dim))
    grad_reshaped_hidden_states = jax.lax.with_sharding_constraint(grad_reshaped_hidden_states, hidden_spec)
    return (
        grad_params,  # grad for params
        grad_reshaped_hidden_states.astype(reshaped_hidden_states.dtype),
        None,  # grad for reshaped_labels
        None,  # grad for reshaped_segmentation
    )

  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)

  total_loss = chunked_cross_entropy_loss(
      gathered_params,
      hidden_states,
      labels,
      segmentation,
  )

  return total_loss


def gradient_accumulation_wrapper(_loss_fn, config, model, params, params_shardings, data, dropout_rng, extra_dpo_args):
  """
  Calculates gradients using gradient accumulation.

  This function computes the gradient of `_loss_fn` over multiple microbatches
  and accumulates them before returning a single, averaged gradient. It uses
  `jax.lax.scan` for efficient accumulation on device.

  It also supports a `shard_optimizer_over_data` mode (e.g., ZeRO-1) where
  parameters are cast to bf16 and sharded *before* the accumulation loop
  to perform the all-gather in lower precision.

  Args:
      _loss_fn: The loss function to differentiate. Its signature is expected
          to be: `(model, config, data, dropout_rng, params, *extra_args, is_train=True)`.
      config: Model and training configuration object. Must contain
          `gradient_accumulation_steps` and `shard_optimizer_over_data`.
      model: The model module.
      params: The model parameters (PyTree).
      params_shardings: The sharding constraints for the parameters (PyTree).
      data: A PyTree of batched data. The leading dimension is assumed
          to be the total batch size (microbatch_size * num_accumulations).
      dropout_rng: JAX PRNGKey for dropout.
      extra_dpo_args: A tuple of extra arguments to pass to the loss function.

  Returns:
      A tuple containing:
      - total_loss (Array): The mean loss, averaged over all microbatches.
      - final_aux (PyTree): Auxiliary outputs, summed across microbatches.
      - raw_grads (PyTree): The accumulated and averaged gradients.
  """
  # When using Zero-1 optimizer sharding, cast params to lower precision and apply sharding constraints
  # so that all-gather is done once in the lower precision before the gradient accumulation loop
  if config.shard_optimizer_over_data:

    def convert_to_bf16(param):
      if param.dtype == jnp.float32:
        return param.astype(jnp.bfloat16)
      return param

    ga_params = jax.tree_util.tree_map(convert_to_bf16, params)
    ga_params = jax.tree.map(jax.lax.with_sharding_constraint, ga_params, params_shardings)
  else:
    ga_params = params

  grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)

  def accumulate_gradient(acc_grad_and_loss, data):
    ga_params = acc_grad_and_loss["ga_params"]

    (_, aux), cur_batch_gradient = grad_func(model, config, data, dropout_rng, ga_params, *extra_dpo_args, is_train=True)
    acc_grad_and_loss["loss"] += aux["total_loss"]
    acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
    acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
    acc_grad_and_loss["grad"] = jax.tree_util.tree_map(lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"])
    acc_grad_and_loss["total_weights"] += aux["total_weights"]
    return acc_grad_and_loss, aux

  def reshape_to_microbatch_accumulations(batch_arr):
    """Reshape global batch to microbatches, assuming batch axis is leading."""
    num_microbatches = config.gradient_accumulation_steps
    microbatch_shape = (batch_arr.shape[0] // num_microbatches, num_microbatches) + batch_arr.shape[1:]
    reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
    return jnp.swapaxes(reshaped_batch_arr, 0, 1)

  data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
  init_grad = jax.tree_util.tree_map(jnp.zeros_like, ga_params)
  init_grad = jax.tree.map(jax.lax.with_sharding_constraint, init_grad, params_shardings)
  init_grad_and_loss = {
      "loss": 0.0,
      "grad": init_grad,
      "total_weights": 0,
      "moe_lb_loss": 0.0,
      "mtp_loss": 0.0,
      "ga_params": ga_params,
  }

  grad_and_loss, aux = jax.lax.scan(
      accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
  )
  loss = (
      grad_and_loss["loss"] / grad_and_loss["total_weights"]
      + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
      + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
  )
  raw_grads = grad_and_loss["grad"]
  if config.shard_optimizer_over_data:
    raw_grads = jax.tree.map(jax.lax.with_sharding_constraint, raw_grads, params_shardings)
  raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], raw_grads)
  aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)  # pytype: disable=module-attr

  return loss, aux, raw_grads
