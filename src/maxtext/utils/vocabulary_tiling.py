# Copyright 2025-2026 Google LLC
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

"""Functions for vocabulary tiling (VT)"""

import functools

from flax import linen as nn
from flax import nnx

import jax
import jax.numpy as jnp
from maxtext.utils.sharding import (
    maybe_shard_with_name,
    all_gather_over_fsdp,
    create_sharding,
)
from maxtext.common.common_types import ShardMode
from maxtext.utils import max_utils


# Submodule names whose params are used by logits_from_hidden_states_for_vocab_tiling:
# the final norm, the LM-head dense, and the embedding table when logits are tied.
# vocab_tiling_nnx_loss splits these out as the only params the loss differentiates.
_OUTPUT_HEAD_PATH_KEYS = ("token_embedder", "shared_embedding", "decoder_norm", "logits_dense")


def _is_output_head_param_path(path, _value):
  """Filter for nnx.split: True when the param path belongs to the output head."""
  keys = [str(getattr(k, "key", k)) for k in path]
  return any(k in keys for k in _OUTPUT_HEAD_PATH_KEYS)


def vocab_tiling_linen_loss(
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
    A tuple of (total_loss, total_z_loss) computed via vocab tiling.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  deterministic = not config.enable_dropout if is_train else True

  param_spec = nn.get_partition_spec(params)
  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      extra_stack_level=1,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)
  # TODO (chengnuojin) all gather only embedding table instead of all params after NNX module is enabled
  gathered_params = all_gather_over_fsdp(params, param_spec, model.mesh, config.logical_axis_rules, config.shard_mode)

  # Customized forward and backward maps for the embedding tiling
  @jax.custom_vjp
  def chunked_cross_entropy_loss(gathered_params, hidden_states, labels, segmentation):
    """
    Calculates the total cross-entropy loss using vocab tiling.
    """
    (total_loss, total_z_loss), _ = _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation)
    return total_loss, total_z_loss

  def _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

    reshaped_hidden_states = _reshape(
        hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
    )
    reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
    reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

    # Scan body accumulates loss from each tile given chunked hidden states and labels
    def _fwd_scan_body(accumulators, chunk_data):
      loss_accumulator, z_loss_accumulator = accumulators
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      # Calculate logits for the current chunk
      chunk_logits = model.apply(
          {"params": gathered_params["params"]},
          hidden_chunk,
          deterministic=deterministic,
          method="logits_from_hidden_states_for_vocab_tiling",
      )
      chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(label_chunk, config.vocab_size)
      chunk_xent, chunk_z_loss = max_utils.cross_entropy_with_logits(
          chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier
      )

      masked_xent = jnp.sum(chunk_xent * (segmentation_chunk != 0))
      masked_z_loss = jnp.sum(chunk_z_loss * (segmentation_chunk != 0))

      loss_accumulator += masked_xent
      z_loss_accumulator += masked_z_loss
      return (loss_accumulator, z_loss_accumulator), None

    initial_acc = (0.0, 0.0)
    (total_loss, total_z_loss), _ = jax.lax.scan(
        _fwd_scan_body, initial_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
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

    return (total_loss, total_z_loss), residuals

  def _chunked_cross_entropy_loss_bwd(residuals, cotangents):
    # Unpack the cotangents tuple. We ignore the z_loss cotangent since the gradients
    # of the z_loss term are already factored into the loss_cotangent.
    loss_cotangent, _ = cotangents

    gathered_params, reshaped_hidden_states, reshaped_labels, reshaped_segmentation, batch_size, seq_len, emb_dim = (
        residuals
    )

    def _single_chunk_loss_fn(input_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk):
      chunk_logits = model.apply(
          {"params": input_params["params"]},
          input_hidden_chunk,
          deterministic=deterministic,
          method="logits_from_hidden_states_for_vocab_tiling",
      )
      chunk_logits = _maybe_shard_with_name(chunk_logits, chunked_logits_spec)
      one_hot_label_chunk = jax.nn.one_hot(input_label_chunk, config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier)
      return jnp.sum(xent * (input_segmentation_chunk != 0))

    def _bwd_scan_body(grad_params_acc, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data

      # Apply sharding constraints to the chunk data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      # Create a loss function closure that captures the current chunk's labels and segmentation.
      # This gives `jax.vjp` a function with the required signature: `loss(params, hidden_states)`.
      # pylint: disable=unnecessary-lambda-assignment
      loss_fn_for_vjp = lambda p, h: _single_chunk_loss_fn(p, h, label_chunk, segmentation_chunk)

      # Get the vector-Jacobian product function wrt both params and hidden states
      _, vjp_fn = jax.vjp(loss_fn_for_vjp, gathered_params, hidden_chunk)

      # 1.0 since total_loss is sum of all individual chunked loss
      (grad_params_update, grad_hidden_chunk) = vjp_fn(1.0)
      grad_hidden_chunk = _maybe_shard_with_name(grad_hidden_chunk, chunked_hidden_spec)

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
    grad_reshaped_hidden_states = _maybe_shard_with_name(grad_reshaped_hidden_states, reshaped_hidden_spec)
    # Chain-rule to accumulate gradients
    grad_params = jax.tree_util.tree_map(lambda g: g * loss_cotangent, grad_params)
    # Cast cotangents back to each primal's dtype; custom_vjp requires dtype match.
    grad_params = jax.tree_util.tree_map(lambda x, y: y.astype(x.dtype), gathered_params, grad_params)
    # Give back sharding constraint
    grad_reshaped_hidden_states = _reshape(grad_reshaped_hidden_states, (batch_size, seq_len, emb_dim), hidden_spec)
    return (
        grad_params,  # grad for params
        grad_reshaped_hidden_states.astype(reshaped_hidden_states.dtype),
        None,  # grad for reshaped_labels
        None,  # grad for reshaped_segmentation
    )

  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)

  total_loss, total_z_loss = chunked_cross_entropy_loss(
      gathered_params,
      hidden_states,
      labels,
      segmentation,
  )

  return total_loss, total_z_loss


def vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train):
  """Computes cross-entropy loss with vocab tiling for NNX models.

  NNX equivalent of `vocab_tiling_linen_loss`. A `custom_vjp` runs the loss in
  vocab chunks via `jax.lax.scan` so the backward only holds one chunk's logits
  at a time, matching the Linen path's memory profile. `nnx.split` separates the
  output-head params (which the loss differentiates) from everything else; the
  rest of the model is passed through but not differentiated, so the scan's
  residuals stay small.

  Args:
    model: NNX model exposing ``logits_from_hidden_states_for_vocab_tiling``.
    hidden_states: Final hidden states from the decoder.
    data: Dict with ``targets`` and ``targets_segmentation``.
    config: Model and training config.
    is_train: Whether the model is in training mode.

  Returns:
    A tuple ``(total_loss, total_z_loss)``.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  deterministic = not config.enable_dropout if is_train else True
  model_mode = "train"

  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      extra_stack_level=1,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)

  # head_params is what the loss differentiates; other_params (transformer layers) and
  # rest (rngs) are passed through the custom_vjp but not differentiated. They go through
  # as primals rather than closure captures: capturing them leaks tracers across the
  # custom_vjp + lax.scan boundary, which fails for tied embeddings.
  graphdef, head_params, other_params, rest = nnx.split(model, _is_output_head_param_path, nnx.Param, ...)

  def _logits_for_chunk(chunk_head_params, chunk_other_params, chunk_rest, hidden_chunk):
    local_model = nnx.merge(graphdef, chunk_head_params, chunk_other_params, chunk_rest, copy=True)
    chunk_logits = local_model.logits_from_hidden_states_for_vocab_tiling(hidden_chunk, deterministic, model_mode)
    return _maybe_shard_with_name(chunk_logits, chunked_logits_spec)

  @jax.custom_vjp
  def chunked_cross_entropy_loss(chunk_head_params, chunk_other_params, chunk_rest, hidden_states, labels, segmentation):
    (total_loss, total_z_loss), _ = _chunked_cross_entropy_loss_fwd(
        chunk_head_params, chunk_other_params, chunk_rest, hidden_states, labels, segmentation
    )
    return total_loss, total_z_loss

  def _chunked_cross_entropy_loss_fwd(
      chunk_head_params, chunk_other_params, chunk_rest, hidden_states, labels, segmentation
  ):
    batch_size, seq_len, emb_dim = hidden_states.shape
    vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

    reshaped_hidden_states = _reshape(
        hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
    )
    reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
    reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

    def _fwd_scan_body(accumulators, chunk_data):
      loss_accumulator, z_loss_accumulator = accumulators
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      chunk_logits = _logits_for_chunk(chunk_head_params, chunk_other_params, chunk_rest, hidden_chunk)
      one_hot_label_chunk = jax.nn.one_hot(label_chunk, config.vocab_size)
      chunk_xent, chunk_z_loss = max_utils.cross_entropy_with_logits(
          chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier
      )

      masked_xent = jnp.sum(chunk_xent * (segmentation_chunk != 0))
      masked_z_loss = jnp.sum(chunk_z_loss * (segmentation_chunk != 0))

      return (loss_accumulator + masked_xent, z_loss_accumulator + masked_z_loss), None

    # Always accumulate in fp32 — `cross_entropy_with_logits` returns fp32 regardless of
    # logits dtype, and a bf16 carry would mismatch the body output type under lax.scan.
    initial_acc = (jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32))
    (total_loss, total_z_loss), _ = jax.lax.scan(
        _fwd_scan_body, initial_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    residuals = (
        chunk_head_params,
        chunk_other_params,
        chunk_rest,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    )
    return (total_loss, total_z_loss), residuals

  def _chunked_cross_entropy_loss_bwd(residuals, cotangents):
    # z_loss is folded into the xent loss inside cross_entropy_with_logits.
    loss_cotangent, _ = cotangents

    (
        chunk_head_params,
        chunk_other_params,
        chunk_rest,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    ) = residuals

    def _single_chunk_loss_fn(input_head_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk):
      chunk_logits = _logits_for_chunk(input_head_params, chunk_other_params, chunk_rest, input_hidden_chunk)
      one_hot_label_chunk = jax.nn.one_hot(input_label_chunk, config.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(chunk_logits, one_hot_label_chunk, z_loss=config.z_loss_multiplier)
      return jnp.sum(xent * (input_segmentation_chunk != 0))

    def _bwd_scan_body(grad_head_acc, chunk_data):
      hidden_chunk, label_chunk, segmentation_chunk = chunk_data
      hidden_chunk = _maybe_shard_with_name(hidden_chunk, chunked_hidden_spec)
      label_chunk = _maybe_shard_with_name(label_chunk, chunked_data_spec)
      segmentation_chunk = _maybe_shard_with_name(segmentation_chunk, chunked_data_spec)

      # pylint: disable=unnecessary-lambda-assignment
      loss_fn_for_vjp = lambda p, h: _single_chunk_loss_fn(p, h, label_chunk, segmentation_chunk)
      _, vjp_fn = jax.vjp(loss_fn_for_vjp, chunk_head_params, hidden_chunk)
      (grad_head_update, grad_hidden_chunk) = vjp_fn(1.0)
      grad_hidden_chunk = _maybe_shard_with_name(grad_hidden_chunk, chunked_hidden_spec)

      grad_head_acc = jax.tree_util.tree_map(lambda acc, update: acc + update, grad_head_acc, grad_head_update)
      return grad_head_acc, grad_hidden_chunk

    initial_grad_head = jax.tree_util.tree_map(jnp.zeros_like, chunk_head_params)

    grad_head, grad_reshaped_hidden_states = jax.lax.scan(
        _bwd_scan_body, initial_grad_head, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    grad_reshaped_hidden_states = _maybe_shard_with_name(grad_reshaped_hidden_states, reshaped_hidden_spec)
    grad_head = jax.tree_util.tree_map(lambda g: g * loss_cotangent, grad_head)
    grad_head = jax.tree_util.tree_map(lambda x, y: y.astype(x.dtype), chunk_head_params, grad_head)
    grad_reshaped_hidden_states = _reshape(grad_reshaped_hidden_states, (batch_size, seq_len, emb_dim), hidden_spec)

    # Return explicit zeros for other_params and rest, not None. With None, JAX builds
    # the zero cotangents with the wrong layer-axis order for scanned params, and the
    # AOT trace fails the cotangent shape check.
    grad_other = jax.tree_util.tree_map(jnp.zeros_like, chunk_other_params)
    grad_rest = jax.tree_util.tree_map(jnp.zeros_like, chunk_rest)
    return (
        grad_head,
        grad_other,
        grad_rest,
        grad_reshaped_hidden_states.astype(reshaped_hidden_states.dtype),
        None,
        None,
    )

  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)

  total_loss, total_z_loss = chunked_cross_entropy_loss(
      head_params, other_params, rest, hidden_states, labels, segmentation
  )
  return total_loss, total_z_loss
