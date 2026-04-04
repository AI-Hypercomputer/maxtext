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

import jax
import jax.numpy as jnp
from maxtext.utils.sharding import (
    maybe_shard_with_name,
    all_gather_over_fsdp,
    create_sharding,
)
from maxtext.common.common_types import ShardMode
from maxtext.utils import max_utils


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
      ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length_no_exp"),
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
  def chunked_cross_entropy_loss(
      gathered_params, hidden_states, labels, segmentation
  ):
    """Calculates the total cross-entropy loss using vocab tiling."""
    # if both batch-sequence tiling and vocab tiling are enabled, call
    # _b_v_chunked_cross_entropy_loss_fwd
    if config.num_vocab_tiling > 1:
      (total_loss, total_z_loss), _ = _b_v_chunked_cross_entropy_loss_fwd(
          gathered_params, hidden_states, labels, segmentation
      )
    else:
      (total_loss, total_z_loss), _ = _chunked_cross_entropy_loss_fwd(
          gathered_params, hidden_states, labels, segmentation
      )
    return total_loss, total_z_loss

  def _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    batch_seq_tile_size = (batch_size * seq_len) // config.num_batch_seq_tiling

    reshaped_hidden_states = _reshape(
        hidden_states,
        (config.num_batch_seq_tiling, batch_seq_tile_size, emb_dim),
        reshaped_hidden_spec,
    )
    reshaped_labels = _reshape(
        labels,
        (config.num_batch_seq_tiling, batch_seq_tile_size),
        reshaped_data_spec,
    )
    reshaped_segmentation = _reshape(
        segmentation,
        (config.num_batch_seq_tiling, batch_seq_tile_size),
        reshaped_data_spec,
    )

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
          method="logits_from_hidden_states",
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

  # Chunked cross entropy loss forward pass, chunk along batch-sequence and
  # vocab dimensions.
  def _b_v_chunked_cross_entropy_loss_fwd(
      gathered_params, hidden_states, labels, segmentation
  ):
    batch_size, seq_len, emb_dim = hidden_states.shape
    v_dim = config.vocab_size

    b_dim = batch_size * seq_len
    b_block_sz = b_dim // config.num_batch_seq_tiling
    v_block_sz = v_dim // config.num_vocab_tiling

    if b_dim % b_block_sz != 0 or v_dim % v_block_sz != 0:
      raise ValueError(
          "Batch/sequence dimension and vocab dimension must be divisible by"
          " their block sizes."
      )

    num_b_blocks = b_dim // b_block_sz
    num_v_blocks = v_dim // v_block_sz

    flat_hidden = _reshape(
        hidden_states,
        (b_dim, emb_dim),
        create_sharding(
            model.mesh,
            ("activation_embed_and_logits_batch_sequence", "activation_embed"),
        ),
    )
    flat_labels = _reshape(
        labels,
        (b_dim,),
        create_sharding(
            model.mesh, ("activation_embed_and_logits_batch_sequence",)
        ),
    )
    flat_segmentation = _reshape(
        segmentation,
        (b_dim,),
        create_sharding(
            model.mesh, ("activation_embed_and_logits_batch_sequence",)
        ),
    )

    if config.logits_via_embedding:
      w = gathered_params["params"]["shared_embedding"]["embedding"]
    else:
      w = gathered_params["params"]["decoder"]["logits_dense"]["kernel"]

    if hasattr(w, "unbox"):
      w = w.unbox()
    elif hasattr(w, "value"):
      w = w.value

    def b_loop_body(i, carry):
      total_loss, total_z_loss = carry
      b_start = i * b_block_sz

      def v_loop_body(j, v_carry):
        lse_b_, b_loss_sum_neg_logits_ = v_carry
        v_start = j * v_block_sz
        labels_b = jax.lax.dynamic_slice(flat_labels, (b_start,), (b_block_sz,))
        x_b = jax.lax.dynamic_slice(
            flat_hidden, (b_start, 0), (b_block_sz, emb_dim)
        )

        # Apply normalization to the batch block
        x_b_norm = model.apply(
            {"params": gathered_params["params"]},
            x_b,
            deterministic=deterministic,
            method="normalize_hidden_states",
        )
        x_b_norm = _maybe_shard_with_name(x_b_norm, chunked_hidden_spec)

        # Extract w_j
        if config.logits_via_embedding:
          # Attend on embedding table. Table is (vocab_size, emb_dim)
          # Transpose to (emb_dim, vocab_size)
          w_j = jax.lax.dynamic_slice(w.T, (0, v_start), (emb_dim, v_block_sz))
        else:
          w_j = jax.lax.dynamic_slice(w, (0, v_start), (emb_dim, v_block_sz))

        # Compute logits for the block
        logits_bv = jnp.dot(x_b_norm, w_j)

        if config.logits_via_embedding and config.normalize_embedding_logits:
          logits_bv = logits_bv / jnp.sqrt(emb_dim)
        if config.final_logits_soft_cap:
          logits_bv = logits_bv / config.final_logits_soft_cap
          logits_bv = jnp.tanh(logits_bv) * config.final_logits_soft_cap

        if config.cast_logits_to_fp32:
          logits_bv = logits_bv.astype(jnp.float32)

        lse_b__ = jnp.logaddexp(lse_b_, jax.nn.logsumexp(logits_bv, axis=-1))

        labels_one_hot = jax.nn.one_hot(
            labels_b - v_start, v_block_sz, dtype=logits_bv.dtype
        )
        b_loss_sum_neg_logits__ = b_loss_sum_neg_logits_ - jnp.sum(
            logits_bv * labels_one_hot, axis=-1
        )
        return lse_b__, b_loss_sum_neg_logits__

      lse_b, b_loss_sum_neg_logits = jax.lax.fori_loop(
          0,
          num_v_blocks,
          v_loop_body,
          (
              jnp.full((b_block_sz,), -jnp.inf, dtype=jnp.float32),
              jnp.zeros((b_block_sz,), dtype=jnp.float32),
          ),
      )

      segmentation_b = jax.lax.dynamic_slice(
          flat_segmentation, (b_start,), (b_block_sz,)
      )
      mask = (segmentation_b != 0).astype(jnp.float32)

      # Z-loss
      z_loss_b = config.z_loss_multiplier * jnp.square(lse_b) * mask
      total_z_loss += jnp.sum(z_loss_b)

      b_loss_sum_neg_logits = b_loss_sum_neg_logits * mask
      lse_b_masked = lse_b * mask

      total_loss += jnp.sum(b_loss_sum_neg_logits) + jnp.sum(lse_b_masked)

      return total_loss, total_z_loss

    initial_acc = (0.0, 0.0)
    total_loss, total_z_loss = jax.lax.fori_loop(
        0,
        num_b_blocks,
        b_loop_body,
        initial_acc,
    )

    # Reshape the flattened 2D tensors `(b_dim, ...)` into 3D chunked tensors
    # `(num_b_blocks, b_block_sz, ...)` so we can process them sequentially
    # over the batch dimension using `jax.lax.scan` in the backward pass.
    # TODO(b/486111493): When we replace the bwd pass, perhaps we can think
    # about what to do with these reshape operations.
    reshaped_hidden_states = _reshape(
        flat_hidden, (num_b_blocks, b_block_sz, emb_dim), reshaped_hidden_spec
    )
    reshaped_labels = _reshape(
        flat_labels, (num_b_blocks, b_block_sz), reshaped_data_spec
    )
    reshaped_segmentation = _reshape(
        flat_segmentation, (num_b_blocks, b_block_sz), reshaped_data_spec
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
          method="logits_from_hidden_states",
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
    # TODO (chengnuojin): we may want to convert grad_params to bf16 to save memory
    # grad_params = jax.tree_util.tree_map(lambda x, y: y.astype(x.dtype), gathered_params, grad_params)
    # Chain-rule to accumulate gradients
    grad_params = jax.tree_util.tree_map(lambda g: g * loss_cotangent, grad_params)
    # Give back sharding constraint
    grad_reshaped_hidden_states = _reshape(grad_reshaped_hidden_states, (batch_size, seq_len, emb_dim), hidden_spec)
    return (
        grad_params,  # grad for params
        grad_reshaped_hidden_states.astype(reshaped_hidden_states.dtype),
        None,  # grad for reshaped_labels
        None,  # grad for reshaped_segmentation
    )

  if config.num_vocab_tiling > 1:
    chunked_cross_entropy_loss.defvjp(
        _b_v_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd
    )
  else:
    chunked_cross_entropy_loss.defvjp(
        _chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd
    )

  total_loss, total_z_loss = chunked_cross_entropy_loss(
      gathered_params,
      hidden_states,
      labels,
      segmentation,
  )

  return total_loss, total_z_loss
