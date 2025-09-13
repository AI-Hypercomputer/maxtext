# Copyright 2023–2025 Google LLC
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

"""JAX implementation of the Multi Token Prediction https://arxiv.org/pdf/2412.19437 """

from typing import Type

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.layers.linears import dense_general
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.decoders import Decoder, DecoderLayer
from MaxText import max_utils
from MaxText import maxtext_utils

from MaxText.globals import EPS


def roll_and_mask(x: jnp.ndarray, shift: int = -1) -> jnp.ndarray:
  """
  Performs a leftward roll on the sequence axis (axis=1) and masks the
  newly created invalid positions at the end of the sequence.
  Assumes input `x` has a batch dimension at axis 0 and sequence at axis 1.

  Args:
    x: The input array of shape [batch, seq_len, ...].
    shift: The number of positions to shift left.

  Returns:
    The rolled array of the same shape as x.
  """
  # If shift is 0, it's a no-op. Return the original array.
  if shift == 0:
    return x

  # to set the last `abs(shift)` elements of the sequence to zero.
  return jnp.roll(x, shift, axis=1).at[:, shift:, ...].set(0)


class MultiTokenPredictionLayer(nn.Module):
  """
  Implements Multi-Token Prediction (MTP) step:
      1. Normalization of previous hidden state and target token embedding.
      2. Concatenation and Projection of normalized features.
      3. Processing through a Transformer Decoder Layer.

      Equation Representation (Conceptual):
          norm_h = RMSNorm(h_prev)
          norm_e = RMSNorm(e_target)
          h_proj = W_p(concat(norm_h, norm_e))
          h_next = TransformerLayer(h_proj, pos_ids, segment_ids, ...)

      It takes the previous hidden state and target embedding as input and outputs the
      processed hidden state from its internal transformer block.
  """

  config: Config
  mesh: Mesh
  layer_number: int
  transformer_layer_module: Type[DecoderLayer] = DecoderLayer

  @nn.compact
  def __call__(
      self,
      prev_hidden_state: jnp.ndarray,
      target_token_embedding: jnp.ndarray,
      position_ids: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jnp.ndarray:
    """
    Applies the MTP combination, projection, and internal transformer processing.

    Args:
        prev_hidden_state: Hidden state from the previous step/layer.
                           Shape: [batch, seq_len, hidden_size]
        target_token_embedding: Embedding of the target token. In the context of MTP,
                                this often refers to a token at a position relative
                                to the current step, where the offset is determined
                                by the layer number `k` (i.e., token t+k).
                                Shape: [batch, seq_len, embed_dim]
        position_ids: Original position IDs for the sequence.
                      Shape: [batch, seq_len]
        decoder_segment_ids: Original segment IDs for the sequence (for attention mask).
                             Shape: [batch, seq_len]
        deterministic: If true, disable dropout.
        model_mode: The current operational mode (train, eval, decode).

    Returns:
        next_hidden_state: The hidden state produced by this MTP step's internal transformer.
                           Shape: [batch, seq_len, hidden_size]
    """
    cfg = self.config
    mesh = self.mesh
    k = self.layer_number

    # --- 1. Normalize Hidden State and Embedding ---
    embedding_norm_layer = rms_norm(
        num_features=target_token_embedding.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"mtp_{k}_embedding_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    embedding_norm = embedding_norm_layer(target_token_embedding)

    hidden_state_norm_layer = rms_norm(
        num_features=prev_hidden_state.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"mtp_{k}_hidden_state_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )

    hidden_state_norm = hidden_state_norm_layer(prev_hidden_state)

    # --- 2. Concatenate Normalized Representations ---
    # Shape: [B, S, 2*H]
    concatenated_features = jnp.concatenate([embedding_norm, hidden_state_norm], axis=-1)

    # --- 3. Project Concatenated Features ---
    # Projects from 2*H back down to H
    projection_layer = dense_general(
        inputs_shape=concatenated_features.shape,
        out_features_shape=cfg.base_emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        use_bias=False,
        kernel_axes=("concat_embed", "embed"),
        name=f"mtp_{k}_projection",
    )
    # Shape: [B, S, H]
    projected_features = projection_layer(concatenated_features)

    # --- 4. Pass through MTP Transformer Block ---
    output = self.transformer_layer_module(
        config=cfg, mesh=mesh, model_mode=model_mode, name=f"mtp_{k}_transformer_layer"
    )(
        inputs=projected_features,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=position_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    if isinstance(output, tuple):
      # Handles the scan=True case, where the output is a tuple.
      next_hidden_state = output[0]
    else:
      # Handles the scan=False case, where the output is a single tensor.
      next_hidden_state = output

    # Shape: [B, S, H]
    # --- Return Processed Hidden State ---
    return next_hidden_state


class MultiTokenPredictionBlock(nn.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  config: Config
  mesh: Mesh
  transformer_layer_module: Type[DecoderLayer]
  decoder: Decoder

  @nn.compact
  def __call__(
      self,
      shared_embedding,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      position_ids,
      decoder_segment_ids,
      deterministic,
  ):
    cfg = self.config
    # The initial hidden state for the MTP chain is the raw output from the main model.
    mtp_hidden_state = main_hidden_state

    # These variables are updated sequentially in each loop iteration,
    # moving the prediction window one token to the right each time.
    rolled_input_ids = input_ids
    rolled_target_ids = target_ids
    rolled_target_mask = target_mask
    rolled_position_id = position_ids

    # Range chosen to align with the naming convention of the paper
    for k in range(1, cfg.mtp_num_layers + 1):
      # Sequentially roll all tensors to prepare data for predicting the k-th future token.
      rolled_input_ids = roll_and_mask(rolled_input_ids)
      rolled_target_ids = roll_and_mask(rolled_target_ids)
      rolled_target_mask = roll_and_mask(rolled_target_mask)
      rolled_position_id = roll_and_mask(rolled_position_id)

      # Embed the k-th future input tokens using the shared embedding module
      target_token_embedding = self.decoder._apply_embedding(
          shared_embedding, rolled_input_ids, rolled_position_id, deterministic, self.decoder.model_mode
      )

      # Instantiate and apply the MTP layer for this step
      mtp_layer = MultiTokenPredictionLayer(
          config=cfg,
          mesh=self.mesh,
          layer_number=k,
          name=f"mtp_layer_{k}",
          transformer_layer_module=self.transformer_layer_module,
      )

      next_mtp_hidden_state = mtp_layer(
          mtp_hidden_state,
          target_token_embedding,
          position_ids,
          decoder_segment_ids,
          deterministic,
          self.decoder.model_mode,
      )

      # Project to logits using the shared embedding transpose
      mtp_logits = self.decoder._apply_output_head(shared_embedding, next_mtp_hidden_state, deterministic)

      # Calculate cross-entropy loss for this specific layer's prediction
      mtp_xent, _ = max_utils.cross_entropy_with_logits(
          mtp_logits, jax.nn.one_hot(rolled_target_ids, cfg.vocab_size), 0.0
      )
      mtp_xent_masked = mtp_xent * rolled_target_mask

      # This logic doesn't run during model initialization to avoid unwated population of the mutable collections.
      if not self.is_initializing():
        # For evaluation, save the top prediction and a valid token mask.
        # This is only active for the target layer during an eval run.
        if cfg.mtp_eval_target_module == k and self.is_mutable_collection("mtp_acceptance"):
          mtp_top_1_pred = jnp.argmax(mtp_logits, axis=-1)
          self.sow("mtp_acceptance", "mtp_preds", mtp_top_1_pred)
          self.sow("mtp_acceptance", "mtp_mask", rolled_target_mask)

        # For training, save the loss components for this MTP head.
        # This is only active during a training run.
        if self.is_mutable_collection("mtp_losses"):
          self.sow("mtp_losses", "losses", jnp.sum(mtp_xent_masked))
          self.sow("mtp_losses", "weights", jnp.sum(rolled_target_mask))

      # The output of this layer is the input for the next, maintaining the causal chain.
      mtp_hidden_state = next_mtp_hidden_state


def calculate_mtp_loss(intermediate_outputs, config):
  """Calculates the Multi Token Prediction loss from intermediate outputs."""
  losses_path = ("mtp_losses", "mtp_block", "losses")
  weights_path = ("mtp_losses", "mtp_block", "weights")

  mtp_losses = maxtext_utils.get_nested_value(intermediate_outputs, losses_path, default=())
  mtp_weights = maxtext_utils.get_nested_value(intermediate_outputs, weights_path, default=())

  if not mtp_losses:  # MTP heads did not run
    return 0.0

  sum_of_all_mtp_losses = jnp.sum(jnp.array(mtp_losses))
  sum_of_all_mtp_weights = jnp.sum(jnp.array(mtp_weights))

  avg_mtp_loss = sum_of_all_mtp_losses / (sum_of_all_mtp_weights + EPS)
  scaled_mtp_loss = avg_mtp_loss * config.mtp_loss_scaling_factor
  return scaled_mtp_loss


def calculate_mtp_acceptance_rate(intermediate_outputs, config):
  """Calculates the MTP acceptance rate from intermediate outputs."""

  sown_data = maxtext_utils.get_nested_value(intermediate_outputs, ("mtp_acceptance", "mtp_block"), {})
  mtp_preds = maxtext_utils.get_nested_value(sown_data, ("mtp_preds",), [None])[0]
  valid_mask = maxtext_utils.get_nested_value(sown_data, ("mtp_mask",), [None])[0]

  # These values are only "sown" (saved) during an evaluation run and only for the specific
  # MTP layer specified by `config.mtp_eval_target_module`. This check handles cases
  # where the required data is absent (e.g., during a training step) and prevents errors.
  if mtp_preds is None or valid_mask is None:
    return 0.0

  # Get the main model's greedy predictions from the logits.
  main_model_preds = jnp.argmax(intermediate_outputs["logits"], axis=-1)

  # Roll the main model's predictions to align them in time with the MTP head's target.
  rolled_main_preds = main_model_preds
  for _ in range(config.mtp_eval_target_module):
    rolled_main_preds = roll_and_mask(rolled_main_preds)

  # Compare the aligned predictions. The `valid_mask` ensures that the comparison
  # only happens on valid tokens, ignoring the placeholder values introduced at the
  # end of the sequence by the `roll_and_mask` operation.
  correct_predictions = jnp.sum((mtp_preds == rolled_main_preds) * valid_mask)
  total_valid_tokens = jnp.sum(valid_mask)

  # Return acceptance rate as a percentage
  return (correct_predictions / (total_valid_tokens + EPS)) * 100
