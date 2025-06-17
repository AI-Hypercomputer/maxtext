"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""JAX implementation of the Multi Token Predicition https://arxiv.org/pdf/2412.19437 """

from typing import Optional, Type

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.layers.attentions import dense_general
from MaxText.layers.blocks import DecoderLayer
from MaxText.layers.normalizations import RMSNorm
from MaxText import max_utils
from MaxText import maxtext_utils


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
      decoder_segment_ids: Optional[jnp.ndarray],
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
    embedding_norm_layer = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"mtp_{k}_embedding_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    embedding_norm = embedding_norm_layer(target_token_embedding)

    hidden_state_norm_layer = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"mtp_{k}_hidden_state_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )

    hidden_state_norm = hidden_state_norm_layer(prev_hidden_state)

    # --- 2. Concatenate Normalized Representations ---
    # Shape: [B, S, 2*H]
    concatenated_features = jnp.concatenate([hidden_state_norm, embedding_norm], axis=-1)

    # --- 3. Project Concatenated Features ---
    # Projects from 2*H back down to H
    projection_layer = dense_general(
        inputs_shape=concatenated_features.shape,
        out_features_shape=cfg.base_emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("concat_embed", "embed"),
        name=f"mtp_{k}_projection",
    )
    # Shape: [B, S, H]
    projected_features = projection_layer(concatenated_features)

    # --- 4. Pass through MTP Transformer Block ---
    next_hidden_state, _ = self.transformer_layer_module(config=cfg, mesh=mesh, name=f"mtp_{k}_transformer_layer")(
        inputs=projected_features,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=position_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    # Shape: [B, S, H]
    # --- Return Processed Hidden State ---
    return next_hidden_state


class MultiTokenPredictionBlock(nn.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  config: Config
  mesh: Mesh
  transformer_layer_module: Type[DecoderLayer]

  @nn.compact
  def __call__(
      self,
      main_hidden_state,
      shared_embedding,
      output_head,
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

    # Range chosen to align with the naming convention of the paper
    for k in range(1, cfg.mtp_num_layers + 1):
      # Sequentially roll all tensors to prepare data for predicting the k-th future token.
      rolled_input_ids = maxtext_utils.roll_and_mask(rolled_input_ids)
      rolled_target_ids = maxtext_utils.roll_and_mask(rolled_target_ids)
      rolled_target_mask = maxtext_utils.roll_and_mask(rolled_target_mask)

      # Embed the k-th future input tokens using the shared embedding module
      target_token_embedding = shared_embedding(rolled_input_ids)

      # Instantiate and apply the MTP layer for this step
      mtp_layer = MultiTokenPredictionLayer(
          config=cfg,
          mesh=self.mesh,
          layer_number=k,
          name=f"mtp_layer_{k}",
          transformer_layer_module=self.transformer_layer_module,
      )

      next_mtp_hidden_state = mtp_layer(
          mtp_hidden_state, target_token_embedding, position_ids, decoder_segment_ids, deterministic
      )

      # Project to logits using the shared output head
      mtp_logits = output_head(hidden_states=next_mtp_hidden_state, deterministic=deterministic, model_mode=MODEL_MODE_TRAIN)

      # Calculate cross-entropy loss for this specific layer's prediction
      mtp_xent, _ = max_utils.cross_entropy_with_logits(mtp_logits, jax.nn.one_hot(rolled_target_ids, cfg.vocab_size), 0.0)
      mtp_xent_masked = mtp_xent * rolled_target_mask

      # This condition ensures loss is only computed during training runs (`.apply`),
      # and not during model initialization (`.init()`).
      if not self.is_initializing():
        # "Sow" the loss values into the 'mtp_losses' collection for the
        self.sow("mtp_losses", "losses", jnp.sum(mtp_xent_masked))
        self.sow("mtp_losses", "weights", jnp.sum(rolled_target_mask))

      # The output of this layer is the input for the next, maintaining the causal chain.
      mtp_hidden_state = next_mtp_hidden_state
