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
from MaxText.layers.blocks import DecoderLayer, Decoder
from MaxText.layers.normalizations import RMSNorm
from MaxText import max_utils
from MaxText import maxtext_utils

import jax
import jax.numpy as jnp

def log_tensor_stats(tensor, name, k=None, full=False):
    if tensor is None:
        jax.debug.print(f"MTP LOG: {'k=' + str(k) if k is not None else ''} {name}: IS NONE")
        return
    try:
        k_prefix = f"k={k} " if k is not None else ""
        name_prefix = f"MTP LOG: {k_prefix}{name}: "
        shape_dtype = f", shape={tensor.shape}, dtype={tensor.dtype}"

        # REMOVED :.4f from the placeholders
        jax.debug.print(name_prefix +
                      "mean={mean}, std={std}, min={min}, max={max}" +
                      shape_dtype,
                      mean=jnp.mean(tensor),
                      std=jnp.std(tensor),
                      min=jnp.min(tensor),
                      max=jnp.max(tensor))
        if full:
             jax.debug.print(name_prefix + "values={x}", x=tensor)
    except Exception as e:
        # This print is outside JAX, so direct f-string is fine for the exception message
        print(f"MTP LOG ERROR in log_tensor_stats for {name}: {e}")
        # Optional: jax.debug.print for the name, if you want it in the device logs
        jax.debug.print("MTP LOG ERROR in log_tensor_stats for name={name}", name=name)

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
    jax.debug.print(f"MTP LOG: --- Enter MultiTokenPredictionLayer k={k} ---")
    log_tensor_stats(prev_hidden_state, "layer_input_prev_h", k)
    log_tensor_stats(target_token_embedding, "layer_input_target_emb", k)

    # --- 1. Normalize Hidden State and Embedding ---
    embedding_norm_layer = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"mtp_{k}_embedding_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    embedding_norm = embedding_norm_layer(target_token_embedding)

    embedding_norm = nn.with_logical_constraint(
        embedding_norm, ('activation_batch', 'activation_length', 'activation_embed')
    )

    log_tensor_stats(embedding_norm, "embedding_norm", k)

    hidden_state_norm_layer = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name=f"mtp_{k}_hidden_state_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )

    hidden_state_norm = hidden_state_norm_layer(prev_hidden_state)

    hidden_state_norm = nn.with_logical_constraint(
        hidden_state_norm, ('activation_batch', 'activation_length', 'activation_embed')
    )

    log_tensor_stats(hidden_state_norm, "hidden_state_norm", k)


    # --- 2. Concatenate Normalized Representations ---
    # Shape: [B, S, 2*H]
    concatenated_features = jnp.concatenate([hidden_state_norm, embedding_norm], axis=-1)

    concatenated_features = nn.with_logical_constraint(
         concatenated_features, ('activation_batch', 'activation_length', None)
    )

    log_tensor_stats(concatenated_features, "concatenated_features", k)

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

    projected_features = nn.with_logical_constraint(
        projected_features, ('activation_batch', 'activation_length', 'activation_embed')
    )

    log_tensor_stats(projected_features, "projected_features", k)

    # --- 4. Pass through MTP Transformer Block ---
    next_hidden_state, _ = self.transformer_layer_module(config=cfg, mesh=mesh, name=f"mtp_{k}_transformer_layer")(
        inputs=projected_features,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=position_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    log_tensor_stats(next_hidden_state, "layer_output_next_h", k)
    jax.debug.print(f"MTP LOG: --- Exit MultiTokenPredictionLayer k={k} ---")
    # Shape: [B, S, H]
    # --- Return Processed Hidden State ---
    return next_hidden_state


class MultiTokenPredictionBlock(nn.Module):
  """Orchestrates the MTP process by running a sequence of MTP layers."""

  config: Config
  mesh: Mesh
  transformer_layer_module: Type[DecoderLayer]
  decoder: Type[Decoder]

  @nn.compact
  def __call__(
      self,
      main_hidden_state,
      input_ids,
      target_ids,
      target_mask,
      position_ids,
      decoder_segment_ids,
      deterministic,
      model_mode: str = MODEL_MODE_TRAIN,
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
      jax.debug.print(f"MTP LOG: ===== START MTP Layer k={k} =====")

      # Sequentially roll all tensors to prepare data for predicting the k-th future token.
      rolled_input_ids = maxtext_utils.roll_and_mask(rolled_input_ids)
      rolled_target_ids = maxtext_utils.roll_and_mask(rolled_target_ids)
      rolled_target_mask = maxtext_utils.roll_and_mask(rolled_target_mask)


      # TODO(@parambole) Not sure if this is requried check 
      rolled_position_ids = position_ids + k
    #   rolled_position_ids = maxtext_utils.roll_and_mask(position_ids)
      log_tensor_stats(rolled_input_ids, "rolled_input_ids", k)
      log_tensor_stats(rolled_position_ids, "rolled_position_ids", k)
      # Embed the k-th future input tokens using the shared embedding module
      target_token_embedding = self.decoder._apply_embedding(rolled_input_ids, rolled_position_ids, deterministic)
      log_tensor_stats(target_token_embedding, "target_token_embedding", k)

      # Instantiate and apply the MTP layer for this step
      mtp_layer = MultiTokenPredictionLayer(
          config=cfg,
          mesh=self.mesh,
          layer_number=k,
          name=f"mtp_layer_{k}",
          transformer_layer_module=self.transformer_layer_module,
      )

      next_mtp_hidden_state = mtp_layer(
          mtp_hidden_state, target_token_embedding, rolled_position_ids, decoder_segment_ids, deterministic, model_mode
      )
      log_tensor_stats(next_mtp_hidden_state, "next_mtp_hidden_state", k)

      # Project to logits using the shared embedding transpose
      mtp_logits = self.decoder._apply_output_head(next_mtp_hidden_state, deterministic, model_mode)
      
      log_tensor_stats(mtp_logits, "mtp_logits", k)

      # This runs only if the feature is enabled (flag > 0), for the specific layer (k == flag),
      # and when the 'mtp_acceptance' collection is mutable (i.e., during eval).
      if not self.is_initializing() and cfg.mtp_eval_target_layer == k and self.is_mutable_collection('mtp_acceptance'):
        mtp_top_1_pred = jnp.argmax(mtp_logits, axis=-1)
        self.sow('mtp_acceptance', 'mtp_preds', mtp_top_1_pred)
        self.sow('mtp_acceptance', 'mtp_mask', rolled_target_mask)

      # This condition ensures loss is only computed during training runs (`.apply`),
      # and not during model initialization (`.init()`).
      if not self.is_initializing() and model_mode == MODEL_MODE_TRAIN:
        true_one_hot = jax.nn.one_hot(rolled_target_ids, cfg.vocab_size)

        # Calculate cross-entropy loss for this specific layer's prediction
        mtp_xent, _ = max_utils.cross_entropy_with_logits(mtp_logits, true_one_hot, 0.0)
        mtp_xent_masked = mtp_xent * rolled_target_mask
        num_valid_tokens = jnp.sum(rolled_target_mask)
        
        avg_mtp_xent = jnp.sum(mtp_xent_masked) / (num_valid_tokens + 1e-8)
        jax.debug.print("MTP LOG: k={k} avg UN SCALED XENT: {x}", k=k, x=avg_mtp_xent)


        # Log probability of the true tokens
        log_probs = jax.nn.log_softmax(mtp_logits)
        true_token_log_probs = jnp.take_along_axis(log_probs, rolled_target_ids[..., None], axis=-1)[..., 0]
        masked_true_token_log_probs = true_token_log_probs * rolled_target_mask
        avg_true_token_log_prob = jnp.sum(masked_true_token_log_probs) / (num_valid_tokens + 1e-8)
        jax.debug.print("MTP LOG: k={k} avg true token log_prob: {x:.4f}", k=k, x=avg_true_token_log_prob)

          # self.sow(...)

        # "Sow" the loss values into the 'mtp_losses' collection for the
        self.sow("mtp_losses", "losses", jnp.sum(mtp_xent_masked))
        self.sow("mtp_losses", "weights", jnp.sum(rolled_target_mask))

      jax.debug.print(f"MTP LOG: ===== END MTP Layer k={k} =====")


      # The output of this layer is the input for the next, maintaining the causal chain.
      mtp_hidden_state = next_mtp_hidden_state
