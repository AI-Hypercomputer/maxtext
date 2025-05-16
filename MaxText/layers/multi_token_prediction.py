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

import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.layers.attentions import DenseGeneral
from MaxText.layers.models import DecoderLayer
from MaxText.layers.normalizations import RMSNorm


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
    projection_layer = DenseGeneral(
        features=cfg.base_emb_dim,
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
