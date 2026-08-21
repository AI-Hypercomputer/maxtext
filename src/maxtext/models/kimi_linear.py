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

"""Kimi K3 Linear Model Backbone in MaxText (NNX)."""

from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.common import common_types as ctypes
from maxtext.layers import linears, quantizations
from maxtext.layers.embeddings import Embed
from maxtext.layers.kimi_decoder_layer import KimiDecoderLayer
from maxtext.layers.normalizations import RMSNorm


class KimiLinearModel(nnx.Module):
  """Kimi K3 text-only backbone in MaxText using NNX."""

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      model_mode: str = ctypes.MODEL_MODE_TRAIN,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    self.token_embedder = Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        dtype=config.dtype,
        config=config,
        mesh=mesh,
        rngs=rngs,
    )

    self.layers = nnx.List([
        KimiDecoderLayer(
            config,
            mesh,
            layer_idx=i,
            model_mode=model_mode,
            quant=quant,
            rngs=rngs,
        )
        for i in range(config.num_decoder_layers)
    ])

    self.decoder_norm = RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

    self.logits_dense = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=config.vocab_size,
        weight_dtype=config.weight_dtype,
        dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
        kernel_axes=("embed_vocab", "vocab"),
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      input_ids: jax.Array,
      *,
      inputs_positions: Optional[jax.Array] = None,
      segment_ids: Optional[jax.Array] = None,
      initial_kda_states: Optional[list[Optional[jax.Array]]] = None,
  ) -> tuple[jax.Array, list[Optional[jax.Array]]]:
    """Executes the Kimi K3 backbone forward pass.

    Args:
      input_ids: Token IDs of shape (batch, seq_len).
      inputs_positions: Token positions of shape (batch, seq_len).
      segment_ids: Optional segment IDs of shape (batch, seq_len).
      initial_kda_states: Optional list of initial KDA recurrent states per layer.

    Returns:
      A tuple of (logits, kda_states) where logits has shape (batch, seq_len, vocab_size)
      and kda_states is a list of length `num_decoder_layers` containing the new KDA states.
    """
    # 1. Token Embeddings
    x = self.token_embedder(input_ids)

    # 2. Sequential Decoder Layers
    kda_states = []
    for i, layer in enumerate(self.layers):
      init_state = initial_kda_states[i] if initial_kda_states is not None else None
      x, kda_state = layer(
          x,
          inputs_positions=inputs_positions,
          segment_ids=segment_ids,
          initial_kda_state=init_state,
      )
      kda_states.append(kda_state)

    # 3. Final RMSNorm
    x = self.decoder_norm(x)

    # 4. Logits Projection
    logits = self.logits_dense(x)

    return logits, kda_states
