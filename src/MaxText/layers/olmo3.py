"""
Copyright 2023-2026 Google LLC

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

"""Decoder layer definition for Olmo 3 models."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from typing import Optional

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.common_types import AttentionType
from MaxText.layers import initializers
from MaxText.layers import attentions
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.layers.attentions import Attention
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers import nnx_wrappers
from MaxText.layers.linears import MlpBlock
from maxtext.utils import max_utils


# -----------------------------------------
# The Decoder Layer for Olmo3 models
# -----------------------------------------

OLMO3_ATTENTION_PATTERN = (
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.LOCAL_SLIDING,
    attentions.AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  """Get attention type based on layer ID."""
  layer_id %= len(OLMO3_ATTENTION_PATTERN)
  return OLMO3_ATTENTION_PATTERN[layer_id]


class Olmo3DecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: models.Config,
      mesh: Mesh,
      model_mode: str,
      attention_type: AttentionType,
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.attention_type = attention_type
    self.quant = quant

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_mlp_layer_norm = RMSNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=jnp.float32,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    current_rope_type = config.rope_type.lower()
    if self.attention_type == attentions.AttentionType.LOCAL_SLIDING:
      current_rope_type = "default"

    # Self-attention block
    self.attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_bias_in_projections=config.attention_bias,
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        query_pre_attn_scalar=(config.head_dim**-0.5),
        model_mode=model_mode,
        use_qk_norm=config.use_qk_norm,
        rope_type=current_rope_type,
        rngs=rngs,
    )

    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    attention_lnx, kv_cache = self.attention(
        inputs,
        inputs,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    # Normalize stream before addition
    attention_lnx = self.post_self_attention_layer_norm(attention_lnx)
    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    mlp_lnx = self.mlp(intermediate_inputs)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    # Normalize stream before addition
    mlp_lnx = self.post_mlp_layer_norm(mlp_lnx)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


Olmo3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Olmo3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Olmo3ScannableBlock(nnx.Module):
  """A repeatable block of Olmo 3 decoder layers.

    This block applies multiple decoder layers sequentially, using the attention
    pattern defined by OLMO3_ATTENTION_PATTERN. It's designed to be
    used with `nn.scan` for efficient compilation.

  Attributes:
    config: Config, MaxText model config
    mesh: Mesh, JAX device mesh (used for sharding)
    num_of_layers: int, number of decoder layers in the block
    quant: Optional[Quant], quantization config
  """

  def __init__(
      self,
      config: models.Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[Quant] = None,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    for layer_id in range(config.inhomogeneous_layer_cycle_interval):
      attention_type = get_attention_type(layer_id)
      layer_name = f"layers_{layer_id}"
      layer = Olmo3DecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          attention_type=attention_type,
          quant=self.quant,
          rngs=rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    cfg = self.config

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs
    for layer_id in range(cfg.inhomogeneous_layer_cycle_interval):
      layer_name = f"layers_{layer_id}"
      layer = getattr(self, layer_name)
      y = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )
      if cfg.scan_layers:
        y = y[0]
    if cfg.scan_layers:
      return y, None
    else:
      return y


Olmo3ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Olmo3ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
