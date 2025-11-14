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

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding

from flax import linen as nn
from flax import nnx

from MaxText.inference import page_manager
from MaxText.common_types import Config
from MaxText import max_utils
from MaxText.sharding import maybe_shard_with_logical
from MaxText.layers.linears import Dropout, MlpBlock
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.attentions import Attention
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm
from MaxText.common_types import MODEL_MODE_PREFILL
from MaxText.layers import moe


# -----------------------------------------
# Routed Experts Only Decoder Layer
# -----------------------------------------


class RoutedExpertsOnlyDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
  ):

    self.config = config
    self.mesh = mesh
    self.quant = quant

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.mlp = moe.RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.mlp_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=self.mesh,
        shard_mode=config.shard_mode,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      previous_chunk=None,
  ):
    cfg = self.config

    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")


    # MLP block.
    mlp_lnx = self.mlp(inputs)[0]
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)
    layer_output = mlp_lnx

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
      return layer_output

RoutedExpertsOnlyDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    RoutedExpertsOnlyDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)