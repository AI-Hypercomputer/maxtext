# Copyright 2023–2026 Google LLC
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

"""Decoder layer definition for mixtral."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from flax import linen as nn
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.attentions import attention_as_linen
from maxtext.layers.normalizations import rms_norm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical

# -----------------------------------------
# The Decoder Layer for Mixtral
# -----------------------------------------


class MixtralDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
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
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    cfg = self.config
    mesh = self.mesh

    activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    def shard(x):
      return maybe_shard_with_logical(
          x, activation_axis_names, mesh=mesh, shard_mode=cfg.shard_mode,
          rules=cfg.logical_axis_rules, skip_trivial_specs=True,
      )

    inputs = shard(inputs)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = rms_norm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )(inputs)
    lnx = shard(lnx)

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)

    attention_lnx, kv_cache = attention_as_linen(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=tuple(map(int, cfg.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, cfg.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, cfg.compute_axis_order.split(","))),
        reshape_q=cfg.reshape_q,
        use_ragged_attention=cfg.use_ragged_attention,
        ragged_block_size=cfg.ragged_block_size,
        model_mode=model_mode,
        name="self_attention",
    )(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = shard(attention_lnx)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = rms_norm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )(intermediate_inputs)
    hidden_states = shard(hidden_states)

    load_balance_loss = None
    # NOTE: the naming mismatch here is to ensure reverse compatibility with existing checkpoints.
    # The `name` represents the weight name in JAX/checkpoints and so the class name
    # is just for readability.
    mlp_lnx, load_balance_loss, _ = moe.get_routed_moe(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        name="MoeBlock_0",
    )(hidden_states)
    mlp_lnx = shard(mlp_lnx)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)
    layer_output = shard(layer_output)

    if cfg.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

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


MixtralDecoderLayerToLinen = MixtralDecoderLayer
