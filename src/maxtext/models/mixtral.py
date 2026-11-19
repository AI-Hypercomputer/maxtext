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


import functools

from flax import nnx
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import Dropout
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical

# -----------------------------------------
# The Decoder Layer for Mixtral
# -----------------------------------------


class MixtralDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # Physical sharding used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO the callees ignore it and let GSPMD infer the layout.
    self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.self_attention = Attention(
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
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        prefill_cache_axis_order=tuple(map(int, config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, config.compute_axis_order.split(","))),
        reshape_q=config.reshape_q,
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.MoeBlock_0 = moe.RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.mlp_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs)

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
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_layer_norm(inputs, out_sharding=self.out_sharding)
    lnx = self._maybe_shard_with_logical(lnx, self.activation_axis_names)

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        out_sharding=self.out_sharding,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = self._maybe_shard_with_logical(attention_lnx, self.activation_axis_names)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    load_balance_loss = None
    # NOTE: the naming mismatch here is to ensure reverse compatibility with existing checkpoints.
    # The `name` represents the weight name in JAX/checkpoints and so the class name
    # is just for readability.
    mlp_lnx, load_balance_loss, _ = self.MoeBlock_0(
        hidden_states,
        out_sharding=self.out_sharding,
        forced_routed_experts=forced_routed_experts,
        token_mask=moe.routing_token_mask(self.config, decoder_segment_ids, model_mode),
    )
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if getattr(self.config, "record_internal_nn_metrics", False):
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache

  # ---------------------------------------------------------------------------
  # Generalized Batch-Split layer adapter (used only when config.enable_batch_split)
  # ---------------------------------------------------------------------------

  def batch_split_supported(self) -> bool:
    """Returns True if this Mixtral MoE layer supports `batch_split`."""
    cfg = self.config
    return (
        getattr(cfg, "num_experts", 1) > 1
        and cfg.sparse_matmul
        and cfg.use_ring_of_experts
        and not cfg.te_moe_block
        and not getattr(cfg, "moe_use_fused_gmm3d", False)
        and cfg.num_moe_emb_chunks <= 0
        and not getattr(cfg, "record_internal_nn_metrics", False)
        and cfg.dropout_rate == 0.0
    )

  def batch_split_attn_and_route(
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
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    """Phase 1/2 TC compute: GQA self-attention + norms + MoE router gate."""
    del page_state, slot, kv_cache, forced_routed_experts
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    lnx = self.pre_self_attention_layer_norm(inputs, out_sharding=self.out_sharding)
    lnx = self._maybe_shard_with_logical(lnx, self.activation_axis_names)
    attention_lnx, _ = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        out_sharding=self.out_sharding,
        kv_cache=None,
        attention_metadata=attention_metadata,
    )
    attention_lnx = self._maybe_shard_with_logical(attention_lnx, self.activation_axis_names)
    intermediate_inputs = inputs + attention_lnx
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)
    moe_in, route_in, seq_lb_loss = self.MoeBlock_0.batch_split_route(hidden_states)
    return intermediate_inputs, moe_in, route_in, seq_lb_loss

  def batch_split_dispatch(self, moe_in, route_in):
    """Weight-free EP All-Gather of microbatch tokens and routing metadata on SC0."""
    return self.MoeBlock_0.batch_split_dispatch(moe_in, route_in)

  def batch_split_dispatch_bwd(self, moe_in, route_in, g_x_val_ag, g_route_ag_diff):
    """Weight-free backward of `batch_split_dispatch` (EP Reduce-Scatter on SC0)."""
    return self.MoeBlock_0.batch_split_dispatch_bwd(moe_in, route_in, g_x_val_ag, g_route_ag_diff)

  def batch_split_experts(
      self, moe_in_ref, x_val_ag, x_scale_ag, route_ag, decoder_input_tokens=None, return_wgrad_hooks: bool = False
  ):
    """Local expert permute + GMMs + unpermute without EP dispatch/combine."""
    del decoder_input_tokens
    return self.MoeBlock_0.batch_split_experts(
        moe_in_ref, x_val_ag, x_scale_ag, route_ag, return_wgrad_hooks=return_wgrad_hooks
    )

  def batch_split_shared_expert(self, moe_in):
    """Mixtral has no shared expert branch."""
    return jnp.zeros_like(moe_in)

  def batch_split_combine(self, y_partial, moe_in_ref):
    """Weight-free EP Reduce-Scatter of routed expert partial outputs on SC0."""
    routed_out = self.MoeBlock_0.batch_split_combine(y_partial, moe_in_ref)
    return self._maybe_shard_with_logical(routed_out, self.activation_axis_names)

  def batch_split_combine_bwd(self, g_out, moe_in_ref, *, defer_dequant: bool = False):
    """Weight-free backward of `batch_split_combine` (EP All-Gather on SC0)."""
    return self.MoeBlock_0.batch_split_combine_bwd(g_out, moe_in_ref, defer_dequant=defer_dequant)

  def batch_split_finalize_bias_updates(self, summed_counts):
    """Finalizes microbatch-summed expert counts into routed bias updates."""
    return self.MoeBlock_0.batch_split_finalize_bias_updates(summed_counts)
