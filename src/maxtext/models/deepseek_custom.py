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

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Optional

from flax import nnx
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import AttentionType
from maxtext.common.common_types import Config
from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.inference import page_manager
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.linears import Dropout
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding
from maxtext.utils.sharding import maybe_shard_with_logical


class CustomAttention(attentions.Attention):
  """Custom GQA attention that supports sub-dimensional output."""

  def init_out_w(self, output_dim: int) -> nnx.Module:
    """Initializes the output projection."""
    if not self.config.attention_output_dim > 0:
      raise ValueError(
          "attention_output_dim must be set to a positive integer for CustomAttention."
      )

    in_features = (self.num_query_heads, self.head_dim)
    out_kernel_axis = (
        (None, None, None)
        if self.config.ici_context_autoregressive_parallelism > 1
        else ("heads", "kv", "embed")
    )
    axis = (-2, -1)

    return linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.config.attention_output_dim,
        axis=axis,
        kernel_init=self.kernel_init,
        kernel_axes=out_kernel_axis,  # trade speed with memory
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )


class DeepSeekGenericLayer(nnx.Module):
  """Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.
  This class follows a pattern of separating module creation from execution.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.is_mhc_enabled = config.mhc_expansion_rate > 1
    self.layer_idx = layer_idx

    # GQA Hybrid routing calculation
    attention_layer_hybrid_ratio = self.config.attention_layer_hybrid_ratio

    # All dense layers use local attention.
    is_global_attention = False
    attention_cls = CustomAttention
    if isinstance(self, DeepSeekDenseLayer):
      attention_cls = attentions.Attention
    elif attention_layer_hybrid_ratio > 0:
      is_global_attention = (self.layer_idx + 1) % attention_layer_hybrid_ratio == 0

    self.attention_type = AttentionType.GLOBAL if is_global_attention else AttentionType.LOCAL_SLIDING

    if is_global_attention and self.config.global_num_query_heads > 0:
      self.num_query_heads = self.config.global_num_query_heads
      self.num_kv_heads = self.config.global_num_kv_heads
      self.sliding_window_size = None
    elif not is_global_attention and self.config.local_num_query_heads > 0:
      self.num_query_heads = self.config.local_num_query_heads
      self.num_kv_heads = self.config.local_num_kv_heads
      self.sliding_window_size = self.config.sliding_window_size
    else:
      self.num_query_heads = self.config.base_num_query_heads
      self.num_kv_heads = self.config.base_num_kv_heads
      self.sliding_window_size = None

    max_logging.log(
        f"Initializing {self.__class__.__name__} - Layer: {layer_idx}, "
        f"Context: {'Global' if is_global_attention else 'Local'} "
        f"(Q_Heads: {self.num_query_heads}, KV_Heads: {self.num_kv_heads})"
    )

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(
        self.config, self.model_mode
    )
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names)
    self.mlp_intermediate_sharding = create_sharding(
        self.mesh, self.mlp_logical_axis_names
    )

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.self_attention = attention_cls(
        config=self.config,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=self.attention_type,
        sliding_window_size=self.sliding_window_size,
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        model_mode=model_mode,
        rngs=rngs,
        attn_logits_soft_cap=self.config.attn_logits_soft_cap,
    )

    self.dropout = Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

    # Optional projection up from intermediate state back to emb dim.
    # This corresponds to the transition from the latent/compressed space back to the main model dimension.
    skip_projection = isinstance(self, DeepSeekDenseLayer)

    if (
        self.config.attention_output_dim > 0
        and self.config.attention_output_dim != self.config.emb_dim
        and not skip_projection
    ):
      out_kernel_axis = (
          (None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("mlp", "embed")
      )
      self.layer_up_projection = linears.DenseGeneral(
          in_features_shape=self.config.attention_output_dim,
          out_features_shape=self.config.emb_dim,
          axis=-1,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=out_kernel_axis,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          quant=quant,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          use_bias=False,
          rngs=self.rngs,
      )
    else:
      self.layer_up_projection = None

  def mlp_op(self, x, deterministic, *args, **kwargs):
    """Executes the MLP operation. To be implemented by subclasses."""
    raise NotImplementedError()

  def with_logical_constraint(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def dropout_op(self, x, deterministic):
    dropout = self.dropout(x, deterministic=deterministic)
    return self.with_logical_constraint(dropout)

  def pre_attention_norm_op(self, x):
    pre_attention_norm = self.pre_self_attention_layer_norm(x)
    return self.with_logical_constraint(pre_attention_norm)

  def post_attention_norm_op(self, x):
    post_attention_norm = self.post_self_attention_layer_norm(x)
    return self.with_logical_constraint(post_attention_norm)

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    """Executes the attention layer."""
    attention_result, _ = self.self_attention(
        x,
        x,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=self.model_mode,
        out_sharding=self.out_sharding,
        previous_chunk=previous_chunk,
        page_state=page_state,
        slot=slot,
    )
    return self.with_logical_constraint(attention_result)

  @property
  def logical_axis_names(self):
    """Generate logical names for activations generally."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_embed"]
    return axis_names

  @property
  def mlp_logical_axis_names(self):
    """Generate logical names for activations in MLP."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_mlp"]
    return axis_names

  def post_process(self, layer_output, load_balance_loss, moe_bias_updates, kv_cache=None):
    """postprocessing."""

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache


class DeepSeekDenseLayer(DeepSeekGenericLayer):
  """DeepSeek-style dense layer with Multi-Head Latent Attention."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)

    # Dense MLP Block uses emb_dim as input and output does not go through the
    # bottleneck.
    # Input Shape:  [Batch, SeqLen, emb_dim] (e.g., [B, S, 7168])
    # Output Shape: [Batch, SeqLen, emb_dim] (e.g., [B, S, 7168])
    mlp_in_features = self.dummy_inputs_shape[-1]

    max_logging.log(f"  [Layer {layer_idx} - Dense] Feature Sizes -> "
                    f"MLP In: {mlp_in_features}, "
                    f"MLP Dim: {self.config.mlp_dim}, ")

    self.mlp = linears.MlpBlock(
        in_features=mlp_in_features,
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=quant,
        model_mode=model_mode,
        mesh=mesh,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic):
    mlp = self.mlp(x, deterministic, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding)
    return self.with_logical_constraint(mlp)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    # 1. Attention: Takes [B, S, base_emb_dim (e.g. 7168)] tokens and outputs
    # [B, S, attention_output_dim (e.g. 3072)] tokens directly.
    attn_out = self.attention_op(
        self.pre_attention_norm_op(x),
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    # 2. MLP processing: Takes [B, S, attention_output_dim (e.g. 3072)] tokens directly
    #    skipping any intermediate residual and feeds into the MLP block.
    #    Outputs [B, S, attention_output_dim] tokens.
    mlp_lnx = self.mlp_op(attn_out, deterministic)
    layer_output = mlp_lnx

    # 3. Final Projection: Maps the [B, S, attention_output_dim]
    #    combined output back to [B, S, base_emb_dim (e.g. 7168)].
    #    This corresponds to the transition from the latent/compressed space back to the main model dimension.
    if self.layer_up_projection is not None:
      layer_output = self.layer_up_projection(layer_output)
      layer_output = self.with_logical_constraint(layer_output)

    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    # 4. Single residual connection for the whole layer
    x = inputs + layer_output
    hidden_states = self.post_attention_norm_op(x)

    return self.post_process(hidden_states, None, None, kv_cache)


DeepSeekDenseLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekMoELayer(DeepSeekGenericLayer):
  """DeepSeek-style MoE layer with Multi-Head Latent Attention.

  Supports dropless and dropping base on configs. Uses a bias in routing instead
  of load balancing loss.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
    if config.attention_output_dim <= 0 or config.attention_output_dim != config.moe_model_dim:
      raise ValueError("attention_output_dim must be positive and equal to moe_model_dim for DeepSeekMoELayer.")

    max_logging.log(f"  [Layer {layer_idx} - MoE] Feature Sizes -> "
                    f"Emb Dim: {self.dummy_inputs_shape[-1]}, "
                    f"Attn Out: {config.attention_output_dim}, "
                    f"MoE In: {config.moe_model_dim}")

    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    # =========================================================================
    # 1. ATTENTION (Down-Projection)
    # Input Shape:  [Batch, SeqLen, emb_dim] (e.g., [B, S, 7168])
    # Output Shape: [Batch, SeqLen, moe_model_dim] (e.g., [B, S, 3072])
    # The attention block implicitly acts as our down-projection bottleneck.
    # =========================================================================
    attn_out = self.attention_op(
        self.pre_attention_norm_op(x),
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    # =========================================================================
    # 2. MIXTURE OF EXPERTS (A2A Communication & Computation)
    # Input Shape:  [Batch, SeqLen, moe_model_dim] (e.g., [B, S, 3072])
    #
    # Routing Flow inside `self.mlp_op`:
    #   a. Token Routing (A2A Dispatch): Tokens are routed to experts across devices.
    #      Because we pass `attn_out` directly without restoring to `emb_dim`,
    #      the A2A payload is only moe_model_dim per token, saving massive bandwidth.
    #   b. Expert Compute: [B, S, moe_model_dim] -> [B, S, expert_hidden] -> [B, S, moe_model_dim]
    #   c. Token Return (A2A Combine): Processed tokens are returned to their
    #      original devices. The payload over the network remains moe_model_dim.
    #
    # Output Shape: [Batch, SeqLen, moe_model_dim] (e.g., [B, S, 3072])
    # =========================================================================

    mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp_op(attn_out, deterministic)
    layer_output = mlp_lnx

    # =========================================================================
    # 3. FINAL UP-PROJECTION
    # Input Shape:  [Batch, SeqLen, moe_model_dim] (e.g., [B, S, 3072])
    # Output Shape: [Batch, SeqLen, emb_dim] (e.g., [B, S, 7168])
    # Restores the token dimension before the outer residual connection.
    # =========================================================================
    if self.layer_up_projection is not None:
      layer_output = self.layer_up_projection(layer_output)
      layer_output = self.with_logical_constraint(layer_output)

    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    # =========================================================================
    # 4. RESIDUAL CONNECTION
    # [Batch, SeqLen, 7168] + [Batch, SeqLen, 7168]
    # =========================================================================
    x = inputs + layer_output
    hidden_states = self.post_attention_norm_op(x)

    return self.post_process(hidden_states, load_balance_loss, moe_bias_updates, kv_cache)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    mlp_lnx, load_balance_loss, moe_bias_updates = self.DeepSeekMoeBlock_0(
        x, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding
    )
    return self.with_logical_constraint(mlp_lnx), load_balance_loss, moe_bias_updates


DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekMoEScannableBlock(nnx.Module):
  """A repeatable block of DeepSeek Custom MoE layers."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs

    for i in range(self.config.inhomogeneous_layer_cycle_interval):
      layer_idx = self.config.first_num_dense_layers + i
      layer_name = f"layers_{i}"
      layer = DeepSeekMoELayer(
          config=config,
          model_mode=model_mode,
          mesh=mesh,
          rngs=rngs,
          quant=quant,
          layer_idx=layer_idx,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    cfg = self.config
    y = inputs
    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layers_{i}")
      y = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          decoder_input_tokens=decoder_input_tokens,
      )
      if cfg.scan_layers:
        y = y[0]
    if cfg.scan_layers:
      return y, None
    else:
      return y


DeepSeekMoEScannableBlockToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoEScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
