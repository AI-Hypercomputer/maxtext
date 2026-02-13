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

"""Llama4 decoder layer definition."""
# pylint: disable=arguments-differ, disable=no-name-in-module, missing-function-docstring

import math

from flax import linen as nn
from flax import nnx
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText.common_types import Array, AttentionType, Config, MODEL_MODE_TRAIN
from MaxText.common_types import MODEL_MODE_PREFILL
from maxtext.inference import page_manager
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import Dropout
from maxtext.layers.linears import MlpBlock
from maxtext.layers.moe import RoutedAndSharedMoE
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils

#### Multi modal model implementation


class Llama4UnfoldConvolution(nnx.Module):
  """implementation of Llama4UnfoldConvolution for Llama4 Multi modal model.

  This module extracts patches from input images and projects them to hidden dimension.

  Attributes:
    config: Config containing model parameters
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    self.config = config
    self.rngs = rngs
    self.vit_unfold_linear = linears.DenseGeneral(
        in_features_shape=(
            self.config.num_channels_for_vit * self.config.patch_size_for_vit * self.config.patch_size_for_vit
        ),
        out_features_shape=self.config.hidden_size_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=False,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, inputs: Array) -> Array:
    batch_size, num_channels, img, _ = inputs.shape
    num_patches = (img // self.config.patch_size_for_vit) ** 2

    patches = lax.conv_general_dilated_patches(
        inputs,
        filter_shape=[self.config.patch_size_for_vit, self.config.patch_size_for_vit],
        window_strides=[self.config.patch_size_for_vit, self.config.patch_size_for_vit],
        padding="VALID",
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
        precision=lax.Precision(self.config.matmul_precision),
        preferred_element_type=self.config.dtype_mm,
    )

    patches = patches.reshape(
        batch_size, num_channels * self.config.patch_size_for_vit * self.config.patch_size_for_vit, num_patches
    )
    patches = patches.transpose(0, 2, 1)

    hidden_states = self.vit_unfold_linear(patches)

    return hidden_states


def pixel_shuffle(input_tensor: Array, shuffle_ratio: float) -> Array:
  """Apply pixel shuffle operation to the input tensor."""
  batch_size, num_patches, channels = input_tensor.shape
  patch_size = int(math.sqrt(num_patches))

  # Reshape to [batch_size, patch_size, patch_size, channels]
  input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
  batch_size, height, width, channels = input_tensor.shape

  # Reshape to [batch_size, height, width * shuffle_ratio, channels / shuffle_ratio]
  reshaped_tensor = input_tensor.reshape(batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio))

  # Transpose to [batch_size, width * shuffle_ratio, height, channels / shuffle_ratio]
  reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

  # Reshape to [batch_size, height * shuffle_ratio, width * shuffle_ratio, channels / (shuffle_ratio^2)]
  reshaped_tensor = reshaped_tensor.reshape(
      batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio), int(channels / (shuffle_ratio**2))
  )

  # Transpose to [batch_size, width * shuffle_ratio, height * shuffle_ratio, channels / (shuffle_ratio^2)]
  reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

  # Reshape back to [batch_size, num_patches, channels]
  output_tensor = reshaped_tensor.reshape(batch_size, -1, reshaped_tensor.shape[-1])
  return output_tensor


class Llama4VisionMLP(nnx.Module):
  """MLP block for Llama4EncoderLayer.

  Attributes:
    config: Config containing model parameters
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    self.config = config
    self.rngs = rngs
    self.vit_encoder_layer_mlp_fc1 = linears.DenseGeneral(
        in_features_shape=self.config.hidden_size_for_vit,
        out_features_shape=self.config.intermediate_size_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=True,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )
    self.vit_encoder_layer_mlp_fc2 = linears.DenseGeneral(
        in_features_shape=self.config.intermediate_size_for_vit,
        out_features_shape=self.config.hidden_size_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=True,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    hidden_states = self.vit_encoder_layer_mlp_fc1(hidden_states)
    hidden_states = nnx.gelu(hidden_states, approximate=False)
    hidden_states = self.vit_encoder_layer_mlp_fc2(hidden_states)
    return hidden_states


class Llama4VisionMLP2(nnx.Module):
  """MLP block for Llama4VisionPixelShuffleMLP.

  Attributes:
    config: Config containing model parameters
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    self.config = config
    self.rngs = rngs
    self.vit_pixel_shuffle_mlp_fc1 = linears.DenseGeneral(
        in_features_shape=self.config.intermediate_size_for_vit,
        out_features_shape=self.config.projector_input_dim_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=False,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )
    self.vit_pixel_shuffle_mlp_fc2 = linears.DenseGeneral(
        in_features_shape=self.config.projector_input_dim_for_vit,
        out_features_shape=self.config.projector_output_dim_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=False,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )
    self.dropout = linears.Dropout(rate=self.config.projector_dropout_for_vit, rngs=self.rngs)

  def __call__(self, hidden_states: Array, deterministic: bool = False) -> Array:
    hidden_states = self.vit_pixel_shuffle_mlp_fc1(hidden_states)
    hidden_states = nnx.gelu(hidden_states, approximate=False)
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    hidden_states = self.vit_pixel_shuffle_mlp_fc2(hidden_states)
    hidden_states = nnx.gelu(hidden_states, approximate=False)
    return hidden_states


class Llama4VisionPixelShuffleMLP(nnx.Module):
  """Implementation of Llama4VisionPixelShuffleMLP for Llama4 Multi modal model.

  This module applies pixel shuffle operation and MLP to encoded patches.

  Attributes:
    config: Config containing model parameters
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    self.config = config
    self.rngs = rngs
    self.pixel_shuffle_ratio = self.config.pixel_shuffle_ratio_for_vit
    self.pixel_shuffle_mlp = Llama4VisionMLP2(config=config, rngs=self.rngs)

  def __call__(self, encoded_patches: Array, deterministic: bool = False) -> Array:
    # Apply pixel shuffle operation
    encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)

    # Apply MLP transformation
    result = self.pixel_shuffle_mlp(encoded_patches, deterministic=deterministic)

    return result


class Llama4MultiModalProjector(nnx.Module):
  """Implementation of Llama4MultiModalProjector for Llama4 Multi modal model.

  This module projects vision features to text hidden dimension.

  Attributes:
    config: Config containing model parameters
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.vit_multi_modal_projector = linears.DenseGeneral(
        in_features_shape=self.config.vision_output_dim_for_vit,
        out_features_shape=self.config.base_emb_dim,
        dtype=self.config.dtype_mm,
        use_bias=False,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

  def __call__(self, image_features: Array) -> Array:
    """Project image features to text hidden dimension.

    Args:
      image_features: Input tensor of shape [batch_size, num_patches, (pixel_shuffle_ratio**2), vision_output_dim]

    Returns:
      Tensor of shape [batch_size, num_patches, (pixel_shuffle_ratio**2), vision_hidden_size]
    """
    b, t, c, d = image_features.shape

    # Reshape image_features to [b * t, c, d] and project to text hidden dimension
    image_features = image_features.reshape(b * t, c, d)
    hidden_states = self.vit_multi_modal_projector(image_features)
    _, c, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, t, c, d)
    return hidden_states


def llama4multimodalprojector_as_linen(config: Config, mesh: Mesh):
  return nnx_wrappers.to_linen(
      Llama4MultiModalProjector,
      config=config,
      mesh=mesh,
      name="Llama4MultiModalProjector_0",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )


def determine_is_nope_layer(layer_id: int, nope_layer_interval: int) -> bool:
  """
  Determines whether the given layer at `layer_id` should use RoPE or not (NoPE).

  Args:
    layer_id: The index of the layer.
    nope_layer_interval: The interval at which layers should use NoPE.

  Returns:
    True if the layer should use NoPE, False otherwise.
  """
  return nope_layer_interval is not None and nope_layer_interval > 0 and (layer_id + 1) % nope_layer_interval == 0


def determine_is_moe_layer(layer_id: int, interleave_moe_layer_step: int) -> bool:
  """
  Determines whether the given layer at `layer_id` is MoE layer.

  This function implements a striding pattern. For example:
  - If moe_layer_stride is 1, all layers are MoE layers.
  - If moe_layer_stride is 2, layers with index 1, 3, 5, ... are MoE layers.

  Args:
    layer_id: The 0-based index of the layer being checked.
    interleave_moe_layer_step: The interval or stride for placing MoE layers.

  Returns:
    True if the layer is MoE layer, False otherwise.
  """
  return (
      interleave_moe_layer_step is not None
      and interleave_moe_layer_step > 0
      and (layer_id + 1) % interleave_moe_layer_step == 0
  )


# -----------------------------------------
# The Decoder Layer specific for LLama4
# -----------------------------------------


class Llama4DecoderLayer(nnx.Module):
  """Transformer decoder layer for Llama4."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      is_nope_layer: bool = False,
      is_moe_layer: bool = False,
  ):
    """Initializes the Llama4 decoder layer.

    Args:
      config: The main model configuration object.
      mesh: The device mesh used for sharding parameters and activations.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: An `nnx.Rngs` object to provide random numbers.
      quant: An optional configuration for quantization. Defaults to None.
      is_nope_layer: If True, this layer will be configured as No Position Embeddings layer. Defaults to False.
      is_moe_layer: If True, this layer will use a MoE block. Defaults to False as Dense.
    """

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.is_nope_layer = is_nope_layer
    self.is_moe_layer = is_moe_layer

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Instead of scaling the query values in the checkpoint conversion (`llama_or_mistral_ckpt`)
    # we'll do it dynamically in the forward pass of Attention
    query_pre_attn_scalar = config.head_dim**-0.5
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
        is_nope_layer=self.is_nope_layer,
        use_qk_norm=config.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        temperature_tuning=config.temperature_tuning,
        temperature_tuning_scale=0.1,
        temperature_tuning_floor_scale=8192.0,
        # note: chunk_attn_window_size is set in the config
        attention_type=AttentionType.GLOBAL if self.is_nope_layer else AttentionType.CHUNK,
        model_mode=model_mode,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    if self.is_moe_layer:
      # NOTE: the name Llama4MoEBlock_0 is to ensure reverse compatibility with
      # existing checkpoints for MoE block.
      self.Llama4MoEBlock_0 = RoutedAndSharedMoE(
          config=config,
          mesh=self.mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=self.quant,
          rngs=self.rngs,
      )
    else:
      self.mlp = MlpBlock(
          mesh=self.mesh,
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          config=config,
          quant=self.quant,
          model_mode=model_mode,
          rngs=self.rngs,
      )

    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

  @property
  def moe_block(self):
    return self.Llama4MoEBlock_0

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
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config
    assert cfg.num_experts >= 1, "Expected the Llama4 config to have `num_experts > 1`."

    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Self-attention block
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        slot=slot,
        page_state=page_state,
        previous_chunk=previous_chunk,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    load_balance_loss = None
    if self.is_moe_layer:
      mlp_lnx, load_balance_loss, _ = self.moe_block(hidden_states)
    else:
      mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
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


Llama4DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Llama4DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Llama4ScannableBlock(nnx.Module):
  """A repeatable block given nope_layer_interval and interleave_moe_layer_step."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      nope_layer_interval: int = 1,
      interleave_moe_layer_step: int = 1,
  ):
    """Initializes the scannable block.

    Args:
      config: The main model configuration object.
      mesh: The device mesh used for sharding parameters and activations.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: An `nnx.Rngs` object to provide random numbers for initialization.
      quant: An optional configuration for quantization. Defaults to None.
      nope_layer_interval: Specifies the interval for inserting a NoPE layer.
      interleave_moe_layer_step: Specifies the interval for inserting a MoE layer.
    """
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.nope_layer_interval = nope_layer_interval
    self.interleave_moe_layer_step = interleave_moe_layer_step

    for layer_id in range(self.config.inhomogeneous_layer_cycle_interval):
      nope_layer = determine_is_nope_layer(layer_id, self.nope_layer_interval)
      moe_layer = determine_is_moe_layer(layer_id, self.interleave_moe_layer_step)
      layer_name = f"layers_{layer_id}"
      layer = Llama4DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          is_nope_layer=nope_layer,
          is_moe_layer=moe_layer,
      )
      setattr(self, layer_name, layer)

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

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs
    for layer_id in range(cfg.inhomogeneous_layer_cycle_interval):
      y = getattr(self, f"layers_{layer_id}")(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
      )
      if cfg.scan_layers:
        y = y[0]
    if cfg.scan_layers:
      return y, None
    else:
      return y


Llama4ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Llama4ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Llama4VisionEncoderLayer(nnx.Module):
  """Transformer encoder layer for Llama4 vision model."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.hidden_states_shape = (
        self.config.per_device_batch_size,
        (self.config.image_size_for_vit // self.config.patch_size_for_vit) ** 2 + 1,
        self.config.hidden_size_for_vit,
    )

    self.input_layer_norm = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit, epsilon=self.config.normalization_layer_epsilon, rngs=self.rngs
    )
    self.self_attention_vision = Attention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit,
        max_target_length=(self.config.image_size_for_vit // self.config.patch_size_for_vit) ** 2 + 1,
        attention_kernel="dot_product",
        inputs_q_shape=self.hidden_states_shape,
        inputs_kv_shape=self.hidden_states_shape,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=self.mesh,
        dropout_rate=0,
        name="self_attention_vision",
        attention_type=AttentionType.FULL,
        is_nope_layer=False,
        use_bias_in_projections=True,
        is_vision=True,
        use_qk_norm=False,
        query_pre_attn_scalar=1 / math.sqrt(self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit),
        # The vision encoder processes an image in a single forward pass to produce
        # embeddings. It doesn't have the concept of "prefill" and "autoregressive"
        # steps that a text decoder has. Therefore, it doesn't need a KV cache for
        # its self-attention mechanism.
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )
    self.post_attention_layer_norm = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit, epsilon=self.config.normalization_layer_epsilon, rngs=self.rngs
    )
    self.Llama4VisionMLP_0 = Llama4VisionMLP(config=self.config, rngs=self.rngs)

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = False,
  ):
    residual = hidden_states
    hidden_states = self.input_layer_norm(hidden_states)
    hidden_states, _ = self.self_attention_vision(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layer_norm(hidden_states)
    hidden_states = self.Llama4VisionMLP_0(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class Llama4VisionEncoder(nnx.Module):
  """Transformer encoder consisting of multiple Llama4VisionEncoderLayer layers.

  This encoder is based on the PyTorch reference implementation and uses multiple
  encoder layers to process vision input.

  Attributes:
    config: Config containing model parameters
    mesh: Mesh, JAX device mesh (used for sharding)
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    for lyr in range(self.config.num_hidden_layers_for_vit):
      layer_name = f"layers_{lyr}"
      layer = Llama4VisionEncoderLayer(
          config=self.config,
          mesh=self.mesh,
          rngs=self.rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(self, hidden_states: Array, deterministic: bool = False):
    for lyr in range(self.config.num_hidden_layers_for_vit):
      layer_name = f"layers_{lyr}"
      layer = getattr(self, layer_name)
      hidden_states = layer(hidden_states, deterministic=deterministic)
    return hidden_states


class Llama4VisionModel(nnx.Module):
  """Llama4 vision model for processing image inputs.

  This model extracts patches from input image tiles and processes them
  through Llama4VisionEncoder and other vision-specific layers.

  Attributes:
    config: Config containing model parameters
    mesh: Mesh, JAX device mesh (used for sharding)
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.scale = self.config.hidden_size_for_vit**-0.5
    self.num_patches = (self.config.tile_size_for_vit // self.config.patch_size_for_vit) ** 2 + 1
    self.initializer = nnx.initializers.normal(self.scale)

    self.class_embedding = nnx.Param(
        self.initializer(self.rngs.params(), (self.config.hidden_size_for_vit,), self.config.dtype_mm)
    )
    self.positional_embedding_vlm = nnx.Param(
        self.initializer(self.rngs.params(), (self.num_patches, self.config.hidden_size_for_vit), self.config.dtype_mm)
    )
    self.layernorm_pre = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit,
        epsilon=self.config.normalization_layer_epsilon,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )
    self.layernorm_post = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit,
        epsilon=self.config.normalization_layer_epsilon,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.Llama4UnfoldConvolution_0 = Llama4UnfoldConvolution(config=self.config, rngs=self.rngs)
    self.Llama4VisionEncoder_0 = Llama4VisionEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs)
    self.Llama4VisionPixelShuffleMLP_0 = Llama4VisionPixelShuffleMLP(config=self.config, rngs=self.rngs)

  def __call__(
      self,
      pixel_values: Array,
      output_attentions: None | bool = None,
      output_hidden_states: None | bool = None,
      return_dict: None | bool = None,
      deterministic: None | bool = False,
  ) -> Array:
    """Forward pass of the Llama4 vision model.

    Args:
      inputs: Input tensor of shape:
              [batch_size * num_images, num_tiles, num_channels_for_vit, tile_size_for_vit, tile_size_for_vit]
      deterministic: Whether to use deterministic mode (disables dropout)

    Returns:
      Final hidden states from the vision encoder of shape:
      [batch_size * num_images, num_tiles, num_patches, vision_output_dim_for_vit]
    """
    # Reshape pixel values to combine batch and num_tiles dimensions
    b, t, c, h, w = pixel_values.shape
    pixel_values = jnp.reshape(pixel_values, [b * t, c, h, w])

    hidden_states = self.Llama4UnfoldConvolution_0(pixel_values)

    # Add class embedding to the beginning of the sequence
    class_embedding_expanded = jnp.expand_dims(jnp.expand_dims(self.class_embedding, axis=0), axis=0)
    class_embedding = jnp.broadcast_to(
        class_embedding_expanded, (hidden_states.shape[0], 1, self.config.hidden_size_for_vit)
    )
    hidden_states = jnp.concatenate([hidden_states, class_embedding], axis=1)

    # Add positional embedding
    hidden_states += self.positional_embedding_vlm

    # Transformation layers
    hidden_states = self.layernorm_pre(hidden_states)
    hidden_states = self.Llama4VisionEncoder_0(hidden_states)
    hidden_states = self.layernorm_post(hidden_states)
    hidden_states = hidden_states[:, :-1, :]

    hidden_states = self.Llama4VisionPixelShuffleMLP_0(hidden_states)

    # Reshape hidden states
    _, patch_num, patch_dim = hidden_states.shape
    hidden_states = jnp.reshape(hidden_states, [b, t, patch_num, patch_dim])

    return hidden_states


def llama4visionmodel_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  return nnx_wrappers.to_linen(
      Llama4VisionModel,
      config=config,
      mesh=mesh,
      name="Llama4VisionModel_0",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
