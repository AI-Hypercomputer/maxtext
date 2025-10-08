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

"""Llama4 decoder layer definition."""
# pylint: disable=arguments-differ, disable=no-name-in-module, missing-function-docstring

import math

import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx

from MaxText.common_types import Config, Array, MODEL_MODE_TRAIN, AttentionType
from MaxText import max_utils
from MaxText.inference import page_manager
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers
from MaxText.layers import linears
from MaxText.layers import quantizations
from MaxText.layers import attentions
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import MlpBlock
from MaxText.layers.linears import Dropout
from MaxText.layers.moe import RoutedAndSharedMoE
from MaxText.common_types import MODEL_MODE_PREFILL


#### Multi modal model implementation


class Llama4UnfoldConvolution(nn.Module):
  """implementation of Llama4UnfoldConvolution for Llama4 Multi modal model.

  This module extracts patches from input images and projects them to hidden dimension.

  Attributes:
    config: Config containing model parameters
  """

  config: Config

  def setup(self):
    """
    Initialize Llama4UnfoldConvolution
    """
    cfg = self.config
    # Linear projection layer using dense_general.
    # patches sent to dense_general with shape:
    # [batch_size, num_patches, num_channels * patch_size * patch_size]
    self.linear = linears.dense_general(
        in_features_shape=(cfg.num_channels_for_vit * cfg.patch_size_for_vit * cfg.patch_size_for_vit),
        out_features_shape=cfg.hidden_size_for_vit,
        dtype=cfg.dtype_mm,
        name="vit_unfold_linear",
        use_bias=False,
        matmul_precision=cfg.matmul_precision,
    )

  def __call__(self, inputs: Array) -> Array:
    """Extract patches and project to hidden dimension.

    Args:
      inputs: Input tensor of shape [batch_size, channels, img, img]

    Returns:
      Tensor of shape [batch_size, num_patches*num_patches, hidden_size]
    """
    cfg = self.config
    # Extract patches using conv_general_dilated_patches
    batch_size, num_channels, img, _ = inputs.shape
    num_patches = (img // cfg.patch_size_for_vit) ** 2

    # Extract patches using conv_general_dilated_patches
    patches = lax.conv_general_dilated_patches(
        inputs,
        filter_shape=[cfg.patch_size_for_vit, cfg.patch_size_for_vit],
        window_strides=[cfg.patch_size_for_vit, cfg.patch_size_for_vit],
        padding="VALID",
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )

    # reshape patches to [batch_size, num_patches, num_channels * patch_size * patch_size]
    patches = patches.reshape(batch_size, num_channels * cfg.patch_size_for_vit * cfg.patch_size_for_vit, num_patches)
    # After transpose, patches shape:
    # [batch_size, num_patches, num_channels * patch_size * patch_size]
    patches = patches.transpose(0, 2, 1)

    # Project patches to hidden dimension using dense_general
    hidden_states = self.linear(patches)

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


class Llama4VisionMLP(nn.Module):
  """MLP block for Llama4EncoderLayer.

  Attributes:
    config: Config containing model parameters
  """

  config: Config

  def setup(self):
    cfg = self.config
    self.fc1 = linears.dense_general(
        in_features_shape=cfg.hidden_size_for_vit,
        out_features_shape=cfg.intermediate_size_for_vit,
        dtype=cfg.dtype_mm,
        name="vit_encoder_layer_mlp_fc1",
        use_bias=True,
        matmul_precision=cfg.matmul_precision,
    )
    self.fc2 = linears.dense_general(
        in_features_shape=cfg.intermediate_size_for_vit,
        out_features_shape=cfg.hidden_size_for_vit,
        dtype=cfg.dtype_mm,
        name="vit_encoder_layer_mlp_fc2",
        use_bias=True,
        matmul_precision=cfg.matmul_precision,
    )

  def __call__(self, hidden_states: Array) -> Array:
    """Apply MLP transformation to hidden states.

    Args:
      hidden_states: Input tensor
      deterministic: If True, disables dropout during inference
    """
    hidden_states = self.fc1(hidden_states)
    hidden_states = nn.gelu(hidden_states, approximate=False)

    hidden_states = self.fc2(hidden_states)

    return hidden_states


class Llama4VisionMLP2(nn.Module):
  """MLP block for Llama4VisionPixelShuffleMLP.

  Attributes:
    config: Config containing model parameters
  """

  config: Config

  def setup(self):
    """
    Initialize Llama4VisionMLP2
    """
    cfg = self.config
    self.fc1 = linears.dense_general(
        in_features_shape=cfg.intermediate_size_for_vit,
        out_features_shape=cfg.projector_input_dim_for_vit,
        dtype=cfg.dtype_mm,
        name="vit_pixel_shuffle_mlp_fc1",
        use_bias=False,
        matmul_precision=cfg.matmul_precision,
    )
    self.fc2 = linears.dense_general(
        in_features_shape=cfg.projector_input_dim_for_vit,
        out_features_shape=cfg.projector_output_dim_for_vit,
        dtype=cfg.dtype_mm,
        name="vit_pixel_shuffle_mlp_fc2",
        use_bias=False,
        matmul_precision=cfg.matmul_precision,
    )
    self.dropout = nn.Dropout(rate=cfg.projector_dropout_for_vit)

  def __call__(self, hidden_states: Array, deterministic: bool = False) -> Array:
    """Apply MLP transformation to hidden states.

    Args:
      hidden_states: Input tensor
      deterministic: If True, disables dropout during inference
    """
    # First linear layer with GELU activation
    hidden_states = self.fc1(hidden_states)
    hidden_states = nn.gelu(hidden_states, approximate=False)

    # Apply dropout
    # in pytorch it's using default Dropout Rate of 0.5
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)

    # Second linear layer with GELU activation
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.gelu(hidden_states, approximate=False)

    return hidden_states


class Llama4VisionPixelShuffleMLP(nn.Module):
  """Implementation of Llama4VisionPixelShuffleMLP for Llama4 Multi modal model.

  This module applies pixel shuffle operation and MLP to encoded patches.

  Attributes:
    config: Config containing model parameters
  """

  config: Config

  def setup(self):
    cfg = self.config
    self.pixel_shuffle_ratio = cfg.pixel_shuffle_ratio_for_vit
    self.pixel_shuffle_mlp = Llama4VisionMLP2(cfg)

  def __call__(self, encoded_patches: Array, deterministic: bool = False) -> Array:
    """Apply pixel shuffle and MLP to encoded patches.

    Args:
      encoded_patches: Input tensor of shape [batch_size, num_patches, hidden_size]
      deterministic: If True, disables dropout during inference

    Returns:
      Tensor of shape [batch_size, num_patches, hidden_size]
    """
    # Apply pixel shuffle operation
    encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)

    # Apply MLP transformation
    result = self.pixel_shuffle_mlp(encoded_patches, deterministic=deterministic)

    return result


class Llama4MultiModalProjector(nn.Module):
  """Implementation of Llama4MultiModalProjector for Llama4 Multi modal model.

  This module projects vision features to text hidden dimension.

  Attributes:
    config: Config containing model parameters
  """

  config: Config
  mesh: Mesh

  def setup(self):
    cfg = self.config
    self.linear = linears.dense_general(
        in_features_shape=cfg.vision_output_dim_for_vit,
        out_features_shape=cfg.base_emb_dim,
        dtype=cfg.dtype_mm,
        name="vit_multi_modal_projector",
        use_bias=False,
        matmul_precision=cfg.matmul_precision,
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
    hidden_states = self.linear(image_features)

    # Reshape hidden states back to [b, t, c, d]
    _, c, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, t, c, d)
    return hidden_states


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
      bidirectional_mask=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      previous_chunk=None,
  ):
    cfg = self.config
    assert cfg.num_experts >= 1, "Expected the Llama4 config to have `num_experts > 1`."

    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Self-attention block
    attention_lnx = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        slot=slot,
        page_state=page_state,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    if self.is_moe_layer:
      mlp_lnx = self.moe_block(hidden_states)
    else:
      mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

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
      bidirectional_mask=None,
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
          bidirectional_mask=bidirectional_mask,
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


class Llama4VisionEncoderLayer(nn.Module):
  """Transformer encoder layer for Llama4 vision model."""

  config: Config
  mesh: Mesh

  @nn.compact
  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = False,
  ) -> Array:
    """Forward pass of the vision encoder layer.

    Args:
      hidden_states: Input hidden states
      deterministic: Whether to use deterministic mode

    Returns:
      Output hidden states
    """
    # Self Attention
    residual = hidden_states

    # Input layer norm
    hidden_states = nn.LayerNorm(name="input_layer_norm", epsilon=1e-5)(hidden_states)

    # Self attention
    attention_layer = attentions.attention_as_linen(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit,
        max_target_length=(self.config.image_size_for_vit // self.config.patch_size_for_vit) ** 2 + 1,
        attention_kernel="dot_product",
        inputs_q_shape=hidden_states.shape,
        inputs_kv_shape=hidden_states.shape,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
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
    )

    hidden_states = attention_layer(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
    )

    hidden_states = residual + hidden_states

    residual = hidden_states

    # Post attention layer norm
    hidden_states = nn.LayerNorm(name="post_attention_layer_norm", epsilon=1e-5)(hidden_states)

    # MLP
    mlp = Llama4VisionMLP(self.config)
    hidden_states = mlp(hidden_states)

    hidden_states = residual + hidden_states

    return hidden_states


class Llama4VisionEncoder(nn.Module):
  """Transformer encoder consisting of multiple Llama4VisionEncoderLayer layers.

  This encoder is based on the PyTorch reference implementation and uses multiple
  encoder layers to process vision input.

  Attributes:
    config: Config containing model parameters
    mesh: Mesh, JAX device mesh (used for sharding)
  """

  config: Config
  mesh: Mesh

  @nn.compact
  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = False,
  ) -> Array:
    """Forward pass of the vision encoder.

    Args:
      hidden_states: Input hidden states
      deterministic: Whether to use deterministic mode (disables dropout)

    Returns:
      Final hidden states
    """
    cfg = self.config

    # Iterate through encoder layers (non-scan version)
    for layer_idx in range(cfg.num_hidden_layers_for_vit):

      # TODO： add scan version
      layer = Llama4VisionEncoderLayer(config=cfg, mesh=self.mesh, name=f"layers_{layer_idx}")

      hidden_states = layer(
          hidden_states=hidden_states,
          deterministic=deterministic,
      )

    return hidden_states


class Llama4VisionModel(nn.Module):
  """Llama4 vision model for processing image inputs.

  This model extracts patches from input image tiles and processes them
  through Llama4VisionEncoder and other vision-specific layers.

  Attributes:
    config: Config containing model parameters
    mesh: Mesh, JAX device mesh (used for sharding)
  """

  config: Config
  mesh: Mesh

  def setup(self):
    self.scale = self.config.hidden_size_for_vit**-0.5
    self.num_patches = (self.config.tile_size_for_vit // self.config.patch_size_for_vit) ** 2 + 1
    self.class_embedding = self.param(
        "class_embedding",
        nn.initializers.normal(stddev=self.scale, dtype=self.config.dtype_mm),
        (self.config.hidden_size_for_vit,),
    )
    self.positional_embedding_vlm = self.param(
        "positional_embedding_vlm",
        nn.initializers.normal(stddev=self.scale, dtype=self.config.dtype_mm),
        (self.num_patches, self.config.hidden_size_for_vit),
    )

  @nn.compact
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
    cfg = self.config
    mesh = self.mesh

    # Reshape pixel values to combine batch and num_tiles dimensions
    b, t, c, h, w = pixel_values.shape
    pixel_values = jnp.reshape(pixel_values, [b * t, c, h, w])

    # Unfold convolution to extract patches
    hidden_states = Llama4UnfoldConvolution(config=cfg)(pixel_values)

    # Add class embedding to the beginning of the sequence
    class_embedding_expanded = jnp.expand_dims(jnp.expand_dims(self.class_embedding, axis=0), axis=0)
    class_embedding = jnp.broadcast_to(class_embedding_expanded, (hidden_states.shape[0], 1, cfg.hidden_size_for_vit))
    hidden_states = jnp.concatenate([class_embedding, hidden_states], axis=1)

    # Add positional embedding
    hidden_states += self.positional_embedding_vlm

    # Transformation layers
    hidden_states = nn.LayerNorm(name="layernorm_pre")(hidden_states)
    hidden_states = Llama4VisionEncoder(config=cfg, mesh=mesh)(hidden_states)
    hidden_states = nn.LayerNorm(name="layernorm_post")(hidden_states)
    hidden_states = hidden_states[:, :-1, :]

    hidden_states = Llama4VisionPixelShuffleMLP(config=cfg)(hidden_states)

    # Reshape hidden states
    _, patch_num, patch_dim = hidden_states.shape
    hidden_states = jnp.reshape(hidden_states, [b, t, patch_num, patch_dim])

    return hidden_states
