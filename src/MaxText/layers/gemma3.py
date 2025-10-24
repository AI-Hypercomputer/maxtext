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

"""Specialised layers for Gemma 3."""

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.common_types import Config, AttentionType, MODEL_MODE_PREFILL
from MaxText import max_utils
from MaxText.layers import quantizations
from MaxText.layers import nnx_wrappers
from MaxText.layers import initializers
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import DenseGeneral, MlpBlock, Dropout
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.initializers import variable_to_logically_partitioned


GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  layer_id %= len(GEMMA3_ATTENTION_PATTERN)
  return GEMMA3_ATTENTION_PATTERN[layer_id]


def get_query_pre_attn_scalar(config) -> float:
  """Returns the scalar to multiply the query by before attention."""
  if config.model_name in ["gemma3-4b", "gemma3-12b"]:
    return config.head_dim**-0.5
  elif config.model_name == "gemma3-27b":
    return (config.base_emb_dim // config.base_num_query_heads) ** -0.5
  else:
    raise ValueError(f"Unsupported model name: {config.model_name}")


# Gemma3 Decoder Layer
class Gemma3DecoderLayer(nnx.Module):
  """Transformer decoder layer for Gemma3."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
  ):
    """Initializes the Gemma3DecoderLayer.

    Args:
      config: The Config object with model hyperparameters.
      mesh: The device mesh for distributed training.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: The random number generators for initialization.
      quant: The quantization configuration.
      attention_type: The type of attention to use.
    """

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.attention_type = attention_type

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    query_pre_attn_scalar = get_query_pre_attn_scalar(config)
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
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        use_qk_norm=True,  # Gemma 3 models use query, key normalizations
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    if self.config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

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

    if self.config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

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
      bidirectional_mask=None,
  ):
    cfg = self.config
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Self-attention block
    attention_lnx = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
    )
    if cfg.use_post_attn_norm:
      attention_lnx = self.post_self_attention_norm(attention_lnx)
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)

    attention_lnx += inputs
    residual = attention_lnx

    attn_output = self.pre_ffw_norm(attention_lnx)

    # MLP block.
    mlp_lnx = self.mlp(attn_output, deterministic=deterministic)
    if cfg.use_post_ffw_norm:
      mlp_lnx = self.post_ffw_norm(mlp_lnx)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    next_layer_addition = mlp_lnx + residual
    next_layer_addition_dropped_out = self.dropout(next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out
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


Gemma3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Gemma3ScannableBlock(nnx.Module):
  """A repeatable block of Gemma3 decoder layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      num_of_layers: int = 1,
  ):
    """Initializes the Gemma3ScannableBlock.

    Args:
      config: The Config object with model hyperparameters.
      mesh: The device mesh for distributed training.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: The random number generators for initialization.
      quant: The quantization configuration.
      num_of_layers: The number of layers in the model.
    """
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.num_of_layers = num_of_layers

    for layer_id in range(self.num_of_layers):
      attention_type = get_attention_type(layer_id)
      layer_name = f"layers_{layer_id}"
      layer = Gemma3DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          attention_type=attention_type,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot=None,
      page_state=None,
      previous_chunk=None,
      bidirectional_mask=None,
  ):

    cfg = self.config
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs

    for layer_id in range(self.num_of_layers):
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


Gemma3ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Gemma3ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    precision: str = "default",
    dtype: jnp.dtype = jnp.float32,
):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]  # pylint: disable=unpacking-non-sequence

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega, precision=precision)
  x = jnp.einsum("m,d->md", x.flatten(), omega, precision=precision)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


class MlpBlockViT(nnx.Module):
  """NNX version of Transformer MLP / feed-forward block."""

  def __init__(
      self,
      config: Config,
      block_id: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.block_id = block_id
    self.rngs = rngs

    self.Dense_0 = DenseGeneral(
        in_features_shape=self.config.hidden_size_for_vit,
        out_features_shape=self.config.intermediate_size_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=True,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )
    self.Dropout_0 = nnx.Dropout(rate=self.config.dropout_rate)
    self.Dense_1 = DenseGeneral(
        in_features_shape=self.config.intermediate_size_for_vit,
        out_features_shape=self.config.hidden_size_for_vit,
        dtype=self.config.dtype_mm,
        use_bias=True,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    """Applies the Transformer MlpBlock module."""
    x = self.Dense_0(x)
    x = nnx.gelu(x)
    x = self.Dropout_0(x, deterministic=deterministic)
    x = self.Dense_1(x)
    return x


class Encoder1DBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      block_id: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.block_id = block_id
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.seq_len = (self.config.image_size_for_vit // self.config.patch_size_for_vit) ** 2

    self.LayerNorm_0 = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit, epsilon=self.config.normalization_layer_epsilon, rngs=self.rngs
    )
    self.MultiHeadDotProductAttention_0 = Attention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit,
        max_target_length=self.seq_len,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(self.config.per_device_batch_size, self.seq_len, self.config.hidden_size_for_vit),
        inputs_kv_shape=(self.config.per_device_batch_size, self.seq_len, self.config.hidden_size_for_vit),
        dropout_rate=0,
        is_nope_layer=True,
        use_bias_in_projections=True,
        attention_type=AttentionType.FULL,
        use_qk_norm=False,
        query_pre_attn_scalar=1 / (self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit) ** 0.5,
        model_mode="train",
        is_vision=True,
        rngs=self.rngs,
    )
    self.LayerNorm_1 = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit, epsilon=self.config.normalization_layer_epsilon, rngs=self.rngs
    )
    self.MlpBlockViT_0 = MlpBlockViT(
        block_id=self.block_id,
        config=self.config,
        rngs=self.rngs,
    )
    self.Dropout_0 = nnx.Dropout(self.config.dropout_rate, rngs=self.rngs)

  def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
    y = self.LayerNorm_0(x)

    y = self.MultiHeadDotProductAttention_0(inputs_q=y, inputs_kv=y, deterministic=deterministic)
    y = self.Dropout_0(y, deterministic=deterministic)
    x = x + y

    y = self.LayerNorm_1(x)
    y = self.MlpBlockViT_0(y, deterministic=deterministic)
    y = self.Dropout_0(y, deterministic=deterministic)
    x = x + y
    return x


class Encoder(nnx.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    for lyr in range(self.config.num_hidden_layers_for_vit):
      layer_name = f"encoderblock_{lyr}"
      layer = Encoder1DBlock(
          block_id=lyr,
          config=self.config,
          mesh=self.mesh,
          rngs=self.rngs,
      )
      setattr(self, layer_name, layer)
    self.encoder_norm = nnx.LayerNorm(
        num_features=self.config.hidden_size_for_vit, epsilon=self.config.normalization_layer_epsilon, rngs=self.rngs
    )

  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    # TODO(aireenmei, hengtaoguo): add if-scan branch to enable scan support for vision encoder
    for lyr in range(self.config.num_hidden_layers_for_vit):
      x = getattr(self, f"encoderblock_{lyr}")(x, deterministic=deterministic)
    x = self.encoder_norm(x)
    return x


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      shape: tuple[int, ...],
      initializer: nnx.initializers.Initializer = nnx.initializers.normal(),
      dtype: jnp.dtype | None = None,
      precision: str = "default",
      *,
      rngs: nnx.Rngs,
  ):
    self.precision = precision
    self.w = nnx.Param(initializer(rngs.params(), shape, dtype))

  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    return jnp.einsum(eqn, x, self.w, precision=self.precision)


class VisionEmbedder(nnx.Module):
  """Projects image embeddings to the embedding space of the text encoder."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.mm_soft_embedding_norm = RMSNorm(
        num_features=self.config.hidden_size_for_vit,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.mm_input_projection = Einsum(
        shape=(self.config.hidden_size_for_vit, self.config.emb_dim),
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array, eqn: str = "...tm,md->...td") -> jax.Array:
    x = self.mm_soft_embedding_norm(x)
    x = self.mm_input_projection(eqn, x)
    return x


def visionembedder_as_linen(
    config: Config,
    mesh: Mesh,
):
  """Creates a VisionEmbedder module."""
  return nnx_wrappers.to_linen(
      VisionEmbedder,
      config,
      mesh=mesh,
      name="VisionEmbedder_0",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )


class VisionExit(nnx.Module):
  """The vision exit layer.

  Possibly downsample the soft tokens to a required output length.

  Attributes:
    output_length: The embed will be spatially avg-pooled to this output length.
  """

  def __init__(self, output_length: int = 256, *, rngs: nnx.Rngs):
    self.output_length = output_length
    self.rngs = rngs

  def __call__(self, x):
    cur_length = x.shape[1]
    if cur_length == self.output_length:
      return x
    cur_width = int(cur_length**0.5)
    assert cur_width**2 == cur_length
    output_width = int(self.output_length**0.5)
    assert output_width**2 == self.output_length, f"Cannot pool {x.shape=} to {self.output_length}=!"
    batch_size = x.shape[0]
    embed_dim = x.shape[-1]
    x = jnp.reshape(x, (batch_size, cur_width, cur_width, embed_dim))
    assert not cur_width % output_width, f"{cur_width=} {output_width=}"
    window = cur_width // output_width
    window_shape = (window, window)
    x = nnx.avg_pool(x, window_shape=window_shape, strides=window_shape)
    batch_size, height, width, embed_dim = x.shape
    return jnp.reshape(x, (batch_size, height * width, embed_dim))


def vision_exit_as_linen(x: jax.Array, output_length: int) -> jax.Array:
  """A wrapper to use VisionExit as a function."""
  return nnx.bridge.to_linen(VisionExit, output_length=output_length)(x)


class Gemma3VisionEncoderLayer(nnx.Module):
  """gemma 3 vision encoder layer"""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.embedding = nnx.Conv(
        in_features=self.config.num_channels_for_vit,
        out_features=self.config.hidden_size_for_vit,
        kernel_size=(self.config.patch_size_for_vit, self.config.patch_size_for_vit),
        strides=self.config.conv_stride_for_vit,
        padding="VALID",
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )
    self.pos_embedding = self._get_posemb(
        self.config.posemb_type_for_vit,
        seqshape=(
            self.config.image_size_for_vit // self.config.patch_size_for_vit,
            self.config.image_size_for_vit // self.config.patch_size_for_vit,
        ),
        width=self.config.hidden_size_for_vit,
        dtype=self.config.dtype_mm,
    )
    self.Dropout_0 = nnx.Dropout(self.config.dropout_rate, rngs=self.rngs)
    self.Transformer = Encoder(
        config=self.config,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.VisionExit = VisionExit(output_length=256, rngs=self.rngs)

  def _get_posemb(
      self,
      typ: str,
      *,
      seqshape: tuple[int, int],
      width: int,
      dtype: jnp.dtype = jnp.float32,
  ):
    """Returns the position embedding."""
    if typ == "learn":
      shape = (1, seqshape[0] * seqshape[1], width)
      initializer = nnx.initializers.normal(stddev=1 / (width**0.5))
      return nnx.Param(initializer(self.rngs.params(), shape, dtype))
    elif typ == "sincos2d":
      return _posemb_sincos_2d(*seqshape, width=width, dtype=dtype, precision=self.config.matmul_precision)
    else:
      raise ValueError(f"Unknown posemb type: {typ}")

  def __call__(self, inputs, deterministic, train=False):
    """ViT model that transforms image inputs to image embeddings.
    Args:
      inputs: jnp.array shaped [B, N, H, W, C], e.g. [4, 1, 896, 896, 3]
    Returns:
      jnp.array for image embeddings, shaped [B, N, P, D], e.g. [4, 1, 256, 1152]
    """
    # currently only supports N=1, the inputs shape is [B, H, W, C]
    if len(inputs.shape) == 4:
      inputs = inputs[:, None, :]
    b, n, h, w, c = inputs.shape
    x = jnp.reshape(inputs, [b * n, h, w, c])
    # Gemma3 uses conv2d with stride 14 and kernel size 14 to extract patches.
    x = self.embedding(x)
    bn, h, w, c = x.shape
    x = jnp.reshape(x, [bn, h * w, c])

    x = self.pos_embedding + x
    x = self.Dropout_0(x)

    # Transformer encoder to extract image features.
    x = self.Transformer(x, deterministic=deterministic)

    # Gemma3 use a vision exit layer to downsample the soft tokens to a required output length.
    x = self.VisionExit(x)
    bn, l, c = x.shape
    x = jnp.reshape(x, [b, n, l, c])
    return x


def gemma3visionencoder_as_linen(
    config: Config,
    mesh: Mesh,
):
  """Creates a Gemma3VisionEncoder module."""
  module = nnx_wrappers.to_linen(
      Gemma3VisionEncoderLayer,
      config=config,
      mesh=mesh,
      name="Gemma3VisionEncoderLayer_0",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module
