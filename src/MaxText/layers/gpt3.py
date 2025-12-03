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

from typing import Any, Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding

from flax import linen as nn
from flax import nnx

from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import Config, DType, AxisNames, BATCH, LENGTH, EMBED, HEAD, D_KV, Array, MODEL_MODE_TRAIN
from MaxText.layers import initializers, nnx_wrappers
from MaxText.layers.linears import DenseGeneral, MlpBlock, canonicalize_tuple, normalize_axes
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.layers import linears
from MaxText.layers.attentions import AttentionOp, KVQuant
from MaxText.layers.initializers import Initializer, NdInitializer, nd_dense_init
from MaxText.layers.quantizations import AqtQuantization as Quant

# -----------------------------------------
# The Normalization Layer specific for GPT3
# -----------------------------------------


class Gpt3LayerNorm(nnx.Module):
  """GPT3 Layer normalization operating on the last axis of the input data."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = nn.initializers.zeros,
      use_bias: bool = True,
      reductions_in_fp32: bool = False,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.use_bias = use_bias
    self.reductions_in_fp32 = reductions_in_fp32
    self.parameter_memory_host_offload = parameter_memory_host_offload

    self.scale = nnx.Param(
        self.scale_init(rngs.params(), (num_features,), self.weight_dtype),
        sharding=self.kernel_axes,
    )
    if self.use_bias:
      self.bias = nnx.Param(
          initializers.default_bias_init(rngs.params(), (num_features,), self.weight_dtype), sharding=self.kernel_axes
      )
    else:
      self.bias = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    if self.reductions_in_fp32:
      x = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=[-1], keepdims=True)
    normed_inputs = (x - mean) * lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(self.dtype)

    scale = self.scale.value
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("gpt3.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    scale = jnp.asarray(scale, self.dtype)
    # broadcast second inputs and element-wise mul
    output = jnp.einsum(
        "i...k,...k->i...k",
        normed_inputs,
        scale + 1,
        out_sharding=out_sharding,
    )

    if self.bias is not None:
      bias = self.bias.value
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output


def gpt3_layer_norm(
    *,
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: tuple[None | str, ...] = (),
    scale_init: Initializer = nn.initializers.zeros,
    use_bias: bool = True,
    reductions_in_fp32: bool = False,
    parameter_memory_host_offload: bool = False,
    name: None | str = None,
):
  """Initializes the gpt3_layer_norm module.

  Args:
    num_features: the number of features.
    epsilon: the epsilon for the layer norm.
    dtype: the dtype of the computation (default: float32).
    weight_dtype: the dtype of the weights (default: float32).
    kernel_axes: logical axes for partitioning the kernel.
    scale_init: initializer for the scale.
    use_bias: whether to add bias in linear transformation.
    reductions_in_fp32: whether to do reductions in fp32.
    parameter_memory_host_offload: Determines whether to offload params to host
    name: name passed to the ToLinen Module
  """

  module = nnx_wrappers.to_linen(
      Gpt3LayerNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      use_bias=use_bias,
      reductions_in_fp32=reductions_in_fp32,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
  return module


# -----------------------------------------
# The Attention Layer specific for GPT3
# -----------------------------------------


class Gpt3MultiHeadAttention(nnx.Module):
  """Multi-head attention in gpt3.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    head_dim: dimension of each head.
    max_target_length: maximum length of output
    max_prefill_predict_length: size of the maximum prefill
    mesh: device mesh
    dtype: the dtype of the computation.
    dropout_rate: dropout rate
    kernel_init: initializer for the kernel of the Dense layers.
    float32_qk_product: bool, if True then compute logits via float32 qk_product to avoid
      numerical issues with bfloat16.
    float32_logits: bool, if True then cast logits to float32 before softmax to avoid
      numerical issues with bfloat16.
    fused_qkv: whether to fuse query, key and value into one projection.
    quant: Quant, stores quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      num_heads: int,
      feature_dim: tuple[int, ...],
      head_dim: int,
      max_target_length: int,
      max_prefill_predict_length: int,
      mesh: Mesh,
      rngs: nnx.Rngs,
      attention_kernel: str,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      dropout_rate: float = 0.0,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      float32_qk_product: bool = False,  # computes logits in float32 for stability.
      float32_logits: bool = True,  # cast logits in float32 for stability.
      fused_qkv: bool = True,
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      use_bias: bool = True,
      input_axis_names: AxisNames = (BATCH, LENGTH, EMBED),
      query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
      key_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
      value_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
      out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
      **kwargs: Any,
  ):
    self.config = config
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.max_target_length = max_target_length
    self.max_prefill_predict_length = max_prefill_predict_length
    self.mesh = mesh
    self.attention_kernel = attention_kernel
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.dropout_rate = dropout_rate
    self.kernel_init = kernel_init
    self.float32_qk_product = float32_qk_product
    self.float32_logits = float32_logits
    self.fused_qkv = fused_qkv
    self.quant = quant
    self.kv_quant = kv_quant
    self.use_bias = use_bias
    self.input_axis_names = input_axis_names
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.out_axis_names = out_axis_names
    self.rngs = rngs
    if self.fused_qkv:
      self.qkv_proj = self.create_projection_layer(
          feature_dim, (3, self.num_heads, self.head_dim), ("embed", "qkv", "heads", "kv")
      )
    else:
      self.query = self.create_projection_layer(feature_dim, (self.num_heads, self.head_dim), ("embed", "heads", "kv"))
      self.key = self.create_projection_layer(feature_dim, (self.num_heads, self.head_dim), ("embed", "heads", "kv"))
      self.value = self.create_projection_layer(feature_dim, (self.num_heads, self.head_dim), ("embed", "heads", "kv"))
    self.out = self.create_projection_layer(
        (self.num_heads, self.head_dim), self.num_heads * self.head_dim, ("heads", "kv", "embed"), axis=(-2, -1)
    )
    self.attention_op = AttentionOp(
        config=config,
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        max_target_length=self.max_target_length,
        float32_qk_product=self.float32_qk_product,
        float32_logits=self.float32_logits,
        quant=self.quant,
        kv_quant=self.kv_quant,
        num_query_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        dtype=self.dtype,
    )

  def create_projection_layer(
      self,
      input_shape: tuple[int, ...],
      output_shape: tuple[int, ...] | int,
      kernel_axes: tuple[str, ...],
      axis: int | tuple[int, ...] = -1,
  ):
    """Create projection layer for Key, Value, Query and Output"""
    axis = canonicalize_tuple(axis)
    in_features_shape = tuple(input_shape[ax] for ax in normalize_axes(axis, len(input_shape)))

    return DenseGeneral(
        in_features_shape=in_features_shape,
        out_features_shape=output_shape,
        axis=axis,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

  def qkv_projection(self, projection_layer: Any, inputs: Array):
    """Fused QKV projection"""
    qkv_proj = projection_layer(inputs)

    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def projection(self, projection_layer: Any, inputs: Array) -> Array:
    """individual projection for one of q, k and v."""
    proj = projection_layer(inputs)
    return proj

  def __call__(
      self,
      inputs_q: Array,
      decoder_segment_ids: Array | None = None,
      *,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      kv_cache: Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
  ):
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    if self.fused_qkv:
      query, key, value = self.qkv_projection(self.qkv_proj, inputs_q)
    else:
      query = self.projection(self.query, inputs_q)
      key = self.projection(self.key, inputs_q)
      value = self.projection(self.value, inputs_q)

    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query /= depth_scaling

    # annotate with sharding constraint.
    query = nn.with_logical_constraint(query, self.query_axis_names)
    query = checkpoint_name(query, "query_proj")
    key = nn.with_logical_constraint(key, self.key_axis_names)
    key = checkpoint_name(key, "key_proj")
    value = nn.with_logical_constraint(value, self.value_axis_names)
    value = checkpoint_name(value, "value_proj")

    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode)

    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.projection(self.out, out)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache


# -----------------------------------------
# The Decoder Layer specific for GPT3
# -----------------------------------------


class Gpt3DecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: models.Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[Quant] = None,
  ):

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.pre_self_attention_norm = Gpt3LayerNorm(
        num_features=dummy_inputs_shape[-1],
        dtype=config.dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        reductions_in_fp32=False,
        use_bias=True,
        rngs=self.rngs,
    )

    self.mlp = MlpBlock(
        mesh=self.mesh,
        in_features=dummy_inputs_shape[-1],
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        use_bias=True,
        use_pre_norm=True,
        config=config,
        quant=self.quant,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    self.self_attention = Gpt3MultiHeadAttention(
        config=config,
        num_heads=config.num_query_heads,
        dtype=config.dtype,
        feature_dim=dummy_inputs_shape,
        weight_dtype=config.weight_dtype,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        mesh=self.mesh,
        dropout_rate=config.dropout_rate,
        name="self_attention",
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        fused_qkv=config.fused_qkv,
        use_bias=True,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        model_mode=model_mode,
        rngs=self.rngs,
    )

    self.dropout = linears.Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

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
      kv_cache=None,
      attention_metadata=None,
  ):
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = self.pre_self_attention_norm(inputs)

    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Self-attention block
    assert (
        self.config.num_query_heads == self.config.num_kv_heads
    ), f"{self.config.num_query_heads=} should be the same as {self.config.num_kv_heads=} in gpt3"

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        deterministic=deterministic,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    attention_lnx += inputs
    # MLP block.
    mlp_lnx = self.mlp(attention_lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = attention_lnx + mlp_lnx
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


Gpt3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gpt3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
