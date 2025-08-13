"""
Copyright 2023 Google LLC

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

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx

from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import Config, DType, AxisNames, BATCH, LENGTH, EMBED, HEAD, D_KV, Array, MODEL_MODE_TRAIN
from MaxText.layers import initializers, nnx_wrappers
from MaxText.layers.linears import mlp_block
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.layers.attention_op import KVQuant, attention_op_as_linen
from MaxText.layers.initializers import Initializer, NdInitializer, nd_dense_init
from MaxText.layers.linears import dense_general
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
      kernel_axes: Tuple[Optional[str], ...] = (),
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

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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
    output = normed_inputs * (scale + 1)

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
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.zeros,
    use_bias: bool = True,
    reductions_in_fp32: bool = False,
    parameter_memory_host_offload: bool = False,
    name: Optional[str] = None,
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


class Gpt3MultiHeadAttention(nn.Module):
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

  config: Config
  num_heads: int
  head_dim: int
  max_target_length: int
  max_prefill_predict_length: int
  mesh: Mesh
  attention_kernel: str
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  dropout_rate: float = 0.0
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  float32_qk_product: bool = False  # computes logits in float32 for stability.
  float32_logits: bool = True  # cast logits in float32 for stability.
  fused_qkv: bool = True
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None
  use_bias: bool = True

  input_axis_names: AxisNames = (BATCH, LENGTH, EMBED)
  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  key_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  value_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  def qkv_projection(self, inputs: Array, proj_name: str):
    """Fused QKV projection"""

    qkv_proj = dense_general(
        inputs_shape=inputs.shape,
        out_features_shape=(3, self.num_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "qkv", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def projection(self, inputs: Array, proj_name: str) -> Array:
    """individual projection for one of q, k and v."""
    proj = dense_general(
        inputs_shape=inputs.shape,
        out_features_shape=(self.num_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    return proj

  def out_projection(self, output_dim: int, out: Array) -> Array:
    """output projection"""
    out_proj = dense_general(
        inputs_shape=out.shape,
        out_features_shape=output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(out)
    return out_proj

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
  ):
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    if self.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
      query = self.projection(inputs_q, proj_name="query")
      key = self.projection(inputs_q, proj_name="key")
      value = self.projection(inputs_q, proj_name="value")

    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query /= depth_scaling

    # annotate with sharding constraint.
    query = nn.with_logical_constraint(query, self.query_axis_names)
    query = checkpoint_name(query, "query_proj")
    key = nn.with_logical_constraint(key, self.key_axis_names)
    key = checkpoint_name(key, "key_proj")
    value = nn.with_logical_constraint(value, self.value_axis_names)
    value = checkpoint_name(value, "value_proj")

    attention_op = attention_op_as_linen(
        config=self.config,
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

    out = attention_op(query, key, value, decoder_segment_ids, model_mode)

    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    out = checkpoint_name(out, "out_proj")
    return out


# -----------------------------------------
# The Decoder Layer specific for GPT3
# -----------------------------------------


class Gpt3DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: models.Config
  mesh: Mesh
  model_mode: str
  quant: Optional[Quant] = None

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
  ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = gpt3_layer_norm(
        num_features=inputs.shape[-1],
        dtype=cfg.dtype,
        name="pre_self_attention_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
        reductions_in_fp32=False,
        use_bias=True,
    )(inputs)

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    # Self-attention block
    assert (
        cfg.num_query_heads == cfg.num_kv_heads
    ), f"{cfg.num_query_heads=} should be the same as {cfg.num_kv_heads=} in gpt3"
    attention_layer = Gpt3MultiHeadAttention(
        config=cfg,
        num_heads=cfg.num_query_heads,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=mesh,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        fused_qkv=cfg.fused_qkv,
        use_bias=True,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
    )

    attention_lnx = attention_layer(
        lnx, decoder_segment_ids=decoder_segment_ids, model_mode=model_mode, deterministic=deterministic
    )

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    attention_lnx += inputs

    # MLP block.
    mlp_lnx = mlp_block(
        in_features=attention_lnx.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        use_bias=True,
        use_pre_norm=True,
        config=cfg,
        quant=self.quant,
    )(attention_lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = attention_lnx + mlp_lnx

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
      return layer_output
