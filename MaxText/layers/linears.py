#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Linear Layers."""

import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.ad_checkpoint import checkpoint_name

from flax import nnx
import flax.linen as nn

from MaxText import max_logging
from MaxText.common_types import MODEL_MODE_PREFILL, DecoderBlockType, DType, Array, Config
from MaxText.layers import quantizations
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.initializers import NdInitializer, nd_dense_init, default_bias_init, variable_to_logically_partitioned
from MaxText.layers.quantizations import AqtQuantization as Quant


def _convert_to_activation_function(fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _compute_dot_general(inputs, kernel, kernel_axes, axis, contract_ind, matmul_precision, quant):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant:
    dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
    dot_general = dot_general_cls()
    return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


class DenseGeneral(nnx.Module):
  """A linear transformation with flexible axes."""

  def __init__(
      self,
      in_features_shape: Union[Iterable[int], int],
      out_features_shape: Union[Iterable[int], int],
      axis: Union[Iterable[int], int] = -1,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      quant: Optional[Quant] = None,
      use_bias: bool = False,
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      *,  # Following arguments are keyword-only
      rngs: nnx.Rngs,
  ):
    """Initializes the DenseGeneral module.

    Args:
      in_features_shape: tuple with numbers of input features for axes specified in
        'axis'.
      out_features_shape: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      quant: quantization config, defaults to None implying no quantization.
      use_bias: whether to add bias in linear transformation.
      matmul_precision: Precision for matrix multiplication.
      parameter_memory_host_offload: Determines whether to offload params to host
      rngs: RNG state for initialization in nnx.
    """
    self.in_features_shape = _canonicalize_tuple(in_features_shape)
    self.out_features_shape = _canonicalize_tuple(out_features_shape)
    self.axis = _canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.quant = quant
    self.use_bias = use_bias
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          sharding=bias_axes,
      )
    else:
      self.bias = None

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = _normalize_axes(self.axis, inputs.ndim)

    for i, ax in enumerate(norm_axis):
      if inputs.shape[ax] != self.in_features_shape[i]:
        raise ValueError(
            f"Input dimension {inputs.shape[ax]} at axis {ax} "
            f"does not match expected input feature size {self.in_features_shape[i]}"
        )

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = jnp.asarray(self.kernel[...], self.dtype)

    # Move logit_dense kernel to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
      kernel = jax.device_put(kernel, jax._src.sharding_impls.TransferToMemoryKind("device"))

    contract_ind = tuple(range(0, len(self.axis)))
    output = _compute_dot_general(
        inputs,
        kernel,
        self.kernel_axes,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant,
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output


def dense_general(
    *,
    inputs_shape: tuple[int, ...] | None = None,
    in_features_shape: tuple[int, ...] | int | None = None,
    out_features_shape: Union[Iterable[int], int],
    axis: Union[Iterable[int], int] = -1,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: Tuple[Optional[str], ...] = (),
    quant: Optional[Quant] = None,
    use_bias: bool = False,
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    name: Optional[str] = None,
):
  """Initializes the dense_general module.

  Args:
    inputs_shape: tuple with the shape of the inputs
    in_features_shape: tuple with numbers of input features for axes specified in
      'axis'.
    out_features_shape: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: logical axes for partitioning the kernel.
    quant: quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
    matmul_precision: Precision for matrix multiplication.
    parameter_memory_host_offload: Determines whether to offload params to host
    name: name passed to the ToLinen Module
  """
  if not (inputs_shape is not None) ^ (in_features_shape is not None):
    raise ValueError("Exactly one of inputs_shape or in_features must be specified.")

  if inputs_shape is not None:
    axis = _canonicalize_tuple(axis)
    in_features_shape = tuple(inputs_shape[ax] for ax in _normalize_axes(axis, len(inputs_shape)))
  else:
    assert in_features_shape is not None
  module = nnx.bridge.to_linen(
      DenseGeneral,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      quant=quant,
      use_bias=use_bias,
      matmul_precision=matmul_precision,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: computation data type for the dense layer.
    weight_dtype: weight data type for the dense layer.
    use_bias: whether to add bias in all feedforward layers.
    use_pre_norm: whether to add pre layer norm in mlp layers.
    quant: Optional quantization config, no quantization if None.
  """

  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable[..., Any]]] = ("relu",)
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  use_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None
  model_mode: Optional[str] = None

  def get_norm_layer(self, num_features: int):
    """get normalization layer."""
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.GEMMA,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(rms_norm, num_features=num_features)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      from MaxText.layers import gpt3  # pylint: disable=import-outside-toplevel

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=self.use_bias)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = self.get_norm_layer(num_features=inputs.shape[-1])(
          name="mlp_layer_norm",
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
      )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = dense_general(
          inputs_shape=inputs.shape,
          out_features_shape=(len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="wi",
          quant=self.quant,
          use_bias=self.use_bias,
          matmul_precision=self.config.matmul_precision,
      )(inputs)
      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        x = dense_general(
            inputs_shape=inputs.shape,
            out_features_shape=self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            name=dense_name,
            quant=self.quant,
            use_bias=self.use_bias,
            matmul_precision=self.config.matmul_precision,
        )(inputs)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.

    if self.model_mode == MODEL_MODE_PREFILL:
      x = nn.with_logical_constraint(x, ("activation_batch", "prefill_activation_length", "activation_mlp"))
    else:
      x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))

    output = dense_general(
        inputs_shape=x.shape,
        out_features_shape=inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        name="wo",
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(x)

    output = checkpoint_name(output, "mlpwo")
    return output
