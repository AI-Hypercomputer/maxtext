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

"""Linear Layers."""

import functools
import operator
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.ad_checkpoint import checkpoint_name

from flax import nnx
import flax.linen as nn

from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import MODEL_MODE_PREFILL, DecoderBlockType, DType, Array, Config
from MaxText.layers import nnx_wrappers, quantizations
from MaxText.layers import normalizations
from MaxText.layers.initializers import NdInitializer, nd_dense_init, default_bias_init, variable_to_logically_partitioned
from MaxText.layers.quantizations import AqtQuantization as Quant


def _convert_to_activation_function(fn_or_string: str | Callable[..., Any]) -> Callable[..., Any]:
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


def normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def canonicalize_tuple(x):
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


def _compute_dot_general_nnx(
    inputs, kernel, axis, contract_ind, matmul_precision, quant_dot_general: nnx_wrappers.ToNNX | None, initializing: bool
):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant_dot_general is not None:
    if initializing:
      quant_dot_general.lazy_init(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
    return quant_dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None, mutable=["aqt"])
  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


class DenseGeneral(nnx.Module):
  """A linear transformation with flexible axes."""

  def __init__(
      self,
      in_features_shape: Iterable[int] | int,
      out_features_shape: Iterable[int] | int,
      axis: Iterable[int] | int = -1,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = (),
      quant: None | Quant = None,
      use_bias: bool = False,
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      *,  # Following arguments are keyword-only
      rngs: nnx.Rngs = None,
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
    self.in_features_shape = canonicalize_tuple(in_features_shape)
    self.out_features_shape = canonicalize_tuple(out_features_shape)
    self.axis = canonicalize_tuple(axis)
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

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      dummy_inputs = jnp.zeros((1, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def __call__(self, inputs: Array, _initializing: bool = False) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = normalize_axes(self.axis, inputs.ndim)

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
      kernel = self.kernel[...]
      # Move logit_dense kernel to device if parameter offloading is enabled
      if self.parameter_memory_host_offload:
        max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
        kernel = jax.device_put(kernel, max_utils.device_space())
      kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(self.axis)))
    output = _compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output


def dense_general(
    *,
    inputs_shape: tuple[int, ...] | None = None,
    in_features_shape: tuple[int, ...] | int | None = None,
    out_features_shape: Iterable[int] | int,
    axis: Iterable[int] | int = -1,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[None | str, ...] = (),
    quant: None | Quant = None,
    use_bias: bool = False,
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    name: None | str = None,
):
  """Creates a DenseGeneral Linen module using nnx.bridge.to_linen.

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
    axis = canonicalize_tuple(axis)
    in_features_shape = tuple(inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape)))
  else:
    assert in_features_shape is not None
  module = nnx_wrappers.to_linen(
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
      abstract_init=False,
  )
  return module


class Dropout(nnx.Dropout):
  """Forked nnx.Dropout that is easier to use with bridge"""
  def __init__( # pylint: disable=super-init-not-called
    self,
    rate: float,
    *,
    broadcast_dims: Sequence[int] = (),
    deterministic: bool = False,
    rng_collection: str = 'dropout',
    rngs: nnx.Rngs| None = None,
  ):
    self.rate = rate
    self.broadcast_dims = broadcast_dims
    self.deterministic = deterministic
    self.rng_collection = rng_collection

    if isinstance(rngs, nnx.Rngs):
      self.rngs = rngs.fork() if hasattr(type(rngs), 'fork') else rngs
    else:
      raise TypeError(
        f'rngs must be a Rngs, RngStream or None, but got {type(rngs)}.'
      )

class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      config: Config,
      in_features: int,
      intermediate_dim: int = 2048,
      activations: Sequence[str | Callable[..., Any]] = ("relu",),
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      intermediate_dropout_rate: float = 0.1,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      use_bias: bool = False,
      use_pre_norm: bool = False,
      quant: None | Quant = None,
      model_mode: None | str = None,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    """A MlpBlock module.

    Args:
      config: Config object containing model parameters.
      in_features: Number of input features.
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
    self.config = config
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.activations = activations
    self.kernel_init = kernel_init
    self.intermediate_dropout_rate = intermediate_dropout_rate
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.use_bias = use_bias
    self.use_pre_norm = use_pre_norm
    self.quant = quant
    self.model_mode = model_mode

    if self.use_pre_norm:
      self.mlp_layer_norm = self.get_norm_layer(num_features=in_features)(
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=config.normalization_layer_epsilon,
          rngs=rngs,
      )
    else:
      self.mlp_layer_norm = None

    if config.fused_mlp:
      self.wi = DenseGeneral(
          in_features_shape=in_features,
          out_features_shape=(len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          quant=self.quant,
          use_bias=self.use_bias,
          matmul_precision=self.config.matmul_precision,
          rngs=rngs,
      )
    else:
      for idx in range(len(self.activations)):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = DenseGeneral(
            in_features_shape=in_features,
            out_features_shape=self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", "mlp"),
            quant=self.quant,
            use_bias=self.use_bias,
            matmul_precision=self.config.matmul_precision,
            rngs=rngs,
        )
        setattr(self, dense_name, module)
    self.dropout = Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,), rngs=rngs)
    self.wo = DenseGeneral(
        in_features_shape=self.intermediate_dim,
        out_features_shape=in_features,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=("mlp", "embed"),
        quant=self.quant,
        use_bias=self.use_bias,
        matmul_precision=self.config.matmul_precision,
        rngs=rngs,
    )

  def get_norm_layer(self, num_features: int):
    """get normalization layer."""
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.QWEN3,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(normalizations.RMSNorm, num_features=num_features)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      from MaxText.layers import gpt3  # pylint: disable=import-outside-toplevel

      return functools.partial(
          gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=self.use_bias
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.mlp_layer_norm is not None:
      inputs = self.mlp_layer_norm(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = self.wi(inputs)
      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = getattr(self, dense_name)
        x = module(inputs)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = self.dropout(x, deterministic=deterministic)  # Broadcast along length.
    if self.model_mode == MODEL_MODE_PREFILL:
      x = nn.with_logical_constraint(x, ("activation_batch", "prefill_activation_length", "activation_mlp"))
    else:
      x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_mlp"))
    output = self.wo(x)

    output = checkpoint_name(output, "mlpwo")
    return output


def mlp_block(
    *,
    config: Config,
    in_features: int,
    intermediate_dim: int = 2048,
    activations: Sequence[str | Callable[..., Any]] = ("relu",),
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    intermediate_dropout_rate: float = 0.1,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    use_bias: bool = False,
    use_pre_norm: bool = False,
    quant: None | Quant = None,
    model_mode: None | str = None,
    name: None | str = None,
):
  """Creates a MlpBlock Linen module using nnx.bridge.to_linen."""
  module = nnx_wrappers.to_linen(
      MlpBlock,
      config=config,
      in_features=in_features,
      intermediate_dim=intermediate_dim,
      activations=activations,
      kernel_init=kernel_init,
      intermediate_dropout_rate=intermediate_dropout_rate,
      dtype=dtype,
      weight_dtype=weight_dtype,
      use_bias=use_bias,
      use_pre_norm=use_pre_norm,
      quant=quant,
      model_mode=model_mode,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
