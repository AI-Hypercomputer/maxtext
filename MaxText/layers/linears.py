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
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import common_types
from layers import initializers
from layers import normalizations
from layers import quantizations
import numpy as np
from jax.ad_checkpoint import checkpoint_name

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init
bias_init = initializers.default_bias_init

RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization

def _convert_to_activation_function(
    fn_or_string: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(f"""Don't know how to convert {fn_or_string}
                         to an activation function""")


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    use_bias: whether to add bias in linear transformation
    quant: quantization config, defaults to None implying no quantization.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  weight_dtype: DType = jnp.float32
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()
  quant: Optional[Quant] = None
  use_bias: bool = False

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """

    def compute_dot_general(inputs, kernel, axis, contract_ind):
      """Computes a dot_general operation that may be quantized."""
      dot_general = lax.dot_general
      if self.quant:
        dot_general_cls = self.quant.dot_general_cls()
        dot_general = dot_general_cls()
      return dot_general(
        inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save memory.
      # Instead they are retreived from the tensors stored in the 'aqt' collection.
      kernel = jnp.zeros(kernel_shape)
    else:
      kernel = self.param(
        'kernel',
        nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
        kernel_shape,
        self.weight_dtype,
        kernel_in_axis,
        kernel_out_axis,
      )
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    output = compute_dot_general(inputs, kernel, axis, contract_ind)

    if self.use_bias:
      bias_axes, bias_shape = self.kernel_axes[-len(features):], kernel_shape[-len(features):]
      bias = self.param(
          'bias',
          nn.with_logical_partitioning(bias_init, bias_axes),
          bias_shape,
          jnp.float32,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output


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
  activations: Sequence[Union[str, Callable[..., Any]]] = ('relu',)
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  use_bias: bool = False
  use_pre_norm: bool = False
  quant: Optional[Quant] = None

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3
      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=self.use_bias)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.use_pre_norm:
      inputs = self.get_norm_layer()(
        name='mlp_layer_norm',
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=('embed',),
        epsilon=cfg.normalization_layer_epsilon,
        )(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = DenseGeneral(
            (len(self.activations), self.intermediate_dim),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=('embed', 'num_activations', 'mlp'),
            name='wi',
            quant=self.quant,
            use_bias=self.use_bias,
      )(inputs)
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:,:,idx,...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
        x = DenseGeneral(
            self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=('embed', 'mlp'),
            name=dense_name,
            quant=self.quant,
            use_bias=self.use_bias,
        )(inputs)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    x = checkpoint_name(x, 'mlpwi')
    # Apply dropout and final dense output projection.
    x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )  # Broadcast along length.
    x = nn.with_logical_constraint(
        x, ('activation_batch', 'activation_length', 'activation_mlp')
    )
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo',
        quant=self.quant,
        use_bias=self.use_bias,
    )(x)

    output = checkpoint_name(output, 'mlpwo')
    return output


class MoeBlock(nn.Module):
  """Mixture of Experts (MoE) block.

  Attributes:
    num_experts: Number of experts.
    num_experts_per_tok: Number of experts for each token.
    kernel_init: Kernel function, passed to the dense layers.
    kernel_axes: Tuple with axes to apply kernel function.
    dtype: Type for the dense layer.
  """

  config: Config
  num_experts: int
  num_experts_per_tok: int
  kernel_init: NdInitializer
  kernel_axes: Tuple[str, ...]
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    gate_logits = DenseGeneral(
            self.num_experts,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name='gate')(inputs)

    weights, selected_experts = lax.top_k(gate_logits, self.num_experts_per_tok)
    weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1)
    mlp_lnx = jnp.zeros_like(inputs)
    weights = weights.astype(self.dtype)
    mlp_lnx = nn.with_logical_constraint(
            mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
        )

    # TODO(ranran): have a better solution to remove the loop here
    for k in range(self.num_experts):
        weights_exp = jnp.sum(jnp.multiply(selected_experts==k, weights), axis=-1)
        mlp_lnx_exp = MlpBlock(
          intermediate_dim=self.config.mlp_dim,
          activations=self.config.mlp_activations,
          intermediate_dropout_rate=self.config.dropout_rate,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name=f'mlp_{k}',
          config=self.config,
          )(inputs, deterministic=deterministic)

        mlp_lnx_exp = nn.with_logical_constraint(
            mlp_lnx_exp, ('activation_batch', 'activation_length', 'activation_embed')
        )
        mlp_lnx_exp = weights_exp[:, :, None] * mlp_lnx_exp
        mlp_lnx += mlp_lnx_exp

    return mlp_lnx
