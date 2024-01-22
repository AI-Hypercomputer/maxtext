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
from typing import Any, Callable, Iterable, Sequence, Tuple, Union

import flax.linen as nn
from jax import lax
import jax.numpy as jnp
import common_types
from layers import initializers
from layers import quantizations
import numpy as np

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
NdInitializer = initializers.NdInitializer

nd_dense_init = initializers.nd_dense_init


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
  """A linear transformation (without bias) with flexible axes.

  Attributes:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
  """

  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()
  use_int8: bool = False

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
      if not self.use_int8:
        return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))
      else:
        aqt_rng = self.make_rng('aqt')
        aqt_dot_general = quantizations.int8_dot_general(aqt_rng)
        return aqt_dot_general(
            inputs, kernel, ((axis, contract_ind), ((), ()))
        )

    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    kernel = self.param(
        'kernel',
        nn.with_logical_partitioning(self.kernel_init, self.kernel_axes),
        kernel_shape,
        jnp.float32,
        kernel_in_axis,
        kernel_out_axis,
    )
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    return compute_dot_general(inputs, kernel, axis, contract_ind)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """

  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable[..., Any]]] = ('relu',)
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = DenseGeneral(
            (len(self.activations), self.intermediate_dim),
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=('embed', 'num_activations', 'mlp'),
            name='wi',
            use_int8=cfg.int8_training,
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
            kernel_init=self.kernel_init,
            kernel_axes=('embed', 'mlp'),
            name=dense_name,
            use_int8=cfg.int8_training,
        )(inputs)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
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
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo',
        use_int8=cfg.int8_training,
    )(x)
    return output
