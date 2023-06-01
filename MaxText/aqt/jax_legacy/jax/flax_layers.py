# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of matrix multiply layers in flax with quantization.

Extends flax layers flax.linen.Dense.
"""

import contextlib
import dataclasses
import typing
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import flags
from aqt.jax_legacy.jax import compute_cost_utils
from aqt.jax_legacy.jax import get_bounds
from aqt.jax_legacy.jax import quant_config
from aqt.jax_legacy.jax import quantization
from aqt.jax_legacy.jax import shape_utils
from aqt.jax_legacy.jax import stats_tag
from aqt.jax_legacy.jax import utils
from aqt.jax_legacy.jax.flax import struct as flax_struct
from aqt.jax_legacy.jax.quantization import QuantOps
from aqt.jax_legacy.jax.quantization import QuantType
from aqt.jax_legacy.jax.sparsity import SparseHParams
from aqt.jax_legacy.jax.sparsity import Sparsity
import flax
from flax import linen as nn
from flax.core import frozen_dict
from flax.linen import partitioning
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS

# Alias for initializer function.
# Should follow the template `def init(key, shape, dtype=dtype) -> ndarray:`.
# See flax.linen.initializers and jax.nn.initializers for more details.
InitializerType = Callable[[jnp.ndarray, Sequence[int], Type[Any]], jnp.ndarray]

default_kernel_init = nn.initializers.lecun_normal()

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Initializer = Callable[[PRNGKey, Shape, DType], Array]
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]


# Based on flax.linen.Dense
class DenseAqt(nn.Module):
  """A linear transformation with optional per-feature weight quantization.

  Attributes:
    features: the number of output features.
    hparams: hyperparameters
    update_bounds: Bool whether to update activation bounds.
    paxis_name: axis_name to which a user `pmaps` the parent module (model),
      refer to jax.pmap() for more documentation. This arg is used for
      get_bounds acts quantization (QuantOps.create_input_fake_quant)
    train: Whether model is training.
    collect_acts_stats: Whether to tag activations to record statistics.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix. Should follow the
      template `def init(key, shape, dtype=dtype): -> array`. See
        flax.linen.initializers and jax.nn.initializers for more details.
    bias_init: initializer function for the bias. Should follow the template
      `def init(key, shape, dtype=dtype): -> array`. See flax.linen.initializers
        and jax.nn.initializers for more details.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details. Defaults to jax.lax.Precision.DEFAULT.
  """

  @dataclass
  class HParams:
    """Hyperparameter class to quantize Dense Layer."""
    # Target integer precision of weights in bits.
    # If None, no weight quantization will be applied.
    weight_prec: Union[None, int, QuantOps.FloatQuant]
    # half_shift flag for weights
    weight_half_shift: bool
    # QuantOps hyperparameter to quantize inputs. If None, no activation
    # quantization will be applied.
    quant_act: Optional[QuantOps.ActHParams]
    # Quantization strategy, one of `fake_quant` or `aqt`.
    quant_type: QuantType
    weight_quant_granularity: quant_config.QuantGranularity


    act_sparsity: Optional[SparseHParams] = None
    weight_sparsity: Optional[SparseHParams] = None

  hparams: HParams
  paxis_name: Optional[str]
  features: Union[int, Tuple[int, ...]]
  train: bool
  dynamic_context: quant_config.DynamicContext
  dtype: Any
  use_bias: bool = True
  kernel_init: InitializerType = default_kernel_init
  bias_init: InitializerType = nn.initializers.zeros
  precision: Optional[lax.Precision] = jax.lax.Precision.DEFAULT
  kernel_axis_names: Optional[Sequence[str]] = None
  possibly_use_quantized_vars: bool = False

  # TODO(shivaniagrawal): Changed the strategy to AQT if quant_type is aqt.

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      padding_mask: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    """Applies a linear transformation to the inputs with optional quantization.

    If weight_prec is not None, scales and quantizes weights to signed int with
    weight_prec bits.

    Args:
      inputs: The nd-array to be transformed.
      padding_mask: boolean tensor of the same shape as 'inputs' specifying
        which values of 'inputs' to use as part of the bounds calculation.
        'True' indicates the corresponding value from 'inputs' should be used.
        If None, all values are used.

    Returns:
      The transformed input.
    """
    batch_size = inputs.shape[0]
    if padding_mask is not None:
      shape_utils.assert_shapes_equal(padding_mask.shape, (batch_size, 1))
    # TODO(wanglisa): Replace fake quant with AQT.

    if self.dynamic_context.collect_acts_stats:
      stats_tag.StatsTag(
          channel_axis=-1, name='inputs', update_stats=self.train)(
              inputs, mask=padding_mask)
    hparams = self.hparams

    if (hparams.weight_prec is not None and
        isinstance(hparams.weight_prec, int) and hparams.weight_prec > 8):
      raise NotImplementedError(
          'If you want to use more than 8bits for quantization, please revisit '
          'jax.lax.Precision.DEFAULT to determine whether it is still sufficient.'
      )

    if not isinstance(self.features, tuple):
      features = (self.features,)
    else:
      features = self.features
    kernel_shape = (inputs.shape[-1],) + features
    if self.kernel_axis_names is None:
      kernel_axis_names = ['unmodeled'] * len(kernel_shape)
    else:
      kernel_axis_names = self.kernel_axis_names
      if len(kernel_axis_names) != len(kernel_shape):
        raise ValueError(f"Kernel axis names {kernel_axis_names} doesn't match "
                         f'kernel shape {kernel_shape}.')

    inputs = jnp.asarray(inputs, self.dtype)

    # Here are 3 possible conditions:
    #   1. not quantization: init normal kernel, quant_w = None
    #   2. quantization and train mode: init normal kernel, quantized kernel
    #      with bfloat16 to avoid gradient on integer and quant_w = None.
    #   3. quantization and inference mode: no normal kernel, init quantized
    #      kernel with int8 and quant_w = (qkernel, qscale).
    quant_w = None
    if (not self.possibly_use_quantized_vars) or (
        self.possibly_use_quantized_vars and self.train):
      kernel = partitioning.param_with_axes(
          'kernel',
          self.kernel_init,
          kernel_shape,
          axes=tuple(kernel_axis_names))
      kernel = jnp.asarray(kernel, self.dtype)

      kernel = Sparsity(
          sparsity_hparams=hparams.weight_sparsity, name='weight_sparsity')(
              kernel,
              update_mask=self.dynamic_context.update_weight_sparsity,
              apply_mask=self.dynamic_context.apply_sparsity,
              num_update_sparsity=self.dynamic_context.num_update_sparsity)

    if self.possibly_use_quantized_vars:
      qkernel_dtype = self.dtype
      qkernel_initializer = nn.initializers.zeros
      if not self.train:
        qkernel_dtype = jax.numpy.int8
        qkernel_initializer = nn.initializers.zeros

      qkernel = partitioning.param_with_axes(
          'qkernel',
          qkernel_initializer,
          kernel_shape,
          qkernel_dtype,
          axes=tuple(kernel_axis_names))
      qkernel = jnp.asarray(qkernel, qkernel_dtype)
      # Initialization of scale does not matter so we are initializing with
      # bias_init. And here only support quantizing on first dimension.
      scale_axis_names = list(kernel_axis_names)
      scale_axis_names[0] += '_qscale'
      qscale = partitioning.param_with_axes(
          'qscale',
          self.bias_init, (1,) + features,
          axes=tuple(scale_axis_names))
      qscale = jnp.asarray(qscale, self.dtype)
      if not self.train:
        quant_w = quantization.QuantW(qkernel, qscale)
        kernel = None

    inputs = Sparsity(
        sparsity_hparams=hparams.act_sparsity, name='act_sparsity')(
            inputs,
            update_mask=self.dynamic_context.update_act_sparsity,
            apply_mask=self.dynamic_context.apply_sparsity,
            num_update_sparsity=self.dynamic_context.num_update_sparsity)

    weight_quant_granularity = hparams.weight_quant_granularity
    # kernel.shape = (channels_in, channels_out)
    if weight_quant_granularity == quant_config.QuantGranularity.PER_CHANNEL:
      # Compute scale factors by reducing over the rows of the weight matrix,
      # resulting in one scale factor per column. This results in one scale
      # factor per output channel.
      expected_scale_shape = (1,) + features
      weight_quant_axis = (0,)
    elif weight_quant_granularity == quant_config.QuantGranularity.PER_TENSOR:
      # Compute a single scale factor for the entire weight matrix.
      expected_scale_shape = (1,) * len(kernel_shape)
      weight_quant_axis = None
    else:
      raise ValueError(
          f'Invalid quantization granularity {weight_quant_granularity}.')

    weight_params = QuantOps.WeightParams(
        prec=hparams.weight_prec,
        half_shift=hparams.weight_half_shift,
        axis=weight_quant_axis,
        expected_scale_shape=expected_scale_shape)

    # TODO(wanglisa): add option to control when scale is being recomputed

    bounds_params = None
    if hparams.quant_act is not None:
      if isinstance(hparams.quant_act.bounds, get_bounds.DynamicBounds.Hyper):
        bounds_params = get_bounds.DynamicBounds.Params(quant_axis=None)
      elif isinstance(hparams.quant_act.bounds, get_bounds.GetBounds.Hyper):
        bounds_params = get_bounds.GetBounds.Params(
            update_bounds=self.dynamic_context.update_bounds,
            update_stats=self.train,
            paxis_name=self.paxis_name,
            mask=padding_mask)

    contracting_dims = ((inputs.ndim - 1,), (0,))
    # `((lhs_contracting_dims, rhs_contracting_dims),
    batch_dims = ((), ())  # (lhs_batch_dims, rhs_batch_dims))`

    if hparams.quant_act is not None and isinstance(
        hparams.quant_act.bounds, get_bounds.DynamicBounds.Hyper):
      y = quantization.flaxformer_dot_general(
          act=inputs,
          w=kernel,
          dimension_numbers=(contracting_dims, batch_dims),
          weight_params=weight_params,
          act_hparams=hparams.quant_act,
          bounds_params=bounds_params,
          dot_precision=self.precision,
          quant_w=quant_w)

    else:
      y = quantization.quantized_dot_general(
          act=inputs,
          w=kernel,
          quant_type=hparams.quant_type,
          weight_params=weight_params,
          act_hparams=hparams.quant_act,
          bounds_params=bounds_params,
          dimension_numbers=(contracting_dims, batch_dims),
          dot_precision=self.precision,
          prefer_int8_to_int32_dot=self.dynamic_context
          .prefer_int8_to_int32_dot,
          quant_w=quant_w)

    # bias
    if self.use_bias:
      bias = partitioning.param_with_axes(
          'bias',
          self.bias_init, (np.prod(features),),
          axes=(kernel_axis_names[-1],))
      bias = jnp.asarray(bias, self.dtype)
      bias = jnp.reshape(bias, features)
      # (batch_size, features)
      y = y + bias[jnp.newaxis, :]
    return y


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


# The Flaxformer sharding API emits some names that are too detailed, so we
# remap them here. Any values that don't match keys here are unchanged.
_RESHAPED_KERNEL_AXIS_NAME_MAP = frozen_dict.freeze({
    'heads * kv': 'joined_kv',
})


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


# Based on flaxformer DenseGeneral
class DenseGeneralAqt(nn.Module):
  """A linear transformation with flexible axes.

    Kernel stored as 2d parameter for compatibility with Adafactor optimizer.

    Attributes:
      features: tuple with numbers of output features.
      use_bias: whether to add a bias to the output (default: False).
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_axis_names: logical axis names to use for kernel sharding. Each
        should be one of _VALID_AXIS_NAMES in sharding.py.
      reshaped_kernel_axis_name_map: Rules for renaming fused kernel axes. We
        keep this as a separate parameter than kernel_axis_names so that
        experiments can toggle `reshape_kernel` without having to keep
        `kernel_axis_names` in sync.
      reshape_kernel: whether to reshape the kernel parameter to 2D for
        Adafactor.
  """

  @dataclass
  class HParams:
    """Hyperparameter class to quantize Dense Layer."""
    # Target integer precision of weights in bits.
    # If None, no weight quantization will be applied.
    weight_prec: Union[None, int, QuantOps.FloatQuant]
    # half_shift flag for weights
    weight_half_shift: bool
    # QuantOps hyperparameter to quantize inputs. If None, no activation
    # quantization will be applied.
    quant_act: Optional[QuantOps.ActHParams]
    weight_quant_granularity: quant_config.QuantGranularity

  hparams: HParams
  train: bool
  features: Union[Iterable[int], int]
  use_bias: bool
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  precision: Any = None
  kernel_axis_names: Optional[Sequence[str]] = None
  reshaped_kernel_axis_name_map: Mapping[str, str] = (
      _RESHAPED_KERNEL_AXIS_NAME_MAP)
  reshape_kernel: bool = True
  possibly_use_quantized_vars: bool = False

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    hparams = self.hparams

    if (hparams.weight_prec is not None and
        isinstance(hparams.weight_prec, int) and hparams.weight_prec > 8):
      raise NotImplementedError(
          'If you want to use more than 8bits for quantization, please revisit '
          'jax.lax.Precision.DEFAULT to determine whether it is still sufficient.'
      )

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    scale_shape = (1,) * len(axis) + features
    if self.reshape_kernel:
      kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                            np.prod(features))
      scale_param_shape = (1, np.prod(features))
    else:
      kernel_param_shape = kernel_shape
      scale_param_shape = scale_shape

    # Determine axes names metadata for partitioning/adafactor rules.
    if self.kernel_axis_names is None:
      kernel_axis_names = ['unmodeled'] * len(kernel_param_shape)
    else:
      kernel_axis_names = self.kernel_axis_names
      if len(kernel_axis_names) != len(kernel_shape):
        raise ValueError(f"Kernel axis names {kernel_axis_names} doesn't match "
                         f'kernel shape {kernel_shape}.')
      if self.reshape_kernel:

        def _reshaped_axis_names(names):
          result = ' * '.join(names)
          return self.reshaped_kernel_axis_name_map.get(result, result)

        kernel_axis_names = (
            _reshaped_axis_names(kernel_axis_names[:len(axis)]),
            _reshaped_axis_names(kernel_axis_names[len(axis):]),
        )

    # Here are 3 possible conditions:
    #   1. not quantization: init normal kernel, quant_w = None
    #   2. quantization and train mode: init normal kernel, quantized kernel
    #      with bfloat16 to avoid gradient on integer and quant_w = None.
    #   3. quantization and inference mode: no normal kernel, init quantized
    #      kernel with int8 and quant_w = (qkernenl, qscale).
    quant_w = None
    if (not self.possibly_use_quantized_vars) or (
        self.possibly_use_quantized_vars and self.train):
      kernel = partitioning.param_with_axes(
          'kernel',
          self.kernel_init,
          kernel_param_shape,
          jnp.float32,
          axes=tuple(kernel_axis_names))
      kernel = jnp.asarray(kernel, self.dtype)
      kernel = jnp.reshape(kernel, kernel_shape)

    if self.possibly_use_quantized_vars:
      qkernel_dtype = self.dtype
      qkernel_initializer = nn.initializers.zeros
      if not self.train:
        qkernel_dtype = jax.numpy.int8
        qkernel_initializer = nn.initializers.zeros
      qkernel = partitioning.param_with_axes(
          'qkernel',
          qkernel_initializer,
          kernel_param_shape,
          qkernel_dtype,
          axes=tuple(kernel_axis_names))
      qkernel = jnp.asarray(qkernel, qkernel_dtype)
      qkernel = jnp.reshape(qkernel, kernel_shape)

      # Initialization of scale does not matter so we are initializing with
      # bias_init.
      scale_axis_names = list(kernel_axis_names)
      len_contracted_dims = 1 if self.reshape_kernel else len(axis)
      for ax in range(0, len_contracted_dims):
        scale_axis_names[ax] += '_qscale'
      qscale = partitioning.param_with_axes(
          'qscale',
          self.bias_init,
          scale_param_shape,
          axes=tuple(scale_axis_names))
      qscale = jnp.asarray(qscale, self.dtype)
      qscale = jnp.reshape(qscale, scale_shape)
      if not self.train:
        quant_w = quantization.QuantW(qkernel, qscale)
        kernel = None

    contract_ind = tuple(range(0, len(axis)))

    weight_quant_granularity = hparams.weight_quant_granularity

    act_quant_axis = None
    if hparams.quant_act:
      if isinstance(hparams.quant_act.bounds, get_bounds.DynamicBounds.Hyper):
        act_quant_granularity = hparams.quant_act.bounds.granularity  # pytype: disable=attribute-error  # jax-ndarray
        if act_quant_granularity == quant_config.QuantGranularity.PER_CHANNEL:
          act_quant_axis = tuple(axis)
        elif act_quant_granularity == quant_config.QuantGranularity.PER_TENSOR:
          act_quant_axis = None
        else:
          raise ValueError(
              f'Invalid quantization granularity {weight_quant_granularity}.')
      elif isinstance(hparams.quant_act.bounds, get_bounds.GetBounds.Hyper):
        raise NotImplementedError(
            'We do not support get_bounds for dot general.')

    if weight_quant_granularity == quant_config.QuantGranularity.PER_CHANNEL:
      # Compute scale factors by reducing over the rows of the weight matrix,
      # resulting in one scale factor per column. This results in one scale
      # factor per output channel.
      expected_scale_shape = (1,) * len(axis) + features
      weight_quant_axis = contract_ind
    elif weight_quant_granularity == quant_config.QuantGranularity.PER_TENSOR:
      # Compute a single scale factor for the entire weight matrix.
      expected_scale_shape = (1,) * len(kernel_shape)
      weight_quant_axis = None
    else:
      raise ValueError(
          f'Invalid quantization granularity {weight_quant_granularity}.')

    bounds_params = get_bounds.DynamicBounds.Params(quant_axis=act_quant_axis)

    weight_params = QuantOps.WeightParams(
        prec=hparams.weight_prec,
        half_shift=hparams.weight_half_shift,
        axis=weight_quant_axis,
        expected_scale_shape=expected_scale_shape)

    out = quantization.flaxformer_dot_general(
        act=inputs,
        w=kernel,
        dimension_numbers=((axis, contract_ind), ((), ())),
        weight_params=weight_params,
        act_hparams=hparams.quant_act,
        bounds_params=bounds_params,
        dot_precision=self.precision,
        quant_w=quant_w)

    if self.use_bias:
      bias = partitioning.param_with_axes(
          'bias',
          self.bias_init, (np.prod(features),),
          jnp.float32,
          axes=(kernel_axis_names[-1],))
      bias = jnp.asarray(bias, self.dtype)
      bias = jnp.reshape(bias, features)
      # Reshape bias for broadcast.
      expand_dims = sorted(set(range(inputs.ndim)) - set(axis))
      for ax in expand_dims:
        bias = jnp.expand_dims(bias, ax)
      out = out + bias
    return out


# Based on flax.linen.Conv
class ConvAqt(nn.Module):
  """Convolution Module with optional quantization.

  Attributes:
    features: number of convolution filters.
    hparams: hyperparameters
    update_bounds: Bool whether to update activation bounds.
    paxis_name: axis_name to which a user `pmaps` the parent module (model),
      refer to jax.pmap() for more documentation. This arg is used for
      get_bounds acts quantization (QuantOps.create_input_fake_quant)
    train: Whether model is training.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of `n` integers, representing the inter-window strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of `inputs`. Convolution with
      input dilation `d` is equivalent to transposed convolution with stride
      `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of the convolution kernel.
      Convolution with kernel dilation is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """

  @dataclass
  class HParams:
    """Hyperparameter class to quantize Conv Layer."""
    # Target integer precision of weights in bits.
    # If None, no weight quantization will be applied.
    # TODO(shivaniagrawal): we could replace this with quant_weight instead.
    weight_prec: Union[None, int, QuantOps.FloatQuant]
    # half_shift flag for weights
    weight_half_shift: bool
    # QuantOps hyperparameter to quantize inputs. If None, no activation
    # quantization will be applied.
    quant_act: Optional[QuantOps.ActHParams]
    # Quantization strategy, one of `fake_quant` or `aqt`.
    quant_type: QuantType

  hparams: HParams
  features: int
  kernel_size: Tuple[int, ...]
  dynamic_context: quant_config.DynamicContext
  train: bool
  paxis_name: Optional[str]
  dtype: Any
  strides: Optional[Tuple[int, ...]] = None
  padding: Union[str, Sequence[Tuple[int, ...]]] = 'SAME'
  input_dilation: Optional[Tuple[int, ...]] = None
  kernel_dilation: Optional[Tuple[int, ...]] = None
  feature_group_count: int = 1
  use_bias: bool = True
  kernel_init: InitializerType = default_kernel_init
  bias_init: InitializerType = flax.linen.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a convolution to the inputs with optional quantization.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).

    Returns:
      The convolved data.
    """
    hparams = self.hparams
    if hparams.weight_prec is not None and hparams.weight_prec > 8:
      raise NotImplementedError(
          'If you want to use more than 8bits for quantization, please revisit '
          'jax.lax.Precision.DEFAULT to determine whether it is still sufficient.'
      )
    jax_precision = jax.lax.Precision.DEFAULT

    if self.strides is None:
      strides = (1,) * (inputs.ndim - 2)
    else:
      strides = self.strides

    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = self.kernel_size + (in_features // self.feature_group_count,
                                       self.features)
    kernel = self.param('kernel', self.kernel_init, kernel_shape)

    inputs = jnp.asarray(inputs, self.dtype)
    kernel = jnp.asarray(kernel, self.dtype)

    # Activation quantization
    if hparams.quant_act is not None:
      inputs = QuantOps.create_inputs_fake_quant(
          inputs=inputs,
          hparams=hparams.quant_act,
          bounds_params=get_bounds.GetBounds.Params(
              update_bounds=self.dynamic_context.update_bounds,
              update_stats=self.train,
              paxis_name=self.paxis_name))

    # Weight quantization
    if hparams.weight_prec is not None:
      kernel_reduction_axis = tuple(range(kernel.ndim - 1))
      expected_scale_shape = (1,) * (kernel.ndim - 1) + (self.features,)
      assert hparams.quant_type == QuantType.FAKE_QUANT, (
          'we only support fake_quant style of aqt for ConvAqt.')
      quantized_type = hparams.quant_type.to_jax_type()
      kernel = QuantOps.create_weights_fake_quant(
          kernel,
          weight_params=QuantOps.WeightParams(
              prec=hparams.weight_prec,
              half_shift=hparams.weight_half_shift,
              axis=kernel_reduction_axis,
              expected_scale_shape=expected_scale_shape),
          quantized_type=quantized_type,
          quantize_weights=self.dynamic_context.quantize_weights)

    # Convolution
    dimension_numbers = flax.linen.linear._conv_dimension_numbers(
        inputs.shape)  # pylint: disable=protected-access
    metadata_context = contextlib.suppress()
    # Use metadata context to annotate op metadata with quantization info
    act_prec = None if hparams.quant_act is None else hparams.quant_act.prec

    if flags.FLAGS.metadata_enabled:
      metadata_context = compute_cost_utils.ConvMetadataMonkeyPatch(
          weight_prec=hparams.weight_prec, act_prec=act_prec)
    with metadata_context:
      y = lax.conv_general_dilated(
          inputs,
          kernel,
          strides,
          self.padding,
          lhs_dilation=self.input_dilation,
          rhs_dilation=self.kernel_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=self.feature_group_count,
          precision=jax_precision)
    # TODO(shivaniagrawal): create quantized conv general dilated.

    # bias
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      # The inputs can have an arbitrary number of spatial dims, so we broadcast
      # the bias to match: (batch_size, spatial_dim,... features)
      # TODO(shivaniagrawal): Consider making ConvAqt rank static (e.g. 2D)
      # or maybe add error checking (e.g. expect inputs to have rank N, but this
      # may already be checked by lax.conv_general_dialated).
      bias = utils.broadcast_rank(bias, inputs)
      y = y + bias
    return y


# From flax.linen.Embed
default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


# This is based on flax.linen.Embed
# (https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/flax/nn/linear.py#L360)
class EmbedAqt(nn.Module):
  """Quantized Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: Number of feature dimensions for each embedding.
    hparams: hyperparameters
    dtype: dtype to use for embedding.
    paxis_name: axis_name to which a user `pmaps` the parent module (model),
      refer to jax.pmap() for more documentation. This arg is used for
      get_bounds acts quantization (QuantOps.create_input_fake_quant)
    train: Whether model is training.
    update_bounds: Bool whether to update activation bounds.
    embedding_init: embedding initializer
  """

  @dataclass
  class HParams:  # pylint: disable=missing-docstring
    # Target integer precision of weights in bits.
    # If None, no quantization will be applied.
    weight_prec: Union[None, int, QuantOps.FloatQuant]
    # half_shift flag for weights
    weight_half_shift: bool
    # QuantOps hyperparameter to quantize inputs for logits. If None, no
    # activation quantization will be applied.
    quant_act: Optional[QuantOps.ActHParams]
    # Quantization strategy, one of `fake_quant` or `aqt`.
    quant_type: QuantType

  num_embeddings: int
  features: int
  hparams: HParams
  dtype: Any
  paxis_name: Optional[str]
  train: bool
  dynamic_context: quant_config.DynamicContext
  embedding_init: InitializerType = default_embed_init

  def setup(self):
    self.embedding = self.param(
        'embedding',
        self.embedding_init,  # pylint: disable=missing-from-attributes
        (self.num_embeddings, self.features))
    hparams = self.hparams
    if hparams.quant_act is not None and isinstance(hparams.quant_act.bounds,
                                                    get_bounds.GetBounds.Hyper):
      self.get_bounds_logits = get_bounds.GetBounds(  # pylint: disable=missing-from-attributes
          hyper=self.hparams.quant_act.bounds)
    self.quantized_dot = quantization.QuantizedDot(  # pylint: disable=missing-from-attributes
        act_hparams=hparams.quant_act,
        quant_type=hparams.quant_type,
        dot_precision=None,
        prefer_int8_to_int32_dot=self.dynamic_context.prefer_int8_to_int32_dot,
        weight_params=QuantOps.WeightParams(
            prec=hparams.weight_prec,
            axis=(0,),
            expected_scale_shape=(1, self.embedding.shape[0]),
            half_shift=hparams.weight_half_shift,
            ))

  def __call__(
      self,
      inputs: jnp.ndarray,
  ) -> jnp.ndarray:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    batch_size, sequence_length = inputs.shape
    if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
      raise ValueError('Input type must be an integer or unsigned integer.')
    embedding = self.embedding

    embedding = jnp.asarray(embedding, self.dtype)

    hparams = self.hparams
    # Initialize state for stats and bounds, this would be required for logits
    # in the following method attend.
    if hparams.quant_act is not None and isinstance(hparams.quant_act.bounds,
                                                    get_bounds.GetBounds.Hyper):
      self.get_bounds_logits(
          inputs,
          bounds_params=get_bounds.GetBounds.Params(
              update_stats=False, update_bounds=False, paxis_name=None),
      )

    weight_prec = hparams.weight_prec
    weight_half_shift = hparams.weight_half_shift
    if weight_prec is not None:
      quantized_type = hparams.quant_type.to_jax_type()
      # In contrast to all other scale factor calculations in this module, we
      # compute per-row instead of per-column (ie, per-output-channel) scale
      # factors here. This is because the embedding matrix might be shared with
      # the output (logit) layer of the transformer, in which case the
      # *transpose* of the embedding matrix will be used as the weight matrix in
      # a mamtul. The per-row scale factors used here would thus correspond to
      # using per-column (because of the transpose) scale factors used by the
      # weight matrix in the logits layer, which is what we need for AQT.
      embedding_quant_ops = QuantOps.create_weights_ops(
          embedding,
          weight_params=QuantOps.WeightParams(
              prec=weight_prec,
              axis=(1,),
              half_shift=weight_half_shift,
              ))
      embedding_quant_ops.assert_scale_shape_is(shape=(self.num_embeddings, 1))

      quantized_embedding = embedding_quant_ops.to_quantized(
          embedding, dtype=quantized_type)
      quantized_embedded_inputs = quantized_embedding[inputs]
      # Since the embedding matrix 'quantized_embedding' is gathered based on
      # 'inputs' to produce the embedding tensor, we apply the same gathering to
      # the per-row scale factors of the embedding matrix so the scale factors
      # will broadcast appropriately in the subsequent call to 'to_quantized'.
      # TODO(malmaud): As part of quantization.py refactor, change
      # 'get_scale_for_aqt' to cleanly support this and hence avoid the need to
      # directly access a protected member of QuantOps.
      scale = embedding_quant_ops._scale[inputs]  # pylint: disable=protected-access
      shape_utils.assert_shapes_equal(scale.shape,
                                      (batch_size, sequence_length, 1))
      shape_utils.assert_shapes_equal(
          quantized_embedded_inputs.shape,
          (batch_size, sequence_length, self.features))
      embedded_inputs = (quantized_embedded_inputs / scale).astype(self.dtype)
    else:
      embedded_inputs = embedding[inputs]
    shape_utils.assert_shapes_equal(
        embedded_inputs.shape, (batch_size, sequence_length, self.features))
    return embedded_inputs

  def attend(self, query: jnp.ndarray, padding_mask: Optional[jnp.ndarray],
             **unused_kwargs) -> jnp.ndarray:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.
      padding_mask: boolean mask indicating which elements of 'query' are
        padding. Used for calculating activation statistics for the dynamic
        bounds quantization algorithm.
      **unused_kwargs: unused arguments passed from the apply method.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    del unused_kwargs

    batch_size = query.shape[0]

    if padding_mask is not None:
      shape_utils.assert_shapes_equal(padding_mask.shape, (batch_size, 1))

    embedding = self.embedding
    embedding = jnp.asarray(embedding, self.dtype)

    # TODO(malmaud): Remove the 'mask' field from this struct so we can
    # make this struct a hyperparameter of the EncoderAqt class.
    bounds_params = get_bounds.GetBounds.Params(
        update_bounds=self.dynamic_context.update_bounds,
        update_stats=self.train,
        paxis_name=self.paxis_name,
        mask=padding_mask,
        module_name='logits')

    out = self.quantized_dot(
        act=query,
        w=jnp.transpose(embedding),
        bounds_params=bounds_params)

    return out


# Forked from Flax LayerNorm module.
class LayerNormAqt(nn.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  Operates on the last axis of the input data.

  Adds quantization support to the Flax LayerNorm layer.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32). Can also be eg
      bfloat16. Note this is the real Jax type that intermediate activations
      will be stored in and is separate from the quantized type specified in
      'hparams' which we are simulating via downcasting operations.
    bias:  If True, bias (beta) is added.
    scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    hparams: An instance of LayerNormAqt.HParams with quantization-related
      hyperparameters.
  """

  @dataclass
  class QuantHParams:
    # TODO(malmaud): Generalize this to other quantization formats.
    prec: QuantOps.FloatQuant.FloatPrec
    reduction_prec: Optional[QuantOps.FloatQuant.FloatPrec]

  @dataclass
  class HParams:
    # We have to refer to the type name with quotes here since we can't directly
    # refer to types being defined in the same class.
    quant_hparams: Optional['LayerNormAqt.QuantHParams']

  hparams: HParams
  dtype: Any
  dynamic_context: quant_config.DynamicContext
  epsilon: float = 1e-6
  use_bias: bool = True
  use_scale: bool = True
  bias_init: InitializerType = nn.initializers.zeros
  scale_init: InitializerType = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input.

    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    if self.hparams.quant_hparams is None:
      return nn.LayerNorm(
          epsilon=self.epsilon,
          dtype=self.dtype,
          use_bias=self.use_bias,
          use_scale=self.use_scale,
          bias_init=self.bias_init,
          scale_init=self.scale_init)(
              x)
    # To match the behavior of the upstream layernorm before the quantization
    # start step, we use float32 for intermediate computations.
    dtype = jnp.float32
    x = x.astype(dtype)

    hparams = self.hparams
    num_features = x.shape[-1]

    if self.use_scale:
      scale_param = self.param('scale', self.scale_init, (num_features,))
    if self.use_bias:
      bias_param = self.param('bias', self.bias_init, (num_features,))

    def quantized_layernorm(x):
      prec = hparams.quant_hparams.prec
      fp_quant = QuantOps.FloatQuant(is_scaled=False, fp_spec=prec)
      quant_ops = QuantOps.create_symmetric_fp(fp_quant=fp_quant, bounds=None)

      def to_quantized(x):
        return quant_ops.to_quantized(x, dtype=dtype)

      # If epsilon is too small to represent in the quantized format, we set it
      # to the minimal representative non-zero value to avoid the possibility of
      # dividing by zero.
      fp_bounds = quantization.fp_cast.get_bounds(prec.exp_min, prec.exp_max,
                                                  prec.sig_bits)
      epsilon = max(self.epsilon, fp_bounds.flush_to_zero_bound)
      quantized_epsilon = to_quantized(jnp.array(epsilon, dtype=dtype))

      # If the reciprocal of the quantized number of features is too small to
      # represent in the quantized format, we set it to the minimal
      # representative nonzero value so that the mean and variance are not
      # trivially 0.
      num_features_quantized = to_quantized(
          jnp.array(num_features, dtype=dtype))
      num_features_recip_quantized = to_quantized(
          jnp.reciprocal(num_features_quantized))
      num_features_recip_quantized = jax.lax.cond(
          jax.lax.eq(num_features_recip_quantized,
                     0.0), lambda _: quantized_epsilon,
          lambda _: num_features_recip_quantized, None)

      x_quantized = to_quantized(x)
      x_sum_quantized_reduction = quantization.quantized_sum(
          x_quantized,
          axis=-1,
          keepdims=True,
          prec=hparams.quant_hparams.reduction_prec)
      x_sum = to_quantized(x_sum_quantized_reduction)
      mean = to_quantized(x_sum * num_features_recip_quantized)
      x_minus_mean = to_quantized(x - mean)
      x_sq = to_quantized(lax.square(x_minus_mean))
      x_sq_sum_quantized_reduction = quantization.quantized_sum(
          x_sq,
          axis=-1,
          keepdims=True,
          prec=hparams.quant_hparams.reduction_prec)
      x_sq_sum = to_quantized(x_sq_sum_quantized_reduction)
      var = to_quantized(x_sq_sum * num_features_recip_quantized)
      # Prevent division by zero.
      var_plus_epsilon = to_quantized(var + quantized_epsilon)
      mul = to_quantized(lax.rsqrt(var_plus_epsilon))
      if self.use_scale:
        quantized_scale_param = to_quantized(scale_param)
        mul = to_quantized(mul * quantized_scale_param)
      y = to_quantized(x_minus_mean * mul)
      if self.use_bias:
        quantized_bias_param = to_quantized(bias_param)
        y = to_quantized(y + quantized_bias_param)
      return y.astype(self.dtype)

    # We keep  this structurally as similar as possible to
    # 'quantized_layernorm' to make it easy to compare their implementations,
    # and thus don't use convenience functions like `jnp.mean`.
    def unquantized_layernorm(x):
      num_features_recip = jnp.reciprocal(num_features)
      x_sum = jnp.sum(x, axis=-1, keepdims=True)
      mean = x_sum * num_features_recip
      x_minus_mean = x - mean
      x_sq = lax.square(x_minus_mean)
      x_sq_sum = jnp.sum(x_sq, axis=-1, keepdims=True)
      var = x_sq_sum * num_features_recip
      var_plus_epsilon = var + self.epsilon
      mul = lax.rsqrt(var_plus_epsilon)
      if self.use_scale:
        mul = mul * scale_param
      y = x_minus_mean * mul
      if self.use_bias:
        y = y + bias_param
      return y.astype(self.dtype)

    quantized_result = quantized_layernorm(x)
    unquantized_result = unquantized_layernorm(x)
    return lax.cond(self.dynamic_context.quantize_acts,
                    lambda _: quantized_result, lambda _: unquantized_result,
                    None)
