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
"""Quantized conv_general."""

import functools
from typing import Optional, Sequence, Tuple, Union

from aqt.common import aqt_config
from aqt.common import aqt_config_utils
from aqt.jax import aqt_tensor
from aqt.jax import aqt_utils
import jax
from jax import lax
import jax.numpy as jnp

# pylint: disable=protected-access
# pytype: disable=attribute-error


@functools.partial(
    jax.custom_jvp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
def _conv_general_aqt(
    lhs: jnp.ndarray,  #
    rhs: jnp.ndarray,
    lhs_quantizer: aqt_tensor.TensorQuantizer,
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    should_int8_quantize: bool,
    train: bool,
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Optional[Sequence[int]],
    rhs_dilation: Optional[Sequence[int]],
    dimension_numbers: lax.ConvGeneralDilatedDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int) -> jnp.ndarray:
  """Wrapper around lax.conv_general_dilated, but with option to use integer conv."""

  def conv_general_float(ops):
    lhs_, rhs_ = ops
    lhs_ = aqt_utils.possibly_use_quantized_variable(lhs_quantizer, lhs_, train)
    rhs_ = aqt_utils.possibly_use_quantized_variable(rhs_quantizer, rhs_, train)
    return lax.conv_general_dilated(
        lhs_,
        rhs_,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count)

  def conv_general_int(ops):
    lhs_, rhs_ = ops
    lhs_int = lhs_.astype(jnp.int8)
    rhs_int = rhs_.astype(jnp.int8)

    lhs_int = aqt_utils.possibly_use_quantized_variable(lhs_quantizer, lhs_int,
                                                        train)
    rhs_int = aqt_utils.possibly_use_quantized_variable(rhs_quantizer, rhs_int,
                                                        train)

    return lax.conv_general_dilated(
        lhs_int,
        rhs_int,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        preferred_element_type=jnp.int32).astype(jnp.float32)

  return lax.cond(should_int8_quantize, conv_general_int, conv_general_float,
                  (lhs, rhs))


@_conv_general_aqt.defjvp
def _conv_general_aqt_jvp(
    lhs_quantizer: aqt_tensor.TensorQuantizer,  #
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    should_int8_quantize: bool,
    train: bool,
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Optional[Sequence[int]],
    rhs_dilation: Optional[Sequence[int]],
    dimension_numbers: lax.ConvGeneralDilatedDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    primals,
    tangents) -> jnp.ndarray:
  """Custom gradient for conv_general_aqt that ignores integer casts."""
  lhs, rhs = primals
  lhs_dot, rhs_dot = tangents
  y = _conv_general_aqt(
      lhs,
      rhs,
      lhs_quantizer,
      rhs_quantizer,
      should_int8_quantize=should_int8_quantize,
      train=train,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count)

  def differentiable_conv_general(lhs_, rhs_):
    # The backward pass op is performed in floating point for now and thus
    # here we use lax.conv_general_dilated to be differentiated by Autodiff.
    return lax.conv_general_dilated(
        lhs_,
        rhs_,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count)

  _, y_tangent = jax.jvp(
      differentiable_conv_general,  #
      (lhs, rhs),
      (lhs_dot, rhs_dot))
  return y, y_tangent  # pytype: disable=bad-return-type  # jax-ndarray


def _validate_dilation_argument(
    lhs_quantizer,  #
    rhs_quantizer,
    lhs_dilation,
    rhs_dilation):
  """Validates preserve_zero=True of AqtIntQuantConfig when the dilation is used."""
  lhs_configs = lhs_quantizer.config.tensor_configs
  rhs_configs = rhs_quantizer.config.tensor_configs
  event_count = lhs_quantizer._last_update.value

  for lhs_config, rhs_config in zip(lhs_configs, rhs_configs):
    if (isinstance(lhs_config.quant_config, aqt_config.IntQuantConfig) and
        isinstance(rhs_config.quant_config, aqt_config.IntQuantConfig) and
        lhs_config.quant_config.bits <= 8 and
        rhs_config.quant_config.bits <= 8):
      is_config_active = aqt_tensor.is_config_active(lhs_config, event_count)
      if is_config_active:
        if lhs_dilation is not None and not lhs_config.quant_config.preserve_zero:
          raise aqt_config.ConfigError(
              f'lhs_config.quant_config.preserve_zero ({lhs_config.quant_config.preserve_zero}) must be True if the input is dilated.'
          )
        if rhs_dilation is not None and not rhs_config.quant_config.preserve_zero:
          raise aqt_config.ConfigError(
              f'rhs_config.quant_config.preserve_zero ({rhs_config.quant_config.preserve_zero}) must be True if the filter is dilated.'
          )


def _validate_inputs(
    input_quantizer: aqt_tensor.TensorQuantizer,  #
    filter_quantizer: aqt_tensor.TensorQuantizer,
    dimension_numbers: lax.ConvGeneralDilatedDimensionNumbers) -> None:
  """Validates configs and inputs for conv_general_dilated.

  Args:
    input_quantizer: the input tensor quantizer.
    filter_quantizer: the filter tensor quantizer.
    dimension_numbers: the conv input tensor argument axes format.

  Raises:
    aqt_config.ConfigError: The input or filter quantizer configurations
      do not share statistics along contraction axes as described by
      `_validate_contraction`, or their quantization schedules are not aligned
      per `aqt_config._validate_alignment`.
  """

  if input_quantizer.config is not None and filter_quantizer.config is not None:
    aqt_config_utils._validate_alignment(
        'input_quantizer.config.tensor_configs',
        input_quantizer.config.tensor_configs,
        'filter_quantizer.config.tensor_configs',
        filter_quantizer.config.tensor_configs)

  input_spec, filter_spec, _ = dimension_numbers  # pytype: disable=attribute-error
  _, *input_contracted_dims = input_spec
  _, *filter_contracted_dims = filter_spec

  if input_quantizer.config is not None:
    for axis in input_contracted_dims:
      if axis not in input_quantizer.config.stats_config.share_stats_axes:
        raise aqt_config.ConfigError(
            f'expected contraction axis ({axis}) to be in '
            f'input_quantizer.config.stats_config.share_stats_axes={input_quantizer.config.stats_config.share_stats_axes}'
        )
  if filter_quantizer.config is not None:
    for axis in filter_contracted_dims:
      if axis not in filter_quantizer.config.stats_config.share_stats_axes:
        raise aqt_config.ConfigError(
            f'expected contraction axis ({axis}) to be in '
            f'filter_quantizer.config.stats_config.share_stats_axes={filter_quantizer.config.stats_config.share_stats_axes}'
        )


def _transpose_inv_scale(x, dimension_numbers_before, dimension_numbers_after):
  """Changes the order of axes in x from dimension_numbers_before to dimension_numbers_after."""
  assert (len(dimension_numbers_before) == len(dimension_numbers_after)), (
      f'len(dimension_numbers_before) ({len(dimension_numbers_before)}) must ',
      f'be equal to len(dimension_numbers_after) ({len(dimension_numbers_after)})'
  )

  axes = [0] * len(dimension_numbers_before)
  for i, axis in enumerate(dimension_numbers_after):
    axes[axis] = dimension_numbers_before[i]

  return jnp.transpose(x, axes)


def conv_general_dilated(lhs: jnp.ndarray,
                         rhs: jnp.ndarray,
                         lhs_quantizer: aqt_tensor.TensorQuantizer,
                         rhs_quantizer: aqt_tensor.TensorQuantizer,
                         window_strides: Sequence[int],
                         padding: Union[str, Sequence[Tuple[int, int]]],
                         lhs_dilation: Optional[Sequence[int]] = None,
                         rhs_dilation: Optional[Sequence[int]] = None,
                         dimension_numbers: Optional[
                             lax.ConvGeneralDilatedDimensionNumbers] = None,
                         feature_group_count: int = 1,
                         batch_group_count: int = 1,
                         train: bool = True) -> jnp.ndarray:
  """Quantized jax.lax.conv_general_dilated.

  Args:
    lhs: Left-hand side of the conv_general_dilated (input).
    rhs: Right-hand side of the conv_general_dilated (filter).
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    lhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of `lhs`. LHS dilation is also
      known as transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the dilation
      factor to apply in each spatial dimension of `rhs`. RHS dilation is also
      known as atrous convolution.
    dimension_numbers: either `None`, a ``ConvDimensionNumbers`` object, or a
      3-tuple ``(lhs_spec, rhs_spec, out_spec)``, where each element is a string
      of length `n+2`.
    feature_group_count: integer, default 1. See XLA HLO docs.
    batch_group_count: integer, default 1. See XLA HLO docs.
    train: If false and `use_quantized_variable` in lhs_quantizer or
      rhs_quantizer, then this indicates `aqt_conv_general` should use the
      quantized variable with the latest quantized, memorized from the most
      recent `TensorQuantizer.update()` in quantized operations rather than the
      float tensor input `lhs` or `rhs` provided to those operations at
      inference time.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  # TODO(jihwanlee): Support quantization for non-2D convolution.
  if len(lhs.shape) != 4:
    raise NotImplementedError('Currently, aqt_conv_general_dilated supports ',
                              'only 2-D convolution.')

  dimension_numbers = lax.conv_dimension_numbers(lhs.shape, rhs.shape,
                                                 dimension_numbers)

  _validate_inputs(lhs_quantizer, rhs_quantizer, dimension_numbers)

  lhs_scale, lhs_inv_scale = lhs_quantizer._get_quant_scale(train)
  rhs_scale, rhs_inv_scale = rhs_quantizer._get_quant_scale(train)

  lhs = lhs_scale * lhs
  rhs = rhs_scale * rhs

  lhs = lhs_quantizer._to_quant(lhs, train)
  rhs = rhs_quantizer._to_quant(rhs, train)

  should_int8_quantize = aqt_utils.should_int8_quantize(lhs_quantizer,
                                                        rhs_quantizer)

  if should_int8_quantize:
    _validate_dilation_argument(lhs_quantizer, rhs_quantizer, lhs_dilation,
                                rhs_dilation)

  conv = _conv_general_aqt(lhs, rhs, lhs_quantizer, rhs_quantizer,
                           should_int8_quantize, train, window_strides, padding,
                           lhs_dilation, rhs_dilation, dimension_numbers,
                           feature_group_count, batch_group_count)

  # Transpose both lhs_inv_scale and rhs_inv_scale such that they have `NHWC`
  # and `HWIO` shapes, respectively, which allows them to be broadcastable no
  # matter how they were shaped originally by `dimension_numbers`.
  dim_numbers = {
      'NHWC': (0, 3, 1, 2),
      'HWIO': (3, 2, 0, 1),
  }
  if lhs_quantizer.config is not None and rhs_quantizer.config is not None:
    lhs_inv_scale = _transpose_inv_scale(lhs_inv_scale, dimension_numbers[0],
                                         dim_numbers['NHWC'])
    rhs_inv_scale = _transpose_inv_scale(rhs_inv_scale, dimension_numbers[1],
                                         dim_numbers['HWIO'])
    assert len(lhs_inv_scale.shape) == len(rhs_inv_scale.shape)

  inv_scale = lhs_inv_scale * rhs_inv_scale
  # Reverse the shape of inv_scale back to one specified by out_spec in
  # `dimension_numbers`
  inv_scale = _transpose_inv_scale(inv_scale, dim_numbers['NHWC'],
                                   dimension_numbers[2])
  return conv * inv_scale
