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
"""Quantized dot_general."""

import functools

from aqt.common import aqt_config
from aqt.common import aqt_config_utils
from aqt.jax import aqt_tensor
from aqt.jax import aqt_utils
import jax
from jax import lax
import jax.numpy as jnp

# TODO(b/220181240): Remove accesses to protected methods. e.g., TQ._to_quant()
# -> TQ.to_quant().
# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4, 5, 6))
def _dot_general_aqt(
    lhs,  #
    rhs,
    lhs_quantizer,
    rhs_quantizer,
    dimension_numbers,
    should_int8_quantize,
    train):
  """Wrapper around lax.dot_general, but with option to use integer dot.

  This function comes equipped with a custom gradient that defines the
  gradient of this function to be the same as the equivalent call to
  lax.dot_general, ignoring casts to and from integer types so that
  quantization-aware-training will work correctly.

  See docstring of lax.dot_general.

  Args:
    lhs: Left-hand side of the dot_general.
    rhs: Right-hand side of the dot_general.
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    dimension_numbers: a tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`
    should_int8_quantize: If true, inputs to lax.dot_general will be cast to
      int8 and results accumulated to int32, then converted back to the original
      input type.
    train: If false and `TensorQuantizer.use_quantized_variable` is True, then
      use the quantized variable, instead of input tensors, for the respective
      input tensor.

  Returns:
    Same as lax.dot_general, but its quantized version.
  """

  def dot_general_float(ops):
    lhs_, rhs_ = ops
    lhs_ = aqt_utils.possibly_use_quantized_variable(lhs_quantizer, lhs_, train)
    rhs_ = aqt_utils.possibly_use_quantized_variable(rhs_quantizer, rhs_, train)
    return lax.dot_general(lhs_, rhs_, dimension_numbers=dimension_numbers)

  def dot_general_int(ops):
    lhs_, rhs_ = ops
    lhs_int = lhs_.astype(jnp.int8)
    rhs_int = rhs_.astype(jnp.int8)

    lhs_int = aqt_utils.possibly_use_quantized_variable(lhs_quantizer, lhs_int,
                                                        train)
    rhs_int = aqt_utils.possibly_use_quantized_variable(rhs_quantizer, rhs_int,
                                                        train)

    return lax.dot_general(
        lhs_int,
        rhs_int,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.int32).astype(jnp.float32)

  return lax.cond(should_int8_quantize, dot_general_int, dot_general_float,
                  (lhs, rhs))


@_dot_general_aqt.defjvp
def _dot_general_aqt_jvp(
    lhs_quantizer,  #
    rhs_quantizer,
    dimension_numbers,
    should_int8_quantize,
    train,
    primals,
    tangents):
  """Custom gradient for dot_general_aqt that ignores integer casts."""
  lhs, rhs = primals
  lhs_dot, rhs_dot = tangents
  y = _dot_general_aqt(
      lhs,
      rhs,
      lhs_quantizer,
      rhs_quantizer,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=should_int8_quantize,
      train=train)

  def differentiable_dot_general(lhs_, rhs_):
    return lax.dot_general(lhs_, rhs_, dimension_numbers=dimension_numbers)

  _, y_tangent = jax.jvp(
      differentiable_dot_general,  #
      (lhs, rhs),
      (lhs_dot, rhs_dot))
  return y, y_tangent


def _validate_inputs(
    lhs_quantizer: aqt_tensor.TensorQuantizer,  #
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    dimension_numbers: lax.DotDimensionNumbers):
  """Validates configs and inputs for dot_general."""

  lhs_config = lhs_quantizer.config
  rhs_config = rhs_quantizer.config

  if lhs_config is not None and rhs_config is not None:
    aqt_config_utils._validate_alignment(
        'lhs_config',  #
        lhs_config.tensor_configs,
        'rhs_config',
        rhs_config.tensor_configs)

  lhs_contracting_dims, rhs_contracting_dims = dimension_numbers[0]

  if not (lhs_config is None or
          any(a in lhs_config.stats_config.share_stats_axes
              for a in lhs_contracting_dims)):
    raise aqt_config.ConfigError(
        f'expected lhs dot_general contraction axis to be in '
        f'share_stats_axes={lhs_config.stats_config.share_stats_axes}')
  if not (rhs_config is None or
          any(a in rhs_config.stats_config.share_stats_axes
              for a in rhs_contracting_dims)):
    raise aqt_config.ConfigError(
        f'expected rhs dot_general contraction axis to be in '
        f'share_stats_axes={rhs_config.stats_config.share_stats_axes}')


def dot(lhs: jnp.ndarray, rhs: jnp.ndarray,
        lhs_quantizer: aqt_tensor.TensorQuantizer,
        rhs_quantizer: aqt_tensor.TensorQuantizer) -> jnp.ndarray:
  """Quantized lax.dot.

  This dot operator is based on lax.dot,
  https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/lax.html#dot.
  For more general contraction, see the `dot_general` operator.

  Args:
    lhs: Left-hand side of the dot.
    rhs: Right-hand side of the dot.
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  if 1 <= lhs.ndim <= 2 and 1 <= rhs.ndim <= 2 and lhs.shape[-1] == rhs.shape[0]:
    return dot_general(
        lhs,
        rhs,
        lhs_quantizer,
        rhs_quantizer,
        dimension_numbers=(((lhs.ndim - 1,), (0,)), ((), ())))
  else:
    raise TypeError('Incompatible shapes for dot: got {} and {}.'.format(
        lhs.shape, rhs.shape))


def dot_general(lhs: jnp.ndarray,
                rhs: jnp.ndarray,
                lhs_quantizer: aqt_tensor.TensorQuantizer,
                rhs_quantizer: aqt_tensor.TensorQuantizer,
                dimension_numbers: lax.DotDimensionNumbers,
                train: bool = True) -> jnp.ndarray:
  """Quantized jax.lax.dot_general.

  Args:
    lhs: Left-hand side of the dot_general.
    rhs: Left-hand side of the dot_general.
    lhs_quantizer: The tensor quantizer for lhs.
    rhs_quantizer: The tensor quantizer for rhs.
    dimension_numbers: a tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`
    train: If false and `use_quantized_variable` in lhs_quantizer or
      rhs_quantizer, then this indicates `aqt_dot_general` should use the
      quantized variable with the latest quantized, memorized from the most
      recent `TensorQuantizer.update()` in quantized operations rather than the
      float tensor input `lhs` or `rhs` provided to those operations at
      inference time.

  Returns:
    An array containing the result with the same dtype as 'lhs' and 'rhs'.
  """
  _validate_inputs(lhs_quantizer, rhs_quantizer, dimension_numbers)

  lhs_scale, lhs_inv_scale = lhs_quantizer._get_quant_scale(train)
  rhs_scale, rhs_inv_scale = rhs_quantizer._get_quant_scale(train)

  lhs = lhs_scale * lhs
  rhs = rhs_scale * rhs

  lhs = lhs_quantizer._to_quant(lhs, train)
  rhs = rhs_quantizer._to_quant(rhs, train)

  should_int8_quantize = aqt_utils.should_int8_quantize(lhs_quantizer,
                                                        rhs_quantizer)

  out = _dot_general_aqt(
      lhs,
      rhs,
      lhs_quantizer,
      rhs_quantizer,
      dimension_numbers=dimension_numbers,
      should_int8_quantize=should_int8_quantize,
      train=train)

  if lhs_quantizer.config is not None and rhs_quantizer.config is not None:
    inv_scale = lax.dot_general(
        lhs_inv_scale, rhs_inv_scale, dimension_numbers=dimension_numbers)
  else:
    inv_scale = lhs_inv_scale * rhs_inv_scale

  return out * inv_scale


def injectable_dot_general(lhs_quantizer, rhs_quantizer, train):
  """Wrapper of aqt_dot_general, supposed to be used in injection API."""
  return lambda a, b, dimension_numbers, precision: dot_general(  # pylint: disable=g-long-lambda
      lhs=a,
      rhs=b,
      lhs_quantizer=lhs_quantizer,
      rhs_quantizer=rhs_quantizer,
      dimension_numbers=dimension_numbers,
      train=train)
