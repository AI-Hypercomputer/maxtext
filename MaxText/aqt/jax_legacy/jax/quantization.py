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

"""Abstraction for quantizing neural networks implemented in jax."""

import contextlib
import dataclasses
import enum
import functools
import logging
import typing
from typing import Iterable, Optional, Tuple, Union

from absl import flags
from aqt.jax_legacy.jax import compute_cost_utils
from aqt.jax_legacy.jax import fp_cast
from aqt.jax_legacy.jax import get_bounds
from aqt.jax_legacy.jax import primitives
from aqt.jax_legacy.jax import shape_utils
from aqt.jax_legacy.jax import utils
from aqt.jax_legacy.jax.flax import struct as flax_struct
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp

# Global bool to control the use of epsilon in the denominator of the scaling
# methods signed_int_scale and unsigned_int_scale. Epsilon is added to avoid
# division by 0. For testing, one may choose to disable the epsilon by setting
# this global to True.
# As this is a global variable, please modify it only before calling any
# functions that use it.
DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = False

# Dtype for quantization computations: scaling; floor and clip; rescaling. this
# is chosen to optimize performance for a given hardware i.e. for TPU we set it
# to float32. It should be matching native dtype of the hardware's
# 'vector unit'.
SCALE_DTYPE = jnp.float32

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass

# ActBounds can be an Jax array of floats with a shape that is broadcastable to
# the shape of activation tensors.
ActsBoundT = Union[float, jnp.ndarray, get_bounds.GetBounds.Hyper,
                   get_bounds.DynamicBounds.Hyper, None]


@dataclass
class _FloatQuant:
  """Parameters for floating-point quantization.

  Floating-point quantization refers to degraded floating-point precision
  below those natively supported, e.g., bfloat16. This quantization scheme
  can either work with, or without scaling (controlled by `is_scaled`).

  With scaling, these quantization steps follow,

  1. Use the maximum representable floating-point value to determine a scale.
  2. This scale is used to "upscale" the argument to the range of the target
     floating-point format.
  3. The precision of the argument is then degraded through a downcast
     operation.
  4. Finally the degraded-precision result is "downscaled" by the inverse
     scale.

  Without scaling, these quantization steps follow,

  1. The argument is downcast to the target fp-format with degraded precision.
     Of importance in this downcast is the saturating behavior, which is
     logically equivalent to clipping by the maximum representable target
     value.
  """

  @dataclass
  class FloatPrec:
    """Parameters for specifying a custom precision floating-point type."""
    # The minimum exponent value of the quantized floating-point format.
    exp_min: int
    # The maximum exponent value of the quantized floating-point format.
    exp_max: int
    # The number of significand bits (excluding hidden bit) of the quantized
    # floating-point format.
    sig_bits: int

  # Whether or not floating-point fake-quant makes use of scaling.
  is_scaled: bool
  # Precision specification for floating-point quantization.
  fp_spec: FloatPrec


_PrecT = Union[None, int, _FloatQuant]  # pylint: disable=invalid-name


class QuantType(str, enum.Enum):
  """Quantization strategy dataclass."""

  # fake_quant strategy ensures that quantized values form an arithmetic
  # sequence e.g.  0*s ... 255*s for 8-bit positive quantization, for some s.
  # it can be implemented as a local op: upscale, floor, clip, downscale.
  FAKE_QUANT = 'fake_quant'
  # fake_quant strategy with quantized inputs/weights type-casted to int.
  FAKE_QUANT_WITH_INT = 'fake_quant_with_int'
  # aqt ensures that MatMul/Conv are in actual integer domain.
  # It can't be implemented as a single op.
  # Before matmul we have upscale, floor and clip, and after matmul we have
  # downscale.
  AQT = 'aqt'

  def to_jax_type(self) -> primitives.jnp_dtype:
    """Returns quantized dtype for the corresponding quantization strategy."""
    # Currently, this function is used to decide the return type for
    # 'QuantOps.to_quantized.' The AQT implementation works by having a
    # conversion to an int dtype and then  back to a fp dtype happen *within*
    # to_quantized, so that Jax backprop works correctly. Thus
    # counter-intuitively, we need this to return a fp dtype for 'aqt' since the
    # return type for 'to_quantized' overall is fp. TODO(malmaud): As part of
    # the refactor of this module, clean this up to eliminate the
    # counter-intuitive behavior.
    if self.value in ['aqt', 'fake_quant']:  # pylint: disable=comparison-with-callable
      return SCALE_DTYPE
    elif self.value == 'fake_quant_with_int':  # pylint: disable=comparison-with-callable
      return jnp.int8
    else:
      raise RuntimeError(f'QuantType {self.value} is unknown.')


class QuantOps:
  """Class for quantizing and dequantizing weights and activations."""

  # Redefined here as nested class attributes to avoid forward-references.
  FloatQuant = _FloatQuant  # pylint: disable=invalid-name
  PrecT = _PrecT  # pylint: disable=invalid-name

  @dataclass
  class WeightParams:
    """Parameters for weight quantization."""
    prec: _PrecT  # expected precision for weight quantization.
    # enable all available values during quantization
    half_shift: bool
    # Axis along which to quantize weights (the non-feature axis).
    axis: Optional[Iterable[int]]


    # expected scale shape for weights quantization. Defaults to None.
    expected_scale_shape: Union[None, int, Tuple[int, ...]] = None

  @dataclass
  class ActHParams:
    """Parameters for activation quantization."""

    # Inheriting from 'str' and making the enums have string values lets us
    # conveniently serialize this class to JSON without a custom JSON encoder.
    class InputDistribution(str, enum.Enum):
      SYMMETRIC = 'symmetric'
      POSITIVE = 'positive'

    input_distribution: InputDistribution
    # float means fixed bound. '-1' means no quantization.
    bounds: ActsBoundT
    prec: _PrecT
    half_shift: bool


  def __init__(
      self,  #
      *,
      prec: Union[int, _FloatQuant],
      scale: Optional[jnp.ndarray],
      symmetric: bool,
      bounds: Optional[jnp.ndarray],
      half_shift: bool):
    """Default constructor, use of named constructors is strongly encoraged.

    Args:
      prec: precision for the QuantOps
      scale: scaling factor to scale the input to quantized precision range
      symmetric: whether the input to quantize is symmetric
      bounds: Optional. The clipping bounds used for calculating scale factors.
      half_shift: Symmetric quantization with all available values enabled
    """
    self._prec = prec
    self._half_shift = half_shift
    if scale is None:
      self._scale = None
    else:
      self._scale = scale.astype(SCALE_DTYPE)

    self._symmetric = symmetric
    # Storing bounds are useful for two reasons: one is debugging, since it
    # makes easy to see how a QuantOps instance came up with its scale factor.
    # Two is that right now, we use a bounds of '-1' as a special value meaning
    # to 'not quantize'. See comment on the 'should_quantize' method for more
    # details.
    self._bounds = bounds

  @classmethod
  def create_symmetric_fp(
      cls,
      *,
      bounds: Optional[primitives.BoundsT],
      fp_quant: _FloatQuant,
  ) -> 'QuantOps':
    """Create QuantOps for symmetric clipping to floating-point bounds.

    Args:
      bounds: The upper (and absolute lower) bound to clip the inputs.
      fp_quant: quantization floating-point specification of the target format.

    Returns:
      QuantOps for quantizing/dequantizing signed activations.
    """
    if bounds is None:
      if fp_quant.is_scaled:
        raise ValueError(
            'bounds can only be None if fp_quant.is_scaled is False.')
      return cls(
          prec=fp_quant,
          scale=None,
          symmetric=True,
          bounds=None,
          half_shift=False)  # disable half_shift for fp quantization
    else:
      initial_bounds = bounds
      # We set bounds = -1 to indicate no quantization.
      # TODO(shivaniagrawal): Move away from the hack of setting bound as -1.
      bounds = jnp.asarray(bounds, SCALE_DTYPE)
      if not DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING:
        # to avoid log2(0)
        bounds = jnp.abs(bounds) + jnp.finfo(SCALE_DTYPE).eps
      scale = jnp.exp2(-jnp.floor(jnp.log2(bounds)))  # Scale to unit binade.
      # NOTE: stop_gradient is needed here to prevent gradient flow through
      # scale when scale is not a constant, but computed as a function of
      # activations or weights.
      scale = lax.stop_gradient(scale)

      return cls(
          prec=fp_quant,
          scale=scale,
          symmetric=True,
          bounds=initial_bounds,
          half_shift=False)  # disable half_shift for fp quantization

  @classmethod
  def create_symmetric(cls, *, bounds: primitives.BoundsT, prec: int,
                       half_shift: bool) -> 'QuantOps':
    """Create QuantOps for symmetric activations clipped to [-bounds, bounds].

    Args:
      bounds: The upper (and absolute lower) bound to clip the inputs.
      prec: Signed int precision for the QuantOps.
      half_shift: Symmetric quantization with all available values enabled

    Returns:
      QuantOps for quantizing/dequantizing signed activations.
    """
    initial_bounds = bounds
    bounds = jnp.asarray(bounds, SCALE_DTYPE)
    if not DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING:
      bounds += jnp.finfo(SCALE_DTYPE).eps  # to avoid div by 0
    scale = primitives.signed_int_bound(
        prec=prec, half_shift=half_shift) / bounds
    # NOTE: stop_gradient is needed here to prevent gradient flow through scale
    # when scale is not a constant, but computed as a function of activations or
    # weights.
    scale = lax.stop_gradient(scale)
    return cls(  # pytype: disable=wrong-arg-types  # jax-ndarray
        prec=prec,
        scale=scale,
        symmetric=True,
        bounds=initial_bounds,
        half_shift=half_shift)

  @classmethod
  def create_positive(cls, *, bounds: primitives.BoundsT,
                      prec: int) -> 'QuantOps':
    """Create QuantOps for positive activations clipped to [0, bounds].

    Args:
      bounds: The upper bound to clip the activations.
      prec: Unsigned int precision for the QuantOps.

    Returns:
      QuantOps for quantizing/dequantizing unsigned activations.
    """
    initial_bounds = bounds
    bounds = jnp.asarray(bounds, SCALE_DTYPE)
    if not DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING:
      bounds += jnp.finfo(SCALE_DTYPE).eps  # to avoid div by 0
    scale = primitives.unsigned_int_bound(prec=prec) / bounds
    # NOTE: stop_gradient is needed here to prevent gradient flow through scale
    # when scale is not a constant, but computed as a function of activations.
    scale = lax.stop_gradient(scale)
    return cls(  # pytype: disable=wrong-arg-types  # jax-ndarray
        prec=prec,
        scale=scale,
        symmetric=False,
        bounds=initial_bounds,
        half_shift=False)  # disable half_shift for positive distribution

  def assert_scale_shape_is(self, *, shape: Union[int, Tuple[int, ...]]):
    # TODO(shivaniagrawal): add option for float scale for fixed bound acts
    # quantization.
    assert self._scale.shape == shape, (
        'scale shape is unexpected, should be %s but got %s' %
        (shape, self._scale.shape))

  def to_quantized(self, x: jnp.ndarray, *,
                   dtype: primitives.jnp_dtype) -> jnp.ndarray:
    """Quantizes the argument to the target format.

    integer: "upscales", rounds or floors and clips.
    floating-point: optionally upscales, then downcasts to target precision.

    Args:
      x: Argument to be quantized.
      dtype: Type of returned quantized value of x. If quantized x is an input
        to a matmul, we might be want to set it to jnp.int8. If quantized x is
        weights stored in memory, same applies. In fake_quant style we might
        prefer to set dtype=SCALE_DTYPE, since quantized x might get constant
        folded with rescale op (`from_quantized`). Please take a look at the
        comment on SCALE_DTYPE.

    Returns:
      Quantized value of x.
    """
    if isinstance(self._prec, _FloatQuant):
      if self._prec.is_scaled:
        x = jnp.multiply(x, self._scale).astype(x.dtype)
      fp_spec = self._prec.fp_spec
      return fp_cast.downcast_sat_ftz(
          x,
          fp_spec.exp_min,
          fp_spec.exp_max,
          fp_spec.sig_bits,
      )
    else:
      if self._symmetric:
        quantize = primitives.round_and_clip_to_signed_int
      else:
        quantize = primitives.floor_and_clip_to_unsigned_int
      scaled_x = jnp.multiply(x, self._scale)
      return quantize(
          scaled_x, prec=self._prec, dtype=dtype, half_shift=self._half_shift)

  # Same as to_quantized but it just "downscales" using the same scale.
  def from_quantized(self, x: jnp.ndarray, *,
                     dtype: primitives.jnp_dtype) -> jnp.ndarray:
    """'Rescales' the quantized value.

    Args:
      x: quantized.
      dtype: return type for rescaled x

    Returns:
      Rescaled x cast to type dtype
    """

    if (isinstance(self._prec, _FloatQuant) and not self._prec.is_scaled):
      return x
    rescaled_x = jnp.divide(x, self._scale)
    return rescaled_x.astype(dtype)

  # Helper fake quantization
  def fake_quant(self,
                 x: jnp.ndarray,
                 *,
                 quantized_type: primitives.jnp_dtype,
                 fake_dependency: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    x_dtype = x.dtype
    quantized_x = self.to_quantized(x, dtype=quantized_type)
    if fake_dependency is not None:
      quantized_x = lax.tie_in(fake_dependency, quantized_x)
    return self.from_quantized(quantized_x, dtype=x_dtype)

  # Assumes weights are unsigned int of precision prec.
  @classmethod
  def create_weights_ops(
      cls,
      w: jnp.ndarray,
      *,
      weight_params: WeightParams,
  ) -> 'QuantOps':
    """Create a QuantOps that can quantize and dequantize a weight tensor.

    Args:
      w: The weights to quantize.
      weight_params: WeightParams Parameters required for weight quantization.

    Returns:
      Quantized and rescaled inputs using fake quant approach.
    """
    weight_bounds = primitives.max_abs_weights(w, axis=weight_params.axis)
    prec = weight_params.prec
    half_shift = weight_params.half_shift
    if isinstance(prec, _FloatQuant):
      ops = cls.create_symmetric_fp(bounds=weight_bounds, fp_quant=prec)
    else:
      ops = cls.create_symmetric(
          bounds=weight_bounds, prec=prec, half_shift=half_shift)


    if weight_params.expected_scale_shape is not None:
      # NOTE: We set keepdim to True when computing weights scale, as a result
      # the axes which are reduced are left in the result as dimensions with
      # size one. User should correctly pass the shape with reduced dimensions
      # set to 1.
      ops.assert_scale_shape_is(shape=weight_params.expected_scale_shape)
    return ops

  # Assumes weights are unsigned int of precision prec.
  @classmethod
  def create_weights_fake_quant(
      cls,
      w: jnp.ndarray,
      *,
      weight_params: WeightParams,
      quantized_type: primitives.jnp_dtype = SCALE_DTYPE,
      fake_dependency: Optional[jnp.ndarray] = None,
      quantize_weights: bool = True,
  ) -> jnp.ndarray:
    """Quantize weights with fake quant approach.

    Args:
      w: The weights to quantize.
      weight_params: WeightParams Parameters required for weight quantization.
      quantized_type: type of intermediate quantized value of weights. Defaults
        to SCALE_DTYPE.
      fake_dependency: dynamic array, quantized weights will have fake
        dependency on. lax.tie_in for more details. This is used in order to
        prevent constant folding of rescale op with quantized weights. Defaults
        to None, in this case  quantized weights would not have a fake
        dependency.
      quantize_weights: whether weights should be quantized or not

    Returns:
      Quantized and rescaled inputs using fake quant approach.
    """
    # TODO(yichi): if weight_params.prec is None or weight_binarize flag True:
    if weight_params.prec is None or not quantize_weights:
      return w

    weight_shape = w.shape


    ops = cls.create_weights_ops(w, weight_params=weight_params)
    weight_rescaled = ops.fake_quant(
        w, quantized_type=quantized_type, fake_dependency=fake_dependency)


    return weight_rescaled

  # TODO(malmaud): rename 'input' to activation here and elsewhere in this file.
  @classmethod
  def create_input_ops(
      cls, inputs: jnp.ndarray, *, hparams: ActHParams,
      bounds_params: Union[get_bounds.GetBounds.Params,
                           get_bounds.DynamicBounds.Params, None]
  ) -> 'QuantOps':
    """Create a QuantOps that can quantize and dequantize an activation tensor.

    Args:
      inputs: The inputs to quantize.
      hparams: Input hyperparameter (ActHParams).
      bounds_params: BoundsParams. Parameters for GetBounds or DynamicBounds.

    Returns:
      Quantized and rescaled inputs using fake quant approach.
    """

    # TODO(shivaniagrawal): investigate why pytype allows types other than
    # ActsBoundT.
    if isinstance(hparams.bounds, int):
      hparams.bounds = float(hparams.bounds)

    # NOTE: if flax module name is None, default name is used.

    # If we want to train with no quantization at first and then turn on
    # GetBounds quantization, we still have to call GetBounds even before
    # quantization is enabled since GetBounds calculates and stores the running
    # statistics that we will use once quantization is enabled. But before
    # quantization is enabled, we want to ignore the returned bounds and just
    # return the original unquantized input. To do so, we take advantage of the
    # fact that GetBounds returns a constant fixed bound for an initial time
    # period and set that initial bound to a special value (-1) to indicate we
    # want to store activation statistics without applying quantization. That
    # will cause clip_bounds will be a tensor of all '-1', which we will check
    # for in a lax.cond call below.

    # TODO(malmaud): Refactor code to separate bounds calculation from tracking
    # activation statistics to avoid the need to rely on special bounds values
    # when disabling quantization.
    if isinstance(hparams.bounds, get_bounds.GetBounds.Hyper):
      if not bounds_params and not isinstance(bounds_params,
                                              get_bounds.GetBounds.Params):
        raise ValueError(
            'act_hparams.bounds is of type GetBounds.Hyper, user must '
            'provide bounds_params, parameters for GetBounds.')
      clip_bounds = get_bounds.GetBounds(
          hyper=hparams.bounds, name=bounds_params.module_name)(
              inputs,
              bounds_params=bounds_params,
          )
    elif isinstance(hparams.bounds, get_bounds.DynamicBounds.Hyper):
      if not bounds_params and not isinstance(bounds_params,
                                              get_bounds.DynamicBounds.Params):
        raise ValueError(
            'act_hparams.bounds is of type DynamicBounds.Hyper, user must '
            'provide bounds_params, parameters for DynamicBounds.')
      clip_bounds = get_bounds.DynamicBounds(
          hyper=hparams.bounds, name=bounds_params.module_name)(
              inputs,
              bounds_params=bounds_params,
          )

    elif isinstance(hparams.bounds, (float, jnp.ndarray)):
      clip_bounds = hparams.bounds
    else:
      assert False, (
          '%s is not a valid type for hparams.bounds, should be float, a list '
          'of floats, DynamicBounds.Hyper or GetBounds.Hyper.' %
          (type(hparams.bounds)))

    if isinstance(hparams.prec, _FloatQuant):
      ops = cls.create_symmetric_fp(bounds=clip_bounds, fp_quant=hparams.prec)
    elif hparams.input_distribution == cls.ActHParams.InputDistribution.SYMMETRIC:
      ops = cls.create_symmetric(
          bounds=clip_bounds, prec=hparams.prec, half_shift=hparams.half_shift)
    elif hparams.input_distribution == cls.ActHParams.InputDistribution.POSITIVE:
      ops = cls.create_positive(bounds=clip_bounds, prec=hparams.prec)
    else:
      assert False, "can't happen."

    if bounds_params and bounds_params.expected_bounds_shape is not None:
      if isinstance(hparams.bounds, get_bounds.GetBounds.Hyper):
        ops.assert_scale_shape_is(shape=bounds_params.expected_bounds_shape)
      else:
        logging.info(
            'Ignoring value of argument expected_scale_shape. Scale for fixed '
            'bounds would be scalar.')
    return ops

  @classmethod
  def create_inputs_fake_quant(
      cls, inputs: jnp.ndarray, *, hparams: ActHParams,
      bounds_params: Union[None, get_bounds.GetBounds.Params,
                           get_bounds.DynamicBounds.Params]
  ) -> jnp.ndarray:
    """Quantize input with fake quant approach.

    Args:
      inputs: The inputs to quantize.
      hparams: Input hyperparameter (ActHParams).
      bounds_params: GetBounds or DynamicBounds Params.

    Returns:
      Quantized and rescaled inputs using fake quant approach.
    """

    if hparams.bounds is None or hparams.prec is None:
      # TODO(lew): support bound-clipping without quantization
      return inputs

    ops = cls.create_input_ops(
        inputs, hparams=hparams, bounds_params=bounds_params)

    quantized_inputs = ops.fake_quant(inputs, quantized_type=SCALE_DTYPE)
    return lax.cond(ops.should_quantize(), lambda _: quantized_inputs,
                    lambda _: inputs, None)

  # When using GetBounds quantization (if hparams.bounds is an instance of
  # GetBounds.Hyper), if we want to disable quantization but continue to
  # collect activation statistics, we have GetBounds return a clip_bounds
  # tensor to all '-1' values as a signal that quantization shoulnd't be
  # applied. See comment on the call to 'GetBounds' above.
  # TODO(malmaud): Find a less hacky way to do this.
  def should_quantize(self) -> jnp.ndarray:
    """Return whether QuantOps should quantize."""
    # We return a scalar jnp.ndarray of dtype bool instead of a Python bool
    # because during the Jax JIT compilation, self._bounds will be a tracer
    # instead of a concrete tensor, which can't be coerced to a Python bool.
    # Since the type of jnp.all is an ndarray, we invert it with '~' instead of
    # 'not'
    return ~jnp.all(self._bounds == -1)

  def get_scale_for_aqt(self, *, allow_per_channel_scales: bool) -> jnp.ndarray:
    """Returns the scale in a shape appropriate for AQT.

    An error is raised if the granularity of the scale factors are incompatible
    with the current AQT implementation and the setting of
    'allow_per_channel_scales'.

    Args:
      allow_per_channel_scales: A boolean indicating whether a separate scale
        factor is allowed for each output channel (True) or if only a scalar
        (ie, per-layer) scale factor is allowed (False).

    Returns:
      Either a scalar array that correspond to a per-layer scale factor, or an
      array of shape (1, num_channels) that correspond to per-channel scale
      factors.
    """

    scale = self._scale

    # If 'scale' is a 1x1x...x1 matrix (ie, only has one element), we
    # canonicalize it to a scalar to simplify the shape-handling code in the AQT
    # implementation.
    if scale.size == 1:
      return scale.reshape(())
    # If the caller requested a a single per-layer scaling factor but the scale
    # factor is non-scalar, raise an error.
    del allow_per_channel_scales
    return scale


PrecisionType = typing.Any


def quantized_dot(*,
                  w: jnp.ndarray,
                  act: jnp.ndarray,
                  quant_type: QuantType,
                  weight_params: QuantOps.WeightParams,
                  act_hparams: Optional[QuantOps.ActHParams],
                  bounds_params: Union[get_bounds.GetBounds.Params,
                                       get_bounds.DynamicBounds.Params, None],
                  prefer_int8_to_int32_dot: bool,
                  dot_precision: Optional[PrecisionType] = None) -> jnp.ndarray:
  """LAX dot with optionally quantized weights and activations.

  Wraps LAX's `Dot
  <https://github.com/google/jax/blob/f65a327c764406db45e95048dfe09209d8ef6d37/jax/_src/lax/lax.py#L632`_
  operator.

  Args:
    w: an array representing weights
    act: an array representing activations
    quant_type: quantization strategy
    weight_params: QuantOps.WeighstParams instance for describing weights
      quantization.
    act_hparams: Optional activation quantization hyperparamers; instance of
      QuantOps.ActHParams. None would mean no activation quantization.
    bounds_params: Optional bounds params for auto activation
      quantization; instance of GetBounds.Params or DynamicBounds.Params.
    prefer_int8_to_int32_dot:  Whether to feed lax.dot inputs with an int8 dtype
      and accumulate to int32 dtype if quantizing to 8bits or 4bits. If False,
      inputs are always foating-point.
    dot_precision: Optional. Either ``None``, which means the default precision
      for the backend, or a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).

  Returns:
    An array containing the result with the same dtype as 'w' and 'act'.

  Raises:
    RuntimeError: 'quant_type' had an unrecognized value.
    TypeError: 'act' and 'w' has different input types.
    ValueError: Shapes of 'act' and 'w' not compatible with quant_type.
  """
  # This code was initially expanded from
  # https://github.com/google/jax/blob/f65a327c764406db45e95048dfe09209d8ef6d37/jax/_src/lax/lax.py#L632
  # We keep the original return-value semantics of lax.dot, which this wraps. In
  # particular, the type of the return value of quantized_dot is the same as the
  # type of the inputs. That means that if the inputs are bfloat16, then the
  # return type of this function will also be bfloat16 even though on current
  # TPUs the underlying bf16*bf16 matrix-multiplication accumulates results to
  # float32. This is potentially undesirable since the user might want the raw
  # float32 result, but it ultimately stems from a limitation of the HLO 'dot'
  # instruction. If that instruction updates to support user-specified output
  # types, we could update quantized_dot accordingly to take a dtype argument to
  # control the return value type. This applies equally to
  # quantized_dynamic_dot_general.
  if not (1 <= act.ndim <= 2 and 1 <= w.ndim <= 2 and
          act.shape[-1] == w.shape[0]):
    raise ValueError('Incompatible shapes for dot: got {} and {}.'.format(
        act.shape, w.shape))
  dot_dimension_numbers = (((act.ndim - 1,), (0,)), ((), ()))
  return quantized_dot_general(
      w=w,
      act=act,
      quant_type=quant_type,
      weight_params=weight_params,
      act_hparams=act_hparams,
      bounds_params=bounds_params,
      prefer_int8_to_int32_dot=prefer_int8_to_int32_dot,
      dimension_numbers=dot_dimension_numbers,
      dot_precision=dot_precision)


def _canonicalize_feature_axes(axis: Union[int, Tuple[int, ...]],
                               ndims: int) -> Tuple[int, ...]:
  """Returns feature axes from reduction axes."""
  if not isinstance(axis, tuple):
    axis = (axis,)
  if not all([dim < ndims and dim >= -ndims for dim in axis]):
    raise ValueError(f'Axis {axis} should correspond to dimensions '
                     f'of the weight.')
  axis = tuple(sorted([(ndims + dim) % ndims for dim in axis]))
  # Feature axes must be either first len(Feature_axis) consecutive
  # dimensions of weight or last len(axis) consecutive axes.
  is_first_consecutive_dims = (axis == tuple(range(len(axis))))
  if not is_first_consecutive_dims:
    raise ValueError(
        f'Reduction axis {axis} should be consecutive first dimensions '
        f'of the weight.')
  return axis


@dataclass
class QuantW:
  quantized_w: jnp.ndarray
  scale: jnp.ndarray


def flaxformer_dot_general(*,
                           act: jnp.ndarray,
                           w: Optional[jnp.ndarray],
                           dimension_numbers: lax.DotDimensionNumbers,
                           weight_params: QuantOps.WeightParams,
                           act_hparams: Optional[QuantOps.ActHParams],
                           bounds_params: Union[None,
                                                get_bounds.DynamicBounds.Params,
                                                get_bounds.GetBounds.Params],
                           dot_precision: Optional[PrecisionType] = None,
                           prefer_int8_to_int32_dot: bool = True,
                           quant_w: Optional[QuantW] = None) -> jnp.ndarray:
  """Flaxformer dot general with optionally quantized weights and activations."""
  input_dtype = act.dtype
  is_weight_quantized = False
  is_act_quantized = False
  (axis, _), (_, _) = dimension_numbers
  act_scale_shape = tuple(
      [act.shape[dim] if dim not in axis else 1 for dim in range(0, act.ndim)])
  if w is not None:
    weight_scale_shape = (1,) * len(axis) + tuple(
        [w.shape[dim] for dim in range(len(axis), w.ndim)])
  elif quant_w is not None:
    weight_scale_shape = (1,) * len(axis) + tuple([
        quant_w.quantized_w.shape[dim]
        for dim in range(len(axis), quant_w.quantized_w.ndim)
    ])
  else:
    raise ValueError('Either of w or quant_w must be not None')

  if act_hparams is not None and act_hparams.prec is not None:
    is_act_quantized = True
    # Calculate 's', the per-column scale factor on activations.
    act_op = QuantOps.create_input_ops(
        act, hparams=act_hparams, bounds_params=bounds_params)
    act_quantized = act_op.to_quantized(act, dtype=input_dtype)

    # Now calculate s^-1.  First we extract s, the activation scale factor,
    # into a  variable called 'act_scale'. We extract it from 'act_op', the
    # QuantOps instance that calculated the scale factors for the activation
    # matrix.
    act_scale = act_op._scale.astype(input_dtype)  # pylint: disable=protected-access
    if act_scale.ndim == 0:
      act_scale = act_scale * jnp.ones(act_scale_shape, act_scale.dtype)
  else:
    # In this case, activations are not being quantized; only weights. There
    # is no need to absorb activation scales into the rows of the weight
    # matrix so 'w_scaled_rows' can just be set to the original weight matrix.
    act_quantized = act
    act_scale = jnp.ones(act_scale_shape, act.dtype)

  if weight_params is not None and weight_params.prec is not None:
    is_weight_quantized = True
    if quant_w is not None:
      weight_quantized = quant_w.quantized_w.astype(input_dtype)
      weight_scale = quant_w.scale
    else:
      # Calculate 'r' from (s^-1) * w
      weight_op = QuantOps.create_weights_ops(w, weight_params=weight_params)
      weight_scale = weight_op._scale.astype(input_dtype)  # pylint: disable=protected-access

      if weight_params.expected_scale_shape:
        shape_utils.assert_shapes_equal(weight_scale.shape,
                                        weight_params.expected_scale_shape)

      # Quantize weight matrix by calculating RoundAndClip(s^-1 * w * t)
      # TODO(malmaud): See comment on 'act_op.to_quantized' above, which
      # applies here as well.
      weight_quantized = weight_op.to_quantized(w, dtype=input_dtype)
  else:
    assert w is not None, ('w can not be None if weight quantization is not '
                           'specified.')
    weight_quantized = w
    weight_scale = jnp.ones(weight_scale_shape, w.dtype)

  weight_prec = None if weight_params is None else weight_params.prec
  metadata_context = contextlib.suppress()
  if flags.FLAGS.metadata_enabled:
    metadata_context = compute_cost_utils.DotMetadataMonkeyPatch(
        lhs_prec=None, rhs_prec=weight_prec, rhs_is_weight=True)
  # Use metadata context to annotate op metadata with quantization info

  # TODO(shivaniagrawal): this is duplicated code with quantized_dot that can
  # be de-duplicated.
  # To decide whether to use an integer-domain dot operation, we first check
  # if the static quantization parameters are compatible with it by seeing if
  # they request that both inputs be quantized 8bits or less. Then check if
  # the dynamic parameters are compatible with it. ie, in a training run with
  # quantization enabled, are we past the activation start step yet.

  # We also do not use int8_to_int32_dot if activation has positive
  # distribution and prec=8, since we would not be able to fit uint8 range in
  # int8.
  # TODO(shivaniagrawal): A proper solution for this would be to have mixed
  # dot(uint8, int8) -> int32 in XLA.

  act_prec = None if act_hparams is None else act_hparams.prec
  act_has_symm_distribution = act_hparams is not None and (
      act_hparams.input_distribution
      == QuantOps.ActHParams.InputDistribution.SYMMETRIC)
  weight_prec = None if weight_params is None else weight_params.prec

  # To decide whether to use an integer-domain dot operation, we first check
  # if the static quantization parameters are compatible with it by seeing if
  # they request that both inputs be quantized 8bits or less. Then check if
  # the dynamic parameters are compatible with it. ie, in a training run with
  # quantization enabled, are we past the activation start step yet.

  # We also do not use int8_to_int32_dot if activation has positive
  # distribution and prec=8, since we would not be able to fit uint8 range in
  # int8.
  # TODO(shivaniagrawal): A proper solution for this would be to have mixed
  # dot(uint8, int8) -> int32 in XLA.
  weight_fits_in_int8 = is_weight_quantized and (weight_prec is not None and
                                                 weight_prec <= 8)
  # is_act_quantized might be an instance of a Jax tracer instead of a
  # Python boolean since it is generally computed from a dynamic input to a
  # JITted Jax function. Thus we use '&' instead of 'and'.
  act_prec_fits_int8 = act_prec is not None and (
      (act_prec == 8 and act_has_symm_distribution) or (act_prec < 8))
  act_fits_in_int8 = is_act_quantized & act_prec_fits_int8
  use_int8_to_int32_dot = prefer_int8_to_int32_dot & weight_fits_in_int8 & act_fits_in_int8

  with metadata_context:
    # Calculate matmul(...)
    # we are not quantizing activations yet, hence not using int8 matmul
    out_quantized = dot_general_aqt(
        act_quantized,
        weight_quantized,
        dimension_numbers=dimension_numbers,
        dot_precision=dot_precision,
        use_int8_to_int32_dot=use_int8_to_int32_dot)

  if is_weight_quantized or is_act_quantized:

    inv_scale = lax.dot_general(
        (1 / act_scale),
        (1 / weight_scale),
        dimension_numbers=dimension_numbers,
    )
    return out_quantized * inv_scale
  else:
    return out_quantized


# TODO(shivaniagrawal): extend it for generic dot_dimenson_numbers
def quantized_dot_general(*,
                          w: Optional[jnp.ndarray],
                          act: jnp.ndarray,
                          quant_type: QuantType,
                          weight_params: QuantOps.WeightParams,
                          act_hparams: Optional[QuantOps.ActHParams],
                          bounds_params: Union[None,
                                               get_bounds.GetBounds.Params,
                                               get_bounds.DynamicBounds.Params],
                          prefer_int8_to_int32_dot: bool,
                          dimension_numbers: lax.DotDimensionNumbers,
                          dot_precision: Optional[PrecisionType] = None,
                          quant_w: Optional[QuantW] = None) -> jnp.ndarray:
  """LAX dot general with optionally quantized weights and activations.

  Wraps LAX's `Dot General
  <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html.

  Args:
    w: an array representing weights
    act: an array representing activations
    quant_type: quantization strategy
    weight_params: QuantOps.WeighstParams instance for describing weights
      quantization.
    act_hparams: Optional activation quantization hyperparamers; instance of
      QuantOps.ActHParams. None would mean no activation quantization.
    bounds_params: Optional bounds params for auto activation
      quantization; instance of GetBounds.Params or DynamicBounds.Params.
    prefer_int8_to_int32_dot:  Whether to feed lax.dot inputs with an int8 dtype
      and accumulate to int32 dtype if quantizing to 8bits or 4bits. If False,
      inputs are always foating-point.
    dimension_numbers: a tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`.
    dot_precision: Optional. Either ``None``, which means the default precision
      for the backend, or a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).
    quant_w: Quantization weights and scales, provided as class QuantW. If
      provided, quantization variables are used directly. Defaults to None.

  Returns:
    An array containing the result with the same dtype as 'w' and 'act'.

  Raises:
    RuntimeError: 'quant_type' had an unrecognized value.
    TypeError: 'act' and 'w' has different input types.
    ValueError: Shapes of 'act' and 'w' not compatible with quant_type.
  """
  supported_contracting_dims = ((act.ndim - 1,), (0,))
  # (lhs_contracting_dims, rhs_contracting_dims)
  if dimension_numbers != (supported_contracting_dims,
                           ((), ())):  # (lhs_batch_dims, rhs_batch_dims)):
    raise ValueError(
        'Quantization is only supported for contracting dimension to be last '
        'dimension of lhs and first dimension of rhs.')
  if quant_type == QuantType.AQT:
    # Let 's' be activation scales and 't' be weight scales. We implement
    # matmul(RoundAndClip(act*s), RoundAndClip(s^-1 * w * t)) *t^-1. In the
    # comments below, we refer to this terminology.
    # lax.dot accepts any combination of 1d and 2d arguments for its lhs and rhs
    # input. To simplify the AQT implementation, we only accept 2d arguments for
    # now.
    if w is not None:
      if act.shape[-1] != w.shape[0]:
        raise ValueError(
            'AQT is currently only implemented for matrix*matrix operations')
      num_input_channels = act.shape[-1]

      if weight_params.axis is None:
        out_channel_shape = ()
      else:
        axes = _canonicalize_feature_axes(weight_params.axis, w.ndim)
        out_channel_shape = tuple(
            [w.shape[i] for i in range(w.ndim) if i not in axes])

      # The ValueError raised in the guard at the beginning of this function
      # should have already checked that the weight matrix has a number of rows
      # equal to the number of channels in the activation.
      assert w.shape[0] == num_input_channels

      # We carry out all intermediate calculations using the same dtype as the
      # inputs. We want to be careful to not take a model configured to be
      # trained in bf16 and accidentally train it in fp32 by virtue of the scale
      # dtype being fp32.
      if act.dtype != w.dtype:
        raise TypeError(
            f'Activations and weight must have the same dtype, but got {act.dtype} and {w.dtype}'
        )
    else:
      assert quant_w is not None
    input_dtype = act.dtype

    is_act_quantized = False
    # In this case, activations will be quantized at some point during training
    # (either now or later) and so we need to gather activation statistics by
    # calling 'QuantOps.create_input_ops', even if activations are not being
    # quantized on this particular training step (see b/174516400).
    if act_hparams is not None and act_hparams.prec is not None:
      # Calculate 's', the per-column scale factor on activations.
      act_op = QuantOps.create_input_ops(
          act, hparams=act_hparams, bounds_params=bounds_params)
      is_act_quantized = act_op.should_quantize()
      # Quantize activation matrix by computing RoundAndClip(w*s)

      # TODO(malmaud): We have to cast quantized activations to an fp format
      # instead of int8 since int8 matmul with int32 accumulation is not yet
      # supported in XLA (and therefore in Jax). See b/170293520. We keep
      # 'act_quantized' in whatever it's original fp format was, typically bf16
      # or fp32, to follow what Fakequant does (see the type cast at the end of
      # QuantOpts.fake_quant).
      act_quantized = act_op.to_quantized(act, dtype=input_dtype)

      # Now calculate s^-1.  First we extract s, the activation scale factor,
      # into a  variable called 'act_scale'. We extract it from 'act_op', the
      # QuantOps instance that calculated the scale factors for the activation
      # matrix.
      act_scale = act_op.get_scale_for_aqt(allow_per_channel_scales=True)
      # act_scale should either be a scalar, corresponding to per-layer
      # quantization, or a matrix with shape (1,.., 1, num_input_channels),
      # corresponding to per-activation-channel scale factors.
      if act_scale.ndim != 0:
        shape_utils.assert_shapes_equal(act_scale.shape,
                                        (1,) * (act_scale.ndim - 1) +
                                        (num_input_channels,))
        # 'w' has one row per column of 'act_scale'. To scale each row of 'w' by
        # the inverse of the corresponding column in 'act_scale', we first have
        # to reshape 'act_scale' from (1, ...,  num_input_channels) to
        # (num_input_channels, 1) so the scale factors will broadcast
        # appropriately across the columns of 'w'.
        act_scale = act_scale.reshape(num_input_channels, 1)
      # Now we calculate s^-1 * w.
      # TODO(shivaniagrawal): This section repeats code from the 'else' block.
      # The code is repeated twice because quantization can either be disabled
      # dynamically by setting the clipping bound to -1 (see comments on
      # 'should_quantize'), or statically by setting the 'prec' hyperparameter
      # to None. This block deals with the dynamic case (hence necessitating the
      # use of the dynamic 'lax.cond') while the 'else' block handles the static
      # case. Ultimately, we should unify them.
      act_quantized, act_scale = lax.cond(
          is_act_quantized,
          lambda _: (act_quantized, act_scale),
          lambda _:  # pylint: disable=g-long-lambda
          (act, jnp.ones_like(act_scale)),
          None)
    else:
      # In this case, activations are not being quantized; only weights. There
      # is no need to absorb activation scales into the rows of the weight
      # matrix so 'w_scaled_rows' can just be set to the original weight matrix.
      act_quantized = act
      act_scale = jnp.array(1.0, dtype=SCALE_DTYPE)

    is_weight_quantized = False
    if weight_params is not None and weight_params.prec is not None:
      is_weight_quantized = True
      if quant_w is not None:
        weight_quantized = quant_w.quantized_w.astype(input_dtype)
        weight_scale = quant_w.scale
      else:
        # Calculate 'r' from (s^-1) * w
        w_scaled_rows = ((1 / act_scale) * w).astype(input_dtype)
        weight_op = QuantOps.create_weights_ops(
            w_scaled_rows, weight_params=weight_params)
        weight_scale = weight_op.get_scale_for_aqt(
            allow_per_channel_scales=True)
        # Similar to 'act_scale' above, the weight_scale can either be a single
        # scalar or be a matrix with shape (1, out_channel_shape), corresponding
        # to a per-channel scale factor for the weight matrix. We verify it
        # here.
        if weight_scale.ndim != 0:
          shape_utils.assert_shapes_equal(weight_scale.shape,
                                          (1,) + out_channel_shape)
          if act.ndim != 0:
            weight_scale_shape = (1,) * (act.ndim - 1) + out_channel_shape
            weight_scale = weight_scale.reshape(weight_scale_shape)

        # Quantize weight matrix by calculating RoundAndClip(s^-1 * w * t)
        # TODO(malmaud): See comment on 'act_op.to_quantized' above, which
        # applies here as well.
        weight_quantized = weight_op.to_quantized(
            w_scaled_rows, dtype=input_dtype)
    else:
      assert w is not None, ('w can not be None if weight quantization is not '
                             'specified.')
      weight_quantized = ((1 / act_scale) * w).astype(input_dtype)
      weight_scale = jnp.array(1.0, dtype=SCALE_DTYPE)

    # Use metadata context to annotate op metadata with quantization info
    act_prec = None if act_hparams is None else act_hparams.prec
    act_has_symm_distribution = act_hparams is not None and (
        act_hparams.input_distribution
        == QuantOps.ActHParams.InputDistribution.SYMMETRIC)
    weight_prec = None if weight_params is None else weight_params.prec

    # To decide whether to use an integer-domain dot operation, we first check
    # if the static quantization parameters are compatible with it by seeing if
    # they request that both inputs be quantized 8bits or less. Then check if
    # the dynamic parameters are compatible with it. ie, in a training run with
    # quantization enabled, are we past the activation start step yet.

    # We also do not use int8_to_int32_dot if activation has positive
    # distribution and prec=8, since we would not be able to fit uint8 range in
    # int8.
    # TODO(shivaniagrawal): A proper solution for this would be to have mixed
    # dot(uint8, int8) -> int32 in XLA.
    weight_fits_in_int8 = is_weight_quantized and (weight_prec is not None and
                                                   weight_prec <= 8)
    # is_act_quantized might be an instance of a Jax tracer instead of a
    # Python boolean since it is generally computed from a dynamic input to a
    # JITted Jax function. Thus we use '&' instead of 'and'.
    act_prec_fits_int8 = act_prec is not None and (
        (act_prec == 8 and act_has_symm_distribution) or (act_prec < 8))
    act_fits_in_int8 = is_act_quantized & act_prec_fits_int8
    use_int8_to_int32_dot = prefer_int8_to_int32_dot & weight_fits_in_int8 & act_fits_in_int8

    metadata_context = contextlib.suppress()
    if flags.FLAGS.metadata_enabled:
      metadata_context = compute_cost_utils.DotMetadataMonkeyPatch(
          lhs_prec=act_prec, rhs_prec=weight_prec, rhs_is_weight=True)

    with metadata_context:
      # Calculate matmul(...)
      out_quantized = dot_general_aqt(
          act_quantized,
          weight_quantized,
          dimension_numbers=dimension_numbers,
          dot_precision=dot_precision,
          use_int8_to_int32_dot=use_int8_to_int32_dot)

    # Scale the columns of the matmul output by computing `matmul(...) * t^-1`
    # TODO(malmaud): Make it possible to return an unquantized matmul to support
    # disabling quantization during initial phase of training.
    #
    # We convert the return value back to input_dtype to ensure the output
    # tensor of quantized_dot has the same dtype as the input tensors to
    # quantized_dot. This explicit cast is necessary since if the inputs are
    # bf16, 'weight_scale' will still fp32 and so multipying out_quantized by
    # (1/weight_scale) will result in a fp32 tensor. We want to convert that
    # back to bf16.

    return (out_quantized * (1 / weight_scale)).astype(input_dtype)

  elif quant_type in (QuantType.FAKE_QUANT, QuantType.FAKE_QUANT_WITH_INT):
    if quant_w is not None:
      raise ValueError(
          'quantized vars are not supported in `fake_quant` style quantization.'
      )
    if quant_type == QuantType.FAKE_QUANT_WITH_INT:
      fake_dependency = act
    # create a dependency on fake input to control constant folding
    else:
      fake_dependency = None

    quantized_type = quant_type.to_jax_type()
    w = QuantOps.create_weights_fake_quant(
        w,
        weight_params=weight_params,
        quantized_type=quantized_type,
        fake_dependency=fake_dependency)

    # TODO(shivaniagrawal): HParams currently allows act_hparams to be NONE.
    # Going forward we can change act_hparams to be required field where if
    # either `prec` or `bounds` is None will result in No activation
    # quantization.
    if act_hparams:
      act = QuantOps.create_inputs_fake_quant(
          act, hparams=act_hparams, bounds_params=bounds_params)

    metadata_context = contextlib.suppress()
    # Use metadata context to annotate op metadata with quantization info
    act_prec = None if act_hparams is None else act_hparams.prec
    weight_prec = None if weight_params is None else weight_params.prec

    if flags.FLAGS.metadata_enabled:
      metadata_context = compute_cost_utils.DotMetadataMonkeyPatch(
          lhs_prec=act_prec, rhs_prec=weight_prec, rhs_is_weight=True)
    with metadata_context:
      out_quantized = lax.dot_general(
          act, w, dimension_numbers=dimension_numbers, precision=dot_precision)
    return out_quantized
  else:
    raise RuntimeError(f'Unsupported quant_type {quant_type}')


class QuantizedDot(nn.Module):
  """Flax module that calculates a quantized 'dot' operation."""

  quant_type: QuantType
  weight_params: QuantOps.WeightParams
  act_hparams: Optional[QuantOps.ActHParams]
  prefer_int8_to_int32_dot: bool
  dot_precision: Optional[PrecisionType] = None

  # TODO(malmaud): Remove the 'padding_mask' field from 'GetBounds.Params'
  # so that 'bounds_params' can be a hyperparameter of this class and
  # only the padding mask will be passed as an argumen to '__call__'.
  @nn.compact
  def __call__(
      self, w: jnp.ndarray, act: jnp.ndarray,
      bounds_params: Union[None, get_bounds.GetBounds.Params,
                           get_bounds.DynamicBounds.Params]
  ) -> jnp.ndarray:
    return quantized_dot(
        w=w,
        act=act,
        bounds_params=bounds_params,
        quant_type=self.quant_type,
        weight_params=self.weight_params,
        act_hparams=self.act_hparams,
        dot_precision=self.dot_precision,
        prefer_int8_to_int32_dot=self.prefer_int8_to_int32_dot)


def quantized_dynamic_dot_general(
    *,
    lhs_act: jnp.ndarray,
    rhs_act: jnp.ndarray,
    quant_type: QuantType,
    lhs_act_hparams: Optional[QuantOps.ActHParams],
    lhs_bounds_params: Union[None, get_bounds.GetBounds.Params,
                             get_bounds.DynamicBounds.Params],
    rhs_act_hparams: Optional[QuantOps.ActHParams],
    rhs_bounds_params: Union[None, get_bounds.GetBounds.Params,
                             get_bounds.DynamicBounds.Params],
    dot_dimension_numbers: lax.DotDimensionNumbers,
    dot_precision: Optional[PrecisionType] = None) -> jnp.ndarray:
  """LAX dot general with optionally quantized dynamic inputs.

  Wraps LAX's `DotGeneral
  <https://github.com/google/jax/blob/f65a327c764406db45e95048dfe09209d8ef6d37/jax/_src/lax/lax.py#L667`_
  operator.

  Args:
    lhs_act: an array representing weights
    rhs_act: an array representing activations
    quant_type: quantization strategy
    lhs_act_hparams: Optional activation quantization hyperparamers for lhs act;
      instance of QuantOps.ActHParams. None means no quantization.
    lhs_bounds_params: Optional get bounds params for lhs act auto
      quantization; instance of GetBounds.Params.
    rhs_act_hparams: Optional activation quantization hyperparamers for rhs act;
      instance of QuantOps.ActHParams. None means no quantization.
    rhs_bounds_params: Optional get bounds params for rhs act auto
      quantization; instance of GetBounds.Params.
    dot_dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
      rhs_batch_dims)).
    dot_precision: Optional. Either ``None``, which means the default precision
      for the backend, or a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).

  Returns:
    An array containing the result.

  Raises:
    RuntimeError: 'quant_type' had an unrecognized value.
    TypeError: Dtypes of lhs_act and rhs_act differed.
  """
  # See comment at the beginning of quantized_dot regarding its return type,
  # which also applies to this function.
  if quant_type == QuantType.AQT:
    # Let 's1' be the scale of 'lhs_act' and 's2' be the scale of 'rhs_act'.  We
    # calculate dot_general(RoundAndClip(s1*lhs_act),
    # RoundAndClip(s2*rhs_act))/(s1*s2). Note that unlike in
    # quantized_dot_general, the scale factors must be scalar (ie, per-tensor
    # quantization) since activations always have static scale factors and so
    # there is no way to absorb per-column scale factor from lhs_act into the
    # rows of rhs_act.

    # See comment on 'input_dtype' in 'quantized_dot'.
    if lhs_act.dtype != rhs_act.dtype:
      raise TypeError('Both activations must have the same dtypes, but got '
                      f'{lhs_act.dtype} and {rhs_act.dtype}')
    input_dtype = lhs_act.dtype

    def get_tensor_and_scale_for_act(
        act: jnp.ndarray, hparams: Optional[QuantOps.ActHParams],
        bounds_params: Union[None, get_bounds.GetBounds.Params,
                             get_bounds.DynamicBounds.Params]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
      # We check whether activations should be quantized based on 'hparams'. If
      # so, we quantize it. If not, we return it unchanged. In either case, we
      # return a scale factor appropriate for unscaling the result of the
      # lax.dot_general.
      if hparams is not None and hparams.prec is not None:
        quant_op = QuantOps.create_input_ops(
            act, hparams=hparams, bounds_params=bounds_params)

        scale = quant_op.get_scale_for_aqt(allow_per_channel_scales=False)
        # Since only per-layer scale factors are supported, we assert that the
        # scale factors are scalars.
        shape_utils.assert_shapes_compatible(scale.shape, ())
        # TODO(malmaud): See comment on 'act_op.to_quantized' earlier in this
        # file, which applies here as well.
        act_quantized = quant_op.to_quantized(act, dtype=input_dtype)

        # TODO(shivaniagrawal): See comment in 'dot_general' above on why this
        # logic is duplicated here and in the 'else' block below.
        return lax.cond(
            quant_op.should_quantize(),  #
            lambda _: (act_quantized, scale),  #
            lambda _: (act, jnp.array(1.0, dtype=SCALE_DTYPE)),  #
            None)
      else:
        # To avoid having a separate code path for every possibility of which of
        # the two input tensors are quantized , we implement not quantizing an
        # activation tensor by simply setting its corresponding scale factor to
        # 1.0.
        return act, jnp.array(1.0, dtype=SCALE_DTYPE)

    lhs_quantized, lhs_scale = get_tensor_and_scale_for_act(
        lhs_act, lhs_act_hparams, lhs_bounds_params)
    rhs_quantized, rhs_scale = get_tensor_and_scale_for_act(
        rhs_act, rhs_act_hparams, rhs_bounds_params)

    metadata_context = contextlib.suppress()
    # Use metadata context to annotate op metadata with quantization info
    lhs_prec = None if lhs_act_hparams is None else lhs_act_hparams.prec
    rhs_prec = None if rhs_act_hparams is None else rhs_act_hparams.prec

    if flags.FLAGS.metadata_enabled:
      metadata_context = compute_cost_utils.DotMetadataMonkeyPatch(
          lhs_prec=lhs_prec, rhs_prec=rhs_prec, rhs_is_weight=False)
    with metadata_context:
      out_quantized = lax.dot_general(
          lhs_quantized,
          rhs_quantized,
          dimension_numbers=dot_dimension_numbers,
          precision=dot_precision)

    # TODO(malmaud): There is an asymmetry here: when we scale the activations
    # to quantize them, the scaling happens in QuantOps.to_quantized. But here,
    # when we dequantize the matrix multiplication of the activations by
    # dividing by the product of the scale factors, we don't use QuantOps. It
    # would be cleaner to do both operations at the same level of abstraction.
    out = (out_quantized / (lhs_scale * rhs_scale)).astype(input_dtype)
  elif quant_type in (QuantType.FAKE_QUANT, QuantType.FAKE_QUANT_WITH_INT):
    # TODO(shivaniagrawal): HParams currently allows act_hparams to be NONE.
    # Going forward we can change act_hparams to be required field where if
    # either `prec` or `bounds` is None will result in No activation
    # quantization.
    if lhs_act_hparams:
      lhs_act = QuantOps.create_inputs_fake_quant(
          lhs_act, hparams=lhs_act_hparams, bounds_params=lhs_bounds_params)

    if rhs_act_hparams:
      rhs_act = QuantOps.create_inputs_fake_quant(
          rhs_act, hparams=rhs_act_hparams, bounds_params=rhs_bounds_params)

    metadata_context = contextlib.suppress()
    # Use metadata context to annotate op metadata with quantization info
    lhs_prec = None if lhs_act_hparams is None else lhs_act_hparams.prec
    rhs_prec = None if rhs_act_hparams is None else rhs_act_hparams.prec

    if flags.FLAGS.metadata_enabled:
      metadata_context = compute_cost_utils.DotMetadataMonkeyPatch(
          lhs_prec=lhs_prec, rhs_prec=rhs_prec, rhs_is_weight=False)
    with metadata_context:
      out = lax.dot_general(
          lhs_act,
          rhs_act,
          dimension_numbers=dot_dimension_numbers,
          precision=dot_precision)
  else:
    raise RuntimeError(f'Unknown quant_type {quant_type}')
  return out


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def quantized_sum(
    x: jnp.ndarray,  #
    axis: Union[int, Tuple[int, ...]],
    keepdims: bool,
    prec: Optional[QuantOps.FloatQuant.FloatPrec]) -> jnp.ndarray:
  """Sums a tensor while quantizing intermediate accumulations.

  This is almost a drop-in replacement for jnp.sum. It only differs in that it
  takes in an 'act_hparams' parameter that controls the quantization of
  intermediate accumulations during the reduction.

  Arguments:
    x: Input, a Jax array
    axis: Which axes to reduce over (see jnp.sum docs)
    keepdims: Whether to keep of drop axes that are reduced (see jnp.sum docs)
    prec: Precision to quantize intermediate to. Currently can only an instance
      of QuantOps.FloatQuant.FloatPrec, corresponding to an unscaled
      floating-point format, or it can be None to indicate no quantization
      should be applied.

  Returns:
    A Jax array with the quantized sum of 'x'.

  """

  # Don't quantize. In this case, this function just wraps jnp.sum.
  if prec is None:
    return jnp.sum(x, axis=axis, keepdims=keepdims)

  # We bypass QuantOps.create_input_ops and directly call
  # QuantOps.create_symmetric_fp because the former creates an instance of
  # GetBounds, which in turn creates state variables to store activation
  # statistics. We do not want to compute statistics for each individual
  # addition within the sum reduction.
  fp_quant = QuantOps.FloatQuant(is_scaled=False, fp_spec=prec)
  quant_ops = QuantOps.create_symmetric_fp(fp_quant=fp_quant, bounds=None)

  if not isinstance(axis, Iterable):
    axis = (axis,)
  axis = utils.normalize_axes(axis, x.ndim)
  dtype = x.dtype

  zero = jnp.zeros((), dtype=dtype)
  x_quantized_sum = lax.reduce(
      x,
      init_values=zero,
      computation=lambda a, b: quant_ops.to_quantized(a + b, dtype=dtype),
      dimensions=axis)

  if keepdims:
    x_quantized_sum = jnp.expand_dims(x_quantized_sum, axis)

  return x_quantized_sum


@quantized_sum.defjvp
def _quantized_sum_jvp(axis, keepdims, prec, primals, tangents):
  (x,), (x_dot,) = primals, tangents
  y = quantized_sum(x, axis=axis, keepdims=keepdims, prec=prec)
  # We calculate the JVP based on the JVP of the original jnp.sum function. That
  # corresponds to using a straight-through-estimator for the quantization
  # operators in 'quantized_sum'.
  _, y_dot = jax.jvp(lambda x: jnp.sum(x, keepdims=keepdims, axis=axis), (x,),
                     (x_dot,))
  return y, y_dot


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4))
def dot_general_aqt(lhs, rhs, dimension_numbers, dot_precision,
                    use_int8_to_int32_dot):
  """Wrapper around lax.dot_general, but with option to use integer dot.

  This function comes equipped with a custom gradient that defines the
  gradient of this function to be the same as the equivalent call to
  lax.dot_general, ignoring casts to and from integer types so that
  quantization-aware-training will work correctly.

  See docstring of lax.dot_general.

  Args:
    lhs: same as in lax.dot_general
    rhs: same as in lax.dot_general
    dimension_numbers: same as in lax.dot_general
    dot_precision: same as in lax.dot_general
    use_int8_to_int32_dot: boolean. If true, inputs to lax.dot_general will be
      cast to int8 and results accumulated to int32, then converted back to the
      original input type.

  Returns:
    Same as lax.dot_general.
  """

  # We define two versions of a dot operation. The first feeds lax.dot_general
  # the original inputs, which are typically bfloat16 or float32. The second
  # converts the inputs to int8 tensors and accumulates results to an int32
  # output.

  def dot_general_fp(ops):
    lhs_, rhs_ = ops
    return lax.dot_general(
        lhs_,
        rhs_,
        dimension_numbers=dimension_numbers,
        precision=dot_precision)

  def dot_general_int(ops):
    lhs_, rhs_ = ops
    input_dtype = lhs_.dtype
    lhs_int = lhs_.astype(jnp.int8)
    rhs_int = rhs_.astype(jnp.int8)
    return lax.dot_general(
        lhs_int,
        rhs_int,
        dimension_numbers=dimension_numbers,
        precision=dot_precision,
        preferred_element_type=jnp.int32).astype(input_dtype)

  return lax.cond(use_int8_to_int32_dot, dot_general_int, dot_general_fp,
                  (lhs, rhs))


@dot_general_aqt.defjvp
def _dot_general_aqt_jvp(dimension_numbers, dot_precision,
                         use_int8_to_int32_dot, primals, tangents):
  """Custom gradient for dot_general_aqt that ignores integer casts."""
  lhs, rhs = primals
  lhs_dot, rhs_dot = tangents
  y = dot_general_aqt(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      dot_precision=dot_precision,
      use_int8_to_int32_dot=use_int8_to_int32_dot)

  def differentiable_dot_general(lhs_, rhs_):
    return lax.dot_general(
        lhs_,
        rhs_,
        dimension_numbers=dimension_numbers,
        precision=dot_precision)

  _, y_tangent = jax.jvp(differentiable_dot_general, (lhs, rhs),
                         (lhs_dot, rhs_dot))
  return y, y_tangent
