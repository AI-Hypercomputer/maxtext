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

"""Emulates various fp8 formats in bfloat16 or float32."""

from aqt.common import emulation_utils
import tensorflow as tf

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import

FPMetadata = emulation_utils.FPMetadata
EXPONENT_BIAS = emulation_utils.EXPONENT_BIAS
get_max_number = emulation_utils.get_max_number



def get_exponent(t):
  """Get exponent of a tensor."""

  assert t.dtype in [tf.bfloat16, tf.float32]

  t_shape = t.shape

  # Make MSB zero
  t = tf.abs(t)

  if t.dtype == tf.bfloat16:
    t = tf.bitcast(t, tf.int16)
    t = tf.bitwise.right_shift(t, tf.constant(7, dtype=tf.int16))
    assert t.dtype == tf.int16
  else:
    t = tf.bitcast(t, tf.int32)
    t = tf.bitwise.right_shift(t, tf.constant(23, dtype=tf.int32))
    assert t.dtype == tf.int32

  assert t.shape == t_shape
  return t - EXPONENT_BIAS


def flush_to_zero(t, exponent, min_exp):
  """Handles underflow when the exponent is less than min_exp.

  Currently flush all underflow to zeros and does not handle subnormals.

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32
    exponent: the exponents of t.
    min_exp: the allowed minimum exponents.

  Returns:
    a tensor in the same dtype with underflow flushed to zeros.
  """

  assert t.shape == exponent.shape
  assert t.dtype in [tf.bfloat16, tf.float32]

  if t.dtype == tf.bfloat16:
    int_dtype = tf.int16
  else:
    int_dtype = tf.int32

  assert exponent.dtype == int_dtype
  if isinstance(min_exp, tf.Tensor):
    min_exp = tf.cast(min_exp, dtype=int_dtype) + EXPONENT_BIAS
  else:
    min_exp = tf.constant(min_exp, dtype=int_dtype) + EXPONENT_BIAS
  zeros = tf.zeros_like(t, dtype=t.dtype)
  mask = tf.greater_equal(exponent + EXPONENT_BIAS, min_exp)
  # The following might change the sign of zeros.
  return tf.where(mask, t, zeros)


def get_max_frac_in_binary(mantissa_bits, dtype):
  """Returns the maximum fraction in binary format (dtype-aware)."""
  if dtype == tf.float32:
    if mantissa_bits == 2:
      return 0x00600000
    else:
      return 0x00700000
  elif dtype == tf.bfloat16:
    if mantissa_bits == 2:
      return 0x0060
    else:
      return 0x0070


def handle_overflow(t, exponent, max_exp, mantissa_bits):
  """Handles exponent overflow of the input tensor.

  Since the definition of NaN/inf is format dependent. This function assumes
  no such special values (NaN or inf).

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32
    exponent: the exponents of t.
    max_exp: the allowed maximum exponents.
    mantissa_bits: the number of mantissa bits in the format.

  Returns:
    a tensor in the same dtype with overflow clamped at the max possible value.
  """

  assert t.shape == exponent.shape
  assert t.dtype in [tf.bfloat16, tf.float32]

  if t.dtype == tf.bfloat16:
    int_dtype = tf.int16
  else:
    int_dtype = tf.int32

  if isinstance(max_exp, tf.Tensor):
    max_exp = tf.cast(max_exp, dtype=int_dtype) + EXPONENT_BIAS
  else:
    max_exp = tf.constant(max_exp, dtype=int_dtype) + EXPONENT_BIAS

  assert exponent.dtype == int_dtype
  max_frac = get_max_frac_in_binary(mantissa_bits, t.dtype)
  if t.dtype == tf.bfloat16:
    abs_max_value = tf.bitwise.left_shift(max_exp,
                                          tf.constant(7, dtype=tf.int16))
    abs_max_value = tf.bitwise.bitwise_or(abs_max_value, max_frac)
    abs_max_value = tf.bitcast(abs_max_value, tf.bfloat16)
  else:
    abs_max_value = tf.bitwise.left_shift(max_exp,
                                          tf.constant(23, dtype=tf.int32))
    abs_max_value = tf.bitwise.bitwise_or(abs_max_value, max_frac)
    abs_max_value = tf.bitcast(abs_max_value, tf.float32)

  return tf.clip_by_value(
      t, clip_value_min=-abs_max_value, clip_value_max=abs_max_value)


def get_mask(dtype, num_lsbs):
  """Get mask with number of LSBs set to 0."""

  assert dtype in [tf.uint16, tf.uint32]
  assert isinstance(num_lsbs, tf.Tensor)

  if dtype == tf.uint16:
    mask = 0xFFFF
  else:
    mask = 0xFFFFFFFF

  mask -= tf.math.pow(2, tf.cast(num_lsbs, tf.int32)) - 1
  mask = tf.cast(mask, dtype)
  return mask


def _round_stochastic(t, mask_bits):
  """Round to a near value with a probability dependent on the proximity.

  Implementations-wise, add a rounding bias that is a random number in [0, 1),
  followed by a truncation. The random number is based on the number of mask
  bits. For example, converting bf16 (E8M7) to E5M2, the number of mask bits is
  5. The random value is a relative term between [0, 1.0) that are
  representable by 5 mantissa bits. That is, the rounding bias is a uniform
  distribution in [0.0000000, 0.0100000) for the mantissa bits. See the below
  example for more details.

  Use the following example, the values are in the format of 1.x * 2^y.
  exponent: y=4 (2^4=16)
  mantissa: 2 bits (in binary): 1.00, 1.01, 1.10, and 1.11.
  representable values: 16, 20, 24, 28. The next representable value is 32
  with an exponent of 4 and a mantissa of 0 (1.00 * 2^5 = 32).

  Take a value of 17 as an example.  The rounding bias is
  [0.0000000, 0.0100000) * 2^4, which is [0, 4). The result of the addition is
  [17, 21), and the following truncation produces a mix of 75% of 16 and 25% of
  20.

  For a negative value of -17, the rounding bias is the same but used in
  subtraction. The result of the subtraction is (-21, -17], and the
  following truncation produces a mix of 75% of -16 and 25% of -20.

  https://en.wikipedia.org/wiki/Rounding#Stochastic_rounding

  Args:
    t: the tensor in integer dtype.
    mask_bits: the number of mask bits.

  Returns:
    a tensor that is rounded with stochastic rounding.
  """
  t_shape = t.shape
  int_dtype = t.dtype

  # add rounding bias.
  if t.dtype is tf.uint16:
    upper = tf.math.pow(2, 7)
  elif t.dtype is tf.uint32:
    upper = tf.math.pow(2, 23)
  rounding_bias = tf.cast(
      tf.random.uniform(
          t_shape,
          minval=0,
          maxval=upper,
          dtype=tf.dtypes.int32,
          seed=0,
      ), int_dtype)
  # Use mask to deal with subnormals.
  rounding_bias_mask = tf.bitwise.left_shift(
      tf.constant(1, int_dtype), mask_bits) - 1
  rounding_bias = tf.bitwise.bitwise_and(rounding_bias, rounding_bias_mask)
  t = t + rounding_bias

  # Truncation
  mask = get_mask(int_dtype, mask_bits)
  t = tf.bitwise.bitwise_and(t, mask)
  assert t.shape == t_shape

  return t


def _round_to_nearest_even(t, mask_bits):
  """Round to nearest even value.

  Add a rounding bias for positive, subtract the bias for negative. The rounding
  bias is depending on whether the value is odd or even. For odd values, the
  addition is "half", which depends on the number of mantissa bits. For 2 bits,
  the "half" value is 0.125 (the third mantissa bit). If the mantissa overflows,
  the mechanism is to increment the exponent. The following is an example. For
  even values, the addition is the value that is smaller than but closest to 0.5
  (for example, 0.4999).

  Use the following example, the values are in the format of 1.x * 2^y.
  exponent: y=3 (2^3=8)
  mantissa: 2 bits: 1.0, 1.25, 1.5, and 1.75. Among these four values, 1.0
            (0x00) and 1.5 (0x10) are considered even values.
  representable values: 8, 10, 12, 14. The next representable value is 16
  with an exponent of 4 and a mantissa of 0. Among these values, 8, 12, and 16
  are even values.
  values in [8, 9] become 8 (round-to-nearest-even).
  values in (9, 11) become 10.
  values in [11, 13] becomes 12
  values in (13, 15) become 14.
  values in [15, 16] become 16 (increment on the exponent).

  https://en.wikipedia.org/wiki/Rounding#Round_half_to_even

  Args:
    t: the tensor in integer dtype.
    mask_bits: the number of mask bits.

  Returns:
    a tensor that is rounded with round-to-nearest-even.
  """
  t_shape = t.shape
  int_dtype = t.dtype

  # add rounding bias.
  rounding_bias = tf.bitwise.left_shift(
      tf.constant(1, int_dtype), mask_bits - 1) - 1
  even_mask = tf.bitwise.left_shift(tf.constant(1, int_dtype), mask_bits)
  is_even = tf.bitwise.bitwise_and(t, even_mask)
  # The rounding bias depends on whether the value is odd or even.
  rounding_bias = tf.where(
      tf.equal(is_even, tf.zeros_like(is_even)), rounding_bias,
      rounding_bias + 1)

  t = t + rounding_bias

  # Truncation
  mask = get_mask(int_dtype, mask_bits)
  t = tf.bitwise.bitwise_and(t, mask)
  assert t.shape == t_shape

  return t


def _round_away_from_zero(t, mask_bits):
  """Round to nearest value with ties away from zero.

  Add half for positive, subtract half for negative. The value of "half"
  depends on the number of mantissa bits. For 2 bits, the "half" value is
  0.125 (the third mantissa bit). If the mantissa overflows, the mechanism is
  to increment the exponent. The following is an example.
  exponent: 3 (2^3)
  mantissa: 2 bits.
  representable values: 8, 10, 12, 14. The next representable value is 16
  with an exponent of 4 and a mantissa of 0.
  values in [8, 9) become 8.
  values in [9, 10] become 10. (round-half-away-from-zero).
  values in [14, 15) become 14.
  values in [15, 16] become 16 (increment on the exponent).

  https://en.wikipedia.org/wiki/Rounding#Round_half_away_from_zero

  Args:
    t: the tensor in integer dtype.
    mask_bits: the number of mask bits.

  Returns:
    a tensor that is rounded with round-half-away-from-zero.
  """
  t_shape = t.shape
  int_dtype = t.dtype

  # add half
  half = tf.bitwise.left_shift(tf.constant(1, int_dtype), mask_bits - 1)
  t = t + half

  # Truncation
  mask = get_mask(int_dtype, mask_bits)
  t = tf.bitwise.bitwise_and(t, mask)
  assert t.shape == t_shape

  return t


def rounding(t, mask_bits, rounding_mode):
  """Rounding given the mask bits."""
  t_shape = t.shape
  dtype = t.dtype

  if t.dtype == tf.bfloat16:
    int_dtype = tf.uint16
  else:
    int_dtype = tf.uint32
  # Use a integer dtype as a container.
  t = tf.bitcast(t, int_dtype)
  assert t.shape == t_shape

  if rounding_mode == emulation_utils.ROUND_AWAY_FROM_ZERO:
    t = _round_away_from_zero(t, mask_bits)
  elif rounding_mode == emulation_utils.ROUND_TO_NEAREST_EVEN:
    t = _round_to_nearest_even(t, mask_bits)
  elif rounding_mode == emulation_utils.ROUND_STOCHASTIC:
    t = _round_stochastic(t, mask_bits)
  else:
    raise NotImplementedError

  # convert to bfloat16/float32
  t = tf.bitcast(t, dtype)
  assert t.shape == t_shape
  return t


def handle_mantissa(t, mantissa_bits, min_exp, rounding_mode):
  """Handles the mantissa of t with the given mantissa bits.

  This function handles the mantissa of t by the given mantissa bits with
  the support of subnormals.

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32
    mantissa_bits: the number of allowed mantissa bits.
    min_exp: an integer number for the minimum exponent.
    rounding_mode: the rounding mode.

  Returns:
    a tensor in the same dtype with manipulated mantissa.
  """

  assert t.dtype in [tf.bfloat16, tf.float32]
  if t.dtype == tf.bfloat16:
    int_dtype = tf.uint16
  else:
    int_dtype = tf.uint32

  exponent = get_exponent(t)
  allowed_mantissa_bits = tf.cast(
      tf.subtract(
          tf.cast(mantissa_bits, exponent.dtype),
          tf.clip_by_value(
              tf.subtract(tf.cast(min_exp, exponent.dtype), exponent), 0,
              mantissa_bits)),
      dtype=int_dtype)

  if t.dtype is tf.bfloat16:
    mask_bits = 7 - allowed_mantissa_bits
  elif t.dtype is tf.float32:
    mask_bits = 23 - allowed_mantissa_bits

  return rounding(t, mask_bits, rounding_mode)


def static_handle_exponent(t, min_exp, max_exp, mantissa_bits):
  """Handles the exponents of inputs by a limiting range [min_exp, max_exp].

  This function limits the range of exponents of inputs in a static way.

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32
    min_exp: the allowed minimum exponents.
    max_exp: the allowed maximum exponents.
    mantissa_bits: the number of mantissa bits in the format.

  Returns:
    a tensor in the same dtype with a limited range of exponents.
  """
  exponent = get_exponent(t)
  value = flush_to_zero(t, exponent, min_exp)
  value = handle_overflow(value, exponent, max_exp, mantissa_bits)
  return value


# Emulated fp8 subnormal handling
# Using E5M2B15 as an example. E5M2B15: 1 sign bit, 5 exponent bits,
# 2 mantissa bits, and the bias is 15.
# The smallest positive normalized number is 2 ** -14: exponent is 1 and
# mantissa are all zeros. That is, 0x04. For subnormals, the exponent is zero,
# and the mantissa is non-zero. This emulated version handles subnormals by
# masking away more and more low order bits as the f32 exponent falls.
# e4m3 variants have different exponent, mantissa, and bias. Therefore, the
# representable subnormals differ.
def emulated_fp(t, fp_metadata):
  """Emulates enumerics in bfloat16 or float32 given the FPMetadata.

  While zeros, NaN, and subnormals are supported, the support of inf is
  specified by the FPMetadata. This function is generic to support a variety
  of FPn formats.

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32.
    fp_metadata: a FPMetadata object.

  Returns:
    a same dtype tensor that emulates floating points.
  """
  assert isinstance(fp_metadata, FPMetadata)
  assert t.dtype in [tf.bfloat16, tf.float32]

  with tf.name_scope('emulated_fp'):
    v = handle_mantissa(
        t,
        mantissa_bits=fp_metadata.mantissa_bits,
        min_exp=fp_metadata.min_exp,
        rounding_mode=fp_metadata.rounding_mode)
    v = static_handle_exponent(
        v,
        min_exp=fp_metadata.min_exp - fp_metadata.mantissa_bits,
        max_exp=fp_metadata.max_exp,
        mantissa_bits=fp_metadata.mantissa_bits)

    # Supporting inf is optional in some of the formats. If inf is supported,
    # the emulation preserves nan and inf and changes the input whose value is a
    # finite number. Otherwise, the emulation changes the input whose value is
    # not nan (and treat inf as a valid number.)
    def cond(t):
      if fp_metadata.support_inf:
        return tf.math.is_finite(t)
      else:
        return tf.math.logical_not(tf.math.is_nan(t))

    return tf.where(cond(t), v, t)


# Emulated E8Mn, n being 1 to 6.
# Use xla reduce precision ops for the emulation.
def emulated_e8mn(t, fp_metadata):
  """Emulates E8Mn in bfloat16 or float32 given the FPMetadata.

  Special value support depends on the underlying hardware.

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32.
    fp_metadata: a FPMetadata object.

  Returns:
    a same dtype tensor that emulates floating points.
  """
  assert isinstance(fp_metadata, FPMetadata)
  assert fp_metadata.exponent_bits == 8
  assert fp_metadata.mantissa_bits <= 6 and fp_metadata.mantissa_bits >= 0
  assert t.dtype in [tf.bfloat16, tf.float32]

  with tf.name_scope('emulated_e8mn'):
    return xla.reduce_precision(t, fp_metadata.exponent_bits,
                                fp_metadata.mantissa_bits)

