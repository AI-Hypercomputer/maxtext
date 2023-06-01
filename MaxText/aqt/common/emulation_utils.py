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

"""Metadata class and utils shared by the TF and JAX implementations."""
import enum


# the bias of exponent in bfloat16/float32.
EXPONENT_BIAS = 127

# Supported rounding mode.
ROUND_AWAY_FROM_ZERO = 'round_away_from_zero'
ROUND_TO_NEAREST_EVEN = 'round_to_nearest_even'
ROUND_STOCHASTIC = 'round_stochastic'


@enum.unique
class EmulatedFromatsEnum(enum.Enum):
  """Enum for FP8 formats with rounding."""
  # E5M2B15 with round-to-nearest-even
  E5M2B15_RTNE = 'e5m2b15_rtne'
  # E5M2B15 with stochastic rounding
  E5M2B15_STOC = 'e5m2b15_stoc'

  # E4M3B11 with round-to-nearest-even
  E4M3B11_RTNE = 'e4m3b11_rtne'
  # E4M3B11 with stochastic rounding
  E4M3B11_STOC = 'e4m3b11_stoc'

  # E4M3B7 with round-to-nearest-even
  E4M3B7_RTNE = 'e4m3b7_rtne'
  # E4M3B7 with stochastic rounding
  E4M3B7_STOC = 'e4m3b7_stoc'

  # The following uses xla reduce precision ops.
  # E8M2B127 with round-to-nearest-even
  E8M2B127_RTNE = 'e8m2b127_rtne'
  # E8M3B127 with round-to-nearest-even
  E8M3B127_RTNE = 'e8m3b127_rtne'
  # E8M4B127 with round-to-nearest-even
  E8M4B127_RTNE = 'e8m4b127_rtne'
  # E8M5B127 with round-to-nearest-even
  E8M5B127_RTNE = 'e8m5b127_rtne'
  # E8M6B127 with round-to-nearest-even
  E8M6B127_RTNE = 'e8m6b127_rtne'

  # INT8 with rounding formats
  # INT8 with round-to-nearest-even
  INT8_RTNE = 'int8_rtne'
  # INT8 with stochastic rounding
  INT8_STOC = 'int8_stoc'

  # Not emulated formats. placed here for comparison.
  BFLOAT16 = 'bfloat16'
  INT8 = 'int8'


class FPMetadata():
  """Defines floating point formats.

  The max_exp and mix_exp are normal exponent. Subnormals are handled
  separately.
  """

  def __init__(self, name, exponent_bits, mantissa_bits, min_exp, max_exp,
               support_inf, rounding_mode):
    self.name = name
    self.exponent_bits = exponent_bits
    self.mantissa_bits = mantissa_bits
    self.min_exp = min_exp
    self.max_exp = max_exp
    self.support_inf = support_inf
    self.rounding_mode = rounding_mode

  name = ''
  exponent_bits = 8
  mantissa_bits = 7
  min_exp = -126
  max_exp = 128
  support_inf = True
  rounding_mode = ROUND_TO_NEAREST_EVEN


def get_metadata(target_format):
  """Return a FPMetadata based on the format."""
  if target_format == EmulatedFromatsEnum.E5M2B15_RTNE:
    return FPMetadata(target_format, 5, 2, -14, 15, True, ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E5M2B15_STOC:
    return FPMetadata(target_format, 5, 2, -14, 15, True, ROUND_STOCHASTIC)
  elif target_format == EmulatedFromatsEnum.E4M3B11_RTNE:
    return FPMetadata(target_format, 4, 3, -10, 4, False, ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E4M3B11_STOC:
    return FPMetadata(target_format, 4, 3, -10, 4, False, ROUND_STOCHASTIC)
  elif target_format == EmulatedFromatsEnum.E4M3B7_RTNE:
    return FPMetadata(target_format, 4, 3, -6, 8, False, ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E4M3B7_STOC:
    return FPMetadata(target_format, 4, 3, -6, 8, False, ROUND_STOCHASTIC)
  elif target_format == EmulatedFromatsEnum.E8M2B127_RTNE:
    return FPMetadata(target_format, 8, 2, -126, 127, True,
                      ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E8M3B127_RTNE:
    return FPMetadata(target_format, 8, 3, -126, 127, True,
                      ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E8M4B127_RTNE:
    return FPMetadata(target_format, 8, 4, -126, 127, True,
                      ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E8M5B127_RTNE:
    return FPMetadata(target_format, 8, 5, -126, 127, True,
                      ROUND_TO_NEAREST_EVEN)
  elif target_format == EmulatedFromatsEnum.E8M6B127_RTNE:
    return FPMetadata(target_format, 8, 6, -126, 127, True,
                      ROUND_TO_NEAREST_EVEN)
  else:
    raise ValueError('Not supported format: ' + target_format)


def is_fp8_format(target_format):
  """Return if the target_format is a supported FP8 format."""
  return target_format in [
      EmulatedFromatsEnum.E5M2B15_RTNE, EmulatedFromatsEnum.E5M2B15_STOC,
      EmulatedFromatsEnum.E4M3B11_RTNE, EmulatedFromatsEnum.E4M3B11_STOC,
      EmulatedFromatsEnum.E4M3B7_RTNE, EmulatedFromatsEnum.E4M3B7_STOC
  ]


def is_e8_format(target_format):
  """Return if the target_format has 8-bit exponent."""
  return get_metadata(target_format).exponent_bits == 8


def get_max_number_from_mantissa_and_max_exp(mantissa_bits, max_exp):
  """Returns the maximum possible number (not inf) allowed by the format."""
  # NOTE(dotzel): max_mantissa_value is always 1.1111111...1 where number of
  # fractional 1s is mantissa_bits. In the limit mantissa_bits => inf, this exp
  # => 2. It only deviates from 2 by its LSB, 2 ** (-mantissa_bits).
  max_mantissa_value = 2 - 2 ** (-mantissa_bits)
  return (2**max_exp) * max_mantissa_value


def get_max_number(target_format):
  """Returns the maximum possible number (not inf) allowed by the format."""
  fp_metadata = get_metadata(target_format)
  return get_max_number_from_mantissa_and_max_exp(
      fp_metadata.mantissa_bits, fp_metadata.max_exp)


