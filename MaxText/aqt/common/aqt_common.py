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

"""Common utility functions used for both TF and Jax."""

from typing import Optional, Sequence, Union

from aqt.common import aqt_config
from aqt.common import emulation_utils

_get_max_number_float = emulation_utils.get_max_number_from_mantissa_and_max_exp


def check_shapes_conformal(actual: Sequence[Optional[int]],
                           expected: Sequence[Optional[int]]):
  """Checks that actual shape conforms to expected.

  The `actual` shape conforms to the `expected` shape if it is equal in
  rank and has identical sizes in all non-None dimensions of `expected`.

  Note this is not a symmetric relation, like tf.ensure_shape. A tensor
  of completely-unknown size meets this `tf.ensure_shape` and
  `tf.TensorShape.is_compatible_with` relation, but is not conformal
  to any tensor with a known dimension.

  Args:
    actual: a list of ints for a tensor shape, with None for unknown dim.
    expected: the requirements for the shape, with None for no requirement.

  Raises:
    ValueError if the actual shape does not conform to expectations.
  """
  expected_shape = [
      ax if ax is not None else given for ax, given in zip(actual, expected)
  ]
  compatible_rank = len(actual) == len(expected)
  compatible_size = list(actual) == expected_shape
  if not compatible_rank or not compatible_size:
    raise ValueError(f'actual shape ({actual}) must be compatible with '
                     f'expected ({expected})')


def _get_clip_bound_int(config: aqt_config.IntQuantConfig):
  """Returns the clip bound when using integer values."""
  assert config.bits <= 23, 'Too many bits, float32 has less precision.'
  bucket_count = 2.0**config.bits
  if config.preserve_zero:
    bucket_count -= 1.0
  return bucket_count / 2.0


def get_clip_bound(
    config: Union[aqt_config.IntQuantConfig, aqt_config.SmallFloatConfig],
) -> float:
  """Returns the clip bound for IntQuantConfig or SmallFloatConfig."""
  config.validate()
  if isinstance(config, aqt_config.IntQuantConfig):
    return _get_clip_bound_int(config)
  elif isinstance(config, aqt_config.SmallFloatConfig):
    return _get_max_number_float(config.mantissa_bits, config.max_exp)
  else:
    raise ValueError(
        '_get_clip_bound called without quantization or emulation.')


def safe_clip_bound(config: aqt_config.IntQuantConfig) -> float:
  """Returns clip-safe clip bound.

  `get_clip_bound()` can be used for rescaling, but not clipping. This is
  because round-to-nearest-even (RTNE) behavior may round the clipped value out
  of range.  For instance, an 8-bit signed integer has a max value of 127; a
  clip of 127.5 may be appropriate for scaling (as we'd like the range [126.5,
  127.5) to map to the integer 127), but the actual clipping of float values
  needs to clip to just below 127.5, to something like 127.499999, such that
  RTNE rounds into range.

  Note: This is not necessary for SmallFloatConfig because the get_clip_bound
  returns the maximum possible value already.

  Args:
    config: the integer quantization config

  Returns:
    A value slightly below `get_clip_bound()`.
  """
  cb_unsafe = get_clip_bound(config)
  # It is essential that x is not clipped to cb_unsafe, but a smaller number.
  # Clipping to cb_unsafe moves the return value to a new and incorrect bucket.
  # The max supported (config.bits)-sized signed int is (2**(config.bits-1)-1).
  # The floor of cb_unsafe is larger than this limit, whereas using
  # cb_unsafe * (1 - eps) for some float32-representable eps >= 2**-23
  # accomplishes this. On the other hand, any eps above bucket resolution
  # 2**(-config.bits) would overcorrect, underutilizing integer range.
  cb = cb_unsafe - 2.0**(-20 + config.bits)
  assert cb < cb_unsafe, 'Internal error, epsilon too small.'
  return cb
