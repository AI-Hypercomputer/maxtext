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

"""General util functions commonly used across different models."""


def get_fp_spec(sig_bit: int, exp_bit: int):
  """Create fp spec which defines precision for floating-point quantization.

  Args:
    sig_bit: the number of bits assigned for significand.
    exp_bit: the number of bits assigned for exponent.

  Returns:
    fp spec
  """
  exp_bound = 2**(exp_bit - 1) - 1
  prec = {'exp_min': -exp_bound, 'exp_max': exp_bound, 'sig_bits': sig_bit}
  return prec
