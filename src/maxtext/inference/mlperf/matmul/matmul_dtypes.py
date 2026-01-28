# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""matrix multiplication data types"""


import jax

from maxtext.inference.mlperf.matmul import timing_util

if __name__ == "__main__":
  _PROFILE = False
  MATMUL_SIZES = [(250, 2048)]

  _INT4 = jax.numpy.int4
  _INT8 = jax.numpy.int8
  _DEFAULT = jax.numpy.bfloat16

  def f(X, Y):
    return jax.lax.batch_matmul(X, Y)

  f_jit = jax.jit(f)

  num_matmuls, matrix_size = MATMUL_SIZES[0]

  for dtypeA, dtypeB in [
      (_INT4, _INT4),
      (_INT4, _INT8),
      (_INT8, _INT4),
      (_INT8, _INT8),
      (_INT8, _DEFAULT),
      (_DEFAULT, _DEFAULT),
  ]:
    A = jax.numpy.ones((num_matmuls, matrix_size, matrix_size), dtype=dtypeA)
    B = jax.numpy.ones((num_matmuls, matrix_size, matrix_size), dtype=dtypeB)

    print(f"A, B shape is {f(A, B).shape}. A dtype is {A.dtype}, B dtype is {B.dtype} and prod type is {f(A, B).dtype}")
    timing_util.simple_timeit(f_jit, A, B, task="matmul_" + str(matrix_size), enable_profile=_PROFILE)
