# Copyright 2024 Google LLC
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

import functools
from typing import Callable

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

import timeit

PROFILE_CAPTURE_PATH = "/tmp/tensorboard"

def matmul_flops(m: int, k: int, n: int):
  return 2 * m * k * n

def matmul_membw(m: int, k: int, n: int, mk_dtype: jnp.dtype, kn_dtype: jnp.dtype, mn_dtype: jnp.dtype):
  return (m * k * np.dtype(mk_dtype).itemsize + 
          k * n * np.dtype(kn_dtype).itemsize + 
          m * n * np.dtype(mn_dtype).itemsize)


def matmul_flops_intensity(m: int, k: int, n: int, mk_dtype: jnp.dtype, kn_dtype: jnp.dtype, mn_dtype: jnp.dtype):
  flops = matmul_flops(m, k, n)
  membw = matmul_membw(m, k, n, mk_dtype, kn_dtype, mn_dtype)
  return flops / membw


def benchmark(f, ntrials: int = 100, capture_profile: bool = True):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials

    if capture_profile:
      with jax.profiler.trace("/tmp/tensorboard"):
        jax.block_until_ready(f(*args, **kwargs))

    return time
  return run


def analyze_matmul(m: int, k: int, n: int, mk_dtype: jnp.dtype, kn_dtype: jnp.dtype, mn_dtype: jnp.dtype,
                   mm_func):
  x = jnp.ones((m, k), dtype=mk_dtype)
  y = jnp.ones((k, n), dtype=kn_dtype)
  time = benchmark(mm_func)(x, y, preferred_element_type=mn_dtype)
  mm_flops = matmul_flops(m, k, n)
  mm_hbm_bw = matmul_membw(m, k, n, mk_dtype, kn_dtype, mn_dtype)

  print(f"----- ({m}, {k})_{np.dtype(mk_dtype).name} x"
        f" ({k}, {n})_{np.dtype(kn_dtype).name} x"
        f" ({m}, {n})_{np.dtype(mn_dtype).name} -----")
  print(f"Matmul FLOP/s: {mm_flops / time: 0.2e}")
  print(f"Matmul HBM BW - GB/s: {mm_hbm_bw / time:.2e}")
  print(f"Matmul arithmetic intensity: {matmul_flops_intensity(m, k, n, mk_dtype, kn_dtype, mn_dtype): .2f}")
  print()


"""
GEMM is simply expressed as the matrix multiplication process of two 2D matrices, A x B = C, 
where the shape of A is [M, K], the shape of B is [K, N], and the shape of C is [M, N].
4 GEMMs with different data types need to be tested:

1. BF16/FP16 GEMM: A, B, and C are all of the same precision (BF16 or FP16), and the MFU is estimated to reach 80%.

2. Mixed Precision GEMM (W8A16, W4A16): A and C are high precision (BF16 or FP16), B is low precision (INT8 or INT4), 
   B is dequantized from low precision to high precision data, and then GEMM calculations of A and B are performed.

2.1. INT8 x BF16/FP16, Per Channel, MFU is estimated to reach 70%.
2.2. INT4 x BF16/FP16, Per Group, group size = 128, MFU estimated to reach 60%.

3. FP8/INT8 GEMM: A, B, C are all of the same precision (FP8 or INT8), MFU estimated to reach 80%.

GEMM MxNxK specifications:
8000x2304x6144
8000x6144x6144
8000x6144x3072
8000x3072x3840
"""
def main():
    matmul_specs = [
      (8000, 2304, 6144),
      (8000, 6144, 6144),
      (8000, 6144, 3072),
      (8000, 3072, 3840),
    ]
    print("================ Trillium matmul benchmark ===================")
    mm = jnp.matmul
    for spec in matmul_specs:
      m, k, n = spec
      analyze_matmul(m, k, n, jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, mm)
      analyze_matmul(m, k, n, jnp.bfloat16, jnp.int8, jnp.bfloat16, mm)
      # bf16 x int4 matmul not supported
      # analyze_matmul(m, k, n, jnp.bfloat16, jnp.int4, jnp.bfloat16, mm)
      analyze_matmul(m, k, n, jnp.int8, jnp.int8, jnp.int8, mm)

if __name__ == "__main__":
    main()
