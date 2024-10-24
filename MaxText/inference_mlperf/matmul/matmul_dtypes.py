import jax

import timing_util

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
