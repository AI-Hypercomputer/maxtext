import jax
from jax import numpy as jnp 

batch_size = 2
channels_in = 2
channels_out = 2

max_int4 = 7.5
max_int8 = 127

def quant_int4(x):
  return jnp.clip(jnp.round(x), -max_int4, max_int4).astype(jnp.int4)

def quant_int8(x):
  return jnp.clip(jnp.round(x), -max_int8, max_int8).astype(jnp.int8)

def matmul_true(lhs, rhs):
  result = jnp.matmul(lhs, rhs, preferred_element_type=jnp.int32)
  assert result.dtype == jnp.int32
  return result

def aqt_matmul_int8(a, w):

  # Calibration. Calibration function is also customizable and injectable.
  a_s = max_int8 / jnp.max(jnp.abs(a), axis=1, keepdims=True)
  w_s = max_int8 / jnp.max(jnp.abs(w), axis=0, keepdims=True)
  assert a_s.shape == (batch_size, 1), f"{a_s.shape} {batch_size}" # shapes checked for illustration
  assert w_s.shape == (1, channels_out), f"{w_s.shape} {channels_out}"

  # int8 matmul with int32 accumulator
  result = matmul_true(quant_int8(a * a_s), quant_int8(w * w_s)) / (a_s * w_s)
  assert result.shape == (batch_size, channels_out)

  return result

def aqt_matmul_int4(a, w):
  # Calibration. Calibration function is also customizable and injectable.
  a_s = max_int4 / jnp.max(jnp.abs(a), axis=1, keepdims=True)
  w_s = max_int4 / jnp.max(jnp.abs(w), axis=0, keepdims=True)
  assert a_s.shape == (batch_size, 1) # shapes checked for illustration
  assert w_s.shape == (1, channels_out)

  # int8 matmul with int32 accumulator
  result = matmul_true(quant_int4(a * a_s), quant_int4(w * w_s)) / (a_s * w_s)
  assert result.shape == (batch_size, channels_out)

  return result



def aqt_matmul_int_a8w4(a, w):
  # Calibration. Calibration function is also customizable and injectable.
  a_s = max_int8 / jnp.max(jnp.abs(a), axis=1, keepdims=True)
  w_s = max_int4 / jnp.max(jnp.abs(w), axis=0, keepdims=True)
  assert a_s.shape == (batch_size, 1) # shapes checked for illustration
  assert w_s.shape == (1, channels_out)

  # int8 matmul with int32 accumulator
  print(a * a_s)
  print(a_s)
  print(w * w_s)
  print(w_s)
  result = matmul_true(quant_int8(a * a_s), quant_int4(w * w_s)) / (a_s * w_s)
  assert result.shape == (batch_size, channels_out)

  return result

def gen_matrix(rows, columns, seed=0):
  np.random.seed(seed)
  return np.random.normal(size=(rows, columns)).reshape((rows, columns))



#a = gen_matrix(batch_size, channels_in) # Activations
#w = gen_matrix(channels_in, channels_out) # Weights

a = jnp.ones((2, 2))*.035
w = jnp.ones((2, 2))*.035

print("int4")
print(aqt_matmul_int4(a, w))
print()
print("int_a8_w4")
print(aqt_matmul_int_a8w4(a, w))
print()
print("int8")
print(aqt_matmul_int8(a, w))
print()

