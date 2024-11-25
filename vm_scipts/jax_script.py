

import jax
import jax.numpy as jnp
import time

import torch

@jax.jit
def multiply_fusion_test():
  x = jnp.ones((1024, 1, 1280, 128), dtype=jnp.float8_e4m3fn)
  y = jnp.ones((1024, 1, 1280, 128), dtype=jnp.bfloat16)
  out = jnp.bfloat16(x) * y
  return out

def torch_fusion_test():
  x = torch.ones((1024, 1, 1280, 128), dtype=torch.float8_e4m3fn)
  y = torch.ones((1024, 1, 1280, 128), dtype=torch.bfloat16)
  out = x.to(torch.bfloat16) * y
  return out

multiply_fusion_test() # warmup run to compile
start_time = time.time_ns()
for i in range(100):
  multiply_fusion_test()
end_time = time.time_ns()
print("number of ms taken for jax: ", (end_time - start_time)/(100 * 1000000))

assert torch.cuda.is_available()
start_time = time.time_ns()
for i in range(100):
  torch_fusion_test()
end_time = time.time_ns()
print("number of ms taken for pytorch: ", (end_time - start_time)/(100 * 1000000))