import jax
from jax import numpy as jnp



def reg_matmuls(weights, input, layer_fnc=None):
  def layer(weights, input):
    outputs = jnp.einsum('bse,eh->bsh',input,weights) # The leading stage dimensions of weights and stages_in is missing because it is vmapped out
    outputs = jnp.tanh(outputs)
    return outputs

  if layer_fnc == None:
    layer_fnc = layer
  for layer_idx in range(weights.shape[0]):
    input = layer_fnc(weights[layer_idx,:,:], input)
  return input

def assert_same_output_and_grad(f1,f2, targets, *inputs):
  def f1_loss(*inputs):
    return jnp.linalg.norm(f1(*inputs) - targets)
  
  def f2_loss(*inputs):
    return jnp.linalg.norm(f2(*inputs) - targets)

  def print_norms(a,b,a_name="a",b_name="b",diff_name="diff"):
    a_norm = jnp.linalg.norm(a)
    b_norm = jnp.linalg.norm(b)
    diff_norm = jnp.linalg.norm(a-b)

    print(f"{diff_name} norm of {diff_norm}")
    print(f"{a_name} norm of {a_norm}")
    print(f"{b_name} norm of {b_norm}")

  f1_value = f1(*inputs)
  f2_value = f2(*inputs)
  _, f1_grad = jax.value_and_grad(f1_loss)(*inputs)
  _, f2_grad = jax.value_and_grad(f2_loss)(*inputs)

  print_norms(f1_value, f2_value, a_name="regular", b_name="pipeline", diff_name="Output difference")
  print_norms(f1_grad, f2_grad, a_name="reg_grad", b_name="pipeline_grad", diff_name="Gradient difference")