import jax
from jax import numpy as jnp
import datetime
import jax
import random
import string


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

def assert_same_output_and_grad(f1,f2, targets, *inputs, f1_extra_inputs=[], f2_extra_inputs=[]):
  f1_inputs = (*inputs, *f1_extra_inputs)
  f2_inputs = (*inputs, *f2_extra_inputs)
  def f1_loss(*f1_inputs):
    return jnp.linalg.norm(f1(*f1_inputs) - targets)
  
  def f2_loss(*f2_inputs):
    return jnp.linalg.norm(f2(*f2_inputs) - targets)

  def print_norms(a,b,a_name="a",b_name="b",diff_name="diff"):
    a_norm = jnp.linalg.norm(a)
    b_norm = jnp.linalg.norm(b)
    diff_norm = jnp.linalg.norm(a-b)

    print(f"{diff_name} norm of {diff_norm}")
    print(f"{a_name} norm of {a_norm}")
    print(f"{b_name} norm of {b_norm}")

  f1_value = f1(*f1_inputs)
  f2_value = f2(*f2_inputs)
  _, f1_grad = jax.value_and_grad(f1_loss)(*f1_inputs)
  _, f2_grad = jax.value_and_grad(f2_loss)(*f2_inputs)

  print_norms(f1_value, f2_value, a_name="regular", b_name="pipeline", diff_name="Output difference")
  print_norms(f1_grad, f2_grad, a_name="reg_grad", b_name="pipeline_grad", diff_name="Gradient difference")

def simple_timeit(f, *args, tries = 10, task = None):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    trace_dir = f"/tmp/traces/{trace_name}"

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    jax.profiler.start_trace(trace_dir)

    for _ in range(tries):
        s = datetime.datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.datetime.now()
        outcomes_ms.append(1000*(e-s).total_seconds())
    jax.profiler.stop_trace()

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
    return average_time_ms