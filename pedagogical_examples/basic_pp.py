
import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
import timing_util

import argparse
from typing import Sequence
import os

def stage(w, x):
  for i in range(w.shape[0]):
    x = layer(w[i], x)
  return x

def layer(w, x):
  x = jnp.tanh(jnp.dot(x, w))
  return x
  
def spmd_pipeline(weights, inputs):
  # M: n_microbatches
  # F: features
  # L: n_stages
  # B: microbatch_size

  # w: [n_stages, n_layers/n_stages, F, F] [L, -1, F , F]
  weights = weights.reshape((args.n_stages, -1 , args.features ,args.features))
  # x: [n_microbatches, microbatch_size, F] [M, B, F]
  inputs = inputs.reshape((args.n_microbatches, args.microbatch_size, args.features))

  # Pad inputs to allow flushing the pipeline
  # x: [M+L-1, B, F]
  # outputs: [M+L-1, B, F]
  inputs = jnp.pad(inputs, [[0, args.n_stages-1], [0, 0], [0, 0]])
  outputs = jnp.zeros((args.n_microbatches + args.n_stages - 1, args.microbatch_size, args.features))

  state = jnp.zeros([args.n_stages, args.microbatch_size, args.features])

  for i in range(args.n_microbatches + args.n_stages - 1):
    state = shift_right_and_insert_input(state, inputs[i])
    state = jax.vmap(stage, in_axes=0, out_axes=0,
                     spmd_axis_name='stage')(weights, state)
    outputs = outputs.at[i].set(state[-1])  # last layer output
  return outputs[args.n_stages - 1:].reshape((args.n_microbatches * args.microbatch_size, args.features))


def shift_right_and_insert_input_new(state, new_microbatch):
    padding = [[1, 0]] + [[0, 0]] * (state.ndim - 1)
    # Use lax.slice to guarantee the gradient is a pad.
    return jax.lax.slice(jnp.pad(state, padding), [0] * state.ndim, state.shape)
  
def shift_right_and_insert_input(state, new_microbatch):
  prev = new_microbatch
  for stage in range(args.n_stages):
    next = state[stage]
    #state[stage] = prev
    state = state.at[stage].set(prev)
    prev = next
  return state

def get_weights_and_inputs():
  k = jax.random.PRNGKey(1)
  k1, k2, k3 = jax.random.split(k, 3)
  inputs = jax.random.normal(k1, (args.batch_size, args.features))
  weights = jax.random.normal(k2, (args.n_layers, args.features, args.features)) 
  # Targets are used for a dummy scalar loss fctn to check gradients, loss = norm(inputs - targets)
  targets = jax.random.normal(k3, (args.batch_size, args.features))
  return weights, inputs, targets

def get_pipelint_jit():
  devices = mesh_utils.create_device_mesh((args.pipeline_axis, args.dp_axis))
  mesh = Mesh(devices, axis_names=('stage', 'data'))

  def S(*specs):
    return NamedSharding(mesh, PartitionSpec(*specs))

  # Configure sharding
  weight_sharding = S('stage', None, None) # weight sharded over stage
  input_sharding = S('data', None)   # inputs sharded over batch
  result_sharding = S('data', None)  # output sharded over batch

  output_jit = jax.jit(spmd_pipeline,
              in_shardings=((weight_sharding, input_sharding)),
              out_shardings=result_sharding)
  return output_jit

def reg_matmuls(weights, input):
  for layer_idx in range(args.n_layers):
    input = layer(weights[layer_idx,:,:], input)
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

def main() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_microbatches', type=int, default=4)
    parser.add_argument('--pipeline_axis', type=int, default=4)
    parser.add_argument('--dp_axis', type=int, default=1)
    parser.add_argument('--features', type=int, default=16)
    
    global args
    args = parser.parse_args()
    args.microbatch_size = args.batch_size // args.n_microbatches


    # Necessary artifacts for the good stuff
    pipeline_func = get_pipelint_jit()
    weights, inputs, targets = get_weights_and_inputs()

    assert_same_output_and_grad(reg_matmuls, pipeline_func, targets, weights, inputs)

    timing_util.simple_timeit(pipeline_func, weights, inputs, tries = 3, task = 'basic_pp')



if __name__ == "__main__":
  main()