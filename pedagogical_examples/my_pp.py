
from absl import app
from absl import flags
import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.compilation_cache import compilation_cache as cc
from jax._src.pjit import with_sharding_constraint
import jax.numpy as jnp

import argparse
import datetime
import numpy as np
import os
from typing import Sequence

def stage(w, x):
  # If we run one layer per stage than this for loop runs just once
  for i in range(w.shape[0]):
    x = layer(w[int(i)], x)
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
  weights = weights.reshape((n_stages, -1 , features ,features))
  # x: [n_microbatches, microbatch_size, F] [M, B, F]
  inputs = inputs.reshape((n_microbatches, microbatch_size, features))

  # Pad inputs to allow flushing the pipeline
  # x: [M+L-1, B, F]
  # outputs: [M+L-1, B, F]
  inputs = jnp.pad(inputs, [[0, n_stages-1], [0, 0], [0, 0]])
  outputs = jnp.zeros((n_microbatches + n_stages - 1, microbatch_size, features))

  state = jnp.zeros([n_stages, microbatch_size, features])

  for i in range(n_microbatches + n_stages - 1):
    state = shift_right_and_insert_input(state, inputs[int(i)])
    state = jax.vmap(stage, in_axes=0, out_axes=0,
                     spmd_axis_name='stage')(weights, state)
    outputs = outputs.at[int(i)].set(state[-1])  # last layer output
  return outputs[n_stages - 1:].reshape((n_microbatches * microbatch_size, features))


def shift_right_and_insert_input_new(state, new_microbatch):
    padding = [[1, 0]] + [[0, 0]] * (state.ndim - 1)
    # Use lax.slice to guarantee the gradient is a pad.
    return jax.lax.slice(jnp.pad(state, padding), [0] * state.ndim, state.shape)
  
def shift_right_and_insert_input(state, new_microbatch):
  prev = new_microbatch
  for stage in range(n_stages):
    next = state[int(stage)]
    #state[stage] = prev
    state = state.at[int(stage)].set(prev)
    prev = next
  return state


# Initialize model weights
batch_size = 64
n_layers=16
n_stages = 2
n_microbatches = 1
microbatch_size = batch_size // n_microbatches

features = 32
k = jax.random.PRNGKey(1)
k1, k2 = jax.random.split(k, 2)

input = jax.random.normal(k1, (batch_size, features))
weights = jax.random.normal(k2, (n_layers, features, features)) 

pipeline_axis = 2
dp_axis = 2
devices = mesh_utils.create_device_mesh((pipeline_axis, dp_axis))
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

output_pipeline = output_jit(weights, input)


def reg_matmuls(weights, input):
  for layer_idx in range(n_layers):
    #input = input * 2
    input = layer(weights[layer_idx,:,:], input)
  return input

def reg_matmuls_loss(weights, input):
  output = reg_matmuls(weights, input)
  return jnp.linalg.norm(output)

def assert_same_output_and_grad(f1,f2,*inputs):
  def f1_loss(*inputs):
    return jnp.sum(f1(*inputs))
  
  def f2_loss(*inputs):
    return jnp.sum(f2(*inputs))

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

assert_same_output_and_grad(reg_matmuls,spmd_pipeline, weights, input)


# def f(x):
#   return x*x

# grad_fn = jax.value_and_grad(f)
# a, b = grad_fn(3.0)
# print(f"{a=}")
# print(f"{b=}")

# a, b = jax.value_and_grad(f)(3.0)
# a, b = grad_fn(3.0)
# print(f"{a=}")
# print(f"{b=}")