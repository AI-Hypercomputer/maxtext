
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

# def mlp(w, x):
#     for i in range(w.shape[0]): # Weights are [layer, embed, embed]
#         x = stage(w[i], x)
#     return x

def stage(w, x):
  for i in range(w.shape[0]):
    x = layer(w[i], x)
  return x

def layer(w, x):
  x = jnp.tanh(jnp.dot(x, w))
  return x
  
def spmd_pipeline(w, x):
  # M: n_microbatches
  # F: features
  # L: n_stages
  # B: microbatch_size

  # w: [n_stages, n_layers/n_stages, F, F] [L, -1, F , F]
  w = w.reshape((n_stages, -1 , features ,features))
  # x: [n_microbatches, microbatch_size, F] [M, B, F]
  x = x.reshape((n_microbatches, microbatch_size, features))

  # Pad inputs to allow flushing the pipeline
  # x: [M+L-1, B, F]
  # outputs: [M+L-1, B, F]
  x = jnp.pad(x, [[0, n_stages-1], [0, 0], [0, 0]])
  outputs = jnp.zeros((n_microbatches + n_stages - 1, microbatch_size, features))

  state = jnp.zeros([n_stages, microbatch_size, features])

  for i in range(n_microbatches + n_stages - 1):
    state = shift_right_and_insert_input(state, x[i])
    state = jax.vmap(stage, in_axes=0, out_axes=0,
                     spmd_axis_name='stage')(w, state)
    outputs = outputs.at[i].set(state[-1])  # last layer output
  return outputs[n_stages - 1:].reshape((n_microbatches * microbatch_size, features))


def shift_right_and_insert_input(state, new_microbatch):
  prev = new_microbatch
  for stage in range(n_stages):
    next = state[stage]
    #state[stage] = prev
    state = state.at[stage].set(prev)
    prev = next
  return state



def mlp(w1, x):
 for i in range(w1.shape[0]):
   x = jnp.tanh(jnp.dot(x, w1[i]))
#    x = jnp.dot(x, w2[i])
 return x

# def mlp(w1, w2, x):
#  for i in range(w1.shape[0]):
#    x = jnp.tanh(jnp.dot(x, w1[i]))
#    x = jnp.dot(x, w2[i])
#  return x

# Initialize model weights
batch_size = 64
n_layers=16
n_stages = 2
n_microbatches = 1
microbatch_size = batch_size // n_microbatches

features = 32
k = jax.random.PRNGKey(1)
k1, k2, k3 = jax.random.split(k, 3)

input = jax.random.normal(k1, (batch_size, features))
weight1 = jax.random.normal(k2, (n_layers, features, features)) 
weight2 = jax.random.normal(k3, (n_layers, features, features)) 

#output = mlp(weight1, weight2, input)
#output = mlp(weight1, input)


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

output_pipeline = output_jit(weight1, input)


def reg_matmuls(weights, input):
  for layer_idx in range(n_layers):
    input = layer(weights[layer_idx,:,:], input)
  return input

output_reg = reg_matmuls(weight1, input)

diff_norm = jnp.linalg.norm(output_pipeline - output_reg)
print(f"{diff_norm=}")

regular_norm = jnp.linalg.norm(output_reg)
print(f"{regular_norm=}")

output_pipeline_norm = jnp.linalg.norm(output_pipeline)
print(f"{output_pipeline_norm=}")