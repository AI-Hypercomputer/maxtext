#!/usr/bin/python3

'''This script is used to measure the performance of different sharding schemes on TPU.'''

import jax
from jax.experimental.pjit import PartitionSpec, pjit
from jax.experimental import maps
from jax.experimental import mesh_utils
from jax.experimental.pjit import with_sharding_constraint


import datetime
import os

from jax.experimental.compilation_cache import compilation_cache as cc

import sys

#os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"

cc.initialize_cache(os.path.expanduser("~/jax_cache_2"))


def activate_profiler(profiler_path):
  if profiler_path:
    jax.profiler.start_trace(profiler_path)

def deactivate_profiler(profiler_path):
  if profiler_path:
    jax.profiler.stop_trace()

def simple_timeit(f, tries = 5, verbose = True):
  '''Simple utility to time a function for multiple runs'''
  outcomes = []
  f() #warm it up!
  for _ in range(tries):
    s = datetime.datetime.now()
    f()
    e = datetime.datetime.now()
    outcomes.append((e-s).total_seconds())
  average_time = sum(outcomes)/len(outcomes)
  if verbose:
    print(f"average time: {average_time}, timings (seconds) {outcomes}")
  return average_time

SHARDING = sys.argv[1]

if SHARDING == "1D_TENSOR_PARALLELISM":
  data_sharding = PartitionSpec(None, ('axis1', 'axis2'))
  parameter_sharding = PartitionSpec( ('axis1', 'axis2'), None)
  devices_array = mesh_utils.create_device_mesh([4, 1])
elif SHARDING == "2D_TENSOR_PARALLELISM":
  data_sharding = PartitionSpec(None, ('axis1', 'axis2'))
  parameter_sharding = PartitionSpec( 'axis1', 'axis2')
  devices_array = mesh_utils.create_device_mesh([2, 2])
elif SHARDING == "FULLY_SHARDED_DATA_PARALLELISM":
  data_sharding = PartitionSpec('axis1')
  parameter_sharding = PartitionSpec('axis1')
  devices_array = mesh_utils.create_device_mesh([4, 1])
elif SHARDING == "DATA_PARALLELISM":
  data_sharding = PartitionSpec('axis1')
  parameter_sharding = PartitionSpec(None)
  devices_array = mesh_utils.create_device_mesh([4, 1])
else:
  assert False, "unknown sharding"

print(f"mesh shape {devices_array}", flush = True)
mesh = maps.Mesh(devices_array, ('axis1', 'axis2'))

BATCH = len(jax.devices()) * 128
D_EMB = 2 * 8192
D_FF =  4 * D_EMB
NUM_LAYERS = 4

parameters = 2 * D_FF * D_EMB * NUM_LAYERS
parameter_bytes = 2 * parameters
activation_bytes = 2 * (  BATCH  * ( D_FF+D_EMB) ) * NUM_LAYERS
memory_bytes = parameter_bytes + activation_bytes

print(f"total {memory_bytes/10**9} GB, parameters {parameter_bytes/10**9} GB, activations {activation_bytes/10**9} GB")

def gen_layer(random_key):
  keys = jax.random.split(random_key, num = 4)
  return {
    "EMB2FF" : 1e-4 * jax.random.normal( keys[0], (D_FF, D_EMB), dtype=jax.numpy.bfloat16),
    "FF2EMB" : 1e-4 * jax.random.normal( keys[1], (D_FF, D_EMB), dtype=jax.numpy.bfloat16),
  }

def gen_layers(random_key):
  layers = []
  for _ in range(NUM_LAYERS):
    random_key, sub_key = jax.random.split(random_key)
    layers.append(gen_layer(sub_key))
  return tuple(layers)

def gen_data(random_key):
  return jax.random.uniform(random_key, (BATCH, D_EMB), dtype=jax.numpy.bfloat16 )


def multiply_layer(in_act, in_layer):
  with jax.named_scope("M1"):
    M1 = jax.nn.sigmoid(in_act @ in_layer["EMB2FF"].T)
    M1 = with_sharding_constraint(M1, data_sharding)
  with jax.named_scope("M2"):
    M2 = jax.nn.sigmoid(M1 @ in_layer["FF2EMB"])
    M2 = with_sharding_constraint(M2, data_sharding)

  return M2

def multiply_layers(in_act, in_layers):
  x = in_act

  for i, layer in enumerate(in_layers):
    with jax.named_scope(f"layer_{i}"):
      x = with_sharding_constraint(multiply_layer(x, layer), data_sharding)

  return x, in_layers

def multiply_layers_with_loss(in_act, in_layers):
  x, _ =  multiply_layers(in_act, in_layers)
  return jax.numpy.sum(x)

multiply_layers_and_grad = jax.value_and_grad(multiply_layers_with_loss, argnums=[1])

def training_step(in_act, in_layers):
  _, grad_layers = multiply_layers_and_grad(in_act, in_layers)
  out_layers = jax.tree_map(lambda param, grad: param - 1e-4 * grad, in_layers, grad_layers[0])
  return out_layers

print("finished includes ", flush = True)

pjit_func = pjit(
        training_step,
        in_axis_resources=(data_sharding, parameter_sharding),
        out_axis_resources=parameter_sharding,
      )

pjit_gen_data = pjit(
        gen_data,
        in_axis_resources=None,
        out_axis_resources=data_sharding
      )

pjit_gen_layers = pjit(
        gen_layers,
        in_axis_resources=None,
        out_axis_resources=parameter_sharding
      )

activate_profiler(sys.argv[2])

with maps.Mesh(mesh.devices, mesh.axis_names):
  key = jax.random.PRNGKey(0)
  presharded_X = jax.block_until_ready(pjit_gen_data(key))
  presharded_layers = jax.block_until_ready(pjit_gen_layers(key))
  TFLOPs_per_device = parameters * 6 * BATCH  / 10**12 / len(jax.devices())
  time = simple_timeit(lambda : jax.block_until_ready(pjit_func(presharded_X, presharded_layers)))
  print(f"time is {time} seconds, TFLOP is {TFLOPs_per_device}, TFLOP/s is {TFLOPs_per_device/time}", flush = True)
deactivate_profiler(sys.argv[2])
