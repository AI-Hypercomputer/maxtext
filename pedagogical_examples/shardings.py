#!/usr/bin/python3

"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

'''This script is used to measure the performance of different sharding schemes on TPU.'''

from absl import app
from absl import flags
import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax._src.pjit import with_sharding_constraint

import argparse
import datetime
import numpy as np
from typing import Sequence

parser = argparse.ArgumentParser(
  description="Experiment different sharding techniques with a simple NN.\
  Ensure 1) The product of dcn dimensions == number of slices \
  2) product of ici dimension = number of devices per slice"
  )
parser.add_argument(
    "--profiler_path", "-p",
    required=False,
    default="",
    help="Path to the profiler where the script will write to.",
    type=str
)
parser.add_argument(
    "--embedding_dimension", "-d",
    required=False,
    default=2048,
    type=int
)
parser.add_argument(
    "--batch_size", "-b",
    required=False,
    default=131072,
    type=int
)
parser.add_argument(
    "--num_layers", "-n",
    required=False,
    default=4,
    type=int
)
parser.add_argument(
    "--dcn_data_parallelism", "-dd",
    help="N-way Data Parallelism across slices",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--dcn_fsdp_parallelism", "-df",
    help="Fsdp parallelism across slices that is expected to be 1 in most cases",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--dcn_tensor_parallelism", "-dt",
    help="Tensor parallelism across slices that is expected to be 1 in most cases",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--ici_data_parallelism", "-id",
    help="Data parallelism within each slice that is expected to be 1 in most cases",
    required=False,
    default=1,
    type=int
)
parser.add_argument(
    "--ici_fsdp_parallelism", "-if",
    help="Number of shards for Fsdp Parallelism within each slice.",
    required=False,
    default=4,
    type=int
)
parser.add_argument(
    "--ici_tensor_parallelism", "-it",
    help="Number of shards for Tensor Parallelism within each slice.",
    required=False,
    default=1,
    type=int
)
args = parser.parse_args()

def main(_argv: Sequence[str]) -> None:
  def activate_profiler(profiler_path):
    if profiler_path:
      print(f"profiling to {profiler_path}")
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

  dcn_parallelism = [args.dcn_data_parallelism, args.dcn_fsdp_parallelism, args.dcn_tensor_parallelism]
  ici_parallelism = [args.ici_data_parallelism, args.ici_fsdp_parallelism, args.ici_tensor_parallelism]

  devices = jax.devices()
  num_devices = len(devices)
  print(f"Devices: {devices} (num_devices: {num_devices})")
  assert len(devices) > 1, "You must have at least two devices"

  # Assert that we have correct inputs of sharding that fit the number of chips
  assert np.product(dcn_parallelism) * np.product(ici_parallelism) == num_devices, f"Number of devices {num_devices} \
        does not match the product of the parallelism {np.product(dcn_parallelism) * np.product(ici_parallelism)}"

  multi_slice_env = hasattr(jax.devices()[0], 'slice_index')
  # Create device mesh

  if multi_slice_env:
    assert args.dcn_data_parallelism == 1 + max(x.slice_index for x in jax.devices()), \
    f"Number of slices given {args.dcn_data_parallelism} \
          does not match the number fetched from jax devices {jax.devices()[0]}"
    devices_array = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
  else:
    devices_array = mesh_utils.create_device_mesh(ici_parallelism)

  print(f"Decided on mesh shape: {devices_array}")

  mesh = Mesh(devices_array, ["data", "fsdp", "tensor"])

  data_sharding = PartitionSpec(("data", "fsdp"),  "tensor")
  # We assume parameters are stored in a decreasing order of dimension size
  parameter_sharding = PartitionSpec("tensor", "fsdp")

  BATCH = len(jax.devices()) * args.batch_size
  D_EMB = args.embedding_dimension
  D_FF =  4 * D_EMB
  NUM_LAYERS = args.num_layers

  parameters = 2 * D_FF * D_EMB * NUM_LAYERS
  parameter_bytes = 2 * parameters
  activation_bytes = 2 * (  BATCH  * ( D_FF+D_EMB) ) * NUM_LAYERS
  memory_bytes = parameter_bytes + activation_bytes

  print(f"total {memory_bytes/1e9} GB, parameters {parameter_bytes/1e9} GB, activations {activation_bytes/1e9} GB")

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

  replicated_sharding = jax.sharding.NamedSharding(mesh, data_sharding)

  parameter_mesh_shardings = jax.tree_map(
        lambda p: jax.sharding.NamedSharding(mesh, p), parameter_sharding)

  data_pspec_shardings = jax.tree_map(
          lambda p: jax.sharding.NamedSharding(mesh, p), parameter_sharding)

  jit_func = jax.jit(
          training_step,
          in_shardings=(replicated_sharding, parameter_mesh_shardings),
          out_shardings=data_pspec_shardings,
        )

  data_mesh_shardings = jax.tree_map(
          lambda p: jax.sharding.NamedSharding(mesh, p), data_sharding)

  jit_gen_data = jax.jit(
          gen_data,
          in_shardings=None,
          out_shardings=data_mesh_shardings
        )

  parameter_mesh_shardings = jax.tree_map(
          lambda p: jax.sharding.NamedSharding(mesh, p), parameter_sharding)

  jit_gen_layers = jax.jit(
          gen_layers,
          in_shardings=None,
          out_shardings=parameter_mesh_shardings
        )

  # starting the profiler outside `with` statement,
  # will call it right before the computation once b/301309635 is resolved
  activate_profiler(args.profiler_path)
  with Mesh(mesh.devices, mesh.axis_names):
    key = jax.random.PRNGKey(0)
    presharded_X = jax.block_until_ready(jit_gen_data(key))
    presharded_layers = jax.block_until_ready(jit_gen_layers(key))
    TFLOPs_per_device = parameters * 6 * BATCH  / 10**12 / len(jax.devices())
    time = simple_timeit(lambda : jax.block_until_ready(jit_func(presharded_X, presharded_layers)))
    print(f"time is {time} seconds, TFLOP is {TFLOPs_per_device}, TFLOP/s is {TFLOPs_per_device/time}", flush = True)
  deactivate_profiler(args.profiler_path)

def parse_flags(argv):
  return parser.parse_args(argv[1:])

if __name__ == "__main__":
  flags.FLAGS.mark_as_parsed()
  app.run(main, flags_parser=parse_flags)
