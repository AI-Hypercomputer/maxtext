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

import datetime
import os
from typing import Sequence

from absl import app
from absl import flags
import jax
from jax.experimental import mesh_utils
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental.pjit import pjit
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
import numpy as np


FLAGS = flags.FLAGS

_profiler_path = flags.DEFINE_string(
    name="profiler_path",
    default=None,
    help="Path to the profiler where the script will write to."
)
_embedding_dimension = flags.DEFINE_integer(
    name="embedding_dimension",
    default=2048,
    help="Dimension of the embedding vector.",
)
_batch_size = flags.DEFINE_integer(
    name="batch_size",
    default=131072,
    help="Batch size.",
)
_num_layers = flags.DEFINE_integer(
    name="num_layers",
    default=4,
    help="Number of layers."
)
_dcn_data_parallelism = flags.DEFINE_integer(
    name="dcn_data_parallelism",
    default=1,
    help="N-way Data Parallelism across slices."
)
_dcn_fsdp_parallelism = flags.DEFINE_integer(
    name="dcn_fsdp_parallelism",
    default=1,
    help="Fsdp parallelism across slices."
)
_dcn_tensor_parallelism = flags.DEFINE_integer(
    name="dcn_tensor_parallelism",
    default=1,
    help="Tensor parallelism within each slice."
)
_ici_data_parallelism = flags.DEFINE_integer(
    name="ici_data_parallelism",
    default=1,
    help="Data parallelism within each slice."
)
_ici_fsdp_parallelism = flags.DEFINE_integer(
    name="ici_fsdp_parallelism",
    default=1,
    help="Fsdp parallelism within each slice."
)
_ici_tensor_parallelism = flags.DEFINE_integer(
    name="ici_tensor_parallelism",
    default=1,
    help="Tensor parallelism within each slice."
)

def main(argv: Sequence[str]) -> None:
  profiler_path = _profiler_path.value
  embedding_dimension = _embedding_dimension.value
  batch_size = _batch_size.value
  num_layers = _num_layers.value
  dcn_data_parallelism = _dcn_data_parallelism.value
  dcn_fsdp_parallelism = _dcn_fsdp_parallelism.value
  dcn_tensor_parallelism = _dcn_tensor_parallelism.value
  ici_data_parallelism = _ici_data_parallelism.value
  ici_fsdp_parallelism = _ici_fsdp_parallelism.value
  ici_tensor_parallelism = _ici_tensor_parallelism.value

  cc.initialize_cache(os.path.expanduser("~/jax_cache_2"))

  def activate_profiler(profiler_path):
    if profiler_path:
      jax.profiler.start_trace(profiler_path)

  def deactivate_profiler(profiler_path):
    if profiler_path:
      jax.profiler.stop_trace()

  def simple_timeit(f, tries = 5, verbose = True):
    """Simple utility to time a function for multiple runs."""
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

  dcn_parallelism = [
      dcn_data_parallelism, dcn_fsdp_parallelism, dcn_tensor_parallelism
  ]
  ici_parallelism = [
      ici_data_parallelism, ici_fsdp_parallelism, ici_tensor_parallelism
  ]

  devices = jax.devices()
  num_devices = len(devices)
  print(f"Devices: {devices} (num_devices: {num_devices})")
  assert len(devices) > 1, "You must have at least two devices"

  # Assert that we have correct inputs of sharding that fit the number of chips
  assert np.product(dcn_parallelism) * np.product(ici_parallelism) == num_devices, (
      f"Number of devices {num_devices} does not match the product of the parallelism \
          {np.product(dcn_parallelism) * np.product(ici_parallelism)}"
  )

  multi_slice_env = hasattr(jax.devices()[0], 'slice_index')
  # Create device mesh

  if multi_slice_env:
    assert dcn_data_parallelism == 1 + max(x.slice_index for x in jax.devices()), (
    f"Number of slices given {dcn_data_parallelism} \
          does not match the number fetched from jax devices {jax.devices()[0]}"
    )
    devices_array = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism, dcn_parallelism
    )
  else:
    devices_array = mesh_utils.create_device_mesh(ici_parallelism)

  print(f"Decided on mesh shape: {devices_array}")

  mesh = Mesh(devices_array, ["data", "fsdp", "tensor"])

  data_sharding = PartitionSpec(("data", "fsdp"),  "tensor")
  # We assume parameters are stored in a decreasing order of dimension size
  parameter_sharding = PartitionSpec("tensor", "fsdp")

  BATCH = len(jax.devices()) * batch_size
  D_EMB = embedding_dimension
  D_FF =  4 * D_EMB
  NUM_LAYERS = num_layers

  parameters = 2 * D_FF * D_EMB * NUM_LAYERS
  parameter_bytes = 2 * parameters
  activation_bytes = 2 * (  BATCH  * ( D_FF+D_EMB) ) * NUM_LAYERS
  memory_bytes = parameter_bytes + activation_bytes

  print(f"total {memory_bytes/10**9} GB, parameters {parameter_bytes/10**9} GB, \
      activations {activation_bytes/10**9} GB")

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
    out_layers = jax.tree_map(
        lambda param, grad: param - 1e-4 * grad, in_layers, grad_layers[0]
    )
    return out_layers

  print("finished includes ", flush = True)

  pjit_func = pjit(
      training_step,
      in_shardings=(data_sharding, parameter_sharding),
      out_shardings=parameter_sharding,
  )

  pjit_gen_data = pjit(gen_data, in_shardings=None, out_shardings=data_sharding)

  pjit_gen_layers = pjit(
      gen_layers, in_shardings=None, out_shardings=parameter_sharding
  )

  with Mesh(mesh.devices, mesh.axis_names):
    key = jax.random.PRNGKey(0)
    presharded_X = jax.block_until_ready(pjit_gen_data(key))
    presharded_layers = jax.block_until_ready(pjit_gen_layers(key))
    activate_profiler(profiler_path)
    TFLOPs_per_device = parameters * 6 * BATCH  / 10**12 / len(jax.devices())
    time = simple_timeit(
        lambda : jax.block_until_ready(pjit_func(presharded_X, presharded_layers))
    )
    print(f"time is {time} seconds, TFLOP is {TFLOPs_per_device},\
        TFLOP/s is {TFLOPs_per_device/time}", flush = True)
  deactivate_profiler(profiler_path)

if __name__ == "__main__":
  app.run(main)


