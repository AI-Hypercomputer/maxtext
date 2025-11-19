# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MaxText sharding."""


import datetime
import numpy as np
import jax

import pytest

from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.lax import with_sharding_constraint

# Global model and data constants
PER_DEVICE_BATCH_SIZE = 131072
BATCH_SIZE = 1  # will be updated inside the test
D_EMB = 2048  # embedding dimension
D_FF = 4 * D_EMB  # feed-forward dimension
NUM_LAYERS = 4  # number of simulated layers in the model

# Global sharding specs, will be updated inside the test
data_sharding = PartitionSpec()
parameter_sharding = PartitionSpec()
multiply_layers_and_grad = None


def simple_timeit(f, tries=5, verbose=True):
  """Simple utility to time a function for multiple runs"""
  outcomes = []
  f()  # warm it up!
  for _ in range(tries):
    s = datetime.datetime.now()
    f()
    e = datetime.datetime.now()
    outcomes.append((e - s).total_seconds())
  average_time = sum(outcomes) / len(outcomes)
  if verbose:
    print(f"average time: {average_time}, timings (seconds) {outcomes}")
  return average_time


# --- Model Simulation Functions ---
def gen_layer(random_key):
  """Generates a single simulated feed-forward layer's parameters."""
  keys = jax.random.split(random_key, num=4)
  return {
      "EMB2FF": 1e-4 * jax.random.normal(keys[0], (D_FF, D_EMB), dtype=jax.numpy.bfloat16),
      "FF2EMB": 1e-4 * jax.random.normal(keys[1], (D_FF, D_EMB), dtype=jax.numpy.bfloat16),
  }


def gen_layers(random_key):
  """Generates all model layers."""
  layers = []
  for _ in range(NUM_LAYERS):
    random_key, sub_key = jax.random.split(random_key)
    layers.append(gen_layer(sub_key))
  return tuple(layers)


def gen_data(random_key):
  """Generates a batch of input data."""
  return jax.random.uniform(random_key, (BATCH_SIZE, D_EMB), dtype=jax.numpy.bfloat16)


# Forward pass simulation functions
def multiply_layer(in_act, in_layer):
  """Simulates a forward pass through one layer, applying sharding constraints."""
  with jax.named_scope("M1"):
    M1 = jax.nn.sigmoid(in_act @ in_layer["EMB2FF"].T)
    M1 = with_sharding_constraint(M1, data_sharding)
  with jax.named_scope("M2"):
    M2 = jax.nn.sigmoid(M1 @ in_layer["FF2EMB"])
    M2 = with_sharding_constraint(M2, data_sharding)
  return M2


def multiply_layers(in_act, in_layers):
  """Passes activations through all simulated layers."""
  x = in_act
  for i, layer in enumerate(in_layers):
    with jax.named_scope(f"layer_{i}"):
      x = with_sharding_constraint(multiply_layer(x, layer), data_sharding)
  return x, in_layers


def multiply_layers_with_loss(in_act, in_layers):
  """Calculates a simple loss."""
  x, _ = multiply_layers(in_act, in_layers)
  return jax.numpy.sum(x)


def training_step(in_act, in_layers):
  """Simulates a single gradient update step."""
  _, grad_layers = multiply_layers_and_grad(in_act, in_layers)
  out_layers = jax.tree_util.tree_map(lambda param, grad: param - 1e-4 * grad, in_layers, grad_layers[0])
  return out_layers


def check_sharding(array_or_dict, expected_sharding):
  """Checks the sharding property only if the item is a JAX array."""
  if isinstance(array_or_dict, jax.Array):
    np.testing.assert_equal(
        array_or_dict.sharding, expected_sharding, "Initial layer parameter sharding does not match expected sharding."
    )


@pytest.mark.tpu_only
@pytest.mark.scheduled_only
def test_fsdp_sharding():
  """Tests FSDP sharding on a simple model simulation."""
  dcn_parallelism = [1, 1, 1]
  ici_parallelism = [1, 4, 1]

  devices = jax.devices()
  num_devices = len(devices)
  assert len(devices) > 1, "Test requires multiple JAX devices."

  # Assert that we have correct inputs of sharding that fit the number of chips
  assert (
      np.prod(dcn_parallelism) * np.prod(ici_parallelism) == num_devices
  ), f"Number of devices {num_devices} \
        does not match the product of the parallelism {np.prod(dcn_parallelism) * np.prod(ici_parallelism)}"

  devices_array = mesh_utils.create_device_mesh(ici_parallelism)
  mesh = Mesh(devices_array, ["data", "fsdp", "tensor"])

  global data_sharding
  global parameter_sharding
  data_sharding = PartitionSpec(("data", "fsdp"), "tensor")
  parameter_sharding = PartitionSpec("tensor", "fsdp")

  global BATCH_SIZE
  BATCH_SIZE = len(jax.devices()) * PER_DEVICE_BATCH_SIZE

  global multiply_layers_and_grad
  multiply_layers_and_grad = jax.value_and_grad(multiply_layers_with_loss, argnums=[1])

  replicated_sharding = jax.sharding.NamedSharding(mesh, data_sharding)
  parameter_mesh_shardings = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), parameter_sharding)
  output_mesh_shardings = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), parameter_sharding)
  jit_gen_layers = jax.jit(gen_layers, in_shardings=None, out_shardings=parameter_mesh_shardings)

  data_mesh_shardings = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_sharding)
  jit_gen_data = jax.jit(gen_data, in_shardings=None, out_shardings=data_mesh_shardings)

  jit_func = jax.jit(
      training_step,
      in_shardings=(replicated_sharding, parameter_mesh_shardings),
      out_shardings=output_mesh_shardings,
  )

  with Mesh(mesh.devices, mesh.axis_names):
    key = jax.random.PRNGKey(0)
    presharded_X = jax.block_until_ready(jit_gen_data(key))
    presharded_layers = jax.block_until_ready(jit_gen_layers(key))

    # assert the data sharding
    np.testing.assert_equal(
        presharded_X.sharding, data_mesh_shardings, "Input data sharding does not match expected sharding."
    )

    # asset the parameter sharding for all layers
    for layer in range(NUM_LAYERS):
      np.testing.assert_equal(
          presharded_layers[layer]["EMB2FF"].sharding,
          parameter_mesh_shardings,
          "The sharding was not applied correctly to the generated layers.",
      )
      np.testing.assert_equal(
          presharded_layers[layer]["FF2EMB"].sharding,
          parameter_mesh_shardings,
          "The sharding was not applied correctly to the generated layers.",
      )

    # Time the training step
    parameters = 2 * D_FF * D_EMB * NUM_LAYERS
    TFLOPs_per_device = parameters * 6 * BATCH_SIZE / 10**12 / len(jax.devices())
    time = simple_timeit(lambda: jax.block_until_ready(jit_func(presharded_X, presharded_layers)))
    print(f"time is {time} seconds, TFLOP is {TFLOPs_per_device}, TFLOP/s is {TFLOPs_per_device/time}", flush=True)
