#!/usr/bin/python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax.experimental.pjit import pjit
from jax._src.partition_spec import PartitionSpec
import numpy as np
from jax._src.mesh import Mesh
import datetime
import os

# os.environ["TPU_STDERR_LOG_LEVEL"] = "0"
# os.environ["TPU_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"


def simple_timeit(f, tries=10, verbose=True):
  outcomes = []
  f()  # warm it up!
  for i in range(tries):
    s = datetime.datetime.now()
    r = f()
    e = datetime.datetime.now()
    outcomes.append((e - s).total_seconds())
  average_time = sum(outcomes) / len(outcomes)
  if verbose:
    print(f"average time: {average_time}, timings (seconds) {outcomes}")
  return average_time


# GPT1
BATCH = len(jax.devices()) * 128
SEQUENCE_LENGTH = 512
D_MODEL = 768
D_HIDDEN = 3072
NUM_LAYERS = 12


parameter_bytes = 2 * (4 * D_MODEL * D_HIDDEN * NUM_LAYERS)
ACTIVATIONS_PER_LAYER = 2
activation_bytes = (
    2 * (BATCH * SEQUENCE_LENGTH * D_MODEL) * NUM_LAYERS * ACTIVATIONS_PER_LAYER
)
memory_bytes = parameter_bytes + activation_bytes

print(
    f"total {memory_bytes/10**9} GB, parameters {parameter_bytes/10**9} GB, all layers of activations {activation_bytes/10**9} GB",
    flush=True,
)


def gen_layer(random_key):
  keys = jax.random.split(random_key, num=4)
  return {
      "WQ": 1e-4
      * jax.random.normal(
          keys[0], (D_MODEL, D_HIDDEN), dtype=jax.numpy.bfloat16
      ),
      "WK": 1e-4
      * jax.random.normal(
          keys[1], (D_MODEL, D_HIDDEN), dtype=jax.numpy.bfloat16
      ),
      "WV": 1e-4
      * jax.random.normal(
          keys[2], (D_MODEL, D_HIDDEN), dtype=jax.numpy.bfloat16
      ),
      "FF": 1e-4
      * jax.random.normal(
          keys[3], (D_HIDDEN, D_MODEL), dtype=jax.numpy.bfloat16
      ),
  }


def gen_layers(random_key):
  layers = []
  for _ in range(NUM_LAYERS):
    random_key, sub_key = jax.random.split(random_key)
    layers.append(gen_layer(sub_key))
  return tuple(layers)


def gen_data(random_key):
  return jax.random.uniform(
      random_key, (BATCH, SEQUENCE_LENGTH, D_MODEL), dtype=jax.numpy.bfloat16
  )


def multiply_layer(in_act, in_layer):
  Q = (
      in_act @ in_layer["WQ"]
  )  # BATCH x SEQUENCE_LENGTH x D_HIDDEN, flops: 2* BATCH * SEQUENCE_LENGTH * D_MODEL * D_HIDDEN
  K = (
      in_act @ in_layer["WK"]
  )  # BATCH x SEQUENCE_LENGTH x D_HIDDEN, flops: 2* BATCH * SEQUENCE_LENGTH * D_MODEL * D_HIDDEN
  V = (
      in_act @ in_layer["WV"]
  )  # BATCH x SEQUENCE_LENGTH x D_HIDDEN, flops: 2* BATCH * SEQUENCE_LENGTH * D_MODEL * D_HIDDEN
  A = jax.numpy.einsum(
      "bsd,btd->bst", Q, K
  )  # BATCH x SEQUENCE_LENGTH x SEQUENCE_LENGTH, flops : 2 * BATCH * SEQUENCE_LENGTH^2 * D_HIDDEN
  A = jax.nn.relu(A)  # TODO(correct low arithmetic intensity manips)
  post_attention = (
      A @ V
  )  # BATCH x SEQUENCE_LENGTH x D_HIDDEN, flops: 2 * BATCH * SEQUENCE_LENGTH^2 * D_HIDDEN

  right_shape = (
      post_attention @ in_layer["FF"]
  )  # BATCH x SEQUENCE_LENGTH x D_MODEL, flops: 2 * BATCH * SEQUENCE_LENGTH * D_HIDDEN * D_MODEL
  right_shape = jax.nn.relu(
      right_shape
  )  # TODO(correct low arithmetic intensity manips)
  return right_shape + 1 + in_act


def multiply_layers(in_act, in_layers):
  x = in_act

  for i in range(len(in_layers)):
    x = multiply_layer(x, in_layers[i])

  return x, in_layers


def multiply_layers_with_loss(in_act, in_layers):
  x, _ = multiply_layers(in_act, in_layers)
  return jax.numpy.sum(x)


def calculate_tflops(f, *args, **kwargs):
  print(
      "Not calculating TFLOPS since MXLA is enabled -- for now just have a stored value for this test"
  )
  return 50


multiply_layers_and_grad = jax.value_and_grad(
    multiply_layers_with_loss, argnums=[1]
)


def training_loop(in_act, in_layers):
  _, grad_layers = multiply_layers_and_grad(in_act, in_layers)
  out_layers = jax.tree_map(
      lambda param, grad: param - 1e-4 * grad, in_layers, grad_layers[0]
  )
  return out_layers


print(f"finished includes ", flush=True)


# pjit NN
devices = jax.devices()
try:
  num_slices = 1 + max([d.slice_index for d in devices])
except:
  num_slices = 1

mesh_shape = [num_slices, len(jax.devices()) // num_slices]
devices_array = np.asarray(jax.devices()).reshape(*mesh_shape)
print(f"mesh shape {mesh_shape}", flush=True)
print(f"device layout {devices_array}", flush=True)
mesh = Mesh(devices_array, ("slices", "tpus"))


pjit_func = pjit(
    training_loop,
    in_shardings=(PartitionSpec(("slices", "tpus")), PartitionSpec("tpus")),
    out_shardings=PartitionSpec("tpus"),
)

pjit_gen_data = pjit(
    gen_data, in_shardings=None, out_shardings=PartitionSpec(("slices", "tpus"))
)

pjit_gen_layers = pjit(
    gen_layers, in_shardings=None, out_shardings=PartitionSpec("tpus")
)

print("compiles completed")

with Mesh(mesh.devices, mesh.axis_names):
  key = jax.random.PRNGKey(0)
  presharded_X = jax.block_until_ready(pjit_gen_data(key))
  presharded_layers = jax.block_until_ready(pjit_gen_layers(key))
  TFLOPs = calculate_tflops(training_loop, presharded_X, presharded_layers)
  with jax.profiler.trace("/tmp/tb12"):
    time = simple_timeit(
        lambda: jax.block_until_ready(
            pjit_func(presharded_X, presharded_layers)
        )
    )
  print(
      f"time is {time} seconds, TFLOP is {TFLOPs}, memory usage is {memory_bytes/10**9} GB, TFLOP/s is {TFLOPs/time}",
      flush=True,
  )

  assert (
      TFLOPs / time > 275 / 2
  ), "make sure that we're hitting the performance target, 50% peakflops"
