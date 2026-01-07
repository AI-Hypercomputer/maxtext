# Copyright 2023–2026 Google LLC
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

"""Matrix multiplication sharding"""

from functools import partial

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh


PREFILL_LENS = [128, 256, 512, 1024]
EMBED = 8192
MLP = 28672
LAYERS = 80
DTYPES = [jax.numpy.int8, jax.numpy.bfloat16]


def matmul(mesh, mesh_dim, dtype=jax.numpy.bfloat16, batch=1024, enable_visual=False):
  """sharded matrix multiplication"""
  A = jax.numpy.ones((batch, EMBED), dtype=dtype)
  activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
  A_ = jax.device_put(A, activation_sharding)
  if enable_visual:
    print("\nSharding A:")
    jax.debug.visualize_array_sharding(A_)

  W1 = jax.numpy.ones((EMBED, MLP), dtype=dtype)
  weight_sharding = jax.sharding.NamedSharding(
      mesh,
      jax.sharding.PartitionSpec(
          "model",
      ),
  )
  W1_ = jax.device_put(W1, weight_sharding)
  if enable_visual:
    print("\nSharding W1:")
    jax.debug.visualize_array_sharding(W1_)

  @partial(jax.jit, out_shardings=activation_sharding)
  def f(_A, _weights):
    _A = jax.lax.with_sharding_constraint(_A @ _weights, activation_sharding)
    return _A


def matmuls(mesh, mesh_dim, enable_visual=False):
  for dtype in DTYPES:
    print()
    for batch in PREFILL_LENS:
      matmul(mesh, mesh_dim, batch=batch, dtype=dtype, enable_visual=enable_visual)


# Start here
def main():
  devices = jax.devices()
  print("Devices:")
  print(devices)
  print()

  mesh_axis_names = ("model", "seq")
  for mesh_dim in [(4, 2), (8, 1)]:
    mesh = Mesh(devices=mesh_utils.create_device_mesh(mesh_dim, devices), axis_names=mesh_axis_names)
    matmuls(mesh, mesh_dim)

  mesh_dim = (4, 2)
  new_devices = [[devices[0], devices[4]], [devices[1], devices[5]], [devices[3], devices[7]], [devices[2], devices[6]]]
  mesh = jax.sharding.Mesh(new_devices, ["model", "seq"])
  print("Optimized device topology for 2x4")
  print(new_devices)
  matmuls(mesh, mesh_dim)


if __name__ == "__main__":
  main()
