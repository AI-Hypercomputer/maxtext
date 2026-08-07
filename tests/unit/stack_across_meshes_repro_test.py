# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the me.

"""Reproduction test for stack_across_meshes_pytree gRPC socket contention issue under Pathways.

This test stress-tests stack_across_meshes_pytree by extracting large model PyTrees
in a background thread while heavy JIT computations run on the main thread,
reproducing host-device gRPC stream contention and socket termination under Pathways.

It is passed. But it shouldn't pass. It is meant to reproduce the crash I see in e2e runs.
"""

import os
import threading
import time
import unittest

try:
  import jax_plugins.xla_proxy  # noqa: F401
except Exception:
  try:
    import pathways  # noqa: F401
  except Exception:
    if os.environ.get("JAX_PLATFORMS") == "proxy":
      os.environ.pop("JAX_PLATFORMS", None)

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.utils.mesh_utils import stack_across_meshes_pytree


class StackAcrossMeshesReproTest(unittest.TestCase):

  def test_concurrent_stack_across_meshes(self):
    """Simulates background syncer thread calling stack_across_meshes_pytree while main thread executes heavy JIT ops."""
    devices = jax.devices()
    num_devices = len(devices)

    if num_devices < 2:
      devices = [devices[0], devices[0]]
      num_devices = 2

    mid = num_devices // 2
    submesh_devices_0 = np.array(devices[:mid]).reshape((mid, 1))
    submesh_devices_1 = np.array(devices[mid:]).reshape((num_devices - mid, 1))
    global_devices = np.array(devices).reshape((num_devices, 1))

    submesh_0 = jax.sharding.Mesh(submesh_devices_0, axis_names=("data", "model"))
    submesh_1 = jax.sharding.Mesh(submesh_devices_1, axis_names=("data", "model"))
    global_mesh = jax.sharding.Mesh(global_devices, axis_names=("diloco", "data"))

    sharding_0 = jax.sharding.NamedSharding(submesh_0, jax.sharding.PartitionSpec())
    sharding_1 = jax.sharding.NamedSharding(submesh_1, jax.sharding.PartitionSpec())

    # Build realistic large parameter PyTrees (30 layers of (4096, 4096) matrices ~2GB total)
    matrix_shape = (4096, 4096)
    num_layers = 30

    param_tree_0 = {}
    param_tree_1 = {}

    for i in range(num_layers):
      param_tree_0[f"layer_{i}_weight_0"] = jax.device_put(jnp.ones(matrix_shape, dtype=jnp.float32), sharding_0)
      param_tree_0[f"layer_{i}_weight_1"] = jax.device_put(jnp.zeros(matrix_shape, dtype=jnp.float32), sharding_0)

      param_tree_1[f"layer_{i}_weight_0"] = jax.device_put(jnp.ones(matrix_shape, dtype=jnp.float32) * 2.0, sharding_1)
      param_tree_1[f"layer_{i}_weight_1"] = jax.device_put(jnp.zeros(matrix_shape, dtype=jnp.float32) * 3.0, sharding_1)

    learner_frags = [param_tree_0, param_tree_1]

    stop_event = threading.Event()
    errors = []
    completed_syncs = [0]

    # Background syncer thread: repeatedly calls stack_across_meshes_pytree in a tight loop
    def syncer_worker():
      iteration = 0
      while not stop_event.is_set() and iteration < 200:
        try:
          stacked = stack_across_meshes_pytree(learner_frags, global_mesh, "diloco")
          # Force evaluation of stacked array leaves
          for key in ["layer_0_weight_0", "layer_10_weight_1", "layer_29_weight_0"]:
            if key in stacked:
              jax.block_until_ready(stacked[key])
          iteration += 1
          completed_syncs[0] = iteration
        except Exception as e:
          errors.append(e)
          print(f"[REPRO TEST] Background syncer thread crashed on iteration {iteration}: {e}")
          break

    # Start background syncer thread
    syncer_thread = threading.Thread(target=syncer_worker)
    syncer_thread.start()

    # Main learner thread: runs heavy JIT matrix multiplications continuously
    @jax.jit
    def heavy_compute_step(x, y):
      z = jnp.dot(x, y) + jnp.sin(x)
      w = jnp.dot(z, y.T) + jnp.cos(y)
      return jnp.dot(w, z.T)

    x_sharded = jax.device_put(jnp.ones(matrix_shape, dtype=jnp.float32), sharding_0)
    y_sharded = jax.device_put(jnp.ones(matrix_shape, dtype=jnp.float32) * 0.5, sharding_0)

    # Run main thread JIT loop for 300 steps
    for step in range(300):
      x_sharded = heavy_compute_step(x_sharded, y_sharded)
      if errors:
        print(f"[REPRO TEST] Main thread detected syncer crash at step {step}")
        break

    stop_event.set()
    syncer_thread.join(timeout=45)

    print(f"[REPRO TEST] Completed {completed_syncs[0]} syncer iterations successfully.")

    if errors:
      self.fail(f"Background syncer thread failed with error: {errors[0]}")


if __name__ == "__main__":
  unittest.main()
