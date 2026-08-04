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

"""Minimal unit test for Pathways null-layout verification on colocated CPU hosts.

This unit test verifies against a live Pathways backend that:
1. Tiled arrays produced by XLA on TPU are normalized when transferred to colocated CPU devices.
2. The resulting CPU array has a null-layout (tiling=None) and executes cleanly under null-layout JIT functions.

Run against a live Pathways cluster:
    RUN_PATHWAYS_REPRO=1 python -m pytest tests/unit/pathways_colocated_cpu_null_layout_test.py -v -s
"""

import os
import unittest

import numpy as np
import jax
import jax.numpy as jnp

from jax.experimental import colocated_python
from jax.experimental.layout import Format, Layout
from maxtext.trainers.diloco.threaded_diloco import _normalize_to_null_layout


@unittest.skipUnless(
    os.environ.get("RUN_PATHWAYS_REPRO") == "1",
    "Only meaningful against a live Pathways proxy backend. Set RUN_PATHWAYS_REPRO=1 "
    "and launch under a Pathways single-controller job to run for real.",
)
class ColocatedCpuNullLayoutTest(unittest.TestCase):
  """Minimal unit test for verifying Pathways null-layout normalization on colocated CPU hosts."""

  NUM_LAYERS = 8
  HIDDEN = 4

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2, "Need >=2 devices for a diloco/model mesh")
    self.mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), ("diloco", "model"))
    self.sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

  def test_minimal_colocated_cpu_null_layout(self):
    """Verifies that placing a tiled TPU array onto colocated CPU hosts
    normalizes it to a null-layout tensor that executes cleanly under a null-layout JIT function.
    """
    # 1. Create a tiled tensor on TPU (produced by JIT/XLA)
    tpu_arr = jax.jit(
        lambda: jnp.ones((self.NUM_LAYERS, self.HIDDEN), dtype=jnp.float32),
        out_shardings=self.sharding,
    )()
    jax.block_until_ready(tpu_arr)

    # 2. Transfer to colocated CPU mesh with null-layout normalization
    cpu_mesh = colocated_python.colocated_cpu_devices(self.mesh)
    cpu_sharding = jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec())
    cpu_arr = _normalize_to_null_layout(jax.device_put(tpu_arr, cpu_sharding))

    # 3. Define JIT function strictly expecting null layout (tiling=None)
    null_layout = Layout(major_to_minor=tuple(range(cpu_arr.ndim)), tiling=None)
    null_format = Format(layout=null_layout, sharding=cpu_sharding)

    @jax.jit(in_shardings=(null_format,), out_shardings=cpu_sharding)
    def cpu_fn(x):
      return x * 2.0

    # 4. Assert execution succeeds without throwing Pathways layout mismatch ValueError
    res = cpu_fn(cpu_arr)
    jax.block_until_ready(res)
    self.assertEqual(res.shape, cpu_arr.shape)
    self.assertEqual(res.sharding, cpu_sharding)


  def test_colocated_cpu_no_null_layout_jit_fails(self):
    """Verifies that placing a tiled TPU array onto colocated CPU hosts
    WITHOUT null-layout normalization causes layout mismatch failure.

    Note: This proves that WITHOUT calling `_normalize_to_null_layout`,
    the jax operations will fail because the tensor retains its TPU tiling
    on the CPU when consumed by a CPU JIT function.
    """
    # 1. Create a tiled tensor on TPU (produced by JIT/XLA)
    tpu_arr = jax.jit(
        lambda: jnp.ones((self.NUM_LAYERS, self.HIDDEN), dtype=jnp.float32),
        out_shardings=self.sharding,
    )()
    jax.block_until_ready(tpu_arr)

    # 2. Transfer to colocated CPU mesh WITHOUT null-layout normalization
    cpu_mesh = colocated_python.colocated_cpu_devices(self.mesh)
    cpu_sharding = jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec())
    
    # We do NOT call _normalize_to_null_layout here.
    cpu_arr = jax.device_put(tpu_arr, cpu_sharding)

    # 3. Define JIT function strictly expecting null layout (tiling=None)
    null_layout = Layout(major_to_minor=tuple(range(cpu_arr.ndim)), tiling=None)
    null_format = Format(layout=null_layout, sharding=cpu_sharding)

    @jax.jit(in_shardings=(null_format,), out_shardings=cpu_sharding)
    def cpu_fn(x):
      return x * 2.0

    # 4. Assert execution fails because cpu_arr still retains TPU tiling
    with self.assertRaises(Exception):
      res = cpu_fn(cpu_arr)
      jax.block_until_ready(res)


if __name__ == "__main__":
  unittest.main()
