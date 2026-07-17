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

"""Unmocked reproduction of the Pathways null-layout crash in the DiLoCo syncer.

This proves (against a *real* Pathways backend, not a mock) that:

  jax.device_put(numpy_array, sharding)

produces an array with a different Pathways-internal layout ("null") than an array of
the identical (shape, dtype, sharding) signature that came out of an actual jit/XLA
compilation ("tiled", e.g. learner transport or `stack_across_meshes_pytree`). Pathways
caches its internal compiled ops (jit__take, jit__scatter, ...) keyed by
(shape, dtype, sharding) WITHOUT layout, so the second array to hit an already-compiled
signature -- if its layout differs from the first -- is rejected at the Pathways/IFRT
proxy boundary. This is exactly the crash `_normalize_to_null_layout` (threaded_diloco.py)
and the shard-level rebuild in `_expand_array_dims_with_mesh` (mesh_utils.py) exist to fix.


Run against a single-host Pathways cluster:

    RUN_PATHWAYS_REPRO=1 python -m pytest tests/unit/pathways_null_layout_repro_test.py -v -s

See scripts/diloco/run_pathways_null_layout_repro.sh for an xpk launch wrapper.
"""

import os
import unittest

import numpy as np
import jax
import jax.numpy as jnp

from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator


@unittest.skipUnless(
    os.environ.get("RUN_PATHWAYS_REPRO") == "1",
    "Only meaningful against a live Pathways proxy backend. Set RUN_PATHWAYS_REPRO=1 "
    "and launch under a Pathways single-controller job to run for real "
    "(see scripts/diloco/run_pathways_null_layout_repro.sh).",
)
class DevicePutNullLayoutRealPathwaysReproTest(unittest.TestCase):
  """Unmocked reproduction: only meaningful run against an actual Pathways backend."""

  NUM_LAYERS = 8
  NUM_FRAGS = 4
  HIDDEN = 4

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2, "Need >=2 devices for a diloco/model mesh")
    self.mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), ("diloco", "model"))
    self.sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

  def _build_tiled_params(self, value):
    """Builds a param tree whose arrays come from JIT/XLA (tiled layout on Pathways)."""
    tiled_arr = jax.jit(
        lambda: jnp.full((self.NUM_LAYERS, self.HIDDEN), value, dtype=jnp.float32),
        out_shardings=self.sharding,
    )()
    return {"layers": {"w": tiled_arr}}

  def _build_null_params(self, value):
    """Builds a param tree whose arrays come from device_put (null layout on Pathways)."""
    np_layers = np.full((self.NUM_LAYERS, self.HIDDEN), value, dtype=np.float32)
    return {"layers": {"w": jax.device_put(np_layers, self.sharding)}}

  def _build_manipulator(self):
    fragment_to_layer_indices = {i + 1: np.array(list(range(i * 2, (i + 1) * 2))) for i in range(self.NUM_FRAGS)}
    keypath_to_is_scanned = {"['layers']['w']": True}
    return FragmentedTreeManipulator(
        keypath_to_is_scanned=keypath_to_is_scanned,
        fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=self.NUM_FRAGS + 1,
        param_scan_axis=0,
    )

  def test_unnormalized_tiled_array_triggers_pathways_layout_mismatch(self):
    """Reproduces the exact Pathways layout error when an unnormalized tiled TPU array
    (from JIT/XLA learner transport) is passed into use_null_layout_jit=True.
    """
    manipulator = self._build_manipulator()
    tiled_params = self._build_tiled_params(1.0)

    # Calling use_null_layout_jit=True directly on an unnormalized tiled array throws
    # the real Pathways layout error: "Layout passed to jit does not match the layout on the respective arg".
    manipulator.get_flat_fragment(tiled_params, fragment_idx=1, use_null_layout_jit=True)

  def test_eager_get_flat_fragment_matches_sharding_without_mismatch(self):
    """Verifies that extracting fragments with use_null_layout_jit=False on TPU arrays
    executes cleanly and extracts the correct sliced array shape without layout errors.
    """
    manipulator = self._build_manipulator()
    params = self._build_null_params(2.0)

    frag = manipulator.get_flat_fragment(params, fragment_idx=1, use_null_layout_jit=False)
    self.assertIn("['layers']['w']", frag)
    w = frag["['layers']['w']"]
    self.assertEqual(w.shape, (2, self.HIDDEN))
    self.assertEqual(w.dtype, jnp.float32)


if __name__ == "__main__":
  unittest.main()
