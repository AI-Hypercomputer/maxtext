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
from maxtext.trainers.diloco.threaded_diloco import _normalize_to_null_layout


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

  def _build_params(self, value):
    np_layers = np.full((self.NUM_LAYERS, self.HIDDEN), value, dtype=np.float32)
    return {"layers": {"w": jax.device_put(np_layers, self.sharding)}}

  def _build_manipulator(self):
    fragment_to_layer_indices = {
        i + 1: jnp.array(list(range(i * 2, (i + 1) * 2))) for i in range(self.NUM_FRAGS)
    }
    keypath_to_is_scanned = {"['layers']['w']": True}
    return FragmentedTreeManipulator(
        keypath_to_is_scanned=keypath_to_is_scanned,
        fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=self.NUM_FRAGS + 1,
        param_scan_axis=0,
    )

  def test_device_put_from_numpy_then_jit_take_crashes_on_mixed_signature(self):
    """Reproduces the exact sequence the pre-fix `_run_syncer_loop` executed:

    1. `tiled_params`: built by an actual jit computation -- stands in for arrays that
       arrive via learner transport / `stack_across_meshes_pytree` (concrete/tiled
       Pathways layout).
    2. `null_params`: built the way the syncer built `global_params` before the fix --
       fresh `jax.device_put(numpy_array, sharding)` (null Pathways layout, per
       `_normalize_to_null_layout`'s docstring in threaded_diloco.py).
    3. Both trees have identical (shape, dtype, sharding). Calling the bare
       `manipulator.get_flat_fragment(..., use_null_layout_jit=False)` -- i.e. eager
       jnp.take, exactly what the old buggy `_run_syncer_loop` code did -- on
       tiled_params first (compiling/caching Pathways' internal jit__take for this
       signature) and then on null_params (same signature, different layout) must crash
       with a Pathways layout-mismatch error on real hardware.
    """
    manipulator = self._build_manipulator()

    tiled_params = jax.jit(lambda t: jax.tree_util.tree_map(lambda x: x + 0.0, t))(
        self._build_params(1.0)
    )
    null_params = self._build_params(2.0)

    # First call compiles/caches Pathways' internal jit for this (shape, dtype, sharding).
    frag_tiled = manipulator.get_flat_fragment(tiled_params, fragment_idx=1)
    self.assertIsNotNone(frag_tiled)

    # Second call: identical signature, different (null) layout -> crashes on Pathways.
    with self.assertRaises(Exception) as ctx:
      manipulator.get_flat_fragment(null_params, fragment_idx=1)
    err = str(ctx.exception).lower()
    self.assertTrue(
        "layout" in err or "tiling" in err,
        f"Expected a layout/tiling mismatch error, got: {ctx.exception!r}",
    )

  def test_normalize_to_null_layout_fixes_the_crash(self):
    """Same sequence as above, but both arrays first pass through the codebase's actual
    fix (`_normalize_to_null_layout`, threaded_diloco.py) -- confirms this is what
    resolves the crash on real Pathways hardware."""
    manipulator = self._build_manipulator()

    tiled_params = jax.jit(lambda t: jax.tree_util.tree_map(lambda x: x + 0.0, t))(
        self._build_params(1.0)
    )
    null_params = self._build_params(2.0)

    tiled_params = _normalize_to_null_layout(tiled_params)
    null_params = _normalize_to_null_layout(null_params)

    frag_a = manipulator.get_flat_fragment(tiled_params, fragment_idx=1)
    frag_b = manipulator.get_flat_fragment(null_params, fragment_idx=1)  # must NOT raise
    np.testing.assert_allclose(np.array(frag_a["['layers']['w']"]), 1.0)
    np.testing.assert_allclose(np.array(frag_b["['layers']['w']"]), 2.0)


if __name__ == "__main__":
  unittest.main()
