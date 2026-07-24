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

"""Live Pathways regression for DiLoCo's compiled-input-format adapter.

Arrays with the same logical shape, dtype, and sharding can reach the syncer with
different physical formats depending on whether they were produced by JIT or by
``jax.device_put``. The format-adapted fragment take compiles the operation, reads
the executable-selected input format, and places each argument into that format
before execution. This test verifies that both array origins work without assuming
that either has a particular physical layout.

Run against a single-host Pathways cluster:

    RUN_PATHWAYS_REPRO=1 python -m pytest tests/unit/pathways_null_layout_repro_test.py -v -s

See scripts/diloco/run_pathways_null_layout_repro.sh for an xpk launch wrapper.
"""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.trainers.diloco.fragmenter import FragmentedTreeManipulator


@unittest.skipUnless(
    os.environ.get("RUN_PATHWAYS_REPRO") == "1",
    "Only meaningful against a live Pathways proxy backend. Set RUN_PATHWAYS_REPRO=1 "
    "and launch under a Pathways single-controller job to run it "
    "(see scripts/diloco/run_pathways_null_layout_repro.sh).",
)
class CompiledInputFormatRealPathwaysRegressionTest(unittest.TestCase):
  """Exercises format-adapted fragment extraction on an actual Pathways backend."""

  NUM_LAYERS = 8
  NUM_FRAGS = 4
  HIDDEN = 4

  def setUp(self):
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2, "Need >=2 devices for a diloco/model mesh")
    self.mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), ("diloco", "model"))
    self.sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

  def _host_values(self, offset):
    values = np.arange(self.NUM_LAYERS * self.HIDDEN, dtype=np.float32)
    return values.reshape(self.NUM_LAYERS, self.HIDDEN) + np.float32(offset)

  def _build_jit_params(self, offset):
    """Builds a parameter tree whose array is produced by JIT/XLA."""
    jit_array = jax.jit(
        lambda: jnp.arange(self.NUM_LAYERS * self.HIDDEN, dtype=jnp.float32).reshape(self.NUM_LAYERS, self.HIDDEN)
        + jnp.float32(offset),
        out_shardings=self.sharding,
    )()
    return {"layers": {"w": jax.block_until_ready(jit_array)}}

  def _build_device_put_params(self, offset):
    """Builds a parameter tree whose array is produced by device_put."""
    array = jax.device_put(self._host_values(offset), self.sharding)
    return {"layers": {"w": jax.block_until_ready(array)}}

  def _build_manipulator(self):
    fragment_to_layer_indices = {i + 1: np.arange(i * 2, (i + 1) * 2, dtype=np.int32) for i in range(self.NUM_FRAGS)}
    keypath_to_is_scanned = {"['layers']['w']": True}
    return FragmentedTreeManipulator(
        keypath_to_is_scanned=keypath_to_is_scanned,
        fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=self.NUM_FRAGS + 1,
        param_scan_axis=0,
    )

  def _assert_adapted_fragment(self, manipulator, params, fragment_idx, expected_full):
    fragment = manipulator.get_flat_fragment(
        params,
        fragment_idx=fragment_idx,
        use_null_layout_jit=True,
    )
    actual = jax.block_until_ready(fragment["['layers']['w']"])
    indices = manipulator.fragment_to_layer_indices[fragment_idx]
    expected = expected_full[indices]

    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_array_equal(np.asarray(actual), expected)

  def test_layout_adapted_take_succeeds_for_jit_and_device_put_arrays(self):
    """Both input origins are converted to the compiled take's required format."""
    manipulator = self._build_manipulator()
    fragment_idx = 2
    cases = (
        ("jit", self._build_jit_params(offset=0), self._host_values(offset=0)),
        (
            "device_put",
            self._build_device_put_params(offset=100),
            self._host_values(offset=100),
        ),
    )

    for source, params, expected_full in cases:
      with self.subTest(source=source):
        self._assert_adapted_fragment(
            manipulator,
            params,
            fragment_idx,
            expected_full,
        )


if __name__ == "__main__":
  unittest.main()
