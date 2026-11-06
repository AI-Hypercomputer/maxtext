# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the DiLoCo mesh helpers in diloco_sharding."""

import itertools
import os
import unittest

from flax import linen as nn
import jax
from jax.sharding import AxisType, PartitionSpec
import numpy as np
import yaml

from maxtext.utils import diloco_sharding
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


def _two_replica_devices(diloco_axis: int) -> np.ndarray:
  """Returns the even prefix of the devices as a 3D array whose `diloco_axis` has size 2.

  The other two axes are sized from `jax.device_count()`, so the test runs on any even number of devices.
  """
  num_devices = jax.device_count() // 2 * 2
  if num_devices < 2:
    raise unittest.SkipTest("Splitting into 2 replicas needs >= 2 devices.")
  per_replica = num_devices // 2
  outer = 2 if per_replica % 2 == 0 else 1
  shape = [outer, per_replica // outer]
  shape.insert(diloco_axis, 2)
  return np.reshape(np.array(jax.devices()[:num_devices]), shape)


class SplitMeshAlongAxisTest(unittest.TestCase):

  def _assert_split(self, devices, axis_names, axis_types):
    """Splits a mesh on `diloco` and checks each submesh's devices, names and axis types."""
    mesh = jax.sharding.Mesh(devices, axis_names, axis_types=axis_types)
    diloco_axis = axis_names.index("diloco")
    keep = [i for i in range(len(axis_names)) if i != diloco_axis]
    submeshes = diloco_sharding.split_mesh_along_axis(mesh, "diloco")
    self.assertEqual(len(submeshes), devices.shape[diloco_axis])
    for i, submesh in enumerate(submeshes):
      self.assertEqual(submesh.axis_names, tuple(axis_names[k] for k in keep))
      self.assertEqual(submesh.axis_types, tuple(axis_types[k] for k in keep))
      np.testing.assert_array_equal(submesh.devices, np.take(devices, i, axis=diloco_axis))
    for a, b in itertools.combinations(submeshes, 2):
      self.assertFalse(set(a.devices.flat) & set(b.devices.flat))
    return submeshes

  def test_submeshes_hold_the_devices_of_each_index(self):
    # Mixed axis types, so dropping `axis_types=` (all-default types on the submesh) is caught.
    self._assert_split(
        _two_replica_devices(diloco_axis=1), ("data", "diloco", "fsdp"), (AxisType.Explicit, AxisType.Auto, AxisType.Auto)
    )

  def test_diloco_at_axis_0(self):
    # base.yml lists `diloco` first in `mesh_axes`.
    self._assert_split(
        _two_replica_devices(diloco_axis=0), ("diloco", "data", "fsdp"), (AxisType.Auto, AxisType.Explicit, AxisType.Auto)
    )

  def test_size_one_diloco_axis(self):
    # Runs on a single device.
    devices = np.reshape(np.array(jax.devices()), (1, jax.device_count(), 1))
    submeshes = self._assert_split(devices, ("diloco", "data", "fsdp"), (AxisType.Auto, AxisType.Explicit, AxisType.Auto))
    self.assertEqual(len(submeshes), 1)

  def test_missing_axis_raises(self):
    mesh = jax.sharding.Mesh(np.reshape(np.array(jax.devices()), (1, -1)), ("data", "fsdp"))
    with self.assertRaisesRegex(ValueError, "diloco"):
      diloco_sharding.split_mesh_along_axis(mesh, "diloco")


class RemoveMeshAxisFromRulesTest(unittest.TestCase):

  def test_drops_axis_and_rules_that_only_map_to_it(self):
    rules = [
        ("diloco", "diloco"),
        ("replica", ["diloco"]),
        ("batch", ["diloco", "data", "fsdp"]),
        ("embed", ("fsdp",)),
        ("heads", []),
        ("norm", None),
        ("activation", "tensor"),
    ]
    self.assertEqual(
        diloco_sharding.remove_mesh_axis_from_rules(rules, "diloco"),
        [("batch", ("data", "fsdp")), ("embed", ("fsdp",)), ("heads", ()), ("norm", None), ("activation", "tensor")],
    )

  def test_logical_axis_falls_through_to_next_matching_rule(self):
    for diloco_rule in ("diloco", ["diloco"]):
      with self.subTest(diloco_rule=diloco_rule):
        rules = [("diloco", diloco_rule), ("diloco", "data"), ("embed", ("fsdp",))]
        stripped = diloco_sharding.remove_mesh_axis_from_rules(rules, "diloco")
        self.assertEqual(nn.logical_to_mesh_axes(("diloco", "embed"), stripped), PartitionSpec("data", "fsdp"))

  def test_base_yml_rules_on_submesh(self):
    with open(os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"), encoding="utf-8") as f:
      base = yaml.safe_load(f)
    mesh_axes = tuple(base["mesh_axes"])
    self.assertEqual(mesh_axes[0], "diloco")
    num_replicas = 2 if jax.device_count() >= 2 else 1
    num_devices = jax.device_count() // num_replicas * num_replicas
    shape = [num_replicas, num_devices // num_replicas] + [1] * (len(mesh_axes) - 2)
    mesh = jax.sharding.Mesh(np.reshape(np.array(jax.devices()[:num_devices]), shape), mesh_axes)
    submesh = diloco_sharding.split_mesh_along_axis(mesh, "diloco")[0]
    rules = base["logical_axis_rules"]
    stripped = diloco_sharding.remove_mesh_axis_from_rules(rules, "diloco")
    embed_axes = tuple(next(physical for logical, physical in rules if logical == "embed"))

    sharding = nn.logical_to_mesh_sharding(PartitionSpec("diloco", "embed"), submesh, stripped)
    self.assertEqual(sharding.mesh, submesh)
    self.assertEqual(sharding.spec, PartitionSpec(None, embed_axes))
    for logical_axis in {logical for logical, _ in rules}:
      nn.logical_to_mesh_sharding(PartitionSpec(logical_axis), submesh, stripped)
    # Unstripped rules name `diloco`, which the submesh does not have.
    with self.assertRaises(ValueError):
      nn.logical_to_mesh_sharding(PartitionSpec("diloco", "embed"), submesh, rules)


if __name__ == "__main__":
  unittest.main()
