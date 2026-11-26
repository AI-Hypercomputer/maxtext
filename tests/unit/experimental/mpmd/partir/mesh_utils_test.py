# Copyright 2026 Google LLC
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

"""Tests for the MPMD pipeline's stage meshes, on CPU devices and fake TPU coordinates."""

# pylint: disable=protected-access

import dataclasses
import unittest

import jax
import numpy as np

from maxtext.experimental.mpmd.partir import mesh_utils as mu


def cpu_devices(count):
  """The first `count` CPU devices, or a skip when there are fewer."""
  devices = jax.devices("cpu")
  if len(devices) < count:
    raise unittest.SkipTest(f"needs {count} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count={count})")
  return devices[:count]


@dataclasses.dataclass(frozen=True)
class FakeTpu:
  id: int
  coords: tuple[int, int, int]
  core_on_chip: int = 0
  slice_index: int = 0


def grid(x, y, z, slice_index=0):
  """Fake chips of an x*y*z box, host by host (2x2 tiles), like jax.devices() lists them."""
  devs = []
  for hz in range(z):
    for hy in range(0, y, 2):
      for hx in range(0, x, 2):
        for dy in range(min(2, y)):
          for dx in range(min(2, x)):
            devs.append(FakeTpu(100_000 * (slice_index + 1) + len(devs), (hx + dx, hy + dy, hz), slice_index=slice_index))
  return devs


class StageMeshTest(unittest.TestCase):

  def test_cpu_devices_split_contiguously(self):
    devices = cpu_devices(4)
    topo = mu.create_stage_meshes(devices, num_stages=2)
    self.assertEqual(topo.devices_per_stage, 2)
    self.assertEqual(topo.fsdp_size, 2)
    self.assertEqual(list(topo.meshes), ["m0", "m1"])
    self.assertEqual(topo.meshes["m0"].axis_names, ("fsdp",))
    self.assertEqual([d.id for d in topo.meshes["m0"].devices.flat], [d.id for d in devices[:2]])
    flat = [d for m in topo.meshes.values() for d in m.devices.flat]
    self.assertEqual(sorted(d.id for d in flat), sorted(d.id for d in devices))

  def test_expert_axis(self):
    devices = cpu_devices(4)
    topo = mu.create_stage_meshes(devices, num_stages=1, expert_size=2)
    self.assertEqual(topo.meshes["m0"].axis_names, ("fsdp", "expert"))
    self.assertEqual(dict(topo.meshes["m0"].shape), {"fsdp": 2, "expert": 2})
    self.assertEqual(topo.data_parallel_size, 4)
    with self.assertRaisesRegex(ValueError, "divisible by num_stages"):
      mu.create_stage_meshes(devices, num_stages=3)
    with self.assertRaisesRegex(ValueError, "exceeds"):
      mu.create_stage_meshes(devices, num_stages=2, fsdp_size=2, expert_size=2)

  def test_4x4x4_stages_are_whole_host_subcubes(self):
    """64 chips into 4 stages: each a 2x2x4 sub-cube (an XLA-supported 3D subslice)."""
    stages = mu._partition_grid_into_stages(np.asarray(grid(4, 4, 4), dtype=object), 4)
    self.assertEqual(len(stages), 4)
    for stage in stages:
      xs, ys, zs = zip(*(d.coords for d in stage))
      box = (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, max(zs) - min(zs) + 1)
      self.assertEqual(box, (2, 2, 4))
      self.assertEqual(min(xs) % 2, 0)
      self.assertEqual(min(ys) % 2, 0)

  def test_2x2x2_v5p16_two_stages_per_host(self):
    """The v5p-16 run on 2026-10-15 (pathways demo): stages 0-1 on host z=0, 2-3 on z=1."""
    stages = mu._partition_grid_into_stages(np.asarray(grid(2, 2, 2), dtype=object), 4)
    for index, stage in enumerate(stages):
      self.assertEqual({d.coords[2] for d in stage}, {index // 2})

  def test_consecutive_stages_are_neighbours(self):
    """The stage blocks follow a snake: each stage's box touches the next one's (a 4x4 v6e-16 in four
    2x2 hosts, a 4x4x4 cube in eight 2x2x2 blocks, 2x2x2 in four 1x2x1 halves)."""

    def box(stage):
      xs, ys, zs = zip(*(d.coords for d in stage))
      return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

    def touching(a, b):  # boxes share a face: adjacent (gap 1) along one axis, overlapping along the others
      gaps = [max(lo2 - hi1, lo1 - hi2) for (lo1, hi1), (lo2, hi2) in zip(a, b)]
      return gaps.count(1) == 1 and all(g <= 0 for g in gaps if g != 1)

    for shape, num_stages in (((4, 4, 1), 4), ((4, 4, 4), 8), ((2, 2, 2), 4), ((4, 4, 4), 4)):
      stages = mu._partition_grid_into_stages(np.asarray(grid(*shape), dtype=object), num_stages)
      boxes = [box(stage) for stage in stages]
      for s in range(num_stages - 1):
        self.assertTrue(touching(boxes[s], boxes[s + 1]), (shape, num_stages, s, boxes[s], boxes[s + 1]))
    self.assertEqual(
        mu._snake_order_blocks(2, 2, 2),
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)],
    )

  def test_stages_never_straddle_slices(self):
    """Every slice's chips share coordinates: two v5p-16 slices into 4 stages put stages 0-1 on
    slice 0 and 2-3 on slice 1, each a whole host as on one slice, instead of mixing both
    slices into every stage."""
    devices = grid(2, 2, 2, slice_index=0) + grid(2, 2, 2, slice_index=1)
    topo = mu.create_stage_meshes(devices, num_stages=4)
    for index, mesh in enumerate(topo.meshes.values()):
      self.assertEqual({d.slice_index for d in mesh.devices.flat}, {index // 2})
      self.assertEqual({d.coords[2] for d in mesh.devices.flat}, {index % 2})
    # More slices than stages: each stage holds whole slices.
    devices = [d for i in range(4) for d in grid(2, 2, 2, slice_index=i)]
    topo = mu.create_stage_meshes(devices, num_stages=2)
    self.assertEqual([sorted({d.slice_index for d in m.devices.flat}) for m in topo.meshes.values()], [[0, 1], [2, 3]])
    with self.assertRaisesRegex(ValueError, "straddles"):
      mu.create_stage_meshes([d for i in range(3) for d in grid(2, 2, 2, slice_index=i)], num_stages=4)

  def test_supported_subslices(self):
    self.assertTrue(mu._is_supported_3d_subslice(2, 2, 4))
    self.assertTrue(mu._is_supported_3d_subslice(4, 4, 8))
    self.assertFalse(mu._is_supported_3d_subslice(1, 4, 4))


class AssignmentTest(unittest.TestCase):

  def test_uniform_and_asymmetric(self):
    a = mu.create_name_to_mesh_assignment(num_layers=8, num_stages=4)
    self.assertEqual([a[mu.layer_name(i)] for i in range(8)], ["m0", "m0", "m1", "m1", "m2", "m2", "m3", "m3"])
    self.assertEqual((a["embed"], a["unembed"]), ("m0", "m3"))
    self.assertEqual((a["pre_m2"], a["post_m2"]), ("m2", "m2"))
    b = mu.create_name_to_mesh_assignment(num_layers=8, num_stages=2, stage_layer_counts=(3, 5))
    self.assertEqual([b[mu.layer_name(i)] for i in range(8)], ["m0"] * 3 + ["m1"] * 5)

  def test_circular(self):
    a = mu.create_name_to_mesh_assignment(num_layers=8, num_stages=2, num_repeats=2)
    self.assertEqual([a[mu.layer_name(i)] for i in range(8)], ["m0", "m0", "m1", "m1", "m0", "m0", "m1", "m1"])
    s = mu.create_name_to_stage_assignment(num_layers=8, num_stages=2, num_repeats=2)
    self.assertEqual([s[mu.layer_name(i)] for i in range(8)], [0, 0, 1, 1, 2, 2, 3, 3])
    # pre/post computations get their own stage ids, so Shardy does not merge them into a call.
    self.assertEqual(len({s["pre_m0"], s["post_m0"], s["embed"]}), 3)


if __name__ == "__main__":
  unittest.main()
