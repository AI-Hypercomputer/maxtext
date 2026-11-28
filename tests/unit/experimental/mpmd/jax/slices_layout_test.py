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

"""--layout slices -- one MPMD stage per TPU slice -- without a TPU.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python3 -m pytest -q tests/unit/experimental/mpmd/jax/slices_layout_test.py

The grouping into slices (`slice_groups`; MPMD_FAKE_SLICES on CPU), the
per-slice topology-aware meshes on fake TPU devices of two v5p-128 slices,
k stages a slice cut into strips of whole chips on fake devices of two
tpu7x-4x4x4 slices and of a tpu7x-8 (two TensorCores a chip: SlicesOnTpu7x),
which stages libtpu keeps synchronous when a stage is a whole slice
(`sync_collective_stages`, libtpu's own answers every process plans by,
`offload_answers`, and main's line comparing them, `offload_check`), and
ds_ep_aot's multi-slice pieces, and the same devices as a Pathways client
sees them (process_index 0 everywhere, the IFRT proxy's client: PathwaysShaped).  Where libtpu is
installed, compile-only topologies of 2 to 4 v5p-8 slices (in a subprocess,
CPU backend: no TPU is used): the boundary copies between slices compile
(send / recv), nothing compiles without a device on the first slice
(libtpu 0.0.46), and a stage or copy moved down onto it
(`ds_ep_aot.first_slice_meshes`, `first_slice_channel`) does; and the plan
ds_ep_aot builds for the Qwen3.8-2.4T-A95B run on 8 x tpu7x-4x4x4; and,
scheduled_only, ds_ep_aot's whole multi-slice AOT of TINY pipelines there,
and of a Qwen3.5-family one on 2 x tpu7x-4x4x4.  The
multi-process runs -- two fake slices of two processes, the boundaries
collective-permutes between them -- are multihost_test.py's
test_one_stage_per_slice{,_fsdp_x_ep}.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_aot
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from tests.unit.experimental.mpmd.jax import multihost_test
from maxtext.experimental.mpmd.jax import pathways
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import transfer

ROOT = testing.import_root()
HAVE_LIBTPU = importlib.util.find_spec("libtpu") is not None


def setUpModule():
  testing.require_cpu()


@dataclasses.dataclass(frozen=True)
class Dev:
  """A device as slice_groups, stage_meshes and sync_collective_stages see it:
  a CPU one by default (no slice, no chip coordinates), a TPU one with
  `coords` (and mesh_utils' device_kind) and its slice's `slice_index`.  No
  `client` (whose devices() are all the job's or topology's) unless given."""

  id: int
  process_index: int
  coords: tuple | None = None
  slice_index: int | None = None
  core_on_chip: int = 0
  platform: str = "cpu"
  device_kind: str = "cpu"
  client: object = dataclasses.field(default=None, repr=False)


def fake_v5p(slices=2, x=4, y=4, z=4):
  """`slices` v5p slices of x by y by z chips (a v5p-128 by default), one
  device a chip, as libtpu's compile-only topology numbers them: ids
  (slice + 1) * 100000 + x + X y + X Y z, a host per 2x2x1 chips, 16
  processes a v5p-128 slice, process ids slice by slice."""
  hosts = (x // 2) * (y // 2) * z
  devs = []
  for k in range(slices):
    for cz in range(z):
      for cy in range(y):
        for cx in range(x):
          host = cx // 2 + (x // 2) * (cy // 2) + (x // 2) * (y // 2) * cz
          devs.append(
              Dev(
                  id=(k + 1) * 100000 + cx + x * cy + x * y * cz,
                  process_index=k * hosts + host,
                  coords=(cx, cy, cz),
                  slice_index=k,
                  platform="tpu",
                  device_kind="TPU v5",
              )
          )
  return devs


def fake_tpu7x(slices=2, x=4, y=4, z=4):
  """`slices` TPU7x slices of x by y by z chips (a tpu7x-4x4x4 by default), two TensorCores a chip, each a device
  (`core_on_chip`), as libtpu's compile-only topology numbers them: ids (slice + 1) * 100000 + 2 (x + X y + X Y z) +
  core, a host per 2x2x1 chips (8 devices), 16 processes a tpu7x-4x4x4 slice, process ids slice by slice."""
  hosts = (x // 2) * (y // 2) * z
  devs = []
  for k in range(slices):
    for cz in range(z):
      for cy in range(y):
        for cx in range(x):
          for core in range(2):
            devs.append(
                Dev(
                    id=(k + 1) * 100000 + 2 * (cx + x * cy + x * y * cz) + core,
                    process_index=k * hosts + cx // 2 + (x // 2) * (cy // 2) + (x // 2) * (y // 2) * cz,
                    coords=(cx, cy, cz),
                    slice_index=k,
                    core_on_chip=core,
                    platform="tpu",
                    device_kind="TPU7x",
                )
            )
  return devs


def ids(meshes):
  return [m.device_ids.tolist() for m in meshes]


def fake_slices(n):
  return mock.patch.dict(os.environ, {"MPMD_FAKE_SLICES": str(n)})


class FakeSlices(unittest.TestCase):
  """MPMD_FAKE_SLICES on CPU devices, which have no slices of their own: four of them, whatever the host has."""

  def setUp(self):
    if len(jax.devices()) < 4:
      self.skipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    self.devs = jax.devices()[:4]
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("MPMD_FAKE_SLICES", None)

  def test_a_stage_per_fake_slice(self):
    """Two fake slices of two devices: stage s is slice s, in (process, id)
    order as (fsdp, ep) -- on CPU in one process 'ep_on_chip''s meshes --
    whatever order the devices come in.  Four slices of one at fsdp 1, ep 1."""
    shuffled = random.Random(0).sample(self.devs, len(self.devs))
    with fake_slices(2):
      self.assertEqual([[d.id for d in g] for g in dsp.slice_groups(shuffled)], [[0, 1], [2, 3]])
      meshes = dsp.stage_meshes(2, 1, 2, devices=shuffled, layout="slices")
      self.assertEqual([dict(m.shape) for m in meshes], [{"fsdp": 1, "expert": 2}] * 2)
      self.assertEqual(ids(meshes), [[[0, 1]], [[2, 3]]])
      self.assertEqual(ids(meshes), ids(dsp.stage_meshes(2, 1, 2, devices=self.devs, layout="ep_on_chip")))
      self.assertEqual(ids(dsp.stage_meshes(2, 2, 1, devices=self.devs, layout="slices")), [[[0], [1]], [[2], [3]]])
    with fake_slices(4):
      self.assertEqual(ids(dsp.stage_meshes(4, 1, 1, devices=self.devs, layout="slices")), [[[0]], [[1]], [[2]], [[3]]])
    self.assertEqual(dsp.sync_collective_stages(meshes), frozenset())  # CPU: nothing to offload

  def test_without_fake_slices_the_devices_are_one_slice(self):
    """One slice: one stage of all of it, or k stages of its devices cut into k strips in (process, id) order."""
    self.assertEqual([len(g) for g in dsp.slice_groups(self.devs)], [4])
    self.assertEqual(ids(dsp.stage_meshes(1, 2, 2, devices=self.devs, layout="slices")), [[[0, 1], [2, 3]]])
    self.assertEqual(ids(dsp.stage_meshes(2, 1, 2, devices=self.devs, layout="slices")), [[[0, 1]], [[2, 3]]])
    with self.assertRaisesRegex(
        ValueError, r"3 stages of fsdp x ep = 1 devices, but the devices are 1 " r"slice\(s\) of \[4\]"
    ):
      dsp.stage_meshes(3, 1, 1, devices=self.devs, layout="slices")

  def test_k_stages_a_slice(self):
    """Two fake slices of two devices, two stages a slice: stages 0 and 1 on slice 0, 2 and 3 on slice 1, each
    strip in (process, id) order, whatever order the devices come in."""
    shuffled = random.Random(0).sample(self.devs, len(self.devs))
    with fake_slices(2):
      meshes = dsp.stage_meshes(4, 1, 1, devices=shuffled, layout="slices")
      self.assertEqual(ids(meshes), [[[0]], [[1]], [[2]], [[3]]])
      self.assertEqual([dict(m.shape) for m in meshes], [{"fsdp": 1, "expert": 1}] * 4)
    with fake_slices(1):
      self.assertEqual(ids(dsp.stage_meshes(4, 1, 1, devices=self.devs, layout="slices")), [[[0]], [[1]], [[2]], [[3]]])
      self.assertEqual(ids(dsp.stage_meshes(2, 2, 1, devices=self.devs, layout="slices")), [[[0], [1]], [[2], [3]]])

  def test_stages_and_slices_must_agree(self):
    """The stages a multiple of the slices, each slice k stages' devices."""
    with fake_slices(2):
      for stages in (3, 2):
        with (
            self.subTest(stages=stages),
            self.assertRaisesRegex(
                ValueError,
                rf"runs k stages on each slice \(n_stages a multiple of the slices\): {stages} stages of fsdp x ep = "
                r"1 devices, but the devices are 2 slice\(s\) of \[2, 2\]",
            ),
        ):
          dsp.stage_meshes(stages, 1, 1, devices=self.devs, layout="slices")
      with self.assertRaisesRegex(ValueError, "need 8 devices, have 4"):
        dsp.stage_meshes(2, 2, 2, devices=self.devs, layout="slices")

  def test_bad_settings(self):
    for bad in ("0", "-2", "two", " 2"):
      with self.subTest(bad=bad), fake_slices(bad), self.assertRaisesRegex(ValueError, "positive number"):
        dsp.slice_groups(self.devs)
    with fake_slices(3), self.assertRaisesRegex(ValueError, "4 devices do not cut into 3 equal slices"):
      dsp.slice_groups(self.devs)

  def test_no_fake_slice_splits_a_process(self):
    """A host belongs to one slice: across processes each fake slice holds
    whole ones (one process holding every device may be cut: a one-process
    run).  Devices of several slices refuse MPMD_FAKE_SLICES; under
    jax.distributed CPU devices carry slice 0, and take it."""
    four = [Dev(id=2048 * q + i, process_index=q, slice_index=0) for q in range(4) for i in range(2)]  # 4 x 2
    self.assertEqual([len(g) for g in dsp.slice_groups(four)], [8])
    with fake_slices(2):
      self.assertEqual([[d.process_index for d in g] for g in dsp.slice_groups(four[::-1])], [[0, 0, 1, 1], [2, 2, 3, 3]])
    with fake_slices(4):
      self.assertEqual([{d.process_index for d in g} for g in dsp.slice_groups(four)], [{0}, {1}, {2}, {3}])
    with (
        fake_slices(8),
        self.assertRaisesRegex(ValueError, r"slices of 1 devices would split processes " r"\[0, 1, 2, 3\]"),
    ):
      dsp.slice_groups(four)
    two = [Dev(id=2048 * q + i, process_index=q) for q in range(2) for i in range(4)]  # 2 hosts x 4
    with fake_slices(4), self.assertRaisesRegex(ValueError, r"would split processes \[0, 1\]"):
      dsp.stage_meshes(4, 2, 1, devices=two, layout="slices")
    with fake_slices(2), self.assertRaisesRegex(ValueError, r"on slices \[0, 1\] of their own"):
      dsp.slice_groups(fake_v5p(x=2, y=2, z=1))


class SlicesOnTpu(unittest.TestCase):
  """Fake devices of two v5p-128 slices: they repeat every chip coordinate."""

  def setUp(self):
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("MPMD_FAKE_SLICES", None)
    self.devs = fake_v5p()

  def test_a_stage_per_slice(self):
    """Stage s is all of slice s, its mesh mesh_utils' topology-aware one over
    that slice alone, whatever order the devices come in; the stages share
    no host."""
    meshes = dsp.stage_meshes(2, 4, 16, devices=random.Random(1).sample(self.devs, 128), layout="slices")
    for s, m in enumerate(meshes):
      self.assertEqual(dict(m.shape), {"fsdp": 4, "expert": 16})
      self.assertEqual({d.slice_index for d in m.devices.flat}, {s})
      own = [d for d in self.devs if d.slice_index == s]
      self.assertEqual(
          m.device_ids.tolist(),
          np.vectorize(lambda d: d.id)(
              mesh_utils.create_device_mesh((4, 16), own, allow_split_physical_axes=True)
          ).tolist(),
      )
    # the slices are alike: slice 1's mesh is slice 0's, chip for chip
    np.testing.assert_array_equal(
        np.vectorize(lambda d: d.id - 100000)(meshes[1].devices), np.vectorize(lambda d: d.id)(meshes[0].devices)
    )
    self.assertEqual(dsp.shared_stage_hosts(meshes), [])
    self.assertEqual(dsp.sync_collective_stages(meshes), frozenset())

  def test_slices_must_be_the_stages(self):
    """k stages on each slice: three stages of 32 devices do not cut two slices of 64."""
    with self.assertRaisesRegex(
        ValueError, r"3 stages of fsdp x ep = 32 devices, but the devices are 2 " r"slice\(s\) of \[64, 64\]"
    ):
      dsp.stage_meshes(3, 2, 16, devices=self.devs, layout="slices")

  def test_two_stages_a_slice(self):
    """PP=4 on two slices (partir's pzb2 layout, the GKE runs of the zero-bubble schedules): stages 0 and 1 on slice
    0, 2 and 3 on slice 1, each a strip of its slice -- its chips cut along z, the last chip axis 2 divides: two
    4x4x2 halves of each 4x4x4 slice, whole hosts each -- its mesh mesh_utils' topology-aware (fsdp, ep) mesh over
    the strip's devices in (process, id) order, whatever order the devices come in.  The strip holding its slice's
    chip (0,0,0) keeps its collectives synchronous (sync_collective_stages: a part of the slice), the other offloads
    them."""
    meshes = dsp.stage_meshes(4, 2, 16, devices=random.Random(2).sample(self.devs, 128), layout="slices")
    self.assertEqual([dict(m.shape) for m in meshes], [{"fsdp": 2, "expert": 16}] * 4)
    for s, m in enumerate(meshes):
      self.assertEqual({d.slice_index for d in m.devices.flat}, {s // 2})
      half = sorted(
          (d for d in self.devs if d.slice_index == s // 2 and d.coords[2] // 2 == s % 2),
          key=lambda d: (d.process_index, d.id),
      )
      grid = mesh_utils.create_device_mesh((2, 16), half, allow_split_physical_axes=True)
      self.assertEqual(m.device_ids.tolist(), np.vectorize(lambda d: d.id)(grid).tolist())
      self.assertEqual(len({d.coords[2] // 2 for d in m.devices.flat}), 1, s)  # a z-half: 4 x 4 x 2 chips
    self.assertEqual(dsp.shared_stage_hosts(meshes), [])
    self.assertEqual(dsp.slice_spanning_stages(meshes), [])
    origin = [s for s, m in enumerate(meshes) if any(map(dsp.on_origin_chip, m.devices.flat))]
    self.assertEqual(origin, [0, 2])
    self.assertEqual(dsp.sync_collective_stages(meshes), frozenset(origin))

  def test_coordinate_layouts_refuse_several_slices(self):
    """'snake' and the physical layouts place by chip coordinates; device-id
    order is slice-major, so the id-order layouts cut a stage per slice too
    (not topology-aware)."""
    for layout in ("snake", "physical", "physical_origin_last"):
      with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, r"span slices \[0, 1\]"):
        dsp.stage_meshes(2, 4, 16, devices=self.devs, layout=layout)
    for layout in ("fsdp_on_chip", "ep_on_chip"):
      with self.subTest(layout=layout):
        meshes = dsp.stage_meshes(2, 4, 16, devices=self.devs, layout=layout)
        self.assertEqual([{d.slice_index for d in m.devices.flat} for m in meshes], [{0}, {1}])
    one = [d for d in self.devs if d.slice_index == 0]
    self.assertEqual(len(dsp.stage_meshes(4, 16, 1, devices=one, layout="physical")), 4)  # one slice: as before

  def test_stages_spanning_slices(self):
    """A stage of the id-order layouts straddles two slices where its share of
    the ids does (main warns; ds_ep_aot refuses it, but for --transfers-only);
    --layout slices makes none, nor do devices without slices."""
    self.assertEqual(dsp.slice_spanning_stages(dsp.stage_meshes(2, 4, 16, devices=self.devs, layout="slices")), [])
    self.assertEqual(dsp.slice_spanning_stages(dsp.stage_meshes(2, 4, 16, devices=self.devs)), [])
    # 96 devices in stages of 48: stage 1 is slice 0's last 16 and slice 1's first 32
    self.assertEqual(dsp.slice_spanning_stages(dsp.stage_meshes(2, 3, 16, devices=self.devs[:96])), [1])
    self.assertEqual(dsp.slice_spanning_stages(dsp.stage_meshes(1, 8, 16, devices=self.devs)), [0])
    cpu = jax.devices()
    self.assertEqual(dsp.slice_spanning_stages(dsp.stage_meshes(len(cpu), 1, 1, devices=cpu)), [])

  def test_cross_slice_warning(self):
    """main warns (and ds_ep_aot) when the stages span slices and the boundary transport does not cross them: not
    for the relay -- the MPMD runs over two v5p-128 slices crossed them with it -- nor for ppermute."""
    meshes = dsp.stage_meshes(2, 4, 16, devices=self.devs, layout="slices")
    for transport in ("relay", "ppermute"):
      self.assertIsNone(dsp.cross_slice_warning(meshes, transport), transport)
    for transport in ("native", "xfer"):
      self.assertRegex(
          dsp.cross_slice_warning(meshes, transport),
          rf"span slices \[0, 1\] and --transport {transport} crosses them: .*--transport relay or "
          "ppermute crosses them",
      )
    one = [d for d in self.devs if d.slice_index == 0]
    self.assertIsNone(dsp.cross_slice_warning(dsp.stage_meshes(4, 16, 1, devices=one, layout="physical"), "native"))

  def test_a_fake_slice_must_be_a_box(self):
    """MPMD_FAKE_SLICES on one TPU slice: a fake slice whose chips are not a
    solid box (here hosts 0 and 1 sit on a diagonal of a 4x4 slice) is a
    ValueError that says so, not mesh_utils' AssertionError listing every
    device."""
    ren = {0: 0, 3: 1, 1: 2, 2: 3}
    diag = [
        dataclasses.replace(d, slice_index=None, process_index=ren[d.process_index])
        for d in fake_v5p(slices=1, x=4, y=4, z=1)
    ]
    with fake_slices(2), self.assertRaisesRegex(ValueError, r"slice 0's \(2, 4\) mesh, its chips not a solid box"):
      dsp.stage_meshes(2, 2, 4, devices=diag, layout="slices")
    with fake_slices(4):  # a host each: 2x2 boxes
      self.assertEqual(len(dsp.stage_meshes(4, 2, 2, devices=diag, layout="slices")), 4)


def by_host_order(devs):
  return sorted(devs, key=lambda d: (d.process_index, d.id))


def chip_mesh(devs, fsdp, ep):
  """mesh_utils' topology-aware (fsdp, ep) mesh over `devs` in (process, id) order -> its device ids."""
  grid = mesh_utils.create_device_mesh((fsdp, ep), by_host_order(devs), allow_split_physical_axes=True)
  return np.vectorize(lambda d: d.id)(grid).tolist()


class SlicesOnTpu7x(unittest.TestCase):
  """Fake devices of two tpu7x-4x4x4 slices (`fake_tpu7x`: two TensorCores a chip, each a device, a host per 2x2x1
  chips), and of a tpu7x-8 VM's one 2x2x1 slice.  At k stages a slice --layout slices cuts each slice's chips into k
  strips of whole chips (ds_ep_pipeline.chip_strips), so every stage holds both TensorCores of each of its chips --
  what Pathways requires of a stage ("a subslice must use all logical devices of each chip") -- and whole hosts of
  its own.  The cut before, the rows of mesh_utils' (k, fsdp x ep) mesh over the slice, put the k axis on the cores:
  each stage one core of every chip, which the Pathways servers of the GKE runs refused (2 x tpu7x-4x4x4, PP=4)."""

  def setUp(self):
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("MPMD_FAKE_SLICES", None)
    self.devs = fake_tpu7x()

  def test_k_stages_a_slice_hold_whole_chips(self):
    """Two and four stages a slice (PP=4 and 8 on two slices; the GKE runs' G4 is fsdp 8 x ep 8): stages k s .. k s +
    k - 1 on slice s, each a 4x4x(4/k) strip of it along z, the last chip axis k divides -- both TensorCores of every
    chip it holds, every device of each host it touches, no host shared with another stage -- its mesh mesh_utils'
    topology-aware (fsdp, ep) mesh over the strip's devices, whatever order the devices come in.  The strip holding
    its slice's chip (0,0,0), the first of each slice, keeps its collectives synchronous under McJAX
    (sync_collective_stages), none does under Pathways, whose client sees the same stages."""
    shuffled = random.Random(3).sample(self.devs, len(self.devs))
    host_devices = collections.Counter(d.process_index for d in self.devs)  # 8 a host
    for k, fsdp, ep in ((2, 8, 8), (4, 4, 8)):
      with self.subTest(k=k):
        meshes = dsp.stage_meshes(2 * k, fsdp, ep, devices=shuffled, layout="slices")
        self.assertEqual([dict(m.shape) for m in meshes], [{"fsdp": fsdp, "expert": ep}] * (2 * k))
        for s, m in enumerate(meshes):
          held = list(m.devices.flat)
          self.assertEqual({d.slice_index for d in held}, {s // k})
          chips = collections.Counter(d.coords for d in held)
          self.assertEqual((len(chips), set(chips.values())), (64 // k, {2}), s)  # both cores of each chip
          self.assertEqual({d.coords[2] for d in held}, set(range(s % k * 4 // k, (s % k + 1) * 4 // k)), s)
          hosts = collections.Counter(d.process_index for d in held)
          self.assertEqual(dict(hosts), {q: host_devices[q] for q in hosts}, s)  # whole hosts
          strip = [d for d in self.devs if d.slice_index == s // k and d.coords in chips]
          self.assertEqual(m.device_ids.tolist(), chip_mesh(strip, fsdp, ep), s)
        self.assertEqual(dsp.shared_stage_hosts(meshes), [])
        self.assertEqual(dsp.slice_spanning_stages(meshes), [])
        origin = [s for s, m in enumerate(meshes) if any(map(dsp.on_origin_chip, m.devices.flat))]
        self.assertEqual(origin, [0, k])
        self.assertEqual(dsp.sync_collective_stages(meshes), frozenset(origin))
        pw = dsp.stage_meshes(2 * k, fsdp, ep, devices=pathways_shaped(shuffled), layout="slices")
        self.assertEqual(ids(pw), ids(meshes))
        self.assertEqual(dsp.sync_collective_stages(pw), frozenset())

  def test_one_stage_a_slice_as_before(self):
    """At one stage a slice nothing changes: stage s is all of slice s, mesh_utils' (fsdp, ep) mesh over it."""
    meshes = dsp.stage_meshes(2, 16, 8, devices=random.Random(4).sample(self.devs, 256), layout="slices")
    for s, m in enumerate(meshes):
      self.assertEqual(m.device_ids.tolist(), chip_mesh([d for d in self.devs if d.slice_index == s], 16, 8))
    self.assertEqual(dsp.sync_collective_stages(meshes), frozenset())

  def test_no_strip_of_whole_chips(self):
    """Eight stages a 4x4x4 slice: no chip axis is a multiple of 8, so no strip of whole chips -- refused, naming the
    k that cut it.  So is a fake slice of devices that are not every TensorCore of a box of chips: two hosts on a
    diagonal of a 4x4x1 slice, or three devices, a chip and a half."""
    with self.assertRaisesRegex(
        ValueError,
        r"at 8 stages a slice cuts each slice's chips into 8 strips of whole chips .*: slice 0's chips are 4x4x4, no "
        r"axis a multiple of 8 \(k = 2 or 4 stages a slice cut it\)",
    ):
      dsp.stage_meshes(16, 4, 4, devices=self.devs, layout="slices")
    ren = {0: 0, 3: 1, 1: 2, 2: 3}
    diag = [dataclasses.replace(d, process_index=ren[d.process_index]) for d in fake_tpu7x(slices=1, x=4, y=4, z=1)]
    with (
        fake_slices(2),
        self.assertRaisesRegex(
            ValueError, r"slice 0's 16 devices are not every TensorCore of a solid box of chips \(8 chips in a 4x4x1"
        ),
    ):
      dsp.stage_meshes(4, 2, 4, devices=diag, layout="slices")
    with self.assertRaisesRegex(ValueError, r"slice 1's 3 devices are not every TensorCore of a solid box"):
      dsp.chip_strips(fake_tpu7x(slices=1, x=2, y=1, z=1)[1:], 3, "slice 1")

  def test_one_tpu7x_8(self):
    """A tpu7x-8 VM: one 2x2x1 slice, one host, ids in chip order.  Two stages a slice are its two y-halves of two
    chips; four do not cut it (no chip axis a multiple of 4) -- but MPMD_FAKE_SLICES=2 makes two slices of two chips
    (by id: (0,0,0) and (1,0,0), then (0,1,0) and (1,1,0)), each cut along x: a chip a stage, both its cores
    (pathways_test.OnTpu.test_zero_bubble's layout).  The stage holding chip (0,0,0) is synchronous under McJAX."""
    one = fake_tpu7x(slices=1, x=2, y=2, z=1)
    halves = dsp.stage_meshes(2, 2, 2, devices=one, layout="slices")
    self.assertEqual(
        [sorted({d.coords for d in m.devices.flat}) for m in halves], [[(0, 0, 0), (1, 0, 0)], [(0, 1, 0), (1, 1, 0)]]
    )
    self.assertEqual([m.device_ids.tolist() for m in halves], [chip_mesh(one[:4], 2, 2), chip_mesh(one[4:], 2, 2)])
    self.assertEqual(dsp.sync_collective_stages(halves), frozenset({0}))
    with self.assertRaisesRegex(ValueError, r"slice 0's chips are 2x2x1, no axis a multiple of 4 \(k = 2 stages"):
      dsp.stage_meshes(4, 1, 2, devices=one, layout="slices")
    with fake_slices(2):
      chips = dsp.stage_meshes(4, 1, 2, devices=one, layout="slices")
    order = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    self.assertEqual([[d.coords for d in m.devices.flat] for m in chips], [[c, c] for c in order])
    self.assertEqual(ids(chips), [[[i, i + 1]] for i in (100000, 100002, 100004, 100006)])
    self.assertEqual(dsp.sync_collective_stages(chips), frozenset({0}))

  def test_stages_over_part_of_a_chip(self):
    """split_chip_stages names each chip a stage holds part of the TensorCores of -- what a Pathways run refuses
    (ds_ep_aot.pathways_problems' message, its servers' words) -- as each stage of the cut before was, one core of
    every chip; the strips hold none, nor do one-core chips (v5p) or CPU devices.  Stages sharing hosts are no
    problem there, but --transport relay and xfer, which move arrays between processes, are."""
    slice0 = [d for d in self.devs if d.slice_index == 0]
    by_core = [
        Mesh(np.array([d for d in slice0 if d.core_on_chip == c], dtype=object).reshape(8, 8), ("fsdp", "expert"))
        for c in (0, 1)
    ]
    got = dsp.split_chip_stages(by_core)
    self.assertEqual(len(got), 128)
    self.assertEqual(got[:2], [(0, (0, (0, 0, 0)), 1, 2), (0, (0, (1, 0, 0)), 1, 2)])
    self.assertEqual({(s, held, of) for s, _, held, of in got}, {(0, 1, 2), (1, 1, 2)})
    self.assertEqual(
        ds_ep_aot.pathways_problems(by_core, "native"),
        [
            f"stage {s} holds 1 of the 2 TensorCores of chip 0,0,0 of slice 0 (and part of 63 more chips): a subslice "
            "must use all logical devices of each chip"
            for s in (0, 1)
        ],
    )
    strips = dsp.stage_meshes(4, 8, 8, devices=self.devs, layout="slices")
    self.assertEqual(dsp.split_chip_stages(strips), [])
    self.assertEqual(
        dsp.split_chip_stages(dsp.stage_meshes(4, 8, 8, devices=pathways_shaped(self.devs), layout="slices")), []
    )
    self.assertEqual(dsp.split_chip_stages(dsp.stage_meshes(4, 2, 16, devices=fake_v5p(), layout="slices")), [])
    cpu = [Dev(id=i, process_index=0) for i in range(4)]  # no chip coordinates
    self.assertEqual(dsp.split_chip_stages(dsp.stage_meshes(2, 1, 2, devices=cpu)), [])
    one_host = dsp.stage_meshes(4, 1, 2, devices=fake_tpu7x(slices=1, x=2, y=2, z=1))  # a chip a stage
    self.assertEqual(dsp.shared_stage_hosts(one_host)[0], (0, 1, [0]))
    self.assertEqual(ds_ep_aot.pathways_problems(one_host, "ppermute"), [])
    for transport in ("relay", "xfer"):
      with self.assertRaises(ValueError) as said:  # main's and mem_probe's words, the message's one copy
        pathways.check_transport(transport)
      self.assertEqual(ds_ep_aot.pathways_problems(one_host, transport), [str(said.exception)])
      self.assertRegex(str(said.exception), rf"^--transport {transport}: under Pathways .* native .* ppermute")


class SyncCollectiveStages(unittest.TestCase):
  """libtpu runs a stage's all-gathers / reduce-scatters synchronously when it
  is a part of a slice holding chip (0,0,0) -- not when it is the whole slice,
  as every stage of --layout slices is."""

  def setUp(self):
    self.devs = fake_v5p()
    self.slice = [[d for d in self.devs if d.slice_index == k] for k in range(2)]

  def mesh(self, devs, shape):
    return Mesh(mesh_utils.create_device_mesh(shape, devs, allow_split_physical_axes=True), ("fsdp", "expert"))

  def halves(self, k):
    """Slice k's z < 2 half (it holds chip (0,0,0)) and its z >= 2 half."""
    return [self.mesh([d for d in self.slice[k] if (d.coords[2] >= 2) == upper], (2, 16)) for upper in (0, 1)]

  def test_whole_slices(self):
    whole = [self.mesh(g, (4, 16)) for g in self.slice]
    self.assertTrue(all(any(map(dsp.on_origin_chip, m.devices.flat)) for m in whole))
    self.assertEqual(dsp.sync_collective_stages(whole), frozenset())

  def test_halves_of_slices(self):
    """Half a slice holding chip (0,0,0) is synchronous, the other half and a
    whole slice are not."""
    self.assertEqual(dsp.sync_collective_stages(self.halves(0) + self.halves(1)), frozenset({0, 2}))
    self.assertEqual(dsp.sync_collective_stages([self.mesh(self.slice[0], (4, 16))] + self.halves(1)), frozenset({1}))
    # one slice: a stage holding chip (0,0,0) is a proper part of it, as before
    self.assertEqual(dsp.sync_collective_stages(self.halves(0)), frozenset({0}))

  def test_one_slice_with_a_slice_index(self):
    """Devices of a one-slice job may carry slice_index 0: no stage of several
    is the whole slice."""
    quarters = [self.mesh([d for d in self.slice[0] if d.coords[2] == z], (1, 16)) for z in range(4)]
    self.assertEqual(dsp.sync_collective_stages(quarters), frozenset({0}))

  def test_slices_counted_among_the_clients_devices(self):
    """A slice is counted among all the devices of the meshes' client -- the
    job's at run time (jax.devices()), a compile-only topology's (its
    client's devices()) -- so a stage holding the meshes' share of a slice is
    a part of it when the slice has more.  Device objects of neither count
    among the meshes' own."""
    whole0, half1, whole1 = self.mesh(self.slice[0], (4, 16)), self.halves(1)[0], self.mesh(self.slice[1], (4, 16))
    self.assertEqual(dsp.sync_collective_stages([whole0, half1]), frozenset())  # among the meshes'
    with mock.patch.object(jax, "devices", lambda *a, **k: list(self.devs)):
      self.assertEqual(dsp.sync_collective_stages([whole0, half1]), frozenset({1}))  # among the job's
      self.assertEqual(dsp.sync_collective_stages([whole0, whole1]), frozenset())

    class Topology:  # a compile-only topology's client: its devices() are all of the topology's

      def devices(self):
        return list(own.values())

    topology = Topology()
    own = {d: dataclasses.replace(d, client=topology) for d in self.devs}
    on = lambda m: Mesh(np.vectorize(own.get, otypes=[object])(m.devices), m.axis_names)
    self.assertEqual(dsp.sync_collective_stages([on(whole0), on(half1)]), frozenset({1}))  # among the topology's
    self.assertEqual(dsp.sync_collective_stages([on(whole0), on(whole1)]), frozenset())

  def test_offload_check(self):
    """main's line for each stage whose offload was asked: libtpu's answer, the plan's, against the rule every
    process planned by before (`sync_collective_stages`), on any layout, compiling nothing."""
    whole = [self.mesh(g, (4, 16)) for g in self.slice]
    self.assertEqual(
        dsp.offload_check(whole, 1, True),
        "stage 1: libtpu probe: its collectives offloaded to SparseCore, as the device rule "
        "(sync_collective_stages) has it",
    )
    self.assertEqual(
        dsp.offload_check(whole, 1, False),
        "warning: stage 1: libtpu probe: its collectives synchronous, where the device rule "
        "(sync_collective_stages) has them offloaded to SparseCore: every process plans by the probe",
    )
    self.assertIn("as the device rule", dsp.offload_check(self.halves(0), 0, False))  # a part with (0,0,0)
    self.assertTrue(dsp.offload_check(self.halves(0), 0, True).endswith("every process plans by the probe"))
    self.assertEqual(
        dsp.offload_check(whole, 0, True, "its libtpu probe failed (XlaRuntimeError: x)"),
        "warning: stage 0: its libtpu probe failed (XlaRuntimeError: x): every process plans its "
        "collectives offloaded to SparseCore, by the device rule (sync_collective_stages)",
    )

  def test_offload_answers(self):
    """Each stage's processes probe its asks; across processes the stage's lowest one publishes its answers
    through the key-value store and every process plans from those, its own stage's too, so all of them build one
    plan from libtpu's answers.  A probe that fails answers by the device rule, and says why (review F63, F27)."""
    import threading

    whole = [self.mesh(g, (4, 16)) for g in self.slice]  # processes 0-15 and 16-31
    asks = [(0, ds_ep.FE), (1, ds_ep.FE), (1, ds_ep.EP_AXIS)]

    class KV:  # the coordination service's key-value store, as offload_answers uses it

      def __init__(self):
        self.d, self.cv = {}, threading.Condition()

      def key_value_set(self, k, v):
        with self.cv:
          assert k not in self.d, k
          self.d[k] = v
          self.cv.notify_all()

      def blocking_key_value_get(self, k, timeout_ms):
        with self.cv:
          assert self.cv.wait_for(lambda: k in self.d, timeout_ms / 1000), f"{k}: nothing published"
          return self.d[k]

    calls = []

    def probe(mesh, axes):  # libtpu's answers here: stage 1 synchronous (against the rule), its EP probe fails
      s = next(i for i, m in enumerate(whole) if m is mesh)
      calls.append((threading.current_thread().name, s, axes))
      if (s, axes) == (1, ds_ep.EP_AXIS):
        raise RuntimeError("RET_CHECK failure\nin the probe")
      return s == 0

    kv, got = KV(), {}
    with mock.patch.object(ds_ep, "sparsecore_offload_available", probe):
      threads = [
          threading.Thread(
              target=lambda pid=pid: got.update(
                  {pid: dsp.offload_answers(whole, asks, kv=kv, key="k", pid=pid, timeout_s=30)}
              ),
              name=f"p{pid}",
          )
          for pid in (0, 5, 16, 20)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()
    want = {
        (0, ds_ep.FE): (True, None),
        (1, ds_ep.FE): (False, None),
        (1, ds_ep.EP_AXIS): (True, "its libtpu probe failed (RuntimeError: RET_CHECK failure in the probe)"),
    }
    self.assertEqual(got, {pid: want for pid in (0, 5, 16, 20)})
    # every process of a stage probes its asks, no other process does; the lowest one's answers are published
    self.assertEqual(
        sorted(calls, key=repr),
        sorted(
            [
                ("p0", 0, ds_ep.FE),
                ("p5", 0, ds_ep.FE),
                ("p16", 1, ds_ep.FE),
                ("p16", 1, ds_ep.EP_AXIS),
                ("p20", 1, ds_ep.FE),
                ("p20", 1, ds_ep.EP_AXIS),
            ],
            key=repr,
        ),
    )
    self.assertEqual(
        kv.d,
        {
            "k/0/fsdp,expert": "1",
            "k/1/fsdp,expert": "0",
            "k/1/expert": "1 its libtpu probe failed (RuntimeError: RET_CHECK failure in the probe)",
        },
    )
    # one process, or ahead of time: every stage's asks probed here
    with mock.patch.object(ds_ep, "sparsecore_offload_available", probe):
      self.assertEqual(dsp.offload_answers(whole, asks), want)

  def test_a_stage_over_two_slices(self):
    """A stage over parts of two slices is a part of each: the rule for parts."""
    both = [self.mesh([d for d in self.devs if d.coords[2] < 2 and d.slice_index == k], (2, 16)) for k in range(2)]
    mixed = Mesh(np.concatenate([both[0].devices, both[1].devices]), ("fsdp", "expert"))
    rest = [self.mesh([d for d in self.slice[k] if d.coords[2] >= 2], (2, 16)) for k in range(2)]
    rest = Mesh(np.concatenate([rest[0].devices, rest[1].devices]), ("fsdp", "expert"))
    self.assertEqual(dsp.sync_collective_stages([mixed, rest]), frozenset({0}))


class ProxyClient:
  """The IFRT proxy's client as the pipeline reads one: runtime_type 'proxy/pathways', and devices() every device of
  the job."""

  runtime_type = "proxy/pathways"
  platform_version = "Pathways\ncl/42"

  def __init__(self):
    self.devs = []

  def devices(self):
    return list(self.devs)


def pathways_shaped(devs):
  """`devs` as a Pathways client sees them: one process holds every device (process_index 0), each keeps its slice's
  slice_index and its chip coordinates, and their client is the IFRT proxy's."""
  client = ProxyClient()
  client.devs = [dataclasses.replace(d, process_index=0, client=client) for d in devs]
  return client.devs


class PathwaysShaped(unittest.TestCase):
  """Devices of two v5p-128 slices as a Pathways client sees them (`pathways_shaped`).  The pipeline runs its
  one-process path there (Pipeline.multi is False: jax.process_count() is 1), on the same stage meshes as the
  multi-controller devices of the same slices; the device rule names no synchronous stage (the Pathways server's
  compiler offloads on chip (0,0,0)'s sub-meshes too); no message claims the native copy hangs between the slices --
  there it is one IFRT CopyArrays the Pathways servers carry -- and main notes how the boundaries cross instead; and
  MPMD_FAKE_SLICES cuts a one-slice job, every device slice_index 0 as local docker Pathways reports them."""

  def setUp(self):
    self.enterContext(mock.patch.dict(os.environ))
    os.environ.pop("MPMD_FAKE_SLICES", None)
    self.mc = fake_v5p()
    self.pw = pathways_shaped(self.mc)

  def test_the_multi_controller_meshes(self):
    self.assertEqual({d.process_index for d in self.pw}, {0})
    self.assertEqual(
        [sorted(d.id for d in g) for g in dsp.slice_groups(self.pw)],
        [sorted(d.id for d in g) for g in dsp.slice_groups(self.mc)],
    )
    for layout in ("slices", "ep_on_chip", "fsdp_on_chip"):
      with self.subTest(layout=layout):
        self.assertEqual(
            ids(dsp.stage_meshes(2, 4, 16, devices=self.pw, layout=layout)),
            ids(dsp.stage_meshes(2, 4, 16, devices=self.mc, layout=layout)),
        )
    meshes = dsp.stage_meshes(2, 4, 16, devices=self.pw, layout="slices")
    # one process holds both stages: what multi-controller JAX refuses (Pipeline, under multi) is the one-process path
    self.assertEqual(dsp.shared_stage_hosts(meshes), [(0, 1, [0])])
    self.assertTrue(dsp.on_pathways(meshes))
    self.assertFalse(dsp.on_pathways(dsp.stage_meshes(2, 4, 16, devices=self.mc, layout="slices")))

  def test_no_stage_synchronous_by_rule(self):
    """Halves of the slices holding chip (0,0,0): synchronous by the rule on McJAX's devices, offloaded on Pathways'
    (the probe, where asked, decides either way) -- and main's line for a probe that offloads agrees with the rule."""
    mesh = lambda devs: Mesh(
        mesh_utils.create_device_mesh((2, 16), devs, allow_split_physical_axes=True), ("fsdp", "expert")
    )
    halves = lambda devs: [
        mesh([d for d in devs if d.slice_index == k and (d.coords[2] >= 2) == upper])
        for k in range(2)
        for upper in (0, 1)
    ]
    self.assertEqual(dsp.sync_collective_stages(halves(self.mc)), frozenset({0, 2}))
    self.assertEqual(dsp.sync_collective_stages(halves(self.pw)), frozenset())
    self.assertIn("as the device rule (sync_collective_stages) has it", dsp.offload_check(halves(self.pw), 0, True))
    self.assertFalse(dsp.rule_offloads(halves(self.mc), 0, "fsdp"))
    self.assertTrue(dsp.rule_offloads(halves(self.pw), 0, "fsdp"))

  def test_cross_slice_messages(self):
    mc = dsp.stage_meshes(2, 4, 16, devices=self.mc, layout="slices")
    pw = dsp.stage_meshes(2, 4, 16, devices=self.pw, layout="slices")
    self.assertRegex(dsp.cross_slice_warning(mc, "native"), "native copy hangs")
    for transport in ("native", "ppermute"):
      self.assertIsNone(dsp.cross_slice_warning(pw, transport), transport)
      self.assertIsNone(dsp.cross_slice_note(mc, transport), transport)
    self.assertEqual(
        dsp.cross_slice_note(pw, "native"),
        "the stages span slices [0, 1]: under Pathways each boundary copy (--transport native) is one IFRT "
        "CopyArrays, which the Pathways servers carry between the slices",
    )
    self.assertRegex(dsp.cross_slice_note(pw, "ppermute"), r"\(--transport ppermute\) is a collective-permute program")
    one = [d for d in self.pw if d.slice_index == 0]
    self.assertIsNone(dsp.cross_slice_note(dsp.stage_meshes(4, 16, 1, devices=one, layout="physical"), "native"))
    # id-order layouts straddle the slices alike in one process: main warns of it there too
    self.assertEqual(dsp.slice_spanning_stages(dsp.stage_meshes(2, 3, 16, devices=self.pw[:96])), [1])

  def test_fake_slices_on_one_pathways_slice(self):
    """Local docker Pathways on one slice: every device slice_index 0 and process 0, so MPMD_FAKE_SLICES=2 cuts it in
    two (whole boxes of chips) and --layout slices runs a stage on each half."""
    one = pathways_shaped([dataclasses.replace(d, slice_index=0) for d in fake_v5p(slices=1, x=2, y=2, z=2)])
    with fake_slices(2):
      groups = dsp.slice_groups(one)
      self.assertEqual([[d.coords[2] for d in g] for g in groups], [[0] * 4, [1] * 4])
      meshes = dsp.stage_meshes(2, 2, 2, devices=one, layout="slices")
    self.assertEqual([{d.coords[2] for d in m.devices.flat} for m in meshes], [{0}, {1}])
    self.assertEqual(dsp.sync_collective_stages(meshes), frozenset())


@testing.needs_devices(4)
class AotPieces(unittest.TestCase):
  """ds_ep_aot's multi-slice pieces that need no libtpu."""

  def test_first_slice_meshes(self):
    """A stage on another slice than the first moves onto the first slice's
    devices at the same chip coordinates, in the same mesh order; a stage on
    the first slice, or over two slices, stays."""
    devs = fake_v5p()
    with mock.patch.dict(os.environ):
      os.environ.pop("MPMD_FAKE_SLICES", None)
      meshes = dsp.stage_meshes(2, 4, 16, devices=devs, layout="slices")
    got, moved = ds_ep_aot.first_slice_meshes(meshes, devs)
    self.assertEqual(moved, [1])
    self.assertIs(got[0], meshes[0])
    self.assertEqual(got[1].axis_names, meshes[1].axis_names)
    coords = lambda m: np.vectorize(lambda d: d.coords, otypes=[object])(m.devices).tolist()
    self.assertEqual(coords(got[1]), coords(meshes[1]))
    self.assertEqual({d.slice_index for d in got[1].devices.flat}, {0})
    self.assertEqual(got[1].device_ids.tolist(), meshes[0].device_ids.tolist())  # slice 0's own mesh
    across = dsp.stage_meshes(1, 8, 16, devices=devs, layout="ep_on_chip")  # one stage, both slices
    self.assertEqual(ds_ep_aot.first_slice_meshes(across, devs), (across, []))
    one = devs[:64]  # one slice: nothing moves
    self.assertEqual(ds_ep_aot.first_slice_meshes(dsp.stage_meshes(2, 2, 16, devices=one, layout="physical"), one)[1], [])
    cpu = jax.devices()[:4]
    self.assertEqual(ds_ep_aot.first_slice_meshes(dsp.stage_meshes(2, 2, 1, devices=cpu), cpu)[1], [])

  def test_shift_to_first_slice(self):
    """One program's meshes move together, as many slices down as puts the
    lowest slice they touch on the first, each device to the one at the same
    chip coordinates: a copy between slices 1 and 3 to one between 0 and 2,
    3 -> 2 to 1 -> 0, the two halves of slice 2 onto slice 0's, still
    disjoint.  Meshes touching the first slice, or devices of one slice, come
    back as they are."""
    devs = fake_v5p(slices=4, x=2, y=2, z=2)  # 4 x v5p-16
    with mock.patch.dict(os.environ):
      os.environ.pop("MPMD_FAKE_SLICES", None)
      s = dsp.stage_meshes(4, 2, 4, devices=devs, layout="slices")
    where = lambda m: np.vectorize(lambda d: (d.slice_index, *d.coords), otypes=[object])(m.devices).tolist()
    (a, b), moved = ds_ep_aot.shift_to_first_slice([s[1], s[3]], devs)
    self.assertTrue(moved)
    self.assertEqual((where(a), where(b)), (where(s[0]), where(s[2])))
    self.assertEqual((a.axis_names, b.axis_names), (s[1].axis_names, s[3].axis_names))
    (a, b), _ = ds_ep_aot.shift_to_first_slice([s[3], s[2]], devs)
    self.assertEqual((where(a), where(b)), (where(s[1]), where(s[0])))
    got, moved = ds_ep_aot.shift_to_first_slice([s[0], s[3]], devs)
    self.assertFalse(moved)
    self.assertTrue(got[0] is s[0] and got[1] is s[3])
    halves = dsp.stage_meshes(2, 1, 4, devices=[d for d in devs if d.slice_index == 2])  # z = 0, z = 1
    h, moved = ds_ep_aot.shift_to_first_slice(halves, devs)
    self.assertTrue(moved)
    coords = lambda m: [(d.slice_index, d.coords) for d in m.devices.flat]
    self.assertEqual([coords(m) for m in h], [[(0, c) for _, c in coords(m)] for m in halves])
    self.assertFalse(set(h[0].devices.flat) & set(h[1].devices.flat))
    one = fake_v5p(slices=1, x=2, y=2, z=2)
    m = dsp.stage_meshes(2, 1, 4, devices=one)
    self.assertEqual(ds_ep_aot.shift_to_first_slice(m, one), (m, False))

  def test_first_slice_channel(self):
    """A boundary copy between two slices other than the first is rebuilt a
    slice down, between the same chip coordinates -- the same name, avals,
    specs and program name; a copy touching the first slice stays the same
    object.  Trees of leaves keep their structure."""
    devs = fake_v5p(slices=3, x=2, y=2, z=1)  # 3 x v5p-8
    with mock.patch.dict(os.environ):
      os.environ.pop("MPMD_FAKE_SLICES", None)
      s = dsp.stage_meshes(3, 2, 2, devices=devs, layout="slices")
    act = P(("fsdp", "expert"), None, None)
    aval = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    ch = transfer.PermuteChannel(aval, NamedSharding(s[2], act), NamedSharding(s[1], act), name="s2->s1")
    got, moved = ds_ep_aot.first_slice_channel(ch, devs)
    self.assertTrue(moved)
    self.assertEqual(
        (got.name, got.avals, got.treedef, got.prog.__name__), (ch.name, ch.avals, ch.treedef, ch.prog.__name__)
    )
    self.assertEqual([x.spec for x in got.src + got.dst], [act, act])
    self.assertEqual(
        (got.src[0].mesh.device_ids.tolist(), got.dst[0].mesh.device_ids.tolist()),
        (s[1].device_ids.tolist(), s[0].device_ids.tolist()),
    )  # slices 2, 1 -> 1, 0
    for a, b in ((0, 2), (1, 0)):
      ch = transfer.PermuteChannel(aval, NamedSharding(s[a], act), NamedSharding(s[b], act))
      self.assertEqual(ds_ep_aot.first_slice_channel(ch, devs), (ch, False))
    tree = {"x": aval, "y": jax.ShapeDtypeStruct((), jnp.float32)}
    on = lambda m: {"x": NamedSharding(m, act), "y": NamedSharding(m, P())}
    got, moved = ds_ep_aot.first_slice_channel(transfer.PermuteChannel(tree, on(s[1]), on(s[2]), name="t"), devs)
    self.assertTrue(moved)
    self.assertEqual(
        [(x.spec, x.mesh.device_ids.tolist()) for x in got.src + got.dst],
        [
            (act, s[0].device_ids.tolist()),
            (P(), s[0].device_ids.tolist()),
            (act, s[1].device_ids.tolist()),
            (P(), s[1].device_ids.tolist()),
        ],
    )

  def test_main_refuses_stages_over_part_of_the_slices(self):
    """A stage over part of the topology's slices does not compile there
    (libtpu 0.0.46: an all-gather's RET_CHECK, a reduce-scatter's SIGILL), so
    main stops, saying so, before it builds the pipeline (whose build may
    probe a stage's collectives); with --transfers-only, or a stage over all
    of them, it warns and goes on."""
    self.addCleanup(jax.config.update, "jax_enable_compilation_cache", jax.config.jax_enable_compilation_cache)
    devs = fake_v5p(slices=4, x=2, y=2, z=1)  # 4 x v5p-8
    self.enterContext(mock.patch.object(ds_ep_aot, "topology_devices", lambda name, n: devs))
    self.enterContext(mock.patch.object(jax, "default_backend", lambda: "tpu"))
    self.enterContext(mock.patch.object(dsp, "build", side_effect=AssertionError("built")))
    args = "--topology v5p-8 --num-slices 4 --layout ep_on_chip --stages 2 --fsdp 2 --ep 4".split()
    span = r"stage 0 spans slices \[0, 1\], stage 1 spans slices \[2, 3\] of 4"
    with self.assertRaisesRegex(
        SystemExit,
        span + r".*compile-only topology compiles no collective over part "
        r"of its slices.*--transfers-only compiles the boundary copies alone",
    ):
      ds_ep_aot.main(args)
    with testing.assert_warns_regex(self, RuntimeWarning, span), self.assertRaisesRegex(AssertionError, "built"):
      ds_ep_aot.main(args + ["--transfers-only"])
    with (
        testing.assert_warns_regex(self, RuntimeWarning, r"stage 0 spans slices \[0, 1, 2, 3\] of 4"),
        self.assertRaisesRegex(AssertionError, "built"),
    ):
      ds_ep_aot.main(args[:-6] + "--stages 1 --fsdp 4 --ep 4".split())

  def test_transfers_only_builds_without_probes(self):
    """--transfers-only over stages that span part of the slices builds the pipeline with the collective choices
    that would make its build compile libtpu's probe over each stage -- an all-gather, which fails there -- pinned
    (EP 2's 'auto'); flavours given explicitly stay (review F24)."""
    self.addCleanup(jax.config.update, "jax_enable_compilation_cache", jax.config.jax_enable_compilation_cache)
    devs = fake_v5p(slices=4, x=2, y=2, z=1)  # 4 x v5p-8
    built = []

    def build(a, **kw):
      built.append(a)
      raise AssertionError("built")

    self.enterContext(mock.patch.object(ds_ep_aot, "topology_devices", lambda name, n: devs))
    self.enterContext(mock.patch.object(jax, "default_backend", lambda: "tpu"))
    self.enterContext(mock.patch.object(dsp, "build", side_effect=build))
    args = "--topology v5p-8 --num-slices 4 --layout ep_on_chip --stages 2 --fsdp 4 --ep 2 --transfers-only".split()
    for extra, want in (
        ([], ("xla", "auto")),
        (["--fsdp-collectives", "xla"], ("xla", "xla")),
        (["--ep-collectives", "ppermute"], ("ppermute", "auto")),
    ):
      with (
          self.subTest(extra=extra),
          testing.assert_warns_regex(self, RuntimeWarning, r"stage 0 spans slices \[0, 1\]"),
          self.assertRaisesRegex(AssertionError, "built"),
      ):
        ds_ep_aot.main(args + extra)
      self.assertEqual((built[-1].ep_collectives, built[-1].fsdp_collectives), want)

  def test_lineage_sparsecore_answers(self):
    """--batch-split on Lineage's layers: main stops before it builds anything where Lineage does not import, but
    for --transfers-only, which compiles none of the programs that import it.  Each stage's answer for Lineage's
    SparseCore regions is asked by the pipeline over the first slice's alike devices (q) and handed to the one over
    the real stages (p), which cannot probe there; under --transfers-only q is given the device rule's, unprobed."""
    self.addCleanup(jax.config.update, "jax_enable_compilation_cache", jax.config.jax_enable_compilation_cache)
    devs = fake_v5p(slices=2, x=2, y=2, z=1)  # 2 x v5p-8, a stage a slice
    calls = []

    def build(a, **kw):
      calls.append(kw)
      if len(calls) % 2 == 0:
        raise AssertionError("built p")
      c = types.SimpleNamespace(fsdp_collectives="xla", ep_collectives="xla")
      return types.SimpleNamespace(stage_scfg=[c, c], lineage_sc=(False, True))  # q, with answers of its own

    self.enterContext(mock.patch.object(ds_ep_aot, "topology_devices", lambda name, n: devs))
    self.enterContext(mock.patch.object(jax, "default_backend", lambda: "tpu"))
    self.enterContext(mock.patch.object(dsp, "build", side_effect=build))
    args = "--topology v5p-8 --num-slices 2 --layout slices --stages 2 --fsdp 2 --ep 2 --batch-split".split()
    with mock.patch.object(dsp, "lineage_missing", lambda a: "no Lineage here"):
      with self.assertRaisesRegex(SystemExit, "^ds_ep_aot: --batch-split: no Lineage here"):
        ds_ep_aot.main(args)
      self.assertEqual(calls, [])
      with self.assertRaisesRegex(AssertionError, "built p"):
        ds_ep_aot.main(args + ["--transfers-only"])
    self.assertEqual([kw["stage_lineage_sparsecore"] for kw in calls], [(True, True), (False, True)])  # rule, q's
    calls.clear()
    with mock.patch.object(dsp, "lineage_missing", lambda a: None), self.assertRaisesRegex(AssertionError, "built p"):
      ds_ep_aot.main(args)
    self.assertEqual([kw["stage_lineage_sparsecore"] for kw in calls], [None, (False, True)])  # q probes; p, q's

  def test_topology_devices_takes_num_slices(self):
    from jax.experimental import topologies

    got = []
    with mock.patch.object(topologies, "get_topology_desc", lambda **kw: got.append(kw) or mock.Mock(devices=["d"])):
      self.assertEqual(ds_ep_aot.topology_devices("v5p-128", 2), ["d"])
      ds_ep_aot.topology_devices("v6e-256")
    self.assertEqual([(kw["topology_name"], kw["num_slices"]) for kw in got], [("v5:4x4x4", 2), ("v6e:16x16", 1)])

  def test_main_checks_num_slices_first(self):
    with self.assertRaises(SystemExit) as cm:
      ds_ep_aot.main(["--num-slices", "0"])
    self.assertIn("--num-slices must be at least 1", str(cm.exception.code))

  def test_main_for_a_pathways_run(self):
    """--pathways: what one Pathways client refuses is refused -- --transport relay or xfer before anything builds, a
    stage over part of a chip's TensorCores once the stages are planned -- and multi-controller JAX's stages sharing
    processes is not (four stages of a chip on a TPU7x-8's one host: refused without --pathways), nor are its checks
    made (multihost_problems, cross_slice_warning); the pipelines are built planned for one Pathways client (build's
    proxy), and compile from there."""
    self.addCleanup(jax.config.update, "jax_enable_compilation_cache", jax.config.jax_enable_compilation_cache)
    self.enterContext(mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": _FLAGS}))
    devs = fake_tpu7x(slices=1, x=2, y=2, z=1)  # one host, 4 chips, 8 devices
    built = []

    class Compiled(Exception):
      pass

    def build(a, **kw):
      built.append(kw)
      return types.SimpleNamespace(summary=lambda: "the plan", precompile=mock.Mock(side_effect=Compiled))

    self.enterContext(mock.patch.object(ds_ep_aot, "topology_devices", lambda name, n: devs))
    self.enterContext(mock.patch.object(jax, "default_backend", lambda: "tpu"))
    self.enterContext(mock.patch.object(dsp, "build", side_effect=build))
    multi_checks = [
        self.enterContext(mock.patch.object(owner, name, wraps=getattr(owner, name)))
        for owner, name in ((ds_ep_aot, "multihost_problems"), (dsp, "cross_slice_warning"))
    ]
    args = "--topology tpu7x-4x4x4 --layout ep_on_chip --stages 4 --fsdp 1 --ep 2".split()
    with self.assertRaisesRegex(
        SystemExit, r"`ds_ep_pipeline --distributed` refuses .*\n  stages 0 and 1 share processes"
    ):
      ds_ep_aot.main(args)
    self.assertIsNone(built[-1]["proxy"])
    self.assertEqual([m.call_count for m in multi_checks], [1, 1])
    with self.assertRaises(Compiled):
      ds_ep_aot.main(args + ["--pathways"])
    self.assertIs(built[-1]["proxy"], True)
    self.assertEqual([m.call_count for m in multi_checks], [1, 1])  # neither made for a Pathways run
    n = len(built)
    for transport in ("relay", "xfer"):
      with (
          self.subTest(transport=transport),
          self.assertRaisesRegex(
              SystemExit, rf"^ds_ep_aot --pathways: --transport {transport}: under Pathways .* native .* ppermute"
          ),
      ):
        ds_ep_aot.main(args + ["--pathways", "--transport", transport])
    self.assertEqual(len(built), n)  # refused before anything builds
    with self.assertRaisesRegex(
        SystemExit,
        r"a Pathways run \(one client driving every stage\) refuses this on tpu7x-4x4x4 at startup:\n  stage 0 holds 1 "
        r"of the 2 TensorCores of chip 0,0,0 of slice 0: a subslice must use all logical devices of each chip",
    ):
      ds_ep_aot.main("--topology tpu7x-4x4x4 --layout ep_on_chip --stages 8 --fsdp 1 --ep 1 --pathways".split())


# One process with a CPU backend: libtpu's compile-only 2 x v5p-8 topology (no TPU is opened) -> a JSON line.
_TOPOLOGY_CHECK = r"""if True:
    import json
    import jax, jax.numpy as jnp, numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from maxtext.experimental.mpmd.jax import ds_ep_aot, ds_ep_pipeline as dsp, transfer
    devs = ds_ep_aot.topology_devices('v5p-8', 2)
    out = {'n': len(devs), 'slices': sorted({d.slice_index for d in devs}),
           'procs': [sorted({d.process_index for d in devs if d.slice_index == k}) for k in range(2)]}
    meshes = dsp.stage_meshes(2, 2, 2, devices=devs, layout='slices')
    out['stage_slices'] = [sorted({d.slice_index for d in m.devices.flat}) for m in meshes]
    out['sync'] = sorted(dsp.sync_collective_stages(meshes))
    # a pipeline over part of the topology: slice 0, and the half of slice 1 holding its chip (0,0,0)
    half = np.array([d for d in devs if d.slice_index == 1 and d.coords[1] == 0], dtype=object).reshape(1, 2)
    out['sync partial'] = sorted(dsp.sync_collective_stages([meshes[0], Mesh(half, ('fsdp', 'expert'))]))
    act = P(('fsdp', 'expert'), None, None)
    aval = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    def compiles(f, mesh):
        x = jax.ShapeDtypeStruct((8, 128), jnp.float32, sharding=NamedSharding(mesh, P(('fsdp', 'expert'))))
        try:
            jax.jit(f).lower(x).compile()
            return 'ok'
        except Exception as e:
            return f'{type(e).__name__}: {e}'.splitlines()[0]
    for a, b in ((0, 1), (1, 0)):
        ch = transfer.PermuteChannel(aval, NamedSharding(meshes[a], act), NamedSharding(meshes[b], act))
        hlo = ch.prog.lower(*ch.lower_args()).compile().as_text()
        out[f'copy {a}->{b}'] = sorted(op for op in ('send', 'recv', 'collective-permute') if f' {op}(' in hlo)
    out['slice 0'] = compiles(lambda x: x * 2 + 1, meshes[0])
    out['slice 1'] = compiles(lambda x: x * 2 + 1, meshes[1])
    moved, which = ds_ep_aot.first_slice_meshes(meshes, devs)
    out['moved'] = which
    out['slice 1 moved'] = compiles(lambda x: x * 2 + 1, moved[1])
    print('RESULT ' + json.dumps(out))
"""

# The compile-only 3 x v5p-8 topology, a stage a slice: every boundary copy compiled as it is and where
# ds_ep_aot.first_slice_channel puts it -> a JSON line, {'a->b': [as it is, moved?, where put]}.
_COPIES_CHECK = r"""if True:
    import json
    import jax, jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from maxtext.experimental.mpmd.jax import ds_ep_aot, ds_ep_pipeline as dsp, transfer
    devs = ds_ep_aot.topology_devices('v5p-8', 3)
    meshes = dsp.stage_meshes(3, 2, 2, devices=devs, layout='slices')
    act = P(('fsdp', 'expert'), None, None)
    aval = jax.ShapeDtypeStruct((4, 256, 128), jnp.bfloat16)
    def compiled(ch):
        try:
            hlo = ch.prog.lower(*ch.lower_args()).compile().as_text()
        except Exception as e:
            return f'{type(e).__name__}: {e}'.splitlines()[0]
        return sorted(op for op in ('send', 'recv', 'collective-permute') if f' {op}(' in hlo)
    out = {}
    for a in range(3):
        for b in range(3):
            if a != b:
                ch = transfer.PermuteChannel(aval, NamedSharding(meshes[a], act), NamedSharding(meshes[b], act))
                put, moved = ds_ep_aot.first_slice_channel(ch, devs)
                out[f'{a}->{b}'] = [compiled(ch), moved, compiled(put)]
    print('RESULT ' + json.dumps(out))
"""

# The stage reduce program of one DeepSeek-V3 chunk (one layer, the embedding and the head, 64 experts, the vocab
# of 129280) on a v5p-128 slice as one --layout slices stage, fsdp 4 x expert 16, compiled for libtpu's compile-only
# topology under the flags in LIBTPU_INIT_ARGS, in both shard modes and both StageConfig.rs_axes -> a JSON line:
# {'<mode> <rs_axes>': [[kind, result bytes, replica group size, 'SC' or 'TC', result shape], ...]} of its
# all-reduces and reduce-scatters, SC where one sits in a computation the SparseCore runs: one that an async start on
# async_execution_thread="sparsecore" calls, or that such a computation calls in turn.  The parse reads replica groups
# listed ('{{0,1,..},..}'), in iota form ('[groups,size]<=[..]') and as mesh axes (jax 0.12.0.1 / libtpu 0.0.51:
# "mesh['fsdp'=4,'expert'=16] {'fsdp'}"), and a SparseCore call as a call-start's to_apply=%c (jax 0.11.1 / libtpu
# 0.0.46) or a custom-call-start's called_computations={%c} (libtpu 0.0.51); on a form it does not know it stops,
# rather than count a group or a SparseCore call wrong.
_REDUCE_AXES_CHECK = r"""if True:
    import collections, json, math, re
    import jax, jax.numpy as jnp
    from maxtext.experimental.mpmd.jax import ds_ep, ds_ep_aot, ds_ep_pipeline as dsp
    from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
    devs = ds_ep_aot.topology_devices('v5p-128')
    cfg = DeepSeekConfig(vocab_size=129280, emb_dim=7168, num_decoder_layers=1, first_num_dense_layers=0,
                         num_query_heads=128, num_kv_heads=128, q_lora_rank=1536, kv_lora_rank=512,
                         qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, mlp_dim=2048, moe_mlp_dim=2048,
                         num_experts=64, num_experts_per_tok=8, shared_experts=1, routed_bias=False)
    ch = ds_ep.ChunkSpec(index=0, layers=(0,), first=True, last=True, stage=0)
    sizes = {'bf16': 2, 'f32': 4}

    def callees(line):
        # the computations an instruction calls: 'calls=%c', 'to_apply=%c', 'called_computations={%c, ..}'
        return re.findall(r'(?:calls|to_apply)=%([\w.\-]+)', line) + [
            c for cs in re.findall(r'called_computations=\{([^{}]*)\}', line) for c in re.findall(r'%([\w.\-]+)', cs)]

    def sparsecore(hlo):
        # the computations a SparseCore call runs, and every computation they call
        body, cur = collections.defaultdict(list), None
        for line in hlo.splitlines():
            m = re.match(r'^(?:ENTRY\s+)?%([\w.\-]+)\s+\(', line)
            if m:
                cur = m.group(1)
            else:
                body[cur].append(line)
        calls = [ln for ln in hlo.splitlines() if 'async_execution_thread="sparsecore"' in ln]
        todo = [c for ln in calls for c in callees(ln)]
        if calls and not todo:
            raise ValueError('SparseCore calls, but no computation of theirs found: ' + calls[0][:300])
        seen = set()
        while todo:
            c = todo.pop()
            if c not in seen:
                seen.add(c)
                todo += [x for ln in body[c] for x in callees(ln)]
        return body, seen

    def group_size(rest):
        # the devices in a replica group; of mesh axes the product of their sizes, a sub-axis "'expert':(4)2" its own
        # (2); a mesh whose device order is no iota prints ', device_ids=(..)' before the axes
        m = re.search(r"replica_groups=(?:\[\d+,(\d+)\]|\{\{([\d,]+)\}|mesh\[([^\]]*)\]"
                      r"(?:, device_ids=\((?:[^()]|\([^()]*\))*\))? \{([^{}]*)\})", rest)
        if m is None:
            raise ValueError('replica_groups not understood: ' + rest[:300])
        if m.group(1) or m.group(2):
            return int(m.group(1)) if m.group(1) else len(m.group(2).split(','))
        axis = {a: int(n) for a, n in re.findall(r"'([^']*)'=(\d+)", m.group(3))}
        spans = re.findall(r"'([^']*)'(?::\(\d+\)(\d+))?", m.group(4))
        return math.prod(int(sub) if sub else axis[a] for a, sub in spans)
    out = {}
    for mode in ('auto', 'explicit'):
        for axes in ('each', 'joint'):
            scfg = ds_ep.StageConfig(model=cfg, fsdp=4, ep=16, fsdp_collectives='xla', ep_collectives='xla',
                                     gmm_impl='ragged_dot', shard_mode=mode, rs_axes=axes)
            mesh, = dsp.typed_meshes(dsp.stage_meshes(1, 4, 16, devices=devs, layout='slices'), mode)
            st = ds_ep.StagePrograms(scfg, [ch], mesh)
            sds = lambda sh, sp: jax.ShapeDtypeStruct(sh, jnp.float32, sharding=jax.NamedSharding(mesh, sp))
            gacc = jax.tree.map(sds, ds_ep.acc_shapes(scfg, ch), ds_ep.acc_specs(scfg, ch), is_leaf=ds_ep._is_shape)
            hlo = st.reduce.lower([gacc]).compile().as_text()
            body, sc = sparsecore(hlo)
            ops = []
            for comp, lines in body.items():
                # a combined collective's result is a tuple: '(bf16[..]{..}, bf16[..]{..})'
                for shape, op, rest in re.findall(r'^\s*(?:ROOT\s+)?%\S+\s+=\s+(\S.*?)\s+(all-reduce|reduce-scatter)'
                                                  r'(?:-start)?\((.*)$', '\n'.join(lines), re.M):
                    n = sum(sizes[t] * math.prod(int(x) for x in d.split(',') if x)
                            for t, d in re.findall(r'(bf16|f32)\[([\d,]*)\]', shape))
                    ops.append([op, n, group_size(rest), 'SC' if comp in sc else 'TC',
                                re.sub(r'\{[^{}]*\}', '', shape)])
            out[f'{mode} {axes}'] = sorted(ops)
    print('RESULT ' + json.dumps(out))
"""

# ds_ep_aot.main on the CPU backend (see test_aot_of_a_pipeline_over_two_slices), and TINY's dims for it.
_AOT_MAIN = (
    'import sys, jax; jax.default_backend = lambda: "tpu"; '
    "from maxtext.experimental.mpmd.jax import ds_ep_aot; ds_ep_aot.main(sys.argv[1:])"
)
_TINY = [x for x in multihost_test.TINY if x != "--no-precompile"]
_Q_TINY = [x for x in multihost_test.Q_TINY if x != "--no-precompile"]
_FLAGS = "--xla_max_concurrent_async_collective_permutes=64"  # the MPMD flag the launchers add to any flag set


def _flag_set(name):
  """The launchers' libtpu flag set `name` (libtpu_flags/<name>.flags: one flag a line, '#' comments) as one string."""
  with open(os.path.join(testing.package_dir(), "libtpu_flags", f"{name}.flags")) as f:
    flags = [ln.split("#", 1)[0].strip() for ln in f]
  return " ".join(x for x in flags if x)


# The Qwen3.8-2.4T-A95B target: 92 layers on 8 stages of a tpu7x-4x4x4 slice each (fsdp 8 x ep 16), V=8, cut
# 1,2,2,2,2,2,2,1 then 2,1,1,1,1,1,1,1, four times over (64 chunks: 12 layers on stages 0-6, 8 on the head's stage
# 7), M=16 of 128 x 4096 tokens, the host relay between the slices, lax collectives (pinned: no libtpu probe
# compiles).
# The run's cut: 64 chunks of 2 and 1 layers, two consecutive chunks of a stage three layers at most -- stage 0
# starting with a one-layer chunk (the embedding's), stages 1-6 with a two-layer one, stage 7 one layer a chunk (the
# last with the head) -- bf16 weights and accumulation, one gather or reduce-scatter piece a host op.
# test_qwen38_target_plan pins the cut.
_QWEN38_SPLIT = ",".join(("1" if s == 7 else "12"[(r + (s > 0)) % 2]) for r in range(8) for s in range(8))
_QWEN38_TARGET = (
    "--model qwen3.8-2.4t-a95b --topology tpu7x-4x4x4 --num-slices 8 --layout slices --stages 8 --vv 8 "
    "--fsdp 8 --ep 16 --mb 16 --gbs 2048 --seq 4096 --transport relay --fsdp-collectives xla "
    "--ep-collectives xla --weight-dtype bfloat16 --acc-dtype bfloat16 --rs-dtype bfloat16 "
    "--pf-split 1 --rs-split 1"
).split() + ["--layer-split", _QWEN38_SPLIT]

# ds_ep_aot.main (argv: its flags) on the CPU backend, stopped where it would start compiling: the two pipelines
# it built -- q, each stage's programs on the first slice's devices, and p, over the real stages -- from the plan
# alone (nothing compiles or runs) -> a JSON line.  check_repeat_plan raises here if the plan breaks its rules.
_TARGET_PLAN = r"""if True:
    import collections, json, sys, unittest
    import jax
    from maxtext.experimental.mpmd.jax import ds_ep, ds_ep_aot, ds_ep_pipeline as dsp
    from tests.unit.experimental.mpmd.jax.ds_ep_test import check_repeat_plan
    jax.default_backend = lambda: 'tpu'
    built, build = [], dsp.build
    dsp.build = lambda *args, **kw: built.append(build(*args, **kw)) or built[-1]

    class Compile(Exception):
        pass

    def precompile(self, threads=64):
        raise Compile
    dsp.Pipeline.precompile = precompile
    try:
        ds_ep_aot.main(sys.argv[1:])
    except Compile:
        pass
    q, p = built[0], built[-1]          # one pipeline when no stage moves
    slices = lambda m: sorted({dsp.slice_index(d) for d in m.devices.flat})
    mixers = lambda ch: [sorted(k for k in v if k in ds_ep.QWEN_KINDS)
                         for v in ds_ep.chunk_shapes(p.scfg, ch)['layers'].values()]
    out = {'q_slices': [slices(m) for m in q.meshes], 'slices': [slices(m) for m in p.meshes],
           'procs': [sorted({d.process_index for d in m.devices.flat}) for m in p.meshes],
           'shapes': [dict(m.shape) for m in p.meshes],
           'chunks': [[list(ch.layers), list(ch.kinds), ch.stage, mixers(ch)] for ch in p.chunks],
           'plan': [p.schedule_name, p.fsdp_scope, p.placement, p.S, p.V, p.C, p.M],
           'peaks': check_repeat_plan(unittest.TestCase(), p) if p.fsdp_scope == 'repeat' else None,
           'gather_slots': [list(g) for g in p.gather_slots], 'fallback': p.pf_fallback,
           'gathers': sorted(collections.Counter(c for c, _ in p.pf_gathers).items()),
           'hosts': sorted({'A' if p.order[j][0] == p.C - 1 else p.order[j][1] for j in p.pf}),
           'rs_self': [p.order[j][0] for j in sorted(p.rs_self)], 'final_slots': [list(f) for f in p.final_slots],
           'dispatched': len(p.dispatched()), 'dispatches': p.dispatches(),
           'channels': [len(p.channels), len(p.sq_channels)],
           'collectives': sorted({(c.fsdp_collectives, c.ep_collectives) for c in p.stage_scfg}),
           'multihost': ds_ep_aot.multihost_problems(p.meshes, p.transport),
           'walk': [ds_ep_aot.hbm_walk(p, s)['peak'] for s in range(p.S)]}
    print('RESULT ' + json.dumps(out))
"""


# The GKE runs' G4 shape at TINY widths -- PP=4 on 2 x tpu7x-4x4x4, two stages a slice of fsdp 8 x ep 8, zbh1 --
# as ds_ep_aot.main builds it before compiling, for multi-controller JAX and for one Pathways client (--pathways):
# one JSON line each, from the plan alone.
_G4_TINY = (
    "--topology tpu7x-4x4x4 --num-slices 2 --layout slices --stages 4 --vv 1 --fsdp 8 --ep 8 --layers 4 --mb 8 "
    "--gbs 512 --schedule zbh1 --transport native"
).split()
_G4_PLANS = r"""if True:
    import json, sys
    import jax
    from maxtext.experimental.mpmd.jax import ds_ep_aot, ds_ep_pipeline as dsp
    jax.default_backend = lambda: 'tpu'
    built, build = [], dsp.build
    dsp.build = lambda *args, **kw: built.append(build(*args, **kw)) or built[-1]

    class Compile(Exception):
        pass

    def precompile(self, threads=64):
        raise Compile
    dsp.Pipeline.precompile = precompile
    for extra in ([], ['--pathways']):
        try:
            ds_ep_aot.main(sys.argv[1:] + extra)
        except Compile:
            pass
        p = built[-1]
        print('RESULT ' + json.dumps({
            'slices': [sorted({dsp.slice_index(d) for d in m.devices.flat}) for m in p.meshes],
            'procs': [sorted({d.process_index for d in m.devices.flat}) for m in p.meshes],
            'multihost': ds_ep_aot.multihost_problems(p.meshes, p.transport),
            'pathways': ds_ep_aot.pathways_problems(p.meshes, p.transport),
            'proxy': p.proxy, 'sync': sorted(p.sync_stages), 'final_slots': [list(f) for f in p.final_slots],
            'split': p.split_chunks}), flush=True)
"""


def _run(code, *args, timeout=600, **env):
  """`python -c code args` on the CPU backend.  libtpu, loaded for the
  compile-only topology, would otherwise take the TPU's lock file (a TPU VM's
  own work may hold it: "The TPU is already in use"); a compile-only topology
  opens no TPU, so it may load next to that (ALLOW_MULTIPLE_LIBTPU_LOAD)."""
  e = dict(
      os.environ,
      JAX_PLATFORMS="cpu",
      ALLOW_MULTIPLE_LIBTPU_LOAD="1",
      PYTHONPATH=testing.child_pythonpath(__name__),
      **env,
  )
  e.pop("MPMD_FAKE_SLICES", None)
  return subprocess.run(
      [sys.executable, "-c", code, *args], cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout
  )


class FakeSlicesZeroBubble(unittest.TestCase):
  """The command line on four CPU devices cut into two fake slices (MPMD_FAKE_SLICES=2), --layout slices at two
  stages a slice -- the GKE layout of PP=4 on two slices -- precompiled, one process each (multihost_test.run_single:
  a child of four devices, XLA:CPU's YNNPACK fusions off): zbh1 with --zb-split-last on, its two middle chunks'
  backwards split into I and W and the last chunk's program into AI and W, against 1F1B, every warmup loss and
  gradient norm bit for bit, and the run record's 'zb' block.  scheduled_only: two command lines of half a minute."""

  ARGS = [x for x in multihost_test.TINY if x != "--no-precompile"] + (
      "--stages 4 --vv 1 --fsdp 1 --ep 1 --gbs 8 --mb 4 --layers 4 --layout slices --transport native --warmup 2"
  ).split()

  def run_cli(self, schedule, *extra):
    with tempfile.TemporaryDirectory() as d:
      rc, out = multihost_test.run_single(
          self.ARGS + ["--schedule", schedule, *extra, "--json", f"{d}/r.json"], 4, env={"MPMD_FAKE_SLICES": "2"}
      )
      self.assertEqual(rc, 0, out[-4000:])
      with open(f"{d}/r.json", encoding="utf-8") as f:
        return json.load(f), out

  @pytest.mark.scheduled_only
  def test_zbh1_is_1f1b(self):
    zb, out = self.run_cli("zbh1", "--zb-split-last", "on")
    ref, _ = self.run_cli("1f1b")
    self.assertRegex(out, r"precompiled \d+ programs")
    for s in range(4):  # stage s: slice s // 2, its device s
      self.assertRegex(zb["summary"], rf"stage {s}: \{{'fsdp': 1, 'expert': 1\}} {s}@\S+ +chunks \[{s}\]")
    self.assertEqual((zb["zb"]["split_chunks"], zb["zb"]["split_last"], zb["zb"]["wbuf_cap"]), ([1, 2, 3], True, 1))
    self.assertNotIn("zb", ref)
    self.assertEqual((zb["warmup_losses"], zb["warmup_gnorms"]), (ref["warmup_losses"], ref["warmup_gnorms"]))
    self.assertLess(zb["warmup_losses"][-1], zb["warmup_losses"][0])


@unittest.skipUnless(HAVE_LIBTPU, "needs libtpu (compile-only TPU topologies)")
class CompileOnlyTopology(unittest.TestCase):

  def _result(self, r):
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    self.assertIsNotNone(line, r.stdout[-3000:] + r.stderr[-3000:])
    return json.loads(line[len("RESULT ") :])

  def test_two_slices(self):
    """A two-slice topology: each slice its own processes, a stage on each
    with --layout slices, the copies between them compiled (send / recv with
    jax 0.11.1 / libtpu 0.0.46).  A program over the second slice alone did
    not compile there ("Slice is empty"); moved onto the first slice it
    does.  A newer jax[tpu] than 0.11.1 may compile it, so
    either outcome passes on the second slice, but no other error.  A
    pipeline over part of the topology counts a slice among all the
    topology's devices: half of slice 1, holding its chip (0,0,0), is not the
    whole slice, so synchronous."""
    got = self._result(_run(_TOPOLOGY_CHECK))
    self.assertEqual((got["n"], got["slices"], got["procs"]), (8, [0, 1], [[0], [1]]))
    self.assertEqual((got["stage_slices"], got["sync"], got["sync partial"]), ([[0], [1]], [], [1]))
    for copy in ("copy 0->1", "copy 1->0"):
      self.assertTrue(got[copy], f"{copy}: no send / recv / collective-permute in its HLO")
    self.assertEqual((got["slice 0"], got["slice 1 moved"], got["moved"]), ("ok", "ok", [1]))
    self.assertTrue(got["slice 1"] == "ok" or "Slice is empty" in got["slice 1"], got["slice 1"])

  def test_copies_off_the_first_slice(self):
    """Three slices, a stage each: a copy between slices 1 and 2 did not
    compile ("Slice is empty", libtpu 0.0.46; a newer libtpu than CI's may
    compile it); moved a slice down (`first_slice_channel`) it does, as
    every copy with slice 0 does where it is."""
    got = self._result(_run(_COPIES_CHECK))
    self.assertEqual(sorted(got), ["0->1", "0->2", "1->0", "1->2", "2->0", "2->1"])
    for copy, (raw, moved, put) in got.items():
      with self.subTest(copy=copy):
        self.assertEqual(moved, "0" not in copy)
        self.assertTrue(isinstance(put, list) and put, f"{copy}: {put}")
        if moved:
          self.assertTrue(isinstance(raw, list) or "Slice is empty" in raw, raw)
        else:
          self.assertEqual(raw, put)

  def test_copies_over_four_slices(self):
    """ds_ep_aot --transfers-only over 4 x v5p-8, a stage a slice: all 18
    copies compile (the activations between neighbours and the gradient
    norm's scalars between every two stages, both ways), the 10 among slices
    1-3 a slice or two down, each a copy over DCN in its HLO."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    args = [
        "--topology",
        "v5p-8",
        "--num-slices",
        "4",
        "--stages",
        "4",
        "--vv",
        "1",
        "--fsdp",
        "2",
        "--ep",
        "2",
        "--layers",
        "4",
        "--mb",
        "4",
        "--gbs",
        "16",
        "--layout",
        "slices",
        "--transport",
        "ppermute",
        "--transfers-only",
        "--hlo-dir",
        d,
        *_TINY,
    ]
    r = _run(_AOT_MAIN, *args, LIBTPU_INIT_ARGS=_FLAGS)
    self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
    self.assertIn("compiled 18 boundary-copy programs", r.stdout)
    off = ["s1->s2", "s2->s1", "s2->s3", "s3->s2"] + [f"sumsq s{a}->s{b}" for a in (1, 2, 3) for b in (1, 2, 3) if a != b]
    self.assertIn(f'copies {", ".join(off)}: compiled moved down until their lower slice is the first', r.stdout)
    hlo = sorted(os.listdir(d))
    self.assertEqual(len(hlo), 18, hlo)
    for f in hlo:
      with open(os.path.join(d, f)) as fh:
        text = fh.read()
      self.assertTrue(any(f" {op}(" in text for op in ("send", "recv", "collective-permute")), f)

  def test_qwen38_target_plan(self):
    """The Qwen3.8-2.4T-A95B run on 8 x tpu7x-4x4x4 (_QWEN38_TARGET) as
    ds_ep_aot.main builds it before compiling (`_TARGET_PLAN`; seconds, from
    the plan alone).  Stage s is slice s, all 128 devices of its 16 hosts
    (fsdp 8 x ep 16), its programs built on slice 0's alike devices; the 92
    layers split in order, by the run's cut (_QWEN38_SPLIT), each chunk's kinds the
    global pattern (layer l gated attention iff l % 4 == 3) and its weights a
    subtree a layer, with that layer's mixer;
    breadth-first with weights per visit, the plan check_repeat_plan holds,
    two chunks' weights at most on a stage, each chunk gathered once a visit
    but the stage's first one's forward visit (the update gathers it) and
    every chunk reduced inside a backward (none in a standalone program);
    the step's dispatches a program per op -- the head's no-op backward
    none -- and the update, a copy per stage crossing, the gradient norm's
    hops; the hosts' checks pass, and the relay compiles no copy.  The
    model is the whole 2.42T one, and the plan's HBM walk (no temporaries:
    the AOT compile prices them) leaves the headroom free on every stage."""
    r = _run(_TARGET_PLAN, *_QWEN38_TARGET)
    got = self._result(r)
    S, V, C, M, L = 8, 8, 64, 16, 92
    self.assertIn("AOT for 8 x tpu7x-4x4x4 (1024 x TPU7x, 128 hosts)", r.stdout)
    self.assertIn(f"MPMD Qwen3.5 family: {L} MoE layers (69 Gated DeltaNet, 23 gated attention), {C} chunks", r.stdout)
    self.assertIn("params 2419.80B", r.stdout)  # MaxText's qwen3.8-2.4t-a95b dims, the head untied
    self.assertIn("boundary copies: --transport relay moves them between the slices on the hosts", r.stdout)
    self.assertEqual(got["q_slices"], [[0]] * S)
    self.assertEqual(got["slices"], [[s] for s in range(S)])
    self.assertEqual(got["procs"], [list(range(16 * s, 16 * s + 16)) for s in range(S)])
    self.assertEqual(got["shapes"], [{"fsdp": 8, "expert": 16}] * S)
    for s in range(S):
      self.assertIn(f"stage {s}: slice {s}, hosts {16 * s}..{16 * s + 15} (16)", r.stdout)
    split = [int(x) for x in _QWEN38_SPLIT.split(",")]
    self.assertEqual(split, [1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1] * 4)  # the cut the comment names
    first = np.cumsum([0] + split)
    for c, (layers, kinds, stage, mixers) in enumerate(got["chunks"]):
      self.assertEqual((layers, stage), (list(range(first[c], first[c + 1])), c % S), c)
      self.assertEqual(kinds, ["attn" if lid % 4 == 3 else "gdn" for lid in layers], c)
      self.assertEqual(mixers, [[k] for k in kinds], c)
    self.assertEqual([sum(len(ch[0]) for ch in got["chunks"] if ch[2] == s) for s in range(S)], [12] * 7 + [8])
    self.assertEqual(got["plan"], ["breadth", "repeat", "circular", S, V, C, M])
    self.assertEqual(got["peaks"], [2] * S)
    self.assertEqual((got["gather_slots"], got["fallback"]), ([[0]] * S, []))
    gathers = dict(got["gathers"])
    for c in range(C):  # a stage's chunks in order: the first (the update gathers it for its forward visit) ...
      want = 1 if c < S or c >= C - S else 2  # ... and the turnaround chunk once, every other one twice
      self.assertEqual(gathers.get(c), want, c)
    self.assertEqual(got["hosts"], ["A", "B", "F"])
    self.assertEqual((sorted(got["rs_self"]), got["final_slots"]), (list(range(S)), [[]] * S))
    self.assertEqual(got["dispatched"], 2 * C * M - M)
    self.assertEqual(got["dispatches"], [2 * C * M - M + S, S * (S - 1) + 2 * (C - 1) * M])
    self.assertIn(f"dispatches/step {2 * C * M - M + S} programs + {S * (S - 1) + 2 * (C - 1) * M} copies", r.stdout)
    self.assertEqual(got["channels"], [0, 0])
    self.assertEqual(got["collectives"], [["xla", "xla"]])
    self.assertEqual(got["multihost"], [])
    hbm = ds_ep_aot.TOPOLOGIES["tpu7x-4x4x4"][4]
    self.assertTrue(all(0 < w < (1 - ds_ep_aot.HEADROOM) * hbm for w in got["walk"]), [w / 2**30 for w in got["walk"]])

  def test_g4_plan_for_a_pathways_run(self):
    """The GKE runs' G4 shape (_G4_TINY) as ds_ep_aot.main builds it before compiling (seconds, from the plan alone),
    and with --pathways.  Either way each stage is a z-half of its slice, eight whole hosts of its own: no
    'stages share processes' refusal for multi-controller JAX (each stage one TensorCore of every chip before
    chip_strips, every host shared), nothing a Pathways run refuses.  For multi-controller JAX the device rule keeps
    the strips holding chip (0,0,0) synchronous, and stage 2 reduces in the drain; planned for one Pathways client
    no stage is (its servers offload), and stage 2 folds its reduce-scatter into its last W."""
    r = _run(_G4_PLANS, *(_G4_TINY + _TINY), LIBTPU_INIT_ARGS=_FLAGS)
    got = [json.loads(ln[len("RESULT ") :]) for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    self.assertEqual(len(got), 2, r.stdout[-3000:] + r.stderr[-3000:])
    mc, pw = got
    for g in got:
      self.assertEqual(g["slices"], [[0], [0], [1], [1]])
      self.assertEqual(g["procs"], [list(range(8 * s, 8 * s + 8)) for s in range(4)])
      self.assertEqual((g["multihost"], g["pathways"], g["split"]), ([], [], [1, 2]))
    self.assertEqual((mc["proxy"], mc["sync"], pw["proxy"], pw["sync"]), (False, [0, 2], True, []))
    self.assertEqual((mc["final_slots"][2], pw["final_slots"][2]), ([0], []))
    self.assertIn("synchronous on stage [0, 2]", r.stdout)
    self.assertIn("planned for one Pathways client (--pathways): stages may share hosts", r.stdout)
    self.assertIn("under Pathways each boundary copy (--transport native) is one IFRT CopyArrays", r.stdout)

  @pytest.mark.scheduled_only
  def test_reduce_scatters_one_axis_at_a_time(self):
    """rs_axes 'each': on the stage of the DeepSeek-V3 runs across two
    v5p-128 slices -- one slice, fsdp 4 x expert 16, under those runs' libtpu
    flags (the v5p flag set, the MPMD flag, then v5p_fsdp_extra, as they
    composed them) -- the stage reduce of a DeepSeek-V3 chunk with its vocab (129280)
    reduce-scatters over 'fsdp' (groups of 4) and 'expert' (16), and
    all-reduces over the stage only the embedding and the head (in one
    psum_scatter each: their per-axis pair libtpu 0.0.46 ran on the
    TensorCore, synchronously) and its sums of squares and norm scales (under
    a MiB); every collective of a MiB or more runs on the SparseCore.  In both
    shard modes, and with jax 0.12.0.1 / libtpu 0.0.51 too.  'joint', one
    psum_scatter over both axes, compiled every leaf stored over both to a
    full-size all-reduce over all 64 devices instead; printed, not held:
    another libtpu may compile it otherwise."""
    flags = f"{_flag_set('v5p')} {_FLAGS} {_flag_set('v5p_fsdp_extra')}"
    got = self._result(_run(_REDUCE_AXES_CHECK, LIBTPU_INIT_ARGS=flags, timeout=1200))
    wide = lambda ops: sorted(shape for op, n, g, _, shape in ops if op == "all-reduce" and g == 64 and n >= 2**20)
    on_tc = lambda ops: [o for o in ops if o[1] >= 2**20 and o[3] != "SC"]
    for mode in ("auto", "explicit"):
      each, joint = got[f"{mode} each"], got[f"{mode} joint"]
      self.assertEqual(wide(each), ["bf16[129280,7168]", "bf16[7168,129280]"], (mode, each))
      self.assertEqual({g for op, n, g, *_ in each if op == "reduce-scatter"}, {4, 16}, (mode, each))
      self.assertEqual(on_tc(each), [], (mode, each))
      print(
          f"{mode}: stage-wide all-reduces of a MiB or more {len(wide(each))} (each), {len(wide(joint))} (joint); "
          f'reduce-scatters {sum(op == "reduce-scatter" for op, *_ in each)} / '
          f'{sum(op == "reduce-scatter" for op, *_ in joint)}; a MiB or more off the SparseCore: '
          f"{len(on_tc(each))} / {len(on_tc(joint))}"
      )
    # the same collectives in both modes (explicit writes a 1-layer leaf's [1, 112, 576] block as [112, 576])
    self.assertEqual(*([o[:4] for o in got[f"{m} each"]] for m in ("auto", "explicit")))

  @pytest.mark.scheduled_only
  def test_aot_of_a_pipeline_over_two_slices(self):
    """ds_ep_aot --num-slices 2 --layout slices --transport ppermute on 2 x
    v5p-8, TINY: stage 1's programs compiled on slice 0's alike devices, the
    copies between the real stages, the report for both stages.  main wants a
    TPU backend for the Pallas kernels, which TINY runs none of (ragged_dot
    experts, reference attention, argsort dispatch): the CPU backend passes
    for one here."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    args = [
        "--topology",
        "v5p-8",
        "--num-slices",
        "2",
        "--stages",
        "2",
        "--vv",
        "2",
        "--fsdp",
        "2",
        "--ep",
        "2",
        "--layers",
        "4",
        "--mb",
        "4",
        "--gbs",
        "16",
        "--schedule",
        "gpipe",
        "--layout",
        "slices",
        "--transport",
        "ppermute",
        "--json",
        f"{d}/aot.json",
        "--hlo-dir",
        f"{d}/hlo",
        "--threads",
        "8",
        *_TINY,
    ]
    r = _run(_AOT_MAIN, *args, LIBTPU_INIT_ARGS=_FLAGS)
    self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
    self.assertIn("AOT for 2 x v5p-8 (8 x TPU v5, 2 hosts)", r.stdout)
    self.assertIn("stage 0: slice 0, hosts 0..0 (1)", r.stdout)
    self.assertIn("stage 1: slice 1, hosts 1..1 (1)", r.stdout)
    self.assertIn("stages [1]: their programs compile on slice 0's devices", r.stdout)
    with open(f"{d}/aot.json") as f:
      rec = json.load(f)
    self.assertEqual(rec["compiled_on_first_slice"], [1])
    self.assertEqual([s["stage"] for s in rec["stages"]], [0, 1])
    self.assertTrue(all(s["fits"] for s in rec["stages"]))
    # Both stages' programs, and the four copies between them (the activations and the gradient norm's
    # scalar, both ways), each over both stages and a send / recv over DCN.
    hlo = sorted(os.listdir(f"{d}/hlo"))
    self.assertTrue(any(f.startswith("s0_") for f in hlo) and any(f.startswith("s1_") for f in hlo), hlo)
    copies = [f for f in hlo if "_xfer_" in f]
    self.assertEqual(
        copies,
        [
            "s0-1_xfer_s0_to_s1.hlo.txt",
            "s0-1_xfer_sumsq_s0_to_s1.hlo.txt",
            "s1-0_xfer_s1_to_s0.hlo.txt",
            "s1-0_xfer_sumsq_s1_to_s0.hlo.txt",
        ],
    )
    for f in copies:
      with open(f"{d}/hlo/{f}") as fh:
        text = fh.read()
      self.assertTrue(" send(" in text and " recv(" in text, f)

  @pytest.mark.scheduled_only
  def test_aot_of_two_stages_on_the_second_slice(self):
    """ds_ep_aot --num-slices 2 --stages 4 --layout ep_on_chip on 2 x v5p-16,
    TINY: two stages a slice, each on a host of its own.  Stages 2 and 3
    compile on slice 0's alike devices, and so do the copies between them,
    inside slice 1 -- still a collective-permute over ICI there -- while the
    copy across, s1 -> s2, compiles where it is, a send / recv over DCN."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    args = [
        "--topology",
        "v5p-16",
        "--num-slices",
        "2",
        "--stages",
        "4",
        "--vv",
        "1",
        "--fsdp",
        "2",
        "--ep",
        "2",
        "--layers",
        "4",
        "--mb",
        "4",
        "--gbs",
        "16",
        "--schedule",
        "gpipe",
        "--layout",
        "ep_on_chip",
        "--transport",
        "ppermute",
        "--json",
        f"{d}/aot.json",
        "--hlo-dir",
        f"{d}/hlo",
        "--threads",
        "8",
        *_TINY,
    ]
    r = _run(_AOT_MAIN, *args, LIBTPU_INIT_ARGS=_FLAGS)
    self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
    for s in range(4):
      self.assertIn(f"stage {s}: slice {s // 2}, hosts {s}..{s} (1)", r.stdout)
    self.assertIn("stages [2, 3]: their programs compile on slice 0's devices", r.stdout)
    with open(f"{d}/aot.json") as f:
      rec = json.load(f)
    self.assertEqual(
        (rec["compiled_on_first_slice"], rec["copies_moved"]),
        ([2, 3], ["s2->s3", "s3->s2", "sumsq s2->s3", "sumsq s3->s2"]),
    )
    self.assertEqual([s["stage"] for s in rec["stages"]], [0, 1, 2, 3])
    self.assertTrue(all(s["fits"] for s in rec["stages"]))
    hlo = sorted(os.listdir(f"{d}/hlo"))
    self.assertTrue(all(any(f.startswith(f"s{s}_") for f in hlo) for s in range(4)), hlo)
    self.assertEqual(len([f for f in hlo if "_xfer_" in f]), 6 + 12, hlo)  # activations both ways, scalars
    with open(f"{d}/hlo/s1-2_xfer_s1_to_s2.hlo.txt") as f1, open(f"{d}/hlo/s2-3_xfer_s2_to_s3.hlo.txt") as f2:
      across, inside = f1.read(), f2.read()
    self.assertTrue(" send(" in across and " recv(" in across)
    self.assertTrue(" collective-permute-start(" in inside and " send(" not in inside)

  @testing.slow
  def test_qwen_aot_over_two_tpu7x_slices(self):
    """ds_ep_aot of a Qwen3.5-family pipeline (Q_TINY) on 2 x tpu7x-4x4x4,
    in the Qwen3.8 target's stage shape and flags: a whole slice a stage
    (fsdp 8 x ep 16 over its 16 hosts), breadth with weights per visit, lax
    collectives, the host relay.  The chunks cut 3,1,1,3 -- Gated DeltaNet
    layers only, gated attention only, a Gated DeltaNet layer, both kinds and
    the head -- each compile their forwards and backwards on their stage,
    stage 1's on slice 0's alike devices, and every stage fits, its walk
    priced from its programs' compiles.  The relay compiles no copy.  The
    CPU backend passes for a TPU one: Q_TINY runs no Pallas kernel (see
    test_aot_of_a_pipeline_over_two_slices).  A minute of TPU7x compiles."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    args = [
        "--topology",
        "tpu7x-4x4x4",
        "--num-slices",
        "2",
        "--stages",
        "2",
        "--vv",
        "2",
        "--fsdp",
        "8",
        "--ep",
        "16",
        "--layers",
        "8",
        "--layer-split",
        "3,1,1,3",
        "--mb",
        "4",
        "--gbs",
        "512",
        "--layout",
        "slices",
        "--transport",
        "relay",
        "--fsdp-collectives",
        "xla",
        "--ep-collectives",
        "xla",
        "--json",
        f"{d}/aot.json",
        "--hlo-dir",
        f"{d}/hlo",
        "--threads",
        "8",
        *_Q_TINY,
        "--experts",
        "16",
    ]
    r = _run(_AOT_MAIN, *args, LIBTPU_INIT_ARGS=_FLAGS)
    self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
    self.assertIn("AOT for 2 x tpu7x-4x4x4 (256 x TPU7x, 32 hosts)", r.stdout)
    self.assertIn(
        "MPMD Qwen3.5 family: 8 MoE layers (6 Gated DeltaNet, 2 gated attention), 4 chunks (V=2, " "layers [3, 1, 1, 3])",
        r.stdout,
    )
    self.assertIn("stage 0: slice 0, hosts 0..15 (16)", r.stdout)
    self.assertIn("stage 1: slice 1, hosts 16..31 (16)", r.stdout)
    self.assertIn("stages [1]: their programs compile on slice 0's devices", r.stdout)
    self.assertIn("boundary copies: --transport relay moves them between the slices on the hosts", r.stdout)
    with open(f"{d}/aot.json") as f:
      rec = json.load(f)
    self.assertEqual((rec["compiled_on_first_slice"], rec["copies_moved"]), ([1], []))
    self.assertEqual([s["stage"] for s in rec["stages"]], [0, 1])
    self.assertTrue(all(s["fits"] for s in rec["stages"]), rec["stages"])
    self.assertEqual([s["uncompiled"] for s in rec["stages"]], [[], []])
    hlo = sorted(os.listdir(f"{d}/hlo"))
    self.assertEqual([f for f in hlo if "_xfer_" in f], [])
    for c, s in enumerate((0, 1, 0, 1)):  # chunk c on stage c % 2; chunk 3 the fused last one (fb / fba)
      for kinds in (("fb_", "fba_"),) if c == 3 else (("fwd_",), ("bwd_", "bwda_")):
        self.assertTrue(any(f.startswith(tuple(f"s{s}_{k}c{c}" for k in kinds)) for f in hlo), (c, kinds, hlo))
    for s in (0, 1):
      self.assertTrue(any(f.startswith(f"s{s}_update_s{s}") for f in hlo), (s, hlo))


if __name__ == "__main__":
  unittest.main()
