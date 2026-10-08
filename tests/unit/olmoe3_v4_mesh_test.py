# Copyright 2023-2026 Google LLC
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

"""Tests for `maxtext_utils.tpu_8x8xz_expert_mesh`: single-slice `(8, 8, Z)` TPU v4 expert-group layouts.

Uses integer stand-ins for devices: entry `x * 8 * Z + y * Z + z` is the chip at physical coordinate `(x, y, z)`.
"""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from maxtext.utils import maxtext_utils


def _physical(z_dim):
  return np.arange(8 * 8 * z_dim).reshape(8, 8, z_dim)


def _coords(ids, z_dim):
  ids = np.asarray(ids)
  return np.stack([ids // (8 * z_dim), (ids // z_dim) % 8, ids % z_dim], axis=-1)


class Tpu8x8xzExpertMeshTest(parameterized.TestCase):
  """Each expert group must be a contiguous block of the torus; data parallelism covers the rest."""

  def _groups(self, z_dim, ep):
    """Returns the expert groups as [DP, EP] device ids for mesh axes (data, expert)."""
    mesh = maxtext_utils.tpu_8x8xz_expert_mesh(_physical(z_dim), ep, (8 * 8 * z_dim // ep, ep))
    self.assertIsNotNone(mesh)
    self.assertEqual(mesh.shape, (8 * 8 * z_dim // ep, ep))
    self.assertCountEqual(mesh.reshape(-1).tolist(), range(8 * 8 * z_dim))
    return mesh

  def _assert_blocks(self, mesh, z_dim, block):
    for group in mesh:
      c = _coords(group, z_dim)
      extent = tuple(int(c[:, i].max() - c[:, i].min() + 1) for i in range(3))
      self.assertEqual(extent, block)
      # Each group starts on a block boundary, so it tiles the torus exactly.
      self.assertEqual(tuple(int(c[:, i].min()) % block[i] for i in range(3)), (0, 0, 0))

  @parameterized.named_parameters(
      ("8x8x48_ep64_planes", 48, 64, (8, 8, 1)),
      ("8x8x48_ep128", 48, 128, (8, 8, 2)),
      ("8x8x48_ep256", 48, 256, (8, 8, 4)),
      ("8x8x16_ep128", 16, 128, (8, 8, 2)),
      ("8x8x16_ep256", 16, 256, (8, 8, 4)),
      ("8x8x8_ep128", 8, 128, (8, 8, 2)),
  )
  def test_expert_groups_are_8x8_slabs(self, z_dim, ep, block):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("OLMOE3_EP64_SUBMESH", None)
      self._assert_blocks(self._groups(z_dim, ep), z_dim, block)

  @parameterized.named_parameters(("8x8x8", 8), ("8x8x16", 16))
  def test_ep64_keeps_cubes_on_1k_and_2k_slices(self, z_dim):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("OLMOE3_EP64_SUBMESH", None)
      self._assert_blocks(self._groups(z_dim, 64), z_dim, (4, 4, 4))

  def test_ep64_submesh_override(self):
    with mock.patch.dict(os.environ, {"OLMOE3_EP64_SUBMESH": "4x4x4"}):
      self._assert_blocks(self._groups(48, 64), 48, (4, 4, 4))
    with mock.patch.dict(os.environ, {"OLMOE3_EP64_SUBMESH": "8x8x1"}):
      self._assert_blocks(self._groups(16, 64), 16, (8, 8, 1))

  def test_unsupported_shapes_return_none(self):
    self.assertIsNone(maxtext_utils.tpu_8x8xz_expert_mesh(np.arange(3072).reshape(12, 16, 16), 128, (24, 128)))
    self.assertIsNone(maxtext_utils.tpu_8x8xz_expert_mesh(_physical(6), 256, (6, 64)))
    self.assertIsNone(maxtext_utils.tpu_8x8xz_expert_mesh(_physical(48), 32, (96, 32)))


if __name__ == "__main__":
  absltest.main()
