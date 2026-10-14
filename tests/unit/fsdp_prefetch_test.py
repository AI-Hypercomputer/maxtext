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

"""Unit tests for the helpers in maxtext.layers.fsdp_prefetch.

Pipeline-level gradient parity against the regular scanned layer stack is tested in
nnx_decoders_test.py (TestNNXDecoderPrefetchGradParity).
"""

import types
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.common.common_types import ShardMode
from maxtext.layers import fsdp_prefetch


def _abstract_mesh(shape, names):
  return AbstractMesh(shape, names, axis_types=(AxisType.Explicit,) * len(names))


class FsdpPrefetchHelpersTest(unittest.TestCase):
  """Tests for the layer-slicing, stacking, and sharding helpers."""

  def test_slice_and_expand0(self):
    tree = {"w": jnp.arange(12.0).reshape(3, 4)}
    layer = fsdp_prefetch._slice_at(tree, 1)  # pylint: disable=protected-access
    np.testing.assert_array_equal(layer["w"], np.arange(4.0, 8.0))
    sub = fsdp_prefetch._slice_range(tree, 1, 3)  # pylint: disable=protected-access
    np.testing.assert_array_equal(sub["w"], np.arange(4.0, 12.0).reshape(2, 4))
    self.assertEqual(fsdp_prefetch._expand0(layer)["w"].shape, (1, 4))  # pylint: disable=protected-access

  def test_explicit_all_gather_without_fsdp_sharding_returns_none(self):
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tensor"))
    w = jnp.ones((4, 8))
    # A size-1 FSDP axis needs no gather; the caller then falls back to resharding.
    out = fsdp_prefetch._explicit_fsdp_all_gather(  # pylint: disable=protected-access
        w, mesh, PartitionSpec("fsdp", "tensor"), PartitionSpec(None, "tensor")
    )
    self.assertIsNone(out)

  def test_mesh_uses_fsdp(self):
    self.assertTrue(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((4, 2), ("fsdp", "tensor"))))
    self.assertTrue(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((1, 2), ("fsdp", "fsdp_transpose"))))
    self.assertFalse(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((1, 8), ("fsdp", "tensor"))))
    self.assertFalse(fsdp_prefetch.mesh_uses_fsdp(_abstract_mesh((8,), ("data",))))

  def test_per_layer_sharding_drops_layer_axis(self):
    # Explicit mesh axes, so that the sharding spec is part of the array's type (jax.typeof).
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tensor"), axis_types=(AxisType.Explicit,) * 2)
    stacked = jax.device_put(jnp.ones((2, 4, 8)), NamedSharding(mesh, PartitionSpec(None, "fsdp", "tensor")))
    s = fsdp_prefetch._per_layer_sharding(stacked, mesh)  # pylint: disable=protected-access
    self.assertEqual(tuple(s.spec), ("fsdp", "tensor"))


class FsdpPrefetchAllGatherTest(unittest.TestCase):
  """Lowers the FSDP weight all-gather against abstract multi-device meshes (no devices needed)."""

  def _lower_gather(self, mesh, spec, shape):
    """Returns (output ShapeDtypeStruct, all_gather StableHLO lines) of `_apply_sharding_hint` in explicit mode."""
    w = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=NamedSharding(mesh, spec))

    def gather(x):
      return fsdp_prefetch._apply_sharding_hint(x, mesh, ShardMode.EXPLICIT)  # pylint: disable=protected-access

    with jax.sharding.use_abstract_mesh(mesh):
      out = jax.eval_shape(gather, w)
      # An abstract mesh has no devices, so the lowering platform must be given explicitly.
      hlo = jax.jit(gather).trace(w).lower(lowering_platforms=("tpu",)).as_text()
    return out, [line for line in hlo.splitlines() if "stablehlo.all_gather" in line]

  def test_explicit_mode_gathers_fsdp_axis_with_tagged_all_gather(self):
    mesh = _abstract_mesh((4, 2), ("fsdp", "tensor"))
    out, gathers = self._lower_gather(mesh, PartitionSpec("fsdp", "tensor"), (8, 6))
    self.assertEqual(out.shape, (8, 6))
    self.assertNotIn("fsdp", str(out.sharding.spec))
    self.assertIn("tensor", str(out.sharding.spec))  # Non-FSDP axes stay sharded.
    self.assertEqual(len(gathers), 1)
    # One all-gather over the 4-way 'fsdp' axis of dim 0; the 2-way 'tensor' shard of dim 1 is kept.
    self.assertIn("all_gather_dim = 0", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertNotIn('name = "tensor"', gathers[0])
    self.assertIn("(tensor<2x3xf32>) -> tensor<8x3xf32>", gathers[0])
    # The tag that lets XLA's FSDP scheduler run the gather on a dedicated SparseCore.
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

  def test_explicit_mode_gathers_multiple_fsdp_axes_of_one_dim(self):
    mesh = _abstract_mesh((2, 2, 2), ("tensor", "fsdp", "fsdp_transpose"))
    out, gathers = self._lower_gather(mesh, PartitionSpec(None, ("tensor", "fsdp", "fsdp_transpose")), (3, 16))
    self.assertEqual(out.shape, (3, 16))
    self.assertNotIn("fsdp", str(out.sharding.spec))
    self.assertEqual(len(gathers), 1)
    self.assertIn("all_gather_dim = 1", gathers[0])
    self.assertIn('name = "fsdp"', gathers[0])
    self.assertIn('name = "fsdp_transpose"', gathers[0])
    self.assertIn("(tensor<3x2xf32>) -> tensor<3x8xf32>", gathers[0])
    self.assertIn('xla_explicit_fsdp = "yes"', gathers[0])

  def test_non_minor_fsdp_axis_falls_back_to_resharding(self):
    mesh = _abstract_mesh((2, 4), ("tensor", "fsdp"))
    # 'fsdp' is not the minor-most axis of dim 0, so a tiled gather would not give the target layout.
    out = fsdp_prefetch._explicit_fsdp_all_gather(  # pylint: disable=protected-access
        jnp.ones((8, 6)), mesh, PartitionSpec(("fsdp", "tensor"), None), PartitionSpec("tensor", None)
    )
    self.assertIsNone(out)

  def test_leaf_without_named_sharding_is_left_alone(self):
    mesh = _abstract_mesh((4,), ("fsdp",))
    leaf = np.ones((2, 3))
    with mock.patch.object(fsdp_prefetch.jax, "typeof", return_value=types.SimpleNamespace(sharding=None)):
      self.assertIs(fsdp_prefetch._apply_sharding_hint(leaf, mesh, ShardMode.EXPLICIT), leaf)  # pylint: disable=protected-access
      self.assertIsNone(fsdp_prefetch._per_layer_sharding(leaf, mesh))  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
