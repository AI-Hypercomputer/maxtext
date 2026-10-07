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

"""Tests for the helpers in maxtext.layers.fsdp_prefetch.

End-to-end gradient parity of the prefetch pipeline against the regular scanned layer stack is
tested in nnx_decoders_test.py (TestNNXDecoderPrefetchGradParity).
"""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.layers import fsdp_prefetch


class FsdpPrefetchHelpersTest(unittest.TestCase):
  """Tests for the static loop-split, ordering, and stacking helpers."""

  def test_fwd_split_covers_all_layers(self):
    for length in range(1, 10):
      s, num_pairs = fsdp_prefetch._prefetch_fwd_split(length)  # pylint: disable=protected-access
      self.assertIn(s, (0, 1))
      # Head layers + loop layers + the last layer (run after the loop) = all layers.
      self.assertEqual(s + 2 * num_pairs + 1, length)

  def test_can_order_requires_multi_trip_loop(self):
    self.assertFalse(fsdp_prefetch._prefetch_can_order(0))  # pylint: disable=protected-access
    self.assertFalse(fsdp_prefetch._prefetch_can_order(1))  # pylint: disable=protected-access
    self.assertTrue(fsdp_prefetch._prefetch_can_order(2))  # pylint: disable=protected-access

  def test_interleave_restores_layer_order(self):
    even = {"w": jnp.array([[0.0], [2.0], [4.0]])}
    odd = {"w": jnp.array([[1.0], [3.0], [5.0]])}
    out = fsdp_prefetch._interleave(even, odd)  # pylint: disable=protected-access
    np.testing.assert_array_equal(out["w"][:, 0], np.arange(6.0))

  def test_take_and_expand0(self):
    tree = {"w": jnp.arange(12.0).reshape(3, 4)}
    layer = fsdp_prefetch._take(tree, 1)  # pylint: disable=protected-access
    np.testing.assert_array_equal(layer["w"], np.arange(4.0, 8.0))
    self.assertEqual(fsdp_prefetch._expand0(layer)["w"].shape, (1, 4))  # pylint: disable=protected-access

  def test_explicit_all_gather_without_fsdp_sharding_returns_none(self):
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tensor"))
    w = jnp.ones((4, 8))
    # A size-1 FSDP axis needs no gather; the caller then falls back to resharding.
    out = fsdp_prefetch._explicit_fsdp_all_gather(  # pylint: disable=protected-access
        w, mesh, PartitionSpec("fsdp", "tensor"), PartitionSpec(None, "tensor")
    )
    self.assertIsNone(out)

  def test_per_layer_sharding_drops_layer_axis(self):
    # Explicit mesh axes, so that the sharding spec is part of the array's type (jax.typeof).
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tensor"), axis_types=(AxisType.Explicit,) * 2)
    stacked = jax.device_put(jnp.ones((2, 4, 8)), NamedSharding(mesh, PartitionSpec(None, "fsdp", "tensor")))
    s = fsdp_prefetch._per_layer_sharding(stacked, mesh)  # pylint: disable=protected-access
    self.assertEqual(tuple(s.spec), ("fsdp", "tensor"))


if __name__ == "__main__":
  unittest.main()
