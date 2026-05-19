# Copyright 2023–2026 Google LLC
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

""" Tests for NNX utilities. """

import unittest
from flax import nnx
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.utils import maxtext_utils_nnx  # pylint: disable=no-name-in-module


class TestMaxtextUtilsNnx(unittest.TestCase):
  """Tests for maxtext_utils_nnx."""

  def test_nnx_sync_moveaxis(self):
    val = jnp.zeros((10, 20, 30))
    sharding = P("layers", "embed", None)

    var = nnx.Param(val)
    var.set_metadata(sharding=sharding, out_sharding=sharding)

    moved_var = maxtext_utils_nnx.nnx_sync_moveaxis(var, 0, 2)

    self.assertEqual(moved_var.get_value().shape, (20, 30, 10))
    self.assertEqual(moved_var.get_metadata().get("sharding"), P("embed", None, "layers"))
    self.assertEqual(moved_var.get_metadata().get("out_sharding"), P("embed", None, "layers"))

  def test_nnx_remove_scan_axis(self):
    val = jnp.zeros((10, 20))
    sharding = P("layers", "embed")
    var = nnx.Param(val)
    var.set_metadata(sharding=sharding)

    # Simulate slicing
    sliced_val = val[0]
    var_with_sliced_val = var.replace(value=sliced_val)

    fixed_var = maxtext_utils_nnx.nnx_remove_scan_axis(var_with_sliced_val, "layers")

    self.assertEqual(fixed_var.get_value().ndim, 1)
    self.assertEqual(fixed_var.get_metadata().get("sharding"), P("embed"))

  def test_nnx_add_scan_axis(self):
    val = jnp.zeros((20,))
    sharding = P("embed")
    var = nnx.Param(val)
    var.set_metadata(sharding=sharding)

    # Simulate stacking
    stacked_val = jnp.stack([val] * 10, axis=0)
    var_with_stacked_val = var.replace(value=stacked_val)

    fixed_var = maxtext_utils_nnx.nnx_add_scan_axis(var_with_stacked_val, "layers", 0)

    self.assertEqual(fixed_var.get_value().ndim, 2)
    self.assertEqual(fixed_var.get_metadata().get("sharding"), P("layers", "embed"))


if __name__ == "__main__":
  unittest.main()
