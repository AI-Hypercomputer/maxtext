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

"""Tests for NNX scan utilities."""

import unittest

from flax import nnx
import jax
import pytest

from maxtext.layers import nnx_scan


class _LinearLayer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (2, 2)))

  def __call__(self, inputs):
    return inputs @ self.kernel.value


@pytest.mark.cpu_only
class TestCreateScannedLayers(unittest.TestCase):
  """Tests for nnx_scan.create_scanned_layers."""

  def test_create_stacks_params_at_param_scan_axis(self):
    """Per-layer params are stacked along param_scan_axis."""
    length = 3
    for axis, expected_shape in ((0, (length, 2, 2)), (1, (2, length, 2))):
      layers = nnx_scan.create_scanned_layers(
          _LinearLayer,
          length=length,
          param_scan_axis=axis,
          metadata_axis_name="layers",
          rngs=nnx.Rngs(0),
      )
      self.assertEqual(layers.kernel.value.shape, expected_shape)

  def test_create_zero_length_returns_none(self):
    """A zero-length stack short-circuits to None."""
    layers = nnx_scan.create_scanned_layers(
        _LinearLayer,
        length=0,
        param_scan_axis=0,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(0),
    )
    self.assertIsNone(layers)


if __name__ == "__main__":
  unittest.main()
