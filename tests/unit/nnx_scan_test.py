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

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import nnx_scan


class _LinearLayer(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
    self.kernel = nnx.Param(jax.random.normal(rngs.params(), (2, 2)))

  def __call__(self, inputs):
    return inputs @ self.kernel.value


def test_nonzero_param_scan_axis_round_trip():
  """The scan dimension is stored at param_scan_axis and restored for application."""
  length = 3
  layers = nnx_scan.create_scanned_layers(
      _LinearLayer,
      length=length,
      param_scan_axis=1,
      metadata_axis_name="layers",
      rngs=nnx.Rngs(0),
  )

  assert layers.kernel.value.shape == (2, length, 2)

  inputs = jnp.array([1.0, -1.0])
  kernels = jnp.moveaxis(layers.kernel.value, 1, 0)
  expected = inputs
  for kernel in kernels:
    expected = expected @ kernel

  actual = nnx_scan.apply_scanned_layers(
      layers,
      inputs,
      length=length,
      param_scan_axis=1,
      apply_fn=lambda layer, carry: layer(carry),
  )

  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
  assert layers.kernel.value.shape == (2, length, 2)
