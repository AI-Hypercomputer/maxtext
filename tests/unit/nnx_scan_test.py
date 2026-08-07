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
from unittest import mock

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


class TestApplyScannedLayers(unittest.TestCase):
  """Tests for nnx_scan.apply_scanned_layers."""

  def test_nonzero_param_scan_axis_round_trip(self):
    """The scan dimension is stored at param_scan_axis and restored for application."""
    length = 3
    layers = nnx_scan.create_scanned_layers(
        _LinearLayer,
        length=length,
        param_scan_axis=1,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(0),
    )

    self.assertEqual(layers.kernel.value.shape, (2, length, 2))

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
    self.assertEqual(layers.kernel.value.shape, (2, length, 2))

  def test_full_remat_checkpoints_scan_body_with_none_policy(self):
    """A None policy means full remat and must not disable jax.checkpoint."""
    layers = nnx_scan.create_scanned_layers(
        _LinearLayer,
        length=2,
        param_scan_axis=1,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(0),
    )

    with mock.patch.object(nnx_scan.jax, "checkpoint", wraps=jax.checkpoint) as checkpoint:
      nnx_scan.apply_scanned_layers(
          layers,
          jnp.array([1.0, -1.0]),
          length=2,
          param_scan_axis=1,
          apply_fn=lambda layer, carry: layer(carry),
          remat=True,
          remat_policy=None,
      )

    checkpoint.assert_called_once()
    self.assertIsNone(checkpoint.call_args.kwargs["policy"])

  def test_pinned_host_parameters_transferred_to_device(self):
    """Parameters on pinned_host are transferred to device before scan."""
    length = 2
    layers = nnx_scan.create_scanned_layers(
        _LinearLayer,
        length=length,
        param_scan_axis=0,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(0),
    )

    mesh = jax.sharding.Mesh(jax.devices()[:1], ("data",))
    sharding_host = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()).with_memory_kind("pinned_host")

    layers.kernel.value = jax.device_put(layers.kernel.value, sharding_host)

    observed_memory_spaces = []

    def apply_fn(layer, carry):
      ms = getattr(jax.typeof(layer.kernel.value), "memory_space", None)
      observed_memory_spaces.append(str(ms))
      return layer(carry)

    actual = nnx_scan.apply_scanned_layers(
        layers,
        jnp.array([1.0, -1.0]),
        length=length,
        param_scan_axis=0,
        apply_fn=apply_fn,
    )

    self.assertTrue(len(observed_memory_spaces) >= 1)
    for ms in observed_memory_spaces:
      self.assertNotIn("host", ms.lower())
    self.assertIsNotNone(actual)


if __name__ == "__main__":
  unittest.main()

