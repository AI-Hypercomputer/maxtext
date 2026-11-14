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

  def test_scan_axis_name_is_inserted_into_tuple_and_list_logical_axes(self):
    """The scan axis name lands at param_scan_axis whether the logical axes are a tuple or a list."""

    class _NamedLayer(nnx.Module):

      def __init__(self, rngs: nnx.Rngs, logical_axes):
        del rngs  # Unused.
        self.kernel = nnx.Param(jnp.zeros((2, 2)), sharding_names=logical_axes)

    for logical_axes in (("embed", "mlp"), ["embed", "mlp"]):
      with self.subTest(type=type(logical_axes).__name__):
        layers = nnx_scan.create_scanned_layers(
            lambda rngs, logical_axes=logical_axes: _NamedLayer(rngs, logical_axes),
            length=3,
            param_scan_axis=1,
            metadata_axis_name="layers",
            rngs=nnx.Rngs(0),
        )
        metadata = layers.kernel.get_metadata()
        stored = metadata.get("sharding_names", metadata.get("out_sharding"))
        self.assertEqual(stored, ("embed", "layers", "mlp"))

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

  def test_new_params_created_in_scan_preserve_param_scan_axis(self):
    """Parameters materialized inside apply_fn get param_scan_axis metadata."""
    length = 3
    layers = nnx_scan.create_scanned_layers(
        _LinearLayer,
        length=length,
        param_scan_axis=1,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(0),
    )

    def materialize_adapter(layer, carry):
      if not hasattr(layer, "adapter"):
        layer.adapter = nnx.Param(jnp.ones((2, 4)))
      return layer(carry)

    inputs = jnp.array([1.0, -1.0])
    nnx_scan.apply_scanned_layers(
        layers,
        inputs,
        length=length,
        param_scan_axis=1,
        apply_fn=materialize_adapter,
    )

    self.assertEqual(layers.adapter.value.shape, (2, length, 4))
    self.assertEqual(layers.adapter.get_metadata().get("param_scan_axis"), 1)

  def test_shared_rngs_not_overwritten_with_scanned_keys(self):
    """Shared unstacked nnx.Rngs attached to layers keeps scalar key shape."""
    length = 3
    layers = nnx_scan.create_scanned_layers(
        _LinearLayer,
        length=length,
        param_scan_axis=1,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(0),
    )
    shared_rngs = nnx.Rngs(42)
    layers.qwix_rngs = shared_rngs

    nnx_scan.apply_scanned_layers(
        layers,
        jnp.array([1.0, -1.0]),
        length=length,
        param_scan_axis=1,
        apply_fn=lambda layer, carry: layer(carry),
    )

    self.assertEqual(shared_rngs.default.key.value.shape, ())
    sample = jax.random.normal(shared_rngs.default(), (2, 2))
    self.assertEqual(sample.shape, (2, 2))

  def test_stacked_per_layer_rngs_advance_in_scan(self):
    """Per-layer stacked RngState advances across calls while broadcast scalar RngState stays scalar."""

    class _DropoutLinearLayer(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (2, 2)))
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

      def __call__(self, inputs):
        return self.dropout(inputs @ self.kernel.value, deterministic=False)

    length = 3
    layers = nnx_scan.create_scanned_layers(
        _DropoutLinearLayer,
        length=length,
        param_scan_axis=1,
        metadata_axis_name="layers",
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    shared_rngs = nnx.Rngs(42)
    layers.qwix_rngs = shared_rngs

    np.testing.assert_array_equal(layers.dropout.rngs.dropout.count.value, np.zeros(length, dtype=np.uint32))

    inputs = jnp.array([1.0, -1.0])
    out1 = nnx_scan.apply_scanned_layers(
        layers,
        inputs,
        length=length,
        param_scan_axis=1,
        apply_fn=lambda layer, carry: layer(carry),
    )
    np.testing.assert_array_equal(layers.dropout.rngs.dropout.count.value, np.ones(length, dtype=np.uint32))
    self.assertEqual(shared_rngs.default.key.value.shape, ())

    out2 = nnx_scan.apply_scanned_layers(
        layers,
        inputs,
        length=length,
        param_scan_axis=1,
        apply_fn=lambda layer, carry: layer(carry),
    )
    np.testing.assert_array_equal(layers.dropout.rngs.dropout.count.value, np.full(length, 2, dtype=np.uint32))
    self.assertEqual(shared_rngs.default.key.value.shape, ())
    self.assertEqual(out1.shape, (2,))
    self.assertEqual(out2.shape, (2,))


if __name__ == "__main__":
  unittest.main()
