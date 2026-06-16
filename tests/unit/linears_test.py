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

"""Tests for linears.py."""

import sys
import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import linears
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class UtilsTest(unittest.TestCase):
  """Tests for utility functions in linears.py."""

  def test_normalize_axes(self):
    self.assertEqual(linears.normalize_axes((1, 2), 4), (1, 2))
    self.assertEqual(linears.normalize_axes((-1, -2), 4), (3, 2))
    self.assertEqual(linears.normalize_axes((0, -1), 3), (0, 2))

  def test_canonicalize_tuple(self):
    self.assertEqual(linears.canonicalize_tuple(1), (1,))
    self.assertEqual(linears.canonicalize_tuple((1, 2)), (1, 2))
    self.assertEqual(linears.canonicalize_tuple([1, 2]), (1, 2))

  # pylint: disable=protected-access
  def test_convert_to_activation_function(self):
    lin_fn = linears._convert_to_activation_function("linear")
    x = jnp.array([1.0, 2.0])
    np.testing.assert_array_equal(lin_fn(x), x)

    relu_fn = linears._convert_to_activation_function("relu")
    x = jnp.array([-1.0, 2.0])
    np.testing.assert_array_equal(relu_fn(x), jnp.array([0.0, 2.0]))

    # Test with callable
    def dummy_fn(x):
      return x + 1

    self.assertEqual(linears._convert_to_activation_function(dummy_fn), dummy_fn)

    with self.assertRaises(ValueError):
      linears._convert_to_activation_function(123)


class DenseGeneralTest(unittest.TestCase):
  """Tests for DenseGeneral."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

  def test_basic_call(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))

  def test_bias(self):
    batch_size = 2
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        use_bias=True,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, out_features))
    self.assertIsNotNone(layer.bias)

  def _run_dense_test(self, axis, in_feat_shape, expected_shape):
    batch_size = 2
    seq_len = 3
    in_features = 4
    out_features = 8

    layer = linears.DenseGeneral(
        in_features_shape=in_feat_shape,
        out_features_shape=out_features,
        axis=axis,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, expected_shape)

  def test_axis_neg_1(self):
    self._run_dense_test(-1, 4, (2, 3, 8))

  def test_axis_1(self):
    self._run_dense_test(1, 3, (2, 4, 8))

  def test_axis_0(self):
    self._run_dense_test(0, 2, (3, 4, 8))


class MlpBlockTest(unittest.TestCase):
  """Tests for MlpBlock."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0, dropout=1)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "fused_mlp": False,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    batch_size = 2
    seq_len = 3
    in_features = 4
    intermediate_dim = 8

    layer = linears.MlpBlock(
        config=self.cfg,
        mesh=self.mesh,
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, in_features))
    self.assertEqual(layer.wi.kernel[...].shape, (in_features, intermediate_dim))

  def test_fused_mlp(self):
    batch_size = 2
    seq_len = 3
    in_features = 4
    intermediate_dim = 8

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
        "fused_mlp": True,
    }
    argv = [sys.argv[0], get_test_config_path()]
    cfg_fused = pyconfig.initialize(argv, **config_arguments)

    layer = linears.MlpBlock(
        config=cfg_fused,
        mesh=self.mesh,
        in_features=in_features,
        intermediate_dim=intermediate_dim,
        rngs=self.rngs,
    )

    inputs = jnp.ones((batch_size, seq_len, in_features))
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, in_features))
    self.assertEqual(layer.wi.kernel[...].shape, (in_features, 1, intermediate_dim))


if __name__ == "__main__":
  unittest.main()
