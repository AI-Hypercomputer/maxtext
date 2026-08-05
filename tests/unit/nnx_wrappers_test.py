# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ToNNX wrappers and layer retrieval."""

import unittest
from unittest import mock
import jax
from flax import linen as nn
from flax import nnx
from maxtext.layers import nnx_wrappers
from maxtext.common.common_types import Config


class DummyLayer(nn.Module):
  """Dummy Linen layer for testing."""

  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=10)(x)


class DummyDecoderScanned(nn.Module):
  """Dummy scanned decoder for testing."""

  config: Config

  def setup(self):
    self.layers = DummyLayer()

  def __call__(self, x):
    return self.layers(x)


class DummyDecoderUnscanned(nn.Module):
  """Dummy unscanned decoder for testing."""

  config: Config

  def setup(self):
    self.layers_0 = DummyLayer()
    self.layers_1 = DummyLayer()

  def __call__(self, x):
    x = self.layers_0(x)
    x = self.layers_1(x)
    return x


class ToNNXGetLayersTest(unittest.TestCase):
  """Tests for ToNNX.get_layers()."""

  def test_get_layers_scanned(self):
    cfg = mock.Mock()
    cfg.scan_layers = True
    cfg.num_decoder_layers = 2

    decoder = DummyDecoderScanned(config=cfg)
    x = jax.numpy.ones((1, 10))

    # Initialize ToNNX
    model = nnx_wrappers.ToNNX(decoder, rngs=nnx.Rngs(0)).lazy_init(x)

    self.assertTrue(hasattr(model, "layers"))
    layers = model.get_layers()
    self.assertEqual(len(layers), 1)
    self.assertIsInstance(layers[0], (nnx.Module, dict))
    self.assertEqual(layers[0], model.layers)

  def test_get_layers_unscanned(self):
    cfg = mock.Mock()
    cfg.scan_layers = False
    cfg.num_decoder_layers = 2

    decoder = DummyDecoderUnscanned(config=cfg)
    x = jax.numpy.ones((1, 10))

    # Initialize ToNNX
    model = nnx_wrappers.ToNNX(decoder, rngs=nnx.Rngs(0)).lazy_init(x)

    self.assertTrue(hasattr(model, "layers_0"))
    self.assertTrue(hasattr(model, "layers_1"))
    layers = model.get_layers()
    self.assertEqual(len(layers), 2)
    self.assertIsInstance(layers[0], (nnx.Module, dict))
    self.assertIsInstance(layers[1], (nnx.Module, dict))
    self.assertEqual(layers[0], model.layers_0)
    self.assertEqual(layers[1], model.layers_1)


if __name__ == "__main__":
  unittest.main()
