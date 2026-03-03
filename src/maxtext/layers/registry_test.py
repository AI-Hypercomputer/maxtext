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

"""Tests for registry.py."""

import unittest
from absl.testing import absltest
from maxtext.layers import registry

class RegistryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Clear registry before each test to avoid side effects
    registry._MODEL_PLUGIN_REGISTRY.clear()

  def test_register_and_get_model_plugin(self):
    class MockLayer:
      pass

    class MockStrategy:
      pass

    @registry.register_model("test_model", MockStrategy)
    class TestLayer(MockLayer):
      pass

    plugin = registry.get_model_plugin("test_model")
    self.assertEqual(plugin.layer_classes, [TestLayer])
    self.assertEqual(plugin.strategy_class, MockStrategy)

  def test_register_multiple_layers(self):
    class MockLayer1:
      pass
    class MockLayer2:
      pass
    class MockStrategy:
      pass

    layers = [MockLayer1, MockLayer2]
    
    registry.register_model("test_multi_layer", MockStrategy)(layers)

    plugin = registry.get_model_plugin("test_multi_layer")
    self.assertEqual(plugin.layer_classes, layers)
    self.assertEqual(plugin.strategy_class, MockStrategy)

  def test_get_unknown_model_plugin(self):
    with self.assertRaisesRegex(ValueError, "Model plugin 'unknown_model' not found"):
      registry.get_model_plugin("unknown_model")

  def test_registry_cleared(self):
    # Ensure setUp works and registry is empty at start of test
    self.assertEmpty(registry._MODEL_PLUGIN_REGISTRY)

if __name__ == '__main__':
  absltest.main()
