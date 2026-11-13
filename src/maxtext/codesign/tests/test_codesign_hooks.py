#!/usr/bin/env python3
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

"""General Co-Design infrastructure unit tests (feature registry, module-path scope, step/layer context)."""

from __future__ import annotations

import unittest

from maxtext.codesign.emulation_base import (
    EmulationFeature,
    get_registered_emulation_features,
    register_emulation_feature,
)
from maxtext.codesign.hooks import install_emulation_hooks
from maxtext.codesign.interception import (
    _compute_active_cond,
    emulation_jax_layer_context,
    emulation_jax_step_context,
    emulation_step_context,
    get_current_emulation_step,
    get_current_jax_layer_idx,
    get_current_jax_step,
    get_current_module_path,
    module_path_scope,
)
from maxtext.codesign.multi_pass.multipass_config import MultipassEmulationConfig


class CodesignInfrastructureTest(unittest.TestCase):
  """Tests general hardware co-design plugin registration and runtime scopes."""

  def test_emulation_feature_plugin_registry(self):
    """Tests registering and instantiating a custom EmulationFeature plugin."""

    @register_emulation_feature("codesign_test_feature")
    class DummyFeature(EmulationFeature):
      """Dummy feature plugin for registry testing."""

      feature_name = "codesign_test_feature"

    registry = get_registered_emulation_features()
    self.assertIn("codesign_test_feature", registry)
    self.assertIs(registry["codesign_test_feature"], DummyFeature)
    feat = registry["codesign_test_feature"]()
    self.assertFalse(feat.enabled)
    self.assertEqual(feat.transform_input_activation(42, "decoder/layers_0"), 42)

  def test_module_path_and_step_scopes(self):
    """Tests nested module_path_scope and step/layer context managers."""
    self.assertEqual(get_current_module_path(), "")
    with module_path_scope("decoder"):
      self.assertEqual(get_current_module_path(), "decoder")
      with module_path_scope("layers_0"):
        self.assertEqual(get_current_module_path(), "decoder/layers_0")
      self.assertEqual(get_current_module_path(), "decoder")
    self.assertEqual(get_current_module_path(), "")

    with emulation_step_context(15):
      self.assertEqual(get_current_emulation_step(), 15)
    cfg = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "layers": "3",
            "start_step": 20,
            "end_step": 30,
        }
    )
    with emulation_jax_step_context(25), emulation_jax_layer_context(3):
      self.assertEqual(get_current_jax_step(), 25)
      self.assertEqual(get_current_jax_layer_idx(), 3)
      self.assertTrue(bool(_compute_active_cond(cfg, cfg.rules[0])))
    with emulation_jax_step_context(10), emulation_jax_layer_context(3):
      self.assertFalse(bool(_compute_active_cond(cfg, cfg.rules[0])))
    with emulation_jax_step_context(25), emulation_jax_layer_context(1):
      self.assertFalse(bool(_compute_active_cond(cfg, cfg.rules[0])))

  def test_install_emulation_hooks_idempotent(self):
    """Tests that install_emulation_hooks is idempotent across repeated calls."""
    install_emulation_hooks(workflow="sft")
    install_emulation_hooks(workflow="sft")


if __name__ == "__main__":
  unittest.main()
