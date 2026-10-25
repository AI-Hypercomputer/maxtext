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

import os
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODESIGN_DIR = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_REPO_SRC = os.path.abspath(os.path.join(_CODESIGN_DIR, "..", "src"))
if not os.path.isdir(_REPO_SRC):
  _REPO_SRC = os.path.abspath(os.path.join(_CODESIGN_DIR, ".."))
_PKG_CODESIGN_DIR = os.path.join(_REPO_SRC, "maxtext", "codesign")
for _p in (_CODESIGN_DIR, _PKG_CODESIGN_DIR, _REPO_SRC):
  if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

from emulation_base import (  # type: ignore  # noqa: E402
    EmulationFeature,
    get_registered_emulation_features,
    register_emulation_feature,
)
from hooks import install_emulation_hooks  # type: ignore  # noqa: E402
from interception import (  # type: ignore  # noqa: E402
    emulation_jax_layer_context,
    emulation_jax_step_context,
    emulation_step_context,
    get_current_emulation_step,
    get_current_module_path,
    module_path_scope,
)


class CodesignInfrastructureTest(unittest.TestCase):
  """Tests general hardware co-design plugin registration and runtime scopes."""

  def test_emulation_feature_plugin_registry(self):
    @register_emulation_feature("codesign_test_feature")
    class DummyFeature(EmulationFeature):
      feature_name = "codesign_test_feature"

    registry = get_registered_emulation_features()
    self.assertIn("codesign_test_feature", registry)
    feat = registry["codesign_test_feature"]()
    self.assertFalse(feat.enabled)
    self.assertEqual(feat.transform_input_activation(42, "decoder/layers_0"), 42)

  def test_module_path_and_step_scopes(self):
    self.assertEqual(get_current_module_path(), "")
    with module_path_scope("decoder"):
      self.assertEqual(get_current_module_path(), "decoder")
      with module_path_scope("layers_0"):
        self.assertEqual(get_current_module_path(), "decoder/layers_0")
      self.assertEqual(get_current_module_path(), "decoder")
    self.assertEqual(get_current_module_path(), "")

    with emulation_step_context(15):
      self.assertEqual(get_current_emulation_step(), 15)
    with emulation_jax_step_context(25), emulation_jax_layer_context(3):
      pass

  def test_install_emulation_hooks_idempotent(self):
    install_emulation_hooks(workflow="sft")
    install_emulation_hooks(workflow="sft")


if __name__ == "__main__":
  unittest.main()
