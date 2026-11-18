# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic tests for `load_hf_reference` / `unpatch_hf_reference`.

Uses a tiny synthetic checkout shaped like upstream `moonshotai/Kimi-K3` (relative
`from .configuration_kimi_k3 import ...`, and a `KimiLinearModel.__init__` that forces
`flash_attention_2`), so no real HF checkout, `fla` or `flash_attn` is needed.
"""

import os
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import pytest

pytest.importorskip("torch")

# pylint: disable=wrong-import-position
from tests.utils import kimi_k3_parity_utils as pu

_CONFIGURATION_SRC = """
class KimiLinearConfig:
  def __init__(self, _attn_implementation=None):
    self._attn_implementation = _attn_implementation
"""

# Mirrors upstream: relative import + unconditional flash_attention_2 override in __init__.
_MODELING_SRC = """
from .configuration_kimi_k3 import KimiLinearConfig


def chunk_kda(*args, **kwargs):
  raise RuntimeError("fla kernel")


def fused_recurrent_kda(*args, **kwargs):
  raise RuntimeError("fla kernel")


class KimiLinearModel:
  def __init__(self, config):
    config._attn_implementation = "flash_attention_2"
    self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
"""

_REF_MODULE_NAMES = ("_kimi_k3_hf_ref", "configuration_kimi_k3", "modeling_kimi_linear")


class LoadHfReferenceTest(unittest.TestCase):
  """`load_hf_reference` must work on an unmodified upstream-shaped checkout and fully unpatch."""

  def setUp(self):
    super().setUp()
    for name in list(sys.modules):
      if name.split(".")[0] in _REF_MODULE_NAMES:
        self.skipTest(f"{name} already imported in this process")
    tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(tmp.cleanup)
    for fname, src in (("configuration_kimi_k3.py", _CONFIGURATION_SRC), ("modeling_kimi_linear.py", _MODELING_SRC)):
      with open(os.path.join(tmp.name, fname), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(src))
    patcher = mock.patch.object(pu, "KIMI_K3_REFERENCE_DIR", tmp.name)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.addCleanup(pu.unpatch_hf_reference)
    self.saved_path = list(sys.path)

  def test_relative_import_resolves_and_config_class_is_shared(self):
    hf_config, hf_model = pu.load_hf_reference(patch_kda=False)
    self.assertIs(hf_model.KimiLinearConfig, hf_config.KimiLinearConfig)
    self.assertEqual(sys.path, self.saved_path)

  def test_requested_attn_implementation_survives_and_kda_is_patched(self):
    hf_config, hf_model = pu.load_hf_reference(patch_kda=True)
    config = hf_config.KimiLinearConfig(_attn_implementation="eager")
    model = hf_model.KimiLinearModel(config)
    self.assertEqual(config._attn_implementation, "eager")  # pylint: disable=protected-access
    self.assertFalse(model._use_flash_attention_2)  # pylint: disable=protected-access
    self.assertIs(hf_model.chunk_kda, pu.chunk_kda_reference)
    self.assertIs(hf_model.fused_recurrent_kda, pu.chunk_kda_reference)
    # Without an explicit request the upstream default is untouched.
    unset = hf_config.KimiLinearConfig()
    hf_model.KimiLinearModel(unset)
    self.assertEqual(unset._attn_implementation, "flash_attention_2")  # pylint: disable=protected-access

  def test_unpatch_restores_module_attributes_and_sys_modules(self):
    _, hf_model = pu.load_hf_reference(patch_kda=True)
    pu.load_hf_reference(patch_kda=True)  # idempotent: must not capture the patched init as "original"
    pu.unpatch_hf_reference()
    config = hf_model.KimiLinearConfig(_attn_implementation="eager")
    hf_model.KimiLinearModel(config)
    self.assertEqual(config._attn_implementation, "flash_attention_2")  # pylint: disable=protected-access
    self.assertIsNot(hf_model.chunk_kda, pu.chunk_kda_reference)
    for name in _REF_MODULE_NAMES + tuple(f"_kimi_k3_hf_ref.{m}" for m in _REF_MODULE_NAMES[1:]):
      self.assertNotIn(name, sys.modules)
    self.assertEqual(sys.path, self.saved_path)


if __name__ == "__main__":
  unittest.main()
