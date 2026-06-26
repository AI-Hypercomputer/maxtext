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

"""qwix + NNX coverage: the config guard and the ToNNX->Linen bridge.

- Config guard: qwix quantization under pure_nnx requires the pure NNX decoder.
  The bridged Linen decoder (pure_nnx_decoder=False) is invisible to qwix, so
  quantization/sparsity would silently no-op; validation must reject that combo.
- Bridge: nnx_attrs_to_linen_vars must skip qwix's non-Variable bookkeeping attrs
  (qwix_path/qwix_rngs/disable_quant_stats_update) instead of raising.
"""

import sys
import unittest

import jax.numpy as jnp
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.layers import nnx_wrappers
from tests.utils.test_helpers import get_test_config_path


class QwixNnxQuantGuardTest(unittest.TestCase):

  def _init(self, **overrides):
    overrides.setdefault("enable_checkpointing", False)
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **overrides)

  def test_bridged_decoder_with_qwix_quant_raises(self):
    with self.assertRaisesRegex(Exception, "pure_nnx_decoder"):
      self._init(pure_nnx=True, pure_nnx_decoder=False, use_qwix_quantization=True, quantization="fp8_full")

  def test_pure_nnx_decoder_with_qwix_quant_ok(self):
    cfg = self._init(pure_nnx=True, pure_nnx_decoder=True, use_qwix_quantization=True, quantization="fp8_full")
    self.assertTrue(cfg.pure_nnx_decoder)

  def test_bridged_decoder_without_quant_ok(self):
    cfg = self._init(pure_nnx=True, pure_nnx_decoder=False, quantization="")
    self.assertEqual(cfg.quantization, "")


class NnxAttrsToLinenVarsBridgeTest(unittest.TestCase):
  """The ToNNX->Linen conversion must skip qwix's non-Variable attrs."""

  def test_non_variable_attrs_are_skipped(self):
    # qwix attaches plain attrs (qwix_path / disable_quant_stats_update) during
    # interception; before the fix these raised "Cannot infer collection name".
    attrs = {
        "kernel": nnx.Param(jnp.ones((2, 3))),
        "qwix_path": ("decoder", "layer"),
        "disable_quant_stats_update": True,
    }
    out = nnx_wrappers.nnx_attrs_to_linen_vars(attrs)  # must not raise
    keys = {k for kp in nnx.traversals.flatten_mapping(out) for k in kp}
    self.assertIn("params", keys)  # the real Variable survived, under its collection
    self.assertIn("kernel", keys)
    self.assertNotIn("qwix_path", keys)
    self.assertNotIn("disable_quant_stats_update", keys)

  def test_only_non_variable_attrs_yields_empty(self):
    out = nnx_wrappers.nnx_attrs_to_linen_vars({"qwix_path": ("x",), "qwix_rngs": 0})
    self.assertEqual(out, {})


if __name__ == "__main__":
  unittest.main()
