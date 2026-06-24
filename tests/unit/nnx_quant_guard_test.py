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

"""Config guard: qwix quantization under pure_nnx requires the pure NNX decoder.

The bridged Linen decoder (pure_nnx_decoder=False) is invisible to qwix, so
quantization/sparsity would silently no-op. Config validation must reject that
combination and accept the supported ones.
"""

import sys
import unittest

from maxtext.configs import pyconfig
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


if __name__ == "__main__":
  unittest.main()
