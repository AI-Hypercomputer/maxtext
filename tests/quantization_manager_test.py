# Copyright 2023–2025 Google LLC
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

"""Tests for quantization_config."""

import unittest
from absl.testing import absltest
import jax.numpy as jnp
from MaxText.kernels.megablox.quantization_config import QuantizationManager, QuantizationConfig
import qwix

class QuantizationManagerTest(unittest.TestCase):

  def test_fallback_quantization(self):
    """Test that the fallback quantization is used when qwix is disabled."""
    fallback_config = QuantizationConfig(
        lhs_quantize_dtype=jnp.int8,
        rhs_quantize_dtype=jnp.int8,
        lhs_calibration_method="absmax",
        rhs_calibration_method="absmax",
    )
    manager = QuantizationManager(
        quantization_rule=None,
        use_qwix_quantization=False,
        fallback=fallback_config,
    )
    self.assertEqual(manager.for_phase("fwd"), fallback_config)
    self.assertEqual(manager.for_phase("dlhs"), fallback_config)
    self.assertEqual(manager.for_phase("drhs"), fallback_config)

  def test_qwix_quantization(self):
    """Test that the qwix quantization is used when qwix is enabled."""
    quantization_rule = qwix.QtRule(
        module_path="decoder/.*layers.*",
        weight_qtype=jnp.int8,
        act_qtype=jnp.int8,
        bwd_qtype=jnp.int8,
        op_names=("dot_general",),
    )
    fallback_config = QuantizationConfig(
        lhs_quantize_dtype=None,
        rhs_quantize_dtype=None,
        lhs_calibration_method="absmax",
        rhs_calibration_method="absmax",
    )
    manager = QuantizationManager(
        quantization_rule=quantization_rule,
        use_qwix_quantization=True,
        fallback=fallback_config,
    )
    self.assertEqual(manager.for_phase("fwd").lhs_quantize_dtype, jnp.int8)
    self.assertEqual(manager.for_phase("fwd").rhs_quantize_dtype, jnp.int8)
    self.assertEqual(manager.for_phase("dlhs").lhs_quantize_dtype, jnp.int8)
    self.assertEqual(manager.for_phase("dlhs").rhs_quantize_dtype, jnp.int8)
    self.assertEqual(manager.for_phase("drhs").lhs_quantize_dtype, jnp.int8)
    self.assertEqual(manager.for_phase("drhs").rhs_quantize_dtype, jnp.int8)

if __name__ == "__main__":
  absltest.main()
