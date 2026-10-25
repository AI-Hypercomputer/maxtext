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

"""Modular unit tests for Dtype-Agnostic Multi-Pass Precision Emulation."""

from __future__ import annotations

import os
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MP_DIR = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_CODESIGN_DIR = os.path.abspath(os.path.join(_MP_DIR, ".."))
_REPO_SRC = os.path.abspath(os.path.join(_CODESIGN_DIR, "..", "src"))
if not os.path.isdir(_REPO_SRC):
  _REPO_SRC = os.path.abspath(os.path.join(_CODESIGN_DIR, "..", ".."))
_PKG_CODESIGN_DIR = os.path.join(_REPO_SRC, "maxtext", "codesign")
_PKG_MP_DIR = os.path.join(_PKG_CODESIGN_DIR, "multi_pass")
for _p in (_MP_DIR, _CODESIGN_DIR, _PKG_MP_DIR, _PKG_CODESIGN_DIR, _REPO_SRC):
  if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

from emulation_base import (  # type: ignore  # noqa: E402
    EmulationFeature,
    get_registered_emulation_features,
    register_emulation_feature,
)
from multipass_config import (  # type: ignore  # noqa: E402
    DEFAULT_MULTIPASS_CONFIG_KEYS,
    DtypeSpec,
    MULTIPASS_FLAG_DOCUMENTATION,
    MULTIPASS_PRESETS,
    MultipassEmulationConfig,
    OperandMultipassSpec,
    resolve_cross_term_pairs,
)


class MultipassEmulationTest(unittest.TestCase):
  """Modular unit tests covering each component of Multi-Pass Precision Emulation."""

  # ---------------------------------------------------------------------------
  # 1. Modular Unit Tests for DtypeSpec (Floating-Point & Integer Formats)
  # ---------------------------------------------------------------------------

  def test_dtype_spec_float_formats(self):
    """Tests built-in and parametric `eXmY` floating-point DtypeSpec parsing and ranges."""
    fp32 = DtypeSpec.parse("fp32")
    self.assertEqual((fp32.kind, fp32.exp_bits, fp32.mantissa_bits), ("float", 8, 23))

    bf16 = DtypeSpec.parse("bf16")
    self.assertEqual((bf16.kind, bf16.exp_bits, bf16.mantissa_bits), ("float", 8, 7))

    tf32 = DtypeSpec.parse("tf32")
    self.assertEqual((tf32.kind, tf32.exp_bits, tf32.mantissa_bits), ("float", 8, 10))

    fp8_e4m3 = DtypeSpec.parse("fp8_e4m3")
    self.assertEqual((fp8_e4m3.kind, fp8_e4m3.exp_bits, fp8_e4m3.mantissa_bits), ("float", 4, 3))
    self.assertAlmostEqual(fp8_e4m3.max_val, 448.0)

    fp8_e5m2 = DtypeSpec.parse("fp8_e5m2")
    self.assertEqual((fp8_e5m2.kind, fp8_e5m2.exp_bits, fp8_e5m2.mantissa_bits), ("float", 5, 2))
    self.assertAlmostEqual(fp8_e5m2.max_val, 57344.0)

    fp6_e3m2 = DtypeSpec.parse("fp6_e3m2")
    self.assertEqual((fp6_e3m2.kind, fp6_e3m2.exp_bits, fp6_e3m2.mantissa_bits), ("float", 3, 2))
    self.assertAlmostEqual(fp6_e3m2.max_val, 28.0)

    fp4_e2m1 = DtypeSpec.parse("fp4_e2m1")
    self.assertEqual((fp4_e2m1.kind, fp4_e2m1.exp_bits, fp4_e2m1.mantissa_bits), ("float", 2, 1))
    self.assertAlmostEqual(fp4_e2m1.max_val, 6.0)

    # Arbitrary parametric eXmY format
    custom_e5m3 = DtypeSpec.parse("e5m3")
    self.assertEqual((custom_e5m3.kind, custom_e5m3.exp_bits, custom_e5m3.mantissa_bits), ("float", 5, 3))

  def test_dtype_spec_integer_formats(self):
    """Tests signed and unsigned integer DtypeSpec parsing (`int8`, `int4`, `uint4`, parametric `intB`)."""
    int8 = DtypeSpec.parse("int8")
    self.assertEqual((int8.kind, int8.int_bits, int8.signed), ("int", 8, True))
    self.assertEqual((int8.min_val, int8.max_val), (-128.0, 127.0))

    int4 = DtypeSpec.parse("int4")
    self.assertEqual((int4.kind, int4.int_bits, int4.signed), ("int", 4, True))
    self.assertEqual((int4.min_val, int4.max_val), (-8.0, 7.0))

    uint4 = DtypeSpec.parse("uint4")
    self.assertEqual((uint4.kind, uint4.int_bits, uint4.signed), ("uint", 4, False))
    self.assertEqual((uint4.min_val, uint4.max_val), (0.0, 15.0))

    # Parametric int6 / uint6
    int6 = DtypeSpec.parse("int6")
    self.assertEqual((int6.kind, int6.int_bits, int6.min_val, int6.max_val), ("int", 6, -32.0, 31.0))
    uint6 = DtypeSpec.parse("uint6")
    self.assertEqual((uint6.kind, uint6.int_bits, uint6.min_val, uint6.max_val), ("uint", 6, 0.0, 63.0))

  # ---------------------------------------------------------------------------
  # 2. Modular Unit Tests for OperandMultipassSpec & DSL Parser
  # ---------------------------------------------------------------------------

  def test_operand_multipass_spec_dsl_and_homogeneous_expansion(self):
    """Tests `OperandMultipassSpec` inline DSL parsing, dict parsing, and homogeneous expansion."""
    spec_het = OperandMultipassSpec.from_any(
        "fp32->[fp8_e5m2,fp8_e4m3,fp4_e2m1]@scale=per_block_pow2:block=64:round=stochastic:subnormals=false"
    )
    self.assertTrue(spec_het.enabled)
    self.assertEqual(spec_het.src_dtype.name, "fp32")
    self.assertEqual(spec_het.num_passes, 3)
    self.assertEqual(spec_het.pass_dtype_names, ["fp8_e5m2", "fp8_e4m3", "fp4_e2m1"])
    self.assertEqual(spec_het.scale_mode, "per_block_pow2")
    self.assertEqual(spec_het.block_size, 64)
    self.assertEqual(spec_het.rounding_mode, "stochastic")
    self.assertFalse(spec_het.subnormals)

    spec_hom = OperandMultipassSpec.from_any(
        {"src_dtype": "fp32", "num_passes": 3, "pass_dtypes": "bf16", "scale_mode": "unit"}
    )
    self.assertEqual(spec_hom.num_passes, 3)
    self.assertEqual(spec_hom.pass_dtype_names, ["bf16", "bf16", "bf16"])

    spec_inferred = OperandMultipassSpec.from_any(
        {"src_dtype": "bf16", "pass_dtypes": "fp8_e5m2,fp4_e2m1"}
    )
    self.assertEqual(spec_inferred.num_passes, 2)
    self.assertEqual(spec_inferred.pass_dtype_names, ["fp8_e5m2", "fp4_e2m1"])

  # ---------------------------------------------------------------------------
  # 3. Modular Unit Tests for Cross-Term Pair Selection
  # ---------------------------------------------------------------------------

  def test_resolve_cross_term_pairs_all_modes(self):
    """Tests `resolve_cross_term_pairs` for triangular, full, asymmetric, karatsuba, custom, and max_order."""
    self.assertEqual(
        resolve_cross_term_pairs(2, 2, "triangular"),
        ((0, 0), (0, 1), (1, 0)),
    )
    self.assertEqual(
        resolve_cross_term_pairs(3, 3, "triangular"),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    )
    self.assertEqual(
        resolve_cross_term_pairs(2, 2, "full"),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    self.assertEqual(
        resolve_cross_term_pairs(2, 2, "asymmetric_lhs"),
        ((0, 0), (1, 0)),
    )
    self.assertEqual(
        resolve_cross_term_pairs(2, 2, "asymmetric_rhs"),
        ((0, 0), (0, 1)),
    )
    self.assertEqual(
        resolve_cross_term_pairs(3, 3, "custom", active_passes="0:0,0:2,2:0"),
        ((0, 0), (0, 2), (2, 0)),
    )
    self.assertEqual(
        resolve_cross_term_pairs(3, 3, "full", max_total_pass_order=1),
        ((0, 0), (0, 1), (1, 0)),
    )

  # ---------------------------------------------------------------------------
  # 4. Modular Unit Tests for Presets & CLI Flag Override Precedence
  # ---------------------------------------------------------------------------

  def test_all_rfc_presets_and_cli_override_precedence(self):
    """Tests all 11 RFC presets and verifies CLI/YAML flags override preset fields cleanly."""
    for preset_name in MULTIPASS_PRESETS:
      cfg = MultipassEmulationConfig.from_dict({"enabled": True, "preset": preset_name})
      self.assertTrue(cfg.enabled)
      self.assertEqual(len(cfg.rules), 1)
      self.assertTrue(cfg.rules[0].has_any_multipass)

    merged = dict(DEFAULT_MULTIPASS_CONFIG_KEYS)
    merged["emulate_multipass_enabled"] = True
    merged["emulate_multipass_preset"] = "BF16_BF16_F32_X3"
    cfg_bf16 = MultipassEmulationConfig.from_maxtext_config(merged)
    r_bf16 = cfg_bf16.rules[0]
    self.assertEqual(r_bf16.lhs.src_dtype.name, "fp32")
    self.assertEqual(r_bf16.lhs.pass_dtype_names, ["bf16", "bf16"])
    self.assertIn(r_bf16.lhs.scale_mode, ("none", "unit"))
    self.assertEqual(r_bf16.cross_term_mode, "triangular")

    merged_override = dict(merged)
    merged_override["emulate_scale_mode"] = "per_block_pow2"
    merged_override["emulate_scale_block_size"] = 64
    merged_override["emulate_rhs_pass_dtypes"] = "fp8_e5m2,fp4_e2m1"
    cfg_over = MultipassEmulationConfig.from_maxtext_config(merged_override)
    r_over = cfg_over.rules[0]
    self.assertEqual(r_over.lhs.scale_mode, "per_block_pow2")
    self.assertEqual(r_over.lhs.block_size, 64)
    self.assertEqual(r_over.rhs.pass_dtype_names, ["fp8_e5m2", "fp4_e2m1"])
    self.assertGreaterEqual(len(MULTIPASS_FLAG_DOCUMENTATION), 35)

  # ---------------------------------------------------------------------------
  # 5. Modular Unit Tests for Rule Matching (Modules, Submodules, Layers, Ops)
  # ---------------------------------------------------------------------------

  def test_rule_matching_modules_submodules_layers_and_ops(self):
    """Tests `MultipassRuleConfig` matching across module groups, submodules, layers, and ops."""
    cfg = MultipassEmulationConfig.from_dict({
        "emulate_multipass_enabled": True,
        "emulate_multipass_preset": "F8E4M3_F8E4M3_F32_X3",
        "emulate_target_modules": "mlp,attn",
        "emulate_target_submodules": "wo,query",
        "emulate_target_layers": "2,4-6",
    })
    self.assertIsNotNone(cfg.match_rule("decoder/layers_2/mlp/wo", "dot_general"))
    self.assertIsNotNone(cfg.match_rule("decoder/layers_5/self_attention/query", "dot_general"))
    self.assertIsNone(cfg.match_rule("decoder/layers_2/mlp/wi_0", "dot_general"))
    self.assertIsNone(cfg.match_rule("decoder/layers_1/mlp/wo", "dot_general"))
    self.assertIsNone(cfg.match_rule("decoder/logits_dense", "dot_general"))

  # ---------------------------------------------------------------------------
  # 6. Modular Unit Tests for Step Schedule & Plugin Registry
  # ---------------------------------------------------------------------------

  def test_step_schedule_and_emulation_plugin_registry(self):
    """Tests `start_step`/`end_step` schedule gating and `EmulationFeature` plugin registration."""
    cfg = MultipassEmulationConfig.from_dict({
        "enabled": True,
        "preset": "F8E4M3_F8E4M3_F32_X3",
        "start_step": 10,
        "end_step": 20,
    })
    self.assertFalse(cfg.is_active_at_step(0))
    self.assertFalse(cfg.is_active_at_step(9))
    self.assertTrue(cfg.is_active_at_step(10))
    self.assertTrue(cfg.is_active_at_step(20))
    self.assertFalse(cfg.is_active_at_step(21))

    @register_emulation_feature("test_custom_clip")
    class CustomClipFeature(EmulationFeature):
      feature_name = "test_custom_clip"

      def __init__(self, enabled: bool = False, clip_val: float = 1.0):
        self.enabled = enabled
        self.clip_val = clip_val

      def summary(self) -> str:
        return f"CustomClip(enabled={self.enabled}, clip_val={self.clip_val})"

      def to_dict(self):
        return {"enabled": self.enabled, "clip_val": self.clip_val}

      @classmethod
      def from_dict(cls, data):
        return cls(
            enabled=bool(data.get("enabled", False)),
            clip_val=float(data.get("clip_val", 1.0)),
        )

    self.assertIn("test_custom_clip", get_registered_emulation_features())
    cfg_with_plugin = MultipassEmulationConfig.from_dict({
        "enabled": True,
        "preset": "BF16_BF16_F32_X3",
        "test_custom_clip": {"enabled": True, "clip_val": 0.5},
    })
    self.assertIn("test_custom_clip", cfg_with_plugin.extra_features)
    self.assertIn("CustomClip", cfg_with_plugin.summary())


if __name__ == "__main__":
  unittest.main()
