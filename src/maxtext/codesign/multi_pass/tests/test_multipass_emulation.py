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

import unittest

from maxtext.codesign.emulation_base import (
    EmulationFeature,
    get_registered_emulation_features,
    register_emulation_feature,
)
from maxtext.codesign.multi_pass.multipass_config import (
    DEFAULT_MULTIPASS_CONFIG_KEYS,
    DtypeSpec,
    MULTIPASS_FLAG_DOCUMENTATION,
    MULTIPASS_PRESETS,
    MultipassEmulationConfig,
    MultipassRuleConfig,
    OperandMultipassSpec,
    TARGET_PRESETS,
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

    spec_inferred = OperandMultipassSpec.from_any({"src_dtype": "bf16", "pass_dtypes": "fp8_e5m2,fp4_e2m1"})
    self.assertEqual(spec_inferred.num_passes, 2)
    self.assertEqual(spec_inferred.pass_dtype_names, ["fp8_e5m2", "fp4_e2m1"])

    spec_direct_list = OperandMultipassSpec(pass_dtypes=["fp8_e4m3", "fp8_e4m3", "fp4_e2m1"])
    self.assertEqual(spec_direct_list.num_passes, 3)
    self.assertEqual(spec_direct_list.pass_dtype_names, ["fp8_e4m3", "fp8_e4m3", "fp4_e2m1"])

    spec_direct_tuple = OperandMultipassSpec(pass_dtypes=("fp8_e4m3", "fp8_e5m2"))
    self.assertEqual(spec_direct_tuple.num_passes, 2)
    self.assertEqual(spec_direct_tuple.pass_dtype_names, ["fp8_e4m3", "fp8_e5m2"])

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
        resolve_cross_term_pairs(2, 2, "full", max_pass_degree=1),
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
    with self.assertRaises(ValueError):
      resolve_cross_term_pairs(2, 2, "custom")
    with self.assertRaises(ValueError):
      resolve_cross_term_pairs(2, 2, "custom", active_passes="5:5")
    with self.assertRaises(ValueError):
      resolve_cross_term_pairs(3, 3, "karatsuba")
    with self.assertRaises(ValueError):
      resolve_cross_term_pairs(1, 2, "karatsuba")

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

    cfg_int8_x2 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "INT8_INT4_INT32_X2"})
    self.assertEqual(cfg_int8_x2.rules[0].lhs.src_dtype.name, "int8")
    self.assertEqual(cfg_int8_x2.rules[0].rhs.src_dtype.name, "int4")
    self.assertEqual(cfg_int8_x2.rules[0].lhs.src_scale_mode, "per_channel_amax")

    cfg_x3_to_full = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "cross_term_mode": "full",
        }
    )
    self.assertEqual(
        cfg_x3_to_full.rules[0].effective_fwd_pairs,
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )

    cfg_rhs_only = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "emulate_lhs": False,
            "emulate_rhs": True,
        }
    )
    self.assertIsNone(cfg_rhs_only.rules[0].lhs)
    self.assertIsNotNone(cfg_rhs_only.rules[0].rhs)
    cfg_rhs_roundtrip = MultipassEmulationConfig.from_dict(cfg_rhs_only.to_dict())
    self.assertIsNone(cfg_rhs_roundtrip.rules[0].lhs)
    self.assertIsNotNone(cfg_rhs_roundtrip.rules[0].rhs)

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
    merged_override["emulate_block_size"] = 64
    merged_override["emulate_rhs_pass_dtypes"] = "fp8_e5m2,fp4_e2m1"
    cfg_over = MultipassEmulationConfig.from_maxtext_config(merged_override)
    r_over = cfg_over.rules[0]
    self.assertEqual(r_over.lhs.scale_mode, "per_block_pow2")
    self.assertEqual(r_over.lhs.block_size, 64)
    self.assertEqual(r_over.rhs.pass_dtype_names, ["fp8_e5m2", "fp4_e2m1"])

    cfg_x2_over = MultipassEmulationConfig.from_dict(
        {
            "emulate_multipass_enabled": True,
            "emulate_multipass_preset": "F8E4M3_F8E4M3_F32_X2",
            "emulate_pass_dtypes": "fp8_e5m2,fp8_e5m2",
            "emulate_num_passes": 2,
        }
    )
    self.assertEqual(cfg_x2_over.rules[0].lhs.pass_dtype_names, ["fp8_e5m2", "fp8_e5m2"])
    self.assertEqual(cfg_x2_over.rules[0].rhs.pass_dtype_names, ["fp8_e5m2", "fp8_e5m2"])
    self.assertGreaterEqual(len(MULTIPASS_FLAG_DOCUMENTATION), 35)

  # ---------------------------------------------------------------------------
  # 5. Modular Unit Tests for Rule Matching (Modules, Submodules, Layers, Ops)
  # ---------------------------------------------------------------------------

  def test_rule_matching_modules_submodules_layers_and_ops(self):
    """Tests `MultipassRuleConfig` matching across module groups, submodules, layers, and ops."""
    cfg_mlp_preset = MultipassEmulationConfig.from_dict(
        {
            "emulate_multipass_enabled": True,
            "emulate_multipass_preset": "F8E4M3_F8E4M3_F32_X3",
            "emulate_target_preset": "mlp",
        }
    )
    self.assertEqual(cfg_mlp_preset.rules[0].module_path, TARGET_PRESETS["mlp"]["module_path"])
    self.assertIsNone(cfg_mlp_preset.match_rule("decoder/layers_0/self_attention/wq_a", "dot_general"))

    cfg_qkv_preset = MultipassEmulationConfig.from_dict(
        {
            "emulate_multipass_enabled": True,
            "emulate_multipass_preset": "F8E4M3_F8E4M3_F32_X3",
            "emulate_target_preset": "qkv",
        }
    )
    for mla_sub in ("wq", "wq_a", "wq_b", "wkv", "wkv_a", "wkv_b"):
      self.assertIsNotNone(
          cfg_qkv_preset.match_rule(f"decoder/layers_0/self_attention/{mla_sub}", "dot_general")
      )

    cfg = MultipassEmulationConfig.from_dict(
        {
            "emulate_multipass_enabled": True,
            "emulate_multipass_preset": "F8E4M3_F8E4M3_F32_X3",
            "emulate_target_modules": "mlp,attn",
            "emulate_target_submodules": "wo,query",
            "emulate_layers": "2,4-6",
        }
    )
    self.assertIsNotNone(cfg.match_rule("decoder/layers_2/mlp/wo", "dot_general"))
    self.assertIsNotNone(cfg.match_rule("decoder/dense_layers_2/mlp/wo", "dot_general"))
    self.assertIsNotNone(cfg.match_rule("decoder/moe_layers_5/self_attention/query", "dot_general"))
    self.assertIsNotNone(cfg.match_rule("decoder/layers_5/self_attention/query", "dot_general"))
    self.assertIsNone(cfg.match_rule("decoder/layers_2/mlp/wi_0", "dot_general"))
    self.assertIsNone(cfg.match_rule("decoder/layers_1/mlp/wo", "dot_general"))
    self.assertIsNone(cfg.match_rule("decoder/logits_dense", "dot_general"))

    with self.assertRaises(ValueError):
      MultipassEmulationConfig.from_dict({"emulate_unknown_typo_flag": True})
    with self.assertRaises(ValueError):
      MultipassRuleConfig.from_dict({"unknown_rule_key": 123})
    with self.assertRaises(ValueError):
      MultipassEmulationConfig.from_dict(
          {
              "emulate_multipass_enabled": True,
              "emulate_multipass_preset": "F8E4M3_F8E4M3_F32_X3",
              "emulate_start_step": 50,
              "emulate_multipass_start_step": 10,
          }
      )

  # ---------------------------------------------------------------------------
  # 6. Modular Unit Tests for Step Schedule & Plugin Registry
  # ---------------------------------------------------------------------------

  def test_step_schedule_and_emulation_plugin_registry(self):
    """Tests `start_step`/`end_step` schedule gating and `EmulationFeature` plugin registration."""
    cfg = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "start_step": 10,
            "end_step": 20,
        }
    )
    self.assertFalse(cfg.is_active_at_step(0))
    self.assertFalse(cfg.is_active_at_step(9))
    self.assertTrue(cfg.is_active_at_step(10))
    self.assertTrue(cfg.is_active_at_step(20))
    self.assertFalse(cfg.is_active_at_step(21))

    @register_emulation_feature("test_custom_clip")
    class CustomClipFeature(EmulationFeature):
      """Custom clipping feature plugin for registry testing."""

      feature_name = "test_custom_clip"

      def __init__(self, enabled: bool = False, clip_val: float = 1.0):
        super().__init__(enabled=enabled)
        self.clip_val = float(clip_val)
        if self.clip_val <= 0:
          raise ValueError("clip_val must be > 0")

      def summary(self) -> str:
        return f"CustomClip(enabled={self.enabled}, clip_val={self.clip_val})"

      def to_dict(self):
        return {"enabled": self.enabled, "clip_val": self.clip_val}

    self.assertIn("test_custom_clip", get_registered_emulation_features())
    self.assertIs(
        get_registered_emulation_features()["test_custom_clip"],
        CustomClipFeature,
    )
    cfg_with_plugin = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "BF16_BF16_F32_X3",
            "test_custom_clip": {"enabled": True, "clip_val": 0.5},
        }
    )
    self.assertIn("test_custom_clip", cfg_with_plugin.extra_features)
    self.assertIn("CustomClip", cfg_with_plugin.summary())
    with self.assertRaises(ValueError):
      MultipassEmulationConfig.from_dict(
          {
              "enabled": True,
              "preset": "BF16_BF16_F32_X3",
              "test_custom_clip": {"enabled": True, "clip_val": -1.0},
          }
      )


if __name__ == "__main__":
  unittest.main()