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

import tempfile
import unittest

try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None  # type: ignore

try:
  import jax
  import jax.numpy as jnp

  HAS_JAX = True
except ImportError:
  jax = None  # type: ignore
  jnp = np  # type: ignore
  HAS_JAX = False

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
)
from maxtext.codesign.multi_pass.checkpoint_utils import (
    apply_multipass_to_pytree,
    evaluate_multipass_error,
    inspect_model_or_pytree_multipass,
    load_multipass_metadata,
    save_multipass_metadata,
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
from maxtext.codesign.multi_pass.multipass_kernel import (
    _numpy_dot_general,
    decompose_tensor_into_passes,
    multipass_dot_general,
    multipass_einsum,
    multipass_gmm_operands,
    quantize_to_dtype_spec,
    reconstruct_multipass_tensor,
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


  # ---------------------------------------------------------------------------
  # 7. Modular Unit Tests for Elementary Quantization, Scaling & Kernel Ops
  # ---------------------------------------------------------------------------

  @unittest.skipIf(np is None, "NumPy is required for array quantization tests")
  def test_quantize_to_dtype_spec_rounding_and_subnormals(self):
    """Tests `quantize_to_dtype_spec` for RNE, RTZ, stochastic rounding, and subnormal flushing."""
    x = jnp.asarray([0.0, 1.25, -3.75, 500.0, 1e-4], dtype=np.float32)
    q_rne = quantize_to_dtype_spec(x, "fp8_e4m3", rounding_mode="rne", subnormals=True)
    q_rtz = quantize_to_dtype_spec(x, "fp8_e4m3", rounding_mode="rtz", subnormals=True)
    self.assertAlmostEqual(float(np.asarray(q_rne)[3]), 448.0)
    self.assertLessEqual(abs(float(np.asarray(q_rtz)[2])), abs(float(np.asarray(q_rne)[2])))

    sub_val = jnp.asarray([0.00390625], dtype=np.float32)
    q_sub_kept = quantize_to_dtype_spec(sub_val, "fp8_e4m3", subnormals=True)
    q_sub_flushed = quantize_to_dtype_spec(sub_val, "fp8_e4m3", subnormals=False)
    self.assertAlmostEqual(float(np.asarray(q_sub_kept)[0]), 0.00390625)
    self.assertAlmostEqual(float(np.asarray(q_sub_flushed)[0]), 0.0)

  @unittest.skipIf(np is None, "NumPy is required for array decomposition tests")
  def test_decompose_and_reconstruct_scaling_modes(self):
    """Tests `decompose_tensor_into_passes` across unit, per_tensor, per_channel, and per_block_pow2 scaling."""
    rng = np.random.default_rng(17)
    x = jnp.asarray(rng.standard_normal((8, 32), dtype=np.float32))
    for scale_mode in ("unit", "per_tensor_amax", "per_channel_amax", "per_block_amax", "per_block_pow2"):
      spec = OperandMultipassSpec.from_any(
          {
              "src_dtype": "fp32",
              "num_passes": 2,
              "pass_dtypes": ["fp8_e5m2", "fp8_e4m3"],
              "scale_mode": scale_mode,
              "block_size": 16,
          }
      )
      slices = decompose_tensor_into_passes(x, spec, contracting_axes=(1,))
      self.assertEqual(len(slices), 2)
      recon = reconstruct_multipass_tensor(x, spec, contracting_axes=(1,))
      self.assertEqual(recon.shape, x.shape)
      rel_err = float(np.linalg.norm(np.asarray(recon - x)) / np.linalg.norm(np.asarray(x)))
      self.assertLess(rel_err, 5e-2, f"Failed error bound for scale_mode={scale_mode}")

  @unittest.skipIf(np is None, "NumPy is required for array tests")
  def test_int8_bit_shift_and_karatsuba_exactness(self):
    """Tests exact `int8 -> [int4, uint4]` decomposition and 3-GEMM Karatsuba vs 4-GEMM full product."""
    rng = np.random.default_rng(7)
    a_int8 = jnp.asarray(rng.integers(-128, 127, size=(16, 32)), dtype=jnp.float32)
    b_int8 = jnp.asarray(rng.integers(-128, 127, size=(32, 16)), dtype=jnp.float32)
    exact_prod = np.matmul(
        np.asarray(a_int8, dtype=np.float32),
        np.asarray(b_int8, dtype=np.float32),
    )

    spec_int = OperandMultipassSpec.from_any("int8->[int4,uint4]@scale=bit_shift")
    slices_a = decompose_tensor_into_passes(a_int8, spec_int)
    self.assertEqual(len(slices_a), 2)
    recon_a = slices_a[0].dequantized + slices_a[1].dequantized
    np.testing.assert_allclose(np.asarray(recon_a), np.asarray(a_int8), atol=0.0)

    cfg_x4_unscaled = MultipassEmulationConfig.from_dict(
        {"enabled": True, "preset": "INT8_INT8_INT32_X4", "src_scale_mode": "none"}
    )
    cfg_x3_unscaled = MultipassEmulationConfig.from_dict(
        {"enabled": True, "preset": "INT8_INT8_INT32_X3", "src_scale_mode": "none"}
    )
    dnums = (((1,), (0,)), ((), ()))
    out_x4_int = multipass_dot_general(a_int8, b_int8, dnums, cfg_x4_unscaled.rules[0])
    out_x3_int = multipass_dot_general(a_int8, b_int8, dnums, cfg_x3_unscaled.rules[0])
    np.testing.assert_allclose(np.asarray(out_x4_int), np.asarray(exact_prod), atol=0.0)
    np.testing.assert_allclose(np.asarray(out_x3_int), np.asarray(exact_prod), atol=0.0)

    # Test INT8_* presets on small-magnitude neural network float tensors (e.g. N(0, 0.05^2))
    a_flt = jnp.asarray(rng.standard_normal((16, 32), dtype=np.float32) * 0.05)
    b_flt = jnp.asarray(rng.standard_normal((32, 16), dtype=np.float32) * 0.05)
    ref_flt = np.matmul(np.asarray(a_flt), np.asarray(b_flt))
    cfg_x4 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "INT8_INT8_INT32_X4"})
    cfg_x3 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "INT8_INT8_INT32_X3"})
    cfg_x2 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "INT8_INT4_INT32_X2"})
    out_x4_flt = np.asarray(multipass_dot_general(a_flt, b_flt, dnums, cfg_x4.rules[0]))
    out_x3_flt = np.asarray(multipass_dot_general(a_flt, b_flt, dnums, cfg_x3.rules[0]))
    out_x2_flt = np.asarray(multipass_dot_general(a_flt, b_flt, dnums, cfg_x2.rules[0]))
    np.testing.assert_allclose(out_x4_flt, out_x3_flt, rtol=1e-5, atol=1e-6)
    rel_err_x4 = float(np.linalg.norm(out_x4_flt - ref_flt) / np.linalg.norm(ref_flt))
    rel_err_x2 = float(np.linalg.norm(out_x2_flt - ref_flt) / np.linalg.norm(ref_flt))
    self.assertLess(rel_err_x4, 0.03)
    self.assertLess(rel_err_x2, 0.25)

  @unittest.skipIf(np is None, "NumPy is required for array tests")
  def test_cross_term_modes_x1_x2_x3_x4_and_custom(self):
    """Tests GEMM error progression across X1, X2, X3, X4 and custom cross-term pairs."""
    rng = np.random.default_rng(99)
    a = jnp.asarray(rng.standard_normal((16, 32), dtype=np.float32))
    b = jnp.asarray(rng.standard_normal((32, 16), dtype=np.float32))
    a_bf16 = quantize_to_dtype_spec(a, "bf16")
    b_bf16 = quantize_to_dtype_spec(b, "bf16")
    ref = np.matmul(np.asarray(a_bf16), np.asarray(b_bf16))
    dnums = (((1,), (0,)), ((), ()))

    cfg_x1 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "F8E4M3_F8E4M3_F32_X1"})
    cfg_x2 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "F8E4M3_F8E4M3_F32_X2"})
    cfg_x3 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "F8E4M3_F8E4M3_F32_X3"})
    cfg_x4 = MultipassEmulationConfig.from_dict({"enabled": True, "preset": "F8E4M3_F8E4M3_F32_X4"})

    out_x1 = np.asarray(multipass_dot_general(a, b, dnums, cfg_x1.rules[0]))
    out_x2 = np.asarray(multipass_dot_general(a, b, dnums, cfg_x2.rules[0]))
    out_x3 = np.asarray(multipass_dot_general(a, b, dnums, cfg_x3.rules[0]))
    out_x4 = np.asarray(multipass_dot_general(a, b, dnums, cfg_x4.rules[0]))

    err_x1 = float(np.linalg.norm(out_x1 - ref) / np.linalg.norm(ref))
    err_x2 = float(np.linalg.norm(out_x2 - ref) / np.linalg.norm(ref))
    err_x3 = float(np.linalg.norm(out_x3 - ref) / np.linalg.norm(ref))
    err_x4 = float(np.linalg.norm(out_x4 - ref) / np.linalg.norm(ref))

    self.assertLess(err_x2, err_x1)
    self.assertLess(err_x3, err_x2)
    self.assertLessEqual(err_x4, err_x3)
    self.assertGreater(float(np.max(np.abs(out_x3 - out_x4))), 0.0)

    cfg_custom = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "src_dtype": "bf16",
            "num_passes": 2,
            "pass_dtypes": ["fp8_e4m3", "fp8_e4m3"],
            "scale_mode": "per_tensor_pow2",
            "cross_term_mode": "custom",
            "active_passes": "0:0,0:1,1:0",
        }
    )
    out_custom = np.asarray(multipass_dot_general(a, b, dnums, cfg_custom.rules[0]))
    np.testing.assert_allclose(out_custom, out_x3, atol=1e-6)

  @unittest.skipUnless(HAS_JAX, "JAX is required for custom_vjp gradient tests")
  def test_custom_vjp_backward_modes(self):
    """Tests `jax.grad` through `multipass_dot_general` for `multipass_bwd` vs `ste`."""
    rng = np.random.default_rng(42)
    a = jnp.asarray(rng.standard_normal((8, 16), dtype=np.float32))
    b = jnp.asarray(rng.standard_normal((16, 8), dtype=np.float32))
    dnums = (((1,), (0,)), ((), ()))

    cfg_bwd = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "bwd_mode": "multipass_bwd",
            "emulate_dlhs": True,
            "emulate_drhs": True,
        }
    )
    cfg_ste = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "bwd_mode": "ste",
        }
    )

    def loss_bwd(x, w):
      y = multipass_dot_general(x, w, dnums, cfg_bwd.rules[0])
      return jnp.sum(y**2)

    def loss_ste(x, w):
      y = multipass_dot_general(x, w, dnums, cfg_ste.rules[0])
      return jnp.sum(y**2)

    da_bwd, db_bwd = jax.grad(loss_bwd, argnums=(0, 1))(a, b)
    da_ste, db_ste = jax.grad(loss_ste, argnums=(0, 1))(a, b)

    self.assertEqual(da_bwd.shape, a.shape)
    self.assertEqual(db_bwd.shape, b.shape)
    self.assertGreater(float(jnp.max(jnp.abs(da_bwd - da_ste))), 0.0)
    self.assertGreater(float(jnp.max(jnp.abs(db_bwd - db_ste))), 0.0)
    np.testing.assert_allclose(np.asarray(da_bwd), np.asarray(da_ste), rtol=5e-2, atol=1e-1)

  # ---------------------------------------------------------------------------
  # 10. Modular Unit Tests for Per-Pass Error/SNR & Checkpoint Utilities
  # ---------------------------------------------------------------------------

  @unittest.skipIf(np is None, "NumPy is required for array tests")
  def test_monotonic_error_reduction_homogeneous_and_heterogeneous(self):
    """Tests strict monotonic error decrease across passes for homogeneous and heterogeneous schedules."""
    rng = np.random.default_rng(123)
    x = jnp.asarray(rng.standard_normal((32, 64), dtype=np.float32))

    # 1. Homogeneous 3-pass FP32 decomposition: fp32 -> [fp8_e4m3] * 3
    spec_3 = OperandMultipassSpec.from_any("fp32->[fp8_e4m3,fp8_e4m3,fp8_e4m3]@scale=per_tensor_pow2")
    report = evaluate_multipass_error(x, spec_3, contracting_axes=(1,))
    errs = report["per_pass_relative_l2_error"]
    self.assertEqual(len(errs), 3)
    self.assertLess(errs[1], errs[0] * 0.15)
    self.assertLess(errs[2], errs[1] * 0.15)

    # 1b. BF16 -> [fp8_e4m3, fp8_e4m3] (2 passes of 4-bit significand recover 8-bit BF16 significand)
    spec_bf16_to_fp8 = OperandMultipassSpec.from_any("bf16->[fp8_e4m3,fp8_e4m3]@scale=per_tensor_pow2")
    rep_bf16_to_fp8 = evaluate_multipass_error(x, spec_bf16_to_fp8, contracting_axes=(1,))
    self.assertGreater(rep_bf16_to_fp8["per_pass_relative_l2_error"][0], 1e-2)
    self.assertLessEqual(rep_bf16_to_fp8["per_pass_relative_l2_error"][1], 1e-6)

    # 2. Heterogeneous passes: fp32 -> [fp8_e5m2, fp8_e4m3, fp4_e2m1]
    spec_het = OperandMultipassSpec.from_any("fp32->[fp8_e5m2,fp8_e4m3,fp4_e2m1]@scale=per_block_pow2:block=16")
    rep_het = evaluate_multipass_error(x, spec_het, contracting_axes=(1,))
    het_errs = rep_het["per_pass_relative_l2_error"]
    self.assertLess(het_errs[1], het_errs[0])
    self.assertLess(het_errs[2], het_errs[1])

    # 3. Henry et al. FP32 -> [bf16, bf16] with scale=unit
    spec_bf16 = OperandMultipassSpec.from_any("fp32->[bf16,bf16]@scale=unit")
    rep_bf16 = evaluate_multipass_error(x, spec_bf16, contracting_axes=(1,))
    self.assertLess(rep_bf16["per_pass_relative_l2_error"][0], 5e-3)
    self.assertLess(rep_bf16["per_pass_relative_l2_error"][1], 5e-5)

  @unittest.skipIf(np is None, "NumPy is required for array tests")
  def test_multipass_einsum_gmm_and_checkpoint_utils(self):
    """Tests `multipass_einsum`, `multipass_gmm_operands`, and checkpoint PyTree utilities."""
    rng = np.random.default_rng(21)
    q = jnp.asarray(rng.standard_normal((2, 4, 8, 16), dtype=np.float32))
    k = jnp.asarray(rng.standard_normal((2, 4, 8, 16), dtype=np.float32))
    cfg = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X3",
            "op_names": ["einsum", "dot_general", "gmm"],
        }
    )
    out = multipass_einsum("bthd,bshd->bhts", q, k, cfg.rules[0])
    ref = jnp.einsum("bthd,bshd->bhts", q, k)
    self.assertEqual(out.shape, ref.shape)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=3e-2, atol=3e-2)

    moe_x = jnp.asarray(rng.standard_normal((8, 16), dtype=np.float32))
    moe_w = jnp.asarray(rng.standard_normal((4, 16, 32), dtype=np.float32))
    q_moe_x, q_moe_w = multipass_gmm_operands(moe_x, moe_w, cfg.rules[0])
    self.assertEqual(q_moe_x.shape, moe_x.shape)
    self.assertEqual(q_moe_w.shape, moe_w.shape)

    pytree = {
        "layers_0": {
            "mlp": {
                "wi": rng.standard_normal((16, 32), dtype=np.float32),
            }
        }
    }
    recon_tree, n_mod = apply_multipass_to_pytree(pytree, cfg)
    self.assertEqual(n_mod, 1)
    self.assertEqual(recon_tree["layers_0"]["mlp"]["wi"].shape, (16, 32))

    insp = inspect_model_or_pytree_multipass(pytree, cfg)
    self.assertEqual(insp["targeted_parameters"], 16 * 32)
    self.assertGreaterEqual(insp["mean_targeted_relative_l2_error"], 0.0)
    self.assertLess(insp["mean_targeted_relative_l2_error"], 1e-2)

    with tempfile.TemporaryDirectory() as tmpdir:
      save_multipass_metadata(tmpdir, cfg)
      loaded = load_multipass_metadata(tmpdir)
      self.assertIsNotNone(loaded)
      self.assertTrue(loaded.enabled)
      self.assertEqual(loaded.rules[0].cross_term_mode, "triangular")

    self.assertGreaterEqual(len(MULTIPASS_PRESETS), 11)
    self.assertGreaterEqual(len(MULTIPASS_FLAG_DOCUMENTATION), 35)
    install_emulation_hooks()

  # ---------------------------------------------------------------------------
  # 11. Modular Unit Tests for Dynamic Step & Layer Gating (`interception.py`)
  # ---------------------------------------------------------------------------

  @unittest.skipIf(np is None, "NumPy is required for array tests")
  def test_dynamic_step_and_layer_gating(self):
    """Tests dynamic `active_cond` step and layer gating inside JIT-compiled functions."""
    rng = np.random.default_rng(11)
    a = jnp.asarray(rng.standard_normal((8, 16), dtype=np.float32))
    b = jnp.asarray(rng.standard_normal((16, 8), dtype=np.float32))
    dnums = (((1,), (0,)), ((), ()))

    cfg = MultipassEmulationConfig.from_dict(
        {
            "enabled": True,
            "preset": "F8E4M3_F8E4M3_F32_X1",
            "layers": "2",
            "start_step": 5,
            "end_step": 10,
        }
    )
    rule = cfg.rules[0]
    dense_ref = _numpy_dot_general(a, b, dnums)
    mp_ref = multipass_dot_general(a, b, dnums, rule)

    def step_and_layer_fn(x, w, step_idx, layer_idx):
      with emulation_jax_step_context(step_idx):
        with emulation_jax_layer_context(layer_idx):
          cond = _compute_active_cond(cfg, rule)
          return multipass_dot_general(x, w, dnums, rule, active_cond=cond)

    if HAS_JAX:
      step_and_layer_fn = jax.jit(step_and_layer_fn)

    out_step2 = step_and_layer_fn(a, b, jnp.int32(2), jnp.int32(2))
    np.testing.assert_allclose(np.asarray(out_step2), np.asarray(dense_ref), rtol=1e-5, atol=1e-5)

    out_layer1 = step_and_layer_fn(a, b, jnp.int32(6), jnp.int32(1))
    np.testing.assert_allclose(np.asarray(out_layer1), np.asarray(dense_ref), rtol=1e-5, atol=1e-5)

    out_active = step_and_layer_fn(a, b, jnp.int32(6), jnp.int32(2))
    np.testing.assert_allclose(np.asarray(out_active), np.asarray(mp_ref), rtol=1e-5, atol=1e-5)

  # ---------------------------------------------------------------------------
  # 12. Scheduled Nightly TPU E2E SFT Smoke Test Hook
  # ---------------------------------------------------------------------------

  def test_tpu_tiny_sft_smoke_custom_multipass(self):
    """End-to-end TPU smoke test: 4-layer tiny debug model SFT (baseline vs custom 3-pass heterogeneous)."""
    if not HAS_JAX:
      self.skipTest("JAX is not installed in this environment")
    if not any(d.platform == "tpu" for d in jax.devices()):
      self.skipTest("Requires TPU hardware (scheduled nightly via sft_multipass_emulation_smoke_test.py)")

    from maxtext.codesign.multi_pass.tests.sft_multipass_emulation_smoke_test import (  # pylint: disable=import-outside-toplevel
        SftMultipassEmulationSmokeTest,
    )

    smoke = SftMultipassEmulationSmokeTest("test_tiny_sft_baseline_vs_custom_heterogeneous_multipass_at_step_k")
    smoke.setUp()
    try:
      smoke.test_tiny_sft_baseline_vs_custom_heterogeneous_multipass_at_step_k()
    finally:
      smoke.tearDown()


if __name__ == "__main__":
  unittest.main()