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

"""Unit tests for Qwix multi-pass quantization, rule scoping, and GMM support in MaxText."""

import json
from types import SimpleNamespace
import unittest
from absl.testing import absltest
import jax
import jax.numpy as jnp
from maxtext.layers import quantizations
import qwix


class QwixQuantizationTest(unittest.TestCase):
  """Tests Qwix rule construction, multipass parsing, scoping, and role filtering."""

  def _make_config(self, **overrides):
    defaults = {
        "quantization": "fp8_full",
        "quantization_calibration_method": "absmax",
        "weight_quantization_calibration_method": "absmax",
        "act_quantization_calibration_method": "absmax",
        "bwd_quantization_calibration_method": "absmax",
        "drhs_grad_quantization_calibration_method": "",
        "quantize_router_proj": True,
        "quantize_logits_proj": False,
        "quantize_mtp": False,
        "sa_qproj_bwd_bf16": False,
        "weight_sparsity_n": 0,
        "weight_sparsity_m": 0,
        "weight_sparsity_update_step": 10,
        "weight_sparsity_start_step": 50,
        "qwix_bwd_weight_grad_tile_count": 1,
        "quantization_multipass_mode": "",
        "quantization_dlhs_multipass_mode": "",
        "quantization_drhs_multipass_mode": "",
        "quantization_exclude_module_path": "",
        "quantization_start_step": 0,
        "quantization_end_step": -1,
        "quantization_target_scan_indices": "",
        "quantization_rules_json": "",
        "quantization_target_roles": ["actor", "student"],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

  def test_parse_scan_indices(self):
    self.assertIsNone(quantizations.parse_scan_indices(""))
    self.assertEqual(quantizations.parse_scan_indices("0, 2,5-7"), (0, 2, 5, 6, 7))
    self.assertEqual(quantizations.parse_scan_indices([1, 3]), (1, 3))

  def test_parse_multipass_mode(self):
    self.assertIsNone(quantizations.parse_multipass_mode(""))
    self.assertEqual(quantizations.parse_multipass_mode("2xint8"), "2xint8")
    spec_json = json.dumps({"lhs": {"num_passes": 3, "pass_dtypes": ["fp8_e4m3", "fp8_e4m3", "fp4_e2m1"]}})
    cfg = quantizations.parse_multipass_mode(spec_json)
    self.assertIsInstance(cfg, qwix.MultipassConfig)
    self.assertLen(cfg.lhs.pass_dtypes, 3)

  def test_get_quantization_rule_scoping_and_multipass(self):
    cfg = self._make_config(
        quantization="fp8_full",
        quantization_multipass_mode="2xint4",
        quantization_dlhs_multipass_mode="2xint8",
        quantization_drhs_multipass_mode="three_pass_fp8",
        quantization_exclude_module_path=".*logits_dense.*",
        quantization_start_step=10,
        quantization_end_step=100,
        quantization_target_scan_indices="1,3-4",
    )
    rules = quantizations.get_quantization_rule(cfg)
    self.assertLen(rules, 1)
    r = rules[0]
    self.assertEqual(r.exclude_module_path, ".*logits_dense.*")
    self.assertEqual(r.start_step, 10)
    self.assertEqual(r.end_step, 100)
    self.assertEqual(r.target_scan_indices, (1, 3, 4))
    self.assertEqual(r.additional_qt_config["multipass_mode"], "2xint4")
    self.assertEqual(r.additional_qt_config["dlhs_multipass_mode"], "2xint8")
    self.assertEqual(r.additional_qt_config["drhs_multipass_mode"], "three_pass_fp8")

  def test_custom_json_rules_and_role_filtering(self):
    rules_json = json.dumps(
        [
            {
                "module_path": ".*mlp.*",
                "exclude_module_path": ".*router.*",
                "weight_qtype": "int8",
                "act_qtype": "int8",
                "multipass_mode": "2xint8",
                "start_step": 5,
                "target_scan_indices": [0, 2],
            }
        ]
    )
    cfg = self._make_config(
        quantization="qwix",
        quantization_rules_json=rules_json,
        quantization_target_roles=["actor", "rollout"],
    )
    rules = quantizations.get_quantization_rule(cfg)
    self.assertLen(rules, 1)
    self.assertEqual(rules[0].module_path, ".*mlp.*")
    self.assertEqual(rules[0].exclude_module_path, ".*router.*")
    self.assertEqual(rules[0].weight_qtype, jnp.int8)
    self.assertEqual(rules[0].act_qtype, jnp.int8)
    self.assertEqual(rules[0].start_step, 5)
    self.assertEqual(rules[0].target_scan_indices, (0, 2))
    self.assertEqual(rules[0].additional_qt_config["multipass_mode"], "2xint8")

    dummy_model = object()
    # Role 'reference' is not in ['actor', 'rollout'] -> returns unmodified model
    self.assertIs(quantizations.maybe_quantize_model(dummy_model, cfg, role="reference"), dummy_model)

  def test_gmm_qwix_multipass_and_step_gating(self):
    from maxtext.kernels.megablox import ops as mblx_ops

    lhs = jnp.full((4, 8), 1.25, dtype=jnp.float32)
    rhs = jnp.ones((2, 8, 4), dtype=jnp.float32)
    group_sizes = jnp.array([2, 2], dtype=jnp.int32)
    tiling = (4, 8, 4, 4, 4, 8, 4, 8, 4)
    rule_1pass = qwix.QtRule(
        start_step=5,
        additional_qt_config={
            "multipass_mode": "1xe3m0",
            "dlhs_multipass_mode": "2xe3m0",
            "drhs_multipass_mode": "2xe3m0",
        },
    )
    rule_2pass = qwix.QtRule(
        start_step=5,
        additional_qt_config={"multipass_mode": "2xe3m0"},
    )
    with qwix.step_context(2):
      # Inactive rule -> unquantized GMM (8 * 1.25 = 10.0)
      out_inactive = mblx_ops.gmm(
          lhs,
          rhs,
          group_sizes,
          preferred_element_type=jnp.float32,
          tiling=tiling,
          interpret=True,
          qwix_rule=rule_1pass,
      )
      self.assertTrue(jnp.allclose(out_inactive, 10.0, atol=1e-3))

    with qwix.step_context(5):
      # Active 1-pass e3m0 rule -> rounds 1.25 to 1.0 (8 * 1.0 = 8.0)
      out_1p, vjp_fn = jax.vjp(
          lambda x, w: mblx_ops.gmm(
              x,
              w,
              group_sizes,
              preferred_element_type=jnp.float32,
              tiling=tiling,
              interpret=True,
              qwix_rule=rule_1pass,
          ),
          lhs,
          rhs,
      )
      self.assertTrue(jnp.allclose(out_1p, 8.0, atol=1e-3))
      dlhs, drhs = vjp_fn(jnp.ones_like(out_1p))
      self.assertEqual(dlhs.shape, lhs.shape)
      self.assertEqual(drhs.shape, rhs.shape)

      # Active 2-pass e3m0 rule -> recovers 0.25 residual in Pass 1 (8 * 1.25 = 10.0)
      out_2p = mblx_ops.gmm(
          lhs,
          rhs,
          group_sizes,
          preferred_element_type=jnp.float32,
          tiling=tiling,
          interpret=True,
          qwix_rule=rule_2pass,
      )
      self.assertTrue(jnp.allclose(out_2p, 10.0, atol=1e-3))

  def assertLen(self, container, expected_len):
    self.assertEqual(len(container), expected_len)


if __name__ == "__main__":
  absltest.main()
