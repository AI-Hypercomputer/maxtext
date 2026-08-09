# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""M2/M5/M6: Gemma-4 E2B/E4B multimodal STATIC contract gate (configs/types.py).

Asserts the config validator ACCEPTS a valid E2B multimodal config and HARD-FAILS on each
contract violation: unknown PLE mode (M2 enum), bidirectional image attention (causal M6),
fused_qkv / fused_mlp (M5), and clipped-linears disabled. These encode the semantics verified
against pinned HF Transformers 5.9.0 and must fail closed rather than silently degrade.
"""
import os
import unittest

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR


def _init(overrides):
  base_yml = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
  argv = [
      "test",
      base_yml,
      "model_name=gemma4-e2b",
      "use_multimodal=true",
      "override_model_config=true",
      "tokenizer_path=/tmp/unused",
      "per_device_batch_size=1",
      "scan_layers=false",
      "enable_checkpointing=false",
      "run_name=gate_test",
      "steps=1",
      "skip_jax_distributed_system=true",
  ] + overrides
  return pyconfig.initialize(argv)


class Gemma4MmContractGateTest(unittest.TestCase):

  def test_valid_e2b_mm_config_builds(self):
    # Clipped linears on, identity PLE, causal image spans, fused off -> the HF-faithful contract.
    cfg = _init(["use_clipped_linears_for_vit=true"])
    self.assertEqual(cfg.model_name, "gemma4-e2b")
    self.assertTrue(cfg.use_multimodal)
    self.assertEqual(str(cfg.ple_pad_mode), "identity")

  def test_unknown_ple_pad_mode_hard_fails(self):
    with self.assertRaises(Exception):
      _init(["use_clipped_linears_for_vit=true", "ple_pad_mode=bogus"])

  def test_both_ple_pad_mode_is_valid_ablation_enum(self):
    cfg = _init(["use_clipped_linears_for_vit=true", "ple_pad_mode=both"])
    self.assertEqual(str(cfg.ple_pad_mode), "both")

  def test_bidirectional_image_attn_hard_fails(self):
    # Gemma-4 E2B/E4B image spans are causal; bidirectional is a Gemma-3 / 26B / 31B feature.
    with self.assertRaises(Exception):
      _init(["use_clipped_linears_for_vit=true", "use_bidirectional_image_attn=true"])

  def test_fused_qkv_hard_fails(self):
    with self.assertRaises(Exception):
      _init(["use_clipped_linears_for_vit=true", "fused_qkv=true"])

  def test_fused_mlp_hard_fails(self):
    with self.assertRaises(Exception):
      _init(["use_clipped_linears_for_vit=true", "fused_mlp=true"])

  def test_clipped_linears_off_under_mm_hard_fails(self):
    with self.assertRaises(Exception):
      _init(["use_clipped_linears_for_vit=false"])

  def test_packing_hard_fails(self):
    # Packed multimodal is unsupported for E2B/E4B in any mode (cross-doc image attention risk).
    with self.assertRaises(Exception):
      _init(["use_clipped_linears_for_vit=true", "packing=true"])


if __name__ == "__main__":
  unittest.main()
