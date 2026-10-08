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

"""Unit tests for Group 5: Kimi-K3 Config & Parameter Mapping Coverage."""

import json
import os
import unittest
import numpy as np
import yaml

from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING, HOOK_FNS
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference

# 0-indexed MLA (full attention) layers of the 93-layer Kimi-K3 model.
_FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92]


class TestKimiK3Group5Config(unittest.TestCase):
  """Tests validating Kimi-K3 configuration and parameter mapping coverage."""

  def setUp(self):
    super().setUp()
    self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    self.config_path = os.path.join(self.base_dir, "src/maxtext/configs/models/kimi-k3.yml")
    self.ref_dir = os.path.join(self.base_dir, "kimi-k3-hf-reference")
    self.index_file = os.path.join(self.ref_dir, "model.safetensors.index.json")
    self.hf_config_file = os.path.join(self.ref_dir, "config.json")

    # Load HF text config
    if os.path.exists(self.hf_config_file):
      with open(self.hf_config_file, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)
      self.text_config = full_cfg["text_config"]
    else:
      self.text_config = {
          "hidden_size": 7168,
          "num_hidden_layers": 93,
          "num_attention_heads": 96,
          "num_key_value_heads": 96,
          "q_lora_rank": 1536,
          "kv_lora_rank": 512,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128,
          "num_experts": 896,
          "num_experts_per_tok": 16,
          "routed_expert_hidden_size": 3584,
          "moe_intermediate_size": 3072,
          "shared_intermediate_size": 6144,
          "rms_norm_eps": 1e-5,
          "first_k_dense_replace": 1,
          "attn_res_block_size": 12,
          "full_attn_layers": list(_FULL_ATTN_LAYERS),
      }

  # ==========================================================================
  # 1. Configuration Verification
  # ==========================================================================
  def test_kimi_k3_yaml_config_hyperparameters(self):
    """Verifies that kimi_k3.yml contains all exact architectural hyperparameters."""
    self.assertTrue(os.path.exists(self.config_path), f"Config file not found: {self.config_path}")

    with open(self.config_path, "r", encoding="utf-8") as f:
      cfg = yaml.safe_load(f)

    # Dimensionality & Layer counts
    self.assertEqual(cfg["base_emb_dim"], 7168)
    self.assertEqual(cfg["base_num_decoder_layers"], 93)
    self.assertEqual(cfg["base_num_query_heads"], 96)
    self.assertEqual(cfg["base_num_kv_heads"], 96)
    self.assertEqual(cfg["head_dim"], 128)
    self.assertEqual(cfg["vocab_size"], 163840)
    self.assertEqual(float(cfg["normalization_layer_epsilon"]), 1.0e-5)

    # MLA & RoPE ranks
    self.assertEqual(cfg["q_lora_rank"], 1536)
    self.assertEqual(cfg["kv_lora_rank"], 512)
    self.assertEqual(cfg["qk_nope_head_dim"], 128)
    self.assertEqual(cfg["qk_rope_head_dim"], 64)
    self.assertEqual(cfg["v_head_dim"], 128)

    # MoE and MLP settings
    self.assertEqual(cfg["num_experts"], 896)
    self.assertEqual(cfg["num_experts_per_tok"], 16)
    self.assertEqual(cfg["routed_expert_hidden_size"], 3584)
    self.assertEqual(cfg["base_moe_mlp_dim"], 3072)
    self.assertEqual(cfg["moe_intermediate_size"], 3072)
    self.assertEqual(cfg["shared_intermediate_size"], 6144)
    self.assertEqual(cfg["first_num_dense_layers"], 1)

    # RoPE & Sequence length
    self.assertEqual(float(cfg["rope_theta"]), 500000.0)
    self.assertEqual(float(cfg["rope_max_timescale"]), 500000.0)
    self.assertEqual(cfg["max_target_length"], 1048576)
    self.assertEqual(cfg["max_position_embeddings"], 1048576)

    # Hybrid Cycle Specifications (23 cycles of 4 layers: 3 KDA + 1 MLA, Layer 92 MLA)
    self.assertEqual(cfg["inhomogeneous_layer_cycle_interval"], 4)
    self.assertEqual(cfg["num_cycles"], 23)
    self.assertEqual(cfg["kda_layers_per_cycle"], 3)
    self.assertEqual(cfg["mla_layers_per_cycle"], 1)
    expected_full_attn_layers = list(_FULL_ATTN_LAYERS)
    self.assertEqual(cfg["full_attn_layers"], expected_full_attn_layers)
    self.assertEqual(len(cfg["full_attn_layers"]), 24)

    # AttnRes
    self.assertEqual(cfg["attn_res_block_size"], 12)

  # ==========================================================================
  # 2. Parameter Mapping Coverage (100% of 96-shard keys in index.json)
  # ==========================================================================
  @requires_kimi_k3_reference
  def test_kimi_k3_param_mapping_100_percent_coverage(self):
    """Verifies 100% parameter key mapping coverage against model.safetensors.index.json."""
    mapping = PARAM_MAPPING["kimi-k3"](self.text_config, None, scan_layers=False)
    hooks = HOOK_FNS["kimi-k3"](self.text_config, None, scan_layers=False, saving_to_hf=False)
    self.assertIn("params-decoder-layers_0-self_attention-A_log", hooks)

    # Check top-level embeddings & norms
    self.assertIn("params-token_embedder-embedding", mapping)
    self.assertEqual(mapping["params-token_embedder-embedding"], "language_model.model.embed_tokens.weight")
    self.assertIn("params-decoder-decoder_norm-scale", mapping)
    self.assertEqual(mapping["params-decoder-decoder_norm-scale"], "language_model.model.norm.weight")
    self.assertIn("params-decoder-logits_dense-kernel", mapping)
    self.assertEqual(mapping["params-decoder-logits_dense-kernel"], "language_model.lm_head.weight")
    self.assertIn("params-decoder-output_attn_res_norm-scale", mapping)
    self.assertEqual(
        mapping["params-decoder-output_attn_res_norm-scale"], "language_model.model.output_attn_res_norm.weight"
    )
    self.assertIn("params-decoder-output_attn_res_proj-kernel", mapping)
    self.assertEqual(
        mapping["params-decoder-output_attn_res_proj-kernel"], "language_model.model.output_attn_res_proj.weight"
    )

    # Check all 93 layers
    raw_full_attn = self.text_config.get("full_attn_layers")
    if raw_full_attn is None and "linear_attn_config" in self.text_config:
      raw_full_attn = self.text_config["linear_attn_config"].get("full_attn_layers")
    if raw_full_attn is not None:
      if 4 in raw_full_attn and 3 not in raw_full_attn:
        full_attn_set = set(x - 1 for x in raw_full_attn)
      else:
        full_attn_set = set(raw_full_attn)
    else:
      full_attn_set = set(_FULL_ATTN_LAYERS)
    num_experts = self.text_config["num_experts"]

    for i in range(93):
      prefix = f"params-decoder-layers_{i}"
      # AttnRes & Norms
      self.assertIn(f"{prefix}-pre_self_attention_layer_norm-scale", mapping)
      self.assertIn(f"{prefix}-post_self_attention_layer_norm-scale", mapping)
      self.assertIn(f"{prefix}-self_attention_res_norm-scale", mapping)
      self.assertIn(f"{prefix}-self_attention_res_proj-kernel", mapping)
      self.assertIn(f"{prefix}-mlp_res_norm-scale", mapping)
      self.assertIn(f"{prefix}-mlp_res_proj-kernel", mapping)

      # Attention (MLA vs KDA)
      if i in full_attn_set:
        self.assertIn(f"{prefix}-self_attention-wq_a-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-q_norm-scale", mapping)
        self.assertIn(f"{prefix}-self_attention-wq_b-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-wkv_a-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-kv_norm-scale", mapping)
        self.assertIn(f"{prefix}-self_attention-wkv_b-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-g_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-out-kernel", mapping)
      else:
        self.assertIn(f"{prefix}-self_attention-q_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-q_conv1d-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-k_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-k_conv1d-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-v_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-v_conv1d-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-f_a_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-f_b_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-A_log", mapping)
        self.assertIn(f"{prefix}-self_attention-dt_bias", mapping)
        self.assertIn(f"{prefix}-self_attention-b_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-g_proj-kernel", mapping)
        self.assertIn(f"{prefix}-self_attention-o_norm-scale", mapping)
        self.assertIn(f"{prefix}-self_attention-out-kernel", mapping)

      # MLP (Layer 0 Dense vs Layers 1-92 Latent MoE)
      if i == 0:
        self.assertIn(f"{prefix}-mlp-wi_0-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-wi_1-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-wo-kernel", mapping)
      else:
        self.assertIn(f"{prefix}-mlp-routed_experts-gate-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-routed_experts-gate-e_score_correction_bias", mapping)
        self.assertIn(f"{prefix}-mlp-routed_expert_down_proj-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-routed_expert_norm-scale", mapping)
        self.assertIn(f"{prefix}-mlp-routed_expert_up_proj-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-shared_expert-wi_0-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-shared_expert-wi_1-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-shared_expert-wo-kernel", mapping)
        self.assertIn(f"{prefix}-mlp-routed_experts-wi_0", mapping)
        self.assertIn(f"{prefix}-mlp-routed_experts-wi_1", mapping)
        self.assertIn(f"{prefix}-mlp-routed_experts-wo", mapping)
        self.assertEqual(len(mapping[f"{prefix}-mlp-routed_experts-wi_0"]), num_experts)
        self.assertEqual(len(mapping[f"{prefix}-mlp-routed_experts-wi_1"]), num_experts)
        self.assertEqual(len(mapping[f"{prefix}-mlp-routed_experts-wo"]), num_experts)

    # Full index.json verification if index file is present
    if os.path.exists(self.index_file):
      with open(self.index_file, "r", encoding="utf-8") as f:
        idx = json.load(f)
      weight_map = idx.get("weight_map", {})

      # Normalize index keys to unquantized target keys
      normalized_index_keys = set()
      for k in weight_map.keys():
        if not k.startswith("language_model."):
          continue
        if k.endswith(".weight_scale"):
          continue
        if k.endswith(".weight_packed"):
          normalized_index_keys.add(k[: -len(".weight_packed")] + ".weight")
        else:
          normalized_index_keys.add(k)

      mapped_hf_keys = set()
      for _, hf_val in mapping.items():
        if isinstance(hf_val, str):
          mapped_hf_keys.add(hf_val)
        elif isinstance(hf_val, list):
          for k in hf_val:
            mapped_hf_keys.add(k)

      missing = normalized_index_keys - mapped_hf_keys
      extra = mapped_hf_keys - normalized_index_keys

      self.assertEqual(len(missing), 0, f"Missing parameter keys in mapping: {sorted(list(missing))[:10]}")
      self.assertEqual(len(extra), 0, f"Extra parameter keys in mapping: {sorted(list(extra))[:10]}")
      self.assertEqual(len(mapped_hf_keys), len(normalized_index_keys))
      self.assertEqual(len(mapped_hf_keys), 249756)

  # ==========================================================================
  # 3. Parameter Transformation Hook Function Verification
  # ==========================================================================
  def test_kimi_k3_param_hook_transformations(self):
    """Verifies that parameter hook transformations correctly reshape and transpose weights."""
    hooks = HOOK_FNS["kimi-k3"](self.text_config, None, scan_layers=False, saving_to_hf=False)

    # 2D Kernel Transpose Hook
    test_2d_tensor = np.arange(12).reshape(3, 4)
    logits_hook = hooks["params-decoder-logits_dense-kernel"]
    transposed_2d = logits_hook(test_2d_tensor)
    self.assertEqual(transposed_2d.shape, (4, 3))
    np.testing.assert_array_equal(transposed_2d, test_2d_tensor.T)

    # 1D Conv Permute Hook: [C, 1, K] <-> [K, 1, C]
    test_conv_tensor = np.arange(24).reshape(6, 1, 4)
    q_conv_hook = hooks["params-decoder-layers_0-self_attention-q_conv1d-kernel"]
    permuted_conv = q_conv_hook(test_conv_tensor)
    self.assertEqual(permuted_conv.shape, (4, 1, 6))

    # 3D Stacked Expert Transpose Hook: [num_experts, in_dim, out_dim] -> [num_experts, out_dim, in_dim]
    test_3d_tensor = np.arange(24).reshape(2, 3, 4)
    expert_hook = hooks["params-decoder-layers_1-mlp-routed_experts-wi_0"]
    transposed_3d = expert_hook(test_3d_tensor)
    self.assertEqual(transposed_3d.shape, (2, 4, 3))


if __name__ == "__main__":
  unittest.main()
