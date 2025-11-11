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

"""Tests for Tunix integration utils."""

import unittest
from unittest import mock

from absl.testing import absltest
from maxtext.src.maxtext.integration.tunix import utils
from MaxText.utils.ckpt_conversion.utils import hf_model_configs
from MaxText.utils.ckpt_conversion.utils import param_mapping


HF_MODEL_CONFIGS = hf_model_configs.HF_MODEL_CONFIGS
VllmWeightMapping = utils.VllmWeightMapping
PARAM_MAPPING = param_mapping.PARAM_MAPPING


class VllmWeightMappingTest(unittest.TestCase):
  """Tests for VllmWeightMapping."""

  def test_convert_hf_map_to_sharding_map_qwen3_8b(self):
    """Tests the convert_hf_map_to_sharding_map method."""
    model_name = "qwen3-8b"
    config = HF_MODEL_CONFIGS[model_name].to_dict()
    vllm_weight_mapping = VllmWeightMapping(model_name, config)
    # Sample hf_mapping to test various correction and generalization rules
    sample_hf_mapping = PARAM_MAPPING[model_name](config, scan_layers=True)

    expected_sharding_map = {
        # Token embeddings - shard vocab dimension
        "base.token_embedder.embedding": (
            "model.embed.embedding",
            ("model", None),
        ),
        # Final layer norm - no sharding needed
        "base.decoder.decoder_norm.scale": (
            "model.norm.scale",
            (None,),
        ),
        # LM head (logits projection) - shard vocab dimension
        "base.decoder.logits_dense.kernel": (
            "model.lm_head",
            (None, "model"),
        ),
        # Layer-specific mappings (scanned -> unscanned)
        # MLP components - shard hidden dimensions
        "base.decoder.layers.mlp.wi_0.kernel": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wi_1.kernel": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wo.kernel": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", "layer", None),
        ),
        # Layer norms - no sharding needed
        "base.decoder.layers.pre_self_attention_layer_norm.scale": (
            "model.layers.*.input_layernorm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.post_self_attention_layer_norm.scale": (
            "model.layers.*.post_attention_layernorm.scale",
            (None, "layer"),
        ),
        # Attention components - shard head dimensions
        "base.decoder.layers.self_attention.query.kernel": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.key.kernel": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.value.kernel": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.out.kernel": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", "layer", None, None),
        ),
        "base.decoder.layers.self_attention.query_norm.scale": (
            "model.layers.*.self_attn.q_norm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.key_norm.scale": (
            "model.layers.*.self_attn.k_norm.scale",
            (None, "layer"),
        ),
    }

    with mock.patch.dict(
        PARAM_MAPPING,
        {model_name: lambda config, scan_layers: sample_hf_mapping},
    ):
      actual_sharding_map = vllm_weight_mapping.to_hf_mapping()
      self.assertEqual(
          sorted(actual_sharding_map.items()),
          sorted(expected_sharding_map.items()),
      )

  def test_convert_hf_map_to_sharding_map_qwen3_8b_standalone_mappings(self):
    """Tests the convert_hf_map_to_sharding_map method."""
    model_name = "qwen3-8b"
    config = HF_MODEL_CONFIGS[model_name].to_dict()
    vllm_weight_mapping = VllmWeightMapping(model_name, config, use_standalone_mappings=True)

    expected_sharding_map = {
        # Token embeddings - shard vocab dimension
        "base.token_embedder.embedding": (
            "model.embed.embedding",
            ("model", None),
        ),
        # Final layer norm - no sharding needed
        "base.decoder.decoder_norm.scale": (
            "model.norm.scale",
            (None,),
        ),
        # LM head (logits projection) - shard vocab dimension
        "base.decoder.logits_dense.kernel": (
            "model.lm_head",
            (None, "model"),
        ),
        # Layer-specific mappings (scanned -> unscanned)
        # MLP components - shard hidden dimensions
        "base.decoder.layers.mlp.wi_0.kernel": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wi_1.kernel": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wo.kernel": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", "layer", None),
        ),
        # Layer norms - no sharding needed
        "base.decoder.layers.pre_self_attention_layer_norm.scale": (
            "model.layers.*.input_layernorm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.post_self_attention_layer_norm.scale": (
            "model.layers.*.post_attention_layernorm.scale",
            (None, "layer"),
        ),
        # Attention components - shard head dimensions
        "base.decoder.layers.self_attention.query.kernel": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.key.kernel": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.value.kernel": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.out.kernel": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", "layer", None, None),
        ),
        "base.decoder.layers.self_attention.query_norm.scale": (
            "model.layers.*.self_attn.q_norm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.key_norm.scale": (
            "model.layers.*.self_attn.k_norm.scale",
            (None, "layer"),
        ),
    }

    actual_sharding_map = vllm_weight_mapping.to_hf_mapping()
    print(f"Debug: actual_sharding_map={actual_sharding_map.values()}")
    self.assertEqual(
        sorted(actual_sharding_map.items()),
        sorted(expected_sharding_map.items()),
    )

  def test_convert_hf_map_to_sharding_map_deepseekv3(self):
    """Tests the convert_hf_map_to_sharding_map method."""
    model_name = "deepseek3-671b"
    config = HF_MODEL_CONFIGS[model_name].to_dict()
    vllm_weight_mapping = VllmWeightMapping(model_name, config)
    # Sample hf_mapping to test various correction and generalization rules

    expected_sharding_map = {
        # --- Non-Layer Parameters ---
        "base.token_embedder.embedding": (
            "embedder.input_embedding_table_VD",
            ("model", None),
        ),
        "base.decoder.decoder_norm.scale": (
            "final_norm.scale",
            (None,),
        ),
        "base.decoder.logits_dense.kernel": (
            "lm_head.input_embedding_table_DV",
            (None, "model"),
        ),

        # ==============================================================================
        # DENSE LAYERS MAPPING
        # ==============================================================================
        "base.decoder.dense_layers.pre_self_attention_layer_norm.scale": (
            "layers.*.pre_attention_norm.scale", (None, "layer")
        ),
        "base.decoder.dense_layers.post_self_attention_layer_norm.scale": (
            "layers.*.pre_mlp_norm.scale", (None, "layer")
        ),
        # --- Attention (MLA) ---
        # Q projections (Down/Up)
        "base.decoder.dense_layers.self_attention.wq_a.kernel": (
            "layers.*.attn.kernel_q_down_proj_DA", (None, "layer", "model", None)
        ),
        "base.decoder.dense_layers.self_attention.wq_b.kernel": (
            "layers.*.attn.kernel_q_up_proj_ANH", (None, "layer", "model", None)
        ),
        # KV projections (Down/Up with MQA)
        "base.decoder.dense_layers.self_attention.wkv_a.kernel": (
            "layers.*.attn.kernel_kv_down_proj_DA", (None, "layer", "model", None)
        ),
        "base.decoder.dense_layers.self_attention.wkv_b.kernel": (
            "layers.*.attn.kernel_kv_up_proj_ANH", (None, "layer", "model", None)
        ),
        # Output projection
         "base.decoder.dense_layers.self_attention.out.kernel": (
            "layers.*.attn.kernel_o_proj_NHD", ("model", "layer", None, None)
        ),
        # MLA Norms
        "base.decoder.dense_layers.self_attention.kv_norm.scale": (
             "layers.*.attn.kv_rms_norm.scale", (None, "layer")
        ),
        "base.decoder.dense_layers.self_attention.q_norm.scale": (
             "layers.*.attn.q_rms_norm.scale", (None, "layer")
        ),
        # --- Dense MLP ---
        "base.decoder.dense_layers.mlp.wi_0.kernel": (
            "layers.*.custom_module.kernel_gating_DF", (None, "layer", "model")
        ),
        "base.decoder.dense_layers.mlp.wi_1.kernel": (
             "layers.*.custom_module.kernel_up_proj_DF", (None, "layer", "model")
        ),
        "base.decoder.dense_layers.mlp.wo.kernel": (
             "layers.*.custom_module.kernel_down_proj_FD", ("model", "layer", None)
        ),

        # ==============================================================================
        # MOE LAYERS MAPPING
        # ==============================================================================
        "base.decoder.moe_layers.pre_self_attention_layer_norm.scale": (
            "layers.*.pre_attention_norm.scale", (None, "layer")
        ),
        "base.decoder.moe_layers.post_self_attention_layer_norm.scale": (
            "layers.*.pre_mlp_norm.scale", (None, "layer")
        ),
        # --- Attention (MLA + Decoupled RoPE) for MoE Layers ---
        "base.decoder.moe_layers.self_attention.wq_a.kernel": (
            "layers.*.attn.kernel_q_down_proj_DA", (None, "layer", "model", None)
        ),
        "base.decoder.moe_layers.self_attention.wq_b.kernel": (
            "layers.*.attn.kernel_q_up_proj_ANH", (None, "layer", "model", None)
        ),
        "base.decoder.moe_layers.self_attention.wkv_a.kernel": (
            "layers.*.attn.kernel_kv_down_proj_DA", (None, "layer", "model", None)
        ),
        "base.decoder.moe_layers.self_attention.wkv_b.kernel": (
            "layers.*.attn.kernel_kv_up_proj_ANH", (None, "layer", "model", None)
        ),
        "base.decoder.moe_layers.self_attention.out.kernel": (
            "layers.*.attn.kernel_o_proj_NHD", ("model", "layer", None, None)
        ),
         "base.decoder.moe_layers.self_attention.kv_norm.scale": (
             "layers.*.attn.kv_rms_norm.scale", (None, "layer")
        ),
        "base.decoder.moe_layers.self_attention.q_norm.scale": (
             "layers.*.attn.q_rms_norm.scale", (None, "layer")
        ),

        # --- DeepSeek MoE Blocks ---
        # Shared Experts
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel": (
            "layers.*.shared_experts.kernel_gating_DF", 
            (None, "layer", "model")
        ),
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel": (
            "layers.*.shared_experts.kernel_up_proj_DF", 
            (None, "layer", "model")
        ),
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.shared_experts.wo.kernel": (
            "layers.*.shared_experts.kernel_down_proj_FD", 
            ("model", "layer", None)
        ),
        # Gating (Router)
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel": (
            "layers.*.custom_module.router.kernel_DE",
            (None, "layer", "model")
        ),
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias": (
            "layers.*.custom_module.router.bias_E",
            (None, "layer", "model")
        ),

        # Routed Experts
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_0": (
            "layers.*.custom_module.kernel_gating_EDF", 
            ("expert", "layer", None, "model")
        ),
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_1": (
            "layers.*.custom_module.kernel_up_proj_EDF", 
            ("expert", "layer", None, "model")
        ),
        "base.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.wo": (
            "layers.*.custom_module.kernel_down_proj_EFD", 
            ("expert", "layer", "model", None)
        ),
    }
    actual_sharding_map = vllm_weight_mapping.to_hf_mapping()
    extra_keys = expected_sharding_map.keys() - actual_sharding_map.keys()
    missing_keys = actual_sharding_map.keys() - expected_sharding_map.keys()
    for key in extra_keys:
      print(f"Extra key: {key}={expected_sharding_map[key]}")

    if missing_keys:
      raise ValueError(f"Missing keys: {missing_keys}")

    # compare the actual sharding map with the expected sharding map without extra keys
    mismatch_keys = []
    for key, value in actual_sharding_map.items():
      if key in expected_sharding_map:
        if value != expected_sharding_map[key]:
          mismatch_keys.append(f"Got={value[0]}, want={expected_sharding_map[key][0]}")

    if mismatch_keys:
      raise ValueError(f"Mismatch keys: got={mismatch_keys}")

if __name__ == "__main__":
  absltest.main()
