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

"""Tests for param_mapping.py"""

import unittest
from unittest import mock
import numpy as np

from maxtext.checkpoint_conversion.to_maxtext import _build_multi_axis_stacked_tensor
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.utils import process_maxtext_param


class ParamMappingTest(unittest.TestCase):

  def test_gemma3_mapping_unscanned(self):
    config = {
        "text_config": {"num_hidden_layers": 2, "hidden_size": 256},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 128},
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_gemma3_mapping_scanned(self):
    config = {
        "text_config": {"num_hidden_layers": 12, "hidden_size": 256},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 128},
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_gemma2_mapping(self):
    config = {
        "num_hidden_layers": 4,
        "hidden_size": 256,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_gemma2_mapping_scanned(self):
    config = {
        "num_hidden_layers": 4,
        "hidden_size": 256,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-layers-pre_self_attention_norm_local-scale", mapping)

  def test_qwen_mapping_dense(self):
    config = {
        "num_hidden_layers": 2,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_qwen_mapping_moe(self):
    config = {
        "num_hidden_layers": 2,
        "num_experts": 4,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-decoder-layers_0-moe_block-wi_0", mapping)

  def test_qwen_mapping_scanned(self):
    config = {
        "num_hidden_layers": 4,
        "hidden_size": 256,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-layers-pre_self_attention_layer_norm-scale", mapping)

  def test_qwen3_next_mapping(self):
    config = {
        "num_hidden_layers": 4,
        "num_experts": 2,
    }
    maxtext_config = mock.Mock()
    maxtext_config.inhomogeneous_layer_cycle_interval = 2
    mapping = param_mapping.QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_qwen3_next_mapping_scanned(self):
    config = {
        "num_hidden_layers": 4,
        "num_experts": 2,
    }
    maxtext_config = mock.Mock()
    maxtext_config.inhomogeneous_layer_cycle_interval = 2
    mapping = param_mapping.QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-scanned_blocks-local_layers-input_layernorm-scale", mapping)
    self.assertIn("params-decoder-scanned_blocks-global_layer-input_layernorm-scale", mapping)
    num_blocks = config["num_hidden_layers"] // maxtext_config.inhomogeneous_layer_cycle_interval
    local_val = mapping["params-decoder-scanned_blocks-local_layers-input_layernorm-scale"]
    global_val = mapping["params-decoder-scanned_blocks-global_layer-input_layernorm-scale"]
    self.assertEqual(len(local_val), num_blocks)
    self.assertEqual(len(local_val[0]), 1)
    self.assertEqual(len(global_val), num_blocks)

  def test_deepseek_mapping(self):
    config = {
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_deepseek_mapping_scanned(self):
    config = {
        "num_hidden_layers": 4,
        "first_k_dense_replace": 1,
        "n_routed_experts": 2,
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-dense_layers-self_attention-query-kernel", mapping)

  def test_gpt_oss_mapping(self):
    config = {
        "num_hidden_layers": 2,
    }
    maxtext_config = mock.Mock()
    maxtext_config.inhomogeneous_layer_cycle_interval = 1
    mapping = param_mapping.GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_gpt_oss_mapping_scanned(self):
    config = {
        "num_hidden_layers": 4,
    }
    maxtext_config = mock.Mock()
    maxtext_config.inhomogeneous_layer_cycle_interval = 2
    mapping = param_mapping.GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-layers-layers_0-pre_self_attention_layer_norm-scale", mapping)

  def test_mixtral_mapping(self):
    config = {
        "num_hidden_layers": 2,
    }
    maxtext_config = mock.Mock()
    maxtext_config.num_experts = 4
    mapping = param_mapping.MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_mixtral_mapping_scanned(self):
    config = {
        "num_hidden_layers": 4,
    }

    class Config:
      num_experts = 4

    maxtext_config = Config()
    mapping = param_mapping.MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-layers-self_attention-query-kernel", mapping)

  def test_gemma4_mapping(self):
    config = {
        "num_hidden_layers": 2,
    }
    maxtext_config = mock.Mock()
    maxtext_config.share_kv_projections = False
    maxtext_config.use_multimodal = False
    maxtext_config.v_norm_with_scale = False
    mapping = param_mapping.GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)

  def test_gemma4_mapping_scanned(self):
    config = {
        "num_hidden_layers": 12,
    }
    maxtext_config = mock.Mock()
    maxtext_config.share_kv_projections = False
    maxtext_config.use_multimodal = False
    maxtext_config.v_norm_with_scale = False
    mapping = param_mapping.GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    # The block scans its 5 local layers (nested [block][local]) then a single global layer.
    self.assertIn("params-decoder-scanned_blocks-local_layers-self_attention-query-kernel", mapping)
    self.assertIn("params-decoder-scanned_blocks-global_layer-self_attention-query-kernel", mapping)
    # local_layers value is nested [block][local]; global_layer is flat over blocks.
    num_blocks = config["num_hidden_layers"] // 6
    local_val = mapping["params-decoder-scanned_blocks-local_layers-self_attention-query-kernel"]
    global_val = mapping["params-decoder-scanned_blocks-global_layer-self_attention-query-kernel"]
    self.assertEqual(len(local_val), num_blocks)
    self.assertEqual(len(local_val[0]), 5)
    self.assertEqual(len(global_val), num_blocks)

  def test_gemma4_local_layers_stack_unstack_roundtrip(self):
    """Stacking HF weights into the nested [block][local] MaxText layout (to_maxtext) and
    un-stacking them back (to_huggingface) must be identity, with the two scan axes placed at
    (param_scan_axis, param_scan_axis + 1) -- not the leading axes used for MoE expert stacking."""
    num_blocks, num_local = 2, 5
    slice_shape = (4, 3)  # per-(block, local) HF weight shape
    cfg = mock.Mock()
    cfg.param_scan_axis = 1
    cfg.scan_layers = True
    cfg.weight_dtype = "float32"
    cfg.rope_type = ""
    cfg.model_name = "gemma4-31b"

    mt_key = "params-decoder-scanned_blocks-local_layers-self_attention-query-kernel"

    def value_of(b, l):
      return np.full(slice_shape, b * 100 + l, dtype=np.float32)

    hf_names = [[f"b{b}_l{l}" for l in range(num_local)] for b in range(num_blocks)]

    def getter(name):
      block_idx, local_idx = name.split("_")
      return value_of(int(block_idx[1:]), int(local_idx[1:]))

    # target: per-slice shape (4, 3) with (blocks, local) inserted at axes (1, 2)
    target_shape = (slice_shape[0], num_blocks, num_local, slice_shape[1])

    # Forward (to_maxtext): stack. Blocks must land at axis 1, local at axis 2.
    stacked = _build_multi_axis_stacked_tensor(hf_names, getter, None, target_shape, cfg, mt_key)
    self.assertEqual(stacked.shape, target_shape)
    for b in range(num_blocks):
      for l in range(num_local):
        np.testing.assert_array_equal(stacked[:, b, l, :], value_of(b, l))

    # Backward (to_huggingface): un-stack and check it round-trips to the originals.
    param_map = {mt_key: hf_names}
    hf_shape_map = {f"b{b}_l{l}": slice_shape for b in range(num_blocks) for l in range(num_local)}
    out = dict(process_maxtext_param(mt_key, stacked, param_map, {}, hf_shape_map, cfg))
    self.assertEqual(len(out), num_blocks * num_local)
    for b in range(num_blocks):
      for l in range(num_local):
        np.testing.assert_array_equal(out[f"b{b}_l{l}"], value_of(b, l))

  # Specific tests with assertions
  def test_reshape_kernel_hook(self):
    config = {
        "text_config": {"num_hidden_layers": 2, "hidden_size": 256},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 128},
    }
    maxtext_config = mock.Mock()
    hooks = param_mapping.GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=True)
    reshape_key = "params-decoder-layers_0-self_attention-query-kernel"
    reshape_hook = hooks[reshape_key]

    dummy_tensor = np.arange(6).reshape(2, 3).astype(np.float32)
    target_shape = (3, 2)
    output = reshape_hook(dummy_tensor, target_shape)
    expected_output = dummy_tensor.T
    np.testing.assert_allclose(output, expected_output)

  def test_scale_rmsnorm_hook(self):
    config = {
        "text_config": {"num_hidden_layers": 2, "hidden_size": 256},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 128},
    }
    maxtext_config = mock.Mock()
    hooks_to_hf = param_mapping.GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        config, maxtext_config, scan_layers=False, saving_to_hf=True
    )
    norm_key = "params-decoder-layers_0-pre_self_attention_norm-scale"
    norm_hook_to_hf = hooks_to_hf[norm_key]

    dummy_tensor = np.array([2.0, 3.0], dtype=np.float32)
    output = norm_hook_to_hf(dummy_tensor, (2,))
    np.testing.assert_allclose(output, np.array([1.0, 2.0]))

  def test_interleave_hook(self):
    config = {
        "num_hidden_layers": 2,
    }
    maxtext_config = mock.Mock()
    maxtext_config.inhomogeneous_layer_cycle_interval = 1
    hooks_to_hf = param_mapping.GPT_OSS_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=True)
    composite_key = ("params-decoder-layers_0-GptOssMlp-wi_0", "params-decoder-layers_0-GptOssMlp-wi_1")
    interleave_hook = hooks_to_hf[composite_key]

    wi_0 = np.array([1, 2], dtype=np.float32)
    wi_1 = np.array([3, 4], dtype=np.float32)

    output = interleave_hook((wi_0, wi_1), (4,))
    expected_output = np.array([1, 3, 2, 4], dtype=np.float32)
    np.testing.assert_allclose(output, expected_output)

  def test_qwen3_vl_fused_moe_hook(self):
    config = {
        "text_config": {"num_hidden_layers": 1, "num_local_experts": 2},
        "vision_config": {"depth": 1, "hidden_size": 128},
    }
    maxtext_config = mock.Mock()
    # Test saving to HF
    hooks_to_hf = param_mapping.QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        config, maxtext_config, scan_layers=False, saving_to_hf=True
    )
    composite_key = ("params-decoder-layers_0-moe_block-wi_0", "params-decoder-layers_0-moe_block-wi_1")
    self.assertIn(composite_key, hooks_to_hf)
    fused_hook_to_hf = hooks_to_hf[composite_key]

    wi_0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    wi_1 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    output_hf = fused_hook_to_hf((wi_0, wi_1), None)
    # Expected: concatenate along last axis
    expected_hf = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    np.testing.assert_allclose(output_hf, expected_hf)

    # Test loading to MaxText
    hooks_to_mt = param_mapping.QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        config, maxtext_config, scan_layers=False, saving_to_hf=False
    )
    self.assertIn(composite_key, hooks_to_mt)
    fused_hook_to_mt = hooks_to_mt[composite_key]

    fused_hf = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    output_mt = fused_hook_to_mt(fused_hf, None)
    # Expected: split along last axis, and stack along a new final axis
    expected_mt = np.stack([wi_0, wi_1], axis=-1)
    np.testing.assert_allclose(output_mt, expected_mt)

  def test_qwen3_vl_mapping(self):
    # Special case for Qwen3-VL: MaxText model has separate wi_0 and wi_1 weights
    # for the MoE block, but the HF model expects a single fused weight.
    config = {
        "text_config": {"num_hidden_layers": 1, "num_local_experts": 2},
        "vision_config": {"depth": 1, "hidden_size": 128},
    }
    maxtext_config = mock.Mock()
    mapping = param_mapping.QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)

    # Check text keys (replaced prefix)
    self.assertIn("params-token_embedder-embedding", mapping)
    self.assertEqual(
        mapping["params-token_embedder-embedding"],
        "model.language_model.embed_tokens.weight",
    )

    # Check MoE keys
    composite_key = ("params-decoder-layers_0-moe_block-wi_0", "params-decoder-layers_0-moe_block-wi_1")
    self.assertIn(composite_key, mapping)
    self.assertEqual(mapping[composite_key], "model.language_model.layers.0.mlp.experts.gate_up_proj")

    # Check vision keys
    self.assertIn("params-vision_encoder-Qwen3VLVisionEncoder_0-patch_embed-proj-kernel", mapping)
    self.assertEqual(
        mapping["params-vision_encoder-Qwen3VLVisionEncoder_0-patch_embed-proj-kernel"],
        "model.visual.patch_embed.proj.weight",
    )


if __name__ == "__main__":
  unittest.main()
