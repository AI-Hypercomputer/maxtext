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
import pytest

pytestmark = [pytest.mark.decoupled_target]


from maxtext.checkpoint_conversion.to_maxtext import _build_multi_axis_stacked_tensor
from maxtext.checkpoint_conversion.utils import hf_shape
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
    num_layers, cycle, num_experts = 8, 4, 2
    config = {
        "num_hidden_layers": num_layers,
        "num_experts": num_experts,
    }
    maxtext_config = mock.Mock()
    maxtext_config.inhomogeneous_layer_cycle_interval = cycle
    mapping = param_mapping.QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)

    # A block covers one period of the hybrid pattern: `cycle - 1` linear-attention
    # layers as an inner scan, then one full-attention layer.
    num_blocks, num_local = num_layers // cycle, cycle - 1
    local_prefix = "params-decoder-layers-local_layers"
    global_prefix = "params-decoder-layers-global_layer"
    self.assertIn(f"{local_prefix}-attention-in_proj_qkvz-kernel", mapping)
    self.assertIn(f"{global_prefix}-attention-attention-query-kernel", mapping)
    # Linear attention only exists on the local layers, full attention only on the global one.
    self.assertNotIn(f"{global_prefix}-attention-in_proj_qkvz-kernel", mapping)
    self.assertNotIn(f"{local_prefix}-attention-attention-query-kernel", mapping)

    # local_layers values are nested [block][local]; global_layer is flat over blocks.
    local_val = mapping[f"{local_prefix}-attention-in_proj_qkvz-kernel"]
    self.assertEqual(len(local_val), num_blocks)
    self.assertEqual(len(local_val[0]), num_local)
    self.assertEqual(local_val[0][0], "model.layers.0.linear_attn.in_proj_qkvz.weight")
    global_val = mapping[f"{global_prefix}-attention-attention-query-kernel"]
    self.assertEqual(len(global_val), num_blocks)
    # The full-attention layer is last in the period.
    self.assertEqual(global_val[0], f"model.layers.{cycle - 1}.self_attn.q_proj.weight")

    # Routed experts add a leading expert axis: [expert][block][local] and [expert][block].
    local_experts = mapping[f"{local_prefix}-mlp-routed_experts-wi_0"]
    self.assertEqual(len(local_experts), num_experts)
    self.assertEqual(len(local_experts[0]), num_blocks)
    self.assertEqual(len(local_experts[0][0]), num_local)
    self.assertEqual(local_experts[1][0][0], "model.layers.0.mlp.experts.1.gate_proj.weight")
    global_experts = mapping[f"{global_prefix}-mlp-routed_experts-wi_0"]
    self.assertEqual(len(global_experts), num_experts)
    self.assertEqual(len(global_experts[0]), num_blocks)

  @staticmethod
  def _indices_from_name(name):
    """Parses a synthetic HF key such as "e1_b0_l2" back into its stacked-axis indices."""
    return tuple(int(part[1:]) for part in name.split("_"))

  def _qwen3_next_conversion_config(self):
    cfg = mock.Mock()
    cfg.param_scan_axis = 1
    cfg.scan_layers = True
    cfg.weight_dtype = "float32"
    cfg.rope_type = ""
    cfg.model_name = "qwen3-next-80b-a3b"
    return cfg

  def _assert_stack_unstack_roundtrip(self, mt_key, hf_names, target_shape, slice_shape, value_of, cfg):
    """Stacking HF weights into `target_shape` (to_maxtext) then un-stacking them back
    (to_huggingface) must be the identity, with every stacked axis in the right place."""

    def getter(name):
      return value_of(*self._indices_from_name(name))

    stacked = _build_multi_axis_stacked_tensor(hf_names, getter, None, target_shape, cfg, mt_key)
    self.assertEqual(stacked.shape, target_shape)

    param_map = {mt_key: hf_names}
    flat_names = []

    def flatten(keys):
      if isinstance(keys, list):
        for sub in keys:
          flatten(sub)
      else:
        flat_names.append(keys)

    flatten(hf_names)
    hf_shape_map = {name: slice_shape for name in flat_names}
    out = dict(process_maxtext_param(mt_key, stacked, param_map, {}, hf_shape_map, cfg))
    self.assertEqual(len(out), len(flat_names))
    for name in flat_names:
      np.testing.assert_array_equal(out[name], value_of(*self._indices_from_name(name)))
    return stacked

  def test_qwen3_next_local_layers_stack_unstack_roundtrip(self):
    """The local layers of a qwen3-next block are an inner scan, so their two stacked axes
    land at (param_scan_axis, param_scan_axis + 1) -- e.g. (emb, blocks, local)."""
    num_blocks, num_local = 2, 3
    slice_shape = (4, 3)  # per-(block, local) HF weight shape
    cfg = self._qwen3_next_conversion_config()
    mt_key = "params-decoder-layers-local_layers-attention-in_proj_qkvz-kernel"

    def value_of(b, l):
      return np.full(slice_shape, b * 100 + l, dtype=np.float32)

    hf_names = [[f"b{b}_l{l}" for l in range(num_local)] for b in range(num_blocks)]
    # blocks at axis 1, local at axis 2.
    target_shape = (slice_shape[0], num_blocks, num_local, slice_shape[1])

    stacked = self._assert_stack_unstack_roundtrip(mt_key, hf_names, target_shape, slice_shape, value_of, cfg)
    for b in range(num_blocks):
      for l in range(num_local):
        np.testing.assert_array_equal(stacked[:, b, l, :], value_of(b, l))

  def test_qwen3_next_routed_experts_stack_unstack_roundtrip(self):
    """qwen3-next's routed experts are expert-stacked *inside* the nested block scan, so they
    need three stacked axes: the expert axis still leads, then (blocks, local)."""
    num_experts, num_blocks, num_local = 2, 2, 3
    slice_shape = (4, 3)  # per-(expert, block, local) HF weight shape
    cfg = self._qwen3_next_conversion_config()
    mt_key = "params-decoder-layers-local_layers-mlp-routed_experts-wi_0"

    def value_of(e, b, l):
      return np.full(slice_shape, e * 10000 + b * 100 + l, dtype=np.float32)

    hf_names = [[[f"e{e}_b{b}_l{l}" for l in range(num_local)] for b in range(num_blocks)] for e in range(num_experts)]
    # experts at axis 0, blocks at axis 1, local at axis 2.
    target_shape = (num_experts, num_blocks, num_local, *slice_shape)

    stacked = self._assert_stack_unstack_roundtrip(mt_key, hf_names, target_shape, slice_shape, value_of, cfg)
    for e in range(num_experts):
      for b in range(num_blocks):
        for l in range(num_local):
          np.testing.assert_array_equal(stacked[e, b, l], value_of(e, b, l))

  def test_qwen3_next_global_layer_experts_stack_unstack_roundtrip(self):
    """The global (full-attention) layer is not inside the inner scan, so its routed experts
    keep the plain scanned-MoE layout with both stacked axes leading: (experts, blocks)."""
    num_experts, num_blocks = 2, 3
    slice_shape = (4, 3)
    cfg = self._qwen3_next_conversion_config()
    mt_key = "params-decoder-layers-global_layer-mlp-routed_experts-wi_0"

    def value_of(e, b):
      return np.full(slice_shape, e * 100 + b, dtype=np.float32)

    hf_names = [[f"e{e}_b{b}" for b in range(num_blocks)] for e in range(num_experts)]
    target_shape = (num_experts, num_blocks, *slice_shape)

    stacked = self._assert_stack_unstack_roundtrip(mt_key, hf_names, target_shape, slice_shape, value_of, cfg)
    for e in range(num_experts):
      for b in range(num_blocks):
        np.testing.assert_array_equal(stacked[e, b], value_of(e, b))

  def test_cosmos3_text_mapping(self):
    config = {
        "text_config": {"num_hidden_layers": 2, "hidden_size": 256},
    }
    maxtext_config = mock.Mock()
    maxtext_config.use_multimodal = False
    mapping = param_mapping.COSMOS3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False)
    self.assertIn("params-token_embedder-embedding", mapping)
    self.assertEqual(mapping["params-decoder-layers_0-self_attention-query-kernel"], "layers.0.self_attn.to_q.weight")

  def test_cosmos3_text_mapping_scanned(self):
    config = {
        "text_config": {"num_hidden_layers": 4, "hidden_size": 256},
    }
    maxtext_config = mock.Mock()
    maxtext_config.use_multimodal = False
    mapping = param_mapping.COSMOS3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=True)
    self.assertIn("params-decoder-layers-self_attention-query-kernel", mapping)
    self.assertEqual(
        mapping["params-decoder-layers-self_attention-query-kernel"],
        [
            "layers.0.self_attn.to_q.weight",
            "layers.1.self_attn.to_q.weight",
            "layers.2.self_attn.to_q.weight",
            "layers.3.self_attn.to_q.weight",
        ],
    )

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


class Hy3ParamMappingTest(unittest.TestCase):
  """Static shape/mapping coverage for Hy3.

  The end-to-end logit check already proves the mapping is right, but it needs
  a real checkpoint and a TPU. These tests catch a renamed key or a dropped
  layer group in CI instead, in both scan modes.
  """

  # 1 dense layer + 3 MoE layers, 4 routed experts -- the smallest config that
  # still exercises the dense/MoE split and the expert stacking.
  HF_CONFIG = {
      "num_hidden_layers": 4,
      "first_k_dense_replace": 1,
      "num_experts": 4,
      "hidden_size": 128,
      "intermediate_size": 256,
      "moe_intermediate_size": 64,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 32,
      "vocab_size": 1000,
      "num_shared_experts": 1,
  }

  def _mapping(self, scan_layers):
    return param_mapping.HY3_MAXTEXT_TO_HF_PARAM_MAPPING(self.HF_CONFIG, mock.Mock(), scan_layers=scan_layers)

  def test_scanned_mapping_covers_both_layer_groups(self):
    mapping = self._mapping(scan_layers=True)
    # Top-level parameters.
    self.assertEqual(mapping["params-token_embedder-embedding"], "model.embed_tokens.weight")
    self.assertEqual(mapping["params-decoder-logits_dense-kernel"], "lm_head.weight")
    # Scanning collapses each group into one key holding a list of HF names.
    self.assertEqual(
        mapping["params-decoder-dense_layers-mlp-wi_0-kernel"],
        ["model.layers.0.mlp.gate_proj.weight"],
    )
    self.assertEqual(
        mapping["params-decoder-moe_layers-Hy3MoeBlock_0-MoeBlock_0-gate-kernel"],
        [f"model.layers.{i}.mlp.router.gate.weight" for i in (1, 2, 3)],
    )

  def test_unscanned_mapping_numbers_layers_per_group(self):
    mapping = self._mapping(scan_layers=False)
    # Dense layers keep the HF index; MoE layers restart at 0.
    self.assertEqual(
        mapping["params-decoder-dense_layers_0-self_attention-query-kernel"],
        "model.layers.0.self_attn.q_proj.weight",
    )
    self.assertEqual(
        mapping["params-decoder-moe_layers_0-self_attention-query-kernel"],
        "model.layers.1.self_attn.q_proj.weight",
    )
    self.assertEqual(
        mapping["params-decoder-moe_layers_2-Hy3MoeBlock_0-MoeBlock_0-gate-bias"],
        "model.layers.3.mlp.expert_bias",
    )

  def test_routed_experts_stack_over_the_expert_axis(self):
    # Experts become a list per MoE layer (unscanned) so the loader can stack them.
    mapping = self._mapping(scan_layers=False)
    self.assertEqual(
        mapping["params-decoder-moe_layers_0-Hy3MoeBlock_0-MoeBlock_0-wi_0"],
        [f"model.layers.1.mlp.experts.{e}.gate_proj.weight" for e in range(4)],
    )

  def test_hy3_specific_hf_names_are_not_deepseek_or_qwen_names(self):
    """These three differ from the DeepSeek/Qwen conventions and were checked against the real checkpoint."""
    mapping = self._mapping(scan_layers=False)
    prefix = "params-decoder-moe_layers_0-Hy3MoeBlock_0"
    self.assertEqual(mapping[f"{prefix}-MoeBlock_0-gate-kernel"], "model.layers.1.mlp.router.gate.weight")
    self.assertEqual(mapping[f"{prefix}-MoeBlock_0-gate-bias"], "model.layers.1.mlp.expert_bias")
    self.assertEqual(mapping[f"{prefix}-shared_experts-wi_0-kernel"], "model.layers.1.mlp.shared_mlp.gate_proj.weight")

  def test_mtp_layer_is_not_mapped(self):
    """Layer `num_hidden_layers` is the MTP layer; it is deliberately out of scope."""
    for scan_layers in (True, False):
      hf_names = []
      for value in self._mapping(scan_layers).values():
        if isinstance(value, str):
          hf_names.append(value)
        else:
          for item in value:
            hf_names.extend(item if isinstance(item, list) else [item])
      self.assertFalse(
          [n for n in hf_names if n.startswith("model.layers.4.")],
          f"MTP layer should not be mapped (scan_layers={scan_layers})",
      )

  def test_shape_map_matches_the_mapping(self):
    """Every HF name the mapping produces must have a shape entry, and vice versa."""
    shapes = set(hf_shape.HY3_HF_WEIGHTS_TO_SHAPE(self.HF_CONFIG))
    mapped = set()
    for value in self._mapping(scan_layers=False).values():
      if isinstance(value, str):
        mapped.add(value)
      else:
        for item in value:
          mapped.update(item if isinstance(item, list) else [item])
    self.assertEqual(mapped - shapes, set(), "mapping references HF names with no shape entry")
    self.assertEqual(shapes - mapped, set(), "shape map has HF names the mapping never produces")


if __name__ == "__main__":
  unittest.main()
