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

"""Tests for kernels"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
from maxtext.utils.max_utils import permute_to_match_maxtext_rope, unpermute_from_match_maxtext_rope
from maxtext.checkpoint_conversion import to_huggingface as to_hf
from maxtext.checkpoint_conversion.to_huggingface import (
    _apply_yarn_rope_config,
    _get_lora_delta,
    _transform_weights_to_adapter,
    _transform_weights_to_full_model,
)
from maxtext.checkpoint_conversion.to_maxtext import (
    convert_hf_lora_key_to_maxtext,
    _process_and_stack_weights,
)


class HFCheckpointConversionTest(unittest.TestCase):

  def test_huggingface_to_maxtext_back_to_huggingface_flow(self):
    base_num_query_heads = 16
    head_dim = 32
    wq = np.arange(base_num_query_heads * head_dim * base_num_query_heads * head_dim, dtype=np.float16).reshape(
        base_num_query_heads * head_dim, base_num_query_heads * head_dim
    )
    wq1 = wq.transpose()
    wq2 = np.reshape(wq1, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

    wq3 = permute_to_match_maxtext_rope(wq2)
    stack_shape = (1,)
    x = np.zeros(stack_shape + wq3.shape, dtype=np.float16)
    x[0, ...] = wq3
    x = np.transpose(x, axes=(1, 0, 2, 3))

    x = x[:, 0, :, :]
    wq4 = unpermute_from_match_maxtext_rope(x, "llama3.1")
    wq5 = wq4.reshape(base_num_query_heads * head_dim, base_num_query_heads * head_dim)
    wq6 = wq5.transpose()

    if not np.array_equal(wq, wq6):
      print("Test failed: wq does not match wq6")

    if not np.array_equal(wq1, wq5):
      print("Test failed: wq1 does not match wq5")

    if not np.array_equal(wq2, wq4):
      print("Test failed: wq2 does not match wq4")

  def test_apply_yarn_rope_config_preserves_existing_hf_fields(self):
    hf_config = SimpleNamespace(
        rope_theta=10_000,
        rope_scaling={
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 40.0,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "rope_theta": 10_000,
            "type": "yarn",
        },
    )
    max_config = SimpleNamespace(
        beta_fast=32,
        beta_slow=1,
        rope_factor=40,
        original_max_position_embeddings=4096,
        rope_max_timescale=10_000,
        rope_truncate=True,
    )

    _apply_yarn_rope_config(hf_config, max_config)

    self.assertEqual(hf_config.rope_theta, 10_000)
    self.assertEqual(hf_config.rope_scaling["type"], "yarn")
    self.assertEqual(hf_config.rope_scaling["mscale"], 0.707)
    self.assertEqual(hf_config.rope_scaling["mscale_all_dim"], 0.707)
    self.assertTrue(hf_config.rope_scaling["truncate"])


class MaxTextToHFLoRAConversionTest(unittest.TestCase):
  """Tests the conversion modes (Base, Adapter, Merged) in to_huggingface with LoRA support."""

  def setUp(self):
    super().setUp()
    self.base_key = "params-decoder-layers-layers_0-self_attention-query-kernel"
    self.a_key = self.base_key + "_lora_a"
    self.b_key = self.base_key + "_lora_b"
    self.scaling = 2.0

    # Simple weights for verification
    # W: (10, 2, 20), A: (10, 2, 4), B: (4, 2, 20)
    self.w_base = np.ones((10, 2, 20), dtype=np.float32)
    self.w_a = np.ones((10, 2, 4), dtype=np.float32) * 0.5
    self.w_b = np.ones((4, 2, 20), dtype=np.float32) * 0.5

    # Expected Merged: W + (B@A)*scaling
    # B@A for each head: (20, 4) @ (4, 10) -> (20, 10) wait, MaxText shapes:
    # MaxText A: (in, heads, rank), B: (rank, heads, out)
    # Merging logic: matmul(A[:, i, :], B[:, i, :]) -> (in, out)
    # head_delta = (0.5 * 0.5) * rank * scaling = 0.25 * 4 * 2.0 = 2.0
    # W_merged head = 1.0 + 2.0 = 3.0
    self.expected_merged_val = 3.0

  def test_get_lora_delta(self):
    lora_dict = {self.a_key: self.w_a, self.b_key: self.w_b}
    delta = _get_lora_delta(self.base_key, lora_dict, self.scaling)

    self.assertEqual(delta.shape, (10, 2, 20))
    self.assertTrue(np.allclose(delta, 2.0))

  def test_transform_weights_to_adapter(self):
    param_map = {self.base_key: "model.layers.0.self_attn.q_proj.weight"}
    lora_dict = {self.a_key: self.w_a, self.b_key: self.w_b}

    weights, modules = _transform_weights_to_adapter(param_map, lora_dict)

    self.assertIn("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", weights)
    self.assertIn("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", weights)
    self.assertEqual(weights["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape, (4, 10))
    self.assertEqual(weights["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"].shape, (20, 4))
    self.assertIn("q_proj", modules)

  def test_transform_weights_to_full_model_merged(self):
    config = MagicMock()
    config.lora.lora_alpha = 32.0
    config.lora.lora_rank = 16.0  # scaling = 2.0

    state_dict = {self.base_key: self.w_base, self.a_key: self.w_a, self.b_key: self.w_b}
    param_map = {self.base_key: "model.layers.0.self_attn.q_proj.weight"}

    # Mock process_maxtext_param to just return the weight
    orig_proc = to_hf.process_maxtext_param
    to_hf.process_maxtext_param = lambda k, w, pm, hfm, sm, c: [(pm[k], w)]

    try:
      weights = _transform_weights_to_full_model(config, [self.base_key], state_dict, param_map, {}, {})
    finally:
      to_hf.process_maxtext_param = orig_proc

    self.assertIn("model.layers.0.self_attn.q_proj.weight", weights)
    self.assertTrue(np.allclose(weights["model.layers.0.self_attn.q_proj.weight"], self.expected_merged_val))


class HFToMaxTextLoRAConversionTest(unittest.TestCase):
  """Tests the conversion logic in to_maxtext with LoRA support."""

  def test_convert_hf_lora_key_to_maxtext(self):
    param_mapping = {
        "params-decoder-layers-layers_0-self_attention-query-kernel": "model.layers.0.self_attn.q_proj.weight",
        "params-decoder-layers-layers_1-mlp-wi_0-kernel": [
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
        ],
    }

    # Simple 1-to-1
    mt_key, idx = convert_hf_lora_key_to_maxtext(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", param_mapping
    )
    self.assertEqual(mt_key, "params-decoder-layers-layers_0-self_attention-query-kernel")
    self.assertIsNone(idx)

    # Scanned/List mapping
    mt_key, idx = convert_hf_lora_key_to_maxtext(
        "base_model.model.model.layers.1.mlp.up_proj.lora_B.weight", param_mapping
    )
    self.assertEqual(mt_key, "params-decoder-layers-layers_1-mlp-wi_0-kernel")
    self.assertEqual(idx, 1)

  def test_process_and_stack_weights(self):
    config = MagicMock()
    config.model_name = "llama3.1-8b"
    config.head_dim = 128

    # 1. Non-scanned case
    indexed = {0: np.ones((10, 20))}
    stacked = _process_and_stack_weights(indexed, False, 1, 0, np.float32, "test", "suffix", config)
    self.assertEqual(stacked.shape, (20, 10))  # Transposed

    # 2. Scanned case (stacking along layers)
    indexed = {0: np.ones((10, 20)) * 1.0, 1: np.ones((10, 20)) * 2.0}
    stacked = _process_and_stack_weights(indexed, True, 2, 0, np.float32, "test", "suffix", config)
    self.assertEqual(stacked.shape, (2, 20, 10))
    self.assertEqual(stacked[1, 0, 0], 2.0)


class ParamKeyPartsFromPathTest(unittest.TestCase):
  """Tests param_key_parts_from_path with different path tuple entries."""

  def test_param_key_parts_from_path(self):
    # pylint: disable=import-outside-toplevel
    from maxtext.checkpoint_conversion.utils.utils import param_key_parts_from_path

    # Mock JAX/NNX path key components
    class MockDictKey:

      def __init__(self, key):
        self.key = key

    class MockSequenceKey:

      def __init__(self, idx):
        self.idx = idx

    class MockGetAttrKey:

      def __init__(self, name):
        self.name = name

    # Test basic string path keys
    path = (
        MockDictKey("decoder"),
        MockGetAttrKey("layers"),
        MockSequenceKey(0),
        MockDictKey("kernel"),
        MockDictKey("value"),
    )
    result = param_key_parts_from_path(path)
    self.assertEqual(result, ["decoder", "layers_0", "kernel"])

    # Test pure string path
    path = ("decoder", "layers", "kernel")
    result = param_key_parts_from_path(path)
    self.assertEqual(result, ["decoder", "layers", "kernel"])

    # Test trailing value segment dropped
    path = ("decoder", "layers", "kernel", "value")
    result = param_key_parts_from_path(path)
    self.assertEqual(result, ["decoder", "layers", "kernel"])

    # Test numeric string round-trip folding
    path = ("decoder", "layers", "0", "kernel")
    result = param_key_parts_from_path(path)
    self.assertEqual(result, ["decoder", "layers_0", "kernel"])

  def test_get_maxtext_model_info(self):
    # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig
    from tests.utils.test_helpers import get_test_config_path
    from maxtext.checkpoint_conversion.to_maxtext import get_maxtext_model_info

    # Initialize a small test config
    cfg = pyconfig.initialize(
        [
            None,
            get_test_config_path(),
            "run_name=gemma4_small_test",
            "decoder_block=gemma4_small",
            "model_name=gemma4-e2b",
            "scan_layers=False",
            "attention=dot_product",
            "num_decoder_layers=2",
            "num_kv_shared_layers=1",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=256",
            "dtype=float32",
            "weight_dtype=float32",
            "hidden_size_per_layer_input=128",
            "vocab_size_per_layer_input=256",
            "vocab_size=256",
        ],
        override_model_config=True,
    )

    model_info, treedef = get_maxtext_model_info(cfg)
    self.assertIsNotNone(model_info)
    self.assertIsNotNone(treedef)
    self.assertTrue(any("decoder" in k for k in model_info))


if __name__ == "__main__":
  unittest.main()
