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

""" Tests for kernels """

import unittest
from unittest.mock import MagicMock
import numpy as np
from maxtext.utils.max_utils import permute_to_match_maxtext_rope, unpermute_from_match_maxtext_rope
from maxtext.checkpoint_conversion import to_huggingface as to_hf
from maxtext.checkpoint_conversion.to_huggingface import (
    _get_lora_delta,
    _transform_weights_to_adapter,
    _transform_weights_to_full_model,
)
from maxtext.checkpoint_conversion.to_maxtext import (
    convert_hf_lora_key_to_maxtext,
    _process_and_stack_weights,
)
from maxtext.checkpoint_conversion.utils.utils import (
    _recursive_update,
    load_orbax_checkpoint,
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

    # 1. Scanned standard linear Case A (3D): [num_layers, input_dim, rank] & [num_layers, rank, output_dim]
    param_map_scanned_a = {
        "params-decoder-scanned_blocks-mlp-wi_0-kernel": [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
        ]
    }
    # num_layers = 2, input_dim = 10, rank = 4, output_dim = 20
    data_a_scanned_a = np.ones((2, 10, 4), dtype=np.float32) * 0.5
    data_b_scanned_a = np.ones((2, 4, 20), dtype=np.float32) * 0.5
    lora_dict_scanned_a = {
        "params-decoder-scanned_blocks-mlp-wi_0-kernel_lora_a": data_a_scanned_a,
        "params-decoder-scanned_blocks-mlp-wi_0-kernel_lora_b": data_b_scanned_a,
    }
    weights_sa, _ = _transform_weights_to_adapter(param_map_scanned_a, lora_dict_scanned_a)
    self.assertIn("base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight", weights_sa)
    self.assertIn("base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight", weights_sa)
    # Since layer dimension is axis 0, layer 0 is data_a_scanned_a[0, :, :], which has shape (10, 4), transpose -> (4, 10)
    self.assertEqual(weights_sa["base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"].shape, (4, 10))
    self.assertEqual(weights_sa["base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight"].shape, (20, 4))

    # 2. Scanned standard linear Case B (3D): [input_dim, num_layers, rank] & [rank, num_layers, output_dim]
    param_map_scanned_b = {
        "params-decoder-scanned_blocks-mlp-wo-kernel": [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
    }
    # num_layers = 2, input_dim = 10, rank = 4, output_dim = 20
    data_a_scanned_b = np.ones((10, 2, 4), dtype=np.float32) * 0.5
    data_b_scanned_b = np.ones((4, 2, 20), dtype=np.float32) * 0.5
    lora_dict_scanned_b = {
        "params-decoder-scanned_blocks-mlp-wo-kernel_lora_a": data_a_scanned_b,
        "params-decoder-scanned_blocks-mlp-wo-kernel_lora_b": data_b_scanned_b,
    }
    weights_sb, _ = _transform_weights_to_adapter(param_map_scanned_b, lora_dict_scanned_b)
    self.assertIn("base_model.model.model.layers.0.mlp.down_proj.lora_A.weight", weights_sb)
    self.assertIn("base_model.model.model.layers.0.mlp.down_proj.lora_B.weight", weights_sb)
    # Since layer dimension is axis 1, layer 0 is data_a_scanned_b[:, 0, :], which has shape (10, 4), transpose -> (4, 10)
    self.assertEqual(weights_sb["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"].shape, (4, 10))
    self.assertEqual(weights_sb["base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"].shape, (20, 4))

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

  def test_get_lora_delta_scanned_and_unscanned_variants(self):
    cases = [
        # (name, key, shape_a, shape_b, expected_shape, expected_val)
        ("2d_linear", "params-decoder-layers-layers_0-mlp-wi_0-kernel", (10, 4), (4, 20), (10, 20), 2.0),
        (
            "3d_unscanned_attn",
            "params-decoder-layers-layers_0-self_attention-query-kernel",
            (10, 2, 4),
            (4, 2, 20),
            (10, 2, 20),
            2.0,
        ),
        (
            "3d_scanned_linear_a",
            "params-decoder-scanned_blocks-mlp-wi_0-kernel",
            (3, 10, 4),
            (3, 4, 20),
            (3, 10, 20),
            2.0,
        ),
        ("3d_scanned_linear_b", "params-decoder-scanned_blocks-mlp-wo-kernel", (10, 3, 4), (4, 3, 20), (10, 3, 20), 2.0),
        (
            "4d_scanned_attn",
            "params-decoder-scanned_blocks-self_attention-query-kernel",
            (3, 10, 2, 4),
            (3, 4, 2, 20),
            (3, 10, 2, 20),
            2.0,
        ),
        ("edge_case_a", "params-decoder-scanned_blocks-mlp-wi_0-kernel", (3, 3, 3), (3, 3, 20), (3, 3, 20), 1.5),
        ("edge_case_b", "params-decoder-scanned_blocks-mlp-wo-kernel", (3, 3, 3), (3, 3, 20), (3, 3, 20), 1.5),
    ]

    for name, key, shape_a, shape_b, expected_shape, expected_val in cases:
      with self.subTest(name=name):
        state_dict = {
            f"{key}_lora_a": np.ones(shape_a, dtype=np.float32) * 0.5,
            f"{key}_lora_b": np.ones(shape_b, dtype=np.float32) * 0.5,
        }
        delta = _get_lora_delta(key, state_dict, 2.0)
        self.assertEqual(delta.shape, expected_shape)
        self.assertTrue(np.allclose(delta, expected_val))


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


class CheckpointMergingTest(unittest.TestCase):
  """Tests the recursive_update and load_orbax_checkpoint functions to ensure we don't overwrite weights."""

  def test_recursive_update(self):

    base = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel": np.ones((4, 4)),
                }
            }
        }
    }
    lora = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel_lora_a": np.ones((4, 2)),
                    "kernel_lora_b": np.ones((2, 4)),
                }
            }
        }
    }

    merged = {}
    _recursive_update(merged, base)
    _recursive_update(merged, lora)

    # Verify that both base and lora weights are present and not overwritten
    self.assertIn("kernel", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_a", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_b", merged["params"]["decoder"]["layers"])
    np.testing.assert_array_equal(merged["params"]["decoder"]["layers"]["kernel"], np.ones((4, 4)))
    np.testing.assert_array_equal(merged["params"]["decoder"]["layers"]["kernel_lora_a"], np.ones((4, 2)))
    np.testing.assert_array_equal(merged["params"]["decoder"]["layers"]["kernel_lora_b"], np.ones((2, 4)))

  @unittest.mock.patch("maxtext.checkpoint_conversion.utils.utils.ocp.Checkpointer")
  @unittest.mock.patch("maxtext.checkpoint_conversion.utils.utils.epath.Path")
  @unittest.mock.patch("maxtext.checkpoint_conversion.utils.utils.jax.devices")
  def test_load_orbax_checkpoint_recursive_merge(self, mock_jax_devices, mock_path, mock_checkpointer_cls):

    # Mock jax devices
    mock_jax_devices.return_value = [MagicMock()]

    # Mock Orbax Checkpointer and its restore results
    mock_ckptr = MagicMock()
    mock_checkpointer_cls.return_value = mock_ckptr

    # Base checkpoint metadata and content
    base_metadata = MagicMock()
    base_metadata.item_metadata.tree = {"params": {"decoder": {"layers": {"kernel": MagicMock(shape=(4, 4))}}}}
    base_restore_content = {"params": {"decoder": {"layers": {"kernel": np.ones((4, 4))}}}}

    # LoRA checkpoint metadata and content
    lora_metadata = MagicMock()
    lora_metadata.item_metadata.tree = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel_lora_a": MagicMock(shape=(4, 2)),
                    "kernel_lora_b": MagicMock(shape=(2, 4)),
                }
            }
        }
    }
    lora_restore_content = {
        "params": {
            "decoder": {
                "layers": {
                    "kernel_lora_a": np.ones((4, 2)),
                    "kernel_lora_b": np.ones((2, 4)),
                }
            }
        }
    }

    # Mock metadata and restore calls
    mock_ckptr.metadata.side_effect = [base_metadata, lora_metadata]
    mock_ckptr.restore.side_effect = [base_restore_content, lora_restore_content]

    # Create dummy config
    config = MagicMock()
    config.checkpoint_storage_concurrent_gb = 8
    config.checkpoint_storage_use_ocdbt = True
    config.checkpoint_storage_use_zarr3 = True
    config.load_parameters_path = "gs://base-bucket/checkpoints"
    config.lora.lora_restore_path = "gs://lora-bucket/checkpoints"

    # Load and merge
    merged = load_orbax_checkpoint(config)

    # Assert checkpointer was called twice and restored both
    self.assertEqual(mock_ckptr.restore.call_count, 2)

    # Verify that the keys are recursively merged correctly!
    self.assertIn("kernel", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_a", merged["params"]["decoder"]["layers"])
    self.assertIn("kernel_lora_b", merged["params"]["decoder"]["layers"])


class Gemma3And4CheckpointConversionTest(unittest.TestCase):
  """Explicitly tests Gemma 3 and Gemma 4 formats for base, adapter-only, and merged weight transformations."""

  def test_gemma3_base_and_adapter_conversion(self):
    # Gemma 3 configuration simulation
    # Scanned layers weight shapes
    # Query weight: [layers, input_dim, heads, head_dim]
    # For Gemma 3: 4D tensor for attention query
    key = "params-decoder-scanned_blocks-self_attention-query-kernel"
    a_key = key + "_lora_a"
    b_key = key + "_lora_b"

    # 4D scanned attention shapes: [num_layers, input_dim, heads, rank]
    # num_layers = 2, input_dim = 16, heads = 2, rank = 4, output_dim = 16
    data_a = np.ones((2, 16, 2, 4), dtype=np.float32) * 0.5
    data_b = np.ones((2, 4, 2, 16), dtype=np.float32) * 0.5

    param_map = {
        key: [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ]
    }
    lora_dict = {a_key: data_a, b_key: data_b}

    # 1. Test Adapter-only transformation
    weights, _ = _transform_weights_to_adapter(param_map, lora_dict)
    self.assertIn("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", weights)
    self.assertIn("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", weights)
    self.assertIn("base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight", weights)
    self.assertIn("base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight", weights)

    # 2. Test Delta contraction / Merged transformation math
    delta = _get_lora_delta(key, lora_dict, lora_scaling=2.0)
    # Expected delta shape matches original query weight shape: [2, 16, 2, 16]
    self.assertEqual(delta.shape, (2, 16, 2, 16))
    # Math: einsum("lipr,lrpo->lipo", A, B) * 2.0
    # For each slice: matmul(0.5, 0.5) * rank * scaling = 0.25 * 4 * 2.0 = 2.0
    self.assertTrue(np.allclose(delta, 2.0))

  def test_gemma4_base_and_adapter_conversion(self):
    # Gemma 4 configuration simulation
    # Scanned layers standard linear weight shapes (e.g. gate_proj, up_proj)
    # 3D scanned linear shape Case A: [num_layers, input_dim, rank] & [num_layers, rank, output_dim]
    key = "params-decoder-scanned_blocks-mlp-wi_0-kernel"
    a_key = key + "_lora_a"
    b_key = key + "_lora_b"

    # num_layers = 2, input_dim = 16, rank = 4, output_dim = 32
    data_a = np.ones((2, 16, 4), dtype=np.float32) * 0.5
    data_b = np.ones((2, 4, 32), dtype=np.float32) * 0.5

    param_map = {
        key: [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
        ]
    }
    lora_dict = {a_key: data_a, b_key: data_b}

    # 1. Test Adapter-only transformation
    weights, _ = _transform_weights_to_adapter(param_map, lora_dict)
    self.assertIn("base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight", weights)
    self.assertIn("base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight", weights)

    # 2. Test Delta contraction / Merged transformation math
    delta = _get_lora_delta(key, lora_dict, lora_scaling=2.0)
    # Expected delta shape matches original gate weight shape: [2, 16, 32]
    self.assertEqual(delta.shape, (2, 16, 32))
    # Math: einsum("lir,lro->lio", A, B) * 2.0 -> 0.25 * 4 * 2.0 = 2.0
    self.assertTrue(np.allclose(delta, 2.0))


if __name__ == "__main__":
  unittest.main()
