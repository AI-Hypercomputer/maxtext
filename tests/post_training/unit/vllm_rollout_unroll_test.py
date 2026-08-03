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

"""Unit tests for MaxText scanned-weight unrolling workarounds."""

from types import SimpleNamespace
import unittest
import numpy as np
import pytest

from maxtext.integration.vllm.maxtext_vllm_rollout import (
    requires_maxtext_scanned_weight_unroll,
    unroll_gemma_scanned_weights,
    unroll_qwen_scanned_weights,
    uses_maxtext_vllm_adapter,
    validate_direct_sync_layer_coverage,
)


class MockWeights:
  """A mock weight container that implements to_pure_dict."""

  def __init__(self, pure_dict):
    self._pure_dict = pure_dict

  def to_pure_dict(self):
    return self._pure_dict


class GemmaScannedWeightsUnrollTest(unittest.TestCase):
  """Verify the correctness of the unroll_gemma_scanned_weights utility."""

  @pytest.mark.cpu_only
  def test_bypasses_non_pytree_weights(self):
    """If the weights object doesn't have `to_pure_dict`, it should be returned unchanged."""
    raw_weights = {"dummy": np.ones(5)}
    result = unroll_gemma_scanned_weights(raw_weights)
    self.assertIs(result, raw_weights)

  @pytest.mark.cpu_only
  def test_bypasses_non_scanned_checkpoints(self):
    """If the checkpoint is not scanned (no 'layers_0' inside 'decoder/layers/'), return unchanged."""
    pure_dict = {
        "decoder": {
            "layers": {
                "0": {"attn": {"wq": np.ones(10)}},
                "1": {"attn": {"wq": np.ones(10)}},
            }
        }
    }
    weights = MockWeights(pure_dict)
    result = unroll_gemma_scanned_weights(weights)
    self.assertIs(result, weights)

  @pytest.mark.cpu_only
  def test_correctly_unrolls_gemma_scanned_weights(self):
    """Verify that scanned layers are properly interleaved and mapped, and remainder layers are appended."""
    # Pattern length = 2 (layers_0 and layers_1)
    # Scan length = 3. In MaxText, param_scan_axis=1, so shape is (feature_dim, scan_length, ...)

    # We want an array where axis 1 has length 3. Let's make it (2, 3, 1)
    # For layers_0, values should be 0, 2, 4
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    # For layers_1, values should be 1, 3, 5
    arr1 = np.zeros((2, 3, 1))
    arr1[:, 0, :] = 1
    arr1[:, 1, :] = 3
    arr1[:, 2, :] = 5

    pure_dict = {
        "decoder": {
            "layers": {
                "layers_0": {
                    "attn": {"wq": arr0},
                },
                "layers_1": {
                    "attn": {"wq": arr1},
                },
            },
            "layers_remainder": {
                "layers_0": {
                    "attn": {"wq": np.array([[6, 6]]).transpose()},  # shape (2, 1)
                }
            },
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_gemma_scanned_weights(weights)

    # Check unrolled structure
    decoder_dict = unrolled["decoder"]

    # Should contain keys 0 to 6 under layers
    self.assertIn(0, decoder_dict["layers"])
    self.assertIn(1, decoder_dict["layers"])
    self.assertIn(2, decoder_dict["layers"])
    self.assertIn(3, decoder_dict["layers"])
    self.assertIn(4, decoder_dict["layers"])
    self.assertIn(5, decoder_dict["layers"])
    self.assertIn(6, decoder_dict["layers"])

    self.assertIsInstance(list(decoder_dict["layers"].keys())[0], int)

    # Check that values are correctly sliced
    np.testing.assert_array_equal(decoder_dict["layers"][0]["attn"]["wq"], np.array([[0], [0]]))
    np.testing.assert_array_equal(decoder_dict["layers"][1]["attn"]["wq"], np.array([[1], [1]]))
    np.testing.assert_array_equal(decoder_dict["layers"][2]["attn"]["wq"], np.array([[2], [2]]))
    np.testing.assert_array_equal(decoder_dict["layers"][3]["attn"]["wq"], np.array([[3], [3]]))
    np.testing.assert_array_equal(decoder_dict["layers"][4]["attn"]["wq"], np.array([[4], [4]]))
    np.testing.assert_array_equal(decoder_dict["layers"][5]["attn"]["wq"], np.array([[5], [5]]))
    np.testing.assert_array_equal(decoder_dict["layers"][6]["attn"]["wq"], np.array([[6], [6]]))

  @pytest.mark.cpu_only
  def test_correctly_unrolls_gemma3_gemma4_scanned_blocks(self):
    """Verify that scanned layers under scanned_blocks are properly interleaved and mapped."""
    arr0 = np.zeros((2, 3, 1))
    arr0[:, 0, :] = 0
    arr0[:, 1, :] = 2
    arr0[:, 2, :] = 4

    arr1 = np.zeros((2, 3, 1))
    arr1[:, 0, :] = 1
    arr1[:, 1, :] = 3
    arr1[:, 2, :] = 5

    pure_dict = {
        "decoder": {
            "scanned_blocks": {
                "layers_0": {
                    "attn": {"wq": arr0},
                },
                "layers_1": {
                    "attn": {"wq": arr1},
                },
            },
            "layers_remainder": {
                "layers_0": {
                    "attn": {"wq": np.array([[6, 6]]).transpose()},
                }
            },
        }
    }
    weights = MockWeights(pure_dict)
    unrolled = unroll_gemma_scanned_weights(weights)

    decoder_dict = unrolled["decoder"]
    self.assertIn(0, decoder_dict["layers"])
    self.assertIn(6, decoder_dict["layers"])
    self.assertIsInstance(list(decoder_dict["layers"].keys())[0], int)
    np.testing.assert_array_equal(decoder_dict["layers"][0]["attn"]["wq"], np.array([[0], [0]]))
    np.testing.assert_array_equal(decoder_dict["layers"][6]["attn"]["wq"], np.array([[6], [6]]))


class QwenScannedWeightsUnrollTest(unittest.TestCase):
  """Verify heterogeneous Qwen blocks map to unscanned decoder attributes."""

  @pytest.mark.cpu_only
  def test_interleaves_slots_and_repetitions(self):
    slot_0 = np.zeros((2, 2, 1), dtype=np.float32)
    slot_0[:, 0, :] = 0
    slot_0[:, 1, :] = 2
    slot_1 = np.zeros((2, 2, 1), dtype=np.float32)
    slot_1[:, 0, :] = 1
    slot_1[:, 1, :] = 3
    weights = MockWeights(
        {
            "base": {
                "decoder": {
                    "layers": {
                        "layer_0": {"probe": slot_0},
                        "layer_1": {"probe": slot_1, "rngs": {"key": np.ones(2, dtype=np.uint32)}},
                    },
                    "decoder_norm": {"scale": np.ones(2)},
                }
            }
        }
    )

    unrolled = unroll_qwen_scanned_weights(weights)

    decoder = unrolled["base"]["decoder"]
    for layer_idx in range(4):
      self.assertIn(f"layers_{layer_idx}", decoder)
      np.testing.assert_array_equal(
          decoder[f"layers_{layer_idx}"]["probe"],
          np.full((2, 1), layer_idx, dtype=np.float32),
      )
    np.testing.assert_array_equal(decoder["decoder_norm"]["scale"], np.ones(2))
    self.assertNotIn("probe", decoder["layers"]["layer_1"])
    np.testing.assert_array_equal(decoder["layers"]["layer_1"]["rngs"]["key"], np.ones(2, dtype=np.uint32))
    target = {
        "model": {
            "decoder": {
                f"layers_{layer_idx}": {"probe": np.zeros((2, 1), dtype=np.float32)} for layer_idx in range(4)
            }
        }
    }
    self.assertEqual(validate_direct_sync_layer_coverage(unrolled, target), 4)

  @pytest.mark.cpu_only
  def test_rejects_inconsistent_scan_lengths(self):
    weights = MockWeights(
        {
            "decoder": {
                "layers": {
                    "layer_0": {"probe": np.ones((2, 2, 1))},
                    "layer_1": {"probe": np.ones((2, 3, 1))},
                }
            }
        }
    )

    with self.assertRaisesRegex(ValueError, "disagree on scan length"):
      unroll_qwen_scanned_weights(weights)

  @pytest.mark.cpu_only
  def test_supports_nondefault_axis_and_sparse_slots(self):
    slot_1 = np.stack(
        [np.full((2, 1), 1, dtype=np.float32), np.full((2, 1), 3, dtype=np.float32)],
        axis=0,
    )
    weights = MockWeights({"decoder": {"layers": {"layer_1": {"probe": slot_1}}}})

    unrolled = unroll_qwen_scanned_weights(weights, scan_axis=0, pattern_length=2)

    self.assertNotIn("layers_0", unrolled["decoder"])
    np.testing.assert_array_equal(unrolled["decoder"]["layers_1"]["probe"], np.full((2, 1), 1))
    np.testing.assert_array_equal(unrolled["decoder"]["layers_3"]["probe"], np.full((2, 1), 3))

  @pytest.mark.cpu_only
  def test_rejects_missing_target_layer_parameters(self):
    source = {"base": {"decoder": {"layers_0": {"probe": np.ones((2, 1))}}}}
    target = {
        "model": {
            "decoder": {
                "layers_0": {"probe": np.zeros((2, 1))},
                "layers_1": {"probe": np.zeros((2, 1))},
            }
        }
    }

    with self.assertRaisesRegex(ValueError, "leave rollout transformer parameters at random initialization"):
      validate_direct_sync_layer_coverage(source, target)


class MaxTextAdapterSelectionTest(unittest.TestCase):
  """Verify only scanned MaxText-adapter rollouts take the custom sync path."""

  @pytest.mark.cpu_only
  def test_detects_dict_and_string_overrides(self):
    dict_config = SimpleNamespace(vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]}, scan_layers=True)
    string_config = SimpleNamespace(vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}', scan_layers=True)

    self.assertTrue(uses_maxtext_vllm_adapter(dict_config))
    self.assertTrue(uses_maxtext_vllm_adapter(string_config))
    self.assertTrue(requires_maxtext_scanned_weight_unroll(dict_config))

  @pytest.mark.cpu_only
  def test_bypasses_native_or_unscanned_rollouts(self):
    native_config = SimpleNamespace(
        vllm_hf_overrides={"architectures": ["Qwen3_5MoeForConditionalGeneration"]}, scan_layers=True
    )
    unscanned_config = SimpleNamespace(vllm_hf_overrides={"architectures": ["MaxTextForCausalLM"]}, scan_layers=False)

    self.assertFalse(requires_maxtext_scanned_weight_unroll(native_config))
    self.assertFalse(requires_maxtext_scanned_weight_unroll(unscanned_config))
