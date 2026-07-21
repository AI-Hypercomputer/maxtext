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

"""Unit tests for the Gemma scanned weights unrolling workaround."""

import unittest
import numpy as np
import pytest

from maxtext.integration.vllm.maxtext_vllm_rollout import unroll_gemma_scanned_weights


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
