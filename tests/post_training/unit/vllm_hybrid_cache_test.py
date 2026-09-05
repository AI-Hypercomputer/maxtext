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

"""Tests for MaxText vLLM hybrid-cache layout helpers."""

from types import SimpleNamespace
import unittest

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from maxtext.integration.vllm.hybrid_cache_utils import (
    build_qwen_gdn_cache_layout,
    map_layer_names_to_indices,
    normalize_vllm_input_positions,
)


pytestmark = [pytest.mark.post_training]


class QwenGdnCacheLayoutTest(unittest.TestCase):
  """Verify the mixed-precision recurrent-cache contract."""

  @pytest.mark.cpu_only
  def test_recurrent_state_is_float32_and_page_size_uses_each_dtype(self):
    cfg = SimpleNamespace(
        gdn_num_value_heads=32,
        gdn_num_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel_dim=4,
    )

    shapes, dtypes, page_size_bytes = build_qwen_gdn_cache_layout(cfg, torch)

    self.assertEqual(shapes, ((3, 8192), (32, 128, 128)))
    self.assertEqual(dtypes, (torch.bfloat16, torch.float32))
    self.assertEqual(page_size_bytes, 2_146_304)


class VllmInputPositionsTest(unittest.TestCase):
  """Verify vLLM position layouts are converted to MaxText decode layouts."""

  @pytest.mark.cpu_only
  def test_flat_positions_gain_singleton_sequence_dimension(self):
    positions = jnp.array([2, 5, 9], dtype=jnp.int32)

    actual = normalize_vllm_input_positions(positions)

    np.testing.assert_array_equal(actual, np.array([[2], [5], [9]], dtype=np.int32))

  @pytest.mark.cpu_only
  def test_mrope_positions_move_channels_to_trailing_dimension(self):
    positions = jnp.array(
        [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=jnp.int32,
    )

    actual = normalize_vllm_input_positions(positions)

    expected = np.array(
        [
            [[0, 10, 20]],
            [[1, 11, 21]],
            [[2, 12, 22]],
            [[3, 13, 23]],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(actual, expected)

  @pytest.mark.cpu_only
  def test_rejects_unknown_position_layout(self):
    with self.assertRaisesRegex(ValueError, r"got \(2, 4\)"):
      normalize_vllm_input_positions(jnp.zeros((2, 4), dtype=jnp.int32))


class MapLayerNamesToIndicesTest(unittest.TestCase):
  """Verify layer name to cache index mappings for hybrid models."""

  @pytest.mark.cpu_only
  def test_none_or_empty_returns_empty_dict(self):
    self.assertEqual(map_layer_names_to_indices(None), {})
    self.assertEqual(map_layer_names_to_indices([]), {})
    self.assertEqual(map_layer_names_to_indices({}), {})
    self.assertEqual(map_layer_names_to_indices("not-a-dict"), {})

  @pytest.mark.cpu_only
  def test_maps_tuple_pairs_from_vllm_layer_names(self):
    input_pairs = [
        ("layer.0", 0),
        ("layer.1", 1),
        ("layer.2", 2),
        ("layer.3", 30),
        ("layer.7", 31),
    ]
    expected = {0: 0, 1: 1, 2: 2, 3: 30, 7: 31}
    self.assertEqual(map_layer_names_to_indices(input_pairs), expected)

  @pytest.mark.cpu_only
  def test_maps_module_names_with_linear_and_self_attn(self):
    input_dict = {
        "model.layers.0.linear_attn": 0,
        "model.layers.1.linear_attn": 1,
        "model.layers.3.self_attn": 30,
        "model.layers.7.self_attn": 31,
    }
    expected = {0: 0, 1: 1, 3: 30, 7: 31}
    self.assertEqual(map_layer_names_to_indices(input_dict), expected)

  @pytest.mark.cpu_only
  def test_maps_numeric_keys(self):
    self.assertEqual(map_layer_names_to_indices({0: 0, 3: 30}), {0: 0, 3: 30})
    self.assertEqual(map_layer_names_to_indices({"0": 0, "3": 30}), {0: 0, 3: 30})


if __name__ == "__main__":
  unittest.main()
