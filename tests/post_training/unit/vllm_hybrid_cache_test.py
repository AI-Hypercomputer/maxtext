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
    gather_layer_kv_caches,
    normalize_vllm_input_positions,
    resolve_layer_kv_cache_indices,
    scatter_layer_kv_caches,
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


if __name__ == "__main__":
  unittest.main()


class LayerOrderedKvCachesTest(unittest.TestCase):
  """Verify the physical-slot -> decoder-layer remapping of vLLM's kv_caches list."""

  @pytest.mark.cpu_only
  def test_type_grouped_hybrid_layout_is_remapped_to_layer_order(self):
    # Qwen3.5-style 4-layer cycle: layers 0-2 are GDN, layer 3 is attention.
    # tpu-inference's hybrid tensor layout groups caches by type, so the
    # physical list is not in layer order; here the attention cache leads.
    kv_caches = ["attn3", "mamba0", "mamba1", "mamba2"]
    index_map = (("layer.0", 1), ("layer.1", 2), ("layer.2", 3), ("layer.3", 0))

    physical_indices = resolve_layer_kv_cache_indices(index_map, len(kv_caches))
    layer_view = gather_layer_kv_caches(kv_caches, physical_indices)

    self.assertEqual(physical_indices, [1, 2, 3, 0])
    self.assertEqual(layer_view, ["mamba0", "mamba1", "mamba2", "attn3"])

  @pytest.mark.cpu_only
  def test_scatter_restores_physical_order_and_passes_through_unmapped_slots(self):
    kv_caches = ["attn3", "mamba0", "mamba1", "mamba2", "aux"]
    physical_indices = [1, 2, 3, 0]
    updated_layer_view = ["mamba0'", "mamba1'", "mamba2'", "attn3'"]

    actual = scatter_layer_kv_caches(kv_caches, updated_layer_view, physical_indices)

    self.assertEqual(actual, ["attn3'", "mamba0'", "mamba1'", "mamba2'", "aux"])
    # The runner's list is not mutated in place.
    self.assertEqual(kv_caches, ["attn3", "mamba0", "mamba1", "mamba2", "aux"])

  @pytest.mark.cpu_only
  def test_shared_physical_cache_is_visible_to_every_sharing_layer(self):
    # KV-sharing: layer 2 redirects to layer 1's cache; only 2 physical caches.
    kv_caches = ["c0", "c1"]
    index_map = {"layer.0": 0, "layer.1": 1, "layer.2": 1}

    physical_indices = resolve_layer_kv_cache_indices(index_map, len(kv_caches))
    layer_view = gather_layer_kv_caches(kv_caches, physical_indices)
    scattered = scatter_layer_kv_caches(kv_caches, ["c0'", "c1'", "c1''"], physical_indices)

    self.assertEqual(layer_view, ["c0", "c1", "c1"])
    # The highest-numbered sharing layer's update wins.
    self.assertEqual(scattered, ["c0'", "c1''"])

  @pytest.mark.cpu_only
  def test_missing_map_keeps_positional_indexing(self):
    kv_caches = ["c0", "c1"]

    self.assertIsNone(resolve_layer_kv_cache_indices(None, 2))
    self.assertIsNone(resolve_layer_kv_cache_indices((), 2))
    self.assertIs(gather_layer_kv_caches(kv_caches, None), kv_caches)
    self.assertEqual(scatter_layer_kv_caches(kv_caches, ["c0'", "c1'"], None), ["c0'", "c1'"])

  @pytest.mark.cpu_only
  def test_non_layer_entries_are_ignored(self):
    index_map = {"layer.0": 0, "layer.1": 1, "layer.1.rope_cache": 2, 42: 3}

    self.assertEqual(resolve_layer_kv_cache_indices(index_map, 3), [0, 1])

  @pytest.mark.cpu_only
  def test_rejects_gaps_and_out_of_range_indices(self):
    with self.assertRaisesRegex(ValueError, r"missing decoder layers \[1\]"):
      resolve_layer_kv_cache_indices({"layer.0": 0, "layer.2": 1}, 2)
    with self.assertRaisesRegex(ValueError, "outside the kv_caches list"):
      resolve_layer_kv_cache_indices({"layer.0": 0, "layer.1": 5}, 2)

  @pytest.mark.cpu_only
  def test_scatter_rejects_length_mismatch(self):
    with self.assertRaisesRegex(ValueError, "Expected layer_kv_caches to have length"):
      scatter_layer_kv_caches(["c0", "c1"], ["c0'"], [0, 1])
