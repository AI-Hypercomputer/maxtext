# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for compressed attention."""

import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax import nnx
from maxtext.configs.pyconfig import initialize
from maxtext.layers import attention_compressed
from maxtext.layers.attention_compressed import (
    CompressedAttention,
    topk_indices_to_selection_mask_compare,
    topk_indices_to_selection_mask_vmap,
)
from tests.utils.test_helpers import get_test_config_path


def _reference_selection_mask(top_k_indices, compressed_len):
  """Broadcast-compare formulation: [batch, seq, k, compressed_len] reduced over k."""
  valid = top_k_indices >= 0
  entry_indices = jnp.arange(compressed_len)[None, None, :]
  is_in_topk = jnp.expand_dims(top_k_indices, axis=-1) == entry_indices[None, ...]
  is_valid_and_in_topk = is_in_topk & jnp.expand_dims(valid, axis=-1)
  return jnp.any(is_valid_and_in_topk, axis=2)


_SELECTION_MASK_FNS = (topk_indices_to_selection_mask_compare, topk_indices_to_selection_mask_vmap)


class TopkSelectionMaskTest(unittest.TestCase):
  """Both selection mask implementations match the broadcast-compare reference."""

  def _assert_matches_reference(self, top_k_indices, compressed_len):
    top_k_indices = jnp.asarray(top_k_indices, dtype=jnp.int32)
    expected = np.asarray(_reference_selection_mask(top_k_indices, compressed_len))
    for mask_fn in _SELECTION_MASK_FNS:
      for fn in (mask_fn, jax.jit(mask_fn, static_argnums=1)):
        with self.subTest(fn=mask_fn.__name__, jit=fn is not mask_fn):
          actual = np.asarray(fn(top_k_indices, compressed_len))
          self.assertEqual(actual.shape, top_k_indices.shape[:2] + (compressed_len,))
          self.assertEqual(actual.dtype, np.bool_)
          np.testing.assert_array_equal(actual, expected)

  def test_random_indices_with_padding_and_duplicates(self):
    batch, seq, k, compressed_len = 2, 16, 8, 12
    key_ids, key_pad = jax.random.split(jax.random.PRNGKey(0))
    # Ids drawn from a range smaller than k force duplicates within rows.
    ids = jax.random.randint(key_ids, (batch, seq, k), 0, compressed_len // 2)
    pad = jax.random.bernoulli(key_pad, 0.3, (batch, seq, k))
    self._assert_matches_reference(jnp.where(pad, -1, ids), compressed_len)

  def test_invalid_ids_select_nothing(self):
    compressed_len = 6
    # -1 must not wrap to the last block; other negatives and ids >= compressed_len are dropped.
    top_k_indices = [[[-1, -1, -1], [-3, 6, 100], [5, -1, 5], [0, 5, -6]]]
    self._assert_matches_reference(top_k_indices, compressed_len)
    for mask_fn in _SELECTION_MASK_FNS:
      with self.subTest(fn=mask_fn.__name__):
        mask = np.asarray(mask_fn(jnp.asarray(top_k_indices), compressed_len))
        np.testing.assert_array_equal(mask[0, 0], np.zeros(compressed_len, dtype=bool))
        np.testing.assert_array_equal(mask[0, 1], np.zeros(compressed_len, dtype=bool))
        np.testing.assert_array_equal(mask[0, 2], [False] * 5 + [True])
        np.testing.assert_array_equal(mask[0, 3], [True] + [False] * 4 + [True])

  def test_k_larger_than_compressed_len(self):
    batch, seq, k, compressed_len = 1, 8, 16, 4
    ids = jax.random.randint(jax.random.PRNGKey(1), (batch, seq, k), -1, compressed_len + 2)
    self._assert_matches_reference(ids, compressed_len)


class CompressedAttentionTest(unittest.TestCase):
  """Tests for Compressed Attention."""

  def setUp(self):
    self.config = self._make_config()
    self.mesh = Mesh(jax.devices(), ("data",))

  def _make_config(self, *overrides):
    return initialize(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek4-284b",
            "attention=dot_product",
            "qk_rope_head_dim=16",
            "v_head_dim=16",
            "qk_nope_head_dim=16",
            "override_model_config=True",
            *overrides,
        ]
    )

  def _make_layer(self, config):
    return CompressedAttention(
        config=config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=4,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

  def test_use_csa_vmap_selection_mask_selects_implementation(self):
    """The flag picks the selection mask implementation, and both give the same output."""
    key_q, key_kv = jax.random.split(jax.random.PRNGKey(0))
    q = jax.random.normal(key_q, (1, 32, 4096))
    kv = jax.random.normal(key_kv, (1, 32, 4096))
    pos = jnp.arange(32)[None, :]
    seg = jnp.zeros((1, 32), dtype=jnp.int32)

    outputs = {}
    for use_vmap in (False, True):
      layer = self._make_layer(self._make_config(f"use_csa_vmap_selection_mask={use_vmap}"))
      with (
          mock.patch.object(
              attention_compressed,
              "topk_indices_to_selection_mask_compare",
              wraps=topk_indices_to_selection_mask_compare,
          ) as compare,
          mock.patch.object(
              attention_compressed,
              "topk_indices_to_selection_mask_vmap",
              wraps=topk_indices_to_selection_mask_vmap,
          ) as vmap,
      ):
        outputs[use_vmap] = layer(q, kv, seg, pos, deterministic=True)
      self.assertEqual((compare.call_count > 0, vmap.call_count > 0), (not use_vmap, use_vmap))

    jax.tree.map(np.testing.assert_array_equal, outputs[False], outputs[True])

  def test_compressed_attention_jaxpr_tag_counts(self):
    layer = CompressedAttention(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=4,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

    q = jnp.ones((1, 32, 4096))
    kv = jnp.ones((1, 32, 4096))
    pos = jnp.arange(32)[None, :]
    seg = jnp.zeros((1, 32), dtype=jnp.int32)

    graphdef, state = nnx.split(layer)

    def forward(state, q, kv, seg, pos):
      layer = nnx.merge(graphdef, state)
      return layer(q, kv, seg, pos, deterministic=True)

    jaxpr = jax.make_jaxpr(forward)(state, q, kv, seg, pos)
    jaxpr_str = str(jaxpr)

    self.assertEqual(jaxpr_str.count("name=query_proj"), 1)
    self.assertEqual(jaxpr_str.count("name=kv_proj"), 1)
    self.assertEqual(jaxpr_str.count("name=attention_out"), 1)
    self.assertEqual(jaxpr_str.count("name=out_proj"), 1)

  def test_compressed_attention_no_double_tagging(self):
    layer = CompressedAttention(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=4,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

    q = jnp.ones((1, 32, 4096))
    kv = jnp.ones((1, 32, 4096))
    pos = jnp.arange(32)[None, :]
    seg = jnp.zeros((1, 32), dtype=jnp.int32)

    graphdef, state = nnx.split(layer)

    def forward(state, q, kv, seg, pos):
      layer = nnx.merge(graphdef, state)
      return layer(q, kv, seg, pos, deterministic=True)

    jaxpr = jax.make_jaxpr(forward)(state, q, kv, seg, pos)
    jaxpr_str = str(jaxpr)

    self.assertEqual(jaxpr_str.count("name=key_proj"), 0)
    self.assertEqual(jaxpr_str.count("name=value_proj"), 0)


if __name__ == "__main__":
  unittest.main()
