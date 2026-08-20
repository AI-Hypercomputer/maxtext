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

"""Unit tests for Gemma 4 small (E2B / E4B) layer-pattern helpers and rematerialization."""

import os
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common import common_types
from maxtext.common.common_types import AttentionType
from maxtext.configs import pyconfig
from maxtext.layers import embeddings
from maxtext.layers import nnx_decoders
from maxtext.models import gemma4_small
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


L = AttentionType.LOCAL_SLIDING
G = AttentionType.GLOBAL


class Gemma4SmallAttentionPatternTest(unittest.TestCase):
  """Per-variant attention-type pattern dispatch."""

  def test_e2b_attention_pattern_period_5(self):
    self.assertEqual(gemma4_small.get_attention_pattern("gemma4-e2b"), (L, L, L, L, G))

  def test_e4b_attention_pattern_period_6(self):
    self.assertEqual(gemma4_small.get_attention_pattern("gemma4-e4b"), (L, L, L, L, L, G))

  def test_default_pattern_period_6(self):
    self.assertEqual(len(gemma4_small.get_attention_pattern(None)), 6)


class Gemma4SmallLayerTypesTest(unittest.TestCase):
  """Per-layer attention-type list across the full stack."""

  def test_e2b_full_layer_types(self):
    layer_types = gemma4_small.build_layer_types(35, "gemma4-e2b")
    self.assertEqual(len(layer_types), 35)
    self.assertEqual(layer_types[0], L)
    # 35 = 7 * 5, so layers 4, 9, ..., 34 are GLOBAL.
    for i in range(4, 35, 5):
      self.assertEqual(layer_types[i], G)

  def test_e4b_full_layer_types(self):
    layer_types = gemma4_small.build_layer_types(42, "gemma4-e4b")
    self.assertEqual(len(layer_types), 42)
    for i in range(42):
      expected = G if i % 6 == 5 else L
      self.assertEqual(layer_types[i], expected, f"layer {i}")


class Gemma4SmallKvSharingTest(unittest.TestCase):
  """KV-sharing donor / shared-layer mapping."""

  def test_e2b_first_kv_shared_layer(self):
    self.assertEqual(gemma4_small.first_kv_shared_layer_idx(35, 20), 15)
    self.assertFalse(gemma4_small.is_kv_shared_layer(14, 35, 20))
    self.assertTrue(gemma4_small.is_kv_shared_layer(15, 35, 20))
    self.assertTrue(gemma4_small.is_kv_shared_layer(34, 35, 20))

  def test_e4b_first_kv_shared_layer(self):
    self.assertEqual(gemma4_small.first_kv_shared_layer_idx(42, 18), 24)
    self.assertFalse(gemma4_small.is_kv_shared_layer(23, 42, 18))
    self.assertTrue(gemma4_small.is_kv_shared_layer(24, 42, 18))

  def test_e2b_kv_donor_mapping(self):
    # E2B has 15 non-shared layers with pattern (L,L,L,L,G). Layer 13 is the
    # last LOCAL_SLIDING and layer 14 the last GLOBAL before sharing starts.
    layer_types = gemma4_small.build_layer_types(35, "gemma4-e2b")
    self.assertEqual(gemma4_small.kv_donor_layer_idx(15, layer_types, 20), 13)
    self.assertEqual(gemma4_small.kv_donor_layer_idx(19, layer_types, 20), 14)
    self.assertIsNone(gemma4_small.kv_donor_layer_idx(0, layer_types, 20))
    self.assertIsNone(gemma4_small.kv_donor_layer_idx(14, layer_types, 20))

  def test_e4b_kv_donor_mapping(self):
    # E4B has 24 non-shared layers with pattern (L,L,L,L,L,G). Layer 22 is
    # the last LOCAL_SLIDING and layer 23 the last GLOBAL before sharing.
    layer_types = gemma4_small.build_layer_types(42, "gemma4-e4b")
    self.assertEqual(gemma4_small.kv_donor_layer_idx(24, layer_types, 18), 22)
    self.assertEqual(gemma4_small.kv_donor_layer_idx(29, layer_types, 18), 23)
    self.assertIsNone(gemma4_small.kv_donor_layer_idx(23, layer_types, 18))

  def test_donor_layer_flag(self):
    layer_types = gemma4_small.build_layer_types(35, "gemma4-e2b")
    self.assertTrue(gemma4_small.is_kv_donor_layer(13, layer_types, 20))
    self.assertTrue(gemma4_small.is_kv_donor_layer(14, layer_types, 20))
    self.assertFalse(gemma4_small.is_kv_donor_layer(12, layer_types, 20))

  def test_no_kv_sharing_when_num_kv_shared_zero(self):
    layer_types = gemma4_small.build_layer_types(10, None)
    self.assertEqual(gemma4_small.first_kv_shared_layer_idx(10, 0), 10)
    for i in range(10):
      self.assertFalse(gemma4_small.is_kv_shared_layer(i, 10, 0))
      self.assertIsNone(gemma4_small.kv_donor_layer_idx(i, layer_types, 0))
      self.assertFalse(gemma4_small.is_kv_donor_layer(i, layer_types, 0))


class Gemma4SmallKvCacheSlotMapTest(unittest.TestCase):
  """Layer -> KV-cache slot mapping used by the vLLM RPA path."""

  def _check_slot_map(self, model_name, num_layers, num_kv_shared):
    """Asserts slot-map invariants for the given model layout."""
    layer_types = gemma4_small.build_layer_types(num_layers, model_name)
    slot_map = gemma4_small.kv_cache_slot_map(layer_types, num_kv_shared)

    num_slots = num_layers - num_kv_shared
    self.assertEqual(len(slot_map), num_layers)
    # Non-shared layers get consecutive slots 0..num_slots-1.
    self.assertEqual([slot_map[i] for i in range(num_slots)], list(range(num_slots)))
    # Shared layers reuse the slot of a donor with the same attention type.
    for lyr in range(num_slots, num_layers):
      donor = gemma4_small.kv_donor_layer_idx(lyr, layer_types, num_kv_shared)
      self.assertEqual(slot_map[lyr], slot_map[donor], f"layer {lyr}")
      self.assertEqual(layer_types[lyr], layer_types[donor], f"layer {lyr}")

  def test_e2b_slot_map(self):
    self._check_slot_map("gemma4-e2b", 35, 20)

  def test_e4b_slot_map(self):
    self._check_slot_map("gemma4-e4b", 42, 18)

  def test_slot_map_without_sharing_is_identity(self):
    layer_types = gemma4_small.build_layer_types(10, None)
    slot_map = gemma4_small.kv_cache_slot_map(layer_types, 0)
    self.assertEqual(slot_map, {i: i for i in range(10)})


class Gemma4SmallDecoderRematTest(unittest.TestCase):
  """Verify that NNXDecoder applies rematerialization per layer for gemma4-small."""

  _BASE_CONFIG_PATH = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  _NUM_LAYERS = 10
  _NUM_KV_SHARED = 5  # Last 5 of 10 layers share K/V (1 full period of pattern).
  _NUM_Q_HEADS = 4
  _NUM_KV_HEADS = 1
  _HEAD_DIM = 32
  _GLOBAL_HEAD_DIM = 64
  _HIDDEN_SIZE = 128
  _PLE_DIM = 32
  _VOCAB = 256

  def _build_jax_config(self, remat_policy="none"):
    """Builds a small E2B-shaped MaxText config with trimmed dimensions for fast tests."""
    return pyconfig.initialize(
        ["", self._BASE_CONFIG_PATH],
        model_name="gemma4-e2b",
        remat_policy=remat_policy,
        scan_layers=False,
        use_multimodal=False,
        override_model_config=True,
        # Override shapes for fast tests:
        base_num_decoder_layers=self._NUM_LAYERS,
        base_num_query_heads=self._NUM_Q_HEADS,
        base_num_kv_heads=self._NUM_KV_HEADS,
        base_emb_dim=self._HIDDEN_SIZE,
        base_mlp_dim=4 * self._HIDDEN_SIZE,
        head_dim=self._HEAD_DIM,
        global_head_dim=self._GLOBAL_HEAD_DIM,
        vocab_size=self._VOCAB,
        vocab_size_per_layer_input=self._VOCAB,
        hidden_size_per_layer_input=self._PLE_DIM,
        num_kv_shared_layers=self._NUM_KV_SHARED,
        max_target_length=64,
        max_prefill_predict_length=8,
        attention="dot_product",  # avoid splash on CPU
        dtype="float32",
        weight_dtype="float32",
        float32_qk_product=True,
        float32_logits=True,
        matmul_precision="highest",
        dropout_rate=0.0,
    )

  def test_gemma4_small_decoder_remat(self):
    cfg_no_remat = self._build_jax_config(remat_policy="none")
    cfg_remat = self._build_jax_config(remat_policy="full")

    mesh = Mesh(np.array(jax.devices()), axis_names=("x",))
    rngs = nnx.Rngs(0)

    decoder_no_remat = nnx_decoders.NNXDecoder(config=cfg_no_remat, mesh=mesh, rngs=rngs)
    decoder_remat = nnx_decoders.NNXDecoder(config=cfg_remat, mesh=mesh, rngs=rngs)
    nnx.update(decoder_remat, nnx.state(decoder_no_remat))

    embed = embeddings.Embed(
        num_embeddings=cfg_no_remat.vocab_size,
        num_features=cfg_no_remat.emb_dim,
        dtype=cfg_no_remat.dtype,
        config=cfg_no_remat,
        mesh=mesh,
        rngs=rngs,
    )

    tokens = jax.random.randint(jax.random.key(1), (2, 8), 0, self._VOCAB)
    positions = jnp.arange(8)[None, :]

    def loss_fn(model):
      out, *_ = model(
          embed,
          tokens,
          decoder_positions=positions,
          deterministic=True,
          model_mode=common_types.MODEL_MODE_TRAIN,
      )
      return jnp.sum(out)

    # Spy on _apply_layer_with_remat (which wraps each layer in jax.checkpoint).
    # When rematerialization is disabled ('none'), layers are called directly
    # without checkpointing (call count is 0).
    with mock.patch.object(
        decoder_no_remat,
        "_apply_layer_with_remat",
        wraps=decoder_no_remat._apply_layer_with_remat,  # pylint: disable=protected-access)
    ) as spy_no_remat:
      loss_no_remat, grad_no_remat = nnx.value_and_grad(loss_fn)(decoder_no_remat)
      self.assertEqual(spy_no_remat.call_count, 0)

    # When rematerialization is enabled ('full'), every unscanned decoder layer
    # must be wrapped in jax.checkpoint via _apply_layer_with_remat once per layer.
    with mock.patch.object(
        decoder_remat,
        "_apply_layer_with_remat",
        wraps=decoder_remat._apply_layer_with_remat,  # pylint: disable=protected-access)
    ) as spy_remat:
      loss_remat, grad_remat = nnx.value_and_grad(loss_fn)(decoder_remat)
      self.assertEqual(spy_remat.call_count, self._NUM_LAYERS)

    # Rematerialization should not alter outputs or gradients.
    np.testing.assert_allclose(loss_no_remat, loss_remat, rtol=1e-4, atol=1e-4)
    jax.tree.map(
        lambda g1, g2: np.testing.assert_allclose(g1, g2, rtol=1e-4, atol=1e-4),
        nnx.state(grad_no_remat),
        nnx.state(grad_remat),
    )


if __name__ == "__main__":
  unittest.main()
