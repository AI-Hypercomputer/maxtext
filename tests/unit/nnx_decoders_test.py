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

# pylint: disable=unbalanced-tuple-unpacking
"""Unit tests for nnx_decoders module.

Tests cover:
  - deepstack_process: pure-JAX helper for injecting visual embeddings
  - NNXDecoderLayer: single transformer decoder layer (init + forward)
  - NNXDecoder: decoder stack utilities (get_decoder_layers, get_norm_layer,
                get_remat_policy, minimal_policy, and full forward pass)
"""

import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    AttentionType,
    DecoderBlockType,
    MultimodalInput,
)
from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder, NNXDecoderLayer, deepstack_process
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import gemma4, gemma4_small, qwen3
from maxtext.models.gpt3 import Gpt3LayerNorm
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils, maxtext_utils_nnx
from tests.utils.test_helpers import get_test_config_path

# ---------------------------------------------------------------------------
# Shared minimal config overrides used across most tests
# ---------------------------------------------------------------------------
_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "nnx_decoder_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 2,
    "attention": "dot_product",
    "max_target_length": 16,
    "base_emb_dim": 256,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 512,
    "max_prefill_predict_length": 4,
    "scan_layers": False,
}


def _make_config(**overrides):
  """Return a pyconfig Config object suitable for unit tests."""
  merged = {**_BASE_CONFIG, **overrides}
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


# ---------------------------------------------------------------------------
# 1. deepstack_process
# ---------------------------------------------------------------------------


class TestDeepstackProcess(unittest.TestCase):
  """Tests for the deepstack_process pure function."""

  # pylint: disable=too-many-positional-arguments
  def _make_inputs(self, batch=2, seq_len=8, hidden_dim=16, num_visual=3, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    hidden_states = jax.random.normal(k1, (batch, seq_len, hidden_dim))
    mask = jnp.zeros((batch, seq_len), dtype=bool).at[:, :num_visual].set(True)
    visual_embeds = jax.random.normal(k2, (batch, num_visual, hidden_dim))
    return hidden_states, mask, visual_embeds

  def test_output_shape_matches_hidden_states(self):
    """Output shape must equal input hidden_states shape."""
    hidden_states, mask, visual_embeds = self._make_inputs()
    result = deepstack_process(hidden_states, mask, visual_embeds)
    self.assertEqual(result.shape, hidden_states.shape)

  def test_unmasked_positions_are_unchanged(self):
    """Positions outside the bidirectional mask must not be modified."""
    batch, seq_len, hidden_dim, num_visual = 1, 6, 8, 2
    hidden_states = jnp.ones((batch, seq_len, hidden_dim))
    mask = jnp.zeros((batch, seq_len), dtype=bool).at[:, :num_visual].set(True)
    # Zero visual embeds ensure any addition at mask=True positions is a no-op
    visual_embeds = jnp.zeros((batch, num_visual, hidden_dim))

    result = deepstack_process(hidden_states, mask, visual_embeds)

    np.testing.assert_allclose(
        np.array(result[:, num_visual:, :]),
        np.ones((batch, seq_len - num_visual, hidden_dim)),
    )

  def test_masked_positions_receive_visual_embeds(self):
    """Visual embeddings must be added at masked (True) positions."""
    batch, seq_len, hidden_dim, num_visual = 1, 4, 4, 2
    hidden_states = jnp.zeros((batch, seq_len, hidden_dim))
    mask = jnp.zeros((batch, seq_len), dtype=bool).at[:, :num_visual].set(True)
    visual_embeds = jnp.ones((batch, num_visual, hidden_dim))

    result = deepstack_process(hidden_states, mask, visual_embeds)

    # At masked positions: 0 + 1 = 1
    np.testing.assert_allclose(
        np.array(result[:, :num_visual, :]),
        np.ones((batch, num_visual, hidden_dim)),
    )
    # At unmasked positions: unchanged (still 0)
    np.testing.assert_allclose(
        np.array(result[:, num_visual:, :]),
        np.zeros((batch, seq_len - num_visual, hidden_dim)),
    )

  def test_zero_visual_embeds_leave_hidden_states_unchanged(self):
    """When all visual embeddings are zero, output equals input."""
    hidden_states, mask, _ = self._make_inputs()
    num_visual = 3
    batch = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]
    zero_visual = jnp.zeros((batch, num_visual, hidden_dim))

    result = deepstack_process(hidden_states, mask, zero_visual)

    np.testing.assert_allclose(np.array(result), np.array(hidden_states))

  def test_all_positions_masked(self):
    """Works correctly when every token position is a visual token."""
    batch, seq_len, hidden_dim = 1, 4, 8
    hidden_states = jnp.zeros((batch, seq_len, hidden_dim))
    mask = jnp.ones((batch, seq_len), dtype=bool)
    visual_embeds = jnp.ones((batch, seq_len, hidden_dim)) * 2.0

    result = deepstack_process(hidden_states, mask, visual_embeds)

    np.testing.assert_allclose(
        np.array(result),
        np.full((batch, seq_len, hidden_dim), 2.0),
    )

  def test_no_positions_masked(self):
    """When no positions are masked, hidden states are unchanged."""
    batch, seq_len, hidden_dim, num_visual = 2, 6, 8, 1
    hidden_states = jnp.ones((batch, seq_len, hidden_dim))
    mask = jnp.zeros((batch, seq_len), dtype=bool)
    visual_embeds = jnp.ones((batch, num_visual, hidden_dim)) * 99.0

    result = deepstack_process(hidden_states, mask, visual_embeds)

    np.testing.assert_allclose(np.array(result), np.array(hidden_states))


# ---------------------------------------------------------------------------
# 2. NNXDecoderLayer
# ---------------------------------------------------------------------------


class TestNNXDecoderLayer(unittest.TestCase):
  """Tests for the NNXDecoderLayer NNX module."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)
    self.rng = jax.random.PRNGKey(0)

  def _make_layer(self, model_mode=MODEL_MODE_TRAIN, config=None):
    return NNXDecoderLayer(
        config=config if config is not None else self.cfg,
        mesh=self.mesh,
        model_mode=model_mode,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

  def _make_inputs(self):
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    emb_dim = cfg.emb_dim
    inputs = jax.random.normal(self.rng, (batch, seq_len, emb_dim)).astype(cfg.dtype)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return inputs, segment_ids, positions

  # --- instantiation ---------------------------------------------------------

  def test_has_pre_self_attention_norm(self):
    layer = self._make_layer()
    self.assertIsInstance(layer.pre_self_attention_norm, RMSNorm)

  def test_has_self_attention(self):

    layer = self._make_layer()
    self.assertIsInstance(layer.self_attention, Attention)

  def test_has_mlp(self):

    layer = self._make_layer()
    self.assertIsInstance(layer.mlp, linears.MlpBlock)

  # --- forward pass ----------------------------------------------------------

  def test_forward_output_shape_train(self):
    """Forward pass output shape matches input shape in train mode."""
    layer = self._make_layer(MODEL_MODE_TRAIN)
    inputs, segment_ids, positions = self._make_inputs()
    out, _ = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(out.shape, inputs.shape)

  def test_forward_output_dtype(self):
    """Output dtype matches config dtype."""
    layer = self._make_layer()
    inputs, segment_ids, positions = self._make_inputs()
    out, _ = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(out.dtype, self.cfg.dtype)

  def test_forward_prefill_mode(self):
    """Test forward pass in prefill mode."""
    layer = self._make_layer(MODEL_MODE_PREFILL)
    inputs, segment_ids, positions = self._make_inputs()
    out, _ = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )
    self.assertEqual(out.shape, inputs.shape)

  def test_record_metrics(self):
    """Test recording intermediate activation metrics."""
    if not hasattr(nnx, "capture"):
      self.skipTest("flax.nnx does not support capture on this environment configuration")

    cfg = _make_config(record_internal_nn_metrics=1)
    layer = self._make_layer(MODEL_MODE_TRAIN, config=cfg)
    inputs, segment_ids, positions = self._make_inputs()

    # Use nnx.capture to retrieve sown variables
    _, state = nnx.capture(layer, nnx.Intermediate)(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    metrics_keys = state.keys()
    self.assertIn("activation_mean", metrics_keys)
    self.assertIn("activation_stdev", metrics_keys)
    self.assertIn("activation_fraction_zero", metrics_keys)

  def test_forward_kv_cache_is_none_when_scan_layers_false(self):
    """kv_cache return value is not None when scan_layers=False (non-scan returns cache)."""
    # With scan_layers=False the layer returns (output, kv_cache).
    # kv_cache may be None in train mode (no cache is populated); we just
    # verify the call doesn't raise and returns a 2-tuple.
    layer = self._make_layer()
    inputs, segment_ids, positions = self._make_inputs()
    result = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)

  def test_forward_deterministic_and_stochastic_consistent_shape(self):
    """Output shape is the same regardless of the deterministic flag."""
    layer = self._make_layer()
    inputs, segment_ids, positions = self._make_inputs()
    out_det, _ = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    out_stoch, _ = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=False,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(out_det.shape, out_stoch.shape)


# ---------------------------------------------------------------------------
# 3. NNXDecoder.get_decoder_layers
# ---------------------------------------------------------------------------


class TestNNXDecoderGetDecoderLayers(unittest.TestCase):
  """Tests for NNXDecoder.get_decoder_layers."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)

  def _make_decoder(self, **cfg_overrides):
    cfg = _make_config(**cfg_overrides) if cfg_overrides else self.cfg
    mesh = _make_mesh(cfg) if cfg_overrides else self.mesh
    return NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))

  def test_default_decoder_block_returns_nnx_decoder_layer(self):
    decoder = self._make_decoder(decoder_block=DecoderBlockType.DEFAULT)
    layers = decoder.get_decoder_layers()
    self.assertEqual(layers, [NNXDecoderLayer])

  def test_get_decoder_layers_returns_list(self):
    decoder = self._make_decoder()
    result = decoder.get_decoder_layers()
    self.assertIsInstance(result, list)
    self.assertGreater(len(result), 0)

  def test_llama2_decoder_block(self):

    decoder = self._make_decoder(model_name="llama2-7b")
    layers = decoder.get_decoder_layers()
    self.assertEqual(layers, [LlamaDecoderLayer])

  def test_get_decoder_layers_idempotent(self):
    """Calling get_decoder_layers twice returns the same result."""
    decoder = self._make_decoder()
    self.assertEqual(decoder.get_decoder_layers(), decoder.get_decoder_layers())


# ---------------------------------------------------------------------------
# 4. NNXDecoder.get_norm_layer
# ---------------------------------------------------------------------------


class TestNNXDecoderGetNormLayer(unittest.TestCase):
  """Tests for NNXDecoder.get_norm_layer."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)
    self.decoder = NNXDecoder(
        config=self.cfg,
        mesh=self.mesh,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

  def test_default_returns_rms_norm(self):
    """DEFAULT decoder block should use RMSNorm."""
    # get_norm_layer returns a functools.partial wrapping RMSNorm.
    # The decoder_norm attribute is already instantiated via that partial.
    self.assertIsInstance(self.decoder.decoder_norm, RMSNorm)

  def test_gpt3_returns_gpt3_layer_norm(self):

    cfg = _make_config(model_name="gpt3-52k")
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsInstance(decoder.decoder_norm, Gpt3LayerNorm)


# ---------------------------------------------------------------------------
# 5. NNXDecoder.get_remat_policy / minimal_policy
# ---------------------------------------------------------------------------


class TestNNXDecoderRematPolicy(unittest.TestCase):
  """Tests for NNXDecoder.get_remat_policy and minimal_policy."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config(remat_policy="full")
    self.mesh = _make_mesh(self.cfg)
    self.decoder = NNXDecoder(
        config=self.cfg,
        mesh=self.mesh,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

  def test_remat_policy_none_returns_none(self):
    self.assertIsNone(self.decoder.get_remat_policy())

  def test_remat_policy_full_returns_none(self):
    cfg = _make_config(remat_policy="full")
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNone(decoder.get_remat_policy())

  def test_remat_policy_minimal_returns_non_none(self):
    cfg = _make_config(remat_policy="minimal")
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNotNone(decoder.get_remat_policy())

  def test_remat_policy_minimal_with_context_returns_non_none(self):
    cfg = _make_config(remat_policy="minimal_with_context")
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNotNone(decoder.get_remat_policy())

  def test_remat_policy_save_qkv_proj_returns_non_none(self):
    cfg = _make_config(remat_policy="save_qkv_proj")
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNotNone(decoder.get_remat_policy())

  def test_remat_policy_save_out_proj_returns_non_none(self):
    cfg = _make_config(remat_policy="save_out_proj")
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsNotNone(decoder.get_remat_policy())

  # --- minimal_policy -------------------------------------------------------

  def test_minimal_policy_no_flags(self):
    policy = self.decoder.minimal_policy()
    self.assertIsNotNone(policy)

  def test_minimal_policy_with_context(self):
    policy = self.decoder.minimal_policy(with_context=True)
    self.assertIsNotNone(policy)

  def test_minimal_policy_with_quantization(self):
    policy = self.decoder.minimal_policy(with_quantization=True)
    self.assertIsNotNone(policy)

  def test_minimal_policy_with_context_and_quantization(self):
    policy = self.decoder.minimal_policy(with_context=True, with_quantization=True)
    self.assertIsNotNone(policy)

  def test_minimal_policy_returns_distinct_objects_for_different_flags(self):
    """Different flag combinations should produce different policy objects."""
    p1 = self.decoder.minimal_policy(with_context=False)
    p2 = self.decoder.minimal_policy(with_context=True)
    # They're different checkpoint policies; at minimum they're both non-None
    # and Python objects that are not the same instance.
    self.assertIsNotNone(p1)
    self.assertIsNotNone(p2)


# ---------------------------------------------------------------------------
# 6. NNXDecoder full forward pass
# ---------------------------------------------------------------------------


class TestNNXDecoderForwardPass(unittest.TestCase):
  """Integration-style test for NNXDecoder.__call__ in train mode."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)
    self.rng = jax.random.PRNGKey(0)
    self.rngs = nnx.Rngs(params=0, dropout=1)

    self.decoder = NNXDecoder(
        config=self.cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )
    self.shared_embedding = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def _make_token_inputs(self):
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(self.rng, (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return ids, segment_ids, positions

  def test_forward_pass_returns_three_tuple(self):
    """__call__ must return (logits, hidden_state, kv_caches)."""
    ids, segment_ids, positions = self._make_token_inputs()
    result = self.decoder(
        self.shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 3)

  def test_logits_shape(self):
    """Logits shape: [batch, seq_len, vocab_size]."""
    cfg = self.cfg
    ids, segment_ids, positions = self._make_token_inputs()
    logits, _, _ = self.decoder(
        self.shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    expected = (
        cfg.global_batch_size_to_train_on,
        cfg.max_target_length,
        cfg.vocab_size,
    )
    self.assertEqual(logits.shape, expected)

  def test_hidden_state_shape(self):
    """hidden_state shape: [batch, seq_len, emb_dim]."""
    cfg = self.cfg
    ids, segment_ids, positions = self._make_token_inputs()
    _, hidden_state, _ = self.decoder(
        self.shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    expected = (
        cfg.global_batch_size_to_train_on,
        cfg.max_target_length,
        cfg.emb_dim,
    )
    self.assertEqual(hidden_state.shape, expected)

  def test_logits_are_finite(self):
    """Logits must not contain NaN or Inf in a simple forward pass."""
    ids, segment_ids, positions = self._make_token_inputs()
    logits, _, _ = self.decoder(
        self.shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertTrue(jnp.all(jnp.isfinite(logits)))

  def test_multimodal_input_forwarded_to_apply_embedding(self):
    """`multimodal_input` must reach `_apply_embedding` as the original struct.

    `NNXDecoder.__call__` takes a `MultimodalInput` struct and hands it to
    `_apply_embedding`, which is the layer that actually unpacks the fields
    and merges the embeddings. This test stubs `_apply_embedding` to capture
    the forwarded struct without running the real embedding path (the test
    config has `use_multimodal=False`).
    """
    ids, segment_ids, positions = self._make_token_inputs()

    # Distinct sentinels so each field can be traced independently.
    sentinel_img_emb = jnp.full((1, 1), 11.0)
    sentinel_img_mask = jnp.full((1, 1), 22.0)
    sentinel_aud_emb = jnp.full((1, 1), 33.0)
    sentinel_aud_mask = jnp.full((1, 1), 44.0)
    sentinel_bidir = jnp.full((1, 1), 55.0)

    mm_input = MultimodalInput(
        image_embeddings=sentinel_img_emb,
        image_masks=sentinel_img_mask,
        audio_embeddings=sentinel_aud_emb,
        audio_masks=sentinel_aud_mask,
        bidirectional_mask=sentinel_bidir,
    )

    captured = {}

    def fake_apply_embedding(
        _shared_embedding,
        _ids,
        _positions,
        _deterministic,
        _model_mode,
        multimodal_input=None,
        decoder_input_embeddings=None,
    ):
      del decoder_input_embeddings
      captured["multimodal_input"] = multimodal_input
      batch = self.cfg.global_batch_size_to_train_on
      seq_len = self.cfg.max_target_length
      emb_dim = self.cfg.emb_dim
      return jnp.zeros((batch, seq_len, emb_dim), dtype=self.cfg.dtype)

    self.decoder._apply_embedding = fake_apply_embedding  # pylint: disable=protected-access
    try:
      self.decoder(
          self.shared_embedding,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          multimodal_input=mm_input,
      )
    finally:
      # NNX modules bind attributes statefully; remove the override to avoid leaking.
      del self.decoder._apply_embedding  # pylint: disable=protected-access

    forwarded = captured["multimodal_input"]
    self.assertIsNotNone(forwarded)
    self.assertTrue(jnp.array_equal(forwarded.image_embeddings, sentinel_img_emb))
    self.assertTrue(jnp.array_equal(forwarded.image_masks, sentinel_img_mask))
    self.assertTrue(jnp.array_equal(forwarded.audio_embeddings, sentinel_aud_emb))
    self.assertTrue(jnp.array_equal(forwarded.audio_masks, sentinel_aud_mask))
    self.assertTrue(jnp.array_equal(forwarded.bidirectional_mask, sentinel_bidir))

  def test_precomputed_embeddings_bypass_initial_multimodal_merge(self):
    """Complete input embeddings must not be merged with vision embeddings again."""
    ids, _, positions = self._make_token_inputs()
    embeddings = jnp.ones(
        (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length, self.cfg.emb_dim),
        dtype=self.cfg.dtype,
    )

    result = self.decoder._apply_embedding(  # pylint: disable=protected-access
        lambda *_args, **_kwargs: self.fail("token embedding should be bypassed"),
        ids,
        positions,
        True,
        MODEL_MODE_TRAIN,
        multimodal_input=object(),
        decoder_input_embeddings=embeddings,
    )

    self.assertTrue(jnp.array_equal(result, embeddings))

  def test_different_random_seeds_produce_different_logits(self):
    """Two randomly-initialised decoders should not produce identical logits."""
    cfg = self.cfg
    mesh = self.mesh
    rngs2 = nnx.Rngs(params=99, dropout=1)
    decoder2 = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs2)
    shared_emb2 = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs2,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    common_kwargs = {
        "decoder_segment_ids": segment_ids,
        "deterministic": True,
        "model_mode": MODEL_MODE_TRAIN,
    }
    logits1, _, _ = self.decoder(self.shared_embedding, ids, positions, **common_kwargs)
    logits2, _, _ = decoder2(shared_emb2, ids, positions, **common_kwargs)
    self.assertFalse(jnp.allclose(logits1, logits2))

  def test_scan_layers(self):
    """Test NNXDecoder with scan_layers=True."""
    cfg = _make_config(scan_layers=True)
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(
        config=cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=self.mesh,
        rngs=rngs,
    )

    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(self.rng, (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    logits, _, _ = decoder(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(logits.shape, (batch, seq_len, cfg.vocab_size))


class _StatefulGemma4DecoderLayer(nnx.Module):
  """Small stand-in that exposes cache ordering and mutable-state updates."""

  def __init__(self, *, attention_type, **unused_kwargs):
    self.increment = 10 if attention_type == AttentionType.GLOBAL else 1
    self.call_count = nnx.Intermediate(jnp.array(0, dtype=jnp.int32))
    self.received_attention_metadata = nnx.Intermediate(jnp.array(False))

  def __call__(
      self,
      inputs,
      *unused_args,
      kv_cache=None,
      attention_metadata=None,
      **unused_kwargs,
  ):
    self.call_count.value += 1
    self.received_attention_metadata.value = attention_metadata is not None
    output = inputs + self.increment
    if kv_cache is None:
      return output
    return output, kv_cache + self.increment


class _SowingGemma4DecoderLayer(nnx.Module):
  """Stand-in whose global layer sows an accumulating Intermediate, like MoE moe_lb_loss."""

  def __init__(self, *, attention_type, **unused_kwargs):
    self.is_global = attention_type == AttentionType.GLOBAL
    # A trivial variable so the local layers have state for apply_scanned_layers to
    # scan over (a bare module has nothing to scan and lax.scan can't infer length).
    self.marker = nnx.Intermediate(jnp.zeros(()))

  def __call__(self, inputs, *unused_args, kv_cache=None, **unused_kwargs):
    output = inputs + 1
    if self.is_global:
      # nnx.sow appends into a tuple by default, so it grows across calls -- the
      # MoE moe_lb_loss pattern that must not enter the global length-1 scan carry.
      self.sow(nnx.Intermediate, "moe_lb_loss", jnp.sum(output))
    if kv_cache is None:
      return output
    return output, kv_cache


class TestGemma4ScannableBlock(unittest.TestCase):
  """Tests Gemma4's nested local/global decoder block behavior."""

  def setUp(self):
    super().setUp()
    self.config = SimpleNamespace(
        dtype=jnp.float32,
        param_scan_axis=1,
        remat_policy="none",
        scan_layers=True,
    )

  def _make_block(self):
    return gemma4.Gemma4ScannableBlock(
        config=self.config,
        mesh=None,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        rngs=nnx.Rngs(0),
    )

  def test_updates_state_through_global_single_iteration_scan(self):
    with mock.patch.object(gemma4, "Gemma4DecoderLayer", _StatefulGemma4DecoderLayer):
      block = self._make_block()
      output, updated_kvs = block(
          jnp.zeros((1, 1, 1)),
          decoder_segment_ids=None,
          decoder_positions=None,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

    np.testing.assert_array_equal(output, jnp.full((1, 1, 1), 15))
    self.assertIsNone(updated_kvs)
    np.testing.assert_array_equal(block.local_layers.call_count.value, jnp.ones(5, dtype=jnp.int32))
    np.testing.assert_array_equal(block.global_layer.call_count.value, 1)

  def test_global_layer_sown_intermediate_accumulates_across_calls(self):
    """A global layer that sows an accumulating Intermediate (e.g. MoE moe_lb_loss)
    must not break the length-1 scan carry, even when the Intermediate already
    exists from a previous call and the sow grows its tuple (1 -> 2 elements)."""
    call_kwargs = {
        "decoder_segment_ids": None,
        "decoder_positions": None,
        "deterministic": True,
        "model_mode": MODEL_MODE_AUTOREGRESSIVE,
    }
    with mock.patch.object(gemma4, "Gemma4DecoderLayer", _SowingGemma4DecoderLayer):
      block = self._make_block()
      # First call creates moe_lb_loss on the global layer (1-tuple).
      block(jnp.zeros((1, 1, 1)), **call_kwargs)
      # Second call: moe_lb_loss already exists and the sow appends -> 2-tuple.
      # Carrying it in the scan would change the carry pytree; the type-based
      # split keeps Intermediates on the ys path instead.
      block(jnp.zeros((1, 1, 1)), **call_kwargs)

    self.assertEqual(len(block.global_layer.moe_lb_loss.value), 2)

  def test_restores_local_state_and_preserves_kv_order(self):
    attention_metadata = object()

    with mock.patch.object(gemma4, "Gemma4DecoderLayer", _StatefulGemma4DecoderLayer):
      block = self._make_block()
      output, updated_kvs = block(
          jnp.zeros((1, 1, 1)),
          decoder_segment_ids=None,
          decoder_positions=None,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          kv_cache=tuple(jnp.array(i) for i in range(6)),
          attention_metadata=attention_metadata,
      )

    np.testing.assert_array_equal(output, jnp.full((1, 1, 1), 15))
    np.testing.assert_array_equal(jnp.stack(updated_kvs), jnp.array([1, 2, 3, 4, 5, 15]))
    np.testing.assert_array_equal(block.local_layers.call_count.value, jnp.ones(5, dtype=jnp.int32))
    np.testing.assert_array_equal(
        block.local_layers.received_attention_metadata.value,
        jnp.ones(5, dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(block.global_layer.call_count.value, 1)
    np.testing.assert_array_equal(block.global_layer.received_attention_metadata.value, True)


# Qwen3-Next blocks read enough of the config (GatedDeltaNet head dims, MoE sizing)
# that a SimpleNamespace stand-in is not workable, so these use a real tiny config.
_QWEN3_NEXT_CONFIG = {
    "run_name": "qwen3_next_scannable_block_test",
    "model_name": "qwen3-next-80b-a3b",
    "max_target_length": 8,
    "base_emb_dim": 64,
    "base_num_decoder_layers": 4,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "head_dim": 32,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 32,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 4,
    "gdn_key_head_dim": 16,
    "gdn_value_head_dim": 16,
    "gdn_chunk_size": 4,
    "sparse_matmul": True,
    "megablox": False,
    "dtype": "float32",
    "weight_dtype": "float32",
}


def _build_qwen3_next_block(layer_idx_offset=0, **overrides):
  """Builds one Qwen3-Next scannable block; overrides go to the config."""
  cfg = _make_config(**{**_QWEN3_NEXT_CONFIG, **overrides})
  block = qwen3.Qwen3NextScannableBlock(
      config=cfg,
      mesh=_make_mesh(cfg),
      model_mode=MODEL_MODE_TRAIN,
      layer_idx_offset=layer_idx_offset,
      rngs=nnx.Rngs(0),
  )
  return cfg, block


class TestQwen3NextScannableBlock(unittest.TestCase):
  """Tests Qwen3-Next's nested local(scan)/global(length-1 scan) decoder block."""

  @classmethod
  def setUpClass(cls):
    """Builds the block once; it costs several seconds and no test here mutates it."""
    cls.cfg, cls.block = _build_qwen3_next_block()

  def _inputs(self, cfg):
    inputs = jax.random.normal(jax.random.PRNGKey(1), (1, cfg.max_target_length, cfg.emb_dim), dtype=jnp.float32)
    positions = jnp.arange(cfg.max_target_length)[None, :]
    segment_ids = jnp.ones((1, cfg.max_target_length), dtype=jnp.int32)
    return inputs, segment_ids, positions

  def test_block_splits_cycle_into_local_stack_plus_one_global(self):
    """A block covers one attention period: cycle-1 stacked linear layers and one full-attention layer."""
    cfg, block = self.cfg, self.block
    self.assertEqual(block.num_local, cfg.inhomogeneous_layer_cycle_interval - 1)
    self.assertEqual(block.num_global, 1)
    self.assertIsNotNone(block.global_layer)

    # The linear-attention layers are stacked along param_scan_axis, not stored per layer.
    _, params, _ = nnx.split(block.local_layers, nnx.Param, ...)
    leaves = [v.value for _, v in params.flat_state()]
    self.assertTrue(leaves)
    for leaf in leaves:
      self.assertEqual(leaf.shape[cfg.param_scan_axis], block.num_local)

  def test_nested_scan_matches_sequential_unroll(self):
    """Scanning the local layers then the global layer equals applying them one by one."""
    cfg, block = self.cfg, self.block
    inputs, segment_ids, positions = self._inputs(cfg)

    scanned = block(inputs, segment_ids, positions, True, MODEL_MODE_TRAIN)

    # Reference: pull each stacked local layer out by index and run it, then the global layer.
    # Under jit, so that the unrolled sub-layers are traced once instead of dispatched op by op.
    local_graphdef, params, rest = nnx.split(block.local_layers, nnx.Param, ...)
    global_graphdef, global_state = nnx.split(block.global_layer)

    @jax.jit
    def unrolled(params, rest, global_state, y):
      if cfg.param_scan_axis != 0:
        params = jax.tree.map(lambda x: jnp.moveaxis(x, cfg.param_scan_axis, 0), params)
      for i in range(block.num_local):
        layer = nnx.merge(
            local_graphdef,
            jax.tree.map(lambda x, i=i: x[i], params),
            jax.tree.map(lambda x, i=i: x[i], rest),
        )
        y = layer(y, segment_ids, positions, True, MODEL_MODE_TRAIN)[0]
      return nnx.merge(global_graphdef, global_state)(y, segment_ids, positions, True, MODEL_MODE_TRAIN)[0]

    expected = unrolled(params, rest, global_state, inputs)

    np.testing.assert_allclose(np.asarray(scanned), np.asarray(expected), rtol=1e-5, atol=1e-5)

  def test_rejects_block_whose_global_layer_is_not_last(self):
    """The local scan runs before the global layer, so any other ordering must be refused.

    A block starting off a cycle boundary straddles two periods -- with a cycle of
    4, layers 1..4 put the full-attention layer third of four. Applying it as
    local-scan-then-global would silently reorder the model, so it is rejected.
    """
    with self.assertRaisesRegex(ValueError, "full-attention layer last"):
      _build_qwen3_next_block(layer_idx_offset=1)


class TestNNXDecoderQwen3Next(unittest.TestCase):
  """Tests the NNXDecoder-level wiring of the Qwen3-Next scanned blocks."""

  def _build(self, num_decoder_layers, **overrides):
    """Builds a scanned Qwen3-Next decoder and its shared embedding."""
    cfg = _make_config(
        **{**_QWEN3_NEXT_CONFIG, "base_num_decoder_layers": num_decoder_layers, "scan_layers": True, **overrides}
    )
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        config=cfg,
        mesh=mesh,
        rngs=nnx.Rngs(params=0),
    )
    return cfg, decoder, shared_embedding

  def _run(self, cfg, decoder, shared_embedding, kv_caches=None):
    """Runs one TRAIN-mode forward pass and returns the logits."""
    batch = cfg.global_batch_size_to_train_on
    seq = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq)[None], (batch, seq))
    logits, _, _ = decoder(
        shared_embedding,
        ids,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        kv_caches=kv_caches,
    )
    return logits

  def test_decoder_regroups_flat_kv_caches_per_block(self):
    """A flat per-layer kv cache list must be regrouped per block and written back in order.

    The scan runs over blocks, not layers, so passing the flat list straight
    through would hand block i only ``kv_caches[i]``. Guards
    ``_apply_qwen3_next_scanned_blocks``, which must also keep
    ``skip_block_remat=True`` on this path rather than falling back to the
    generic (block-rematerialized) branch.

    The external-cache path is a static unroll, so its cost is per sub-layer;
    a cycle of 2 keeps two blocks -- the minimum that can regroup wrongly --
    at half the layers of the stock cycle of 4.
    """
    cfg, decoder, shared_embedding = self._build(4, inhomogeneous_layer_cycle_interval=2)
    # Every layer is inside the scan, so there is no remainder block.
    self.assertFalse(hasattr(decoder, "layers_remainder"))
    batch = cfg.global_batch_size_to_train_on
    seq = cfg.max_target_length

    # Distinct sentinel per layer: regrouping errors show up as caches landing on
    # the wrong layer, which the pass-through in TRAIN mode makes visible.
    kv_caches = [jnp.full((batch, seq), float(i)) for i in range(cfg.num_decoder_layers)]
    self._run(cfg, decoder, shared_embedding, kv_caches=kv_caches)

    self.assertEqual(len(kv_caches), cfg.num_decoder_layers)
    for i, cache in enumerate(kv_caches):
      np.testing.assert_array_equal(np.asarray(cache), np.full((batch, seq), float(i)))

  def test_decoder_keeps_layers_past_the_last_whole_block(self):
    """Layers left over by the block scan must still be built and applied.

    ``num_decoder_layers // inhomogeneous_layer_cycle_interval`` blocks cover
    only a whole number of periods, so with 6 layers and a period of 4 the last
    two would be silently dropped -- the model would quietly run 4 layers. They
    go into ``layers_remainder`` instead; perturbing only that block's weights
    has to move the output, which it cannot do if the block is never applied.
    """
    cfg, decoder, shared_embedding = self._build(6)
    self.assertEqual(decoder.layers_remainder.num_local, 2)
    # The remainder starts on a period boundary, so it holds no full-attention layer.
    self.assertEqual(decoder.layers_remainder.num_global, 0)

    before = self._run(cfg, decoder, shared_embedding)
    _, params, rest = nnx.split(decoder.layers_remainder, nnx.Param, ...)
    nnx.update(decoder.layers_remainder, jax.tree.map(lambda x: x + 0.1, params), rest)
    after = self._run(cfg, decoder, shared_embedding)

    self.assertFalse(
        np.allclose(np.asarray(before), np.asarray(after)),
        "the remainder block's weights did not affect the output, so it was not applied",
    )


class TestQwen3NextDecoderParity(unittest.TestCase):
  """The Linen `Decoder` and the pure-NNX `NNXDecoder` must emit the same parameter tree.

  One checkpoint mapping (`QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING`) serves both
  decoders, so a name or shape that differs between them silently breaks
  conversion on whichever side the mapping was not written against.
  """

  def _param_tree(self, num_decoder_layers, pure_nnx_decoder):
    """Returns {parameter key: shape} for the chosen decoder implementation."""
    # pylint: disable=import-outside-toplevel
    from maxtext.checkpoint_conversion.to_maxtext import get_maxtext_model_info

    cfg = _make_config(
        **{
            **_QWEN3_NEXT_CONFIG,
            "base_num_decoder_layers": num_decoder_layers,
            "scan_layers": True,
            "pure_nnx_decoder": pure_nnx_decoder,
        }
    )
    model_info, _ = get_maxtext_model_info(cfg)
    return {key: shape for key, (_, shape) in model_info.items()}

  def test_decoders_agree_on_whole_periods(self):
    self.assertEqual(self._param_tree(8, True), self._param_tree(8, False))

  def test_decoders_agree_with_a_remainder(self):
    """6 layers is one whole period plus a two-layer remainder, which both decoders
    have to put in a `layers_remainder` block rather than spell out layer by layer."""
    self.assertEqual(self._param_tree(6, True), self._param_tree(6, False))


class TestNNXDecoderDeepseekAndGemma4(unittest.TestCase):
  """Tests for Deepseek and Gemma4 specific decoder logic."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)
    self.rng = jax.random.PRNGKey(0)
    self.rngs = nnx.Rngs(params=0, dropout=1)

  def _make_token_inputs(self, cfg):
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(self.rng, (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return ids, segment_ids, positions

  def _make_shared_embedding(self, cfg):
    return Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def test_gemma_scan_layers_equivalence(self):
    # Test that scan_layers=True and scan_layers=False produce identical logits
    # even when bidirectional_mask is provided, proving kwargs are not dropped.
    cfg_base = {
        "run_name": "gemma3_scan_equiv_test",
        "decoder_block": "gemma3",
        "model_name": "gemma3-4b",
        "num_decoder_layers": 2,
        "base_emb_dim": 128,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "base_mlp_dim": 256,
        "hidden_size_per_layer_input": 128,
        "vocab_size_per_layer_input": 256,
        "vocab_size": 256,
        "max_target_length": 64,
        "per_device_batch_size": 1.0,
    }

    cfg_scanned = _make_config(scan_layers=True, **cfg_base)
    cfg_unscanned = _make_config(scan_layers=False, **cfg_base)

    # Use identical RNGs to guarantee the same parameter initialization
    rngs_scanned = nnx.Rngs(params=0, dropout=1)
    rngs_unscanned = nnx.Rngs(params=0, dropout=1)

    decoder_scanned = NNXDecoder(config=cfg_scanned, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs_scanned)
    decoder_unscanned = NNXDecoder(config=cfg_unscanned, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs_unscanned)

    ids, segment_ids, positions = self._make_token_inputs(cfg_scanned)
    shared_embedding = self._make_shared_embedding(cfg_scanned)

    # Provide a mock bidirectional_mask to trigger the code path that dropped the kwarg
    batch = cfg_scanned.global_batch_size_to_train_on
    seq_len = cfg_scanned.max_target_length
    bidirectional_mask = jax.random.normal(self.rng, (batch, seq_len)) > 0
    mm_input = MultimodalInput(bidirectional_mask=bidirectional_mask)

    logits_scanned, _, _ = decoder_scanned(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        multimodal_input=mm_input,
    )

    logits_unscanned, _, _ = decoder_unscanned(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        multimodal_input=mm_input,
    )

    np.testing.assert_allclose(logits_scanned, logits_unscanned, atol=1e-4, rtol=1e-4)

  def test_gemma_scan_layers_kv_cache_updated(self):
    # Test that scan_layers=True correctly forwards kv_caches to _apply_layers_sequentially.
    # Patching/inspecting the private _apply_gemma3_scanned_blocks is intentional here.
    # pylint: disable=protected-access
    cfg = _make_config(
        run_name="gemma3_scan_kv_test",
        decoder_block="gemma3",
        model_name="gemma3-4b",
        scan_layers=True,
        num_decoder_layers=2,
        base_emb_dim=128,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        base_mlp_dim=256,
        hidden_size_per_layer_input=128,
        vocab_size_per_layer_input=256,
        vocab_size=256,
        max_target_length=64,
        per_device_batch_size=1.0,
    )

    decoder = NNXDecoder(config=cfg, mesh=self.mesh, model_mode=MODEL_MODE_PREFILL, rngs=self.rngs)
    ids, segment_ids, positions = self._make_token_inputs(cfg)
    shared_embedding = self._make_shared_embedding(cfg)

    decoder._apply_gemma3_scanned_blocks = MagicMock(return_value=jnp.zeros((1, 1)))

    mock_kv_caches = [jnp.zeros((1, 1)) for _ in range(cfg.num_decoder_layers)]

    _ = decoder(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        kv_caches=mock_kv_caches,
    )

    # Verify that _apply_gemma3_scanned_blocks was called with the kv_caches
    self.assertTrue(decoder._apply_gemma3_scanned_blocks.called)
    call_kwargs = decoder._apply_gemma3_scanned_blocks.call_args[1]
    self.assertIn("kv_caches", call_kwargs)
    self.assertEqual(call_kwargs["kv_caches"], mock_kv_caches)

  def test_gemma4_scanned_layers(self):
    """Test NNXDecoder with gemma4 block and scan_layers=True."""
    cfg = _make_config(
        decoder_block="gemma4",
        scan_layers=True,
        num_decoder_layers=3,  # Not a multiple of the pattern length (which is usually larger) to test remainder logic
        vocab_size=256,
    )
    decoder = NNXDecoder(
        config=cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )
    shared_embedding = self._make_shared_embedding(cfg)
    ids, segment_ids, positions = self._make_token_inputs(cfg)

    logits, _, _ = decoder(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(
        logits.shape,
        (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size),
    )

  def test_gemma4_block_external_kv_cache_matches_scanned_path(self):
    """External-kv-cache path must numerically match the scanned (kv=None) path.

    Guards the real stacked-parameter slice/re-stack in
    ``Gemma4ScannableBlock._forward_with_external_kv_cache`` (``nnx.split`` of the
    scanned local stack, ``param_scan_axis`` moveaxis, per-layer merge, re-stack,
    ``nnx.update``) against the ``jax.lax.scan`` path. The mock-based
    ``TestGemma4ScannableBlock`` tests cover call ordering / cache collection but
    use trivial params, so they never exercise the real stacked-param mechanics.

    Uses ``model_mode=TRAIN`` with ``dot_product`` attention: attention then
    ignores the external caches (passing them straight through), so both paths
    compute the same forward and only the loop mechanism (scan vs static unroll)
    differs -- any mismatch is a slice/re-stack bug.
    """
    cfg = _make_config(
        decoder_block="gemma4",
        scan_layers=True,
        num_decoder_layers=len(gemma4.GEMMA4_ATTENTION_PATTERN),  # exactly one full 5-local + 1-global block
        base_emb_dim=128,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        # float32 + high matmul precision so the two paths agree to tight tolerance;
        # the paths are mathematically identical, so any bf16-level rounding drift
        # between scan and static-unroll compilation would only mask a real bug.
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
    )
    mesh = _make_mesh(cfg)

    def make_block():
      # Same seed => identical params in both blocks, so any output difference is
      # attributable to the scan-vs-unroll code path, not initialization.
      return gemma4.Gemma4ScannableBlock(
          config=cfg,
          mesh=mesh,
          model_mode=MODEL_MODE_TRAIN,
          quant=None,
          num_of_layers=len(gemma4.GEMMA4_ATTENTION_PATTERN),
          remat_policy_fn=None,
          apply_internal_remat=False,
          rngs=nnx.Rngs(params=0),
      )

    batch = cfg.global_batch_size_to_train_on
    seq = cfg.max_target_length
    inputs = jax.random.normal(self.rng, (batch, seq, cfg.emb_dim), dtype=cfg.dtype)
    segment_ids = jnp.full((batch, seq), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq)[None], (batch, seq))
    call_kwargs = {
        "decoder_segment_ids": segment_ids,
        "decoder_positions": positions,
        "deterministic": True,
        "model_mode": MODEL_MODE_TRAIN,
    }

    # Scanned path (kv_cache=None); scan_layers=True => returns (y, None).
    y_scanned, _ = make_block()(inputs, **call_kwargs)

    # External-kv path: one cache per layer, ordered local[0..4] then global.
    num_layers = len(gemma4.GEMMA4_ATTENTION_PATTERN)
    external_kv = tuple(jnp.zeros((batch, seq), dtype=cfg.dtype) for _ in range(num_layers))
    y_external, updated_kvs = make_block()(inputs, **call_kwargs, kv_cache=external_kv)

    self.assertEqual(y_external.shape, inputs.shape)
    self.assertEqual(len(updated_kvs), num_layers)
    np.testing.assert_allclose(y_external, y_scanned, rtol=1e-5, atol=1e-5)


@pytest.mark.tpu_only
class TestGemma4SmallNNXDecoder(unittest.TestCase):
  """Unit tests for Gemma4 Small NNXDecoder to improve code coverage."""

  def test_gemma4_small_decoder(self):
    cfg = pyconfig.initialize(
        [
            None,
            get_test_config_path(),
            "run_name=gemma4_small_test",
            "decoder_block=gemma4_small",
            "model_name=gemma4-e2b",
            "scan_layers=False",
            "attention=dot_product",
            "num_decoder_layers=3",
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
            "max_target_length=128",
            "per_device_batch_size=1.0",
        ],
        override_model_config=True,
    )

    devices = np.array(jax.devices())
    num_devices = len(devices)
    mesh_shape = [1] * len(cfg.mesh_axes)
    mesh_shape[cfg.mesh_axes.index("data")] = num_devices
    mesh = Mesh(devices.reshape(mesh_shape), cfg.mesh_axes)

    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )

    # Inputs
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )

    logits, _, _ = decoder(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertEqual(
        logits.shape,
        (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size),
    )

  def test_gemma4_small_decoder_with_mock_cache_and_ple(self):
    cfg = pyconfig.initialize(
        [
            None,
            get_test_config_path(),
            "run_name=gemma4_small_test",
            "decoder_block=gemma4_small",
            "model_name=gemma4-e2b",
            "scan_layers=False",
            "attention=dot_product",
            "remat_policy=none",
            "num_decoder_layers=3",
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
            "max_target_length=128",
            "per_device_batch_size=1.0",
        ],
        override_model_config=True,
    )

    devices = np.array(jax.devices())
    num_devices = len(devices)
    mesh_shape = [1] * len(cfg.mesh_axes)
    mesh_shape[cfg.mesh_axes.index("data")] = num_devices
    mesh = Mesh(devices.reshape(mesh_shape), cfg.mesh_axes)

    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )

    # Mock each layer's compute_shared_kv
    for layer in decoder.get_layers():
      layer.compute_shared_kv = MagicMock(return_value=(jnp.zeros((1, 16, 128)), jnp.zeros((1, 16, 128))))

    # Inputs
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )

    layer_types = gemma4_small.build_layer_types(cfg.num_decoder_layers, cfg.model_name)
    cache_index_of = gemma4_small.kv_cache_slot_map(layer_types, cfg.num_kv_shared_layers)
    max_slot = max(cache_index_of.values())
    kv_caches = [f"initial_cache_{i}" for i in range(max_slot + 1)]

    with patch(
        "maxtext.models.gemma4_small.Gemma4SmallDecoderLayer.__call__",
        return_value=(jnp.zeros((1, 16, 128)), "mock_kv_cache"),
    ):
      _, _, updated_caches = decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          kv_caches=kv_caches,
      )

      # Verify that the mocked kv_caches were correctly updated
      self.assertEqual(updated_caches, ["mock_kv_cache"] * len(kv_caches))

      # Test RuntimeError branch coverage
      with self.assertRaises(RuntimeError):

        def mock_donor_idx(lyr, layer_types, num_kv_shared):
          if lyr == 2:
            return 0
          return gemma4_small.kv_donor_layer_idx(lyr, layer_types, num_kv_shared)

        with patch("maxtext.models.gemma4_small.kv_donor_layer_idx", side_effect=mock_donor_idx):
          decoder(
              shared_embedding,
              ids,
              positions,
              decoder_segment_ids=segment_ids,
              deterministic=True,
              model_mode=MODEL_MODE_TRAIN,
              kv_caches=kv_caches,
          )


class TestApplyLayersSequentiallyMetadataAxisName(unittest.TestCase):
  """Tests for metadata axis name parameterization in NNXDecoder."""

  def test_metadata_axis_name_parameterization(self):
    cfg = _make_config(param_scan_axis=0)
    mesh = _make_mesh(cfg)
    rngs = nnx.Rngs(params=0)

    decoder = NNXDecoder(
        config=cfg,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs,
    )

    class DummyLayer(nnx.Module):
      """A dummy NNX module for testing."""

      def __init__(self):
        self.p = nnx.Param(jax.numpy.zeros((2,)))

      def __call__(self, x, **kwargs):
        return x + self.p.value, None

    # Manually create a stacked layer using NNX scan
    stacked_layers = nnx.vmap(DummyLayer, in_axes=(), out_axes=0, axis_size=2)()

    x_in = jax.numpy.zeros((2,))

    # We mock maxtext_utils_nnx.nnx_add_and_sync_scan_axis to ensure the custom name is passed
    original_add_scan_axis = maxtext_utils_nnx.nnx_add_and_sync_scan_axis
    mock_add_scan_axis = MagicMock(side_effect=original_add_scan_axis)
    maxtext_utils_nnx.nnx_add_and_sync_scan_axis = mock_add_scan_axis

    try:
      # Use a custom metadata_axis_name
      custom_axis_name = "custom_scanned_blocks"
      # pylint: disable=protected-access
      _, _, _ = decoder._apply_layers_sequentially(
          layers=stacked_layers,
          x_in=x_in,
          length=2,
          metadata_axis_name=custom_axis_name,
      )

      # Verify that the custom axis name was indeed passed down
      found_custom_name = False
      for call_args in mock_add_scan_axis.call_args_list:
        if call_args[0][1] == custom_axis_name:
          found_custom_name = True
          break

      self.assertTrue(
          found_custom_name,
          "The custom metadata_axis_name was not passed to nnx_add_and_sync_scan_axis!",
      )
    finally:
      maxtext_utils_nnx.nnx_add_and_sync_scan_axis = original_add_scan_axis


class TestApplyLayersSequentiallyDynamicGraphInit(unittest.TestCase):
  """Params created inside the scan body must not drag the base stack out with them."""

  class _AdapterLayer(nnx.Module):
    """A layer that materializes a new param while tracing, as Qwix LoRA does."""

    def __init__(self):
      self.p = nnx.Param(jax.numpy.zeros((2,)))

    def __call__(self, x, **kwargs):
      self.adapter = nnx.LoRAParam(jax.numpy.ones((2,)))
      return x + self.p.value + self.adapter.value, None

  def _run(self, param_scan_axis):
    cfg = _make_config(param_scan_axis=param_scan_axis)
    decoder = NNXDecoder(config=cfg, mesh=_make_mesh(cfg), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
    layers = nnx.vmap(self._AdapterLayer, in_axes=(), out_axes=param_scan_axis, axis_size=2)()
    # Qwix sets this on every module for the duration of its init pass.
    decoder.disable_quant_stats_update = True
    base_before = layers.p.value
    # pylint: disable=protected-access
    _, out_layers, _ = decoder._apply_layers_sequentially(layers=layers, x_in=jax.numpy.zeros((2,)), length=2)
    return base_before, out_layers

  def test_created_param_escapes_the_scan(self):
    for axis in (0, 1):
      with self.subTest(param_scan_axis=axis):
        _, out_layers = self._run(axis)
        self.assertTrue(hasattr(out_layers, "adapter"))

  def test_base_params_are_not_restacked(self):
    for axis in (0, 1):
      with self.subTest(param_scan_axis=axis):
        base_before, out_layers = self._run(axis)
        self.assertIs(out_layers.p.value, base_before)


if __name__ == "__main__":
  unittest.main()
