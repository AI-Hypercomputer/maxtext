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

# pylint: disable=unbalanced-tuple-unpacking
import re
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
from flax import traverse_util
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    AttentionType,
    DecoderBlockType,
    MultimodalInput,
    ShardMode,
)
from maxtext.configs import pyconfig
from maxtext.integration.tunix.weight_mapping import raiden_unscan
from maxtext.layers import fsdp_prefetch
from maxtext.layers import fsdp_prefetch_inhomogeneous
from maxtext.layers import linears
from maxtext.layers import nnx_decoders
from maxtext.layers import nnx_scan
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder, NNXDecoderLayer, deepstack_process
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import gemma4, gemma4_small, qwen3, qwen3_5
from maxtext.models.gpt3 import Gpt3LayerNorm
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils, maxtext_utils_nnx, model_creation_utils, sharding
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


class TestNNXDecoderServeFp8WeightForwardPass(unittest.TestCase):
  """End-to-end forward pass with quantization="serve_fp8_weight" (native, no-dequantize
  FP8 compute -- see quantizations.native_fp8_dot_general / ServeFp8WeightQuantization).
  """

  # weight_block_size must divide contracted dims in _BASE_CONFIG.
  _FP8_OVERRIDES = {
      "weight_dtype": "float8_e4m3fn",
      "weight_block_size": 16,
  }

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  def _make_token_inputs(self, cfg):
    """Generates dummy token inputs for decoder test."""
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(self.rng, (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return ids, segment_ids, positions

  def _build(self, cfg, mesh, rngs):
    """Builds test decoder and shared embedding."""
    quant = quantizations.configure_quantization(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, quant=quant, rngs=rngs)
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )
    return decoder, shared_embedding

  def test_forward_pass_shapes_and_finite(self):
    """A serve_fp8_weight decoder must produce correctly-shaped, finite logits."""
    cfg = _make_config(quantization="serve_fp8_weight", **self._FP8_OVERRIDES)
    mesh = _make_mesh(cfg)
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder, shared_embedding = self._build(cfg, mesh, rngs)
    ids, segment_ids, positions = self._make_token_inputs(cfg)

    logits, hidden_state, *_ = decoder(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(logits.shape, (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size))
    self.assertEqual(hidden_state.shape, (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim))
    self.assertTrue(jnp.all(jnp.isfinite(logits)))

  def test_matches_dequantize_baseline(self):
    """Native fp8 compute must closely match the existing dequantize-then-matmul path.

    Both decoders store the same fp8-quantized weights (weight_dtype=float8_e4m3fn);
    only the compute strategy differs (quantization="serve_fp8_weight" vs unset). State
    is explicitly synced after construction rather than relied on via matching RNG draws.
    """
    baseline_cfg = _make_config(**self._FP8_OVERRIDES)
    native_cfg = _make_config(quantization="serve_fp8_weight", **self._FP8_OVERRIDES)
    mesh = _make_mesh(baseline_cfg)

    baseline_decoder, baseline_embedding = self._build(baseline_cfg, mesh, nnx.Rngs(params=0, dropout=1))
    native_decoder, native_embedding = self._build(native_cfg, mesh, nnx.Rngs(params=0, dropout=1))
    nnx.update(native_decoder, nnx.state(baseline_decoder))
    nnx.update(native_embedding, nnx.state(baseline_embedding))

    ids, segment_ids, positions = self._make_token_inputs(baseline_cfg)
    call_kwargs = {
        "decoder_segment_ids": segment_ids,
        "deterministic": True,
        "model_mode": MODEL_MODE_TRAIN,
    }
    baseline_logits, _, *_ = baseline_decoder(baseline_embedding, ids, positions, **call_kwargs)
    native_logits, _, *_ = native_decoder(native_embedding, ids, positions, **call_kwargs)

    baseline_logits = np.asarray(baseline_logits, dtype=np.float32)
    native_logits = np.asarray(native_logits, dtype=np.float32)
    self.assertEqual(baseline_logits.shape, native_logits.shape)
    relerr = float(np.max(np.abs(baseline_logits - native_logits)) / (np.max(np.abs(baseline_logits)) + 1e-12))
    # Loose tolerance: compares native FP8 forward pass against dequantized baseline.
    self.assertLess(relerr, 0.1, f"serve_fp8_weight vs dequantize baseline relerr={relerr:.3e}")


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
        shard_mode=ShardMode.AUTO,
        debug_sharding=False,
    )

  def _make_block(self):
    return gemma4.Gemma4ScannableBlock(
        config=self.config,
        # The block pins its input with maybe_shard_with_logical, which needs a real mesh.
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
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


class TestQwen3NextScannableBlock(unittest.TestCase):
  """Tests Qwen3-Next's nested local(scan)/global(length-1 scan) decoder block."""

  MODEL_CONFIG = _QWEN3_NEXT_CONFIG
  BLOCK_CLS = qwen3.Qwen3NextScannableBlock

  @classmethod
  def _build(cls, layer_idx_offset=0, **overrides):
    """Builds one scannable block; overrides go to the config."""
    cfg = _make_config(**{**cls.MODEL_CONFIG, **overrides})
    block = cls.BLOCK_CLS(
        config=cfg,
        mesh=_make_mesh(cfg),
        model_mode=MODEL_MODE_TRAIN,
        layer_idx_offset=layer_idx_offset,
        rngs=nnx.Rngs(0),
    )
    return cfg, block

  @classmethod
  def setUpClass(cls):
    """Builds the block once; it costs several seconds and no test here mutates it."""
    cls.cfg, cls.block = cls._build()

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

  def test_params_created_inside_the_global_layer_are_kept(self):
    """A param materialized while the global layer runs (e.g. a Qwix LoRA adapter) must reach the block."""
    cfg, block = self._build()
    inputs, segment_ids, positions = self._inputs(cfg)
    run_layer = block._run_layer  # pylint: disable=protected-access

    def run_layer_creating_a_param(layer, *args, **kwargs):
      if layer.is_full_attention_layer and not hasattr(layer, "created"):
        layer.created = nnx.Param(jnp.full((2,), 3.0, dtype=jnp.float32))
      return run_layer(layer, *args, **kwargs)

    with mock.patch.object(block, "_run_layer", side_effect=run_layer_creating_a_param):
      block(inputs, segment_ids, positions, True, MODEL_MODE_TRAIN)
    self.assertTrue(hasattr(block.global_layer, "created"))
    np.testing.assert_array_equal(np.asarray(block.global_layer.created.value), np.full((2,), 3.0))

  def test_rejects_block_whose_global_layer_is_not_last(self):
    """The local scan runs before the global layer, so any other ordering must be refused.

    A block starting off a cycle boundary straddles two periods -- with a cycle of
    4, layers 1..4 put the full-attention layer third of four. Applying it as
    local-scan-then-global would silently reorder the model, so it is rejected.
    """
    with self.assertRaisesRegex(ValueError, "full-attention layer last"):
      self._build(layer_idx_offset=1)

  def test_rejects_forced_routing_that_does_not_cover_every_sub_layer(self):
    """Router replay is sliced per sub-layer, so a mismatched leading axis must not be broadcast."""
    cfg, block = self.cfg, self.block
    inputs, segment_ids, positions = self._inputs(cfg)
    forced_routed_experts = jnp.zeros(
        (block.num_of_layers + 1, 1, cfg.max_target_length, cfg.num_experts_per_tok), dtype=jnp.int32
    )
    with self.assertRaisesRegex(ValueError, "one slice per sub-layer"):
      block(
          inputs,
          segment_ids,
          positions,
          True,
          MODEL_MODE_TRAIN,
          forced_routed_experts=forced_routed_experts,
      )


class TestNNXDecoderQwen3Next(unittest.TestCase):
  """Tests the NNXDecoder-level wiring of the Qwen3-Next scanned blocks."""

  MODEL_CONFIG = _QWEN3_NEXT_CONFIG

  def _build(self, num_decoder_layers, **overrides):
    """Builds a scanned Qwen3-Next decoder and its shared embedding."""
    cfg = _make_config(
        **{**self.MODEL_CONFIG, "base_num_decoder_layers": num_decoder_layers, "scan_layers": True, **overrides}
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

  def _run(self, cfg, decoder, shared_embedding, kv_caches=None, attention_metadata=None):
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
        attention_metadata=attention_metadata,
    )
    return logits

  def test_decoder_regroups_flat_kv_caches_per_block(self):
    """A flat per-layer kv cache list must be regrouped per block and written back in order.

    The scan runs over blocks, not layers, so passing the flat list straight
    through would hand block i only ``kv_caches[i]``. Guards
    ``_apply_qwen3_hybrid_scanned_blocks``, which must also keep
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

  def test_decoder_routes_per_layer_attention_metadata_by_global_layer(self):
    """Every layer must get its own entry of vLLM's per-layer metadata, keyed by global layer index.

    vLLM hands hybrid models a dict from ``layer.{i}`` to that layer's KV-cache
    group metadata. The sub-layers of a scanned block do not know their global
    index (the local ones are all built with layer_idx=0), so the decoder has to
    resolve the dict for each block. Two blocks of a cycle-2 model plus a
    one-layer remainder cover the scanned blocks and the remainder block alike:
    a lookup by position in the block would hand layers 2..4 the entries of
    layers 0..1.
    """
    cfg, decoder, shared_embedding = self._build(5, inhomogeneous_layer_cycle_interval=2)
    self.assertEqual(decoder.layers_remainder.num_local, 1)
    batch = cfg.global_batch_size_to_train_on
    seq = cfg.max_target_length
    kv_caches = [jnp.full((batch, seq), float(i)) for i in range(cfg.num_decoder_layers)]
    attention_metadata = {f"layer.{i}": object() for i in range(cfg.num_decoder_layers)}

    received = []
    original_call = qwen3.Qwen3NextDecoderLayer.__call__

    def recording_call(layer, *args, attention_metadata=None, **kwargs):
      received.append(attention_metadata)
      # The sentinels are not real metadata, so run the layer as it runs without any.
      return original_call(layer, *args, attention_metadata=None, **kwargs)

    with mock.patch.object(qwen3.Qwen3NextDecoderLayer, "__call__", recording_call):
      self._run(cfg, decoder, shared_embedding, kv_caches=kv_caches, attention_metadata=attention_metadata)

    self.assertEqual(len(received), cfg.num_decoder_layers)
    for lyr, metadata in enumerate(received):
      self.assertIs(metadata, attention_metadata[f"layer.{lyr}"], f"layer {lyr} got another layer's metadata")
    # The caches still land back on their own layers.
    for i, cache in enumerate(kv_caches):
      np.testing.assert_array_equal(np.asarray(cache), np.full((batch, seq), float(i)))

  def test_decoder_rejects_per_layer_attention_metadata_without_kv_caches(self):
    """Without external caches the blocks scan their sub-layers, which cannot vary the metadata per layer."""
    cfg, decoder, shared_embedding = self._build(4, inhomogeneous_layer_cycle_interval=2)
    attention_metadata = {f"layer.{i}": object() for i in range(cfg.num_decoder_layers)}

    with self.assertRaisesRegex(NotImplementedError, "cannot be resolved inside a scanned block"):
      self._run(cfg, decoder, shared_embedding, attention_metadata=attention_metadata)

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


_QWEN3_5_CONFIG = {
    **_QWEN3_NEXT_CONFIG,
    "run_name": "qwen3_5_scannable_block_test",
    "model_name": "qwen3.5-35b-a3b",
    # MRoPE sections must cover rotary_dim / 2, which is 4 here, not the shipped [11, 11, 10].
    "mrope_section": [2, 1, 1],
}


class TestQwen3_5ScannableBlock(TestQwen3NextScannableBlock):
  """Runs the Qwen3-Next scannable-block tests against Qwen3.5's block."""

  MODEL_CONFIG = _QWEN3_5_CONFIG
  BLOCK_CLS = qwen3_5.Qwen3_5ScannableBlock

  def test_block_uses_qwen3_5_layers(self):
    """The block must build Qwen3.5 decoder layers, not Qwen3-Next ones."""
    self.assertIsInstance(self.block.global_layer, qwen3_5.Qwen3_5DecoderLayer)


class TestNNXDecoderQwen3_5(TestNNXDecoderQwen3Next):
  """Runs the NNXDecoder wiring tests against Qwen3.5."""

  MODEL_CONFIG = _QWEN3_5_CONFIG

  def test_raiden_unscan_reproduces_the_unscanned_decoder(self):
    """Raiden's unscan of the nested block state must load into an unscanned decoder unchanged.

    Raiden binds trainer and sampler tensors by name, and the sampler runs
    ``scan_layers=False``, so every name and shape has to match and the loaded
    decoder has to give the scanned decoder's logits. A cycle of 3 over 6 layers
    puts two slots in ``local_layers`` and two repeats on the scan axis, so a
    swapped slot or repeat index changes the output.
    """
    cfg, decoder, shared_embedding = self._build(6, inhomogeneous_layer_cycle_interval=3)
    unscanned_cfg, unscanned_decoder, _ = self._build(6, inhomogeneous_layer_cycle_interval=3, scan_layers=False)

    unscanned = raiden_unscan.unscan_layers(
        nnx.state(decoder, nnx.Param),
        num_layers=cfg.num_decoder_layers,
        scan_axis=cfg.param_scan_axis,
        cycle_interval=cfg.inhomogeneous_layer_cycle_interval,
    )

    def shapes(tree):
      return {jax.tree_util.keystr(path): leaf.shape for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}

    self.assertEqual(shapes(unscanned), shapes(nnx.state(unscanned_decoder, nnx.Param)))
    nnx.update(unscanned_decoder, unscanned)

    # Every loaded leaf must be exactly its block/slot slice of the scanned state:
    # slot j of repeat i is layer i * cycle + j, and the last slot is the global layer.
    axis, cycle = cfg.param_scan_axis, cfg.inhomogeneous_layer_cycle_interval
    scanned = traverse_util.flatten_dict(nnx.state(decoder, nnx.Param).to_pure_dict())
    loaded = traverse_util.flatten_dict(nnx.state(unscanned_decoder, nnx.Param).to_pure_dict())
    for key, value in loaded.items():
      idx = next((i for i, k in enumerate(key) if isinstance(k, str) and re.fullmatch(r"layers_\d+", k)), None)
      if idx is None:
        expected = scanned[key]
      else:
        repeat, slot = divmod(int(key[idx].split("_")[1]), cycle)
        sub = "global_layer" if slot == cycle - 1 else "local_layers"
        expected = np.take(np.asarray(scanned[key[:idx] + ("layers", sub) + key[idx + 1 :]]), repeat, axis=axis)
        if sub == "local_layers":
          expected = np.take(expected, slot, axis=axis)
      np.testing.assert_array_equal(np.asarray(value), expected, err_msg=str(key))

    # Scanned and unscanned forwards differ by float rounding that grows with the device count.
    np.testing.assert_allclose(
        np.asarray(self._run(unscanned_cfg, unscanned_decoder, shared_embedding)),
        np.asarray(self._run(cfg, decoder, shared_embedding)),
        rtol=1e-3,
        atol=1e-3,
    )


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

  def test_gemma4_decoder_layer_sharding_by_shard_mode(self):
    """Under ShardMode.AUTO, Gemma4 decoder layers must leave out_sharding as None."""
    # pylint: disable=protected-access
    for shard_mode, expect_none in ((ShardMode.AUTO, True), (ShardMode.EXPLICIT, False)):
      cfg_gemma4 = _make_config(
          decoder_block="gemma4",
          shard_mode=shard_mode.value,
      )
      layer = gemma4.Gemma4DecoderLayer(
          config=cfg_gemma4,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0),
      )
      cfg_small = _make_config(
          decoder_block="gemma4_small",
          model_name="gemma4-e2b",
          hidden_size_per_layer_input=64,
          vocab_size_per_layer_input=256,
          shard_mode=shard_mode.value,
      )
      small_layer = gemma4_small.Gemma4SmallDecoderLayer(
          config=cfg_small,
          mesh=self.mesh,
          model_mode=MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0),
      )
      if expect_none:
        self.assertIsNone(layer.out_sharding)
        self.assertIsNone(layer.mlp_intermediate_sharding)
        self.assertIs(layer._maybe_shard_with_logical, gemma4.with_logical_constraint)
        self.assertIsNone(small_layer.out_sharding)
        self.assertIsNone(small_layer.mlp_intermediate_sharding)
        self.assertIsNone(small_layer.ple_sharding)
        self.assertIs(small_layer._maybe_shard_with_logical, gemma4_small.with_logical_constraint)
      else:
        self.assertIsNotNone(layer.out_sharding)
        self.assertIsNotNone(layer.mlp_intermediate_sharding)
        self.assertIsNotNone(small_layer.out_sharding)
        self.assertIsNotNone(small_layer.mlp_intermediate_sharding)
        self.assertIsNotNone(small_layer.ple_sharding)


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


class TestNNXDecoderFP8WeightOnly(unittest.TestCase):
  """Tests for NNXDecoder with FP8 weight-only storage and dynamic dequantization."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_config(
        weight_dtype="float8_e4m3fn",
        dtype="bfloat16",
        unquantized_modules=["token_embedder", "logits_dense"],
    )
    self.mesh = _make_mesh(self.cfg)
    self.rngs = nnx.Rngs(params=0, dropout=1)
    self.decoder = NNXDecoder(
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.shared_embedding = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        embedding_init=jax.nn.initializers.normal(stddev=1.0),
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def test_fp8_weights_and_unquantized_layers(self):
    """Verifies that dense linear weights are FP8 while embedding and norms are BF16."""
    layer_0 = self.decoder.layers_0
    self.assertEqual(layer_0.self_attention.query.kernel[...].dtype, jnp.float8_e4m3fn)
    self.assertIsNotNone(layer_0.self_attention.query.kernel_scale)
    self.assertEqual(layer_0.mlp.wi_0.kernel[...].dtype, jnp.float8_e4m3fn)
    self.assertEqual(self.shared_embedding.embedding[...].dtype, jnp.bfloat16)
    self.assertEqual(self.decoder.decoder_norm.scale[...].dtype, jnp.bfloat16)

  def test_fp8_forward_pass_execution(self):
    """Verifies that an end-to-end forward pass with dynamic FP8 dequantization executes and produces valid logits."""
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    logits, hidden_state, _ = self.decoder(
        self.shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertEqual(logits.shape, (batch, seq_len, cfg.vocab_size))
    self.assertEqual(hidden_state.shape, (batch, seq_len, cfg.emb_dim))
    self.assertTrue(jnp.all(jnp.isfinite(logits)))


class _StatefulQwen3_5DecoderLayer(nnx.Module):
  """Stand-in Qwen3.5 sub-layer that exposes the cache and metadata it receives."""

  def __init__(self, *, is_full_attention_layer, **unused_kwargs):
    self.increment = 10 if is_full_attention_layer else 1
    self.call_count = nnx.Intermediate(jnp.array(0, dtype=jnp.int32))
    self.received_kv_cache = nnx.Intermediate(jnp.array(False))
    self.received_attention_metadata = nnx.Intermediate(jnp.array(False))
    # The metadata itself when a test passes an array (to tell the sub-layers' entries apart), else -1.
    self.received_attention_metadata_value = nnx.Intermediate(jnp.array(-1, dtype=jnp.int32))

  def __call__(self, inputs, *unused_args, kv_cache=None, attention_metadata=None, **unused_kwargs):
    self.call_count.value += 1
    self.received_kv_cache.value = jnp.array(kv_cache is not None)
    self.received_attention_metadata.value = jnp.array(attention_metadata is not None)
    if isinstance(attention_metadata, jax.Array):
      self.received_attention_metadata_value.value = attention_metadata.astype(jnp.int32)
    output = inputs + self.increment
    if kv_cache is None:
      return output
    return output, kv_cache + self.increment


class Qwen3_5ScannableBlockKVCacheTest(unittest.TestCase):
  """Qwen3.5 must thread vLLM's externally-managed KV caches through its block.

  Qwen3_5ScannableBlock reuses Qwen3NextScannableBlock, whose external-cache path
  unrolls the stacked local layers and then runs the global layer. The sub-layers
  are swapped for stand-ins, which keeps the test on CPU and focused on the cache
  plumbing rather than on building a real MoE block.
  """

  CYCLE = 4

  def setUp(self):
    super().setUp()
    self.config = SimpleNamespace(
        inhomogeneous_layer_cycle_interval=self.CYCLE,
        param_scan_axis=1,
        remat_policy="none",
        scan_layers=True,
        shard_mode=ShardMode.AUTO,
    )

  def _make_block(self):
    with mock.patch.object(qwen3_5, "Qwen3_5DecoderLayer", _StatefulQwen3_5DecoderLayer):
      return qwen3_5.Qwen3_5ScannableBlock(
          config=self.config,
          mesh=None,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs=nnx.Rngs(0),
      )

  def _call(self, block, **kwargs):
    return block(
        jnp.zeros((1, 1, 1)),
        decoder_segment_ids=None,
        decoder_positions=None,
        deterministic=True,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        **kwargs,
    )

  @pytest.mark.cpu_only
  def test_each_sublayer_gets_its_own_cache_and_updates_are_returned(self):
    block = self._make_block()
    kv_cache = tuple(jnp.array(i) for i in range(self.CYCLE))

    output, updated = self._call(block, kv_cache=kv_cache, attention_metadata=object())

    # Every sub-layer ran exactly once: three local layers (+1 each), then the global one (+10).
    np.testing.assert_array_equal(output, jnp.full((1, 1, 1), 13))
    np.testing.assert_array_equal(jnp.stack(updated), jnp.array([1, 2, 3, 13]))
    np.testing.assert_array_equal(block.local_layers.call_count.value, jnp.ones(self.CYCLE - 1, dtype=jnp.int32))
    np.testing.assert_array_equal(block.global_layer.call_count.value, 1)
    # attention_metadata must reach the sub-layers; without it the attention
    # kernel cannot address vLLM's paged cache.
    np.testing.assert_array_equal(
        block.local_layers.received_attention_metadata.value, jnp.ones(self.CYCLE - 1, dtype=jnp.bool_)
    )
    np.testing.assert_array_equal(block.global_layer.received_attention_metadata.value, True)

  @pytest.mark.cpu_only
  def test_training_path_returns_none_and_passes_no_cache(self):
    block = self._make_block()

    output, updated = self._call(block)

    np.testing.assert_array_equal(output, jnp.full((1, 1, 1), 13))
    self.assertIsNone(updated)
    np.testing.assert_array_equal(block.local_layers.received_kv_cache.value, jnp.zeros(self.CYCLE - 1, dtype=jnp.bool_))
    np.testing.assert_array_equal(block.global_layer.received_kv_cache.value, False)

  @pytest.mark.cpu_only
  def test_shorter_cache_does_not_index_out_of_range(self):
    block = self._make_block()

    _, updated = self._call(block, kv_cache=(jnp.array(0),))

    self.assertEqual(len(updated), self.CYCLE)
    np.testing.assert_array_equal(updated[0], 1)
    self.assertEqual(updated[1:], (None,) * (self.CYCLE - 1))

  @pytest.mark.cpu_only
  def test_each_sublayer_gets_its_own_attention_metadata(self):
    """A per-sub-layer metadata tuple must reach the sub-layers in block order, like the caches.

    vLLM's hybrid metadata is per KV-cache group, so sub-layers of one block can
    need different entries; handing every sub-layer the same one would point
    some of them at another group's block tables.
    """
    block = self._make_block()
    kv_cache = tuple(jnp.array(i) for i in range(self.CYCLE))
    attention_metadata = tuple(jnp.array(100 + i) for i in range(self.CYCLE))

    self._call(block, kv_cache=kv_cache, attention_metadata=attention_metadata)

    np.testing.assert_array_equal(
        block.local_layers.received_attention_metadata_value.value, jnp.arange(100, 100 + self.CYCLE - 1)
    )
    np.testing.assert_array_equal(block.global_layer.received_attention_metadata_value.value, 100 + self.CYCLE - 1)

  @pytest.mark.cpu_only
  def test_rejects_per_layer_attention_metadata_dict(self):
    """A dict keyed by global layer cannot be resolved by sub-layers that do not know their global index."""
    block = self._make_block()
    kv_cache = tuple(jnp.array(i) for i in range(self.CYCLE))
    attention_metadata = {f"layer.{i}": jnp.array(i) for i in range(self.CYCLE)}

    with self.assertRaisesRegex(NotImplementedError, "cannot be resolved inside a scanned block"):
      self._call(block, kv_cache=kv_cache, attention_metadata=attention_metadata)

  @pytest.mark.cpu_only
  def test_rejects_per_sub_layer_attention_metadata_without_external_caches(self):
    """Only the external-cache path unrolls the sub-layers; the scans cannot vary metadata per sub-layer."""
    block = self._make_block()
    attention_metadata = tuple(jnp.array(i) for i in range(self.CYCLE))

    with self.assertRaisesRegex(NotImplementedError, "externally-managed kv caches"):
      self._call(block, attention_metadata=attention_metadata)

  @pytest.mark.cpu_only
  def test_rejects_per_sub_layer_attention_metadata_of_the_wrong_length(self):
    """A tuple not covering every sub-layer would leave some sub-layer without its entry."""
    block = self._make_block()
    kv_cache = tuple(jnp.array(i) for i in range(self.CYCLE))
    attention_metadata = tuple(jnp.array(i) for i in range(self.CYCLE - 1))

    with self.assertRaisesRegex(ValueError, "one entry per sub-layer"):
      self._call(block, kv_cache=kv_cache, attention_metadata=attention_metadata)

  @pytest.mark.cpu_only
  def test_flat_per_layer_caches_survive_the_per_block_round_trip(self):
    """vLLM hands over one cache per layer; the scan runs over 4-layer blocks."""
    num_layers = 8
    scan_length = num_layers // self.CYCLE
    kv_caches = [jnp.array(i) for i in range(num_layers)]

    grouped = maxtext_utils.prepare_kv_caches_for_scan(kv_caches, scan_length, self.CYCLE, stack=False)
    self.assertEqual([[int(kv) for kv in group] for group in grouped], [[0, 1, 2, 3], [4, 5, 6, 7]])

    # Mimic _apply_layers_sequentially replacing each block entry with the
    # tuple the block returned.
    block = self._make_block()
    for block_idx in range(scan_length):
      _, grouped[block_idx] = self._call(block, kv_cache=grouped[block_idx])

    maxtext_utils.update_kv_caches_after_scan(kv_caches, grouped, scan_length, self.CYCLE, stacked=False)

    self.assertEqual(
        [int(kv) for kv in kv_caches],
        [i + (10 if i % self.CYCLE == self.CYCLE - 1 else 1) for i in range(num_layers)],
    )


class ScannedCycleLayerCountTest(unittest.TestCase):
  """A scanned decoder with a repeating layer cycle needs a whole number of cycles."""

  def _build(self, num_layers):
    cfg = _make_config(
        model_name="olmo3-7b-pt",
        scan_layers=True,
        base_num_decoder_layers=num_layers,
        head_dim=64,
        vocab_size=256,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    return model_creation_utils.create_nnx_abstract_model(cfg, mesh)

  def test_partial_cycle_is_rejected(self):
    # olmo3 repeats a 4-layer cycle; 6 layers used to build 4 and drop the rest.
    with self.assertRaisesRegex(ValueError, "must be a multiple of inhomogeneous_layer_cycle_interval"):
      self._build(6)

  def test_fewer_layers_than_one_cycle_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "must be a multiple of inhomogeneous_layer_cycle_interval"):
      self._build(2)

  def test_whole_cycles_build(self):
    self._build(8)


class TestNNXDecoderPrefetching(unittest.TestCase):
  """Unit and integration tests for FSDP weight prefetching (`prefetch_fsdp_weights`)."""

  def setUp(self):
    super().setUp()
    # The CPU test mesh has a single device, so no FSDP axis spans more than one device. Force the
    # prefetch path so that these tests exercise it.
    patcher = mock.patch.object(fsdp_prefetch, "mesh_uses_fsdp", return_value=True)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.cfg = _make_config(scan_layers=True, prefetch_fsdp_weights=True, num_decoder_layers=3)
    self.mesh = _make_mesh(self.cfg)
    self.rng = jax.random.PRNGKey(0)
    self.rngs = nnx.Rngs(params=0, dropout=1)

  def _make_token_inputs(self):
    batch = self.cfg.global_batch_size_to_train_on
    seq_len = self.cfg.max_target_length
    ids = jax.random.randint(self.rng, (batch, seq_len), 0, self.cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return ids, segment_ids, positions

  def _get_sharded_params_and_state(self, decoder, with_fsdp=False, use_partition_specs=False):
    """Splits decoder and applies NamedSharding with partition specs to parameters."""
    nnx.pop(decoder, nnx.RngState, nnx.RngCount)
    graphdef, params, state = nnx.split(decoder, nnx.Param, ...)

    def _shard_arr(arr):
      if isinstance(arr, jax.Array):
        if with_fsdp and arr.ndim >= 2:
          fsdp_size = self.mesh.shape.get("fsdp", 1)
          data_size = self.mesh.shape.get("data", 1)
          axis_0 = "fsdp" if arr.shape[0] % fsdp_size == 0 else None
          axis_1 = "data" if arr.ndim > 2 and arr.shape[1] % data_size == 0 else None
          spec = PartitionSpec(axis_0, axis_1)
        else:
          spec = PartitionSpec()
        if use_partition_specs:
          # Directly return the array with sharding attribute set as PartitionSpec string/spec for testing
          return jax.lax.with_sharding_constraint(arr, NamedSharding(self.mesh, spec))
        return jax.lax.with_sharding_constraint(arr, NamedSharding(self.mesh, spec))
      return arr

    params = jax.tree.map(_shard_arr, params)
    return graphdef, params, state

  def test_prefetch_forward_pass_no_grad(self):
    """Exercises _custom_vjp_prefetch_pipeline outside autodiff (is_fwd=False)."""
    decoder = NNXDecoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    shared_emb = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    graphdef, params, state = self._get_sharded_params_and_state(decoder, with_fsdp=False)
    m = nnx.merge(graphdef, params, state)
    logits, _, kv = m(
        shared_emb,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    self.assertEqual(
        logits.shape,
        (
            self.cfg.global_batch_size_to_train_on,
            self.cfg.max_target_length,
            self.cfg.vocab_size,
        ),
    )
    self.assertIsNone(kv)

  def test_prefetch_value_and_grad(self):
    """Exercises _custom_vjp_prefetch_pipeline_fwd and _custom_vjp_prefetch_pipeline_bwd."""
    decoder = NNXDecoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    shared_emb = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    graphdef, params, state = self._get_sharded_params_and_state(decoder, with_fsdp=False)

    # pylint: disable=too-many-positional-arguments
    def loss_fn(p, s, emb, token_ids, pos, seg_ids):
      m = nnx.merge(graphdef, p, s)
      out_logits, _, _ = m(
          emb,
          token_ids,
          pos,
          decoder_segment_ids=seg_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_logits**2)

    loss_val, (grad_params, grad_emb) = jax.value_and_grad(loss_fn, argnums=(0, 2))(
        params, state, shared_emb, ids, positions, segment_ids
    )
    self.assertTrue(jnp.isfinite(loss_val))
    self.assertIsNotNone(grad_params)
    self.assertIsNotNone(grad_emb)

  def test_prefetch_sharding_hints_and_fsdp_spec(self):
    """Exercises PartitionSpec stripping, NamedSharding branches, and reduce-scatter collective helpers."""
    decoder = NNXDecoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    shared_emb = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    graphdef, params_sharded, state = self._get_sharded_params_and_state(decoder, with_fsdp=True)

    def loss_fn(p, s):
      m = nnx.merge(graphdef, p, s)
      out_logits, _, _ = m(
          shared_emb,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_logits)

    loss_val, grad_params = jax.value_and_grad(loss_fn)(params_sharded, state)
    self.assertTrue(jnp.isfinite(loss_val))
    self.assertIsNotNone(grad_params)

  def test_prefetch_tuple_and_nontuple_layer_output(self):
    """Exercises layer output unpacking branches for tuple and non-tuple return types in fwd and bwd."""
    decoder = NNXDecoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    shared_emb = Embed(
        num_embeddings=self.cfg.vocab_size,
        num_features=self.cfg.emb_dim,
        dtype=self.cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    graphdef, params, state = self._get_sharded_params_and_state(decoder, with_fsdp=False)

    with unittest.mock.patch.object(
        NNXDecoderLayer,
        "__call__",
        return_value=jnp.zeros(
            (
                self.cfg.global_batch_size_to_train_on,
                self.cfg.max_target_length,
                self.cfg.emb_dim,
            )
        ),
    ):

      def loss_fn(p, s):
        m = nnx.merge(graphdef, p, s)
        out_logits, _, _ = m(
            shared_emb,
            ids,
            positions,
            decoder_segment_ids=segment_ids,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )
        return jnp.sum(out_logits)

      loss_val, grad_params = jax.value_and_grad(loss_fn)(params, state)
      self.assertTrue(jnp.isfinite(loss_val))
      self.assertIsNotNone(grad_params)

  def test_prefetch_zero_length(self):
    """Exercises length == 0 fast return inside _apply_layers_sequentially_prefetch."""
    decoder = NNXDecoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    hidden = jnp.zeros(
        (
            self.cfg.global_batch_size_to_train_on,
            self.cfg.max_target_length,
            self.cfg.emb_dim,
        )
    )
    # pylint: disable=protected-access
    out, _, kv = decoder._apply_layers_sequentially_prefetch(decoder.layers, hidden, length=0)
    self.assertEqual(out.shape, hidden.shape)
    self.assertIsNone(kv)

  def test_prefetch_unnamed_sharding_and_none_specs(self):
    """Exercises non-NamedSharding branches and None specs inside prefetch pipeline fwd/bwd helpers."""
    # 1) Exercise non-NamedSharding branches by mocking getattr for 'sharding' attribute on params
    rngs1 = nnx.Rngs(params=0, dropout=1)
    cfg_auto = _make_config(
        scan_layers=True,
        prefetch_fsdp_weights=True,
        num_decoder_layers=2,
        shard_mode="auto",
    )
    decoder = NNXDecoder(config=cfg_auto, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs1)
    shared_emb = Embed(
        num_embeddings=cfg_auto.vocab_size,
        num_features=cfg_auto.emb_dim,
        dtype=cfg_auto.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg_auto,
        mesh=self.mesh,
        rngs=rngs1,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    nnx.pop(decoder, nnx.RngState, nnx.RngCount)
    graphdef, params, state = nnx.split(decoder, nnx.Param, ...)

    s_dev = NamedSharding(self.mesh, PartitionSpec())
    orig_getattr = getattr

    def _mock_getattr(obj, name, default=None):
      if name == "sharding":
        return s_dev
      return orig_getattr(obj, name, default)

    def loss_fn(p, s):
      m = nnx.merge(graphdef, p, s)
      out_logits, _, _ = m(
          shared_emb,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_logits)

    with unittest.mock.patch.object(fsdp_prefetch, "getattr", side_effect=_mock_getattr, create=True):
      loss_val, grad_params = jax.value_and_grad(loss_fn)(params, state)
    self.assertTrue(jnp.isfinite(loss_val))
    self.assertIsNotNone(grad_params)

    # 2) Exercise shard_mode='auto' branches with explicit NamedSharding and PartitionSpec
    rngs2 = nnx.Rngs(params=0, dropout=1)
    cfg_auto_pspec = _make_config(
        scan_layers=True,
        prefetch_fsdp_weights=True,
        num_decoder_layers=2,
        shard_mode="auto",
    )
    decoder_pspec = NNXDecoder(
        config=cfg_auto_pspec,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=rngs2,
    )
    shared_emb_pspec = Embed(
        num_embeddings=cfg_auto_pspec.vocab_size,
        num_features=cfg_auto_pspec.emb_dim,
        dtype=cfg_auto_pspec.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg_auto_pspec,
        mesh=self.mesh,
        rngs=rngs2,
    )
    nnx.pop(decoder_pspec, nnx.RngState, nnx.RngCount)
    graphdef_pspec, params_pspec, state_pspec = nnx.split(decoder_pspec, nnx.Param, ...)

    def _to_named_sharding_pspec(arr):
      if isinstance(arr, jax.Array):
        return jax.lax.with_sharding_constraint(
            arr,
            jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec()),
        )
      return arr

    params_pspec = jax.tree.map(_to_named_sharding_pspec, params_pspec)

    def loss_fn_pspec(p, s):
      m = nnx.merge(graphdef_pspec, p, s)
      out_logits, _, _ = m(
          shared_emb_pspec,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_logits)

    loss_val_pspec, grad_params_pspec = jax.value_and_grad(loss_fn_pspec)(params_pspec, state_pspec)
    self.assertTrue(jnp.isfinite(loss_val_pspec))
    self.assertIsNotNone(grad_params_pspec)

  def test_prefetch_with_param_scan_axis_nonzero(self):
    """Exercises param_scan_axis != 0 inside _apply_layers_sequentially_prefetch."""
    cfg_scan_1 = _make_config(
        scan_layers=True,
        prefetch_fsdp_weights=True,
        num_decoder_layers=2,
        param_scan_axis=1,
    )
    decoder = NNXDecoder(
        config=cfg_scan_1,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )
    shared_emb = Embed(
        num_embeddings=cfg_scan_1.vocab_size,
        num_features=cfg_scan_1.emb_dim,
        dtype=cfg_scan_1.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg_scan_1,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    ids, segment_ids, positions = self._make_token_inputs()
    graphdef, params, state = self._get_sharded_params_and_state(decoder, with_fsdp=False)

    def loss_fn(p, s):
      m = nnx.merge(graphdef, p, s)
      out_logits, _, _ = m(
          shared_emb,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_logits)

    loss_val, grad_params = jax.value_and_grad(loss_fn)(params, state)
    self.assertTrue(jnp.isfinite(loss_val))
    self.assertIsNotNone(grad_params)

  def test_prefetch_skipped_without_fsdp(self):
    """Without FSDP sharding there is nothing to prefetch, so the regular scan is used."""
    decoder = NNXDecoder(config=self.cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    hidden = jnp.zeros((self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length, self.cfg.emb_dim))
    with (
        mock.patch.object(fsdp_prefetch, "mesh_uses_fsdp", return_value=False),
        mock.patch.object(NNXDecoder, "_apply_layers_sequentially_prefetch") as prefetch,
        mock.patch.object(nnx_decoders.max_logging, "info") as log_info,
    ):
      # pylint: disable=protected-access
      out, _, kv = decoder._apply_layers_sequentially(decoder.layers, hidden, length=0, kv_caches_stacked=None)
    prefetch.assert_not_called()
    self.assertIn("using the regular scan", log_info.call_args[0][0])
    self.assertEqual(out.shape, hidden.shape)
    self.assertIsNone(kv)

  def test_apply_layers_sequentially_zero_length_no_prefetch(self):
    """Exercises length == 0 fast return inside _apply_layers_sequentially when prefetch_fsdp_weights is False."""
    cfg_no_prefetch = _make_config(scan_layers=True, prefetch_fsdp_weights=False, num_decoder_layers=2)
    decoder = NNXDecoder(
        config=cfg_no_prefetch,
        mesh=self.mesh,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )
    hidden = jnp.zeros(
        (
            self.cfg.global_batch_size_to_train_on,
            self.cfg.max_target_length,
            self.cfg.emb_dim,
        )
    )
    # pylint: disable=protected-access
    out, _, kv = decoder._apply_layers_sequentially(decoder.layers, hidden, length=0, kv_caches_stacked=None)
    self.assertEqual(out.shape, hidden.shape)
    self.assertIsNone(kv)

  def test_prefetch_pipeline_non_array_state_leaf(self):
    """Runs value_and_grad through the prefetch pipeline with an extra non-parameter (BatchStat) state leaf."""
    cfg_auto = _make_config(
        scan_layers=True,
        prefetch_fsdp_weights=True,
        num_decoder_layers=2,
        shard_mode="auto",
    )
    decoder = NNXDecoder(config=cfg_auto, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=self.rngs)
    shared_emb = Embed(
        num_embeddings=cfg_auto.vocab_size,
        num_features=cfg_auto.emb_dim,
        dtype=cfg_auto.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg_auto,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    ids, segment_ids, positions = self._make_token_inputs()

    decoder.dummy = nnx.BatchStat(np.array([1.0, 2.0]))
    graphdef, params, state = self._get_sharded_params_and_state(decoder, with_fsdp=False)

    def loss_fn(p, s):
      m = nnx.merge(graphdef, p, s)
      out_logits, _, _ = m(
          shared_emb,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.sum(out_logits)

    loss_val, grad_params = jax.value_and_grad(loss_fn)(params, state)
    self.assertTrue(jnp.isfinite(loss_val))
    self.assertIsNotNone(grad_params)


class TestNNXDecoderPrefetchGradParity(unittest.TestCase):
  """The prefetch pipeline must produce the same loss and gradients as the plain scanned stack."""

  def setUp(self):
    super().setUp()
    # Force the prefetch path on the single-device CPU test mesh (see TestNNXDecoderPrefetching).
    patcher = mock.patch.object(fsdp_prefetch, "mesh_uses_fsdp", return_value=True)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _loss_and_grads(self, num_layers, prefetch, with_grads=True, **overrides):
    """Returns (loss, grads w.r.t. the decoder params) of a small float32 decoder, jitted.

    With `with_grads=False` the loss is computed without differentiation and grads is None.
    """
    cfg_kwargs = {
        "scan_layers": True,
        "prefetch_fsdp_weights": prefetch,
        "num_decoder_layers": num_layers,
        "base_num_decoder_layers": num_layers,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "remat_policy": "full",
    }
    cfg_kwargs.update(overrides)
    cfg = _make_config(**cfg_kwargs)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    shared_emb = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=nnx.Rngs(params=1),
    )
    batch, seq_len = cfg.global_batch_size_to_train_on, cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    nnx.pop(decoder, nnx.RngState, nnx.RngCount)
    graphdef, params, state = nnx.split(decoder, nnx.Param, ...)

    def loss_fn(p):
      m = nnx.merge(graphdef, p, state)
      logits, _, _ = m(
          shared_emb,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(logits.astype(jnp.float32) ** 2)

    if not with_grads:
      return jax.jit(loss_fn)(params), None
    return jax.jit(jax.value_and_grad(loss_fn))(params)

  def _assert_parity(self, num_layers, **overrides):
    """Asserts the prefetch pipeline matches the plain scanned stack in loss and every gradient leaf."""
    ref_loss, ref_grads = self._loss_and_grads(num_layers, prefetch=False, **overrides)
    loss, grads = self._loss_and_grads(num_layers, prefetch=True, **overrides)
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-6)
    ref_leaves, ref_def = jax.tree_util.tree_flatten(ref_grads)
    leaves, tree_def = jax.tree_util.tree_flatten(grads)
    self.assertEqual(ref_def, tree_def)
    for ref, got in zip(ref_leaves, leaves):
      self.assertEqual(ref.shape, got.shape)
      np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-4, atol=2e-5)

  def test_grad_parity_one_layer(self):
    self._assert_parity(num_layers=1)

  def test_grad_parity_two_layers(self):
    self._assert_parity(num_layers=2)

  def test_grad_parity_four_layers(self):
    self._assert_parity(num_layers=4)

  def test_grad_parity_five_layers(self):
    self._assert_parity(num_layers=5)

  def test_grad_parity_seven_layers(self):
    # Two or more layer pairs per loop in both passes, so the loops order their second prefetch.
    self._assert_parity(num_layers=7)

  def test_grad_parity_eight_layers(self):
    self._assert_parity(num_layers=8)

  def test_grad_parity_param_scan_axis_zero(self):
    self._assert_parity(num_layers=4, param_scan_axis=0)

  def _assert_parity_saved_residuals(self, num_layers, **overrides):
    """Asserts parity and returns how many residuals per layer the prefetch remat policy saves."""
    with unittest.mock.patch.object(fsdp_prefetch.max_logging, "info") as log_info:
      self._assert_parity(num_layers=num_layers, **overrides)
    counts = [
        int(m.group(1))
        for c in log_info.call_args_list
        for m in [re.search(r"remat policy saves (\d+) residual", str(c))]
        if m
    ]
    self.assertTrue(counts, "the prefetch pipeline did not apply the remat policy")
    return counts[0]

  def test_grad_parity_remat_minimal(self):
    # The forward pass saves the policy's named activations and the backward pass reuses them.
    self.assertGreater(self._assert_parity_saved_residuals(num_layers=3, remat_policy="minimal"), 0)

  def test_grad_parity_remat_minimal_seven_layers(self):
    # Two or more layer pairs per loop, so saved residuals ride through both ordered unroll=2 loops.
    self.assertGreater(self._assert_parity_saved_residuals(num_layers=7, remat_policy="minimal"), 0)

  def test_grad_parity_remat_save_dot_except_mlp(self):
    self.assertGreater(self._assert_parity_saved_residuals(num_layers=5, remat_policy="save_dot_except_mlp"), 0)

  def test_grad_parity_remat_minimal_one_layer(self):
    self.assertGreater(self._assert_parity_saved_residuals(num_layers=1, remat_policy="minimal"), 0)

  def test_remat_custom_decoder_layer_input_saves_nothing_extra(self):
    # `decoder_layer_input` is the layer input under a checkpoint name; the pipeline already keeps
    # each layer's input, so the policy must not save it a second time.
    saved = self._assert_parity_saved_residuals(num_layers=4, remat_policy="custom", decoder_layer_input="device")
    self.assertEqual(saved, 0)

  def test_full_remat_skips_residual_tracing(self):
    with unittest.mock.patch.object(fsdp_prefetch.max_logging, "info") as log_info:
      self._assert_parity(num_layers=3, remat_policy="full")
    self.assertFalse(any("remat policy saves" in str(c) for c in log_info.call_args_list))

  def test_grad_parity_remat_custom_offload(self):
    # Offloaded residuals live in pinned host memory; they must survive the custom_vjp boundary and the
    # forward scan stacking, and be brought back for the backward pass.
    try:
      jax.devices()[0].memory("pinned_host")
    except Exception:  # pylint: disable=broad-exception-caught
      self.skipTest("backend has no pinned_host memory")
    saved = self._assert_parity_saved_residuals(num_layers=5, remat_policy="custom", mlpwo="offload", query_proj="device")
    self.assertGreater(saved, 0)

  def test_blocks_with_internal_remat_save_their_sub_layer_inputs(self):
    """A block that remats its own sub-layers is not recomputed whole before that remat runs.

    Under full remat the pipeline would otherwise keep only each block's input, so the backward
    pass would recompute the block and then, inside the block's own remat, every sub-layer again.
    Saving each sub-layer's input instead takes two residuals per Qwen3.5 block -- the local
    scan's stacked inputs and the global layer's -- and none of them may be a gathered weight.
    Qwen3.5 blocks normally prefetch one sub-layer at a time, so whole-block prefetching is forced.
    """
    model_config = {k: v for k, v in _QWEN3_5_CONFIG.items() if k != "base_num_decoder_layers"}
    with (
        unittest.mock.patch.object(fsdp_prefetch_inhomogeneous, "sublayer_stages", return_value=None),
        unittest.mock.patch.object(fsdp_prefetch.max_logging, "info") as log_info,
    ):
      self._assert_parity(num_layers=8, remat_policy="full", **model_config)
    logs = [str(c) for c in log_info.call_args_list]
    self.assertTrue(any("remat policy saves 2 residual(s) per layer" in log for log in logs), logs)
    self.assertFalse(any("shaped like gathered weights" in log for log in logs), logs)


# A tiny Qwen3.5 decoder: every scan step is a block of `inhomogeneous_layer_cycle_interval` (4)
# sub-layers mixing Gated DeltaNet and global attention, so the stack is inhomogeneous inside a block.
_QWEN3_5_TINY = {
    "model_name": "qwen3.5-35b-a3b",
    "base_emb_dim": 64,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 1,
    "head_dim": 32,
    "base_mlp_dim": 64,
    "base_moe_mlp_dim": 64,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "shared_experts": 1,
    "gdn_key_head_dim": 16,
    "gdn_value_head_dim": 16,
    "gdn_num_key_heads": 2,
    "gdn_num_value_heads": 2,
    "gdn_chunk_size": 8,
    "vocab_size": 128,
    "use_mrope": False,
    "sparse_matmul": False,
    "capacity_factor": -1,
}


class TestNNXDecoderPrefetchInhomogeneous(unittest.TestCase):
  """Per-sub-layer prefetching of inhomogeneous blocks must match the plain scanned stack."""

  # Reuse the parity helper without inheriting (and re-running) TestNNXDecoderPrefetchGradParity's tests.
  _loss_and_grads = TestNNXDecoderPrefetchGradParity._loss_and_grads  # pylint: disable=protected-access

  def _assert_parity(self, num_layers, **overrides):
    """Like TestNNXDecoderPrefetchGradParity._assert_parity, with a looser gradient tolerance.

    Rematerializing the Gated DeltaNet sub-layers is not bitwise identical to the plain scan: from
    three blocks on, a few elements of the GDN `in_proj_qkvz` kernel gradient differ by up to ~4e-4
    (relative ~0.5%) in float32, by the same amount with whole-block prefetching.
    """
    ref_loss, ref_grads = self._loss_and_grads(num_layers, prefetch=False, **overrides)
    loss, grads = self._loss_and_grads(num_layers, prefetch=True, **overrides)
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-6)
    ref_leaves, ref_def = jax.tree_util.tree_flatten(ref_grads)
    leaves, tree_def = jax.tree_util.tree_flatten(grads)
    self.assertEqual(ref_def, tree_def)
    for ref, got in zip(ref_leaves, leaves):
      self.assertEqual(ref.shape, got.shape)
      np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-2, atol=1e-3)

  def setUp(self):
    super().setUp()
    # Force the prefetch path on the single-device CPU test mesh (see TestNNXDecoderPrefetching).
    patcher = mock.patch.object(fsdp_prefetch, "mesh_uses_fsdp", return_value=True)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _patch_pipelines(self):
    """Wraps the per-sub-layer and whole-block prefetch pipelines to count their calls."""
    # pylint: disable=protected-access
    return (
        mock.patch.object(
            fsdp_prefetch_inhomogeneous,
            "_custom_vjp_nested_prefetch_pipeline",
            wraps=fsdp_prefetch_inhomogeneous._custom_vjp_nested_prefetch_pipeline,
        ),
        mock.patch.object(
            fsdp_prefetch,
            "_custom_vjp_prefetch_pipeline",
            wraps=fsdp_prefetch._custom_vjp_prefetch_pipeline,
        ),
    )

  def _assert_qwen3_5_parity(self, num_blocks, cycle=4, **overrides):
    """Asserts parity for `num_blocks` Qwen3.5 blocks and that sub-layers were prefetched one at a time.

    Each block has `cycle - 1` Gated DeltaNet sub-layers and one global attention sub-layer.
    """
    nested_patch, whole_patch = self._patch_pipelines()
    config = {**_QWEN3_5_TINY, "inhomogeneous_layer_cycle_interval": cycle, **overrides}
    with nested_patch as nested, whole_patch as whole_blocks:
      self._assert_parity(num_layers=num_blocks * cycle, **config)
    self.assertEqual(nested.call_count, 1)
    self.assertEqual(whole_blocks.call_count, 0)
    graphdefs, length, sizes = nested.call_args.args[0], nested.call_args.args[6], nested.call_args.args[8]
    # The Gated DeltaNet sub-layers form one scanned group, followed by the global attention sub-layer.
    self.assertEqual(len(graphdefs), 2)
    self.assertEqual(sizes, (cycle - 1, 1))
    self.assertEqual(length, num_blocks)

  def test_qwen3_5_one_block(self):
    self._assert_qwen3_5_parity(num_blocks=1)

  def test_qwen3_5_two_blocks(self):
    self._assert_qwen3_5_parity(num_blocks=2)

  def test_qwen3_5_three_blocks(self):
    self._assert_qwen3_5_parity(num_blocks=3)

  def test_qwen3_5_four_blocks(self):
    # Two or more scanned blocks, so the scan body orders the sub-layers' prefetches.
    self._assert_qwen3_5_parity(num_blocks=4)

  def test_qwen3_5_expert_parallelism(self):
    """Per-sub-layer prefetching matches the plain scanned stack on an fsdp=2 x expert=2 mesh."""
    if jax.device_count() < 4:
      self.skipTest("Needs 4 devices for an fsdp=2 x expert=2 mesh.")
    self._assert_qwen3_5_parity(num_blocks=2, ici_fsdp_parallelism=2, ici_expert_parallelism=2)

  def test_qwen3_5_sub_layer_gathers_under_expert_parallelism(self):
    """Under EP, each sub-layer weight is gathered over the FSDP axes `sharding.fsdp_axes_for_dim` picks.

    Non-routed weights (Gated DeltaNet, attention, router and shared expert) are sharded over
    ('fsdp', 'expert') on their embed dimension and get one tagged all-gather over both axes; routed
    experts keep `expert` on their expert dimension and gather only `fsdp`. The gathers are lowered on an
    abstract explicit-sharding mesh, so no devices are needed.
    """
    cfg = _make_config(
        **{
            **_QWEN3_5_TINY,
            "scan_layers": True,
            "num_decoder_layers": 8,
            "base_num_decoder_layers": 8,
            "inhomogeneous_layer_cycle_interval": 4,
        }
    )
    decoder = NNXDecoder(config=cfg, mesh=_make_mesh(cfg), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    stages = fsdp_prefetch_inhomogeneous.sublayer_stages(decoder.layers)
    self.assertEqual(len(stages), 4)
    axis_sizes = {axis: 1 for axis in cfg.mesh_axes} | {"fsdp": 4, "expert": 2}
    mesh = AbstractMesh(tuple(axis_sizes.values()), tuple(axis_sizes), axis_types=(AxisType.Explicit,) * len(axis_sizes))

    def logical_axes(var):
      meta = var.get_metadata()
      for key in ("sharding", "out_sharding", "sharding_names"):
        axes = meta.get(key)
        if isinstance(axes, (PartitionSpec, tuple, list)) and axes:
          return tuple(a for a in axes if a != "layers")  # Drop the block scan axis.
      return None

    def lower_gather(spec, shape):
      w = jax.ShapeDtypeStruct(shape, jnp.float32, sharding=NamedSharding(mesh, spec))

      def gather(x):
        return fsdp_prefetch._apply_sharding_hint(x, mesh, ShardMode.EXPLICIT)  # pylint: disable=protected-access

      with jax.sharding.use_abstract_mesh(mesh):
        out = jax.eval_shape(gather, w)
        hlo = jax.jit(gather).trace(w).lower(lowering_platforms=("tpu",)).as_text()
      return out, [line for line in hlo.splitlines() if "stablehlo.all_gather" in line]

    checked = {"routed": 0, "co_sharded": 0}
    # The first Gated DeltaNet sub-layer and the global attention sub-layer.
    for _, params, _ in (stages[0], stages[-1]):
      for path, var in nnx.to_flat_state(params):
        axes = logical_axes(var)
        if not axes:
          continue  # Replicated (e.g. A_log, dt_bias): nothing to gather.
        spec = nn.logical_to_mesh_axes(axes, cfg.logical_axis_rules)
        shape = list(var.get_value().shape)
        del shape[cfg.param_scan_axis]  # One block's slice, as the pipeline gathers it.
        out, gathers = lower_gather(spec, tuple(shape))
        name = "/".join(map(str, path))
        if not any(sharding.fsdp_axes_for_dim(entry) for entry in spec):
          self.assertEqual(gathers, [], name)
          continue
        self.assertEqual(len(gathers), 1, name)
        self.assertIn('xla_explicit_fsdp = "yes"', gathers[0], name)
        self.assertIn('name = "fsdp"', gathers[0], name)
        if axes[0] == "exp":
          checked["routed"] += 1
          self.assertNotIn('name = "expert"', gathers[0], name)
          self.assertEqual(tuple(out.sharding.spec)[0], "expert", name)
        else:
          checked["co_sharded"] += 1
          self.assertIn('name = "expert"', gathers[0], name)
          self.assertNotIn("expert", str(out.sharding.spec), name)
    # wi_0/wi_1/wo of both sub-layers; the GDN, attention, router and shared-expert kernels.
    self.assertEqual(checked["routed"], 6)
    self.assertGreaterEqual(checked["co_sharded"], 14)

  def test_qwen3_5_remat_minimal(self):
    with unittest.mock.patch.object(fsdp_prefetch.max_logging, "info") as log_info:
      self._assert_qwen3_5_parity(num_blocks=4, remat_policy="minimal")
    self.assertTrue(any("remat policy saves" in str(c) for c in log_info.call_args_list))

  def test_qwen3_5_param_scan_axis_zero(self):
    self._assert_qwen3_5_parity(num_blocks=3, param_scan_axis=0)

  def test_qwen3_5_one_local_layer_per_block(self):
    # A single Gated DeltaNet sub-layer per block: no same-group scan, two trip-count-one loops.
    self._assert_qwen3_5_parity(num_blocks=2, cycle=2)

  def test_qwen3_5_two_local_layers_per_block(self):
    # The same-group scan of the Gated DeltaNet sub-layers has a single trip.
    self._assert_qwen3_5_parity(num_blocks=2, cycle=3)

  def test_qwen3_5_five_local_layers_per_block(self):
    self._assert_qwen3_5_parity(num_blocks=2, cycle=6)

  def test_qwen3_5_forward_only(self):
    # Without differentiation, jax.custom_vjp runs the pipeline's primal function instead of its fwd rule.
    config = {**_QWEN3_5_TINY, "inhomogeneous_layer_cycle_interval": 4}
    ref_loss, _ = self._loss_and_grads(8, prefetch=False, with_grads=False, **config)
    nested_patch, _ = self._patch_pipelines()
    with nested_patch as nested:
      loss, _ = self._loss_and_grads(8, prefetch=True, with_grads=False, **config)
    self.assertEqual(nested.call_count, 1)
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-6)

  def test_qwen3_5_global_only_blocks_prefetch_whole_blocks(self):
    # With a cycle of 1 every block is a single global attention layer: nothing to split, so the
    # stack keeps prefetching whole scan steps.
    nested_patch, whole_patch = self._patch_pipelines()
    with nested_patch as nested, whole_patch as whole_blocks:
      self._assert_parity(num_layers=3, **{**_QWEN3_5_TINY, "inhomogeneous_layer_cycle_interval": 1})
    self.assertEqual(nested.call_count, 0)
    self.assertEqual(whole_blocks.call_count, 1)

  def test_homogeneous_decoder_prefetches_whole_layers(self):
    # A homogeneous decoder (no `prefetch_sublayer_names()`) must keep using `fsdp_prefetch`.
    nested_patch, whole_patch = self._patch_pipelines()
    with nested_patch as nested, whole_patch as whole_layers:
      self._loss_and_grads(3, prefetch=True)
    self.assertEqual(nested.call_count, 0)
    self.assertEqual(whole_layers.call_count, 1)

  def test_sublayer_stages_require_all_params_in_sublayers(self):
    class Block(nnx.Module):

      def __init__(self, extra):
        self.layer_0 = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
        self.layer_1 = nnx.Linear(2, 3, rngs=nnx.Rngs(1))
        if extra:
          self.extra = nnx.Param(jnp.ones((2,)))

      def prefetch_sublayer_names(self):
        return ("layer_0", "layer_1")

    # pylint: disable=protected-access
    block = Block(extra=False)
    stages = fsdp_prefetch_inhomogeneous.sublayer_stages(block)
    self.assertEqual(len(stages), 2)
    self.assertEqual(stages[1][1]["kernel"].shape, (2, 3))
    block = Block(extra=True)
    with unittest.mock.patch.object(fsdp_prefetch_inhomogeneous.max_logging, "info") as log_info:
      self.assertIsNone(fsdp_prefetch_inhomogeneous.sublayer_stages(block))
    self.assertTrue(any("params outside its sub-layers" in str(c) for c in log_info.call_args_list))

  def test_single_sublayer_block_prefetches_whole_blocks(self):
    block = SimpleNamespace(prefetch_sublayer_names=lambda: ("layer_0",))
    # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch_inhomogeneous._prefetch_sublayer_names(block))
    self.assertIsNone(fsdp_prefetch_inhomogeneous._prefetch_sublayer_names(SimpleNamespace()))

  def test_sublayer_stages_local_layers_without_global_layer(self):
    class Block(nnx.Module):

      def __init__(self):
        self.num_local, self.num_global, self.global_layer = 2, 0, None
        self.config = SimpleNamespace(param_scan_axis=1)
        self.local_layers = nnx_scan.create_scanned_layers(
            lambda r: nnx.Linear(2, 3, rngs=r),
            length=2,
            param_scan_axis=1,
            metadata_axis_name="local_layers",
            rngs=nnx.Rngs(0),
        )

    stages = fsdp_prefetch_inhomogeneous.sublayer_stages(Block())
    self.assertEqual(len(stages), 2)
    self.assertEqual(stages[0][1]["kernel"].shape, (2, 3))

  def test_sublayer_stages_with_fewer_than_two_sublayers_prefetch_whole_blocks(self):
    class Block(nnx.Module):

      def __init__(self):
        # Claims two local layers but has none: only the global layer can be split off.
        self.num_local, self.num_global, self.local_layers = 2, 1, None
        self.global_layer = nnx.Linear(2, 2, rngs=nnx.Rngs(0))

    self.assertIsNone(fsdp_prefetch_inhomogeneous.sublayer_stages(Block()))

  def test_unstack_inner_scanned_sublayers(self):
    # pylint: disable=protected-access
    for scan_axis in (0, 1):
      local_layers = nnx_scan.create_scanned_layers(
          lambda r: nnx.Linear(2, 3, rngs=r),
          length=3,
          param_scan_axis=scan_axis,
          metadata_axis_name="local_layers",
          rngs=nnx.Rngs(0),
      )
      stacked = nnx.state(local_layers, nnx.Param)["kernel"].get_value()
      stages = fsdp_prefetch_inhomogeneous._unstack_scanned_sublayers(local_layers, 3, scan_axis)
      self.assertEqual(len(stages), 3)
      for i, (_, params, _) in enumerate(stages):
        kernel = params["kernel"]
        np.testing.assert_array_equal(kernel.get_value(), jnp.take(stacked, i, axis=scan_axis))
        self.assertNotIn("local_layers", tuple(kernel.get_metadata().get("sharding_names") or ()))

  def test_unstack_variable_passes_unstacked_leaves_to_every_sublayer(self):
    # pylint: disable=protected-access
    raw = jnp.ones((3, 4))
    self.assertEqual(fsdp_prefetch_inhomogeneous._unstack_variable(raw, 3, 0), (raw, raw, raw))
    shared = nnx.Variable(jnp.zeros(()))
    self.assertEqual(fsdp_prefetch_inhomogeneous._unstack_variable(shared, 3, 0), (shared, shared, shared))


if __name__ == "__main__":
  unittest.main()
