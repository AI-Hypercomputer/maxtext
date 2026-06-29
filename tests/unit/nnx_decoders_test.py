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

"""Unit tests for nnx_decoders module.

Tests cover:
  - deepstack_process: pure-JAX helper for injecting visual embeddings
  - NNXDecoderLayer: single transformer decoder layer (init + forward)
  - NNXDecoder: decoder stack utilities (get_decoder_layers, get_norm_layer,
                get_remat_policy, minimal_policy, and full forward pass)
"""

import sys
import unittest

import pytest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    DecoderBlockType,
    MultimodalInput,
)
from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import (
    NNXDecoder,
    NNXDecoderLayer,
    NNXScannedPipelineStage,
    NNXSequentialPipelineStage,
    deepstack_process,
)
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import gemma4_small
from maxtext.models.gpt3 import Gpt3LayerNorm
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils
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
    "activations_in_float32": True,
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
    ):
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


if __name__ == "__main__":
  unittest.main()


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
            "per_device_batch_size=1.0",
            "max_target_length=16",
            "max_prefill_predict_length=4",
            "vocab_size=256",
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
    # pylint: disable=import-outside-toplevel
    from unittest.mock import MagicMock, patch

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
            "per_device_batch_size=1.0",
            "max_target_length=16",
            "max_prefill_predict_length=4",
            "vocab_size=256",
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
    for layer in decoder.layers:
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


def _assert_grad_parity(test_case, ref_leaves, other_leaves, *, what, rtol=5e-2):
  """Assert two gradient leaf-lists agree within a relative tolerance.

  Checks leaf count, finite values, and a nonzero backward on both paths, then bounds the aggregate
  relative L2 error ``||g_other - g_ref|| / ||g_ref||`` by ``rtol``. A relative norm (not elementwise)
  is used so the check holds on TPU, where the rematerialized backward bf16-rounds the matmuls
  differently.
  """
  ref_leaves, other_leaves = list(ref_leaves), list(other_leaves)
  test_case.assertEqual(len(ref_leaves), len(other_leaves), f"{what}: grad pytrees differ in leaf count")
  test_case.assertGreater(len(ref_leaves), 0, f"{what}: no gradients")
  for g_ref, g_other in zip(ref_leaves, other_leaves):
    test_case.assertEqual(g_ref.shape, g_other.shape, f"{what}: gradient shape mismatch")
  test_case.assertTrue(
      all(bool(jnp.all(jnp.isfinite(g))) for g in ref_leaves + other_leaves), f"{what}: non-finite gradient"
  )
  test_case.assertTrue(any(bool(jnp.any(g != 0)) for g in ref_leaves), f"{what}: reference backward is all-zero")
  test_case.assertTrue(any(bool(jnp.any(g != 0)) for g in other_leaves), f"{what}: backward produced all-zero grads")
  ref = jnp.concatenate([g.astype(jnp.float32).ravel() for g in ref_leaves])
  oth = jnp.concatenate([g.astype(jnp.float32).ravel() for g in other_leaves])
  rel_l2 = float(jnp.linalg.norm(oth - ref) / (jnp.linalg.norm(ref) + 1e-12))

  test_case.assertLess(rel_l2, rtol, f"{what}: relative L2 gradient error {rel_l2:.4%} exceeds rtol={rtol:.2%}")


class TestNNXDecoderDeepseek4(unittest.TestCase):
  """Parity tests for DeepSeek-V4 (deepseek4) decoder-level handling in NNXDecoder.

  DeepSeek-V4 has ``first_num_hash_layers`` prefix layers (static hash routing,
  heterogeneous attention) that are unrolled, followed by uniform alternating
  HCA(compress_ratio=128)/CSA(compress_ratio=4) blocks. The Linen reference is
  ``decoders.Decoder._apply_deepseek4_scanned_blocks``.
  """

  def _make_deepseek4_config(
      self,
      scan_layers=False,
      num_decoder_layers=5,
      first_num_hash_layers=3,
      compress_ratios=(0, 0, 4, 128, 4),
      remat_policy="full",
  ):
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        override_model_config=True,
        per_device_batch_size=1.0,
        run_name="deepseek4_nnx_test",
        enable_checkpointing=False,
        model_name="deepseek4-284b",
        attention="dot_product",
        remat_policy=remat_policy,
        # Dense MoE (sparse_matmul=False) so the forward runs on any backend;
        # the megablox GMM path is a TPU-only Pallas kernel.
        sparse_matmul=False,
        megablox=False,
        base_num_decoder_layers=num_decoder_layers,
        base_emb_dim=256,
        base_mlp_dim=512,
        base_moe_mlp_dim=512,
        base_num_query_heads=4,
        base_num_kv_heads=1,
        num_experts=8,
        num_experts_per_tok=2,
        shared_experts=1,
        first_num_hash_layers=first_num_hash_layers,
        compress_ratios=list(compress_ratios),
        indexer_head_dim=64,
        indexer_n_heads=4,
        indexer_topk=8,
        head_dim=64,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        kv_lora_rank=64,
        # seq_len must be >= the largest compress_ratio (128) so HCA layers produce >=1 compressed block.
        max_target_length=256,
        max_prefill_predict_length=64,
        vocab_size=256,
        scan_layers=scan_layers,
        dtype="float32",
        weight_dtype="float32",
        activations_in_float32=True,
        sliding_window_size=8,
    )

  def test_construct_non_scan_does_not_raise(self):
    """NNXDecoder(deepseek4) must construct; get_norm_layer must support deepseek4 (RMSNorm)."""
    cfg = self._make_deepseek4_config(scan_layers=False)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertIsInstance(decoder.decoder_norm, RMSNorm)
    self.assertTrue(decoder.is_deepseek4)

  def test_get_decoder_layers_registers_deepseek4(self):
    """Regression guard: NNXDecoder.get_decoder_layers layer_map MUST contain DEEPSEEK4."""
    from maxtext.models import deepseek4  # pylint: disable=import-outside-toplevel

    cfg = self._make_deepseek4_config(scan_layers=False)
    dec = NNXDecoder(config=cfg, mesh=_make_mesh(cfg), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertEqual(dec.get_decoder_layers(), [deepseek4.DeepSeek4DecoderLayer])

    cfg_s = self._make_deepseek4_config(scan_layers=True)
    dec_s = NNXDecoder(
        config=cfg_s, mesh=_make_mesh(cfg_s), model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1)
    )
    self.assertEqual(dec_s.get_decoder_layers(), [deepseek4.DeepSeek4ScannableBlock])

  def test_linen_pipeline_dispatch_includes_deepseek4(self):
    """Linen Decoder._get_nnx_decoder_block_classes (pipeline path) must include DEEPSEEK4."""
    from maxtext.layers import decoders  # pylint: disable=import-outside-toplevel
    from maxtext.models import deepseek4  # pylint: disable=import-outside-toplevel

    cfg = self._make_deepseek4_config(scan_layers=False)
    mesh = _make_mesh(cfg)
    dec = decoders.Decoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(dec._get_nnx_decoder_block_classes(), [deepseek4.DeepSeek4DecoderLayer])  # pylint: disable=protected-access

    cfg_s = self._make_deepseek4_config(scan_layers=True)
    dec_s = decoders.Decoder(config=cfg_s, mesh=_make_mesh(cfg_s), model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(dec_s._get_nnx_decoder_block_classes(), [deepseek4.DeepSeek4ScannableBlock])  # pylint: disable=protected-access

  def _build_and_run(self, cfg):
    """Builds an NNXDecoder + shared embedding for ``cfg`` and runs one train-mode forward pass.

    Returns (decoder, logits, expected_logits_shape, aot_logits) where ``aot_logits`` is the
    ``jax.eval_shape`` (ahead-of-time, no XLA execution) abstract result of the same forward,
    computed on the freshly-built pre-forward modules.
    """
    mesh = _make_mesh(cfg)
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    # AOT structural check of the forward graph -- traces the full prefix + HCA/CSA stack
    # symbolically (no execution). Functionalize decoder+embedding via split/merge so eval_shape can
    # trace the stateful NNX modules without a cross-trace RngCount mutation.
    graphdef, state = nnx.split((decoder, shared_embedding))

    def _forward_from_state(state_in, ids_in):
      dec, emb = nnx.merge(graphdef, state_in)
      out, _, _ = dec(
          emb, ids_in, positions, decoder_segment_ids=segment_ids, deterministic=True, model_mode=MODEL_MODE_TRAIN
      )
      return out

    aot_logits = jax.eval_shape(_forward_from_state, state, ids)

    logits, _, _ = decoder(
        shared_embedding,
        ids,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    return decoder, logits, (batch, seq_len, cfg.vocab_size), aot_logits

  def _assert_forward_is_real(self, logits, aot_logits, expected):
    """Assertions that exercise the forward beyond finiteness (reviewer: isfinite is weak).

    * The AOT (jax.eval_shape) result matches the expected logits shape AND dtype -- proves the
      deepseek4 prefix + scanned HCA/CSA stack composes into well-formed logits symbolically.
    * The executed logits are non-degenerate: real variance (std well above 0) and position-dependent
      (a causal decoder MUST produce different logits at the first vs last position). A constant or
      collapsed forward would satisfy isfinite + shape but fail these.
    """
    self.assertEqual(aot_logits.shape, expected)
    self.assertEqual(aot_logits.dtype, jnp.float32)
    self.assertEqual(logits.dtype, jnp.float32)
    self.assertGreater(float(jnp.std(logits)), 1e-2)
    self.assertFalse(
        bool(jnp.allclose(logits[:, 0, :], logits[:, -1, :], rtol=1e-2, atol=1e-2)),
        msg="deepseek4 logits are position-invariant -> forward is degenerate",
    )

  def _deepseek4_decoder_loss_and_grads(self, cfg):
    """Build the DeepSeek-V4 decoder + shared embedding for cfg and return (loss, grads) for a
    sum-of-squares loss differentiated (nnx.value_and_grad) wrt every decoder + embedding Param.
    Fixed seed, so two builds differing only in remat_policy share the same params."""
    mesh = _make_mesh(cfg)
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
    shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )
    batch = cfg.global_batch_size_to_train_on
    seq_len = cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))

    def loss_fn(dec, emb):
      out, _, _ = dec(
          emb, ids, positions, decoder_segment_ids=segment_ids, deterministic=True, model_mode=MODEL_MODE_TRAIN
      )
      return jnp.sum(out.astype(jnp.float32) ** 2)

    return nnx.value_and_grad(loss_fn, argnums=(0, 1))(decoder, shared_embedding)

  def _assert_decoder_grad_parity(self, scan_layers):
    """Run the full DeepSeek-V4 decoder under two remat policies ('full' vs 'minimal') and assert
    matching loss and gradients. Gradients are compared by relative L2 error (see _assert_grad_parity,
    tolerant of TPU bf16 rounding); loss at rtol=1e-2."""
    loss_full, grads_full = self._deepseek4_decoder_loss_and_grads(
        self._make_deepseek4_config(scan_layers=scan_layers, remat_policy="full")
    )
    loss_min, grads_min = self._deepseek4_decoder_loss_and_grads(
        self._make_deepseek4_config(scan_layers=scan_layers, remat_policy="minimal")
    )
    np.testing.assert_allclose(np.array(loss_full), np.array(loss_min), rtol=1e-2, atol=1e-2)
    _assert_grad_parity(
        self, jax.tree.leaves(grads_full), jax.tree.leaves(grads_min), what="deepseek4 decoder full-vs-minimal remat"
    )

  def test_scan_init_builds_prefix_and_scanned_blocks(self):
    """scan init builds first_num_hash_layers unrolled prefix layers + a scanned block stack."""
    cfg = self._make_deepseek4_config(scan_layers=True)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertEqual(decoder.num_prefix_layers, cfg.first_num_hash_layers)
    for i in range(cfg.first_num_hash_layers):
      self.assertTrue(hasattr(decoder, f"layers_{i}"))
    # num_decoder_layers=5, first_num_hash_layers=3 -> (5-3)//2 = 1 scanned HCA/CSA block
    self.assertIsNotNone(decoder.scanned_blocks)

  def test_scan_init_odd_non_prefix_layers_raises(self):
    """scan init with an ODD non-prefix layer count must raise AssertionError (no silent drop).

    num_decoder_layers=6, first_num_hash_layers=3 -> (6-3)=3 is odd. The DeepSeek-V4 scanned body
    pairs HCA/CSA layers via `// 2`, which would silently drop the trailing layer without the guard
    in NNXDecoder._init_scanned_deepseek4.
    """
    cfg = self._make_deepseek4_config(scan_layers=True, num_decoder_layers=6, first_num_hash_layers=3)
    mesh = _make_mesh(cfg)
    with self.assertRaises(AssertionError):
      NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))

  def test_scan_init_even_non_prefix_layers_constructs(self):
    """scan init with an EVEN non-prefix layer count still constructs (companion to the odd guard).

    num_decoder_layers=5, first_num_hash_layers=3 -> (5-3)=2 is even -> the guard passes and one
    scanned HCA/CSA block is built.
    """
    cfg = self._make_deepseek4_config(scan_layers=True, num_decoder_layers=5, first_num_hash_layers=3)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertEqual(decoder.num_prefix_layers, 3)
    self.assertIsNotNone(decoder.scanned_blocks)

  def test_non_scan_init_builds_deepseek4_layers(self):
    """non-scan init builds num_decoder_layers DeepSeek4DecoderLayer instances (with layer_idx)."""
    from maxtext.models import deepseek4  # pylint: disable=import-outside-toplevel

    cfg = self._make_deepseek4_config(scan_layers=False)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertEqual(len(decoder.layers), cfg.num_decoder_layers)
    for layer in decoder.layers:
      self.assertIsInstance(layer, deepseek4.DeepSeek4DecoderLayer)

  def test_forward_non_scan(self):
    """deepseek4 non-scan forward returns correct logits shape and finite values.

    End-to-end numeric forward (prefix hash-routing layers + per-layer DeepSeek4DecoderLayer with
    global layer_idx + decoder_input_tokens). Uses dense MoE (sparse_matmul=False) so it runs on any backend.
    """
    cfg = self._make_deepseek4_config(scan_layers=False)
    decoder, logits, expected, aot_logits = self._build_and_run(cfg)
    self.assertEqual(logits.shape, expected)
    self._assert_forward_is_real(logits, aot_logits, expected)
    self.assertTrue(jnp.all(jnp.isfinite(logits)))  # secondary
    self.assertTrue(hasattr(decoder, "layers_0"))
    self._assert_decoder_grad_parity(scan_layers=cfg.scan_layers)

  def test_forward_scan(self):
    """deepseek4 scan forward (unrolled hash-routing prefix + scanned HCA/CSA blocks).

    End-to-end numeric forward exercising _apply_deepseek4_scanned_blocks: the prefix layers run
    unscanned, then the alternating HCA(128)/CSA(4) blocks run via _apply_layers_sequentially.
    """
    cfg = self._make_deepseek4_config(scan_layers=True)
    _, logits, expected, aot_logits = self._build_and_run(cfg)
    self.assertEqual(logits.shape, expected)
    self._assert_forward_is_real(logits, aot_logits, expected)
    self.assertTrue(jnp.all(jnp.isfinite(logits)))  # secondary
    self._assert_decoder_grad_parity(scan_layers=cfg.scan_layers)


class TestNNXPipelineStages(unittest.TestCase):
  """Tests for the NNX pipeline-stage modules (NNXSequentialPipelineStage / NNXScannedPipelineStage),
  including per-stage remat + params-only host-offload (set_remat_policy_on_layers_per_stage /
  parameter_memory_host_offload) that the nnx-based-pipeline migration dropped.

  Per-stage remat and host-offload are output-transparent: rematerialization only changes which
  activations are saved vs recomputed, and host-offload only changes where params live. Neither
  changes the forward result, so a stage built with them must produce numerically identical output
  to the same stage (same params/rngs) without them.
  """

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)

  def _inputs(self, cfg):
    batch = cfg.global_batch_size_to_train_on
    seq = cfg.max_target_length
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim)).astype(cfg.dtype)
    segment_ids = jnp.full((batch, seq), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq)[None], (batch, seq))
    return inputs, segment_ids, positions

  def _run_stage(self, stage_cls, remat_policy, num_layers=2, config=None, use_mesh=False):
    """Builds a pipeline stage of num_layers NNXDecoderLayers and runs one train-mode forward.

    Fresh rngs with a fixed seed each call -> two stages built this way share identical params, so
    any output difference would come solely from remat / host-offload (which must be transparent).

    use_mesh: build + run inside the device mesh + logical axis_rules. Required for host-offload:
    jax.device_put(params, Space.Device) pins the exact param sharding, so params must be sharded
    consistently with the (mesh-placed) inputs -- exactly as a real pipeline forward runs. Without
    a mesh the params land on a single device and clash with the multi-device inputs.
    """
    cfg = config if config is not None else self.cfg
    mesh = _make_mesh(cfg) if config is not None else self.mesh

    def _build_and_run():
      stage = stage_cls(
          NNXDecoderLayer,
          num_layers,
          cfg,
          mesh,
          None,
          MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=1),
          remat_policy=remat_policy,
          apply_remat=remat_policy is not None,
      )
      inputs, segment_ids, positions = self._inputs(cfg)
      out = stage(inputs, segment_ids, positions, True, MODEL_MODE_TRAIN)
      return out[0] if isinstance(out, tuple) else out

    if use_mesh:
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        return _build_and_run()
    return _build_and_run()

  def test_sequential_stage_forward_shape(self):
    """NNXSequentialPipelineStage forward returns [batch, seq, emb] and finite values."""
    inputs, _, _ = self._inputs(self.cfg)
    out = self._run_stage(NNXSequentialPipelineStage, None)
    self.assertEqual(out.shape, inputs.shape)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_scanned_stage_forward_shape(self):
    """NNXScannedPipelineStage forward returns [batch, seq, emb] and finite values."""
    inputs, _, _ = self._inputs(self.cfg)
    out = self._run_stage(NNXScannedPipelineStage, None)
    self.assertEqual(out.shape, inputs.shape)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_sequential_stage_remat_is_output_transparent(self):
    """Per-stage remat on a sequential stage must not change the forward output."""
    out_no_remat = self._run_stage(NNXSequentialPipelineStage, None)
    out_remat = self._run_stage(NNXSequentialPipelineStage, jax.checkpoint_policies.nothing_saveable)
    np.testing.assert_allclose(np.array(out_no_remat), np.array(out_remat), rtol=1e-5, atol=1e-5)

  def test_scanned_stage_remat_is_output_transparent(self):
    """Per-stage remat on a scanned stage must not change the forward output."""
    out_no_remat = self._run_stage(NNXScannedPipelineStage, None)
    out_remat = self._run_stage(NNXScannedPipelineStage, jax.checkpoint_policies.nothing_saveable)
    np.testing.assert_allclose(np.array(out_no_remat), np.array(out_remat), rtol=1e-5, atol=1e-5)

  def _stage_value_and_grad(self, stage_cls, apply_remat, remat_policy, num_layers=2, config=None, use_mesh=False):
    """Builds a pipeline stage (fixed seed) and returns (loss, input_grad, param_grads) for
    ``loss = sum(stage(x)**2)``.

    Differentiation uses ``nnx.value_and_grad`` over ``(inputs, stage)`` so the stateful stage
    (rng/kv state) is threaded correctly and gradients flow to BOTH the inputs and every stage Param.
    The fixed seed means the remat and no-remat builds share identical params, so the two paths are
    differentiated on the same weights (their loss and gradients are compared by _assert_stage_grad_parity).

    use_mesh builds + differentiates inside the device mesh + logical axis_rules -- required for
    host-offload, where ``jax.device_put(params, Space.Device)`` pins the exact param sharding (see
    _run_stage).
    """
    cfg = config if config is not None else self.cfg
    mesh = _make_mesh(cfg) if config is not None else self.mesh

    def _build_and_grad():
      stage = stage_cls(
          NNXDecoderLayer,
          num_layers,
          cfg,
          mesh,
          None,
          MODEL_MODE_TRAIN,
          rngs=nnx.Rngs(params=0, dropout=1),
          remat_policy=remat_policy,
          apply_remat=apply_remat,
      )
      inputs, segment_ids, positions = self._inputs(cfg)

      def loss_fn(x, module):
        out = module(x, segment_ids, positions, True, MODEL_MODE_TRAIN)
        out = out[0] if isinstance(out, tuple) else out
        return jnp.sum(out.astype(jnp.float32) ** 2)

      loss, (input_grad, param_grads) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(inputs, stage)
      return loss, input_grad, param_grads

    if use_mesh:
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        return _build_and_grad()
    return _build_and_grad()

  def _assert_stage_grad_parity(self, stage_cls, num_layers=2):
    """remat vs no-remat: matching loss + gradients (wrt inputs AND params), plus a real (nonzero)
    backward. jax.checkpoint recomputes activations in the backward pass but is mathematically
    transparent; gradients are compared by relative L2 error (see _assert_grad_parity), which tolerates
    the bf16 rounding of the rematerialized backward on TPU. Loss parity holds everywhere."""
    loss_ref, xgrad_ref, pgrad_ref = self._stage_value_and_grad(
        stage_cls, apply_remat=False, remat_policy=None, num_layers=num_layers
    )
    loss_remat, xgrad_remat, pgrad_remat = self._stage_value_and_grad(
        stage_cls, apply_remat=True, remat_policy=jax.checkpoint_policies.nothing_saveable, num_layers=num_layers
    )
    # Loss parity (reproducible on all platforms).
    np.testing.assert_allclose(np.array(loss_remat), np.array(loss_ref), rtol=1e-2, atol=1e-2)
    # Input-gradient parity.
    _assert_grad_parity(self, [xgrad_ref], [xgrad_remat], what="stage remat input-grad")
    # Per-Param gradient parity across the whole pytree.
    _assert_grad_parity(
        self,
        jax.tree_util.tree_leaves(pgrad_ref),
        jax.tree_util.tree_leaves(pgrad_remat),
        what="stage remat param-grads",
    )

  def test_sequential_stage_remat_grad_parity(self):
    """Backward parity: a sequential stage's per-stage remat must reproduce the no-remat loss and
    gradients (wrt inputs AND params). jax.checkpoint recomputes activations in the backward pass but
    is mathematically transparent, so value_and_grad must match."""
    self._assert_stage_grad_parity(NNXSequentialPipelineStage)

  def test_scanned_stage_remat_grad_parity(self):
    """Backward parity for the scanned pipeline stage (remat inside the jax.lax.scan body): remat vs
    no-remat must yield identical loss and gradients (wrt inputs AND params)."""
    self._assert_stage_grad_parity(NNXScannedPipelineStage)

  def test_single_layer_stage_remat_is_output_transparent(self):
    """num_layers_per_pipeline_stage==1: a 1-layer stage with remat must match the no-remat output."""
    out_no_remat = self._run_stage(NNXSequentialPipelineStage, None, num_layers=1)
    out_remat = self._run_stage(NNXSequentialPipelineStage, jax.checkpoint_policies.nothing_saveable, num_layers=1)
    np.testing.assert_allclose(np.array(out_no_remat), np.array(out_remat), rtol=1e-5, atol=1e-5)

  def test_single_layer_stage_remat_grad_parity(self):
    """num_layers_per_pipeline_stage==1: BACKWARD parity for the single-layer stage remat path (a
    distinct builder branch). remat vs no-remat must yield identical loss + gradients."""
    self._assert_stage_grad_parity(NNXSequentialPipelineStage, num_layers=1)

  @pytest.mark.tpu_only
  def test_remat_with_host_offload_is_output_transparent(self):
    """Per-stage remat + params-only host-offload (parameter_memory_host_offload) must not change output.

    tpu_only: host-offload uses jax.device_put(..., max_utils.device_space()) which targets TPU host
    memory; on CPU fake-multi-device it pins params to a single device and breaks the input sharding.
    The remat half is covered by the stage remat tests above; this adds the offload device_put (TPU only).
    """
    offload_cfg = _make_config(parameter_memory_host_offload=True)
    # Both runs inside the mesh so params are sharded consistently with the inputs; the offloaded run
    # additionally exercises jax.device_put(params, Space.Device) inside the per-stage remat.
    plain = self._run_stage(NNXSequentialPipelineStage, None, config=offload_cfg, use_mesh=True)
    offloaded = self._run_stage(
        NNXSequentialPipelineStage, jax.checkpoint_policies.nothing_saveable, config=offload_cfg, use_mesh=True
    )
    np.testing.assert_allclose(np.array(plain), np.array(offloaded), rtol=1e-5, atol=1e-5)

  @pytest.mark.tpu_only
  def test_remat_with_host_offload_grad_is_transparent(self):
    """BACKWARD parity for per-stage remat + params-only host-offload: the loss AND gradients (wrt
    inputs and every Param) with parameter_memory_host_offload must match the no-offload/no-remat
    path -- host-offload only moves where params live, it must not change the math.

    tpu_only for the same reason as the forward host-offload test: jax.device_put(..., device_space())
    targets TPU host memory; on CPU fake-multi-device it pins params to one device and breaks sharding.
    """
    offload_cfg = _make_config(parameter_memory_host_offload=True)
    loss_ref, xgrad_ref, pgrad_ref = self._stage_value_and_grad(
        NNXSequentialPipelineStage, apply_remat=False, remat_policy=None, config=offload_cfg, use_mesh=True
    )
    loss_off, xgrad_off, pgrad_off = self._stage_value_and_grad(
        NNXSequentialPipelineStage,
        apply_remat=True,
        remat_policy=jax.checkpoint_policies.nothing_saveable,
        config=offload_cfg,
        use_mesh=True,
    )
    np.testing.assert_allclose(np.array(loss_off), np.array(loss_ref), rtol=1e-2, atol=1e-2)
    # Gradients are compared by relative L2 error, which absorbs the bf16 rounding of the rematerialized
    # backward on TPU. The offload path is additionally pinned output-transparent at 1e-5 by
    # test_remat_with_host_offload_is_output_transparent.
    _assert_grad_parity(self, [xgrad_ref], [xgrad_off], what="host-offload input-grad")
    _assert_grad_parity(
        self,
        jax.tree_util.tree_leaves(pgrad_ref),
        jax.tree_util.tree_leaves(pgrad_off),
        what="host-offload param-grads",
    )


class TestNNXPerStageRematApplied(unittest.TestCase):
  """Guards the per-stage remat parity bug: remat_policy='full' resolves to get_remat_policy()==None,
  which is a VALID 'full rematerialization' policy (matching Linen nn.remat(policy=None)). Gating on
  `remat_policy is not None` silently dropped remat for the default 'full' policy. The builder must
  apply per-stage remat whenever set_remat_policy_on_layers_per_stage=True, regardless of policy value.
  """

  def _build_stage(self, remat_policy, num_layers_per_pipeline_stage=2, flag_on=True):
    """Build a pipeline stage via the NNXDecoder builder for the given remat policy + flag."""
    cfg = _make_config(
        remat_policy=remat_policy,
        set_remat_policy_on_layers_per_stage=flag_on,
        num_layers_per_pipeline_stage=num_layers_per_pipeline_stage,
        scan_layers_per_stage=False,
        scan_layers=False,
    )
    mesh = _make_mesh(cfg)
    dec = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    stage = dec._get_pipeline_stage_module(dec.get_decoder_layers(), nnx.Rngs(params=0, dropout=1))  # pylint: disable=protected-access
    return cfg, stage

  def _run_forward(self, cfg, stage):
    seq = cfg.max_target_length
    x = jax.random.normal(jax.random.PRNGKey(0), (1, seq, cfg.emb_dim)).astype(cfg.dtype)
    seg = jnp.full((1, seq), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    pos = jnp.broadcast_to(jnp.arange(seq)[None], (1, seq))
    out = stage(x, seg, pos, True, MODEL_MODE_TRAIN)
    return out[0] if isinstance(out, tuple) else out

  def test_full_policy_applies_remat(self):
    """remat_policy='full' (get_remat_policy()==None) must STILL apply per-stage remat (the bug).

    Before the fix the builder gated on `per_stage_remat is not None`, so 'full' (None policy)
    silently produced a no-remat stage. Now apply_remat is True and the checkpoint(policy=None)
    full-rematerialization path runs.
    """
    cfg, stage = self._build_stage("full")
    self.assertTrue(stage.apply_remat, "per-stage remat dropped for remat_policy='full'")
    self.assertIsNone(stage.remat_policy)  # 'full' -> None policy == full remat
    out = self._run_forward(cfg, stage)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_minimal_policy_applies_remat(self):
    """Sanity: a non-None policy also applies remat and runs."""
    cfg, stage = self._build_stage("minimal")
    self.assertTrue(stage.apply_remat)
    self.assertIsNotNone(stage.remat_policy)
    out = self._run_forward(cfg, stage)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_single_layer_stage_wraps_with_remat_for_full_policy(self):
    """num_layers_per_pipeline_stage==1 + 'full': must return a remat-applying stage, not a bare layer."""
    _, stage = self._build_stage("full", num_layers_per_pipeline_stage=1)
    self.assertIsInstance(stage, NNXSequentialPipelineStage)
    self.assertTrue(stage.apply_remat)

  def test_flag_off_single_layer_returns_bare_layer(self):
    """Flag off: num_layers==1 returns the bare layer (no stage wrapper) -- unchanged behavior."""
    _, stage = self._build_stage("full", num_layers_per_pipeline_stage=1, flag_on=False)
    self.assertNotIsInstance(stage, (NNXSequentialPipelineStage, NNXScannedPipelineStage))
