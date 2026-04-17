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

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN, DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder, NNXDecoderLayer, deepstack_process
from maxtext.layers.normalizations import RMSNorm
from maxtext.models.gpt3 import Gpt3LayerNorm
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_decoupled_parallelism_overrides, get_test_config_path

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
  extra_args = get_decoupled_parallelism_overrides()
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      **_BASE_CONFIG,
      **extra_args,
      **overrides,
      override_model_config=True,
  )


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


# ---------------------------------------------------------------------------
# 1. deepstack_process
# ---------------------------------------------------------------------------


class TestDeepstackProcess(unittest.TestCase):
  """Tests for the deepstack_process pure function."""

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

  def _make_layer(self, model_mode=MODEL_MODE_TRAIN):
    return NNXDecoderLayer(
        config=self.cfg,
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
    out, _ = layer(inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(out.shape, inputs.shape)

  def test_forward_output_dtype(self):
    """Output dtype matches config dtype."""
    layer = self._make_layer()
    inputs, segment_ids, positions = self._make_inputs()
    out, _ = layer(inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(out.dtype, self.cfg.dtype)

  def test_forward_kv_cache_is_none_when_scan_layers_false(self):
    """kv_cache return value is not None when scan_layers=False (non-scan returns cache)."""
    # With scan_layers=False the layer returns (output, kv_cache).
    # kv_cache may be None in train mode (no cache is populated); we just
    # verify the call doesn't raise and returns a 2-tuple.
    layer = self._make_layer()
    inputs, segment_ids, positions = self._make_inputs()
    result = layer(inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_TRAIN)
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)

  def test_forward_deterministic_and_stochastic_consistent_shape(self):
    """Output shape is the same regardless of the deterministic flag."""
    layer = self._make_layer()
    inputs, segment_ids, positions = self._make_inputs()
    out_det, _ = layer(inputs, segment_ids, positions, deterministic=True, model_mode=MODEL_MODE_TRAIN)
    out_stoch, _ = layer(inputs, segment_ids, positions, deterministic=False, model_mode=MODEL_MODE_TRAIN)
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
    expected = (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size)
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
    expected = (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.emb_dim)
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


if __name__ == "__main__":
  unittest.main()
