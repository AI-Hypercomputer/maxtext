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
from flax.linen import partitioning as nn_partitioning
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
from maxtext.layers import linears, mhc
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import (
    NNXDecoder,
    NNXDecoderLayer,
    NNXScannedPipelineStage,
    NNXSequentialPipelineStage,
    _make_single_layer_remat_stage_cls,
    deepstack_process,
)
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import gemma4, gemma4_small
from maxtext.models.gpt3 import Gpt3LayerNorm
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# ---------------------------------------------------------------------------
# Shared minimal config overrides used across most tests
# ---------------------------------------------------------------------------
# jax.checkpoint lowers to this primitive. Match it by IDENTITY, never by substring-searching the
# printed jaxpr: JAX renames primitives' printed names for cosmetics (pjit_p went "pjit" -> "jit" in a
# commit that also touched ad_checkpoint.py), and "remat2" is a leftover marker from the 2021 remat
# rewrite. A rename would break present-checks loudly and make absent-checks pass vacuously forever.
from jax._src.ad_checkpoint import remat_p as _REMAT_PRIMITIVE  # pylint: disable=wrong-import-position
from jax._src import core as _jax_core  # pylint: disable=wrong-import-position


def _jaxpr_contains_primitive(jaxpr, primitive):
  """True if `primitive` appears anywhere in `jaxpr`, including nested sub-jaxprs (scan/cond bodies)."""
  inner = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
  if any(eqn.primitive is primitive for eqn in inner.eqns):
    return True
  return any(_jaxpr_contains_primitive(sub, primitive) for sub in _jax_core.subjaxprs(inner))


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

  def _assert_layer_matches_submodule_composition(self, model_mode):
    """Forward pass output shape matches input shape, AND the layer composes its sub-modules exactly
    as norm -> self_attention(lnx,lnx) + mlp(lnx) -> dropout -> +residual .
    """
    layer = self._make_layer(model_mode)
    inputs, segment_ids, positions = self._make_inputs()
    out, _ = layer(
        inputs,
        segment_ids,
        positions,
        deterministic=True,
        model_mode=model_mode,
    )
    self.assertEqual(out.shape, inputs.shape)

    ref_layer = self._make_layer(model_mode)
    lnx = ref_layer.pre_self_attention_norm(inputs)
    attention_lnx, _ = ref_layer.self_attention(
        lnx,
        lnx,
        positions,
        decoder_segment_ids=segment_ids,
        deterministic=True,
        model_mode=model_mode,
        kv_cache=None,
        attention_metadata=None,
    )
    mlp_lnx = ref_layer.mlp(lnx, deterministic=True)
    combined = ref_layer.dropout(mlp_lnx + attention_lnx, deterministic=True)
    expected = combined + inputs
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-5, atol=1e-5)

  def test_forward_output_shape_train(self):
    """Forward pass in train mode matches shape and the sub-module composition (see helper docstring)."""
    self._assert_layer_matches_submodule_composition(MODEL_MODE_TRAIN)

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
    """Forward pass in prefill mode matches shape and the sub-module composition (see helper
    docstring). Prefill selects a different sharding-axis-name branch inside __call__ (values
    unaffected) and a different attention code path; the composition still must hold."""
    self._assert_layer_matches_submodule_composition(MODEL_MODE_PREFILL)

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

  def test_logits_depend_on_input_tokens(self):
    """The decoder's output must actually depend on the token ids it was given."""
    _, segment_ids, positions = self._make_token_inputs()
    cfg = self.cfg
    batch, seq_len = cfg.global_batch_size_to_train_on, cfg.max_target_length
    key_a, _ = jax.random.split(jax.random.PRNGKey(1234))
    ids_a = jax.random.randint(key_a, (batch, seq_len), 0, cfg.vocab_size)
    ids_b = (ids_a + 1) % cfg.vocab_size  # guaranteed different at every position

    def run(ids):
      logits, _, _ = self.decoder(
          self.shared_embedding,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return np.array(logits)

    self.assertFalse(
        bool(jnp.allclose(run(ids_a), run(ids_b), rtol=1e-4, atol=1e-4)),
        "logits are identical for two different token sequences -> the decoder is ignoring its input "
        "tokens (e.g. the embedding lookup is not using decoder_input_tokens)",
    )

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
    """hidden_state shape: [batch, seq_len, emb_dim], AND it must actually be the decoder LAYER
    STACK's output, not the raw embeddings passed through untouched.
    """
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

    embeds = self.decoder._apply_embedding(  # pylint: disable=protected-access
        self.shared_embedding, ids, positions, True, MODEL_MODE_TRAIN
    )
    self.assertFalse(
        bool(jnp.allclose(hidden_state, embeds, rtol=1e-3, atol=1e-3)),
        msg="hidden_state equals the raw embeddings -> the decoder layer stack was not applied",
    )

  def test_logits_are_finite(self):
    """Logits must not contain NaN or Inf, AND be non-degenerate."""
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
    self.assertGreater(float(jnp.std(logits)), 1e-2)
    self.assertFalse(
        bool(jnp.allclose(logits[:, 0, :], logits[:, -1, :], rtol=1e-2, atol=1e-2)),
        msg="logits are position-invariant -> forward is degenerate",
    )

  def test_multimodal_input_forwarded_to_apply_embedding(self):
    """`multimodal_input` must reach `_apply_embedding` as the original struct."""
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

  def test_scan_forward_uses_combined_scan_axis_helper_regression(self):
    """Every scan_layers=True pure-NNX forward must run."""
    cfg = _make_config(scan_layers=True)
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = NNXDecoder(config=cfg, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
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

    call_kwargs = {
        "decoder_segment_ids": segment_ids,
        "deterministic": True,
        "model_mode": MODEL_MODE_TRAIN,
    }
    # Force the dynamic_graph_init rebuild path so nnx.Param leaves (with real, non-default
    # param_scan_axis metadata) actually flow through nnx_add_and_sync_scan_axis -- see docstring.
    decoder.disable_quant_stats_update = True
    logits1, _, _ = decoder(shared_embedding, ids, positions, **call_kwargs)
    logits2, _, _ = decoder(shared_embedding, ids, positions, **call_kwargs)

    self.assertEqual(logits1.shape, (batch, seq_len, cfg.vocab_size))
    self.assertTrue(jnp.all(jnp.isfinite(logits1)))
    # The scanned params must survive the post-scan axis restoration + write-back round trip: a
    # second forward re-reads the persisted params, so a mis-restored scan axis would crash or drift.
    np.testing.assert_allclose(np.array(logits1), np.array(logits2), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()


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

    def _forward():
      out, _, _ = decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return out

    logits = _forward()
    self.assertEqual(
        logits.shape,
        (cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.vocab_size),
    )

    _, remainder_params, remainder_rest = nnx.split(decoder.layers_remainder, nnx.Param, ...)
    self.assertGreater(len(jax.tree_util.tree_leaves(remainder_params)), 0, "layers_remainder has no params to perturb")
    perturbed_params = jax.tree.map(lambda x: x + 10.0, remainder_params)
    nnx.update(decoder.layers_remainder, nnx.State.merge(perturbed_params, remainder_rest))
    logits_perturbed = _forward()

    self.assertFalse(
        bool(jnp.allclose(logits, logits_perturbed)),
        msg="gemma4 remainder-only forward is invariant to layers_remainder params -> remainder block not applied",
    )

  def test_gemma4_block_external_kv_cache_matches_scanned_path(self):
    """External-kv-cache path must numerically match the scanned (kv=None) path."""
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
            "per_device_batch_size=1.0",
            "max_target_length=16",
            "max_prefill_predict_length=4",
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


def _assert_grad_parity(
    test_case, ref_leaves, other_leaves, *, what, rtol=2e-2, per_leaf_rtol=5e-2, per_leaf_atol_frac=1e-3
):
  """Assert two gradient leaf-lists agree, by BOTH an aggregate and a per-leaf relative bound."""
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
  total_norm = float(jnp.linalg.norm(ref))
  for i, (g_ref, g_other) in enumerate(zip(ref_leaves, other_leaves)):
    r = g_ref.astype(jnp.float32)
    leaf_norm = float(jnp.linalg.norm(r))
    leaf_err = float(jnp.linalg.norm(g_other.astype(jnp.float32) - r))
    allowed = per_leaf_rtol * leaf_norm + per_leaf_atol_frac * max(total_norm, 1e-12)
    test_case.assertLessEqual(
        leaf_err,
        allowed,
        f"{what}: leaf {i} (shape {tuple(g_ref.shape)}, {leaf_norm / max(total_norm, 1e-12):.2%} of total "
        f"gradient norm) has error {leaf_err:.3e}, exceeding the allowed "
        f"{per_leaf_rtol:.0%}*leaf + {per_leaf_atol_frac:.0e}*total = {allowed:.3e}. "
        "The aggregate check can miss this when the leaf is small.",
    )


class TestNNXDecoderDeepseek4(unittest.TestCase):
  """Parity tests for DeepSeek-V4 (deepseek4) decoder-level handling in NNXDecoder."""

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

  def test_scan_construction_registers_prefix_via_existing_helper_regression(self):
    """Test that NNXDecoder(deepseek4, scan_layers=True) registers the prefix layers via the existing helper"""
    cfg = self._make_deepseek4_config(scan_layers=True)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertEqual(decoder.num_prefix_layers, cfg.first_num_hash_layers)
    for i in range(cfg.first_num_hash_layers):
      self.assertTrue(hasattr(decoder, f"layers_{i}"), f"prefix layer layers_{i} was not registered")

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
    """Builds an NNXDecoder + shared embedding for ``cfg`` and runs one train-mode forward pass."""
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
    """Assertions that exercise the forward beyond finiteness (reviewer: isfinite is weak)."""
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
    """scan init with an ODD non-prefix layer count must raise AssertionError (no silent drop)."""
    cfg = self._make_deepseek4_config(scan_layers=True, num_decoder_layers=6, first_num_hash_layers=3)
    mesh = _make_mesh(cfg)
    with self.assertRaises(AssertionError):
      NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))

  def test_scan_init_even_non_prefix_layers_constructs(self):
    """scan init with an EVEN non-prefix layer count still constructs (companion to the odd guard)."""
    cfg = self._make_deepseek4_config(scan_layers=True, num_decoder_layers=5, first_num_hash_layers=3)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertEqual(decoder.num_prefix_layers, 3)
    self.assertIsNotNone(decoder.scanned_blocks)

  def test_non_scan_init_builds_deepseek4_layers(self):
    """non-scan init builds num_decoder_layers DeepSeek4DecoderLayer instances, each with the
    correct GLOBAL layer_idx (0..num_decoder_layers-1) baked in at construction.
    """
    from maxtext.models import deepseek4  # pylint: disable=import-outside-toplevel

    cfg = self._make_deepseek4_config(scan_layers=False)
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    for i in range(cfg.num_decoder_layers):
      layer = getattr(decoder, f"layers_{i}")
      self.assertIsInstance(layer, deepseek4.DeepSeek4DecoderLayer)
      self.assertEqual(layer.layer_idx, i, f"layers_{i} has the wrong global layer_idx")
      self.assertEqual(
          layer.self_attention.compress_ratio,
          cfg.compress_ratios[i],
          f"layers_{i}.self_attention.compress_ratio does not match config.compress_ratios[{i}] -- "
          "layer_idx was not correctly routed to this layer's construction",
      )

  def test_forward_non_scan(self):
    """deepseek4 non-scan forward returns correct logits shape and finite values."""
    cfg = self._make_deepseek4_config(scan_layers=False)
    decoder, logits, expected, aot_logits = self._build_and_run(cfg)
    self.assertEqual(logits.shape, expected)
    self._assert_forward_is_real(logits, aot_logits, expected)
    self.assertTrue(jnp.all(jnp.isfinite(logits)))  # secondary
    self.assertTrue(hasattr(decoder, "layers_0"))
    # _assert_decoder_grad_parity below compares remat_policy='full' vs 'minimal' on the SAME
    # construction, so it is blind to any construction bug shared by both (e.g. every layer
    # silently getting layer_idx=-1 -- see test_non_scan_init_builds_deepseek4_layers). Anchor this
    # forward test to real per-layer construction: layers_0 must route to compress_ratios[0].
    self.assertEqual(decoder.layers_0.self_attention.compress_ratio, cfg.compress_ratios[0])
    self._assert_decoder_grad_parity(scan_layers=cfg.scan_layers)

  def test_forward_scan(self):
    """deepseek4 scan forward (unrolled hash-routing prefix + scanned HCA/CSA blocks)."""
    cfg = self._make_deepseek4_config(scan_layers=True)
    _, logits, expected, aot_logits = self._build_and_run(cfg)
    self.assertEqual(logits.shape, expected)
    self._assert_forward_is_real(logits, aot_logits, expected)
    self.assertTrue(jnp.all(jnp.isfinite(logits)))  # secondary
    self._assert_decoder_grad_parity(scan_layers=cfg.scan_layers)

  def test_deepseek4_builds_learned_hc_head_collapse(self):
    """DeepSeek-V4 must build a LEARNED hyper-head to collapse its mhc_expansion_rate mHC streams."""
    cfg = self._make_deepseek4_config(scan_layers=False)
    self.assertGreater(cfg.mhc_expansion_rate, 1)  # deepseek4-284b -> mhc_expansion_rate=4
    mesh = _make_mesh(cfg)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    self.assertTrue(hasattr(decoder, "hc_head"), "deepseek4 decoder is missing the learned hc_head collapse module")
    self.assertIsInstance(decoder.hc_head, mhc.DeepSeek4HyperHead)
    for name in ("hc_fn", "hc_base", "hc_scale"):
      self.assertIsInstance(getattr(decoder.hc_head, name), nnx.Param, f"hc_head.{name} must be a materialized nnx.Param")

  def test_deepseek4_collapse_is_wired_to_hc_head(self):
    """The final mHC-stream collapse must FLOW THROUGH ``self.hc_head``, not the unweighted mhc_reduce."""
    cfg = self._make_deepseek4_config(scan_layers=False)
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

    def _forward():
      out, _, _ = decoder(
          shared_embedding,
          ids,
          positions,
          decoder_segment_ids=segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return out

    hidden_before = _forward()
    # Perturb ONLY the learned hyper-head; nothing else in the graph changes.
    decoder.hc_head.hc_scale.value += 10.0
    decoder.hc_head.hc_fn.value *= 5.0
    hidden_after = _forward()

    self.assertFalse(
        bool(jnp.allclose(hidden_before, hidden_after)),
        msg="forward output is invariant to hc_head params -> collapse is NOT wired through hc_head",
    )


class TestNNXPipelineStages(unittest.TestCase):
  """Tests for the NNX pipeline-stage modules (NNXSequentialPipelineStage / NNXScannedPipelineStage),
  including per-stage remat + params-only host-offload (set_remat_policy_on_layers_per_stage /
  parameter_memory_host_offload) that the nnx-based-pipeline migration dropped.
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
    """Builds a pipeline stage of num_layers NNXDecoderLayers and runs one train-mode forward."""
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

  def _assert_stage_forward_is_real(self, out, inputs):
    """Validate that a pipeline stage forward returns finite values and is not a degenerate passthrough."""
    self.assertTrue(jnp.all(jnp.isfinite(out)))
    self.assertGreater(float(jnp.std(out)), 1e-2)
    self.assertFalse(
        bool(jnp.allclose(out, inputs, rtol=1e-3, atol=1e-3)),
        msg="stage output equals the raw input -> layers were not applied (stage is a passthrough)",
    )

  def test_sequential_stage_forward_shape(self):
    """NNXSequentialPipelineStage forward returns [batch, seq, emb] and finite values."""
    inputs, _, _ = self._inputs(self.cfg)
    out = self._run_stage(NNXSequentialPipelineStage, None)
    self.assertEqual(out.shape, inputs.shape)
    self._assert_stage_forward_is_real(out, inputs)

  def test_scanned_stage_forward_shape(self):
    """NNXScannedPipelineStage forward returns [batch, seq, emb] and finite values."""
    inputs, _, _ = self._inputs(self.cfg)
    out = self._run_stage(NNXScannedPipelineStage, None)
    self.assertEqual(out.shape, inputs.shape)
    self._assert_stage_forward_is_real(out, inputs)

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

  def test_scanned_stage_remat_does_not_stack_params(self):
    """The scanned pipeline stage must NOT stack read-only params across its layers inside jax.lax.scan."""
    real_scan = jax.lax.scan
    captured = {}

    def spy_scan(*args, **kwargs):
      carry, ys = real_scan(*args, **kwargs)
      # The pipeline-stage scan is the only one whose stacked output is an nnx.State.
      if isinstance(ys, nnx.State):
        captured["ys"] = ys
      return carry, ys

    for policy in (None, jax.checkpoint_policies.nothing_saveable):
      captured.clear()
      with patch("jax.lax.scan", spy_scan):
        self._run_stage(NNXScannedPipelineStage, policy)
      self.assertIn("ys", captured, f"jax.lax.scan produced no nnx.State ys for policy={policy!r}")
      stacked_params, _ = captured["ys"].split(nnx.Param, ...)
      param_leaves = jax.tree_util.tree_leaves(stacked_params)
      self.assertEqual(
          len(param_leaves),
          0,
          f"scanned stage stacked {len(param_leaves)} nnx.Param leaf/leaves across layers for "
          f"policy={policy!r} (apply_remat={policy is not None}); the scan body must drop read-only "
          f"params from its returned state to avoid a [num_layers, *param] transient allocation.",
      )

  def _stage_value_and_grad(self, stage_cls, apply_remat, remat_policy, num_layers=2, config=None, use_mesh=False):
    """Builds a pipeline stage (fixed seed) and returns (loss, input_grad, param_grads) for
    ``loss = sum(stage(x)**2)``.

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
    """Per-stage remat + params-only host-offload (parameter_memory_host_offload) must not change output."""
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
    """remat_policy='full' (get_remat_policy()==None) must STILL apply per-stage remat (the bug)."""
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

  @staticmethod
  def _param_key_paths(module):
    """Sorted '/'-joined nnx.Param key paths (the leaves the pipeline stacks via nnx.split)."""
    _, params, _ = nnx.split(module, nnx.Param, ...)
    return sorted(
        "/".join(str(getattr(k, "key", k)) for k in path) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]
    )

  def test_single_layer_stage_keeps_params_top_level_linen_parity(self):
    """num_layers_per_pipeline_stage==1 + set_remat_policy_on_layers_per_stage: the remat-applying
    stage must keep params TOP-LEVEL -- identical nnx.Param key paths to the flag-off bare layer,
    NO 'layers_0' nesting -- matching Linen nn.remat (param-tree transparent). It must be a subclass
    of the base decoder layer (IS-A layer), not a NNXSequentialPipelineStage wrapper. Inverts the
    prior test that accepted the layers_0-nesting wrapper; also pins remat output-/grad-transparency
    vs the bare layer (same seed => identical params)."""
    cfg, stage_on = self._build_stage("full", num_layers_per_pipeline_stage=1, flag_on=True)
    _, bare_off = self._build_stage("full", num_layers_per_pipeline_stage=1, flag_on=False)

    # Per-stage remat still applied ('full' policy -> None == full rematerialization).
    self.assertTrue(stage_on.apply_remat)
    self.assertIsNone(stage_on.remat_policy)

    # IS-A base layer, NOT the wrapper -> no layers_0 nesting. base_stage_cls resolves to the
    # concrete model decoder layer (LlamaDecoderLayer for base.yml), which is exactly the class of
    # the flag-off bare layer; the remat stage must subclass it (Linen nn.remat is a same-class wrap).
    self.assertIsInstance(stage_on, type(bare_off))
    self.assertNotIsInstance(stage_on, (NNXSequentialPipelineStage, NNXScannedPipelineStage))

    # Core parity: identical param key paths, none under layers_0.
    on_paths = self._param_key_paths(stage_on)
    off_paths = self._param_key_paths(bare_off)
    self.assertEqual(on_paths, off_paths)
    self.assertFalse(
        any("layers_0" in p for p in on_paths),
        f"flag-on single-layer stage nested params under layers_0: {on_paths}",
    )

    # Output transparency: remat must not change the forward result vs the bare layer (same params).
    out_on = self._run_forward(cfg, stage_on)
    out_off = self._run_forward(cfg, bare_off)
    np.testing.assert_allclose(np.array(out_on), np.array(out_off), rtol=1e-5, atol=1e-5)

    # Grad transparency: gradients wrt every Param match the bare (no-remat) layer.
    seq = cfg.max_target_length
    x = jax.random.normal(jax.random.PRNGKey(0), (1, seq, cfg.emb_dim)).astype(cfg.dtype)
    seg = jnp.full((1, seq), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    pos = jnp.broadcast_to(jnp.arange(seq)[None], (1, seq))

    def _param_grads(module):
      def loss_fn(m):
        out = m(x, seg, pos, True, MODEL_MODE_TRAIN)
        out = out[0] if isinstance(out, tuple) else out
        return jnp.sum(out.astype(jnp.float32) ** 2)

      return jax.tree_util.tree_leaves(nnx.grad(loss_fn)(module))

    _assert_grad_parity(
        self,
        _param_grads(bare_off),
        _param_grads(stage_on),
        what="single-layer top-level remat param-grads",
    )

  def test_flag_off_single_layer_returns_bare_layer(self):
    """Flag off: num_layers==1 returns the bare layer (no stage wrapper) -- unchanged behavior."""
    _, stage = self._build_stage("full", num_layers_per_pipeline_stage=1, flag_on=False)
    self.assertNotIsInstance(stage, (NNXSequentialPipelineStage, NNXScannedPipelineStage))


class TestNNXStageRematAppliedInJaxpr(unittest.TestCase):
  """Guards against a silently no-op jax.checkpoint wrap in the shared helper
  ``_run_stage_layer_with_remat`` (used by NNXSequentialPipelineStage, NNXScannedPipelineStage, and
  the single-layer remat stage from ``_make_single_layer_remat_stage_cls``).

  """

  def setUp(self):
    super().setUp()
    self.cfg = _make_config()
    self.mesh = _make_mesh(self.cfg)

  def _inputs(self):
    cfg = self.cfg
    batch = cfg.global_batch_size_to_train_on
    seq = cfg.max_target_length
    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim)).astype(cfg.dtype)
    segment_ids = jnp.full((batch, seq), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq)[None], (batch, seq))
    return inputs, segment_ids, positions

  def _stage_fwd_jaxpr_text(self, stage):
    """``str(jax.make_jaxpr(...))`` of one stage forward pass, via nnx.split/merge -> pure fn (the
    same split/merge shape _run_stage_layer_with_remat itself uses internally, so make_jaxpr traces a
    real, representative call)."""
    inputs, segment_ids, positions = self._inputs()
    graphdef, state = nnx.split(stage)

    def fwd(state_in, x_in):
      merged = nnx.merge(graphdef, state_in)
      out = merged(x_in, segment_ids, positions, True, MODEL_MODE_TRAIN)
      return out[0] if isinstance(out, tuple) else out

    return jax.make_jaxpr(fwd)(state, inputs)

  def _assert_remat_applied(self, stage, apply_remat):
    """Asserts that the stage's forward jaxpr contains a remat primitive iff apply_remat is True."""
    has_remat = _jaxpr_contains_primitive(self._stage_fwd_jaxpr_text(stage), _REMAT_PRIMITIVE)
    if apply_remat:
      self.assertTrue(
          has_remat,
          f"{type(stage).__name__} forward jaxpr has no remat primitive with apply_remat=True -- the "
          "jax.checkpoint(pure_fn, ...) wrap inside _run_stage_layer_with_remat appears to have been "
          "bypassed (e.g. replaced by a direct pure_fn(params, state, inputs) call).",
      )
    else:
      self.assertFalse(
          has_remat,
          f"{type(stage).__name__} forward jaxpr unexpectedly contains a remat primitive with " "apply_remat=False.",
      )

  def test_sequential_stage_remat_applied_in_jaxpr(self):
    """NNXSequentialPipelineStage: a `remat2` primitive is present iff apply_remat=True."""
    stage_on = NNXSequentialPipelineStage(
        NNXDecoderLayer,
        2,
        self.cfg,
        self.mesh,
        None,
        MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
        remat_policy=jax.checkpoint_policies.nothing_saveable,
        apply_remat=True,
    )
    self._assert_remat_applied(stage_on, apply_remat=True)

    stage_off = NNXSequentialPipelineStage(
        NNXDecoderLayer,
        2,
        self.cfg,
        self.mesh,
        None,
        MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
        remat_policy=None,
        apply_remat=False,
    )
    self._assert_remat_applied(stage_off, apply_remat=False)

  def test_scanned_stage_remat_applied_in_jaxpr(self):
    """NNXScannedPipelineStage: a `remat2` primitive is present iff apply_remat=True. The checkpoint
    wraps the per-layer scan body, but jax's jaxpr pretty-printer recurses into a scan's nested jaxpr,
    so the primitive still shows up in the top-level jaxpr text (same reasoning the pipeline-level
    TestNNXCircularRepeatRemat test in nnx_pipeline_test.py relies on for its own scan-nested remat)."""
    stage_on = NNXScannedPipelineStage(
        NNXDecoderLayer,
        2,
        self.cfg,
        self.mesh,
        None,
        MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
        remat_policy=jax.checkpoint_policies.nothing_saveable,
        apply_remat=True,
    )
    self._assert_remat_applied(stage_on, apply_remat=True)

    stage_off = NNXScannedPipelineStage(
        NNXDecoderLayer,
        2,
        self.cfg,
        self.mesh,
        None,
        MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
        remat_policy=None,
        apply_remat=False,
    )
    self._assert_remat_applied(stage_off, apply_remat=False)

  def test_single_layer_stage_remat_applied_in_jaxpr(self):
    """Single-layer remat stage (_make_single_layer_remat_stage_cls, the
    num_layers_per_pipeline_stage==1 builder path): a `remat2` primitive is present iff
    apply_remat=True."""
    stage_cls = _make_single_layer_remat_stage_cls(LlamaDecoderLayer)
    stage_on = stage_cls(
        config=self.cfg,
        mesh=self.mesh,
        quant=None,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
        remat_policy=jax.checkpoint_policies.nothing_saveable,
        apply_remat=True,
    )
    self._assert_remat_applied(stage_on, apply_remat=True)

    stage_off = stage_cls(
        config=self.cfg,
        mesh=self.mesh,
        quant=None,
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0, dropout=1),
        remat_policy=None,
        apply_remat=False,
    )
    self._assert_remat_applied(stage_off, apply_remat=False)
