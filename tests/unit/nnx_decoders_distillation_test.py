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

"""Tests for the distillation-specific outputs and rematerialization options of NNXDecoder."""

import sys
import types
import unittest

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers.embeddings import Embed
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.trainers.post_train.distillation import distillation_utils
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "nnx_decoders_distillation_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 3,
    "attention": "dot_product",
    "max_target_length": 16,
    "max_prefill_predict_length": 4,
    "base_emb_dim": 64,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 128,
    "head_dim": 32,
    "vocab_size": 128,
    "scan_layers": False,
    "remat_policy": "full",
}


def _make_config(**overrides):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()], override_model_config=True, **{**_BASE_CONFIG, **overrides}
  )


class UnscannedDecoderDistillationTest(unittest.TestCase):
  """Covers `remat_unscanned_layers_with_scan` and block-boundary feature capture."""

  def _build(self, cfg):
    """Returns (decoder, embedding, inputs) for `cfg`."""
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
    embedding = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        mesh=mesh,
        rngs=nnx.Rngs(0),
    )
    batch, seq_len = cfg.global_batch_size_to_train_on, cfg.max_target_length
    ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
    segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
    positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
    return decoder, embedding, (ids, segment_ids, positions)

  @staticmethod
  def _forward(decoder, embedding, inputs):
    ids, segment_ids, positions = inputs
    return decoder(
        embedding, ids, positions, decoder_segment_ids=segment_ids, deterministic=True, model_mode=MODEL_MODE_TRAIN
    )

  def test_remat_with_scan_matches_default_remat(self):
    """The length-1-scan remat path is numerically identical to plain per-layer remat (loss and gradients)."""
    results = []
    for flag in (False, True):
      # float32 so that differences in XLA fusion between the two paths do not show up as bfloat16 rounding.
      cfg = _make_config(
          remat_unscanned_layers_with_scan=flag, dtype="float32", weight_dtype="float32", matmul_precision="highest"
      )
      decoder, embedding, inputs = self._build(cfg)

      # Differentiate through an explicit split/merge, as the distillation trainer does.
      graphdef, params, rest = nnx.split(decoder, nnx.Param, ...)

      def loss_fn(params_in, graphdef=graphdef, rest=rest, embedding=embedding, inputs=inputs):
        dec = nnx.merge(graphdef, params_in, rest, copy=True)
        logits = self._forward(dec, embedding, inputs)[0]
        return jnp.mean(jnp.square(logits.astype(jnp.float32)))

      results.append(jax.jit(jax.value_and_grad(loss_fn))(params))

    loss_off, grads_off = results[0]
    loss_on, grads_on = results[1]
    np.testing.assert_allclose(float(loss_on), float(loss_off), rtol=1e-6)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6), grads_on, grads_off)

  def test_block_boundary_features_unscanned(self):
    """Unscanned decoders sow the outputs of distill_layer_indices layers, in that order."""
    cfg = _make_config(distill_beta=0.1, distill_layer_indices=[2, 0])
    decoder, embedding, inputs = self._build(cfg)
    hidden_state = self._forward(decoder, embedding, inputs)[1]

    model = types.SimpleNamespace(decoder=decoder)
    features = maxtext_utils.get_intermediate_value(model, "block_boundary_activations", clear=True)
    batch, seq_len = inputs[0].shape
    self.assertEqual(features.shape, (2, batch, seq_len, cfg.emb_dim))
    # Layer 2 is the last layer, so its output is the (pre-norm) final hidden state.
    np.testing.assert_allclose(np.asarray(features[0]), np.asarray(hidden_state), rtol=1e-6)
    self.assertFalse(hasattr(decoder, "block_boundary_activations"))
    # Attention out-projection activations are only sown with scanned layers.
    intermediates = jax.tree_util.tree_leaves_with_path(nnx.state(decoder, nnx.Intermediate))
    self.assertFalse(any("out_projection_activations" in jax.tree_util.keystr(p) for p, _ in intermediates))

  def test_block_boundary_features_default_to_all_layers(self):
    cfg = _make_config(distill_beta=0.1)
    decoder, embedding, inputs = self._build(cfg)
    self._forward(decoder, embedding, inputs)
    features = maxtext_utils.get_intermediate_value(types.SimpleNamespace(decoder=decoder), "block_boundary_activations")
    self.assertEqual(features.shape[0], cfg.num_decoder_layers)

  def test_out_of_range_distill_layer_index_raises(self):
    with self.assertRaisesRegex(ValueError, "out of range"):
      _make_config(distill_beta=0.1, distill_layer_indices=[3])
    cfg = _make_config(distill_beta=0.1)
    cfg.distill_layer_indices = [3]
    decoder, embedding, inputs = self._build(cfg)
    with self.assertRaisesRegex(ValueError, "out of range"):
      self._forward(decoder, embedding, inputs)

  def test_scanned_layers_sow_out_projection_features(self):
    cfg = _make_config(distill_beta=0.1, scan_layers=True)
    decoder, embedding, inputs = self._build(cfg)
    self._forward(decoder, embedding, inputs)
    self.assertFalse(hasattr(decoder, "block_boundary_activations"))
    intermediates = jax.tree_util.tree_leaves_with_path(nnx.state(decoder, nnx.Intermediate))
    self.assertTrue(any("out_projection_activations" in jax.tree_util.keystr(p) for p, _ in intermediates))

  def test_feature_distillation_allows_unscanned_layers(self):
    cfg = _make_config(distill_beta=0.1, scan_layers=False)
    self.assertFalse(cfg.scan_layers)


class FusedLMHeadDecoderTest(unittest.TestCase):
  """Covers the `distill_fused_lm_head` output of the decoder."""

  def test_fused_head_outputs_reproduce_logits(self):
    """In training mode the decoder returns (hidden, embedding) that reproduce its logits via the LM head spec."""
    outputs = {}
    for fused in (False, True):
      cfg = _make_config(
          distill_fused_lm_head=fused, logits_via_embedding=True, final_logits_soft_cap=5.0, logits_dot_in_fp32=True
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      decoder = NNXDecoder(config=cfg, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0, dropout=1))
      embedding = Embed(
          num_embeddings=cfg.vocab_size,
          num_features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          config=cfg,
          mesh=mesh,
          rngs=nnx.Rngs(0),
      )
      batch, seq_len = cfg.global_batch_size_to_train_on, cfg.max_target_length
      ids = jax.random.randint(jax.random.PRNGKey(0), (batch, seq_len), 0, cfg.vocab_size)
      segment_ids = jnp.full((batch, seq_len), DECODING_ACTIVE_SEQUENCE_INDICATOR)
      positions = jnp.broadcast_to(jnp.arange(seq_len)[None], (batch, seq_len))
      logits = decoder(
          embedding, ids, positions, decoder_segment_ids=segment_ids, deterministic=True, model_mode=MODEL_MODE_TRAIN
      )[0]
      outputs[fused] = (logits, cfg)

    reference, _ = outputs[False]
    (hidden, table), cfg = outputs[True]
    head = distillation_utils.LMHeadSpec(
        normalize_embedding_logits=cfg.normalize_embedding_logits,
        final_logits_soft_cap=cfg.final_logits_soft_cap,
        logits_dot_in_fp32=cfg.logits_dot_in_fp32,
    )
    np.testing.assert_allclose(
        np.asarray(distillation_utils.project_to_logits(hidden, table, head), np.float32),
        np.asarray(reference, np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
  unittest.main()
