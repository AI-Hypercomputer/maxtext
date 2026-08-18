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

"""Tests for the DeepSeek-V4 compressed-attention mask flags."""

import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext import pyconfig
from maxtext.common.common_types import DEFAULT_MASK_VALUE
from maxtext.layers import initializers
from maxtext.layers.attention_compressed import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
    build_compressed_segment_mask,
    build_csa_indexer_causal,
    build_deepseek4_hoisted_masks,
    build_hca_causal_mask,
    combine_compressed_and_segment_mask,
    topk_threshold_membership,
)
from maxtext.layers.attention_op import AttentionOp, build_compressed_splash_mask, build_local_sliding_splash_mask
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.models.deepseek4 import DeepSeek4DecoderLayer, DeepSeek4ScannableBlock


def make_config(**overrides):
  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_target_length": 128,
      "base_emb_dim": 64,
      "head_dim": 64,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 1,
      "dtype": "float32",
      "weight_dtype": "float32",
      "q_lora_rank": 16,
      "indexer_n_heads": 2,
      "indexer_head_dim": 64,
      "indexer_topk": 8,
      "sliding_window_size": 8,
      "compress_ratios": [0, 0, 4, 128],
  }
  config_arguments.update(overrides)
  argv = [sys.argv[0], "src/maxtext/configs/base.yml"]
  return pyconfig.initialize(argv, **config_arguments)


def make_compressor(config, cls, compress_rate, seed=0):
  rotary = DeepSeekV4RotaryEmbedding(
      head_dim=config.head_dim,
      partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
      rope_theta=config.compressed_rope_max_timescale,
      fprop_dtype=config.dtype,
  )
  return cls(
      config=config,
      compress_ratio=compress_rate,
      rotary_embedding=rotary,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs=nnx.Rngs(seed),
  )


def golden_membership(index_scores, k, future_mask):
  """The pre-change op sequence: top_k indices -> invalidation -> one-hot -> any."""
  top_k_indices = jax.lax.top_k(index_scores, k)[1]
  invalid = jnp.take_along_axis(future_mask, top_k_indices, axis=-1)
  top_k_indices = jnp.where(invalid, jnp.full_like(top_k_indices, -1), top_k_indices)
  valid = top_k_indices >= 0
  entry_indices = jnp.arange(index_scores.shape[-1])[None, None, :]
  is_in_topk = jnp.expand_dims(top_k_indices, axis=-1) == entry_indices[None, ...]
  return jnp.any(is_in_topk & jnp.expand_dims(valid, axis=-1), axis=2)


class ThresholdMembershipTest(unittest.TestCase):
  """Checks threshold membership against the index-based top-k path."""

  BATCH, SEQ, WINDOWS, K = 2, 16, 16, 5

  def _future_mask(self):
    positions = jnp.broadcast_to(jnp.arange(self.SEQ)[None, :], (self.BATCH, self.SEQ))
    return build_csa_indexer_causal(positions, self.WINDOWS, 1)

  def _check(self, scores):
    future_mask = self._future_mask()
    masked = jnp.where(future_mask, jnp.full_like(scores, -jnp.inf), scores)
    got = jnp.logical_and(topk_threshold_membership(masked, self.K), jnp.logical_not(future_mask))
    want = golden_membership(masked, self.K, future_mask)
    np.testing.assert_array_equal(np.array(got), np.array(want))

  def test_continuous_scores(self):
    rng = np.random.default_rng(0)
    self._check(jnp.array(rng.normal(size=(self.BATCH, self.SEQ, self.WINDOWS)), dtype=jnp.float32))

  def test_quantized_scores_force_ties(self):
    rng = np.random.default_rng(1)
    q = np.round(rng.normal(size=(self.BATCH, self.SEQ, self.WINDOWS)) * 2) / 2
    self._check(jnp.array(q, dtype=jnp.float32))

  def test_signed_zeros_at_boundary(self):
    rng = np.random.default_rng(2)
    z = np.where(rng.random((self.BATCH, self.SEQ, self.WINDOWS)) < 0.5, 0.0, -0.0)
    self._check(jnp.array(z, dtype=jnp.float32))

  def test_all_equal_rows(self):
    self._check(jnp.ones((self.BATCH, self.SEQ, self.WINDOWS), dtype=jnp.float32))

  def test_k_zero_and_k_full(self):
    scores = jnp.ones((self.BATCH, self.SEQ, self.WINDOWS), dtype=jnp.float32)
    self.assertFalse(bool(topk_threshold_membership(scores, 0).any()))
    self.assertTrue(bool(topk_threshold_membership(scores, self.WINDOWS).all()))

  def test_topk_tiebreak_semantics(self):
    """Canary for the jax.lax.top_k semantics the threshold path depends on."""
    row = jnp.array([[1.0, 2.0, 2.0, 0.0, 2.0]])
    np.testing.assert_array_equal(np.array(jax.lax.top_k(row, 3)[1]), [[1, 2, 4]])
    zeros = jnp.array([[0.0, -0.0, 0.0, -0.0]])
    np.testing.assert_array_equal(np.array(jax.lax.top_k(zeros, 2)[1]), [[0, 2]])


class IndexerMembershipModuleTest(unittest.TestCase):
  """Checks indexer membership through the CSA compressor."""

  BATCH, SEQ, RATE = 2, 64, 4

  def _run(self, packed):
    cfg_off = make_config()
    cfg_on = make_config(indexer_threshold_membership=True)
    comp_off = make_compressor(cfg_off, DeepseekV4CSACompressor, self.RATE)
    comp_on = make_compressor(cfg_on, DeepseekV4CSACompressor, self.RATE)

    rng = np.random.default_rng(0)
    hidden = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg_off.emb_dim)), dtype=jnp.float32)
    q_latent = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg_off.q_lora_rank)), dtype=jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(self.SEQ)[None, :], (self.BATCH, self.SEQ))
    attention_mask = None
    if packed:
      ids = jnp.array(np.repeat([[1, 2]], self.SEQ // 2, axis=1).reshape(1, self.SEQ).repeat(self.BATCH, 0))
      attention_mask = build_compressed_segment_mask(ids, self.RATE)

    kv_off, mask_off = comp_off(hidden, q_latent, positions, attention_mask)
    kv_on, mask_on = comp_on(hidden, q_latent, positions, attention_mask)
    kv_bool, mask_bool = comp_on(hidden, q_latent, positions, attention_mask, emit_bool_mask=True)
    np.testing.assert_array_equal(np.array(kv_off), np.array(kv_on))
    np.testing.assert_array_equal(np.array(kv_on), np.array(kv_bool))
    np.testing.assert_array_equal(np.array(mask_off), np.array(mask_on))
    np.testing.assert_array_equal(np.array(mask_on == 0.0), np.array(mask_bool))

  def test_unpacked(self):
    self._run(packed=False)

  def test_packed(self):
    self._run(packed=True)


class HoistedMasksTest(unittest.TestCase):
  """Checks hoisted masks against the per-layer builders."""

  BATCH, SEQ = 2, 64

  def _inputs(self):
    positions = jnp.broadcast_to(jnp.arange(self.SEQ)[None, :], (self.BATCH, self.SEQ))
    ids = jnp.array(np.repeat([[1, 2]], self.SEQ // 2, axis=1).reshape(1, self.SEQ).repeat(self.BATCH, 0))
    return positions, ids

  def test_bundle_matches_per_layer_builders(self):
    cfg = make_config(hoist_static_attention_masks=True, compress_ratios=[0, 0, 4, 8])
    positions, ids = self._inputs()
    hoisted = build_deepseek4_hoisted_masks(cfg, None, ids, positions, "train")

    for rate in (4, 8):
      np.testing.assert_array_equal(
          np.array(hoisted[f"segment_mask_{rate}"]), np.array(build_compressed_segment_mask(ids, rate))
      )
    np.testing.assert_array_equal(
        np.array(hoisted["csa_future_mask_4"]), np.array(build_csa_indexer_causal(positions, self.SEQ // 4, 4))
    )
    hca = build_hca_causal_mask(positions, self.SEQ // 8, 8, cfg.dtype)
    hca = combine_compressed_and_segment_mask(hca, build_compressed_segment_mask(ids, 8))
    np.testing.assert_array_equal(np.array(hoisted["hca_mask_8"]), np.array(hca))

  def _check_model_forwarding(self, hoisted):
    inputs = jnp.zeros((self.BATCH, self.SEQ, 4))
    positions, ids = self._inputs()

    decoder = mock.Mock()
    decoder.with_logical_constraint.side_effect = lambda x: x
    decoder.self_attention_with_norm_op.return_value = (inputs, inputs)
    decoder.mhc_mlp.return_value = (inputs, {})
    decoder.dropout_op.side_effect = lambda x, deterministic: x
    decoder.post_process.return_value = (inputs, None)
    DeepSeek4DecoderLayer.__call__(decoder, inputs, ids, positions, True, "train", hoisted_masks=hoisted)
    self.assertIs(decoder.self_attention_with_norm_op.call_args.kwargs["hoisted_masks"], hoisted)

    block = mock.Mock()
    block.layers_0.side_effect = lambda x, *args, **kwargs: (x, None)
    block.layers_1.side_effect = lambda x, *args, **kwargs: (x, None)
    DeepSeek4ScannableBlock.__call__(block, inputs, ids, positions, True, "train", hoisted_masks=hoisted)
    self.assertIs(block.layers_0.call_args.kwargs["hoisted_masks"], hoisted)
    self.assertIs(block.layers_1.call_args.kwargs["hoisted_masks"], hoisted)

  def test_gating_and_model_forwarding(self):
    positions, ids = self._inputs()
    self.assertIsNone(build_deepseek4_hoisted_masks(make_config(), None, ids, positions, "train"))
    cfg_on = make_config(hoist_static_attention_masks=True)
    self.assertIsNone(build_deepseek4_hoisted_masks(cfg_on, None, ids, positions, "autoregressive"))
    self._check_model_forwarding({"csa_future_mask_4": jnp.zeros((self.BATCH, self.SEQ, self.SEQ // 4))})

  def test_hca_compressor_with_hoisted_mask(self):
    cfg = make_config()
    comp = make_compressor(cfg, DeepseekV4HCACompressor, 8)
    rng = np.random.default_rng(0)
    hidden = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg.emb_dim)), dtype=jnp.float32)
    positions, _ = self._inputs()

    kv_default, mask_default = comp(hidden, None, positions, "train")
    kv_bool, mask_bool = comp(hidden, None, positions, "train", emit_bool_mask=True)
    kv_skip, mask_skip = comp(hidden, None, positions, "train", skip_mask=True)
    self.assertIsNone(mask_skip)
    np.testing.assert_array_equal(np.array(kv_default), np.array(kv_bool))
    np.testing.assert_array_equal(np.array(kv_default), np.array(kv_skip))
    np.testing.assert_array_equal(np.array(mask_default == 0.0), np.array(mask_bool))
    hoisted_mask = build_hca_causal_mask(positions, self.SEQ // 8, 8, cfg.dtype)
    np.testing.assert_array_equal(np.array(mask_default), np.array(hoisted_mask))

  def test_indexer_with_hoisted_future_mask(self):
    cfg = make_config()
    comp = make_compressor(cfg, DeepseekV4CSACompressor, 4)
    rng = np.random.default_rng(0)
    hidden = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg.emb_dim)), dtype=jnp.float32)
    q_latent = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg.q_lora_rank)), dtype=jnp.float32)
    positions, _ = self._inputs()
    hoisted = {"csa_future_mask_4": build_csa_indexer_causal(positions, self.SEQ // 4, 4)}

    sel_default = comp.indexer(hidden, q_latent, positions, None)
    sel_hoisted = comp.indexer(hidden, q_latent, positions, None, hoisted_masks=hoisted)
    np.testing.assert_array_equal(np.array(sel_default), np.array(sel_hoisted))


class BoolMaskTest(unittest.TestCase):
  """Checks boolean compressed masks against additive masks."""

  BATCH, SEQ = 2, 64

  def test_builders_agree(self):
    positions = jnp.broadcast_to(jnp.arange(self.SEQ)[None, :], (self.BATCH, self.SEQ))
    ids = jnp.array(np.repeat([[1, 2]], self.SEQ // 2, axis=1).reshape(1, self.SEQ).repeat(self.BATCH, 0))

    seg_add = build_compressed_segment_mask(ids, 4)
    seg_bool = build_compressed_segment_mask(ids, 4, emit_bool=True)
    np.testing.assert_array_equal(np.array(seg_add == 0.0), np.array(seg_bool))

    hca_add = build_hca_causal_mask(positions, self.SEQ // 8, 8, jnp.float32)
    hca_bool = build_hca_causal_mask(positions, self.SEQ // 8, 8, jnp.float32, emit_bool=True)
    np.testing.assert_array_equal(np.array(hca_add == 0.0), np.array(hca_bool))

    combined_add = combine_compressed_and_segment_mask(hca_add, seg_add[:, :, : self.SEQ // 8])
    combined_bool = combine_compressed_and_segment_mask(hca_bool, seg_bool[:, :, : self.SEQ // 8])
    np.testing.assert_array_equal(np.array(combined_add >= DEFAULT_MASK_VALUE * 0.5), np.array(combined_bool))

    prefix = build_local_sliding_splash_mask(self.BATCH, ids, positions, self.SEQ, self.SEQ, 8)
    additive_splash = build_compressed_splash_mask(
        jnp.expand_dims(combined_add, axis=2), ids, positions, self.SEQ, self.SEQ + self.SEQ // 8, 8
    )
    bool_splash = build_compressed_splash_mask(
        jnp.expand_dims(combined_bool, axis=2),
        ids,
        positions,
        self.SEQ,
        self.SEQ + self.SEQ // 8,
        8,
        precomputed_uncompressed=prefix,
    )
    np.testing.assert_array_equal(np.array(additive_splash), np.array(bool_splash))

  def test_indexer_selection_same_under_bool_segment_mask(self):
    cfg = make_config()
    comp = make_compressor(cfg, DeepseekV4CSACompressor, 4)
    rng = np.random.default_rng(0)
    hidden = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg.emb_dim)), dtype=jnp.float32)
    q_latent = jnp.array(rng.normal(size=(self.BATCH, self.SEQ, cfg.q_lora_rank)), dtype=jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(self.SEQ)[None, :], (self.BATCH, self.SEQ))
    ids = jnp.array(np.repeat([[1, 2]], self.SEQ // 2, axis=1).reshape(1, self.SEQ).repeat(self.BATCH, 0))

    sel_add = comp.indexer(hidden, q_latent, positions, build_compressed_segment_mask(ids, 4))
    sel_bool = comp.indexer(hidden, q_latent, positions, build_compressed_segment_mask(ids, 4, emit_bool=True))
    np.testing.assert_array_equal(np.array(sel_add), np.array(sel_bool))

  def test_dense_path_rejects_bool_mask(self):
    with self.assertRaises(ValueError):
      AttentionOp.generate_attention_mask(
          None,
          query=jnp.zeros((1, 4, 1, 8)),
          key=jnp.zeros((1, 4, 1, 8)),
          decoder_segment_ids=None,
          model_mode="train",
          compressed_mask=jnp.zeros((1, 1, 4, 4), dtype=jnp.bool_),
      )


if __name__ == "__main__":
  unittest.main()
