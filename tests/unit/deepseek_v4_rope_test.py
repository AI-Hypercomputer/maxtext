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

"""Unit tests for DeepSeek-V4 YaRN RoPE rescaling and per-layer theta selection."""

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs.pyconfig import initialize
from maxtext.layers.attention_compressed import CompressedAttention
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.cpu_only

# Mirrors the HF DeepSeek-V4-Flash config.json rope_scaling block, as pinned in
# src/maxtext/configs/models/deepseek4-284b.yml.
_HEAD_DIM = 512
_QK_ROPE_HEAD_DIM = 64
_PARTIAL_ROTARY_FACTOR = _QK_ROPE_HEAD_DIM / _HEAD_DIM
_COMPRESS_ROPE_THETA = 160000.0
_MAIN_ROPE_THETA = 10000.0
_ROPE_FACTOR = 16.0
_BETA_FAST = 32.0
_BETA_SLOW = 1.0
_ORIGINAL_MAX_POSITION_EMBEDDINGS = 65536

# YaRN correction range for the constants above, from
# find_correction_dim(beta) = dim * log(orig_max_pos / (beta * 2 * pi)) / (2 * log(theta))
# with dim = 64: floor(15.45) = 15 and ceil(24.71) = 25. Below `low` the ramp leaves the
# frequency untouched; at or above `high` it is fully divided by rope_factor.
_EXPECTED_LOW = 15
_EXPECTED_HIGH = 25


def _unscaled_inv_freq(rope_theta, dim):
  """Recomputes the pre-YaRN baseline exactly as DeepSeekV4RotaryEmbedding.inv_freq does."""
  half_dim = dim // 2
  fraction = 2 * jnp.arange(0, half_dim, dtype=jnp.float32) / dim
  return np.asarray(1.0 / (rope_theta**fraction))


def _rotary_embedding(rope_type, rope_theta=_COMPRESS_ROPE_THETA):
  return DeepSeekV4RotaryEmbedding(
      head_dim=_HEAD_DIM,
      partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
      rope_theta=rope_theta,
      rope_type=rope_type,
      rope_factor=_ROPE_FACTOR,
      beta_fast=_BETA_FAST,
      beta_slow=_BETA_SLOW,
      original_max_position_embeddings=_ORIGINAL_MAX_POSITION_EMBEDDINGS,
      truncate=True,
  )


class DeepSeekV4RotaryEmbeddingInvFreqTest(unittest.TestCase):
  """Locks in the inverse frequencies produced by DeepSeekV4RotaryEmbedding."""

  def test_default_rope_type_is_bit_identical_to_the_unscaled_baseline(self):
    for rope_theta in (_MAIN_ROPE_THETA, _COMPRESS_ROPE_THETA):
      with self.subTest(rope_theta=rope_theta):
        embedding = _rotary_embedding("default", rope_theta=rope_theta)
        np.testing.assert_array_equal(
            np.asarray(embedding.inv_freq),
            _unscaled_inv_freq(rope_theta, embedding.dim),
        )

  def test_yarn_leaves_high_frequency_bands_untouched(self):
    embedding = _rotary_embedding("yarn")
    scaled = np.asarray(embedding.inv_freq)
    baseline = _unscaled_inv_freq(_COMPRESS_ROPE_THETA, embedding.dim)

    np.testing.assert_array_equal(scaled[: _EXPECTED_LOW + 1], baseline[: _EXPECTED_LOW + 1])

  def test_yarn_divides_low_frequency_bands_by_rope_factor(self):
    embedding = _rotary_embedding("yarn")
    scaled = np.asarray(embedding.inv_freq)
    baseline = _unscaled_inv_freq(_COMPRESS_ROPE_THETA, embedding.dim)

    np.testing.assert_array_equal(scaled[_EXPECTED_HIGH:], baseline[_EXPECTED_HIGH:] / _ROPE_FACTOR)

  def test_yarn_ramp_interpolates_monotonically_between_the_two_regimes(self):
    embedding = _rotary_embedding("yarn")
    scaled = np.asarray(embedding.inv_freq)
    baseline = _unscaled_inv_freq(_COMPRESS_ROPE_THETA, embedding.dim)
    ratio = scaled / baseline

    self.assertEqual(scaled.shape, (embedding.dim // 2,))
    # YaRN only ever stretches wavelengths, so every frequency shrinks or stays put.
    self.assertTrue(np.all(ratio <= 1.0 + 1e-6), f"ratio exceeded 1: {ratio}")
    self.assertTrue(np.all(ratio >= 1.0 / _ROPE_FACTOR - 1e-6), f"ratio below 1/factor: {ratio}")
    self.assertTrue(np.all(np.diff(ratio) <= 1e-7), f"ratio is not non-increasing: {ratio}")
    # The ramp region must be strictly interior, otherwise the test would pass vacuously.
    ramp = ratio[_EXPECTED_LOW + 1 : _EXPECTED_HIGH]
    self.assertTrue(np.all(ramp < 1.0) and np.all(ramp > 1.0 / _ROPE_FACTOR), f"degenerate ramp: {ramp}")

  def test_yarn_actually_changes_the_frequencies(self):
    baseline = np.asarray(_rotary_embedding("default").inv_freq)
    scaled = np.asarray(_rotary_embedding("yarn").inv_freq)

    self.assertFalse(np.allclose(scaled, baseline))


class CompressedAttentionRopeSelectionTest(unittest.TestCase):
  """Checks the per-layer rope_theta / rope_type selection in CompressedAttention."""

  def setUp(self):
    super().setUp()
    self.config = initialize(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek4-284b",
            "attention=dot_product",
            "qk_rope_head_dim=16",
            "v_head_dim=16",
            "qk_nope_head_dim=16",
            "override_model_config=True",
        ]
    )
    self.mesh = Mesh(jax.devices(), ("data",))

  def _build(self, compress_ratio):
    return CompressedAttention(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=512,
        max_target_length=128,
        mesh=self.mesh,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 32, 4096),
        inputs_kv_shape=(1, 32, 4096),
        compress_ratio=compress_ratio,
        q_lora_rank=1024,
        rngs=nnx.Rngs(0),
    )

  def test_model_config_pins_yarn_scaling(self):
    self.assertEqual(self.config.rope_type, "yarn")
    self.assertEqual(self.config.compressed_rope_max_timescale, 160000)
    self.assertEqual(self.config.rope_max_timescale, 10000)

  def test_compressed_layer_uses_compressed_theta_with_yarn(self):
    rotary = self._build(compress_ratio=4).rotary_embedding

    self.assertEqual(rotary.rope_theta, self.config.compressed_rope_max_timescale)
    self.assertEqual(rotary.rope_type, "yarn")

  def test_sliding_window_layer_uses_main_theta_without_yarn(self):
    rotary = self._build(compress_ratio=0).rotary_embedding

    self.assertEqual(rotary.rope_theta, self.config.rope_max_timescale)
    self.assertEqual(rotary.rope_type, "default")


if __name__ == "__main__":
  unittest.main()
