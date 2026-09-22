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

"""Tests for .

Three layers of coverage:
  * Functional correctness: prefill caching, autoregressive step-by-step ring buffer update.
  * Bit-exact parity against legacy .
  * Input validation: ensures invalid modes and shapes fail loudly.
"""

import unittest
from absl.testing import absltest
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import pytest

from maxtext.common.common_types import (
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
)
from maxtext.inference import kvcache as legacy_kvcache
from maxtext.m3.core.kvcache import KVCache as M3KVCache

_BATCH = 2
_PREFILL_LEN = 8
_TARGET_LEN = 16
_HEADS = 4
_HEAD_DIM = 16
_DTYPE = jnp.float32
_RULES = (
    ('cache_batch_prefill', None),
    ('cache_batch', None),
    ('cache_sequence', None),
    ('cache_heads', None),
    ('cache_kv', None),
)


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class KVCacheTest(unittest.TestCase):
  """Tests for m3 model-agnostic KVCache."""

  def setUp(self):
    """Sets up device mesh for KV cache unit tests."""
    super().setUp()
    self.devices = jax.devices()[:1]
    self.mesh = jax.sharding.Mesh(self.devices, ('data',))

  def _create_caches(self, model_mode):
    """Creates instances of M3 and legacy KVCache for comparative testing."""
    with jax.set_mesh(self.mesh), logical_axis_rules(_RULES):
      cache_m3 = M3KVCache(
          max_prefill_length=_PREFILL_LEN,
          max_target_length=_TARGET_LEN,
          batch=_BATCH,
          key_heads=_HEADS,
          value_heads=_HEADS,
          key_head_size=_HEAD_DIM,
          value_head_size=_HEAD_DIM,
          dtype=_DTYPE,
          model_mode=model_mode,
      )
      cache_legacy = legacy_kvcache.KVCache(
          max_prefill_length=_PREFILL_LEN,
          max_target_length=_TARGET_LEN,
          batch=_BATCH,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=_HEADS,
          value_heads=_HEADS,
          key_head_size=_HEAD_DIM,
          value_head_size=_HEAD_DIM,
          dtype=_DTYPE,
          model_mode=model_mode,
      )
      return cache_m3, cache_legacy

  def test_prefill_parity_against_legacy(self):
    """Prefill caching in M3 must produce bit-identical results to legacy KVCache."""
    cache_m3, cache_legacy = self._create_caches(MODEL_MODE_PREFILL)

    key = jax.random.normal(jax.random.PRNGKey(0), (_BATCH, _PREFILL_LEN, _HEADS, _HEAD_DIM), dtype=_DTYPE)
    value = jax.random.normal(jax.random.PRNGKey(1), (_BATCH, _PREFILL_LEN, _HEADS, _HEAD_DIM), dtype=_DTYPE)
    seg_ids = jnp.ones((_BATCH, _PREFILL_LEN), dtype=jnp.int32)

    with jax.set_mesh(self.mesh), logical_axis_rules(_RULES):
      out_m3, ar_m3 = cache_m3(key, value, decoder_segment_ids=seg_ids, model_mode=MODEL_MODE_PREFILL)
      out_leg, ar_leg = cache_legacy(key, value, decoder_segment_ids=seg_ids, model_mode=MODEL_MODE_PREFILL)

    self.assertIsNone(ar_m3)
    self.assertIsNone(ar_leg)

    k_m3, v_m3, seg_m3 = out_m3
    k_leg, v_leg, seg_leg = out_leg

    self.assertTrue(jnp.array_equal(k_m3, k_leg), 'Prefill key mismatch')
    self.assertTrue(jnp.array_equal(v_m3, v_leg), 'Prefill value mismatch')
    self.assertTrue(jnp.array_equal(seg_m3, seg_leg), 'Prefill segment id mismatch')

  def test_autoregressive_step_parity_against_legacy(self):
    """Autoregressive decoding steps in M3 must match legacy KVCache ring buffer."""
    cache_m3, cache_legacy = self._create_caches(MODEL_MODE_AUTOREGRESSIVE)

    # Initial prefill setup
    prompt_key = jax.random.normal(jax.random.PRNGKey(2), (_BATCH, _PREFILL_LEN, _HEADS, _HEAD_DIM), dtype=_DTYPE)
    prompt_value = jax.random.normal(jax.random.PRNGKey(3), (_BATCH, _PREFILL_LEN, _HEADS, _HEAD_DIM), dtype=_DTYPE)
    prompt_seg = jnp.ones((_BATCH, _PREFILL_LEN), dtype=jnp.int32)

    with jax.set_mesh(self.mesh), logical_axis_rules(_RULES):
      cache_m3.kv_cache_prefill(prompt_key, prompt_value, prompt_seg)
      cache_legacy.kv_cache_prefill(prompt_key, prompt_value, prompt_seg)

      # Run 3 decoding steps
      for step in range(3):
        token_key = jax.random.normal(jax.random.PRNGKey(10 + step), (_BATCH, 1, _HEADS, _HEAD_DIM), dtype=_DTYPE)
        token_value = jax.random.normal(jax.random.PRNGKey(20 + step), (_BATCH, 1, _HEADS, _HEAD_DIM), dtype=_DTYPE)

        token_seg = jnp.ones((_BATCH, 1), dtype=jnp.int32)
        pref_m3, ar_m3 = cache_m3(token_key, token_value, decoder_segment_ids=token_seg, model_mode=MODEL_MODE_AUTOREGRESSIVE)
        pref_leg, ar_leg = cache_legacy(token_key, token_value, decoder_segment_ids=token_seg, model_mode=MODEL_MODE_AUTOREGRESSIVE)

        # Prefill cache retrieved in AR mode should be identical
        self.assertTrue(jnp.array_equal(pref_m3[0], pref_leg[0]), f'AR step {step} prefill key mismatch')
        self.assertTrue(jnp.array_equal(pref_m3[1], pref_leg[1]), f'AR step {step} prefill value mismatch')

        # AR cache (key, value, segment_id, lengths) should be identical
        k_ar_m3, v_ar_m3, seg_ar_m3, len_ar_m3 = ar_m3
        k_ar_leg, v_ar_leg, seg_ar_leg, len_ar_leg = ar_leg

        self.assertTrue(jnp.array_equal(k_ar_m3, k_ar_leg), f'AR step {step} key mismatch')
        self.assertTrue(jnp.array_equal(v_ar_m3, v_ar_leg), f'AR step {step} value mismatch')
        self.assertTrue(jnp.array_equal(seg_ar_m3, seg_ar_leg), f'AR step {step} segment id mismatch')
        self.assertTrue(jnp.array_equal(len_ar_m3, len_ar_leg), f'AR step {step} length mismatch')

  def test_validation_errors(self):
    """Ensures illegal lengths and operational modes fail with clear errors."""
    with self.assertRaises(ValueError):
      M3KVCache(
          max_prefill_length=16,
          max_target_length=16,  # target must be > prefill
          batch=_BATCH,
          key_head_size=_HEAD_DIM,
          value_head_size=_HEAD_DIM,
          key_heads=_HEADS,
          value_heads=_HEADS,
          dtype=_DTYPE,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

    cache_m3, _ = self._create_caches(MODEL_MODE_AUTOREGRESSIVE)
    multi_token_key = jnp.zeros((_BATCH, 2, _HEADS, _HEAD_DIM), dtype=_DTYPE)
    multi_token_val = jnp.zeros((_BATCH, 2, _HEADS, _HEAD_DIM), dtype=_DTYPE)

    with self.assertRaises(ValueError):
      cache_m3(multi_token_key, multi_token_val, model_mode=MODEL_MODE_AUTOREGRESSIVE)


if __name__ == '__main__':
  absltest.main()
