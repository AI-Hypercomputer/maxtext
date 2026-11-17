# Copyright 2023-2026 Google LLC
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

"""EMo's causal pool (emo_causal_pool): each token's pool comes from its document's tokens up to itself."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.models import olmoe3


class EmoCausalPoolTest(unittest.TestCase):
  """With ``emo_causal_pool`` the pool totals are prefix sums within each document."""

  def _mask(self, key, causal=False, logits=None, segments=None):
    """(logits, EMo-masked logits) for a router with pools in [2, 8] of 8 experts, sampled with `key`."""
    cfg = types.SimpleNamespace(
        emo_causal_pool=causal,
        emo_document_expert_pool_choices=[],
        emo_document_expert_pool_probs=[],
        emo_min_document_expert_pool=2,
        emo_max_document_expert_pool=8,
        emo_eval_document_expert_pool=8,
        emo_threshold_by_bisection=False,
        model_call_mode="train",
        moe_lean_routing=False,
    )
    router = types.SimpleNamespace(config=cfg)
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 8)) if logits is None else logits
    segments = jnp.ones((2, 16), jnp.int32) if segments is None else segments
    return logits, olmoe3.OLMoE3LatentRoutedMoE._emo_mask_logits(  # pylint: disable=protected-access
        router, logits, segments, key
    )

  def test_off_by_default_matches_the_whole_document_pool(self):
    """Without the flag the mask is the whole-document one: every token of a document keeps the same experts."""
    segments = jnp.asarray([[1] * 6 + [2] * 10, [1] * 16], jnp.int32)
    _, whole = self._mask(jax.random.PRNGKey(1), segments=segments)
    keep = np.isfinite(np.asarray(whole))
    for b, start, end in ((0, 0, 6), (0, 6, 16), (1, 0, 16)):
      np.testing.assert_array_equal(keep[b, start:end], np.broadcast_to(keep[b, start], keep[b, start:end].shape))

  def test_causal_pool_ignores_later_tokens(self):
    """With ``emo_causal_pool`` a token's pool depends only on its document's tokens up to itself."""
    segments = jnp.asarray([[1] * 6 + [2] * 10, [1] * 16], jnp.int32)
    key = jax.random.PRNGKey(1)
    logits, masked = self._mask(key, causal=True, segments=segments)
    later = logits.at[:, 9:].add(5.0 * jax.random.normal(jax.random.PRNGKey(2), (2, 7, 8)))
    _, masked_later = self._mask(key, causal=True, logits=later, segments=segments)
    keep, keep_later = np.isfinite(np.asarray(masked)), np.isfinite(np.asarray(masked_later))
    np.testing.assert_array_equal(keep[:, :9], keep_later[:, :9])
    self.assertFalse(np.array_equal(keep[:, 9:], keep_later[:, 9:]), "later pools should follow the changed tokens")

  def test_causal_pool_first_token_keeps_its_own_top_experts(self):
    """A document's first token keeps its own highest-scoring experts; its last token's pool is the whole-document one."""
    segments = jnp.asarray([[1] * 6 + [2] * 10, [1] * 16], jnp.int32)
    key = jax.random.PRNGKey(1)
    logits, causal = self._mask(key, causal=True, segments=segments)
    _, whole = self._mask(key, causal=False, segments=segments)
    causal, whole, logits = np.asarray(causal), np.asarray(whole), np.asarray(logits)
    for b, first, last in ((0, 0, 5), (0, 6, 15), (1, 0, 15)):
      kept = np.isfinite(causal[b, first])
      self.assertTrue(np.all(logits[b, first][kept].min() >= logits[b, first][~kept]))  # its own top-|pool| logits
      np.testing.assert_array_equal(np.isfinite(causal[b, last]), np.isfinite(whole[b, last]))


if __name__ == "__main__":
  unittest.main()
