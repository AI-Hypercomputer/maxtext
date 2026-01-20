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

"""Tests for QK-Clip utilities."""

import unittest
from collections import namedtuple
import jax.numpy as jnp
from maxtext.utils.qk_clip_utils import apply_qk_clip


class QKClipTest(unittest.TestCase):

  def _get_config_and_state(self, threshold, nope_dim, params_dict):
    """Helper to create mock Config and State objects."""
    Config = namedtuple("Config", ["qk_clip_threshold", "qk_nope_head_dim"])
    config = Config(qk_clip_threshold=threshold, qk_nope_head_dim=nope_dim)

    State = namedtuple("State", ["params", "replace"])
    state = State(params=params_dict, replace=lambda params: State(params, None))
    return config, state

  def test_apply_qk_clip_logic(self):
    """Tests QK Clip math and application logic on CPU."""
    # 1. Setup Mock Data
    wq_b = jnp.ones((2, 2, 6))  # [rank, heads, dim]
    wkv_b = jnp.ones((2, 2, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}, "wkv_b": {"kernel": wkv_b}}}}}
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    # 2. Setup Mock Intermediates
    # Head 0: max_logit = 20.0 (>10.0) -> Gamma = 0.5
    # Head 1: max_logit = 5.0  (<10.0) -> Gamma = 1.0
    max_logits = jnp.array([[20.0, 5.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    # 3. Run Apply Clip
    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]
    new_wkv = new_state.params["decoder"]["layers_0"]["self_attention"]["wkv_b"]["kernel"]

    # 4. Verify Results
    # Head 0: Scale = 0.5. W_c * sqrt(0.5), W_r * 0.5
    self.assertTrue(jnp.allclose(new_wq[:, 0, :4], 1.0 * jnp.sqrt(0.5)))
    self.assertTrue(jnp.allclose(new_wq[:, 0, 4:], 0.5))
    self.assertTrue(jnp.allclose(new_wkv[:, 0, :4], 1.0 * jnp.sqrt(0.5)))

    # Head 1: Scale = 1.0. No change.
    self.assertTrue(jnp.allclose(new_wq[:, 1, :], 1.0))
    self.assertTrue(jnp.allclose(new_wkv[:, 1, :], 1.0))

  def test_verify_per_head_clipping(self):
    """Explicitly verifies that clipping is applied independently per head."""
    # Setup 3 Heads:
    # Head 0: 40.0 (Way above threshold) -> Scale 0.25
    # Head 1: 9.9  (Just below threshold) -> Scale 1.0
    # Head 2: 1.0  (Way below threshold)  -> Scale 1.0
    wq_b = jnp.ones((1, 3, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}}}}}
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    max_logits = jnp.array([[40.0, 9.9, 1.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]

    # Head 0: Scale 0.25. W_qc = 1.0 * sqrt(0.25) = 0.5
    self.assertTrue(jnp.allclose(new_wq[0, 0, 0], 0.5))
    # Head 1: Unchanged
    self.assertTrue(jnp.allclose(new_wq[0, 1, 0], 1.0))
    # Head 2: Unchanged
    self.assertTrue(jnp.allclose(new_wq[0, 2, 0], 1.0))

  def test_no_clipping_when_below_threshold(self):
    """Verifies that weights are unchanged when max_logits < tau."""
    wq_b = jnp.ones((2, 1, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}}}}}
    config, state = self._get_config_and_state(threshold=100.0, nope_dim=4, params_dict=params)

    # Max logits = 50.0 (Below threshold 100.0)
    max_logits = jnp.array([[50.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]

    # Assert exact equality
    self.assertTrue(jnp.array_equal(new_wq, wq_b))

  def test_shared_keys_are_untouched(self):
    """Verifies that wkv_a (Shared Key) is ignored by the clipper."""
    wkv_a = jnp.full((2, 1, 6), 5.0)  # Arbitrary value
    params = {
        "decoder": {
            "layers_0": {
                "self_attention": {
                    "wkv_a": {"kernel": wkv_a},  # Should be ignored
                    "wkv_b": {"kernel": jnp.ones((2, 1, 6))},  # Should be clipped
                }
            }
        }
    }
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    # Trigger clipping with high logits
    max_logits = jnp.array([[100.0]])
    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    new_state = apply_qk_clip(state, intermediates, config)
    new_wkv_a = new_state.params["decoder"]["layers_0"]["self_attention"]["wkv_a"]["kernel"]

    # Assert wkv_a is completely unchanged
    self.assertTrue(jnp.array_equal(new_wkv_a, wkv_a))

  def test_resilience_to_missing_stats(self):
    """Verifies that code handles layers without max_logits gracefully."""
    wq_b = jnp.ones((2, 1, 6))
    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}}}}}
    config, state = self._get_config_and_state(threshold=10.0, nope_dim=4, params_dict=params)

    # Intermediates dict is empty
    intermediates = {}

    # Should not crash, should return original params
    new_state = apply_qk_clip(state, intermediates, config)
    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]

    self.assertTrue(jnp.array_equal(new_wq, wq_b))

  def test_dot_product_max_logits(self):
    """Verifies max_logits calculation logic used in AttentionOp."""
    # [Batch=1, Heads=1, Len_Q=2, Dim=4]
    q = jnp.array([[[[10.0, 0, 0, 0], [0, 10.0, 0, 0]]]])
    k = jnp.array([[[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]]])

    # Standard Einsum for dot product attention
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)

    # Max logits over Q and K dimensions
    computed_max = jnp.max(logits, axis=(-2, -1))

    self.assertEqual(computed_max.shape, (1, 1))
    self.assertEqual(computed_max[0, 0], 10.0)


if __name__ == "__main__":
  unittest.main()
