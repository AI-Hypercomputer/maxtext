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
import jax
import jax.numpy as jnp
from MaxText.utils.qk_clip_utils import apply_qk_clip
from collections import namedtuple


class QKClipTest(unittest.TestCase):

  def test_apply_qk_clip_logic(self):
    """Tests QK Clip math and application logic on CPU."""

    # 1. Setup Mock Config
    Config = namedtuple("Config", ["qk_clip_threshold", "qk_nope_head_dim"])
    config = Config(qk_clip_threshold=10.0, qk_nope_head_dim=4)

    # 2. Setup Mock Params (simulating MLA structure)
    # Shape: [rank, heads, dim]
    # We use 2 heads. Head 0 will explode, Head 1 will be safe.
    # rank=2, nope=4, rope=2 -> total_q=6
    wq_b = jnp.ones((2, 2, 6))
    # rank=2, nope=4, v=2 -> total_kv=6
    wkv_b = jnp.ones((2, 2, 6))

    params = {"decoder": {"layers_0": {"self_attention": {"wq_b": {"kernel": wq_b}, "wkv_b": {"kernel": wkv_b}}}}}

    # Mock State object
    State = namedtuple("State", ["params", "replace"])
    state = State(params=params, replace=lambda params: State(params, None))

    # 3. Setup Mock Intermediates (max_logits)
    # Head 0: max_logit = 20.0 (Threshold is 10.0, so gamma = 10/20 = 0.5)
    # Head 1: max_logit = 5.0  (Threshold is 10.0, so gamma = 1.0)
    # Shape: [batch=1, heads=2]
    max_logits = jnp.array([[20.0, 5.0]])

    intermediates = {"decoder": {"layers_0": {"self_attention": {"max_logits": (max_logits,)}}}}

    # 4. Run Apply Clip
    # We mock pmax to just return the input (single device behavior)
    with jax.check_tracer_leaks(False):  # helper for raw numpy/jax mix if needed
      new_state = apply_qk_clip(state, intermediates, config)

    new_wq = new_state.params["decoder"]["layers_0"]["self_attention"]["wq_b"]["kernel"]
    new_wkv = new_state.params["decoder"]["layers_0"]["self_attention"]["wkv_b"]["kernel"]

    # 5. Verify Results

    # -- Check W_q (wq_b) --
    # Head 0: Scale = 0.5
    # W_qc (first 4 dims): * sqrt(0.5) approx 0.7071
    # W_qr (last 2 dims): * 0.5

    # W_qc check
    self.assertTrue(jnp.allclose(new_wq[:, 0, :4], 1.0 * jnp.sqrt(0.5)))
    # W_qr check
    self.assertTrue(jnp.allclose(new_wq[:, 0, 4:], 0.5))

    # Head 1: Scale = 1.0 (No Change)
    self.assertTrue(jnp.allclose(new_wq[:, 1, :], 1.0))

    # -- Check W_k/v (wkv_b) --
    # Head 0: Scale = 0.5
    # W_kc (first 4 dims): * sqrt(0.5)
    # W_v (last 2 dims): No change (1.0)

    # W_kc check
    self.assertTrue(jnp.allclose(new_wkv[:, 0, :4], 1.0 * jnp.sqrt(0.5)))
    # W_v check
    self.assertTrue(jnp.allclose(new_wkv[:, 0, 4:], 1.0))

  def test_dot_product_max_logits(self):
    """Verifies max_logits calculation using standard dot product attention."""
    # This simulates the logic added to AttentionOp.apply_attention_dot

    # [Batch=1, Heads=1, Len_Q=2, Dim=4]
    q = jnp.array([[[[10.0, 0, 0, 0], [0, 10.0, 0, 0]]]])
    k = jnp.array([[[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]]])  # Key matches dim 0

    # Q @ K.T
    # q[0] dot k[0] = 10*1 = 10
    # q[0] dot k[1] = 10*1 = 10
    # q[1] dot k[0] = 0

    # Standard Einsum for dot product attention
    # shape: [b, heads, q_len, k_len]
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)

    # Max logits over Q and K dimensions
    computed_max = jnp.max(logits, axis=(-2, -1))  # [b, heads]

    self.assertEqual(computed_max.shape, (1, 1))
    self.assertEqual(computed_max[0, 0], 10.0)


if __name__ == "__main__":
  unittest.main()
