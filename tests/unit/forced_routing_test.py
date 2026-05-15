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
"""Unit tests for forced routing in moe.py."""

import unittest
import jax
import jax.numpy as jnp
from maxtext.layers import moe
from maxtext.common import common_types as ctypes


class DummyConfig:

  def __init__(self, model_name="default", decoder_block=ctypes.DecoderBlockType.DEFAULT):
    self.model_name = model_name
    self.decoder_block = decoder_block
    self.norm_topk_prob = False
    self.use_random_routing = False
    self.shard_mode = ctypes.ShardMode.AUTO


class DummyRoutedMoE:

  def __init__(self, config):
    self.config = config
    self.dtype = jnp.float32
    self.num_experts_per_tok = 2
    self.num_experts = 3

  def _maybe_shard_with_logical(self, x, spec):
    return x


class ForcedRoutingTest(unittest.TestCase):

  def test_basic_override(self):
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    pre_bias_logits = gate_logits  # Not DeepSeek
    forced_routed_experts = jnp.array([[[2, 1], [0, 2]]])  # (1, 2, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model, gate_logits, pre_bias_logits, forced_routed_experts=forced_routed_experts
    )

    # Check that indices are overridden
    self.assertTrue((top_k_indices == forced_routed_experts).all())
    # Check that weights are extracted correctly and softmaxed
    # For token 0: indices 2, 1 -> logits 3.0, 2.0 -> softmax([3.0, 2.0])
    # For token 1: indices 0, 2 -> logits 4.0, 6.0 -> softmax([4.0, 6.0])
    expected_weights = jax.nn.softmax(jnp.array([[[3.0, 2.0], [4.0, 6.0]]]).astype(jnp.float32), axis=-1)
    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_gemma4_softmax(self):
    config = DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4)
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    pre_bias_logits = gate_logits
    forced_routed_experts = jnp.array([[[2, 1], [0, 2]]])  # (1, 2, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model, gate_logits, pre_bias_logits, forced_routed_experts=forced_routed_experts
    )

    # Check that indices are overridden
    self.assertTrue((top_k_indices == forced_routed_experts).all())

    # For Gemma 4, it applies softmax to gate_logits first!

    expected_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
    expected_weights = jnp.take_along_axis(expected_probs, forced_routed_experts, axis=-1)

    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_reshape_and_update_weights(self):
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    weights = jnp.array([[[0.1, 0.2], [0.3, 0.4]]])  # (1, 2, 2)
    indices = jnp.array([[[2, -1], [-1, 1]]])  # (1, 2, 2)

    update_weights = moe.RoutedMoE.reshape_and_update_weights(model, weights, indices, safe_updates=True)

    # Expected shape: (1, 2, 3) where 3 is num_experts!
    # For token 0: index 2 -> 0.1. Index -1 -> mapped to 0 but weight 0.0!
    # So for expert 0: 0.0. Expert 1: 0.0. Expert 2: 0.1.
    # For token 1: index -1 -> mapped to 0 but weight 0.0! Index 1 -> 0.4.
    # So for expert 0: 0.0. Expert 1: 0.4. Expert 2: 0.0.
    expected_update_weights = jnp.array([[[0.0, 0.0, 0.1], [0.0, 0.4, 0.0]]])

    self.assertTrue(jax.numpy.allclose(update_weights, expected_update_weights, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
  unittest.main()
