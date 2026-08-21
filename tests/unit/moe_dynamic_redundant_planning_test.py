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
"""Unit tests for MoonEP-style dynamic redundant expert planning in MaxText."""

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers.moe import plan_expert_duplication


class MoeDynamicRedundantPlanningTest(unittest.TestCase):
  """Tests for the pure-JAX JIT-compiled greedy load balancer."""

  def setUp(self):
    super().setUp()
    self.num_ep = 4
    self.experts_per_rank = 16  # Total experts = 64
    self.num_slots_B = 2
    self.total_experts = self.num_ep * self.experts_per_rank

  def test_balanced_distribution_no_duplication(self):
    """Under balanced routing, no experts should be duplicated."""
    # Each rank routes exactly 100 tokens to each expert
    token_counts = jnp.full((self.num_ep, self.total_experts), 100, dtype=jnp.int32)
    
    slot_assignments, reroute_fractions = plan_expert_duplication(
        token_counts,
        num_ep=self.num_ep,
        experts_per_rank=self.experts_per_rank,
        num_slots_B=self.num_slots_B,
        rebalance_threshold=1.15,
    )

    # All slots should be unused (-1)
    np.testing.assert_array_equal(np.array(slot_assignments), -1)
    # No tokens should be rerouted
    np.testing.assert_allclose(np.array(reroute_fractions), 0.0)

  def test_single_hot_expert_rebalancing(self):
    """When a single expert is heavily overloaded, it should be assigned to an idle rank."""
    # Rank 0, Expert 0 gets 8000 tokens; all other experts get 50 tokens
    counts = np.full((self.num_ep, self.total_experts), 50, dtype=np.int32)
    counts[:, 0] = 2000  # Total for Expert 0 = 8000 tokens
    token_counts = jnp.array(counts)

    slot_assignments, reroute_fractions = plan_expert_duplication(
        token_counts,
        num_ep=self.num_ep,
        experts_per_rank=self.experts_per_rank,
        num_slots_B=self.num_slots_B,
        rebalance_threshold=1.15,
    )

    slots_np = np.array(slot_assignments)
    fractions_np = np.array(reroute_fractions)

    # Expert 0 MUST be assigned to at least one rank's slot
    self.assertIn(0, slots_np)
    # The home rank for Expert 0 is Rank 0 (0 // 16 = 0). It should NOT be assigned to Rank 0's own slot
    self.assertNotIn(0, slots_np[0])
    # The reroute fraction for Expert 0 should be positive (diverting load)
    self.assertGreater(fractions_np[0], 0.0)
    self.assertLessEqual(fractions_np[0], 0.5)

  def test_multi_expert_slot_capacity(self):
    """Verifies that no rank exceeds its slot capacity B."""
    counts = np.full((self.num_ep, self.total_experts), 10, dtype=np.int32)
    # Overload multiple experts on Rank 0
    counts[:, 0] = 1000
    counts[:, 1] = 1000
    counts[:, 2] = 1000
    counts[:, 3] = 1000
    token_counts = jnp.array(counts)

    slot_assignments, reroute_fractions = plan_expert_duplication(
        token_counts,
        num_ep=self.num_ep,
        experts_per_rank=self.experts_per_rank,
        num_slots_B=self.num_slots_B,
        rebalance_threshold=1.15,
    )

    slots_np = np.array(slot_assignments)
    # Shape must be strictly [num_ep, num_slots_B]
    self.assertEqual(slots_np.shape, (self.num_ep, self.num_slots_B))
    # Each slot can hold at most 1 assigned expert
    for r in range(self.num_ep):
      assigned = [e for e in slots_np[r] if e >= 0]
      self.assertLessEqual(len(assigned), self.num_slots_B)

  def test_jit_compilation_and_numerical_equivalence(self):
    """Verifies JIT compilation succeeds with static shapes and matches un-jitted output."""
    counts = np.random.RandomState(42).randint(10, 500, size=(self.num_ep, self.total_experts)).astype(np.int32)
    counts[:, 5] = 4000  # Make expert 5 hot
    token_counts = jnp.array(counts)

    jitted_plan = jax.jit(
        plan_expert_duplication,
        static_argnames=("num_ep", "experts_per_rank", "num_slots_B"),
    )

    slots_unjitted, fracs_unjitted = plan_expert_duplication(
        token_counts, self.num_ep, self.experts_per_rank, self.num_slots_B
    )
    slots_jitted, fracs_jitted = jitted_plan(
        token_counts, self.num_ep, self.experts_per_rank, self.num_slots_B
    )

    np.testing.assert_array_equal(np.array(slots_jitted), np.array(slots_unjitted))
    np.testing.assert_allclose(np.array(fracs_jitted), np.array(fracs_unjitted), rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
