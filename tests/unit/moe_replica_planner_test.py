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

"""CPU tests for the hot-expert replication planners (pure jnp)."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from maxtext.layers import moe_replica_planner as mrp

NUM_EP = 8
NUM_EXPERTS = 64
LOCAL = NUM_EXPERTS // NUM_EP
NUM_REPLICAS = 2
MAX_SEND = 2
BOARD_SIZE = 4

PLANNERS = {
    "greedy": mrp.compute_plan,
    "rounds": functools.partial(mrp.compute_plan_iter, rounds=2 * NUM_REPLICAS),
}


def _skewed_counts(seed=0):
  """[EP(src), E] token counts with a few very hot experts concentrated on two devices."""
  rng = np.random.default_rng(seed)
  counts = rng.integers(10, 20, size=(NUM_EP, NUM_EXPERTS)).astype(np.int32)
  counts[:, 0:3] += 400  # hot experts homed on device 0
  counts[:, LOCAL : LOCAL + 2] += 250  # hot experts homed on device 1
  return jnp.asarray(counts)


def _device_loads(counts, route_table, num_slots):
  """Per-device token load when each (source, logical expert) is sent to route_table[source, expert]."""
  dev = np.asarray(route_table) // num_slots
  loads = np.zeros(counts.shape[0])
  np.add.at(loads, dev.ravel(), np.asarray(counts).ravel())
  return loads


class ReplicaPlannerTest(parameterized.TestCase):

  def test_identity_route_table(self):
    table = np.asarray(mrp.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS))
    self.assertEqual(table.shape, (NUM_EP, NUM_EXPERTS))
    e = np.arange(NUM_EXPERTS)
    expected = (e // LOCAL) * (LOCAL + NUM_REPLICAS) + e % LOCAL
    for src in range(NUM_EP):
      np.testing.assert_array_equal(table[src], expected)
    # Home slots only: never a replica slot.
    self.assertTrue(np.all(table % (LOCAL + NUM_REPLICAS) < LOCAL))

  @parameterized.parameters("greedy", "rounds")
  def test_plan_is_valid(self, planner):
    counts = _skewed_counts()
    plan = PLANNERS[planner](counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    num_slots = LOCAL + NUM_REPLICAS
    self.assertEqual(plan.num_slots, num_slots)
    route = np.asarray(plan.route_table)
    rep_home = np.asarray(plan.rep_home)
    rep_slot = np.asarray(plan.rep_slot)
    self.assertEqual(route.shape, (NUM_EP, NUM_EXPERTS))
    self.assertEqual(rep_home.shape, (NUM_EP, NUM_REPLICAS))
    for src in range(NUM_EP):
      for e in range(NUM_EXPERTS):
        phys = int(route[src, e])
        dev, slot = divmod(phys, num_slots)
        if slot < LOCAL:
          # Home slot: must be the expert's own home.
          self.assertEqual(phys, (e // LOCAL) * num_slots + e % LOCAL)
        else:
          # Replica slot: must hold a replica of exactly this expert, on a device other than its home.
          r = slot - LOCAL
          self.assertEqual(int(rep_home[dev, r]), e // LOCAL)
          self.assertEqual(int(rep_slot[dev, r]), e % LOCAL)
          self.assertNotEqual(dev, e // LOCAL)
    # Each home sends at most max_send replicas.
    valid = rep_home >= 0
    sends = np.bincount(rep_home[valid], minlength=NUM_EP)
    self.assertTrue(np.all(sends <= MAX_SEND))
    # No source shard of a replicated expert is routed to its home device's own replica.
    self.assertGreater(valid.sum(), 0)

  @parameterized.parameters("greedy", "rounds")
  def test_plan_reduces_max_load(self, planner):
    counts = _skewed_counts()
    num_slots = LOCAL + NUM_REPLICAS
    identity = mrp.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS)
    before = _device_loads(counts, identity, num_slots)
    plan = PLANNERS[planner](counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    after = _device_loads(counts, plan.route_table, num_slots)
    self.assertEqual(after.sum(), before.sum())  # every token is routed exactly once
    self.assertLess(after.max(), 0.9 * before.max())

  @parameterized.parameters("greedy", "rounds")
  def test_disabled_plan_is_identity(self, planner):
    counts = _skewed_counts()
    plan = PLANNERS[planner](counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE, enabled=False)
    np.testing.assert_array_equal(
        np.asarray(plan.route_table), np.asarray(mrp.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS))
    )
    self.assertTrue(np.all(np.asarray(plan.rep_home) < 0))

  @parameterized.parameters("greedy", "rounds")
  def test_balanced_counts_need_no_replicas(self, planner):
    counts = jnp.full((NUM_EP, NUM_EXPERTS), 16, jnp.int32)
    plan = PLANNERS[planner](counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    num_slots = LOCAL + NUM_REPLICAS
    after = _device_loads(counts, plan.route_table, num_slots)
    np.testing.assert_allclose(after, float(counts.sum()) / NUM_EP)

  @parameterized.parameters("greedy", "rounds", "lagged")
  def test_more_replica_slots_than_experts(self, mode):
    """Regression test: num_ep * num_replicas (16) > num_experts (8) clamps top_k candidates cleanly."""
    num_ep, num_experts, num_replicas, max_send, board_size = 4, 8, 4, 3, 2
    local = num_experts // num_ep
    num_slots = local + num_replicas
    rng = np.random.default_rng(7)
    counts_np = rng.integers(5, 15, size=(num_ep, num_experts)).astype(np.int32)
    counts_np[:, 0] += 300
    counts = jnp.asarray(counts_np)

    if mode == "greedy":
      plan = mrp.compute_plan(counts, num_ep, num_replicas, max_send, board_size)
    elif mode == "rounds":
      plan = mrp.compute_plan_iter(counts, num_ep, num_replicas, max_send, board_size, rounds=2 * num_replicas)
    else:
      rep_e, order, n_rep = mrp.compute_placement(jnp.sum(counts, axis=0), num_ep, num_replicas, max_send)
      self.assertEqual(order.shape, (num_experts, 2))
      plan = mrp.plan_from_placement(counts, rep_e, order, n_rep, num_ep, num_replicas, max_send, board_size)

    before = _device_loads(counts, mrp.identity_route_table(num_ep, num_experts, num_replicas), num_slots)
    after = _device_loads(counts, plan.route_table, num_slots)
    self.assertEqual(after.sum(), before.sum())
    self.assertLess(after.max(), before.max())


if __name__ == "__main__":
  absltest.main()
