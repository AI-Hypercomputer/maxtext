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

"""CPU tests for the online hot-expert replication planner."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import moe_replica_planner as mrp

NUM_EP = 8
NUM_EXPERTS = 64
LOCAL = NUM_EXPERTS // NUM_EP
NUM_REPLICAS = 2
MAX_SEND = 2
BOARD_SIZE = 4


def _greedy_reference_plan(counts_src, num_ep, num_replicas, max_send, board_size):
  """Reference serial greedy planner over top-`EP*R` hot experts used as a quality oracle."""
  _, num_experts = counts_src.shape
  local = num_experts // num_ep
  counts_src = counts_src.astype(jnp.float32)
  c = jnp.sum(counts_src, axis=0)
  dev_load = jnp.sum(c.reshape(num_ep, local), axis=1)
  num_cand = min(num_ep * num_replicas, num_experts)
  _, cand = jax.lax.top_k(c, num_cand)
  cand = cand.astype(jnp.int32)
  src_ids = jnp.arange(num_ep, dtype=jnp.int32)

  def body(i, state):
    dev_load, free, send_cnt, pair_used, rep_e, rep_mask = state
    e = cand[i]
    h = e // local
    blocked = (free <= 0) | (src_ids == h) | pair_used[h]
    r = jnp.argmin(jnp.where(blocked, jnp.inf, dev_load)).astype(jnp.int32)
    ok = (~blocked[r]) & (send_cnt[h] < max_send)
    target = (dev_load[h] - dev_load[r]) / 2.0
    same_board = (src_ids // board_size) == (r // board_size)
    key = jnp.where(src_ids == h, 3 * num_ep, jnp.where(same_board, 0, num_ep) + (src_ids - r) % num_ep)
    order = jnp.argsort(key)
    contrib = counts_src[order, e]
    cum = jnp.cumsum(contrib)
    take_sorted = (cum <= target) & (jnp.arange(num_ep) < num_ep - 1)
    moved = jnp.sum(jnp.where(take_sorted, contrib, 0.0))
    ok = ok & (moved > 0)
    take = jnp.zeros((num_ep,), bool).at[order].set(take_sorted) & ok
    slot = jnp.minimum((num_replicas - free[r]).astype(jnp.int32), num_replicas - 1)
    dev_load = dev_load.at[h].add(jnp.where(ok, -moved, 0.0)).at[r].add(jnp.where(ok, moved, 0.0))
    free = free.at[r].add(jnp.where(ok, -1, 0))
    send_cnt = send_cnt.at[h].add(jnp.where(ok, 1, 0))
    pair_used = pair_used.at[h, r].set(pair_used[h, r] | ok)
    rep_e = rep_e.at[r, slot].set(jnp.where(ok, e, rep_e[r, slot]))
    rep_mask = rep_mask.at[r, slot].set(jnp.where(ok, take, rep_mask[r, slot]))
    return dev_load, free, send_cnt, pair_used, rep_e, rep_mask

  init = (
      dev_load,
      jnp.full((num_ep,), num_replicas, jnp.int32),
      jnp.zeros((num_ep,), jnp.int32),
      jnp.zeros((num_ep, num_ep), bool),
      jnp.full((num_ep, num_replicas), -1, jnp.int32),
      jnp.zeros((num_ep, num_replicas, num_ep), bool),
  )
  _, _, _, _, rep_e, rep_mask = jax.lax.fori_loop(0, num_cand, body, init)
  return mrp._finalize_plan(rep_e, rep_mask, num_ep, num_experts, num_replicas, max_send)  # pylint: disable=protected-access


def _skewed_counts(seed=0):
  """[EP(src), E] token counts with hot experts concentrated on two devices."""
  rng = np.random.default_rng(seed)
  counts = rng.integers(10, 20, size=(NUM_EP, NUM_EXPERTS)).astype(np.int32)
  counts[:, 0:3] += 400
  counts[:, LOCAL : LOCAL + 2] += 250
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
    self.assertTrue(np.all(table % (LOCAL + NUM_REPLICAS) < LOCAL))

  def test_plan_is_valid(self):
    counts = _skewed_counts()
    plan = mrp.compute_plan_iter(counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    num_slots = LOCAL + NUM_REPLICAS
    self.assertEqual(plan.num_slots, num_slots)
    self.assertEqual(plan.num_replicas, NUM_REPLICAS)
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
          self.assertEqual(phys, (e // LOCAL) * num_slots + e % LOCAL)
        else:
          r = slot - LOCAL
          self.assertEqual(int(rep_home[dev, r]), e // LOCAL)
          self.assertEqual(int(rep_slot[dev, r]), e % LOCAL)
          self.assertNotEqual(dev, e // LOCAL)
          self.assertNotEqual(src, e // LOCAL)
    valid = rep_home >= 0
    sends = np.bincount(rep_home[valid], minlength=NUM_EP)
    self.assertTrue(np.all(sends <= MAX_SEND))
    self.assertGreater(valid.sum(), 0)
    # Verify send_peer_index and send_slots agree with (rep_home, rep_slot).
    spi = np.asarray(plan.send_peer_index)
    send_slots = np.asarray(plan.send_slots)
    for r in range(NUM_EP):
      for s in range(NUM_REPLICAS):
        h = int(rep_home[r, s])
        if h >= 0:
          pos = int(spi[h, r])
          self.assertGreaterEqual(pos, 0)
          self.assertLess(pos, MAX_SEND)
          self.assertEqual(int(send_slots[h, pos]), int(rep_slot[r, s]))

  @parameterized.parameters(0, 1, 2, 3)
  def test_plan_reduces_max_load_vs_greedy_oracle(self, seed):
    counts = _skewed_counts(seed=seed)
    num_slots = LOCAL + NUM_REPLICAS
    identity = mrp.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS)
    before = _device_loads(counts, identity, num_slots)
    rounds_plan = mrp.compute_plan_iter(counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    greedy_plan = _greedy_reference_plan(counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    after_rounds = _device_loads(counts, rounds_plan.route_table, num_slots)
    after_greedy = _device_loads(counts, greedy_plan.route_table, num_slots)
    self.assertEqual(after_rounds.sum(), before.sum())
    self.assertLess(after_rounds.max(), 0.75 * before.max())
    # Rounds matches or beats greedy within 5% across skewed profiles.
    self.assertLessEqual(after_rounds.max(), 1.05 * after_greedy.max())

  def test_disabled_plan_is_identity(self):
    counts = _skewed_counts()
    plan = mrp.compute_plan_iter(counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE, enabled=False)
    np.testing.assert_array_equal(
        np.asarray(plan.route_table), np.asarray(mrp.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS))
    )
    self.assertTrue(np.all(np.asarray(plan.rep_home) == -1))
    self.assertTrue(np.all(np.asarray(plan.send_peer_index) == -1))

  def test_expert_zero_not_clobbered_by_empty_slots(self):
    """Regression test: when expert 0 is replicated and later replica slots stay empty (-1), route[:, 0] stays valid."""
    counts = jnp.ones((NUM_EP, NUM_EXPERTS), dtype=jnp.int32).at[:, 0].set(1000)
    plan = mrp.compute_plan_iter(counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    route_e0 = np.asarray(plan.route_table)[:, 0]
    # Home shard 0 keeps its own shard of expert 0; at least one non-home shard is rerouted off device 0.
    self.assertEqual(int(route_e0[0]), 0)
    self.assertTrue(np.any(route_e0[1:] >= LOCAL + NUM_REPLICAS))
    loads_before = _device_loads(
        counts, mrp.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS), LOCAL + NUM_REPLICAS
    )
    loads_after = _device_loads(counts, plan.route_table, plan.num_slots)
    self.assertEqual(loads_after.sum(), loads_before.sum())
    self.assertLess(loads_after[0], 0.65 * loads_before[0])

  def test_same_board_sources_preferred(self):
    """When a hot expert is replicated to board 1, sources on board 1 reroute before sources on board 0."""
    # Device 0 (board 0) owns hot expert 0; devices 4..7 (board 1) are cold, with uniform per-source counts on expert 0.
    counts = np.full((NUM_EP, NUM_EXPERTS), 20, dtype=np.int32)
    counts[:, LOCAL : 4 * LOCAL] = 30  # keep board 0 peers slightly busier than board 1
    counts[:, 4 * LOCAL :] = 5
    counts[:, 0] = 200
    plan = mrp.compute_plan_iter(jnp.asarray(counts), NUM_EP, 1, 1, BOARD_SIZE, rounds=1, shift=False)
    route_e0 = np.asarray(plan.route_table)[:, 0]
    rerouted_board0 = int(np.sum(route_e0[1:4] != 0))
    rerouted_board1 = int(np.sum(route_e0[4:8] != 0))
    self.assertGreater(rerouted_board1, 0)
    self.assertGreaterEqual(rerouted_board1, rerouted_board0)

  def test_more_replica_slots_than_experts_supported(self):
    counts = jnp.asarray([[500, 10, 10, 10], [500, 10, 10, 10], [500, 10, 10, 10], [500, 10, 10, 10]], jnp.int32)
    plan = mrp.compute_plan_iter(counts, num_ep=4, num_replicas=3, max_send=2, board_size=2)
    self.assertEqual(plan.num_slots, 1 + 3)
    loads = _device_loads(counts, plan.route_table, plan.num_slots)
    self.assertEqual(loads.sum(), float(counts.sum()))
    self.assertLess(loads.max(), 2000.0)

  def test_invalid_arguments_raise(self):
    counts = _skewed_counts()
    with self.assertRaisesRegex(ValueError, "num_ep >= 2"):
      mrp.compute_plan_iter(counts[:1], 1, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    with self.assertRaisesRegex(ValueError, "dividing num_experts"):
      mrp.compute_plan_iter(counts[:, :63], NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    with self.assertRaisesRegex(ValueError, "num_replicas >= 1"):
      mrp.compute_plan_iter(counts, NUM_EP, 0, MAX_SEND, BOARD_SIZE)
    with self.assertRaisesRegex(ValueError, "max_send >= 1"):
      mrp.compute_plan_iter(counts, NUM_EP, NUM_REPLICAS, 0, BOARD_SIZE)
    with self.assertRaisesRegex(ValueError, "board_size >= 1"):
      mrp.compute_plan_iter(counts, NUM_EP, NUM_REPLICAS, MAX_SEND, 0)

  def test_plan_satisfies_weight_move_invariants(self):
    cases = [
        (_skewed_counts(seed=0), NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE),
        (_skewed_counts(seed=3), NUM_EP, NUM_REPLICAS, 2, BOARD_SIZE),
        (
            jnp.asarray([[500, 10, 10, 10], [500, 10, 10, 10], [500, 10, 10, 10], [500, 10, 10, 10]], jnp.int32),
            4,
            3,
            2,
            2,
        ),
    ]
    for counts, ep, r_slots, max_send, board in cases:
      local = counts.shape[1] // ep
      plan = mrp.compute_plan_iter(counts, ep, r_slots, max_send, board)
      rep_home = np.asarray(plan.rep_home)
      rep_slot = np.asarray(plan.rep_slot)
      send_slots = np.asarray(plan.send_slots)
      spi = np.asarray(plan.send_peer_index)
      pairs = set()
      for r in range(ep):
        valid_homes = [int(h) for h in rep_home[r] if h >= 0]
        # Invariant 2: distinct valid home devices per receiver's R slots.
        self.assertEqual(len(valid_homes), len(set(valid_homes)))
        for s, h in enumerate(rep_home[r]):
          if h >= 0:
            # Invariant 1: at most one replica per (home, receiver) pair.
            self.assertNotIn((int(h), r), pairs)
            pairs.add((int(h), r))
            idx = int(spi[int(h), r])
            self.assertTrue(0 <= idx < max_send)
            self.assertEqual(int(send_slots[int(h), idx]), int(rep_slot[r, s]))
      for h in range(ep):
        valid_rcvs = [r for r in range(ep) if spi[h, r] >= 0]
        k = len(valid_rcvs)
        self.assertLessEqual(k, max_send)
        # Invariant 3: valid sends compacted at front [0..k-1] in ascending receiver order, within [0, local).
        self.assertEqual([int(spi[h, r]) for r in valid_rcvs], list(range(k)))
        if k > 0:
          self.assertTrue(np.all((send_slots[h, :k] >= 0) & (send_slots[h, :k] < local)))
        self.assertTrue(np.all(send_slots[h, k:] == 0))


if __name__ == "__main__":
  absltest.main()
