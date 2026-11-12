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

"""CPU tests for the hot-expert replication planners and expert-replication config flags.

The ragged all-to-all weight move is not implemented on CPU, so only the planners (pure jnp) and
config parsing are tested here.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import pydantic

from maxtext.configs import pyconfig
from maxtext.layers import moe_expert_replication as mer
from tests.utils.test_helpers import get_test_config_path

NUM_EP = 8
NUM_EXPERTS = 64
LOCAL = NUM_EXPERTS // NUM_EP
NUM_REPLICAS = 2
MAX_SEND = 2
BOARD_SIZE = 4

PLANNERS = {
    "greedy": mer.compute_plan,
    "rounds": functools.partial(mer.compute_plan_iter, rounds=2 * NUM_REPLICAS),
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
  loads = np.zeros(NUM_EP)
  np.add.at(loads, dev.ravel(), np.asarray(counts).ravel())
  return loads


class ReplicaPlannerTest(parameterized.TestCase):

  def test_identity_route_table(self):
    table = np.asarray(mer.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS))
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
    identity = mer.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS)
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
        np.asarray(plan.route_table), np.asarray(mer.identity_route_table(NUM_EP, NUM_EXPERTS, NUM_REPLICAS))
    )
    self.assertTrue(np.all(np.asarray(plan.rep_home) < 0))

  @parameterized.parameters("greedy", "rounds")
  def test_balanced_counts_need_no_replicas(self, planner):
    counts = jnp.full((NUM_EP, NUM_EXPERTS), 16, jnp.int32)
    plan = PLANNERS[planner](counts, NUM_EP, NUM_REPLICAS, MAX_SEND, BOARD_SIZE)
    num_slots = LOCAL + NUM_REPLICAS
    after = _device_loads(counts, plan.route_table, num_slots)
    np.testing.assert_allclose(after, float(counts.sum()) / NUM_EP)


class ExpertReplicationConfigTest(absltest.TestCase):

  def _init(self, **kw):
    return pyconfig.initialize(
        [None, get_test_config_path()], run_name="moe_expert_replication_test", enable_checkpointing=False, **kw
    )

  def test_defaults_off(self):
    cfg = self._init()
    self.assertEqual(cfg.expert_replicas_per_device, 0)
    self.assertEqual(cfg.expert_replica_plan, "online")
    self.assertEqual(cfg.expert_replica_planner, "greedy")
    self.assertEqual(cfg.expert_replica_impl, "auto")
    self.assertEqual(cfg.expert_replica_move, "gather")
    self.assertFalse(cfg.expert_replica_identity)
    self.assertFalse(cfg.expert_replica_overhead_control)
    self.assertFalse(cfg.log_ra2a_required_rbf)
    self.assertIn(cfg.synthetic_routing, (None, "none"))  # YAML parses `none` as null.

  def test_flags_parse(self):
    cfg = self._init(
        expert_replicas_per_device=4,
        expert_replica_max_send=3,
        expert_replica_plan="lagged",
        expert_replica_planner="rounds",
        expert_replica_impl="concat",
        expert_replica_move="direct",
        expert_replica_overhead_control=True,
        log_ra2a_required_rbf=True,
        synthetic_routing="lognormal",
        synthetic_routing_sigmas="0.5,0.8",
        synthetic_routing_row_taus=1.2,
        synthetic_routing_row_len=768,
        synthetic_routing_pool_size=2,
    )
    self.assertEqual(cfg.expert_replicas_per_device, 4)
    self.assertEqual(cfg.expert_replica_max_send, 3)
    self.assertEqual(cfg.expert_replica_plan, "lagged")
    self.assertEqual(cfg.expert_replica_planner, "rounds")
    self.assertEqual(cfg.expert_replica_impl, "concat")
    self.assertEqual(cfg.expert_replica_move, "direct")
    self.assertTrue(cfg.expert_replica_overhead_control)
    self.assertTrue(cfg.log_ra2a_required_rbf)
    self.assertEqual(cfg.synthetic_routing, "lognormal")
    self.assertEqual(cfg.synthetic_routing_row_len, 768)
    self.assertEqual(cfg.synthetic_routing_pool_size, 2)

  def test_invalid_choices_rejected(self):
    for kw in (
        {"expert_replica_planner": "bogus"},
        {"expert_replica_impl": "bogus"},
        {"expert_replica_move": "bogus"},
        {"expert_replica_plan": "bogus"},
        {"synthetic_routing": "bogus"},
    ):
      with self.subTest(kw=kw):
        with self.assertRaises((pydantic.ValidationError, ValueError)):
          self._init(**kw)


if __name__ == "__main__":
  absltest.main()
