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

"""CPU tests for the expert-replication config flags.

The ragged all-to-all weight move is not implemented on CPU; the planners and the weight-move index
math are tested in moe_replica_planner_test.py and moe_replica_weights_test.py.
"""

from absl.testing import absltest
import pydantic

from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path


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
    )
    self.assertEqual(cfg.expert_replicas_per_device, 4)
    self.assertEqual(cfg.expert_replica_max_send, 3)
    self.assertEqual(cfg.expert_replica_plan, "lagged")
    self.assertEqual(cfg.expert_replica_planner, "rounds")
    self.assertEqual(cfg.expert_replica_impl, "concat")
    self.assertEqual(cfg.expert_replica_move, "direct")
    self.assertTrue(cfg.expert_replica_overhead_control)
    self.assertTrue(cfg.log_ra2a_required_rbf)

  def test_invalid_choices_rejected(self):
    for kw in (
        {"expert_replica_planner": "bogus"},
        {"expert_replica_impl": "bogus"},
        {"expert_replica_move": "bogus"},
        {"expert_replica_plan": "bogus"},
    ):
      with self.subTest(kw=kw):
        with self.assertRaises((pydantic.ValidationError, ValueError)):
          self._init(**kw)


if __name__ == "__main__":
  absltest.main()
