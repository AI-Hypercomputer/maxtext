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

"""Unit tests for threaded DiLoCo components."""

import sys
import unittest
import threading
import time
from maxtext.configs import pyconfig
from maxtext.trainers.diloco.threaded_diloco import make_learner_config
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager

class ThreadedDilocoUnitTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Need to add src to path if not already there, but maxtext imports usually assume src is in path.
    # We will initialize config with base.yml
    self.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="test",
        enable_diloco=True,
        enable_streaming_diloco=True,
        num_diloco_replicas=2,
    )

  def test_make_learner_config(self):
    learner_config = make_learner_config(self.config, learner_idx=1, num_learners=2)

    # Check that diloco is removed from mesh_axes
    self.assertNotIn("diloco", learner_config.mesh_axes)

    # Check logical_axis_rules
    for _, physical_axes in learner_config.logical_axis_rules:
      if isinstance(physical_axes, str):
        self.assertNotEqual(physical_axes, "diloco")
      elif isinstance(physical_axes, (list, tuple)):
        self.assertNotIn("diloco", physical_axes)

    # Check other flags
    self.assertTrue(learner_config.enable_local_data_loading)
    self.assertEqual(learner_config.learner_idx, 1)
    self.assertEqual(learner_config.num_learners, 2)
    self.assertFalse(learner_config.enable_streaming_diloco)
    self.assertFalse(learner_config.enable_diloco)

  def test_transport_manager_basic(self):
    manager = ThreadedTransportManager(num_learners=2)

    # Test learner to syncer
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="l0_s1_f1")
    manager.send_to_syncer(learner_idx=1, step=1, fragment_id=1, data="l1_s1_f1")

    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1), "l0_s1_f1")
    self.assertEqual(manager.recv_from_learner(learner_idx=1, step=1, fragment_id=1), "l1_s1_f1")

    # Test syncer to learner
    manager.send_to_learner(learner_idx=0, step=1, fragment_id=1, data="s_l0_s1_f1")
    manager.send_to_learner(learner_idx=1, step=1, fragment_id=1, data="s_l1_s1_f1")

    self.assertEqual(manager.recv_from_syncer(learner_idx=0, step=1, fragment_id=1), "s_l0_s1_f1")
    self.assertEqual(manager.recv_from_syncer(learner_idx=1, step=1, fragment_id=1), "s_l1_s1_f1")

  def test_transport_manager_out_of_order(self):
    manager = ThreadedTransportManager(num_learners=1)

    # Send out of order
    manager.send_to_syncer(learner_idx=0, step=2, fragment_id=1, data="step2")
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="step1")

    # Receive in order
    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1), "step1")
    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=2, fragment_id=1), "step2")

  def test_transport_manager_blocking(self):
    manager = ThreadedTransportManager(num_learners=1)
    results = {}

    def worker():
      results['data'] = manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1)

    t = threading.Thread(target=worker)
    t.start()

    # Sleep to ensure worker is blocked
    time.sleep(0.1)
    self.assertTrue(t.is_alive())
    self.assertNotIn('data', results)

    # Send data
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="blocked_data")
    t.join(timeout=1.0)

    self.assertFalse(t.is_alive())
    self.assertEqual(results['data'], "blocked_data")

if __name__ == "__main__":
  unittest.main()
