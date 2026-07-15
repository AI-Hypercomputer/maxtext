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

"""Tests for decomposed_transport.py."""

import threading
import time
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager, LearnerTransport, SyncerTransport


def test_threaded_transport_basic():
  manager = ThreadedTransportManager(num_learners=2)
  learner0 = LearnerTransport(manager, 0)
  learner1 = LearnerTransport(manager, 1)
  syncer = SyncerTransport(manager)

  # Test Learner -> Syncer
  learner0.send_to_syncer(step=0, fragment_id=0, data="L0_S0_F0")
  learner1.send_to_syncer(step=0, fragment_id=0, data="L1_S0_F0")

  assert syncer.recv_from_learner(0, step=0, fragment_id=0) == "L0_S0_F0"
  assert syncer.recv_from_learner(1, step=0, fragment_id=0) == "L1_S0_F0"

  # Test Syncer -> Learner
  syncer.send_to_learner(0, step=0, fragment_id=0, data="S_L0_S0_F0")
  syncer.send_to_learner(1, step=0, fragment_id=0, data="S_L1_S0_F0")

  assert learner0.recv_from_syncer(step=0, fragment_id=0) == "S_L0_S0_F0"
  assert learner1.recv_from_syncer(step=0, fragment_id=0) == "S_L1_S0_F0"


def test_threaded_transport_blocking():
  manager = ThreadedTransportManager(num_learners=1)
  learner = LearnerTransport(manager, 0)
  syncer = SyncerTransport(manager)

  received_data = []

  def reader_thread():
    data = learner.recv_from_syncer(step=5, fragment_id=0)
    received_data.append(data)

  t = threading.Thread(target=reader_thread)
  t.start()

  # Sleep to ensure reader thread is blocking
  time.sleep(0.1)
  assert not received_data

  # Send data
  syncer.send_to_learner(0, step=5, fragment_id=0, data="late_data")
  t.join(timeout=2)

  assert received_data == ["late_data"]


def test_threaded_transport_out_of_order():
  manager = ThreadedTransportManager(num_learners=1)
  learner = LearnerTransport(manager, 0)
  syncer = SyncerTransport(manager)

  # Send step 2 then step 1
  learner.send_to_syncer(step=2, fragment_id=0, data="step2")
  learner.send_to_syncer(step=1, fragment_id=0, data="step1")

  # Receive step 1 first
  assert syncer.recv_from_learner(0, step=1, fragment_id=0) == "step1"
  assert syncer.recv_from_learner(0, step=2, fragment_id=0) == "step2"
