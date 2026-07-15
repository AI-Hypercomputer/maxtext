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

"""Thread-safe transport layer for non-SPMD multi-threaded DiLoCo."""

import queue
from typing import Any


class ThreadedTransportManager:
  """Manages in-memory communication between learner threads and the syncer thread."""

  def __init__(self, num_learners: int):
    self.num_learners = num_learners

    # Thread-safe FIFO queues for each learner.
    # Stores tuples of (step, fragment_id, data).
    self._learner_to_syncer_queues = [queue.Queue() for _ in range(num_learners)]
    self._syncer_to_learner_queues = [queue.Queue() for _ in range(num_learners)]

    # Local buffers for out-of-order storage.
    # Since only the receiving thread (syncer for learner_to_syncer, and respective
    # learner for syncer_to_learner) accesses these buffers, we do not need locks.
    self._learner_to_syncer_buffers = [{} for _ in range(num_learners)]
    self._syncer_to_learner_buffers = [{} for _ in range(num_learners)]

  def send_to_syncer(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    """Learner sends data to the syncer."""
    self._learner_to_syncer_queues[learner_idx].put((step, fragment_id, data))

  def recv_from_learner(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    """Syncer receives data from a specific learner. Blocks if not available."""
    key = (step, fragment_id)
    buffer = self._learner_to_syncer_buffers[learner_idx]
    if key in buffer:
      return buffer.pop(key)

    while True:
      rec_step, rec_frag, data = self._learner_to_syncer_queues[learner_idx].get()
      if rec_step == step and rec_frag == fragment_id:
        return data
      buffer[(rec_step, rec_frag)] = data

  def send_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    """Syncer sends data to a specific learner."""
    self._syncer_to_learner_queues[learner_idx].put((step, fragment_id, data))

  def recv_from_syncer(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    """Learner receives data from the syncer. Blocks if not available."""
    key = (step, fragment_id)
    buffer = self._syncer_to_learner_buffers[learner_idx]
    if key in buffer:
      return buffer.pop(key)

    while True:
      rec_step, rec_frag, data = self._syncer_to_learner_queues[learner_idx].get()
      if rec_step == step and rec_frag == fragment_id:
        return data
      buffer[(rec_step, rec_frag)] = data


class LearnerTransport:
  """Wrapper for learner threads to communicate with the syncer."""

  def __init__(self, manager: ThreadedTransportManager, learner_idx: int):
    self.manager = manager
    self.learner_idx = learner_idx

  def send_to_syncer(self, step: int, fragment_id: int, data: Any):
    self.manager.send_to_syncer(self.learner_idx, step, fragment_id, data)

  def recv_from_syncer(self, step: int, fragment_id: int) -> Any:
    return self.manager.recv_from_syncer(self.learner_idx, step, fragment_id)


class SyncerTransport:
  """Wrapper for the syncer thread to communicate with learners."""

  def __init__(self, manager: ThreadedTransportManager):
    self.manager = manager

  def send_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    self.manager.send_to_learner(learner_idx, step, fragment_id, data)

  def recv_from_learner(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    return self.manager.recv_from_learner(learner_idx, step, fragment_id)
