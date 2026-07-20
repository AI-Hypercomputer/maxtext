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
import threading
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import traceback
import jax
import jax.numpy as jnp
from jax.experimental import colocated_python
from maxtext.utils import max_logging


_JAX_DISPATCH_LOCK = threading.Lock()


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

  def __init__(
      self,
      manager: ThreadedTransportManager,
      learner_idx: int,
      local_cpu_mesh: jax.sharding.Mesh,
  ):
    self.manager = manager
    self.learner_idx = learner_idx
    self.local_cpu_mesh = local_cpu_mesh
    self._executor = ThreadPoolExecutor(max_workers=1)

  def send_to_syncer_async(self, step: int, fragment_id: int, data: Any):
    """Asynchronously offloads TPU data to local CPU mesh and sends to syncer."""
    # 1. Asynchronously offload to CPU colocated mesh (non-blocking on main thread)
    cpu_sharding = jax.tree_util.tree_map(
        lambda s: jax.sharding.NamedSharding(self.local_cpu_mesh, s.spec),
        jax.tree_util.tree_map(lambda x: x.sharding, data),
    )
    # concatenate_by_mesh_axis always donates its inputs.  Force this transfer
    # to own its buffers so donation in the syncer cannot invalidate learner
    # state when source and destination happen to share a backend (notably in
    # multi-device CPU tests, where the "colocated CPU" is the learner CPU).
    # Dispatching device_put concurrently from Python threads is unsafe on the
    # CPU PJRT backend.  Only dispatch is serialized; completion remains async.
    with _JAX_DISPATCH_LOCK:
      frag_cpu = jax.device_put(data, cpu_sharding, may_alias=False)

    # 2. Block and send in the background executor thread
    def _send():
      try:
        max_logging.log(f"Learner {self.learner_idx}: async send starting for step {step} frag {fragment_id}")
        jax.block_until_ready(frag_cpu)
        max_logging.log(f"Learner {self.learner_idx}: async send block_until_ready done")
        self.manager.send_to_syncer(self.learner_idx, step, fragment_id, frag_cpu)
        max_logging.log(f"Learner {self.learner_idx}: async send sent to syncer")
      except Exception as e:
        max_logging.error(f"Learner {self.learner_idx}: async send failed: {e}")
        max_logging.error(traceback.format_exc())
        raise e

    self._executor.submit(_send)

  def send_to_syncer(self, step: int, fragment_id: int, data: Any):
    """Synchronously offloads TPU data to local CPU mesh and sends to syncer."""
    cpu_sharding = jax.tree_util.tree_map(
        lambda s: jax.sharding.NamedSharding(self.local_cpu_mesh, s.spec),
        jax.tree_util.tree_map(lambda x: x.sharding, data),
    )
    with _JAX_DISPATCH_LOCK:
      frag_cpu = jax.device_put(data, cpu_sharding, may_alias=False)
    jax.block_until_ready(frag_cpu)
    self.manager.send_to_syncer(self.learner_idx, step, fragment_id, frag_cpu)

  def recv_from_syncer(self, step: int, fragment_id: int) -> Any:
    return self.manager.recv_from_syncer(self.learner_idx, step, fragment_id)

  def close(self):
    """Shutdown background thread executor."""
    self._executor.shutdown(wait=True)


class SyncerTransport:
  """Wrapper for the syncer thread to communicate with learners."""

  def __init__(self, manager: ThreadedTransportManager):
    self.manager = manager

  def send_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    self.manager.send_to_learner(learner_idx, step, fragment_id, data)

  def recv_from_learner(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    return self.manager.recv_from_learner(learner_idx, step, fragment_id)
