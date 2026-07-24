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

from concurrent.futures import Future, ThreadPoolExecutor
import queue
import threading
import traceback
from typing import Any

import jax

from maxtext.utils import max_logging


class TransportClosedError(RuntimeError):
  """Raised when a blocked transport operation is cancelled."""


class TransportProtocolError(RuntimeError):
  """Raised when a producer violates the per-channel FIFO protocol."""


class _BoundedChannel:
  """A cancelable FIFO whose capacity includes unpublished transfers."""

  _POLL_SECONDS = 0.1

  def __init__(self, capacity: int):
    self._queue = queue.Queue(maxsize=capacity)
    self._slots = threading.BoundedSemaphore(capacity)
    self._closed = threading.Event()

  def reserve(self):
    """Reserves capacity before the caller allocates a payload."""
    while not self._closed.is_set():
      if self._slots.acquire(timeout=self._POLL_SECONDS):  # pylint: disable=consider-using-with
        if self._closed.is_set():
          self._slots.release()
          break
        return
    raise TransportClosedError("Transport closed while waiting for payload capacity")

  def cancel_reservation(self):
    self._slots.release()

  def publish_reserved(self, item):
    """Publishes an item whose capacity was reserved by ``reserve``."""
    try:
      while not self._closed.is_set():
        try:
          self._queue.put(item, timeout=self._POLL_SECONDS)
          return
        except queue.Full:
          continue
      raise TransportClosedError("Transport closed while publishing a payload")
    except Exception:
      self.cancel_reservation()
      raise

  def send(self, item):
    self.reserve()
    self.publish_reserved(item)

  def recv(self):
    while True:
      if self._closed.is_set() and self._queue.empty():
        raise TransportClosedError("Transport closed while waiting for a payload")
      try:
        item = self._queue.get(timeout=self._POLL_SECONDS)
        self._slots.release()
        return item
      except queue.Empty:
        continue

  def close(self):
    self._closed.set()


class ThreadedTransportManager:
  """Manages in-memory communication between learner threads and the syncer thread."""

  def __init__(self, num_learners: int, max_pending_fragments: int = 1):
    if max_pending_fragments < 1:
      raise ValueError(f"max_pending_fragments must be positive, got {max_pending_fragments}")
    self.num_learners = num_learners

    # Each direction has one producer and consumes monotonically. Strict FIFO
    # avoids an out-of-order buffer silently defeating the memory bound.
    self._learner_to_syncer = [_BoundedChannel(max_pending_fragments) for _ in range(num_learners)]
    self._syncer_to_learner = [_BoundedChannel(max_pending_fragments) for _ in range(num_learners)]

  def reserve_to_syncer(self, learner_idx: int):
    self._learner_to_syncer[learner_idx].reserve()

  def cancel_to_syncer_reservation(self, learner_idx: int):
    self._learner_to_syncer[learner_idx].cancel_reservation()

  def publish_to_syncer(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    self._learner_to_syncer[learner_idx].publish_reserved((step, fragment_id, data))

  def send_to_syncer(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    """Learner sends data to the syncer."""
    self._learner_to_syncer[learner_idx].send((step, fragment_id, data))

  def recv_from_learner(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    """Syncer receives data from a specific learner. Blocks if not available."""
    rec_step, rec_frag, data = self._learner_to_syncer[learner_idx].recv()
    if rec_step != step or rec_frag != fragment_id:
      raise TransportProtocolError(f"Learner {learner_idx} sent ({rec_step}, {rec_frag}); expected ({step}, {fragment_id})")
    return data

  def reserve_to_learner(self, learner_idx: int):
    self._syncer_to_learner[learner_idx].reserve()

  def cancel_to_learner_reservation(self, learner_idx: int):
    self._syncer_to_learner[learner_idx].cancel_reservation()

  def publish_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    self._syncer_to_learner[learner_idx].publish_reserved((step, fragment_id, data))

  def send_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    """Syncer sends data to a specific learner."""
    self._syncer_to_learner[learner_idx].send((step, fragment_id, data))

  def recv_from_syncer(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    """Learner receives data from the syncer. Blocks if not available."""
    rec_step, rec_frag, data = self._syncer_to_learner[learner_idx].recv()
    if rec_step != step or rec_frag != fragment_id:
      raise TransportProtocolError(
          f"Syncer sent ({rec_step}, {rec_frag}); expected ({step}, {fragment_id}) for learner {learner_idx}"
      )
    return data

  def close(self):
    """Cancels blocked producers and consumers."""
    for channel in self._learner_to_syncer + self._syncer_to_learner:
      channel.close()


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
    self._send_futures: list[Future] = []

  def _raise_completed_send_errors(self):
    pending = []
    for future in self._send_futures:
      if future.done():
        future.result()
      else:
        pending.append(future)
    self._send_futures = pending

  def send_to_syncer_async(self, step: int, fragment_id: int, data: Any):
    """Asynchronously offloads TPU data to local CPU mesh and sends to syncer."""
    self._raise_completed_send_errors()

    # Reserve before device_put. Bounding only Queue.put would still allow an
    # unbounded number of large CPU transfers in the executor's work queue.
    self.manager.reserve_to_syncer(self.learner_idx)
    try:
      cpu_sharding = jax.tree_util.tree_map(
          lambda s: jax.sharding.NamedSharding(
              self.local_cpu_mesh,
              s.spec,
              memory_kind=s.memory_kind,
          ),
          jax.tree_util.tree_map(lambda x: x.sharding, data),
      )
      # The syncer donates received CPU payloads. Prevent fragment 0 from
      # aliasing whole, live learner parameter buffers.
      frag_cpu = jax.device_put(data, cpu_sharding, may_alias=False)
    except Exception:
      self.manager.cancel_to_syncer_reservation(self.learner_idx)
      raise

    def _send():
      reservation_consumed = False
      try:
        max_logging.log(f"Learner {self.learner_idx}: async send starting for step {step} frag {fragment_id}")
        jax.block_until_ready(frag_cpu)
        max_logging.log(f"Learner {self.learner_idx}: async send block_until_ready done")
        # publish_reserved owns (and releases on failure) the reservation once
        # it is called.
        reservation_consumed = True
        self.manager.publish_to_syncer(self.learner_idx, step, fragment_id, frag_cpu)
        max_logging.log(f"Learner {self.learner_idx}: async send sent to syncer")
      except Exception as e:
        if not reservation_consumed:
          self.manager.cancel_to_syncer_reservation(self.learner_idx)
        # The protocol cannot make progress without this payload. Cancel all
        # peer waits immediately instead of waiting for the learner thread to
        # reach its next send and poll this Future.
        self.manager.close()
        max_logging.error(f"Learner {self.learner_idx}: async send failed: {e}")
        max_logging.error(traceback.format_exc())
        raise e

    try:
      future = self._executor.submit(_send)
    except Exception:
      # ThreadPoolExecutor.submit can fail after shutdown. At that point no
      # worker owns either the reservation or the freshly allocated CPU tree.
      self.manager.cancel_to_syncer_reservation(self.learner_idx)
      raise
    self._send_futures.append(future)

  def send_to_syncer(self, step: int, fragment_id: int, data: Any):
    """Synchronously offloads TPU data to local CPU mesh and sends to syncer."""
    self._raise_completed_send_errors()
    self.manager.reserve_to_syncer(self.learner_idx)
    reservation_consumed = False
    try:
      cpu_sharding = jax.tree_util.tree_map(
          lambda s: jax.sharding.NamedSharding(
              self.local_cpu_mesh,
              s.spec,
              memory_kind=s.memory_kind,
          ),
          jax.tree_util.tree_map(lambda x: x.sharding, data),
      )
      frag_cpu = jax.device_put(data, cpu_sharding, may_alias=False)
      jax.block_until_ready(frag_cpu)
      reservation_consumed = True
      self.manager.publish_to_syncer(self.learner_idx, step, fragment_id, frag_cpu)
    finally:
      if not reservation_consumed:
        self.manager.cancel_to_syncer_reservation(self.learner_idx)

  def send_control_to_syncer(self, step: int, fragment_id: int, data: Any = None):
    """Sends a small host control message without a device transfer."""
    self._raise_completed_send_errors()
    self.manager.send_to_syncer(self.learner_idx, step, fragment_id, data)

  def recv_from_syncer(self, step: int, fragment_id: int) -> Any:
    return self.manager.recv_from_syncer(self.learner_idx, step, fragment_id)

  def close(self):
    """Waits for background sends and propagates their exceptions."""
    self._executor.shutdown(wait=True)
    for future in self._send_futures:
      future.result()
    self._send_futures.clear()


class SyncerTransport:
  """Wrapper for the syncer thread to communicate with learners."""

  def __init__(self, manager: ThreadedTransportManager):
    self.manager = manager

  def send_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    self.manager.send_to_learner(learner_idx, step, fragment_id, data)

  def reserve_to_learner(self, learner_idx: int):
    self.manager.reserve_to_learner(learner_idx)

  def cancel_to_learner_reservation(self, learner_idx: int):
    self.manager.cancel_to_learner_reservation(learner_idx)

  def publish_to_learner(self, learner_idx: int, step: int, fragment_id: int, data: Any):
    self.manager.publish_to_learner(learner_idx, step, fragment_id, data)

  def recv_from_learner(self, learner_idx: int, step: int, fragment_id: int) -> Any:
    return self.manager.recv_from_learner(learner_idx, step, fragment_id)

  def close(self):
    self.manager.close()
