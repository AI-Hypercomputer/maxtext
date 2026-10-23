# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process message passing between threaded DiLoCo learners and the syncer."""

import queue
import threading
import time
import traceback
from typing import Any

from maxtext.utils import max_logging

# Longest a waiting thread goes without checking for an abort.
_POLL_SECONDS = 1.0

# Every message is keyed by (step, fragment index).
MessageKey = tuple[int, int]


class TransportAborted(RuntimeError):
  """Raised in every waiting thread once any participant has aborted the transport.

  Its `__cause__` is the exception the first aborting participant passed to `ThreadedTransport.abort`, if any.
  """


class Mailbox:
  """Single-consumer queue that delivers messages by key and buffers ones that arrive early."""

  def __init__(self, transport: "ThreadedTransport", timeout_seconds: float):
    self._queue: queue.Queue[tuple[MessageKey, Any]] = queue.Queue()
    self._early: dict[MessageKey, Any] = {}
    self._transport = transport
    self._timeout_seconds = timeout_seconds

  def put(self, key: MessageKey, payload: Any) -> None:
    self._queue.put((key, payload))

  def get_next(self, deadline: float | None = None) -> tuple[MessageKey, Any]:
    """Returns the oldest `(key, payload)`.

    Args:
      deadline: `time.monotonic()` value to wait until; defaults to now plus the transport timeout.

    Raises:
      TransportAborted: The transport was aborted before a message arrived.
      TimeoutError: No message arrived before the deadline.
    """
    if deadline is None:
      deadline = time.monotonic() + self._timeout_seconds
    while True:
      self._transport.raise_if_aborted("while waiting for a message")
      remaining = deadline - time.monotonic()
      try:
        return self._queue.get(timeout=max(0.0, min(_POLL_SECONDS, remaining)))
      except queue.Empty:
        if time.monotonic() >= deadline:
          raise TimeoutError(f"No message within {self._timeout_seconds} s.") from None

  def get(self, key: MessageKey) -> Any:
    """Returns the payload for `key`, buffering messages with other keys until they are asked for.

    The transport timeout bounds the whole wait, not the gap between two messages.
    """
    deadline = time.monotonic() + self._timeout_seconds
    while key not in self._early:
      k, payload = self.get_next(deadline)
      self._early[k] = payload
    return self._early.pop(key)


class ThreadedTransport:
  """One mailbox per direction per learner, plus a shared abort flag that records the root cause.

  `to_syncer[i]` is consumed by the syncer, `to_learner[i]` by learner `i`.
  """

  def __init__(self, num_learners: int, timeout_seconds: float):
    self._abort_event = threading.Event()
    self._abort_lock = threading.Lock()
    self._cause: BaseException | None = None
    self.to_syncer = [Mailbox(self, timeout_seconds) for _ in range(num_learners)]
    self.to_learner = [Mailbox(self, timeout_seconds) for _ in range(num_learners)]

  def abort(self, cause: BaseException | None = None) -> None:
    """Makes every current and future wait raise `TransportAborted`.

    Args:
      cause: The failure that made the caller abort. Only the first abort's cause is kept and logged; it is the root
        cause, and every later failure is a consequence of the abort.
    """
    with self._abort_lock:
      if self._abort_event.is_set():
        return
      self._cause = cause
      self._abort_event.set()
    if cause is not None:
      details = "".join(traceback.format_exception(cause)).rstrip()
      max_logging.log(f"Threaded DiLoCo: stopping every thread because of {type(cause).__name__}: {cause}\n{details}")

  @property
  def aborted(self) -> bool:
    return self._abort_event.is_set()

  @property
  def cause(self) -> BaseException | None:
    """The cause passed to the first `abort` call, or None."""
    return self._cause

  def raise_if_aborted(self, context: str) -> None:
    """Raises `TransportAborted`, chained to the root cause, if the transport has been aborted."""
    if not self._abort_event.is_set():
      return
    cause = self._cause
    suffix = f": {type(cause).__name__}: {cause}" if cause is not None else "."
    raise TransportAborted(f"Transport aborted by another threaded DiLoCo participant {context}{suffix}") from cause
