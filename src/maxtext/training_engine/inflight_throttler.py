# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inflight computation throttler."""

import queue
from typing import Any

import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import metrics as metrics_module


class InflightThrottler:
  """Rate limits the number of inflight computations on TPU."""

  def __init__(self, config: pyconfig.HyperParameters):
    """Initializes the inflight throttler.

    Args:
      config: The training configuration.
    """
    self._inflight_queue = queue.Queue[Any](maxsize=config.max_inflight_computations)
    self._metrics_logger = metrics_module.MetricsLogger(config=config)
    # Popped by `wait_for_next` but not yet written; see `_flush_pending_metrics`.
    self._pending_metrics: abstract_engine.MetricsBuffer | None = None

  def add_computation(self, computation: Any, metrics: abstract_engine.MetricsBuffer | None) -> None:
    """Adds an active on-device computation to the queue."""
    self._inflight_queue.put((jax.tree.leaves(computation), metrics))
    # The caller has just dispatched, so the device has work queued behind this point and the
    # blocking read inside `write_metrics` overlaps it instead of running against an idle
    # device. This is the whole reason the write is deferred rather than done in place.
    self._flush_pending_metrics()

  def _flush_pending_metrics(self) -> None:
    """Writes the buffer stashed by the last `wait_for_next`, if any."""
    if self._pending_metrics is None:
      return
    metrics, self._pending_metrics = self._pending_metrics, None
    self._metrics_logger.write_metrics(metrics)

  def wait_for_next(self) -> None:
    """If the limit is reached, wait for the next computation to finish.

    Blocks, but does not log. `write_metrics` reduces each `WeightedMetric` on device and then
    pulls the result to host with `np.asarray`, and those reduction ops are dispatched behind
    whatever is already in the device's queue -- so doing it here, before the caller dispatches
    the step it just made room for, stalls the host on the full backlog with nothing new
    running. Deferring to the following `add_computation` costs one dispatch of staleness in
    the log and hands the same work a busy device to hide behind. Metrics carry their own step
    id (`MetricsBuffer.id`), so nothing downstream can tell the difference.
    """
    if self._inflight_queue.full():
      computation, metrics = self._inflight_queue.get()
      jax.block_until_ready(computation)
      if metrics is not None:
        # Never hold two: the engine only attaches metrics to one of its two computations per
        # step, but a caller that attached them to both would otherwise silently lose a buffer.
        self._flush_pending_metrics()
        self._pending_metrics = metrics

  def wait_for_all(self) -> None:
    """Wait for all inflight computations to finish and log their metrics."""
    while not self._inflight_queue.empty():
      computation, metrics = self._inflight_queue.get()
      jax.block_until_ready(computation)
      # Write metrics for the completed computation.
      if metrics is not None:
        self._flush_pending_metrics()
        self._pending_metrics = metrics
    # Draining is the one place that must not leave a write outstanding: callers use it to
    # reach a quiescent state before checkpointing or shutting down.
    self._flush_pending_metrics()

  def cleanup(self) -> None:
    """Closes the underlying metrics logger and releases resources."""
    self.wait_for_all()
    self._metrics_logger.cleanup()
