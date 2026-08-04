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
    max_inflight = getattr(config, "max_inflight_computations", 2)
    self._inflight_queue = queue.Queue[Any](maxsize=max_inflight)
    self._metrics_logger = metrics_module.MetricsLogger(config=config)

  def add_computation(self, computation: Any, metrics: abstract_engine.MetricsBuffer | None) -> None:
    """Adds an active on-device computation to the queue."""
    self._inflight_queue.put((jax.tree.leaves(computation), metrics))

  def wait_for_next(self) -> None:
    """If the limit is reached, wait for the next computation to finish."""
    if self._inflight_queue.full():
      computation, metrics = self._inflight_queue.get()
      jax.block_until_ready(computation)
      # Write metrics for the completed computation.
      if metrics is not None:
        self._metrics_logger.write_metrics(metrics)

  def wait_for_all(self) -> None:
    """Wait for all inflight computations to finish and log their metrics."""
    while not self._inflight_queue.empty():
      computation, metrics = self._inflight_queue.get()
      jax.block_until_ready(computation)
      # Write metrics for the completed computation.
      if metrics is not None:
        self._metrics_logger.write_metrics(metrics)

  def cleanup(self) -> None:
    """Closes the underlying metrics logger and releases resources."""
    self.wait_for_all()
    self._metrics_logger.cleanup()
