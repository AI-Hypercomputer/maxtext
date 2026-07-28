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

"""Metric utilities for MaxRL."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

from absl import logging
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.utils import max_utils
import numpy as np


class MetricsLogger:
  """Logger and buffering service for metrics."""

  def __init__(self, config: pyconfig.HyperParameters):
    self._metrics_buffer: list[abstract_engine.MetricsBuffer] = []
    self.tb_writer = max_utils.initialize_summary_writer(
        config.tensorboard_dir, config.run_name, config.enable_tensorboard
    )

  def buffer_metrics(
      self,
      train_step: int,
      name: str,
      metric: jax.Array | float | int | None = None,
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ):
    """Buffers metrics for the given train step.

    Args:
      train_step: The train step for which to buffer metrics.
      name: The name of the metric.
      metric: The metric to buffer.
      aggregation_fn: Optional aggregation function to apply to the metric.
    """
    if not self._metrics_buffer or self._metrics_buffer[-1].id != train_step:
      new_buffer = abstract_engine.MetricsBuffer(id=train_step, mode="train")
      self._metrics_buffer.append(new_buffer)

    # Record the new metric in the buffer for the current step.
    self._record_metric(name, metric, aggregation_fn=aggregation_fn)

  def _record_metric(
      self,
      name: str,
      metric: abstract_engine.WeightedMetric | jax.Array | float | int,
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ) -> None:
    """Records a metric into the buffer, appending to JAX arrays.

    Args:
      name: The name of the metric.
      metric: The metric to record.
      aggregation_fn: The aggregation function to apply to the metric.
    """
    buffer = self._metrics_buffer[-1]

    if aggregation_fn is not None:
      buffer.aggregation_fns[name] = aggregation_fn

    if isinstance(metric, abstract_engine.WeightedMetric):
      if name not in buffer.weighted_metrics:
        buffer.weighted_metrics[name] = abstract_engine.WeightedMetric(
            unreduced_sum=jax.numpy.atleast_1d(metric.unreduced_sum),
            denominator=jax.numpy.atleast_1d(metric.denominator),
            eps=metric.eps,
            min_denom=metric.min_denom,
        )
      else:
        current_val = buffer.weighted_metrics[name]
        new_sum = jax.numpy.append(current_val.unreduced_sum, metric.unreduced_sum)
        new_denom = jax.numpy.append(current_val.denominator, metric.denominator)
        buffer.weighted_metrics[name] = dataclasses.replace(current_val, unreduced_sum=new_sum, denominator=new_denom)
    else:
      if name not in buffer.scalar_metrics:
        buffer.scalar_metrics[name] = jax.numpy.atleast_1d(jax.numpy.asarray(metric))
      else:
        buffer.scalar_metrics[name] = jax.numpy.append(buffer.scalar_metrics[name], metric)
    self._metrics_buffer[-1] = buffer

  def get_metrics(self, clear_cache: bool = True) -> list[abstract_engine.MetricsBuffer]:
    """Returns cached metrics and optionally clears the metrics cache.

    Args:
      clear_cache: Whether to reset cached metrics after retrieval.

    Returns:
      The accumulated on-device MetricsBuffer.
    """
    metrics_to_return = self._metrics_buffer
    if clear_cache:
      # Clear the metrics buffer.
      self._metrics_buffer = []
    return metrics_to_return

  def log_metrics(self, metrics: dict[str, Any]) -> None:
    """Log metrics.

    Args:
      metrics: Dictionary mapping metric names to reduced Python floats/numpy
        arrays.
    """
    for metric_name, value in metrics.items():
      try:
        agg_value = np.array(value)
        logging.info("Metric %s: %s", metric_name, agg_value)
      except Exception:  # pylint: disable=broad-except
        logging.warning(
            "Skipping metric %s: Could not convert to numpy array.",
            metric_name,
        )
        continue

  def write_metrics(self, metrics: dict[str, Any]) -> None:
    """Write metrics to TensorBoard.

    Args:
      metrics: Dictionary mapping metric names to reduced Python floats/numpy
        arrays.
    """
    self.log_metrics(metrics)
    self.write_metrics_to_tensorboard(metrics)

  def write_metrics_to_tensorboard(self, metrics: dict[str, Any]) -> None:
    """Write metrics to TensorBoard.

    Args:
      metrics: Dictionary mapping metric names to reduced Python floats/numpy
        arrays.
    """
    if jax.process_index() == 0:
      for metric_name, value in metrics.items():
        self.tb_writer.add_scalar(metric_name, value)
        self.tb_writer.flush()

  def process_metrics(self, metrics: abstract_engine.MetricsBuffer) -> None:
    """Reduction and processing pipeline for MetricsBuffer.

    Unpacks on-device weighted metrics by invoking safe compute(), extracts
    scalar metrics, and applies any host-side aggregation callbacks.

    Args:
      metrics: MetricsBuffer retrieved from device.
    """
    processed: dict[str, Any] = {}

    # Process weighted metrics via safe division
    for name, weighted_metric in metrics.weighted_metrics.items():
      reduced_val = weighted_metric.compute()
      host_val = np.asarray(reduced_val)
      if host_val.ndim == 0:
        host_val = float(host_val)
      if name in metrics.aggregation_fns:
        host_val = metrics.aggregation_fns[name](host_val)
      processed[name] = host_val

    # Process scalar metrics
    for name, scalar_val in metrics.scalar_metrics.items():
      host_val = np.asarray(scalar_val)
      if host_val.ndim == 0:
        host_val = float(host_val)
      if name in metrics.aggregation_fns:
        host_val = metrics.aggregation_fns[name](host_val)
      processed[name] = host_val

    logging.info("Logging metrics for step %s", metrics.id)
    self.write_metrics(processed)

  def flush_metrics_and_cleanup(self) -> list[abstract_engine.MetricsBuffer] | None:
    """Flushes metrics and cleans up the metrics buffer."""
    if self._metrics_buffer is None:
      return None
    self.process_metrics(self._metrics_buffer)
    self._metrics_buffer = None
    return None
