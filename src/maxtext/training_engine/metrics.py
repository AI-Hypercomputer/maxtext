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

"""Metric utilities for MaxText training engine."""

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


_METRICS_TO_LOG = [
    "learning_rate",
    "loss",
    "gradient_norm",
    "step_time",
    "tflops",
]


class MetricsRecorder:
  """Synchronous frontend for buffering and aggregating step metrics on-device.

  Acts as the producer in the metrics pipeline, owned by `MaxTextTrainingEngine`.
  During forward-backward micro-steps (`fwd_bwd`) and weight updates (`update`),
  it buffers uncomputed JAX tensors into an on-device `MetricsBuffer` grouped by step ID.

  Once a step completes, the engine retrieves this buffer and enqueues it with the
  TPU computation via `InflightThrottler`, which later hands it to `MetricsLogger`
  for processing.
  """

  def __init__(self):
    self._metrics_buffer: list[abstract_engine.MetricsBuffer] = []

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

  def get_step_metrics(self, step: int) -> abstract_engine.MetricsBuffer | None:
    """Returns the latest metrics from the metrics buffer."""
    if not self._metrics_buffer:
      return None
    latest = self._metrics_buffer[-1]
    if latest.id == step:
      return latest
    return None

  def cleanup(self) -> None:
    """Cleans up the metrics recorder."""
    self._metrics_buffer = []


class MetricsLogger:
  """Asynchronous backend consumer for processing and logging completed metrics.

  Owned and invoked by `InflightThrottler` when a TPU step finishes executing on device.

  It receives the completed `MetricsBuffer` (produced earlier by `MetricsRecorder`),
  reduces the on-device tensors, converts them to host Python types, and writes the
  results to TensorBoard and console stdout.
  """

  def __init__(self, config: pyconfig.HyperParameters):
    """Initializes the metrics logger.

    Args:
      config: The training configuration.
    """

    self._tb_writer = None
    if getattr(config, "enable_tensorboard", False):
      self._tb_writer = max_utils.initialize_summary_writer(
          config.tensorboard_dir, config.run_name, config.enable_tensorboard
      )
    self._config = config

  def _log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
    """Logs the metrics to the console.

    Args:
      step: The train step for which to log the metrics.
      metrics: Dictionary mapping metric names to reduced Python floats/numpy
        arrays.
    """
    log_message = [f"Completed step: {step}"]
    log_message.extend(f"{k}: {metrics[k]:.3f}" for k in _METRICS_TO_LOG if k in metrics)

    logging.info(", ".join(log_message))

  def write_metrics(self, metrics: abstract_engine.MetricsBuffer) -> None:
    """Write metrics to the console and TensorBoard.

    Args:
      metrics: MetricsBuffer containing the metrics to write.
    """
    processed_metrics = self.process_metrics(metrics)
    self._log_metrics(metrics.id, processed_metrics)
    self._write_metrics_to_tensorboard(metrics.id, processed_metrics)

  def _write_metrics_to_tensorboard(self, step: int, metrics: dict[str, Any]) -> None:
    """Write metrics to TensorBoard.

    Args:
      step: The train step for which to write the metrics.
      metrics: Dictionary mapping metric names to reduced Python floats/numpy
        arrays.
    """
    if self._tb_writer is None:
      return

    if jax.process_index() == 0:
      for metric_name, value in metrics.items():
        self._tb_writer.add_scalar(metric_name, value, step)

    if step % self._config.log_period == 0:
      logging.info(
          "Flushing TensorBoard writer at step %d to %s",
          step,
          self._config.tensorboard_dir,
      )
      self._tb_writer.flush()

  def process_metrics(self, metrics: abstract_engine.MetricsBuffer) -> dict[str, Any]:
    """Reduction and processing pipeline for MetricsBuffer.

    Unpacks on-device weighted metrics by invoking safe compute(), extracts
    scalar metrics, and applies any host-side aggregation callbacks.

    Args:
      metrics: MetricsBuffer retrieved from device.
    Returns:
      A dictionary of metric names to processed metric values.
    """
    processed: dict[str, Any] = {}

    # Process weighted metrics via safe division
    for name, weighted_metric in metrics.weighted_metrics.items():
      if name not in _METRICS_TO_LOG:
        continue
      reduced_val = weighted_metric.compute()
      host_val = np.asarray(reduced_val)
      if name in metrics.aggregation_fns:
        host_val = metrics.aggregation_fns[name](host_val)
      elif host_val.size > 1:
        host_val = np.mean(host_val)
      if isinstance(host_val, (np.ndarray, jax.Array)):
        host_val = host_val.item() if host_val.size == 1 else float(np.mean(host_val))
      processed[name] = host_val

    # Process scalar metrics
    for name, scalar_val in metrics.scalar_metrics.items():
      if name not in _METRICS_TO_LOG:
        continue
      host_val = np.asarray(scalar_val)
      if name in metrics.aggregation_fns:
        host_val = metrics.aggregation_fns[name](host_val)
      elif host_val.size > 1:
        host_val = np.mean(host_val)
      if isinstance(host_val, (np.ndarray, jax.Array)):
        host_val = host_val.item() if host_val.size == 1 else float(np.mean(host_val))
      processed[name] = host_val
    return processed

  def cleanup(self) -> None:
    """This is a terminal operation that closes the writer objects.

    Once called, the logger instance should not be used to add or write more
    metrics as the underlying writer objects (e.g., TensorBoard SummaryWriter)
    will be closed.
    """
    if self._tb_writer is not None:
      max_utils.close_summary_writer(self._tb_writer)
