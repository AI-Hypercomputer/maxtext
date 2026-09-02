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
import os
from typing import Any

from absl import logging
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from maxtext.utils import max_logging, max_utils, maxtext_utils
import numpy as np


_METRICS_TO_LOG = [
    "learning_rate",
    "loss",
    "perplexity",
    "total_weights",
    "gradient_norm",
    "step_skipped",
    "step_time",
    "tflops",
]

# How many completed step buffers stay resident before the oldest are evicted.
#
# Every buffer holds live device arrays -- one leaf per scalar metric, plus one per aux
# metric when the engine runs with `has_aux`, which for MoE/MTP configs is tens of leaves --
# so nothing in the list is free. Nothing on the engine's own step path removes an entry
# either: `get_step_metrics` hands out the newest by reference, and only
# `get_metrics_history(clear_cache=True)` and `cleanup()` clear. A driver that reads the
# engine's own TensorBoard output rather than calling `get_metrics()` therefore never
# clears, and both HBM and -- because `save_checkpoint` serializes the whole retained
# history -- checkpoint size and save latency grow linearly in steps.
#
# Tunix v2 keeps exactly one prior step (`_prev_buffered_train_metrics` in
# `peft_trainer_v2.py`). This keeps a window instead so batched readers of
# `get_metrics_history` still work, while making the footprint constant in step count.
_DEFAULT_MAX_BUFFERED_STEPS = 128


class MetricsRecorder:
  """Synchronous frontend for buffering and aggregating step metrics on-device.

  Acts as the producer in the metrics pipeline, owned by `MaxTextTrainingEngine`.
  During forward-backward micro-steps (`fwd_bwd`) and weight updates (`update`),
  it buffers uncomputed JAX tensors into an on-device `MetricsBuffer` grouped by step ID.

  Once a step completes, the engine retrieves this buffer and enqueues it with the
  TPU computation via `InflightThrottler`, which later hands it to `MetricsLogger`
  for processing.
  """

  def __init__(self, max_buffered_steps: int = _DEFAULT_MAX_BUFFERED_STEPS):
    """Initializes the recorder.

    Args:
      max_buffered_steps: How many completed step buffers to retain; older ones are evicted
        as new steps start. Pass 0 or a negative value to retain everything, which is only
        safe when the driver drains the history itself.
    """
    self._metrics_buffer: list[abstract_engine.MetricsBuffer] = []
    self._max_buffered_steps = max_buffered_steps
    self._dropped_buffer_count = 0

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
      self._evict_old_buffers()

    # Record the new metric in the buffer for the current step.
    self._record_metric(name, metric, aggregation_fn=aggregation_fn)

  def _evict_old_buffers(self) -> None:
    """Drops the oldest step buffers once the retention window is full.

    Only ever runs when a *new* step starts, so the buffer the current step is writing into
    and the one `get_step_metrics` is about to hand the throttler are never the ones dropped.
    """
    if self._max_buffered_steps <= 0 or len(self._metrics_buffer) <= self._max_buffered_steps:
      return
    num_dropped = len(self._metrics_buffer) - self._max_buffered_steps
    oldest_dropped_id = self._metrics_buffer[0].id
    del self._metrics_buffer[:num_dropped]
    self._dropped_buffer_count += num_dropped
    # Dropping metrics silently is the pattern that produced the fabricated 0.0 in the parity
    # harness, so say so -- but once per window, not once per step.
    logging.log_every_n(
        logging.WARNING,
        "Metrics history is full at %d step(s); evicting buffers from step %s onwards "
        "(%d dropped so far). Drain it with MetricsRecorder.get_metrics_history() or the "
        "engine's get_metrics() if you need every step.",
        self._max_buffered_steps,
        self._max_buffered_steps,
        oldest_dropped_id,
        self._dropped_buffer_count,
    )

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

  def get_metrics_history(self, clear_cache: bool = True) -> list[abstract_engine.MetricsBuffer]:
    """Returns every cached step buffer and optionally clears the metrics cache.

    The engine's own `get_metrics` returns only the most recent buffer, per the trainer
    contract. This is the accessor that keeps the history reachable -- the last
    `max_buffered_steps` of it; see `_DEFAULT_MAX_BUFFERED_STEPS` for why it is a window.

    Args:
      clear_cache: Whether to reset cached metrics after retrieval.

    Returns:
      One on-device MetricsBuffer per retained train step, oldest first.
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
    if config.enable_tensorboard:
      self._tb_writer = max_utils.initialize_summary_writer(
          config.tensorboard_dir, config.run_name, config.enable_tensorboard
      )
    self._config = config

  def write_setup_info_to_tensorboard(self, params: Any) -> None:
    """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
    num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
    if self._tb_writer is None:
      return
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self._tb_writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.getenv("LIBTPU_INIT_ARGS", ""), self._tb_writer)
    maxtext_utils.add_config_to_summary_writer(self._config, self._tb_writer)

  def _log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
    """Logs the metrics to the console.

    Args:
      step: The train step for which to log the metrics.
      metrics: Dictionary mapping metric names to reduced Python floats/numpy
        arrays.
    """
    log_message = [f"Completed step: {step}"]
    for k in _METRICS_TO_LOG:
      if k in metrics:
        if k == "learning_rate":
          log_message.append(f"{k}: {metrics[k]:.3e}")
        else:
          log_message.append(f"{k}: {metrics[k]:.3f}")

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

    if "loss" in processed:
      processed["perplexity"] = float(np.exp(np.asarray(processed["loss"], dtype=np.float64)))
    return processed

  def cleanup(self) -> None:
    """This is a terminal operation that closes the writer objects.

    Once called, the logger instance should not be used to add or write more
    metrics as the underlying writer objects (e.g., TensorBoard SummaryWriter)
    will be closed.
    """
    if self._tb_writer is not None:
      max_utils.close_summary_writer(self._tb_writer)
