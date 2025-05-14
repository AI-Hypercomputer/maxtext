"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=bare-except, consider-using-generator
"""Logger that saves metrics to a local file, GCS and TensorBoard."""

import json
import os

import numpy as np

import jax

from MaxText import max_logging
from MaxText.utils import gcs_utils


def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {val: float(metrics["scalar"][val]) for val in metrics["scalar"]}
  metrics_dict["step"] = float(step)
  metrics_dict["run_name"] = run_name
  return metrics_dict


class MetricLogger:
  """
  Logger for saving metrics to a local file, GCS and TensorBoard.
  """

  def __init__(self, writer, config):
    self.buffered_step = None
    self.buffered_metrics = None
    self.writer = writer
    self.config = config

  def write_metrics(self, running_gcs_metrics, metrics, step, is_training=True):
    """Entry point for all metrics writing in Train's Main.

    To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
    onto the last metrics and step and only publish when we receive a new metrics and step.
    The logic is that this ensures that Jax is able to queues train_steps and we
    don't block when turning "lazy" Jax arrays into real Python numbers.
    """
    metrics_to_write, steps_to_write = None, None
    if is_training:
      if self.buffered_metrics is not None:
        if self.buffered_step is None:
          raise ValueError(f"When writing metrics, {self.buffered_step=} was none")
        metrics_to_write = self.buffered_metrics
        steps_to_write = self.buffered_step
        self.log_metrics(metrics_to_write, steps_to_write)
      self.buffered_metrics = metrics
      self.buffered_step = step
    else:
      metrics_to_write = metrics
      steps_to_write = step

    if metrics_to_write:
      if self.config.enable_tensorboard:
        self.write_metrics_to_tensorboard(metrics_to_write, steps_to_write, is_training)

      if self.config.metrics_file:
        self.write_metrics_locally(metrics_to_write, steps_to_write)

      if self.config.gcs_metrics and jax.process_index() == 0:
        running_gcs_metrics = self.write_metrics_for_gcs(metrics_to_write, steps_to_write, running_gcs_metrics, is_training)

  def log_metrics(self, metrics, step):
    """Logs metrics via max_logging"""
    max_logging.log(
        f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
        f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
        f"Tokens/s/device: {metrics['scalar']['perf/per_device_tokens_per_sec']:.3f}, "
        f"total_weights: {metrics['scalar']['learning/total_weights']}, "
        f"loss: {metrics['scalar']['learning/loss']:.3f}"
    )

  def write_metrics_locally(self, metrics, step):
    """Writes metrics locally for testing"""
    with open(self.config.metrics_file, "a", encoding="utf8") as local_metrics_file:
      if step == 0:
        local_metrics_file.truncate(0)

      metrics_dict = _prepare_metrics_for_json(metrics, step, self.config.run_name)
      local_metrics_file.write(str(json.dumps(metrics_dict)) + "\n")

  def write_metrics_for_gcs(self, metrics, step, running_metrics, is_training):
    """Writes metrics to gcs"""
    metrics_dict_step = _prepare_metrics_for_json(metrics, step, self.config.run_name)
    running_metrics.append(metrics_dict_step)
    if is_training and (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      start_step = (step // self.config.log_period) * self.config.log_period
      metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
      with open(metrics_filename, "wt", encoding="utf8") as metrics_for_gcs:
        for metrics_step in running_metrics:
          metrics_for_gcs.write(str(json.dumps(metrics_step)) + "\n")

      gcs_filename = os.path.join(self.config.metrics_dir, metrics_filename)
      max_logging.log(f"Moving file {metrics_filename} to GCS...")
      gcs_utils.upload_blob(gcs_filename, metrics_filename)
      max_logging.log(f"File {metrics_filename} moved successfully!")
      running_metrics = []  # reset running_metrics to empty list
    return running_metrics

  def write_metrics_to_tensorboard(self, metrics, step, is_training):
    """Writes metrics to TensorBoard"""
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        self.writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        self.writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    if is_training:
      full_log = step % self.config.log_period == 0

      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={self.config.tensorboard_dir}'")
        self.writer.flush()
