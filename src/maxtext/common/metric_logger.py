# Copyright 2023–2025 Google LLC
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

# pylint: disable=bare-except, consider-using-generator
# pytype: disable=attribute-error
"""Logger that saves metrics to a local file, GCS and TensorBoard."""

import json
import os
import queue
import enum
import wandb
import socket

import numpy as np

import jax

import google_cloud_mldiagnostics as mldiag

from MaxText.globals import EPS
from maxtext.common.gcp_workload_monitor import GCPWorkloadMonitor
from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from collections import defaultdict

# Mapping MaxText metrics to managed profiler metrics
_METRICS_TO_MANAGED = {
    "learning/current_learning_rate": "learning_rate",
    "learning/loss": "loss",
    "learning/grad_norm": "gradient_norm",
    "learning/total_weights": "total_weights",
    "perf/step_time_seconds": "step_time",
    "perf/per_device_tokens_per_sec": "throughput",
    "perf/per_device_tflops_per_sec": "tflops",
    # There are no mappings to the following metrics yet:
    # "latency", "mfu"
}


def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {val: float(metrics["scalar"][val]) for val in metrics["scalar"]}
  metrics_dict["step"] = float(step)
  metrics_dict["run_name"] = run_name
  return metrics_dict


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs["intermediates"]["decoder"]["decoder"]

    for layer_num in range(config.num_decoder_layers):
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = metrics_dict["activation_fraction_zero"][
          0
      ][layer_num]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs["intermediates"]["decoder"][f"layers_{layer_num}"]
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = layer["activation_fraction_zero"][0]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = layer["activation_mean"][0]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = layer["activation_stdev"][0]


class MetadataKey(enum.Enum):
  PER_DEVICE_TFLOPS = "per_device_tflops"
  PER_DEVICE_TOKENS = "per_device_tokens"


class MetricLogger:
  """
  Logger for saving metrics to a local file, GCS and TensorBoard.
  """

  def __init__(self, config, learning_rate_schedule):
    self.writer = max_utils.initialize_summary_writer(config.tensorboard_dir, config.run_name)
    self.config = config
    self.metadata = {}
    self.running_gcs_metrics = [] if config.gcs_metrics else None
    self.performance_metric_queue = self.get_performance_metric_queue(config)
    self.learning_rate_schedule = learning_rate_schedule
    self.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    self.buffered_train_metrics = None
    
    if self.config.managed_mldiagnostics:
      ManagedMLDiagnostics(config)  # Initialize the MLRun instance.
      
    if self.config.enable_wandb: 
      wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_name,
        resume="allow",
      ) # Initialize wandb logger.

  def reset_eval_metrics(self):
    """Resets the cumulative metrics dictionary for a new evaluation run."""
    self.cumulative_eval_metrics = {"scalar": defaultdict(float)}

  def write_metrics(self, metrics, step, is_training=True):
    """Entry point for all metrics writing in Train's Main."""
    if metrics:
      self.log_metrics(metrics, step, is_training)

      if self.config.enable_tensorboard:
        self.write_metrics_to_tensorboard(metrics, step, is_training)

      if self.config.metrics_file:
        self.write_metrics_locally(metrics, step)

      if self.config.gcs_metrics and jax.process_index() == 0:
        self.write_metrics_for_gcs(metrics, step, is_training)

      if self.config.managed_mldiagnostics:
        self.write_metrics_to_managed_mldiagnostics(metrics, step)
        
      if self.config.enable_wandb and jax.process_index() == 0:
        self.write_metrics_to_wandb(metrics, step)

  def log_metrics(self, metrics, step, is_training):
    """Logs metrics via max_logging."""
    if is_training:
      self._log_training_metrics(metrics, step)
    else:
      self._log_eval_metrics(metrics, step)

  def _log_training_metrics(self, metrics, step):
    """Handles training-specific metric logging."""
    # Skip logging if in profiler activation/deactivation steps
    # TODO(b/456828037): Switch to subprocess profiling to avoid timing artifacts at boundary steps.
    scalars = metrics["scalar"]
    loss = scalars["learning/loss"]
    is_rampup = step < self.config.rampup_end_step
    is_metric_hidden_step = self.config.hide_profiler_step_metric and self._is_profiler_boundary_step(step)

    # Start building the log parts
    log_parts = []
    if is_rampup:
      log_parts.append("[Rampup Batch Size Phase]")

    if is_metric_hidden_step:
      log_parts.append(
          f"completed profiler activation/deactivation step: {step}",
      )
    else:
      log_parts.extend(
          [
              f"completed step: {step}",
              f"seconds: {scalars['perf/step_time_seconds']:.3f}",
          ]
      )

    # Add performance metrics only if strictly NOT in rampup phase
    # TODO(b/452468482): Enable performance metric (TFLOPs, Tokens/s) tracking during batch size rampup.
    if not is_rampup and not is_metric_hidden_step:
      log_parts.extend(
          [
              f"TFLOP/s/device: {scalars['perf/per_device_tflops_per_sec']:.3f}",
              f"Tokens/s/device: {scalars['perf/per_device_tokens_per_sec']:.3f}",
          ]
      )

    log_parts.extend(
        [
            f"total_weights: {scalars['learning/total_weights']}",
            f"loss: {loss:.3f}",
        ]
    )

    if self.config.mtp_num_layers > 0:
      mtp_loss = scalars.get("learning/mtp_loss", 0.0)
      log_parts.append(f"main_model_loss: {loss - mtp_loss:.3f}")
      log_parts.append(f"mtp_loss: {mtp_loss:.3f}")

    max_logging.log(", ".join(log_parts))

  def _log_eval_metrics(self, metrics, step):
    """Handles evaluation-specific metric logging."""
    scalars = metrics["scalar"]
    log_parts = [
        f"eval metrics after step: {step}",
        f"loss={scalars['eval/avg_loss']:.3f}",
        f"total_weights={scalars['eval/total_weights']}",
    ]

    if self.config.mtp_num_layers > 0:
      log_parts.extend(
          [
              f"avg_mtp_loss={scalars['eval/avg_mtp_loss']:.3f}",
              f"avg_mtp_acceptance_rate={scalars['eval/avg_mtp_acceptance_rate_percent']:.2f}%",
          ]
      )

    max_logging.log(", ".join(log_parts))

  def _is_profiler_boundary_step(self, step):
    """Determines if the current step is a profiler start/stop boundary that should be hidden."""
    if len(self.config.profiler) == 0:
      return False
    skip_steps = self.config.skip_first_n_steps_for_profiler
    profiler_steps = self.config.profiler_steps
    # Steps immediately before/at start, and at/immediately after end of profiling
    boundary_steps = {
        skip_steps,
        skip_steps + 1,
        skip_steps + profiler_steps,
        skip_steps + profiler_steps + 1,
    }
    return step in boundary_steps

  def write_metrics_locally(self, metrics, step):
    """Writes metrics locally for testing."""
    with open(self.config.metrics_file, "a", encoding="utf8") as local_metrics_file:
      if step == 0:
        local_metrics_file.truncate(0)

      metrics_dict = _prepare_metrics_for_json(metrics, step, self.config.run_name)
      local_metrics_file.write(str(json.dumps(metrics_dict)) + "\n")

  def write_metrics_for_gcs(self, metrics, step, is_training):
    """Writes metrics to GCS."""
    metrics_dict_step = _prepare_metrics_for_json(metrics, step, self.config.run_name)
    self.running_gcs_metrics.append(metrics_dict_step)
    if is_training and (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      start_step = (step // self.config.log_period) * self.config.log_period
      metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
      with open(metrics_filename, "wt", encoding="utf8") as metrics_for_gcs:
        for metrics_step in self.running_gcs_metrics:
          metrics_for_gcs.write(str(json.dumps(metrics_step)) + "\n")

      gcs_filename = os.path.join(self.config.metrics_dir, metrics_filename)
      max_logging.log(f"Moving file {metrics_filename} to GCS...")
      gcs_utils.upload_blob(gcs_filename, metrics_filename)
      max_logging.log(f"File {metrics_filename} moved successfully!")
      self.running_gcs_metrics = []  # reset running_metrics to empty list

  def write_metrics_to_tensorboard(self, metrics, step, is_training):
    """Writes metrics to TensorBoard."""
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

  def write_metrics_to_managed_mldiagnostics(self, metrics, step):
    """Write metrics to managed profiler."""
    if (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      for metric_name in metrics.get("scalar", []):
        value = metrics["scalar"][metric_name]
        # For NumPy/JAX array objects (including single-element arrays), use .item()
        # to extract the native Python scalar.
        if hasattr(value, "item"):
          value = value.item()
        mapped_metric_name = _METRICS_TO_MANAGED.get(metric_name, metric_name)
        mldiag.metrics.record(mapped_metric_name, value, step=int(step))

  def write_metrics_to_wandb(self, metrics, step):
    """Write metrics to weights and biases (wandb)."""
    flat_metrics = {}
    for key, val in metrics.get("scalar", {}).items():
      flat_metrics[key] = float(val)
    for key, val in metrics.get("scalars", {}).items():
      for subkey, subval in val.items():
        flat_metrics[f"{key}/{subkey}"] = float(subval)
    wandb.log(flat_metrics, step=step)

  def write_setup_info_to_tensorboard(self, params):
    """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
    num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
    self.metadata[MetadataKey.PER_DEVICE_TFLOPS], _, _ = maxtext_utils.calculate_tflops_training_per_device(self.config)
    self.metadata[MetadataKey.PER_DEVICE_TOKENS] = maxtext_utils.calculate_tokens_training_per_device(self.config)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self.writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.getenv("LIBTPU_INIT_ARGS", ""), self.writer)
    maxtext_utils.add_config_to_summary_writer(self.config, self.writer)

  def get_performance_metric_queue(self, config):
    """Records heartbeat metrics and performance metrics to GCP."""
    performance_metric_queue = None
    if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
      gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
      if config.report_heartbeat_metric_for_gcp_monitoring:
        gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
      if config.report_performance_metric_for_gcp_monitoring:
        performance_metric_queue = queue.Queue()
        gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)
    return performance_metric_queue

  def buffer_and_write_train_metrics(self, metrics, step, step_time_delta):
    """
    Buffers metrics for the current training step and simultaneously writes the training metrics
    for the previous step to GCS and/or TensorBoard. This buffering strategy allows for back-to-back
    execution of training steps, by overlapping data loading for step n with the execution of step n−1.
    This significantly boosts training efficiency.
    """
    if self.buffered_train_metrics is not None:
      (step_to_write, metrics_to_write) = self.buffered_train_metrics
      self.write_metrics(metrics_to_write, step_to_write)

    self.record_train_metrics(metrics, step, step_time_delta.total_seconds())
    self.buffered_train_metrics = (step, metrics)

  def record_train_metrics(self, metrics, step, step_time):
    """Records training metrics for the current step."""
    metrics["scalar"].update({"perf/step_time_seconds": step_time})
    metrics["scalar"].update({"learning/current_learning_rate": self.learning_rate_schedule(step)})
    if step >= self.config.rampup_end_step:
      metrics["scalar"].update({"perf/per_device_tflops": self.metadata[MetadataKey.PER_DEVICE_TFLOPS]})
      metrics["scalar"].update(
          {"perf/per_device_tflops_per_sec": (self.metadata[MetadataKey.PER_DEVICE_TFLOPS] / step_time)}
      )
      metrics["scalar"].update({"perf/per_device_tokens": self.metadata[MetadataKey.PER_DEVICE_TOKENS]})
      metrics["scalar"].update(
          {"perf/per_device_tokens_per_sec": (self.metadata[MetadataKey.PER_DEVICE_TOKENS] / step_time)}
      )
    if self.performance_metric_queue:
      self.performance_metric_queue.put(step_time)

  def record_eval_metrics(self, step, metrics=None, eval_step_count=None):
    """Records eval metrics and writes the metrics to GCS and/or to TensorBoard."""
    if metrics:
      self.cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(
          metrics["scalar"].get("evaluation/total_loss", 0.0)
      )
      self.cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(
          metrics["scalar"].get("evaluation/total_weights", 0.0)
      )
      self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(
          metrics["scalar"].get("evaluation/moe_lb_loss", 0.0)
      )
      self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"] += float(metrics["scalar"].get("evaluation/mtp_loss", 0.0))
      self.cumulative_eval_metrics["scalar"]["eval/mtp_acceptance_rate_percent"] += float(
          metrics["scalar"].get("evaluation/mtp_acceptance_rate_percent", 0.0)
      )
      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] += float(
            metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0)
        )

    if eval_step_count:
      eval_loss = self.cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          self.cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      self.cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      self.cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      self.cumulative_eval_metrics["scalar"]["eval/avg_mtp_loss"] = (
          self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"] / eval_step_count
      )
      self.cumulative_eval_metrics["scalar"]["eval/avg_mtp_acceptance_rate_percent"] = (
          self.cumulative_eval_metrics["scalar"]["eval/mtp_acceptance_rate_percent"] / eval_step_count
      )
      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = (
            self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] / eval_step_count
        )

      self.write_metrics(self.cumulative_eval_metrics, step, is_training=False)

  def flush_metrics_and_cleanup(self):
    """
    This is a terminal operation that uploads any buffered metrics to GCS
    and/or TensorBoard before closing the writer objects. Once called, the
    logger instance should not be used to add or write more metrics as the
    underlying writer objects (e.g., TensorBoard SummaryWriter) will be closed.
    """
    if self.buffered_train_metrics is not None:
      (step_to_write, metrics_to_write) = self.buffered_train_metrics
      self.write_metrics(metrics_to_write, step_to_write)

    max_utils.close_summary_writer(self.writer)
