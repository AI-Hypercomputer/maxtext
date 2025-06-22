from datetime import datetime
import queue
import contextlib
from typing import Any

import jax

from MaxText import profiler
from MaxText.gcp_workload_monitor import GCPWorkloadMonitor
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
  record_scalar_metrics,
)
from MaxText.utils import gcs_utils
from MaxText import maxtext_utils, max_utils, max_logging

# TODO: we likely should defer accessing metrics until the n+1th training step has completed
# to not force blocking
class MetricsRecorder:
  def __init__(self, config, train_ctx):
    self.per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
    self.per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)
    self.running_gcs_metrics = [] if config.gcs_metrics else None
    self.performance_metric_queue = self.get_performance_metric_queue(config)
    self.metric_logger = MetricLogger(train_ctx.writer, config)
    self.learning_rate_schedule = train_ctx.learning_rate_schedule
    self.last_step_completion = datetime.now()

  def get_performance_metric_queue(self, config):
    performance_metric_queue = None

    if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
      gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
      if config.report_heartbeat_metric_for_gcp_monitoring:
        gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
      if config.report_performance_metric_for_gcp_monitoring:
        performance_metric_queue = queue.Queue()
        gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)

    return performance_metric_queue

  def record_train_metrics(self, metrics, step):
    now = datetime.now()
    step_time_delta = now - self.last_step_completion
    self.last_step_completion = now

    lr_schedule = self.learning_rate_schedule(step)
    record_scalar_metrics(metrics, step_time_delta, self.per_device_tflops, lr_schedule, self.per_device_tokens)
    if self.performance_metric_queue:
      self.performance_metric_queue.put(step_time_delta.total_seconds())

    self.metric_logger.write_metrics(self.running_gcs_metrics, metrics, step)

  def record_eval_metrics(self, aggregated_metrics, step):
    self.metric_logger.write_metrics(self.running_gcs_metrics, aggregated_metrics, step, is_training=False)


class Profiler:
  def __init__(self, config, state, start_step):
    self.config = config
    self.state = state
    self.prof = profiler.Profiler(config, offset_step=start_step)

    if config.profiler != "" and self.prof.start_initial_profile_step >= config.steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")

  def _start_train_step(self, step):
    if step == self.prof.start_initial_profile_step or self.prof.should_activate_periodic_profile(step):
      postfix = f"step_{step}" if self.config.profile_periodically_period > 0 else ""
      self.prof.activate(blocking_object=self.state, optional_postfix=postfix)


  def _end_train_step(self, step):
    if step == self.prof.finished_initial_profile_step or self.prof.should_deactivate_periodic_profile(step):
        self.prof.deactivate(blocking_object=self.state)

  @contextlib.contextmanager
  def train_step(self, step):
    try:
      self._start_train_step(step)
      yield
    finally:
      self._end_train_step(step)

  @contextlib.contextmanager
  def train_loop(self):
    try:
      yield
    finally:
      self._end_train_loop()

  def _end_train_loop(self):
    self.prof.deactivate()


def maybe_upload_hlo(config, state):
  if config.dump_hlo:
    jax.block_until_ready(state)  # Ensure compilation has finished.
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


def assert_params_sufficiently_sharded(config, train_ctx, state):
  # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage in PP
  if not config.using_pipeline_parallelism:
    maxtext_utils.assert_params_sufficiently_sharded(state.params, train_ctx.mesh, config.sharding_tolerance)


def log_statistics(config, train_ctx, state):
  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")

  # write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), train_ctx.writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], train_ctx.writer)
  maxtext_utils.add_config_to_summary_writer(config, train_ctx.writer)

  max_utils.print_mem_stats("After params initialized")


class TrainingHooks:
  # TODO: train_ctx or whatever we pass here should probably contain state and start_step
  #       we should also minimize train_ctx to what hooks need vs. training loop
  def __init__(self, config):
    self.config = config
    self.training_step_metrics = None
    self.eval_step_metrics = None

  def pre_training(self, train_ctx, state, start_step):
    self.train_ctx = train_ctx
    self.state = state
    self.start_step = start_step

    self.profiler = Profiler(self.config, state, start_step)
    self.metrics_recorder = MetricsRecorder(self.config, train_ctx)

  @contextlib.contextmanager
  def training_loop(self):
      with self.profiler.train_loop():
        try:
          assert_params_sufficiently_sharded(self.config, self.train_ctx, self.state)
          maybe_upload_hlo(self.config, self.state)
          log_statistics(self.config, self.train_ctx, self.state)

          yield
        finally:
          # TODO: this should be created in this class, not by setup_mesh_and_model
          max_utils.close_summary_writer(self.train_ctx.writer)

  @contextlib.contextmanager
  def prepare_inputs(self, input_data: Any) -> Any:
    return input_data

  @contextlib.contextmanager
  def training_step(self, step):
    with self.profiler.train_step(step):
      try:
        yield
      finally:
        self.metrics_recorder.record_train_metrics(self.training_step_metrics, step)

  @contextlib.contextmanager
  def eval_step(self, step):
    # TODO: add this to profiler
    # with self.profiler.eval_step():
    try:
      pass
    finally:
      self.metrics_recorder.record_eval_metrics(self.eval_step_metrics, step)

