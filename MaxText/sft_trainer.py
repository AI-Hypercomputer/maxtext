# Copyright 2023 Google LLC
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

"Training loop for Supervised Fine-Tuning (SFT)."

from datetime import datetime
import os
import queue
from typing import Sequence
from dataclasses import dataclass
from typing import Any, Callable
import contextlib

from absl import app
import tensorflow as tf
import jax
from flax.linen import partitioning as nn_partitioning

from MaxText import checkpointing
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import max_logging
from MaxText import profiler
from MaxText import pyconfig
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.gcp_workload_monitor import GCPWorkloadMonitor
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
  check_example_batch,
  train_step as inner_train_step,
  eval_step as inner_eval_step,
  EPS,
  get_first_step,
  record_scalar_metrics,
  save_checkpoint,
  setup_mesh_and_model,
  train_step,
  validate_train_config,
)
from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import maybe_start_goodput_monitoring, create_goodput_recorder, maybe_record_goodput


class StopTraining(Exception):
  pass


class DataLoader:
  def __init__(self, config, data_iterator, recorder):
    self.reuse_example_batch = config.reuse_example_batch
    self.data_iterator = data_iterator
    self.recorder = recorder
    self.last_batch = None

  def try_load_next_batch(self):
    with maybe_record_goodput(self.recorder, "data_loading"):
      if self.reuse_example_batch and self.last_batch:
        return self.last_batch
      
      try:
        self.last_batch = next(self.data_iterator)
      except Exception as e:  # pylint: disable=broad-except
        max_logging.log(f"load_next_batch failed, you may have run out of data. Error message: {e}")
        self.last_batch = None
        raise StopTraining()

      return self.last_batch


class EvalMetrics:
  def __init__(self):
    self.all_eval_metrics = []

  def __bool__(self):
    return bool(self.all_eval_metrics)

  def append(self, metrics):
    self.all_eval_metrics.append(metrics)

  def aggregate(self):
    cumulative_eval_metrics = {
          "eval/total_loss": 0.0,
          "eval/total_weights": 0.0,
          "eval/avg_loss": 0.0,
          "eval/moe_lb_loss": 0.0,
    }

    for eval_metrics in self.all_eval_metrics:
      cumulative_eval_metrics["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
      cumulative_eval_metrics["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
      cumulative_eval_metrics["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])

    eval_loss = cumulative_eval_metrics["eval/total_loss"] / (
      cumulative_eval_metrics["eval/total_weights"] + EPS
    )

    eval_step_count = len(self.all_eval_metrics)
    cumulative_eval_metrics["eval/avg_loss"] = eval_loss
    cumulative_eval_metrics["eval/avg_moe_lb_loss"] = (
        cumulative_eval_metrics["eval/moe_lb_loss"] / (eval_step_count + EPS)
    )

    total_weights = cumulative_eval_metrics["eval/total_weights"]

    return {"scalar": cumulative_eval_metrics}, eval_loss, total_weights


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


@dataclass
class TrainingContext:
  writer: Any
  checkpoint_manager: Any
  mesh: Any    
  model: Any
  learning_rate_schedule: Any
  tx: Any
  recorder: Any


@dataclass
class LoopContext:
  # truly immutable objects
  train_rng: jax.random.PRNGKey
  p_train_step: Callable 
  p_eval_step: Callable
  start_step: int
  # NOTE: when used, these objects are side-effecting
  profiler: Profiler
  metrics_recorder: MetricsRecorder
  data_loader: Any
  eval_data_iterator: Any
  # NOTE: this field is mutable and will be updated by the inner train step function
  state: Any


def set_up_environment(config):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  # TF allocates extraneous GPU memory when using TFDS and this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + 
      " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )


def get_step_rng(init_rng, step, eval_step = None):
  # we need to generate a unique key per train step in the case of training 
  # and per eval step, per train step, in the case of eval
  if eval_step is None:
    return jax.random.fold_in(init_rng, 2 * step + 1)
  else:
    interval_base_rng = jax.random.fold_in(init_rng, 2 * step)
    return jax.random.fold_in(interval_base_rng, eval_step)


def print_compiled_train_step_stats(compiled_train):
  compiled_stats = compiled_train.memory_analysis()
  if compiled_stats is not None:
    max_logging.log(
        f"Output size: {compiled_stats.output_size_in_bytes}, "
        f"temp size: {compiled_stats.temp_size_in_bytes}, "
        f"argument size: {compiled_stats.argument_size_in_bytes}, "
        f"host temp size: {compiled_stats.host_temp_size_in_bytes}, in bytes."
    )


def get_and_compile_eval(config, train_ctx, state_mesh_shardings):
  get_partial_eval = maxtext_utils.get_functional_eval_with_signature

  (
      functional_eval,
      in_shard_eval,
      out_shard_eval,
      static_argnums_eval,
      donate_argnums_eval,
  ) = get_partial_eval(inner_eval_step, train_ctx.mesh, state_mesh_shardings, train_ctx.model, config)

  p_eval_step = jax.jit(
      functional_eval,
      in_shardings=in_shard_eval,
      out_shardings=out_shard_eval,
      static_argnums=static_argnums_eval,
      donate_argnums=donate_argnums_eval,
  )

  return p_eval_step


def get_and_compile_step_functions(config, train_ctx, state_mesh_shardings, do_eval, rng, state):
  get_partial_train = maxtext_utils.get_functional_train_with_signature

  (
    functional_train,
    in_shard_train,
    out_shard_train,
    static_argnums_train,
    donate_argnums_train,
  ) = get_partial_train(inner_train_step, train_ctx.mesh, state_mesh_shardings, train_ctx.model, config)
  
  p_train_step = jax.jit(
    functional_train,
    in_shardings=in_shard_train,
    out_shardings=out_shard_train,
    static_argnums=static_argnums_train,
    donate_argnums=donate_argnums_train,
  )

  # TODO: is this used when it's already JITted/compiled?
  with train_ctx.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):  
    compiled_train = p_train_step.lower(state, create_dummy_batch_for_train(config), rng).compile()
    print_compiled_train_step_stats(compiled_train)

  if do_eval:
    p_eval_step = get_and_compile_eval(config, train_ctx, state_mesh_shardings)

  return p_train_step, p_eval_step


# TODO: move to max_utils if not already there 
def create_dummy_batch_for_train(config):
  dummy_shape = (config.global_batch_size_to_train_on, config.max_target_length)
  dummy_dtype = jax.numpy.int32

  batch_keys = [
    'inputs', 'inputs_position', 'inputs_segmentation',
    'targets', 'targets_position', 'targets_segmentation'
  ]

  return {key: jax.numpy.zeros(dummy_shape, dtype=dummy_dtype) for key in batch_keys}


def log_statistics(config, train_ctx, state):
  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")

  # write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), train_ctx.writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], train_ctx.writer)
  maxtext_utils.add_config_to_summary_writer(config, train_ctx.writer)

  max_utils.print_mem_stats("After params initialized")


def maybe_write_final_checkpoint(checkpoint_manager, state, config):
  checkpoint_manager = checkpoint_manager
  step = int(state.step) - 1
  checkpoint_not_written_on_final_step = step % config.checkpoint_period != 0

  if checkpoint_manager is not None and checkpoint_not_written_on_final_step:
    if save_checkpoint(checkpoint_manager, step, state, config.dataset_type, None, config, force=True):
      checkpointing.print_save_message(step, config.async_checkpointing)

    checkpoint_manager.wait_until_finished()


def maybe_write_checkpoint(checkpoint_manager, step, state, config):
  checkpoint_manager = checkpoint_manager
  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, int(step), state, config.dataset_type, None, config):
      checkpointing.print_save_message(step, config.async_checkpointing)

    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(step):
      checkpoint_manager.wait_until_finished()
      raise StopTraining()


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


def train_step(step, config, train_ctx, loop_ctx):
  with jax.profiler.StepTraceAnnotation("train", step_num=step):
    example_batch = loop_ctx.data_loader.try_load_next_batch()
    if example_batch:
      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      step_rng = get_step_rng(loop_ctx.train_rng, step)
      with maybe_record_goodput(train_ctx.recorder, "step", step):
        # FIXME: can we remove this since it's compiled
        with train_ctx.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          loop_ctx.state, metrics = loop_ctx.p_train_step(loop_ctx.state, example_batch, step_rng)

  return metrics


def maybe_eval_step(config, train_ctx, loop_ctx, step):
  eval_metrics = EvalMetrics()
  if config.eval_interval > 0 and step >= loop_ctx.start_step and (step + 1) % config.eval_interval == 0:
    assert loop_ctx.eval_data_iterator
    
    # pylint: disable=not-callable
    for eval_step, eval_batch in enumerate(loop_ctx.eval_data_iterator):
      if config.eval_steps > 0 and eval_step >= config.eval_steps:
        break
      
      step_rng = get_step_rng(loop_ctx.init_rng, step, eval_step)
      with train_ctx.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        step_metrics = loop_ctx.p_eval_step(loop_ctx.state, eval_batch, step_rng)
        eval_metrics.append(step_metrics)

      max_logging.log(f"Completed eval step {eval_step}")

    loop_ctx.eval_data_iterator.reset()
    
  return eval_metrics


def train_and_maybe_eval_step(config, train_ctx, loop_ctx, step):
  with profiler.train_step(step):
    metrics = train_step(step, config, train_ctx, loop_ctx)
  loop_ctx.metrics_recorder.record_train_metrics(metrics, step)

  maybe_write_checkpoint(train_ctx.checkpoint_manager, step, loop_ctx.state, config)

  eval_metrics = maybe_eval_step(config, train_ctx, loop_ctx, step)
  if eval_metrics:
    aggregated_metrics, eval_loss, total_weights = eval_metrics.aggregate()
    loop_ctx.metrics_recorder.record_eval_metrics(aggregated_metrics, step)
    max_logging.log(f"Average loss after {step=}: {step=}, {eval_loss=},"f" {total_weights=}")

    if eval_loss <= config.target_eval_loss:
      max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
      raise StopTraining()


def train_loop(config, train_ctx, loop_ctx):
    try:
      for step in range(loop_ctx.start_step, config.steps):
          train_and_maybe_eval_step(config, train_ctx, loop_ctx, step)
    except StopTraining:
      max_logging.log("Training stopped due to hitting loss target, pre-emption or end of training data")

    maybe_write_final_checkpoint(train_ctx.checkpoint_manager, loop_ctx.state, config)


def train(config, recorder):
  with maybe_record_goodput(recorder, "tpu_init"):
    init_rng, *train_ctx_args= setup_mesh_and_model(config)
    train_ctx = TrainingContext(*train_ctx_args, recorder)
  
  with maybe_record_goodput(recorder, "training_preparation"):
    data_it, eval_data_it = create_data_iterator(config, train_ctx.mesh)

    state, _, shd, data_it = maxtext_utils.setup_training_state(
      train_ctx.model, data_it, train_ctx.tx, config, init_rng, train_ctx.mesh, train_ctx.checkpoint_manager
    )
    assert_params_sufficiently_sharded(config, train_ctx, state)

    compile_rng, train_rng = jax.random.split(init_rng, 2)
    p_train_step, p_eval_step = get_and_compile_step_functions(config, train_ctx, shd, eval_data_it is not None,
                                                               compile_rng, state)

    maybe_upload_hlo(config, state)
    log_statistics(config, state, train_ctx)

    start_step = get_first_step(state)
    profiler = Profiler(config, state, start_step)    
    metrics_recorder = MetricsRecorder(config, train_ctx)

    data_loader = DataLoader(config, data_it, train_ctx.recorder)
    loop_ctx = LoopContext(train_rng, p_train_step, p_eval_step, start_step, profiler,
                          metrics_recorder, data_loader, eval_data_it, state)

  with profiler.train_loop():
    state = train_loop(config, train_ctx, loop_ctx)

  max_utils.close_summary_writer(train_ctx.writer)


def main(argv: Sequence[str]) -> None:
  config = pyconfig.initialize(argv)
  validate_train_config(config)
  
  set_up_environment(config)
  
  max_utils.print_system_information()

  # create a goodput recorder (which will only log if enabled) and start monitoring if configured
  recorder = create_goodput_recorder(config)
  maybe_start_goodput_monitoring(config)

  with maybe_record_goodput(recorder, "job"):  
    train(config, recorder)


if __name__ == "__main__":
  app.run(main)
