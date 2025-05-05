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
import sys
import queue
from typing import Sequence

from absl import app

import numpy as np

import tensorflow as tf

import jax
from flax.linen import partitioning as nn_partitioning

from ml_goodput_measurement import monitoring

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
    load_next_batch,
    record_scalar_metrics,
    save_checkpoint,
    setup_mesh_and_model,
    train_step,
    validate_train_config,
)

from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import maybe_start_goodput_monitoring, create_goodput_recorder, maybe_record_goodput

def rjx_logging(state, metrics, step):
  jax.block_until_ready(state) # Ensure computation is finished before getting values

  try:
      params_on_host = jax.device_get(state.params)
      weight_slice = params_on_host['params']['decoder']['logits_dense']['kernel'][42, 88]  
      max_logging.log(f"rjx: State after step {step} - Sample weight: {weight_slice}")
      # step 6: 6.34193e-05 (others steps: 5.60284e-05, 1.94311e-05)

      metrics_on_host = jax.device_get(metrics)
      max_logging.log(f"rjx: State after step {step} - Metrics: {metrics_on_host}")
      # step 6: Metrics: {'scalar': {'learning/grad_norm': array(0.996094, dtype=bfloat16), 'learning/loss': array(9.86612, dtype=float32), 'learning/moe_lb_loss': array(0., dtype=float32), 'learning/param_norm': array(14464, dtype=bfloat16), 'learning/raw_grad_norm': array(7.5625, dtype=bfloat16), 'learning/total_weights': array(3615, dtype=int32)}, 'scalars': {}}
  except Exception as e:
      max_logging.log(f"rjx: Error logging state details after step {step}: {e}")
  max_logging.log(f"rjx: ---------- End step {step} ----------")


class TrainingMetrics:
  def __init__(self, config, writer, learning_rate_schedule):
    self.per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
    self.per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)
    self.running_gcs_metrics = [] if config.gcs_metrics else None
    self.performance_metric_queue = self.get_performance_metric_queue(config)
    self.metric_logger = MetricLogger(writer, config)
    self.learning_rate_schedule = learning_rate_schedule
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
  
    record_scalar_metrics(metrics, step_time_delta, self.per_device_tflops, self.learning_rate_schedule(step), self.per_device_tokens)
    if self.performance_metric_queue:
      self.performance_metric_queue.put(step_time_delta.total_seconds())

    self.metric_logger.write_metrics(self.running_gcs_metrics, metrics, step)

  def record_eval_metrics(self, all_eval_metrics, step):
    cumulative_eval_metrics = {
      "scalar": {
          "eval/total_loss": 0.0,
          "eval/total_weights": 0.0,
          "eval/avg_loss": 0.0,
          "eval/moe_lb_loss": 0.0,
      }
    }

    for eval_metrics in all_eval_metrics:
      cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
      cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
      cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])

    eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
      cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
    )

    eval_step_count = len(all_eval_metrics)
    cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
    cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
        cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
    )

    self.metric_logger.write_metrics(self.running_gcs_metrics, cumulative_eval_metrics, step, is_training=False)

    return eval_loss, cumulative_eval_metrics['scalar']['eval/total_weights']


class Profiler:
  def __init__(self, config, state, start_step):
    self.config = config
    self.state = state
    self.prof = profiler.Profiler(config, offset_step=start_step)

    if config.profiler != "" and self.prof.start_initial_profile_step >= config.steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
  
  def start_train_step(self, step):
    if step == self.prof.start_initial_profile_step or self.prof.should_activate_periodic_profile(step):
      self.prof.activate(blocking_object=self.state, optional_postfix=f"step_{step}" if self.config.profile_periodically_period > 0 else "")


  def end_train_step(self, step):
    if step == self.prof.finished_initial_profile_step or self.prof.should_deactivate_periodic_profile(step):
        self.prof.deactivate(blocking_object=self.state)

  def end_train_loop(self):
    self.prof.deactivate()


def try_load_next_batch(recorder, config, data_iterator, example_batch):
  with maybe_record_goodput(recorder, "data_loading"):
    try:
      return load_next_batch(data_iterator, example_batch, config)
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"load_next_batch failed, you may have run out of data. Error message: {e}")
      return None
    

def get_step_rng(init_rng, step, eval_step = None):
  # we need to generate a unique key per train step in the case of training 
  # and per eval step, per train step, in the case of eval
  if eval_step is None:
    return jax.random.fold_in(init_rng, 2 * step + 1)
  else:
    interval_base_rng = jax.random.fold_in(init_rng, 2 * step)
    return jax.random.fold_in(interval_base_rng, eval_step)


def train_step(step, config, mesh, recorder, data_iterator, init_rng, p_train_step, example_batch, state):
  with jax.profiler.StepTraceAnnotation("train", step_num=step):
    example_batch = try_load_next_batch(recorder, config, data_iterator, example_batch)
    if example_batch:
      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      step_rng = get_step_rng(init_rng, step)
      with maybe_record_goodput(recorder, "step", step):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          state, metrics = p_train_step(state, example_batch, step_rng)

      rjx_logging(state, metrics, step)

  return state, metrics, example_batch


def maybe_eval(config, step, start_step, eval_data_iterator, p_eval_step, mesh, init_rng, state):
  all_eval_metrics = []
  # FIXME: this should be >=?
  if config.eval_interval > 0 and step >= start_step and (step + 1) % config.eval_interval == 0:
    assert eval_data_iterator
    
    # pylint: disable=not-callable
    for eval_step, eval_batch in enumerate(eval_data_iterator):
      if config.eval_steps > 0 and eval_step >= config.eval_steps:
        break
      
      step_rng = get_step_rng(init_rng, step, eval_step)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        eval_metrics = p_eval_step(state, eval_batch, step_rng)
        all_eval_metrics.append(eval_metrics)

      max_logging.log(f"Completed eval step {eval_step}")
    
  return all_eval_metrics


def get_and_compile_eval(config, mesh, state_mesh_shardings, model):
  (
      functional_eval,
      in_shard_eval,
      out_shard_eval,
      static_argnums_eval,
      donate_argnums_eval,
  ) = maxtext_utils.get_functional_eval_with_signature(inner_eval_step, mesh, state_mesh_shardings, model, config)

  p_eval_step = jax.jit(
      functional_eval,
      in_shardings=in_shard_eval,
      out_shardings=out_shard_eval,
      static_argnums=static_argnums_eval,
      donate_argnums=donate_argnums_eval,
  )

  return p_eval_step


def get_and_compile_step_functions(config, state, mesh, state_mesh_shardings, model, do_eval, rng):
  # get the compilation of functional_train and eval, either by loading the compiled version or wrapping a new one in a jit
  (
    functional_train,
    in_shard_train,
    out_shard_train,
    static_argnums_train,
    donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(inner_train_step, mesh, state_mesh_shardings, model, config)

  if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...")
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    max_logging.log("Loaded compiled function")
    # TODO: p_eval_step is not yet supported in load_compiled

    return p_train_step, None

  p_train_step = jax.jit(
    functional_train,
    in_shardings=in_shard_train,
    out_shardings=out_shard_train,
    static_argnums=static_argnums_train,
    donate_argnums=donate_argnums_train,
  )

  compiled_train = p_train_step.lower(state, create_dummy_batch_for_train(config), rng).compile()

  if do_eval:
    p_eval_step = get_and_compile_eval(config, mesh, state_mesh_shardings, model)

  return p_train_step, compiled_train, p_eval_step


# TODO: move to max_utils if not already there 
def create_dummy_batch_for_train(config):
  dummy_shape = (config.global_batch_size_to_train_on, config.max_target_length)
  dummy_dtype = jax.numpy.int32

  batch_keys = [
      'inputs', 'inputs_position', 'inputs_segmentation',
      'targets', 'targets_position', 'targets_segmentation'
  ]

  return {key: jax.numpy.zeros(dummy_shape, dtype=dummy_dtype) for key in batch_keys}


def log_statistics(config, state, writer, mesh, compiled_train):
  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")

  # write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  maxtext_utils.add_config_to_summary_writer(config, writer)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):  # TODO: is this used when it's already JITted/compiled?
    compiled_stats = compiled_train.memory_analysis()
    if compiled_stats is not None:
      max_logging.log(
          f"Output size: {compiled_stats.output_size_in_bytes}, "
          f"temp size: {compiled_stats.temp_size_in_bytes}, "
          f"argument size: {compiled_stats.argument_size_in_bytes}, "
          f"host temp size: {compiled_stats.host_temp_size_in_bytes}, in bytes."
      )

  max_utils.print_mem_stats("After params initialized")

def maybe_write_final_checkpoint(checkpoint_manager, state, config, data_iterator):
  checkpoint_not_written_on_final_step = (int(state.step) - 1) % config.checkpoint_period != 0

  if checkpoint_manager is not None and checkpoint_not_written_on_final_step:
    if save_checkpoint(checkpoint_manager, int(state.step) - 1, state, config.dataset_type, data_iterator, config, force=True):
      checkpointing.print_save_message(int(state.step) - 1, config.async_checkpointing)

    checkpoint_manager.wait_until_finished()


def maybe_write_checkpoint(checkpoint_manager, step, state, config, data_iterator):
  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, int(step), state, config.dataset_type, data_iterator, config):
      checkpointing.print_save_message(step, config.async_checkpointing)

    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(step):
      checkpoint_manager.wait_until_finished()
      return True
    
  return False


def maybe_dump_hlo(config, state):
  if config.dump_hlo:
    jax.block_until_ready(state)  # Ensure compilation has finished.
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


def assert_params_sufficiently_sharded(config, state, mesh):
  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)


def train_loop(config, recorder):
  # TODO: think abuot moving these lines out of here into caller
  with maybe_record_goodput(recorder, "job"):  

    with maybe_record_goodput(recorder, "tpu_init"):
      init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
    
    with maybe_record_goodput(recorder, "training_preparation"):
      data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
      state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
        model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
      )
      assert_params_sufficiently_sharded(config, state, mesh)
    
    do_eval = eval_data_iterator is not None
    compile_rng, train_rng = jax.random.split(init_rng, 2)
    # FIXME: should now use compiled_train only?
    p_train_step, compiled_train, p_eval_step = get_and_compile_step_functions(config, state, mesh, state_mesh_shardings, model, do_eval, compile_rng)
    # FIXME: will this include eval or should we compile that too?
    maybe_dump_hlo(config, state)
    log_statistics(config, state, writer, mesh, compiled_train)

    start_step = get_first_step(state)
    profiler = Profiler(config, state, start_step)    
    training_metrics = TrainingMetrics(config, writer, learning_rate_schedule)

    example_batch = None
    for step in range(start_step, config.steps):
      profiler.start_train_step(step)

      state, metrics, example_batch = train_step(step, config, mesh, recorder, data_iterator, train_rng, p_train_step, example_batch, state)
      if example_batch is None:
        break
      training_metrics.record_train_metrics(metrics, step)

      if maybe_write_checkpoint(checkpoint_manager, step, state, config, data_iterator):
        # we are being pre-empted; shut-down
        break

      all_eval_metrics = maybe_eval(config, step, start_step, eval_data_iterator, p_eval_step, mesh, init_rng, state)
      if all_eval_metrics:
        eval_loss, total_weights = training_metrics.record_eval_metrics(all_eval_metrics, step)
        max_logging.log(f"Average loss after {step=}: {step=}, {eval_loss=},"f" {total_weights=}")

        if eval_loss <= config.target_eval_loss:
          max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
          break
    
      profiler.end_train_step(step)

    profiler.end_train_loop()
    maybe_write_final_checkpoint(checkpoint_manager, state, config, data_iterator)
    max_utils.close_summary_writer(writer)

def set_up_environment(config):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data and this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  os.environ["TFDS_DATA_DIR"] = config.dataset_path


def main(argv: Sequence[str]) -> None:
  config = pyconfig.initialize(argv)
  validate_train_config(config)
  
  set_up_environment(config)
  
  max_utils.print_system_information()

  # create a goodput recorder (which will only log if enabled) and start monitoring if configured
  recorder = create_goodput_recorder(config)
  maybe_start_goodput_monitoring(config)

  train_loop(config, recorder)


if __name__ == "__main__":
  app.run(main)
