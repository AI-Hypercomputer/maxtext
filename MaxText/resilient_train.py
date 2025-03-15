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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import queue
import sys
import time
import ray
import asyncio
import random as py_rand

from typing import Sequence
from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import profiler
import pyconfig
import tensorflow as tf
import ray_cluster

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

import jax.numpy as jnp

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from ml_goodput_measurement import monitoring
import orbax.checkpoint as ocp
from gcp_workload_monitor import GCPWorkloadMonitor

from train import (
  EPS,
  validate_train_config,
  get_first_step,
  load_next_batch,
  record_scalar_metrics,
  write_metrics,
  clear_buffered_metrics,
  save_checkpoint,
  _split_dpo_state,
  _merge_dpo_state,
  train_step,
  eval_step,
  create_goodput_recorder,
  record_goodput,
  check_example_batch,
  setup_train_loop
)
# pylint: disable=too-many-positional-arguments


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------

@ray.remote
class MaxtextTrainer(ray_cluster.ResilientWorker):
  def __init__(self, process_id, physical_node_id, physical_node_ip):
    super().__init__(process_id, physical_node_id, physical_node_ip)

  def initialize(self, coordinator_addr, num_processes, **kwargs):
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    tf.config.set_visible_devices([], "GPU")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    super().initialize(coordinator_addr, num_processes)
    ocp.multihost.initialize_runtime_to_distributed_ids()
    ocp.multihost.initialize_distributed_to_device_ids()
    maxtext_args = kwargs['maxtext_args']
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
      os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    pyconfig.initialize(maxtext_args)
    max_utils.print_system_information()
    self.config = pyconfig.config
    validate_train_config(self.config)
    os.environ["TFDS_DATA_DIR"] = self.config.dataset_path
    self.vertex_tensorboard_manager = VertexTensorboardManager()
    if self.config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
      self.vertex_tensorboard_manager.configure_vertex_tensorboard(self.config)

    if self.config.monitor_goodput and jax.process_index() == 0:
      logger_name = f"goodput_{self.config.run_name}"
      self.goodput_monitor = monitoring.GoodputMonitor(
        job_name=self.config.run_name,
        logger_name=logger_name,
        tensorboard_dir=self.config.tensorboard_dir,
        upload_interval=self.config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        pathway_enabled=self.config.enable_pathways_goodput,
        include_badput_breakdown=True,
        include_step_deviation=self.config.monitor_step_time_deviation,
        step_deviation_interval_seconds=self.config.step_deviation_interval_seconds,
      )
      self.goodput_monitor.start_goodput_uploader()
      max_logging.log("Started Goodput upload to Tensorboard in the background!")
      if self.config.monitor_step_time_deviation:
        self.goodput_monitor.start_step_deviation_uploader()
        max_logging.log("Started step time deviation upload to Tensorboard in the background!")
      
    debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=self.config.collect_stack_trace,
          stack_trace_to_cloud=self.config.stack_trace_to_cloud,
          stack_trace_interval_seconds=self.config.stack_trace_interval_seconds,
      )
    )
    self.diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  
  def _fail(self, fail_prob, hang_prob):
    if py_rand.random() >= (1 - hang_prob):
      max_logging.log("Gonna hang")
      time.sleep(3600)

    if py_rand.random() >= (1 - fail_prob):
      exception = False if py_rand.random() < 0.5 else True
      max_logging.log(f"Failing with exception = {exception}")
      if exception:
        raise Exception("Failure")
      else:
        # Cause a seg fault, no graceful exception propagation
        eval((lambda:0).__code__.replace(co_consts=()))

  def _train_loop(self, state=None):
    """Main Training loop.
    Args:
      config:
      state:
      ckpt_path:
    Returns:
    """
    # Create a GoodputRecorder to log information
    recorder = create_goodput_recorder(self.config)
    record_goodput(recorder, self.config, recorder.record_job_start_time if recorder else None)

    (
        init_rng,
        writer,
        checkpoint_manager,
        state_mesh_shardings,
        model,
        mesh,
        learning_rate_schedule,
        data_iterator,
        eval_data_iterator,
        state,
    ) = setup_train_loop(self.config)

    if self.config.use_dpo:
      if "reference_params" not in state.params:
        reference_params = jax.tree.map(jnp.copy, state.params["params"])
        state = _merge_dpo_state(state, reference_params)
      state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

    # pylint: disable=line-too-long
    (
        functional_train,
        in_shard_train,
        out_shard_train,
        static_argnums_train,
        donate_argnums_train,
    ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, self.config)

    if eval_data_iterator:
      # pylint: disable=line-too-long
      (
          functional_eval,
          in_shard_eval,
          out_shard_eval,
          static_argnums_eval,
          donate_argnums_eval,
      ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, self.config)

    num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
    per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(self.config)
    per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(self.config)

    # Write train config params, num model params, and XLA flags to tensorboard
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
    if self.config.compiled_trainstep_file != "":
      print("Loading the compiled function...", flush=True)
      # Need to pass train signature and state to determine i/o shapes of train_state for now.
      p_train_step = maxtext_utils.load_compiled(self.config, functional_train, state)
      # TODO: p_eval_step is not yet supported in load_compiled
      p_eval_step = None
      print("Loaded compiled function!", flush=True)
    else:
      p_train_step = jax.jit(
          functional_train,
          in_shardings=in_shard_train,
          out_shardings=out_shard_train,
          static_argnums=static_argnums_train,
          donate_argnums=donate_argnums_train,
      )

      if eval_data_iterator:
        p_eval_step = jax.jit(
            functional_eval,
            in_shardings=in_shard_eval,
            out_shardings=out_shard_eval,
            static_argnums=static_argnums_eval,
            donate_argnums=donate_argnums_eval,
        )
      else:
        p_eval_step = None

    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None

    start_step = get_first_step(state)  # this is the start_step for training
    prof = profiler.Profiler(self.config, offset_step=start_step)
    first_profiling_step = prof.start_initial_profile_step
    if self.config.profiler != "" and first_profiling_step >= self.config.steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = prof.finished_initial_profile_step

    example_batch = None
    last_step_completion = datetime.datetime.now()
    failure_timer_start = datetime.datetime.now()
    performance_metric_queue = None
    if self.config.report_heartbeat_metric_for_gcp_monitoring or self.config.report_performance_metric_for_gcp_monitoring:
      gcp_workload_monitor = GCPWorkloadMonitor(self.config.run_name)
      if self.config.report_heartbeat_metric_for_gcp_monitoring:
        gcp_workload_monitor.start_heartbeat_reporting_thread(self.config.heartbeat_reporting_interval_in_seconds)
      if self.config.report_performance_metric_for_gcp_monitoring:
        performance_metric_queue = queue.Queue()
        gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)

    for step in np.arange(start_step, self.config.steps):
      with self.EnableHeartbeat():
        if step == first_profiling_step or prof.should_activate_periodic_profile(step):
          optional_postfix = f"step_{step}" if self.config.profile_periodically_period > 0 else ""
          prof.activate(blocking_object=state, optional_postfix=optional_postfix)

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          record_goodput(recorder, self.config, recorder.record_data_loading_start_time if recorder else None)
          example_batch = load_next_batch(data_iterator, example_batch, self.config)
          record_goodput(recorder, self.config, recorder.record_data_loading_end_time if recorder else None)
          check_example_batch(self.config, example_batch=example_batch)
          # pylint: disable=not-callable
          nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
          record_goodput(recorder, self.config, recorder.record_step_start_time if recorder else None, step)
          with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state, metrics = p_train_step(state, example_batch, nextrng)

        step_time_delta = datetime.datetime.now() - last_step_completion
        last_step_completion = datetime.datetime.now()
        record_scalar_metrics(metrics, step_time_delta, per_device_tflops, learning_rate_schedule(step), per_device_tokens)
        if performance_metric_queue:
          performance_metric_queue.put(step_time_delta.total_seconds())
        
        if checkpoint_manager is not None:
          state_to_save = state if not self.config.use_dpo else _split_dpo_state(state)[0]
          if save_checkpoint(checkpoint_manager, int(step), state_to_save, self.config.dataset_type, data_iterator, self.config):
            checkpointing.print_save_message(step, self.config.async_checkpointing)

          # Upon preemption, exit when and only when all ongoing saves are complete.
          if checkpoint_manager.reached_preemption(step):
            checkpoint_manager.wait_until_finished()
            sys.exit()

        write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, self.config)

        if self.config.dump_hlo and step == start_step:
          jax.block_until_ready(state)  # Ensure compilation has finished.
          max_utils.upload_dump(
              self.config.dump_hlo_local_dir,
              self.config.dump_hlo_gcs_dir,
              module_name=self.config.dump_hlo_module_name,
              delete_local_after=self.config.dump_hlo_delete_local_after,
              all_host_upload=self.config.dump_hlo_upload_all,
          )

        if self.config.eval_interval > 0 and step > start_step and (step + 1) % self.config.eval_interval == 0:
          assert eval_data_iterator
          cumulative_eval_metrics = {
              "scalar": {
                  "eval/total_loss": 0.0,
                  "eval/total_weights": 0.0,
                  "eval/avg_loss": 0.0,
                  "eval/moe_lb_loss": 0.0,
              }
          }
          eval_dpo_reward_accuracy = 0.0
          eval_step_count = 0
          # pylint: disable=not-callable
          for eval_batch in eval_data_iterator:
            if self.config.eval_steps > 0 and eval_step_count >= self.config.eval_steps:
              break
            with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
              eval_metrics = p_eval_step(state, eval_batch, nextrng)
            cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
            cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
            cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
            eval_dpo_reward_accuracy += float(eval_metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))  # for dpo only
            max_logging.log(f"Completed eval step {eval_step_count}")
            eval_step_count += 1
          eval_loss = (
              cumulative_eval_metrics["scalar"]["eval/total_loss"]
              / (cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS)
              + cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
          )
          cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
          cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
              cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
          )
          if self.config.use_dpo:
            cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = eval_dpo_reward_accuracy / eval_step_count
          write_metrics(
              writer, local_metrics_file, running_gcs_metrics, cumulative_eval_metrics, step, self.config, is_training=False
          )
          max_logging.log(
              f"average loss after {step=}: {eval_step_count=}, {eval_loss=},"
              f" total_weights={cumulative_eval_metrics['scalar']['eval/total_weights']}"
          )
          if eval_loss <= self.config.target_eval_loss:
            max_logging.log(f"Early stop and exit loop after reaching {self.config.target_eval_loss=}")
            prof.deactivate()
            break

        if step == last_profiling_step or prof.should_deactivate_periodic_profile(step):
          prof.deactivate(blocking_object=state)

        if step == start_step:
          max_utils.print_mem_stats("After params initialized")
        
        current_time = datetime.datetime.now()
        time_since_failure_sim = (current_time - failure_timer_start).total_seconds()
        if time_since_failure_sim >= self.config.failure_sim_time:
          self._fail(self.config.crash_prob, self.config.hang_prob)

    if checkpoint_manager is not None:
      checkpoint_manager.wait_until_finished()
    write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, self.config.steps - 1, self.config)  # final step metrics
    max_utils.close_summary_writer(writer)
    record_goodput(recorder, self.config, recorder.record_job_end_time if recorder else None)
    clear_buffered_metrics()
    with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      # pytype: disable=attribute-error
      compiled = p_train_step.lower(state, example_batch, nextrng).compile()
      compiled_stats = compiled.memory_analysis()
      if compiled_stats is not None:
        max_logging.log(
            f"Output size: {compiled_stats.output_size_in_bytes}, "
            f"temp size: {compiled_stats.temp_size_in_bytes}, "
            f"argument size: {compiled_stats.argument_size_in_bytes}, "
            f"host temp size: {compiled_stats.host_temp_size_in_bytes}, in bytes."
        )
    return state

  def run(self):
    with diagnostic.diagnose(self.diagnostic_config):
      self._train_loop()

def main(argv: Sequence[str]) -> None:
  ray.init(address='auto', logging_level=0)

  hang_time_threshold = None
  # Get hang time threshold
  for arg in argv:
    if arg.startswith('hang_time_threshold='):
      hang_time_threshold = int(arg.split('=')[1])
      break

  cluster_coordinator = ray_cluster.RayClusterCoordinator(MaxtextTrainer, hang_time_threshold=hang_time_threshold)
  cluster_coordinator.initialize_workers(maxtext_args=argv)
  cluster_coordinator.log("Initialized workers")
  asyncio.run(cluster_coordinator.run())



if __name__ == "__main__":
  app.run(main)
