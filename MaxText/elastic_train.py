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

"""Training loop for elastic training model."""
import datetime
import logging
import os
import sys
import traceback
import queue

from collections.abc import Sequence
from absl import app
from flax.linen import partitioning as nn_partitioning
import jax

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import profiler
import pyconfig
import pathwaysutils  # pylint: disable=unused-import
import tensorflow as tf

from metric_logger import MetricLogger

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from input_pipeline.input_pipeline_interface import create_data_iterator

from gcp_workload_monitor import GCPWorkloadMonitor

import jax.numpy as jnp

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from ml_goodput_measurement import monitoring

from elastic.debug import timing
from elastic.debug import watchdog

from train import (
    check_example_batch,
    create_goodput_recorder,
    eval_step,
    get_first_step,
    load_next_batch,
    record_goodput, record_scalar_metrics,
    save_checkpoint, setup_mesh_and_model, setup_train_loop,
    train_step,
    validate_train_config,
    EPS,
)

logging.basicConfig()
logging.getLogger('elastic.manager').setLevel(logging.DEBUG)
logging.getLogger('elastic.simulated_manager').setLevel(logging.DEBUG)
logging.getLogger('elastic.reshard').setLevel(logging.DEBUG)
logging.getLogger('elastic.debug.timing').setLevel(logging.DEBUG)
logging.getLogger('elastic.debug.watchdog').setLevel(logging.DEBUG)


@timing.timeit
def elastic_handler(
    config: pyconfig.HyperParameters,
    checkpoint_manager,
):
  """Reshard function."""
  if checkpoint_manager is not None:
    checkpoint_manager.close()

  with jax.default_device(config.em.default_device):
    init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = (
        setup_mesh_and_model(config)
    )
    with mesh:
      data_iterator, _ = create_data_iterator(config, mesh)

      step, snapshot = config.em.get_resharded_snapshot(mesh)

      if checkpoint_manager is not None:
        # Confirm this is the right thing to do
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None and latest_step >= step:
          max_logging.log(
              f"Deleting checkpoint from step {latest_step} since we are "
              f"rewinding to step {step}."
          )
          checkpoint_manager.delete(latest_step)

      state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
          model,
          data_iterator,
          tx,
          config,
          jax.random.fold_in(init_rng, step),
          mesh,
          checkpoint_manager=None,
      )

      state = state.replace(**snapshot)
      state = state.replace(step=state.step.at[None].set(step))

      slice_indices = jax.tree.map(lambda x: {d.slice_index for d in x.sharding.device_set}, state.params)

      (
          functional_train,
          in_shard_train,
          out_shard_train,
          static_argnums_train,
          donate_argnums_train,
      ) = maxtext_utils.get_functional_train_with_signature(
          train_step, mesh, state_mesh_shardings, model, config
      )

      p_train_step = jax.jit(
          functional_train,
          in_shardings=in_shard_train,
          out_shardings=out_shard_train,
          static_argnums=static_argnums_train,
          donate_argnums=donate_argnums_train,
      )

      example_batch = None
      metric_logger = MetricLogger(writer, config)

      jax.block_until_ready(state)

  return (
      config,
      step,
      state,
      mesh,
      checkpoint_manager,
      data_iterator,
      p_train_step,
      example_batch,
      learning_rate_schedule,
      metric_logger,
      writer,
  )


def train_loop(config, state=None):
  """Main Training loop.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  # Create a GoodputRecorder to log information
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_job_start_time if recorder else None)

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
  ) = setup_train_loop(config)

  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  p_train_step = jax.jit(
      functional_train,
      in_shardings=in_shard_train,
      out_shardings=out_shard_train,
      static_argnums=static_argnums_train,
      donate_argnums=donate_argnums_train,
  )

  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  max_utils.add_config_to_summary_writer(config, writer)

  local_metrics_file = open(config.metrics_file, "a", encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  first_profiling_step = prof.start_initial_profile_step
  if config.profiler != "" and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = prof.finished_initial_profile_step

  example_batch = None
  last_step_completion = datetime.datetime.now()

  performance_metric_queue = None
  if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
    gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
    if config.report_heartbeat_metric_for_gcp_monitoring:
      gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
    if config.report_performance_metric_for_gcp_monitoring:
      performance_metric_queue = queue.Queue()
      gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)

  metric_logger = MetricLogger(writer, config)
  step = start_step

  config.em.initialize_snapshot(
      step,
      snapshot=dict(
          params=state.params,
          opt_state=state.opt_state,
      ),
  )

  # step_down = {10, 30, 44}
  # step_up = {14, 40, 45}
  while True:
    with watchdog.watchdog(120):
      try:
        # if step in step_down:
        #   step_down.remove(step)
        #   # Remove a slice
        #   config.em.update_good_slice_indices(set(range(config.em.total_slice_count))
        #   - {step % config.em.total_slice_count})
        #   raise jax.errors.JaxRuntimeError("DATA_LOSS: Fake")
        # elif step in step_up:
        #   step_up.remove(step)

        #   config.em.update_good_slice_indices(set(range(config.em.total_slice_count)))


        if step == first_profiling_step or prof.should_activate_periodic_profile(step):
          optional_postfix = f"step_{step}" if config.profile_periodically_period > 0 else ""
          prof.activate(blocking_object=state, optional_postfix=optional_postfix)

        if step >= config.steps:
          break

        max_logging.log(f"{step=} {config.em.failure_count=} {config.em.good_slice_count=}")
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules), jax.default_device(config.em.default_device):
          with jax.profiler.StepTraceAnnotation("train", step_num=step):
            record_goodput(recorder, config, recorder.record_data_loading_start_time if recorder else None)
            example_batch = load_next_batch(data_iterator, example_batch, config)
            record_goodput(recorder, config, recorder.record_data_loading_end_time if recorder else None)
            check_example_batch(config, example_batch=example_batch)
            # pylint: disable=not-callable
            nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
            record_goodput(recorder, config, recorder.record_step_start_time if recorder else None, step)
            state, metrics = p_train_step(state, example_batch, nextrng)

          step_time_delta = datetime.datetime.now() - last_step_completion
          last_step_completion = datetime.datetime.now()
          record_scalar_metrics(metrics, step_time_delta, per_device_tflops, learning_rate_schedule(step), per_device_tokens)
          if performance_metric_queue:
            performance_metric_queue.put(step_time_delta.total_seconds())

          if checkpoint_manager is not None:
            state_to_save = state
            if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
              checkpointing.print_save_message(step, config.async_checkpointing)

            # Upon preemption, exit when and only when all ongoing saves are complete.
            if checkpoint_manager.reached_preemption(step):
              checkpoint_manager.wait_until_finished()
              sys.exit()

          metric_logger.write_metrics(running_gcs_metrics, metrics, step)

          if step == last_profiling_step or prof.should_deactivate_periodic_profile(step):
            prof.deactivate(blocking_object=state)

        config.em.maybe_snapshot(
            step=step,
            snapshot=dict(
                params=state.params,
                opt_state=state.opt_state,
            ),
            block=True,
        )

        ret = config.em.maybe_reshard_up(
            step=step,
            snapshot=dict(
                params=state.params,
                opt_state=state.opt_state,
            ),
            elastic_handler=elastic_handler,
            handler_kwargs=dict(
                config=config,
                checkpoint_manager=checkpoint_manager,
            ),
        )
        if ret is not None:
          (config,
           step,
           state,
           mesh,
           checkpoint_manager,
           data_iterator,
           p_train_step,
           example_batch,
           learning_rate_schedule,
           metric_logger,
           writer) = ret

        if step == start_step:
          max_utils.print_mem_stats("After params initialized")

        step += 1

      except jax.errors.JaxRuntimeError as error:
        ret = config.em.maybe_reshard_down(
            error=error,
            elastic_handler=elastic_handler,
            handler_kwargs=dict(
                config=config,
                checkpoint_manager=checkpoint_manager,
            ),
        )
        if ret is not None:
          (config,
           step,
           state,
           mesh,
           checkpoint_manager,
           data_iterator,
           p_train_step,
           example_batch,
           learning_rate_schedule,
           metric_logger,
           writer) = ret

  if checkpoint_manager is not None:
    checkpoint_manager.wait_until_finished()
  metric_logger.write_metrics(running_gcs_metrics, metrics, config.steps - 1)  # final step metrics
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, recorder.record_job_end_time if recorder else None)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
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


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  if config.monitor_goodput and jax.process_index() == 0:
    logger_name = f"goodput_{config.run_name}"
    goodput_monitor = monitoring.GoodputMonitor(
        job_name=config.run_name,
        logger_name=logger_name,
        tensorboard_dir=config.tensorboard_dir,
        upload_interval=config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        pathway_enabled=config.enable_pathways_goodput,
        include_badput_breakdown=True,
        include_step_deviation=config.monitor_step_time_deviation,
        step_deviation_interval_seconds=config.step_deviation_interval_seconds,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard in the background!")
    if config.monitor_step_time_deviation:
      goodput_monitor.start_step_deviation_uploader()
      max_logging.log("Started step time deviation upload to Tensorboard in the background!")
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)


if __name__ == "__main__":
  app.run(main)
