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

from elastic import utils

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


@utils.timeit
def elastic_handler(
    config: pyconfig.HyperParameters,
    step: int,
    state,
    mesh: jax.sharding.Mesh,
    checkpoint_manager,
    data_iterator,
    p_train_step,
    example_batch,
    learning_rate_schedule,
    metric_logger,
    writer,
):
  """Reshard function."""
  if checkpoint_manager is not None:
    checkpoint_manager.close()

  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = (
      setup_mesh_and_model(config)
  )

  data_iterator, _ = create_data_iterator(config, mesh)

  restore_step = config.eu.data["save_step"]

  if checkpoint_manager is not None:
    # Confirm this is the right thing to do
    latest_step = checkpoint_manager.latest_step()
    max_logging.log(f"{latest_step=}")
    if latest_step is not None and latest_step >= restore_step:
      max_logging.log(
          f"Deleting checkpoint from step {latest_step} since we are "
          f"rewinding to step {restore_step}."
      )
      checkpoint_manager.delete(latest_step)

  state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
      model,
      data_iterator,
      tx,
      config,
      jax.random.fold_in(init_rng, restore_step),
      mesh,
      checkpoint_manager=None,
  )

  def reshard(arr):
    sharding_pinned_host = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(mesh, x.sharding.spec, memory_kind="pinned_host"),
        arr,
    )
    resharded_pinned_host = config.eu.reshard(
        arr,
        sharding_pinned_host,
        put_array=config.eu.put_array_device_put2,
        donate_input=True,
    )

    sharding_device = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(mesh, x.sharding.spec, memory_kind="device"),
        resharded_pinned_host,
    )

    resharded_device = config.eu.reshard(
        resharded_pinned_host,
        sharding_device,
        put_array=config.eu.put_array_device_put0,
        donate_input=False,
    )

    return resharded_pinned_host, resharded_device

  config.eu.data["params"], params = reshard(config.eu.data["params"])
  config.eu.data["opt_state"], opt_state = reshard(config.eu.data["opt_state"])

  state = state.replace(step=restore_step, params=params, opt_state=opt_state)
  jax.block_until_ready(state)

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
      restore_step,
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

  if config.use_dpo:
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
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config)

  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  max_utils.add_config_to_summary_writer(config, writer)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
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

  config.eu.maybe_snapshot(
      step,
      force=True,
      params=state.params,
      opt_state=state.opt_state,
  )

  # step_down = {10, 30, 44}
  # step_up = {14, 16, 40, 45}
  while True:
    with utils.watchdog(120):
      try:
        # if step in step_down:
        #   step_down.remove(step)
        #   # Remove a slice
        #   config.eu.update_good_slice_indices(set(range(config.eu.total_slice_count)) - {step % config.eu.total_slice_count})
        #   raise jax.errors.JaxRuntimeError("DATA_LOSS: Fake")
        # elif step in step_up:
        #   step_up.remove(step)

        #   config.eu.update_good_slice_indices(set(range(config.eu.total_slice_count)))


        if step == first_profiling_step or prof.should_activate_periodic_profile(step):
          optional_postfix = f"step_{step}" if config.profile_periodically_period > 0 else ""
          prof.activate(blocking_object=state, optional_postfix=optional_postfix)

        if step >= config.steps:
          break

        max_logging.log(f"{step=} {config.eu.failure_count=} {config.eu.good_slice_count=}")
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules), jax.default_device(config.eu.default_device):
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
            state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
            if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
              checkpointing.print_save_message(step, config.async_checkpointing)

            # Upon preemption, exit when and only when all ongoing saves are complete.
            if checkpoint_manager.reached_preemption(step):
              checkpoint_manager.wait_until_finished()
              sys.exit()

          metric_logger.write_metrics(running_gcs_metrics, metrics, step)

          if config.dump_hlo and step == start_step:
            jax.block_until_ready(state)  # Ensure compilation has finished.
            max_utils.upload_dump(
                config.dump_hlo_local_dir,
                config.dump_hlo_gcs_dir,
                module_name=config.dump_hlo_module_name,
                delete_local_after=config.dump_hlo_delete_local_after,
                all_host_upload=config.dump_hlo_upload_all,
            )

          if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
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
              if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
                break
              with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                eval_metrics = p_eval_step(state, eval_batch, nextrng)
              cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
              cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
              cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
              eval_dpo_reward_accuracy += float(eval_metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))  # for dpo only
              max_logging.log(f"Completed eval step {eval_step_count}")
              eval_step_count += 1
            eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
                cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
            )
            cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
            cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
                cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
            )
          if config.use_dpo:
            cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = eval_dpo_reward_accuracy / eval_step_count
            metric_logger.write_metrics(running_gcs_metrics, cumulative_eval_metrics, step, is_training=False)
            max_logging.log(
                f"average loss after {step=}: {eval_step_count=}, {eval_loss=},"
                f" total_weights={cumulative_eval_metrics['scalar']['eval/total_weights']}"
            )
            if eval_loss <= config.target_eval_loss:
              max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
              prof.deactivate()
              break

          if step == last_profiling_step or prof.should_deactivate_periodic_profile(step):
            prof.deactivate(blocking_object=state)

          config.eu.maybe_snapshot(
              step,
              params=state.params,
              opt_state=state.opt_state,
          )

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
           writer) = config.eu.maybe_reshard_up(
               step,
               elastic_handler,
               save_args=dict(
                   params=state.params,
                   opt_state=state.opt_state,
               ),
               handler_args=(
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
               ),
           )

          if step == start_step:
            max_utils.print_mem_stats("After params initialized")

          step += 1

      except jax.errors.JaxRuntimeError as error:
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
         writer) = config.eu.maybe_reshard_down(
             error,
             elastic_handler,
             handler_args=(
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
             ),
         )

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
