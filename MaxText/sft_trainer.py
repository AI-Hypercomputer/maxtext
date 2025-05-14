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

import datetime
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
    create_goodput_recorder,
    eval_step,
    EPS,
    get_first_step,
    load_next_batch,
    record_goodput,
    record_scalar_metrics,
    save_checkpoint,
    setup_mesh_and_model,
    train_step,
    validate_train_config,
)


def setup_train_loop(config):
  """Set up prerequisites for the training loop -
    checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
    Set up data iterator and tokenizer, initialize the model.

  Args:
      config

  Returns:
      init_rng:
      writer: Summary writer for tensorboard
      checkpoint_manager: Orbax checkpointer
      state_mesh_annotations: the mesh annotations for the train state
      model:
      mesh:
      learning_rate_schedule:
      data_iterator:
      state: the initialized train state
  """
  recorder = create_goodput_recorder(config)

  record_goodput(recorder, config, recorder.record_tpu_init_start_time if recorder else None)
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  record_goodput(recorder, config, recorder.record_tpu_init_end_time if recorder else None)

  record_goodput(recorder, config, recorder.record_training_preparation_start_time if recorder else None)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )
  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)
  record_goodput(recorder, config, recorder.record_training_preparation_end_time if recorder else None)
  return (
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
  )


def train_loop(config, state=None):
  """Main training loop for SFT."""
  if not config.use_sft:
    raise TypeError("Set use_sft to True to run Supervised Fine Tuning.")

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
  maxtext_utils.add_config_to_summary_writer(config, writer)

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

    p_eval_step = None
    if eval_data_iterator:
      p_eval_step = jax.jit(
          functional_eval,
          in_shardings=in_shard_eval,
          out_shardings=out_shard_eval,
          static_argnums=static_argnums_eval,
          donate_argnums=donate_argnums_eval,
      )

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
  input_data_shardings = maxtext_utils.get_input_data_sharding(config, mesh)
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step or prof.should_activate_periodic_profile(step):
      optional_postfix = f"step_{step}" if config.profile_periodically_period > 0 else ""
      prof.activate(blocking_object=state, optional_postfix=optional_postfix)

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      record_goodput(recorder, config, recorder.record_data_loading_start_time if recorder else None)
      try:
        example_batch = load_next_batch(data_iterator, example_batch, config)
        example_batch = jax.lax.with_sharding_constraint(example_batch, input_data_shardings)
      except Exception as e:  # pylint: disable=broad-except
        max_logging.log(f"load_next_batch failed, you may have run out of data. Error message: {e}")
        break
      record_goodput(recorder, config, recorder.record_data_loading_end_time if recorder else None)
      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
      record_goodput(recorder, config, recorder.record_step_start_time if recorder else None, step)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, nextrng)

    step_time_delta = datetime.datetime.now() - last_step_completion
    last_step_completion = datetime.datetime.now()
    record_scalar_metrics(metrics, step_time_delta, per_device_tflops, learning_rate_schedule(step), per_device_tokens)
    if performance_metric_queue:
      performance_metric_queue.put(step_time_delta.total_seconds())

    if checkpoint_manager is not None:
      if save_checkpoint(checkpoint_manager, int(step), state, config.dataset_type, data_iterator, config):
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
        max_logging.log(f"Completed eval step {eval_step_count}")
        eval_step_count += 1
      eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
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

    if step == start_step:
      max_utils.print_mem_stats("After params initialized")

  if checkpoint_manager is not None:
    if (int(state.step) - 1) % config.checkpoint_period != 0:
      try:
        if save_checkpoint(
            checkpoint_manager, int(state.step) - 1, state, config.dataset_type, data_iterator, config, force=True
        ):
          checkpointing.print_save_message(int(state.step) - 1, config.async_checkpointing)
      except Exception:  # pylint: disable=broad-except
        max_logging.log(f"Checkpoint already saved for step {int(state.step)-1}.")

    checkpoint_manager.wait_until_finished()
  metric_logger.write_metrics(running_gcs_metrics, metrics, config.steps - 1)  # final step metrics
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, recorder.record_job_end_time if recorder else None)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
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

  if config.monitor_goodput and jax.process_index() == 0:
    logger_name = f"goodput_{config.run_name}"
    # Workload monitoring and Goodput monitoring both uses /workload/performance
    # GCM metric to publish step_time and step_deviation metrics. For now, we
    # will disable publishing step deviation metrics to GCM if workload
    # monitoring is enabled. Will reconcile this in the future.
    if config.report_performance_metric_for_gcp_monitoring:
      config.enable_gcp_step_deviation_metrics = False

    gcp_options = monitoring.GCPOptions(
        enable_gcp_goodput_metrics=config.enable_gcp_goodput_metrics,
        enable_gcp_step_deviation_metrics=config.enable_gcp_step_deviation_metrics,
    )
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
        gcp_options=gcp_options,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard & GCM in the background!")
    if config.monitor_step_time_deviation:
      goodput_monitor.start_step_deviation_uploader()
      max_logging.log("Started step time deviation upload to Tensorboard & GCM in the background!")

  train_loop(config)


if __name__ == "__main__":
  app.run(main)
