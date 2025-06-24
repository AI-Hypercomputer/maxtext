# Copyright 2025 Google LLC
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

"""Training loop for elastic training model.

Elastic training via Pathways on Cloud allows for slices to go down without
crashing the main program. In this way, elastic events (slices going up and
down) can be caught or polled for and reacted to within an elastic_handler
function without causing the whole workload to restart. This involves
creating a new mesh with the available slices, reinitializing variables,
and recompiling functions in addition to restoring from a host offloaded
snapshot of the state.

The purpose of this training loop is to serve as an example for how to
support elasticity and is actively in development. As such, there are some
performance optimizations that have yet to be added as well as some features
not supported.

Current limitations:
- The host offloaded snapshot is currently blocking
- Elastic event handling during async checkpointing is not tested
- Elastic event handling during profiling is not tested
- Elastic manager configuration values are hard coded
- Debug logging statements for elasticity are hard coded as enabled

See https://github.com/AI-Hypercomputer/pathways-utils/tree/main/pathwaysutils/elastic
for more details about the elastic manager.
"""
from collections.abc import Sequence
import datetime
import logging
import os
import sys
import time

from absl import app

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

import jax

from flax.linen import partitioning as nn_partitioning

import pathwaysutils
from pathwaysutils.elastic import manager
from pathwaysutils.debug import timing

import tensorflow as tf

from MaxText import checkpointing
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import max_logging
from MaxText import profiler
from MaxText import pyconfig
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.metric_logger import MetricLogger
from MaxText.train import check_example_batch
from MaxText.train import get_first_step
from MaxText.train import load_next_batch
from MaxText.train import save_checkpoint
from MaxText.train import setup_mesh_and_model
from MaxText.train import setup_train_loop
from MaxText.train import train_step
from MaxText.train import validate_train_config
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager

logging.basicConfig()
logging.getLogger("pathwaysutils.elastic.manager").setLevel(logging.INFO)
logging.getLogger("pathwaysutils.debug.timing").setLevel(logging.DEBUG)


@timing.timeit
def elastic_handler(
    config: pyconfig.HyperParameters,
    elastic_manager,
    checkpoint_manager,
):
  """Reconfigures the workload onto the currently available slices.

  This is called by the elastic manager's maybe_reshard_up/down
  functions and is responsible for creating a new mesh,
  reinitializing the state and any objects that depend on the mesh.

  It returns all of the reinitialized objects.

  maybe_reshard_up/down take this function and its arguments and if
  there is an elastic event, those functions will call this function
  and return its returns.
  """
  # We reuse setup_mesh_and_model because it contains most of the
  # reconfiguration. Depending on the configuration, the checkpoint
  # manager depends on the mesh and must be recreated. Therefore, we
  # close the previous checkpoint manager and get a new checkpoint
  # manager from setup_mesh_and_model.
  if checkpoint_manager is not None:
    checkpoint_manager.close()

  with jax.default_device(elastic_manager.default_device):
    init_rng, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(
        config, elastic_manager.good_devices
    )
    with mesh:
      data_iterator, _ = create_data_iterator(config, mesh)

      step, snapshot_jax_arrays, _ = elastic_manager.get_resharded_snapshot(mesh)

      # We do not want to restore from the previous checkpoint but instead
      # restore from the host offloaded snapshot.
      if checkpoint_manager is not None:
        latest_step = checkpoint_manager.latest_step()

        # If we checkpointed after the latest snapshot, the checkpoint manager
        # will try to take another checkpoint and fail because it already
        # exists. Therefore, we delete the checkpoint and let the checkpoint
        # manager re-take the checkpoint.
        if latest_step is not None and latest_step >= step:
          max_logging.log(f"Deleting checkpoint from step {latest_step} since we are rewinding to step {step}.")
          checkpoint_manager.delete(latest_step)

      state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
          model,
          data_iterator,
          tx,
          config,
          jax.random.fold_in(init_rng, step),
          mesh,
          checkpoint_manager=None,
      )

      state = state.replace(**snapshot_jax_arrays)
      state = state.replace(step=state.step.at[None].set(step))

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

      example_batch = None
      metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

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
  )


def train_loop(config, elastic_manager, recorder, state=None):
  """Main Training loop."""
  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      _,
      state,
  ) = setup_train_loop(config, recorder)

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

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)

  example_batch = None
  step = start_step

  elastic_manager.maybe_snapshot(
      step,
      snapshot_jax_arrays={
          "params": state.params,
          "opt_state": state.opt_state,
      },
      force=True,
      block=True,
  )

  input_data_shardings = maxtext_utils.get_input_data_sharding(config, mesh)

  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  # Using while loop instead of a for loop because with elasticity
  # the step is restored back to the latest snapshot when a slice is lost
  while step < config.steps:
    try:
      step_start_time = datetime.datetime.now()
      prof.maybe_activate_profiler(step, state)

      max_logging.log(f"{step=} {elastic_manager.elastic_down_event_count=} {elastic_manager.good_slice_count=}")
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules), jax.default_device(elastic_manager.default_device):
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          with maybe_record_goodput(recorder, GoodputEvent.DATA_LOADING):
            try:
              example_batch = load_next_batch(data_iterator, example_batch, config)
              example_batch = jax.lax.with_sharding_constraint(example_batch, input_data_shardings)
            except Exception as e:  # pylint: disable=broad-except
              max_logging.log(f"load_next_batch failed, you may have run out of data. Error message: {e}")
              break

          check_example_batch(config, example_batch=example_batch)
          # pylint: disable=not-callable
          nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
          with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
            state, metrics = p_train_step(state, example_batch, nextrng)

        if checkpoint_manager is not None:
          state_to_save = state
          if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
            checkpointing.print_save_message(step, config.async_checkpointing)

          # Upon preemption, exit when and only when all ongoing saves are complete.
          if checkpoint_manager.reached_preemption(step):
            checkpoint_manager.wait_until_finished()
            sys.exit()

        prof.maybe_deactivate_profiler(step, state)

      elastic_manager.maybe_snapshot(
          step=step,
          snapshot_jax_arrays={
              "params": state.params,
              "opt_state": state.opt_state,
          },
          block=True,
      )

      ret = elastic_manager.maybe_reshard_up(
          step=step,
          snapshot_jax_arrays={
              "params": state.params,
              "opt_state": state.opt_state,
          },
          elastic_handler=elastic_handler,
          handler_kwargs={
              "config": config,
              "elastic_manager": elastic_manager,
              "checkpoint_manager": checkpoint_manager,
          },
      )
      if ret is not None:
        (
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
        ) = ret

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      jax.block_until_ready(state)  # ensure training step is completed

      step_time_delta = datetime.datetime.now() - step_start_time
      metric_logger.record_train_metrics(metrics, step, step_time_delta)

      step += 1

    except jax.errors.JaxRuntimeError as error:
      ret = elastic_manager.maybe_reshard_down(
          error=error,
          elastic_handler=elastic_handler,
          handler_kwargs={
              "config": config,
              "elastic_manager": elastic_manager,
              "checkpoint_manager": checkpoint_manager,
          },
      )
      if ret is not None:
        (
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
        ) = ret

  if checkpoint_manager is not None:
    if ((int(state.step) - 1) % config.checkpoint_period != 0) and (int(state.step) != 0):
      try:
        state_to_save = state
        if save_checkpoint(
            checkpoint_manager,
            int(state.step) - 1,
            state_to_save,
            config.dataset_type,
            data_iterator,
            config,
            force=True,
        ):
          checkpointing.print_save_message(int(state.step) - 1, config.async_checkpointing)
      except Exception:  # pylint: disable=broad-except
        max_logging.log(f"Checkpoint is already saved for step {int(state.step)-1}.")

    checkpoint_manager.wait_until_finished()
  metric_logger.cleanup()

  if example_batch:
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


def wait_for_all_slices(elastic_manager: manager.Manager) -> None:
  elastic_manager.good_slice_indices = elastic_manager.get_slice_availability()
  while len(elastic_manager.good_slice_indices) < elastic_manager.total_slice_count:
    max_logging.log(
        f"Only {elastic_manager.good_slice_count} slices out of {elastic_manager.total_slice_count} available. "
        "Sleeping for 5 seconds."
    )
    time.sleep(5)
    elastic_manager.good_slice_indices = elastic_manager.get_slice_availability()
  max_logging.log("All slices are available")


def elastic_initialize(devices: Sequence[jax.Device]) -> manager.Manager:
  """Initializes the elastic manager and pyconfig to support elastic training

  Args:
    devices: The devices used for training

  Returns:
    The initialized elastic manager
  """
  elastic_manager = manager.Manager(
      devices,
      reshard_check_period=1,
      snapshot_period=5,
      max_elastic_down_event_count=100,
      max_reshard_retry_count=3,
  )

  # Do not start training until all slices are available
  # TODO: b/408455557 - Migrate to pathwaysutils and make configurable
  wait_for_all_slices(elastic_manager)

  pyconfig.HyperParameters.global_batch_size_to_train_on = property(
      lambda self: elastic_manager.scale_by_good_slices(self.get_keys()["global_batch_size_to_train_on"])
  )
  pyconfig.HyperParameters.global_batch_size_to_load = property(
      lambda self: elastic_manager.scale_by_good_slices(self.get_keys()["global_batch_size_to_load"])
  )
  pyconfig.HyperParameters.micro_batch_size_to_train_on = property(
      lambda self: elastic_manager.scale_by_good_slices(self.get_keys()["micro_batch_size_to_train_on"])
  )
  pyconfig.HyperParameters.num_slices = property(lambda self: elastic_manager.good_slice_count)

  return elastic_manager


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"

  elastic_manager = elastic_initialize(jax.devices())

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Goodput configurations
  maybe_monitor_goodput(config)
  recorder = create_goodput_recorder(config)

  # Stack traces configurations
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)

  with diagnostic.diagnose(diagnostic_config):
    with maybe_record_goodput(recorder, GoodputEvent.JOB):
      train_loop(config, elastic_manager, recorder)


if __name__ == "__main__":
  app.run(main)
