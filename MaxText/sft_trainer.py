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
from typing import Sequence

from absl import app

import numpy as np

import tensorflow as tf

import jax

from flax.linen import partitioning as nn_partitioning

from MaxText import checkpointing
from MaxText import exceptions
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import max_logging
from MaxText import profiler
from MaxText import pyconfig
from MaxText.data_loader import DataLoader
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
    eval_step,
    get_first_step,
    save_checkpoint,
    setup_train_loop,
    train_step,
    validate_train_config,
)
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)


def train_loop(config, recorder, state=None):
  """Main training loop for SFT."""
  if not config.use_sft:
    raise TypeError("Set use_sft to True to run Supervised Fine Tuning.")

  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
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

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config)

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

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)

  example_batch = None

  data_loader = DataLoader(config, mesh, data_iterator, recorder)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  try:
    for step in np.arange(start_step, config.steps):
      step_start_time = datetime.datetime.now()
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch()
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            state, metrics = p_train_step(state, example_batch, nextrng)

      if checkpoint_manager is not None:
        if save_checkpoint(checkpoint_manager, int(step), state, config.dataset_type, data_iterator, config):
          checkpointing.print_save_message(step, config.async_checkpointing)

        # Upon preemption, exit when and only when all ongoing saves are complete.
        if checkpoint_manager.reached_preemption(step):
          checkpoint_manager.wait_until_finished()
          sys.exit()

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
        eval_step_count = 0
        # pylint: disable=not-callable
        for eval_batch in eval_data_iterator:
          if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
            break
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(state, eval_batch, nextrng)
          metric_logger.record_eval_metrics(step, metrics=eval_metrics)
          max_logging.log(f"Completed eval step {eval_step_count}")
          eval_step_count += 1
        metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)
        if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
          prof.deactivate()
          raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      jax.block_until_ready(state)  # ensure training step is completed

      step_time_delta = datetime.datetime.now() - step_start_time
      metric_logger.record_train_metrics(metrics, step, step_time_delta)
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")

  if checkpoint_manager is not None:
    if ((int(state.step) - 1) % config.checkpoint_period != 0) and (int(state.step) != 0):
      try:
        if save_checkpoint(
            checkpoint_manager, int(state.step) - 1, state, config.dataset_type, data_iterator, config, force=True
        ):
          checkpointing.print_save_message(int(state.step) - 1, config.async_checkpointing)
      except Exception:  # pylint: disable=broad-except
        max_logging.log(f"Checkpoint already saved for step {int(state.step)-1}.")

    checkpoint_manager.wait_until_finished()
  metric_logger.cleanup()

  if example_batch:
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

  maybe_monitor_goodput(config)
  recorder = create_goodput_recorder(config)
  with maybe_record_goodput(recorder, GoodputEvent.JOB):
    train_loop(config, recorder)


if __name__ == "__main__":
  app.run(main)
