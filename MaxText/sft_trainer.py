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
from typing import Sequence

from absl import app

import numpy as np

import tensorflow as tf

import jax

from flax.linen import partitioning as nn_partitioning

from MaxText import checkpointing
from MaxText import exceptions
from MaxText import max_utils
from MaxText import max_logging
from MaxText import maxtext_utils
from MaxText import profiler
from MaxText import pyconfig
from MaxText import train_utils
from MaxText.data_loader import DataLoader
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
    eval_step,
    get_first_step,
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

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config, model, mesh, state, state_mesh_shardings, train_step, eval_step, eval_data_iterator
  )

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
    compiled_stats = compiled.memory_analysis()
    max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
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

      checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step)

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

    checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator)
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
  finally:
    metric_logger.cleanup()

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
