# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Training loop for Supervised Fine-Tuning (SFT)."

from typing import Sequence

from absl import app

import numpy as np

import tensorflow as tf
import jax

from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.trainers.pre_train.train import (
    eval_step,
    get_first_step,
    train_step,
)
from maxtext.common import checkpointing, profiler
from maxtext.common.data_loader import DataLoader
from maxtext.common.goodput import (
    RECORD_JOB_START_TIME,
    maybe_monitor_goodput,
    record_goodput,
)
from maxtext.common.metric_logger import MetricLogger
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_utils
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils


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
      _,
      _,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  # NNX jits over the GraphDef + a flat nnx.State, so split the TrainStateNNX
  # here (mirrors trainers/pre_train/train.py). Linen jits over the module.
  if config.pure_nnx:
    jit_model, state = nnx.split(state)
  else:
    jit_model = model

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config, jit_model, mesh, state, state_mesh_shardings, train_step, eval_step, eval_data_iterator, params_shardings
  )

  # Only the Linen step takes a dropout rng; pass it only there so the args
  # match the jitted in_shardings (see get_functional_train_with_signature).
  rng_args = () if config.pure_nnx else (init_rng,)

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    data_sharding = sharding.get_input_data_sharding(config, mesh)
    shaped_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding)
    compiled = p_train_step.lower(state, shaped_batch, *rng_args).compile()
    compiled_stats = compiled.memory_analysis()
    max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(model, state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  data_loader = DataLoader(config, mesh, data_iterator, recorder)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  if config.pure_nnx:
    _, setup_params, _ = nnx.split(state.model, nnx.Param, ...)
  else:
    setup_params = state.params
  metric_logger.write_setup_info_to_tensorboard(setup_params)

  _job_completed_gracefully = False
  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch()
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        step_rng_args = () if config.pure_nnx else (nextrng,)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            state, metrics = p_train_step(state, example_batch, *step_rng_args)

      step_time_delta = datetime.datetime.now() - last_step_completion

      checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step)

      if config.dump_hlo and step == start_step:
        jax.block_until_ready(state)  # Ensure compilation has finished.
        gcs_utils.upload_dump(
            config.dump_hlo_local_dir,
            config.dump_hlo_gcs_dir,
            module_name=config.dump_hlo_module_name,
            delete_local_after=config.dump_hlo_delete_local_after,
            all_host_upload=config.dump_hlo_upload_all,
        )

      if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
        assert eval_data_iterator
        # Explicitly reset the eval iterator and counters before starting the eval loop
        eval_data_iterator.reset()
        metric_logger.reset_eval_metrics()
        max_logging.log(f"Starting eval after train step {step}")
        eval_step_count = 0
        last_eval_step_completion = datetime.datetime.now()
        # pylint: disable=not-callable
        for eval_batch in eval_data_iterator:
          if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
            break
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
          eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
          last_eval_step_completion = datetime.datetime.now()
          metric_logger.buffer_and_write_metrics(
              eval_metrics, eval_step_count, step_time_delta=eval_step_time_delta, is_training=False
          )
          max_logging.log(f"Completed eval step {eval_step_count}")
          eval_step_count += 1

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      last_step_completion = datetime.datetime.now()
      metric_logger.buffer_and_write_metrics(metrics, step, step_time_delta)

    if config.save_checkpoint_on_completion:
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator)
    elif checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
      checkpoint_manager.wait_until_finished()
    _job_completed_gracefully = True
  except exceptions.StopTraining as e:
    prof.deactivate()
    max_logging.log(f"Training stopped: {str(e)}")
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
    metric_logger.flush_metrics_and_cleanup()

  return state


def main(argv: Sequence[str]) -> None:
  argv = list(argv)
  argv.append("use_sft=True")
  argv.append("use_tunix_gradient_accumulation=False")
  config, recorder = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
