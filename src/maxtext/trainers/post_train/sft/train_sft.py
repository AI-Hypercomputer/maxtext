#  Copyright 2023-2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
SFT training script that calls a trainer in Tunix to run SFT on a MaxText model
using `HuggingFaceH4/ultrachat_200k` dataset. The configurations for the dataset
are defined inside `src/maxtext/configs/post_train/sft.yml`.

Example command:
Training & Evaluation:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL_NAME?} load_parameters_path=${CHECKPOINT_PATH?} \
    hf_access_token=${HF_ACCESS_TOKEN?} tokenizer_path=${TOKENIZER_PATH?} \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane weight_dtype=bfloat16

Training:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/maxtext/configs/post_train/sft.yml \
    run_name=${RUN_NAME?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    model_name=${MODEL_NAME?} load_parameters_path=${CHECKPOINT_PATH?} \
    hf_access_token=${HF_ACCESS_TOKEN?} tokenizer_path=${TOKENIZER_PATH?} \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=-1 steps=10 profiler=xplane weight_dtype=bfloat16
"""

from collections import defaultdict
from typing import Sequence, Dict

from absl import app
import os
import jax
import optax
import pathwaysutils

from flax.linen import partitioning as nn_partitioning

from orbax import checkpoint as ocp

from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft import metrics_logger, peft_trainer, profiler

from maxtext.configs import pyconfig
from maxtext.trainers.pre_train.train import loss_fn
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.optimizers import optimizers
from maxtext.trainers.post_train.sft import hooks
from maxtext.utils import max_utils
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


class SFTPerfTracer(perf_tracer_lib.PerfTracer):
  """Custom PerfTracer for SFT that aggregates step durations and supports clearing."""

  def export(self) -> Dict[str, float]:
    """Aggregates durations of all PEFT_TRAIN spans in the tracer."""
    # pylint: disable=protected-access
    timelines = self._get_timeline_snapshots()
    micro_batch_durations = defaultdict(float)

    for timeline in timelines.values():
      for span in timeline.spans.values():
        if span.name == perf_constants.PEFT_TRAIN:
          micro = span.tags.get(perf_constants.MICRO_BATCH, 0)
          micro_batch_durations[micro] = max(micro_batch_durations[micro], span.duration)

    total_duration = sum(micro_batch_durations.values())
    return {"step_time": total_duration}

  def clear(self):
    """Safely clears spans from all timelines in the tracer."""
    # pylint: disable=protected-access
    with self._timelines_lock:
      for tl in self._host_timelines.values():
        tl.spans.clear()
      for tl in self._device_timelines.values():
        tl.spans.clear()


def get_tunix_config(mt_config):
  """Gets the Tunix training configurations from the MaxText config.

  Args:
    mt_config: MaxText config.

  Returns:
    A Tunix `TrainingConfig` object.
  """
  # Checkpointing configurations
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=mt_config.checkpoint_period,
      enable_async_checkpointing=mt_config.async_checkpointing,
  )

  # Metrics configurations
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=mt_config.tensorboard_dir)

  # Profiler configurations
  profiler_options = None
  if mt_config.profiler:
    set_profile_options = True
    platform_version = jax.extend.backend.get_backend().platform_version.strip()
    if platform_version.startswith("Pathways"):
      max_logging.log("Pathways backend detected. Disabling setting profile options.")
      set_profile_options = False
    profiler_options = profiler.ProfilerOptions(
        log_dir=mt_config.tensorboard_dir,
        skip_first_n_steps=mt_config.skip_first_n_steps_for_profiler,
        profiler_steps=mt_config.profiler_steps,
        set_profile_options=set_profile_options,
    )

  return peft_trainer.TrainingConfig(
      eval_every_n_steps=mt_config.eval_interval,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps,
      checkpoint_root_directory=mt_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
  )


def use_maxtext_loss_function(trainer, mt_config):
  """Configures the trainer to use MaxText's loss function.

  This function creates a wrapper around MaxText's `loss_fn` to make it
  compatible with the Tunix trainer's expected loss function signature.

  Args:
    trainer: The PeftTrainer instance.
    mt_config: MaxText config.

  Returns:
    The trainer configured with the MaxText loss function.
  """

  def loss_func(model, inputs, inputs_position, inputs_segmentation, targets, targets_position, targets_segmentation):
    data = {
        "inputs": inputs,
        "inputs_position": inputs_position,
        "inputs_segmentation": inputs_segmentation,
        "targets": targets,
        "targets_position": targets_position,
        "targets_segmentation": targets_segmentation,
    }
    return loss_fn(model, mt_config, data, dropout_rng=None, params=None, is_train=True)

  trainer = trainer.with_loss_fn(loss_func, has_aux=True)
  return trainer


def setup_trainer_state(mt_config, goodput_recorder=None):
  """Set up prerequisites for training loop."""
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.create_nnx_model(mt_config)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    # pass in model for muon
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)

    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold),
          optimizer,
      )

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    # Initialize the Tunix performance tracer to be able to get step_time metrics in `training_hooks.on_train_step_end()`.
    perf_tracer_v2 = SFTPerfTracer(devices=mesh.devices)
    training_hooks = hooks.SFTTrainingHooks(
        mt_config, mesh, learning_rate_schedule, goodput_recorder, perf_tracer_v2=perf_tracer_v2
    )
    data_hooks = hooks.SFTDataHooks(mt_config, mesh, goodput_recorder)

    trainer = peft_trainer.PeftTrainer(model, optimizer, tunix_config, perf_tracer_v2=perf_tracer_v2)
    trainer.with_training_hooks(training_hooks)
    trainer.with_data_hooks(data_hooks)
    trainer = use_maxtext_loss_function(trainer, mt_config)

  return trainer, mesh


def train_model(mt_config, trainer, mesh):
  """Runs the SFT training loop in Tunix."""
  with mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    trainer.train(trainer.data_hooks.train_data_iterator, trainer.data_hooks.eval_data_iterator)
  return trainer


def train(mt_config, goodput_recorder=None):
  """Main method for SFT training.

  Args:
    mt_config: MaxText config.
    goodput_recorder: An optional GoodputRecorder to record performance metrics.
  """
  trainer, mesh = setup_trainer_state(mt_config, goodput_recorder)
  _job_completed_gracefully = False
  try:
    trainer = train_model(mt_config, trainer, mesh)
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  return trainer, mesh


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  goodput_recorder = create_goodput_recorder(mt_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
