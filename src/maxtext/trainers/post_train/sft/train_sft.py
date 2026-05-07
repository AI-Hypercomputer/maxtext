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

import inspect
from typing import Any, Sequence

from absl import app
import os
import jax
import optax
import pathwaysutils

from flax import nnx
from flax.linen import partitioning as nn_partitioning

from orbax import checkpoint as ocp

from tunix.sft import metrics_logger, peft_trainer, profiler

from maxtext.optimizers import optimizers
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
from maxtext.trainers.post_train.sft import hooks
from maxtext.utils import lora_utils
from maxtext.utils import max_utils
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


class MaxTextPeftTrainer(peft_trainer.PeftTrainer):
  """MaxText-specific PeftTrainer that avoids nested NNX transformations.

  Tunix's default PeftTrainer._train_step creates nnx.value_and_grad inside
  nnx.jit. This nesting causes Flax NNX to assign conflicting outer_index
  values to graph nodes, resulting in:
    ValueError: The graph structure of a node added to cached_partial was
    mutated inside the transformation.

  This subclass overrides create_train_step_fn to use jax.value_and_grad
  with an explicit split/merge pattern (matching MaxText's pre-training NNX
  train_step), which avoids the nested NNX transformation issue entirely.
  """

  def create_train_step_fn(self):
    """Creates a train step using jax.value_and_grad with explicit NNX split/merge."""
    loss_fn_ref = self.loss_fn
    has_aux = self._has_aux
    gen_fn = self.gen_model_input_fn
    is_lora_enabled = self._lora_enabled
    wrt = nnx.LoRAParam if is_lora_enabled else nnx.Param

    # Detect whether Tunix's train() expects (loss, aux, grad_norm) or just
    # (loss, aux) by inspecting the source of PeftTrainer._train_step.
    tunix_expects_grad_norm = False
    try:
      source = inspect.getsource(peft_trainer.PeftTrainer._train_step)  # pylint: disable=protected-access
      tunix_expects_grad_norm = "grad_norm" in source
    except (TypeError, OSError):
      pass

    # Capture the graphdef once outside of JIT so that split/merge inside
    # jax.value_and_grad can use a stable (non-traced) structural descriptor.
    graphdef, _, _ = nnx.split(self.model, wrt, ...)

    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any):
      inputs = gen_fn(inputs)

      # Split model into differentiable params and non-differentiable rest.
      # Using jax.value_and_grad (not nnx.value_and_grad) avoids nesting NNX
      # transforms inside nnx.jit, which would corrupt outer_index tracking.
      _, diff_params, rest = nnx.split(model, wrt, ...)

      def loss_wrapper(diff_params, rest, **inputs_kw):
        local_model = nnx.merge(graphdef, diff_params, rest, copy=True)
        out = loss_fn_ref(local_model, **inputs_kw)
        # Capture updated non-param state (e.g. RNG counters) from local_model.
        _, _, new_rest = nnx.split(local_model, wrt, ...)
        if has_aux:
          loss, aux = out
          return loss, (aux, new_rest)
        else:
          return out, (None, new_rest)

      grad_fn = jax.value_and_grad(loss_wrapper, argnums=0, has_aux=True)
      (out_val, (aux, new_rest)), grads = grad_fn(diff_params, rest, **inputs)

      # Propagate updated non-param state (RNG counters, etc.) back to model.
      nnx.update(model, new_rest)

      # Apply optimizer update. grads has the same nnx.State(wrt) structure
      # as diff_params, which is compatible with optimizer.update.
      optimizer.update(model, grads)

      aux_out = aux if has_aux else None
      if tunix_expects_grad_norm:
        return out_val, aux_out, optax.global_norm(grads)
      return out_val, aux_out

    return train_step


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
      data_sharding_axis=tuple(mt_config.data_sharding),
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

  def loss_func(
      model,
      inputs,
      inputs_position,
      inputs_segmentation,
      targets,
      targets_position,
      targets_segmentation,
  ):
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

    model, mesh = model_creation_utils.from_pretrained(mt_config)
    if mt_config.lora.enable_lora:
      model = lora_utils.apply_lora_to_model(model, mesh, mt_config)

    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    # pass in model for muon
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)

    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold),
          optimizer,
      )

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule, goodput_recorder)
    data_hooks = hooks.SFTDataHooks(mt_config, mesh, goodput_recorder)

    # Provide rules context so 'norm' is translated to mesh axes during maybe_restore
    with nn_partitioning.axis_rules(mt_config.logical_axis_rules):
      trainer = MaxTextPeftTrainer(model, optimizer, tunix_config)
      if mt_config.lora.lora_restore_path:
        trainer = lora_utils.restore_lora_from_path(trainer, mt_config)
      trainer.with_training_hooks(training_hooks)
      trainer.with_data_hooks(data_hooks)
      trainer = use_maxtext_loss_function(trainer, mt_config)

  return trainer, mesh


def train_model(mt_config, trainer, mesh):
  """Runs the SFT training loop in Tunix."""
  with mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    trainer.train(
        trainer.data_hooks.train_data_iterator,
        trainer.data_hooks.eval_data_iterator,
    )
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

  mt_config = pyconfig.initialize_pydantic(argv)
  max_utils.print_system_information()

  goodput_recorder = create_goodput_recorder(mt_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
