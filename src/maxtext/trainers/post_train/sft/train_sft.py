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
are defined inside `src/MaxText/configs/sft.yml`.

Example command:
Training & Evaluation:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane weight_dtype=bfloat16

Training:
  python3 -m maxtext.trainers.post_train.sft.train_sft src/MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=-1 steps=10 profiler=xplane weight_dtype=bfloat16
"""

from typing import Sequence

from absl import app
import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import pathwaysutils

from flax.linen import partitioning as nn_partitioning

from orbax import checkpoint as ocp

from tunix.sft import metrics_logger, peft_trainer, profiler
from tunix.rl import reshard

from MaxText import optimizers
from MaxText import pyconfig
from MaxText.train import loss_fn
from maxtext.common.goodput import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from maxtext.trainers.post_train.sft import hooks
from maxtext.utils import max_utils
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


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


def maybe_apply_lora(model, mesh, mt_config):
  """Optionally applies LoRA/QLoRA to a MaxText model using Qwix."""
  if not getattr(mt_config, "enable_lora", False):
    return model

  import qwix

  if mt_config.lora_rank <= 0:
    raise ValueError("enable_lora is True but lora_rank is not set to a positive value.")
  if not mt_config.lora_module_path:
    raise ValueError("enable_lora is True but lora_module_path is empty.")

  lora_kwargs = {
      "module_path": mt_config.lora_module_path,
      "rank": mt_config.lora_rank,
      "alpha": mt_config.lora_alpha,
  }
  if mt_config.lora_tile_size is not None:
    lora_kwargs["tile_size"] = mt_config.lora_tile_size
  if mt_config.lora_weight_qtype is not None:
    lora_kwargs["weight_qtype"] = mt_config.lora_weight_qtype
    max_logging.log("QLoRA is enabled with weight_qtype=%s", mt_config.lora_weight_qtype)
  else:
    max_logging.log("LoRA is enabled.")

  lora_provider = qwix.LoraProvider(**lora_kwargs)

  batch_size = getattr(mt_config, "per_device_batch_size", 1)
  seq_len = getattr(mt_config, "max_target_length", 1)
  if batch_size <= 0 or seq_len <= 0:
    raise ValueError(
        "per_device_batch_size and max_target_length must be positive when LoRA is enabled."
    )

  devices_data_fsdp = 1
  if mesh is not None:
    devices_data_fsdp = mesh.shape.get("data", 1) * mesh.shape.get("fsdp", 1)

  dummy_bs = (max(batch_size, devices_data_fsdp) + devices_data_fsdp - 1) // devices_data_fsdp
  dummy_bs *= devices_data_fsdp

  decoder_input_tokens = jnp.zeros((dummy_bs, seq_len), dtype=jnp.int32)
  decoder_positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (dummy_bs, seq_len))

  lora_model = qwix.apply_lora_to_model(
    model,
    lora_provider,
    decoder_input_tokens=decoder_input_tokens,
    decoder_positions=decoder_positions,
  )

  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)

  def _key_to_str(key):
    if isinstance(key, str):
      return key
    if hasattr(key, "name"):
      return str(key.name)
    return str(key)

  lora_state = nnx.state(lora_model, nnx.LoRAParam)
  lora_leaves = jax.tree_util.tree_leaves(lora_state)
  lora_count = len(lora_leaves)

  if lora_count == 0:
    full_state = nnx.state(lora_model)
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(full_state)
    for path, _ in paths_and_leaves:
      path_str = "/".join(_key_to_str(k) for k in path).lower()
      if "lora" in path_str or "adapter" in path_str:
        lora_count += 1

  if lora_count == 0:
    module_paths = []
    for path, _ in lora_model.iter_modules():
      module_paths.append("/".join(str(p) for p in path))
      if len(module_paths) >= 50:
        break
    max_logging.log(
      f"LoRA module_path='{mt_config.lora_module_path}' did not match any weights. "
      f"Sample module paths: {module_paths}"
    )
    raise ValueError("LoRA enabled but no LoRA parameters found in decoder/model state.")
  max_logging.log("LoRA verification: found %d LoRA parameter entries.", lora_count)

  return lora_model


def setup_trainer_state(mt_config, goodput_recorder=None):
  """Set up prerequisites for training loop."""
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.create_nnx_model(mt_config)
    model = maybe_apply_lora(model, mesh, mt_config)
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

    trainer = peft_trainer.PeftTrainer(model, optimizer, tunix_config)
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
  trainer = train_model(mt_config, trainer, mesh)
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

  with maybe_record_goodput(goodput_recorder, GoodputEvent.JOB), maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
