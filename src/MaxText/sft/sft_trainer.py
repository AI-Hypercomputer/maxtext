#  Copyright 2025 Google LLC
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
  python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane weight_dtype=bfloat16

Training:
  python3 -m MaxText.sft.sft_trainer src/MaxText/configs/sft.yml \
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
import pathwaysutils
from flax import nnx

from flax.linen import partitioning as nn_partitioning

from orbax import checkpoint as ocp

from tunix.sft import peft_trainer, profiler
import qwix

from MaxText import max_utils
from MaxText import max_logging
from MaxText import maxtext_utils
from MaxText import optimizers
from MaxText import pyconfig
from MaxText import model_creation_utils
from MaxText.train import loss_fn
from MaxText.sft import hooks
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)


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
  metrics_logging_options = peft_trainer.metrics_logger.MetricsLoggerOptions(log_dir=mt_config.tensorboard_dir)

  # Profiler configurations
  profiler_options = None
  if mt_config.profiler:
    set_profile_options = True
    if jax.extend.backend.get_backend().platform_version == "Pathways":
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


def create_model_input_for_lora(base_model):
  """Creates dummy model input for LoRA tracing.
  
  Creates the model input structure that qwix expects to trace through
  the model's computation graph when applying LoRA adapters.
  
  The parameter names must exactly match the Transformer's __call__ signature.
  
  Args:
    base_model: The base model to extract configuration from.
    
  Returns:
    A dictionary with keys matching Transformer.__call__ parameters:
    'decoder_input_tokens', 'decoder_positions', 'decoder_segment_ids', 'cache'.
  """
  # Get batch and sequence length from model config
  batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(
      config=base_model.config,
      model_mode=base_model.model_mode
  )
  
  # Create dummy inputs matching Transformer.__call__ signature
  return {
      "decoder_input_tokens": jnp.ones(
          (batch_size, seq_len), dtype=jnp.int32
      ),
      "decoder_positions": jnp.ones(
          (batch_size, seq_len), dtype=jnp.int32
      ),
      # "cache": None,
  }


def apply_lora_to_model(base_model, mesh, mt_config, quantize=False):
  """Applies LoRA to the base model.

  Args:
    base_model: The base MaxText model to apply LoRA to.
    mesh: The device mesh for sharding.
    mt_config: MaxText config containing LoRA parameters.
    quantize: Whether to use quantized LoRA (NF4).

  Returns:
    The model with LoRA applied and properly sharded.
  """
  # Extract LoRA parameters from config
  rank = getattr(mt_config, "lora_rank", 8)
  alpha = getattr(mt_config, "lora_alpha", 16)
  
  # Define which modules to apply LoRA to
  module_path = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj"
  
  if quantize:
    lora_provider = qwix.LoraProvider(
        module_path=module_path,
        rank=rank,
        alpha=alpha,
        weight_qtype="nf4",
        tile_size=256,
    )
  else:
    lora_provider = qwix.LoraProvider(
        module_path=module_path,
        rank=rank,
        alpha=alpha,
    )

  # Get model inputs from the model itself, just like Gemma's pattern:
  # model_input = base_model.get_model_input()
  # lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)
  #
  # The get_model_input() method returns dummy inputs that match the model's expected signature.
  # For NNX models, qwix needs to call the model to trace the computation graph.
  model_input = create_model_input_for_lora(base_model)
  
  # Apply LoRA to the model using the inputs from create_model_input_for_lora()
  lora_model = qwix.apply_lora_to_model(
      base_model, 
      lora_provider,
      **model_input
  )

  # Apply sharding constraints
  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


def train(mt_config, goodput_recorder=None):
  """Runs the SFT training loop.

  Args:
    mt_config: MaxText config.
    goodput_recorder: An optional GoodputRecorder to record performance metrics.
  """
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = model_creation_utils.create_nnx_model(mt_config)
    
    # Apply LoRA if enabled
    use_lora = getattr(mt_config, "use_lora", False)
    if use_lora:
      max_logging.log("Applying LoRA to the model...")
      quantize_lora = getattr(mt_config, "quantize_lora", False)
      model = apply_lora_to_model(model, mesh, mt_config, quantize=quantize_lora)
      max_logging.log("LoRA applied successfully")
    
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule, goodput_recorder)
    data_hooks = hooks.SFTDataHooks(mt_config, mesh, goodput_recorder)

    trainer = peft_trainer.PeftTrainer(model, optimizer, tunix_config)
    trainer.with_training_hooks(training_hooks)
    trainer.with_data_hooks(data_hooks)
    trainer = use_maxtext_loss_function(trainer, mt_config)

  with mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    trainer.train(data_hooks.train_data_iterator, data_hooks.eval_data_iterator)

  return trainer, mesh


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  maybe_monitor_goodput(mt_config)
  goodput_recorder = create_goodput_recorder(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.JOB):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
