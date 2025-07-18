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
SFT training script that calls a trainer in Tunix to run SFT on a MaxText model.

This script sets up the MaxText model, data iterators, and hooks, and then
passes them to the Tunix `PeftTrainer` to execute the training loop.

Example command:
  python3 -m MaxText.sft.sft_trainer MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=llama2-7b load_parameters_path=gs://maxtext-model-checkpoints/llama2-7b-chat/scanned/0/items \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=meta-llama/Llama-2-7b-chat-hf \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=5
"""

from typing import Sequence

from absl import app
from flax import nnx
import jax
from tunix.sft import peft_trainer, profiler
import optax
import os

from orbax import checkpoint as ocp
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText.sft import hooks
from MaxText import pyconfig
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.integrations.tunix.tunix_utils import build_tunix_wrapper


def get_tunix_config(mt_config):
  # Checkpointing configurations
  checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=mt_config.checkpoint_period,
    enable_async_checkpointing=mt_config.async_checkpointing,
  )

  # Metrics configurations
  metrics_logging_options = peft_trainer.metrics_logger.MetricsLoggerOptions(
      log_dir=mt_config.tensorboard_dir,
      flush_every_n_steps=1
  )

  # Profiler configurations
  profiler_options = None
  if mt_config.profiler:
    profiler_options = profiler.ProfilerOptions(
      log_dir=mt_config.tensorboard_dir,
      skip_first_n_steps=mt_config.skip_first_n_steps_for_profiler,
      profiler_steps=mt_config.profiler_steps)

  return peft_trainer.TrainingConfig(
      eval_every_n_steps=mt_config.eval_interval,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps,
      checkpoint_root_directory=mt_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
  )

def get_maxtext_nnx_model(mt_config):
  rngs = nnx.Rngs(1234)
  model = build_tunix_wrapper(
      mt_config,
      rngs,
      enable_dropout=False,   # deterministic SFT (you can override at runtime)
      init_batch_size=1,
      init_seq_len=1,
      use_attention_mask=False,  # trust Tunix loss masking
  )
  mesh  = model.base.mesh

  # Add these lines to properly get the graph definition and state
  graphdef, state = nnx.split(model)
  model = nnx.merge(graphdef, state)  # Recreate model in proper NNX format
  return model.base, mesh

def gen_model_input_fn(x):
  return {
      "input_tokens": x["inputs"],
      "input_mask": x["inputs_segmentation"],
      "positions": x["inputs_position"],
      "attention_mask": x["inputs_segmentation"],
  }

def run_sft_training(mt_config):
  model, mesh = get_maxtext_nnx_model(mt_config)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
  training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule)

  train_data_iterator, eval_data_iterator = create_data_iterator(mt_config, mesh)
  data_hooks = hooks.SFTDataHooks(mt_config, mesh, train_data_iterator)

  tunix_config = get_tunix_config(mt_config)
  trainer = peft_trainer.PeftTrainer(model, optax.sgd(learning_rate_schedule), tunix_config, training_hooks=training_hooks, data_hooks=data_hooks)
  trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)
  trainer.train(train_data_iterator, eval_data_iterator)

def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"

  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  run_sft_training(mt_config)

if __name__ == "__main__":
  app.run(main)
