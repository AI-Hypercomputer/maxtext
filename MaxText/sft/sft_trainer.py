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

Example command:
Training & Evaluation:
  python3 -m MaxText.sft.sft_trainer MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane

Training:
  python3 -m MaxText.sft.sft_trainer MaxText/configs/sft.yml \
    run_name=$RUN_NAME base_output_directory=$BASE_OUTPUT_DIRECTORY \
    model_name=$MODEL_NAME load_parameters_path=$CHECKPOINT_PATH \
    hf_access_token=$HF_ACCESS_TOKEN tokenizer_path=$TOKENIZER_PATH \
    per_device_batch_size=1 max_target_length=1024 \
    eval_interval=-1 steps=10 profiler=xplane
"""

from typing import Sequence
from absl import app
from flax import linen as nn
from flax import nnx
from functools import partial
import jax
from tunix.sft import peft_trainer, profiler
import optax
import os
import MaxText as mt
from orbax import checkpoint as ocp
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adaptor import TunixMaxTextLlama
from MaxText.sft import hooks
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)


def get_tunix_config(mt_config):
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
    profiler_options = profiler.ProfilerOptions(
        log_dir=mt_config.tensorboard_dir,
        skip_first_n_steps=mt_config.skip_first_n_steps_for_profiler,
        profiler_steps=mt_config.profiler_steps,
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


def gen_model_input_fn(x):
  return {
      "input_tokens": x["inputs"],
      "input_mask": x["inputs_segmentation"],
      "positions": x["inputs_position"],
      "attention_mask": x["inputs_segmentation"],
  }


def get_maxtext_model(config):
  def create_model():
    return mt.from_pretrained(config, rngs=nnx.Rngs(params=0, dropout=1))

  abstract_model = nnx.eval_shape(create_model)
  graphdef, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)
  mesh = abstract_model.mesh

  # JIT a function that creates the model state with proper sharding from the start.
  # By providing out_shardings, we instruct JAX to produce sharded output directly,
  # avoiding a large intermediate allocation on a single device.
  with nn.logical_axis_rules(config.logical_axis_rules):
    out_shardings = nn.logical_to_mesh_sharding(specs, mesh)

  @partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_state():
    # This will be JIT-compiled. JAX knows the output sharding and can
    # initialize the parameters directly on the target devices in a sharded way.
    model = create_model()
    return nnx.state(model)

  with jax.sharding.use_mesh(mesh):
    # Create the model with sharded parameters.
    sharded_state = create_sharded_state()
    model = nnx.merge(graphdef, sharded_state)

    target_for_restore = jax.tree.map(lambda v: v.value, sharded_state, is_leaf=lambda n: isinstance(n, nnx.Variable))
    checkpoint = mt.checkpointing.load_params_from_path(
        load_parameters_from_path=config.load_parameters_path,
        abstract_unboxed_params=target_for_restore,
        checkpoint_storage_concurrent_gb=None,
    )

    if checkpoint:
      nnx.update(model, checkpoint)

    tunix_model = TunixMaxTextLlama(
        base_model=model,
        use_attention_mask=False,  # trust Tunix loss masking
    )
  return tunix_model, mesh


def run_sft_training(mt_config, goodput_recorder):
  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    model, mesh = get_maxtext_model(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    training_hooks = hooks.SFTTrainingHooks(mt_config, mesh, learning_rate_schedule, goodput_recorder)
    data_hooks = hooks.SFTDataHooks(mt_config, mesh, goodput_recorder)

    tunix_config = get_tunix_config(mt_config)
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(learning_rate_schedule), tunix_config)
    trainer.with_training_hooks(training_hooks)
    trainer.with_data_hooks(data_hooks)
    trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

  trainer.train(data_hooks.train_data_iterator, data_hooks.eval_data_iterator)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"

  mt_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  maybe_monitor_goodput(mt_config)
  goodput_recorder = create_goodput_recorder(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.JOB):
    run_sft_training(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
