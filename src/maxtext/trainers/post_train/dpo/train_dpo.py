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

"""DPO Training script that uses Tunix DPOTrainer on a MaxText model.

Example command:
Training & Evaluation:
  python3 -m maxtext.trainers.post_train.dpo.train_dpo \
    run_name=${WORKLOAD?} base_output_directory=${BASE_OUTPUT_DIRECTORY?} \
    tokenizer_path="google/gemma-2-2b-it" tokenizer_type=huggingface \
    dataset_type="hf" hf_path="Anthropic/hh-rlhf" hf_eval_split="test" \
    train_data_columns="['chosen', 'rejected']" eval_data_columns="['chosen', 'rejected']" \
    model_name=${MODEL?} load_parameters_path=${MAXTEXT_CONVERTED_CHECKPOINT?}/0/items \
    hf_access_token=${HF_TOKEN?} per_device_batch_size=1 max_target_length=1024 \
    eval_interval=2 eval_steps=2 steps=10 profiler=xplane weight_dtype=bfloat16
"""

from absl import app
import jax
import optax
from orbax import checkpoint as ocp
import pathwaysutils

from flax import nnx
from flax.linen import partitioning as nn_partitioning

from tunix.sft import metrics_logger, profiler
from tunix.sft.dpo.dpo_trainer import DPOTrainer, DPOTrainingConfig

from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.configs import pyconfig
from maxtext.configs.types import MaxTextConfig
from maxtext.input_pipeline import tokenizer
from maxtext.optimizers import optimizers
from maxtext.trainers.post_train.dpo import hooks
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def get_tunix_config(mt_config: MaxTextConfig) -> DPOTrainingConfig:
  """Gets the Tunix training configurations from the MaxText config.

  Args:
    mt_config: MaxText config.

  Returns:
    A Tunix `DPOTrainingConfig` object.
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

  return DPOTrainingConfig(
      eval_every_n_steps=mt_config.eval_interval,
      max_steps=mt_config.steps,
      gradient_accumulation_steps=mt_config.gradient_accumulation_steps,
      checkpoint_root_directory=mt_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
      algorithm=mt_config.dpo.algo,
      lambda_orpo=mt_config.dpo.orpo_lambda,
      beta=mt_config.dpo.dpo_beta,
      label_smoothing=mt_config.dpo.dpo_label_smoothing,
      max_prompt_length=mt_config.dpo.max_prompt_length,
      max_response_length=mt_config.max_target_length - mt_config.dpo.max_prompt_length,
  )


def setup_trainer_state(mt_config, goodput_recorder=None, test_only_training_hooks_class=None):
  """Set up prerequisites for training loop."""
  tunix_config = get_tunix_config(mt_config)

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TPU_INIT):
    # We need pad_id to correctly initialize the MaxTextTunixAdapter in the from_pretrained call below.
    # To do that we instantiate a separate copy of the tokenizer here, which is somewhat redundant.
    # However, there is no clean way to get the pad_id from the tokenizer in the datapipiline here.
    tok = tokenizer.build_tokenizer(
        mt_config.tokenizer_path,
        mt_config.tokenizer_type,
        False,
        False,
        mt_config.hf_access_token,
    )
    model, mesh = model_creation_utils.from_pretrained(
        mt_config,
        wrap_with_tunix_adapter=True,
        tokenizer_pad_id=tok.pad_id,
    )

    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(mt_config)
    # pass in model for muon
    optimizer = optimizers.get_optimizer(mt_config, learning_rate_schedule, model)

    if mt_config.gradient_clipping_threshold > 0:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold),
          optimizer,
      )

  # ORPO does not require a reference model.
  ref_model = nnx.clone(model) if mt_config.dpo.algo == "dpo" else None

  with maybe_record_goodput(goodput_recorder, GoodputEvent.TRAINING_PREPARATION):
    training_hooks_class = test_only_training_hooks_class or hooks.DPOTrainingHooks
    training_hooks = training_hooks_class(mt_config, mesh, learning_rate_schedule, goodput_recorder)
    data_hooks = hooks.DPODataHooks(mt_config, mesh, goodput_recorder)

    # Provide rules context so logical axes (e.g. 'norm') are translated to mesh axes during maybe_restore
    with nn_partitioning.axis_rules(mt_config.logical_axis_rules):
      trainer = DPOTrainer(
          model=model, ref_model=ref_model, optimizer=optimizer, training_config=tunix_config, tokenizer=None
      )
      trainer.with_training_hooks(training_hooks)
      trainer.with_data_hooks(data_hooks)

  return trainer, mesh


def train_model(mt_config: MaxTextConfig, trainer, mesh):
  """Runs the DPO training loop in Tunix."""
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    trainer.train(trainer.data_hooks.train_data_iterator, trainer.data_hooks.eval_data_iterator)
  return trainer


def train(mt_config, goodput_recorder=None, test_only_training_hooks_class=None):
  """Main method for DPO training.

  Args:
    mt_config: MaxText config.
    goodput_recorder: An optional GoodputRecorder to record performance metrics.
    test_only_training_hooks_class: An optional DPOTrainingHooks subclass to override hooks.
  """
  trainer, mesh = setup_trainer_state(mt_config, goodput_recorder, test_only_training_hooks_class)
  _job_completed_gracefully = False
  try:
    trainer = train_model(mt_config, trainer, mesh)
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(goodput_recorder, RECORD_JOB_END_TIME)
  return trainer, mesh


def validate_config(config):
  """Validates the configuration parameters for DPO training."""
  if config.optimizer_memory_host_offload:
    raise ValueError(
        "optimizer_memory_host_offload=True is not supported on the post-training "
        "DPO path because the underlying Tunix DPOTrainer does not support "
        "host offloading of the optimizer state."
    )

  if config.num_vocab_tiling > 1:
    raise ValueError(
        f"Vocab Tiling is not supported with DPO. "
        f"num_vocab_tiling was configured to {config.num_vocab_tiling}, but it must be 1 when running train_dpo."
    )


def main(argv: list[str]) -> None:
  """Main function to run DPO training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()

  mt_config = pyconfig.initialize_pydantic(argv)
  validate_config(mt_config)
  max_utils.print_system_information()

  goodput_recorder = create_goodput_recorder(mt_config)
  record_goodput(goodput_recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(mt_config):
    train(mt_config, goodput_recorder)


if __name__ == "__main__":
  app.run(main)
