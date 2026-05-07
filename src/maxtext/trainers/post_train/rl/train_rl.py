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

"""
RL Trainer

This module provides a unified `rl_train` function that consolidates the common
RL training logic. It handles model loading, reward function setup, dataset
processing, and training orchestration. By default, we run Group Relative Policy Optimization (GRPO) on
GSM8K math reasoning benchmark. The script is also flexible enough to run Group Sequence Policy Optimization (GSPO).

Usage Examples:

# GRPO on Llama3.1-8B-Instruct
python3 -m maxtext.trainers.post_train.rl.train_rl src/maxtext/configs/post_train/rl.yml \
  model_name=llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=${WORKLOAD?} \
  base_output_directory=${OUTPUT_PATH?} \
  hf_access_token=${HF_TOKEN?}

# GSPO on Llama3.1-70B-Instruct
python3 -m maxtext.trainers.post_train.rl.train_rl src/maxtext/configs/post_train/rl.yml \
  model_name=llama3.1-70b \
  tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \
  load_parameters_path=gs://path/to/checkpoint/0/items \
  run_name=${WORKLOAD?} \
  base_output_directory=${OUTPUT_PATH?} \
  hf_access_token=${HF_TOKEN?} \
  loss_algo=gspo-token

"""

from __future__ import annotations
from typing import Sequence

import jax
import logging
import os
import pathwaysutils

from absl import app
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from pprint import pprint
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "0"

from maxtext.trainers.post_train.rl.data_pipeline import prepare_datasets
from maxtext.trainers.post_train.rl.trainer_setup import create_rl_components
from maxtext.trainers.post_train.rl.evaluate_rl import EvalPhase, run_eval
from maxtext.utils import max_logging, max_utils, model_creation_utils


def get_max_train_steps(trainer_config) -> int:
  """Calculate the total number of training steps."""
  return int(
      trainer_config.num_batches
      * trainer_config.rl.num_iterations
      * trainer_config.train_fraction
      * trainer_config.num_epoch
  )


def rl_train(argv: Sequence[str], kwargs: dict) -> None:
  """
  Run RL training with the provided configuration.

  Args:
    argv: Command-line arguments.
    kwargs: Additional keyword arguments.
  """
  trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
      argv, kwargs
  )

  reference_model, reference_mesh, actor_model, actor_mesh, rollout_mesh = model_creation_utils.create_models_and_meshes(
      trainer_config, sampler_config, trainer_devices, sampler_devices
  )

  if not trainer_config.debug.rl:
    # Apply filter to suppress noisy logs
    noise_filter = max_logging.NoisyLogFilter()
    logging.getLogger().addFilter(noise_filter)
    absl_logging.get_absl_logger().addFilter(noise_filter)
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

  if not epath.Path(trainer_config.tensorboard_dir).exists():
    epath.Path(trainer_config.tensorboard_dir).mkdir(parents=True, exist_ok=True)

  if not epath.Path(trainer_config.checkpoint_dir).exists():
    epath.Path(trainer_config.checkpoint_dir).mkdir(parents=True)

  max_train_steps = get_max_train_steps(trainer_config)

  # Create model tokenizer
  model_tokenizer = AutoTokenizer.from_pretrained(trainer_config.tokenizer_path)

  train_dataset, test_dataset = prepare_datasets(trainer_config, model_tokenizer)

  if trainer_config.debug.rl:
    max_logging.log("Train dataset samples:")
    for i, ele in enumerate(train_dataset):
      if i >= 5:
        break
      pprint(ele)
    if trainer_config.num_test_batches > 0:
      max_logging.log("Test dataset samples:")
      for i, ele in enumerate(test_dataset):
        if i >= 5:
          break
        pprint(ele)

  if trainer_config.debug.rl:
    max_logging.log("Reference Model initialized successfully")
    nnx.display(reference_model)
    max_logging.log(f"Reference mesh shape: {reference_mesh.shape}")
    max_logging.log("Policy Model initialized successfully")
    nnx.display(actor_model)
    max_logging.log(f"Policy mesh shape: {actor_mesh.shape}")
    max_logging.log(f"Rollout_mesh shape: {rollout_mesh.shape}")

  rl_cluster, rl_trainer, _ = create_rl_components(
      trainer_config,
      sampler_config,
      sampler_devices,
      actor_model,
      actor_mesh,
      reference_model,
      reference_mesh,
      rollout_mesh,
      model_tokenizer,
      max_train_steps,
  )

  # Run evaluation before training
  if trainer_config.num_test_batches > 0:
    # Update vllm with model parameters from checkpoint
    rl_cluster.rollout.update_params(nnx.state(actor_model))
    run_eval(EvalPhase.PRE_TRAINING, trainer_config, test_dataset, rl_cluster)

  # Start training
  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Capturing reference model state before training.")
    ref_state_before = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))

  max_logging.warning("Starting RL training...")
  rl_trainer.train(train_dataset)

  if trainer_config.load_checkpoint_only_once:
    max_logging.log("Checking if reference model state changed during training.")
    ref_state_after = nnx.to_pure_dict(nnx.state(reference_model.base, nnx.Param))
    check = jax.tree_util.tree_map(jax.numpy.array_equal, ref_state_before, ref_state_after)
    if not jax.tree_util.tree_all(check):
      raise ValueError("Reference model parameters changed during training!")
    max_logging.log("Reference model parameters verified to be unchanged during training.")

  max_logging.warning("RL Training Completed Successfully!")

  # Run evaluation after training
  if trainer_config.num_test_batches > 0:
    run_eval(EvalPhase.POST_TRAINING, trainer_config, test_dataset, rl_cluster)


def main(argv: Sequence[str], kwargs: dict = None) -> None:
  """Main function to run RL training.

  Args:
    argv: Command-line arguments.
  """
  kwargs = kwargs or {}
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  max_utils.print_system_information()
  rl_train(argv, kwargs)


if __name__ == "__main__":
  app.run(main)
