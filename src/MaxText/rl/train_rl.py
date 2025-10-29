# Copyright 2023–2025 Google LLC
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
GRPO Trainer

This module provides a unified `rl_train` function that consolidates the common
RL training logic. It handles model loading, reward function setup, dataset
processing, and training orchestration. By default, we run Group Relative Policy Optimization (GRPO) on 
GSM8K math reasoning benchmark. GRPO can enhance your model's problem-solving skills on mathematical word problems,
coding problems, etc. 

Usage:
  Usage Examples:

  # Llama3.1-8B (single host)
  python3 src/MaxText/examples/rl_trainer.py \\
    --model_name=llama3.1-8b \\
    --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
    --load_parameters_path=gs://path/to/checkpoint \\
    --base_output_directory=/tmp/grpo_output \\
    --hf_access_token=$HF_TOKEN \\
    --steps=100

  # Llama3.1-70B with Pathways (multi-host)
  python3 src/MaxText/examples/rl_trainer.py \\
    --model_name=llama3.1-70b \\
    --tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \\
    --load_parameters_path=gs://path/to/checkpoint \\
    --base_output_directory=gs://path/to/output \\
    --hf_access_token=$HF_TOKEN \\
    --use_pathways=true \\
    --steps=100

  # Custom dataset
  python3 src/MaxText/examples/rl_trainer.py \\
    --model_name=llama3.1-8b \\
    --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
    --load_parameters_path=gs://path/to/checkpoint \\
    --base_output_directory=/tmp/grpo_output \\
    --hf_access_token=$HF_TOKEN \\
    --hf_path=custom/dataset \\
    --steps=100
"""

from absl import app
import os
import re

import jax
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

import grain

import pathwaysutils

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger
from tunix.models.llama3 import model as llama3_lib

from MaxText import max_logging, max_utils, maxtext_utils, pyconfig
from MaxText import model_creation_utils
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter


# ====== Reproducibility ======
SEED = 42

# We use OpenAI's GSM8K dataset. GSM8K comprises grade school math word problems.


def extract_hash_answer(text: str, debug: bool = False) -> str | None:
  if debug:
    print(f"Extracting answer from: {text}")
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(data_dir, split="train") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  loaded_dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=SEED)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": model_tokenizer.apply_chat_template(
                  [
                      {
                          "role": "user",
                          "content": TEMPLATE.format(
                              system_prompt=SYSTEM_PROMPT,
                              question=x["question"].decode("utf-8"),
                          ),
                      },
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              # passed to reward functions
              "question": x["question"].decode("utf-8"),
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"].decode("utf-8"), debug=mt_config.debug),
          }
      )
  )
  return loaded_dataset


def get_maxtext_model(config, devices=None):
  """
  Load MaxText model with Tunix adapter.
  # Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
  # To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
  """
  model, mesh = model_creation_utils.create_nnx_model(config, devices)
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model)
    tunix_model.config = None
  return tunix_model, mesh


def setup_device_allocation(mt_config, use_pathways: bool = False):
  """Setup device allocation for training and inference."""

  devices = jax.devices()
  num_vms = len(devices) // mt_config.chips_per_vm
  trainer_devices = devices
  sampler_devices = devices
  if num_vms >= 2 and use_pathways:
    # Multiple hosts with Pathways - potentially split devices for trainer and sampler
    # based on trainer_devices_fraction and sampler_devices_fraction
    print(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
    num_devices = len(devices)
    num_trainer_devices = int(num_devices * mt_config.trainer_devices_fraction)
    num_sampler_devices = int(num_devices * mt_config.sampler_devices_fraction)
    trainer_devices = devices[:num_trainer_devices]
    sampler_devices = devices[num_devices - num_sampler_devices :]
    if mt_config.trainer_devices_fraction!=1.0:
      print(f"Using first {len(trainer_devices)} devices as Trainer devices")
    if mt_config.sampler_devices_fraction != 1.0:
      print(f"Using last {len(sampler_devices)} devices as Sampler devices")
  
  return trainer_devices, sampler_devices, num_vms


# Reward Functions
def match_format_exactly(prompts, completions, mt_config, **kwargs):
  """Reward exact format matching."""
  scores = []
  match_format = re.compile(
      (
          r"^[\s]{0,}}"
          rf"{mt_config.reasoning_start_token}.+?{mt_config.reasoning_end_token}.*?"
          rf"{mt_config.solution_start_token}(.+?){mt_config.solution_end_token}"
          r"[\s]{0,}$"
      ),
      flags=re.MULTILINE | re.DOTALL,
  )

  for completion in completions:
    score = 0
    if match_format.search(completion) is not None:
      score += mt_config.reward_exact_format_match
    scores.append(score)
  return scores


def match_format_approximately(prompts, completions, mt_config, **kwargs):
  """Reward approximate format matching."""
  scores = []
  for completion in completions:
    score = 0
    score += (
        mt_config.reward_partial_format_match
        if completion.count(mt_config.reasoning_start_token) == 1
        else mt_config.penalty_incorrect_format
    )
    score += (
        mt_config.reward_partial_format_match
        if completion.count(mt_config.reasoning_end_token) == 1
        else mt_config.penalty_incorrect_format
    )
    score += (
        mt_config.reward_partial_format_match
        if completion.count(mt_config.solution_start_token) == 1
        else mt_config.penalty_incorrect_format
    )
    score += (
        mt_config.reward_partial_format_match
        if completion.count(mt_config.solution_end_token) == 1
        else mt_config.penalty_incorrect_format
    )
    scores.append(score)
  return scores


def check_answer(prompts, completions, answer, mt_config, **kwargs):
  """Reward correct answers."""
  match_format = re.compile(
      (
          r"^[\s]{0,}}"
          rf"{mt_config.reasoning_start_token}.+?{mt_config.reasoning_end_token}.*?"
          rf"{mt_config.solution_start_token}(.+?){mt_config.solution_end_token}"
          r"[\s]{0,}$"
      ),
      flags=re.MULTILINE | re.DOTALL,
  )

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None for r in completions
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue

    if guess == true_answer:
      score += mt_config.reward_exact_format_match
    elif guess.strip() == true_answer.strip():
      score += mt_config.reward_white_space_format_match
    else:
      try:
        ratio = float(guess) / float(true_answer)
        if 0.9 <= ratio <= 1.1:
          score += mt_config.reward_ratio_guess_to_answer_high
        elif 0.8 <= ratio <= 1.2:
          score += mt_config.reward_ratio_guess_to_answer_low
        else:
          score += mt_config.penalty_incorrect_answer
      except (ValueError, ZeroDivisionError):
        score += mt_config.penalty_incorrect_format
    scores.append(score)
  return scores


def check_numbers(prompts, completions, answer, mt_config, **kwargs):
  """Reward correct numerical answers."""
  match_numbers = re.compile(
      rf"{mt_config.solution_start_token}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
  )
  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in completions
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except (ValueError, TypeError):
      scores.append(0)
      continue
  return scores


def rl_train(mt_config):
  """
  Run RL training with the provided configuration.

  Args:
    mt_config: MaxText configuration object
  """
  # ====== Debug flag for verbose logs ======
  DEBUG = mt_config.debug

  print("Starting GRPO Training")

  # Number of training steps.
  max_train_steps = int(mt_config.num_batches * mt_config.num_iterations * mt_config.train_fraction * mt_config.num_epochs)

  # ====== Data ======
  # Setup data directories
  home = os.path.expanduser("~") + "/"
  train_data_dir = f"{home}/data/train"
  test_data_dir = f"{home}/data/test"
  if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
  if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
 

  # Load datasets
  print("Loading GSM8K dataset...")
  train_dataset, tokenizer = get_gsm8k_dataset(
      train_data_dir,
      split="train",
      batch_size=mt_config.per_device_batch_size,
      num_batches=getattr(mt_config, "num_batches", 4),
  )

  # Load test dataset for evaluation (currently not used in training loop)
  get_gsm8k_dataset(
      test_data_dir,
      split="test",
      batch_size=mt_config.per_device_batch_size,
      num_batches=getattr(mt_config, "num_test_batches", 5),
  )

  
  # Setup device allocation
  if jax.extend.backend.get_backend().platform_version == "Pathways":
    max_logging.log("Pathways backend detected. Disabling setting profile options.")
    use_pathways = True
  else:
    use_pathways = False
  print(f"Use Pathways: {use_pathways}")
  trainer_devices, sampler_devices, num_vms = setup_device_allocation(mt_config, use_pathways)

  # Load reference model
  print("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = get_ref_maxtext_model(mt_config, trainer_devices)
  devices_array = maxtext_utils.create_device_mesh(mt_config, sampler_devices)
  # if trainer_devices=sampler_devices, then rollout_mesh=reference_mesh
  # else rollout_mesh uses sampler_devices
  rollout_mesh = Mesh(devices_array, mt_config.mesh_axes)
  if mt_config.debug:
      print("Reference Model initialized successfully")
      nnx.display(reference_model)
      print(f"Reference mesh shape: {reference_mesh.shape}")

      # Sanity check that weights are loaded correctly
      _maxtext_state_flatten = nnx.state(llama3_1_70b).flat_state()
      maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
      print(
          f"maxtext_state_flatten[base.token_embedder.embedding].value={maxtext_state_flatten['base.token_embedder.embedding'].value}"
      )

  # TODO: @mazumdera: change this to use lora
  # Load policy model
  print("Creating policy model with same config as reference model on trainer mesh")
  policy_model, policy_mesh = get_maxtext_model(mt_config, trainer_devices)
  actor_mesh = policy_mesh

  if mt_config.debug:
      print("Policy Model initialized successfully")
      nnx.display(policy_model)
      print(f"Policy mesh shape: {policy_mesh.shape}")

  # Setup optimizer
  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=mt_config.learning_rate,
          # Linearly increase learning rate from 0. to learning_rate in the first 
          # warmup_steps_fraction training steps, and then gradually decrease the 
          # learning rate to 0 using cosine scheduler.
          warmup_steps=int(mt_config.warmup_steps_fraction*mt_config.max_train_steps),
          decay_steps=max_train_steps,
      ),
      b1=mt_config.adam_b1,
      b2=mt_config.adam_b2,
      weight_decay=mt_config.adam_weight_decay,
  )

  # TODO: @mazumdera: try optimizer offloading with adamw
  # Add gradient clipping if specified
  # Grad clipping to prevent large gradients. We find this
  # important to keep KL divergence in check.
  if mt_config.gradient_clipping_threshold > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=mt_config.gradient_clipping_threshold),
        optimizer,
    )

  # Setup checkpointing
  ckpt_dir = mt_config.checkpoint_dir
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=mt_config.checkpoint_period, mt_config.max_num_checkpoints_to_keep
  )

  # Setup metrics logging
  log_dir=mt_config.tensorboard_dir
  print(f"TensorBoard logs directory: {LOG_DIR}")
  print(f"tensorboard --logdir {LOG_DIR} --port=8086")
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=log_dir, flush_every_n_steps=20)

  # Profiler configurations
  # TODO: xfgu@: add profiling
  profiler_options = None

  # RL Cluster config
  # Note that we use vLLM as the rollout engine.
  # and we are using Tensor Parallelism for rollout
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: actor_mesh,
          rl_cluster_lib.Role.REFERENCE: reference_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      rollout_engine="vllm",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=mt_config.eval_interval,
          max_steps=max_train_steps,
          # metrics logging
          metrics_logging_options=metrics_logging_options,
          # profiling
          profiler_options=profiler_options,
          # checkpoint saving
          checkpoint_root_directory=ckpt_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=mt_config.max_target_length,
          max_prompt_length=mt_config.max_prefill_predict_length,
          kv_cache_size=mt_config.max_prefill_predict_length
          + mt_config.max_target_length
          + mt_config.kv_cache_buffer,
          temperature=mt_config.decode_sampling_temperature,
          top_p=mt_config.decode_sampling_nucleus_p,
          top_k=mt_config.decode_sampling_top_k,
      ),
      rollout_vllm_model_version=mt_config.hf_model_name,
      rollout_vllm_hbm_utilization=mt_config.hbm_utilization_vllm,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_swap_space_size_gb=mt_config.swap_space_vllm_gb,
  )

  # Setup GRPO config
  grpo_config = GrpoConfig(
      num_generations=mt_config.num_generations,
      num_iterations=mt_config.num_iterations,
      beta=mt_config.grpo_beta,
      epsilon=mt_config.grpo_epsilon,
      loss_algo=mt_config.loss_algo,
  )

  # Create RL cluster
  print("Creating RL cluster...")
  with nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

  # Create GRPO trainer
  print("Setting up GRPO trainer...")
  rl_trainer = GrpoLearner(
      rl_cluster=rl_cluster,
      reward_fns=[
          lambda **kwargs: match_format_exactly(mt_config=mt_config, **kwargs),
          lambda **kwargs: match_format_approximately(mt_config=mt_config, **kwargs),
          lambda **kwargs: check_answer(mt_config=mt_config, **kwargs),
          lambda **kwargs: check_numbers(mt_config=mt_config, **kwargs),
      ],
      grpo_config=grpo_config,
  )

  # Start training
  print("Starting GRPO training...")
  with policy_mesh, nn_partitioning.axis_rules(mt_config.logical_axis_rules):
    rl_trainer.train(train_dataset)
  
  profile_dir = mt_config.tensorboard_dir
  max_logging.log(f"Saving profiles to {profile_dir}")

  jax.profiler.start_trace(profile_dir)
  with mesh, nn_partitioning.axis_rules(config_policy.logical_axis_rules):
    rl_trainer.train(dataset)
  jax.profiler.stop_trace()

  print("=" * 80)
  print("GRPO Training Completed Successfully!")
  print("=" * 80)

  return rl_trainer, rl_cluster

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

  rl_train(mt_config)


if __name__ == "__main__":
  app.run(main)
