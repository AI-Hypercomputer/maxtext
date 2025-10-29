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

# ====== Reward-specific constants ======
REWARD_EXACT_FORMAT_MATCH = 3.0
REWARD_WHITE_SPACE_FORMAT_MATCH = 1.5
REWARD_PARTIAL_FORMAT_MATCH = 0.5
REWARD_RATIO_GUESS_TO_ANSWER_HIGH = 0.5
REWARD_RATIO_GUESS_TO_ANSWER_LOW = 0.25
PENALTY_INCORRECT_FORMAT = -0.5
PENALTY_INCORRECT_ANSWER = -1.0

# ====== Special tokens for GSM8K reasoning ======
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

# ====== System prompt and Templates ======

SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {REASONING_START} and \
{REASONING_END}. Then, provide the final answer (i.e., just one numerical \
value) between {SOLUTION_START} and {SOLUTION_END}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

# ====== Debug flag for verbose logs ======
DEBUG=False

# ====== Reproducibility ======
SEED = 42

# We use OpenAI's GSM8K dataset. GSM8K comprises grade school math word problems.


def extract_hash_answer(text: str) -> str | None:
  if DEBUG:
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
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return loaded_dataset


def get_maxtext_model(config, devices=None):
  """Load MaxText model with Tunix adapter."""
  model, mesh = model_creation_utils.create_nnx_model(config, devices)
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model)
    model_config = llama3_lib.ModelConfig.llama3_1_8b()
    tunix_model.config = model_config
  return tunix_model, mesh


def setup_device_allocation(config, use_pathways: bool = False):
  """Setup device allocation for training and inference."""
  devices = jax.devices()

  # Get device allocation parameters from config
  trainer_devices_fraction = getattr(config, "trainer_devices_fraction", 0.5)
  sampler_devices_fraction = getattr(config, "sampler_devices_fraction", 0.5)
  chips_per_vm = getattr(config, "chips_per_vm", 4)

  num_vms = len(devices) // chips_per_vm

  trainer_devices = devices
  sampler_devices = devices
  if num_vms >= 2 and use_pathways:
    # Multiple hosts with Pathways - split devices for trainer and sampler
    print(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
    num_devices = len(devices)
    num_trainer_devices = int(num_devices * trainer_devices_fraction)
    num_sampler_devices = int(num_devices * sampler_devices_fraction)
    trainer_devices = devices[:num_trainer_devices]
    sampler_devices = devices[num_devices - num_sampler_devices :]

    print("Creating reference model and also meshes for reference and rollout")
    llama3_1_70b, reference_mesh = get_maxtext_model(config_ref, trainer_devices)
    devices_array = maxtext_utils.create_device_mesh(config_ref, sampler_devices)
    rollout_mesh = Mesh(devices_array, config_ref.mesh_axes)
    mesh = reference_mesh
  
  return trainer_devices, sampler_devices, num_vms


# Reward Functions
def match_format_exactly(prompts, completions, **kwargs):
  """Reward exact format matching."""
  scores = []
  match_format = re.compile(
      rf"^[\s]{{0,}}" rf"{REASONING_START}.+?{REASONING_END}.*?" rf"{SOLUTION_START}(.+?){SOLUTION_END}" rf"[\s]{{0,}}$",
      flags=re.MULTILINE | re.DOTALL,
  )

  for completion in completions:
    score = 0
    if match_format.search(completion) is not None:
      score += REWARD_EXACT_FORMAT_MATCH
    scores.append(score)
  return scores


def match_format_approximately(prompts, completions, **kwargs):
  """Reward approximate format matching."""
  scores = []
  for completion in completions:
    score = 0
    score += REWARD_PARTIAL_FORMAT_MATCH if completion.count(REASONING_START) == 1 else PENALTY_INCORRECT_FORMAT
    score += REWARD_PARTIAL_FORMAT_MATCH if completion.count(REASONING_END) == 1 else PENALTY_INCORRECT_FORMAT
    score += REWARD_PARTIAL_FORMAT_MATCH if completion.count(SOLUTION_START) == 1 else PENALTY_INCORRECT_FORMAT
    score += REWARD_PARTIAL_FORMAT_MATCH if completion.count(SOLUTION_END) == 1 else PENALTY_INCORRECT_FORMAT
    scores.append(score)
  return scores


def check_answer(prompts, completions, answer, **kwargs):
  """Reward correct answers."""
  match_format = re.compile(
      rf"^[\s]{{0,}}" rf"{REASONING_START}.+?{REASONING_END}.*?" rf"{SOLUTION_START}(.+?){SOLUTION_END}" rf"[\s]{{0,}}$",
      flags=re.MULTILINE | re.DOTALL,
  )

  extracted_responses = [guess.group(1) if (guess := match_format.search(r)) is not None else None for r in completions]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue

    if guess == true_answer:
      score += REWARD_EXACT_FORMAT_MATCH
    elif guess.strip() == true_answer.strip():
      score += REWARD_WHITE_SPACE_FORMAT_MATCH
    else:
      try:
        ratio = float(guess) / float(true_answer)
        if 0.9 <= ratio <= 1.1:
          score += REWARD_RATIO_GUESS_TO_ANSWER_HIGH
        elif 0.8 <= ratio <= 1.2:
          score += REWARD_RATIO_GUESS_TO_ANSWER_LOW
        else:
          score += PENALTY_INCORRECT_ANSWER
      except (ValueError, ZeroDivisionError):
        score += PENALTY_INCORRECT_FORMAT
    scores.append(score)
  return scores


def check_numbers(prompts, completions, answer, **kwargs):
  """Reward correct numerical answers."""
  match_numbers = re.compile(rf"{SOLUTION_START}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL)
  extracted_responses = [guess.group(1) if (guess := match_numbers.search(r)) is not None else None for r in completions]

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
  print("Starting GRPO Training")


  # Number of training steps.
  max_steps = int(mt_config.num_batches * mt_config.num_iterations * TRAIN_FRACTION * NUM_EPOCHS)
  # Setup device allocation
  if jax.extend.backend.get_backend().platform_version == "Pathways":
    max_logging.log("Pathways backend detected. Disabling setting profile options.")
    use_pathways = True
  else:
    use_pathways = False

  trainer_devices, sampler_devices, num_vms = setup_device_allocation(mt_config, use_pathways)

  print(f"Device allocation: {len(trainer_devices)} trainer, {len(sampler_devices)} sampler")
  print(f"Use Pathways: {use_pathways}")

  # ====== Data ======
  # Setup data directories
  home = os.path.expanduser("~") + "/"
  train_data_dir = f"{home}/data/train"
  test_data_dir = f"{home}/data/test"
  if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
  if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)
  TRAIN_FRACTION = 1.0

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

  # Load reference model
  print("Loading reference model...")
  reference_model, reference_mesh = get_maxtext_model(mt_config, trainer_devices)
  reference_model.config = None

  # Load policy model
  print("Loading policy model...")
  policy_model, policy_mesh = get_maxtext_model(mt_config, trainer_devices)
  policy_model.config = None

  
  # Setup meshes
  if num_vms >= 2 and not use_pathways:
    actor_mesh = policy_mesh
    rollout_mesh = Mesh(maxtext_utils.create_device_mesh(mt_config, sampler_devices), mt_config.mesh_axes)
  else:
    actor_mesh = policy_mesh
    rollout_mesh = policy_mesh

  # Setup optimizer
  learning_rate = getattr(mt_config, "learning_rate", 3e-6)
  max_steps = getattr(mt_config, "steps", 100)
  warmup_steps = int(0.1 * max_steps)

  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=learning_rate,
          warmup_steps=warmup_steps,
          decay_steps=max_steps,
          end_value=0.0,
      ),
      b1=0.9,
      b2=0.99,
      weight_decay=0.1,
  )

  # Add gradient clipping if specified
  max_grad_norm = getattr(mt_config, "max_grad_norm", 0.1)
  if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=max_grad_norm),
        optimizer,
    )

  # Setup checkpointing
  ckpt_dir = mt_config.base_output_directory

  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=getattr(mt_config, "checkpoint_period", 50), max_to_keep=4
  )

  # Setup metrics logging
  log_dir = mt_config.base_output_directory
  max_logging.log(f"Logging to {log_dir}")

  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=log_dir, flush_every_n_steps=20)

  # Setup RL cluster config
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
          eval_every_n_steps=getattr(mt_config, "eval_interval", 10),
          max_steps=max_steps,
          metrics_logging_options=metrics_logging_options,
          profiler_options=None,
          checkpoint_root_directory=ckpt_dir,
          checkpointing_options=checkpointing_options,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=getattr(mt_config, "max_target_length", 768),
          max_prompt_length=getattr(mt_config, "max_prefill_predict_length", 256),
          kv_cache_size=getattr(mt_config, "max_prefill_predict_length", 256)
          + getattr(mt_config, "max_target_length", 768)
          + 256,
          temperature=getattr(mt_config, "decode_sampling_temperature", 0.9),
          top_p=getattr(mt_config, "decode_sampling_top_p", 1.0),
          top_k=getattr(mt_config, "decode_sampling_top_k", 50),
      ),
      rollout_vllm_model_version="meta-llama/Meta-Llama-3.1-8B-Instruct",
      rollout_vllm_hbm_utilization=0.2,
      rollout_vllm_tpu_backend_type="jax",
  )

  # Setup GRPO config
  grpo_config = GrpoConfig(
      num_generations=getattr(mt_config, "num_generations", 2),
      num_iterations=1,
      beta=getattr(mt_config, "grpo_beta", 0.08),
      epsilon=getattr(mt_config, "grpo_epsilon", 0.2),
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
          match_format_exactly,
          match_format_approximately,
          check_answer,
          check_numbers,
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
