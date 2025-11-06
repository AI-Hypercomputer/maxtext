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

# pylint: disable=bare-except, consider-using-generator
"""
GRPO (Group Relative Policy Optimization) Tutorial
===================================================

This tutorial demonstrates training the Llama 3.1 8B model on the GSM8K math reasoning
benchmark using Group Relative Policy Optimization (GRPO).

What is GRPO?
-------------
GRPO is a Reinforcement Learning algorithm designed to enhance reasoning abilities
of Large Language Models. It's a memory-efficient variant of PPO (Proximal Policy
Optimization) that:
  - Eliminates the need for a separate value function model (saves memory)
  - Generates multiple responses per prompt (the "group")
  - Evaluates responses using reward functions
  - Calculates relative advantage based on group performance
  - Updates the policy to improve future generations

Why use GRPO?
-------------
GRPO can enhance your model's problem-solving skills on:
  - Mathematical word problems (like GSM8K)
  - Coding challenges
  - Reasoning tasks
  - Any task where you can define a reward function

Architecture Overview:
---------------------
  - Tunix: Main library for GRPO training orchestration
  - vLLM: Efficient inference engine for generating responses during training
  - MaxText: Model implementation (supports Qwen3, Llama, Gemma, etc.)
  - JAX/Flax: Backend for training

Hardware Requirements:
---------------------
This tutorial uses a single host TPU VM (e.g., v6e-8 or v5p-8)

Let's get started!
"""

# ==============================================================================
# STEP 0: IMPORTS AND ENVIRONMENT SETUP
# ==============================================================================

# Standard library imports
from pprint import pprint
import functools
import os
import re

# Third-party imports
from tqdm.auto import tqdm
import grain
import humanize
import jax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

# Tunix imports (GRPO training framework)
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.base_rollout import RolloutConfig
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger
from tunix.models.llama3 import model as llama3_lib

# MaxText imports (model implementation)
from MaxText.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_PKG_DIR
from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter

# Environment setup for vLLM
# Skip JAX precompilation to speed up startup when using vLLM
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

# For Colab/notebook environments (uncomment if needed):
# import nest_asyncio
# nest_asyncio.apply()  # Fixes "This event loop is already running" error

# Initialize JAX devices (TPU/GPU)
jax.devices()

# Global settings
DEBUG = True  # Set to True for detailed debug output during training
HOME = os.path.join(os.path.expanduser("~"), "")
print(f"Home directory (from Python): {HOME}")

# ==============================================================================
# INSTALLATION REQUIREMENTS
# ==============================================================================
# Before running this tutorial, ensure you have installed all required packages.
# For detailed instructions, refer to:
# https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html
#
# Quick setup:
# 1. Run the initial setup script:
#    bash tools/setup/setup.sh
#
# 2. Activate your virtual environment:
#    venv_name="maxtext_venv"  # Replace with your venv name if different
#    source ~/$venv_name/bin/activate
#
# 3. Install vLLM and tpu-commons dependencies:
#    bash ~/maxtext/src/MaxText/examples/install_tunix_vllm_requirement.sh
#    Note: This installation may take several minutes. Monitor the logs for any errors.

# ==============================================================================
# STEP 1: CONFIGURE HYPERPARAMETERS
# ==============================================================================
#
# This section defines all hyperparameters for the GRPO training pipeline.
# These are organized into logical categories for easy understanding and tuning.
#
# Note: These are not "perfect" hyperparameters. For production results,
# you may need to tune these values and train for longer.

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
# Directories for storing training and test datasets
TRAIN_DATA_DIR = os.path.join(HOME, "data", "train")
TEST_DATA_DIR = os.path.join(HOME, "data", "test")
if not os.path.exists(TRAIN_DATA_DIR):
  os.makedirs(TRAIN_DATA_DIR)
if not os.path.exists(TEST_DATA_DIR):
  os.makedirs(TEST_DATA_DIR)

# Fraction of training data to use (1.0 = 100%, use all data)
# Set < 1.0 to create a validation split
TRAIN_FRACTION = 1.0

# ==============================================================================
# MODEL & CHECKPOINT CONFIGURATION
# ==============================================================================
# Path to pre-trained model checkpoint (can be local or GCS path)
# For Llama 3 8B, you'll need to convert the HuggingFace checkpoint to MaxText format
# See: /maxtext/src/MaxText/utils/ckpt_conversion/to_maxtext.py
MODEL_CHECKPOINT_PATH = "gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items"

# Directory for TensorBoard logs (training metrics visualization)
LOG_DIR = os.path.join(HOME, "content", "tensorboard", "grpo", "logs_llama3", "")
if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)

# Directory for JAX profiling traces (performance analysis)
PROFILE_DIR = os.path.join(HOME, "content", "jax_traces", "grpo", "profiles_llama3", "")
if not os.path.exists(PROFILE_DIR):
  os.makedirs(PROFILE_DIR)

# Directory for saving training checkpoints
CKPT_DIR = os.path.join(HOME, "content", "ckpts_llama3", "")
if not os.path.exists(CKPT_DIR):
  os.makedirs(CKPT_DIR)

# Checkpoint saving frequency (save every N steps)
SAVE_INTERVAL_STEPS = 500

# Maximum number of checkpoints to retain (older ones are deleted)
MAX_TO_KEEP = 4

# Random seed for reproducibility (data shuffling, sampling, etc.)
SEED = 42

# ==============================================================================
# GRPO ALGORITHM PARAMETERS
# ==============================================================================
# Number of responses generated per prompt in each training step
# This is the "G" (group size) in GRPO Algorithm 1
# Larger values provide better advantage estimates but increase compute
NUM_GENERATIONS = 2

# Number of optimization iterations per batch (μ in GRPO Algorithm 1)
# Higher values = more gradient steps per batch of data
NUM_ITERATIONS = 1

# KL divergence penalty coefficient (β in GRPO loss function)
# Controls how much the policy can deviate from the reference model
# Too low: policy may diverge too much; Too high: policy updates too conservative
BETA = 0.08

# PPO-style clipping parameter (ε in GRPO loss)
# Prevents excessively large policy updates for training stability
EPSILON = 0.2

# ==============================================================================
# GENERATION/SAMPLING PARAMETERS (During Training)
# ==============================================================================
# Maximum length of input prompts (tokens)
MAX_PROMPT_LENGTH = 512

# Maximum number of tokens to generate per response
TOTAL_GENERATION_STEPS = 1024

# Sampling temperature during training rollouts
# Higher values (0.9) encourage diversity and exploration
# This is important for GRPO to generate varied responses
TEMPERATURE = 0.9

# Top-p (nucleus) sampling parameter
# 1.0 = consider all tokens in the distribution
TOP_P = 1.0

# Top-k sampling parameter
# Only sample from the top K most likely tokens
TOP_K = 50

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
# Batch size per device (number of prompts processed together)
BATCH_SIZE = 1

# Number of batches to train on
# Increase for better results (original: 3738, reduced for demo)
NUM_BATCHES = 500  # 200

# Number of batches to use for testing/evaluation
# Keep low for quick evaluation (max 330 if batch_size=4)
NUM_TEST_BATCHES = 200  # 200

# Evaluate on validation set every N steps
# (Not used if TRAIN_FRACTION = 1.0, no validation split)
EVAL_EVERY_N_STEPS = 10

# Number of times to iterate over the entire dataset
NUM_EPOCHS = 1

# Total number of training steps (computed from other params)
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# ==============================================================================
# OPTIMIZER & LEARNING RATE SCHEDULE
# ==============================================================================
# Peak learning rate for AdamW optimizer
LEARNING_RATE = 3e-6

# AdamW beta1 parameter (momentum for first moment estimates)
B1 = 0.9

# AdamW beta2 parameter (momentum for second moment estimates)
B2 = 0.99

# Weight decay coefficient for L2 regularization
WEIGHT_DECAY = 0.1

# Number of warmup steps for learning rate schedule
# LR linearly increases from 0 to LEARNING_RATE over this period
# Then cosine decays to 0 over remaining steps
WARMUP_STEPS = int(0.1 * MAX_STEPS)

# Maximum gradient norm for gradient clipping
# Prevents exploding gradients and helps maintain stable KL divergence
# Set to None to disable gradient clipping
MAX_GRAD_NORM = 0.1

# ==============================================================================
# EVALUATION/INFERENCE CONFIGURATIONS
# ==============================================================================
# Different sampling strategies for evaluation
GENERATION_CONFIGS = {
    # Greedy decoding: deterministic, always picks most likely token
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},

    # Standard sampling: balanced exploration
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},

    # Liberal sampling: high diversity
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ==============================================================================
# REWARD FUNCTION PARAMETERS
# ==============================================================================
# Rewards for correct formatting
REWARD_EXACT_FORMAT_MATCH = 3.0          # Perfect format match
REWARD_WHITE_SPACE_FORMAT_MATCH = 1.5    # Match with whitespace differences
REWARD_PARTIAL_FORMAT_MATCH = 0.5        # Partial format compliance

# Rewards for answer correctness
REWARD_RATIO_GUESS_TO_ANSWER_HIGH = 0.5  # Answer within 10% of correct value
REWARD_RATIO_GUESS_TO_ANSWER_LOW = 0.25  # Answer within 20% of correct value

# Penalties for mistakes
PENALTY_INCORRECT_FORMAT = -0.5          # Wrong formatting
PENALTY_INCORRECT_ANSWER = -1.0          # Wrong answer


# ==============================================================================
# STEP 2: DEFINE UTILITY FUNCTIONS
# ==============================================================================
#
# Helper functions for monitoring training progress and system resources


def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


# ==============================================================================
# STEP 3: DATA PREPROCESSING & TOKENIZER SETUP
# ==============================================================================
#
# This section:
# 1. Loads the tokenizer for Llama 3.1 8B
# 2. Defines special tokens for structured output (reasoning + answer format)
# 3. Creates prompt templates for the GSM8K math reasoning task
#
# We instruct the model to use a specific format:
#   <reasoning>...model's reasoning...</reasoning>
#   <answer>...final numerical answer...</answer>
#
# This structured format helps with:
#   - Evaluating the reasoning process
#   - Extracting and verifying the final answer
#   - Providing clearer rewards during RL training

# Load tokenizer for Llama 3.1 8B from HuggingFace
model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Define special tokens for structured output format
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

# System prompt that instructs the model on the expected format
SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

# Chat template format (plain text - will be formatted by tokenizer's chat template)
TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

# ==============================================================================
# STEP 4: DATASET CREATION
# ==============================================================================
#
# We use GSM8K (Grade School Math 8K) - a dataset of grade school math word
# problems requiring multi-step reasoning. Perfect for testing GRPO's ability
# to improve reasoning capabilities!


def extract_hash_answer(text: str) -> str | None:
  """Extracts the answer from a string that contains '####'.

  Args:
    text: The string to extract the answer from.

  Returns:
    The extracted answer as a string, or None if '####' is not found.
  """
  if DEBUG:
    print(f"Extracting answer from: {text}")
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_dataset(data_dir, split="train") -> grain.MapDataset:
  """Gets and preprocesses the GSM8K dataset.

  Args:
    data_dir: The directory to download and store the dataset.
    split: The dataset split to use (e.g., 'train', 'test').

  Returns:
    A grain.MapDataset containing the preprocessed data.
  """
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
              # Prompts are passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=x["question"].decode("utf-8"),
              ),
              # Question and answer are passed to reward functions
              "question": x["question"].decode("utf-8"),
              "answer": extract_hash_answer(x["answer"].decode("utf-8")),
          }
      )
  )
  return loaded_dataset


DATASET = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = DATASET.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = DATASET[: int(len(DATASET) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = DATASET[int(len(DATASET) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test").batch(BATCH_SIZE)[:NUM_TEST_BATCHES]


# Debug: Print sample batch to verify data preprocessing
if DEBUG:
  print("Sample batch from dataset:")
  for ele in train_dataset[:1]:
    pprint(ele)


# ==============================================================================
# STEP 5: LOAD POLICY AND REFERENCE MODELS
# ==============================================================================
#
# GRPO requires TWO models:
#
# 1. POLICY MODEL (Actor):
#    - This is the model being trained
#    - Weights are updated during training
#    - Generates responses during rollouts
#
# 2. REFERENCE MODEL:
#    - Frozen copy of the original model
#    - Used to compute KL divergence penalty
#    - Prevents the policy from deviating too far from original behavior
#    - Ensures training stability
#
# Training Strategy:
# ------------------
# - This script uses FULL model training (all parameters updated)
# - For memory efficiency, you could use LoRA (Low-Rank Adaptation):
#   * Freeze base model weights
#   * Only train small LoRA adapters
#   * Significantly reduces memory usage
#
# Precision:
# ---------
# - Using bfloat16 for memory efficiency
# - For even lower precision, consider Qwix for Quantization-Aware Training

print("HBM usage before loading models:")
show_hbm_usage()


# ### Helper function to create MaxText models


def get_ref_maxtext_model(config):
  """Creates and returns a TunixMaxTextAdapter model and mesh.

  Args:
    config: The model configuration.

  Returns:
    A tuple containing the TunixMaxTextAdapter model and the mesh.
  """

  model, this_mesh = model_creation_utils.create_nnx_model(config)
  with this_mesh:
    tunix_model = TunixMaxTextAdapter(
        base_model=model,
    )

    this_model_config = llama3_lib.ModelConfig.llama3_1_8b()
    tunix_model.config = this_model_config

  return tunix_model, this_mesh


model_config = llama3_lib.ModelConfig.llama3_1_8b()

# Configure and load the reference model
# Note: pass the path to your scanned checkpoint for "load_parameters_path".
# To create a scanned checkpoint, you can use /maxtext/src/MaxText/utils/ckpt_conversion/to_maxtext.py
config_ref = pyconfig.initialize(
    [
        "",
        f"{HOME}/maxtext/src/MaxText/configs/base.yml",
    ],
    base_output_directory="dummy",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",
    tokenizer_type="huggingface",
    tokenizer_path="meta-llama/Llama-3.1-8B",
    load_parameters_path=MODEL_CHECKPOINT_PATH,
    per_device_batch_size=1,
    max_prefill_predict_length=4,
    max_target_length=1024,
    steps=10,
    async_checkpointing="false",
    model_name="llama3.1-8b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
)

llama3_8b, mesh = get_ref_maxtext_model(config_ref)

llama3_8b.config = model_config

nnx.display(llama3_8b)


if DEBUG:
  print("Model initialized successfully")
  print(f"Model mesh shape: {mesh.shape}")
  print(f"Model config: {model_config}")

  # Sanity check that weights are loaded correctly
  _maxtext_state_flatten = nnx.state(llama3_8b).flat_state()
  maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
  print(
      f"maxtext_state_flatten[base.token_embedder.embedding].value="
      f"{maxtext_state_flatten['base.token_embedder.embedding'].value}"
  )


# See the memory use after loading the reference model:
print("HBM usage after loading ref model:")
show_hbm_usage()


# Configure and load the policy model
config_policy = pyconfig.initialize(
    [
        "",
        f"{HOME}/maxtext/src/MaxText/configs/base.yml",
    ],
    base_output_directory="dummy",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",  # This is not used in Tunix.
    tokenizer_type="huggingface",
    tokenizer_path="meta-llama/Llama-3.1-8B",
    load_parameters_path=MODEL_CHECKPOINT_PATH,
    per_device_batch_size=1,
    max_prefill_predict_length=4,
    max_target_length=1024,
    steps=10,
    async_checkpointing="false",
    model_name="llama3.1-8b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
)
llama3_8b_policy, mesh_policy = get_ref_maxtext_model(config_policy)

llama3_8b_policy.config = model_config

nnx.display(llama3_8b_policy)

if DEBUG:
  print("Model initialized successfully")
  print(f"Model mesh shape: {mesh_policy.shape}")

  # Sanity check that weights are loaded correctly
  _maxtext_state_flatten = nnx.state(llama3_8b_policy).flat_state()
  maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
  print(
      f"maxtext_state_flatten[base.token_embedder.embedding].value="
      f"{maxtext_state_flatten['base.token_embedder.embedding'].value}"
  )

# See memory usage after loading the policy model:
print("HBM usage after loading policy model:")
show_hbm_usage()


# ==============================================================================
# STEP 6: DEFINE REWARD FUNCTIONS
# ==============================================================================
#
# Reward functions are the heart of GRPO - they tell the model what behavior
# to reinforce. We define FOUR reward functions for the GSM8K task:
#
# 1. match_format_exactly: Rewards exact format compliance
#    - Checks if output has <reasoning>...</reasoning> and <answer>...</answer>
#    - Reward: +3.0 points
#
# 2. match_format_approximately: Rewards partial format compliance
#    - Checks if special tokens appear once each (no duplicates/missing)
#    - Reward: +0.5 per correct token, -0.5 penalty per incorrect
#
# 3. check_answer: Rewards correct numerical answers
#    - Exact match: +3.0
#    - With whitespace differences: +1.5
#    - Within 10% of correct: +0.5
#    - Within 20% of correct: +0.25
#    - Wrong answer: -1.0
#
# 4. check_numbers: Fallback for extracting numbers from verbose answers
#    - Extracts first number from <answer> section
#    - Exact match: +1.5
#
# Inspiration: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

# Regular expression to match the expected format
match_format = re.compile(
    rf"^[\s]{{0,}}" rf"{reasoning_start}.+?{reasoning_end}.*?" rf"{solution_start}(.+?){solution_end}" rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Test the regex (optional verification)
match_format.search(
    f"{reasoning_start}Let me" f" think!{reasoning_end}{solution_start}2{solution_end}",
)


# --- Reward Function 1: Exact Format Match ---


def match_format_exactly(prompts, completions, **kargs):
  """Rewards completions that exactly match the specified format.

  Args:
    prompts: The prompts used to generate completions.
    completions: The generated completions.
    **kargs: Additional keyword arguments.

  Returns:
    A list of scores for each completion.
  """
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += REWARD_EXACT_FORMAT_MATCH
    scores.append(score)
  return scores


# --- Reward Function 2: Approximate Format Match ---


def match_format_approximately(prompts, completions, **kargs):
  """Rewards completions that approximately match the specified format.

  Args:
    prompts: The prompts used to generate completions.
    completions: The generated completions.
    **kargs: Additional keyword arguments.

  Returns:
    A list of scores for each completion.
  """
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += REWARD_PARTIAL_FORMAT_MATCH if response.count(reasoning_start) == 1 else PENALTY_INCORRECT_FORMAT
    score += REWARD_PARTIAL_FORMAT_MATCH if response.count(reasoning_end) == 1 else PENALTY_INCORRECT_FORMAT
    score += REWARD_PARTIAL_FORMAT_MATCH if response.count(solution_start) == 1 else PENALTY_INCORRECT_FORMAT
    score += REWARD_PARTIAL_FORMAT_MATCH if response.count(solution_end) == 1 else PENALTY_INCORRECT_FORMAT
    scores.append(score)
  return scores


# --- Reward Function 3: Answer Correctness Check ---
#
# This function rewards correct answers with partial credit for close answers


def check_answer(prompts, completions, answer, **kargs):
  """Checks if the answer in the completion is correct and rewards accordingly.

  Args:
    prompts: The prompts used to generate completions.
    completions: The generated completions.
    answer: The ground truth answers.
    **kargs: Additional keyword arguments.

  Returns:
    A list of scores for each completion.
  """
  responses = completions

  extracted_responses = [guess.group(1) if (guess := match_format.search(r)) is not None else None for r in responses]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += REWARD_EXACT_FORMAT_MATCH
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += REWARD_WHITE_SPACE_FORMAT_MATCH
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if 0.9 <= ratio <= 1.1:
          score += REWARD_RATIO_GUESS_TO_ANSWER_HIGH
        elif 0.8 <= ratio <= 1.2:
          score += REWARD_RATIO_GUESS_TO_ANSWER_LOW
        else:
          score += PENALTY_INCORRECT_ANSWER  # Penalize wrong answers
      except (ValueError, TypeError, ZeroDivisionError):
        score += PENALTY_INCORRECT_FORMAT  # Penalize
    scores.append(score)
  return scores


# --- Reward Function 4: Number Extraction Fallback ---
#
# Sometimes the answer section contains text instead of just a number.
# This function extracts the first number found and checks correctness.
# Useful when the model provides verbose answers like "<answer>The answer is 42</answer>"

# Regex to extract the first number from the answer section
match_numbers = re.compile(rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")


def check_numbers(prompts, completions, answer, **kargs):
  """Extracts numbers from completions and rewards if they match the answer.

  Args:
    prompts: The prompts used to generate completions.
    completions: The generated completions.
    answer: The ground truth answers.
    **kargs: Additional keyword arguments.

  Returns:
    A list of scores for each completion.
  """
  question = kargs["question"]
  responses = completions

  extracted_responses = [guess.group(1) if (guess := match_numbers.search(r)) is not None else None for r in responses]

  scores = []
  if DEBUG:
    print("START ============================")
    print(f"Question: {question[0]}")
    print(f"Answer: {answer[0]}")
    print(f"Response: {responses[0]}")
    print(f"Extracted: {extracted_responses[0]}")
    print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except (ValueError, TypeError):
      scores.append(0)
      continue
  return scores


# ==============================================================================
# STEP 7: DEFINE EVALUATION FUNCTIONS
# ==============================================================================
#
# Evaluation helps us measure model performance before and after training.
# We'll run evaluation both BEFORE and AFTER GRPO training to measure improvement.
#
# Evaluation Metrics:
# ------------------
# QUANTITATIVE:
#   1. Answer Accuracy: % of samples with exact correct numerical answer
#   2. Answer Partial Accuracy: % of samples where answer is within ±10% of truth
#   3. Format Accuracy: % of samples with correct <reasoning> and <answer> format
#
# QUALITATIVE:
#   - We can also sample and manually inspect specific model outputs
#   - Useful for understanding HOW the model's reasoning improves
#
# The evaluation functions:
#   - generate_responses(): Uses vLLM to generate model outputs
#   - score_responses(): Applies reward functions to score outputs
#   - evaluate(): Runs full evaluation pipeline and returns metrics


def generate_responses(
    prompts,
    rl_cluster,
    num_passes=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
):
  """
  Generate responses for a batch of prompts across multiple passes.

  Args:
      prompts: List of prompts to generate responses for
      rl_cluster: Model cluster for generation
      num_passes: Number of generation passes
      temperature: Sampling temperature
      top_k: Top-k sampling parameter
      top_p: Top-p sampling parameter

  Returns:
      List of lists containing responses for each prompt across passes
  """
  multiple_call_responses = [[] for _ in range(len(prompts))]

  for p in range(num_passes):
    responses = rl_cluster.rollout.generate(
        prompts,
        rollout_config=RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ),
    )
    responses = responses.text

    if DEBUG:
      print(f"Pass {p+1}/{num_passes}, responses: {responses}")

    for idx, response in enumerate(responses):
      multiple_call_responses[idx].append(response)

  return multiple_call_responses


def score_responses(question, responses, answer):
  """
  Score a set of responses for a single question.

  Args:
      question: The evaluation question
      responses: List of generated responses for this question
      answer: The correct answer

  Returns:
      Tuple of (is_correct, is_partially_correct, has_correct_format)
  """
  if DEBUG:
    print("========================================")
    print(f"Evaluation Question: {question}")
    print(f"Evaluation Answer: {answer}")
    print(f"Evaluation Responses: {responses}")
    print("========================================")

  is_correct = False
  is_partially_correct = False
  has_correct_format = False

  for response in responses:
    # Extract numerical response
    extracted_response = guess.group(1) if (guess := match_numbers.search(response)) is not None else "-1000000"

    if DEBUG:
      print(f"Evaluation extracted_response: {extracted_response}")

    # Check exact correctness
    try:
      if float(extracted_response.strip()) == float(answer.strip()):
        is_correct = True

      # Check partial correctness (within 10%)
      ratio = float(extracted_response.strip()) / float(answer.strip())
      if 0.9 <= ratio <= 1.1:
        is_partially_correct = True
    except (ValueError, TypeError, ZeroDivisionError) as e:
      if DEBUG:
        print(f"Evaluation Exception: {e}")
        print("SKIPPED")

    # Check format correctness
    if match_format.search(response) is not None:
      has_correct_format = True

    # Early exit if all criteria are met
    if is_correct and is_partially_correct and has_correct_format:
      break

  return is_correct, is_partially_correct, has_correct_format


def evaluate(
    dataset,
    rl_cluster,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """
  Computes accuracy and percentage of outputs matching the format.

  Args:
      dataset: The evaluation dataset
      rl_cluster: Model cluster for generation
      temperature: Sampling temperature
      top_k: Top-k sampling parameter
      top_p: Top-p sampling parameter
      num_passes: Number of generation passes
      corr_lst: If True, only include correct responses in the list
      make_lst: If True, return a list of (question, answer, responses)

  Returns:
      Tuple of statistics and optionally the response list
  """
  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]
    prompts = batch["prompts"]

    # Generate responses for all prompts in the batch
    multiple_call_responses = generate_responses(
        prompts=prompts,
        rl_cluster=rl_cluster,
        num_passes=num_passes,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Score each question-answer pair
    for question, responses, answer in zip(questions, multiple_call_responses, answers):
      is_correct, is_partially_correct, has_correct_format = score_responses(
          question=question,
          responses=responses,
          answer=answer,
      )

      # Update counters
      if is_correct:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, responses))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, responses))

      if is_partially_correct:
        partially_corr += 1

      if has_correct_format:
        corr_format += 1

      total += 1

      # Print progress every 10 items
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  # Prepare return values
  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )

  if make_lst:
    return to_return, response_lst
  return to_return


# ==============================================================================
# STEP 8: MAIN TRAINING PIPELINE
# ==============================================================================
#
# The main() function orchestrates the entire GRPO training workflow:
#
# 1. Setup Infrastructure:
#    - Configure checkpointing (save model every N steps)
#    - Setup metrics logging (TensorBoard)
#    - Configure profiling
#
# 2. Create Optimizer:
#    - AdamW optimizer with warmup + cosine decay learning rate schedule
#    - Gradient clipping for stability
#
# 3. Setup RL Cluster:
#    - Combines policy model, reference model, and tokenizer
#    - Configures vLLM for rollout (response generation)
#    - Sets up mesh for distributed training
#
# 4. Initialize GRPO Learner:
#    - Combines RL cluster with reward functions
#    - Configures GRPO-specific hyperparameters (beta, epsilon, etc.)
#
# 5. Pre-Training Evaluation:
#    - Measure baseline performance before training
#
# 6. Training Loop:
#    - GRPO trainer runs for MAX_STEPS
#    - Each step: generate responses → compute rewards → update policy
#
# 7. Post-Training Evaluation:
#    - Measure final performance to see improvement
#
# Let's begin!

def main():
  # --- 1. Setup Infrastructure ---

  # Checkpoint manager: saves model weights periodically
  checkpointing_options = ocp.CheckpointManagerOptions(save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP)

  # Metrics logger: tracks training metrics for TensorBoard
  metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=LOG_DIR, flush_every_n_steps=20)

  # Print TensorBoard command for monitoring
  print(f"TensorBoard logs directory: {LOG_DIR}")
  print(f"tensorboard --logdir {LOG_DIR} --port=8086")

  # --- 2. Create Optimizer with Learning Rate Schedule ---

  # AdamW optimizer with warmup + cosine decay schedule
  # LR starts at 0, increases to LEARNING_RATE over WARMUP_STEPS,
  # then decreases to 0 following a cosine curve
  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
          init_value=0.0,                # Start LR at zero
          peak_value=LEARNING_RATE,      # Peak LR after warmup
          warmup_steps=WARMUP_STEPS,     # Linear warmup period
          decay_steps=MAX_STEPS,         # Total steps for cosine decay
          end_value=0.0,                 # End LR at zero
      ),
      b1=B1,                             # Adam beta1 (momentum)
      b2=B2,                             # Adam beta2 (variance)
      weight_decay=WEIGHT_DECAY,         # L2 regularization
  )

  # Add gradient clipping for training stability
  # Prevents exploding gradients and helps control KL divergence
  if MAX_GRAD_NORM is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
        optimizer,
    )

  # --- 3. Setup RL Cluster Configuration ---

  # The RL Cluster manages three roles:
  # - ACTOR (policy model): generates responses and gets trained
  # - REFERENCE: frozen model for KL divergence computation
  # - ROLLOUT: vLLM engine for efficient generation during training
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,        # Policy model mesh
          rl_cluster_lib.Role.REFERENCE: mesh,    # Reference model mesh
          rl_cluster_lib.Role.ROLLOUT: mesh,      # vLLM rollout mesh
      },
      rollout_engine="vllm",                      # Use vLLM for fast generation
      offload_to_cpu=False,                       # Keep everything on TPU/GPU
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,              # Optimizer for policy model
          eval_every_n_steps=EVAL_EVERY_N_STEPS,  # Validation frequency
          max_steps=MAX_STEPS,                    # Total training steps
          gradient_accumulation_steps=None,       # No gradient accumulation
          metrics_logging_options=metrics_logging_options,  # TensorBoard logging
          checkpoint_root_directory=CKPT_DIR,                # Checkpoint save dir
          checkpointing_options=checkpointing_options,       # Checkpoint frequency
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=TOTAL_GENERATION_STEPS,    # Max response length
          max_prompt_length=MAX_PROMPT_LENGTH,              # Max input length
          kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,  # Cache size
          temperature=TEMPERATURE,                          # Sampling temperature
          top_p=TOP_P,                                      # Nucleus sampling
          top_k=TOP_K,                                      # Top-k sampling
      ),
      rollout_vllm_model_version="meta-llama/Llama-3.1-8B",  # HuggingFace model ID for vLLM
      rollout_vllm_hbm_utilization=0.2,            # vLLM memory usage (20%)
      rollout_vllm_tpu_backend_type="jax",         # Use JAX backend for vLLM
  )

  # --- 4. Initialize GRPO Configuration ---

  # GRPO-specific hyperparameters
  grpo_config = GrpoConfig(
      num_generations=NUM_GENERATIONS,  # Responses per prompt (group size)
      num_iterations=NUM_ITERATIONS,    # Optimization iterations per batch
      beta=BETA,                        # KL divergence penalty coefficient
      epsilon=EPSILON,                  # PPO-style clipping parameter
  )

  # Create RL Cluster: combines models and configuration
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=llama3_8b_policy,          # Policy model (trainable)
      reference=llama3_8b,             # Reference model (frozen)
      tokenizer=model_tokenizer,       # Tokenizer for both models
      cluster_config=cluster_config,   # Cluster configuration
  )

  # Create GRPO Trainer: combines RL cluster with reward functions
  grpo_trainer = GrpoLearner(
      rl_cluster=rl_cluster,
      reward_fns=[                     # List of reward functions to use
          match_format_exactly,        # Reward 1: Exact format match
          match_format_approximately,  # Reward 2: Approximate format match
          check_answer,                # Reward 3: Answer correctness
          check_numbers,               # Reward 4: Number extraction fallback
      ],
      grpo_config=grpo_config,         # GRPO hyperparameters
  )

  # Debug: Test vLLM generation (optional sanity check)
  if DEBUG:
    print("Testing vLLM generation...")
    output = rl_cluster.rollout.generate(
        ["The capital of France is"],
        rollout_config=RolloutConfig(max_tokens_to_generate=64, temperature=0.1),
    )
    print(f"vLLM test output: {output}")

  # --- 5. Pre-Training Evaluation ---
  #
  # Evaluate model BEFORE training to establish a baseline
  # This helps us measure how much GRPO improves the model

  print("\n" + "="*80)
  print("EVALUATING MODEL BEFORE GRPO TRAINING")
  print("="*80 + "\n")

  # pylint: disable=unbalanced-tuple-unpacking
  (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
      test_dataset,
      rl_cluster,
      **GENERATION_CONFIGS["greedy"],  # Use greedy decoding for deterministic eval
  )
  print(f"\nPre-Training Results:")
  print(f"  Correct: {corr}/{total}")
  print(f"  Answer Accuracy: {accuracy:.2f}%")
  print(f"  Partial Accuracy: {partial_accuracy:.2f}%")
  print(f"  Format Accuracy: {format_accuracy:.2f}%\n")

  # --- 6. Training Loop ---
  #
  # This is where the magic happens! GRPO training loop:
  # For each batch:
  #   1. Generate multiple responses per prompt (using vLLM)
  #   2. Score responses using reward functions
  #   3. Compute advantages (how much better than group average)
  #   4. Update policy to increase probability of high-reward responses
  #   5. Apply KL penalty to prevent drift from reference model

  print("="*80)
  print("STARTING GRPO TRAINING")
  print("="*80 + "\n")

  # Start JAX profiler for performance analysis
  # Uncomment to enable profiling (note: generates large trace files for long training runs)
  # jax.profiler.start_trace(PROFILE_DIR)

  # Run training with proper mesh and axis rules for distributed training
  with mesh, nn_partitioning.axis_rules(config_policy.logical_axis_rules):
    grpo_trainer.train(DATASET)

  # Stop profiler (uncomment if profiling is enabled above)
  # jax.profiler.stop_trace()

  print("\n" + "="*80)
  print("TRAINING COMPLETE")
  print("="*80 + "\n")

  # Check memory usage after training
  print("HBM usage after training:")
  show_hbm_usage()

  # --- 7. Post-Training Evaluation ---
  #
  # Evaluate model AFTER training to measure improvement

  print("\n" + "="*80)
  print("EVALUATING MODEL AFTER GRPO TRAINING")
  print("="*80 + "\n")

  # pylint: disable=unbalanced-tuple-unpacking
  (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
      test_dataset,
      rl_cluster,
      **GENERATION_CONFIGS["greedy"],
  )
  print(f"\nPost-Training Results:")
  print(f"  Correct: {corr}/{total}")
  print(f"  Answer Accuracy: {accuracy:.2f}%")
  print(f"  Partial Accuracy: {partial_accuracy:.2f}%")
  print(f"  Format Accuracy: {format_accuracy:.2f}%\n")

  print("="*80)
  print("GRPO TUTORIAL COMPLETE!")
  print("="*80)


if __name__ == "__main__":
  main()
