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
DEPRECATED: This file is deprecated and kept for reference only.
Please use the new unified CLI interface: rl_trainer.py

See GRPO_README.md for migration guide and usage examples.

This tutorial demonstrates training the Llama3.1 8B-IT model on
 the GSM8K math reasoning benchmark using Group Relative Policy Optimization (GRPO).
   GRPO can enhance your model's problem-solving skills on mathematical word problems,
     coding problems, etc. """

# This tutorial demonstrates training the Llama3.1 8B-IT model on the GSM8K math
# reasoning benchmark using Group Relative Policy Optimization (GRPO). GRPO can
# enhance your model's problem-solving skills on mathematical word problems,
# coding problems, etc.
#
# GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It
# is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
# eliminating the need for a separate value function model. GRPO works by
# generating multiple responses for a given prompt, evaluating these responses
# using a reward model, and then calculating a relative advantage based on the
# group's performance to update the policy.
#
# We use Tunix as the library for GRPO.
# And we use vLLM as the library for efficient model inference and generation.
#
# In this tutorial we use a single host TPUVM such as `v6e-8/v5p-8`. Let's get started!


# ## Install necessary libraries

# ## Imports

import functools
import os
from pprint import pprint
import re
import sys

from datetime import datetime
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import grain
import humanize


import jax
from jax.sharding import Mesh
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.sft import metrics_logger

from transformers import AutoTokenizer

from flax import linen as nn
from tunix.models.llama3 import model as llama3_lib
import numpy as np
from etils import epath

from tunix.rl.rollout.base_rollout import RolloutConfig

from MaxText.globals import MAXTEXT_ASSETS_ROOT
import pathwaysutils

pathwaysutils.initialize()

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

# add the parent directory (two levels up to say ~/HOME/maxtext) to sys.path if currenlt runnig from
# ~/HOME/maxtext/MaxText/examples

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to get the project root
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from MaxText import maxtext_utils, model_creation_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter

# This is for running the script in a colab or notebook environment.
# import nest_asyncio
# nest_asyncio.apply()  # To fix "This event loop is already running" error in Colab
# Run `pip install nest_asyncio` if not already installed.

print(f"JAX devices: {jax.devices()}")

DEBUG = True  # set to True to run in debug mode, for more print statements

run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

HOME = os.path.expanduser("~") + "/"
print(f"Home directory (from Python): {HOME}")

# Look for base.yml in two possible locations.
path1 = os.path.join(HOME, "maxtext/src/MaxText/configs/base.yml")
path2 = "/deps/src/MaxText/configs/base.yml"
if os.path.exists(path1):
  BASE_YAML_PATH = path1
elif os.path.exists(path2):
  BASE_YAML_PATH = path2
else:
  raise FileNotFoundError("Could not find base.yml in the expected locations: " f"{path1} or {path2}")

# ## Hyperparameters
#
# Let's define the configuration we are going to use. Note that this is by no
# means a "perfect" set of hyperparameters. To get good results, you might have
# to train the model for longer.

# ====== Hardware =====
CHIPS_PER_VM = 4  # depends on hardware, for v5p this is 4

# ====== Data ======
TRAIN_DATA_DIR = f"{HOME}/data/train"
TEST_DATA_DIR = f"{HOME}/data/test"
if not os.path.exists(TRAIN_DATA_DIR):
  os.makedirs(TRAIN_DATA_DIR)
if not os.path.exists(TEST_DATA_DIR):
  os.makedirs(TEST_DATA_DIR)
TRAIN_FRACTION = 1.0


# ====== Input Checkpoint directory =====
MODEL_CHECKPOINT_PATH = "/path/to/scanned/model/ckpt_load_dir/"

# ====== Checkpoint directory =====
LOG_DIR = f"{HOME}/content/tensorboard/grpo/logs_llama3/"
if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)

# ===== Profiling =====
PROFILE_DIR = f"/path/to/profile_dir/{run_id}/profiles_llama3/"
if not epath.Path(PROFILE_DIR).exists():
  epath.Path(PROFILE_DIR).mkdir(parents=True)

# ====== Checkpoint saving ======
CKPT_DIR = f"/path/to/ckpt_save_dir/{run_id}/ckpts_llama3/"

if not epath.Path(CKPT_DIR).exists():
  epath.Path(CKPT_DIR).mkdir(parents=True)

SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Reproducibility ======
SEED = 42


# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
# NUM_BATCHES = 3738
NUM_BATCHES = 4  # 200
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 5  # 200

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = int(0.1 * MAX_STEPS)
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1


# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 0.01, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}
TRAINER_DEVICES_FRACTION = 0.5
SAMPLER_DEVICES_FRACTION = 0.5
HBM_UTILIZATION_VLLM = 0.2

# ====== Reward ======
REWARD_EXACT_FORMAT_MATCH = 3.0
REWARD_WHITE_SPACE_FORMAT_MATCH = 1.5
REWARD_PARTIAL_FORMAT_MATCH = 0.5
REWARD_RATIO_GUESS_TO_ANSWER_HIGH = 0.5
REWARD_RATIO_GUESS_TO_ANSWER_LOW = 0.25
PENALTY_INCORRECT_FORMAT = -0.5
PENALTY_INCORRECT_ANSWER = -1.0


# ## Data preprocessing
#
# First, let's define some special tokens. We instruct the model to first reason
# between the `<reasoning>` and `</reasoning>` tokens. After
# reasoning, we expect it to provide the answer between the `<answer>` and
# `</answer>` tokens.

# model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


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


dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test").batch(BATCH_SIZE)[:NUM_TEST_BATCHES]


# Let's see how one batch of the dataset looks like!


if DEBUG:
  for ele in train_dataset[:1]:
    pprint(ele)


# ## Load the policy model and the reference model
#
# The policy model is the model which is actually trained and whose weights are
# updated. The reference model is the model with which we compute KL divergence.
# This is to ensure that the policy updates are not huge and that it does not
# deviate too much from the reference model.
#
# Typically, the reference model is the base model, and the policy model is the
# same base model, but with potentially LoRA parameters where only the LoRA parameters are updated.
# This script is not using LoRA, so both the reference and policy models are the same.
#
# Note: We perform full precision (fp32) training. You can, however, leverage
# Qwix for QAT.


# ### Load MaxText model

# TODO: @mazumdera: create a installation script for GRPO
# ! uv pip install -r ../../maxtext/requirements.txt


def get_ref_maxtext_model(config, devices=None):

  model, mesh = model_creation_utils.create_nnx_model(config, devices)
  with mesh:
    tunix_model = TunixMaxTextAdapter(
        base_model=model,
    )

    model_config = llama3_lib.ModelConfig.llama3_1_8b()
    tunix_model.config = model_config

  return tunix_model, mesh


# Load the reference model
# Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
# To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py
config_ref = pyconfig.initialize(
    [
        "",
        BASE_YAML_PATH,
    ],
    base_output_directory="dummy",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",
    tokenizer_type="tiktoken",
    tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer_llama3.tiktoken"),
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

devices = jax.devices()
num_vms = len(devices) // CHIPS_PER_VM

if num_vms >= 2:
  print(f"{num_vms} VMs detected, allocating trainer and sampler devices")
  num_devices = len(devices)
  num_trainer_devices = int(num_devices * TRAINER_DEVICES_FRACTION)
  num_sampler_devices = int(num_devices * SAMPLER_DEVICES_FRACTION)
  trainer_devices = devices[:num_trainer_devices]
  sampler_devices = devices[num_devices - num_sampler_devices :]

  print("Creating reference model and also meshes for reference and rollout")
  llama3_1_8b, reference_mesh = get_ref_maxtext_model(config_ref, trainer_devices)
  devices_array = maxtext_utils.create_device_mesh(config_ref, sampler_devices)
  rollout_mesh = Mesh(devices_array, config_ref.mesh_axes)
  mesh = reference_mesh
else:
  llama3_1_8b, mesh = get_ref_maxtext_model(config_ref, devices)
  actor_mesh = mesh
  reference_mesh = mesh
  rollout_mesh = mesh


llama3_1_8b.config = None

nnx.display(llama3_1_8b)


if DEBUG:
  print("Model initialized successfully")
  print(f"Model mesh shape: {mesh.shape}")
  print(f"Model config: {None}")

  # Sanity check that weights are loaded correctly
  _maxtext_state_flatten = nnx.state(llama3_1_8b).flat_state()
  maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
  print(
      f"maxtext_state_flatten[base.token_embedder.embedding].value={maxtext_state_flatten['base.token_embedder.embedding'].value}"
  )


# Load the policy model
# Note: pass the path to your scanned checkpoint for "load_parameters_path". To generate a scanned checkpoint, you can use the `scanned_checkpoint.py` script in MaxText.
# To create a scanned checkpoint, you can use /maxtext/MaxText/utils/ckpt_conversion/to_maxtext.py

# TODO: @mazumdera: change this to use lora

config_policy = pyconfig.initialize(
    [
        "",
        BASE_YAML_PATH,
    ],
    base_output_directory="dummy",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",  # This is not used in Tunix.
    tokenizer_type="tiktoken",
    tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizer_llama3.tiktoken"),
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

if num_vms >= 2:
  # For the policy model, override the config to create a 4-device mesh.
  print("Creating policy model on trainer mesh")
  llama3_1_8b_policy, policy_mesh = get_ref_maxtext_model(config_policy, trainer_devices)
  actor_mesh = policy_mesh
else:
  llama3_1_8b_policy, policy_mesh = get_ref_maxtext_model(config_policy, devices)


llama3_1_8b_policy.config = None

nnx.display(llama3_1_8b_policy)

if DEBUG:
  print("Model initialized successfully")
  print(f"Model mesh shape: {policy_mesh.shape}")

  # Sanity check that weights are loaded correctly
  _maxtext_state_flatten = nnx.state(llama3_1_8b_policy).flat_state()
  maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
  print(
      f"maxtext_state_flatten[base.token_embedder.embedding].value={maxtext_state_flatten['base.token_embedder.embedding'].value}"
  )


# ## Define reward functions
#
# We define four reward functions:
#
# - reward if the format of the output exactly matches the instruction given in
# `TEMPLATE`;
# - reward if the format of the output approximately matches the instruction given
# in `TEMPLATE`;
# - reward if the answer is correct/partially correct;
# - Sometimes, the text between `<answer>`, `</answer>` might not be one
#   number. So, extract the number, and reward the model if the answer is correct.
#
# The reward functions are inspired from
# [here](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb).
#
# First off, let's define a RegEx for checking whether the format matches.
#

match_format = re.compile(
    rf"^[\s]{{0,}}" rf"{reasoning_start}.+?{reasoning_end}.*?" rf"{solution_start}(.+?){solution_end}" rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me" f" think!{reasoning_end}{solution_start}2{solution_end}",
)


def match_format_exactly(prompts, completions, **kargs):
  """
  Give the model a reward of REWARD_EXACT_FORMAT_MATCH points if the format matches exactly.
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


def match_format_approximately(prompts, completions, **kargs):
  """
  We also reward the model if the format of the output matches partially.
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


def check_answer(prompts, completions, answer, **kargs):
  """
  Reward the model if the answer is correct. A reward is also given if the answer
  does not match exactly, i.e., based on how close the answer is to the correct
  value.
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
        if ratio >= 0.9 and ratio <= 1.1:
          score += REWARD_RATIO_GUESS_TO_ANSWER_HIGH
        elif ratio >= 0.8 and ratio <= 1.2:
          score += REWARD_RATIO_GUESS_TO_ANSWER_LOW
        else:
          score += PENALTY_INCORRECT_ANSWER  # Penalize wrong answers
      except:
        score += PENALTY_INCORRECT_FORMAT  # Penalize
    scores.append(score)
  return scores


# Sometimes, the text between `<answer>` and `</answer>` might not be one
# number; it can be a sentence. So, we extract the number and compare the answer.


match_numbers = re.compile(rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")


def check_numbers(prompts, completions, answer, **kargs):
  """
  Reward the model if the answer is correct.
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
    except:
      scores.append(0)
      continue
  return scores


# ## Evaluate
#
#
# Before we train the model, let's evaluate the model on the test set so we can
# see the improvement post training.
#
# We evaluate it in two ways:
#
# **Quantitative**
#
# * **Answer Accuracy**: percentage of samples for which the model predicts the
# correct final numerical answer
# * **Answer (Partial) Accuracy**: percentage of samples for which the model
# predicts a final numerical answer such that the \`model answer / answer\`
# ratio lies between 0.9 and 1.1.
# * **Format Accuracy**: percentage of samples for which the model outputs the
# correct format, i.e., reasoning between the reasoning special tokens, and the
# final answer between the \`\<start\_answer\>\`, \`\<end\_answer\>\` tokens.
#
# **Qualitative**
#
# We'll also print outputs for a few given questions so that we can compare the generated output later.
#


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
    except Exception as e:
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


# ## Train
#
# Let's set up all the configs first - checkpointing, metric logging and training.
# We then train the model.


# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(log_dir=LOG_DIR, flush_every_n_steps=20)

# Profiler
profiler_options = None

# Logs
print(f"TensorBoard logs directory: {LOG_DIR}")
print(f"tensorboard --logdir {LOG_DIR} --port=8086")


# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
# TODO: @mazumdera: try optimizer offloading with adamw

if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )


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
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # profiling
        profiler_options=profiler_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
    rollout_vllm_model_version="meta-llama/Meta-Llama-3.1-8B-Instruct",
    rollout_vllm_hbm_utilization=HBM_UTILIZATION_VLLM,
    rollout_vllm_tpu_backend_type="jax",
)

grpo_config = GrpoConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)


# RL cluster


with nn_partitioning.axis_rules(config_policy.logical_axis_rules):
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=llama3_1_8b_policy,
      reference=llama3_1_8b,
      tokenizer=model_tokenizer,
      cluster_config=cluster_config,
  )

# GRPO Trainer
grpo_trainer = GrpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    grpo_config=grpo_config,
)


if DEBUG:
  # verify if vllm sampler works
  output = rl_cluster.rollout.generate(
      ["The capital of France is"],
      rollout_config=RolloutConfig(max_tokens_to_generate=64, temperature=0.1),
  )

  print(f"Output: {output}")


# ## Evaluate before training
#


(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    rl_cluster,
    **GENERATION_CONFIGS["greedy"],
)
print(f"Pre GRPO Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%," f" {format_accuracy=}%")


# ## Start training
#

# jax.profiler.start_trace(PROFILE_DIR)
with mesh, nn_partitioning.axis_rules(config_policy.logical_axis_rules):
  grpo_trainer.train(dataset)
# jax.profiler.stop_trace()

# ## Evaluate
#
# Let's evaluate our model!


(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    rl_cluster,
    **GENERATION_CONFIGS["greedy"],
)
print(f"Post GRPO Training: {corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%," f" {format_accuracy=}%")
