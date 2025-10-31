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
import numpy as np
from etils import epath

from tunix.rl.rollout.base_rollout import RolloutConfig

from MaxText.globals import MAXTEXT_ASSETS_ROOT
from maxtext.src.MaxText import utils_rl

# ## Evaluate
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
    tmvp_config,
    prompts,
    rl_cluster,
    num_passes=1,
):
  """
  Generate responses for a batch of prompts across potentially multiple passes.

  Args:
      tmvp_config: Configuration object
      prompts: List of prompts to generate responses for
      rl_cluster: Model cluster for generation
      num_passes: Number of generation passes

  Returns:
      List of lists containing responses for each prompt across passes
  """
  multiple_call_responses = [[] for _ in range(len(prompts))]

  eval_strategy = tmvp_config.generation_configs[tmvp_config.eval_sampling_strategy]
  for p in range(num_passes):
    responses = rl_cluster.rollout.generate(
        prompts,
        rollout_config=RolloutConfig(
            max_tokens_to_generate=tmvp_config.max_target_length,
            temperature=eval_strategy["eval_temperature"],
            top_k=eval_strategy["eval_top_k"],
            top_p=eval_strategy["eval_top_p"],
        ),
    )
    responses = responses.text

    if tmvp_config.debug:
      print(f"Pass {p+1}/{num_passes}, responses: {responses}")

    for idx, response in enumerate(responses):
      multiple_call_responses[idx].append(response)

  return multiple_call_responses


def score_responses(tmvp_config, question, responses, answer):
  """
  Score a set of responses for a single question.

  Args:
      question: The evaluation question
      responses: List of generated responses for this question
      answer: The correct answer

  Returns:
      Tuple of (is_correct, is_partially_correct, has_correct_format)
  """
  match_format = utils_rl.get_match_format_regex(tmvp_config)
  match_numbers = utils_rl.get_match_numbers_regex(tmvp_config)

  if tmvp_config.debug:
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

    if tmvp_config.debug:
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
      if tmvp_config.debug:
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
    tmvp_config,
    dataset,
    rl_cluster,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """
  Computes accuracy and percentage of outputs matching the format.

  Args:
      tmvp_config: Configuration object
      dataset: The evaluation dataset
      rl_cluster: Model cluster for generation.
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
        tmvp_config=tmvp_config,
        prompts=prompts,
        rl_cluster=rl_cluster,
        num_passes=num_passes,
    )

    # Score each question-answer pair
    for question, responses, answer in zip(questions, multiple_call_responses, answers):
      is_correct, is_partially_correct, has_correct_format = score_responses(
          tmvp_config=tmvp_config,
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
