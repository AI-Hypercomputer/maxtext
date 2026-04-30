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
RL Evaluation Module.
"""
import collections
import json
import re
from typing import Any

from tqdm.auto import tqdm
from tunix.rl.rollout.base_rollout import RolloutConfig

from maxtext.trainers.post_train.rl import utils_rl
from maxtext.utils import max_logging

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
# pylint: disable=broad-exception-caught


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
            max_tokens_to_generate=tmvp_config.max_target_length - tmvp_config.max_prefill_predict_length,
            temperature=eval_strategy["eval_temperature"],
            top_k=eval_strategy["eval_top_k"],
            top_p=eval_strategy["eval_top_p"],
        ),
    )
    responses = responses.text

    if tmvp_config.debug.rl:
      max_logging.log(f"Pass {p+1}/{num_passes}, responses: {responses}")

    for idx, response in enumerate(responses):
      multiple_call_responses[idx].append(response)

  return multiple_call_responses


def _score_single(
    extracted_response: str,
    raw_response: str,
    answers: list[str],
    tmvp_config: Any,
    match_format: re.Pattern[str],
) -> tuple[bool, bool, bool]:
  """Score one (extracted answer, raw response) pair. Returns (is_correct, is_partially_correct, has_correct_format)."""
  has_correct_format = match_format.search(raw_response) is not None
  try:
    is_correct, is_partially_correct = utils_rl.check_correctness(extracted_response, answers, tmvp_config)
    if tmvp_config.debug.rl:
      max_logging.log(f"Result has_correct_format: {has_correct_format}")
      max_logging.log(f"Result is_correct: {is_correct}")
      max_logging.log(f"Result is_partially_correct: {is_partially_correct}")
  except Exception as e:  # pylint: disable=broad-exception-caught
    is_correct, is_partially_correct = False, False
    if tmvp_config.debug.rl:
      max_logging.log(f"Evaluation Exception: {e} — SKIPPED")
  return is_correct, is_partially_correct, has_correct_format


def score_responses(tmvp_config, question, responses, answers):
  """Score a set of responses for a single question.

  Args:
      tmvp_config: Configuration object
      question: The evaluation question
      responses: List of generated responses for this question
      answers: List of correct answers

  Returns:
      Tuple of (is_correct, is_partially_correct, has_correct_format)
  """
  if tmvp_config.debug.rl:
    max_logging.log("========================================")
    max_logging.log(f"Evaluation Question: {question}")
    max_logging.log(f"Evaluation Answer: {answers}")
    max_logging.log(f"Evaluation Responses: {responses}")
    max_logging.log("========================================")

  eval_mode = getattr(tmvp_config, "eval_mode", "pass")
  match_format = utils_rl.get_match_format_regex(tmvp_config)
  extracted_responses = [utils_rl.extract_answer(r, tmvp_config) for r in responses]

  if not extracted_responses:
    return False, False, False

  if eval_mode == "maj":
    # extract the single-most frequent response
    counter = collections.Counter(extracted_responses)
    majority = counter.most_common(1)[0][0]
    if tmvp_config.debug.rl:
      max_logging.log(f"Majority Response: {majority} (Count: {counter[majority]})")

    # Check the format for the majority response
    has_correct_format = any(
        match_format.search(responses[idx]) is not None
        for idx, response in enumerate(extracted_responses)
        if response == majority
    )
    is_correct, is_partially_correct, _ = _score_single(majority, responses[0], answers, tmvp_config, match_format)
    return is_correct, is_partially_correct, has_correct_format

  if eval_mode == "pass":
    result = False, False, False
    for extracted, response in zip(extracted_responses, responses):
      result = _score_single(extracted, response, answers, tmvp_config, match_format)
      # Early exit if all criteria are met
      if all(result):
        return result
    return result

  if eval_mode == "pass_at_1":
    # Estimate pass@1: fraction of N samples that are correct per problem.
    # Returns floats in [0, 1] instead of booleans.
    scores = [
        _score_single(extracted_response, response, answers, tmvp_config, match_format)
        for extracted_response, response in zip(extracted_responses, responses)
    ]
    n_samples = len(scores)
    frac_correct = sum(s[0] for s in scores) / n_samples
    frac_partial = sum(s[1] for s in scores) / n_samples
    frac_format = sum(s[2] for s in scores) / n_samples
    if tmvp_config.debug.rl:
      max_logging.log(f"{frac_correct*n_samples:.0f}/{n_samples} correct")
      max_logging.log(f"{frac_partial*n_samples:.0f}/{n_samples} partial")
      max_logging.log(f"{frac_format*n_samples:.0f}/{n_samples} format")
    return frac_correct, frac_partial, frac_format

  raise ValueError(f"Unknown eval_mode: {eval_mode!r}")


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
      # decode the json-encoded list of acceptable answers
      answer = list(dict.fromkeys(json.loads(answer)))
      is_correct, is_partially_correct, has_correct_format = score_responses(
          tmvp_config=tmvp_config,
          question=question,
          responses=responses,
          answers=answer,
      )

      # Update counters. For "pass" and "maj" modes, scores are booleans
      # (True=1, False=0). For "pass_at_1" mode, scores are floats in [0, 1]
      # representing the fraction of samples correct. Using += works for both:
      # bool is a subtype of int in Python, so True += is the same as += 1.
      corr += is_correct
      partially_corr += is_partially_correct
      corr_format += has_correct_format

      if make_lst:
        if corr_lst and is_correct:
          response_lst.append((question, answer, responses))
        elif not corr_lst and not is_correct:
          response_lst.append((question, answer, responses))

      total += 1

      # Print progress every 10 items
      if total % 10 == 0:
        max_logging.log(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  # Prepare return values
  to_return = (
      corr,
      total,
      corr / total * 100 if total > 0 else 0,
      partially_corr / total * 100 if total > 0 else 0,
      corr_format / total * 100 if total > 0 else 0,
  )

  return to_return, response_lst
