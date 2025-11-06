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

# pylint: disable=bare-except, consider-using-generator, chained-comparison
"""RL Utils Module."""
import re
import optax
from MaxText import max_logging


# Let's define a RegEx for checking whether the format matches.
#
def get_match_format_regex(tmvp_config):
  """Returns a compiled regex to extract the answer from a completion."""
  match_format = re.compile(
      (
          r"^[\s]{0,}"
          rf"{tmvp_config.reasoning_start_token}.+?{tmvp_config.reasoning_end_token}.*?"
          rf"{tmvp_config.solution_start_token}(.+?){tmvp_config.solution_end_token}"
          r"[\s]{0,}$"
      ),
      flags=re.MULTILINE | re.DOTALL,
  )
  if tmvp_config.debug["rl"]:
    match_format.search(
        f"{tmvp_config.reasoning_start_token}Let me"
        f" think!{tmvp_config.reasoning_end_token}{tmvp_config.solution_start_token}2{tmvp_config.solution_end_token}",
    )
  return match_format


def match_format_exactly(prompts, completions, tmvp_config, **kargs):
  """
  Give the model a reward of tmvp_config.reward_exact_format_match points if the format matches exactly.
  """
  scores = []
  match_format = get_match_format_regex(tmvp_config)
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += tmvp_config.reward_exact_format_match
    scores.append(score)
  return scores


def match_format_approximately(prompts, completions, tmvp_config, **kargs):
  """
  We also reward the model if the format of the output matches partially.
  """
  scores = []

  for completion in completions:
    score = 0
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += (
        tmvp_config.reward_partial_format_match
        if completion.count(tmvp_config.reasoning_start_token) == 1
        else tmvp_config.penalty_incorrect_format
    )
    score += (
        tmvp_config.reward_partial_format_match
        if completion.count(tmvp_config.reasoning_end_token) == 1
        else tmvp_config.penalty_incorrect_format
    )
    score += (
        tmvp_config.reward_partial_format_match
        if completion.count(tmvp_config.solution_start_token) == 1
        else tmvp_config.penalty_incorrect_format
    )
    score += (
        tmvp_config.reward_partial_format_match
        if completion.count(tmvp_config.solution_end_token) == 1
        else tmvp_config.penalty_incorrect_format
    )
    scores.append(score)
  return scores


def check_answer(prompts, completions, answer, tmvp_config, **kargs):
  """
  Reward the model if the answer is correct. A reward is also given if the answer
  does not match exactly, i.e., based on how close the answer is to the correct
  value.
  """
  match_format = get_match_format_regex(tmvp_config)
  extracted_responses = [guess.group(1) if (guess := match_format.search(c)) is not None else None for c in completions]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets tmvp_config.reward_exact_format_match points!
    if guess == true_answer:
      score += tmvp_config.reward_exact_format_match
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += tmvp_config.reward_white_space_format_match
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += tmvp_config.reward_ratio_guess_to_answer_high
        elif ratio >= 0.8 and ratio <= 1.2:
          score += tmvp_config.reward_ratio_guess_to_answer_low
        else:
          score += tmvp_config.penalty_incorrect_answer  # Penalize wrong answers
      except:
        score += tmvp_config.penalty_incorrect_format  # Penalize
    scores.append(score)
  return scores


# Sometimes, the text between `<answer>` and `</answer>` might not be one
# number; it can be a sentence. So, we extract the number and compare the answer.


def get_match_numbers_regex(tmvp_config):
  """Returns a compiled regex to extract the answer from a completion."""
  match_numbers = re.compile(rf"{tmvp_config.solution_start_token}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL)
  if tmvp_config.debug["rl"]:
    match_numbers.findall(f"{tmvp_config.solution_start_token}  0.34  {tmvp_config.solution_end_token}")
  return match_numbers


def check_numbers(prompts, completions, answer, tmvp_config, **kargs):
  """
  Reward the model if the answer is correct.
  """
  question = kargs["question"]

  match_numbers = get_match_numbers_regex(tmvp_config)
  extracted_responses = [guess.group(1) if (guess := match_numbers.search(c)) is not None else None for c in completions]

  scores = []
  if tmvp_config.debug["rl"]:
    max_logging.log("START ============================")
    max_logging.log(f"Question: {question[0]}")
    max_logging.log(f"Answer: {answer[0]}")
    max_logging.log(f"Response: {completions[0]}")
    max_logging.log(f"Extracted: {extracted_responses[0]}")
    max_logging.log("END ==============================")
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


def extract_hash_answer(text: str, debug: bool = False) -> str | None:
  """Function to extract only the answer hash from the text."""
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def get_optimizer(tmvp_config, max_train_steps):
  """Function to obtain an optax optimizer, currently we use adamw."""
  optimizer = optax.adamw(
      learning_rate=optax.schedules.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=tmvp_config.learning_rate,
          # Linearly increase learning rate from 0. to learning_rate in the first
          # warmup_steps_fraction training steps, and then gradually decrease the
          # learning rate to 0 using cosine scheduler.
          warmup_steps=int(tmvp_config.warmup_steps_fraction * max_train_steps),
          decay_steps=max_train_steps,
          end_value=0.0,
      ),
      b1=tmvp_config.adam_b1,
      b2=tmvp_config.adam_b2,
      weight_decay=tmvp_config.adam_weight_decay,
  )

  # TODO: @mazumdera: try optimizer offloading with adamw
  # Add gradient clipping if specified
  # Grad clipping to prevent large gradients. We find this
  # important to keep KL divergence in check.
  if tmvp_config.gradient_clipping_threshold > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=tmvp_config.gradient_clipping_threshold),
        optimizer,
    )
  return optimizer
