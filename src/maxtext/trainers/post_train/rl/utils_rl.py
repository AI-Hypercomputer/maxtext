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

# pylint: disable=bare-except, consider-using-generator, chained-comparison, broad-exception-caught
"""RL Utils Module."""
import re
import optax
from maxtext.utils import max_logging


from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify import parse

# initialize math_verify_func once
math_verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    ),
)


def boxed(x):
  """Wraps the input string in a LaTeX boxed command if it's not already wrapped."""
  return "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


EPSILON = 1e-6
# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


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
  if tmvp_config.debug.rl:
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


def normalize_final_answer(final_answer: str) -> str:
  """Normalize a final answer to a quantitative reasoning question.

  Args:
      final_answer: The answer string to normalize

  Returns:
      Normalized answer string
  """
  final_answer = final_answer.split("=")[-1]

  # Apply substitutions and removals
  for before, after in SUBSTITUTIONS:
    final_answer = final_answer.replace(before, after)
  for expr in REMOVED_EXPRESSIONS:
    final_answer = final_answer.replace(expr, "")

  # Extract and normalize LaTeX math
  final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
  final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

  # Normalize shorthand TeX:
  #  \fracab -> \frac{a}{b}
  #  \frac{abc}{bef} -> \frac{abc}{bef}
  #  \fracabc -> \frac{a}{b}c
  #  \sqrta -> \sqrt{a}
  #  \sqrtab -> sqrt{a}b
  final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
  final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
  final_answer = final_answer.replace("$", "")

  # Normalize numbers
  if final_answer.replace(",", "").isdigit():
    final_answer = final_answer.replace(",", "")

  return final_answer


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
    # Normalize for certain datasets
    if "DAPO" in tmvp_config.dataset_name or "OpenMathInstruct" in tmvp_config.dataset_name:
      guess = normalize_final_answer(guess)
      true_answer = normalize_final_answer(true_answer)
    # Try math_verify first for robust comparison
    verified_correct = False
    mv_output = None
    true_answer_fixed = true_answer
    guess_fixed = guess
    try:
      # Fix LaTeX escaping issues for both ground truth and extracted answer
      true_answer_fixed = fix_latex_escaping(true_answer)
      guess_fixed = fix_latex_escaping(guess)

      mv_output = math_verify_func([boxed(true_answer_fixed)], [boxed(guess_fixed)])
      if mv_output and mv_output[0] > 0.1:
        verified_correct = True
    except (TimeoutException, Exception):
      pass

    # Correct answer gets tmvp_config.reward_exact_format_match points!
    if guess == true_answer:
      score += tmvp_config.reward_exact_format_match
    # Give credit if spaces are seen but otherwise the answers match (useful for simple datasets like gsm8k)
    elif guess.strip() == true_answer.strip():
      score += tmvp_config.reward_white_space_format_match
    # Answers match upon robust comparison with math_verify
    elif verified_correct:
      score += tmvp_config.reward_exact_format_match
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        # Fix LaTeX escaping issues for both ground truth and extracted answer
        true_answer_fixed = fix_latex_escaping(true_answer)
        guess_fixed = fix_latex_escaping(guess)
        val_true = parse(boxed(true_answer_fixed.strip()))
        val_guess = parse(boxed(guess_fixed.strip()))

        ratio = (val_guess[0] + EPSILON) / (val_true[0] + EPSILON)
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
  if tmvp_config.debug.rl:
    match_numbers.findall(f"{tmvp_config.solution_start_token}  0.34  {tmvp_config.solution_end_token}")
  return match_numbers


def fix_latex_escaping(text: str) -> str:
  """Fix common LaTeX commands that lost their backslashes due to Python string escaping.

  This handles cases where someone writes "\frac" in a regular string and it becomes "frac".
  Also handles cases where escape sequences like \f, \n, \t, \r, \b, \a, \v were interpreted by Python.
  """
  # First, try to recover from Python escape sequences that were interpreted
  # Map of (escape_char, LaTeX_suffix) -> LaTeX_command
  escape_fixes = [
      ("\f", "rac", r"\frac"),  # \f (form feed) → \frac
      ("\n", "ewline", r"\newline"),  # \n (newline) → \newline
      ("\n", "e", r"\ne"),  # \n (newline) → \ne (not equal)
      ("\t", "heta", r"\theta"),  # \t (tab) → \theta
      ("\t", "an", r"\tan"),  # \t (tab) → \tan
      ("\t", "o", r"\to"),  # \t (tab) → \to
      ("\t", "imes", r"\times"),  # \t (tab) → \times
      ("\t", "ext", r"\text"),  # \t (tab) → \text
      ("\t", "extbf", r"\textbf"),  # \t (tab) → \textbf
      ("\t", "extit", r"\textit"),  # \t (tab) → \textit
      ("\r", "ightarrow", r"\rightarrow"),  # \r (carriage return) → \rightarrow
      ("\r", "ightarrow", r"\Rightarrow"),  # \r (carriage return) → \Rightarrow (capital R handled separately)
      ("\b", "eta", r"\beta"),  # \b (backspace) → \beta
      ("\b", "ar", r"\bar"),  # \b (backspace) → \bar
      ("\b", "inom", r"\binom"),  # \b (backspace) → \binom
      ("\b", "oxed", r"\boxed"),  # \b (backspace) → \boxed
      ("\a", "lpha", r"\alpha"),  # \a (bell) → \alpha
      ("\a", "pprox", r"\approx"),  # \a (bell) → \approx
      ("\v", "ec", r"\vec"),  # \v (vertical tab) → \vec
  ]

  for escape_char, suffix, latex_cmd in escape_fixes:
    if escape_char in text:
      text = text.replace(escape_char + suffix, latex_cmd)

  # Common LaTeX commands that might have lost their backslashes
  latex_commands = [
      "frac",
      "sqrt",
      "pi",
      "theta",
      "alpha",
      "beta",
      "gamma",
      "delta",
      "sum",
      "int",
      "infty",
      "cdot",
      "times",
      "div",
      "pm",
      "mp",
      "leq",
      "geq",
      "neq",
      "approx",
      "equiv",
      "sin",
      "cos",
      "tan",
      "log",
      "ln",
      "exp",
      "lim",
      "to",
      "rightarrow",
      "leftarrow",
      "Rightarrow",
      "Leftarrow",
      "overline",
      "underline",
      "hat",
      "bar",
      "vec",
      "dot",
      "ddot",
      "mathbb",
      "mathbf",
      "mathrm",
      "text",
      "textbf",
      "textit",
      "boxed",
      "left",
      "right",
      "choose",
      "binom",
  ]

  # Add backslashes to commands that appear to be missing them
  for cmd in latex_commands:
    # Match the command if it appears as a word boundary (not already escaped)
    # and not already preceded by a backslash
    text = re.sub(rf"(?<!\\)\b{cmd}\b", rf"\\{cmd}", text)

  return text


def check_numbers(prompts, completions, answer, tmvp_config, **kargs):
  """
  Reward the model if the answer is correct using math_verify for robust comparison.
  Handles both numeric values and mathematical expressions with LaTeX.
  """
  question = kargs["question"]

  # Extract full answer content from solution tags (not just first number)
  match_format = get_match_format_regex(tmvp_config)
  extracted_responses = [guess.group(1) if (guess := match_format.search(c)) is not None else None for c in completions]

  scores = []
  if tmvp_config.debug.rl:
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

    # Try math_verify first for robust comparison of both numbers and expressions
    try:
      # Fix LaTeX escaping issues for both ground truth and extracted answer
      true_answer_fixed = fix_latex_escaping(true_answer)
      guess_fixed = fix_latex_escaping(guess)

      # Normalize for certain datasets
      if "DAPO" in tmvp_config.dataset_name or "OpenMathInstruct" in tmvp_config.dataset_name:
        true_answer_fixed = normalize_final_answer(true_answer_fixed)
        guess_fixed = normalize_final_answer(guess_fixed)

      # Use math_verify to compare answers (handles both numeric and expression comparison)
      score, _ = math_verify_func([boxed(true_answer_fixed)], [boxed(guess_fixed)])
      # Return scaled score: 1.5 for exact/correct, 0 otherwise
      scores.append(1.5 if score > 0.1 else 0.0)
    except (TimeoutException, Exception):
      # Fallback to simple numeric comparison if math_verify fails
      try:
        guess_val = float(normalize_final_answer(guess).strip())
        true_val = float(normalize_final_answer(true_answer).strip())
        scores.append(1.5 if guess_val == true_val else 0.0)
      except:
        scores.append(0)

  return scores


def extract_hash_answer(text: str) -> str | None:
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


def process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x):
  """Function to process input dataset"""

  def _to_str(val):
    if isinstance(val, bytes):
      return val.decode("utf-8")
    return str(val)

  # Handle DAPO dataset schema
  # originally (prompt is a list, answer is in reward_model)
  # https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/viewer/default/train?row=0
  # but using https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed/viewer/all/train?row=1
  # so question is prompt and answer is solution

  question = x.get("question", x.get("prompt"))
  answer = x.get("answer")
  if answer is None and "solution" in x:
    answer = x["solution"]

  # Handle OpenMathInstruct-2
  if "problem" in x:
    question = x["problem"]
  if "expected_answer" in x:
    answer = x["expected_answer"]

  # Handle AIME-2024
  if "extra_info" in x and isinstance(x["extra_info"], dict) and "raw_problem" in x["extra_info"]:
    question = x["extra_info"]["raw_problem"]

  if "reward_model" in x and isinstance(x["reward_model"], dict) and "ground_truth" in x["reward_model"]:
    answer = x["reward_model"]["ground_truth"]

  question = _to_str(question)
  answer = _to_str(answer)

  if dataset_name == "gsm8k":
    answer = extract_hash_answer(answer)

  return {
      # passed to model forward pass
      "prompts": model_tokenizer.apply_chat_template(
          [
              {
                  "role": "user",
                  "content": template_config["TEMPLATE"].format(
                      system_prompt=template_config["SYSTEM_PROMPT"].format(
                          reasoning_start_token=tmvp_config.reasoning_start_token,
                          reasoning_end_token=tmvp_config.reasoning_end_token,
                          solution_start_token=tmvp_config.solution_start_token,
                          solution_end_token=tmvp_config.solution_end_token,
                      ),
                      question=question,
                  ),
              },
          ],
          tokenize=False,
          add_generation_prompt=True,
      ),
      # passed to reward functions
      "question": question,
      # passed to reward functions
      "answer": answer,
  }
