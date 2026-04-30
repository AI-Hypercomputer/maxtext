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
import itertools
import json
import re
import uuid
from typing import Any, Callable, Optional
from etils import epath
import optax
import numpy as np

from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify import parse

from tunix.rl.agentic.parser.chat_template_parser import parser as agentic_chat_template_parser

from maxtext.trainers.post_train.rl.math_verify_pool import math_verify_pool, verify_math_worker
from maxtext.utils import max_logging


EPSILON = 1e-6

FALLBACK_ANSWER = "-1000000"

# Constants for normalization
SUBSTITUTIONS = [
    # Collapse double backslashes first so subsequent rules see canonical form
    # (mirrors Tunix `_strip_string` line 116).
    ("\\\\", "\\"),
    # Tunix `_strip_string` lines 120-121: tfrac/dfrac → frac.
    ("\\tfrac", "\\frac"),
    ("\\dfrac", "\\frac"),
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    # Tunix `_normalize` lines 281-282: set-style answers.
    (" or ", ","),
    (" and ", ","),
    # Tunix `_normalize` lines 284-286: scale words.
    ("million", "*10^6"),
    ("billion", "*10^9"),
    ("trillion", "*10^12"),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

UNITS = [
    "yard",
    "foot",
    "feet",
    "mile",
    "day",
    "week",
    "month",
    "year",
    "hour",
    "minute",
    "second",
    "centimeter",
    "meter",
    "cm",
    "mm",
    "km",
    "inch",
    "degree",
    "pound",
    "cent",
    "mph",
]

REMOVED_EXPRESSIONS = [
    "\\left",
    "\\right",
    "\\!",
    "square",
    "ways",
    "integers",
    "dollars",
    "units",
    "\\ldots",
    "sue",
    "points",
    "digits",
    "gm",
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


def math_verify_func(
    items: list[tuple[int, list[str], list[str]]],
    scores: list[float],
    timeout: float = 300,
    trainer_config: Optional[Any] = None,
) -> list[float]:
  """Verifies a batch of math problems, handling timeouts and exceptions.

  Dispatches to a spawn-based multiprocessing pool (`math_verify_pool`)
  so that hung sympy calls inside `math_verify` can be
  killed via `pool.terminate()` and so the workers cannot contend for the
  trainer's TPU.
  """
  if not items:
    return scores

  num_procs = None
  if trainer_config is not None:
    timeout = getattr(trainer_config, "math_verify_timeout", timeout)
    num_procs = getattr(trainer_config, "math_verify_num_procs", None)

  return math_verify_pool(
      trainer_config,
      items,
      scores,
      timeout=timeout,
      num_procs=num_procs,
      log_fn=max_logging.log,
  )


def boxed(x: str) -> str:
  """Wraps the input string in a LaTeX boxed command if it's not already wrapped."""
  return "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


def get_match_format_regex(tmvp_config: Any) -> re.Pattern[str]:
  """Returns a compiled regex to extract the answer from a completion."""
  match_format = re.compile(
      (
          rf"{tmvp_config.reasoning_start_token}.+{tmvp_config.reasoning_end_token}.*?"
          rf"{tmvp_config.solution_start_token}(.+?){tmvp_config.solution_end_token}"
      ),
      flags=re.MULTILINE | re.DOTALL,
  )
  if tmvp_config.debug.rl:
    match_format.search(
        f"{tmvp_config.reasoning_start_token}Let me"
        f" think!{tmvp_config.reasoning_end_token}{tmvp_config.solution_start_token}2{tmvp_config.solution_end_token}",
    )
  return match_format


def get_answer_fallback_regex(tmvp_config: Any) -> re.Pattern[str]:
  """Returns a compiled regex that finds the *last* answer tag in a completion.

  Used as a fallback when the full <reasoning>...</reasoning><answer>...</answer>
  format is incomplete (e.g. missing the closing reasoning tag).  The result
  reward can still be computed independently from the format reward.
  """
  return re.compile(
      rf"{re.escape(tmvp_config.solution_start_token)}(.+?){re.escape(tmvp_config.solution_end_token)}",
      flags=re.MULTILINE | re.DOTALL,
  )


def match_format_exactly(prompts: list[str], completions: list[str], tmvp_config: Any, **kargs: Any) -> list[float]:
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


def match_format_approximately(prompts: list[str], completions: list[str], tmvp_config: Any, **kargs: Any) -> list[float]:
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

  # Inject implicit mixed numbers BEFORE the substitutions strip spaces
  # (mirrors Tunix `_inject_implicit_mixed_number`): "7 3/4" -> "7+3/4".
  final_answer = re.sub(r"([0-9]) +([0-9])", r"\1+\2", final_answer)

  # Apply substitutions and removals
  for before, after in SUBSTITUTIONS:
    final_answer = final_answer.replace(before, after)
  for unit in UNITS:
    final_answer = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", final_answer)
  for expr in REMOVED_EXPRESSIONS:
    final_answer = final_answer.replace(expr, "")

  # Extract and normalize LaTeX math
  final_answer = re.sub(
      r".*?(\d+)?\s*\$\s*(\d+)?\s*(\\frac\{.*?\}\{.*?\}|\d+/\d+)\s*\$.*",
      lambda m: f"${w}{m.group(3)}$" if (w := (m.group(1) or m.group(2))) else f"${m.group(3)}$",
      final_answer,
  )
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

  # Leading-zero fixups (Tunix `_strip_string` lines 143-149). Spaces have
  # already been stripped above, so we only need the start-of-string and
  # post-`{` cases.
  if final_answer.startswith("."):
    final_answer = "0" + final_answer
  final_answer = final_answer.replace("{.", "{0.")

  # Strip a single layer of outer braces (Tunix `_normalize` lines 309-310):
  # "{42}" -> "42".
  if len(final_answer) >= 2 and final_answer[0] == "{" and final_answer[-1] == "}":
    final_answer = final_answer[1:-1]

  # Integer-float collapse (Tunix `_normalize` lines 313-314): "2.0" -> "2".
  try:
    f = float(final_answer)
    if abs(f - round(f)) < 1e-7:
      final_answer = str(int(round(f)))
  except (ValueError, OverflowError):
    pass

  # Normalize numbers
  if final_answer.replace(",", "").isdigit():
    final_answer = final_answer.replace(",", "")

  return final_answer


def preprocess_math_string(text: str) -> str:
  """Fix common formatting issues in text."""
  # Normalize text
  text = normalize_final_answer(text).strip()
  # Fix LaTeX escaping issues
  text = fix_latex_escaping(text)
  return text


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
      ("\t", "extit", r"\textit"),  # \t (tab) → \textit  # codespell:ignore
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


def check_numbers(
    prompts: list[str], completions: list[str], answer: list[str], tmvp_config: Any, **kargs: Any
) -> list[float]:
  """
  Reward the model if the answer is correct using math_verify for robust comparison.
  Handles both numeric values and mathematical expressions with LaTeX.
  """
  question = kargs["question"]

  # Extract full answer content from solution tags (not just first number)
  extracted_responses = [extract_answer(c, tmvp_config) for c in completions]
  true_answers = [list(dict.fromkeys(json.loads(acceptable_answers))) for acceptable_answers in answer]

  scores = [tmvp_config.penalty_incorrect_format] * len(completions)  # Default to penalty for incorrect format
  math_verify_queue = []
  for gen_idx, (guess, unique_answers) in enumerate(zip(extracted_responses, true_answers)):
    if guess is None:
      continue

    if guess == FALLBACK_ANSWER:
      scores[gen_idx] = tmvp_config.penalty_incorrect_answer
      continue

    has_exact_match = False
    for true_answer in unique_answers:
      # 1. Check for exact or whitespace-normalized match first for a quick reward
      if guess == true_answer:
        scores[gen_idx] = max(scores[gen_idx], tmvp_config.reward_exact_answer)
        has_exact_match = True
      elif guess.strip() == true_answer.strip():
        scores[gen_idx] = max(scores[gen_idx], tmvp_config.reward_white_space_format_match)
        has_exact_match = True

    if not has_exact_match:
      norm_guess = preprocess_math_string(guess)
      norm_answers = []
      for true_answer in unique_answers:
        norm_answer = preprocess_math_string(true_answer)
        norm_answers.append(boxed(norm_answer))
      math_verify_queue.append((gen_idx, norm_answers, [boxed(norm_guess)]))

  if math_verify_queue:
    # 2. Try math_verify for robust mathematical correctness checking
    scores = math_verify_func(math_verify_queue, scores, trainer_config=tmvp_config)

  if tmvp_config.debug.rl:
    debug_log_path = epath.Path(tmvp_config.base_output_directory) / tmvp_config.run_name / "debug_rl_logs"
    debug_log_path.mkdir(parents=True, exist_ok=True)
    log_file = debug_log_path / f"check_numbers_{uuid.uuid4().hex}.txt"
    log_content = (
        "START ============================\n"
        f"Question: {question[0]}\n"
        f"Answer: {answer[0]}\n"
        f"Response: {completions[0]}\n"
        f"Extracted: {extracted_responses[0]}\n"
        f"Reward Score: {scores[0]}\n"
        "END ==============================\n"
    )
    log_file.write_text(log_content)

  return scores


def extract_answer(response: str, tmvp_config: Any) -> str:
  """Function to extract the answer from the text based on the tmvp_config format."""
  answer_fallback = get_answer_fallback_regex(tmvp_config)
  # Find the *last* occurrence of the answer tag (most likely the final answer).
  fallback_matches = answer_fallback.findall(response)
  extracted_response = fallback_matches[-1].strip() if fallback_matches else FALLBACK_ANSWER
  return extracted_response


def extract_hash_answer(text: str) -> str | None:
  """Function to extract only the answer hash from the text."""
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def check_correctness(extracted_response: str, acceptable_answers: list[str], tmvp_config: Any) -> tuple[bool, bool]:
  """Handles math verification and partial correctness logic."""
  norm_response = preprocess_math_string(extracted_response)
  norm_answers = []
  for answer in acceptable_answers:
    norm_answers.append(preprocess_math_string(answer))

  # Check exact correctness first
  score = verify_math_worker([boxed(norm_answer) for norm_answer in norm_answers], [boxed(norm_response)])
  if score > 0.0:
    return True, True  # Exact correctness implies partial correctness

  # Check partial correctness if values can be extracted (within 10%)
  is_partially_correct = False
  try:
    predictions = parse(boxed(norm_response), (ExprExtractionConfig(), LatexExtractionConfig()))
    golds = list(
        itertools.chain.from_iterable(
            parse(boxed(norm_answer), (ExprExtractionConfig(), LatexExtractionConfig())) for norm_answer in norm_answers
        )
    )
    is_partially_correct = any(
        0.9 <= (float(pred) + EPSILON) / (float(gold) + EPSILON) <= 1.1 for pred in predictions for gold in golds
    )
  except:
    if tmvp_config.debug.rl:
      max_logging.log(
          f"check_correctness failed for extracted response: {extracted_response} and answers: {acceptable_answers}"
      )

  return False, is_partially_correct


def get_optimizer(tmvp_config: Any, max_train_steps: int) -> optax.GradientTransformation:
  """Function to obtain an optax optimizer, currently we use adamw."""
  schedule = optax.schedules.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=tmvp_config.learning_rate,
      # Linearly increase learning rate from 0. to learning_rate in the first
      # warmup_steps_fraction training steps, and then gradually decrease the
      # learning rate to 0 using cosine scheduler.
      warmup_steps=int(tmvp_config.warmup_steps_fraction * max_train_steps),
      decay_steps=max_train_steps,
      end_value=0.0,
  )

  # TODO: @mazumdera: try optimizer offloading with adamw
  # Add gradient clipping if specified
  # Grad clipping to prevent large gradients. We find this
  # important to keep KL divergence in check.
  def make_optimizer(learning_rate):
    transforms = []
    if tmvp_config.gradient_clipping_threshold > 0:
      transforms.append(optax.clip_by_global_norm(max_norm=tmvp_config.gradient_clipping_threshold))
    transforms.append(
        optax.adamw(
            learning_rate=learning_rate,
            b1=tmvp_config.adam_b1,
            b2=tmvp_config.adam_b2,
            weight_decay=tmvp_config.adam_weight_decay,
        )
    )
    return optax.chain(*transforms)

  # Wrap the entire optimizer (including gradient clipping) with
  # inject_hyperparams so opt_state.hyperparams['learning_rate'] is at the
  # top level of the state tree. This is required for tunix's peft_trainer to
  # automatically read and log the per-step learning rate.
  return optax.inject_hyperparams(make_optimizer)(learning_rate=schedule)


def format_maxtext_messages(
    messages: list[str], template_config: dict[str, Any], tmvp_config: Any
) -> list[dict[str, str]]:
  """Helper to inject MaxText's system prompt into the input user messages."""
  if template_config is None:
    raise ValueError("template_config cannot be None for format_maxtext_messages.")

  formatted_messages = []
  for msg in messages:
    formatted_content = template_config["TEMPLATE"].format(
        system_prompt=template_config["SYSTEM_PROMPT"].format(
            reasoning_start_token=tmvp_config.reasoning_start_token,
            reasoning_end_token=tmvp_config.reasoning_end_token,
            solution_start_token=tmvp_config.solution_start_token,
            solution_end_token=tmvp_config.solution_end_token,
        ),
        question=msg,
    )
    formatted_messages.append({"role": "user", "content": formatted_content})
  return formatted_messages


def process_answer(question: str, answer: str, question_type: str) -> list[str]:
  """Function to process the answer based on the question type."""
  if question_type == "MCQ":
    # For MCQs, we need to process the response to get the acceptable answers
    # e.g., returns "10" and "A" if answer="A" and question="What is 5+5? (A) 10, (B) 11, ..."
    return process_mcq(question, answer)

  return [answer, answer]


def process_mcq(question: str, answer: str) -> list[str]:
  """Extracts options from MCQ question and returns a list of acceptable answers based on the provided answer key."""
  pattern = r"""
    (?:
        \(?([A-E])\)?              # Matches (A) or A
        |                          # OR
        \\(?:text|mathrm|textbf)   # Matches \text, \mathrm, or \textbf
        \{([A-E])\}                # Matches {A} or {B} or {C} etc.
    )
    [\s\.\:\}\$]*\s*               # Matches trailing punctuation/whitespace
    (.*?)                          # The actual content of the option
    (?=                            # Lookahead for the next option or end
        \s*
        (?:\\quad|\\qquad|\\hspace|\n|\r)
        \s*
        (?:\(?|\\(?:text|mathrm|textbf))
        | $
    )
  """
  matches = re.findall(pattern, question, re.DOTALL | re.VERBOSE)
  options = {}
  for m in matches:
    letter = m[0] or m[1]
    value = m[2].strip()
    clean_value = re.sub(r"\\q?quad|\\textbf{|[~$]", "", value).strip()
    options[letter] = clean_value

  # List of answer formats that should be accepted
  acceptable_answers = [answer]
  if answer in options:
    # If the answer is a Letter (e.g., "B"), add the corresponding value
    acceptable_answers.append(options[answer])
  else:
    # If the answer is a value, find the corresponding letter
    for letter, value in options.items():
      norm_value = normalize_final_answer(value)
      norm_answer = normalize_final_answer(answer)
      if norm_answer == norm_value:
        acceptable_answers.append(letter)
        break

  return acceptable_answers


def process_data(
    dataset_name: str,
    model_tokenizer: Any,
    template_config: dict[str, Any],
    tmvp_config: Any,
    x: dict[str, Any],
) -> dict[str, str]:
  """Function to process input dataset"""

  def _to_str(val):
    if isinstance(val, bytes):
      return val.decode("utf-8")
    return str(val)

  for key in ["problem", "prompt", "question"]:
    if key in x:
      question = _to_str(x[key])
      break

  for key in ["answer", "solution", "expected_answer"]:
    if key in x:
      answer = _to_str(x[key])
      break

  # Handle AIME-2024
  if "extra_info" in x and isinstance(x["extra_info"], dict) and "raw_problem" in x["extra_info"]:
    question = _to_str(x["extra_info"]["raw_problem"])
  if "reward_model" in x and isinstance(x["reward_model"], dict) and "ground_truth" in x["reward_model"]:
    answer = _to_str(x["reward_model"]["ground_truth"])

  if dataset_name == "openai/gsm8k":
    answer = extract_hash_answer(answer)

  question_type = "default"
  if "question_type" in x:
    question_type = _to_str(x["question_type"])
  processed_answer = process_answer(question, answer, question_type)

  messages = [question]
  formatted_messages = format_maxtext_messages(messages, template_config, tmvp_config)

  prompts = model_tokenizer.apply_chat_template(
      formatted_messages,
      tokenize=False,
      add_generation_prompt=True,
  )

  return {
      # passed to model forward pass
      "prompts": prompts,
      # passed to reward functions
      "question": question,
      # list of acceptable answers passed to reward functions
      "answer": json.dumps(processed_answer),  # string-encode the list to prevent grain from flattening it while batching
  }


def get_correctness_metrics(
    prompts: Any,
    completions: Any,
    rewards: np.ndarray,
    advantages: Any,
    **kwargs: Any,
) -> dict[str, tuple[float | int, Callable[..., Any]]]:
  """Compute correctness statistics metrics based on rewards."""
  del prompts, completions, advantages, kwargs
  solve_all = (rewards > 0.1).all()
  solve_none = (rewards == 0).all()
  solve_partial = (~solve_all) and (~solve_none)
  solve_ratio = (rewards > 0.1).mean()
  return {
      "rewards/solve_all": (
          1 if solve_all else 0,
          np.mean,
      ),
      "rewards/solve_none": (
          1 if solve_none else 0,
          np.mean,
      ),
      "rewards/solve_partial": (
          1 if solve_partial else 0,
          np.mean,
      ),
      "rewards/solve_ratio": (
          solve_ratio,
          np.mean,
      ),
  }


class MaxTextChatParser(agentic_chat_template_parser.DefaultChatTemplateParser):
  """
  Custom Chat Parser for MaxText that intercepts message lists dynamically
  during agentic rollouts and injects the necessary system templates and
  special tokens using the shared helper.
  """

  def __init__(self, model_tokenizer: Any, template_config: dict[str, Any], tmvp_config: Any) -> None:
    super().__init__(model_tokenizer)
    self.template_config = template_config
    self.tmvp_config = tmvp_config

  def parse(
      self,
      messages: list[dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    """Overrides the default parse method to apply MaxText-specific formatting to the messages."""
    # Apply MaxText specific formatting to the messages
    formatted_messages = format_maxtext_messages(messages, self.template_config, self.tmvp_config)

    # Delegate to Tunix default parser to apply the tokenizer's chat template
    return super().parse(
        messages=formatted_messages, add_generation_prompt=add_generation_prompt, is_first_msg=is_first_msg
    )
