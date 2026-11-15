"""Clean raw-text 2-shot GSM8K processor for base pretrained OLMo 3.5 checkpoints.

Avoids wrapping base pretrained models in instruct-only chat template tags
(`<|im_start|>user ...`) that were not seen during pretraining, while keeping
the exact `<reasoning>...</reasoning><answer>...</answer>` tags scored by
MaxText's built-in GSM8K reward stack (`match_format_exactly`,
`match_format_approximately`, `check_numbers`).
"""

import json
from typing import Any

from maxtext.trainers.post_train.rl import utils_rl


def process_data(
    dataset_name: str,
    model_tokenizer: Any,
    template_config: dict[str, Any],
    tmvp_config: Any,
    x: dict[str, Any],
) -> dict[str, str]:
  """Formats GSM8K examples as clean raw-text few-shot prompts."""
  del model_tokenizer  # Base checkpoint uses plain text without chat tags.

  def _to_str(val: Any) -> str:
    return val.decode("utf-8") if isinstance(val, bytes) else str(val)

  question = ""
  for key in ("problem", "prompt", "question"):
    if key in x:
      question = _to_str(x[key])
      break

  answer = ""
  for key in ("answer", "solution", "expected_answer"):
    if key in x:
      answer = _to_str(x[key])
      break

  if dataset_name == "openai/gsm8k":
    answer = utils_rl.extract_hash_answer(answer)

  question_type = _to_str(x["question_type"]) if "question_type" in x else "default"
  processed_answer = utils_rl.process_answer(question, answer, question_type)

  system_prompt = template_config["SYSTEM_PROMPT"].format(
      reasoning_start_token=tmvp_config.reasoning_start_token,
      reasoning_end_token=tmvp_config.reasoning_end_token,
      solution_start_token=tmvp_config.solution_start_token,
      solution_end_token=tmvp_config.solution_end_token,
  )
  prompt = template_config["TEMPLATE"].format(
      system_prompt=system_prompt,
      question=question,
  )
  return {
      "prompts": prompt,
      "question": question,
      "answer": json.dumps(processed_answer),
  }
