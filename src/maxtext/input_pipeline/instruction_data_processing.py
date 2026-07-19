# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing for instruction dataset."""

import json
import importlib
import os
import re

from maxtext.utils import max_logging


def load_data_template_from_file(template_path):
  """Loads a data template from a file."""
  if not template_path:
    return None

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  template_full_path = os.path.join(repo_root, template_path)

  if not os.path.isfile(template_full_path):
    return None

  if template_full_path.endswith(".json"):
    with open(template_full_path, "r", encoding="utf-8") as f:
      try:
        return json.load(f)
      except json.JSONDecodeError:
        return None

  return None


def load_chat_template_from_file(template_path):
  """Loads a chat template from a file."""
  if not template_path:
    return None

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  template_full_path = os.path.join(repo_root, template_path)

  if not os.path.isfile(template_full_path):
    return None

  if template_full_path.endswith((".jinja", ".j2", ".txt")):
    with open(template_full_path, "r", encoding="utf-8") as f:
      return f.read()

  if template_full_path.endswith(".json"):
    with open(template_full_path, "r", encoding="utf-8") as f:
      try:
        template_config = json.load(f)
        if isinstance(template_config, dict) and "chat_template" in template_config:
          return template_config["chat_template"]
      except json.JSONDecodeError:
        return None

  return None


def get_template_placeholders(template):
  """Dynamically extracts the format keys (placeholders) from a template string."""
  # Finds all names inside {...}
  return set(re.findall(r"(?<!{){([a-zA-Z0-9_]+)}(?!})", template))


def extract_reasoning_and_answer(text, separator):
  if separator not in text:
    return None, None
  [reasoning, answer] = text.split(separator)
  return reasoning, answer


def math_qa_formatting(example, template_config=None):
  """Maps question-answer pairs to conversational format."""

  # Initialize prompt and completion with fallback templates
  prompt = {"role": "user", "content": example["question"]}
  completion = {"role": "assistant", "content": example["answer"]}

  # Apply templates to prompt and completion, if provided
  if template_config:
    # Apply PROMPT_TEMPLATE to prompt, if provided
    if "PROMPT_TEMPLATE" in template_config:
      placeholders = get_template_placeholders(template_config["PROMPT_TEMPLATE"])
      if "question" not in placeholders:
        max_logging.log("PROMPT_TEMPLATE has no 'question' placeholder. No template will be applied to prompt.")
      else:
        prompt = {
            "role": "user",
            "content": template_config["PROMPT_TEMPLATE"].format(question=example["question"].strip()),
        }
    else:
      max_logging.log("PROMPT_TEMPLATE is empty. No template will be applied to prompt.")

    # Apply COMPLETION_TEMPLATE to completion, if provided
    if "COMPLETION_TEMPLATE" in template_config:
      placeholders = get_template_placeholders(template_config["COMPLETION_TEMPLATE"])
      if "REASONING_ANSWER_SEPARATOR" in template_config:
        reasoning, answer = extract_reasoning_and_answer(example["answer"], template_config["REASONING_ANSWER_SEPARATOR"])
        if "reasoning" not in placeholders or "answer" not in placeholders:
          max_logging.log(
              "COMPLETION_TEMPLATE is missing 'reasoning' or 'answer' placeholder."
              " No template will be applied to completion."
              " Remove REASONING_ANSWER_SEPARATOR from template or update COMPLETION_TEMPLATE."
          )
        elif reasoning is None or answer is None:
          max_logging.log(
              "REASONING_ANSWER_SEPARATOR is present in template but not found in answer."
              " No template will be applied to completion."
              " Update REASONING_ANSWER_SEPARATOR in the template."
          )
        else:
          completion = {
              "role": "assistant",
              "content": template_config["COMPLETION_TEMPLATE"].format(
                  reasoning=reasoning.strip(), answer=answer.strip()
              ),
          }
      else:
        max_logging.log(
            "REASONING_ANSWER_SEPARATOR not found in chat template."
            " Using only 'answer' placeholder for COMPLETION_TEMPLATE."
        )
        if "answer" not in placeholders:
          max_logging.log(
              "COMPLETION_TEMPLATE is missing 'answer' placeholder. No template will be applied to completion."
          )
        else:
          completion = {
              "role": "assistant",
              "content": template_config["COMPLETION_TEMPLATE"].format(answer=example["answer"].strip()),
          }
    else:
      max_logging.log("COMPLETION_TEMPLATE is empty. No template will be applied to completion.")

  example["messages"] = [prompt, completion]
  return example


def load_formatter(formatting_func_path, **kwargs):
  """Loads a formatter function from a given path.

  Returns a callable that takes a dataset and applies the formatter via .map().
  """
  module_path, method_name = formatting_func_path.rsplit(".", 1)
  module = importlib.import_module(module_path)
  func = getattr(module, method_name)

  def formatter(dataset, dataset_features):
    remove_cols = []
    if kwargs:
      remove_cols = kwargs.pop("remove_columns", None)
    return dataset.map(
        func,
        fn_kwargs=kwargs if kwargs else None,
        features=dataset_features,
        remove_columns=remove_cols,
    )

  return formatter


def convert_to_conversational_format(
    dataset,
    data_columns,
    formatting_func_path=None,
    formatting_func_kwargs=None,
):
  """Converts instruction dataset to conversational format."""
  import datasets  # pylint: disable=import-outside-toplevel

  dataset_features = datasets.Features(
      {"messages": [{"content": datasets.Value("string"), "role": datasets.Value("string")}]}
  )

  if formatting_func_path:
    if not formatting_func_kwargs:
      formatting_func_kwargs = {}
    formatting_func_kwargs["remove_columns"] = data_columns
    template_path = formatting_func_kwargs.pop("template_path", None)
    if template_path:
      formatting_func_kwargs["template_config"] = load_data_template_from_file(template_path)
    formatter = load_formatter(formatting_func_path, **(formatting_func_kwargs))
    dataset = formatter(dataset, dataset_features)
    data_columns = ["messages"]
    return dataset, data_columns

  if "question" in data_columns and "answer" in data_columns:
    dataset = dataset.map(
        math_qa_formatting,
        fn_kwargs={},
        remove_columns=data_columns,
        features=dataset_features,
    )
    data_columns = ["messages"]
    return dataset, data_columns

  return dataset, data_columns
