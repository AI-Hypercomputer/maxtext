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

import datasets
import json
import os
import re

from MaxText import max_logging


def load_template_from_file(template_path):
  """Loads a template from a file."""
  template_config = None
  if os.path.isfile(template_path) and template_path.endswith(".json"):
    with open(template_path, encoding="utf-8") as f:
      template_config = json.load(f)
  return template_config


def get_template_placeholders(template):
  """Dynamically extracts the format keys (placeholders) from a template string."""
  # Finds all names inside {...}
  return set(re.findall(r"(?<!{){([a-zA-Z0-9_]+)}(?!})", template))


def extract_reasoning_and_answer(text, separator):
  if separator not in text:
    return None, None
  [reasoning, answer] = text.split(separator)
  return reasoning, answer


def map_qa_data_to_conversation(example, template_config):
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


def convert_to_conversational_format(
    dataset,
    data_columns,
    chat_template_path,
):
  """Converts instruction dataset to conversational format."""
  template_config = None
  if chat_template_path:
    template_config = load_template_from_file(chat_template_path)
  if "question" in data_columns and "answer" in data_columns:
    dataset_features = datasets.Features(
        {"messages": [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
    )
    dataset = dataset.map(
        map_qa_data_to_conversation,
        fn_kwargs={"template_config": template_config},
        remove_columns=data_columns,
        features=dataset_features,
    )
    data_columns = ["messages"]
  return dataset, data_columns
