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

REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

QA_PROMPT_TEMPLATE = f"""
You are given a problem. Think about the problem and \
provide your reasoning. Place it between {REASONING_START} and \
{REASONING_END}. Then, provide the final answer (i.e., just one numerical \
value) between {ANSWER_START} and {ANSWER_END}.

{{question}}"""

QA_COMPLETION_TEMPLATE = f"""
{REASONING_START}
{{reasoning}}
{REASONING_END}

{ANSWER_START}{{answer}}{ANSWER_END}
"""


def extract_reasoning_and_answer(text):
  if "####" not in text:
    return None, None
  [reasoning, answer] = text.split("####")
  return reasoning, answer


def map_qa_data_to_conversation(example):
  """Maps question-answer pairs to conversational format."""
  reasoning, answer = extract_reasoning_and_answer(example["answer"])
  prompt = {"role": "user", "content": QA_PROMPT_TEMPLATE.format(question=example["question"])}
  completion = {"role": "assistant", "content": QA_COMPLETION_TEMPLATE.format(reasoning=reasoning, answer=answer)}
  example["messages"] = [prompt, completion]
  return example


def convert_to_conversational_format(
    dataset,
    data_columns,
):
  """Converts instruction dataset to conversational format."""
  dataset_features = datasets.Features(
      {"messages": [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
  )
  data_column_names = ["messages"]
  if "question" in data_columns and "answer" in data_columns:
    dataset = dataset.map(
        map_qa_data_to_conversation,
        remove_columns=data_columns,
        features=dataset_features,
    )
  return dataset, data_column_names
