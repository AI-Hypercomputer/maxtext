#  Copyright 2023-2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Utility functions for SFT Qwen3 demo notebook."""

from tqdm.auto import tqdm

import datasets
import grain
import os
import re

from maxtext.input_pipeline import instruction_data_processing

# Suppress vLLM logging with a severity level below ERROR
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from tunix.rl.rollout import base_rollout


DATASET_NAME = "openai/gsm8k"
DATASET_DATA_DIR = "main"
DATASET_TRAIN_SPLIT = "train"
DATASET_TEST_SPLIT = "test"
DATASET_DATA_COLUMN = ["question", "answer"]
TRAIN_STEPS = 10000
SEED = 42
BATCH_SIZE = 4
NUM_TEST_SAMPLES = 1320
MAX_TOKENS_TO_GENERATE = 768
MAX_PROMPT_LENGTH = 256
EVALUATION_CONFIG = {"temperature": 1e-4, "top_k": 1, "top_p": 1.0}
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
# Regex to check the full format (reasoning + answer markers)
MATCH_FORMAT = re.compile(
    rf"^.*?" rf"{REASONING_START}.+?{REASONING_END}.*?" rf"{ANSWER_START}(.+?){ANSWER_END}" rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
# Regex to extract the final numerical answer
MATCH_ANSWER = re.compile(rf"{ANSWER_START}.*?([\d\.\,\$]{{1,}})", flags=re.MULTILINE | re.DOTALL)


def get_test_dataset(config, tokenizer, data_template_path):
  """Loads and prepares the test dataset from Hugging Face.

  Args:
    config: The pyconfig object containing run configurations, including
      `hf_access_token`.
    tokenizer: The tokenizer for processing the text data.
    data_template_path: The path to the template config for formatting the data.

  Returns:
    A grain.MapDataset instance for the test split, with prompts and target
    answers.
  """

  template_config = instruction_data_processing.load_data_template_from_file(data_template_path)
  dataset = datasets.load_dataset(
      DATASET_NAME,
      data_dir=DATASET_DATA_DIR,
      split=DATASET_TEST_SPLIT,
      token=config.hf_access_token,
  )

  return (
      grain.MapDataset.source(dataset)
      .shuffle(seed=SEED)
      .map(
          lambda x: {
              "question": x["question"],
              "prompt": tokenizer.apply_chat_template(
                  [
                      {
                          "role": "user",
                          "content": (template_config["PROMPT_TEMPLATE"].format(question=x["question"].strip())),
                      }
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              "target_answer": (
                  instruction_data_processing.extract_reasoning_and_answer(
                      x["answer"], template_config["REASONING_ANSWER_SEPARATOR"]
                  )[1]
              ),
          }
      )
  )


def evaluate_model(dataset, vllm_rollout, debug=True):
  """Runs evaluation on the model using vLLM.

  Args:
    dataset: The dataset to evaluate on.
    vllm_rollout: The vLLM rollout object for generating responses.
    debug: If True, prints debug information for each sample.

  Returns:
    A dictionary containing evaluation scores: 'correct', 'partially_correct',
    and 'correct_format' percentages.
  """
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
      max_prompt_length=MAX_PROMPT_LENGTH,
      temperature=EVALUATION_CONFIG["temperature"],
      top_p=EVALUATION_CONFIG["top_p"],
      top_k=EVALUATION_CONFIG["top_k"],
      data_type="bfloat16",
  )

  total, total_correct, total_partially_correct, total_correct_format = 0, 0, 0, 0
  for batch in tqdm(dataset):
    batch_response = vllm_rollout.generate(batch["prompt"], rollout_config)
    for i, question in enumerate(batch["question"]):
      if debug:
        print("========================================")
        print(f"Question: {question}")
        print("----------------------------------------")
        print(f"Model Generated Response: {batch_response.text[i]}")
        print("----------------------------------------")
        print(f"Target Response: {batch["target_answer"][i]}")
        print("========================================")

      is_correct, is_partially_correct, has_correct_format = score_response(
          target=batch["target_answer"][i], prediction=batch_response.text[i], debug=debug
      )
      if is_correct:
        total_correct += 1
      if is_partially_correct:
        total_partially_correct += 1
      if has_correct_format:
        total_correct_format += 1
      total += 1

  return {
      "correct": (total_correct / total) * 100,
      "partially_correct": (total_partially_correct / total) * 100,
      "correct_format": (total_correct_format / total) * 100,
  }


def safe_string_to_float(text):
  """Cleans a string to make it safely convertible to a float.

  Removes commas, spaces, and dollar signs.

  Args:
    text: The input string.

  Returns:
    The cleaned string.
  """
  text = text.replace(",", "").replace(" ", "")  # converts "2,125" to "2125"
  text = text.replace("$", "")  # converts "$50" to "50"
  return text


def score_response(target, prediction, debug=True):
  """Scores the model's prediction against the target answer.

  It checks for exact correctness, partial correctness (within 10%), and
  whether the response follows the expected format.

  Args:
    target: The ground truth answer string.
    prediction: The model's generated response string.
    debug: If True, prints exceptions during scoring.

  Returns:
    A tuple of booleans: (is_correct, is_partially_correct, has_correct_format).
  """
  is_correct, is_partially_correct, has_correct_format = False, False, False
  extracted_response = guess.group(1) if (guess := MATCH_ANSWER.search(prediction)) is not None else ""
  extracted_response = safe_string_to_float(extracted_response)
  target = safe_string_to_float(target)
  try:
    # Check exact correctness
    if float(extracted_response.strip()) == float(target.strip()):
      is_correct = True

    # Check partial correctness (within 10%)
    ratio = float(extracted_response.strip()) / float(target.strip())
    if 0.9 <= ratio <= 1.1:
      is_partially_correct = True

    if MATCH_FORMAT.search(prediction) is not None:
      has_correct_format = True
  except (ValueError, TypeError, ZeroDivisionError) as e:
    if debug:
      print("Evaluation exception: ", e)

  return is_correct, is_partially_correct, has_correct_format
