# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a simple script for MMLU benchmark for a trained checkpoint.
Dataset: https://huggingface.co/datasets/lighteval/mmlu

To get optimal performance the prompt template needs to be adjusted (e.g. CoT or 5-shot prompt) per model.


To run the MMLU benchmark:
# Default is zero-shot prompting
python3 -m benchmarks.mmlu.mmlu_eval MaxText/configs/base.yml \
  tokenizer_type=tiktoken tokenizer_path=assets/tokenizer_llama3.tiktoken \
  load_parameters_path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/unscanned/checkpoints/0/items model_name=llama3.1-8b \
  max_prefill_predict_length=256 max_target_length=512 per_device_batch_size=1 ici_tensor_parallelism=8

# Example of using the prompt_template flag for Chain-of-Thought (CoT) prompting:
python3 -m benchmarks.mmlu.mmlu_eval MaxText/configs/base.yml \
  tokenizer_type=tiktoken tokenizer_path=assets/tokenizer_llama3.tiktoken \
  load_parameters_path=check_point_path model_name=llama3.1-8b \
  max_prefill_predict_length=1024 max_target_length=2048 ici_tensor_parallelism=4 per_device_batch_size=1 \
  prompt_template="The following are multiple choice questions (with answers) about {subject}.\n\n{question}\n
  {choices}\nAnswer: Let's think step by step."

# Example of using the prompt_template flag for 5-shot prompting (replace with actual examples):
python3 -m benchmarks.mmlu.mmlu_eval MaxText/configs/base.yml \
  tokenizer_type=tiktoken tokenizer_path=assets/tokenizer_llama3.tiktoken \
  load_parameters_path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/unscanned/checkpoints/0/items model_name=llama3.1-8b \
  max_prefill_predict_length=1024 max_target_length=2048 ici_tensor_parallelism=4 per_device_batch_size=1 \
  prompt_template='Example 1:\nQuestion: What is the capital of France?\nChoices:\nA. London\nB. Paris\nC. Rome\nD. Berlin\nAnswer: B\n\nExample 2:\nQuestion: What is the highest mountain in the world?\nChoices:\nA. K2\nB. Kangchenjunga\nC. Mount Everest\nD. Lhotse\nAnswer: C\n\nExample 3:\nQuestion: What is the chemical symbol for water?\nChoices:\nA. H2O\nB. CO2\nC. O2\nD. NaCl\nAnswer: A\n\nExample 4:\nQuestion: Who painted the Mona Lisa?\nChoices:\nA. Michelangelo\nB. Leonardo da Vinci\nC. Raphael\nD. Donatello\nAnswer: B\n\nExample 5:\nQuestion: Which planet is known as the Red Planet?\nChoices:\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer: B\n\nThe following are multiple choice questions (with answers) about {subject}.\n\n{question}\n{choices}\nAnswer:'   # pylint: disable=line-too-long
"""

import collections
import re
import sys

from absl import flags

import datasets
import numpy as np
from typing import List, Optional

import jax

from dataclasses import dataclass
from benchmarks.mmlu.mmlu_categories import categories
from benchmarks.mmlu.mmlu_categories import subcategories

from tqdm import tqdm

from MaxText import pyconfig
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxengine
from MaxText import multimodal_utils

ASCII_UPPERCASE_A = ord("A")  # ASCII value for uppercase 'A'
SUPPORTED_DATASETS = ["HuggingFaceM4/ChartQA"]

# DEFAULT_PROMPT_TEMPLATE = """{image_placeholder}{question}
# {choices}
# Give me answer directly in the format <answer>your_answer</answer>."""

DEFAULT_PROMPT_TEMPLATE = """You are an expert at answering questions based 
on provided charts. Your task is to extract the exact answer from the 
given context or determine that it's not present.
{image_placeholder} Question: {question}
For numerical answers, provide only the number. 
For text answers, provide only the exact text. 
For judgement questions, respond with "Yes" or "No".
If not found, output "N/A". 
Your output must be only the exact answer within <answer></answer>, with no extra contents.
"""


@dataclass
class ParsedDatasetExample:
  """Parsed example from the HuggingFace dataset."""
  question: Optional[str] = None
  image_np: Optional[np.ndarray] = None
  choices: Optional[List[str]] = None
  answer: Optional[str] = None


def parse_dataset_example(example, hf_dataset_name):
  """Parse a single example from the HuggingFace dataset."""
  parsed_example = ParsedDatasetExample()
  if hf_dataset_name == "HuggingFaceM4/ChartQA":
    parsed_example.question = example["query"]
    parsed_example.image_np = np.asarray(example["image"].convert("RGB")) # Convert PIL object to np array
    parsed_example.answer = example["label"][0]
  else:
    raise ValueError(f"Unsupported dataset: {hf_dataset_name}")
  return parsed_example


def construct_prompt(parsed_dataset_example: ParsedDatasetExample, config, system_message: Optional[str] = None):
  """Construct prompt from a parsed dataset example."""
  image_placeholder = multimodal_utils.get_image_placeholder(config.model_name) if config.use_multimodal else ""
  choices_text = "\n".join(f"{chr(ASCII_UPPERCASE_A + idx)}. {choice}" for idx, choice in enumerate(parsed_dataset_example.choices)) if parsed_dataset_example.choices else ""
  prompt = DEFAULT_PROMPT_TEMPLATE.format(
    image_placeholder=image_placeholder, 
    question=parsed_dataset_example.question, 
    choices=choices_text if choices_text else "N/A"
  )
  # Add extra model-specific formatting such as user/model/assistant tags
  prompt = multimodal_utils.reformat_prompt(prompt, "<|image|>", config.model_name)
  prompt = system_message + "\n\n" + prompt if system_message else prompt
  return prompt


def parse_answer(output_string):
  match = re.search(r"<answer>(.*?)</answer>", output_string, re.DOTALL) # re.s makes . match newlines as well
  return match.group(1) if match else None


def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)

  max_prefill_predict_length = getattr(config, "max_prefill_predict_length", 1024)
  max_target_length = getattr(config, "max_target_length", 2048)

  # Initialize counters for overall and per-subject accuracies
  correct_count = 0
  total_count = 0
  hf_data_dir = config.hf_data_dir

  test_ds = datasets.load_dataset(hf_data_dir, "default", split="test")
  for idx, example in enumerate(tqdm(test_ds, desc=f"Evaluating {hf_data_dir} dataset")):
    prefill_length = config.max_prefill_predict_length
    parsed_dataset_example = parse_dataset_example(example, hf_data_dir)
    prompt = construct_prompt(parsed_dataset_example, config)
    processor_output = multimodal_utils.pre_process_image(parsed_dataset_example.image_np, model_name=config.model_name)
    prefill_length -= multimodal_utils.get_image_offsets(config.model_name, processor_output=processor_output)
    print("\n" + "*"*50)

    # Tokenize the input
    tokens, true_length = tokenizer.encode(prompt, is_bos=True, prefill_lengths=[prefill_length])
    if config.use_multimodal:
      tokens = multimodal_utils.prepare_text_for_image_fusion(
          tokens, model_name=config.model_name, processor_output=processor_output
      )
      true_length += multimodal_utils.get_image_offsets(config.model_name, processor_output=processor_output)
    if true_length > max_prefill_predict_length:
      max_logging.log(
          f"Warning: Prompt length {true_length} exceeds max prefill length" f" {max_prefill_predict_length}. Truncating."
      )
      tokens = tokens[:max_prefill_predict_length]
      true_length = max_prefill_predict_length
    assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"

    # Perform prefill
    prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, images=processor_output.pixel_values, true_length=true_length)
    slot = 0

    # Initialize decode state
    decode_state = engine.init_decode_state()
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    steps = range(max_prefill_predict_length, max_target_length)
    sampled_tokens = [first_token.get_result_at_slot(slot).tokens.item()]

    predicted_answer = ""

    for _ in steps:
      # Decode generated tokens so far
      output = tokenizer.decode(sampled_tokens)
      predicted_answer = parse_answer(output)
      if predicted_answer:
        break

      # Generate next token
      decode_state, sampled_token = engine.generate(params, decode_state)
      sampled_tokens.append(sampled_token.get_result_at_slot(slot).tokens.item())
      if sampled_tokens[-1] == tokenizer.eos_id:
        break

    # if not predicted_answer:
    #   max_logging.log("Could not extract an answer from the model's output for example" f" {total_count + 1}")
    # elif predicted_answer not in {chr(ASCII_UPPERCASE_A + idx) for idx in range(len(choices))}:
    #   max_logging.log(f"Invalid or missing predicted answer for subject '{subject}' in example {total_count + 1}")

    correct_answer = parsed_dataset_example.answer
    # For open-ended answers, we do substring matching
    # TODO(hengtaoguo): More robust correctness checking (e.g. numerical tolerance for numbers, Gemini APIs)
    is_correct = correct_answer in predicted_answer if predicted_answer is not None else False

    # Log answer
    max_logging.log(
        f"{total_count + 1} | {parsed_dataset_example.question}\n"
        f"[Model output] {output}\n"
        f"[Label answer] {correct_answer}\n"
        f"Matching: {is_correct}"
    )

    # Update accuracy for overall and per-subject
    if is_correct:
      correct_count += 1
      # subject_correct[subject] += 1
    total_count += 1
    # subject_total[subject] += 1
    max_logging.log(f"Running accuracy: {correct_count / (total_count):.4f} | Processed: {total_count}/{len(test_ds)}")

    if idx >= 19: # For debugging, limit to first 20 examples
      break

    if idx % 50 == 0:
      max_logging.log(f" Accuracy: {correct_count / total_count:.4f} | Processed: {total_count}/{len(test_ds)}")

  # Final accuracy
  if total_count > 0:
    accuracy = correct_count / total_count
    max_logging.log(f"\nFinal accuracy on {hf_data_dir} dataset: {accuracy:.4f}")
  else:
    max_logging.log("No valid predictions were made.")


def validate_config(config):
  assert not config.load_full_state_path, (
      "Decode doesn't operate on full states! Convert to parameter checkpoint"
      " first. Using generate_param_only_checkpoint."
  )
  assert config.hf_data_dir, (
    "For benchmark evaluation, please specify the HuggingFace dataset name using the hf_data_dir config field."
  )
  assert config.hf_data_dir in SUPPORTED_DATASETS, (
    f"Unsupported dataset {config.hf_data_dir}. Supported datasets are: {SUPPORTED_DATASETS}."
    " Please add support for your desired dataset in the code of multimodal_eval.py."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  flags.FLAGS(sys.argv)
  cfg = pyconfig.initialize(sys.argv)
  validate_config(cfg)
  max_utils.print_system_information()
  main(cfg)
