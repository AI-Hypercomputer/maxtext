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

"""This is a simple script for MMLU benchmark for a trained checkpoint.
Dataset: https://huggingface.co/datasets/lighteval/mmlu

To get optimal performance the prompt template needs to be adjusted (e.g. CoT or 5-shot prompt) per model.


To run the MMLU benchmark:
# Default is zero-shot prompting
python3 -m benchmarks.mmlu.mmlu_eval src/MaxText/configs/base.yml \
  tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  load_parameters_path=check_point_path model_name=llama3.1-8b \
  max_prefill_predict_length=1024 max_target_length=2048 ici_tensor_parallelism=4 per_device_batch_size=1

# Example of using the prompt_template flag for Chain-of-Thought (CoT) prompting:
python3 -m benchmarks.mmlu.mmlu_eval src/MaxText/configs/base.yml \
  tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  load_parameters_path=check_point_path model_name=llama3.1-8b \
  max_prefill_predict_length=1024 max_target_length=2048 ici_tensor_parallelism=4 per_device_batch_size=1 \
  prompt_template="The following are multiple choice questions (with answers) about {subject}.\n\n{question}\n
  {choices}\nAnswer: Let's think step by step."

# Example of using the prompt_template flag for 5-shot prompting (replace with actual examples):
python3 -m benchmarks.mmlu.mmlu_eval src/MaxText/configs/base.yml \
  tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken \
  load_parameters_path=check_point_path model_name=llama3.1-8b \
  max_prefill_predict_length=1024 max_target_length=2048 ici_tensor_parallelism=4 per_device_batch_size=1 \
  prompt_template='Example 1:\nQuestion: What is the capital of France?\nChoices:\nA. London\nB. Paris\nC. Rome\nD. Berlin\nAnswer: B\n\nExample 2:\nQuestion: What is the highest mountain in the world?\nChoices:\nA. K2\nB. Kangchenjunga\nC. Mount Everest\nD. Lhotse\nAnswer: C\n\nExample 3:\nQuestion: What is the chemical symbol for water?\nChoices:\nA. H2O\nB. CO2\nC. O2\nD. NaCl\nAnswer: A\n\nExample 4:\nQuestion: Who painted the Mona Lisa?\nChoices:\nA. Michelangelo\nB. Leonardo da Vinci\nC. Raphael\nD. Donatello\nAnswer: B\n\nExample 5:\nQuestion: Which planet is known as the Red Planet?\nChoices:\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn\nAnswer: B\n\nThe following are multiple choice questions (with answers) about {subject}.\n\n{question}\n{choices}\nAnswer:'   # pylint: disable=line-too-long
"""

import collections
import re
import sys

from absl import flags

import datasets

import jax

from benchmarks.mmlu.mmlu_categories import categories
from benchmarks.mmlu.mmlu_categories import subcategories

from tqdm import tqdm

from MaxText import pyconfig
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxengine

ASCII_UPPERCASE_A = ord("A")  # ASCII value for uppercase 'A'

DEFAULT_PROMPT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

{question}
{choices}
Answer:"""


_PROMPT_TEMPLATE = flags.DEFINE_string(
    "prompt_template",
    default=DEFAULT_PROMPT_TEMPLATE,
    help="prompt template",
)


def construct_prompt(subject, question, choices):
  subject = subject.replace("_", " ")
  choices_text = "\n".join(f"{chr(ASCII_UPPERCASE_A + idx)}. {choice}" for idx, choice in enumerate(choices))
  prompt = _PROMPT_TEMPLATE.value.format(subject=subject, question=question, choices=choices_text)
  return prompt


def parse_answer(output):
  match = re.search(r"Answer:\s*([A-D])|(?:The answer is)\s*([A-D])", output, re.IGNORECASE)
  predicted_answer = match.group(1) or match.group(2) if match else None
  return predicted_answer


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
  subject_correct = collections.defaultdict(int)
  subject_total = collections.defaultdict(int)
  subcat_correct = collections.defaultdict(int)
  subcat_total = collections.defaultdict(int)

  mmlu_test_ds = datasets.load_dataset("lighteval/mmlu", "all", split="test")
  for idx, example in enumerate(tqdm(mmlu_test_ds, desc="Evaluating MMLU dataset")):
    subject = example["subject"]
    question = example["question"]
    choices = example["choices"]
    label = example["answer"]
    prompt = construct_prompt(subject, question, choices)

    # Tokenize the input
    tokens, true_length = tokenizer.encode(prompt, is_bos=True, prefill_lengths=[max_prefill_predict_length])
    if true_length > max_prefill_predict_length:
      max_logging.log(
          f"Warning: Prompt length {true_length} exceeds max prefill length" f" {max_prefill_predict_length}. Truncating."
      )
      tokens = tokens[:max_prefill_predict_length]
      true_length = max_prefill_predict_length
    assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"

    # Perform prefill
    prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
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
      predicted_answer = parse_answer(prompt + output)
      if predicted_answer:
        break

      # Generate next token
      decode_state, sampled_token = engine.generate(params, decode_state)
      sampled_tokens.append(sampled_token.get_result_at_slot(slot).tokens.item())
      if sampled_tokens[-1] == tokenizer.eos_id:
        break

    if not predicted_answer:
      max_logging.log("Could not extract an answer from the model's output for example" f" {total_count + 1}")
    elif predicted_answer not in {chr(ASCII_UPPERCASE_A + idx) for idx in range(len(choices))}:
      max_logging.log(f"Invalid or missing predicted answer for subject '{subject}' in example {total_count + 1}")

    # Convert the label index to the corresponding letter
    correct_answer = chr(65 + label)

    # Log answer
    max_logging.log(
        f"{total_count + 1} | {prompt}\n[Model output] {output}\n"
        f"[Correct answer] {correct_answer}, Matching: {predicted_answer == correct_answer}"
    )

    # Update accuracy for overall and per-subject
    if predicted_answer == correct_answer:
      correct_count += 1
      subject_correct[subject] += 1
    total_count += 1
    subject_total[subject] += 1

    if idx % 50 == 0:
      max_logging.log(f" Accuracy: {correct_count / total_count:.4f} | Processed: {total_count}/{len(mmlu_test_ds)}")

  # Final accuracy
  if total_count > 0:
    accuracy = correct_count / total_count
    max_logging.log(f"\nFinal accuracy on MMLU dataset: {accuracy:.4f}")
  else:
    max_logging.log("No valid predictions were made.")

  # Calculate subject accuracies
  subject_acc = {subject: subject_correct[subject] / subject_total[subject] for subject in subject_total}

  # Map subject accuracies to subcategories
  for subject in subject_acc:
    if subject in subcategories:
      subcat_labels = subcategories[subject]
      for subcat_label in subcat_labels:
        subcat_correct[subcat_label] += subject_correct[subject]
        subcat_total[subcat_label] += subject_total[subject]
    else:
      max_logging.log(f"Warning: Subject '{subject}' not found in subcategories.")

  # Calculate subcategory accuracies
  max_logging.log("\nSubcategory Accuracies:")
  for subcat_label in subcat_total:
    acc = subcat_correct[subcat_label] / subcat_total[subcat_label]
    max_logging.log(f"Accuracy for subcategory '{subcat_label}': {acc:.4f}")

  # Calculate and print category accuracies
  cat_correct = collections.defaultdict(int)
  cat_total = collections.defaultdict(int)

  for category_name, subcat_labels in categories.items():
    for subcat_label in subcat_labels:
      cat_correct[category_name] += subcat_correct[subcat_label]
      cat_total[category_name] += subcat_total[subcat_label]

  max_logging.log("\nCategory Accuracies:")
  for category_name in cat_total:
    if cat_total[category_name] > 0:
      acc = cat_correct[category_name] / cat_total[category_name]
      max_logging.log(f"Accuracy for category '{category_name}': {acc:.4f}")
    else:
      max_logging.log(f"Accuracy for category '{category_name}': No data available.")


def validate_config(config):
  assert not config.load_full_state_path, (
      "Decode doesn't operate on full states! Convert to parameter checkpoint"
      " first. Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  flags.FLAGS(sys.argv)
  cfg = pyconfig.initialize(sys.argv)
  validate_config(cfg)
  max_utils.print_system_information()
  main(cfg)
