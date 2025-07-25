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

"""This is a simple script for multimodal benchmark for a trained checkpoint.
HuggingFaceM4/ChartQA: https://huggingface.co/datasets/HuggingFaceM4/ChartQA

Usage:
# Gemma3-4b on a single TPU v4-8 VM
python3 -m benchmarks.multimodal.multimodal_eval MaxText/configs/base.yml \
  model_name=gemma3-4b tokenizer_path=assets/tokenizer.gemma3 \
  load_parameters_path=gs://maxtext-model-checkpoints/gemma3-4b/multimodal/2025-05-21-23-23-59/checkpoints/0/items \
  base_output_directory=$YOUR_GCS_PATH \
  per_device_batch_size=1 run_name=mmeval_test steps=1 async_checkpointing=false \
  scan_layers=false use_multimodal=true attention=\'dot_product\' \
  max_prefill_predict_length=550 max_target_length=570 per_device_batch_size=1 \
  hf_data_dir=HuggingFaceM4/ChartQA hf_eval_split=test

# Llama4-17b-16e on a TPU v5p-128 cluster (images resized to 336x336 for simplicity)
python -m benchmarks.multimodal.multimodal_eval \
  MaxText/configs/base.yml model_name=llama4-17b-16e image_resize=336 \
  tokenizer_path=meta-llama/Llama-4-Scout-17B-16E \
  load_parameters_path=gs://maxtext-model-checkpoints/llama4-17b-16e/hybrid/2025-07-22-11-03-20/0/items \
  base_output_directory=$YOUR_GCS_PATH \
  per_device_batch_size=1 run_name=mmeval_test steps=1 async_checkpointing=false \
  scan_layers=true use_multimodal=true attention=\'dot_product\' \
  max_prefill_predict_length=350 max_target_length=370 per_device_batch_size=1 \
  hf_data_dir=HuggingFaceM4/ChartQA hf_eval_split=test hf_access_token=\'$YOUR_HF_ACCESS_TOKEN\' \
  ici_fsdp_parallelism=1 ici_expert_parallelism=16 ici_tensor_parallelism=4
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from absl import flags
import datasets
import jax
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from MaxText import max_logging
from MaxText import maxengine
from MaxText import max_utils
from MaxText import multimodal_utils
from MaxText import pyconfig
from MaxText.utils import gcs_utils


ASCII_UPPERCASE_A = ord("A")  # ASCII value for uppercase 'A'
SUPPORTED_DATASETS = ["HuggingFaceM4/ChartQA"]

DEFAULT_PROMPT_TEMPLATE = """You are an expert at answering questions based 
on provided charts. Your task is to extract the exact answer from the 
given context or determine that it's not present.
{image_placeholder} Question: {question}
For numerical answers, provide only the number. 
For text answers, provide only the exact text. 
For judgement questions, respond with "Yes" or "No".
If not found, output "N/A". 
Your output must be only the exact answer within <answer></answer>, with no extra contents.

Example:
Question: What is the capital of France?
Your answer: <answer>Paris</answer>
"""

SFT_PROMPT_TEMPLATE = "{image_placeholder}{question}"


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

  # Resize the image if specified. This helps simplify the llama4's tiling, so we have a fixed input size
  if cfg.image_resize != -1:
    pil_img = Image.fromarray(parsed_example.image_np)
    pil_img = pil_img.resize((cfg.image_resize, cfg.image_resize))
    parsed_example.image_np = np.asarray(pil_img.convert("RGB"))

  return parsed_example


def construct_prompt(parsed_dataset_example: ParsedDatasetExample, config, system_message: Optional[str] = None):
  """Construct prompt from a parsed dataset example."""
  image_placeholder = multimodal_utils.get_image_placeholder(config.model_name) if config.use_multimodal else ""
  choices_text = "\n".join(f"{chr(ASCII_UPPERCASE_A + idx)}. {choice}" for idx, choice in enumerate(parsed_dataset_example.choices)) if parsed_dataset_example.choices else ""
  # # Prompt for raw pretrained checkpoints
  # prompt = DEFAULT_PROMPT_TEMPLATE.format(
  #   image_placeholder=image_placeholder, 
  #   question=parsed_dataset_example.question, 
  #   choices=choices_text if choices_text else "N/A"
  # )
  # Prompt for SFT checkpoints, same as the original SFT prompt
  prompt = SFT_PROMPT_TEMPLATE.format(
    image_placeholder=image_placeholder, 
    question=parsed_dataset_example.question
  )
  # Add extra model-specific formatting such as user/model/assistant tags
  prompt = multimodal_utils.reformat_prompt(prompt, image_placeholder, config.model_name)
  prompt = system_message + "\n\n" + prompt if system_message else prompt
  return prompt


def parse_answer(output_string):
  # Try to match the <answer>?</answer> template (e.g., <answer>Paris</answer> from any pretrained models)
  match_xml = re.search(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
  if match_xml:
    return match_xml.group(1).strip()

  # If not found, try to match the ['?'] template (e.g., ['Paris'] from HuggingFaceM4/ChartQA SFT)
  match_list = re.search(r"\['(.*?)'\]", output_string)
  if match_list:
    return match_list.group(1).strip()

  # If neither template is found, return None
  return None


def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)

  max_prefill_predict_length = getattr(config, "max_prefill_predict_length", 1024)
  max_target_length = getattr(config, "max_target_length", 2048)

  # Initialize counters for overall accuracy
  correct_count = 0
  total_count = 0

  # Get the HuggingFace dataset path and name from the config
  hf_data_dir = config.hf_data_dir
  hf_data_name = hf_data_dir.split("/")[-1] if "/" in hf_data_dir else hf_data_dir
  hf_eval_split = config.hf_eval_split

  # Config for saving csv results
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  results_file_name = f"{hf_data_name}_results.csv"  # Choose an appropriate name
  result_gcs_path = f"{cfg.base_output_directory}/{timestamp}.csv" if cfg.base_output_directory else None
  max_logging.log(f"Results will be saved to {results_file_name} and uploaded to GCS: {result_gcs_path}")
  results_data = []

  test_ds = datasets.load_dataset(hf_data_dir, "default", split=hf_eval_split)
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
      image_offsets = multimodal_utils.get_image_offsets(config.model_name, processor_output=processor_output)
      true_length += image_offsets
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

    correct_answer = parsed_dataset_example.answer
    # For open-ended answers, we do substring matching
    # TODO(hengtaoguo): More robust correctness checking (e.g. numerical tolerance for numbers, Gemini APIs)
    # is_correct = correct_answer in predicted_answer if predicted_answer is not None else False
    is_correct = correct_answer in output

    # Log answer
    max_logging.log(
        f"{total_count + 1} | {parsed_dataset_example.question}\n"
        f"[Model output] {output}\n"
        f"[Label answer] {correct_answer}\n"
        f"Matching: {is_correct}"
    )

    # Save results for CSV
    results_data.append({
        "question ID": total_count + 1,
        "question": parsed_dataset_example.question,
        "label": parsed_dataset_example.answer,
        "output": output,
        "is_correct": is_correct
    })

    # Update accuracy for overall
    if is_correct:
      correct_count += 1
    total_count += 1
    max_logging.log(f"Running accuracy: {correct_count / (total_count):.4f} | Processed: {total_count}/{len(test_ds)}")

    # if idx >= 4: # For debugging, limit to first 5 examples
    #   break

    # Every 100 rows, save intermediate results to CSV and upload to GCS
    if idx % 100 == 0 and result_gcs_path is not None and jax.process_index() == 0:
      results_df = pd.DataFrame(results_data)
      results_df.to_csv(results_file_name, index=False)
      gcs_utils.upload_blob(result_gcs_path, results_file_name)
      max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")

  # Final accuracy
  if total_count > 0:
    accuracy = correct_count / total_count
    max_logging.log(f"\nFinal accuracy on {hf_data_dir} dataset: {accuracy:.4f}")
  else:
    max_logging.log("No valid predictions were made.")

  # Save predictions to CSV and upload to GCS
  if result_gcs_path is not None and jax.process_index() == 0:
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_file_name, index=False)
    max_logging.log(f"Saved predictions to {results_file_name}")
    gcs_utils.upload_blob(result_gcs_path, results_file_name)
  max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")


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
