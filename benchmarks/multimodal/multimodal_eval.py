# Copyright 2026 Google LLC
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

"""This script runs a multimodal quality benchmark for a trained checkpoint.

Usage:
# Gemma3-4b on a single TPU v4-8 VM
python3 -m benchmarks.multimodal.multimodal_eval MaxText/configs/base.yml \
  model_name=gemma3-4b tokenizer_path=assets/tokenizer.gemma3 \
  load_parameters_path=gs://maxtext-model-checkpoints/gemma3-4b/multimodal/2025-05-21-23-23-59/checkpoints/0/items \
  base_output_directory=$YOUR_GCS_PATH \
  per_device_batch_size=1 run_name=mmeval_test steps=1 async_checkpointing=false \
  scan_layers=false use_multimodal=true attention=\'dot_product\' \
  max_prefill_predict_length=550 max_target_length=570 per_device_batch_size=1 \
  hf_path=HuggingFaceM4/ChartQA hf_eval_split=test

# Llama4-17b-16e on a TPU v5p-128 cluster (images resized to 336x336 for simplicity)
python -m benchmarks.multimodal.multimodal_eval \
  MaxText/configs/base.yml model_name=llama4-17b-16e \
  tokenizer_path=meta-llama/Llama-4-Scout-17B-16E \
  load_parameters_path=gs://maxtext-model-checkpoints/llama4-17b-16e/hybrid/2025-07-22-11-03-20/0/items \
  base_output_directory=$YOUR_GCS_PATH \
  per_device_batch_size=1 run_name=mmeval_test steps=1 async_checkpointing=false \
  scan_layers=true use_multimodal=true attention=\'dot_product\' \
  max_prefill_predict_length=350 max_target_length=370 per_device_batch_size=1 \
  hf_path=HuggingFaceM4/ChartQA hf_eval_split=test hf_access_token=\'$YOUR_HF_ACCESS_TOKEN\' \
  ici_fsdp_parallelism=1 ici_expert_parallelism=16 ici_tensor_parallelism=4 \
  --image_resize=336
"""


import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import absl
from maxtext.inference.inference_utils import str2bool

import datasets
import jax
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.multimodal import processor as mm_processor
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.trainers.post_train.rl import utils_rl


@dataclass
class DebugConfig:
  rl: bool = False


@dataclass
class TmvpConfig:
  solution_start_token: str = "<answer>"
  solution_end_token: str = "</answer>"
  debug: DebugConfig = field(default_factory=DebugConfig)


absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log


ASCII_UPPERCASE_A = ord("A")  # ASCII value for uppercase 'A'
SUPPORTED_DATASETS = ["HuggingFaceM4/ChartQA"]

# To guide any ckpts converted from HF to answer in the desired format, use a default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are an expert at answering questions based
on provided charts. Your task is to extract the exact answer from the
given context or determine that it's not present.
For numerical answers, provide only the number.
For text answers, provide only the exact text.
For judgement questions, respond with "Yes" or "No".
If not found, output "N/A".
Your output must be only the exact answer within <answer></answer>, with no extra contents.

Example:
Question: What is the capital of France?
Your answer: <answer>Paris</answer>

Chart: {image_placeholder} Question: {question}
"""

# For MaxText SFT ckpts, use a simpler prompt (aligned with input_pipeline_utils.reformat_prompt)
SFT_PROMPT_TEMPLATE = "{image_placeholder}{question}"


@dataclass
class ParsedDatasetExample:
  """Parsed example from the HuggingFace dataset."""

  question: Optional[str] = None
  image_np: Optional[np.ndarray] = None
  choices: Optional[List[str]] = None
  answer: Optional[str] = None


def parse_dataset_example(example, hf_dataset_name, local_args):
  """Parse a single example from the HuggingFace dataset."""
  parsed_example = ParsedDatasetExample()
  if hf_dataset_name == "HuggingFaceM4/ChartQA":
    parsed_example.question = example["query"]
    parsed_example.image_np = np.asarray(example["image"].convert("RGB"))  # Convert PIL object to np array
    parsed_example.answer = example["label"][0]
  else:
    raise ValueError(f"Unsupported dataset: {hf_dataset_name}")

  # Resize the image if specified. This helps simplify the llama4's tiling, so we have a fixed input size
  if local_args.image_resize != -1:
    pil_img = Image.fromarray(parsed_example.image_np)
    pil_img = pil_img.resize((local_args.image_resize, local_args.image_resize))
    parsed_example.image_np = np.asarray(pil_img.convert("RGB"))

  return parsed_example


def construct_prompt(
    parsed_dataset_example: ParsedDatasetExample, config, local_args, system_message: Optional[str] = None
):
  """Construct prompt from a parsed dataset example."""
  image_placeholder = config.image_placeholder
  choices_text = (
      "\n".join(f"{chr(ASCII_UPPERCASE_A + idx)}. {choice}" for idx, choice in enumerate(parsed_dataset_example.choices))
      if parsed_dataset_example.choices
      else ""
  )
  if local_args.ckpt_type == "base":
    prompt = DEFAULT_PROMPT_TEMPLATE.format(
        image_placeholder=image_placeholder,
        question=parsed_dataset_example.question,
        choices=choices_text if choices_text else "N/A",
    )
  elif local_args.ckpt_type == "sft":
    prompt = mm_processor.reformat_prompt(
        parsed_dataset_example.question,
        image_placeholder,
        config.model_name,
        num_images=1 if config.use_multimodal else 0,
    )
  else:
    raise ValueError(f"Unsupported ckpt_type: {local_args.ckpt_type}")

  prompt = system_message + "\n\n" + prompt if system_message else prompt
  return prompt


def main(config, local_args):
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
  hf_path = config.hf_path
  hf_eval_split = config.hf_eval_split

  # Config for saving csv results
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  results_file_name = local_args.tmp_results_file
  result_gcs_path = f"{config.base_output_directory}/{timestamp}.csv" if config.base_output_directory else None
  max_logging.log(f"Results will be saved to {results_file_name} and uploaded to GCS: {result_gcs_path}")
  results_data = []
  tmvp_config = TmvpConfig()

  test_ds = datasets.load_dataset(hf_path, "default", split=hf_eval_split)
  for idx, example in enumerate(tqdm(test_ds, desc=f"Evaluating {hf_path} dataset")):
    prefill_length = config.max_prefill_predict_length
    parsed_dataset_example = parse_dataset_example(example, hf_path, local_args)
    prompt = construct_prompt(parsed_dataset_example, config, local_args)
    processor_output = mm_processor.preprocess_image_for_training(parsed_dataset_example.image_np, config.model_name)
    prefill_length -= mm_processor.get_image_offsets(config=config, processor_output=processor_output)
    print("\n" + "*" * 50)

    # Tokenize the input
    tokens, true_length = tokenizer.encode(prompt, is_bos=True, prefill_lengths=[prefill_length])
    if config.use_multimodal:
      tokens = mm_processor.prepare_text_for_image_fusion(tokens=tokens, config=config, processor_output=processor_output)
      image_offsets = mm_processor.get_image_offsets(config=config, processor_output=processor_output)
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
    prefill_result, first_token = engine.prefill(
        params=params, padded_tokens=tokens, images=processor_output.pixel_values, true_length=true_length
    )
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
      predicted_answer = utils_rl.extract_answer(output, tmvp_config)
      if predicted_answer != utils_rl.FALLBACK_ANSWER:
        break

      # Generate next token
      decode_state, sampled_token = engine.generate(params, decode_state)
      sampled_tokens.append(sampled_token.get_result_at_slot(slot).tokens.item())
      if sampled_tokens[-1] == tokenizer.eos_id:
        break

    correct_answer = parsed_dataset_example.answer
    if predicted_answer == utils_rl.FALLBACK_ANSWER:
      predicted_answer = utils_rl.extract_answer(output, tmvp_config)

    exact_correct, _ = utils_rl.check_correctness(predicted_answer, [correct_answer], tmvp_config)
    is_correct = exact_correct

    # Log answer
    max_logging.log(
        f"{total_count + 1} | {parsed_dataset_example.question}\n"
        f"[Model output] {output}\n"
        f"[Label answer] {correct_answer}\n"
        f"Matching: {is_correct}"
    )

    # Save results for CSV
    results_data.append(
        {
            "question ID": total_count + 1,
            "question": parsed_dataset_example.question,
            "label": parsed_dataset_example.answer,
            "output": output,
            "is_correct": is_correct,
        }
    )

    # Update accuracy for overall
    if is_correct:
      correct_count += 1
    total_count += 1
    max_logging.log(f"Running accuracy: {correct_count / (total_count):.4f} | Processed: {total_count}/{len(test_ds)}")

    if local_args.num_examples != -1 and total_count >= local_args.num_examples:
      break

    # Every 100 rows, save intermediate results to CSV and upload to GCS
    if idx % 100 == 0 and result_gcs_path is not None and jax.process_index() == 0:
      results_df = pd.DataFrame(results_data)
      results_df.to_csv(results_file_name, index=False)
      gcs_utils.upload_blob(result_gcs_path, results_file_name)
      max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")

  # Final accuracy
  if total_count > 0:
    accuracy = correct_count / total_count
    max_logging.log(f"\nFinal accuracy on {hf_path} dataset: {accuracy:.4f}")
  else:
    max_logging.log("No valid predictions were made.")

  # Save predictions to CSV and upload to GCS
  if result_gcs_path is not None and jax.process_index() == 0:
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_file_name, index=False)
    max_logging.log(f"Saved predictions to {results_file_name}")
    gcs_utils.upload_blob(result_gcs_path, results_file_name)
    max_logging.log(f"Uploaded the results file to GCS bucket: {result_gcs_path}")

    if local_args.remove_tmp_results and os.path.exists(results_file_name):
      os.remove(results_file_name)
      max_logging.log(f"Removed temporary results file: {results_file_name}")


def validate_config(config):
  assert not config.load_full_state_path, (
      "Decode doesn't operate on full states! Convert to parameter checkpoint"
      " first. Using generate_param_only_checkpoint."
  )
  assert (
      config.hf_path
  ), "For benchmark evaluation, please specify the HuggingFace dataset name using the hf_path config field."
  assert config.hf_path in SUPPORTED_DATASETS, (
      f"Unsupported dataset {config.hf_path}. Supported datasets are: {SUPPORTED_DATASETS}."
      " Please add support for your desired dataset in the code of multimodal_eval.py."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--num_examples", type=int, required=False, default=-1, help="Number of examples to evaluate. Default to -1 (all)."
  )
  parser.add_argument(
      "--tmp_results_file",
      type=str,
      required=False,
      default="mm_eval_results.csv",
      help="Temporary results CSV file path.",
  )
  parser.add_argument(
      "--remove_tmp_results",
      type=str2bool,
      required=False,
      default=True,
      help="Whether to remove the temporary results CSV file after uploading to GCS.",
  )
  parser.add_argument(
      "--ckpt_type",
      type=str,
      required=False,
      default="base",
      choices=["base", "sft"],
      help=(
          "Checkpoint type: 'base' (uses DEFAULT_PROMPT_TEMPLATE) or 'sft' (uses"
          " SFT_PROMPT_TEMPLATE with model-specific reformat_prompt)."
      ),
  )
  parser.add_argument(
      "--image_resize",
      type=int,
      required=False,
      default=-1,
      help="Resize images to this square size before processing. -1 disables resizing.",
  )

  _local_args, remaining_args = parser.parse_known_args()
  model_args = [sys.argv[0]] + remaining_args

  cfg = pyconfig.initialize(model_args)
  validate_config(cfg)
  max_utils.print_system_information()
  main(cfg, _local_args)
