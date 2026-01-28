# Copyright 2023–2026 Google LLC
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

"""
This script executes the data generation step for Response-based Knowledge Distillation.
Knowledge Distillation is a compression technique that transfers knowledge
from a larger (teacher) model to a smaller (student) model.
The script runs inference on a teacher model using vLLM to create output samples.
This generated dataset can be used to fine-tune a student model.

Example command:
  python3 -m tools.data_generation.generate_distillation_data_vllm \
      --dataset-path HuggingFaceH4/ultrachat_200k \
      --data-split train_sft \
      --data-columns messages \
      --hf-access-token $HF_TOKEN \
      --teacher-model ${BASE_DIRECTORY}/qwen3-32b \
      --use-chat-template \
      --num-prompts 5120 \
      --output-file ${BASE_DIRECTORY}/datasets/distillation_data.parquet

This processes 5120 prompts, generating the specified number of samples per prompt.
Some prompts may be filtered out if prompt tokens are longer than `max-prefill-length`.
`max-target-length` is the max length of prompt tokens and expected completion tokens.
"""

import argparse
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
import transformers


def main():
  parser = argparse.ArgumentParser(description="Generate distillation data using vLLM.")
  parser.add_argument(
      "--dataset-path",
      type=str,
      default="HuggingFaceH4/ultrachat_200k",
      help="Path to Hugging Face dataset.",
  )
  parser.add_argument("--data-split", type=str, default="train_sft", help="Subset of data to load.")
  parser.add_argument(
      "--data-columns",
      nargs="+",
      default=["messages"],
      help="Columns names that contain relevant data.",
  )
  parser.add_argument(
      "--hf-access-token",
      type=str,
      required=True,
      help="Access token for Hugging Face.",
  )
  parser.add_argument(
      "--use-chat-template",
      action="store_true",
      help="Enable tokenizer to apply a chat template.",
  )
  parser.add_argument("--max-prefill-length", type=int, default=256, help="The maximum prompt length.")
  parser.add_argument(
      "--max-target-length",
      type=int,
      default=2048,
      help="The maximum prompt length plus the output completion length.",
  )
  parser.add_argument(
      "--num-generations",
      type=int,
      default=1,
      help="Number of samples to generate per prompt.",
  )
  parser.add_argument(
      "--num-prompts",
      type=int,
      default=5120,
      help="Number of prompts to process.",
  )
  parser.add_argument(
      "--output-file",
      type=str,
      default=os.path.join(os.environ.get("BASE_DIRECTORY", "."), "datasets", "distillation_data.parquet"),
      help="Output Parquet file path.",
  )
  parser.add_argument(
      "--teacher-model",
      type=str,
      default=os.path.join(os.environ.get("BASE_DIRECTORY", "."), "qwen3-32b"),
      help="Local path to downloaded teacher model.",
  )
  parser.add_argument("--tp-size", type=int, default=4, help="Number of TPU chips.")
  parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model length.")
  parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate.")
  parser.add_argument(
      "--max-num-batched-tokens",
      type=int,
      default=2048,
      help="Maximum number of batched tokens.",
  )
  parser.add_argument("--max-num-seqs", type=int, default=256, help="Maximum number of sequences.")
  parser.add_argument(
      "--gpu-memory-utilization",
      type=float,
      default=0.98,
      help="GPU memory utilization.",
  )

  config = parser.parse_args()

  # --- Configuration ---
  TEACHER_MODEL = config.teacher_model
  DATASET_NAME = config.dataset_path
  DATASET_SPLIT = config.data_split
  PROMPT_COLUMN = config.data_columns[0] if config.data_columns else "messages"
  OUTPUT_FILE = config.output_file
  TP_SIZE = config.tp_size
  MAX_MODEL_LEN = config.max_model_len
  MAX_NEW_TOKENS = config.max_new_tokens
  MAX_NUM_BATCHED_TOKENS = config.max_num_batched_tokens
  MAX_NUM_SEQS = config.max_num_seqs
  GPU_MEMORY_UTILIZATION = config.gpu_memory_utilization
  NUM_PROMPTS = config.num_prompts
  NUM_GENERATIONS = config.num_generations
  # ---------------------

  def apply_chat_template(example, tokenizer, prompt_column):
    messages = example[prompt_column]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"formatted_prompt": prompt}

  print(f"Loading dataset {DATASET_NAME} ({DATASET_SPLIT})...")
  dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

  # Limit dataset for tutorial
  dataset = dataset.select(range(min(len(dataset), NUM_PROMPTS)))

  print(f"Loading tokenizer {TEACHER_MODEL}...")
  tokenizer = transformers.AutoTokenizer.from_pretrained(TEACHER_MODEL)

  if config.use_chat_template:
    print("Formatting prompts...")
    dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer, PROMPT_COLUMN),
        desc="Applying chat template",
    )
    prompts = dataset["formatted_prompt"]
  else:
    prompts = dataset[PROMPT_COLUMN]

  print(f"Initializing vLLM with model {TEACHER_MODEL}...")
  llm = LLM(
      model=TEACHER_MODEL,
      max_model_len=MAX_MODEL_LEN,
      tensor_parallel_size=TP_SIZE,
      max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
      max_num_seqs=MAX_NUM_SEQS,
      gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
      enforce_eager=False,
  )

  sampling_params = SamplingParams(
      temperature=1.0,
      top_p=1.0,
      max_tokens=MAX_NEW_TOKENS,
      n=NUM_GENERATIONS,
  )

  print("Running inference...")
  outputs = llm.generate(prompts, sampling_params)

  # Collect results and save directly to Parquet.
  results = []
  for output, original_item in zip(outputs, dataset):
    for completion in output.outputs:
      msgs = list(original_item[PROMPT_COLUMN])
      msgs.append({"role": "assistant", "content": completion.text})
      results.append({"messages": msgs})

  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
  print(f"Saving results to {OUTPUT_FILE} (Parquet)")
  ds = Dataset.from_list(results)
  ds.to_parquet(OUTPUT_FILE)


if __name__ == "__main__":
  main()
