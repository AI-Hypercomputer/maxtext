# Copyright 2024 Google LLC
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
""" Evaluation script based on MLPerf requirements"""

import argparse
from transformers import AutoTokenizer, LlamaTokenizer
import nltk
import evaluate
import numpy as np
import json
import pandas as pd


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--checkpoint-path",
      default=None,
      help="Path to Llama2-70b-hf-chat checkpoint (optional if tokenizer_path is provided)",
  )
  parser.add_argument(
      "--tokenizer-path", default=None, help="Path to tokenizer files (optional if checkpoint-path is provided)"
  )
  parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
  parser.add_argument("--dataset-file", required=True, help="path to processed openorca validation set (full dataset)")
  parser.add_argument(
      "--verbose", action="store_true", help="Print verbose messages including prompt, ground truth, and prediction."
  )
  parser.add_argument("--show_tokens", action="store_true", help="In verbose mode, also print the raw predicted token IDs.")
  parser.add_argument(
      "--dtype", default="int64", help="dtype of the accuracy log's 'data' field", choices=["int32", "int64", "float"]
  )
  parser.add_argument("--python_seed", type=int, default=42, help="Seed used for sampling the dataset during inference.")
  parser.add_argument(
      "--total_sample_count", type=int, required=True, help="Number of samples used during inference (from the SUT)."
  )
  parser.add_argument(
      "--prompt_column_name",
      type=str,
      default="input",
      help="Name of the column containing input prompts/text in the dataset (default: 'input').",
  )
  args = parser.parse_args()
  return args


def get_dataset_for_evaluation(processed_dataset_file, total_sample_count, seed, prompt_column_name):
  print(f"Loading full dataset from: {processed_dataset_file}")
  full_dataset = pd.read_pickle(processed_dataset_file)
  print(f"Full dataset loaded with {len(full_dataset)} samples. Columns: {full_dataset.columns.tolist()}")

  if total_sample_count < len(full_dataset):
    print(f"Sampling {total_sample_count} samples from the full dataset using seed {seed}.")
    eval_dataset = full_dataset.sample(n=total_sample_count, random_state=seed)
  else:
    print(f"Using the full dataset ({len(full_dataset)} samples) as total_sample_count ({total_sample_count}) is not less.")
    eval_dataset = full_dataset

  eval_dataset.reset_index(drop=True, inplace=True)
  print(f"Final dataset for evaluation has {len(eval_dataset)} samples (indices 0 to {len(eval_dataset)-1}).")

  if "output" not in eval_dataset.columns:
    raise ValueError("Dataset must contain an 'output' column for ground truths.")
  if prompt_column_name not in eval_dataset.columns:
    raise ValueError(
        f"Dataset must contain a '{prompt_column_name}' column for prompts. Available columns: {eval_dataset.columns.tolist()}"
    )

  return eval_dataset


def postprocess_text(preds, targets):
  preds = [str(pred).strip() for pred in preds]
  targets = [str(target).strip() for target in targets]

  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

  return preds, targets


def main():
  args = get_args()
  metric = evaluate.load("rouge")
  nltk.download("punkt", quiet=True)

  if args.checkpoint_path:
    print(f"Loading tokenizer from checkpoint: {args.checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )
  elif args.tokenizer_path:
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
  else:
    raise ValueError("Either --checkpoint-path or --tokenizer-path must be provided")

  eval_dataset = get_dataset_for_evaluation(
      args.dataset_file, args.total_sample_count, args.python_seed, args.prompt_column_name
  )

  ground_truths_list = eval_dataset["output"].tolist()
  prompts_list = eval_dataset[args.prompt_column_name].tolist()

  target_required_for_rouge = []
  preds_token_ids_all = []
  verbose_data_list = []

  eval_dtype = np.int64
  if args.dtype == "int32":
    eval_dtype = np.int32
  elif args.dtype == "float":
    eval_dtype = np.float32

  try:
    with open(args.mlperf_accuracy_file, "rt", encoding="utf8") as f:
      results_from_log = json.load(f)
  except FileNotFoundError:
    print(f"Error: MLPerf accuracy file not found at {args.mlperf_accuracy_file}")
    return
  except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from {args.mlperf_accuracy_file}. Error: {e}")
    return

  if not results_from_log:
    print(f"Warning: No results found in {args.mlperf_accuracy_file}.")
    return

  seen_qsl_indices = set()
  gen_tok_len_total = 0

  for pred_entry in results_from_log:
    qsl_idx = pred_entry["qsl_idx"]
    if qsl_idx in seen_qsl_indices:
      if args.verbose:
        print(f"Skipping duplicate qsl_idx: {qsl_idx}")
      continue
    seen_qsl_indices.add(qsl_idx)

    if qsl_idx >= len(ground_truths_list):
      print(
          f"Warning: qsl_idx {qsl_idx} from accuracy log is out of bounds for the loaded dataset (size: {len(ground_truths_list)}). Skipping this entry."
      )
      continue

    current_ground_truth = ground_truths_list[qsl_idx]
    current_prompt = prompts_list[qsl_idx]
    target_required_for_rouge.append(current_ground_truth)

    try:
      pred_tokens = np.frombuffer(bytes.fromhex(pred_entry["data"]), eval_dtype)
    except ValueError as e:
      print(f"Warning: Could not decode hex data for qsl_idx {qsl_idx}. Error: {e}. Using EOS token as prediction.")
      pred_tokens = np.array([tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2], dtype=eval_dtype)

    if len(pred_tokens) > 0 and (pred_tokens[0] > tokenizer.vocab_size or pred_tokens[0] < 0):
      if args.verbose:
        print(
            f"qsl_idx {qsl_idx}: First token {pred_tokens[0]} invalid (vocab size: {tokenizer.vocab_size}). Replacing with BOS (ID: {tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1}). Original: {pred_tokens[:5]}"
        )
      bos_token_id_to_use = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
      pred_tokens = np.concatenate(([bos_token_id_to_use], pred_tokens[1:])).astype(eval_dtype)
    elif len(pred_tokens) == 0:
      if args.verbose:
        print(
            f"qsl_idx {qsl_idx}: Predicted token sequence empty. Using EOS (ID: {tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2})."
        )
      eos_token_id_to_use = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
      pred_tokens = np.array([eos_token_id_to_use], dtype=eval_dtype)

    gen_tok_len_total += len(pred_tokens)
    preds_token_ids_all.append(pred_tokens)

    if args.verbose:
      verbose_data_list.append(
          {
              "qsl_idx": qsl_idx,
              "prompt": current_prompt,
              "ground_truth": current_ground_truth,
              "predicted_tokens_list": pred_tokens.tolist(),
          }
      )

  if not preds_token_ids_all:
    print("No valid predictions were processed to decode or evaluate after filtering.")
    return

  preds_decoded_text_all = tokenizer.batch_decode(preds_token_ids_all, skip_special_tokens=True)

  if args.verbose:
    print("\n--- Verbose Comparison (Sampled Data) ---")
    for i, data_item in enumerate(verbose_data_list):
      data_item["predicted_text"] = preds_decoded_text_all[i]
      print(f"\nSample (qsl_idx from log file: {data_item['qsl_idx']})")
      print(f"\n\tPrompt: {data_item['prompt']}")
      print(f"\n\tOriginal Output (Ground Truth): {data_item['ground_truth']}")
      print(f"\n\tPredicted Output (Decoded): {data_item['predicted_text']}")
      if args.show_tokens:
        print(f"\n\tPredicted Tokens: {data_item['predicted_tokens_list']}")
      print("\n----------------------------------\n")
    print("--- End of Verbose Comparison ---\n")

  preds_for_rouge, targets_for_rouge = postprocess_text(preds_decoded_text_all, target_required_for_rouge)

  if not preds_for_rouge or not targets_for_rouge:
    print("No data available for ROUGE calculation after postprocessing.")
    return

  rouge_scores = metric.compute(
      predictions=preds_for_rouge, references=targets_for_rouge, use_stemmer=True, use_aggregator=False
  )
  # Convert NumPy floats from np.mean to Python floats for JSON serialization
  final_result = {k: float(round(np.mean(v) * 100, 4)) for k, v in rouge_scores.items()}

  prediction_lens_chars = [len(pred_str) for pred_str in preds_for_rouge]
  gen_num_processed = len(preds_for_rouge)

  final_result.update(
      {
          "gen_len_chars": int(np.sum(prediction_lens_chars)),
          "gen_num_evaluated": gen_num_processed,
          "gen_tok_len_total": gen_tok_len_total,
          "tokens_per_sample": float(round(gen_tok_len_total / gen_num_processed, 1))
          if gen_num_processed > 0
          else 0.0,
      }
  )

  print("\nResults\n")
  print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
  main()
