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
""" Evaluation script based on MLPerf requirements, extended for multiple file comparison."""

import argparse
from transformers import AutoTokenizer, LlamaTokenizer
import nltk
import evaluate
import numpy as np
import json
import pandas as pd
from itertools import combinations  # For pairwise comparisons
import os  # For basename


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
  parser.add_argument(
      "--mlperf-accuracy-files", required=True, nargs="+", help="Path(s) to one or more mlperf_log_accuracy.json files."
  )
  parser.add_argument(
      "--accuracy-file-labels",
      nargs="+",
      default=None,
      help="Optional labels for each accuracy file, in the same order. If not provided, filenames will be used.",
  )
  parser.add_argument("--dataset-file", required=True, help="path to processed openorca validation set (full dataset)")
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print verbose messages including prompt, ground truth, and prediction for individual files.",
  )
  parser.add_argument(
      "--show_tokens",
      action="store_true",
      help="In verbose mode, also print the raw predicted token IDs for individual files.",
  )
  parser.add_argument(
      "--verbose_pairwise_limit",
      type=int,
      default=0,
      help="Number of samples to print verbosely for pairwise comparisons (0 for none).",
  )
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

  if args.accuracy_file_labels and len(args.accuracy_file_labels) != len(args.mlperf_accuracy_files):
    raise ValueError("Number of --accuracy-file-labels must match the number of --mlperf-accuracy-files.")
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


def postprocess_text(preds, targets=None):
  preds = [str(pred).strip() for pred in preds]
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

  if targets is not None:
    targets = [str(target).strip() for target in targets]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]
    return preds, targets
  return preds


def process_single_accuracy_log(accuracy_file_path, tokenizer, eval_dataset, prompt_column_name, args):
  """
  Processes a single MLPerf accuracy log file.
  Returns a tuple: (results_summary_dict, predictions_by_qsl_idx_dict)
  """
  print(f"\n--- Processing accuracy log: {accuracy_file_path} ---")

  ground_truths_list = eval_dataset["output"].tolist()
  prompts_list = eval_dataset[prompt_column_name].tolist()

  target_required_for_rouge = []
  preds_token_ids_all = []
  verbose_data_list = []
  predictions_by_qsl_idx = {}

  eval_dtype = {"int32": np.int32, "int64": np.int64, "float": np.float32}[args.dtype]

  try:
    with open(accuracy_file_path, "rt", encoding="utf8") as f:
      results_from_log = json.load(f)
  except FileNotFoundError:
    print(f"Error: File not found at {accuracy_file_path}")
    return None, None
  except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from {accuracy_file_path}. Error: {e}")
    return None, None

  if not results_from_log:
    print(f"Warning: No results found in {accuracy_file_path}.")
    return

  seen_qsl_indices = set()
  gen_tok_len_total = 0

  for pred_entry in results_from_log:
    qsl_idx = pred_entry["qsl_idx"]
    if qsl_idx in seen_qsl_indices:
      if args.verbose:
        print(f"Skipping duplicate qsl_idx: {qsl_idx} in {accuracy_file_path}")
      continue
    seen_qsl_indices.add(qsl_idx)

    if qsl_idx >= len(ground_truths_list):
      print(
          f"Warning (file: {accuracy_file_path}): qsl_idx {qsl_idx} is out of bounds for dataset (size: {len(ground_truths_list)}). Skipping."
      )
      continue

    current_ground_truth = ground_truths_list[qsl_idx]
    current_prompt = prompts_list[qsl_idx]
    target_required_for_rouge.append(current_ground_truth)

    try:
      pred_tokens = np.frombuffer(bytes.fromhex(pred_entry["data"]), eval_dtype)
    except ValueError as e:
      print(f"Warning (file: {accuracy_file_path}, qsl_idx: {qsl_idx}): Could not decode hex. Error: {e}. Using EOS.")
      pred_tokens = np.array([tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2], dtype=eval_dtype)

    if len(pred_tokens) > 0 and (pred_tokens[0] > tokenizer.vocab_size or pred_tokens[0] < 0):
      if args.verbose:
        print(f"qsl_idx {qsl_idx} (file: {accuracy_file_path}): First token {pred_tokens[0]} invalid. Replacing with BOS.")
      bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
      pred_tokens = np.concatenate(([bos_id], pred_tokens[1:])).astype(eval_dtype)
    elif len(pred_tokens) == 0:
      if args.verbose:
        print(f"qsl_idx {qsl_idx} (file: {accuracy_file_path}): Empty prediction. Using EOS.")
      eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
      pred_tokens = np.array([eos_id], dtype=eval_dtype)

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
    print(f"No valid predictions processed from {accuracy_file_path}.")
    return

  preds_decoded_text_all = tokenizer.batch_decode(preds_token_ids_all, skip_special_tokens=True)

  idx_for_decoded_text = 0
  for item in verbose_data_list:  # This list only contains entries for successfully processed qsl_idx
    item["predicted_text"] = preds_decoded_text_all[idx_for_decoded_text]
    predictions_by_qsl_idx[item["qsl_idx"]] = preds_decoded_text_all[idx_for_decoded_text]
    idx_for_decoded_text += 1

  processed_qsl_indices_in_order = []
  # Re-iterate results_from_log to get the qsl_idx for predictions that were actually decoded
  # This ensures the qsl_idx lines up with preds_decoded_text_all
  temp_seen_for_order = set()
  for pred_entry_re_eval in results_from_log:
    qsl_idx_re_eval = pred_entry_re_eval["qsl_idx"]
    if qsl_idx_re_eval in temp_seen_for_order:
      continue
    if qsl_idx_re_eval >= len(ground_truths_list):
      continue  # Skip out of bounds
    try:  # Check if hex data was valid (mimic earlier check)
      _ = np.frombuffer(bytes.fromhex(pred_entry_re_eval["data"]), eval_dtype)
      processed_qsl_indices_in_order.append(qsl_idx_re_eval)
      temp_seen_for_order.add(qsl_idx_re_eval)
    except ValueError:
      continue  # Skip if hex was bad, as it would have been skipped for preds_token_ids_all

  if len(processed_qsl_indices_in_order) == len(preds_decoded_text_all):
    for i, q_idx in enumerate(processed_qsl_indices_in_order):
      predictions_by_qsl_idx[q_idx] = preds_decoded_text_all[i]
  else:
    print(
        f"Warning (file: {accuracy_file_path}): Mismatch in processed qsl_idx count ({len(processed_qsl_indices_in_order)}) and decoded text count ({len(preds_decoded_text_all)}). Predictions dictionary might be incomplete."
    )
    if verbose_data_list:
      for item in verbose_data_list:
        if "predicted_text" in item:
          predictions_by_qsl_idx[item["qsl_idx"]] = item["predicted_text"]

  if args.verbose:
    print(f"\n--- Verbose Comparison for: {accuracy_file_path} ---")
    for data_item in verbose_data_list:
      print(f"\nSample (qsl_idx: {data_item['qsl_idx']})")
      print(f"  Prompt: {data_item['prompt']}")
      print(f"  Original Output (Ground Truth): {data_item['ground_truth']}")
      print(
          f"  Predicted Output (Decoded): {data_item.get('predicted_text', '[Error - text not decoded for verbose print]')}"
      )
      if args.show_tokens:
        print(f"  Predicted Tokens: {data_item['predicted_tokens_list']}")
    print(f"--- End of Verbose Comparison for {accuracy_file_path} ---\n")

  preds_for_rouge, targets_for_rouge_processed = postprocess_text(preds_decoded_text_all, target_required_for_rouge)

  if not preds_for_rouge or not targets_for_rouge_processed:
    print(f"No data for ROUGE calculation for {accuracy_file_path}.")
    return 

  metric = evaluate.load("rouge")  # Load metric here to be thread-safe if we ever parallelize
  rouge_scores = metric.compute(
      predictions=preds_for_rouge, references=targets_for_rouge_processed, use_stemmer=True, use_aggregator=False
  )
  summary_result = {k: float(round(np.mean(v) * 100, 4)) for k, v in rouge_scores.items()}

  prediction_lens_chars = [len(pred_str) for pred_str in preds_for_rouge]
  gen_num_processed = len(preds_for_rouge)

  summary_result.update(
      {
          "gen_len_chars": int(np.sum(prediction_lens_chars)),
          "gen_num_evaluated": gen_num_processed,
          "gen_tok_len_total": int(gen_tok_len_total),  # Ensure Python int
          "tokens_per_sample": float(round(gen_tok_len_total / gen_num_processed, 1)) if gen_num_processed > 0 else 0.0,
      }
  )

  print(f"Summary for {accuracy_file_path}:")
  print(json.dumps(summary_result, indent=2))
  return summary_result, predictions_by_qsl_idx


def main():
  args = get_args()
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
    print(f"Loading tokenizer from path: {args.tokenizer_path}")
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
  else:
    raise ValueError("Either --checkpoint-path or --tokenizer-path must be provided for the tokenizer.")

  eval_dataset = get_dataset_for_evaluation(
      args.dataset_file, args.total_sample_count, args.python_seed, args.prompt_column_name
  )

  all_results_summary = {}
  all_predictions_data = {}

  file_labels = (
      args.accuracy_file_labels if args.accuracy_file_labels else [os.path.basename(f) for f in args.mlperf_accuracy_files]
  )

  for i, acc_file_path in enumerate(args.mlperf_accuracy_files):
    label = file_labels[i]
    summary, preds_by_qsl = process_single_accuracy_log(
        acc_file_path, tokenizer, eval_dataset, args.prompt_column_name, args
    )
    if summary and preds_by_qsl is not None:
      all_results_summary[label] = summary
      all_predictions_data[label] = preds_by_qsl
    else:
      print(f"Skipping {label} due to processing errors.")

  print("\n\n--- Overall Summary (vs Ground Truth) ---")
  for label, summary in all_results_summary.items():
    print(f"\nResults for: {label}")
    print(json.dumps(summary, indent=2))

  # Pairwise Comparisons
  if len(all_predictions_data) > 1:
    print("\n\n--- Pairwise ROUGE Comparisons ---")
    metric_for_pairwise = evaluate.load("rouge")

    for (label1, preds1_dict), (label2, preds2_dict) in combinations(all_predictions_data.items(), 2):
      print(f"\nComparing: '{label1}' vs '{label2}'")

      common_qsl_indices = sorted(list(set(preds1_dict.keys()) & set(preds2_dict.keys())))

      if not common_qsl_indices:
        print("  No common QSL indices found between these two files.")
        continue

      print(f"  Found {len(common_qsl_indices)} common samples for comparison.")

      preds_from_file1 = [preds1_dict[idx] for idx in common_qsl_indices]
      preds_from_file2 = [preds2_dict[idx] for idx in common_qsl_indices]

      # Postprocess text for ROUGE
      # Treating file1 as "predictions" and file2 as "references" for this comparison
      processed_preds1 = postprocess_text(preds_from_file1)
      processed_preds2 = postprocess_text(preds_from_file2)

      try:
        pairwise_rouge_scores = metric_for_pairwise.compute(
            predictions=processed_preds1, references=processed_preds2, use_stemmer=True, use_aggregator=False
        )
        pairwise_summary = {k: float(round(np.mean(v) * 100, 4)) for k, v in pairwise_rouge_scores.items()}
        print(f"  ROUGE of '{label1}' (as pred) vs '{label2}' (as ref):")
        print(f"  {json.dumps(pairwise_summary, indent=4)}")

      except Exception as e:
        print(f"  Error computing pairwise ROUGE for '{label1}' vs '{label2}': {e}")

      if args.verbose_pairwise_limit > 0:
        print(f"\n  --- Verbose Pairwise Sample Comparison (limit {args.verbose_pairwise_limit}) ---")
        prompts_list_common = [eval_dataset[args.prompt_column_name][idx] for idx in common_qsl_indices]

        for i in range(min(args.verbose_pairwise_limit, len(common_qsl_indices))):
          q_idx = common_qsl_indices[i]
          print(f"\n  Sample (qsl_idx: {q_idx})")
          print(f"    Prompt: {prompts_list_common[i]}")
          print(f"    '{label1}': {preds_from_file1[i]}")
          print(f"    '{label2}': {preds_from_file2[i]}")
        print(f"  --- End of Verbose Pairwise Sample Comparison ---")


if __name__ == "__main__":
  main()
