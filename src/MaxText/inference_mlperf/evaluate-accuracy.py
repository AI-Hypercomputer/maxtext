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
""" Evaluation script based on MLPerf requirements"""

import argparse
import json

from transformers import AutoTokenizer, LlamaTokenizer

import nltk

import evaluate

import numpy as np

import pandas as pd


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint-path", required=True, help="Path to Llama2-70b-hf-chat checkpoint")
  parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
  parser.add_argument("--dataset-file", required=True, help="path to processed openorca validation set")
  parser.add_argument("--verbose", action="store_true", help="verbose messages")
  parser.add_argument("--dtype", default="int64", help="dtype of the accuracy log", choices=["int32", "int64", "float"])
  args = parser.parse_args()
  return args


def get_groundtruth(processed_dataset_file):
  data = pd.read_pickle(processed_dataset_file)
  ground_truths = data["output"]
  return ground_truths


def postprocess_text(preds, targets):
  preds = [pred.strip() for pred in preds]
  targets = [target.strip() for target in targets]

  # rougeLSum expects newline after each sentence
  preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
  targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

  return preds, targets


def main():

  args = get_args()
  metric = evaluate.load("rouge")
  nltk.download("punkt")

  if args.checkpoint_path:
    print(f"Loading checkpoint from {args.checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )
  elif args.tokenizer_path:
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = LlamaTokenizer(args.tokenizer_path)
  else:
    raise ValueError("Either --checkpoint-path or --tokenizer-path must be provided")

  targets = get_groundtruth(args.dataset_file)

  target_required = []
  preds_token_ids = []

  eval_dtype = np.int64
  if args.dtype == "int32":
    eval_dtype = np.int32
  elif args.dtype == "float":
    eval_dtype = np.float32

  with open(args.mlperf_accuracy_file, "rt", encoding="utf8") as f:
    results = json.load(f)

  seen = set()
  gen_tok_len = 0
  for pred in results:
    qsl_idx = pred["qsl_idx"]
    if qsl_idx in seen:
      continue

    seen.add(qsl_idx)
    target = targets[qsl_idx]
    target_required.append(target)
    pred = np.frombuffer(bytes.fromhex(pred["data"]), eval_dtype)
    if pred[0] > 32000 or pred[0] < 0:
      pred = [1, *pred[1:]]
    gen_tok_len += len(pred)
    preds_token_ids.append(pred)

  preds_decoded_text = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True)

  preds, targets = postprocess_text(preds_decoded_text, target_required)

  result = metric.compute(predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
  result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
  prediction_lens = [len(pred) for pred in preds]
  gen_num = len(preds)

  result = {
      **result,
      "gen_len": np.sum(prediction_lens),
      "gen_num": gen_num,
      "gen_tok_len": gen_tok_len,
      "tokens_per_sample": round(gen_tok_len / gen_num, 1),
  }

  print("\nResults\n")
  print(result)


if __name__ == "__main__":
  main()
