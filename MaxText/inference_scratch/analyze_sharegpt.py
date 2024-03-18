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

import argparse
import json

PREFILL_BUCKET_SIZE_TO_MS = {64: 9.174, 128: 11.087, 256: 18.468, 512: 29.128, 1024: 58.386}
SYSTEM_TIME_PER_DECODE_TOKEN_MS = 0.32591875
MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 1024

def next_power_of_2(x):
  return 1 if x == 0 else 2**(x - 1).bit_length()

def tokens_in_input_str(s):
  return_val =  int(1.3 * len(s.split()))
  return return_val

def get_prefill_and_generate_times(filename=""):
  if filename == "":
    return PREFILL_BUCKET_SIZE_TO_MS, SYSTEM_TIME_PER_DECODE_TOKEN_MS

  prefill_bucket_size_to_ms = {}
  with open(filename, "r") as f:
    microbenchmark_results = json.load(f)
  for k, v in microbenchmark_results["Prefill"].items():
    prefill_bucket_size_to_ms[int(k)] = round(v["prefill_time_in_ms"], 3)

  return prefill_bucket_size_to_ms, microbenchmark_results['AutoRegressive']['ar_step_in_ms_per_seq']

def get_conversations_from_file(filename, max_input_tokens, max_output_tokens):
  convo_token_numbers = []
  with open(filename, 'r') as f:
    loaded_share_gpt = json.load(f)
  for example in loaded_share_gpt:
    if len(example['conversations']) < 2:
      continue
    num_input_tokens = tokens_in_input_str(example['conversations'][0]['value'])
    num_output_tokens = tokens_in_input_str(example['conversations'][1]['value'])
    convo_token_numbers.append((num_input_tokens, num_output_tokens))

  num_convos = len(convo_token_numbers)
  kept_convos = [c for c in convo_token_numbers if c[0] <= max_input_tokens and c[1] <= max_output_tokens]

  mean_input = sum(c[0] for c in kept_convos) / len(kept_convos)
  mean_output = sum(c[1] for c in kept_convos) / len(kept_convos)

  print(f"Kept {len(kept_convos)} of {num_convos} total convos. {len(100*kept_convos)/num_convos:.3f}%")
  print(f"Out of kept convos, mean input tokens: {mean_input:.3f}, mean output tokens: {mean_output:.3f}")
  return kept_convos


def compute_times(convos, prefill_bucket_size_to_ms, system_time_per_decode_token_ms, verbose=False):
  total_prefill_system_ms = 0
  total_generate_system_ms = 0
  for convo in convos:
    input_tok, output_tok = convo
    bucket = max(128, next_power_of_2(input_tok))
    generate_system_ms = output_tok * system_time_per_decode_token_ms
    prefill_system_ms = prefill_bucket_size_to_ms[bucket]
    total_prefill_system_ms += prefill_system_ms
    total_generate_system_ms += generate_system_ms
    if verbose:
      print(f"{convo} {bucket}, {prefill_system_ms:.2f}, {generate_system_ms:.2f}")

  total_prefill_time_seconds = total_prefill_system_ms / 1000
  total_generate_time_seconds = total_generate_system_ms / 1000
  total_time_s = total_prefill_time_seconds + total_generate_time_seconds

  print(f"\nTotal time {total_time_s:.3f} seconds: "
        f"\n\tPrefill time: {total_prefill_time_seconds:.3f} seconds"
        f"\n\tGenerate time: {total_generate_time_seconds:.3f} seconds")
  return total_time_s, total_prefill_time_seconds, total_generate_time_seconds


def get_num_tokens_in_convos(convos):
  num_input_tokens = sum(c[0] for c in convos)
  num_output_tokens = sum(c[1] for c in convos)
  return num_input_tokens, num_output_tokens


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('convo_file', type=str,
                      help='a json file containing conversations')
  parser.add_argument('-t', '--mb_timing_file', type=str, default="",
                      help='a json file containing microbenchmark timing results')
  parser.add_argument('-v', '--verbose', action="store_true")
  args = parser.parse_args()

  convos = get_conversations_from_file(args.convo_file, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)
  total_input_tokens, total_output_tokens = get_num_tokens_in_convos(convos)
  prefill_time_ms_buckets, generate_time_ms = get_prefill_and_generate_times(filename=args.mb_timing_file)
  total_time_seconds, _, _ = compute_times(convos, prefill_time_ms_buckets, generate_time_ms, args.verbose)

  print(f"Output {total_output_tokens} tokens in {total_time_seconds:.3f} seconds "
        f"= {total_output_tokens/total_time_seconds:.3f} out tok/s")
