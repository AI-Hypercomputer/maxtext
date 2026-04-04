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
Verification script to check the correctness of saved top-k teacher logits.

Example usage:
python3 python3 src/maxtext/trainers/post_train/distillation/verify_saved_logits.py \
  --output_dir=/tmp/save_logits_dir \
  --expected_steps=140
"""

import functools
import sys

import argparse
import pickle
from absl import app
import tensorflow as tf
from array_record.python import array_record_module
from maxtext.utils import max_logging


def verify_array_records(output_dir, expected_steps, expected_k, expected_keys):
  """Verifies the contents of ArrayRecord files containing top-k teacher logits."""

  file_pattern = f"{output_dir}/*.array_record"
  files = tf.io.gfile.glob(file_pattern)

  if not files:
    assert False, f"Error: No ArrayRecord files found matching {file_pattern}"

  max_logging.log(f"Found {len(files)} ArrayRecord files. Starting verification...")

  total_records_processed = 0
  all_keys_verified = set()

  for file_path in files:
    max_logging.log(f"Verifying: {file_path}")
    reader = array_record_module.ArrayRecordReader(file_path)
    num_records_in_file = reader.num_records()

    if num_records_in_file == 0:
      max_logging.log(f"Warning: {file_path} is empty.")
      continue

    for record_idx in range(num_records_in_file):
      record = reader.read()
      data = pickle.loads(record)

      # Verify all required keys are present
      required_keys = ["tokens", "top_k_logits", "top_k_indices"]
      for key in required_keys:
        assert key in data, f"Missing required key '{key}' in record {record_idx} in {file_path}"

      # Verify all optional keys are present
      for key in expected_keys:
        assert key in data, f"Missing optional key '{key}' in record {record_idx} in {file_path}"

      # Verify shapes for Top-K outputs
      actual_k_logits = data["top_k_logits"].shape[-1]
      actual_k_indices = data["top_k_indices"].shape[-1]
      assert actual_k_logits == expected_k, f"Expected top_k={expected_k}, got {actual_k_logits} for logits"
      assert actual_k_indices == expected_k, f"Expected top_k={expected_k}, got {actual_k_indices} for indices"

      if not all_keys_verified:
        all_keys_verified.update(data.keys())

    total_records_processed += num_records_in_file
    max_logging.log(f"Verified {num_records_in_file} records in {file_path}")

  # Verify the total number of steps processed across all files
  assert (
      total_records_processed == expected_steps
  ), f"Expected a total of {expected_steps} steps across all files, but found {total_records_processed}."

  max_logging.log("-----------------------------------------")
  max_logging.log("Verification Successful!")
  max_logging.log(f"- Total files verified: {len(files)}")
  max_logging.log(f"- Total steps verified: {total_records_processed} (Matches expected)")
  max_logging.log(f"- Top-K dimension: {expected_k}")
  max_logging.log(f"- Keys verified in records: {sorted(list(all_keys_verified))}")


def main(argv, local_args):
  verify_array_records(local_args.output_dir, local_args.expected_steps, local_args.top_k, local_args.optional_keys)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the array_record files.")
  parser.add_argument(
      "--expected_steps", type=int, required=True, help="Number of expected steps (matches config.steps)."
  )
  parser.add_argument("--top_k", type=int, default=128, help="Expected top K value.")
  parser.add_argument(
      "--optional_keys",
      type=str,
      nargs="*",
      default=["inputs_position", "inputs_segmentation", "targets_segmentation", "targets"],
      help="Optional keys expected to be in the record.",
  )

  local_arg, remaining_args = parser.parse_known_args()
  main_wrapper = functools.partial(main, local_args=local_arg)
  app.run(main_wrapper, argv=[sys.argv[0]] + remaining_args)
