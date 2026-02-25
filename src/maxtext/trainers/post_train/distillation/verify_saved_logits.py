"""
Verification script to check the correctness of saved top-k teacher logits.

Example usage:
python3 src/maxtext/trainers/post_train/distillation/verify_saved_logits.py \
  --output_dir=/path/to/your/output \
  --expected_steps=1000 \
  --top_k=128
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

  file_pattern = f"{output_dir}/teacher_top_k_process_*.array_record"
  files = tf.io.gfile.glob(file_pattern)

  if not files:
    max_logging.log(f"Error: No ArrayRecord files found matching {file_pattern}")
    return

  max_logging.log(f"Found {len(files)} ArrayRecord files. Starting verification...")

  for file_path in files:
    max_logging.log(f"Verifying: {file_path}")
    reader = array_record_module.ArrayRecordReader(file_path)
    num_records = reader.num_records()

    step_count = 0
    for _ in range(num_records):
      record = reader.read()
      data = pickle.loads(record)

      # Verify all required keys are present
      for key in ["tokens", "top_k_logits", "top_k_indices"]:
        assert key in data, f"Missing required key '{key}' at step {step_count} in {file_path}"

      # Verify all optional keys are present
      for key in expected_keys:
        assert key in data, f"Missing optional key '{key}' at step {step_count} in {file_path}"

      # Verify shapes for Top-K outputs
      actual_k_logits = data["top_k_logits"].shape[-1]
      actual_k_indices = data["top_k_indices"].shape[-1]
      assert actual_k_logits == expected_k, f"Expected top_k={expected_k}, got {actual_k_logits} for logits"
      assert actual_k_indices == expected_k, f"Expected top_k={expected_k}, got {actual_k_indices} for indices"

      step_count += 1

    # Verify the total number of steps processed
    assert step_count == expected_steps, f"Expected {expected_steps} steps, but found {step_count} in {file_path}."

    max_logging.log(f"Successfully verified {file_path}")
    max_logging.log(f"- Total steps: {step_count} (Matches expected)")
    max_logging.log(f"- Top-K dimension: {expected_k}")
    max_logging.log(f"- Keys verified: {list(data.keys())}")


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
