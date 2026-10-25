# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validates structural consistency between a MaxText blueprint and an Orbax checkpoint."""

import argparse
import json
import time
import absl.logging
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging as logger

# Initialize logging verbosity to INFO so logger.info is actually printed
absl.logging.set_verbosity(absl.logging.INFO)


def load_shapes(filepath):
  """Parses a file to extract key-shape pairs."""
  import os

  if not os.path.exists(filepath):
    raise FileNotFoundError(
        f"Required shape file '{filepath}' does not exist. "
        "Please ensure the upstream checkpoint inspection task executed successfully."
    )
  shapes = {}
  with open(filepath, "r", encoding="utf-8") as file_handle:
    for line in file_handle:
      if "key:" in line and "|" in line:
        parts = line.split("|", 1)
        shapes[parts[0].replace("key:", "").strip()] = parts[1].replace("shape:", "").strip()
  return shapes


def check_mismatches(ideal, actual):
  """Compares dictionaries and returns True if mismatches exist."""
  if not ideal or not actual:
    logger.info("MISMATCH: One or both shape dictionaries are empty. This is likely an upstream failure.")
    return True, ["FAILED_EMPTY_DICTIONARY"]

  all_keys = sorted(set(ideal.keys()) | set(actual.keys()))
  has_mismatch = False
  mismatched_layers = []
  match_count = 0

  for k in all_keys:
    exp = ideal.get(k, "MISSING")
    got = actual.get(k, "MISSING")
    if exp == got:
      match_count += 1
    else:
      logger.info(f"MISMATCH: {k} | Expected: {exp} -> Got: {got}")
      has_mismatch = True
      mismatched_layers.append(k)

  if match_count > 0:
    logger.info(f"Verification complete: {match_count} parameter layers matched perfectly.")
  return has_mismatch, mismatched_layers


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS dir to upload report")
  parser.add_argument("--run_name", type=str, default="unknown_run", help="Unique Airflow run identifier")
  parser.add_argument(
      "--ideal_shapes_path",
      type=str,
      default="/tmp/ideal_shapes.txt",
      help="Path to ideal shapes text file",
  )
  parser.add_argument(
      "--actual_shapes_path",
      type=str,
      default="/tmp/actual_shapes.txt",
      help="Path to actual shapes text file",
  )
  args = parser.parse_args()

  _has_mismatch = False
  _mismatched_layers = []
  error_message = None

  try:
    ideal_shapes = load_shapes(args.ideal_shapes_path)
    actual_shapes = load_shapes(args.actual_shapes_path)
    _has_mismatch, _mismatched_layers = check_mismatches(ideal_shapes, actual_shapes)
  except Exception as e:  # pylint: disable=broad-exception-caught
    _has_mismatch = True
    error_message = str(e) if str(e) else type(e).__name__

  report = {
      "run_name": args.run_name,
      "task": "checkpoint_shape_validation",
      "timestamp": time.time(),
      "status": "FAILURE" if _has_mismatch else "SUCCESS",
      "mismatches_found": _has_mismatch,
      "mismatched_layers": _mismatched_layers,
  }
  if error_message:
    report["error_message"] = error_message

  if args.report_gcs_dir:
    report_name = f"shape_validation_report_run_name_{args.run_name}_{int(time.time())}.json"
    gcs_dir = args.report_gcs_dir
    if not gcs_dir.endswith("/"):
      gcs_dir += "/"
    local_report_path = f"/tmp/{report_name}"
    try:
      with open(local_report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
      gcs_utils.upload_blob(f"{gcs_dir}{report_name}", local_report_path)
    except Exception as e:
      logger.error(f"Failed to write or upload shape validation report to GCS: {e}")

  if _has_mismatch:
    if error_message:
      raise RuntimeError(error_message)
    raise ValueError(f"ERROR: Structural mismatches found in {len(_mismatched_layers)} layers: {_mismatched_layers}")

  logger.info("\nSUCCESS: All parameters match perfectly.")
