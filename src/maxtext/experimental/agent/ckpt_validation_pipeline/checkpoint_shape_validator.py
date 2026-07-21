# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "innovation" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validates structural consistency between a MaxText blueprint and an Orbax checkpoint."""

import argparse
import json
import time
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging as logger


def load_shapes(filepath):
  """Parses a file to extract key-shape pairs."""
  shapes = {}
  with open(filepath, "r", encoding="utf-8") as file_handle:
    for line in file_handle:
      if "key:" in line and "|" in line:
        parts = line.split("|")
        shapes[parts[0].replace("key:", "").strip()] = parts[1].replace("shape:", "").strip()
  return shapes


def check_mismatches(ideal, actual):
  """Compares dictionaries and returns True if mismatches exist."""
  all_keys = sorted(set(ideal.keys()) | set(actual.keys()))
  has_mismatch = False

  for k in all_keys:
    exp = ideal.get(k, "MISSING")
    got = actual.get(k, "MISSING")
    if exp == got:
      logger.info(f"MATCH: {k} | Expected: {exp} -> Got: {got}")
    else:
      logger.info(f"MISMATCH: {k} | Expected: {exp} -> Got: {got}")
      has_mismatch = True

  return has_mismatch



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS dir to upload report")
  parser.add_argument("--ideal_shapes_path", type=str, default="/tmp/ideal_shapes.txt", help="Path to ideal shapes text file")
  parser.add_argument("--actual_shapes_path", type=str, default="/tmp/actual_shapes.txt", help="Path to actual shapes text file")
  args = parser.parse_args()

  ideal_shapes = load_shapes(args.ideal_shapes_path)
  actual_shapes = load_shapes(args.actual_shapes_path)

  _has_mismatch = check_mismatches(ideal_shapes, actual_shapes)

  report = {
      "task": "checkpoint_shape_validation",
      "timestamp": time.time(),
      "status": "FAILURE" if _has_mismatch else "SUCCESS",
      "mismatches_found": _has_mismatch,
  }

  if args.report_gcs_dir:
    report_name = f"shape_validation_report_{int(time.time())}.json"
    gcs_dir = args.report_gcs_dir
    if not gcs_dir.endswith("/"):
      gcs_dir += "/"
    local_report_path = f"/tmp/{report_name}"
    with open(local_report_path, "w", encoding="utf-8") as report_file:
      json.dump(report, report_file, indent=2)
    gcs_utils.upload_blob(f"{gcs_dir}{report_name}", local_report_path)

  if _has_mismatch:
    raise ValueError("ERROR: Structural mismatches found!")

  logger.info("\nSUCCESS: All parameters match perfectly.")
