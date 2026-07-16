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

import sys
import argparse
import json
import time


def load_shapes(filepath):
  """Parses a file to extract key-shape pairs."""
  shapes = {}
  with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
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
      print(f"MATCH: {k} | Expected: {exp} -> Got: {got}")
    else:
      print(f"MISMATCH: {k} | Expected: {exp} -> Got: {got}")
      has_mismatch = True

  return has_mismatch


def upload_to_gcs(report_data, gcs_dir):
  """Uploads the JSON report to the specified GCS directory."""
  if not gcs_dir:
    return

  if not gcs_dir.startswith("gs://"):
    print(f"GCS path must start with gs://, got: {gcs_dir}")
    return

  filename = f"shape_validation_report_{int(time.time())}.json"
  local_path = f"/tmp/{filename}"

  with open(local_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

  try:
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    # parse gs://bucket-name/path/to/dir
    gcs_dir_stripped = gcs_dir[5:]  # remove gs://
    parts = gcs_dir_stripped.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
      prefix += "/"

    blob_name = f"{prefix}{filename}"
    print(f"Uploading report to gs://{bucket_name}/{blob_name}...")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    print("Upload successful.")
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Failed to upload report to GCS: {e}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS dir to upload report")
  args = parser.parse_args()

  ideal_shapes = load_shapes("/tmp/ideal_shapes.txt")
  actual_shapes = load_shapes("/tmp/actual_shapes.txt")

  _has_mismatch = check_mismatches(ideal_shapes, actual_shapes)

  report = {
      "task": "checkpoint_shape_validation",
      "timestamp": time.time(),
      "status": "FAILURE" if _has_mismatch else "SUCCESS",
      "mismatches_found": _has_mismatch,
  }

  upload_to_gcs(report, args.report_gcs_dir)

  if _has_mismatch:
    print("\nERROR: Structural mismatches found!")
    sys.exit(1)

  print("\nSUCCESS: All parameters match perfectly.")
