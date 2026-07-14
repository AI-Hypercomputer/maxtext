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

if __name__ == "__main__":
  ideal_shapes = load_shapes("/tmp/ideal_shapes.txt")
  actual_shapes = load_shapes("/tmp/actual_shapes.txt")

  if check_mismatches(ideal_shapes, actual_shapes):
    print("\nERROR: Structural mismatches found!")
    sys.exit(1)

  print("\nSUCCESS: All parameters match perfectly.")
