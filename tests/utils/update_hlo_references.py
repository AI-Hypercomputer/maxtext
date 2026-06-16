# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper script to dynamically recreate reference files for HLO validations checks rules.

This tool dynamically removes existing reference HLO files and executes the test suite
in order to recreate and validate them. The secure CI workflow `.github/workflows/update_reference_hlo.yml`
uses this script from an isolated test runner environment to bridge dynamic artifact
extractions setup logic and commit auto updates to PR.
"""

import os
import subprocess
import glob


def main():
  base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
  test_dir = os.path.join(base_dir, "tests/integration/")
  test_file = os.path.join(test_dir, "hlo_diff_test.py")

  reference_pattern = os.path.join(base_dir, "tests/utils/reference_hlo_*.txt")
  existing_files = glob.glob(reference_pattern)

  if existing_files:
    for reference_file in existing_files:
      print(f"Removing existing reference file: {reference_file}")
      os.remove(reference_file)
  else:
    print(f"No existing reference files found matching {reference_pattern}.")

  print(f"Running test suite to generate new references: {test_file}")

  env = os.environ.copy()
  env["PYTHONPATH"] = os.path.join(base_dir, "src/")

  result = subprocess.run(["pytest", test_file, "-v"], env=env, capture_output=True, text=True, check=False)

  print("STDOUT:", result.stdout)
  print("STDERR:", result.stderr)

  if result.returncode == 0:
    print("Reference files updated successfully.")
  else:
    print(f"Failed to update reference files. Test exited with code {result.returncode}")


if __name__ == "__main__":
  main()
