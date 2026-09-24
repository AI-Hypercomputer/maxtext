# Copyright 2026 Google LLC
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

"""Tool for the Fixer agent to run pyink and pylint safely."""

import argparse
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description="Run linters (pyink and pylint) on a target file.")
  parser.add_argument("--file", type=str, required=True, help="Target Python file")

  args = parser.parse_args()
  file_path = args.file

  # Run Pyink
  print(f"Running pyink on {file_path}...")
  try:
    subprocess.run(["pyink", "--pyink-indentation=2", "--line-length=122", file_path], check=True)
    print("pyink completed successfully.")
  except FileNotFoundError:
    print("Note: pyink not installed in this environment. Skipping formatting check.")
  except subprocess.CalledProcessError as e:
    print(f"pyink failed with error code {e.returncode}")
    sys.exit(e.returncode)

  # Run Pylint
  print(f"Running pylint on {file_path}...")
  try:
    res = subprocess.run(["pylint", file_path], check=False)
    # pylint exit codes: 1=Fatal, 2=Error, 4=Warning, 8=Refactor, 16=Convention
    if res.returncode & 1 or res.returncode & 2 or res.returncode & 32:
      print(f"pylint failed with fatal/error code {res.returncode}")
      sys.exit(res.returncode)
    else:
      print(f"pylint completed (no syntax/fatal errors, code {res.returncode}).")
  except FileNotFoundError:
    print("Note: pylint not installed in this environment. Skipping lint check.")


if __name__ == "__main__":
  main()
