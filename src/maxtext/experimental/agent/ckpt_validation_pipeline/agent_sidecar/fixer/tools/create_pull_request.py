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

"""Tool for the Verifier agent to safely open Pull Requests."""

import argparse
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description="Create a Pull Request using GitHub CLI.")
  parser.add_argument("--title", type=str, required=True, help="PR Title")
  parser.add_argument("--body", type=str, required=True, help="PR Body/Description")
  parser.add_argument("--base", type=str, required=True, help="Base branch (e.g., main or maxtext_branch)")

  args = parser.parse_args()

  cmd = ["gh", "pr", "create", "--title", args.title, "--body", args.body, "--base", args.base]

  try:
    subprocess.run(cmd, check=True)
    print(f"Successfully opened Pull Request targeting {args.base}.")
  except subprocess.CalledProcessError as e:
    print(f"Failed to create PR using gh CLI. Error: {e}")
    sys.exit(e.returncode)


if __name__ == "__main__":
  main()
