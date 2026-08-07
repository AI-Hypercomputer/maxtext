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

"""Helper script for the Fixer agent to locally run shape inspections."""

import argparse
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description="Run inspect_checkpoint locally.")
  parser.add_argument("--mode", type=str, required=False, default="maxtext", choices=["hf", "maxtext", "orbax"])
  parser.add_argument("--model", type=str, required=False, default=None, help="MaxText model name")
  parser.add_argument("--run_id", type=str, required=False, default=None, help="Run ID")
  args, unknown = parser.parse_known_args()

  import os

  script_path = "/app/src/maxtext/checkpoint_conversion/inspect_checkpoint.py"
  if not os.path.exists(script_path):
    script_path = "src/maxtext/checkpoint_conversion/inspect_checkpoint.py"

  cmd = ["python3", script_path, args.mode]
  if args.model:
    cmd.append(f"model_name={args.model}")
  cmd.extend(unknown)

  try:
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as e:
    print(f"Error: inspect_checkpoint.py failed with return code {e.returncode}")
    sys.exit(e.returncode)


if __name__ == "__main__":
  main()
