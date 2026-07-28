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

"""Tool for the Fixer agent to safely manage GitHub branches."""

import argparse
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description="Manage GitHub branches.")
  parser.add_argument("--action", choices=["create", "checkout", "delete"], required=True)
  parser.add_argument("--branch", type=str, required=True, help="Branch name")

  args = parser.parse_args()

  cmd = []
  if args.action == "create":
    # Create and checkout
    cmd = ["git", "checkout", "-b", args.branch]
  elif args.action == "checkout":
    cmd = ["git", "checkout", args.branch]
  elif args.action == "delete":
    cmd = ["git", "branch", "-D", args.branch]

  try:
    subprocess.run(cmd, check=True)
    print(f"Successfully executed {args.action} for branch {args.branch}")
  except subprocess.CalledProcessError as e:
    print(f"Failed to execute git command. Error: {e}")
    sys.exit(e.returncode)


if __name__ == "__main__":
  main()
