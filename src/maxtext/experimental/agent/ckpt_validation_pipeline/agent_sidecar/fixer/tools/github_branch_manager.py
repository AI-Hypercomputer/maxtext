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

  try:
    subprocess.run(["git", "fetch", "origin"], check=False)
    if args.action == "create":
      # Create branch based on HEAD or origin/main if available
      subprocess.run(["git", "checkout", "-b", args.branch], check=True)
    elif args.action == "checkout":
      # Checkout local branch or track remote branch
      res = subprocess.run(["git", "checkout", args.branch], check=False)
      if res.returncode != 0:
        subprocess.run(["git", "checkout", "-b", args.branch, f"origin/{args.branch}"], check=True)
    elif args.action == "delete":
      subprocess.run(["git", "branch", "-D", args.branch], check=True)
    print(f"Successfully executed {args.action} for branch {args.branch}")
  except subprocess.CalledProcessError as e:
    print(f"Failed to execute git command. Error: {e}")
    sys.exit(e.returncode)


if __name__ == "__main__":
  main()
