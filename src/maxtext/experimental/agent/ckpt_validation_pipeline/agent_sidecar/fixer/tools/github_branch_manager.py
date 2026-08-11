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
    import os

    repo_dir = "/tmp/maxtext_repo"
    base_branch = "main"

    if not os.path.exists(repo_dir):
      print(f"Cloning {base_branch} into {repo_dir}...")
      subprocess.run(
          ["git", "clone", "-b", base_branch, "https://github.com/AI-Hypercomputer/maxtext.git", repo_dir], check=True
      )

    subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, check=False)
    if args.action == "create":
      # Check out the base branch first so we branch off the correct code
      res = subprocess.run(["git", "checkout", base_branch], cwd=repo_dir, check=False)
      if res.returncode != 0:
        subprocess.run(["git", "checkout", "-b", base_branch, f"origin/{base_branch}"], cwd=repo_dir, check=True)
      subprocess.run(["git", "checkout", "-b", args.branch], cwd=repo_dir, check=True)
    elif args.action == "checkout":
      # Checkout local branch or track remote branch
      res = subprocess.run(["git", "checkout", args.branch], cwd=repo_dir, check=False)
      if res.returncode != 0:
        subprocess.run(["git", "checkout", "-b", args.branch, f"origin/{args.branch}"], cwd=repo_dir, check=True)
    elif args.action == "delete":
      subprocess.run(["git", "branch", "-D", args.branch], cwd=repo_dir, check=True)
    print(f"Successfully executed {args.action} for branch {args.branch}")
  except subprocess.CalledProcessError as e:
    print(f"Failed to execute git command. Error: {e}")
    sys.exit(e.returncode)


if __name__ == "__main__":
  main()
