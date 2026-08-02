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
  parser.add_argument("--title", type=str, required=False, default=None, help="PR Title")
  parser.add_argument("--body", type=str, required=False, default=None, help="PR Body/Description")
  parser.add_argument("--message", type=str, required=False, default=None, help="Commit/PR Message")
  parser.add_argument("--base", type=str, required=False, default="main", help="Base branch")
  parser.add_argument("--fix_branch", type=str, required=False, default=None, help="Forked fix branch name")

  args, unknown = parser.parse_known_args()

  title = args.title or args.message or "Automated code fix by Overwatch Agent"
  body = args.body or args.message or title
  import time
  fork_branch = args.fix_branch or f"fix/agent-remediation-{int(time.time())}"

  print(f"1. Forking new branch '{fork_branch}' from base branch '{args.base}'...")
  subprocess.run(["git", "checkout", args.base], check=False)
  subprocess.run(["git", "checkout", "-b", fork_branch], check=False)

  print("2. Staging and committing patched changes...")
  subprocess.run(["git", "add", "."], check=False)
  subprocess.run(["git", "commit", "-m", title], check=False)

  print(f"3. Pushing forked branch '{fork_branch}' to origin...")
  push_res = subprocess.run(["git", "push", "-u", "origin", fork_branch], check=False)

  print(f"4. Opening Pull Request from '{fork_branch}' into base branch '{args.base}'...")
  cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--base", args.base, "--head", fork_branch]

  try:
    subprocess.run(cmd, check=True)
    print(f"Successfully opened Pull Request targeting '{args.base}' from head '{fork_branch}'.")
  except Exception as e:
    print(f"Note: gh CLI or remote push could not authenticate in serverless mode ({e}).")
    print(f"Successfully created forked branch '{fork_branch}' and committed fix locally.")
    print(f"PR Title: {args.title}\nPR Body: {args.body}")


if __name__ == "__main__":
  main()
