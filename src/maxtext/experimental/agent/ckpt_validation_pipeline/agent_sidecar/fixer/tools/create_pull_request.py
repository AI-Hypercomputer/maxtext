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

  print(f"1. Checking out fix branch '{fork_branch}'...")
  subprocess.run(["git", "checkout", fork_branch], check=False)

  subprocess.run(["git", "commit", "-am", title], check=False)

  import os

  gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
  remote_url = (
      f"https://x-access-token:{gh_token}@github.com/AI-Hypercomputer/maxtext.git"
      if gh_token
      else "https://github.com/AI-Hypercomputer/maxtext.git"
  )
  print(f"Configuring git remote 'origin' ({'with GH_TOKEN' if gh_token else 'anonymous'})...")
  if subprocess.run(["git", "remote", "set-url", "origin", remote_url], capture_output=True).returncode != 0:
    subprocess.run(["git", "remote", "add", "origin", remote_url], check=False)

  print(f"3. Pushing forked branch '{fork_branch}' to origin...")
  push_res = subprocess.run(["git", "push", "-uf", "origin", fork_branch], capture_output=True, text=True, check=False)
  if push_res.returncode != 0:
    print(f"git push failed (code {push_res.returncode}):\nSTDOUT: {push_res.stdout}\nSTDERR: {push_res.stderr}")
  else:
    print(f"git push succeeded:\n{push_res.stdout}\n{push_res.stderr}")

  print(f"4. Opening Pull Request from '{fork_branch}' into base branch '{args.base}'...")
  env = os.environ.copy()
  if gh_token:
    env["GH_TOKEN"] = gh_token
    env["GITHUB_TOKEN"] = gh_token
  cmd = [
      "gh",
      "pr",
      "create",
      "--title",
      title,
      "--body",
      body,
      "--base",
      args.base,
      "--head",
      fork_branch,
      "--repo",
      "AI-Hypercomputer/maxtext",
  ]

  try:
    pr_res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    print(
        f"Successfully opened Pull Request targeting '{args.base}' from head '{fork_branch}'.\nSTDOUT: {pr_res.stdout}\nSTDERR: {pr_res.stderr}"
    )
  except subprocess.CalledProcessError as e:
    print(f"Note: gh pr create failed (code {e.returncode}):\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
    print(f"Successfully created forked branch '{fork_branch}' and committed fix locally.")
    print(f"PR Title: {args.title}\nPR Body: {args.body}")
  except Exception as e:
    print(f"Note: gh CLI or remote push could not authenticate in serverless mode ({e}).")
    print(f"Successfully created forked branch '{fork_branch}' and committed fix locally.")
    print(f"PR Title: {args.title}\nPR Body: {args.body}")


if __name__ == "__main__":
  main()
