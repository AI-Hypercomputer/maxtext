# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Queries GitHub Actions CI status and classifies blocking vs. non-blocking failures for MaxText PRs."""

import argparse
import json
import sys
import time
from typing import Any
import urllib.error
import urllib.request

API_BASE = "https://api.github.com/repos/AI-Hypercomputer/maxtext"
NON_BLOCKING_JOBS = {"Track Test Performance / Track Test Duration"}
TRANSIENT_HTTP_CODES = {500, 502, 503, 504}


def api_get(path: str, throttle_s: float = 1.5, max_retries: int = 3) -> Any:
  """Performs a throttled unauthenticated GitHub REST GET with rate-limit and 5xx retry handling."""
  url = path if path.startswith("http") else f"{API_BASE}{path}"
  req = urllib.request.Request(
      url,
      headers={"Accept": "application/vnd.github+json", "User-Agent": "maxtext-ci-status"},
  )
  for attempt in range(max_retries):
    time.sleep(throttle_s)
    try:
      with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
      if e.code == 403 and e.headers.get("x-ratelimit-remaining") == "0":
        reset = int(e.headers.get("x-ratelimit-reset", "0"))
        wait_s = max(1, reset - int(time.time()))
        if attempt + 1 < max_retries and wait_s <= 120:
          time.sleep(wait_s)
          continue
        raise RuntimeError(f"GitHub API rate limit exhausted; resets in {wait_s}s.") from e
      if e.code in TRANSIENT_HTTP_CODES and attempt + 1 < max_retries:
        time.sleep(2.0 ** (attempt + 1))
        continue
      raise
    except urllib.error.URLError:
      if attempt + 1 < max_retries:
        time.sleep(2.0 ** (attempt + 1))
        continue
      raise
  raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def classify_job_steps(job: dict[str, Any]) -> str:
  """Classifies a failed Actions job into test_failure, runner_interrupted, or setup_failure."""
  steps = job.get("steps") or []
  failed_steps = [s.get("name", "") for s in steps if s.get("conclusion") in ("failure", "cancelled")]
  upload_ok = any("Upload" in s.get("name", "") and s.get("conclusion") == "success" for s in steps)
  if any("Run Tests" in name or "Test" in name for name in failed_steps):
    return "test_failure" if upload_ok else "runner_interrupted"
  if failed_steps:
    return f"setup_failure({failed_steps[0]})"
  return job.get("conclusion") or "unknown"


def summarize_pr_ci(pr_num: int, throttle_s: float = 1.5) -> dict[str, Any]:
  """Fetches PR metadata, workflow runs on the head SHA, and blocking vs. non-blocking job status."""
  pr = api_get(f"/pulls/{pr_num}", throttle_s=throttle_s)
  head_sha = pr["head"]["sha"]
  commits = pr.get("commits", 0)
  labels = [lbl["name"] for lbl in pr.get("labels", [])]

  runs_resp = api_get(f"/actions/runs?head_sha={head_sha}&per_page=20", throttle_s=throttle_s)
  runs = runs_resp.get("workflow_runs", [])

  blocking_failures: list[str] = []
  non_blocking_failures: list[str] = []
  pending_runs = 0

  for run in runs:
    if run.get("event") not in ("pull_request", "pull_request_target"):
      continue
    status = run.get("status")
    conclusion = run.get("conclusion")
    if status != "completed":
      pending_runs += 1
      continue
    if conclusion in ("success", "skipped", "neutral"):
      continue

    jobs_resp = api_get(f"/actions/runs/{run['id']}/jobs?per_page=100", throttle_s=throttle_s)
    for job in jobs_resp.get("jobs", []):
      jname = job.get("name", "")
      jconc = job.get("conclusion")
      if jconc in ("success", "skipped", "neutral", None):
        continue
      full_name = f"{run.get('name')} / {jname}"
      if full_name in NON_BLOCKING_JOBS or jname in NON_BLOCKING_JOBS:
        non_blocking_failures.append(full_name)
      else:
        blocking_failures.append(f"{full_name} [{classify_job_steps(job)}]")

  if commits != 1:
    verdict = f"BLOCKED_COMMITS({commits})"
  elif blocking_failures:
    verdict = "FAILING"
  elif pending_runs > 0:
    verdict = f"PENDING({pending_runs})"
  else:
    verdict = "PULL_READY" if "pull ready" in labels else "GREEN_NEEDS_APPROVAL"

  return {
      "pr": pr_num,
      "head_sha": head_sha[:9],
      "commits": commits,
      "verdict": verdict,
      "blocking_failures": blocking_failures,
      "non_blocking_failures": non_blocking_failures,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("prs", nargs="+", type=int, help="PR numbers to inspect")
  parser.add_argument("--throttle", type=float, default=1.5, help="Seconds between API requests")
  args = parser.parse_args()

  for pr_num in args.prs:
    info = summarize_pr_ci(pr_num, throttle_s=args.throttle)
    print(f"PR #{info['pr']} ({info['head_sha']}, commits={info['commits']}): {info['verdict']}")
    for bf in info["blocking_failures"]:
      print(f"  [BLOCKING] {bf}")
    for nbf in info["non_blocking_failures"]:
      print(f"  [NON-BLOCKING] {nbf}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
