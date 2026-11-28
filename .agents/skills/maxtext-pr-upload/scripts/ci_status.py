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

"""Queries GitHub Actions CI status and classifies blocking vs.

non-blocking failures for MaxText PRs.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_REPO = "AI-Hypercomputer/maxtext"
API_BASE = f"https://api.github.com/repos/{DEFAULT_REPO}"
NON_BLOCKING_JOBS = {"Track Test Performance / Track Test Duration"}
STANDALONE_REQUIRED_CHECKS = {
    "Require PR Checklist",
    "cla/google",
    "All Required Tests Passed",
}
TRANSIENT_HTTP_CODES = {500, 502, 503, 504}


def _default_github_repo() -> str:
  """Infers owner/repo from git remote origin, defaulting to AI-Hypercomputer/maxtext."""
  try:
    proc = subprocess.run(
        ["git", "-c", "safe.directory=*", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode == 0:
      m = re.search(
          r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?\s*$",
          proc.stdout,
      )
      if m:
        return m.group(1)
  except OSError:
    pass
  return DEFAULT_REPO


def api_get(
    path: str,
    throttle_s: float = 1.5,
    max_retries: int = 3,
    repo: str = DEFAULT_REPO,
) -> Any:
  """Performs a throttled unauthenticated GitHub REST GET with rate-limit and 5xx retry handling."""
  base_url = f"https://api.github.com/repos/{repo}" if repo else API_BASE
  url = path if path.startswith("http") else f"{base_url}{path}"
  req = urllib.request.Request(
      url,
      headers={
          "Accept": "application/vnd.github+json",
          "User-Agent": "maxtext-ci-status",
      },
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


def api_get_paginated_list(
    path_prefix: str,
    list_key: str | None,
    throttle_s: float = 1.5,
    max_pages: int = 10,
    *,
    page_size: int = 100,
    repo: str = DEFAULT_REPO,
) -> list[dict[str, Any]]:
  """Fetches up to max_pages of paginated GitHub REST items."""
  out: list[dict[str, Any]] = []
  sep = "&" if "?" in path_prefix else "?"
  for page in range(1, max_pages + 1):
    resp = api_get(
        f"{path_prefix}{sep}per_page={page_size}&page={page}",
        throttle_s=throttle_s,
        repo=repo,
    )
    if list_key is not None:
      items = resp.get(list_key, []) if isinstance(resp, dict) else []
    else:
      items = resp if isinstance(resp, list) else []
    if not items:
      break
    out.extend(items)
    if len(items) < page_size:
      break
  return out


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


def summarize_pr_ci(
    pr_num: int,
    throttle_s: float = 1.5,
    repo: str = DEFAULT_REPO,
) -> dict[str, Any]:
  """Fetches PR metadata, workflow runs on the head SHA, and blocking vs.

  non-blocking job status.
  """
  pr = api_get(f"/pulls/{pr_num}", throttle_s=throttle_s, repo=repo)
  head_sha = pr["head"]["sha"]
  commits = pr.get("commits", 0)
  labels = [lbl["name"] for lbl in pr.get("labels", [])]

  all_runs = api_get_paginated_list(
      f"/actions/runs?head_sha={head_sha}",
      "workflow_runs",
      throttle_s=throttle_s,
      repo=repo,
  )
  latest_runs: dict[tuple[Any, Any], dict[str, Any]] = {}
  for run in all_runs:
    key = (run.get("name"), run.get("event"))
    prev = latest_runs.get(key)
    if prev is None or (run.get("run_number") or 0, run.get("id") or 0) > (
        prev.get("run_number") or 0,
        prev.get("id") or 0,
    ):
      latest_runs[key] = run
  runs = list(latest_runs.values())

  blocking_failures: list[str] = []
  non_blocking_failures: list[str] = []
  pending_runs = 0
  seen_job_names: set[str] = set()

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

    attempt = run.get("run_attempt")
    jobs_path = (
        f"/actions/runs/{run['id']}/attempts/{attempt}/jobs"
        if isinstance(attempt, int) and attempt >= 1
        else f"/actions/runs/{run['id']}/jobs"
    )
    jobs = api_get_paginated_list(
        jobs_path,
        "jobs",
        throttle_s=throttle_s,
        page_size=25,
        repo=repo,
    )
    for job in jobs:
      jname = job.get("name", "")
      seen_job_names.add(jname)
      jconc = job.get("conclusion")
      if jconc in ("success", "skipped", "neutral", None):
        continue
      full_name = f"{run.get('name')} / {jname}"
      if full_name in NON_BLOCKING_JOBS or jname in NON_BLOCKING_JOBS:
        non_blocking_failures.append(full_name)
      else:
        blocking_failures.append(f"{full_name} [{classify_job_steps(job)}]")

  check_runs = api_get_paginated_list(
      f"/commits/{head_sha}/check-runs",
      "check_runs",
      throttle_s=throttle_s,
      repo=repo,
  )
  latest_checks: dict[str, dict[str, Any]] = {}
  for cr in check_runs:
    cname = cr.get("name", "")
    if not cname:
      continue
    prev = latest_checks.get(cname)
    if prev is None or (cr.get("id") or 0) > (prev.get("id") or 0):
      latest_checks[cname] = cr

  for cname in sorted(STANDALONE_REQUIRED_CHECKS):
    cr = latest_checks.get(cname)
    if not cr:
      continue
    if cr.get("status") != "completed":
      if cname not in seen_job_names and not any(r.get("name") == cname and r.get("status") != "completed" for r in runs):
        pending_runs += 1
      continue
    cconc = cr.get("conclusion")
    if cconc in ("success", "skipped", "neutral", None):
      continue
    if not any(f" / {cname} [" in bf or bf.startswith(f"{cname} [") for bf in blocking_failures):
      blocking_failures.append(f"{cname} [{cconc}]")

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
  parser.add_argument(
      "--repo",
      default=_default_github_repo(),
      help="GitHub owner/repo (default: origin remote or AI-Hypercomputer/maxtext)",
  )
  parser.add_argument("--throttle", type=float, default=1.5, help="Seconds between API requests")
  args = parser.parse_args()

  for pr_num in args.prs:
    info = summarize_pr_ci(pr_num, throttle_s=args.throttle, repo=args.repo)
    print(f"PR #{info['pr']} ({info['head_sha']}, commits={info['commits']}):" f" {info['verdict']}")
    for bf in info["blocking_failures"]:
      print(f"  [BLOCKING] {bf}")
    for nbf in info["non_blocking_failures"]:
      print(f"  [NON-BLOCKING] {nbf}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
