"""Collector prototype for the CI Pulse dashboard.

Reads workflow runs, jobs and pull requests of the "MaxText Package Tests" pipeline
from the GitHub API and writes the per-run records the static site is built from.
"""

import argparse
import requests
import os
import re
from datetime import datetime
from collections import defaultdict

# Setup
OWNER = "AI-Hypercomputer"
REPO = "maxtext"
TARGET_WORKFLOW = "MaxText Package Tests"  # Display name only; also the fallback matcher if the path lookup fails
TARGET_WORKFLOW_PATH = ".github/workflows/ci_pipeline.yml"
MAX_PUSH_COMMITS = 30  # Cap on how many of the PR's most recent commit shas are swept for workflow runs

# Configure headers and safely load the GitHub token
HEADERS = {"Accept": "application/vnd.github.v3+json"}
github_token = os.environ.get("GITHUB_TOKEN")

if github_token:
  HEADERS["Authorization"] = f"Bearer {github_token}"
else:
  print(
      "WARNING: GITHUB_TOKEN environment variable not set. You will quickly hit the API rate limit of 60 requests/hour.\n"
  )

# Module-level cache for the workflow id resolved from TARGET_WORKFLOW_PATH
_target_workflow_id = None
_target_workflow_id_resolved = False


def get_target_workflow_id():
  """Resolves TARGET_WORKFLOW_PATH to its numeric workflow id (cached; returns None if the lookup fails)."""
  global _target_workflow_id, _target_workflow_id_resolved
  if _target_workflow_id_resolved:
    return _target_workflow_id
  _target_workflow_id_resolved = True
  try:
    page = 1
    while True:
      url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows?per_page=100&page={page}"
      response = requests.get(url, headers=HEADERS)
      response.raise_for_status()
      workflows = response.json().get("workflows", [])
      if not workflows:
        break
      for workflow in workflows:
        if workflow.get("path") == TARGET_WORKFLOW_PATH:
          _target_workflow_id = workflow["id"]
          return _target_workflow_id
      page += 1
  except requests.RequestException:
    pass
  print(
      f"WARNING: Could not resolve '{TARGET_WORKFLOW_PATH}' to a workflow id. "
      f"Falling back to matching runs by name '{TARGET_WORKFLOW}'."
  )
  return _target_workflow_id


def filter_target_runs(runs):
  """Keeps only runs of the target workflow, matching by workflow id (falling back to the display name)."""
  workflow_id = get_target_workflow_id()
  if workflow_id is not None:
    return [r for r in runs if r.get("workflow_id") == workflow_id]
  return [r for r in runs if r.get("name") == TARGET_WORKFLOW]


def clean_job_name(name):
  """Shortens the first segment by extracting text in () and removes redundant trailing elements."""
  if not name:
    return "Unknown Job"

  parts = [p.strip() for p in name.split("/")]

  # 1. Shorten the first part if it matches the pattern "Some Text (actual-name)".
  #    Skip bare numbers like "(2)" (matrix shard indexes), which would produce names like "2 / tpu-pathways-unit".
  if parts:
    match = re.search(r"\(([^)]+)\)", parts[0])
    if match and re.search(r"\D", match.group(1)):
      parts[0] = match.group(1)

  if len(parts) > 1:
    last_part = parts[-1]
    prefix = " / ".join(parts[:-1])
    # 2. If the last segment is already mentioned earlier in the name, remove it
    if last_part in prefix:
      return prefix
    else:
      return " / ".join(parts)

  return parts[0]


def get_merged_prs(limit=None, target_date_str=None):
  """Fetches merged PRs, optionally filtering strictly by a target_date (YYYY-MM-DD) or a count limit."""
  prs = []
  page = 1

  if target_date_str:
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
  else:
    target_date = None

  while True:
    url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls"
        f"?state=closed&sort=updated&direction=desc&per_page=100&page={page}"
    )
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    page_prs = response.json()

    if not page_prs:
      break

    for pr in page_prs:
      if target_date:
        updated_at_date = datetime.strptime(pr["updated_at"], "%Y-%m-%dT%H:%M:%SZ").date()
        if updated_at_date < target_date:
          return prs[:limit] if limit else prs

      if pr.get("merged_at"):
        if target_date:
          merged_at_date = datetime.strptime(pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ").date()
          if merged_at_date == target_date:
            prs.append(pr)
        else:
          prs.append(pr)

        if limit and not target_date and len(prs) >= limit:
          return prs
    page += 1

  return prs[:limit] if limit else prs


def get_pr(pr_number):
  """Fetches a specific PR by its number."""
  url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}"
  response = requests.get(url, headers=HEADERS)
  response.raise_for_status()
  return response.json()


def get_pr_commit_shas(pr_number):
  """Fetches the shas of all commits on a PR, oldest first (paginated)."""
  shas = []
  page = 1
  while True:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}/commits?per_page=100&page={page}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    page_commits = response.json()
    if not page_commits:
      break
    shas.extend(commit["sha"] for commit in page_commits)
    if len(page_commits) < 100:
      break
    page += 1
  return shas


def get_workflow_runs(commit_sha):
  """Fetches all GitHub Actions workflow runs for a specific commit (paginated)."""
  runs = []
  page = 1
  while True:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs?head_sha={commit_sha}&per_page=100&page={page}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    page_runs = response.json().get("workflow_runs", [])
    if not page_runs:
      break
    runs.extend(page_runs)
    if len(page_runs) < 100:
      break
    page += 1
  return runs


def get_jobs_for_attempt(run_id, attempt_number):
  """Fetches jobs for a specific workflow attempt (handles re-runs, paginated)."""
  jobs = []
  page = 1
  while True:
    url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}"
        f"/attempts/{attempt_number}/jobs?per_page=100&page={page}"
    )
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    page_jobs = response.json().get("jobs", [])
    if not page_jobs:
      break
    jobs.extend(page_jobs)
    if len(page_jobs) < 100:
      break
    page += 1
  return jobs


def print_pushes_summary(runs_by_sha):
  """Prints the cross-push summary using approximate run-level durations (updated_at - run_started_at)."""
  all_runs = [run for sha_runs in runs_by_sha.values() for run in sha_runs]
  approx_compute_seconds = 0
  for run in all_runs:
    started_str = run.get("run_started_at")
    updated_str = run.get("updated_at")
    if started_str and updated_str:
      started = datetime.strptime(started_str, "%Y-%m-%dT%H:%M:%SZ")
      updated = datetime.strptime(updated_str, "%Y-%m-%dT%H:%M:%SZ")
      approx_compute_seconds += max(0, (updated - started).total_seconds())
  print(
      f"\nAcross all pushes: {len(runs_by_sha)} pushes analyzed, {len(all_runs)} workflow runs, "
      f"total compute ~{approx_compute_seconds / 60:.2f} min (approximate, from run-level durations)"
  )


def main():
  parser = argparse.ArgumentParser(description="Fetch CI statistics for merged MaxText PRs.")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--pr", type=int, help="Fetch statistics for a specific PR number.")
  group.add_argument("--limit", type=int, help="Number of latest merged PRs to fetch.")
  group.add_argument("--date", type=str, help="Fetch all merged PRs exactly on this date (YYYY-MM-DD).")
  args = parser.parse_args()

  if args.pr:
    print(f"Fetching data for PR #{args.pr}...")
    prs = [get_pr(args.pr)]
  elif args.date:
    print(f"Fetching all merged PRs on {args.date}...")
    prs = get_merged_prs(target_date_str=args.date)
  else:
    limit = args.limit or 5
    print(f"Fetching data for the latest {limit} merged PRs...")
    prs = get_merged_prs(limit=limit)

  if not prs:
    print("No PRs found matching your criteria.")
    return

  print("\n" + "=" * 100 + "\n")

  for pr in prs:
    pr_number = pr["number"]
    head_sha = pr["head"]["sha"]

    pr_status = "Merged" if pr.get("merged_at") else pr.get("state", "unknown").capitalize()

    print(f"--- PR #{pr_number}: {pr['title']} ---")
    print(f"Status: {pr_status}")
    if pr.get("merged_at"):
      print(f"Merged at: {pr['merged_at']}")

    # Sweep every push of the PR: collect the target workflow's runs for each commit sha
    commit_shas = get_pr_commit_shas(pr_number)
    if len(commit_shas) > MAX_PUSH_COMMITS:
      print(f"Note: PR has {len(commit_shas)} commits; sweeping only the {MAX_PUSH_COMMITS} most recent.")
      commit_shas = commit_shas[-MAX_PUSH_COMMITS:]
    if head_sha not in commit_shas:
      commit_shas.append(head_sha)

    runs_by_sha = {}
    for sha in commit_shas:
      sha_runs = filter_target_runs(get_workflow_runs(sha))
      if sha_runs:
        runs_by_sha[sha] = sha_runs

    if not runs_by_sha:
      print(f"No '{TARGET_WORKFLOW}' workflow runs found on any commit of this PR.")
      print("\n" + "=" * 100 + "\n")
      continue

    target_runs = runs_by_sha.get(head_sha, [])
    if not target_runs:
      print(f"No '{TARGET_WORKFLOW}' workflow runs found for the head commit; skipping detailed job analysis.")
      print_pushes_summary(runs_by_sha)
      print("\n" + "=" * 100 + "\n")
      continue

    latest_run = target_runs[0]
    run_id = latest_run["id"]
    total_attempts = latest_run.get("run_attempt", 1)

    attempt_1_jobs = get_jobs_for_attempt(run_id, 1)

    total_checks = len(attempt_1_jobs)
    total_compute_seconds = 0
    failed_checks_attempt_1 = []
    cancelled_checks_attempt_1 = []
    start_times = []
    end_times = []
    wait_times = []
    run_times = []

    for job in attempt_1_jobs:
      raw_job_name = job.get("name")
      job_name = clean_job_name(raw_job_name)

      # Extract Machine Type (filtering out generic 'self-hosted' label if possible)
      labels = job.get("labels") or []
      machine_labels = [lbl for lbl in labels if lbl != "self-hosted"]
      machine_type = ", ".join(machine_labels) if machine_labels else ", ".join(labels)
      if not machine_type:
        machine_type = "Unknown"

      if job.get("started_at"):
        start_times.append(datetime.strptime(job.get("started_at"), "%Y-%m-%dT%H:%M:%SZ"))
      if job.get("completed_at"):
        end_times.append(datetime.strptime(job.get("completed_at"), "%Y-%m-%dT%H:%M:%SZ"))

      queued_str = job.get("queued_at") or job.get("created_at")
      started_str = job.get("started_at")

      if queued_str and started_str:
        queued = datetime.strptime(queued_str, "%Y-%m-%dT%H:%M:%SZ")
        started = datetime.strptime(started_str, "%Y-%m-%dT%H:%M:%SZ")
        wait_seconds = max(0, (started - queued).total_seconds())
        wait_times.append({"name": job_name, "wait_seconds": wait_seconds, "machine_type": machine_type})

      if job.get("started_at") and job.get("completed_at"):
        start = datetime.strptime(job.get("started_at"), "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.strptime(job.get("completed_at"), "%Y-%m-%dT%H:%M:%SZ")
        # Skipped/instant jobs can report completed_at slightly before started_at
        run_seconds = max(0, (end - start).total_seconds())
        total_compute_seconds += run_seconds

        run_times.append({"name": job_name, "run_seconds": run_seconds})

      conclusion = job.get("conclusion")
      if conclusion == "cancelled":
        # Cancelled jobs are usually superseded (newer push) or collateral of another failure:
        # count them separately instead of reporting them as failures.
        cancelled_checks_attempt_1.append(job_name)
      elif conclusion in ["failure", "timed_out", "action_required"]:
        fail_reason = "Unknown failure"
        for step in job.get("steps", []):
          if step.get("conclusion") in ["failure", "cancelled"]:
            fail_reason = f"Step '{step.get('name')}' failed."
            break
        failed_checks_attempt_1.append(
            {"name": job_name, "raw_name": raw_job_name, "reason": fail_reason, "conclusion": conclusion}
        )

    wall_clock_seconds = 0
    earliest_start = None
    latest_end = None

    if start_times and end_times:
      earliest_start = min(start_times)
      latest_end = max(end_times)
      if latest_end > earliest_start:
        wall_clock_seconds = (latest_end - earliest_start).total_seconds()

    if wait_times:
      avg_wait = sum(w["wait_seconds"] for w in wait_times) / len(wait_times)
      sorted_waits = sorted(wait_times, key=lambda x: x["wait_seconds"], reverse=True)
      max_wait = sorted_waits[0]["wait_seconds"]
      longest_waits = sorted_waits[:3]
    else:
      avg_wait = max_wait = 0
      longest_waits = []

    if run_times:
      avg_run = sum(r["run_seconds"] for r in run_times) / len(run_times)
      sorted_runs = sorted(run_times, key=lambda x: x["run_seconds"], reverse=True)
      max_run = sorted_runs[0]["run_seconds"]
    else:
      avg_run = max_run = 0
      sorted_runs = []

    # Cross-attempt matching keys on the RAW job name (stable across attempts);
    # names are shortened with clean_job_name() only when printed.
    flaky_history = defaultdict(list)
    if total_attempts > 1 and failed_checks_attempt_1:
      for fail in failed_checks_attempt_1:
        flaky_history[fail["raw_name"]].append(f"Attempt 1: {fail['conclusion']}")

      for attempt in range(2, total_attempts + 1):
        attempt_jobs = get_jobs_for_attempt(run_id, attempt)
        for job in attempt_jobs:
          raw_job_name = job.get("name")
          if raw_job_name in flaky_history:
            flaky_history[raw_job_name].append(f"Attempt {attempt}: {job.get('conclusion')}")

    flaky_successes = {
        name: history for name, history in flaky_history.items() if any("success" in step for step in history)
    }

    print(f"Target Workflow: {TARGET_WORKFLOW}")
    print(f"Total Attempts (Re-runs): {total_attempts}")

    print("\n--- Attempt 1 Statistics ---")
    print(f"Total Checks: {total_checks}")

    if earliest_start and latest_end:
      print(f"First Job Started: {earliest_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
      print(f"Last Job Ended:    {latest_end.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    print(f"Wall-Clock Time: {wall_clock_seconds / 60:.2f} minutes")
    print(f"Total Compute Time: {total_compute_seconds / 60:.2f} minutes")

    print(f"\nAverage Job Run Time: {avg_run / 60:.2f} minutes")
    print(f"Max Job Run Time: {max_run / 60:.2f} minutes")

    print(f"\nAverage Runner Wait Time: {avg_wait / 60:.2f} minutes")
    print(f"Max Runner Wait Time: {max_wait / 60:.2f} minutes")

    if longest_waits:
      print("\nLongest Waiting Jobs (Top 3):")
      print(f"{'Job Name':<75} | {'Wait (min)':>10} | {'Machine Type'}")
      print("-" * 115)
      for w in longest_waits:
        print(f"{w['name']:<75} | {w['wait_seconds'] / 60:>10.2f} | {w['machine_type']}")

    if sorted_runs:
      print("\nAll Jobs Running Times (Longest to Shortest):")
      print(f"{'Job Name':<75} | {'Run (min)':>10}")
      print("-" * 88)
      for r in sorted_runs:
        print(f"{r['name']:<75} | {r['run_seconds'] / 60:>10.2f}")

    print(f"\nInitial Failed Checks: {len(failed_checks_attempt_1)}")
    for fail in failed_checks_attempt_1:
      print(f"  -> {fail['name']}: {fail['reason']}")

    print(f"Cancelled/superseded: {len(cancelled_checks_attempt_1)} (excluded from failure stats)")
    for name in cancelled_checks_attempt_1:
      print(f"  -> {name}")

    if flaky_successes:
      print("\nFlakiness / Re-run Success:")
      for name, history in flaky_successes.items():
        history_str = " -> ".join(history)
        print(f"  * {clean_job_name(name)}")
        print(f"      {history_str}")

    print_pushes_summary(runs_by_sha)

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
  main()
