# collect.py

Here is a complete guide to the `tools/ci_metrics/collect.py` script, including a detailed explanation of its mechanics, how to run it, and what the output looks like.

### 1. Detailed Explanation

The script is a specialized CI (Continuous Integration) analytics tool designed to interact with the GitHub API. It analyzes the **"MaxText Package Tests"** workflow (defined by `.github/workflows/ci_pipeline.yml`) in the `AI-Hypercomputer/maxtext` repository.

Here is how it works under the hood:
*   **Targeting PRs:** It fetches Pull Requests based on your criteria (a specific PR number, a limit of recent merged PRs, or a specific merge date). It safely paginates through GitHub's API by sorting PRs by their `updated_at` timestamp. It only fetches `merged` PRs.
*   **Workflow Isolation by ID:** The script resolves `.github/workflows/ci_pipeline.yml` to its numeric workflow id once per process (via `GET /repos/{owner}/{repo}/actions/workflows`, cached in a module variable) and filters runs by `workflow_id`. This is robust against display-name renames and against unrelated workflows sharing a similar name. The display name `"MaxText Package Tests"` is used only for output — and as a fallback matcher if the path lookup fails.
*   **Sweeping All Pushes:** Instead of looking only at the last commit, the script fetches every commit sha of the PR (`GET /pulls/{n}/commits`, paginated with `per_page=100`) and collects the target workflow's runs for each sha (also paginated with `per_page=100`). The sweep is capped at the **30 most recent shas**; a note is printed if the PR has more commits than that.
*   **Attempt 1 Baseline:** The detailed per-job analysis still isolates **Attempt 1** of the **latest run on the PR's head commit**. This prevents manual re-runs from artificially inflating the total compute time or skewing the wait times.
*   **Cross-Push Summary:** Each PR's report ends with a line of the form `Across all pushes: N pushes analyzed, M workflow runs, total compute ~X min`. Summing per-job durations across every run would be too expensive on the API, so this total is an **approximation**: it sums run-level durations (`updated_at - run_started_at`) and is labeled as approximate in the output.
*   **Time Calculations:**
    *   **Wait Time:** Calculated as the difference between when a job was queued (`queued_at` or `created_at`) and when the runner actually picked it up (`started_at`).
    *   **Run Time:** Calculated as the difference between `started_at` and `completed_at`.
    *   **Wall-Clock Time:** Calculates the true end-to-end duration by finding the absolute earliest start time and absolute latest end time across the entire suite.
*   **Cancelled vs Failed:** Jobs (or runs) that conclude as `cancelled` — typically superseded by a newer push, or collateral of a sibling failure — are **not** counted in "Initial Failed Checks". They are reported separately as `Cancelled/superseded: N (excluded from failure stats)`. Only `failure`, `timed_out`, and `action_required` count as failures.
*   **Job Name Cleaning:** It uses regular expressions to shorten matrix job names. For example, it automatically turns `TPU Posttrain Tests (tpu-post-training-unit) / Execute Tests (1) / tpu-post-training-unit` into `tpu-post-training-unit / Execute Tests (1)`. The parenthetical substitution is skipped when the captured text is a bare number (a matrix shard index): `TPU Pathways Unit Tests (2) / tpu-pathways-unit` stays as-is instead of becoming the nonsensical `2 / tpu-pathways-unit`.
*   **Runner Extraction:** It inspects the `labels` array provided by GitHub to identify the required machine type (e.g., `linux-x86-ct6e-180-4tpu`), filtering out generic labels like `"self-hosted"`.
*   **Flakiness Detection:** If the workflow has multiple attempts (re-runs), the script compares the failed jobs from Attempt 1 against the later attempts. If a job failed initially but passed on a subsequent attempt, it flags it under "Flakiness / Re-run Success". Because job ids differ between attempts, cross-attempt matching keys on the **raw** (uncleaned) job name, which is stable across attempts; `clean_job_name()` is applied only at print time. This prevents two distinct jobs that shorten to the same string from having their histories merged.

---

### 2. Usage Guide

**Prerequisites:**
You need the Python `requests` library installed and a GitHub Personal Access Token (PAT) exported to your environment to avoid hitting the 60 requests/hour rate limit.

```bash
pip install requests
export GITHUB_TOKEN="ghp_your_actual_token_here"
```

**Command-Line Arguments:**
The script accepts mutually exclusive arguments so you can pivot your analysis:
*   `--pr <NUMBER>`: Analyzes one specific Pull Request, whether open, closed, or merged.
*   `--limit <NUMBER>`: Analyzes the latest N *merged* Pull Requests (defaults to 5 if no arguments are provided).
*   `--date <YYYY-MM-DD>`: Analyzes all Pull Requests that were merged exactly on this specific date.

**Example Commands:**
```bash
# Analyze a specific PR
python3 tools/ci_metrics/collect.py --pr 4952

# Analyze the 10 most recently merged PRs
python3 tools/ci_metrics/collect.py --limit 10

# Analyze all PRs merged on August 1st, 2026
python3 tools/ci_metrics/collect.py --date 2026-08-01
```

---

### 3. Example Output

The output below is **illustrative only** — the PR, timings, job names, and runner labels are made up for the example (real runner labels look like `linux-x86-n2-32` or `linux-x86-ct6e-180-4tpu`). It shows a PR that experienced runner delays, a test failure, a job superseded by cancellation, and a successful manual re-run.

```text
Fetching data for PR #4952...

================================================================================

--- PR #4952: Fix RL Training pipeline: Restrict scanned weight unrolling to MaxText vLLM syncs ---
Status: Merged
Merged at: 2026-08-20T14:32:10Z
Target Workflow: MaxText Package Tests
Total Attempts (Re-runs): 2

--- Attempt 1 Statistics ---
Total Checks: 42
First Job Started: 2026-08-20 12:15:00 UTC
Last Job Ended:    2026-08-20 12:40:31 UTC
Wall-Clock Time: 25.52 minutes
Total Compute Time: 294.23 minutes

Average Job Run Time: 7.00 minutes
Max Job Run Time: 19.43 minutes

Average Runner Wait Time: 0.35 minutes
Max Runner Wait Time: 8.93 minutes

Longest Waiting Jobs (Top 3):
Job Name                                                                    | Wait (min) | Machine Type
-------------------------------------------------------------------------------------------------------------------
gpu-unit / Execute Tests (3)                                                |       8.93 | linux-x86-g2-48-l4-4gpu
cpu-integration / Execute Tests (2)                                         |       4.88 | linux-x86-n2-32
tpu-integration / Setup Parameters                                          |       1.37 | linux-x86-ct6e-180-4tpu

All Jobs Running Times (Longest to Shortest):
Job Name                                                                    | Run (min) 
----------------------------------------------------------------------------------------
Jupyter Notebook Tests / Execute sft_qwen3_demo.ipynb                       |      19.43
Jupyter Notebook Tests / Execute sft_multimodal_gemma3_demo.ipynb           |       7.02
Jupyter Notebook Tests / Execute sft_llama3_demo_tpu.ipynb                  |       5.80
tpu-integration / Execute Tests (1)                                         |       4.12
...

Initial Failed Checks: 2
  -> tpu-unit / Execute Tests (1): Step 'Run PyTest' failed.
  -> All Required Tests Passed: Step 'Check test results' failed.
Cancelled/superseded: 1 (excluded from failure stats)

Flakiness / Re-run Success:
  * tpu-unit / Execute Tests (1)
      Attempt 1: failure -> Attempt 2: success

Across all pushes: 3 pushes analyzed, 4 workflow runs, total compute ~78.10 min (approximate, from run-level durations)

================================================================================
```
