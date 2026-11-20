#!/bin/bash

# Copyright 2026 Google LLC
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

# Cancels any in-progress `MaxText Package Tests` (`ci_pipeline.yml`) pull_request
# run on HEAD_SHA and re-runs the newest pull_request run on HEAD_SHA so CI picks
# up the PR's current labels while keeping check runs attached to the PR checks box.

set -euo pipefail

if [[ ! "${GITHUB_REPOSITORY:-}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] ||
  [[ ! "${HEAD_SHA:-}" =~ ^[0-9a-f]{40}$ ]] ||
  [[ ! "${PR_NUMBER:-}" =~ ^[0-9]+$ ]]; then
  echo "::error::Invalid repository ('${GITHUB_REPOSITORY:-}'), HEAD_SHA ('${HEAD_SHA:-}'), or PR_NUMBER ('${PR_NUMBER:-}')."
  exit 1
fi

POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-5}"
FIND_RUN_ATTEMPTS="${FIND_RUN_ATTEMPTS:-6}"
CANCEL_WAIT_ATTEMPTS="${CANCEL_WAIT_ATTEMPTS:-24}"
FORCE_CANCEL_AFTER_ATTEMPT="${FORCE_CANCEL_AFTER_ATTEMPT:-12}"
RERUN_ATTEMPTS="${RERUN_ATTEMPTS:-5}"

# Exits 0 cleanly if the PR was closed or a newer commit was pushed (whose `synchronize`
# run already reads live labels, and would be cancelled if we re-ran an older commit).
verify_pr_still_matches_head() {
  local pr_meta="" pr_state="" live_head_sha="" attempt
  for attempt in 1 2 3; do
    if pr_meta=$(gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}" \
      --jq '[.state // "none", .head.sha // "none"] | @tsv') && [[ -n "${pr_meta}" ]]; then
      IFS=$'\t' read -r pr_state live_head_sha <<< "${pr_meta}"
      if [[ "${pr_state}" != "open" || "${live_head_sha}" != "${HEAD_SHA}" ]]; then
        echo "PR #${PR_NUMBER} state='${pr_state}' head='${live_head_sha}' no longer matches open ${HEAD_SHA}; skipping CI re-run."
        exit 0
      fi
      return 0
    fi
    if (( attempt < 3 )); then
      sleep "${POLL_INTERVAL_SECONDS}"
    fi
  done
  echo "::error::Could not read live state for PR #${PR_NUMBER} from GitHub API."
  exit 1
}

find_latest_pr_run() {
  gh api \
    "repos/${GITHUB_REPOSITORY}/actions/workflows/ci_pipeline.yml/runs?head_sha=${HEAD_SHA}&event=pull_request&per_page=100" \
    --jq '.workflow_runs | max_by(.id) | select(. != null) | [.id, .status // "none", .conclusion // "none", .referenced_workflows[0].sha // "none"] | @tsv'
}

verify_pr_still_matches_head

latest_run=""
api_ok="false"
for (( attempt = 1; attempt <= FIND_RUN_ATTEMPTS; attempt++ )); do
  if latest_run=$(find_latest_pr_run); then
    api_ok="true"
    if [[ -n "${latest_run}" ]]; then
      break
    fi
  fi
  if (( attempt < FIND_RUN_ATTEMPTS )); then
    sleep "${POLL_INTERVAL_SECONDS}"
  fi
done

if [[ "${api_ok}" != "true" ]]; then
  echo "::error::Could not list 'MaxText Package Tests' runs for commit ${HEAD_SHA} on PR #${PR_NUMBER}."
  exit 1
fi

if [[ -z "${latest_run}" ]]; then
  echo "No 'MaxText Package Tests' pull_request run found for commit ${HEAD_SHA} on PR #${PR_NUMBER}; nothing to re-run."
  exit 0
fi

IFS=$'\t' read -r run_id run_status run_conclusion run_merge_sha <<< "${latest_run}"

# Preserve GitHub's fork approval gate on first-time external contributors.
if [[ "${run_status}" == "action_required" || "${run_conclusion}" == "action_required" ]]; then
  echo "Run ${run_id} on PR #${PR_NUMBER} is awaiting maintainer approval (action_required); skipping automatic re-run."
  exit 0
fi

# `/rerun` re-executes the run's frozen merge commit (`referenced_workflows[0].sha`).
# Runs started before live API label support landed lack `fetch_pr_metadata_via_api`
# and would reuse stale initial labels; fail fast before cancelling active jobs.
if [[ "${run_merge_sha}" =~ ^[0-9a-f]{40}$ ]] &&
  raw_script=$(gh api -H "Accept: application/vnd.github.raw" \
    "repos/${GITHUB_REPOSITORY}/contents/.github/scripts/analyze_code_changes.sh?ref=${run_merge_sha}" 2>/dev/null) &&
  [[ -n "${raw_script}" && "${raw_script}" != *"fetch_pr_metadata_via_api"* ]]; then
  echo "::error::Run ${run_id} on PR #${PR_NUMBER} predates live PR label re-runs and would reuse stale initial labels. Rebase PR #${PR_NUMBER} onto main (or amend its commit) and force-push to start a fresh CI run."
  exit 1
fi

if [[ "${run_status}" != "completed" ]]; then
  echo "Run ${run_id} on PR #${PR_NUMBER} is currently '${run_status}'; cancelling before re-run."
  cancel_sent="false"
  force_cancel_sent="false"
  for (( attempt = 1; attempt <= CANCEL_WAIT_ATTEMPTS; attempt++ )); do
    if (( attempt >= FORCE_CANCEL_AFTER_ATTEMPT )) && [[ "${force_cancel_sent}" == "false" ]]; then
      echo "Run ${run_id} is still '${run_status}'; escalating to force-cancel."
      if gh api -X POST "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}/force-cancel" --silent 2>/dev/null; then
        force_cancel_sent="true"
      fi
    elif (( attempt < FORCE_CANCEL_AFTER_ATTEMPT )) && [[ "${cancel_sent}" == "false" ]]; then
      if gh api -X POST "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}/cancel" --silent 2>/dev/null; then
        cancel_sent="true"
      fi
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
    if state_line=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}" \
      --jq '[.status // "none", .conclusion // "none"] | @tsv') && [[ -n "${state_line}" ]]; then
      IFS=$'\t' read -r run_status run_conclusion <<< "${state_line}"
      if [[ "${run_status}" == "completed" ]]; then
        break
      fi
    else
      echo "Warning: poll attempt ${attempt} for run ${run_id} failed; continuing."
    fi
  done

  if [[ "${run_status}" != "completed" ]]; then
    echo "::error::Timed out waiting for run ${run_id} to stop before re-running (status: '${run_status}')."
    exit 1
  fi

  verify_pr_still_matches_head
fi

output=""
for (( attempt = 1; attempt <= RERUN_ATTEMPTS; attempt++ )); do
  if output=$(gh api -X POST "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}/rerun" --silent 2>&1); then
    echo "Triggered re-run of 'MaxText Package Tests' run ${run_id} for PR #${PR_NUMBER}."
    exit 0
  fi
  if [[ "${output,,}" == *"over a month ago"* ]]; then
    # `allow_update_branch` is disabled on this repo and `auto_label_pr.yml` requires 1 commit per PR.
    echo "::error::Run ${run_id} is over 30 days old and can no longer be re-run by GitHub. Rebase PR #${PR_NUMBER} onto main (or amend its commit) and force-push to start a fresh CI run."
    exit 1
  fi
  if (( attempt < RERUN_ATTEMPTS )); then
    echo "Attempt ${attempt} to re-run ${run_id} failed (${output}); retrying."
    sleep "$(( attempt * POLL_INTERVAL_SECONDS ))"
  fi
done

echo "::error::Could not re-run 'MaxText Package Tests' run ${run_id}: ${output}"
exit 1
