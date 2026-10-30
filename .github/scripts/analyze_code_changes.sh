#!/bin/bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Analyzes git diff changes in a Pull Request against a base reference
# to selectively enable or disable test suites and notebook executions.
# This optimizes CI run times by only triggering relevant test environments
# based on the specific files modified.
#
# Behavior & Logic Flow:
#   0. FORCE_ALL_TESTS=true: Bypasses all rules and enables every test suite and
#      notebook (used by the `scheduled-only` PR label).
#   1. Non-PR Events: If not a pull request, enables all test suites and notebooks.
#   2. Changed-file source: Uses the GitHub REST API (paginated `pulls/N/files`) when
#      PR_NUMBER, GITHUB_REPOSITORY and the gh CLI are available; falls back to
#      `git diff origin/<base>...HEAD` if the API is unavailable, fails, returns no
#      files, or hits its 3000-file cap.
#   3. Empty Diff / Error: If no files are detected or diff fails, runs all core
#      test suites (excluding expensive notebook tests) as a fail-safe.
#   4. Default State: All individual test and notebook flags are initialized to 'false'.
#   5. File Evaluation Loop: Iterates through each changed file:
#      - Evaluates against specific domain rules (notebook workflows, pathways,
#        TPU pre/post-training dependencies, GPU files, inference, etc.) and
#        cumulatively enables corresponding flags.
#      - Tracks any unmatched files.
#   6. Exclusion Filtering: Filters out pre-configured excluded patterns/directories
#      from the unmatched file list.
#   7. Fallback Check: If any truly unmatched files remain, triggers a general fallback
#      enabling all core test suites (excluding notebooks).
#   8. Heavy-Test Gate: Every successful run ends by writing run_heavy_tests,
#      heavy_tests_gated and heavy_tests_gate_message, which skip the heavy TPU/GPU
#      suites on drafts and while a PR has `skip-ci-tests` (see gate_heavy_tests).

set -e

# The latest value of each flag, for the heavy-test gate (GITHUB_OUTPUT keeps the last value written).
declare -A FLAG_VALUES=()

# Helper to output a key-value flag to GITHUB_OUTPUT (if set) and stdout
emit_flag() {
  local key="$1"
  local val="$2"
  FLAG_VALUES["$key"]="$val"
  # Only log if value is true
  if [[ "$val" == "true" ]]; then
    echo "$key=$val"
  fi
  # Always write to GITHUB_OUTPUT so GitHub Actions steps have the key
  if [[ -n "$GITHUB_OUTPUT" ]]; then
    echo "$key=$val" >> "$GITHUB_OUTPUT"
  fi
}

# Helper to enable specific test flags and set all others to 'false'
set_test_flags() {
  local enable_tests="$1"
  local enable_notebooks="$2"
  for flag in run_tests run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests; do
    emit_flag "$flag" "$enable_tests"
  done
  emit_flag "run_notebooks" "$enable_notebooks"
}

# Helper to enable specific flags and set all others to 'false'
enable_flags() {
  local enabled=" $* "
  for flag in run_tests run_notebooks run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests; do
    if [[ "$enabled" == *" $flag "* ]]; then
      emit_flag "$flag" "true"
    fi
  done
}

# Helper to check if a changed file matches a specific domain pattern
matches_pattern() {
  local changed_file="$1"
  local pattern="$2"
  [[ "$changed_file" =~ $pattern ]]
}

# Heavy-test gate (b/571566462): decides whether the heavy TPU/GPU suites (the TPU/GPU jobs and the
# notebooks in ci_pipeline.yml) may run, and writes run_heavy_tests, heavy_tests_gated and
# heavy_tests_gate_message. Non-PR events always run them. For a PR, the first matching rule wins:
#   1. Drafts never run them. The message says whether marking the PR ready is enough.
#   2. `scheduled-only` (FORCE_ALL_TESTS) runs them: it asks for the full scheduled pipeline.
#   3. `skip-ci-tests` skips them. add_skip_ci_tests_label.yml adds it to new PRs; auto_label_pr.yml
#      removes it when the PR gets the "pull ready" label, and the author can remove it sooner.
#   4. The first run of a new PR (`opened`) skips them: the labeler adds the label in parallel, so
#      this run's payload doesn't have it yet. The labeler skips the PRs that Copybara exports
#      (Copybara merges them itself, without approvals here), so they run every suite.
#   5. Otherwise they run.
gate_heavy_tests() {
  local run_heavy_tests=true heavy_tests_gated=false message="" gets_label_now=false
  local needs_heavy_tests=false flag
  if [[ "$EVENT_NAME" == "pull_request" ]]; then
    if [[ "${EVENT_ACTION:-}" == "opened" && "${IS_COPYBARA_PR:-}" != "true" ]]; then
      gets_label_now=true
    fi
    if [[ "${IS_DRAFT:-}" == "true" ]]; then
      run_heavy_tests=false
      # Once the PR is ready, rule 2 or 5 runs all tests; under rule 3 or 4 it has, or is getting, the label.
      if [[ "$FORCE_ALL_TESTS" == "true" ]] ||
        [[ "${HAS_SKIP_CI_TESTS_LABEL:-}" != "true" && "$gets_label_now" != "true" ]]; then
        message="Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' to run all tests before merging."
      else
        message="Heavy tests skipped because this PR is a draft. Mark it 'Ready for review'; they then run once the PR gets the 'pull ready' label, which a bot adds after 2 approvals (a maintainer adds it to PRs from forks)."
      fi
    elif [[ "$FORCE_ALL_TESTS" == "true" ]]; then
      run_heavy_tests=true
    elif [[ "${HAS_SKIP_CI_TESTS_LABEL:-}" == "true" ]]; then
      run_heavy_tests=false
      message="Heavy tests skipped because this PR has the 'skip-ci-tests' label. They run once the PR gets the 'pull ready' label, which a bot adds after 2 approvals (a maintainer adds it to PRs from forks). To run them sooner, remove 'skip-ci-tests' or ask a maintainer to."
    elif [[ "$gets_label_now" == "true" ]]; then
      run_heavy_tests=false
      message="Heavy tests skipped on the first run of a new PR, which gets the 'skip-ci-tests' label. They run once the PR gets the 'pull ready' label, which a bot adds after 2 approvals (a maintainer adds it to PRs from forks). To run them sooner, remove 'skip-ci-tests' or ask a maintainer to."
    fi
    # Only a change that needs the heavy suites is gated: notebooks run on TPUs, and the test suites that
    # have a TPU/GPU flavor run when run_tests is true (the flags that the heavy jobs check). If you add a
    # heavy job or change the flags a heavy job depends on, update this rule too.
    if [[ "${FLAG_VALUES[run_notebooks]:-}" == "true" ]]; then
      needs_heavy_tests=true
    elif [[ "${FLAG_VALUES[run_tests]:-}" == "true" ]]; then
      for flag in run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests; do
        if [[ "${FLAG_VALUES[$flag]:-}" == "true" ]]; then
          needs_heavy_tests=true
        fi
      done
    fi
    if [[ "$run_heavy_tests" == "false" && "$needs_heavy_tests" == "true" ]]; then
      heavy_tests_gated=true
    else
      message=""
    fi
  fi
  # All Required Tests Passed fails on missing or invalid outputs, so always write all three.
  local outputs=(
    "run_heavy_tests=${run_heavy_tests}"
    "heavy_tests_gated=${heavy_tests_gated}"
    "heavy_tests_gate_message=${message}"
  )
  printf '%s\n' "${outputs[@]}"
  if [[ -n "$GITHUB_OUTPUT" ]]; then
    printf '%s\n' "${outputs[@]}" >> "$GITHUB_OUTPUT"
  fi
}

# Every successful run ends here, so the gate outputs come last. If the script fails earlier, they
# stay missing and All Required Tests Passed fails, instead of the heavy suites running ungated.
finish() {
  gate_heavy_tests
  exit 0
}

EVENT_NAME="${EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
BASE_REF="${1:-${GITHUB_BASE_REF:-main}}"
# Set to "true" to bypass the per-file rules and run everything, e.g. when a PR
# carries the `scheduled-only` label and should behave like a scheduled run.
# Lowercased so that values such as "True" from manual runs are also honored.
FORCE_ALL_TESTS="${FORCE_ALL_TESTS:-false}"
FORCE_ALL_TESTS="${FORCE_ALL_TESTS,,}"

if [[ "$FORCE_ALL_TESTS" == "true" ]]; then
  echo "FORCE_ALL_TESTS is set, running all tests and notebooks"
  set_test_flags "true" "true"
  finish
fi

if [ "$EVENT_NAME" != "pull_request" ]; then
  echo "Not a pull request (event: $EVENT_NAME), running all tests and notebooks"
  set_test_flags "true" "true"
  finish
fi

# GitHub's "List pull requests files" REST endpoint returns at most this many files.
MAX_API_FILES=3000

CHANGED_FILES=""
DIFF_SOURCE=""

# Preferred source: ask GitHub for the PR's changed files. The paginated REST endpoint
# is used on purpose; `gh pr view --json files` silently truncates at 100 files.
# Returns non-zero (and leaves CHANGED_FILES empty) whenever the list cannot be trusted.
list_changed_files_via_api() {
  if [[ -z "${PR_NUMBER:-}" || -z "${GITHUB_REPOSITORY:-}" ]]; then
    return 1
  fi
  if ! command -v gh > /dev/null 2>&1; then
    echo "Note: gh CLI not found; listing changed files with git diff instead."
    return 1
  fi
  local api_files api_count
  # stderr is not captured so it can never be parsed as a filename; it still reaches the log.
  if ! api_files=$(gh api --paginate "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/files" --jq '.[].filename'); then
    echo "Warning: GitHub API file listing failed for PR #${PR_NUMBER} (see gh error above); falling back to git diff."
    return 1
  fi
  api_count=$(grep -c . <<< "$api_files" || true)
  if (( api_count == 0 )); then
    echo "Warning: GitHub API returned no files for PR #${PR_NUMBER}; falling back to git diff."
    return 1
  fi
  if (( api_count >= MAX_API_FILES )); then
    echo "Warning: GitHub API returned ${api_count} files, at the endpoint's ${MAX_API_FILES}-file cap, so the list may be truncated; falling back to git diff."
    return 1
  fi
  CHANGED_FILES="$api_files"
  DIFF_SOURCE="GitHub API, PR #${PR_NUMBER}"
}

# Fallback source: diff against the base branch in the local clone.
list_changed_files_via_git() {
  local diff_target="origin/$BASE_REF"
  local fetch_ok="true"
  local fetch_err fetch_errors diff_error
  if ! git rev-parse --verify "$diff_target" > /dev/null 2>&1; then
    fetch_ok="false"
    fetch_errors=""
    # Attempt 1: explicit refspec so the remote-tracking ref is created even on
    # clones whose fetch refspec does not cover the base branch.
    if fetch_err=$(git fetch origin "${BASE_REF}:refs/remotes/origin/${BASE_REF}" 2>&1); then
      fetch_ok="true"
    else
      fetch_errors="refspec fetch: ${fetch_err}"
      # Attempt 2: plain fetch (updates FETCH_HEAD, and on modern git the tracking ref too).
      if fetch_err=$(git fetch origin "$BASE_REF" 2>&1); then
        fetch_ok="true"
      else
        fetch_errors="${fetch_errors}; plain fetch: ${fetch_err}"
        echo "Warning: unable to fetch base ref '${BASE_REF}' from origin: ${fetch_errors}"
      fi
    fi
  fi
  # Only trust FETCH_HEAD if one of OUR fetches succeeded; otherwise it may be a stale
  # ref left by actions/checkout (e.g. the PR merge ref), which would yield a wrong diff.
  if [[ "$fetch_ok" == "true" ]] && ! git rev-parse --verify "$diff_target" > /dev/null 2>&1; then
    if git rev-parse --verify "FETCH_HEAD" > /dev/null 2>&1; then
      echo "Warning: ${diff_target} is not available; falling back to FETCH_HEAD ($(git rev-parse --short FETCH_HEAD))."
      diff_target="FETCH_HEAD"
    fi
  fi

  diff_error=""
  if ! CHANGED_FILES=$(git diff --name-only "${diff_target}...HEAD" 2>/dev/null); then
    diff_error=$(git diff --name-only "${diff_target}...HEAD" 2>&1) || true
    CHANGED_FILES=""
  fi

  if [[ -n "$diff_error" ]]; then
    echo "Warning: git diff encountered an error: $diff_error"
  fi
  DIFF_SOURCE="git diff against ${diff_target}"
}

if ! list_changed_files_via_api; then
  list_changed_files_via_git
fi

echo "Changed files (source: ${DIFF_SOURCE}):"
echo "$CHANGED_FILES"

if [ -z "$CHANGED_FILES" ]; then
  echo "No files detected or diff failed. Running core test suites (excluding notebooks) as a fail-safe."
  set_test_flags "true" "false"
  finish
fi

# Disable all tests by default
set_test_flags "false" "false"

# Array to track files that didn't match any known pattern
UNMATCHED_FILES=()

# Pre-populated list of excluded patterns/files that shouldn't trigger tests
EXCLUDED_FILES=(
  '^\.agents/'
  '^\.gemini/'
  '\.github/workflows/update_changelog.yml$'
  '^\.github/scripts/'
  '^tools/'
  '^tests/dev/'
  '\.md$'
  'src/maxtext/version.py$'
)

# Loop through every changed file
while IFS= read -r file; do
  [[ -z "$file" ]] && continue

  matched=false

  # Notebook workflows changes
  if matches_pattern "$file" "\.github/workflows/run_jupyter_notebooks.yml$"; then
    echo "Notebook workflow changed, enabling notebook tests."
    enable_flags run_notebooks
    matched=true
  fi

  # Pathways workflow changes
  if matches_pattern "$file" "\.github/workflows/run_pathways_tests.yml$"; then
    echo "Pathways workflow changed, enabling all pathways tests."
    enable_flags run_tests run_pathways_tests
    matched=true
  fi

  # TPU pre-training dependencies changes
  if matches_pattern "$file" "src/dependencies/requirements/generated_requirements/tpu-requirements.txt$"; then
    echo "TPU pre-training dependencies changed, enabling TPU pre-training tests."
    enable_flags run_tests run_pretrain_tests run_pathways_tests
    matched=true
  fi

  # TPU post-training dependencies changes
  if matches_pattern "$file" "src/dependencies/requirements/generated_requirements/tpu-post-train-requirements.txt$"; then
    echo "TPU post-training dependencies changed, enabling TPU post-training tests."
    enable_flags run_tests run_posttrain_tests run_pathways_tests
    matched=true
  fi

  # GPU dependencies changes
  if matches_pattern "$file" "src/dependencies/requirements/generated_requirements/cuda12-requirements.txt$"; then
    echo "GPU dependencies changed, enabling GPU tests."
    enable_flags run_tests run_gpu_tests
    matched=true
  fi

  # GPU configs/source/test changes
  if matches_pattern "$file" "src/maxtext/configs/gpu/|src/maxtext/inference/gpu/|tests/end_to_end/gpu/"; then
    echo "GPU files changed, enabling GPU tests."
    enable_flags run_tests run_gpu_tests
    matched=true
  fi

  # Post-training source/test changes
  if matches_pattern "$file" "src/maxtext/trainers/post_train/|tests/post_training/"; then
    echo "Post-training files changed, enabling TPU post-training tests."
    enable_flags run_tests run_posttrain_tests
    matched=true
  fi

  # General inference only changes
  if matches_pattern "$file" "src/maxtext/inference/|tests/inference/"; then
    echo "Inference files changed, enabling TPU pre-training tests."
    enable_flags run_tests run_pretrain_tests
    matched=true
  fi

  # Notebook files changed
  if matches_pattern "$file" "\.ipynb$"; then
    echo "Notebook files changed, enabling notebook tests."
    enable_flags run_notebooks
    matched=true
  fi

  # If no rule matched this file, track it as unmatched
  if [[ "$matched" == "false" ]]; then
    UNMATCHED_FILES+=("$file")
  fi

done <<< "$CHANGED_FILES"

# Filter out pre-populated EXCLUDED_FILES from UNMATCHED_FILES
FINAL_UNMATCHED_FILES=()
for file in "${UNMATCHED_FILES[@]}"; do
  is_excluded="false"
  for excluded in "${EXCLUDED_FILES[@]}"; do
    if matches_pattern "$file" "$excluded"; then
      is_excluded="true"
      break
    fi
  done
  if [[ "$is_excluded" == "false" ]]; then
    FINAL_UNMATCHED_FILES+=("$file")
  fi
done

# If there are any unmatched files, trigger general fallback
if [ ${#FINAL_UNMATCHED_FILES[@]} -gt 0 ]; then
  echo "The following changed files did not match any specific domain rules:"
  printf '  - %s\n' "${FINAL_UNMATCHED_FILES[@]}"
  echo "Enabling all test suites except notebook tests as a fallback."
  enable_flags run_tests run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests
fi

finish
