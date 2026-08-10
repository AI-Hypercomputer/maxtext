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
#   1. Non-PR Events: If not a pull request, enables all test suites and notebooks.
#   2. Empty Diff / Error: If no files are detected or diff fails, runs everything 
#      as a fail-safe.
#   3. Default State: All individual test and notebook flags are initialized to 'false'.
#   4. File Evaluation Loop: Iterates through each changed file:
#      - Evaluates against specific domain rules (notebook workflows, pathways, 
#        TPU pre/post-training dependencies, GPU files, inference, etc.) and 
#        cumulatively enables corresponding flags.
#      - Tracks any unmatched files.
#   5. Exclusion Filtering: Filters out pre-configured excluded patterns/directories 
#      from the unmatched file list.
#   6. Fallback Check: If any truly unmatched files remain, triggers a general fallback 
#      enabling all core test suites (excluding notebooks).

set -e

# Helper to output a key-value flag to GITHUB_OUTPUT (if set) and stdout
emit_flag() {
  local key="$1"
  local val="$2"
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

EVENT_NAME="${EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
BASE_REF="${1:-${GITHUB_BASE_REF:-main}}"

if [ "$EVENT_NAME" != "pull_request" ]; then
  echo "Not a pull request (event: $EVENT_NAME), running all tests and notebooks"
  set_test_flags "true" "true"
  exit 0
fi

if ! git rev-parse --verify "origin/$BASE_REF" > /dev/null 2>&1; then
  git fetch origin "$BASE_REF" 2>/dev/null || true
fi

CHANGED_FILES=$(git diff --name-only "origin/${BASE_REF}...HEAD" 2>/dev/null || true)

echo "Changed files against origin/${BASE_REF}:"
echo "$CHANGED_FILES"

if [ -z "$CHANGED_FILES" ]; then
  echo "No files detected or diff failed. Running everything as a fail-safe."
  set_test_flags "true" "true"
  exit 0
fi

# Disable all tests by default
set_test_flags "false" "false"

# Array to track files that didn't match any known pattern
UNMATCHED_FILES=()

# Pre-populated list of excluded patterns/files that shouldn't trigger tests
EXCLUDED_FILES=(
  '^\.github/scripts/'
  '^tools/'
  '\.md$'
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
