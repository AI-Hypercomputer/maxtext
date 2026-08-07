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

set -e

# Sequence of checks:
# - Non-PR / Diff error: Run all suites (fail-open)
# - Docs only: Skip all tests and notebooks
# - Notebooks only: Run notebooks only
# - Core CI & dependencies: Run all suites
# - GPU only: Run GPU tests only (no notebooks, as notebooks run on TPU)
# - Pathways only: Run Pathways tests only
# - Post-training only: Run post-training tests and notebooks only
# - General inference only: Run pretrain TPU/CPU suites only
# - Tests only: Run test suites only (no notebooks)
# - Default fallback: Run all suites (fail-open for core/shared code)

# Helper to output a key-value flag to GITHUB_OUTPUT (if set) and stdout
emit_flag() {
  local key="$1"
  local val="$2"
  echo "$key=$val"
  if [[ -n "$GITHUB_OUTPUT" ]]; then
    echo "$key=$val" >> "$GITHUB_OUTPUT"
  fi
}

# Helper to set all flags to 'true' or 'false'
set_all_flags() {
  local val="$1"
  for flag in run_tests run_notebooks run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests; do
    emit_flag "$flag" "$val"
  done
}

# Helper to enable specific flags and set all others to 'false'
enable_flags() {
  local enabled=" $* "
  for flag in run_tests run_notebooks run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests; do
    if [[ "$enabled" =~ " $flag " ]]; then
      emit_flag "$flag" "true"
    else
      emit_flag "$flag" "false"
    fi
  done
}

# Helper to check if changed files exclusively match a specific domain pattern
matches_only_domain() {
  local pattern="$1"
  echo "$CHANGED_FILES" | grep -E "$pattern" > /dev/null && \
    ! echo "$CHANGED_FILES" | grep -v -E "($pattern|\.md$)" > /dev/null
}

EVENT_NAME="${EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
BASE_REF="${1:-${GITHUB_BASE_REF:-main}}"

if [ "$EVENT_NAME" != "pull_request" ]; then
  echo "Not a pull request (event: $EVENT_NAME), running all tests"
  set_all_flags "true"
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
  set_all_flags "true"
  exit 0
fi

# Documentation only changes
if ! echo "$CHANGED_FILES" | grep -v -E '\.md$' > /dev/null; then
  echo "Documentation-only files changed, skipping all tests and notebooks."
  set_all_flags "false"
  exit 0
fi

# Notebook only changes
if ! echo "$CHANGED_FILES" | grep -v -E '\.(ipynb|md)$' > /dev/null; then
  echo "Only notebook/doc files changed, running notebooks only."
  enable_flags run_notebooks
  exit 0
fi

# Core CI workflows or dependencies changes
if echo "$CHANGED_FILES" | grep -E '(^|/)(src/dependencies/|\.github/workflows/)' | grep -v -E '(\.github/workflows/run_pathways_tests\.yml)' > /dev/null; then
  echo "Core files (dependencies, workflows) changed, enabling all tests and notebooks."
  set_all_flags "true"
  exit 0
fi

# GPU only changes (note: notebooks run on TPU, so GPU changes skip notebooks)
if matches_only_domain 'src/maxtext/configs/gpu/|src/maxtext/inference/gpu/|tests/end_to_end/gpu/'; then
  echo "Only GPU files changed, enabling GPU tests only."
  enable_flags run_tests run_gpu_tests
  exit 0
fi

# Pathways only changes
if matches_only_domain 'src/maxtext/inference/jetstream_pathways/|\.github/workflows/run_pathways_tests\.yml'; then
  echo "Only Pathways files changed, enabling Pathways tests only."
  enable_flags run_tests run_pathways_tests
  exit 0
fi

# Post-training only changes
if matches_only_domain 'src/maxtext/trainers/post_train/|tests/post_training/'; then
  echo "Only post-training files changed, enabling post-training tests only."
  enable_flags run_tests run_notebooks run_posttrain_tests
  exit 0
fi

# General inference only changes
if matches_only_domain 'src/maxtext/inference/|tests/inference/'; then
  echo "Only inference files changed, enabling pretrain unit/integration tests only."
  enable_flags run_tests run_pretrain_tests
  exit 0
fi

# Tests and test-tooling only changes (skips notebooks as tutorials are unaffected)
if matches_only_domain '(^tests/|^\.github/scripts/|^pytest\.ini$|^\.coveragerc$)'; then
  echo "Only test files and test configurations changed, enabling test suites (skipping notebooks)."
  enable_flags run_tests run_pretrain_tests run_posttrain_tests run_pathways_tests run_gpu_tests
  exit 0
fi

# Default fallback: run all domain suites
echo "General source changes detected, enabling all domain suites."
set_all_flags "true"
