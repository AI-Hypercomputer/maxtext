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

# Decides whether the heavy TPU/GPU test suites may run in this CI run, and
# whether the "Full CI Gatekeeper" job has to block the merge (b/571566462).
#
# On pull requests the heavy suites run only when the PR is not a draft AND
# carries the `ready_for_full_ci` label (or `scheduled-only`, which already asks
# for the full scheduled pipeline). Other events (e.g. schedule,
# workflow_dispatch, or a release run that calls ci_pipeline.yml, which reports
# the caller's event) always run them.
#
# Inputs (environment variables). Boolean inputs count as false unless they are
# "true" in any letter case; EVENT_NAME is compared exactly:
#   EVENT_NAME                github.event_name (defaults to GITHUB_EVENT_NAME,
#                             then pull_request, like analyze_code_changes.sh)
#   IS_DRAFT                  github.event.pull_request.draft
#   HAS_FULL_CI_LABEL         whether the PR has the `ready_for_full_ci` label
#   HAS_SCHEDULED_ONLY_LABEL  whether the PR has the `scheduled-only` label
#   RUN_TESTS, RUN_NOTEBOOKS, RUN_PRETRAIN_TESTS, RUN_POSTTRAIN_TESTS,
#   RUN_PATHWAYS_TESTS, RUN_GPU_TESTS
#                             the flags emitted by analyze_code_changes.sh
#
# Outputs (written to GITHUB_OUTPUT when it is set, and echoed to stdout):
#   run_heavy_tests           "true" if the heavy TPU/GPU suites may run
#   heavy_tests_gated         "true" if the change needs heavy suites that the
#                             gate skipped
#   heavy_tests_gate_message  why they were skipped and what the PR still needs
#                             to run them; empty when nothing was gated
#
# The script only decides and always exits 0 (unless GITHUB_OUTPUT is set but
# cannot be written). The Full CI Gatekeeper job in ci_pipeline.yml fails the
# run when heavy_tests_gated is "true", and also when heavy_tests_gated or
# run_heavy_tests is missing or not "true"/"false", so every run must emit all
# three outputs with these values.

set -euo pipefail

FULL_CI_LABEL="ready_for_full_ci"
MISSING_LABEL_MESSAGE="Heavy tests skipped. Please apply the '${FULL_CI_LABEL}' label to run all tests before merging."
DRAFT_MESSAGE="Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' and apply the '${FULL_CI_LABEL}' label to run all tests before merging."
DRAFT_LABELED_MESSAGE="Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' to run all tests before merging."

# Prints "true" if the argument is "true" in any letter case, otherwise "false".
normalize_bool() {
  local val="${1:-}"
  if [[ "${val,,}" == "true" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

# Writes key=value to GITHUB_OUTPUT (if set) and to stdout.
emit_output() {
  local key="$1"
  local value="$2"
  echo "${key}=${value}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "${key}=${value}" >> "$GITHUB_OUTPUT"
  fi
}

EVENT_NAME="${EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
IS_DRAFT="$(normalize_bool "${IS_DRAFT:-}")"
HAS_FULL_CI_LABEL="$(normalize_bool "${HAS_FULL_CI_LABEL:-}")"
HAS_SCHEDULED_ONLY_LABEL="$(normalize_bool "${HAS_SCHEDULED_ONLY_LABEL:-}")"
RUN_TESTS="$(normalize_bool "${RUN_TESTS:-}")"
RUN_NOTEBOOKS="$(normalize_bool "${RUN_NOTEBOOKS:-}")"
RUN_PRETRAIN_TESTS="$(normalize_bool "${RUN_PRETRAIN_TESTS:-}")"
RUN_POSTTRAIN_TESTS="$(normalize_bool "${RUN_POSTTRAIN_TESTS:-}")"
RUN_PATHWAYS_TESTS="$(normalize_bool "${RUN_PATHWAYS_TESTS:-}")"
RUN_GPU_TESTS="$(normalize_bool "${RUN_GPU_TESTS:-}")"

# Heavy suites are needed when notebooks run (on TPU) or when tests run with any
# suite that has a TPU/GPU flavor. This mirrors the flag conditions of the heavy
# jobs in ci_pipeline.yml (fork and dependabot PRs never run notebooks, but a
# notebook-only change still counts as heavy here).
heavy_suite_selected="false"
for flag in "$RUN_PRETRAIN_TESTS" "$RUN_POSTTRAIN_TESTS" "$RUN_PATHWAYS_TESTS" "$RUN_GPU_TESTS"; do
  if [[ "$flag" == "true" ]]; then
    heavy_suite_selected="true"
  fi
done
heavy_tests_needed="false"
if [[ "$RUN_NOTEBOOKS" == "true" ]] || [[ "$RUN_TESTS" == "true" && "$heavy_suite_selected" == "true" ]]; then
  heavy_tests_needed="true"
fi

run_heavy_tests="true"
skip_reason=""
if [[ "$EVENT_NAME" == "pull_request" ]]; then
  if [[ "$IS_DRAFT" == "true" ]]; then
    run_heavy_tests="false"
    if [[ "$HAS_FULL_CI_LABEL" == "true" || "$HAS_SCHEDULED_ONLY_LABEL" == "true" ]]; then
      skip_reason="$DRAFT_LABELED_MESSAGE"
    else
      skip_reason="$DRAFT_MESSAGE"
    fi
  elif [[ "$HAS_FULL_CI_LABEL" != "true" && "$HAS_SCHEDULED_ONLY_LABEL" != "true" ]]; then
    run_heavy_tests="false"
    skip_reason="$MISSING_LABEL_MESSAGE"
  fi
fi

heavy_tests_gated="false"
heavy_tests_gate_message=""
if [[ "$run_heavy_tests" == "false" && "$heavy_tests_needed" == "true" ]]; then
  heavy_tests_gated="true"
  heavy_tests_gate_message="$skip_reason"
fi

emit_output "run_heavy_tests" "$run_heavy_tests"
emit_output "heavy_tests_gated" "$heavy_tests_gated"
emit_output "heavy_tests_gate_message" "$heavy_tests_gate_message"
