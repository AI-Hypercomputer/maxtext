#!/usr/bin/env bash

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

# Resolves the source run ID for fast-rebuild mode.
#
# Three-tier priority:
#   1. Explicit input  – the user provided a source_run_id.
#   2. Current branch  – most recent completed run on this branch
#                        where every "Build and Push" job succeeded.
#   3. Main fallback   – same check on main (skipped when already on main).
#
# Required environment variables:
#   EXPLICIT_ID          User-provided source_run_id (may be empty).
#   BRANCH               github.ref_name of the dispatch.
#   GH_TOKEN             GitHub token with actions:read scope.
#   WORKFLOW_FILE        Workflow filename to search (e.g. tpu_docker_images_pipeline.yml).
#   GITHUB_REPOSITORY    owner/repo.
#   GITHUB_OUTPUT        Path to the step output file.

set -euo pipefail

# Priority 1: explicit input
if [ -n "${EXPLICIT_ID}" ]; then
  echo "Using explicit source_run_id: ${EXPLICIT_ID}"
  echo "source_run_id=${EXPLICIT_ID}" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "No source_run_id provided. Auto-detecting..."

# Helper: check if all Build-and-Push jobs in a run succeeded
builds_ok() {
  gh api "repos/${GITHUB_REPOSITORY}/actions/runs/$1/jobs?per_page=100" \
    --jq '[.jobs[] | select(.name | test("Build and Push")) | .conclusion] | if length == 0 then false else all(. == "success") end'
}

# Priority 2: most recent completed run on this branch
echo "Searching branch '${BRANCH}'..."
for rid in $(gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/${WORKFLOW_FILE}/runs" \
    -f branch="${BRANCH}" -f status="completed" -F per_page=5 \
    --jq '.workflow_runs[]?.id'); do
  if [ "$(builds_ok "$rid")" = "true" ]; then
    echo "Auto-resolved from branch ${BRANCH}: run ${rid}"
    echo "source_run_id=${rid}" >> "$GITHUB_OUTPUT"
    exit 0
  fi
done

# Priority 3: most recent completed run on main (skip if already on main)
if [ "${BRANCH}" != "main" ]; then
  echo "No valid run on branch '${BRANCH}'. Searching main..."
  for rid in $(gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/${WORKFLOW_FILE}/runs" \
      -f branch="main" -f status="completed" -F per_page=5 \
      --jq '.workflow_runs[]?.id'); do
    if [ "$(builds_ok "$rid")" = "true" ]; then
      echo "::warning::No source run found on branch ${BRANCH}. Using main run ${rid}."
      echo "source_run_id=${rid}" >> "$GITHUB_OUTPUT"
      exit 0
    fi
  done
fi

echo "::error::No source run with successful builds found on branch '${BRANCH}' or main. Run build-all first."
exit 1
