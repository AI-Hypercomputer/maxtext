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

# Dependency drift guard for fast-rebuild mode.
#
# Compares dependency-related files between a source commit and a current commit.
# If any dependency files changed, the fast-rebuild guarantee ("same deps, new code")
# is violated and this script exits non-zero.
#
# The script only reads git data for the two commits (fetching them if the clone is
# shallow); it never checks out or executes code from either commit, so it is safe to
# run from a trusted checkout of the workflow's own ref.
#
# Required environment variables:
#   SOURCE_SHA          - Git SHA of the source image's commit
#                         (if empty, SOURCE_RUN_ID + GITHUB_TOKEN + GITHUB_REPOSITORY
#                         are used to look it up via the GitHub API)
#
# Optional environment variables:
#   CURRENT_SHA         - Git SHA of the commit being built (defaults to HEAD)
#   SOURCE_RUN_ID       - GitHub Actions run ID (used to look up SOURCE_SHA if not provided)
#   GITHUB_TOKEN        - GitHub token for API access (required if SOURCE_SHA not provided)
#   GITHUB_REPOSITORY   - owner/repo (required if SOURCE_SHA not provided)
#   ALLOW_DRIFT         - Set to "true" to warn instead of fail when drift is detected
#   WORKFLOW            - "pre-training" (default) or "post-training". Post-training images
#                         need the clean overlay when src/maxtext/integration/vllm/ changed.
#   GITHUB_OUTPUT       - If set, "overlay_variant=fast|clean" is appended to this file so the
#                         workflow can pick the overlay Dockerfile (see select_overlay_variant).
#
# Exit codes:
#   0 - No drift detected, or drift allowed (ALLOW_DRIFT=true)
#   1 - Drift detected and ALLOW_DRIFT is not "true"
#   2 - Input validation error (missing required variables)

set -eo pipefail

# Paths that must not change between source and target for "same deps" guarantee.
# See spec Section 5.2 for rationale on each path.
GUARD_PATHS=(
  "src/dependencies/"
  "pyproject.toml"
  ".dockerignore"
)

# --- Resolve SOURCE_SHA ---
if [ -z "${SOURCE_SHA}" ]; then
  if [ -z "${SOURCE_RUN_ID}" ] || [ -z "${GITHUB_TOKEN}" ] || [ -z "${GITHUB_REPOSITORY}" ]; then
    echo "::error::Either SOURCE_SHA or (SOURCE_RUN_ID + GITHUB_TOKEN + GITHUB_REPOSITORY) must be set."
    exit 2
  fi

  echo "Resolving source commit SHA from run ${SOURCE_RUN_ID}..."
  SOURCE_SHA=$(curl -s -f -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runs/${SOURCE_RUN_ID}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('head_sha',''))" 2>/dev/null) || {
    echo "::error::Failed to fetch run ${SOURCE_RUN_ID} from GitHub API. Check that the run ID exists and GITHUB_TOKEN has actions:read permission."
    exit 2
  }

  if [ -z "${SOURCE_SHA}" ]; then
    echo "::error::Could not resolve head_sha for run ${SOURCE_RUN_ID}."
    exit 2
  fi
fi

# --- Resolve CURRENT_SHA (defaults to HEAD) ---
if [ -z "${CURRENT_SHA}" ]; then
  CURRENT_SHA=$(git rev-parse HEAD)
fi

echo "Source commit:  ${SOURCE_SHA}"
echo "Current commit: ${CURRENT_SHA}"

# --- Ensure both commits are in local history (shallow clones may lack them) ---
for sha in "${SOURCE_SHA}" "${CURRENT_SHA}"; do
  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "Commit ${sha} not in local history, fetching..."
    git fetch --depth=1 origin "${sha}" || {
      echo "::error::Could not fetch commit ${sha}. It may have been force-pushed away."
      exit 2
    }
  fi
done

# --- Overlay Dockerfile variant ---
# fast:  maxtext_code_overlay_fast.Dockerfile, COPY --link only. BuildKit never pulls the
#        source image's layers, so the build takes seconds. Nothing is deleted from the
#        source image, so a file removed or renamed since the source commit would survive.
#        Only used when no code file was removed.
# clean: maxtext_code_overlay.Dockerfile. Removes the old code directories first and
#        re-installs the vLLM adapter for post-training images. Slower, always exact.
CODE_PATHS=(
  "src/maxtext/"
  "tests/"
  "benchmarks/"
  "pytest.ini"
)

select_overlay_variant() {
  local variant="fast"
  local reason="no code files were deleted or renamed"
  local removed=""
  local vllm_changed=""
  removed=$(git diff --no-renames --diff-filter=D --name-only "${SOURCE_SHA}" "${CURRENT_SHA}" -- "${CODE_PATHS[@]}")
  if [ -n "${removed}" ]; then
    variant="clean"
    reason="code files were deleted or renamed since the source commit"
  elif [ "${WORKFLOW}" = "post-training" ]; then
    vllm_changed=$(git diff --no-renames --name-only "${SOURCE_SHA}" "${CURRENT_SHA}" -- "src/maxtext/integration/vllm/")
    if [ -n "${vllm_changed}" ]; then
      variant="clean"
      reason="src/maxtext/integration/vllm/ changed, the installed vLLM adapter must be refreshed"
    fi
  fi
  echo "Overlay variant: ${variant} (${reason})"
  if [ -n "${removed}" ]; then
    echo "${removed}" | while IFS= read -r f; do echo "  - ${f} (removed)"; done
  fi
  if [ -n "${GITHUB_OUTPUT}" ]; then
    echo "overlay_variant=${variant}" >> "${GITHUB_OUTPUT}"
  fi
}

# --- Check for dependency file changes ---
CHANGED=$(git diff --no-renames --name-only "${SOURCE_SHA}" "${CURRENT_SHA}" -- "${GUARD_PATHS[@]}")

if [ -n "${CHANGED}" ]; then
  echo "::error::Dependency files changed between source (${SOURCE_SHA:0:7}) and current commit (${CURRENT_SHA:0:7}):"
  echo "${CHANGED}" | while IFS= read -r f; do echo "  - ${f}"; done

  if [ "${ALLOW_DRIFT}" = "true" ]; then
    echo ""
    echo "::warning::Proceeding with fast-rebuild despite dependency drift (ALLOW_DRIFT=true)."
    echo "The resulting image may have DIFFERENT dependencies than the source image."
    select_overlay_variant
    exit 0
  else
    echo ""
    echo "Dependencies changed — the source image's deps would not match the current commit."
    echo "Options:"
    echo "  1. Use build-all mode to rebuild with current dependencies."
    echo "  2. Set allow_dependency_drift=true to override (not recommended)."
    exit 1
  fi
else
  echo "No dependency changes detected. Fast-rebuild is safe."
  select_overlay_variant
  exit 0
fi
