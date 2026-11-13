#!/bin/bash

# Copyright 2023-2026 Google LLC
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

# Resolves RELEASE_SHA, RC_SHA (parent 7-char SHA), and PACKAGE_VERSION for a
# release tag, validating tag naming, and PyPI availability.

set -euo pipefail

RELEASE_TAG="${RELEASE_TAG:?Error: RELEASE_TAG is required.}"
MODE="${MODE:-dry-run}"
case "${MODE}" in
  dry-run|publish|force-publish) ;;
  *) echo "Error: Invalid MODE '${MODE}' (expected dry-run, publish, or force-publish)."; exit 1 ;;
esac

# On real publishes, fetch explicitly from refs/tags/ so branches cannot shadow tags.
if [[ "${MODE}" != "dry-run" ]]; then
  git fetch --no-tags --depth=2 origin "+refs/tags/${RELEASE_TAG}:refs/tags/${RELEASE_TAG}"
else
  git fetch --no-tags --depth=2 origin "${RELEASE_TAG}"
fi

RELEASE_SHA=$(git rev-parse "FETCH_HEAD^{commit}")
RC_FULL_SHA=$(git rev-parse "${RELEASE_SHA}~1")
RC_SHA="${RC_FULL_SHA:0:7}"

PACKAGE_VERSION=$(git show "${RELEASE_SHA}:src/maxtext/version.py" | sed -nE "s/^__version__[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"].*/\1/p")
: "${PACKAGE_VERSION:?Error: Could not parse __version__ from src/maxtext/version.py.}"

echo "Release Commit: ${RELEASE_SHA} (version: ${PACKAGE_VERSION}, RC image tag: ${RC_SHA})"

# Ensure RELEASE_TAG matches maxtext-v<version> in version.py (allow branch/SHA names only during dry-runs).
if [[ "${MODE}" != "dry-run" || "${RELEASE_TAG}" == maxtext-v* ]] && [[ "${RELEASE_TAG}" != "maxtext-v${PACKAGE_VERSION}" ]]; then
  echo "Error: Release tag '${RELEASE_TAG}' does not match version '${PACKAGE_VERSION}' (expected 'maxtext-v${PACKAGE_VERSION}')."
  exit 1
fi

# On real publishes, fail fast before manual approval if this version is already published on PyPI.
if [[ "${MODE}" != "dry-run" ]]; then
  HTTP_STATUS=$(curl -sS --retry 3 --retry-delay 5 -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/maxtext/${PACKAGE_VERSION}/json")
  if [[ "${HTTP_STATUS}" == "200" ]]; then
    echo "Error: maxtext ${PACKAGE_VERSION} already exists on PyPI."
    exit 1
  fi
fi

# Ensure the release commit has identical runtime code to its parent RC_SHA (certified Docker image) by allowing changes only to version.py and release_notes.md.
EXPECTED_FILES=$'docs/release_notes.md\nsrc/maxtext/version.py'
CHANGED_FILES=$(git diff --name-only "${RC_FULL_SHA}" "${RELEASE_SHA}" | LC_ALL=C sort)
if [[ "${CHANGED_FILES}" != "${EXPECTED_FILES}" ]]; then
  ACTUAL_FILES="${CHANGED_FILES//$'\n'/, }"
  MSG="Release commit must modify exactly docs/release_notes.md and src/maxtext/version.py (found: ${ACTUAL_FILES:-none})."
  if [[ "${MODE}" == "force-publish" ]]; then
    echo "::warning::${MSG} Proceeding because force-publish is enabled."
  else
    echo "Error: ${MSG}"
    exit 1
  fi
fi

{
  echo "release_sha=${RELEASE_SHA}"
  echo "rc_sha=${RC_SHA}"
  echo "package_version=${PACKAGE_VERSION}"
} >> "${GITHUB_OUTPUT:-/dev/null}"
