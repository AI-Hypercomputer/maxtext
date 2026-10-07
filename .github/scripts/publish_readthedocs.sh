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

# Publishes versioned MaxText documentation on Read the Docs via API v3.
# Waits for PyPI CDN propagation first so .readthedocs.yml pre_build checks pass.
#
# Required environment variables:
#   READTHEDOCS_TOKEN - Read the Docs API v3 token
#   PYPI_VERSION      - MaxText package version (e.g., 0.2.6)

set -euo pipefail

if [[ -z "${READTHEDOCS_TOKEN:-}" ]]; then
  echo "Error: READTHEDOCS_TOKEN secret is not set."
  exit 1
fi
PYPI_VERSION="${PYPI_VERSION:?Error: PYPI_VERSION is required.}"

RTD_PROJECT="maxtext"
RTD_VERSION="maxtext-v${PYPI_VERSION}"
API_BASE="https://app.readthedocs.org/api/v3/projects/${RTD_PROJECT}"

# Wait for PyPI CDN propagation so .readthedocs.yml pre_build check sees HTTP 200.
echo "Waiting for maxtext ${PYPI_VERSION} to become visible on PyPI..."
for i in {1..24}; do
  PYPI_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/maxtext/${PYPI_VERSION}/json") || PYPI_STATUS="000"
  if [[ "${PYPI_STATUS}" == "200" ]]; then
    echo "Confirmed maxtext ${PYPI_VERSION} is live on PyPI."
    break
  fi
  if [[ "${i}" == "24" ]]; then
    echo "Error: Timed out waiting for maxtext ${PYPI_VERSION} on PyPI (HTTP ${PYPI_STATUS})."
    exit 1
  fi
  sleep 5
done

echo "Syncing versions on Read the Docs..."
curl -f -sS --retry 3 --retry-delay 5 -X POST \
  -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
  "${API_BASE}/sync-versions/"

for i in {1..24}; do
  STATUS=$(curl -sS -o /dev/null -w "%{http_code}" \
    -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
    "${API_BASE}/versions/${RTD_VERSION}/") || STATUS="000"
  if [[ "${STATUS}" == "200" ]]; then
    break
  fi
  if [[ "${i}" == "24" ]]; then
    echo "Error: Timed out waiting for version ${RTD_VERSION} to appear in Read the Docs (HTTP ${STATUS})."
    exit 1
  fi
  echo "Waiting for version ${RTD_VERSION} to appear in Read the Docs (attempt ${i}/24)..."
  sleep 5
done

IS_ACTIVE=$(curl -f -sS --retry 3 --retry-delay 5 \
  -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
  "${API_BASE}/versions/${RTD_VERSION}/" | \
  python3 -c "import sys, json; print(json.load(sys.stdin)['active'])")

BUILD_ID=""
if [[ "${IS_ACTIVE}" != "True" ]]; then
  echo "Activating version ${RTD_VERSION} on Read the Docs..."
  curl -f -sS --retry 3 --retry-delay 5 -X PATCH \
    -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"active": true, "hidden": false}' \
    "${API_BASE}/versions/${RTD_VERSION}/"
  sleep 5
  BUILD_ID=$(curl -f -sS --retry 3 --retry-delay 5 \
    -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
    "${API_BASE}/versions/${RTD_VERSION}/builds/?running=true&limit=1" | \
    python3 -c "import sys, json; results = json.load(sys.stdin).get('results', []); print(results[0]['id'] if results else '')")
fi

if [[ -z "${BUILD_ID}" ]]; then
  echo "Triggering Read the Docs build for ${RTD_VERSION}..."
  BUILD_RESPONSE=$(curl -f -sS --retry 3 --retry-delay 5 -X POST \
    -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
    "${API_BASE}/versions/${RTD_VERSION}/builds/")
  BUILD_ID=$(echo "${BUILD_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin)['build']['id'])")
fi

echo "Waiting for Read the Docs build ${BUILD_ID} to finish..."
while true; do
  BUILD_STATUS=$(curl -f -sS --retry 3 --retry-delay 5 \
    -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
    "${API_BASE}/builds/${BUILD_ID}/")
  STATE=$(echo "${BUILD_STATUS}" | python3 -c "import sys, json; print(json.load(sys.stdin)['state']['code'])")
  SUCCESS=$(echo "${BUILD_STATUS}" | python3 -c "import sys, json; print(json.load(sys.stdin)['success'])")

  echo "Build ${BUILD_ID} state: ${STATE} (success: ${SUCCESS})"
  if [[ "${STATE}" == "finished" ]]; then
    if [[ "${SUCCESS}" == "True" ]]; then
      echo "Read the Docs build ${BUILD_ID} succeeded."
      break
    else
      echo "Error: Read the Docs build ${BUILD_ID} failed."
      exit 1
    fi
  elif [[ "${STATE}" == "cancelled" ]]; then
    echo "Error: Read the Docs build ${BUILD_ID} was cancelled."
    exit 1
  fi
  sleep 15
done

echo "Setting ${RTD_VERSION} as the default version on Read the Docs..."
curl -f -sS --retry 3 --retry-delay 5 -X PATCH \
  -H "Authorization: Token ${READTHEDOCS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"default_version\": \"${RTD_VERSION}\"}" \
  "${API_BASE}/"
echo "Successfully published and set ${RTD_VERSION} as default on Read the Docs."
