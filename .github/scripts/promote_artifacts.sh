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

# Promotes a Docker image version from a staging Artifact Registry repository
# to a production Artifact Registry repository via the :promoteArtifact API
# (Exit Gate) and waits for the long-running operation to complete.
#
# Usage:
#   .github/scripts/promote_artifacts.sh <project_name> <staging_repo> <prod_repo> <image_name> <image_digest> [location]

set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <project_name> <staging_repo> <prod_repo> <image_name> <image_digest> [location]" >&2
  exit 1
fi

PROJECT_NAME="$1"
STAGING_REPO="$2"
PROD_REPO="$3"
IMAGE_NAME="$4"
IMAGE_DIGEST="$5"
LOCATION="${6:-us}"

echo "Promoting ${IMAGE_NAME}@${IMAGE_DIGEST} from ${STAGING_REPO} to ${PROD_REPO} in project ${PROJECT_NAME}..."

ACCESS_TOKEN="$(gcloud auth print-access-token)"
PROMOTE_URL="https://artifactregistry.googleapis.com/v1/projects/${PROJECT_NAME}/locations/${LOCATION}/repositories/${PROD_REPO}:promoteArtifact"
SOURCE_REPO="projects/${PROJECT_NAME}/locations/${LOCATION}/repositories/${STAGING_REPO}"
SOURCE_VERSION="${SOURCE_REPO}/packages/${IMAGE_NAME}/versions/${IMAGE_DIGEST}"
export SOURCE_REPO SOURCE_VERSION

PAYLOAD=$(python3 -c 'import json, os
print(json.dumps({
    "source_repository": os.environ["SOURCE_REPO"],
    "source_version": os.environ["SOURCE_VERSION"],
    "attachment_behavior": "PUBLIC_BCID_VSA_ONLY",
}))')

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -X POST "${PROMOTE_URL}" \
  -d "${PAYLOAD}")

HTTP_STATUS=$(echo "${RESPONSE}" | tail -n 1)
BODY=$(echo "${RESPONSE}" | sed '$d')
echo "Promote API response (HTTP ${HTTP_STATUS}): ${BODY}"

if [[ "${HTTP_STATUS}" -lt 200 || "${HTTP_STATUS}" -ge 300 ]]; then
  if echo "${BODY}" | grep -qi "already exists"; then
    echo "Artifact version already exists in ${PROD_REPO}; skipping promotion."
    exit 0
  fi
  echo "Error: promoteArtifact API failed with HTTP ${HTTP_STATUS}" >&2
  exit 1
fi

OP_NAME=$(echo "${BODY}" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("name", ""))')
if [[ -z "${OP_NAME}" ]]; then
  echo "No operation name returned; promotion completed synchronously."
  exit 0
fi

echo "Waiting for promotion operation ${OP_NAME} to complete..."
for attempt in {1..60}; do
  OP_STATUS=$(curl -s \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    "https://artifactregistry.googleapis.com/v1/${OP_NAME}")
  read -r DONE HAS_ERR < <(python3 -c 'import json, sys; d = json.load(sys.stdin); print(str(d.get("done", False)).lower(), str("error" in d).lower())' <<< "${OP_STATUS}")
  if [[ "${DONE}" == "true" ]]; then
    echo "Final operation status: ${OP_STATUS}"
    if [[ "${HAS_ERR}" == "true" ]]; then
      if echo "${OP_STATUS}" | grep -qi "already exists"; then
        echo "Artifact version already exists in ${PROD_REPO}; skipping promotion."
        exit 0
      fi
      echo "Error: Promotion operation failed." >&2
      exit 1
    fi
    exit 0
  fi
  if [[ "${attempt}" -eq 60 ]]; then
    echo "Error: Timed out waiting for promotion operation ${OP_NAME} to complete." >&2
    exit 1
  fi
  echo "Promotion still in progress, retrying in 5 seconds..."
  sleep 5
done
