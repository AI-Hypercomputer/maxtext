#!/bin/bash

# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to build and upload the MaxText colocated Python Docker image.
# This script should be run from the root directory of the MaxText repository.
#
# Arguments can be provided as KEY=VALUE pairs, e.g.:
#   bash build_upload_colocated_python_image.sh PROJECT=my-gcp LOCAL_IMAGE_NAME=my-colocated
#
# Supported KEYs:
#   PROJECT: GCP project for gcr.io. 
#   LOCAL_IMAGE_NAME: Overrides the local tag used during docker build. Defaults to 'maxtext-colocated-python'.
#   IMAGE_LOCATION: The full URL for the final image in the registry.
#                   Defaults to gcr.io/${PROJECT}/${USER}_${LOCAL_IMAGE_NAME}:latest.
#                   Setting this variable fully overrides the default construction using PROJECT and LOCAL_IMAGE_NAME.

# Example 1: Use defaults for PROJECT and LOCAL_IMAGE_NAME
# bash src/dependencies/scripts/build_upload_colocated_python_image.sh PROJECT=my-tpu-dev

# Example 2: Specify a custom project and local name
# bash src/dependencies/scripts/build_upload_colocated_python_image.sh PROJECT=my-tpu-dev LOCAL_IMAGE_NAME=maxtext-cp

# Example 3: Provide a complete IMAGE_LOCATION, overriding defaults
# bash src/dependencies/scripts/build_upload_colocated_python_image.sh IMAGE_LOCATION=us-docker.pkg.dev/my-artifact-repo/images/colocated-python:stable

for ARGUMENT in "$@"; do
    if [[ "$ARGUMENT" == *"="* ]]; then
        IFS='=' read -r RAW_KEY VALUE <<< "$ARGUMENT"
        KEY=$(echo "$RAW_KEY" | tr '[:lower:]' '[:upper:]')
        export "$KEY"="$VALUE"
        echo "  Parsed: $KEY=$VALUE"
    else
        echo "Warning: Ignoring argument '$ARGUMENT'. Arguments should be in KEY=VALUE format (e.g., PROJECT=my-proj)."
    fi
done

# Set default values using parameter expansion.
: "${LOCAL_IMAGE_NAME:=maxtext-colocated-python}"
if [[ -z "${IMAGE_LOCATION}" ]]; then
  # IMAGE_LOCATION was not provided, so PROJECT is required for the default.
  if [[ -z "${PROJECT}" ]]; then
    echo "Error: Either PROJECT or IMAGE_LOCATION must be provided." >&2
    echo "  - To use a default location in GCR, set PROJECT=your-gcp-project-id." >&2
    echo "  - To specify a full custom location, set IMAGE_LOCATION=your/registry/path/image:tag." >&2
    exit 1
  fi
  IMAGE_LOCATION="gcr.io/${PROJECT}/${USER}_${LOCAL_IMAGE_NAME}:latest"
fi

DOCKERFILE_TEMPLATE="src/dependencies/dockerfiles/colocated_python.Dockerfile.template"
CONTEXT="maxtext/"

echo "$(date): Building and pushing MaxText Colocated Python image..."
echo "  PROJECT: ${PROJECT}"
echo "  LOCAL_IMAGE_NAME: ${LOCAL_IMAGE_NAME}"
echo "  IMAGE_LOCATION: ${IMAGE_LOCATION}"
echo "  Dockerfile Template: ${DOCKERFILE_TEMPLATE}"
echo "  Build Context: ${CONTEXT}"

# --- Step 1: Extract JAX Version from requirements.txt ---
echo "$(date): Extracting JAX version from requirements..."
REQ_FILE="src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Error: Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi
# Extracts version like "0.10.0" from lines like "jax>=0.10.0" or "jax==0.10.0"
JAX_VERSION=$(grep -E "^jax(>=|==)" "${REQ_FILE}" | head -1 | sed -E 's/^jax(>=|==)([0-9.]+).*/\2/')

if [[ -z "${JAX_VERSION}" ]]; then
  echo "Error: Could not extract jax version from ${REQ_FILE}. Ensure it's in the format 'jax>=X.Y.Z' or 'jax==X.Y.Z'." >&2
  exit 1
fi
echo "  Detected required JAX version: ${JAX_VERSION}"

# --- Step 2: Find the Latest Compatible Base Image Tag ---
BASE_IMAGE_REPO="us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar"
TARGET_JAX_TAG_PART="jax_${JAX_VERSION}"
echo "$(date): Searching for base image tag in '${BASE_IMAGE_REPO}' containing '${TARGET_JAX_TAG_PART}'..."

# Authenticate Docker for the base image registry (us-docker.pkg.dev)
BASE_REGISTRY=$(echo "${BASE_IMAGE_REPO}" | cut -d/ -f1)
echo "$(date): Configuring Docker for base image registry: ${BASE_REGISTRY}"
gcloud auth configure-docker --quiet "${BASE_REGISTRY}"

# List tags, filter by JAX version, sort by date (desc), and take the latest.
BASE_IMAGE_TAG=$(gcloud artifacts docker images list "${BASE_IMAGE_REPO}" --include-tags --format=json | \
                 jq -r '.[] | .tags[]?' | \
                 grep "${TARGET_JAX_TAG_PART}" | \
                 sort -r | \
                 head -1)

if [[ -z "${BASE_IMAGE_TAG}" ]]; then
  echo "Error: Could not find a suitable base image tag in ${BASE_IMAGE_REPO} for JAX version '${JAX_VERSION}'." >&2
  echo "  Searched for tags containing '${TARGET_JAX_TAG_PART}'." >&2
  echo "  Available matching tags found:" >&2
  gcloud artifacts docker images list "${BASE_IMAGE_REPO}" --include-tags --format=json | \
    jq -r '.[] | .tags[]?' | sort -r >&2
  exit 1
fi
FULL_BASE_IMAGE="${BASE_IMAGE_REPO}:${BASE_IMAGE_TAG}"
echo "  Found latest compatible base image: ${FULL_BASE_IMAGE}"

# --- Step 3: Build the Docker image using --build-arg ---
echo "$(date): Running docker build with local tag '${LOCAL_IMAGE_NAME}' using ${TMP_DOCKERFILE}..."
docker build --no-cache \
  -f ${DOCKERFILE_TEMPLATE} \
  --build-arg BASE_IMAGE="${FULL_BASE_IMAGE}" \
  -t "${LOCAL_IMAGE_NAME}" \
  .

# --- Step 4: Tag and Push the image ---
echo "$(date): Tagging '${LOCAL_IMAGE_NAME}' as '${IMAGE_LOCATION}'..."
docker tag "${LOCAL_IMAGE_NAME}" "${IMAGE_LOCATION}"

echo "$(date): Pushing '${IMAGE_LOCATION}'..."
docker push "${IMAGE_LOCATION}"

# --- Step 5: Cleanup ---
echo "$(date): Cleaning up local tag '${LOCAL_IMAGE_NAME}'..."
docker image rm "${LOCAL_IMAGE_NAME}"

echo "$(date): Build and push complete for ${IMAGE_LOCATION}"
