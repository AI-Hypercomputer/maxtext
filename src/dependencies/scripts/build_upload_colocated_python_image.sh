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
#   PROJECT: Overrides the default GCP project for gcr.io. Defaults to 'cloud-tpu-multipod-dev'.
#   LOCAL_IMAGE_NAME: Overrides the local tag used during docker build. Defaults to 'maxtext-colocated-python'.
#   IMAGE_LOCATION: The full URL for the final image in the registry.
#                   Defaults to gcr.io/${PROJECT}/${USER}_${LOCAL_IMAGE_NAME}:latest.
#                   Setting this variable fully overrides the default construction using PROJECT and LOCAL_IMAGE_NAME.
# For instructions on building the MaxText Docker image, please refer to the https://maxtext.readthedocs.io/en/latest/build_maxtext.html.

# Example 1: Use defaults for PROJECT and LOCAL_IMAGE_NAME
# bash src/dependencies/scripts/build_upload_colocated_python_image.sh

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
: "${PROJECT:=cloud-tpu-multipod-dev}"
: "${LOCAL_IMAGE_NAME:=maxtext-colocated-python}"
: "${IMAGE_LOCATION:=gcr.io/${PROJECT}/${USER}_${LOCAL_IMAGE_NAME}:latest}"

echo "$(date): Building and pushing MaxText Colocated Python image..."
echo "  PROJECT: ${PROJECT}"
echo "  LOCAL_IMAGE_NAME: ${LOCAL_IMAGE_NAME}"
echo "  IMAGE_LOCATION: ${IMAGE_LOCATION}"
echo "  Dockerfile: src/dependencies/dockerfiles/colocated_python.Dockerfile"
echo "  Build Context: maxtext/"

# Extract JAX Version from requirements.txt
echo "$(date): Extracting JAX version from requirements..."
REQ_FILE="src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Error: Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi
# Extracts version like "0.10.0" from lines like "jax==0.10.0"
JAX_VERSION=$(grep "^jax>=" "${REQ_FILE}" | head -1 | sed -E 's/.*>=([0-9.]+).*/\1/')

if [[ -z "${JAX_VERSION}" ]]; then
  echo "Error: Could not extract jax version from ${REQ_FILE}. Ensure it's in the format 'jax==X.Y.Z'." >&2
  exit 1
fi
echo "  Detected required JAX version: ${JAX_VERSION}"

# Find the Latest Compatible Base Image Tag
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

# Create a Temporary Dockerfile with the Dynamic Base Image
ORIGINAL_DOCKERFILE="src/dependencies/dockerfiles/colocated_python.Dockerfile"
TMP_DOCKERFILE=$(mktemp maxtext_colocated_python_Dockerfile.XXXXXX)
# Ensure the temporary file is removed on script exit
trap 'rm -f "${TMP_DOCKERFILE}"' EXIT

echo "$(date): Creating temporary Dockerfile: ${TMP_DOCKERFILE}"
# Replace the hardcoded FROM line with the dynamically determined base image
sed "s|^FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:.*|FROM ${FULL_BASE_IMAGE}|" "${ORIGINAL_DOCKERFILE}" > "${TMP_DOCKERFILE}"

echo "$(date): Running docker build with local tag '${LOCAL_IMAGE_NAME}' using ${TMP_DOCKERFILE}..."
# The build context '.' is the maxtext/ root directory.
# The Dockerfile should contain 'COPY . /app/maxtext/'.
docker build --no-cache \
  -f "${TMP_DOCKERFILE}" \
  -t "${LOCAL_IMAGE_NAME}" \
  .

# Tag the locally built image with the final IMAGE_LOCATION.
echo "$(date): Tagging '${LOCAL_IMAGE_NAME}' as '${IMAGE_LOCATION}'..."
docker tag "${LOCAL_IMAGE_NAME}" "${IMAGE_LOCATION}"

# Push the image to the specified IMAGE_LOCATION.
echo "$(date): Pushing '${IMAGE_LOCATION}'..."
docker push "${IMAGE_LOCATION}"

# Clean up the local Docker image tag used during build.
echo "$(date): Cleaning up local tag '${LOCAL_IMAGE_NAME}'..."
docker image rm "${LOCAL_IMAGE_NAME}"

echo "$(date): Build and push complete for ${IMAGE_LOCATION}"