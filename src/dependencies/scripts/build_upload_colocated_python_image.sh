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

# GCP Project for the image registry.
: "${PROJECT:=cloud-tpu-multipod-dev}"

# The base name for the local Docker image tag.
: "${LOCAL_IMAGE_NAME:=maxtext-colocated-python}"

# The full URI for the final image location.
# Defaults to gcr.io/${PROJECT}/${USER}_${LOCAL_IMAGE_NAME}:latest
# if IMAGE_LOCATION is not already set.
: "${IMAGE_LOCATION:=gcr.io/${PROJECT}/${USER}_${LOCAL_IMAGE_NAME}:latest}"

echo "$(date): Building and pushing MaxText Colocated Python image..."
echo "  PROJECT: ${PROJECT}"
echo "  LOCAL_IMAGE_NAME: ${LOCAL_IMAGE_NAME}"
echo "  IMAGE_LOCATION: ${IMAGE_LOCATION}"
echo "  Dockerfile: src/dependencies/dockerfiles/colocated_python.Dockerfile"
echo "  Build Context: maxtext/"

# Extract the registry from IMAGE_LOCATION (e.g., gcr.io or us-docker.pkg.dev)
REGISTRY=$(echo "${IMAGE_LOCATION}" | cut -d/ -f1)

# Build the Docker image. The build context is the current directory (maxtext/).
# The Dockerfile should contain 'COPY . /app/maxtext/'.
# Tag the image locally with ${LOCAL_IMAGE_NAME}.
echo "$(date): Running docker build with local tag '${LOCAL_IMAGE_NAME}'..."
docker build --no-cache \
  -f src/dependencies/dockerfiles/colocated_python.Dockerfile \
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