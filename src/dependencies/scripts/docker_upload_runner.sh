#!/bin/bash

# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This scripts takes a docker image that already contains the MaxText dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# For instructions on building and uploading the MaxText Docker image, please refer to the https://maxtext.readthedocs.io/en/latest/build_maxtext.html.

# Each time you update the `maxtext_base_image`` via `build_maxtext_docker_image`, there will be a slow upload process.
# However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

PACKAGE_DIR="${PACKAGE_DIR:-src}"
echo "PACKAGE_DIR: $PACKAGE_DIR"

set -e

export LOCAL_IMAGE_NAME=maxtext_base_image
export PROJECT=$(gcloud config get-value project)

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r RAW_KEY VALUE <<< "$ARGUMENT"
    KEY=$(echo "$RAW_KEY" | tr '[:lower:]' '[:upper:]')
    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done

export LOCAL_IMAGE_NAME_RUNNER=${LOCAL_IMAGE_NAME}__runner

echo "In PROJECT=$PROJECT, uploading local image ${LOCAL_IMAGE_NAME} to CLOUD_IMAGE_NAME as ${CLOUD_IMAGE_NAME}"

if [[ ! -v CLOUD_IMAGE_NAME ]]; then
  echo "Erroring out because CLOUD_IMAGE_NAME is unset, please set it!"
  exit 1
fi

# In the following sections we will want to check that the files that we are
# packaging in the Docker image don't have outside symbolic links.
# However, we want to exclude files and dirs listed in `.dockerignore` from this check

# Build find exclusion arguments
EXCLUDE_PATHS=()
if [ -f .dockerignore ]; then
  while IFS= read -r pattern; do
    # Ignore empty lines and comments
    if [[ -n "$pattern" && ! "$pattern" =~ ^# ]]; then
      # Strip leading ./ if present
      pattern="${pattern#./}"
      # Strip trailing / if present
      pattern="${pattern%/}"
      EXCLUDE_PATHS+=(-o -path "./$pattern")
    fi
  done < .dockerignore
fi

PRUNE_ARGS=()
if [ ${#EXCLUDE_PATHS[@]} -gt 0 ]; then
  # Remove leading -o and create the prune expression for find
  # This tells find not to descend into these paths.
  PRUNE_ARGS=(\( "${EXCLUDE_PATHS[@]:1}" \) -prune -o)
fi

# Check for dangling symbolic links (target does not exist), excluding .dockerignore paths.
DANGLING_LINKS=$(find -L . "${PRUNE_ARGS[@]}" -type l -print)
if [ -n "$DANGLING_LINKS" ]; then
  echo "ERROR: Found dangling symbolic links in the build context:"
  echo "$DANGLING_LINKS"
  echo ""
  echo "These can cause 'failed to compute cache key' errors during 'docker build'."
  echo "To fix this, please either:"
  echo "  1. Remove or fix these links."
  echo "  2. Add the following lines to your .dockerignore file:"
  while IFS= read -r link; do
    if [ -n "$link" ]; then
      echo "     ${link#./}"
    fi
  done <<< "$DANGLING_LINKS"
  exit 1
fi

# Check for absolute symbolic links, which Docker can't follow outside the build context.
ABSOLUTE_LINKS=$(find . "${PRUNE_ARGS[@]}" -type l -lname '/*' -print)
if [ -n "$ABSOLUTE_LINKS" ]; then
  echo "ERROR: Found symbolic links with absolute paths in the build context:"
  echo "$ABSOLUTE_LINKS"
  echo ""
  echo "Docker cannot follow absolute paths outside of the build context, which causes 'failed to compute cache key' errors."
  echo "To fix this, please either:"
  echo "  1. Remove these links from your local directory."
  echo "  2. Add the following lines to your .dockerignore file:"
  while IFS= read -r link; do
    if [ -n "$link" ]; then
      echo "     ${link#./}"
    fi
  done <<< "$ABSOLUTE_LINKS"
  exit 1
fi

# Check if the base image exists locally
if ! docker image inspect "${LOCAL_IMAGE_NAME}" &> /dev/null; then
  echo "ERROR: Base image '${LOCAL_IMAGE_NAME}' not found locally."
  echo "Please build it first by running 'build_maxtext_docker_image'."
  exit 1
fi

docker build --no-cache --build-arg BASEIMAGE=${LOCAL_IMAGE_NAME} \
             --build-arg PACKAGE_DIR=${PACKAGE_DIR} \
             -f "$PACKAGE_DIR"'/dependencies/dockerfiles/maxtext_runner.Dockerfile' \
             -t ${LOCAL_IMAGE_NAME_RUNNER} .

# Determine the full target image path
export FULL_IMAGE_PATH=""
if [[ "${CLOUD_IMAGE_NAME}" == *"/"* ]]; then
  # If it contains '/', assume it's a full path (e.g., us-docker.pkg.dev/project/repo/image)
  export FULL_IMAGE_PATH="${CLOUD_IMAGE_NAME}"
else
  # Otherwise, default to GCR
  export FULL_IMAGE_PATH="gcr.io/${PROJECT}/${CLOUD_IMAGE_NAME}"
fi

# Append :latest if no tag is specified
if [[ "${FULL_IMAGE_PATH}" != *":"* ]]; then
  export FULL_IMAGE_PATH="${FULL_IMAGE_PATH}:latest"
fi

echo "Tagging local image ${LOCAL_IMAGE_NAME_RUNNER} as ${FULL_IMAGE_PATH}..."
docker tag ${LOCAL_IMAGE_NAME_RUNNER} ${FULL_IMAGE_PATH}

echo "Pushing image to registry..."
docker push ${FULL_IMAGE_PATH}

# Since 'set -e' is active, if we reach this point, the push was successful.
echo ""
echo "========================================================================================================="
echo "✅ SUCCESS: MaxText Docker image uploaded successfully!"
echo "========================================================================================================="
echo "Your image is available at:"
echo "👉 ${FULL_IMAGE_PATH}"
echo ""
echo "You can copy-paste the path above directly into your XPK or GKE workload configurations."
echo "========================================================================================================="
