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

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Example command:
# bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner

if [ "${BASH_SOURCE-}" ]; then
  this_file="${BASH_SOURCE[0]}"
elif [ "${ZSH_VERSION-}" ]; then
  # shellcheck disable=SC2296
  this_file="${(%):-%x}"
else
  this_file="${0}"
fi

MAXTEXT_REPO_ROOT="${MAXTEXT_REPO_ROOT:-$(CDPATH='' cd -- "$(dirname -- "${this_file}")"'/../..' && pwd)}"

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

# Check for dangling symbolic links (target does not exist).
DANGLING_LINKS=$(find -L . -type l)
if [ -n "$DANGLING_LINKS" ]; then
  echo "ERROR: Found dangling symbolic links in the build context:"
  echo "$DANGLING_LINKS"
  echo "These can cause 'failed to compute cache key' errors during 'docker build'."
  echo "Please remove or fix them before building the Docker image."
  exit 1
fi

# Check for absolute symbolic links, which Docker can't follow outside the build context.
ABSOLUTE_LINKS=$(find . -type l -lname '/*')
if [ -n "$ABSOLUTE_LINKS" ]; then
  echo "ERROR: Found symbolic links with absolute paths in the build context:"
  echo "$ABSOLUTE_LINKS"
  echo "Docker cannot follow absolute paths outside of the build context, which can cause 'failed to compute cache key' errors."
  echo "Please remove these links or convert them to relative paths before building the Docker image."
  exit 1
fi

# Download other test assets from GCS into "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/test_assets
# if ! gcloud storage cp gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/test_assets}"; then
#   echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."
# fi

# Check if the base image exists locally
if ! docker image inspect "${LOCAL_IMAGE_NAME}" &> /dev/null; then
  echo "ERROR: Base image '${LOCAL_IMAGE_NAME}' not found locally."
  echo "Please build it first by running 'bash docker_build_dependency_image.sh'."
  exit 1
fi

docker build --no-cache --build-arg BASEIMAGE=${LOCAL_IMAGE_NAME} \
             -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_runner.Dockerfile' \
             -t ${LOCAL_IMAGE_NAME_RUNNER} .

docker tag ${LOCAL_IMAGE_NAME_RUNNER} gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest

echo "All done, check out your artifacts at: gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}"
