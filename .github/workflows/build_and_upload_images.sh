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

# This script builds and uploads two images - one with only dependencies, the other also has a code snapshot.
# These images are tagged in GCR with both "latest" and date in format YYYY-MM-DD via $(date +%Y-%m-%d)

# This script wraps several steps meant for the github runners to push nightly images.
# If you want to create and push your own images you should instead use docker_build_dependency_image and
# docker_upload_runner in the maxtext root directory.

# Example command:
# bash build_and_upload_images.sh PROJECT=<project> MODE=stable DEVICE=tpu CLOUD_IMAGE_NAME=${USER}_runner

if [ "${BASH_SOURCE-}" ]; then
  this_file="${BASH_SOURCE[0]}"
elif [ "${ZSH_VERSION-}" ]; then
  # shellcheck disable=SC2296
  this_file="${(%):-%x}"
else
  this_file="${0}"
fi

MAXTEXT_REPO_ROOT="${MAXTEXT_REPO_ROOT:-$(CDPATH='' cd -- "$(dirname -- "${this_file}")"'/../..' && pwd)}"
export LOCAL_IMAGE_NAME=maxtext_base_image

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ ! -v CLOUD_IMAGE_NAME ]] || [[ ! -v PROJECT ]] || [[ ! -v MODE ]] || [[ ! -v DEVICE ]]; then
  echo "You must set CLOUD_IMAGE_NAME, PROJECT, MODE, and DEVICE"
  exit 1
fi

gcloud auth configure-docker us-docker.pkg.dev --quiet
bash "$MAXTEXT_REPO_ROOT"'/dependencies/scripts/docker_build_dependency_image.sh' LOCAL_IMAGE_NAME=$LOCAL_IMAGE_NAME MODE="$MODE" DEVICE="$DEVICE"
image_date=$(date +%Y-%m-%d)

# Upload only dependencies image
dependency_image_name=${CLOUD_IMAGE_NAME}_dependencies
docker tag ${LOCAL_IMAGE_NAME} gcr.io/$PROJECT/${dependency_image_name}:latest
docker push gcr.io/$PROJECT/${dependency_image_name}:latest
docker tag ${LOCAL_IMAGE_NAME} gcr.io/$PROJECT/${dependency_image_name}:${image_date}
docker push gcr.io/$PROJECT/${dependency_image_name}:${image_date}

# Download other test assets from GCS into ${MAXTEXT_TEST_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/tests/assets/golden_logits}
if ! gcloud storage cp gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/tests/assets/golden_logits}"; then
  echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."
fi

# Build then upload "dependencies + code" image
docker build --build-arg BASEIMAGE=${LOCAL_IMAGE_NAME} -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_runner.Dockerfile' -t ${LOCAL_IMAGE_NAME}_runner .
docker tag ${LOCAL_IMAGE_NAME}_runner gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
docker tag ${LOCAL_IMAGE_NAME}_runner gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:${image_date}
docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:${image_date}
