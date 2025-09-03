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

set -e

export LOCAL_IMAGE_NAME=src/MaxText_base_image
export PROJECT=$(gcloud config get-value project)

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

export LOCAL_IMAGE_NAME_RUNNER=${LOCAL_IMAGE_NAME}__runner

echo "In PROJECT=$PROJECT, uploading local image ${LOCAL_IMAGE_NAME} to CLOUD_IMAGE_NAME as ${CLOUD_IMAGE_NAME}"

if [[ ! -v CLOUD_IMAGE_NAME ]]; then
  echo "Erroring out because CLOUD_IMAGE_NAME is unset, please set it!"
  exit 1
fi

# Download other test assets from GCS into "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/test_assets
if ! gcloud storage cp gs://src/MaxText-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/test_assets}"; then
  echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."
fi

docker build --build-arg BASEIMAGE=${LOCAL_IMAGE_NAME} -f ./src/MaxText_runner.Dockerfile -t ${LOCAL_IMAGE_NAME_RUNNER} .

docker tag ${LOCAL_IMAGE_NAME_RUNNER} gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest

echo "All done, check out your artifacts at: gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}"
