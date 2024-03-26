#!/bin/bash

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
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

bash docker_build_dependency_image.sh LOCAL_IMAGE_NAME=$LOCAL_IMAGE_NAME MODE=$MODE DEVICE=$DEVICE
gcloud auth configure-docker --quiet
image_date=$(date +%Y-%m-%d)

# Upload only dependencies image
dependency_image_name=${CLOUD_IMAGE_NAME}_dependencies
docker tag ${LOCAL_IMAGE_NAME} gcr.io/$PROJECT/${dependency_image_name}:latest
docker push gcr.io/$PROJECT/${dependency_image_name}:latest
docker tag ${LOCAL_IMAGE_NAME} gcr.io/$PROJECT/${dependency_image_name}:${image_date}
docker push gcr.io/$PROJECT/${dependency_image_name}:${image_date}

# Build then upload "dependencies + code" image
docker build --build-arg BASEIMAGE=${LOCAL_IMAGE_NAME} -f ./maxtext_runner.Dockerfile -t ${LOCAL_IMAGE_NAME}_runner .
docker tag ${LOCAL_IMAGE_NAME}_runner gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:latest
docker tag ${LOCAL_IMAGE_NAME}_runner gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:${image_date}
docker push gcr.io/$PROJECT/${CLOUD_IMAGE_NAME}:${image_date}