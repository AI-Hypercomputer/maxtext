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

# Description:
# bash build_docker_image.sh MODE=stable
# bash build_docker_image.sh MODE=nightly
# bash build_docker_image.sh MODE=head

# bash build_docker_image.sh MODE=stable
# bash build_docker_image.sh MODE=stable JAX_VERSION=0.4.13
# bash build_docker_image.sh MODE=stable JAX_VERSION=0.4.13 IMAGE_PREFIX=jax0.4.13


# Enable "exit immediately if any command fails" option
set -e

export PROJECT=$(gcloud config get-value project)
export IMAGE_PREFIX=gke_img

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done
echo "PROJECT=$PROJECT"

if [[ "$MODE" == "stable" || ! -v MODE ]]; then
    # Stable mode
  if [[ ! -v JAX_VERSION ]]; then
    docker build --build-arg MODE="stable" -f ./maxtext.Dockerfile -t ${IMAGE_PREFIX}_maxtext_stable .
  else
    docker build --build-arg MODE="stable" --build-arg JAX_VERSION="$JAX_VERSION" -f ./maxtext.Dockerfile -t ${IMAGE_PREFIX}_maxtext_stable .
  fi
  docker tag ${IMAGE_PREFIX}_maxtext_stable gcr.io/$PROJECT/${IMAGE_PREFIX}_maxtext_stable:latest
  sudo docker push gcr.io/$PROJECT/${IMAGE_PREFIX}_maxtext_stable:latest
elif [[ $MODE == "nightly" ]]; then
  # Nightly mode
  docker build --build-arg MODE="nightly" -f ./maxtext.Dockerfile -t ${IMAGE_PREFIX}_maxtext_nightly .
  docker tag ${IMAGE_PREFIX}_maxtext_nightly gcr.io/$PROJECT/${IMAGE_PREFIX}_maxtext_nightly:latest
  docker push gcr.io/$PROJECT/${IMAGE_PREFIX}_maxtext_nightly:latest
elif [[ $MODE == "head" ]]; then
  # Head mode
  docker build --build-arg MODE="head" -f ./maxtext.Dockerfile -t ${IMAGE_PREFIX}_maxtext_head .
  docker tag ${IMAGE_PREFIX}_maxtext_head gcr.io/$PROJECT/${IMAGE_PREFIX}_maxtext_head:latest
  docker push gcr.io/$PROJECT/${IMAGE_PREFIX}_maxtext_head:latest
fi
