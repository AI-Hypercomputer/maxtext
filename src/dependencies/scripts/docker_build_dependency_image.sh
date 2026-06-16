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

# This script is used to build the MaxText Docker image, supporting
# different environments (stable, nightly) and use cases (pre-training, post-training).
# IMPORTANT: This script must be executed from the root directory of the MaxText repository.

# For instructions on building the MaxText Docker image, please refer to the https://maxtext.readthedocs.io/en/latest/build_maxtext.html.

PACKAGE_DIR="${PACKAGE_DIR:-src}"
echo "PACKAGE_DIR: $PACKAGE_DIR"

# Enable "exit immediately if any command fails" option
set -e

# Check for docker permissions
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Permission denied while trying to connect to the Docker daemon." >&2
  echo "You can fix this by:" >&2
  echo "1. Running this script with sudo: 'sudo bash $0 $@'" >&2
  echo "2. Adding your user to the 'docker' group: 'sudo usermod -aG docker \${USER}' (requires a new login session)." >&2
  echo "3. Running `newgrp docker` in your current terminal." >&2
  exit 1
fi

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

export LOCAL_IMAGE_NAME=maxtext_base_image
echo "Building docker image: $LOCAL_IMAGE_NAME. This will take a few minutes but the image can be reused as you iterate."

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r RAW_KEY VALUE <<< "$ARGUMENT"
    KEY=$(echo "$RAW_KEY" | tr '[:lower:]' '[:upper:]')
    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done

# Set default values if not provided
if [[ -z ${JAX_VERSION+x} ]] ; then
  export JAX_VERSION=NONE
fi
if [[ -z ${MODE} ]]; then
  export MODE=stable
fi
if [[ -z ${DEVICE} ]]; then
  export DEVICE=tpu
fi
if [[ -z ${WORKFLOW} ]]; then
  export WORKFLOW=pre-training
fi
if [[ -z ${INCLUDE_TEST_ASSETS} ]]; then
  export INCLUDE_TEST_ASSETS=false
fi

# Create docker build arguments array
docker_build_args=(
  "DEVICE=${DEVICE}"
  "WORKFLOW=${WORKFLOW}"
  "MODE=${MODE}"
  "JAX_VERSION=${JAX_VERSION}"
  "PACKAGE_DIR=${PACKAGE_DIR}"
  "INCLUDE_TEST_ASSETS=${INCLUDE_TEST_ASSETS}"
)

run_docker_build() {
  local dockerfile_path="$1"
  shift 1 # Move past the first argument, the rest are build-args
  docker build --network host $(printf -- '--build-arg %q ' "$@") -f "$dockerfile_path" -t "$LOCAL_IMAGE_NAME" .
}

# Function to build image for GPUs
build_gpu_image() {
  if [[ ${MODE} == "pinned" ]]; then
    local base_image=ghcr.io/nvidia/jax:base-2024-12-04
    docker_build_args+=("BASEIMAGE=${base_image}")
  fi

  echo "Building docker image with arguments: ${docker_build_args[*]}"
  run_docker_build "$PACKAGE_DIR/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile" "${docker_build_args[@]}"
}

# Function to build image for TPUs
build_tpu_image() {
  if [[ -n "$LIBTPU_VERSION" ]]; then
    docker_build_args+=("LIBTPU_VERSION=${LIBTPU_VERSION}")
  else
    docker_build_args+=("LIBTPU_VERSION=NONE")
  fi

  if [[ ${MANTARAY} == "true" ]]; then
    local base_image=gcr.io/tpu-prod-env-one-vm/benchmark-db:2025-02-14
    docker_build_args+=("BASEIMAGE=${base_image}")
  fi

  echo "Building docker image with arguments: ${docker_build_args[*]}"
  run_docker_build "$PACKAGE_DIR/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile" "${docker_build_args[@]}"
}

if [[ ${DEVICE} == "gpu" ]]; then
  build_gpu_image
else
  build_tpu_image
fi

echo ""
echo "*************************"
echo ""

echo "Built your base docker image and named it ${LOCAL_IMAGE_NAME}.
It only has the dependencies installed. Assuming you're on a TPUVM, to run the
docker image locally and mirror your local working directory run:"
echo "docker run -v $(pwd):/deps --rm -it --privileged --entrypoint bash ${LOCAL_IMAGE_NAME}"
echo ""
echo "You can run MaxText and your development tests inside of the docker image. Changes to your workspace will automatically
be reflected inside the docker container."
echo "Once you want to upload your docker container to GCR, run 'upload_maxtext_docker_image CLOUD_IMAGE_NAME=your_image_name'."
