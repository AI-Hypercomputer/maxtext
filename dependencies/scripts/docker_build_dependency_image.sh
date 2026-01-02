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

# ==================================
# PRE-TRAINING BUILD EXAMPLES
# ==================================

# Build docker image with stable dependencies
## bash dependencies/scripts/docker_build_dependency_image.sh DEVICE={{gpu|tpu}} MODE=stable

# Build docker image with nightly dependencies
## bash dependencies/scripts/docker_build_dependency_image.sh DEVICE={{gpu|tpu}} MODE=nightly

# Build docker image with stable dependencies and, a pinned JAX_VERSION for TPUs
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=stable JAX_VERSION=0.4.13

# Build docker image with a pinned JAX_VERSION and, a pinned LIBTPU_VERSION for TPUs
## bash dependencies/scripts/docker_build_dependency_image.sh MODE={{stable|nightly}} JAX_VERSION=0.8.1 LIBTPU_VERSION=0.0.31.dev20251119+nightly

# Build docker image with a custom libtpu.so for TPUs
# Note: libtpu.so file must be present in the root directory of the MaxText repository
## bash dependencies/scripts/docker_build_dependency_image.sh MODE={{stable|nightly}}

# Build docker image with nightly dependencies and, a pinned JAX_VERSION for GPUs
# Available versions listed at https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax
## bash dependencies/scripts/docker_build_dependency_image.sh DEVICE=gpu MODE=nightly JAX_VERSION=0.4.36.dev20241109

# ==================================
# POST-TRAINING BUILD EXAMPLES
# ==================================

# Build docker image with stable pre-training dependencies and stable post-training dependencies
## bash dependencies/scripts/docker_build_dependency_image.sh WORKFLOW=post-training

# Build docker image with stable pre-training dependencies and post-training dependencies from GitHub head
## bash dependencies/scripts/docker_build_dependency_image.sh WORKFLOW=post-training POST_TRAINING_SOURCE=local

if [ "${BASH_SOURCE-}" ]; then
  this_file="${BASH_SOURCE[0]}"
elif [ "${ZSH_VERSION-}" ]; then
  # shellcheck disable=SC2296
  this_file="${(%):-%x}"
else
  this_file="${0}"
fi

MAXTEXT_REPO_ROOT="${MAXTEXT_REPO_ROOT:-$(CDPATH='' cd -- "$(dirname -- "${this_file}")"'/../..' && pwd)}"

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
# TODO: Remove 'custom_wheels' mode support when tpu-recipes migration is complete.
elif [[ ${MODE} == "custom_wheels" ]]; then
  export WORKFLOW=custom-wheels
  export MODE=nightly
fi
if [[ -z ${DEVICE} ]]; then
  export DEVICE=tpu
fi
if [[ -z ${WORKFLOW} ]]; then
  export WORKFLOW=pre-training
fi

# Create docker build arguments array
docker_build_args=(
  "DEVICE=${DEVICE}"
  "WORKFLOW=${WORKFLOW}"
  "MODE=${MODE}"
  "JAX_VERSION=${JAX_VERSION}"
)

run_docker_build() {
  local dockerfile_path="$1"
  shift 1 # Move past the first argument, the rest are build-args
  docker build --network host $(printf -- '--build-arg %q ' "$@") -f "$dockerfile_path" -t "$LOCAL_IMAGE_NAME" .
}

# Function to build post-training image
build_post_training_image() {
  DOCKERFILE_NAME=""
  if [[ ${POST_TRAINING_SOURCE} == "local" ]] ; then
    # To install vllm, tunix, tpu-inference from a local path, we copy it into the build context, excluding __pycache__.
    # This assumes vllm, tunix, tpu-inference is a sibling directory to the current one (maxtext).
    rsync -a --exclude='__pycache__' ../tpu-inference .
    rsync -a --exclude='__pycache__' ../vllm .
    rsync -a --exclude='__pycache__' ../tunix .

    # The cleanup is set to run even if the build fails to remove the copied directory.
    trap "rm -rf ./tpu-inference ./vllm ./tunix" EXIT INT TERM

    DOCKERFILE_NAME='maxtext_post_training_local_dependencies.Dockerfile'
    echo "Building local post-training dependencies: $DOCKERFILE_NAME"
  else
    DOCKERFILE_NAME='maxtext_post_training_dependencies.Dockerfile'
    echo "Building remote post-training dependencies: $DOCKERFILE_NAME"
  fi
  run_docker_build "$MAXTEXT_REPO_ROOT/dependencies/dockerfiles/$DOCKERFILE_NAME" \
    "MODE=${WORKFLOW}" "BASEIMAGE=${LOCAL_IMAGE_NAME}"
}

# Function to build image for GPUs
build_gpu_image() {
  if [[ ${MODE} == "pinned" ]]; then
    local base_image=ghcr.io/nvidia/jax:base-2024-12-04
    docker_build_args+=("BASEIMAGE=${base_image}")
  fi

  echo "Building docker image with arguments: ${docker_build_args[*]}"
  run_docker_build "$MAXTEXT_REPO_ROOT/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile" "${docker_build_args[@]}"
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
  run_docker_build "$MAXTEXT_REPO_ROOT/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile" "${docker_build_args[@]}"

  # Handle post-training workflow if specified
  if [[ ${WORKFLOW} == "post-training" || ${WORKFLOW} == "post-training-experimental" ]]; then
    build_post_training_image
  fi

  # TODO: Remove 'custom_wheels' mode support when tpu-recipes migration is complete.
  if [[ ${WORKFLOW} == "custom-wheels" ]]; then
    echo "Building custom wheels dependencies."
    run_docker_build "$MAXTEXT_REPO_ROOT/dependencies/dockerfiles/maxtext_custom_wheels.Dockerfile" "BASEIMAGE=${LOCAL_IMAGE_NAME}"
  fi
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
echo "Once you want you upload your docker container to GCR, take a look at docker_upload_runner.sh"
