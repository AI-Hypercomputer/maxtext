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
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=stable
## bash dependencies/scripts/docker_build_dependency_image.sh DEVICE={{gpu|tpu}} MODE=stable_stack BASEIMAGE={{JAX_STABLE_STACK BASEIMAGE FROM ARTIFACT REGISTRY}}

# Build docker image with nightly dependencies
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=nightly

# Build docker image with stable dependencies and, a pinned JAX_VERSION for TPUs
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=stable JAX_VERSION=0.4.13

# Build docker image with stable dependencies and, a pinned JAX_VERSION for GPUs
# Available versions listed at https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax
## bash dependencies/scripts/docker_build_dependency_image.sh DEVICE=gpu MODE=nightly JAX_VERSION=0.4.36.dev20241109

# MODE=custom_wheels builds the nightly environment, then reinstalls any
# additional wheels present in the maxtext directory.
# Use this mode to install custom dependencies, such as custom JAX or JAXlib builds.
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=custom_wheels

# ==================================
# POST-TRAINING BUILD EXAMPLES
# ==================================

# Build docker image with stable pre-training dependencies and stable post-training dependencies
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training

# Build docker image with stable pre-training dependencies and post-training dependencies from GitHub head
## bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training POST_TRAINING_SOURCE=local

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

export LOCAL_IMAGE_NAME=maxtext_base_image
echo "Building to $LOCAL_IMAGE_NAME"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r RAW_KEY VALUE <<< "$ARGUMENT"
    KEY=$(echo "$RAW_KEY" | tr '[:lower:]' '[:upper:]')
    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done


if [[ -z ${JAX_VERSION+x} ]] ; then
  export JAX_VERSION=NONE
  echo "Default JAX_VERSION=${JAX_VERSION}"
fi

if [[ -z ${MODE} ]]; then
  export MODE=stable
  echo "Default MODE=${MODE}"
  export CUSTOM_JAX=0
  export INSTALL_POST_TRAINING=0
elif [[ ${MODE} == "custom_wheels" ]] ; then
  export MODE=nightly
  export CUSTOM_JAX=1
  export INSTALL_POST_TRAINING=0
elif [[ ${MODE} == "post-training" || ${MODE} == "post-training-experimental" ]] ; then
  export INSTALL_POST_TRAINING=1
  export CUSTOM_JAX=0
else
  export CUSTOM_JAX=0
  export INSTALL_POST_TRAINING=0
fi

if [[ -z ${DEVICE} ]]; then
  export DEVICE=tpu
  echo "Default DEVICE=${DEVICE}"
fi

# New flag for post-training source
if [[ -z ${POST_TRAINING_SOURCE} ]]; then
 export POST_TRAINING_SOURCE=remote # Default to the original Dockerfile
 echo "Default POST_TRAINING_SOURCE=${POST_TRAINING_SOURCE}"
fi

# Function to build with MODE=jax_ai_image
build_ai_image() {
    if [[ -z ${BASEIMAGE+x} ]]; then
        echo "Error: BASEIMAGE is unset, please set it!"
        exit 1
    fi
    COMMIT_HASH=$(git rev-parse --short HEAD)
    echo "Building JAX AI MaxText Imageat commit hash ${COMMIT_HASH}..."

    docker build \
        --build-arg JAX_AI_IMAGE_BASEIMAGE=${BASEIMAGE} \
        --build-arg COMMIT_HASH=${COMMIT_HASH} \
        --build-arg DEVICE="$DEVICE" \
        --network=host \
        -t ${LOCAL_IMAGE_NAME} \
        -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_jax_ai_image.Dockerfile' \
        .
}

if [[ -z ${LIBTPU_GCS_PATH+x} ]] ; then
  export LIBTPU_GCS_PATH=NONE
  echo "Default LIBTPU_GCS_PATH=${LIBTPU_GCS_PATH}"
  if [[ ${DEVICE} == "gpu" ]]; then
    if [[ ${MODE} == "stable_stack" || ${MODE} == "jax_ai_image" ]]; then
      build_ai_image
    else
      if [[ ${MODE} == "pinned" ]]; then
        export BASEIMAGE=ghcr.io/nvidia/jax:base-2024-12-04
      else
        export BASEIMAGE=ghcr.io/nvidia/jax:base
      fi
      docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION \
                   --build-arg DEVICE=$DEVICE --build-arg BASEIMAGE=$BASEIMAGE \
                   -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_gpu_dependencies.Dockerfile' \
                   -t ${LOCAL_IMAGE_NAME} .
    fi
  else
    if [[ ${MODE} == "stable_stack" || ${MODE} == "jax_ai_image" ]]; then
      build_ai_image
    elif [[ ${MANTARAY} == "true" ]]; then
      echo "Building with benchmark-db"
      docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION \
                   --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH --build-arg DEVICE=$DEVICE \
                   -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_db_dependencies.Dockerfile' \
                   -t ${LOCAL_IMAGE_NAME} .
    elif [[ ${INSTALL_POST_TRAINING} -eq 1 && ${DEVICE} == "tpu" ]]; then
      echo "Installing MaxText stable mode dependencies for post-training"
      docker build --network host --build-arg MODE=stable --build-arg JAX_VERSION=$JAX_VERSION \
                   --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH --build-arg DEVICE=$DEVICE \
                   -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_dependencies.Dockerfile' \
                   -t ${LOCAL_IMAGE_NAME} .
    else
      docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION \
                   --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH --build-arg DEVICE=$DEVICE \
                   -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_dependencies.Dockerfile' \
                   -t ${LOCAL_IMAGE_NAME} .
    fi
  fi
else
  docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION \
               --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH \
               -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_dependencies.Dockerfile' \
               -t ${LOCAL_IMAGE_NAME} .
  docker build --network host --build-arg CUSTOM_LIBTPU=true \
               -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_libtpu_path.Dockerfile' \
               -t ${LOCAL_IMAGE_NAME} .
fi

if [[ ${INSTALL_POST_TRAINING} -eq 1 ]] ; then
  if [[ ${DEVICE} != "tpu" ]] ; then
    echo "Error: MODE=post-training is only supported for DEVICE=tpu"
    exit 1
  fi

 DOCKERFILE_NAME=""
  if [[ ${POST_TRAINING_SOURCE} == "local" ]] ; then
    
  # To install tpu-inference from a local path, we copy it into the build context, excluding __pycache__.
  # This assumes vllm, tunix, tpu-inference is a sibling directory to the current one (maxtext).
  rsync -a --exclude='__pycache__' ../tpu-inference .
  # To install vllm from a local path, we copy it into the build context, excluding __pycache__.
  # This assumes vllm is a sibling directory to the current one (maxtext).
  rsync -a --exclude='__pycache__' ../vllm .

  rsync -a --exclude='__pycache__' ../tunix .

  # The cleanup is set to run even if the build fails to remove the copied directory.
  trap "rm -rf ./tpu-inference ./vllm ./tunix" EXIT INT TERM

  DOCKERFILE_NAME='maxtext_post_training_local_dependencies.Dockerfile'
  echo "Using local post-training dependencies Dockerfile: $DOCKERFILE_NAME"
 else
  DOCKERFILE_NAME='maxtext_post_training_dependencies.Dockerfile'
  echo "Using remote post-training dependencies Dockerfile: $DOCKERFILE_NAME"
 fi

 docker build \
 --network host \
 --build-arg BASEIMAGE=${LOCAL_IMAGE_NAME} \
 --build-arg MODE=${MODE} \
 -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/'"$DOCKERFILE_NAME" \
 -t ${LOCAL_IMAGE_NAME} .
fi

if [[ ${CUSTOM_JAX} -eq 1 ]] ; then
  echo "Installing custom jax and jaxlib"
  docker build --network host \
               -f "$MAXTEXT_REPO_ROOT"'/dependencies/dockerfiles/maxtext_custom_wheels.Dockerfile' \
               -t ${LOCAL_IMAGE_NAME} .
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
