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

# Example command:
# bash docker_build_dependency_image.sh MODE=stable
# bash docker_build_dependency_image.sh DEVICE={{gpu|tpu}} MODE=stable_stack BASEIMAGE={{JAX_STABLE_STACK BASEIMAGE FROM ARTIFACT REGISTRY}}
# bash docker_build_dependency_image.sh MODE=nightly
# bash docker_build_dependency_image.sh MODE=stable JAX_VERSION=0.4.13
# Nightly build with JAX_VERSION for GPUs. Available versions listed at https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/jax:
# bash docker_build_dependency_image.sh DEVICE=gpu MODE=nightly JAX_VERSION=0.4.36.dev20241109 # Note: this sets both jax-nightly and jaxlib-nightly 
# MODE=custom_wheels is the same as nightly except that it reinstalls any
# additional wheels that are present in the maxtext directory.
# The main use case is to install custom jax or jaxlib wheels but it also
# works with any custom wheels.
# bash docker_build_dependency_image.sh MODE=custom_wheels

# Enable "exit immediately if any command fails" option
set -e

export LOCAL_IMAGE_NAME=maxtext_base_image
echo "Building to $LOCAL_IMAGE_NAME"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done


if [[ -z ${JAX_VERSION+x} ]] ; then
  export JAX_VERSION=NONE
  echo "Default JAX_VERSION=${JAX_VERSION}"
fi

if [[ -z ${MODE} ]]; then
  export MODE=stable
  echo "Default MODE=${MODE}"
elif [[ ${MODE} == "custom_wheels" ]] ; then
  export MODE=nightly
  export CUSTOM_JAX=1
else
  export CUSTOM_JAX=0
fi

if [[ -z ${DEVICE} ]]; then
  export DEVICE=tpu
  echo "Default DEVICE=${DEVICE}"
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
        -f ./maxtext_jax_ai_image.Dockerfile .
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
      docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION --build-arg DEVICE=$DEVICE --build-arg BASEIMAGE=$BASEIMAGE -f ./maxtext_gpu_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
    fi
  else
    if [[ ${MODE} == "stable_stack" || ${MODE} == "jax_ai_image" ]]; then
      build_ai_image
    elif [[ ${MANTARAY} == "true" ]]; then
      echo "Building with benchmark-db"
      docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH --build-arg DEVICE=$DEVICE -f ./maxtext_db_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
    else
      docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH --build-arg DEVICE=$DEVICE -f ./maxtext_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
    fi
  fi
else
  docker build --network host --build-arg MODE=${MODE} --build-arg JAX_VERSION=$JAX_VERSION --build-arg LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH -f ./maxtext_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
  docker build --network host --build-arg CUSTOM_LIBTPU=true -f ./maxtext_libtpu_path.Dockerfile -t ${LOCAL_IMAGE_NAME} .
fi

if [[ ${CUSTOM_JAX} -eq 1 ]] ; then
  echo "Installing custom jax and jaxlib"
  docker build --network host -f ./maxtext_custom_wheels.Dockerfile -t ${LOCAL_IMAGE_NAME} .
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
