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

export LOCAL_IMAGE_NAME=maxtext_base_image

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ "$MODE" == "stable" || ! -v MODE ]]; then
  echo "STABLE"
    # Stable mode
  if [[ ! -v JAX_VERSION ]]; then
    docker build --build-arg MODE="stable" -f ./maxtext_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
  else
    docker build --build-arg MODE="stable" --build-arg JAX_VERSION="$JAX_VERSION" -f ./maxtext.Dockerfile -t ${LOCAL_IMAGE_NAME} .
  fi
elif [[ $MODE == "nightly" ]]; then
  echo "NIGHTLY"

  # Nightly mode
  docker build --build-arg MODE="nightly" -f ./maxtext_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
elif [[ $MODE == "head" ]]; then
  echo "HEAD"

  # Head mode
  docker build --build-arg MODE="head" -f ./maxtext_dependencies.Dockerfile -t ${LOCAL_IMAGE_NAME} .
fi

echo ""
echo "*************************"
echo ""

echo "Built your base docker image and named it ${LOCAL_IMAGE_NAME}.
It only has the dependencies installed. Assuming you're on a TPUVM, to run the
docker image locally and mirror your local working directory run:"
echo "docker run -v $(pwd):/app --rm -it --privileged --entrypoint bash maxtext_base_image"
echo ""
echo "You can run MaxText and your development tests inside of the docker image. Changes to your workspace will automatically
be reflected inside the docker container."
echo "Once you want you upload your docker container to GCR, take a look at docker_upload_runner.sh"