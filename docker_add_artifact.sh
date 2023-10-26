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

# This scripts takes a docker image that already contains the MaxText dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Example command:
# IMAGE_NAME=gcr.io/my-project/my-existing-image 
# COMMAND="export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true' && python3 MaxText/train_compile.py MaxText/configs/base.yml compile_topology='v4-8' compiled_trainstep_file='compiled_v4_8.pickle' compile_topology_num_slices=1"
# bash docker_add_artifact.sh ${IMAGE_NAME} "${COMMAND}"
set -e 


IMAGE_NAME=${1} # e.g. gcr.io/my-project/my-existing-image 
COMMAND=${2} # e.g. "export LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true' && python3 MaxText/train_compile.py MaxText/configs/base.yml compile_topology='v4-128' compiled_trainstep_file='compiled_v4_128.pickle' compile_topology_num_slices=1 per_device_batch_size=1"

echo "Running docker_add_artifact.sh..."
echo "Image Name is ${IMAGE_NAME}"
echo "Command is ${COMMAND}"

tmp_local_container=tmp_local_container

# Step 1: Build a Docker Container
docker build -t ${tmp_local_container} -f ./agi.Dockerfile -t ${IMAGE_NAME} .

# Step 2: Run a command inside in the Docker Container
docker run --privileged ${tmp_local_container} /bin/sh -c "${COMMAND}"

# Step 3: Commit changes to the container
container_id=$(docker ps -lq)  # Get the ID of the last running container
docker commit "$container_id" ${IMAGE_NAME}:latest

# Step 4: Push the New Docker Image to GCR
docker push ${IMAGE_NAME}:latest

# Step 5: Clean up - remove the temporary container
docker rm "$container_id"