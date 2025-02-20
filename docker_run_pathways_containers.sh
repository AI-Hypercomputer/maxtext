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

# Example - # bash docker_run_pathways_containers.sh maxtext_image=<>

# bash docker_run_pathways_containers.sh maxtext_image=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest

# Stop execution if any command exits with error


echo "Running docker_run_pathways_containers.sh"

# Prerequisites on the self-hosted runner
echo "Setting up the prerequisites"
apt-get install docker
apt-get install docker-compose-plugin
docker compose version # To ensure docker compose is installed
gcloud auth configure-docker us-docker.pkg.dev --quiet

set -e
# Default - 
maxtext_image=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest


# Parse input variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

cd utils_pathways
MAXTEXT_IMAGE=${maxtext_image} docker compose down
sleep 10
echo "MAXTEXT image ----------------"
echo $MAXTEXT_IMAGE
MAXTEXT_IMAGE=${maxtext_image} docker compose up --exit-code-from maxtext
sleep 10
MAXTEXT_IMAGE=${maxtext_image} docker compose down
# echo "Exit code from Maxtext..."
# echo $?
# sleep 20
# docker compose ps
# MAXTEXT_IMAGE=${maxtext_image} docker compose logs -f maxtext