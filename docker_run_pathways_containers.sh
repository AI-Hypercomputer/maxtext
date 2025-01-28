#!/bin/bash

# Copyright 2025 Google LLC
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

# Examples -
# TPU unit tests - bash docker_run_pathways_containers.sh maxtext_image=<your Maxtext image> command="cd MaxText ; python3 -m pytest tests -m 'not gpu_only and not integration_test' -s"
# Subset of unit tests - bash docker_run_pathways_containers.sh maxtext_image=<your Maxtext image> command="cd MaxText ; python3 -m pytest tests/train_tests.py -m 'not gpu_only and not integration_test' -s"

echo "Running docker_run_pathways_containers.sh"

# Stop execution if any command exits with error
set -e

# Parse input variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

cd utils_pathways

# Error handling - Run 'docker compose down' if 'docker compose up' errors.
# Exit the script with the exit code of docker compose up.
clean_up () {
    overall_test_exit_code=$?
    echo "Tests are not successful, exit code is $overall_test_exit_code"
    MAXTEXT_IMAGE=${maxtext_image} COMMAND=${command} docker compose down
    exit $overall_test_exit_code

}
trap clean_up EXIT

# Setting up and tearing down the test setup -
MAXTEXT_IMAGE=${maxtext_image} COMMAND=${command} docker compose up --exit-code-from maxtext
MAXTEXT_IMAGE=${maxtext_image} COMMAND=${command} docker compose down
