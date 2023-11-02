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

# Run this on 256 chips to achieve a loss value of ~2.6
echo "Running 1b_convergence.sh"

RUN_NAME=${1}
OUTPUT_PATH=${2}
DATASET_PATH=${3}

LOSS_THRESHOLD=100.0 # Set to large value so test is guaranteed to pass, since we are not running as a test.
bash end_to_end/test_convergence_1b_params.sh ${RUN_NAME} ${LOSS_THRESHOLD} ${OUTPUT_PATH} ${DATASET_PATH}
