#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

LOG_FILE_IN_GCS=$1
filename=$(basename $LOG_FILE_IN_GCS)
output_file=$(date "+%Y-%m-%d-%H:%M:%S")_${filename}

CNS_PATH=/cns/pi-d/home/${USER}/tensorboard/multislice/
fileutil mkdir -p ${CNS_PATH}
/google/data/ro/projects/cloud/bigstore/mpm/fileutil_bs/stable/bin/fileutil_bs cp /bigstore/${LOG_FILE_IN_GCS} ${CNS_PATH}/$output_file
echo file to put into xprof: ${CNS_PATH}/$output_file
