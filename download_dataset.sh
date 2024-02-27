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

# This script downloads c4/en/3.0.1 to your gcs bucket directory
# Usage bash download_dataset.sh <<gcp project>> <<gcs bucket name>>
# Usage example: bash download_dataset.sh cloud-tpu-multipod-dev gs://maxtext-dataset

function remove_trailing_slash {
  if [[ $1 =~ /$ ]]; then  # Check if the path ends with a slash
    echo "${1::-1}"       # Remove the last character
  else
    echo "$1"              # Output the path as-is
  fi
}

gsutil -u $1 -m cp 'gs://allennlp-tensorflow-datasets/c4/en/3.0.1/*' $(remove_trailing_slash $2)/c4/en/3.0.1
