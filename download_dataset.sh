#!/bin/bash

# SPDX-License-Identifier: Apache-2.0

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
