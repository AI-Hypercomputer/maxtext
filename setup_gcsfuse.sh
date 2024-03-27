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
# bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=dataset

set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ -z ${DATASET_GCS_BUCKET} || -z ${MOUNT_PATH} ]]; then
  echo "Please set arguments: DATASET_GCS_BUCKET and MOUNT_PATH"
  exit 1
fi

if [[ "$DATASET_GCS_BUCKET" =~ gs:\/\/ ]] ; then
    DATASET_GCS_BUCKET="${DATASET_GCS_BUCKET/gs:\/\//}"
    echo "Removed gs:// from GCS bucket name, GCS bucket is $DATASET_GCS_BUCKET"
fi

if [[ -d ${MOUNT_PATH} ]]; then
  echo "$MOUNT_PATH exists, removing..."
  fusermount -u $MOUNT_PATH || rm -rf $MOUNT_PATH
fi

mkdir -p $MOUNT_PATH

# see https://cloud.google.com/storage/docs/gcsfuse-cli for all configurable options of gcsfuse CLI
# Grain uses _PROCESS_MANAGEMENT_MAX_THREADS = 64 (https://github.com/google/grain/blob/main/grain/_src/python/grain_pool.py)
# Please make sure max-conns-per-host > grain_worker_count * _PROCESS_MANAGEMENT_MAX_THREADS

gcsfuse -o ro --implicit-dirs --http-client-timeout=5s --max-conns-per-host=2000 \
        --debug_fuse_errors --debug_fuse --debug_gcs --debug_invariants --debug_mutex \
        --log-file=$HOME/gcsfuse.json "$DATASET_GCS_BUCKET" "$MOUNT_PATH"
