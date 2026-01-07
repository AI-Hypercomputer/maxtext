#!/bin/bash

# Copyright 2023–2026 Google LLC
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

# Description:
# bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=/tmp/gcsfuse FILE_PATH=/tmp/gcsfuse/my_dataset

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
MAXTEXT_VERSION=$(pip list | grep '^MaxText ' | awk '{print $2}')
if [[ -z "$MAXTEXT_VERSION" ]]; then
  MAXTEXT_VERSION="unknown"
fi
GRAIN_VERSION=$(pip list | grep '^grain ' | awk '{print $2}')
if [[ -z "$GRAIN_VERSION" ]]; then
  GRAIN_VERSION="unknown"
fi

APP_NAME="maxtext-gcsfuse/maxtext-$MAXTEXT_VERSION/grain-$GRAIN_VERSION"



# see https://cloud.google.com/storage/docs/gcsfuse-cli for all configurable options of gcsfuse CLI
TIMESTAMP=$(date +%Y%m%d-%H%M)
gcsfuse -o ro --implicit-dirs --log-severity=debug \
        --type-cache-max-size-mb=-1 --stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 \
        --log-file=$HOME/gcsfuse_$TIMESTAMP.json --app-name=$APP_NAME "$DATASET_GCS_BUCKET" "$MOUNT_PATH"

# Use ls to prefill the metadata cache: https://cloud.google.com/storage/docs/cloud-storage-fuse/performance#improve-first-time-reads
if [[ ! -z ${FILE_PATH} ]] ; then 
  FILE_COUNT=$(ls -R $FILE_PATH | wc -l)
  echo $FILE_COUNT files found in $FILE_PATH
fi
