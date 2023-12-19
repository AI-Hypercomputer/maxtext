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
# sudo bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=dataset

set -e -x

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

if [[ $GCS_BUCKET == gs://* ]] ; then
    echo "Remove gs:// from GCS bucket name"
    exit 1
fi

if ! command -v gcsfuse &> /dev/null ; then
  apt-get update -y && \
  apt-get install -y lsb-release && \
  apt-get install -y gnupg && \
  apt-get install -y curl
  export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
  apt-get update -y && apt-get -y install gcsfuse
  rm -rf /var/lib/apt/lists/*
fi

mkdir -p $MOUNT_PATH

# gcsfuse --implicit-dirs "$DATASET_GCS_BUCKET" "$MOUNT_PATH"

gcsfuse --implicit-dirs --http-client-timeout=5s --max-conns-per-host=2000 \
        --debug_fuse_errors --debug_fuse --debug_gcs --debug_invariants --debug_mutex \
        --log-file=$HOME/gcsfuse.json "$DATASET_GCS_BUCKET" "$MOUNT_PATH"