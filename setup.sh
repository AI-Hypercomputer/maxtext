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
# bash setup.sh MODE={stable,nightly,libtpu-only} LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu}

# You need to specificy a MODE, default value stable. 
# You have the option to provide a LIBTPU_GCS_PATH that points to a libtpu.so provided to you by Google. 
# In libtpu-only MODE, the LIBTPU_GCS_PATH is mandatory.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.13


# Enable "exit immediately if any command fails" option
set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ $JAX_VERSION == NONE ]]; then
  unset JAX_VERSION
fi

if [[ $LIBTPU_GCS_PATH == NONE ]]; then
  unset LIBTPU_GCS_PATH
fi

if [[ -n $JAX_VERSION && ! ($MODE == "stable" || -z $MODE) ]]; then
     echo -e "\n\nError: You can only specify a JAX_VERSION with stable mode.\n\n"
     exit 1
fi

libtpu_path="$HOME/custom_libtpu/libtpu.so"

if [[ "$MODE" == "libtpu-only" ]]; then
    # Only update custom libtpu.
    if [[ -n "$LIBTPU_GCS_PATH" ]]; then
        # Install custom libtpu
        echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
        # Install required dependency
        pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
        exit 0
    else
        echo -e "\n\nError: You must provide a custom libtpu for libtpu-only mode.\n\n"
        exit 1
    fi
fi

# Save the script folder path of maxtext
run_name_folder_path=$(pwd)

# Uninstall existing jax, jaxlib and  libtpu-nightly
pip3 show jax && pip3 uninstall -y jax 
pip3 show jaxlib && pip3 uninstall -y jaxlib
pip3 show libtpu-nightly && pip3 uninstall -y libtpu-nightly

# Delete jax folder if it exists
if [[ -d $HOME/jax ]]; then
    rm -rf $HOME/jax
fi

# Delete xla folder if it exists
if [[ -d $HOME/xla ]]; then
    rm -rf $HOME/xla
fi

# Delete custom libtpu if it exists
if [ -e "$libtpu_path" ]; then
    rm "$libtpu_path"
fi

if [[ "$MODE" == "stable" || ! -v MODE ]]; then
# Stable mode
    if [[ -n "$JAX_VERSION" ]]; then
        echo "Installing stable jax, jaxlib, libtpu version ${JAX_VERSION}"
        pip3 install jax[tpu]==${JAX_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    else
        echo "Installing stable jax, jaxlib, libtpu"
        pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    fi

    if [[ -n "$LIBTPU_GCS_PATH" ]]; then 
        # Install custom libtpu
        echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
        # Install required dependency
        pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
    fi
elif [[ $MODE == "nightly" ]]; then 
# Nightly mode
    echo "Installing jax-head, jaxlib-nightly"
    # Install jax from GitHub head
    pip3 install git+https://github.com/google/jax
    # Install jaxlib-nightly
    pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

    if [[ -n "$LIBTPU_GCS_PATH" ]]; then
        # Install custom libtpu
        echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
        # Install required dependency
        pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
    else
        # Install libtpu-nightly
        echo "Installing libtpu-nightly"
        pip3 install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre
    fi
else
    echo -e "\n\nError: You can only set MODE to [stable,nightly,libtpu-only].\n\n"
    exit 1
fi

# Install dependencies from requirements.txt
cd $run_name_folder_path && pip3 install -r requirements.txt
