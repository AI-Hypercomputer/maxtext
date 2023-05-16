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

# How to use:
# stable/default mode will install jax,jaxlib,libtpu-nightly from stable release.
# Example Command:
# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh"
# or
# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh MODE=stable"

# nightly mode will install jax-head, jaxlib-nightly, libtpu-nightly or custom libtpu from gcs bucket.
# Example Command:
# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh MODE=nightly"
# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh MODE=nightly LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu}"

# head mode will install jax-head, jaxlib-head, custom libtpu from gcs bucket.
# Example Command:
# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh MODE=head LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu}"

# libtpu-only mode will install custom libtpu from gcs bucket.
# Example Command:
# python3 multihost_runner.py --TPU_PREFIX=$TPU_PREFIX --COMMAND="bash setup.sh MODE=libtpu-only LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu}"


# Enable "exit immediately if any command fails" option
set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

libtpu_path="/lib/libtpu.so"

if [[ "$MODE" == "libtpu-only" ]]; then
    # Only update custom libtpu.
    if [[ -n "$LIBTPU_GCS_PATH" ]]; then
        # Install custom libtpu
        echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
        # Install required dependency
        sudo pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        sudo gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
        exit 0
    else
        echo -e "\n\nError: You must provide a custom libtpu for libtpu-only mode.\n\n"
        exit 1
    fi
fi

# Save the script folder path of maxtext
run_name_folder_path=$(pwd)

# Uninstall existing jax, jaxlib, and libtpu-nightly
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
    sudo rm "$libtpu_path"
fi

if [[ "$MODE" == "stable" || ! -v MODE ]]; then
# Stable mode
    if [[ -n "$LIBTPU_GCS_PATH" ]]; then 
        echo -e "\n\nError: You can't use stable mode with custom libtpu.\n\n"
        exit 1
    else
        echo "Installing stable jax, jaxlib, libtpu"
        pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
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
        sudo pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        sudo gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
    else
        # Install libtpu-nightly
        echo "Installing libtpu-nightly"
        pip3 install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre
    fi
elif [[ $MODE == "head" ]]; then 
# Head mode
    if [[ -n "$LIBTPU_GCS_PATH" ]]; then
        # Install custom libtpu
        echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
        # Install required dependency
        sudo pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        sudo gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
    else
        echo -e "\n\nError: You must provide a custom libtpu for head mode.\n\n"
        exit 1
    fi

    echo "Installing jax-head, jaxlib-head"
    # Install jax from GitHub head
    echo "Installing jax from HEAD..."
    # Install jax from GitHub head
    pip3 install git+https://github.com/google/jax
    # Install jaxlib from GitHub head
    echo "Installing jaxlib from HEAD..."
    cd $HOME && git clone https://github.com/openxla/xla
    cd $HOME && git clone https://github.com/google/jax.git
    cd $HOME/jax
    python3 build/build.py --enable_tpu --bazel_options="--override_repository=xla=$HOME/xla"
    pip3 install dist/jaxlib-*-cp*-manylinux2014_x86_64.whl --force-reinstall --no-deps
else
    echo -e "\n\nError: You can only set MODE to [stable,nightly,head].\n\n"
    exit 1
fi

# Install dependencies from requirements.txt
cd $run_name_folder_path && pip3 install -r requirements.txt
