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

# Uninstall existing JAX and JAXlib
pip3 show jax && pip3 uninstall -y jax 
pip3 show jaxlib && pip3 uninstall -y jaxlib

libtpu_path="/lib/libtpu.so"

# Delete libtpu if it exists
if [ -e "$libtpu_path" ]; then
    sudo rm "$libtpu_path"
fi

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ $MODE == "nightly" ]]; then 
    # Nightly mode
    echo "Installing jax-head, jaxlib-nightly"
    # Install JAX from GitHub head
    pip3 install git+https://github.com/google/jax
    # Install jaxlib-nightly
    pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

    if [[ -n "$LIBTPU_GCS_PATH" ]]; then
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
elif [[ ! -n "$LIBTPU_GCS_PATH" ]]; then 
    # Stable mode
    echo "Installing stable jax, jaxlib, libtpu"
    pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
    echo -e "\n\nError: You can't use stable mode with customized libtpu.\n\n"
    exit 1
fi

# Install dependencies from requirements.txt
pip3 install -r requirements.txt
