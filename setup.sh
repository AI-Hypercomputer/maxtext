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

# Install dependencies from requirements.txt
pip3 install -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ $MODE == "nightly" ]]; then
    echo "Installing jax-head, jaxlib-nightly, and libtpu-nightly"
    # Install JAX from GitHub head
    pip3 install git+https://github.com/google/jax
    # Install jaxlib-nightly
    pip3 install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
    # Install libtpu-nightly
    pip3 install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre
fi

if [[ -n "$LIBTPU_GCS_PATH" ]]; then
    sudo pip3 install -U crcmod
    echo "Copying libtpu.so from $LIBTPU_GCS_PATH to /lib/libtpu.so"
    sudo gsutil cp "$LIBTPU_GCS_PATH" /lib/libtpu.so
fi
