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
# bash setup.sh MODE={stable,nightly,libtpu-only} LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu} DEVICE={tpu,gpu}


# You need to specify a MODE, default value stable.
# You have the option to provide a LIBTPU_GCS_PATH that points to a libtpu.so provided to you by Google.
# In libtpu-only MODE, the LIBTPU_GCS_PATH is mandatory.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.13


# Enable "exit immediately if any command fails" option
set -e
export DEBIAN_FRONTEND=noninteractive

(sudo bash || bash) <<'EOF'
apt update && \
apt install -y numactl && \
apt install -y lsb-release && \
apt install -y gnupg && \
apt install -y curl
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update -y && apt -y install gcsfuse
rm -rf /var/lib/apt/lists/*
EOF

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ -z "$DEVICE" ]]; then
        export DEVICE="tpu"
fi

if [[ $LIBTPU_GCS_PATH == NONE ]]; then
  unset LIBTPU_GCS_PATH
fi

# Save the script folder path of maxtext
run_name_folder_path=$(pwd)

# Install dependencies from requirements.txt
# cd $run_name_folder_path && pip install --upgrade pip &&  pip3 install -r cpu_requirements.txt

# Install custom jaxlib
pip3 install /deps/jaxlib-*-cp*-manylinux2014_x86_64.whl --force-reinstall
# Install custom jax
pip3 install /deps/jax-*-py3-none-any.whl --force-reinstall


# # Uninstall existing jax, jaxlib and  libtpu-nightly
# pip3 show jax && pip3 uninstall -y jax
# pip3 show jaxlib && pip3 uninstall -y jaxlib
# pip3 show libtpu-nightly && pip3 uninstall -y libtpu-nightly


# pip install numpy wheel build
# echo "Clone jax and xla"
# git clone https://github.com/google/jax
# cd jax
# git clone https://github.com/openxla/xla.git
# # sed -i '451i\  gloo_context->setTimeout(std::chrono::seconds(60));' xla/xla/pjrt/cpu/gloo_collectives.cc
# echo "Building jaxlib"
# echo "ls $(pwd)"
# ls $(pwd)
# python build/build.py --bazel_options=--override_repository=xla=$(pwd)/xla
# echo "Finished building jaxlib"
# echo "Installing libtpu  nightly"
# pip install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre
# echo "install JAX"
# pip install -e .  # installs jax

# Install dependencies from requirements.txt
cd $run_name_folder_path && pip install --upgrade pip &&  pip3 install -r cpu_requirements.txt
