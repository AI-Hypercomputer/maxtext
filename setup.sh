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

if [[ $DEVICE == "tpu" ]]; then
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
fi

# Save the script folder path of maxtext
run_name_folder_path=$(pwd)

# Uninstall existing jax, jaxlib and  libtpu-nightly
pip3 show jax && pip3 uninstall -y jax
pip3 show jaxlib && pip3 uninstall -y jaxlib
pip3 show libtpu-nightly && pip3 uninstall -y libtpu-nightly

# Delete custom libtpu if it exists
if [ -e "$libtpu_path" ]; then
    rm "$libtpu_path"
fi

if [[ "$MODE" == "pinned" ]]; then
  if [[ "$DEVICE" != "gpu" ]]; then
    echo "pinned mode is supported for GPU builds only."
    exit 1
  fi
  echo "Installing pinned jax, jaxlib for NVIDIA gpu."
  pip3 install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -c constraints_gpu.txt
  pip3 install "transformer-engine==1.5.0+297459b" \
    --extra-index-url https://us-python.pkg.dev/gce-ai-infra/maxtext-build-support-packages/simple/ \
    -c constraints_gpu.txt
elif [[ "$MODE" == "stable" || ! -v MODE ]]; then
# Stable mode
    if [[ $DEVICE == "tpu" ]]; then
        echo "Installing stable jax, jaxlib for tpu"
        if [[ -n "$JAX_VERSION" ]]; then
            echo "Installing stable jax, jaxlib, libtpu version ${JAX_VERSION}"
            pip3 install jax[tpu]==${JAX_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        else
            echo "Installing stable jax, jaxlib, libtpu for tpu"
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
        if [[ -n "$LIBTPU_GCS_PATH" ]]; then
            # Install custom libtpu
            echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
            # Install required dependency
            pip3 install -U crcmod
            # Copy libtpu.so from GCS path
            gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
        fi
    elif [[ $DEVICE == "gpu" ]]; then
        echo "Installing stable jax, jaxlib for NVIDIA gpu"
        if [[ -n "$JAX_VERSION" ]]; then
            echo "Installing stable jax, jaxlib ${JAX_VERSION}"
            pip3 install -U "jax[cuda12_pip]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        else
            echo "Installing stable jax, jaxlib, libtpu for NVIDIA gpu"
            pip3 install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        fi
        pip3 install "transformer-engine==1.5.0+297459b" \
          --extra-index-url https://us-python.pkg.dev/gce-ai-infra/maxtext-build-support-packages/simple/
    fi
elif [[ $MODE == "nightly" ]]; then
# Nightly mode
    if [[ $DEVICE == "gpu" ]]; then
        echo "Installing jax-nightly, jaxlib-nightly"
        # Install jax-nightly
        pip3 install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
        # Install jaxlib-nightly
        pip3 install -U --pre jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html
        # Install prebuilt Transformer Engine for GPU builds.
        pip3 install "transformer-engine==1.5.0+297459b" \
          --extra-index-url https://us-python.pkg.dev/gce-ai-infra/maxtext-build-support-packages/simple/
    elif [[ $DEVICE == "tpu" ]]; then
        echo "Installing jax-nightly, jaxlib-nightly"
        # Install jax-nightly
        pip3 install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
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
        echo "Installing nightly tensorboard plugin profile"
        pip3 install tbp-nightly --upgrade
    fi
    echo "Installing nightly tensorboard plugin profile"
    pip3 install tbp-nightly --upgrade
else
    echo -e "\n\nError: You can only set MODE to [stable,nightly,libtpu-only].\n\n"
    exit 1
fi

# Install dependencies from requirements.txt
cd $run_name_folder_path && pip install --upgrade pip
if [[ "$MODE" == "pinned" ]]; then
    pip3 install -r requirements.txt -c constraints_gpu.txt
else
    pip3 install -U -r requirements.txt
fi
