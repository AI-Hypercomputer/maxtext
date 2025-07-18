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
# For DEVICE=gpu, you may also specify JAX_VERSION when MODE=nightly, e.g. JAX_VERSION=0.4.36.dev20241109


# Enable "exit immediately if any command fails" option
set -e
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_SUSPEND=1
export NEEDRESTART_MODE=l

apt-get update && apt-get install -y sudo
(sudo bash || bash) <<'EOF'
apt update && \
apt install -y numactl lsb-release gnupg curl net-tools iproute2 procps lsof git ethtool && \
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt update -y && apt -y install gcsfuse
rm -rf /var/lib/apt/lists/*
EOF

# We need to pin specific versions of setuptools, see b/402501203 for more.
python3 -m pip install setuptools==65.5.0 wheel==0.45.1

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

if [[ -n $JAX_VERSION && ! ($MODE == "stable" || -z $MODE || ($MODE == "nightly" && $DEVICE == "gpu")) ]]; then
     echo -e "\n\nError: You can only specify a JAX_VERSION with stable mode (plus nightly mode on GPU).\n\n"
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
            python3 -m pip install -U crcmod
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

# Install dependencies from requirements.txt
cd "$run_name_folder_path" && python3 -m pip install --upgrade pip
if [[ "$MODE" == "pinned" ]]; then
    python3 -m pip install --no-cache-dir -U -r requirements.txt -c constraints_gpu.txt
else
    python3 -m pip install --no-cache-dir -U -r requirements.txt
fi

# Uninstall existing jax, jaxlib and  libtpu-nightly
python3 -m pip show jax && python3 -m pip uninstall -y jax
python3 -m pip show jaxlib && python3 -m pip uninstall -y jaxlib
python3 -m pip show libtpu-nightly && python3 -m pip uninstall -y libtpu-nightly

# Delete custom libtpu if it exists
if [ -e "$libtpu_path" ]; then
    rm "$libtpu_path"
fi

if [[ "$MODE" == "pinned" ]]; then
  if [[ "$DEVICE" != "gpu" ]]; then
    echo "pinned mode is supported for GPU builds only."
    exit 1
  fi
  echo "Installing Jax and Transformer Engine."
  python3 -m pip install "jax[cuda12]" -c constraints_gpu.txt
  python3 -m pip install transformer-engine[jax]==1.13.0

elif [[ "$MODE" == "stable" || ! -v MODE ]]; then
# Stable mode
    if [[ $DEVICE == "tpu" ]]; then
        echo "Installing stable jax, jaxlib for tpu"
        if [[ -n "$JAX_VERSION" ]]; then
            echo "Installing stable jax, jaxlib, libtpu version ${JAX_VERSION}"
            python3 -m pip install jax[tpu]==${JAX_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        else
            echo "Installing stable jax, jaxlib, libtpu for tpu"
            python3 -m pip install 'jax[tpu]>0.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        fi

        if [[ -n "$LIBTPU_GCS_PATH" ]]; then
            # Install custom libtpu
            echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
            # Install required dependency
            python3 -m pip install -U crcmod
            # Copy libtpu.so from GCS path
            gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
        fi
        if [[ -n "$LIBTPU_GCS_PATH" ]]; then
            # Install custom libtpu
            echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
            # Install required dependency
            python3 -m pip install -U crcmod
            # Copy libtpu.so from GCS path
            gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
        fi
    elif [[ $DEVICE == "gpu" ]]; then
        echo "Installing stable jax, jaxlib for NVIDIA gpu"
        if [[ -n "$JAX_VERSION" ]]; then
            echo "Installing stable jax, jaxlib ${JAX_VERSION}"
            python3 -m pip install -U "jax[cuda12]==${JAX_VERSION}"
        else
            echo "Installing stable jax, jaxlib, libtpu for NVIDIA gpu"
            python3 -m pip install "jax[cuda12]"
        fi
    fi
elif [[ $MODE == "nightly" ]]; then
# Nightly mode
    if [[ $DEVICE == "gpu" ]]; then
        # Install jax-nightly
        if [[ -n "$JAX_VERSION" ]]; then
            echo "Installing jax-nightly, jaxlib-nightly ${JAX_VERSION}"
            python3 -m pip install -U --pre jax==${JAX_VERSION} jaxlib==${JAX_VERSION} jax-cuda12-plugin[with-cuda] jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        else
            echo "Installing latest jax-nightly, jaxlib-nightly"
            python3 -m pip install -U --pre jax jaxlib jax-cuda12-plugin[with-cuda] jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        fi
    elif [[ $DEVICE == "tpu" ]]; then
        echo "Installing jax-nightly, jaxlib-nightly"
        # Install jax-nightly
        python3 -m pip install --pre -U jax -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        # Install jaxlib-nightly
        python3 -m pip install --pre -U jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/

        if [[ -n "$LIBTPU_GCS_PATH" ]]; then
            # Install custom libtpu
            echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
            # Install required dependency
            python3 -m pip install -U crcmod
            # Copy libtpu.so from GCS path
            gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
        else
            # Install libtpu-nightly
            echo "Installing libtpu-nightly"
            python3 -m pip install -U --pre libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        fi
        echo "Installing nightly tensorboard plugin profile"
        python3 -m pip install tbp-nightly --upgrade
    fi
    echo "Installing nightly tensorboard plugin profile"
    python3 -m pip install tbp-nightly --upgrade
else
    echo -e "\n\nError: You can only set MODE to [stable,nightly,libtpu-only].\n\n"
    exit 1
fi
