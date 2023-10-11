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
# bash setup.sh MODE={stable,nightly,head,libtpu-only} LIBTPU_GCS_PATH={gcs_path_to_custom_libtpu} SKIP_REQUIREMENTS={true,false}

# You need to specificy a MODE, default value stable. 
# You have the option to provide a LIBTPU_GCS_PATH that points to a libtpu.so provided to you by Google. 
# In head MODE and libtpu-only MODE, the LIBTPU_GCS_PATH is mandatory.
# For MODE=stable you may additionally specify JAX_VERSION, e.g. JAX_VERSION=0.4.13
# Set SKIP_REQUIREMENTS=true if you don't want to install packages from requirements.txt, any other value
# will install packages from requirements.txt.


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

if [[ $SKIP_REQUIREMENTS == NONE ]]; then
    SKIP_REQUIREMENTS="false"
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
    echo "Installing stable tensorboard plugin profile"
    pip3 install tensorboard-plugin-profile --upgrade
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
    echo "Installing nightly tensorboard plugin profile"
    pip3 install tbp-nightly --upgrade
elif [[ $MODE == "head" ]]; then 
# Head mode
    echo "Installing jax and jaxlib from head"
    if [ -d "$HOME/jax" ]; then
        cd "$HOME/jax" && git pull origin main
    else
        cd $HOME && git clone https://github.com/google/jax.git
    fi
    if [ -d "$HOME/xla" ]; then
        cd "$HOME/xla" && git pull origin main
    else
        cd $HOME && git clone https://github.com/openxla/xla.git
    fi
    cd $HOME/jax
    pip3 install numpy wheel build
    python3 build/build.py --enable_tpu --bazel_options=--override_repository=xla=$HOME/xla
    pip3 install dist/*.whl --force-reinstall --no-deps # installs jaxlib (includes XLA)
    pip3 install -e .  # installs jax
    if [[ -n "$LIBTPU_GCS_PATH" ]]; then
        installed_libtpu_path=$(python3 -c '
        import os; import jax; print(os.environ["TPU_LIBRARY_PATH"])
        ')
        # Install custom libtpu
        echo "Installing libtpu.so from $LIBTPU_GCS_PATH to $installed_libtpu_path"
        # Install required dependency
        pip3 install -U crcmod
        # Copy libtpu.so from GCS path
        gsutil cp "$LIBTPU_GCS_PATH" "$installed_libtpu_path"
    else
        echo -e "\n\nError: You must provide a custom libtpu for head mode.\n\n"
        exit 1
    fi

    echo "Installing nightly tensorboard plugin profile"
    pip3 install tbp-nightly --upgrade
else
    echo -e "\n\nError: You can only set MODE to [stable,nightly,head,libtpu-only].\n\n"
    exit 1
fi

if [[ $SKIP_REQUIREMENTS != "true" ]]; then 
    # Install dependencies from requirements.txt
    cd $run_name_folder_path && pip3 install -r requirements.txt
fi
