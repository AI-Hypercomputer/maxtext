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

# ==================================
# TPU EXAMPLES
# ==================================

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt
## bash tools/setup/setup.sh MODE=stable

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt + specified jax, jaxlib, libtpu
## bash tools/setup/setup.sh MODE=stable JAX_VERSION=0.8.0

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt + custom libtpu
## bash tools/setup/setup.sh MODE=stable LIBTPU_GCS_PATH=gs://my_custom_libtpu/libtpu.so

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt + jax-nightly, jaxlib-nightly, libtpu-nightly
## bash tools/setup/setup.sh MODE=nightly

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt + specified jax-nightly, jaxlib-nightly + latest libtpu-nightly
## bash tools/setup/setup.sh MODE=nightly JAX_VERSION=0.8.2.dev20251211

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt + specified jax-nightly, jaxlib-nightly + specific libtpu-nightly
## bash tools/setup/setup.sh MODE=nightly JAX_VERSION=0.8.1 LIBTPU_VERSION=0.0.31.dev20251119+nightly

# Install dependencies in dependencies/generated_requirements/tpu-requirements.txt + jax-nightly, jaxlib-nightly + custom libtpu
## bash tools/setup/setup.sh MODE=nightly LIBTPU_GCS_PATH=gs://my_custom_libtpu/libtpu.so

# Install custom libtpu only
## bash tools/setup/setup.sh MODE=libtpu-only LIBTPU_GCS_PATH=gs://my_custom_libtpu/libtpu.so

# ==================================
# GPU EXAMPLES
# ==================================

# Install dependencies in dependencies/generated_requirements/cuda12-requirements.txt
## bash tools/setup/setup.sh MODE=stable DEVICE=gpu

# Install dependencies in dependencies/generated_requirements/cuda12-requirements.txt + specified jax, jaxlib, jax-cuda12-plugin, jax-cuda12-pjrt
## bash tools/setup/setup.sh MODE=stable DEVICE=gpu JAX_VERSION=0.4.13

# Install dependencies in dependencies/generated_requirements/cuda12-requirements.txt + jax-nightly, jaxlib-nightly
## bash tools/setup/setup.sh MODE=nightly DEVICE=gpu

# Install dependencies in dependencies/generated_requirements/cuda12-requirements.txt + specified jax, jaxlib, jax-cuda12-plugin, jax-cuda12-pjrt
## bash tools/setup/setup.sh MODE=nightly DEVICE=gpu JAX_VERSION=0.4.36.dev20241109


# Enable "exit immediately if any command fails" option
set -e
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_SUSPEND=1
export NEEDRESTART_MODE=l

# Directory Validation Check
echo "Checking current directory..."
if [[ ! -d "dependencies" || ! -d "src" ]]; then
    echo -e "\n\e[31mERROR: Critical directories not found!\e[0m"
    echo "Please run this script from the root of the MaxText repository."
    echo "Expected to find './dependencies' and './src' folders."
    exit 1
fi
echo "Directory check passed."

# Enable automatic restart of services without the need for prompting 
if command -v sudo &> /dev/null && [ -f /etc/needrestart/needrestart.conf ]; then
    sudo sed -i "s/#\$nrconf{restart} = 'i';/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf
else
   echo "Skipping editing needrestart.conf"
fi

echo "Checking Python version..."
PY_VERSION=$(python3 --version 2>&1)
if [ -z "$PY_VERSION" ]; then
  # shellcheck disable=SC2016
  >&2 printf 'No python3 is installed, installing `uv` from its install script\n'
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PY_VERSION='Python 2.7.5'
fi
PY_VERSION=${PY_VERSION##* }
PY_VERSION=${PY_VERSION%.*}
# shellcheck disable=SC2072
if [[ '3.12' > "$PY_VERSION" ]]; then
    echo -e "\n\e[31mERROR: Outdated Python Version! You are currently using ${PY_VERSION}.*, but MaxText requires Python version 3.12 or higher.\e[0m"
    # Ask the user if they want to create a virtual environment with uv
    read -p "Would you like to create a Python 3.12 virtual environment using uv? (y/n) " -n 1 -r
    echo # Move to a new line after input
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Check if uv is installed first; if not, install uv
        if ! command -v uv &> /dev/null; then
            pip install uv
        fi
        # Ask for the venv name
        read -rp "Please enter a name for your new virtual environment (default: venv-maxtext): " venv_name
        # Use a default name if the user provides no input
        if [ -z "$venv_name" ]; then
            venv_name="$HOME"'/venv-maxtext'
            echo "No name provided. Using default name: '$venv_name'"
        fi
        echo "Creating virtual environment '$venv_name' with Python 3.12..."
        python3 -m uv venv --python 3.12 "$venv_name" --seed
        printf '%s\n' "$(realpath -- "$venv_name")" >> /tmp/venv_created
        echo -e "\n\e[32mVirtual environment '$venv_name' created successfully!\e[0m"
        echo "To activate it, run the following command:"
        echo -e "\e[33m  source $venv_name/bin/activate\e[0m"
        . "$venv_name"/bin/activate
    else
        echo "Exiting. Please upgrade your Python environment to continue."
        exit 1
    fi
fi
echo "Python version check passed. Continuing with script."
echo "--------------------------------------------------"

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

python3 -m pip install -U setuptools wheel uv

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set default value for $DEVICE
if [[ -z "$DEVICE" ]]; then
    export DEVICE=tpu
fi

# Set default value for $MODE
if [[ -z "$MODE" ]]; then
  export MODE=stable
fi

# Unset optional variables if set to NONE
unset_optional_vars() {
    local optional_vars=("JAX_VERSION" "LIBTPU_VERSION" "LIBTPU_GCS_PATH")
    for var_name in "${optional_vars[@]}"; do
        if [[ ${!var_name} == NONE ]]; then
            unset "$var_name"
        fi
    done
}
unset_optional_vars

version_mismatch_warning() {
    echo -e "\n\nWARNING: You are installing a $1 version that is different from the one pinned by MaxText. This can lead to the following issues:"
    echo -e "1. Compatibility: The dependencies in the requirements file are tested and compatible with the pinned $1 version. We cannot guarantee that they will work correctly with a different $1 version."
    echo -e "2. Consistency: Installing a custom $1 version can pull in different transitive dependencies over time, making the environment non-reproducible and potentially affecting performance.\n\n"
}

install_custom_libtpu() {
    libtpu_path="$HOME/custom_libtpu/libtpu.so"
    echo -e "\nInstalling libtpu.so from $LIBTPU_GCS_PATH to $libtpu_path"
    version_mismatch_warning "libtpu"
    # Delete custom libtpu if it exists
    if [ -e "$libtpu_path" ]; then
        rm -v "$libtpu_path"
    fi
    # Install 'crcmod' to download 'libtpu.so' from GCS reliably
    python3 -m uv pip install -U crcmod
    # Copy libtpu.so from GCS path
    gsutil cp "$LIBTPU_GCS_PATH" "$libtpu_path"
}

install_maxtext_with_deps() {
    if [[ "$DEVICE" != "tpu" && "$DEVICE" != "gpu" ]]; then
      echo -e "\n\nError: DEVICE must be either 'tpu' or 'gpu'.\n\n"
      exit 1
    fi
    echo "Setting up MaxText in $MODE mode for $DEVICE device"
    if [ "$DEVICE" = "gpu" ]; then
        dep_name='dependencies/requirements/generated_requirements/cuda12-requirements.txt'
    else
        dep_name='dependencies/requirements/generated_requirements/tpu-requirements.txt'
    fi
    echo "Installing requirements from $dep_name"
    python3 -m uv pip install --resolution=lowest -r "$dep_name" \
        -r 'src/install_maxtext_extra_deps/extra_deps_from_github.txt'

    # The MaxText package is installed separately from its dependencies to optimize
    # docker image rebuild times by leveraging docker's layer caching.
    # Dependencies are installed in a separate step before MaxText code is
    # copied. This means that if MaxText code changes, but the
    # dependencies do not, docker can reuse the cached dependency layer, leading
    # to significantly faster image builds.
    if [ -f 'pyproject.toml' ]; then
        echo "Installing MaxText package without installing the dependencies (already installed)"
        python3 -m uv pip install --no-deps -e .
    fi
}

# stable mode installation
if [[ "$MODE" == "stable" ]]; then
    install_maxtext_with_deps

    if [[ $DEVICE == "tpu" ]]; then
        if [[ -n "$JAX_VERSION" ]]; then
            echo -e "\nInstalling stable jax, jaxlib, libtpu version ${JAX_VERSION}"
            version_mismatch_warning "jax"
            python3 -m uv pip install -U jax[tpu]==${JAX_VERSION}
        fi
        if [[ -n "$LIBTPU_GCS_PATH" ]]; then
            install_custom_libtpu
        elif [[ -n "$LIBTPU_VERSION" ]]; then
            echo -e "\nInstalling libtpu ${LIBTPU_VERSION}"
            version_mismatch_warning "libtpu"
            python3 -m uv pip install -U --no-deps libtpu==${LIBTPU_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        fi
    elif [[ $DEVICE == "gpu" ]]; then
        if [[ -n "$JAX_VERSION" ]]; then
            echo -e "\nInstalling stable jax, jaxlib ${JAX_VERSION}"
            version_mismatch_warning "jax"
            python3 -m uv pip install -U "jax[cuda12]==${JAX_VERSION}"
        fi
    fi
    exit 0
fi

# nightly mode installation
if [[ $MODE == "nightly" ]]; then
    install_maxtext_with_deps

    # Uninstall existing jax, jaxlib and libtpu
    python3 -m uv pip show jax && python3 -m uv pip uninstall jax
    python3 -m uv pip show jaxlib && python3 -m uv pip uninstall jaxlib
    python3 -m uv pip show libtpu && python3 -m uv pip uninstall libtpu

    if [[ $DEVICE == "tpu" ]]; then
        if [[ -n "$JAX_VERSION" ]]; then
            echo -e "\nInstalling jax-nightly, jaxlib-nightly ${JAX_VERSION}"
            python3 -m uv pip install -U --pre --no-deps jax==${JAX_VERSION} jaxlib==${JAX_VERSION} -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        else
            echo -e "\nInstalling the latest jax-nightly, jaxlib-nightly"
            python3 -m uv pip install --pre -U --no-deps jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        fi
        if [[ -n "$LIBTPU_GCS_PATH" ]]; then
            install_custom_libtpu
        elif [[ -n "$LIBTPU_VERSION" ]]; then
            echo -e "\nInstalling libtpu ${LIBTPU_VERSION}"
            version_mismatch_warning "libtpu"
            python3 -m uv pip install -U --no-deps libtpu==${LIBTPU_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        else
            echo -e "\nInstalling the latest libtpu-nightly"
            python3 -m uv pip install -U --pre --no-deps libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        fi
    elif [[ $DEVICE == "gpu" ]]; then
        if [[ -n "$JAX_VERSION" ]]; then
            echo -e "\nInstalling jax-nightly, jaxlib-nightly ${JAX_VERSION}"
            python3 -m uv pip install -U --pre --no-deps jax==${JAX_VERSION} jaxlib==${JAX_VERSION} \
                jax-cuda12-plugin[with-cuda]==${JAX_VERSION} jax-cuda12-pjrt==${JAX_VERSION} -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        else
            echo -e "\nInstalling the latest jax-nightly, jaxlib-nightly"
            python3 -m uv pip install -U --pre --no-deps jax jaxlib \
                jax-cuda12-plugin[with-cuda] jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
        fi
    fi
    exit 0
fi

# libtpu-only mode installation
if [[ "$MODE" == "libtpu-only" ]]; then
    if [[ "$DEVICE" != "tpu" ]]; then
      echo -e "\n\nError: MODE=libtpu-only is only supported for DEVICE=tpu.\n\n"
      exit 1
    fi
    if [[ -z "$LIBTPU_GCS_PATH" ]]; then
        echo -e "\n\nError: LIBTPU_GCS_PATH must be set when MODE is libtpu-only.\n\n"
        exit 1
    fi
    install_custom_libtpu
    exit 0
fi

echo -e "\n\nError: MODE must be either 'stable', 'nightly', or 'libtpu-only'.\n\n"
exit 1
