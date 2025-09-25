# syntax=docker/dockerfile:experimental
# Use Python 3.12 as the base image
FROM python:3.12-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the default Python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

# Set environment variables for Google Cloud SDK and Python 3.12
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.12:${PATH}"

# Set environment variables via build arguments
ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

ARG LIBTPU_GCS_PATH
ENV ENV_LIBTPU_GCS_PATH=$LIBTPU_GCS_PATH

ARG DEVICE
ENV ENV_DEVICE=$DEVICE

ENV MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/src/MaxText/test_assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

COPY libtpu-0.0.24.dev20250918+tpu7x-cp314-cp314t-manylinux_2_31_x86_64.whl ./
# Copy setup files and dependency files separately for better caching
COPY setup.sh ./
COPY requirements.txt requirements_with_jax_ai_image.txt ./

# Install dependencies - these steps are cached unless the copied files change
RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION LIBTPU_GCS_PATH=${ENV_LIBTPU_GCS_PATH} DEVICE=${ENV_DEVICE}"
RUN --mount=type=cache,target=/root/.cache/pip bash setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION} LIBTPU_GCS_PATH=${ENV_LIBTPU_GCS_PATH} DEVICE=${ENV_DEVICE}

# Now copy the remaining code (source files that may change frequently)
COPY . .

# Install (editable) MaxText
RUN test -f '/tmp/venv_created' && "$(tail -n1 /tmp/venv_created)"/bin/activate ; pip install --no-dependencies -e .
