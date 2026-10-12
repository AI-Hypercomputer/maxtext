# This Dockerfile is used to build a Docker image containing all the
# system-level dependencies required for maxtext. Since the maxtext
# package itself includes all the necessary Python dependencies, this
# image is designed to have a clean Python environment with just pip
# and uv installed.
#
# Usage:
#   docker build --build-arg DEVICE=cuda12 -t maxtext-unit-test-cuda12:py312 -f src/dependencies/dockerfiles/clean_py_env_gpu.Dockerfile .
#   gcloud auth login
#   gcloud config set project tpu-prod-env-multipod
#   gcloud auth configure-docker us-docker.pkg.dev
#   docker tag maxtext-unit-test-cuda12:py312 us-docker.pkg.dev/tpu-prod-env-multipod/maxtext-images/maxtext-unit-test-cuda12:py312
#   docker push us-docker.pkg.dev/tpu-prod-env-multipod/maxtext-images/maxtext-unit-test-cuda12:py312

# Default to Python 3.12 and Ubuntu 24.04 (can be overridden with Airlock mirror).
ARG PYTHON_VERSION=3.12
ARG BASEIMAGE=ubuntu:24.04

FROM ${BASEIMAGE}

# Set the working directory in the container
WORKDIR /maxtext

# Arguments
ARG DEVICE

# Environment variables
ENV DEVICE=$DEVICE
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CLOUD_SDK_VERSION=latest
# Ensure apt package installations run without manual intervention
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# See all env variables
RUN env

# Update the package lists and install Python 3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip
# Remove any existing pip at /usr/bin/pip
RUN rm -f /usr/bin/pip
# Set python3 and pip3 as the default python and pip commands
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1


# System level dependencies
RUN apt-get update && apt-get install -y apt-utils git build-essential cmake curl pkg-config gnupg procps iproute2 ethtool lsb-release && rm -rf /var/lib/apt/lists/*
# Add the Google Cloud SDK package repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-cli && rm -rf /var/lib/apt/lists/*
# Install gcsfuse
RUN export GCSFUSE_REPO=gcsfuse-bullseye && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update -y && apt-get -y install gcsfuse && \
    rm -rf /var/lib/apt/lists/*
# See all installed system level packages
RUN apt list --installed

RUN python3 --version
RUN python3 -m pip list
RUN python3 -m pip install uv
RUN rm -rf /root/.cache/pip

# Clean python env
RUN python3 -m uv pip list
