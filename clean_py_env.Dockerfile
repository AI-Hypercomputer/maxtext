# This Dockerfile is used to build a Docker image containing all the
# system-level dependencies required for maxtext. Since the maxtext
# package itself includes all the necessary Python dependencies, this
# image is designed to have a clean Python environment with just pip
# and uv installed.
#
# Build a docker image by running:
# `docker build --build-arg DEVICE=tpu -t <docker_image_name>:<tag> -f clean_py_env.Dockerfile .`
# Or
# `docker build --build-arg PYTHON_VERSION=3.11 --build-arg DEVICE=tpu -t <docker_image_name>:<tag> -f clean_py_env.Dockerfile .`
#
# How to upload the image to Google Container Registry (GCR):
# e.g., DEVICE=tpu and PYTHON_VERSION=3.12
#.  gcloud init
#   gcloud auth configure-docker
#   docker tag <docker_image_name>:<tag> gcr.io/tpu-prod-env-multipod/maxtext-unit-test-tpu:py312
#   docker push gcr.io/tpu-prod-env-multipod/maxtext-unit-test-tpu:py312

# Default to Python 3.12. This ARG must be declared before FROM.
ARG PYTHON_VERSION=3.12

# Use the PYTHON_VERSION ARG to specify the base image tag
FROM python:${PYTHON_VERSION}-slim-bullseye

# Set the working directory in the container
WORKDIR /maxtext

# Arguments
ARG DEVICE

# Enviroment variables
ENV DEVICE=$DEVICE
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CLOUD_SDK_VERSION=latest
# Ensure apt package installations run without manual intervention
ENV DEBIAN_FRONTEND=noninteractive
# See all env variables
RUN env

# System level dependencies
RUN apt-get update && apt-get install -y apt-utils git curl gnupg procps iproute2 ethtool cmake pkg-config build-essential dnsutils && rm -rf /var/lib/apt/lists/*
# Add the Google Cloud SDK package repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk && rm -rf /var/lib/apt/lists/*
# Install gcsfuse
RUN export GCSFUSE_REPO=gcsfuse-bullseye && \
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt update -y && apt -y install gcsfuse && \
    rm -rf /var/lib/apt/lists/*
# See all installed system level packages
RUN apt list --installed

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip uv
RUN rm -rf /root/.cache/pip

# Clean python env
RUN python -m uv pip list
