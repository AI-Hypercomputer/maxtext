# syntax=docker/dockerfile:experimental
ARG BASEIMAGE=ghcr.io/nvidia/jax:base
FROM $BASEIMAGE

# Stopgaps measure to circumvent gpg key setup issue.
RUN echo "deb [trusted=yes] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" > /etc/apt/sources.list.d/devtools-ubuntu2204-amd64.list

# Install dependencies for adjusting network rto
RUN apt-get update && apt-get install -y iproute2 ethtool lsof

# Install DNS util dependencies
RUN apt-get install -y dnsutils

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set environment variables for Google Cloud SDK
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Upgrade libcusprase to work with Jax
RUN apt-get update && apt-get install -y libcusparse-12-3

ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

ARG DEVICE
ENV ENV_DEVICE=$DEVICE

RUN mkdir -p /deps

# Set the working directory in the container
WORKDIR /deps

# Copy all files from local workspace into docker container
COPY . .
RUN ls .

RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION DEVICE=${ENV_DEVICE}"
RUN --mount=type=cache,target=/root/.cache/pip bash setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION} DEVICE=${ENV_DEVICE}


WORKDIR /deps
