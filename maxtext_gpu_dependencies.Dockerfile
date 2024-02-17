FROM ghcr.io/nvidia/jax:maxtext

# Install dependencies for adjusting network rto
RUN apt-get update && apt-get install -y iproute2 ethtool lsof

# Install the Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash

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

WORKDIR /app