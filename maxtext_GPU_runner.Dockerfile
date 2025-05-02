# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image

# Set the working directory in the container
WORKDIR /workspace

# Copy assets separately 
COPY assets/ .
COPY MaxText/test_assets/ MaxText/.

# Copy all files except assets from local workspace into docker container
COPY --exclude=assets --exclude=MaxText/test_assets . .

RUN python3 -m pip install mlperf-logging@git+https://github.com/mlperf/logging.git
WORKDIR /workspace
