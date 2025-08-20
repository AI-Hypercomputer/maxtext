# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image

# Set the working directory in the container
WORKDIR /deps

# Copy assets separately
COPY assets assets/
COPY MaxText/test_assets/ MaxText/test_assets/

# Copy all files except assets from local workspace into docker container
COPY --exclude=assets --exclude=MaxText/test_assets . .
