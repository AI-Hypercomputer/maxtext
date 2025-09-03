# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=src/MaxText_base_image
FROM $BASEIMAGE

#FROM src/MaxText_base_image

ENV MAXTEXT_ASSETS_ROOT=/deps/assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy assets separately
COPY assets assets/
COPY src/MaxText/test_assets src/MaxText/test_assets/

# Copy all files except assets from local workspace into docker container
COPY --exclude=assets --exclude=src/MaxText/test_assets . .
