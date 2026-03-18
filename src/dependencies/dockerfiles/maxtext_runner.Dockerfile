# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

ARG PACKAGE_DIR
ENV PACKAGE_DIR=$PACKAGE_DIR

ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy assets separately
COPY ${PACKAGE_DIR}/maxtext/assets/ "${MAXTEXT_ASSETS_ROOT}"

# Copy all files except assets from local workspace into docker container
COPY --exclude=${PACKAGE_DIR}/maxtext/assets/ ${PACKAGE_DIR}/maxtext/ src/MaxText/
