# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image

ENV MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy assets separately
COPY src/MaxText/assets/ "${MAXTEXT_ASSETS_ROOT}"
COPY tests/assets/ "${MAXTEXT_TEST_ASSETS_ROOT}"

# Copy all files except assets from local workspace into docker container
COPY --exclude="${MAXTEXT_ASSETS_ROOT}" --exclude="${MAXTEXT_TEST_ASSETS_ROOT}" . .
