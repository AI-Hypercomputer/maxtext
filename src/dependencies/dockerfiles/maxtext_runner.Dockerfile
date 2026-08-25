# syntax=docker.io/docker/dockerfile:1.7-labs

ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

ARG PACKAGE_DIR
ENV PACKAGE_DIR=$PACKAGE_DIR

ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/maxtext
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Install GDN v3 Tokamax commit
RUN pip install --no-cache-dir xprof && pip install --no-deps --no-cache-dir --force-reinstall git+https://github.com/openxla/tokamax.git@e777ee5087d1e05145ca320cda9439c8ac62e64a

# Copy assets separately
COPY ${PACKAGE_DIR}/maxtext/assets/ "${MAXTEXT_ASSETS_ROOT}"

# Copy all files except assets from local workspace into docker container
COPY --exclude=${PACKAGE_DIR}/maxtext/assets/ ${PACKAGE_DIR}/maxtext/ src/maxtext/
