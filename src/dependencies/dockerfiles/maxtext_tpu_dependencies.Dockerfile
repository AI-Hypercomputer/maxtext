# syntax=docker/dockerfile:experimental

ARG BASEIMAGE=python:3.12-slim-bullseye
FROM $BASEIMAGE

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the default Python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

# Set environment variables for Google Cloud SDK and Python 3.12
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.12:${PATH}"

# Set environment variables via build arguments
ARG MODE
ENV ENV_MODE=$MODE

ARG WORKFLOW
ENV ENV_WORKFLOW=$WORKFLOW

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

ARG LIBTPU_VERSION
ENV ENV_LIBTPU_VERSION=$LIBTPU_VERSION

ARG DEVICE
ENV ENV_DEVICE=$DEVICE

ARG PACKAGE_DIR
ENV PACKAGE_DIR=$PACKAGE_DIR

ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/maxtext
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy setup files and dependency files separately for better caching
COPY ${PACKAGE_DIR}/dependencies/extra_deps/ src/dependencies/extra_deps/
COPY ${PACKAGE_DIR}/dependencies/requirements/ src/dependencies/requirements/
COPY ${PACKAGE_DIR}/dependencies/scripts/ src/dependencies/scripts/
COPY ${PACKAGE_DIR}/maxtext/integration/vllm/ src/maxtext/integration/vllm/

# Copy the custom libtpu.so file if it exists
COPY libtpu.so* /root/custom_libtpu/

# Install dependencies - these steps are cached unless the copied files change
RUN echo "Running command: bash setup.sh MODE=$ENV_MODE WORKFLOW=$ENV_WORKFLOW JAX_VERSION=$ENV_JAX_VERSION LIBTPU_VERSION=$ENV_LIBTPU_VERSION DEVICE=${ENV_DEVICE}"
RUN --mount=type=cache,target=/root/.cache/uv \
    export UV_LINK_MODE=copy && \
    bash /deps/src/dependencies/scripts/setup.sh MODE=${ENV_MODE} WORKFLOW=${ENV_WORKFLOW} JAX_VERSION=${ENV_JAX_VERSION} LIBTPU_VERSION=${ENV_LIBTPU_VERSION} DEVICE=${ENV_DEVICE}

# Now copy the remaining code (source files that may change frequently)
COPY ${PACKAGE_DIR}/maxtext/ src/maxtext/
COPY ${PACKAGE_DIR}/MaxText/ src/MaxText/
COPY tests*/ tests/
COPY benchmarks*/ benchmarks/

# Download test assets from GCS if building image with test assets
ARG INCLUDE_TEST_ASSETS=false
RUN if [ "$INCLUDE_TEST_ASSETS" = "true" ]; then \
        echo "Downloading test assets from GCS..."; \
        if ! gcloud storage cp -r gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT}/golden_logits"; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
        fi; \
    fi

ENV PYTHONPATH="/deps/src:${PYTHONPATH}"
