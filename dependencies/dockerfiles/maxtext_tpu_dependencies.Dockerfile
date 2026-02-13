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

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

ARG LIBTPU_VERSION
ENV ENV_LIBTPU_VERSION=$LIBTPU_VERSION

ARG DEVICE
ENV ENV_DEVICE=$DEVICE

ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy setup files and dependency files separately for better caching
COPY tools/setup tools/setup/
COPY dependencies/requirements/ dependencies/requirements/
COPY src/install_maxtext_extra_deps/extra_deps_from_github.txt src/install_maxtext_extra_deps/

# Copy the custom libtpu.so file if it exists inside maxtext repository
COPY libtpu.so* /root/custom_libtpu/

# Install dependencies - these steps are cached unless the copied files change
RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION LIBTPU_VERSION=$ENV_LIBTPU_VERSION DEVICE=${ENV_DEVICE}"
RUN --mount=type=cache,target=/root/.cache/pip bash /deps/tools/setup/setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION} LIBTPU_VERSION=${ENV_LIBTPU_VERSION} DEVICE=${ENV_DEVICE}

# Now copy the remaining code (source files that may change frequently)
COPY . .

# Download test assets from GCS if building image with test assets
ARG INCLUDE_TEST_ASSETS=false
RUN if [ "$INCLUDE_TEST_ASSETS" = "true" ]; then \
        echo "Downloading test assets from GCS..."; \
        if ! gcloud storage cp -r gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT}/golden_logits"; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
        fi; \
    fi

# Install (editable) MaxText
RUN test -f '/tmp/venv_created' && "$(tail -n1 /tmp/venv_created)"/bin/activate ; pip install --no-dependencies -e .
