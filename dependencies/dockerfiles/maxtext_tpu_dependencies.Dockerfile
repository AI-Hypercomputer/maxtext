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

# Install uv
RUN pip install --no-cache-dir -U uv

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

ENV MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/src/MaxText/test_assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy setup files and dependency files separately for better caching
COPY dependencies/requirements/ dependencies/requirements/
COPY src/install_maxtext_extra_deps/ src/install_maxtext_extra_deps/
COPY src/MaxText/__init__.py src/MaxText/__init__.py
COPY pyproject.toml .
COPY build_hooks.py .
COPY README.md .

# Copy the custom libtpu.so file if it exists inside maxtext repository
COPY libtpu.so* /root/custom_libtpu/

# Install MaxText with TPU dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .[tpu] --resolution=lowest; \
    uv pip install --system -r src/install_maxtext_extra_deps/extra_deps_from_github.txt;

# Version overrides for JAX, JAXLIB and LIBTPU
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$MODE" = "stable" ]; then \
        if [ "$JAX_VERSION" != "NONE" ]; then \
            echo -e "\nInstalling jax, jaxlib, libtpu version ${JAX_VERSION}"; \
            uv pip install --system -U jax[tpu]==${JAX_VERSION}; \
        fi; \
        if [ "$LIBTPU_VERSION" != "NONE" ]; then \
            echo -e "\nInstalling libtpu version ${LIBTPU_VERSION}"; \
            uv pip install --system -U --no-deps libtpu==${LIBTPU_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
        fi; \
    elif [ "$MODE" = "nightly" ]; then \
        uv pip uninstall --system jax jaxlib libtpu || true; \
        if [ "$JAX_VERSION" != "NONE" ]; then \
            echo -e "\nInstalling jax-nightly, jaxlib-nightly ${JAX_VERSION}"; \
            uv pip install --system -U --pre --no-deps jax==${JAX_VERSION} jaxlib==${JAX_VERSION} -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/; \
        else \
            echo -e "\nInstalling the latest jax-nightly, jaxlib-nightly"; \
            uv pip install --system -U --pre --no-deps jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/; \
        fi; \
        if [ "$LIBTPU_VERSION" != "NONE" ]; then \
            echo -e "\nInstalling libtpu version ${LIBTPU_VERSION}"; \
            uv pip install --system -U --no-deps libtpu==${LIBTPU_VERSION} -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
        else \
            echo -e "\nInstalling the latest libtpu-nightly"; \
            uv pip install --system -U --pre --no-deps libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
        fi; \
    fi

# Now copy the remaining code (source files that may change frequently)
COPY . .

# Download test assets from GCS if building image with test assets
ARG INCLUDE_TEST_ASSETS=false
RUN if [ "$INCLUDE_TEST_ASSETS" = "true" ]; then \
        echo "Downloading test assets from GCS..."; \
        if ! gcloud storage cp -r gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT}"; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
        fi; \
    fi