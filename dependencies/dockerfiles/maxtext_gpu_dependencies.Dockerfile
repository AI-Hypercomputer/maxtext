# syntax=docker/dockerfile:experimental
ARG BASEIMAGE=ghcr.io/nvidia/jax:base
FROM $BASEIMAGE

# Move the 'EXTERNALLY-MANAGED' file to allow system-wide pip installs
RUN if [ -f /usr/lib/python3.12/EXTERNALLY-MANAGED ]; then \
    mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.old; \
fi

# Stopgaps measure to circumvent gpg key setup issue.
RUN echo "deb [trusted=yes] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" > /etc/apt/sources.list.d/devtools-ubuntu2204-amd64.list

# Install dependencies for adjusting network rto
RUN apt-get update && apt-get install -y iproute2 ethtool lsof

# Install DNS util dependencies
RUN apt-get install -y dnsutils

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Install uv
RUN pip install --no-cache-dir -U uv

# Set environment variables for Google Cloud SDK
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Upgrade libcusprase to work with Jax
RUN apt-get update && apt-get install -y libcusparse-12-6

ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

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

# Install MaxText with GPU dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .[cuda12] --resolution=lowest; \
    uv pip install --system -r src/install_maxtext_extra_deps/extra_deps_from_github.txt;

# Version overrides for JAX & JAXLIB
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$MODE" = "stable" ]; then \
        if [ "$JAX_VERSION" != "NONE" ]; then \
            echo -e "\nInstalling jax, jaxlib version ${JAX_VERSION}"; \
            uv pip install --system -U jax[cuda12]==${JAX_VERSION}; \
        fi; \
    elif [ "$MODE" = "nightly" ]; then \
        uv pip uninstall --system jax jaxlib libtpu || true; \
        if [ "$JAX_VERSION" != "NONE" ]; then \
            echo -e "\nInstalling jax-nightly, jaxlib-nightly ${JAX_VERSION}"; \
            uv pip install --system -U --pre --no-deps jax==${JAX_VERSION} jaxlib==${JAX_VERSION} \
                jax-cuda12-plugin[with-cuda]==${JAX_VERSION} jax-cuda12-pjrt==${JAX_VERSION} -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/; \
        else \
            echo -e "\nInstalling the latest jax-nightly, jaxlib-nightly"; \
            uv pip install --system -U --pre --no-deps jax jaxlib \
                jax-cuda12-plugin[with-cuda] jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/; \
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
