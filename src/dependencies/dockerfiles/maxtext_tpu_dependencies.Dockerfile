# syntax=docker/dockerfile:experimental

ARG BASEIMAGE=python:3.12-slim-trixie

# Stage 0: Bootstrap apt-transport-artifact-registry (ar+https) when USE_AIRLOCK=true
FROM $BASEIMAGE AS airlock-bootstrap
ARG USE_AIRLOCK=false
RUN mkdir -p /airlock-apt-methods && \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates gpg && \
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-transport-artifact-registry && \
        cp -a /usr/lib/apt/methods/ar+https /airlock-apt-methods/; \
    fi

FROM $BASEIMAGE
ARG USE_AIRLOCK=false
COPY --from=airlock-bootstrap /airlock-apt-methods/ /usr/lib/apt/methods/

# Install system dependencies including C++20 compiler for vLLM (uses Airlock apt repo when USE_AIRLOCK=true)
RUN --mount=type=secret,id=credentials,target=/root/.config/gcloud/application_default_credentials.json,required=false \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        if [ -s /root/.config/gcloud/application_default_credentials.json ]; then export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json; fi && \
        CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME:-trixie}") && \
        rm -f /etc/apt/sources.list.d/debian.sources && \
        printf "deb [trusted=yes] ar+https://us-apt.pkg.dev/remote/artifact-foundry-prod/debian-3p-remote-%s %s main\ndeb [trusted=yes] ar+https://us-apt.pkg.dev/remote/artifact-foundry-prod/debian-3p-remote-%s-security %s-security main\n" "$CODENAME" "$CODENAME" "$CODENAME" "$CODENAME" > /etc/apt/sources.list; \
    fi && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc-12 g++-12 build-essential cmake ninja-build curl gnupg && \
    rm -rf /var/lib/apt/lists/*

# Add the Google Cloud SDK package repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg > /tmp/apt-key.gpg && \
    gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg /tmp/apt-key.gpg && \
    rm /tmp/apt-key.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Install the Google Cloud SDK
RUN --mount=type=secret,id=credentials,target=/root/.config/gcloud/application_default_credentials.json,required=false \
    if [ "$USE_AIRLOCK" = "true" ] && [ -s /root/.config/gcloud/application_default_credentials.json ]; then export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json; fi && \
    apt-get update && apt-get install -y google-cloud-cli && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.12 and default GCC/G++ to 12 (matching bookworm)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Upgrade pip, setuptools, wheel, uv and clean up ensurepip bundled cache
RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel uv && \
    python3 -c 'import ensurepip, os, shutil; shutil.rmtree(os.path.join(os.path.dirname(ensurepip.__file__), "_bundled"), ignore_errors=True)'

# Set environment variables for Google Cloud SDK, Python 3.12, and GCC 12
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.12:${PATH}"
ENV CC=gcc-12
ENV CXX=g++-12

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

# TODO: remove default once separate TF and TF-free nightly image workflows are established
ARG TF=true
ENV ENV_TF=$TF

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
RUN echo "Running command: bash setup.sh MODE=$ENV_MODE WORKFLOW=$ENV_WORKFLOW JAX_VERSION=$ENV_JAX_VERSION LIBTPU_VERSION=$ENV_LIBTPU_VERSION DEVICE=${ENV_DEVICE} TF=${ENV_TF}"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=credentials,target=/root/.config/gcloud/application_default_credentials.json,required=false \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        if [ -s /root/.config/gcloud/application_default_credentials.json ]; then export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json; fi && \
        TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null || true) && \
        if [ -n "$TOKEN" ] && curl -fsSL -H "Authorization: Bearer ${TOKEN}" "https://us-python.pkg.dev/artifact-foundry-prod/python-3p-trusted/simple/" >/dev/null 2>&1; then \
            export UV_INDEX_URL="https://oauth2accesstoken:${TOKEN}@us-python.pkg.dev/artifact-foundry-prod/python-3p-trusted/simple/" && \
            export UV_EXTRA_INDEX_URL="https://pypi.org/simple/" && \
            export UV_INDEX_STRATEGY="unsafe-best-match"; \
        fi; \
    fi && \
    export UV_LINK_MODE=copy && \
    bash /deps/src/dependencies/scripts/setup.sh MODE=${ENV_MODE} WORKFLOW=${ENV_WORKFLOW} JAX_VERSION=${ENV_JAX_VERSION} LIBTPU_VERSION=${ENV_LIBTPU_VERSION} DEVICE=${ENV_DEVICE} TF=${ENV_TF} && \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME:-trixie}") && \
        echo "deb http://deb.debian.org/debian ${CODENAME} main" > /etc/apt/sources.list; \
    fi

# Now copy the remaining code (source files that may change frequently)
COPY ${PACKAGE_DIR}/maxtext/ src/maxtext/
# Now copy resource needed for pytest:
COPY tests*/ tests/
COPY pytest.ini* pytest.ini
COPY benchmarks*/ benchmarks/

# Download test assets from GCS if building image with test assets
ARG INCLUDE_TEST_ASSETS=false
RUN if [ "$INCLUDE_TEST_ASSETS" = "true" ]; then \
        echo "Downloading test assets from GCS..."; \
        if ! gcloud storage cp -r gs://maxtext-test-assets/* "${MAXTEXT_TEST_ASSETS_ROOT}/golden_logits"; then \
        echo "WARNING: Failed to download test assets from GCS. These files are only used for end-to-end tests; you may not have access to the bucket."; \
        fi; \
    fi

ENV PYTHONPATH="/deps/src${PYTHONPATH:+:${PYTHONPATH}}"
