# syntax=docker/dockerfile:experimental
ARG BASEIMAGE=ubuntu:24.04

# Stage 0: Bootstrap apt-transport-artifact-registry when USE_AIRLOCK=true
FROM $BASEIMAGE AS airlock-bootstrap
ARG USE_AIRLOCK=false
RUN mkdir -p /airlock-apt-methods && \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg > /tmp/apt-key.gpg && \
        gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg /tmp/apt-key.gpg && \
        rm /tmp/apt-key.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-transport-artifact-registry && \
        cp -a /usr/lib/apt/methods/ar+https /airlock-apt-methods/ 2>/dev/null || true; \
    fi

FROM $BASEIMAGE

ARG USE_AIRLOCK=false
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=airlock-bootstrap /airlock-apt-methods/ /usr/lib/apt/methods/

# Install Python 3.12, build tools, and network/DNS utilities on Ubuntu 24.04 (uses Airlock apt repo when USE_AIRLOCK=true)
RUN --mount=type=secret,id=credentials,target=/root/.config/gcloud/application_default_credentials.json,required=false \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        if [ -s /root/.config/gcloud/application_default_credentials.json ]; then export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json; fi && \
        rm -f /etc/apt/sources.list.d/ubuntu.sources && \
        printf "deb [trusted=yes] ar+https://us-apt.pkg.dev/remote/artifact-foundry-prod/ubuntu-3p-remote-noble noble main restricted universe multiverse\ndeb [trusted=yes] ar+https://us-apt.pkg.dev/remote/artifact-foundry-prod/ubuntu-3p-remote-noble-updates noble-updates main restricted universe multiverse\ndeb [trusted=yes] ar+https://us-apt.pkg.dev/remote/artifact-foundry-prod/ubuntu-3p-remote-noble-security noble-security main restricted universe multiverse\n" > /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        build-essential gcc-12 g++-12 cmake ninja-build pkg-config \
        git curl gnupg ca-certificates iproute2 ethtool lsof bind9-dnsutils && \
    rm -rf /var/lib/apt/lists/*

# Set default python/python3 and gcc/g++ alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Move the 'EXTERNALLY-MANAGED' file to allow system-wide pip/uv installs
RUN if [ -f /usr/lib/python3.12/EXTERNALLY-MANAGED ]; then \
    mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.old; \
fi

# Add the Google Cloud SDK package repository
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg > /tmp/apt-key.gpg && \
    gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg /tmp/apt-key.gpg && \
    rm /tmp/apt-key.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Install the Google Cloud SDK
RUN --mount=type=secret,id=credentials,target=/root/.config/gcloud/application_default_credentials.json,required=false \
    if [ "$USE_AIRLOCK" = "true" ] && [ -s /root/.config/gcloud/application_default_credentials.json ]; then export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json; fi && \
    apt-get update && apt-get install -y google-cloud-cli && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel, uv, install nvidia-nvtx-cu12/nvidia-nccl-cu12 for transformer-engine-jax build, and clean up ensurepip cache
RUN python3 -m pip install --upgrade --no-cache-dir --ignore-installed pip setuptools wheel uv keyrings.google-artifactregistry-auth nvidia-nvtx-cu12 nvidia-nccl-cu12 && \
    ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/nvtx/include/nvtx3 /usr/local/include/nvtx3 && \
    ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2 && \
    ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so && \
    python3 -c 'import ensurepip, os, shutil; shutil.rmtree(os.path.join(os.path.dirname(ensurepip.__file__), "_bundled"), ignore_errors=True)'

# Set environment variables for Google Cloud SDK, GCC 12, and NVIDIA container runtime
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvcc/bin:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
ENV CC=gcc-12
ENV CXX=g++-12
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ARG MODE
ENV ENV_MODE=$MODE

ARG JAX_VERSION
ENV ENV_JAX_VERSION=$JAX_VERSION

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

# Install dependencies (including CUDA and JAX wheels from Airlock when USE_AIRLOCK=true)
RUN echo "Running command: bash setup.sh MODE=$ENV_MODE JAX_VERSION=$ENV_JAX_VERSION DEVICE=${ENV_DEVICE} TF=${ENV_TF}"
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
    bash /deps/src/dependencies/scripts/setup.sh MODE=${ENV_MODE} JAX_VERSION=${ENV_JAX_VERSION} DEVICE=${ENV_DEVICE} TF=${ENV_TF} && \
    uv pip install --system nvidia-curand-cu12 && \
    ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime /usr/local/lib/python3.12/dist-packages/nvidia/cuda_cudart && \
    for so in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib/lib*.so.*; do \
        if [ -f "$so" ]; then \
            base=$(basename "$so" | sed 's/\.so\..*/.so/'); \
            ln -sf "$so" "/usr/lib/x86_64-linux-gnu/$base"; \
            ln -sf "$so" "/usr/lib/x86_64-linux-gnu/$(basename "$so")"; \
        fi; \
    done && \
    if [ "$USE_AIRLOCK" = "true" ]; then \
        echo "deb http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse" > /etc/apt/sources.list; \
    fi && \
    find /usr/local/lib -type d -path "*/nvidia/*/lib" > /etc/ld.so.conf.d/nvidia-wheels.conf && \
    ldconfig

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
