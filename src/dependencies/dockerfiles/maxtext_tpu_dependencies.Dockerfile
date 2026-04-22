# syntax=docker/dockerfile:1.4

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

ENV MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/tests/assets
ENV MAXTEXT_PKG_DIR=/deps/src/maxtext
ENV MAXTEXT_REPO_ROOT=/deps

# Set the working directory in the container
WORKDIR /deps

# Copy setup files and dependency files separately for better caching
COPY tools/setup tools/setup/
COPY src/dependencies/requirements/ src/dependencies/requirements/
COPY src/install_maxtext_extra_deps/ src/install_maxtext_extra_deps/
COPY src/maxtext/integration/vllm/ src/maxtext/integration/vllm/

# Copy the custom libtpu.so file if it exists inside maxtext repository
COPY libtpu.so* /root/custom_libtpu/

# Install dependencies - these steps are cached unless the copied files change
RUN echo "Running command: bash setup.sh MODE=$ENV_MODE WORKFLOW=$ENV_WORKFLOW JAX_VERSION=$ENV_JAX_VERSION LIBTPU_VERSION=$ENV_LIBTPU_VERSION DEVICE=${ENV_DEVICE}"
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.cache/uv bash /deps/tools/setup/setup.sh MODE=${ENV_MODE} WORKFLOW=${ENV_WORKFLOW} JAX_VERSION=${ENV_JAX_VERSION} LIBTPU_VERSION=${ENV_LIBTPU_VERSION} DEVICE=${ENV_DEVICE}

# Monkey-patch tpu_inference.utils.hbm_usages_bytes for multi-host TPU support.
# Workaround for https://github.com/vllm-project/tpu-inference/pull/2268 —
# remove once a tpu-inference release with this fix is in the post-training deps.
RUN python3 - <<'EOF'
import pathlib, sys

site_packages = next(p for p in sys.path if "site-packages" in p and pathlib.Path(p).is_dir())
sitecustomize = pathlib.Path(site_packages) / "sitecustomize.py"

patch = """
try:
    import tpu_inference.utils as _tpi_utils
    def _hbm_usages_bytes_multihost(devices):
        import jax as _jax
        current_process = _jax.process_index()
        usage = []
        for device in devices:
            if device.process_index != current_process:
                continue
            hbm_used = device.memory_stats()["bytes_in_use"]
            hbm_limit = device.memory_stats()["bytes_limit"]
            usage.append((hbm_used, hbm_limit))
        if usage and len(usage) < len(list(devices)):
            usage = [usage[0]] * len(list(devices))
        return usage
    _tpi_utils.hbm_usages_bytes = _hbm_usages_bytes_multihost
except ImportError:
    pass
"""

existing = sitecustomize.read_text() if sitecustomize.exists() else ""
sitecustomize.write_text(existing + patch)
print(f"Wrote monkey-patch to {sitecustomize}")
EOF

# Install lm-eval before copying source so this layer is cached across source changes.
# vLLM + tpu-inference are installed by setup.sh when WORKFLOW=post-training.
# The MaxText vLLM adapter install (pip install -e) is deferred until after COPY . . below.
RUN pip install "lm-eval[api]"

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

# Install the MaxText vLLM adapter (requires source to be present)
RUN pip install --no-deps -e src/maxtext/integration/vllm/

# Install (editable) MaxText
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.cache/uv test -f '/tmp/venv_created' && "$(tail -n1 /tmp/venv_created)"/bin/activate ; pip install --no-dependencies -e .
