# Use the base image identified from your running container
FROM gcr.io/tpu-prod-env-multipod/mohit-ep-repl:20260402-a

# Set working directory
WORKDIR /deps

# Install UV for faster and cleaner dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone and install tpu-inference to get the compatible vLLM version
RUN git clone https://github.com/vllm-project/tpu-inference.git /tpu-inference && \
    cd /tpu-inference && \
    git checkout e5991df669dc27d7cab92e4aeff55ae44f015f40 && \
    # Extract the LKG vLLM commit hash
    VLLM_COMMIT_HASH=$(cat .buildkite/vllm_lkg.version) && \
    echo "Using vLLM commit: ${VLLM_COMMIT_HASH}" && \
    # Clone and pin vLLM
    git clone https://github.com/vllm-project/vllm.git /vllm && \
    cd /vllm && \
    git checkout "${VLLM_COMMIT_HASH}" && \
    # Install vLLM with TPU requirements
    uv pip install --system -r requirements/tpu.txt && \
    VLLM_TARGET_DEVICE="tpu" uv pip install --system -e . && \
    # Cleanup vLLM build artifacts
    rm -rf build/ dist/ *.egg-info/ && \
    find . -name "*.o" -type f -delete && \
    find . -name "*.a" -type f -delete && \
    # Go back and install tpu-inference
    pip install --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html libtpu==0.0.39.dev20260403 && \ 
    cd /tpu-inference && \
    uv pip install --system -e . && \
    # Cleanup tpu-inference build artifacts
    rm -rf build/ dist/ *.egg-info/ && \
    find . -name "*.o" -type f -delete && \
    find . -name "*.a" -type f -delete && \
    # Global cleanup
    uv cache clean && \
    pip cache purge

# Copy the current MaxText folder into /deps/maxtext
COPY . /deps
WORKDIR /deps

# Install MaxText dependencies and final cleanup
RUN uv pip install --system -e . && \
    rm -rf build/ dist/ *.egg-info/ && \
    uv cache clean && \
    pip cache purge && \
    rm -rf /root/.cache/pip /root/.cache/uv /root/.cache/huggingface

# Environment setup for Pathways and custom paths
ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
ENV PYTHONPATH=$PYTHONPATH:/vllm:/tpu-inference:/deps/src

