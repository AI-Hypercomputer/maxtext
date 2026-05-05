FROM gcr.io/cloud-tpu-multipod-dev/mohit-rl:0428-tp-0.9.2

# Set working directory
WORKDIR /deps

# Install UV for faster and cleaner dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system -U orbax-checkpoint

# Copy the current MaxText folder into /deps/maxtext
COPY . /deps
WORKDIR /deps





# 1. Clone and install tpu-inference to get the compatible vLLM version
RUN cd /tpu-inference && \
    git fetch && \
    git checkout da4b9c8d313ed8a70b5701175ccab0c05645d3f6 && \
    # Extract the LKG vLLM commit hash
    VLLM_COMMIT_HASH=$(cat .buildkite/vllm_lkg.version) && \
    echo "Using vLLM commit: ${VLLM_COMMIT_HASH}" && \
    # Clone and pin vLLM
    cd /vllm && \
    git fetch && \
    git checkout "${VLLM_COMMIT_HASH}" && \
    # Install vLLM with TPU requirements
    uv pip install --system -r requirements/tpu.txt && \
    VLLM_TARGET_DEVICE="tpu" uv pip install --system -e . && \
    # Cleanup vLLM build artifacts
    rm -rf build/ dist/ *.egg-info/ && \
    find . -name "*.o" -type f -delete && \
    find . -name "*.a" -type f -delete && \
    # Go back and install tpu-inference
    # pip install -U \
    # libtpu==0.0.39.dev20260403 \
    # -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    cd /tpu-inference && \
    uv pip install --system -e . && \

    # Cleanup tpu-inference build artifacts
    rm -rf build/ dist/ *.egg-info/ && \
    find . -name "*.o" -type f -delete && \
    find . -name "*.a" -type f -delete && \
    # Global cleanup
    uv cache clean && \
    pip cache purge

# Install MaxText dependencies and final cleanup
RUN uv pip install --system -e .[tpu] && \
    rm -rf build/ dist/ *.egg-info/ && \
    uv cache clean && \
    pip cache purge && \
    rm -rf /root/.cache/pip /root/.cache/uv /root/.cache/huggingface



# Environment setup for Pathways and custom paths
ENV JAX_PLATFORMS=proxy,cpu
ENV JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
ENV PYTHONPATH=$PYTHONPATH:/vllm:/tpu-inference:/deps/src
