# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
echo "Installing GRPO dependencies (vLLM, tpu-inference) with MODE=${MODE}"
uv pip uninstall -y jax jaxlib libtpu

uv pip install aiohttp==3.12.15

# Install Python packages that enable uv pip to authenticate with Google Artifact Registry automatically.
uv pip install keyring keyrings.google-artifactregistry-auth

uv pip install numba==0.61.2

uv pip uninstall tunix
uv pip install -e ../tunix --no-cache-dir


VLLM_TARGET_DEVICE="tpu" uv pip install -e ../vllm --no-cache-dir --pre \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html \
    --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html 


uv pip install -e ../tpu-inference --no-cache-dir --pre \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html

# # Install vLLM for Jax and TPUs from the artifact registry
# VLLM_TARGET_DEVICE="tpu" uv pip install --no-cache-dir --pre \
#     --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
#     --extra-index-url https://pypi.org/simple/ \
#     --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
#     --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
#     --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
#     --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
#     --find-links https://storage.googleapis.com/libtpu-releases/index.html \
#     --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
#     --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html \
#     vllm==0.11.1rc1.dev292+g1b86bd8e1.tpu

# # Install tpu-commons from the artifact registry
# uv pip install --no-cache-dir --pre \
#     --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
#     --extra-index-url https://pypi.org/simple/ \
#     --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
#     --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
#     tpu-commons==0.1.2

# # Uninstall existing jax to avoid conflicts
# # uv pip uninstall -y jax jaxlib libtpu

# # --- STAGE 1: Install Static Dependencies ---
# # Install any packages *not* defined in your project dependency files
# --mount=type=cache,target=/root/.cache/uv pip uv pip install \
#     aiohttp==3.12.15\
#     keyring \
#     keyrings.google-artifactregistry-auth

# --mount=type=cache,target=/root/.cache/uv pip uv pip install \
#     numba==0.61.2

# # VLLM_TARGET_DEVICE="tpu" uv pip install vllm
# # --- STAGE 2: Install Project Dependencies (The Main Cached Layer) ---

# # Copy *only* the dependency definition files.
# # This assumes vllm and tpu-inference are in the build context, copied from the parent directory.
# COPY vllm/requirements/tpu.txt /tmp/
# COPY vllm/requirements/build.txt /tmp/
# COPY vllm/requirements/common.txt /tmp/
# COPY tpu-inference/requirements.txt /tmp/

# # the full dependency installation.
# # This entire layer is cached and will *only* be rebuilt if
# # these .txt files change.
# --mount=type=cache,target=/root/.cache/uv pip bash -c ' \ 
#     # Set the target device so uv pip installs the right JAX/libtpu
#     # Install tpu-inference dependencies
#     export VLLM_TARGET_DEVICE="tpu" && \
#     uv pip install -r /tmp/tpu.txt -r /tmp/build.txt -r /tmp/common.txt -r /tmp/requirements.txt --no-cache-dir --pre \
#         --extra-index-url https://pypi.org/simple/ \
#         --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
#         --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
#         --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
#         --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
#         --find-links https://storage.googleapis.com/libtpu-releases/index.html \
#         --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
#         --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html'

#     # Install tpu-inference dependencies
#  --mount=type=cache,target=/root/.cache/pip bash -c ' \
#         pip install -r /tmp/requirements.txt --no-cache-dir --pre \
#         --extra-index-url https://pypi.org/simple/ \
#         --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
#         --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
#         --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
#         --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
#         --find-links https://storage.googleapis.com/libtpu-releases/index.html \
#         --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
#         --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html'

# # --- STAGE 3: Install Project Source Code ---

# # Now, copy the full source code. This invalidates cache frequently,
# # but the next step is fast.
# COPY vllm /vllm/
# COPY tpu-inference /tpu-inference/
# COPY tunix /tunix


# # Install in editable mode. This is lightning-fast because all
# # dependencies were installed and cached in STAGE 2.
# --mount=type=cache,target=/root/.cache/pip VLLM_TARGET_DEVICE="tpu" pip install -e /vllm/
# --mount=type=cache,target=/root/.cache/pip pip install -e /tpu-inference/

# --mount=type=cache,target=/root/.cache/pip pip install --no-deps /tunix/
# # --mount=type=cache,target=/root/.cache/pip VLLM_TARGET_DEVICE="tpu" pip install -e /tpu-inference/

if [ "$MODE" = "grpo-experimental" ]; then \
    echo "MODE=grpo-experimental: Re-installing JAX/libtpu"; \
    pip uninstall -y jax jaxlib libtpu && \
    pip install --pre -U jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ && \
    pip install -U --pre libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
    fi
