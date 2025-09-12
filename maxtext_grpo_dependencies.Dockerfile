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

ARG BASEIMAGE
FROM ${BASEIMAGE}

# Uninstall existing jax to avoid conflicts
RUN pip uninstall -y jax jaxlib libtpu

# Install GRPO dependencies that are less likely to change
RUN pip install aiohttp==3.12.15 keyring keyrings.google-artifactregistry-auth

# Install vLLM for Jax and TPUs from the artifact registry
RUN VLLM_TARGET_DEVICE="tpu" pip install --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html \
    --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html \
    vllm==0.10.2rc2.dev59+gdcb28a332.tpu

# Install specific JAX version for GRPO
RUN pip install jax==0.7.1.dev20250813 jaxlib==0.7.1.dev20250813 libtpu==0.0.21.dev20250813+nightly --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    --find-links https://storage.googleapis.com/libtpu-wheels/index.html \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html \
    --find-links https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    --find-links https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Install tpu-commons from local source
COPY tpu_commons /tpu_commons
RUN pip install -e /tpu_commons --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html

RUN pip install numba==0.61.2