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

RUN pip install aiohttp==3.12.15

# Install Python packages that enable pip to authenticate with Google Artifact Registry automatically.
RUN pip install keyring keyrings.google-artifactregistry-auth

RUN pip install numba==0.61.2

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
    vllm==0.11.1rc1.dev292+g1b86bd8e1.tpu

# Install tpu-commons from the artifact registry
RUN pip install --no-cache-dir --pre \
    --index-url https://us-python.pkg.dev/cloud-tpu-images/maxtext-rl/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    tpu-commons==0.1.2

COPY tunix ./tunix/



RUN pip install --no-cache-dir --no-deps \
    ./tunix 
