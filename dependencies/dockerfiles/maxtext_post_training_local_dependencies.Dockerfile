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

ARG BASEIMAGE=maxtext_base_image
FROM ${BASEIMAGE}

ARG MODE
ENV MODE=$MODE

RUN echo "Installing Post-Training dependencies (tunix, vLLM, tpu-inference) with MODE=${MODE}"
RUN pip uninstall -y jax jaxlib libtpu

RUN pip install aiohttp==3.12.15

# Install Python packages that enable pip to authenticate with Google Artifact Registry automatically.
RUN pip install keyring keyrings.google-artifactregistry-auth

RUN pip install numba==0.61.2

COPY tunix /tunix
RUN pip uninstall -y google-tunix
RUN pip install -e /tunix --no-cache-dir

COPY vllm /vllm
RUN VLLM_TARGET_DEVICE="tpu" pip install -e /vllm --no-cache-dir

COPY tpu-inference /tpu-inference
RUN pip install -e /tpu-inference --no-cache-dir

RUN pip install --no-deps qwix==0.1.4

RUN pip install math-verify==0.9.0

RUN if [ "$MODE" = "post-training-experimental" ]; then \
    echo "MODE=post-training-experimental: Re-installing JAX/libtpu"; \
    pip uninstall -y jax jaxlib libtpu && \
    pip install --pre -U jax jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ && \
    pip install -U --pre libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
    fi
