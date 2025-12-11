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
ARG MODE
ENV MODE=$MODE

RUN echo "Installing Post-Training dependencies (tunix, vLLM, tpu-inference) with MODE=${MODE}"
RUN pip uninstall -y jax jaxlib libtpu

RUN pip install aiohttp==3.12.15

# Install Python packages that enable pip to authenticate with Google Artifact Registry automatically.
RUN pip install keyring keyrings.google-artifactregistry-auth

RUN pip install numba==0.61.2

RUN pip install vllm==0.12.0

RUN pip uninstall -y tpu-inference &&  git clone https://github.com/abhinavclemson/tpu-inference.git && cd tpu-inference && pip install -e .

RUN pip uninstall -y tunix && git clone -b moe https://github.com/abhinavclemson/tunix.git && cd tunix && pip install -e . && cd ..

RUN pip install --no-deps qwix==0.1.4

RUN pip install google-metrax numpy==2.2

RUN if [ "$MODE" = "post-training-experimental" ]; then \
    echo "MODE=post-training-experimental: Re-installing JAX/libtpu"; \
    pip uninstall -y jax jaxlib libtpu && \
    pip install --pre jax==0.8.0.dev20251013 jaxlib==0.8.0.dev20251013 libtpu==0.0.25.dev20251012+nightly  -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
    fi
