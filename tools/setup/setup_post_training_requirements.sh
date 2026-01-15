# !/bin/bash

# Copyright 2023–2026 Google LLC
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

# This script installs the dependencies for running post-training with MaxText+Tunix+vLLM on TPUs

set -e
set -x

uv pip uninstall jax jaxlib libtpu

uv pip install aiohttp==3.12.15

# Install Tunix
uv pip install google-tunix==0.1.5

# Install vLLM for Jax and TPUs
uv pip install vllm-tpu

uv pip install numba==0.61.2

uv pip install --no-deps qwix==0.1.4
