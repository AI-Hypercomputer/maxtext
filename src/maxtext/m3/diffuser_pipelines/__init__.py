# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion pipelines (checkpoint loading, denoising loop, VAE decode), driven by `maxtext/inference/sample.py`.

Import the modules directly (`pipeline_utils`, `wan_pipeline`): this package imports nothing eagerly, so
`import maxtext.m3.diffuser_pipelines` works without the diffusion dependencies (diffusers, transformers).
"""
