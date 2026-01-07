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

"""Setup for MaxText vLLM adapter package."""

from setuptools import setup

setup(
    name="maxtext_vllm_adapter",
    version="0.1.0",
    packages=["maxtext_vllm_adapter"],
    entry_points={"vllm.general_plugins": ["register_maxtext_vllm_adapter = maxtext_vllm_adapter:register"]},
)
