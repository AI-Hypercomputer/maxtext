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

"""Provides a centralized access point for vLLM weight mappings.

This module defines the `StandaloneVllmWeightMapping` class, which acts as a
dispatcher to retrieve the correct weight mapping configuration for a given
model name. This allows for easy extension to support new models.
"""

from maxtext.src.maxtext.integration.tunix.weight_mapping.deepseek3 import DEEPSEEK_VLLM_MAPPING
from maxtext.src.maxtext.integration.tunix.weight_mapping.llama3 import LLAMA3_VLLM_MAPPING
from maxtext.src.maxtext.integration.tunix.weight_mapping.qwen3 import QWEN3_VLLM_MAPPING


class StandaloneVllmWeightMapping:
  """Mapping MaxText model weights to vLLM's model weights."""

  def __getattr__(self, name):
    if name.startswith("llama3.1"):
      return LLAMA3_VLLM_MAPPING
    elif name.startswith("qwen3"):
      return QWEN3_VLLM_MAPPING
    elif name.startswith("deepseek3"):
      return DEEPSEEK_VLLM_MAPPING
    else:
      raise ValueError(f"{name} vLLM weight mapping not found.")

  def __getitem__(self, key):
    return getattr(self, key)

  @classmethod
  def __class_getitem__(cls, key):
    instance = cls()
    return instance[key]
