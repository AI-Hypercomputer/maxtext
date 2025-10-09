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
from MaxText.integration.tunix.weight_mapping.llama3 import LLAMA3_VLLM_MAPPING
from MaxText.integration.tunix.weight_mapping.qwen3 import QWEN3_VLLM_MAPPING


class VLLM_WEIGHT_MAPPING:
  """Mapping MaxText model weights to vLLM's model weights"""

  def __getattr__(self, name):
    if name.startswith("llama3.1"):
      return LLAMA3_VLLM_MAPPING
    elif name.startswith("qwen3"):
      return QWEN3_VLLM_MAPPING
    else:
      raise ValueError(f"{name} vLLM weight mapping not found.")

  def __getitem__(self, key):
    return getattr(self, key)

  @classmethod
  def __class_getitem__(cls, key):
    instance = cls()
    return instance[key]