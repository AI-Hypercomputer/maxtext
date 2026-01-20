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

"""Base interface for Parameter Mapping Strategies."""

import abc
from typing import Any, Dict, Callable


class ParamMapperStrategy(abc.ABC):
  """Abstract base class for model parameter mapping strategies."""

  @abc.abstractmethod
  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Returns the parameter mapping dictionary from MaxText to HuggingFace.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.

    Returns:
      A dictionary mapping MaxText parameter keys to HuggingFace keys.
    """

  @abc.abstractmethod
  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Returns the hook functions dictionary for parameter transformation.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.
      saving_to_hf: Boolean indicating direction (True for MT->HF, False for HF->MT).

    Returns:
      A dictionary mapping keys to hook functions.
    """

  def get_vllm_hooks(self) -> Dict[str, Callable]:
    """Returns hooks specific to VLLM integration if applicable."""
    return {}
