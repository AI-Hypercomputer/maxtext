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

"""Adapter for integrating MaxText Transformer models with Tunix.

This module provides the `TunixMaxTextAdapter` class, which wraps a MaxText
Transformer model to expose a call signature compatible with Tunix Trainers.
It also handles weight mapping for compatibility with Hugging Face models.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

from jax import Array
from flax import nnx
from MaxText.layers.models import Transformer
from maxtext.src.maxtext.integration.tunix.utils import VllmWeightMapping


class TunixMaxTextAdapter(nnx.Module):
  """Adapter exposing Tunix Trainer call signature over a Transformer model."""

  def __init__(
      self,
      base_model: Transformer,
      hf_config=None,
  ):
    super().__init__()
    self.base = base_model
    self.hf_config = hf_config
    self._vllm_weight_mapping = VllmWeightMapping(
        self.base.config.model_name, self.hf_config
    )

  # ------------------------------------------------------------------ #
  # Tunix call signature
  # ------------------------------------------------------------------ #
  def __call__(
      self,
      input_tokens: Array,  # [B, L]
      positions: Array,  # [B, L]
      cache: Optional[Any],  # Tunix currently passes None from Trainers
      attention_mask: Optional[Array],  # [B, L, L] or None
      output_hidden_states: bool = False,  # ignored
  ) -> Tuple[Array, None]:
    """Forward compatible with Tunix Trainers default loss.
    Returns logits, None.
    """
    logits = self.base(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        # TODO: @mazumdera - add support for packing
        decoder_segment_ids=None,
    )
    return logits, None

  def to_hf_mappings(self):
    return self._vllm_weight_mapping(
        self.base.config.model_name, self.hf_config
    ).to_hf_mapping()

  def to_hf_transpose_keys(self):
    return self._vllm_weight_mapping.to_hf_transpose_keys()

  def to_hf_hook_fns(self):
    return self._vllm_weight_mapping.to_hf_hook_fns()

  def lora_to_hf_mappings(self):
    return self._vllm_weight_mapping.lora_to_hf_mappings()


