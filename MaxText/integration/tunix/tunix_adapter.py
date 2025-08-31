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

from __future__ import annotations

from typing import Optional, Tuple, Any

from jax import Array
from flax import nnx
from MaxText.layers.models import Transformer
from MaxText.integration.tunix.weight_mapping import VLLM_WEIGHT_MAPPING

class TunixMaxTextAdapter(nnx.Module):
  """Adapter exposing Tunix Trainer call signature over a Transformer model."""

  def __init__(
      self,
      base_model: Transformer,
  ):
    super().__init__()
    self.base = base_model

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
    return VLLM_WEIGHT_MAPPING[self.base.config.model_name].to_hf_mapping()

  def to_hf_transpose_keys(self):
    return VLLM_WEIGHT_MAPPING[self.base.config.model_name].to_hf_transpose_keys()

  def to_hf_hook_fns(self):
    return VLLM_WEIGHT_MAPPING[self.base.config.model_name].to_hf_hook_fns()

  def lora_to_hf_mappings(self):
    return VLLM_WEIGHT_MAPPING[self.base.config.model_name].lora_to_hf_mappings()
