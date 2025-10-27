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

from typing import Any, Optional, Tuple

from flax import nnx
from jax import Array
from MaxText.layers.models import Transformer
from tunix.model.llama3 import BACKEND_MAPPINGS as llama3_backend_mappings
from tunix.model.qwen2 import BACKEND_MAPPINGS as qwen2_backend_mappings
from tunix.model.qwen3 import BACKEND_MAPPINGS as qwen3_backend_mappings


def _get_backend_mappings(model_name: str):
  """Returns the backend mappings for the given model name."""
  if model_name in qwen3_backend_mappings:
    return qwen3_backend_mappings["vllm_jax"]
  elif model_name in qwen2_backend_mappings:
    return qwen2_backend_mappings["vllm_jax"]
  elif model_name in llama3_backend_mappings:
    return llama3_backend_mappings["vllm_jax"]
  else:
    raise ValueError(f'Unsupported model name: {model_name}')


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

    return _get_backend_mappings(self.base.config.model_name).to_hf_mappings()

  def to_hf_transpose_keys(self):
    return _get_backend_mappings(
        self.base.config.model_name
    ).to_hf_transpose_keys()

  def to_hf_hook_fns(self):
    return _get_backend_mappings(self.base.config.model_name).to_hf_hook_fns()

  def lora_to_hf_mappings(self):
    return _get_backend_mappings(
        self.base.config.model_name
    ).lora_to_hf_mappings()
