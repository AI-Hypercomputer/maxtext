# Copyright 2026 Google LLC
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

"""Model-specific helpers shared by MaxText's vLLM adapter."""

import math
import re
from typing import Any

import jax.numpy as jnp


def map_layer_names_to_indices(
    layer_name_to_kvcache_index: Any,
) -> dict[int, int]:
  """Builds a mapping from layer index (int) to KV cache index (int).

  In hybrid models (e.g. Qwen3.5 GDN + Full Attention), vLLM groups KV cache
  tensors by layer type, so the physical cache list does not match 1:1 with
  decoder layer index (e.g. GDN layers 0..29 followed by Full Attention 30..39).
  `layer_name_to_kvcache_index` provides the mapping from layer name (e.g.
  'layer.3', 'model.layers.3.self_attn') to cache index.
  """
  if layer_name_to_kvcache_index is None:
    return {}
  try:
    mapping = dict(layer_name_to_kvcache_index)
  except (TypeError, ValueError):
    return {}
  lyr_to_cache_idx: dict[int, int] = {}
  for k, idx in mapping.items():
    parsed_lyr = None
    if isinstance(k, int):
      parsed_lyr = k
    elif isinstance(k, str):
      if k.isdigit():
        parsed_lyr = int(k)
      else:
        m = re.search(r"(?:layers?|layer)[._](\d+)", k)
        if m:
          parsed_lyr = int(m.group(1))
        else:
          m = re.search(r"\b(\d+)\b", k)
          if m:
            parsed_lyr = int(m.group(1))
    if parsed_lyr is not None:
      lyr_to_cache_idx[parsed_lyr] = int(idx)
  return lyr_to_cache_idx


def normalize_vllm_input_positions(input_positions: Any):
  """Converts vLLM's position layout to MaxText decode/prefill layout.

  vLLM supplies ordinary positions as ``(num_tokens,)``, decode MRoPE positions
  as ``(3, num_tokens)``, and batched MRoPE positions as ``(3, batch, seq)``.
  MaxText treats flattened decode tokens as a batch with a singleton sequence dimension,
  expecting ``(num_tokens, 1)`` or ``(num_tokens, 1, 3)``, and batched MRoPE as
  ``(batch, seq, 3)``.
  """
  if input_positions is None:
    return None
  if input_positions.ndim == 1:
    return input_positions.reshape((-1, 1))
  if input_positions.ndim == 2 and input_positions.shape[0] == 3:
    return input_positions.T[:, None, :]
  if input_positions.ndim == 3 and input_positions.shape[0] == 3:
    return jnp.transpose(input_positions, (1, 2, 0))
  raise ValueError(
      "vLLM input positions must have shape (num_tokens,), (3, num_tokens), or (3, batch, seq); "
      f"got {input_positions.shape}."
  )


def build_qwen_gdn_cache_layout(cfg: Any, torch_module: Any):
  """Returns the shapes, dtypes, and unpadded bytes for a Qwen GDN cache."""
  key_dim = cfg.gdn_key_head_dim * cfg.gdn_num_key_heads
  value_dim = cfg.gdn_value_head_dim * cfg.gdn_num_value_heads
  conv_dim = key_dim * 2 + value_dim

  shapes = (
      (cfg.gdn_conv_kernel_dim - 1, conv_dim),
      (cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim),
  )
  # This is the TPU Inference / upstream vLLM contract regardless of model
  # weight or attention-KV dtype.
  dtypes = (torch_module.bfloat16, torch_module.float32)
  page_size_bytes = sum(
      math.prod(shape) * torch_module.empty((), dtype=dtype).element_size()
      for shape, dtype in zip(shapes, dtypes, strict=True)
  )
  return shapes, dtypes, page_size_bytes
