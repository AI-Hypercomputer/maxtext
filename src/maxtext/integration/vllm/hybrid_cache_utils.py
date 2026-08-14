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
from typing import Any


def normalize_vllm_input_positions(input_positions: Any):
  """Converts vLLM's flattened position layout to MaxText decode layout.

  vLLM supplies ordinary positions as ``(num_tokens,)`` and MRoPE positions
  as ``(3, num_tokens)``. MaxText treats flattened decode tokens as a batch
  with a singleton sequence dimension, so it expects ``(num_tokens, 1)`` or
  ``(num_tokens, 1, 3)``, respectively.
  """
  if input_positions.ndim == 1:
    return input_positions.reshape((-1, 1))
  if input_positions.ndim == 2 and input_positions.shape[0] == 3:
    return input_positions.T[:, None, :]
  raise ValueError(
      "vLLM input positions must have shape (num_tokens,) or (3, num_tokens); " f"got {input_positions.shape}."
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
