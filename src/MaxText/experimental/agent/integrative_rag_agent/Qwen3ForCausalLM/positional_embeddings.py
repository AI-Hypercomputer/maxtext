
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions to compute RoPE parameters, for internal use.
"""

import math
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

from MaxText.common_types import Config


def _compute_longrope_parameters(config: Config, seq_len: Optional[int] = None) -> Tuple[Array, float]:
  """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length.
    Returns:
        Tuple of (Array, float), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
  # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
  base = config.rope_theta
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
  dim = int(head_dim * partial_rotary_factor)
  long_factor = config.rope_scaling["long_factor"]
  short_factor = config.rope_scaling["short_factor"]
  factor = config.rope_scaling.get("factor")
  attention_factor = config.rope_scaling.get("attention_factor")

  # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
  # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
  # values to compute the default attention scaling factor, instead of using `factor`.
  if hasattr(config, "original_max_position_embeddings"):
    original_max_position_embeddings = config.original_max_position_embeddings
    factor = config.max_position_embeddings / config.original_max_position_embeddings
  else:
    original_max_position_embeddings = config.max_position_embeddings

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if factor is not None and factor <= 1.0:
      attention_factor = 1.0
    else:
      attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

  # Compute the inverse frequencies -- scaled based on the target sequence length
  if seq_len and seq_len > original_max_position_embeddings:
    ext_factors = jnp.array(long_factor, dtype=jnp.float32)
  else:
    ext_factors = jnp.array(short_factor, dtype=jnp.float32)
  inv_freq_shape = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
  inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

  return inv_freq, attention_factor
