# Copyright 2026 Google LLC
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

"""Attention Residuals (AttnRes) softmax pooling for Kimi-K3.

Kimi-K3 replaces the plain residual stream with a learned softmax pooling over
a list of block residuals plus the running prefix sum. This module provides only
the pooling primitive; the decoder layer owns the per-site weights (RMSNorm scale
and a [D, 1] projection) and decides when a new block residual is committed.

  V = [block_residual_0, ..., block_residual_{K-1}, prefix_sum]
  k = V * rsqrt(mean(V**2, -1) + eps)          (RMSNorm without scale, fp32)
  scores = sum(k * (norm_weight * proj_weight), -1)
  out = sum(softmax(scores) * V, over K+1)
"""

from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp


def _apply_attn_res(
    prefix_sum: jnp.ndarray,
    block_residual: Optional[jnp.ndarray],
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    epsilon: float = 1e-5,
    num_blocks: None = None,
) -> jnp.ndarray:
  """Softmax-pools `block_residual` and `prefix_sum` (port of HF `_apply_attn_res`).

  Args:
    prefix_sum: Running prefix sum of shape [..., D].
    block_residual: Committed block residuals of shape [..., K, D], or None.
      With no blocks (None or K == 0) the pooling is the identity and
      `prefix_sum` is returned unchanged.
    proj_weight: Projection kernel of shape [D, 1], [1, D] or [D].
    norm_weight: RMSNorm scale of shape [D].
    epsilon: RMSNorm variance epsilon.
    num_blocks: Must be None. Accepted only for call-site compatibility; the
      blocks are always taken from the (dynamically grown) `block_residual`.

  Returns:
    Pooled hidden states of shape [..., D] in `prefix_sum.dtype`.
  """
  if num_blocks is not None:
    raise NotImplementedError("Static-padded AttnRes (num_blocks != None) is not supported.")
  # No blocks: softmax over a single slot is 1, so the result is prefix_sum itself.
  if block_residual is None or block_residual.shape[-2] == 0:
    return prefix_sum

  orig_dtype = prefix_sum.dtype
  v = jnp.concatenate([block_residual, jnp.expand_dims(prefix_sum, axis=-2)], axis=-2)  # [..., K+1, D]

  # fp32 materialization of V is deliberate, to match the HF reference numerics.
  v_float = v.astype(jnp.float32)
  k = v_float * lax.rsqrt(jnp.mean(jnp.square(v_float), axis=-1, keepdims=True) + epsilon)

  score_weight = norm_weight.astype(jnp.float32).reshape(-1) * proj_weight.astype(jnp.float32).reshape(-1)  # [D]
  scores = jnp.sum(k * score_weight, axis=-1)  # [..., K+1]
  probs = jax.nn.softmax(scores, axis=-1)
  hidden_states = jnp.sum(jnp.expand_dims(probs, axis=-1) * v_float, axis=-2)
  return hidden_states.astype(orig_dtype)
