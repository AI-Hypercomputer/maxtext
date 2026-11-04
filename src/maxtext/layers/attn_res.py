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
    num_blocks: Optional[int] = None,
) -> jnp.ndarray:
  """Softmax-pools `block_residual` and `prefix_sum` (port of HF `_apply_attn_res`).

  Callers must pass `epsilon=cfg.normalization_layer_epsilon` (the HF RMSNorm
  `variance_epsilon`, 1e-5 for Kimi-K3; `KimiLinearConfig` defaults to 1e-6).
  In each decoder layer the pre-attention pooling (`self_attention_res`) runs on
  the incoming blocks *before* `prefix_sum` is committed as a new block at
  `layer_idx % attn_res_block_size == 0`; the pre-MLP pooling then sees the
  updated block list.

  Args:
    prefix_sum: Running prefix sum of shape [..., D].
    block_residual: Committed block residuals of shape [..., K, D], or None.
      With no blocks (None or K == 0) the pooling is the identity and
      `prefix_sum` is returned unchanged.
    proj_weight: Projection kernel of shape [D, 1] (MaxText `DenseGeneral`
      layout, i.e. the transpose of the HF [1, D] `nn.Linear` weight).
    norm_weight: RMSNorm scale of shape [D].
    epsilon: RMSNorm variance epsilon.
    num_blocks: Must be None. Accepted only for call-site compatibility; the
      blocks are always taken from the (dynamically grown) `block_residual`.

  Returns:
    Pooled hidden states of shape [..., D] in the promoted dtype of
    `block_residual` and `prefix_sum` (HF casts to `v.dtype`).
  """
  if num_blocks is not None:
    raise NotImplementedError("Static-padded AttnRes (num_blocks != None) is not supported.")
  # No blocks: softmax over a single slot is 1, so the result is prefix_sum itself.
  if block_residual is None or block_residual.shape[-2] == 0:
    return prefix_sum

  d = prefix_sum.shape[-1]
  if block_residual.shape[-1] != d:
    raise ValueError(f"block_residual hidden size {block_residual.shape[-1]} != prefix_sum hidden size {d}")
  if tuple(norm_weight.shape) != (d,):
    raise ValueError(f"norm_weight must have shape ({d},), got {tuple(norm_weight.shape)}")
  if tuple(proj_weight.shape) != (d, 1):
    raise ValueError(f"proj_weight must have shape ({d}, 1), got {tuple(proj_weight.shape)}")

  v = jnp.concatenate([block_residual, jnp.expand_dims(prefix_sum, axis=-2)], axis=-2)  # [..., K+1, D]

  # fp32 materialization of V is deliberate, to match the HF reference numerics.
  v_float = v.astype(jnp.float32)
  k = v_float * lax.rsqrt(jnp.mean(jnp.square(v_float), axis=-1, keepdims=True) + epsilon)

  score_weight = norm_weight.astype(jnp.float32) * proj_weight.astype(jnp.float32)[:, 0]  # [D]
  scores = jnp.sum(k * score_weight, axis=-1)  # [..., K+1]
  probs = jax.nn.softmax(scores, axis=-1)
  # Mirrors HF `torch.matmul(probs, v_float)`; HIGHEST keeps the fp32 contraction exact on TPU.
  hidden_states = jnp.einsum("...k,...kd->...d", probs, v_float, precision=lax.Precision.HIGHEST)
  return hidden_states.astype(v.dtype)
