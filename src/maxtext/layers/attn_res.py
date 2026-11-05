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

For lax.scan over layers the growing block list is replaced by a fixed-size
`AttnResBuffer` (padded slots + committed count) and a masked softmax.
"""

import math
from typing import NamedTuple, Optional

import jax
from jax import lax
import jax.numpy as jnp


class AttnResBuffer(NamedTuple):
  """Fixed-shape AttnRes highway state, for lax.scan over decoder layers.

  Attributes:
    blocks: Committed block residuals padded to a static slot count, [..., num_slots, D].
      Slots at index >= `count` hold don't-care (finite) values that the masked
      pooling ignores.
    count: int32 scalar, the number of committed blocks (valid slots).
    pending: Optional [..., D] block committed since `blocks` was last written, logically
      slot `count` (used inside a scanned cycle so the big `blocks` array stays the same
      value for every sub-layer and is saved for backward once per cycle). None outside.
    has_pending: bool scalar, whether `pending` is a committed block. None iff `pending` is.
  """

  blocks: jnp.ndarray
  count: jnp.ndarray
  pending: Optional[jnp.ndarray] = None
  has_pending: Optional[jnp.ndarray] = None


def attn_res_num_commits(num_decoder_layers: int, attn_res_block_size: int) -> int:
  """Number of block commits over the whole stack (one at every layer_idx % block_size == 0)."""
  return math.ceil(num_decoder_layers / attn_res_block_size)


def attn_res_num_buffer_slots(num_decoder_layers: int, attn_res_block_size: int) -> int:
  """Static slot count of an `AttnResBuffer`.

  One more than the number of commits: `commit_attn_res_block` writes the current
  prefix sum into slot `count` on *every* layer (so a traced layer index needs no
  conditional write) and only advances `count` on commit layers. The spare slot keeps
  those speculative writes after the last commit from clobbering a committed block.
  """
  return attn_res_num_commits(num_decoder_layers, attn_res_block_size) + 1


def init_attn_res_buffer(prefix_sum: jnp.ndarray, num_slots: int) -> AttnResBuffer:
  """Returns an empty buffer ([..., num_slots, D] zeros, count 0) shaped like `prefix_sum`."""
  shape = prefix_sum.shape[:-1] + (num_slots, prefix_sum.shape[-1])
  return AttnResBuffer(blocks=jnp.zeros(shape, prefix_sum.dtype), count=jnp.zeros((), jnp.int32))


def commit_attn_res_block(buffer: AttnResBuffer, prefix_sum: jnp.ndarray, is_commit) -> AttnResBuffer:
  """Appends `prefix_sum` as a new block when `is_commit` (a bool or traced bool scalar).

  The write into slot `count` is unconditional; only `count` depends on `is_commit`.
  Uncommitted writes land in a masked slot and are overwritten later.

  With a `pending` slot, the commit goes there instead (at most one commit per pending
  window; see `flush_attn_res_pending`) and `blocks` is left untouched.
  """
  if buffer.pending is not None:
    is_commit = jnp.asarray(is_commit)
    return buffer._replace(
        pending=jnp.where(is_commit, prefix_sum.astype(buffer.pending.dtype), buffer.pending),
        has_pending=jnp.logical_or(buffer.has_pending, is_commit),
    )
  value = jnp.expand_dims(prefix_sum.astype(buffer.blocks.dtype), axis=-2)
  blocks = lax.dynamic_update_slice_in_dim(buffer.blocks, value, buffer.count, axis=-2)
  count = buffer.count + jnp.asarray(is_commit, jnp.int32)
  return AttnResBuffer(blocks=blocks, count=count)


def flush_attn_res_pending(buffer: AttnResBuffer) -> AttnResBuffer:
  """Writes the pending block (if any) into `blocks` and drops the pending slot."""
  if buffer.pending is None:
    return buffer
  return commit_attn_res_block(AttnResBuffer(buffer.blocks, buffer.count), buffer.pending, buffer.has_pending)


def _check_attn_res_shapes(prefix_sum, block_residual, proj_weight, norm_weight) -> None:
  d = prefix_sum.shape[-1]
  if block_residual.shape[-1] != d:
    raise ValueError(f"block_residual hidden size {block_residual.shape[-1]} != prefix_sum hidden size {d}")
  if tuple(norm_weight.shape) != (d,):
    raise ValueError(f"norm_weight must have shape ({d},), got {tuple(norm_weight.shape)}")
  if tuple(proj_weight.shape) != (d, 1):
    raise ValueError(f"proj_weight must have shape ({d}, 1), got {tuple(proj_weight.shape)}")


def _pool(v: jnp.ndarray, proj_weight, norm_weight, epsilon, valid=None) -> jnp.ndarray:
  """Softmax pooling over axis -2 of `v` [..., K, D] in fp32; `valid` [K] masks out slots."""
  # fp32 materialization of V is deliberate, to match the HF reference numerics.
  v_float = v.astype(jnp.float32)
  k = v_float * lax.rsqrt(jnp.mean(jnp.square(v_float), axis=-1, keepdims=True) + epsilon)

  score_weight = norm_weight.astype(jnp.float32) * proj_weight.astype(jnp.float32)[:, 0]  # [D]
  scores = jnp.sum(k * score_weight, axis=-1)  # [..., K]
  if valid is not None:
    scores = jnp.where(valid, scores, -jnp.inf)
  probs = jax.nn.softmax(scores, axis=-1)
  # Mirrors HF `torch.matmul(probs, v_float)`; HIGHEST keeps the fp32 contraction exact on TPU.
  return jnp.einsum("...k,...kd->...d", probs, v_float, precision=lax.Precision.HIGHEST)


def _apply_attn_res_masked(
    prefix_sum: jnp.ndarray,
    block_buffer: jnp.ndarray,
    num_committed: jnp.ndarray,
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    *,
    epsilon: float = 1e-5,
    pending: Optional[jnp.ndarray] = None,
    has_pending: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Static-shape AttnRes: pools the first `num_committed` slots of `block_buffer` and `prefix_sum`.

  Equal (up to fp summation order) to `_apply_attn_res(prefix_sum, block_buffer[..., :n, :])`
  with n = num_committed, including n == 0 (identity, up to the promoted output dtype).
  Empty slots get a -inf score, i.e. an exactly-zero softmax weight. With `pending`
  ([..., D]) and `has_pending`, `pending` is pooled as block n when `has_pending` is true.

  Args:
    prefix_sum: [..., D].
    block_buffer: [..., M, D] fixed-size block buffer.
    num_committed: int scalar (may be traced), 0 <= num_committed <= M.
    proj_weight: [D, 1].
    norm_weight: [D].
    epsilon: RMSNorm variance epsilon.
    pending: Optional extra block [..., D] (see `AttnResBuffer.pending`).
    has_pending: bool scalar; required with `pending`.

  Returns:
    Pooled hidden states of shape [..., D] in the promoted dtype of `block_buffer` and `prefix_sum`.
  """
  _check_attn_res_shapes(prefix_sum, block_buffer, proj_weight, norm_weight)
  if (pending is None) != (has_pending is None):
    raise ValueError("pending and has_pending must be given together.")
  num_slots = block_buffer.shape[-2]
  parts = [block_buffer]
  valid = [jnp.arange(num_slots) < num_committed]
  if pending is not None:
    parts.append(jnp.expand_dims(pending.astype(block_buffer.dtype), axis=-2))
    valid.append(jnp.reshape(jnp.asarray(has_pending, bool), (1,)))
  # The prefix-sum slot (last) is always valid, so the softmax max is finite.
  parts.append(jnp.expand_dims(prefix_sum, axis=-2))
  valid.append(jnp.ones((1,), dtype=bool))
  v = jnp.concatenate(parts, axis=-2)  # [..., M(+1)+1, D]
  valid = jnp.concatenate(valid)
  return _pool(v, proj_weight, norm_weight, epsilon, valid).astype(v.dtype)


def _apply_attn_res(
    prefix_sum: jnp.ndarray,
    block_residual: Optional[jnp.ndarray],
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    epsilon: float = 1e-5,
    num_blocks: Optional[int] = None,
    *,
    num_committed: Optional[jnp.ndarray] = None,
    pending: Optional[jnp.ndarray] = None,
    has_pending: Optional[jnp.ndarray] = None,
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
    num_blocks: None for the dynamic-concat path (unscanned). An int selects the
      static-padded path (`_apply_attn_res_masked`, used under lax.scan); it must
      equal `block_residual.shape[-2]`.
    num_committed: Number of valid leading slots of the static buffer (may be
      traced); required iff `num_blocks` is set.
    pending: Optional pending block of the static path (see `AttnResBuffer.pending`).
    has_pending: Whether `pending` is valid; required with `pending`.

  Returns:
    Pooled hidden states of shape [..., D] in the promoted dtype of
    `block_residual` and `prefix_sum` (HF casts to `v.dtype`).
  """
  if num_blocks is not None:
    if block_residual is None or block_residual.shape[-2] != num_blocks:
      raise ValueError(
          f"Static-padded AttnRes expects a [..., {num_blocks}, D] block buffer, got "
          f"{None if block_residual is None else tuple(block_residual.shape)}."
      )
    if num_committed is None:
      raise ValueError("Static-padded AttnRes (num_blocks != None) requires num_committed.")
    return _apply_attn_res_masked(
        prefix_sum,
        block_residual,
        num_committed,
        proj_weight,
        norm_weight,
        epsilon=epsilon,
        pending=pending,
        has_pending=has_pending,
    )
  if num_committed is not None or pending is not None or has_pending is not None:
    raise ValueError("num_committed / pending / has_pending are only valid with num_blocks (static-padded AttnRes).")
  # No blocks: softmax over a single slot is 1, so the result is prefix_sum itself.
  if block_residual is None or block_residual.shape[-2] == 0:
    return prefix_sum

  _check_attn_res_shapes(prefix_sum, block_residual, proj_weight, norm_weight)
  v = jnp.concatenate([block_residual, jnp.expand_dims(prefix_sum, axis=-2)], axis=-2)  # [..., K+1, D]
  return _pool(v, proj_weight, norm_weight, epsilon).astype(v.dtype)
