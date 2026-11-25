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
a list of block residuals plus the running prefix sum. This module provides the
pooling primitive and fused `AttnRes` + sublayer `RMSNorm` entry point, backed
by the Pallas TPU kernel in `maxtext.kernels.attn_res` when enabled on TPU.

  V = [block_residual_0, ..., block_residual_{K-1}, prefix_sum]
  k = V * rsqrt(mean(V**2, -1) + eps)          (RMSNorm without scale, fp32)
  scores = sum(k * (norm_weight * proj_weight), -1)
  out = sum(softmax(scores) * V, over K+1)

For lax.scan over layers the growing block list is replaced by a fixed-size
`AttnResBuffer` (padded slots + committed count) and a masked softmax.
"""

import math
from typing import Any, NamedTuple, Optional, Sequence

import jax
from jax import lax
import jax.numpy as jnp

from maxtext.common.common_types import ShardMode
from maxtext.kernels.attn_res import pallas_kernel
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import sharding as maxtext_sharding

_DEFAULT_PALLAS_CONFIG = pallas_kernel.AttnResKernelConfig()


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


def _resolve_activation_pspec(
    prefix_sum: jnp.ndarray,
    mesh: Any,
    shard_mode: Any,
    logical_axis_names: Optional[Sequence[str]],
    logical_axis_rules: Optional[Any] = None,
) -> Optional[jax.sharding.PartitionSpec]:
  """Resolves the 3D activation `PartitionSpec` for `ShardMode.AUTO`."""
  if shard_mode != ShardMode.AUTO or mesh is None or getattr(mesh, "empty", True):
    return None
  if logical_axis_names is None:
    return None
  rules = maxtext_sharding.get_logical_axis_rules() or logical_axis_rules
  if not rules:
    return None
  act_axes = maxtext_sharding.logical_to_mesh_axes(tuple(logical_axis_names), mesh, rules=rules)
  parts = tuple(act_axes) + (None,) * max(0, 3 - len(act_axes))
  pspec = jax.sharding.PartitionSpec(*parts[:3])
  return maxtext_sharding.adjust_pspec_for_indivisible_shapes(pspec, prefix_sum.shape, mesh)


def _try_pallas_attn_res(
    prefix_sum: jnp.ndarray,
    block_buffer: jnp.ndarray,
    num_committed: int | jnp.ndarray,
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    *,
    epsilon: float,
    rms_norm: Optional[RMSNorm] = None,
    pending: Optional[jnp.ndarray] = None,
    has_pending: Optional[jnp.ndarray] = None,
    mesh: Any = None,
    shard_mode: Any = ShardMode.AUTO,
    logical_axis_names: Optional[Sequence[str]] = None,
    logical_axis_rules: Optional[Any] = None,
    kernel_config: Optional[pallas_kernel.AttnResKernelConfig] = None,
) -> Optional[jnp.ndarray]:
  """Dispatches the Pallas kernel when inputs satisfy its contract, else returns None."""
  if block_buffer.shape[-2] <= 0:
    return None
  k_cfg = _DEFAULT_PALLAS_CONFIG if kernel_config is None else kernel_config
  act_pspec = _resolve_activation_pspec(prefix_sum, mesh, shard_mode, logical_axis_names, logical_axis_rules)
  sub_scale = rms_norm.scale.get_value() if rms_norm is not None else None
  norm_dtype = (jnp.float32 if rms_norm.dtype is None else rms_norm.dtype) if rms_norm is not None else None
  if not pallas_kernel.can_use_attn_res_pallas(
      prefix_sum,
      block_buffer,
      pending=pending,
      sublayer_norm_scale=sub_scale,
      norm_dtype=norm_dtype,
      mesh=mesh,
      activation_pspec=act_pspec,
      config=k_cfg,
  ):
    return None
  if rms_norm is not None:
    assert sub_scale is not None
    return pallas_kernel.attn_res_rms_norm_pallas(
        prefix_sum,
        block_buffer,
        num_committed,
        norm_weight,
        proj_weight,
        sub_scale,
        pending=pending,
        has_pending=has_pending,
        res_epsilon=epsilon,
        norm_epsilon=rms_norm.epsilon,
        norm_dtype=norm_dtype,
        mesh=mesh,
        activation_pspec=act_pspec,
        config=k_cfg,
    )
  return pallas_kernel.attn_res_pallas(
      prefix_sum,
      block_buffer,
      num_committed,
      norm_weight,
      proj_weight,
      pending=pending,
      has_pending=has_pending,
      epsilon=epsilon,
      mesh=mesh,
      activation_pspec=act_pspec,
      config=k_cfg,
  )


def _validate_attn_res_args(
    prefix_sum: jnp.ndarray,
    block_residual: Optional[jnp.ndarray],
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    *,
    num_blocks: Optional[int],
    num_committed: Optional[jnp.ndarray],
    pending: Optional[jnp.ndarray],
    has_pending: Optional[jnp.ndarray],
) -> tuple[bool, int | jnp.ndarray]:
  """Validates AttnRes args and returns `(has_blocks, resolved_num_committed)`."""
  if num_blocks is not None:
    if block_residual is None or block_residual.shape[-2] != num_blocks:
      raise ValueError(
          f"Static-padded AttnRes expects a [..., {num_blocks}, D] block buffer, got "
          f"{None if block_residual is None else tuple(block_residual.shape)}."
      )
    if num_committed is None:
      raise ValueError("Static-padded AttnRes (num_blocks != None) requires num_committed.")
    _check_attn_res_shapes(prefix_sum, block_residual, proj_weight, norm_weight)
    if (pending is None) != (has_pending is None):
      raise ValueError("pending and has_pending must be given together.")
    return True, num_committed

  if num_committed is not None or pending is not None or has_pending is not None:
    raise ValueError("num_committed / pending / has_pending are only valid with num_blocks (static-padded AttnRes).")
  if block_residual is None or block_residual.shape[-2] == 0:
    return False, 0
  _check_attn_res_shapes(prefix_sum, block_residual, proj_weight, norm_weight)
  return True, block_residual.shape[-2]


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
    use_pallas_kernel: bool = False,
    mesh: Any = None,
    shard_mode: Any = ShardMode.AUTO,
    logical_axis_names: Optional[Sequence[str]] = None,
    logical_axis_rules: Optional[Any] = None,
    kernel_config: Optional[pallas_kernel.AttnResKernelConfig] = None,
) -> jnp.ndarray:
  """Static-shape AttnRes: pools the first `num_committed` slots of `block_buffer` and `prefix_sum`."""
  _check_attn_res_shapes(prefix_sum, block_buffer, proj_weight, norm_weight)
  if (pending is None) != (has_pending is None):
    raise ValueError("pending and has_pending must be given together.")

  if use_pallas_kernel:
    out = _try_pallas_attn_res(
        prefix_sum,
        block_buffer,
        num_committed,
        proj_weight,
        norm_weight,
        epsilon=epsilon,
        pending=pending,
        has_pending=has_pending,
        mesh=mesh,
        shard_mode=shard_mode,
        logical_axis_names=logical_axis_names,
        logical_axis_rules=logical_axis_rules,
        kernel_config=kernel_config,
    )
    if out is not None:
      return out

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
    use_pallas_kernel: bool = False,
    mesh: Any = None,
    shard_mode: Any = ShardMode.AUTO,
    logical_axis_names: Optional[Sequence[str]] = None,
    logical_axis_rules: Optional[Any] = None,
    kernel_config: Optional[pallas_kernel.AttnResKernelConfig] = None,
) -> jnp.ndarray:
  """Softmax-pools `block_residual` and `prefix_sum` (port of HF `_apply_attn_res`)."""
  has_blocks, resolved_committed = _validate_attn_res_args(
      prefix_sum,
      block_residual,
      proj_weight,
      norm_weight,
      num_blocks=num_blocks,
      num_committed=num_committed,
      pending=pending,
      has_pending=has_pending,
  )
  if not has_blocks:
    return prefix_sum
  assert block_residual is not None
  if num_blocks is not None:
    assert num_committed is not None
    return _apply_attn_res_masked(
        prefix_sum,
        block_residual,
        num_committed,
        proj_weight,
        norm_weight,
        epsilon=epsilon,
        pending=pending,
        has_pending=has_pending,
        use_pallas_kernel=use_pallas_kernel,
        mesh=mesh,
        shard_mode=shard_mode,
        logical_axis_names=logical_axis_names,
        logical_axis_rules=logical_axis_rules,
        kernel_config=kernel_config,
    )
  if use_pallas_kernel:
    out = _try_pallas_attn_res(
        prefix_sum,
        block_residual,
        resolved_committed,
        proj_weight,
        norm_weight,
        epsilon=epsilon,
        mesh=mesh,
        shard_mode=shard_mode,
        logical_axis_names=logical_axis_names,
        logical_axis_rules=logical_axis_rules,
        kernel_config=kernel_config,
    )
    if out is not None:
      return out
  v = jnp.concatenate([block_residual, jnp.expand_dims(prefix_sum, axis=-2)], axis=-2)  # [..., K+1, D]
  return _pool(v, proj_weight, norm_weight, epsilon).astype(v.dtype)


def _can_fuse_rms_norm(rms_norm: RMSNorm) -> bool:
  return (
      isinstance(rms_norm, RMSNorm)
      and rms_norm.with_scale
      and rms_norm.scale is not None
      and rms_norm.scale_offset == 0.0
      and not rms_norm.defer_scale_grad_allreduce
      and not rms_norm.parameter_memory_host_offload
  )


def apply_attn_res_rms_norm(
    prefix_sum: jnp.ndarray,
    block_residual: Optional[jnp.ndarray],
    proj_weight: jnp.ndarray,
    norm_weight: jnp.ndarray,
    rms_norm: RMSNorm,
    *,
    epsilon: float = 1e-5,
    num_blocks: Optional[int] = None,
    num_committed: Optional[jnp.ndarray] = None,
    pending: Optional[jnp.ndarray] = None,
    has_pending: Optional[jnp.ndarray] = None,
    use_pallas_kernel: bool = False,
    mesh: Any = None,
    shard_mode: Any = ShardMode.AUTO,
    logical_axis_names: Optional[Sequence[str]] = None,
    logical_axis_rules: Optional[Any] = None,
    kernel_config: Optional[pallas_kernel.AttnResKernelConfig] = None,
) -> jnp.ndarray:
  """Fuses `_apply_attn_res` and sublayer `RMSNorm` on TPU when `use_pallas_kernel=True`."""
  if use_pallas_kernel and _can_fuse_rms_norm(rms_norm):
    has_blocks, resolved_committed = _validate_attn_res_args(
        prefix_sum,
        block_residual,
        proj_weight,
        norm_weight,
        num_blocks=num_blocks,
        num_committed=num_committed,
        pending=pending,
        has_pending=has_pending,
    )
    if has_blocks:
      assert block_residual is not None
      out = _try_pallas_attn_res(
          prefix_sum,
          block_residual,
          resolved_committed,
          proj_weight,
          norm_weight,
          epsilon=epsilon,
          rms_norm=rms_norm,
          pending=pending,
          has_pending=has_pending,
          mesh=mesh,
          shard_mode=shard_mode,
          logical_axis_names=logical_axis_names,
          logical_axis_rules=logical_axis_rules,
          kernel_config=kernel_config,
      )
      if out is not None:
        return out

  pooled = _apply_attn_res(
      prefix_sum=prefix_sum,
      block_residual=block_residual,
      proj_weight=proj_weight,
      norm_weight=norm_weight,
      epsilon=epsilon,
      num_blocks=num_blocks,
      num_committed=num_committed,
      pending=pending,
      has_pending=has_pending,
      use_pallas_kernel=use_pallas_kernel,
      mesh=mesh,
      shard_mode=shard_mode,
      logical_axis_names=logical_axis_names,
      logical_axis_rules=logical_axis_rules,
      kernel_config=kernel_config,
  )
  return rms_norm(pooled)
