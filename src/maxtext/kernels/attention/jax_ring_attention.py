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
"""JAX implementation without using Pallas for Ring Attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
from typing import Any, TypeAlias

import jax
from jax import lax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
from maxtext.kernels.attention import jax_flash_attention
from maxtext.kernels.attention import splash_attention_kernel
from maxtext.kernels.tokamax_splash_attention import base as tokamax_base
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as tokamax_mask_lib
import numpy as np

DEFAULT_MASK_VALUE = splash_attention_kernel.DEFAULT_MASK_VALUE
SegmentIds = tokamax_base.SegmentIds
SegmentIdsType: TypeAlias = splash_attention_kernel.SegmentIds | tokamax_base.SegmentIds
MaskType: TypeAlias = jax.Array | np.ndarray | mask_lib.Mask | tokamax_mask_lib.Mask


def _materialize_mask(mask: MaskType) -> jax.Array | np.ndarray:
  """Materializes a lazy Splash mask into a dense array if needed."""
  if isinstance(mask, (mask_lib.Mask, tokamax_mask_lib.Mask)):
    return mask[:, :]
  return mask


def ring_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: MaskType,
    segment_ids: SegmentIdsType | None = None,
    *,
    mask_value: float = -1e30,
    block_q: int = 1024,
    block_kv: int = 1024,
    axis_name: str = "ring",
    cap: float | None = None,
    save_residuals: bool = False,
    unroll: bool = False,
    fusion_specs: Mapping[str, Any] | None = None,
) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
  """Computes masked ring attention forward pass using FlashAttention blocks."""
  del fusion_specs
  q_is_3d = q.ndim == 3
  if q_is_3d:
    q = jnp.expand_dims(q, axis=0)
    if (
        segment_ids is not None
        and isinstance(segment_ids.q, (jax.Array, np.ndarray))
        and isinstance(segment_ids.kv, (jax.Array, np.ndarray))
        and segment_ids.q.ndim == 1
    ):
      segment_ids = SegmentIds(
          jnp.expand_dims(segment_ids.q, axis=0),
          jnp.expand_dims(segment_ids.kv, axis=0),
      )
    if k.ndim == 3:
      k = jnp.expand_dims(k, axis=0)
      v = jnp.expand_dims(v, axis=0)
    elif k.ndim == 2:
      k = jnp.expand_dims(k, axis=(0, 1))
      v = jnp.expand_dims(v, axis=(0, 1))
    else:
      raise ValueError(f"Unsupported k ndim {k.ndim} for 3D q")
  else:
    if k.ndim == 3:
      k = jnp.expand_dims(k, axis=1)
      v = jnp.expand_dims(v, axis=1)

  ring_size = lax.axis_size(axis_name)
  ring_idx = lax.axis_index(axis_name)
  if ring_size <= 1:
    raise ValueError("Ring size must be greater than 1.")

  shift_fn = functools.partial(
      lax.ppermute,
      axis_name=axis_name,
      perm=[(i, (i + 1) % ring_size) for i in range(ring_size)],
  )

  q_shard_len = q.shape[-2]
  kv_shard_len = k.shape[-2]
  eff_block_q = min(block_q, q_shard_len)
  eff_block_kv = min(block_kv, kv_shard_len)

  o_shape = (*q.shape[:-1], v.shape[-1])
  o_init = jnp.zeros(o_shape, dtype=jnp.float32)
  l_init = jnp.zeros(q.shape[:-1], dtype=jnp.float32)
  m_init = jnp.full(q.shape[:-1], mask_value, dtype=jnp.float32)
  q_start = ring_idx * q_shard_len

  mask_arr = _materialize_mask(mask)

  def body_fn(
      carry: tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          Any,
      ],
      i: jax.Array,
  ) -> tuple[
      tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          Any,
      ],
      None,
  ]:
    m_prev, l_prev, o_prev, k_curr, v_curr, seg_curr = carry
    current_kv_shard_idx = (ring_idx - i) % ring_size

    kv_start = current_kv_shard_idx * kv_shard_len
    if mask_arr.ndim == 2:
      local_mask = lax.dynamic_slice(
          mask_arr,
          (q_start, kv_start),
          (q_shard_len, kv_shard_len),
      )
    elif mask_arr.ndim == 3:
      if not q_is_3d and mask_arr.shape[0] == q.shape[0]:
        local_mask = lax.dynamic_slice(
            mask_arr,
            (0, q_start, kv_start),
            (mask_arr.shape[0], q_shard_len, kv_shard_len),
        )
      else:
        local_mask = lax.dynamic_slice(
            mask_arr[0],
            (q_start, kv_start),
            (q_shard_len, kv_shard_len),
        )
    elif mask_arr.ndim == 4:
      local_mask = lax.dynamic_slice(
          mask_arr[0, 0],
          (q_start, kv_start),
          (q_shard_len, kv_shard_len),
      )
    else:
      raise ValueError(f"Unsupported mask ndim: {mask_arr.ndim}")

    def compute_fwd() -> tuple[jax.Array, jax.Array, jax.Array]:
      flash_seg = splash_attention_kernel.SegmentIds(seg_curr.q, seg_curr.kv) if seg_curr is not None else None
      out_curr, stats = jax_flash_attention.flash_attention_block_masked(
          q=q,
          k=k_curr,
          v=v_curr,
          segment_ids=flash_seg,
          block_kv=eff_block_kv,
          block_q=eff_block_q,
          mask=local_mask,
          mask_value=mask_value,
          cap=cap,
          save_residuals=True,
          loop_unroll=unroll,
      )
      assert isinstance(stats, dict)
      lse_curr = stats["logsumexp"].astype(jnp.float32)
      m_curr = stats["max_logits"].astype(jnp.float32)

      l_curr = jnp.where(lse_curr == -jnp.inf, 0.0, jnp.exp(lse_curr - m_curr))
      o_curr = out_curr.astype(jnp.float32) * l_curr[..., None]

      m_next = jnp.maximum(m_prev, m_curr)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_next = alpha * l_prev + beta * l_curr
      o_next = alpha[..., None] * o_prev + beta[..., None] * o_curr
      return m_next, l_next, o_next

    def skip_fwd() -> tuple[jax.Array, jax.Array, jax.Array]:
      return m_prev, l_prev, o_prev

    has_valid_tokens = jnp.any(local_mask)
    m_next, l_next, o_next = lax.cond(has_valid_tokens, compute_fwd, skip_fwd)

    k_next = shift_fn(k_curr)
    v_next = shift_fn(v_curr)
    if seg_curr is not None:
      kv_seg_next = shift_fn(seg_curr.kv)
      seg_next = type(seg_curr)(seg_curr.q, kv_seg_next)
    else:
      seg_next = None

    return (m_next, l_next, o_next, k_next, v_next, seg_next), None

  (m_final, l_final, o_final, _, _, _), _ = lax.scan(
      body_fn,
      (m_init, l_init, o_init, k, v, segment_ids),
      xs=jnp.arange(ring_size),
      length=ring_size,
      unroll=unroll,
  )

  l_inv = jnp.where(l_final == 0.0, 0.0, 1.0 / l_final)
  out = (o_final * l_inv[..., None]).astype(q.dtype)

  lse = jnp.log(l_final) + m_final
  lse = jnp.where(l_final == 0.0, mask_value, lse)
  stats = {
      "logsumexp": lse.astype(jnp.float32),
      "max_logits": m_final.astype(jnp.float32),
  }
  stats = jax.tree_util.tree_map(jax.lax.stop_gradient, stats)

  if q_is_3d:
    out = jnp.squeeze(out, axis=0)
    stats = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), stats)

  if not save_residuals:
    return out
  return out, stats


def ring_attention_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    do: jax.Array,
    stats: Mapping[str, jax.Array],
    mask: MaskType,
    segment_ids: SegmentIdsType | None = None,
    *,
    mask_value: float = -1e30,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: int | None = None,
    block_kv_head: int | None = None,
    axis_name: str = "ring",
    cap: float | None = None,
    unroll: bool = False,
    fusion_specs: Mapping[str, Any] | None = None,
    is_causal: bool = False,
    use_log2: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes masked ring attention backward pass using FlashAttention blocks."""
  del fusion_specs, unroll, is_causal
  q_is_3d = q.ndim == 3
  k_was_2d = False
  k_was_3d_in_4d = False

  if q_is_3d:
    q = jnp.expand_dims(q, axis=0)
    out = jnp.expand_dims(out, axis=0)
    do = jnp.expand_dims(do, axis=0)
    logsumexp = jnp.expand_dims(stats["logsumexp"], axis=0)
    if (
        segment_ids is not None
        and isinstance(segment_ids.q, (jax.Array, np.ndarray))
        and isinstance(segment_ids.kv, (jax.Array, np.ndarray))
        and segment_ids.q.ndim == 1
    ):
      segment_ids = SegmentIds(
          jnp.expand_dims(segment_ids.q, axis=0),
          jnp.expand_dims(segment_ids.kv, axis=0),
      )
    if k.ndim == 3:
      k = jnp.expand_dims(k, axis=0)
      v = jnp.expand_dims(v, axis=0)
    elif k.ndim == 2:
      k_was_2d = True
      k = jnp.expand_dims(k, axis=(0, 1))
      v = jnp.expand_dims(v, axis=(0, 1))
    else:
      raise ValueError(f"Unsupported k ndim {k.ndim} for 3D q")
  else:
    logsumexp = stats["logsumexp"]
    if k.ndim == 3:
      k_was_3d_in_4d = True
      k = jnp.expand_dims(k, axis=1)
      v = jnp.expand_dims(v, axis=1)

  ring_size = lax.axis_size(axis_name)
  ring_idx = lax.axis_index(axis_name)
  if ring_size <= 1:
    raise ValueError("Ring size must be greater than 1.")

  shift_fn = functools.partial(
      lax.ppermute,
      axis_name=axis_name,
      perm=[(i, (i + 1) % ring_size) for i in range(ring_size)],
  )

  q_shard_len = q.shape[-2]
  kv_shard_len = k.shape[-2]
  eff_block_q = min(block_q, q_shard_len)
  eff_block_kv = min(block_kv, kv_shard_len)

  dq_accum = jnp.zeros_like(q, dtype=q.dtype)
  dk_accum = jnp.zeros_like(k, dtype=k.dtype)
  dv_accum = jnp.zeros_like(v, dtype=v.dtype)
  q_start = ring_idx * q_shard_len

  di = jnp.sum(out * do, axis=-1)
  d_k = k.shape[-1]

  mask_arr = _materialize_mask(mask)

  def body_fn(
      carry: tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          Any,
      ],
      i: jax.Array,
  ) -> tuple[
      tuple[
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          jax.Array,
          Any,
      ],
      None,
  ]:
    dq_acc, dk_acc, dv_acc, k_curr, v_curr, seg_curr = carry
    current_kv_shard_idx = (ring_idx - i) % ring_size

    kv_start = current_kv_shard_idx * kv_shard_len
    if mask_arr.ndim == 2:
      local_mask = lax.dynamic_slice(
          mask_arr,
          (q_start, kv_start),
          (q_shard_len, kv_shard_len),
      )
    elif mask_arr.ndim == 3:
      if not q_is_3d and mask_arr.shape[0] == q.shape[0]:
        local_mask = lax.dynamic_slice(
            mask_arr,
            (0, q_start, kv_start),
            (mask_arr.shape[0], q_shard_len, kv_shard_len),
        )
      else:
        local_mask = lax.dynamic_slice(
            mask_arr[0],
            (q_start, kv_start),
            (q_shard_len, kv_shard_len),
        )
    elif mask_arr.ndim == 4:
      local_mask = lax.dynamic_slice(
          mask_arr[0, 0],
          (q_start, kv_start),
          (q_shard_len, kv_shard_len),
      )
    else:
      raise ValueError(f"Unsupported mask ndim: {mask_arr.ndim}")

    def compute_bwd() -> tuple[jax.Array, jax.Array, jax.Array]:
      flash_seg = splash_attention_kernel.SegmentIds(seg_curr.q, seg_curr.kv) if seg_curr is not None else None
      return jax_flash_attention.flash_attention_bwd(
          q=q,
          k=k_curr,
          v=v_curr,
          o=out,
          do=do,
          logsumexp=logsumexp,
          mask=local_mask,
          segment_ids=flash_seg,
          mask_value=mask_value,
          cap=cap,
          block_q=eff_block_q,
          block_kv=eff_block_kv,
          block_q_head=block_q_head,
          block_kv_head=block_kv_head,
          dtype=q.dtype,
          is_causal=False,
          use_log2=use_log2,
          di=di,
          fuse=False,
      )

    def skip_bwd() -> tuple[jax.Array, jax.Array, jax.Array]:
      return (
          jnp.zeros_like(q, dtype=q.dtype),
          jnp.zeros_like(k, dtype=k.dtype),
          jnp.zeros_like(v, dtype=v.dtype),
      )

    has_valid_tokens = jnp.any(local_mask)
    dq_curr, dk_curr, dv_curr = lax.cond(
        has_valid_tokens,
        compute_bwd,
        skip_bwd,
    )

    dq_next = (dq_acc + dq_curr).astype(q.dtype)
    dk_updated = (dk_acc + dk_curr).astype(k.dtype)
    dv_updated = (dv_acc + dv_curr).astype(v.dtype)

    kv_curr = jnp.concatenate([k_curr, v_curr], axis=-1)
    kv_next = shift_fn(kv_curr)
    k_next, v_next = jnp.split(kv_next, [d_k], axis=-1)

    dk_dv_updated = jnp.concatenate([dk_updated, dv_updated], axis=-1)
    dk_dv_next = shift_fn(dk_dv_updated)
    dk_next, dv_next = jnp.split(dk_dv_next, [d_k], axis=-1)

    if seg_curr is not None:
      kv_seg_next = shift_fn(seg_curr.kv)
      seg_next = type(seg_curr)(seg_curr.q, kv_seg_next)
    else:
      seg_next = None

    return (dq_next, dk_next, dv_next, k_next, v_next, seg_next), None

  (dq_final, dk_final, dv_final, _, _, _), _ = lax.scan(
      body_fn,
      (dq_accum, dk_accum, dv_accum, k, v, segment_ids),
      xs=jnp.arange(ring_size),
      length=ring_size,
      unroll=True,
  )

  dq = dq_final.astype(q.dtype)
  dk = dk_final.astype(k.dtype)
  dv = dv_final.astype(v.dtype)

  if q_is_3d:
    dq = jnp.squeeze(dq, axis=0)
    if k_was_2d:
      dk = jnp.squeeze(dk, axis=(0, 1))
      dv = jnp.squeeze(dv, axis=(0, 1))
    else:
      dk = jnp.squeeze(dk, axis=0)
      dv = jnp.squeeze(dv, axis=0)
  elif k_was_3d_in_4d:
    dk = jnp.squeeze(dk, axis=1)
    dv = jnp.squeeze(dv, axis=1)

  return dq, dk, dv


@functools.partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "mask_value",
        "block_q",
        "block_kv",
        "block_q_head",
        "block_kv_head",
        "axis_name",
        "cap",
        "unroll",
        "fusion_specs",
        "is_causal",
        "use_log2",
    ),
)
def ring_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: MaskType,
    segment_ids: SegmentIdsType | None = None,
    mask_value: float = -1e30,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: int | None = None,
    block_kv_head: int | None = None,
    axis_name: str = "ring",
    cap: float | None = None,
    unroll: bool = False,
    fusion_specs: Mapping[str, Any] | None = None,
    is_causal: bool = False,
    use_log2: bool = False,
) -> jax.Array:
  """Computes masked ring attention using FlashAttention blocks."""
  del block_q_head, block_kv_head, is_causal, use_log2
  out, _ = ring_attention_fwd(
      q=q,
      k=k,
      v=v,
      mask=mask,
      segment_ids=segment_ids,
      mask_value=mask_value,
      block_q=block_q,
      block_kv=block_kv,
      axis_name=axis_name,
      cap=cap,
      save_residuals=True,
      unroll=unroll,
      fusion_specs=fusion_specs,
  )
  assert isinstance(out, jnp.ndarray)
  return out


def _ring_attention_fwd_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: MaskType,
    segment_ids: SegmentIdsType | None = None,
    mask_value: float = -1e30,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: int | None = None,
    block_kv_head: int | None = None,
    axis_name: str = "ring",
    cap: float | None = None,
    unroll: bool = False,
    fusion_specs: Mapping[str, Any] | None = None,
    is_causal: bool = False,
    use_log2: bool = False,
) -> tuple[jax.Array, tuple[Any, ...]]:
  """Custom VJP forward rule for ring_attention."""
  del block_q_head, block_kv_head, is_causal, use_log2
  out, stats = ring_attention_fwd(
      q=q,
      k=k,
      v=v,
      mask=mask,
      segment_ids=segment_ids,
      mask_value=mask_value,
      block_q=block_q,
      block_kv=block_kv,
      axis_name=axis_name,
      cap=cap,
      save_residuals=True,
      unroll=unroll,
      fusion_specs=fusion_specs,
  )
  res = (q, k, v, mask, segment_ids, out, stats)
  return out, res


def _ring_attention_bwd_rule(
    mask_value: float,
    block_q: int,
    block_kv: int,
    block_q_head: int | None,
    block_kv_head: int | None,
    axis_name: str,
    cap: float | None,
    unroll: bool,
    fusion_specs: Mapping[str, Any] | None,
    is_causal: bool,
    use_log2: bool,
    res: tuple[Any, ...],
    do: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None]:
  """Custom VJP backward rule for ring_attention."""
  q, k, v, mask, segment_ids, out, stats = res
  dq, dk, dv = ring_attention_bwd(
      q=q,
      k=k,
      v=v,
      out=out,
      do=do,
      stats=stats,
      mask=mask,
      segment_ids=segment_ids,
      mask_value=mask_value,
      block_q=block_q,
      block_kv=block_kv,
      block_q_head=block_q_head,
      block_kv_head=block_kv_head,
      axis_name=axis_name,
      cap=cap,
      unroll=unroll,
      fusion_specs=fusion_specs,
      is_causal=is_causal,
      use_log2=use_log2,
  )
  return dq, dk, dv, None, None


ring_attention.defvjp(_ring_attention_fwd_rule, _ring_attention_bwd_rule)


@jax.tree_util.register_pytree_node_class
class RingJaxAttentionKernel:
  """Implements Ring Attention using native JAX operations."""

  def __init__(
      self,
      mask: MaskType | jax.sharding.PartitionSpec,
      ring_axis: str,
      block_q: int = 1024,
      block_kv: int = 1024,
      mask_value: float = DEFAULT_MASK_VALUE,
      cap: float | None = None,
      is_mqa: bool = False,
      save_residuals: bool = False,
      unroll: bool = False,
      is_causal: bool = False,
      use_log2: bool = False,
      **kwargs: Any,
  ):
    del is_mqa, save_residuals
    if isinstance(mask, (mask_lib.Mask, tokamax_mask_lib.Mask)):
      self._mask = mask[:, :]
    else:
      self._mask = mask
    self._ring_axis = ring_axis
    self._block_q = block_q
    self._block_kv = block_kv
    self._mask_value = mask_value
    self._cap = cap
    self._unroll = unroll
    self._is_causal = is_causal
    self._use_log2 = use_log2
    self._kwargs = kwargs

  def __call__(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      segment_ids: SegmentIdsType | None = None,
      sinks: jax.Array | None = None,
      indexer_mask: jax.Array | None = None,
  ) -> jax.Array:
    if sinks is not None:
      raise NotImplementedError("Sinks are not supported in JAX ring attention.")
    if indexer_mask is not None:
      raise NotImplementedError("Indexer mask is not supported in JAX ring attention.")
    assert not isinstance(self._mask, jax.sharding.PartitionSpec)
    return ring_attention(
        q,
        k,
        v,
        mask=self._mask,
        segment_ids=segment_ids,
        mask_value=self._mask_value,
        block_q=self._block_q,
        block_kv=self._block_kv,
        axis_name=self._ring_axis,
        cap=self._cap,
        unroll=self._unroll,
        is_causal=self._is_causal,
        use_log2=self._use_log2,
    )

  def manual_sharding_spec(self) -> RingJaxAttentionKernel:
    """Returns a kernel with replicated PartitionSpec for shard_map."""
    return RingJaxAttentionKernel(
        mask=jax.sharding.PartitionSpec(),
        ring_axis=self._ring_axis,
        block_q=self._block_q,
        block_kv=self._block_kv,
        mask_value=self._mask_value,
        cap=self._cap,
        unroll=self._unroll,
        is_causal=self._is_causal,
        use_log2=self._use_log2,
        **self._kwargs,
    )

  def tree_flatten(
      self,
  ) -> tuple[
      tuple[np.ndarray | jax.Array | jax.sharding.PartitionSpec],
      dict[str, Any],
  ]:
    """Flattens RingJaxAttentionKernel for JAX PyTree registration."""
    children = (self._mask,)
    aux_data = {
        "ring_axis": self._ring_axis,
        "block_q": self._block_q,
        "block_kv": self._block_kv,
        "mask_value": self._mask_value,
        "cap": self._cap,
        "unroll": self._unroll,
        "is_causal": self._is_causal,
        "use_log2": self._use_log2,
        **self._kwargs,
    }
    return children, aux_data

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: Mapping[str, Any],
      children: Sequence[np.ndarray | jax.Array | jax.sharding.PartitionSpec],
  ) -> RingJaxAttentionKernel:
    """Unflattens RingJaxAttentionKernel from JAX PyTree representation."""
    (mask,) = children
    return cls(mask=mask, **aux_data)


def make_ring_attention(
    mask: MaskType,
    *,
    config: Any | None = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    is_mqa: bool = False,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    ring_axis: str = "ring",
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
    unroll: bool = False,
    is_causal: bool = False,
    use_log2: bool = False,
) -> RingJaxAttentionKernel:
  """Creates a RingJaxAttentionKernel instance.

  Args:
    mask: The attention mask (2D array or Splash Mask).
    config: Optional SplashConfig to extract block sizes and soft cap from.
    block_q: Block size along the query sequence dimension.
    block_kv: Block size along the key and value sequence dimension.
    is_mqa: Whether the model uses Multi Query Attention.
    save_residuals: Whether to save residuals for the backward pass.
    mask_value: The value to use for masked out attention scores.
    ring_axis: The name of the JAX axis used for the ring.
    q_seq_shards: The number of shards for the query sequence dimension.
    kv_seq_shards: The number of shards for the key and value sequence
      dimension.
    unroll: Whether to unroll the forward scan loop.
    is_causal: Whether causal masking is used.
    use_log2: Whether to use base 2 logarithm in backward attention.

  Returns:
    A RingJaxAttentionKernel instance.
  """
  del q_seq_shards, kv_seq_shards
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")
  if not isinstance(
      mask,
      (
          np.ndarray,
          jax.Array,
          mask_lib.NumpyMask,
          mask_lib.FullMask,
          mask_lib.CausalMask,
          tokamax_mask_lib.NumpyMask,
          tokamax_mask_lib.FullMask,
          tokamax_mask_lib.CausalMask,
      ),
  ):
    raise NotImplementedError("Only NumpyMask, FullMask, and CausalMask are supported, got" f" {type(mask)}.")

  if config is not None:
    block_q = config.block_q
    block_kv = config.block_kv
    cap = config.attn_logits_soft_cap
  else:
    cap = None

  return RingJaxAttentionKernel(
      mask=mask,
      ring_axis=ring_axis,
      block_q=block_q,
      block_kv=block_kv,
      mask_value=mask_value,
      cap=cap,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      unroll=unroll,
      is_causal=is_causal,
      use_log2=use_log2,
  )
