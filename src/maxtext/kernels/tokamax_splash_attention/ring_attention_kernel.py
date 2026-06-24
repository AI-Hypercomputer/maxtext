# pylint: skip-file
# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Implementation of Ring Attention."""

import dataclasses
import functools
from typing import Any

import jax
from jax import lax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.tokamax_splash_attention import base
from maxtext.kernels.tokamax_splash_attention import ring_attention_utils
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as splash_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as mask_lib
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info as mask_info_lib

P = jax.P
MaskInfo = mask_info_lib.MaskInfo
partial = functools.partial

SegmentIds = base.SegmentIds
SplashConfig = splash_kernel.SplashConfig
SplashResidualsType = base.SplashResidualsType
SplashCustomReturnType = base.SplashCustomReturnType
MaskFunctionType = splash_kernel.MaskFunctionType
_splash_attention_forward = splash_kernel._splash_attention_forward  # pylint: disable=protected-access
_splash_attention_bwd = splash_kernel._splash_attention_bwd  # pylint: disable=protected-access

_dynamic_slice_mask_info = ring_attention_utils.dynamic_slice_mask_info
_offset_q_sequence_for_kv_shard = ring_attention_utils.offset_q_sequence_for_kv_shard
_has_no_active_blocks = ring_attention_utils.has_no_active_blocks
_has_empty_attention_rows = ring_attention_utils.has_empty_attention_rows
_mask_sparsity = ring_attention_utils.mask_sparsity
_has_axis = ring_attention_utils.has_axis


def _validate_ring_axis_size(ring_axis: str, ring_axis_size: int, expected_ring_size: int) -> None:
  if ring_axis_size != expected_ring_size:
    raise ValueError(
        f"Ring axis {ring_axis} has size {ring_axis_size}, but ring attention "
        f"was built for {expected_ring_size} sequence shards."
    )


def _ring_attention_forward(
    fwd_mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    *,
    sinks: jax.Array | None = None,
    ring_axis: str,
    expected_ring_size: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:

  if q.shape[-1] != k.shape[-1]:
    raise NotImplementedError("Queries and keys must have the same head dimension.")

  if sinks is not None:
    raise NotImplementedError("Sinks aren't supported yet.")

  ring_axis_size = lax.axis_size(ring_axis)
  _validate_ring_axis_size(ring_axis, ring_axis_size, expected_ring_size)
  ring_axis_idx = lax.axis_index(ring_axis)

  shift = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
  )
  # for example, if ring size is 4
  # Device 3 => permute_idx 0, offset (3-0) % 4 = 3,
  #             permute_idx 1, offset (3-1) % 4 = 2, etc.
  # Device 2 => permute_idx 0, offset (2-0) % 4 = 2,
  #             permute_idx 1, offset (2-1) % 4 = 1, etc.
  # Device 1 => permute_idx 0, offset (1-0) % 4 = 1,
  #             permute_idx 1, offset (1-1) % 4 = 0, etc.
  # Device 0 => permute_idx 0, offset (0-0) % 4 = 0,
  #             permute_idx 1, offset (0-1) % 4 = 3, etc.

  splash_fwd_partial = partial(
      _splash_attention_forward,
      save_residuals=True,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=None,
  )
  # Initial accumulator values
  o_shape = q.shape[:-1] + (v.shape[-1],)
  o_init = jnp.zeros(o_shape, dtype=jnp.float32)
  l_init = jnp.zeros((o_shape[0], o_shape[1]), jnp.float32)
  m_init = jnp.full_like(l_init, mask_value, dtype=jnp.float32)

  def body(carry, i: int):
    m_prev, l_prev, o_prev, k_current, v_current, segment_ids_current = carry

    current_kv_shard_idx = (ring_axis_idx - i) % ring_axis_size
    local_fwd_mask_info = _dynamic_slice_mask_info(fwd_mask_info, current_kv_shard_idx, ring_axis_size)
    local_fwd_mask_info = _offset_q_sequence_for_kv_shard(
        local_fwd_mask_info, current_kv_shard_idx, k_current.shape[-2]
    )
    k_next = shift(k_current)
    v_next = shift(v_current)

    if segment_ids is not None:
      kv_segment_ids_next = shift(segment_ids_current.kv)
      segment_ids_next = SegmentIds(segment_ids.q, kv_segment_ids_next)
    else:
      segment_ids_next = None

    out_curr, stats = splash_fwd_partial(
        local_fwd_mask_info,
        q,
        k_current,
        v_current,
        segment_ids=segment_ids_current,
        sinks=sinks,
    )
    no_active_blocks = _has_no_active_blocks(local_fwd_mask_info)
    out_curr = jnp.where(no_active_blocks, jnp.zeros_like(out_curr), out_curr)
    lse_curr = jnp.where(
        no_active_blocks,
        jnp.full_like(stats["logsumexp"], mask_value),
        stats["logsumexp"],
    )
    m_curr = jnp.where(
        no_active_blocks,
        jnp.full_like(stats["max_logits"], mask_value),
        stats["max_logits"],
    )
    empty_rows = _has_empty_attention_rows(lse_curr, m_curr, mask_value)
    l_curr = jnp.exp(lse_curr - m_curr)
    l_curr = jnp.where(empty_rows, jnp.zeros_like(l_curr), l_curr)
    o_curr = out_curr.astype(jnp.float32) * l_curr[..., None]
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    o_next = alpha[..., None] * o_prev + beta[..., None] * o_curr
    return (m_next, l_next, o_next, k_next, v_next, segment_ids_next), None

  initial_carry = (m_init, l_init, o_init, k, v, segment_ids)
  (m_final, l_final, o_final, _, _, _), _ = lax.scan(
      body,
      initial_carry,
      xs=jnp.arange(0, ring_axis_size),
      length=ring_axis_size,
      unroll=False,
  )  # type: ignore[arg-type]
  # Final normalization
  assert l_final.dtype == jnp.float32
  l_inv = jnp.where(l_final == 0.0, 0.0, 1.0 / l_final)
  out = (o_final * l_inv[..., None]).astype(q.dtype)
  # Final logsumexp for residuals
  lse = jnp.log(l_final) + m_final
  lse = jnp.where(l_final == 0.0, mask_value, lse)

  return out, (lse, m_final)


def _ring_attention_bwd(
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    expected_ring_size: int,
    # Residuals and gradients
    res: Any,
    do: jax.Array,
):
  del save_residuals
  (q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info) = res
  do = do.astype(jnp.float32)
  if dkv_mask_info is None:
    raise ValueError("Need to specify backward blocks.")

  ring_axis_size = lax.axis_size(ring_axis)
  _validate_ring_axis_size(ring_axis, ring_axis_size, expected_ring_size)
  ring_axis_idx = lax.axis_index(ring_axis)

  shift = partial(
      lax.ppermute,
      axis_name=ring_axis,
      perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
  )
  dq_accum = jnp.zeros(q.shape, dtype=jnp.float32)
  dk_accum = jnp.zeros(k.shape, dtype=jnp.float32)
  dv_accum = jnp.zeros(v.shape, dtype=jnp.float32)
  dsinks = sinks

  def body(carry, i: int):
    (
        dq_accum,
        dk_accum,
        dv_accum,
        k_current,
        v_current,
        segment_ids_current,
        _,
    ) = carry
    k_next = shift(k_current)
    v_next = shift(v_current)

    current_kv_shard_idx = (ring_axis_idx - i) % ring_axis_size
    local_dkv_mask_info = _dynamic_slice_mask_info(dkv_mask_info, current_kv_shard_idx, ring_axis_size)
    local_dkv_mask_info = _offset_q_sequence_for_kv_shard(
        local_dkv_mask_info, current_kv_shard_idx, k_current.shape[-2]
    )
    if segment_ids is not None:
      kv_segment_ids_next = shift(segment_ids_current.kv)
      segment_ids_next = SegmentIds(segment_ids.q, kv_segment_ids_next)
    else:
      segment_ids_next = None

    residuals_for_chunk = (
        q,
        k_current,
        v_current,
        segment_ids_current,
        sinks,
        out,
        logsumexp,
        local_dkv_mask_info,
    )

    attn_bwd = functools.partial(
        _splash_attention_bwd,
        save_residuals=False,
        mask_value=mask_value,
        is_mqa=is_mqa,
        config=config,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        dkv_mask_sparsity=dkv_mask_sparsity,
        return_fp32_grads=True,
    )
    _, _, dq_i, dk_i, dv_i, _, dsinks, _ = attn_bwd(res=residuals_for_chunk, grads=do)
    dv_next = shift(dv_accum + dv_i.astype(jnp.float32))
    dk_next = shift(dk_accum + dk_i.astype(jnp.float32))
    dq_accum = dq_accum + dq_i.astype(jnp.float32)

    return (
        dq_accum,
        dk_next,
        dv_next,
        k_next,
        v_next,
        segment_ids_next,
        dsinks,
    ), None

  initial_carry = (dq_accum, dk_accum, dv_accum, k, v, segment_ids, dsinks)
  (dq, dk, dv, _, _, _, dsinks), _ = lax.scan(
      body,
      initial_carry,
      xs=jnp.arange(ring_axis_size),
      length=ring_axis_size,
      unroll=False,
  )

  if sinks is not None:
    dsinks = jax.lax.psum(dsinks, axis_name=ring_axis)

  return (
      None,  # fwd_mask_info
      None,  # dkv_mask_info
      dq.astype(q.dtype),
      dk.astype(k.dtype),
      dv.astype(v.dtype),
      None,
      dsinks,
  )


def _ring_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    # nondiff_args
    mask_value: float,  # 1
    is_mqa: bool,  # 2
    config: SplashConfig | None,  # 3
    mask_function: MaskFunctionType | None,  # 4
    fwd_mask_sparsity: float,  # 5
    dkv_mask_sparsity: float,  # 6
    save_residuals: bool,  # 7
    ring_axis: str,  # 8
    expected_ring_size: int,  # 9
) -> tuple[jax.Array, SplashResidualsType]:
  """Forward pass for the custom VJP of ring attention.

  This function is used by `jax.custom_vjp` to define the forward pass
  of the ring attention computation, also returning residuals needed for
  the backward pass.

  Args:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    q: Query array.
    k: Key array.
    v: Value array.
    segment_ids: Optional segment IDs for packed sequences.
    sinks: Optional sink tokens.
    mask_value: The value used for masked-out attention scores.
    is_mqa: Whether Multi-Query Attention is used.
    config: SplashAttention configuration.
    mask_function: Optional function to apply additional masking.
    fwd_mask_sparsity: Sparsity level of the forward mask.
    save_residuals: Whether to save residuals for the backward pass.
    ring_axis: The name of the jax axis used for the ring.

  Returns:
    A tuple containing:
      - The output of the ring attention computation.
      - Residuals needed for the backward pass (`SplashResidualsType`).
  """
  del dkv_mask_sparsity
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported.")

  out, (logsumexp, max_logits) = _ring_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      ring_axis=ring_axis,
      expected_ring_size=expected_ring_size,
  )
  residuals = (q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info)
  return out, residuals


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "save_residuals",
        "ring_axis",
        "expected_ring_size",
    ),
)
def _ring_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool,
    ring_axis: str,
    expected_ring_size: int,
) -> SplashCustomReturnType:
  """Performs ring attention with a custom VJP.

  This function is a wrapper around `_ring_attention_forward` and is used
  to define the custom gradient for ring attention.

  Args:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    q: Query array.
    k: Key array.
    v: Value array.
    segment_ids: Optional segment IDs for packed sequences.
    sinks: Optional sink tokens.
    mask_value: The value used for masked-out attention scores.
    is_mqa: Whether Multi-Query Attention is used.
    config: SplashAttention configuration.
    mask_function: Optional function to apply additional masking.
    fwd_mask_sparsity: Sparsity level of the forward mask.
    save_residuals: Whether to save residuals for the backward pass.
    ring_axis: The name of the jax axis used for the ring.

  Returns:
    The output of the ring attention computation.
  """
  del dkv_mask_info, dkv_mask_sparsity
  out, _ = _ring_attention_forward(
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks=sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      ring_axis=ring_axis,
      expected_ring_size=expected_ring_size,
  )
  return out


_ring_attention_custom.defvjp(_ring_attention_fwd, _ring_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
        "save_residuals",
        "ring_axis",
        "expected_ring_size",
    ],
)
def _ring_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    save_residuals: bool = False,
    ring_axis: str,
    expected_ring_size: int,
) -> SplashCustomReturnType:
  """Performs ring attention using SplashAttention kernels.

  This function orchestrates the ring attention mechanism by iterating through
  shards of keys and values across devices along the specified `ring_axis`,
  using `_splash_attention_forward` for each chunk.

  Args:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    q: Query array.
    k: Key array.
    v: Value array.
    segment_ids: Optional segment IDs for packed sequences.
    sinks: Optional sink tokens.
    is_mqa: Whether Multi-Query Attention is used.
    config: SplashAttention configuration.
    mask_value: The value used for masked-out attention scores.
    mask_function: Optional function to apply additional masking.
    fwd_mask_sparsity: Sparsity level of the forward mask.
    save_residuals: Whether to save residuals for the backward pass.
    ring_axis: The name of the jax axis used for the ring.

  Returns:
    The output of the ring attention computation.

  Raises:
    ValueError: If the specified `ring_axis` does not exist.
  """
  if not _has_axis(ring_axis):
    raise ValueError(f"Ring axis {ring_axis} does not exist")

  return _ring_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      is_mqa=is_mqa,
      config=config,
      mask_value=mask_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
      save_residuals=save_residuals,
      ring_axis=ring_axis,
      expected_ring_size=expected_ring_size,
  )


@jax.tree_util.register_pytree_node_class
class RingSplashAttentionKernel:
  """Implements Ring Attention using SplashAttention for sequence parallelism.

  This kernel computes global attention by keeping Keys and Values distributed
  across the `ring_axis`. Instead of gathering full sequences, it rotates K/V
  shards between devices and accumulates results incrementally. This allows
  processing sequence lengths that exceed single-device memory limits.

  Attributes:
    fwd_mask_info: Mask information for the forward pass.
    dkv_mask_info: Mask information for the backward pass for dK/dV.
    ring_axis: The name of the jax axis used for the ring.
    expected_ring_size: Number of sequence shards used when building the
      kernel.
    kwargs: Additional keyword arguments passed to the SplashAttentionKernel
      constructor.
  """

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      ring_axis: str,
      expected_ring_size: int,
      **kwargs,
  ):
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info
    self.ring_axis = ring_axis
    self.expected_ring_size = expected_ring_size
    self.kwargs = kwargs

  def __call__(self, *args, **kwargs):
    reserved_kwargs = set(self.kwargs) | {"ring_axis", "expected_ring_size"}
    overrides = reserved_kwargs.intersection(kwargs)
    if overrides:
      raise ValueError(
          "Ring attention call-time kwargs cannot override validated kernel " f"configuration: {sorted(overrides)}."
      )
    return _ring_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(
            self.kwargs,
            **kwargs,
            ring_axis=self.ring_axis,
            expected_ring_size=self.expected_ring_size,
        ),
    )

  def manual_sharding_spec(self):
    """Ring attention expects MaskInfo metadata to be sharded by Q shard.

    Each local Q shard slices the current KV shard metadata at each ring step.
    Partial mask blocks stay replicated because they are shared by all shards.
    """

    spec = jax.sharding.PartitionSpec(self.ring_axis)
    _resolve_spec = lambda x: spec if x is not None else None

    def mask_info_spec(mask_info):
      if mask_info is None:
        return None
      return MaskInfo(  # pytype: disable=wrong-arg-types
          mask_next=_resolve_spec(mask_info.mask_next),
          active_rows=_resolve_spec(mask_info.active_rows),
          active_cols=_resolve_spec(mask_info.active_cols),
          num_active_blocks=_resolve_spec(mask_info.num_active_blocks),
          block_mask=_resolve_spec(mask_info.block_mask),
          partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated
          if mask_info.partial_mask_blocks is not None
          else None,
          q_sequence=_resolve_spec(mask_info.q_sequence),
      )

    return RingSplashAttentionKernel(
        mask_info_spec(self.fwd_mask_info),
        mask_info_spec(self.dkv_mask_info),
        ring_axis=self.ring_axis,
        expected_ring_size=self.expected_ring_size,
        **self.kwargs,
    )

  def tree_flatten(self):
    children = (self.fwd_mask_info, self.dkv_mask_info)
    aux_data = self.kwargs.copy()
    aux_data["ring_axis"] = self.ring_axis
    aux_data["expected_ring_size"] = self.expected_ring_size
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    fwd_mask_info, dkv_mask_info = children
    dkv_mask_info = mask_info_lib.MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    return cls(
        mask_info_lib.MaskInfo(*fwd_mask_info),
        dkv_mask_info,
        **aux_data,
    )


def make_ring_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = base.DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    ring_axis: str,
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
):
  """Creates a RingSplashAttentionKernel.

  Args:
    mask: The attention mask. Ring attention supports square FullMask and
      CausalMask with offset=0.
    config: SplashAttention configuration. If None, uses the default config.
    is_mqa: Whether the model uses Multi-Query Attention.
    save_residuals: Whether to save residuals for the backward pass.
    mask_value: The value to use for masked-out attention scores.
    downcast_smem_data: Whether to downcast data in shared memory.
    partial_mask_blocks_dtype: The dtype for partial mask blocks.
    ring_axis: The name of the jax scan axis used for the ring.
    q_seq_shards: The number of shards for the query sequence dimension.
    kv_seq_shards: The number of shards for the key/value sequence dimension.

  Returns:
    A RingSplashAttentionKernel instance.

  Raises:
    ValueError: If the mask shape is unexpected or ring_axis is not specified.
    NotImplementedError: If a requested mask, config, or residual mode is not
      supported by the ring attention implementation.
  """

  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")
  if mask.shape[0] != mask.shape[1]:
    raise NotImplementedError(
        "Ring attention currently supports only square self-attention masks; " f"got shape {mask.shape}."
    )
  if save_residuals:
    raise NotImplementedError("Ring attention does not support save_residuals=True.")
  if q_seq_shards <= 0 or kv_seq_shards <= 0:
    raise ValueError("q_seq_shards and kv_seq_shards must be positive.")
  if q_seq_shards != kv_seq_shards:
    raise NotImplementedError("Ring attention currently requires q_seq_shards == kv_seq_shards.")

  if isinstance(mask, (np.ndarray, mask_lib.NumpyMask)):
    raise NotImplementedError(
        "Ring attention does not support dense NumpyMask yet; use FullMask or " "CausalMask with offset=0."
    )

  if not isinstance(mask, (mask_lib.FullMask, mask_lib.CausalMask)):
    raise NotImplementedError(
        "Only FullMask and CausalMask are supported for ring attention; "
        "other lazy masks are intentionally unsupported until rotated-shard "
        f"semantics are covered. Got {type(mask)}."
    )
  if isinstance(mask, mask_lib.CausalMask) and mask.offset != 0:
    raise NotImplementedError("Ring attention supports CausalMask only with offset=0; " f"got offset={mask.offset}.")

  default_config = SplashConfig.get_default()
  if config is None:
    config = dataclasses.replace(default_config, use_base2_exp=False)
  elif config.use_base2_exp:
    raise NotImplementedError(
        "Ring attention does not support use_base2_exp=True because ring "
        "normalization currently combines natural-log softmax statistics."
    )
  if not config.fuse_reciprocal:
    raise NotImplementedError(
        "Ring attention requires fuse_reciprocal=True because ring "
        "normalization combines already-normalized per-shard outputs."
    )

  process_fn = partial(
      mask_info_lib.process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
      kv_seq_shards=kv_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
      return_dynamic_grid=True,
  )
  fwd_mask_sparsity = _mask_sparsity(fwd_mask_info)
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  dkv_mask_sparsity = 0.0
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)
    dkv_mask_sparsity = _mask_sparsity(dkv_mask_info)
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

  return RingSplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      ring_axis=ring_axis,
      expected_ring_size=q_seq_shards,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )
