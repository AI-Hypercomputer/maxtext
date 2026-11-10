#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""JAX implementation without using Pallas for Flash Attention."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from jax import lax
from jax.experimental import layout
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.xla_metadata import must_fuse_call
import jax.numpy as jnp
from maxtext.kernels.attention.splash_attention_kernel import SegmentIds

DLL = layout.Layout
Layout = layout.Format

_LONG_SEQUENCE_THRESHOLD = 4096


def mask_blocker(mask: jnp.ndarray, block_q: int, block_kv: int) -> jnp.ndarray:
  """Creates a blocked mask from a full mask.

  Args:
    mask: The attention mask with shape of (batch_size, q_seq_len, kv_seq_len).
    block_q: Block size for the query sequence dimension.
    block_kv: Block size for the key/value sequence dimension.

  Returns:
    A blocked mask where each element indicates the number of non-zero
    elements in the corresponding block of the original mask.
  """
  batch_size, q_seq_len, kv_seq_len = mask.shape

  if q_seq_len % block_q != 0:
    raise ValueError(f"q_seq_len {q_seq_len} must be divisible by block_q {block_q}")
  if kv_seq_len % block_kv != 0:
    raise ValueError(f"kv_seq_len {kv_seq_len} must be divisible by block_kv {block_kv}")
  q_blocks = q_seq_len // block_q
  kv_blocks = kv_seq_len // block_kv

  blocked_mask = mask.reshape(batch_size, q_blocks, block_q, kv_blocks, block_kv)
  return jnp.count_nonzero(blocked_mask, axis=(2, 4)).astype(jnp.int32)


def _prepare_mask_and_block_predicate(
    mask: mask_lib.Mask | jax.Array,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
    segment_ids: SegmentIds | None,
) -> Tuple[jax.Array, Callable[[int, int], Any]]:
  """Extracts the full mask and constructs the block computation predicate."""
  mask_array = mask[:, :]
  if mask_array.ndim == 2:
    # A rank-2 mask is shared by every item in the batch.
    mask_full = jnp.broadcast_to(mask_array[None, :, :], (batch_size, q_seq_len, kv_seq_len))
  else:
    if mask_array.shape != (batch_size, q_seq_len, kv_seq_len):
      raise ValueError(
          "Batched attention mask must have shape" f" {(batch_size, q_seq_len, kv_seq_len)}, got {mask_array.shape}."
      )
    mask_full = jnp.asarray(mask_array)

  if segment_ids is not None:
    segment_ids_q = segment_ids.q[:, :, None] if segment_ids.q.ndim == 2 else segment_ids.q[None, :, None]
    segment_ids_kv = segment_ids.kv[:, None, :] if segment_ids.kv.ndim == 2 else segment_ids.kv[None, None, :]
    mask_full = jnp.logical_and(mask_full, segment_ids_q == segment_ids_kv)

  if isinstance(mask, mask_lib.CausalMask):
    # In the case of a causal mask, the compute_attention_block should be executed
    # if the current block (i, j) falls within the lower triangle. This means that
    # the maximum query index in block i must be greater than or equal to the
    # minimum key/value index in block j.
    # Max q_idx in block i: (i + 1) * block_q - 1
    # Min kv_idx in block j: j * block_kv
    # Condition: (i + 1) * block_q - 1 >= j * block_kv
    # Which simplifies to: (i + 1) * block_q > j * block_kv
    def should_compute_block(i: int, j: int) -> bool | jax.Array:
      return (i + 1) * block_q > j * block_kv

  else:
    mask_blocked = jax.jit(mask_blocker, static_argnums=[1, 2])(mask_full, block_q, block_kv)

    def should_compute_block(i: int, j: int) -> bool | jax.Array:
      mask_i_j_slice = jax.lax.dynamic_slice(mask_blocked, (0, i, j), (batch_size, 1, 1))
      # The compute_attention_block should be executed if at least one element
      # in the slice is non-zero, meaning at least one batch requires work for
      # this block.
      return jnp.any(jnp.not_equal(mask_i_j_slice, 0))

  return mask_full, should_compute_block


def _compute_residual_stats(
    l: jnp.ndarray,
    m: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
  """Computes logsumexp and max_logits residual statistics with stopped gradients."""
  stats = {"logsumexp": m + jnp.log(l), "max_logits": m}
  return jax.tree.map(jax.lax.stop_gradient, stats)


def _flash_attention_block_masked_short_seq(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: SegmentIds | None,
    block_kv: int,
    block_q: int,
    mask: mask_lib.Mask | jax.Array,
    mask_value: float,
    cap: Optional[float] = None,
    save_residuals: bool = False,
    logits_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    loop_unroll: int | bool = True,
    fuse_logits: bool = True,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
  """Computes masked flash attention for short sequences (q_seq_len <= 4096)."""
  batch_size, num_q_heads, q_seq_len, qk_head_dim_size = q.shape
  _, num_kv_heads, kv_seq_len, _ = k.shape
  v_head_dim_size = v.shape[-1]
  data_type = q.dtype
  q_groups = num_q_heads // num_kv_heads
  q = q.reshape(
      (
          batch_size,
          num_kv_heads,
          q_groups,
          q_seq_len,
          qk_head_dim_size,
      )
  )

  # Calculate the number of key/value and query blocks.
  num_kv_blocks = kv_seq_len // block_kv
  num_q_blocks = q_seq_len // block_q

  mask_full, should_compute_block = _prepare_mask_and_block_predicate(
      mask=mask,
      batch_size=batch_size,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      block_q=block_q,
      block_kv=block_kv,
      segment_ids=segment_ids,
  )

  # Initialize `l` (logsumexp) and `m` (max_logits) for the online softmax.
  # `l` is initialized to 0 since no blocks have been processed yet and the sum
  # is 0.
  l = jnp.zeros((batch_size, num_kv_heads, q_groups, q_seq_len), dtype=data_type)
  # `m` is initialized to the mask_value so that the first block's maximum logit
  # correctly becomes the running maximum.
  m = jnp.full(
      (batch_size, num_kv_heads, q_groups, q_seq_len),
      mask_value,
      dtype=data_type,
  )

  output = jnp.zeros(
      (
          batch_size,
          num_kv_heads,
          q_groups,
          q_seq_len,
          v_head_dim_size,
      ),
      dtype=data_type,
  )

  # Outer loop over the key/value blocks.
  def outer_loop_body(j, carried):
    output, l, m = carried

    # Inner loop over the query blocks.
    def inner_loop_body(i, carried_inner):
      output, l, m = carried_inner

      # Calculates the attention computation (Q@K.T)@V with online softmax for
      # the current query and key/value blocks.
      def compute_attention_block(output, l, m):
        output_i_slice = jax.lax.dynamic_slice_in_dim(output, i * block_q, block_q, axis=-2)
        l_i_slice = jax.lax.dynamic_slice_in_dim(l, i * block_q, block_q, axis=-1)
        m_i_slice = jax.lax.dynamic_slice_in_dim(m, i * block_q, block_q, axis=-1)
        full_mask_i_j_slice = jax.lax.dynamic_slice(
            mask_full,
            (0, i * block_q, j * block_kv),
            (batch_size, block_q, block_kv),
        )
        broadcasted_mask = jnp.broadcast_to(
            full_mask_i_j_slice[:, None, None, :, :],
            (batch_size, num_kv_heads, q_groups, block_q, block_kv),
        )

        k_j_slice = jax.lax.dynamic_slice_in_dim(k, j * block_kv, block_kv, axis=-2)
        v_j_slice = jax.lax.dynamic_slice_in_dim(v, j * block_kv, block_kv, axis=-2)

        # let's get the slice of Q in N dimension
        q_slice = jax.lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=-2)

        s_i_j_dup = jnp.einsum(
            "bxhqc,bxkc->bxhqk",
            q_slice,
            k_j_slice,
            preferred_element_type=data_type,
        )
        if cap is not None:
          s_i_j_dup = jnp.tanh(s_i_j_dup / cap)
          s_i_j_dup = s_i_j_dup * cap
        s_i_j_dup = jnp.where(broadcasted_mask, s_i_j_dup, mask_value)
        m_i_j = s_i_j_dup.max(axis=-1)

        def fuse_this(q_slice, k_j_slice, mask_value, broadcasted_mask, m_i_j):
          s_i_j = jnp.einsum(
              "bxhqc,bxkc->bxhqk",
              q_slice,
              k_j_slice,
              preferred_element_type=logits_dtype,
          )
          if cap is not None:
            s_i_j = jnp.tanh(s_i_j / cap)
            s_i_j = s_i_j * cap
          s_i_j = jnp.where(broadcasted_mask, s_i_j, mask_value)
          p_i_j = jnp.exp(s_i_j - m_i_j[..., None])
          return p_i_j

        if fuse_logits:
          p_i_j = must_fuse_call("1")(fuse_this)(q_slice, k_j_slice, mask_value, broadcasted_mask, m_i_j)
        else:
          p_i_j = fuse_this(q_slice, k_j_slice, mask_value, broadcasted_mask, m_i_j)
        l_i_j = p_i_j.sum(axis=-1)
        assert m_i_j.shape == m_i_slice.shape
        m_i_new = jnp.maximum(m_i_slice, m_i_j)
        m_i_difference = jnp.exp(m_i_slice - m_i_new)
        m_i_j_difference = jnp.exp(m_i_j - m_i_new)
        l_i_new = m_i_difference * l_i_slice + m_i_j_difference * l_i_j

        divider = l_i_new[..., None]
        pv = jnp.einsum(
            "bxhqk,bxkc->bxhqc",
            p_i_j,
            v_j_slice,
            preferred_element_type=data_type,
        )
        # This forces the layout of the final @V to have better utilization by
        # avoiding using XLU:
        # Changes conv emitter type from
        # EmitAllInputFeatureInSublanesOutputBatchInSublanesXposeReuse to
        # EmitInputBatchInLanes
        pv = layout.with_layout_constraint(pv, DLL(major_to_minor=(0, 1, 2, 4, 3)))
        numerator = l_i_slice[..., None] * m_i_difference[..., None] * output_i_slice + m_i_j_difference[..., None] * pv

        output_i_slice_new = numerator / divider
        output = jax.lax.dynamic_update_index_in_dim(output, output_i_slice_new, i * block_q, axis=-2)
        l = jax.lax.dynamic_update_index_in_dim(l, l_i_new, i * block_q, axis=-1)
        m = jax.lax.dynamic_update_index_in_dim(m, m_i_new, i * block_q, axis=-1)
        return output, l, m

      def identity(output, l, m):
        """A no-op identity function."""

        return output, l, m

      output, l, m = jax.lax.cond(
          should_compute_block(i, j),
          compute_attention_block,
          identity,
          output,
          l,
          m,
      )

      return output, l, m

    output, l, m = jax.lax.fori_loop(0, num_q_blocks, inner_loop_body, (output, l, m), unroll=loop_unroll)

    return (output, l, m)

  output, l, m = jax.lax.fori_loop(0, num_kv_blocks, outer_loop_body, (output, l, m), unroll=loop_unroll)

  # Fold q_groups (dimension 2) back into the query heads dimension correctly
  # for both MHA (q_groups=1) and GQA (q_groups > 1).
  output = output.reshape((batch_size, num_q_heads, q_seq_len, v_head_dim_size))
  if not save_residuals:
    # To avoid remat of the output, we can use context=hbm remat policy as in
    # maxtext/configs/types.py
    return output

  l = l.reshape((batch_size, num_q_heads, q_seq_len))
  m = m.reshape((batch_size, num_q_heads, q_seq_len))
  return output, _compute_residual_stats(l, m)


def _flash_attention_block_masked_long_seq(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: SegmentIds | None,
    block_kv: int,
    block_q: int,
    mask: mask_lib.Mask | jax.Array,
    mask_value: float,
    cap: Optional[float] = None,
    save_residuals: bool = False,
    logits_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    loop_unroll: int | bool = True,
    fuse_logits: bool = True,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
  """Computes masked flash attention for long sequences (q_seq_len > 4096)."""
  batch_size, num_q_heads, q_seq_len, qk_head_dim_size = q.shape
  _, num_kv_heads, kv_seq_len, _ = k.shape
  v_head_dim_size = v.shape[-1]
  data_type = q.dtype
  q_groups = num_q_heads // num_kv_heads
  q = q.reshape(
      (
          batch_size,
          num_kv_heads,
          q_groups,
          q_seq_len,
          qk_head_dim_size,
      )
  )

  # Calculate the number of key/value and query blocks.
  num_kv_blocks = kv_seq_len // block_kv
  num_q_blocks = q_seq_len // block_q

  mask_full, should_compute_block = _prepare_mask_and_block_predicate(
      mask=mask,
      batch_size=batch_size,
      q_seq_len=q_seq_len,
      kv_seq_len=kv_seq_len,
      block_q=block_q,
      block_kv=block_kv,
      segment_ids=segment_ids,
  )

  l = jnp.zeros((batch_size, num_kv_heads, q_groups, q_seq_len), dtype=data_type)
  m = jnp.full(
      (batch_size, num_kv_heads, q_groups, q_seq_len),
      mask_value,
      dtype=data_type,
  )
  output = jnp.zeros(
      (
          batch_size,
          num_kv_heads,
          q_groups,
          q_seq_len,
          v_head_dim_size,
      ),
      dtype=data_type,
  )

  # Augment V with a column of ones before the loop so that l_i_j = p_i_j.sum(-1)
  # is computed as the last column of the P @ V MXU contraction rather than a
  # separate reduce instruction that cannot fuse with DUS. With this we get rid
  # of one time-consuming loop fusion on long sequence lengths.
  v_ones = jnp.ones(v.shape[:-1] + (1,), dtype=v.dtype)
  v_to_slice = jnp.concatenate([v, v_ones], axis=-1)

  # Outer loop over the key/value blocks.
  def outer_loop_body(j, carried):
    output, l, m = carried

    # Inner loop over the query blocks.
    def inner_loop_body(i, carried_inner):
      output, l, m = carried_inner

      def compute_attention_block(output, l, m):
        output_i_slice = jax.lax.dynamic_slice_in_dim(output, i * block_q, block_q, axis=-2)
        l_i_slice = jax.lax.dynamic_slice_in_dim(l, i * block_q, block_q, axis=-1)
        m_i_slice = jax.lax.dynamic_slice_in_dim(m, i * block_q, block_q, axis=-1)
        full_mask_i_j_slice = jax.lax.dynamic_slice(
            mask_full,
            (0, i * block_q, j * block_kv),
            (batch_size, block_q, block_kv),
        )
        broadcasted_mask = jnp.broadcast_to(
            full_mask_i_j_slice[:, None, None, :, :],
            (batch_size, num_kv_heads, q_groups, block_q, block_kv),
        )

        k_j_slice = jax.lax.dynamic_slice_in_dim(k, j * block_kv, block_kv, axis=-2)
        v_j_slice = jax.lax.dynamic_slice_in_dim(v_to_slice, j * block_kv, block_kv, axis=-2)
        q_i_slice = jax.lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=-2)

        def fusion_qk(q_i_slice, k_j_slice, broadcasted_mask):
          s_i_j = jnp.einsum(
              "bxhqc,bxkc->bxhqk",
              q_i_slice,
              k_j_slice,
              preferred_element_type=logits_dtype,
          )
          if cap is not None:
            s_i_j = jnp.tanh(s_i_j / cap)
            s_i_j = s_i_j * cap
          s_i_j = jnp.where(broadcasted_mask, s_i_j, mask_value)
          m_i_j = s_i_j.max(axis=-1)
          return s_i_j, m_i_j

        if fuse_logits:
          s_i_j, m_i_j = must_fuse_call("qk")(fusion_qk)(q_i_slice, k_j_slice, broadcasted_mask)
        else:
          s_i_j, m_i_j = fusion_qk(q_i_slice, k_j_slice, broadcasted_mask)
        assert m_i_j.shape == m_i_slice.shape

        def fusion_pv(
            s_i_j,
            v_j_slice,
            l_i_slice,
            m_i_slice,
            output_i_slice,
            m_i_j,
        ):
          p_i_j = jnp.exp(s_i_j - m_i_j[..., None])
          m_i_new = jnp.maximum(m_i_slice, m_i_j)
          m_i_difference = jnp.exp(m_i_slice - m_i_new)
          m_i_j_difference = jnp.exp(m_i_j - m_i_new)

          pv_aug = jnp.einsum(
              "bxhqk,bxkc->bxhqc",
              p_i_j,
              v_j_slice,
              preferred_element_type=data_type,
          )
          # This forces the layout of the final @V to have better utilization
          # by avoiding using XLU:
          # Changes conv emitter type from
          # EmitAllInputFeatureInSublanesOutputBatchInSublanesXposeReuse to
          # EmitInputBatchInLanes
          pv_aug = layout.with_layout_constraint(pv_aug, DLL(major_to_minor=(0, 1, 2, 4, 3)))
          pv = pv_aug[..., :-1]
          l_i_j = pv_aug[..., -1]

          l_i_new = m_i_difference * l_i_slice + m_i_j_difference * l_i_j
          output_i_slice_new = m_i_difference[..., None] * output_i_slice + m_i_j_difference[..., None] * pv
          return output_i_slice_new, l_i_new, m_i_new

        if fuse_logits:
          output_i_slice_new, l_i_new, m_i_new = must_fuse_call("pv")(fusion_pv)(
              s_i_j, v_j_slice, l_i_slice, m_i_slice, output_i_slice, m_i_j
          )
        else:
          output_i_slice_new, l_i_new, m_i_new = fusion_pv(s_i_j, v_j_slice, l_i_slice, m_i_slice, output_i_slice, m_i_j)

        output = jax.lax.dynamic_update_index_in_dim(output, output_i_slice_new, i * block_q, axis=-2)
        l = jax.lax.dynamic_update_index_in_dim(l, l_i_new, i * block_q, axis=-1)
        m = jax.lax.dynamic_update_index_in_dim(m, m_i_new, i * block_q, axis=-1)
        return output, l, m

      def identity(output, l, m):
        """A no-op identity function."""
        return output, l, m

      output, l, m = jax.lax.cond(
          should_compute_block(i, j),
          compute_attention_block,
          identity,
          output,
          l,
          m,
      )
      return output, l, m

    output, l, m = jax.lax.fori_loop(0, num_q_blocks, inner_loop_body, (output, l, m), unroll=loop_unroll)
    return (output, l, m)

  output, l, m = jax.lax.fori_loop(0, num_kv_blocks, outer_loop_body, (output, l, m), unroll=loop_unroll)

  # Fold q_groups (dimension 2) back into the query heads dimension correctly
  # for both MHA (q_groups=1) and GQA (q_groups > 1).
  output = output.reshape((batch_size, num_q_heads, q_seq_len, v_head_dim_size))
  l = l.reshape((batch_size, num_q_heads, q_seq_len))
  output = output / l[..., None]
  if not save_residuals:
    # To avoid remat of the output, we can use context=hbm remat policy as in
    # maxtext/configs/types.py
    return output

  m = m.reshape((batch_size, num_q_heads, q_seq_len))
  return output, _compute_residual_stats(l, m)


# This function computes masked flash attention using a block-sparse approach.
# This implementation keeps the full batch and number of heads dimensions
# throughout the attention computation while iterating through blocks of the
# key/value sequence and, within each, iterates through blocks of the query
# sequence. The `mask_blocked` is used to skip computations for blocks where all
# attention scores are masked out, improving efficiency for sparse masks.
def flash_attention_block_masked(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: SegmentIds | None,
    block_kv: int,
    block_q: int,
    mask: mask_lib.Mask | jax.Array,
    mask_value: float,
    cap: Optional[float] = None,
    save_residuals: bool = False,
    logits_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    loop_unroll: int | bool = True,
    fuse_logits: bool = True,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
  """Computes masked flash attention using block-sparse masking.

  Dispatches dynamically to the short-sequence implementation (q_seq_len <=
  4096)
  or the long-sequence implementation (q_seq_len > 4096).

  Args:
    q: Query tensor with shape (batch_size, num_kv_heads,
      num_q_heads_per_kv_head, q_seq_len, head_dim).
    k: Key tensor with shape (batch_size, num_kv_heads, kv_seq_len, head_dim).
    v: Value tensor with shape (batch_size, num_kv_heads, kv_seq_len,
      v_head_dim).
    segment_ids: SegmentIds are a mechanism to ensure that there is no
      cross-attention between segments (fraction of a sequence) that have been
      concatenated together into a sequence. Each array is a list of ids
      (integers). Only tokens with the same id are allowed to attend to each
      other. It stores the segment ids of the query and key/value sequences.
    block_kv: Block size for the key/value sequence dimension.
    block_q: Block size for the query sequence dimension.
    mask: The full attention mask. A rank-2 mask with shape (q_seq_len,
      kv_seq_len) is shared by every batch item. A rank-3 mask with shape
      (batch_size, q_seq_len, kv_seq_len) supplies an independent mask for each
      batch item.
    mask_value: The value to use for masked-out attention scores.
    cap: Optional cap for attention logits. This helps to prevent extremely
      large logits: capped_logits = jnp.tanh(logits / attn_logits_soft_cap) *
      attn_logits_soft_cap
    save_residuals: Whether to save residuals. If True, returns a tuple of
      (output, dict=(logsumexp, max_logits)). Both `logsumexp` and `max_logits`
      are of shape (batch_size, num_kv_heads, num_q_heads // num_kv_heads,
      q_seq_len).
    logits_dtype: Preferred element type for the fused query-key product that
      forms the attention probabilities; pass the query dtype when
      full-precision logits are required.
    loop_unroll: Unroll setting for the block loops; True fully unrolls, which
      is prohibitive for long sequences with a runtime mask.
    fuse_logits: Wrap the logits computation in a must-fuse XLA metadata call.
      The wrapper does not support reverse-mode differentiation inside a sharded
      computation; pass False when gradients are required.

  Returns:
    If save_residuals is True, returns a tuple containing:
      - The output of the attention computation.
      - A dict of (logsumexp, max_logits)
    Otherwise, returns the output of the attention computation.
  """
  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]

  if q_seq_len % block_q != 0:
    raise ValueError(f"q_seq_len {q_seq_len} must be divisible by block_q {block_q}")
  if kv_seq_len % block_kv != 0:
    raise ValueError(f"kv_seq_len {kv_seq_len} must be divisible by block_kv {block_kv}")

  if q_seq_len <= _LONG_SEQUENCE_THRESHOLD:
    return _flash_attention_block_masked_short_seq(
        q=q,
        k=k,
        v=v,
        segment_ids=segment_ids,
        block_kv=block_kv,
        block_q=block_q,
        mask=mask,
        mask_value=mask_value,
        cap=cap,
        save_residuals=save_residuals,
        logits_dtype=logits_dtype,
        loop_unroll=loop_unroll,
        fuse_logits=fuse_logits,
    )
  return _flash_attention_block_masked_long_seq(
      q=q,
      k=k,
      v=v,
      segment_ids=segment_ids,
      block_kv=block_kv,
      block_q=block_q,
      mask=mask,
      mask_value=mask_value,
      cap=cap,
      save_residuals=save_residuals,
      logits_dtype=logits_dtype,
      loop_unroll=loop_unroll,
      fuse_logits=fuse_logits,
  )


def _apply_mask_and_soft_cap_bwd(
    qk: jax.Array,
    mask: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    mask_value: float,
    attn_logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array]:
  """Applies mask and soft cap to qk logits."""
  qk_uncapped = qk
  if attn_logits_soft_cap is not None:
    qk = jnp.tanh(qk_uncapped / attn_logits_soft_cap)
    qk = qk * attn_logits_soft_cap

  if q_segment_ids is not None and kv_segment_ids is not None:
    seg_mask = q_segment_ids[..., :, None] == kv_segment_ids[..., None, :]
    if mask is not None:
      mask = jnp.logical_and(mask, seg_mask)
    else:
      mask = seg_mask

  if mask is not None:
    qk = jnp.where(mask, qk, mask_value)

  return qk, qk_uncapped


@functools.partial(
    jax.jit,
    static_argnames=[
        "block_q",
        "block_kv",
        "block_head",
        "dtype",
        "is_causal",
        "use_log2",
        "fuse",
        "unroll_loops",
    ],
)
def _flash_attention_mha_bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    do: jax.Array,
    logsumexp: jax.Array,
    segment_ids: SegmentIds | None = None,
    mask: jax.Array | None = None,
    mask_value: float = -1e30,
    attn_logits_soft_cap: float | None = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_head: int = 1,
    dtype: jax.typing.DTypeLike = jnp.float32,
    is_causal: bool = False,
    use_log2: bool = False,
    di: jax.Array | None = None,
    fuse: bool = True,
    unroll_loops: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Memory efficient JAX implementation of Flash Attention backward pass for MHA."""
  dp_dup_must_fuse = False
  dp_dedup_must_fuse = fuse
  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]

  if di is None:
    di_arr = jnp.sum(o * do, axis=-1)
  else:
    di_arr = di

  def should_compute_block(i, j):
    return not is_causal or (j + 1) * block_q > i * block_kv

  num_q_heads = q.shape[-3]
  eff_block_head = min(block_head, num_q_heads)
  if num_q_heads % eff_block_head != 0:
    raise ValueError(f"num_q_heads {num_q_heads} must be divisible by block_head" f" {block_head}")
  if q_seq_len % block_q != 0:
    raise ValueError(f"q_seq_len {q_seq_len} must be divisible by block_q {block_q}")
  if kv_seq_len % block_kv != 0:
    raise ValueError(f"kv_seq_len {kv_seq_len} must be divisible by block_kv {block_kv}")

  slice_sizes_k = k.shape[:-3] + (eff_block_head, block_kv, k.shape[-1])
  slice_sizes_v = v.shape[:-3] + (eff_block_head, block_kv, v.shape[-1])
  slice_sizes_q = q.shape[:-3] + (eff_block_head, block_q, q.shape[-1])
  slice_sizes_do = do.shape[:-3] + (eff_block_head, block_q, do.shape[-1])
  slice_sizes_log = logsumexp.shape[:-2] + (eff_block_head, block_q)

  if mask is None:
    mask_arr = None
  elif isinstance(mask, mask_lib.Mask):
    mask_arr = mask[:, :]
  else:
    mask_arr = mask

  def head_kernel(head_idx, head_carry):
    head_start = head_idx * eff_block_head

    def k_block_kernel(i, carry):
      dq_carry, dk_carry, dv_carry = carry
      kv_slice_start = i * block_kv

      start_indices_k = (0,) * (k.ndim - 3) + (head_start, kv_slice_start, 0)
      start_indices_v = (0,) * (v.ndim - 3) + (head_start, kv_slice_start, 0)

      k_block = lax.dynamic_slice(k, start_indices_k, slice_sizes_k)
      v_block = lax.dynamic_slice(v, start_indices_v, slice_sizes_v)
      kv_segment_ids_block = (
          lax.dynamic_slice_in_dim(segment_ids.kv, kv_slice_start, block_kv, axis=-1) if segment_ids is not None else None
      )

      k_block_prec = k_block.astype(dtype)
      v_block_prec = v_block.astype(dtype)

      dk_block = jnp.zeros_like(k_block_prec)
      dv_block = jnp.zeros_like(v_block_prec)

      def q_block_kernel(j, inner_carry):
        dk_block_carry, dv_block_carry, dq_carry_inner = inner_carry
        q_slice_start = j * block_q

        start_indices_q = (0,) * (q.ndim - 3) + (head_start, q_slice_start, 0)
        start_indices_do = (0,) * (do.ndim - 3) + (head_start, q_slice_start, 0)
        start_indices_log = (0,) * (logsumexp.ndim - 2) + (
            head_start,
            q_slice_start,
        )

        q_block = lax.dynamic_slice(q, start_indices_q, slice_sizes_q)
        do_block = lax.dynamic_slice(do, start_indices_do, slice_sizes_do)
        logsumexp_block = lax.dynamic_slice(logsumexp, start_indices_log, slice_sizes_log).astype(dtype)
        di_block = lax.dynamic_slice(di_arr, start_indices_log, slice_sizes_log)
        q_segment_ids_block = (
            lax.dynamic_slice_in_dim(segment_ids.q, q_slice_start, block_q, axis=-1) if segment_ids is not None else None
        )

        if mask_arr is None:
          mask_block_qkv = None
        elif mask_arr.ndim == 2:
          mask_block_qkv = lax.dynamic_slice(
              mask_arr,
              (q_slice_start, kv_slice_start),
              (block_q, block_kv),
          )
        elif mask_arr.ndim == 3:
          if mask_arr.shape[0] == q.shape[0] and q.ndim == 4 and mask_arr.shape[0] != num_q_heads:
            mask_slice = lax.dynamic_slice(
                mask_arr,
                (0, q_slice_start, kv_slice_start),
                (mask_arr.shape[0], block_q, block_kv),
            )
            mask_block_qkv = mask_slice[:, None, :, :]
          else:
            mask_block_qkv = lax.dynamic_slice(
                mask_arr,
                (head_start, q_slice_start, kv_slice_start),
                (eff_block_head, block_q, block_kv),
            )
        else:
          mask_block_qkv = lax.dynamic_slice(
              mask_arr,
              (0, head_start, q_slice_start, kv_slice_start),
              (mask_arr.shape[0], eff_block_head, block_q, block_kv),
          )

        q_block_prec = q_block.astype(dtype)
        do_block_prec = do_block.astype(dtype)

        def compute_updates():
          qk = jnp.einsum("...hd,...kd->...hk", q_block_prec, k_block_prec)
          qk_masked, qk_uncapped = _apply_mask_and_soft_cap_bwd(
              qk,
              mask_block_qkv,
              q_segment_ids_block,
              kv_segment_ids_block,
              mask_value,
              attn_logits_soft_cap,
          )

          if use_log2:
            p = 2 ** (qk_masked - logsumexp_block[..., None])
          else:
            p = jnp.exp(qk_masked - logsumexp_block[..., None])

          dv_block_update = jnp.einsum("...hk,...hd->...kd", p, do_block_prec)

          def fuse_dk_compute(
              do_block_prec,
              v_block_prec,
              di_block,
              p,
              attn_logits_soft_cap,
              q_block_prec,
          ):
            dp = jnp.einsum("...hd,...kd->...hk", do_block_prec, v_block_prec)
            ds = p * (dp - di_block[..., None])
            if attn_logits_soft_cap is not None:
              normalized = qk_uncapped / attn_logits_soft_cap
              d = jnp.tanh(normalized)
              g = ds * (1 - d)
              ds = g + g * d
            dk_block_update = jnp.einsum("...hk,...hd->...kd", ds, q_block_prec)
            return dk_block_update

          def fuse_dq_compute(
              do_block_prec,
              v_block_prec,
              di_block,
              p,
              attn_logits_soft_cap,
              k_block_prec,
          ):
            dp = jnp.einsum("...hd,...kd->...hk", do_block_prec, v_block_prec)
            ds = p * (dp - di_block[..., None])
            if attn_logits_soft_cap is not None:
              normalized = qk_uncapped / attn_logits_soft_cap
              d = jnp.tanh(normalized)
              g = ds * (1 - d)
              ds = g + g * d
            dq_block_update = jnp.einsum("...hk,...kd->...hd", ds, k_block_prec)
            return dq_block_update

          def fuse_dq_dk_compute(
              do_block_prec,
              v_block_prec,
              di_block,
              p,
              attn_logits_soft_cap,
              q_block_prec,
              k_block_prec,
          ):
            dp = jnp.einsum("...hd,...kd->...hk", do_block_prec, v_block_prec)
            ds = p * (dp - di_block[..., None])
            if attn_logits_soft_cap is not None:
              normalized = qk_uncapped / attn_logits_soft_cap
              d = jnp.tanh(normalized)
              g = ds * (1 - d)
              ds = g + g * d
            dk_block_update = jnp.einsum("...hk,...hd->...kd", ds, q_block_prec)
            dq_block_update = jnp.einsum("...hk,...kd->...hd", ds, k_block_prec)
            return dk_block_update, dq_block_update

          if dp_dup_must_fuse:
            dk_block_update = must_fuse_call("1")(fuse_dk_compute)(
                do_block_prec,
                v_block_prec,
                di_block,
                p,
                attn_logits_soft_cap,
                q_block_prec,
            )
            dq_block = must_fuse_call("2")(fuse_dq_compute)(
                do_block_prec,
                v_block_prec,
                di_block,
                p,
                attn_logits_soft_cap,
                k_block_prec,
            )
          elif dp_dedup_must_fuse:
            dk_block_update, dq_block = must_fuse_call("3")(fuse_dq_dk_compute)(
                do_block_prec,
                v_block_prec,
                di_block,
                p,
                attn_logits_soft_cap,
                q_block_prec,
                k_block_prec,
            )
          else:
            dp = jnp.einsum("...hd,...kd->...hk", do_block_prec, v_block_prec)
            ds = (dp - di_block[..., None]) * p
            if attn_logits_soft_cap is not None:
              normalized = qk_uncapped / attn_logits_soft_cap
              d = jnp.tanh(normalized)
              g = ds * (1 - d)
              ds = g + g * d
            dk_block_update = jnp.einsum("...hk,...hd->...kd", ds, q_block_prec)
            dq_block = jnp.einsum("...hk,...kd->...hd", ds, k_block_prec)

          return dk_block_update, dv_block_update, dq_block

        def skip_updates():
          return (
              jnp.zeros_like(k_block_prec),
              jnp.zeros_like(v_block_prec),
              jnp.zeros_like(q_block_prec),
          )

        if is_causal:
          dk_block_update, dv_block_update, dq_block = lax.cond(
              should_compute_block(i, j),
              compute_updates,
              skip_updates,
          )
        else:
          dk_block_update, dv_block_update, dq_block = compute_updates()

        dk_block_carry = dk_block_carry + dk_block_update
        dv_block_carry = dv_block_carry + dv_block_update
        dq_carry_inner = lax.dynamic_update_slice(
            dq_carry_inner,
            dq_block + lax.dynamic_slice(dq_carry_inner, start_indices_q, slice_sizes_q),
            start_indices_q,
        )
        return dk_block_carry, dv_block_carry, dq_carry_inner

      dk_block, dv_block, dq_carry = lax.fori_loop(
          0,
          q_seq_len // block_q,
          q_block_kernel,
          (dk_block, dv_block, dq_carry),
          unroll=unroll_loops,
      )

      dk_carry = lax.dynamic_update_slice(dk_carry, dk_block, start_indices_k)
      dv_carry = lax.dynamic_update_slice(dv_carry, dv_block, start_indices_v)
      return dq_carry, dk_carry, dv_carry

    num_kv_blocks = kv_seq_len // block_kv
    dq_out, dk_out, dv_out = lax.fori_loop(
        0,
        num_kv_blocks,
        k_block_kernel,
        head_carry,
        unroll=unroll_loops,
    )
    return dq_out, dk_out, dv_out

  dk = jnp.zeros_like(k, dtype=dtype)
  dv = jnp.zeros_like(v, dtype=dtype)
  dq = jnp.zeros_like(q, dtype=dtype)
  num_head_blocks = num_q_heads // eff_block_head
  dq, dk, dv = lax.fori_loop(
      0,
      num_head_blocks,
      head_kernel,
      (dq, dk, dv),
      unroll=unroll_loops,
  )
  return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


def flash_attention_mha_bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    do: jax.Array,
    logsumexp: jax.Array,
    segment_ids: SegmentIds | None = None,
    mask: jax.Array | mask_lib.Mask | None = None,
    mask_value: float = -1e30,
    attn_logits_soft_cap: float | None = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_head: int = 1,
    dtype: jax.typing.DTypeLike = jnp.float32,
    is_causal: bool = False,
    use_log2: bool = False,
    di: jax.Array | None = None,
    fuse: bool = True,
    unroll_loops: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Memory efficient JAX implementation of Flash Attention backward pass for MHA."""
  if mask is not None:
    mask_arr = jnp.asarray(mask[:, :]) if isinstance(mask, mask_lib.Mask) else jnp.asarray(mask)
  else:
    mask_arr = None
  return _flash_attention_mha_bwd_jax(
      q=q,
      k=k,
      v=v,
      o=o,
      do=do,
      logsumexp=logsumexp,
      segment_ids=segment_ids,
      mask=mask_arr,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
      block_q=block_q,
      block_kv=block_kv,
      block_head=block_head,
      dtype=dtype,
      is_causal=is_causal,
      use_log2=use_log2,
      di=di,
      fuse=fuse,
      unroll_loops=unroll_loops,
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        "block_q",
        "block_kv",
        "block_q_head",
        "block_kv_head",
        "dtype",
        "is_causal",
        "use_log2",
        "fuse",
    ],
)
def _flash_attention_gqa_bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    do: jax.Array,
    logsumexp: jax.Array,
    segment_ids: SegmentIds | None = None,
    mask: jax.Array | None = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: int = 1,
    block_kv_head: int = 1,
    mask_value: float = -1e30,
    attn_logits_soft_cap: float | None = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
    is_causal: bool = False,
    use_log2: bool = False,
    di: jax.Array | None = None,
    fuse: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Memory efficient JAX implementation of Flash Attention backward pass for GQA."""
  dp_dup_must_fuse = fuse
  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]
  num_q_heads = q.shape[-3]
  num_kv_heads = k.shape[-3]
  head_multiplier = num_q_heads // num_kv_heads

  eff_block_kv_head = min(block_kv_head, num_kv_heads)
  eff_block_q_head = min(block_q_head, head_multiplier)
  num_kv_head_blocks = num_kv_heads // eff_block_kv_head
  num_q_head_blocks = head_multiplier // eff_block_q_head

  if q_seq_len % block_q != 0:
    raise ValueError(f"q_seq_len {q_seq_len} must be divisible by block_q {block_q}")
  if kv_seq_len % block_kv != 0:
    raise ValueError(f"kv_seq_len {kv_seq_len} must be divisible by block_kv {block_kv}")
  if num_kv_heads % eff_block_kv_head != 0:
    raise ValueError(f"num_kv_heads {num_kv_heads} must be divisible by block_kv_head" f" {block_kv_head}")
  if head_multiplier % eff_block_q_head != 0:
    raise ValueError(f"head_multiplier {head_multiplier} must be divisible by block_q_head" f" {block_q_head}")

  if mask is None:
    mask_arr = None
  elif isinstance(mask, mask_lib.Mask):
    mask_arr = mask[:, :]
  else:
    mask_arr = mask

  if mask_arr is not None:
    if mask_arr.ndim == 2:
      mask_arr = jnp.broadcast_to(mask_arr[None, :, :], (num_q_heads, *mask_arr.shape))
    elif mask_arr.ndim == 3 and mask_arr.shape[0] != num_q_heads:
      mask_arr = jnp.broadcast_to(mask_arr, (num_q_heads, *mask_arr.shape[1:]))

  q_gqa = q.reshape(q.shape[:-3] + (num_kv_heads, head_multiplier, q_seq_len, q.shape[-1]))
  do_gqa = do.reshape(do.shape[:-3] + (num_kv_heads, head_multiplier, q_seq_len, do.shape[-1]))
  logsumexp_gqa = logsumexp.reshape(logsumexp.shape[:-2] + (num_kv_heads, head_multiplier, q_seq_len))

  di_arr = jnp.sum(o * do, axis=-1, dtype=dtype) if di is None else di
  di_gqa = di_arr.reshape(di_arr.shape[:-2] + (num_kv_heads, head_multiplier, q_seq_len))

  def should_compute_block(i, j):
    return not is_causal or (j + 1) * block_q > i * block_kv

  slice_sizes_k = k.shape[:-3] + (eff_block_kv_head, block_kv, k.shape[-1])
  slice_sizes_v = v.shape[:-3] + (eff_block_kv_head, block_kv, v.shape[-1])
  slice_sizes_q = q_gqa.shape[:-4] + (
      eff_block_kv_head,
      eff_block_q_head,
      block_q,
      q_gqa.shape[-1],
  )
  slice_sizes_do = do_gqa.shape[:-4] + (
      eff_block_kv_head,
      eff_block_q_head,
      block_q,
      do_gqa.shape[-1],
  )
  slice_sizes_log = logsumexp_gqa.shape[:-3] + (
      eff_block_kv_head,
      eff_block_q_head,
      block_q,
  )

  def kv_head_kernel(kv_head_idx, kv_head_carry):
    dk_kv, dv_kv, dq_kv = kv_head_carry
    kv_head_start = kv_head_idx * eff_block_kv_head

    def q_head_kernel(q_head_idx, q_head_carry):
      dk_q, dv_q, dq_q = q_head_carry
      q_head_start = q_head_idx * eff_block_q_head

      def k_block_kernel(i, k_carry):
        dk_k, dv_k, dq_k = k_carry
        kv_slice_start = i * block_kv

        start_indices_k = (0,) * (k.ndim - 3) + (
            kv_head_start,
            kv_slice_start,
            0,
        )
        start_indices_v = (0,) * (v.ndim - 3) + (
            kv_head_start,
            kv_slice_start,
            0,
        )

        k_block = lax.dynamic_slice(k, start_indices_k, slice_sizes_k)
        v_block = lax.dynamic_slice(v, start_indices_v, slice_sizes_v)
        kv_segment_ids_block = (
            lax.dynamic_slice_in_dim(segment_ids.kv, kv_slice_start, block_kv, axis=-1)
            if segment_ids is not None
            else None
        )

        k_block_prec = k_block.astype(dtype)
        v_block_prec = v_block.astype(dtype)

        dk_block = jnp.zeros_like(k_block_prec)
        dv_block = jnp.zeros_like(v_block_prec)

        def q_block_kernel(j, inner_carry):
          dk_inner, dv_inner, dq_inner = inner_carry
          q_slice_start = j * block_q

          start_indices_q = (0,) * (q_gqa.ndim - 4) + (
              kv_head_start,
              q_head_start,
              q_slice_start,
              0,
          )
          start_indices_do = (0,) * (do_gqa.ndim - 4) + (
              kv_head_start,
              q_head_start,
              q_slice_start,
              0,
          )
          start_indices_log = (0,) * (logsumexp_gqa.ndim - 3) + (
              kv_head_start,
              q_head_start,
              q_slice_start,
          )

          q_block = lax.dynamic_slice(q_gqa, start_indices_q, slice_sizes_q)
          do_block = lax.dynamic_slice(do_gqa, start_indices_do, slice_sizes_do)
          logsumexp_block = lax.dynamic_slice(logsumexp_gqa, start_indices_log, slice_sizes_log).astype(dtype)
          di_block = lax.dynamic_slice(di_gqa, start_indices_log, slice_sizes_log)

          q_segment_ids_block = (
              lax.dynamic_slice_in_dim(segment_ids.q, q_slice_start, block_q, axis=-1)
              if segment_ids is not None
              else None
          )

          if mask_arr is None:
            mask_block_qkv = None
          else:
            mask_block_qkv = lax.dynamic_slice(
                mask_arr,
                (
                    kv_head_start * head_multiplier + q_head_start,
                    q_slice_start,
                    kv_slice_start,
                ),
                (eff_block_kv_head * eff_block_q_head, block_q, block_kv),
            )
            mask_block_qkv = mask_block_qkv.reshape(eff_block_kv_head, eff_block_q_head, block_q, block_kv)

          q_block_prec = q_block.astype(dtype)
          do_block_prec = do_block.astype(dtype)

          def compute_updates():
            qk = jnp.einsum("...hqmd,...hkd->...hqmk", q_block_prec, k_block_prec)
            qk_masked, qk_uncapped = _apply_mask_and_soft_cap_bwd(
                qk,
                mask_block_qkv,
                q_segment_ids_block,
                kv_segment_ids_block,
                mask_value,
                attn_logits_soft_cap,
            )

            if use_log2:
              p = 2 ** (qk_masked - logsumexp_block[..., None])
            else:
              p = jnp.exp(qk_masked - logsumexp_block[..., None])

            dv_block_update = jnp.einsum("...hqmk,...hqmd->...hkd", p, do_block_prec)

            def fuse_dk_compute(
                do_block_prec,
                v_block_prec,
                di_block,
                p,
                attn_logits_soft_cap,
                q_block_prec,
            ):
              dp = jnp.einsum("...hqmd,...hkd->...hqmk", do_block_prec, v_block_prec)
              ds = p * (dp - di_block[..., None])
              if attn_logits_soft_cap is not None:
                normalized = qk_uncapped / attn_logits_soft_cap
                d = jnp.tanh(normalized)
                g = ds * (1 - d)
                ds = g + g * d

              dk_block_update = jnp.einsum("...hqmk,...hqmd->...hkd", ds, q_block_prec)
              return dk_block_update

            def fuse_dq_compute(
                do_block_prec,
                v_block_prec,
                di_block,
                p,
                attn_logits_soft_cap,
                k_block_prec,
            ):
              dp = jnp.einsum("...hqmd,...hkd->...hqmk", do_block_prec, v_block_prec)
              ds = p * (dp - di_block[..., None])
              if attn_logits_soft_cap is not None:
                normalized = qk_uncapped / attn_logits_soft_cap
                d = jnp.tanh(normalized)
                g = ds * (1 - d)
                ds = g + g * d

              dq_block_update = jnp.einsum("...hqmk,...hkd->...hqmd", ds, k_block_prec)
              return dq_block_update

            if dp_dup_must_fuse:
              dk_block_update = must_fuse_call("1")(fuse_dk_compute)(
                  do_block_prec,
                  v_block_prec,
                  di_block,
                  p,
                  attn_logits_soft_cap,
                  q_block_prec,
              )
              dq_block = must_fuse_call("2")(fuse_dq_compute)(
                  do_block_prec,
                  v_block_prec,
                  di_block,
                  p,
                  attn_logits_soft_cap,
                  k_block_prec,
              )
            else:
              dp = jnp.einsum("...hqmd,...hkd->...hqmk", do_block_prec, v_block_prec)
              ds = p * (dp - di_block[..., None])
              if attn_logits_soft_cap is not None:
                normalized = qk_uncapped / attn_logits_soft_cap
                d = jnp.tanh(normalized)
                g = ds * (1 - d)
                ds = g + g * d
              dk_block_update = jnp.einsum("...hqmk,...hqmd->...hkd", ds, q_block_prec)
              dq_block = jnp.einsum("...hqmk,...hkd->...hqmd", ds, k_block_prec)

            return dk_block_update, dv_block_update, dq_block

          def skip_updates():
            return (
                jnp.zeros_like(k_block_prec),
                jnp.zeros_like(v_block_prec),
                jnp.zeros_like(q_block_prec),
            )

          if is_causal:
            dk_block_update, dv_block_update, dq_block = lax.cond(
                should_compute_block(i, j),
                compute_updates,
                skip_updates,
            )
          else:
            dk_block_update, dv_block_update, dq_block = compute_updates()

          dk_inner = dk_inner + dk_block_update
          dv_inner = dv_inner + dv_block_update
          dq_inner = lax.dynamic_update_slice(
              dq_inner,
              dq_block + lax.dynamic_slice(dq_inner, start_indices_q, slice_sizes_q),
              start_indices_q,
          )
          return dk_inner, dv_inner, dq_inner

        dk_block, dv_block, dq_k = lax.fori_loop(
            0,
            q_seq_len // block_q,
            q_block_kernel,
            (dk_block, dv_block, dq_k),
        )

        dk_k = lax.dynamic_update_slice(
            dk_k,
            dk_block + lax.dynamic_slice(dk_k, start_indices_k, slice_sizes_k),
            start_indices_k,
        )
        dv_k = lax.dynamic_update_slice(
            dv_k,
            dv_block + lax.dynamic_slice(dv_k, start_indices_v, slice_sizes_v),
            start_indices_v,
        )
        return dk_k, dv_k, dq_k

      num_kv_blocks = kv_seq_len // block_kv
      dk_q, dv_q, dq_q = lax.fori_loop(
          0,
          num_kv_blocks,
          k_block_kernel,
          (dk_q, dv_q, dq_q),
      )
      return dk_q, dv_q, dq_q

    dk_kv, dv_kv, dq_kv = lax.fori_loop(
        0,
        num_q_head_blocks,
        q_head_kernel,
        (dk_kv, dv_kv, dq_kv),
    )
    return dk_kv, dv_kv, dq_kv

  dk = jnp.zeros_like(k, dtype=dtype)
  dv = jnp.zeros_like(v, dtype=dtype)
  dq_gqa = jnp.zeros_like(q_gqa, dtype=dtype)

  dk, dv, dq_gqa = lax.fori_loop(
      0,
      num_kv_head_blocks,
      kv_head_kernel,
      (dk, dv, dq_gqa),
  )

  dq = dq_gqa.reshape(q.shape)
  return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


def flash_attention_gqa_bwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    do: jax.Array,
    logsumexp: jax.Array,
    segment_ids: SegmentIds | None = None,
    mask: jax.Array | mask_lib.Mask | None = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: int = 1,
    block_kv_head: int = 1,
    mask_value: float = -1e30,
    attn_logits_soft_cap: float | None = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
    is_causal: bool = False,
    use_log2: bool = False,
    di: jax.Array | None = None,
    fuse: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Memory efficient JAX implementation of Flash Attention backward pass for GQA."""
  if mask is not None:
    mask_arr = jnp.asarray(mask[:, :]) if isinstance(mask, mask_lib.Mask) else jnp.asarray(mask)
  else:
    mask_arr = None
  return _flash_attention_gqa_bwd_jax(
      q=q,
      k=k,
      v=v,
      o=o,
      do=do,
      logsumexp=logsumexp,
      segment_ids=segment_ids,
      mask=mask_arr,
      block_q=block_q,
      block_kv=block_kv,
      block_q_head=block_q_head,
      block_kv_head=block_kv_head,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
      dtype=dtype,
      is_causal=is_causal,
      use_log2=use_log2,
      di=di,
      fuse=fuse,
  )


def flash_attention_bwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    o: jnp.ndarray,
    do: jnp.ndarray,
    logsumexp: jnp.ndarray,
    mask: jnp.ndarray | mask_lib.Mask | None = None,
    segment_ids: SegmentIds | None = None,
    mask_value: float = -1e30,
    cap: Optional[float] = None,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: Optional[int] = None,
    block_kv_head: Optional[int] = None,
    block_head: Optional[int] = None,
    dtype: jax.typing.DTypeLike | None = None,
    is_causal: bool = False,
    use_log2: bool = False,
    di: Optional[jnp.ndarray] = None,
    fuse: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Computes backward pass for Flash Attention supporting MHA, GQA, and MQA."""
  if mask is not None:
    mask_arr = jnp.asarray(mask[:, :]) if isinstance(mask, mask_lib.Mask) else jnp.asarray(mask)
  else:
    mask_arr = None
  q_is_3d = q.ndim == 3
  k_was_2d = False
  k_was_3d_in_4d = False

  if q_is_3d:
    q = jnp.expand_dims(q, axis=0)
    o = jnp.expand_dims(o, axis=0)
    do = jnp.expand_dims(do, axis=0)
    logsumexp = jnp.expand_dims(logsumexp, axis=0)
    if di is not None:
      di = jnp.expand_dims(di, axis=0)
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
    if k.ndim == 3:
      k_was_3d_in_4d = True
      k = jnp.expand_dims(k, axis=1)
      v = jnp.expand_dims(v, axis=1)

  num_q_heads = q.shape[-3]
  num_kv_heads = k.shape[-3]
  is_mqa = num_kv_heads == 1 and num_q_heads > 1
  use_gqa = (num_kv_heads < num_q_heads) or is_mqa

  comp_dtype = dtype if dtype is not None else q.dtype
  eff_block_q = min(block_q, q.shape[-2])
  eff_block_kv = min(block_kv, k.shape[-2])

  if use_gqa:
    head_multiplier = num_q_heads // num_kv_heads
    eff_block_kv_head = min(block_kv_head, num_kv_heads) if block_kv_head is not None else num_kv_heads
    eff_block_q_head = min(block_q_head, head_multiplier) if block_q_head is not None else head_multiplier

    dq, dk, dv = flash_attention_gqa_bwd_jax(
        q=q,
        k=k,
        v=v,
        o=o,
        do=do,
        logsumexp=logsumexp,
        segment_ids=segment_ids,
        mask=mask_arr,
        block_q=eff_block_q,
        block_kv=eff_block_kv,
        block_q_head=eff_block_q_head,
        block_kv_head=eff_block_kv_head,
        mask_value=mask_value,
        attn_logits_soft_cap=cap,
        dtype=comp_dtype,
        is_causal=is_causal,
        use_log2=use_log2,
        di=di,
        fuse=fuse,
    )
  else:
    eff_block_head = (
        min(block_head, num_q_heads)
        if block_head is not None
        else (
            min(block_q_head, num_q_heads)
            if block_q_head is not None
            else (min(block_kv_head, num_q_heads) if block_kv_head is not None else num_q_heads)
        )
    )
    dq, dk, dv = flash_attention_mha_bwd_jax(
        q=q,
        k=k,
        v=v,
        o=o,
        do=do,
        logsumexp=logsumexp,
        segment_ids=segment_ids,
        mask=mask_arr,
        mask_value=mask_value,
        attn_logits_soft_cap=cap,
        block_q=eff_block_q,
        block_kv=eff_block_kv,
        block_head=eff_block_head,
        dtype=comp_dtype,
        is_causal=is_causal,
        use_log2=use_log2,
        di=di,
        fuse=fuse,
    )

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


splash_attention_mha_bwd_jax = flash_attention_mha_bwd_jax
splash_attention_gqa_bwd_jax = flash_attention_gqa_bwd_jax
flash_attention_block_masked_bwd = flash_attention_bwd
