#  Copyright 2025 Google LLC
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

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from MaxText.kernels import splash_attention_kernel

SegmentIds = splash_attention_kernel.SegmentIds


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
    mask: jnp.ndarray,
    mask_value: float,
    cap: Optional[float] = None,
    save_residuals: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]]:
  """Computes masked flash attention using block-sparse masking.

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
    mask: The full attention mask with shape of (q_seq_len, kv_seq_len). This
      mask will be used for all batches.
    mask_value: The value to use for masked-out attention scores.
    cap: Optional cap for attention logits. This helps to prevent extremely
      large logits: capped_logits = jnp.tanh(logits / attn_logits_soft_cap) *
      attn_logits_soft_cap
    save_residuals: Whether to save residuals. If True, returns a tuple of
      (output, dict=(logsumexp, max_logits)). Both `logsumexp` and `max_logits`
      are of shape (batch_size, num_kv_heads, num_q_heads // num_kv_heads,
      q_seq_len).

  Returns:
    If save_residuals is True, returns a tuple containing:

    * The output of the attention computation.
    * A dict of (logsumexp, max_logits)

    Otherwise, returns the output of the attention computation.
  """
  batch_size, num_q_heads, q_seq_len, qk_head_dim_size = q.shape
  _, num_kv_heads, kv_seq_len, _ = k.shape
  v_head_dim_size = v.shape[-1]
  data_type = q.dtype
  q_groups = num_q_heads // num_kv_heads
  q = q.reshape((
      batch_size,
      num_kv_heads,
      q_groups,
      q_seq_len,
      qk_head_dim_size,
  ))

  # Calculate the number of key/value and query blocks.
  num_kv_blocks = kv_seq_len // block_kv
  num_q_blocks = q_seq_len // block_q

  # Before applying the segment mask, we need to broadcast the mask in batch
  # dimension since we have same logic for all batches.
  mask_full = jnp.broadcast_to(
      mask[None, :, :], (batch_size, q_seq_len, kv_seq_len)
  )

  if segment_ids is not None:
    segment_ids_q = segment_ids.q[:, :, None]
    segment_ids_kv = segment_ids.kv[:, None, :]
    mask_full = jnp.logical_and(mask_full, segment_ids_q == segment_ids_kv)
  mask_blocked = jax.jit(mask_blocker, static_argnums=[1, 2])(
      mask_full, block_q, block_kv
  )

  # Initialize `l` (logsumexp) and `m` (max_logits) for the online softmax.
  # `l` is initialized to 0 since no blocks have been processed yet and the sum
  # is 0.
  l = jnp.zeros(
      (batch_size, num_kv_heads, q_groups, q_seq_len), dtype=jnp.float32
  )
  # `m` is initialized to the mask_value so that the first block's maximum logit
  # correctly becomes the running maximum.
  m = jnp.full(
      (batch_size, num_kv_heads, q_groups, q_seq_len),
      mask_value,
      dtype=jnp.float32,
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
    k_j_slice = jax.lax.dynamic_slice_in_dim(k, j * block_kv, block_kv, axis=-2)
    v_j_slice = jax.lax.dynamic_slice_in_dim(v, j * block_kv, block_kv, axis=-2)

    # Inner loop over the query blocks.
    def inner_loop_body(i, carried_inner):
      output, l, m = carried_inner

      # Calculates the attention computation (Q@K.T)@V with online softmax for
      # the current query and key/value blocks.
      def compute_attention_block(output, l, m):
        # let's get the slice of Q in N dimension
        q_slice = jax.lax.dynamic_slice_in_dim(q, i * block_q, block_q, axis=-2)
        output_i_slice = jax.lax.dynamic_slice_in_dim(
            output, i * block_q, block_q, axis=-2
        )
        l_i_slice = jax.lax.dynamic_slice_in_dim(
            l, i * block_q, block_q, axis=-1
        )
        m_i_slice = jax.lax.dynamic_slice_in_dim(
            m, i * block_q, block_q, axis=-1
        )
        s_i_j = jnp.einsum(
            "bxhqc,bxkc->bxhqk",
            q_slice,
            k_j_slice,
            preferred_element_type=jnp.float32,
        )
        full_mask_i_j_slice = jax.lax.dynamic_slice(
            mask_full,
            (0, i * block_q, j * block_kv),
            (batch_size, block_q, block_kv),
        )
        broadcasted_mask = jnp.broadcast_to(
            full_mask_i_j_slice[:, None, None, :, :],
            (batch_size, num_kv_heads, q_groups, block_q, block_kv),
        )

        s_i_j = jnp.where(broadcasted_mask, s_i_j, mask_value)
        if cap is not None:
          s_i_j = jnp.tanh(s_i_j / cap)
          s_i_j = s_i_j * cap
        m_i_j = s_i_j.max(axis=-1)
        p_i_j = jnp.exp(s_i_j - m_i_j[..., None])
        l_i_j = p_i_j.sum(axis=-1)
        assert m_i_j.shape == m_i_slice.shape
        m_i_new = jnp.maximum(m_i_slice, m_i_j)
        m_i_difference = jnp.exp(m_i_slice - m_i_new)
        m_i_j_difference = jnp.exp(m_i_j - m_i_new)
        l_i_new = m_i_difference * l_i_slice + m_i_j_difference * l_i_j

        divider = l_i_new[..., None]
        numerator = l_i_slice[..., None] * m_i_difference[
            ..., None
        ] * output_i_slice + m_i_j_difference[..., None] * jnp.einsum(
            "bxhqk,bxkc->bxhqc",
            p_i_j,
            v_j_slice,
            preferred_element_type=data_type,
        )

        output_i_slice_new = numerator / divider
        output = jax.lax.dynamic_update_index_in_dim(
            output, output_i_slice_new.astype(data_type), i * block_q, axis=-2
        )
        l = jax.lax.dynamic_update_index_in_dim(
            l, l_i_new, i * block_q, axis=-1
        )
        m = jax.lax.dynamic_update_index_in_dim(
            m, m_i_new, i * block_q, axis=-1
        )
        return output, l, m

      def identity(output, l, m):
        """A no-op identity function."""

        return output, l, m

      batch_size = mask_blocked.shape[0]
      mask_i_j_slice = jax.lax.dynamic_slice(
          mask_blocked, (0, i, j), (batch_size, 1, 1)
      )
      # The compute_attention_block should be executed if at least one element
      # in the slice is non-zero, meaning at least one batch requires work for
      # this block.
      output, l, m = jax.lax.cond(
          jnp.any(jnp.not_equal(mask_i_j_slice, 0)),
          compute_attention_block,
          identity,
          output,
          l,
          m,
      )

      return output, l, m

    output, l, m = jax.lax.fori_loop(
        0, num_q_blocks, inner_loop_body, (output, l, m), unroll=True
    )

    return (output, l, m)

  output, l, m = jax.lax.fori_loop(
      0, num_kv_blocks, outer_loop_body, (output, l, m), unroll=True
  )

  # Reshape the output to drop the size one dimension at index 2,
  # which corresponds to `num_q_heads // num_kv_heads` when
  # num_q_heads == num_kv_heads.
  output = output.squeeze(axis=2)
  if not save_residuals:
    # To avoid remat of the output, we can use context=hbm remat policy as in
    # maxtext/configs/types.py
    return output

  l = l.squeeze(axis=2)
  m = m.squeeze(axis=2)
  stats = {"logsumexp": m + jnp.log(l), "max_logits": m}
  stats = jax.tree.map(jax.lax.stop_gradient, stats)
  return output, stats


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
    raise ValueError(
        f"q_seq_len {q_seq_len} must be divisible by block_q {block_q}"
    )
  if kv_seq_len % block_kv != 0:
    raise ValueError(
        f"kv_seq_len {kv_seq_len} must be divisible by block_kv {block_kv}"
    )
  q_blocks = q_seq_len // block_q
  kv_blocks = kv_seq_len // block_kv

  blocked_mask = mask.reshape(
      batch_size, q_blocks, block_q, kv_blocks, block_kv
  )
  return jnp.count_nonzero(blocked_mask, axis=(2, 4)).astype(jnp.int32)
