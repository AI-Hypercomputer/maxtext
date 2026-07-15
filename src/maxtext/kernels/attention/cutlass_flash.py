# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUTLASS FlashAttention-2 wrapper (via flash-attn-jax) for MaxText.

Feature parity with the GPU ``flash`` (pallas) path:
  - causal self-attention with optional sliding window (LOCAL_SLIDING),
  - native GQA/MQA (num_kv_heads < num_query_heads, no KV repeat),
  - sequence packing via segment ids, lowered to FA2's varlen kernel with
    cu_seqlens built inside jit (fixed-size, empty tail sequences are skipped
    by the kernel).

Constraints (callers must fall back for these): head_dim <= 256, sm80+,
training/prefill only (no decode), no attention sinks / logit soft-cap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flash_attn_jax import flash_mha, flash_mha_varlen


def _cu_seqlens_from_segment_ids(segment_ids: jax.Array, max_segments_per_row: int) -> jax.Array:
  """Builds FA2 cu_seqlens (int32, [num_seqs + 1]) from MaxText segment ids.

  A new sequence starts at every row start and at every within-row segment-id
  change (packing never spans rows). Each row holds at most max_segments_per_row
  segments plus one trailing padding run (id 0), which also counts as a run
  start, so the result has a fixed size of ``batch * (max_segments_per_row + 1)
  + 1``; unused tail entries repeat the total token count, i.e. zero-length
  sequences that FA2 skips.
  """
  batch, seq_len = segment_ids.shape
  total = batch * seq_len
  flat = segment_ids.reshape(total)
  positions = jnp.arange(total, dtype=jnp.int32)
  prev = jnp.concatenate([jnp.full((1,), -1, dtype=flat.dtype), flat[:-1]])
  is_start = (positions % seq_len == 0) | (flat != prev)
  # Valid starts sort to the front, padding (=total) to the back.
  start_positions = jnp.where(is_start, positions, total)
  num_seqs = batch * (max_segments_per_row + 1)
  starts = jax.lax.sort(start_positions)[:num_seqs]
  return jnp.concatenate([starts, jnp.full((1,), total, dtype=jnp.int32)])


def cutlass_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    segment_ids: jax.Array | None,
    *,
    sm_scale: float = 1.0,
    window: int | None = None,
    max_segments_per_row: int = 1,
) -> jax.Array:
  """Causal (optionally sliding-window) attention on [B, S, H, D] tensors.

  ``window`` follows the MaxText LOCAL_SLIDING convention: query i attends
  keys j with ``i - window < j <= i``. FA2's window_size=(left, right) counts
  additional positions, hence ``left = window - 1``.
  """
  batch, seq_len, num_q_heads, head_dim = query.shape
  window_size = (window - 1, 0) if window is not None else (-1, -1)

  # With at most one segment per row, causal masking alone already isolates
  # rows (right-padding is never attended by earlier tokens), so the cheaper
  # dense kernel is exact — this avoids the ~3% varlen scheduling overhead.
  if segment_ids is None or max_segments_per_row <= 1:
    return flash_mha(query, key, value, softmax_scale=sm_scale, is_causal=True, window_size=window_size)

  cu_seqlens = _cu_seqlens_from_segment_ids(segment_ids, max_segments_per_row)
  q_flat = query.reshape(batch * seq_len, num_q_heads, head_dim)
  k_flat = key.reshape(batch * seq_len, key.shape[-2], head_dim)
  v_flat = value.reshape(batch * seq_len, value.shape[-2], head_dim)
  out = flash_mha_varlen(
      q_flat,
      k_flat,
      v_flat,
      cu_seqlens,
      max_seqlen_q=seq_len,
      max_seqlen_k=seq_len,
      softmax_scale=sm_scale,
      is_causal=True,
      window_size=window_size,
  )
  return out.reshape(batch, seq_len, num_q_heads, head_dim)
