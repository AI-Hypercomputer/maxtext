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

"""Causal softmax attention as a pure function.

This is the arithmetic every m3 attention module ends with once its own
projections, norms and position encodings are done: grouped-query scores,
a causal mask that also respects packed-segment boundaries, softmax, and the
weighted sum over values. It is a function, not a module -- it has no
parameters and no configuration beyond its arguments -- which is what keeps
it out of the "shared attention class" category that `m3/README.md` rules
out.

The `kernel` argument is how a fused implementation gets in without the
model knowing: a model resolves `kernels.attention_from_config(config,
sharding)` once at construction and passes the result here. The default is
the reference arithmetic below.

Numerics follow legacy MaxText's `apply_attention_dot` so that parity holds in
bf16: scores stay in the input dtype unless `float32_logits`, the mask is
applied before any cast, and the normalizer is applied after the value
contraction.
"""

import jax
import jax.numpy as jnp

from maxtext.m3.core import kernels


def causal_mask(positions: jax.Array, segment_ids: jax.Array | None) -> jax.Array:
  """`(batch, 1, 1, q_length, kv_length)` boolean mask: attend to earlier positions in the same segment."""
  mask = positions[:, None, :] <= positions[:, :, None]
  if segment_ids is not None:
    mask = mask & (segment_ids[:, :, None] == segment_ids[:, None, :])
  return mask[:, None, None, :, :]


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    positions: jax.Array,
    segment_ids: jax.Array | None = None,
    float32_logits: bool = False,
    kernel: kernels.AttentionKernel = kernels.XLA_ATTENTION,
) -> jax.Array:
  """Causal grouped-query attention.

  Args:
    query: `(batch, length, heads, head_dim)`, already scaled by the model.
    key: `(batch, length, kv_heads, head_dim)`; `heads` must be a multiple of
      `kv_heads`.
    value: `(batch, length, kv_heads, value_dim)`.
    positions: `(batch, length)` token positions; the mask attends to keys at
      positions no later than the query's. Fused kernels mask by sequence
      index instead, which agrees with this whenever positions count up by
      one within a segment -- true of m3's packing, and the same assumption
      legacy's flash path makes.
    segment_ids: `(batch, length)` ids for packed sequences, or `None`.
    float32_logits: Compute the softmax in float32. Fused kernels always do.
    kernel: Which implementation runs; see `core/kernels.py`.

  Returns:
    `(batch, length, heads, value_dim)` in `value.dtype`.
  """
  if kernel.backend != "xla":
    return kernels.fused_attention(query, key, value, segment_ids=segment_ids, kernel=kernel)

  b, t, n_q, d = query.shape
  n_kv = key.shape[2]
  groups = n_q // n_kv
  query = jnp.reshape(query, (b, t, n_kv, groups, d))

  scores = jnp.einsum("btkgd,bskd->bkgts", query, key)
  scores = jnp.where(causal_mask(positions, segment_ids), scores, -1e10)
  if float32_logits:
    scores = scores.astype(jnp.float32)
  scores = jnp.reshape(scores, (b, n_q, t, -1))
  scores = scores - jnp.max(scores, axis=-1, keepdims=True)
  weights = jnp.exp(scores)
  normalizer = jnp.sum(weights, axis=-1, keepdims=True)

  weights = jnp.reshape(weights.astype(value.dtype), (b, n_kv, groups, t, -1))
  out = jnp.einsum("bkgts,bskd->btkgd", weights, value)
  out = jnp.reshape(out, (b, t, n_q, value.shape[-1]))
  normalizer = jnp.transpose(normalizer, (0, 2, 1, 3))
  return (out / normalizer).astype(value.dtype)
