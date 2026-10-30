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
"""Tests for the block-masked JAX flash attention custom VJP."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from maxtext.kernels.attention import jax_flash_attention
import numpy as np

_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)


def _reference_attention(q, k, v, segment_ids, mask, cap):
  """Dense attention with a runtime mask, segment ids and GQA broadcasting."""
  num_q_heads, num_kv_heads = q.shape[1], k.shape[1]
  k = jnp.repeat(k, num_q_heads // num_kv_heads, axis=1)
  v = jnp.repeat(v, num_q_heads // num_kv_heads, axis=1)
  logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)
  if cap is not None:
    logits = jnp.tanh(logits / cap) * cap
  full_mask = mask[:, None] if mask.ndim == 3 else mask[None, None]
  if segment_ids is not None:
    seg_mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    full_mask = jnp.logical_and(full_mask, seg_mask[:, None])
  logits = jnp.where(full_mask, logits, _MASK_VALUE)
  return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, axis=-1), v)


class FlashAttentionBlockMaskedCustomVjpTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="mha_3d_mask", num_q_heads=4, num_kv_heads=4, mask_ndim=3, segments=True, cap=None),
      dict(testcase_name="mqa_3d_mask", num_q_heads=4, num_kv_heads=1, mask_ndim=3, segments=True, cap=None),
      dict(testcase_name="gqa_2d_mask", num_q_heads=4, num_kv_heads=2, mask_ndim=2, segments=False, cap=None),
      dict(testcase_name="batch_equals_heads", num_q_heads=2, num_kv_heads=2, mask_ndim=3, segments=True, cap=None),
      dict(testcase_name="soft_cap", num_q_heads=4, num_kv_heads=4, mask_ndim=3, segments=False, cap=5.0),
  )
  def test_matches_dense_reference(self, num_q_heads, num_kv_heads, mask_ndim, segments, cap):
    batch, q_len, kv_len, head_dim, v_head_dim = 2, 32, 64, 16, 8
    keys = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(keys[0], (batch, num_q_heads, q_len, head_dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, num_kv_heads, kv_len, head_dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, num_kv_heads, kv_len, v_head_dim), jnp.float32)
    cotangent = jax.random.normal(keys[3], (batch, num_q_heads, q_len, v_head_dim), jnp.float32)

    mask_shape = (batch, q_len, kv_len) if mask_ndim == 3 else (q_len, kv_len)
    mask = jax.random.bernoulli(keys[4], 0.5, mask_shape)
    # Keep the diagonal so that no query row is fully masked.
    mask = jnp.logical_or(mask, jnp.eye(q_len, kv_len, dtype=jnp.bool_))

    segment_ids = None
    if segments:
      q_seg = jnp.broadcast_to((jnp.arange(q_len) >= q_len // 2).astype(jnp.int32), (batch, q_len))
      kv_seg = jnp.broadcast_to((jnp.arange(kv_len) >= q_len // 2).astype(jnp.int32), (batch, kv_len))
      segment_ids = jax_flash_attention.SegmentIds(q=q_seg, kv=kv_seg)

    def loss(fn, q, k, v):
      return jnp.sum(fn(q, k, v) * cotangent)

    def flash(q, k, v):
      out, _ = jax_flash_attention.flash_attention_block_masked_custom_vjp(
          q, k, v, segment_ids, mask, 16, 16, _MASK_VALUE, cap
      )
      return out

    def reference(q, k, v):
      return _reference_attention(q, k, v, segment_ids, mask, cap)

    np.testing.assert_allclose(jax.jit(flash)(q, k, v), reference(q, k, v), rtol=1e-4, atol=1e-4)
    flash_grads = jax.jit(jax.grad(lambda *a: loss(flash, *a), argnums=(0, 1, 2)))(q, k, v)
    reference_grads = jax.grad(lambda *a: loss(reference, *a), argnums=(0, 1, 2))(q, k, v)
    for name, actual, expected in zip("qkv", flash_grads, reference_grads):
      np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, err_msg=f"d{name}")


if __name__ == "__main__":
  absltest.main()
