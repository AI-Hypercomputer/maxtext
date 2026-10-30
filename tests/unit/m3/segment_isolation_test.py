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

"""Unit tests for Workstream 2: hybrid-model segment isolation (Conv1d, PLE dilated conv, n-gram, recurrence)."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.layers import (
    segment_causal_conv1d,
    segment_chunk_decay_mask,
    segment_linear_recurrence,
    segment_ngram_hash_ids,
)


class SegmentIsolationTest(absltest.TestCase):
  """Verifies packed multi-document execution matches separate per-document execution (`<= 1e-5` fp32)."""

  def test_segment_causal_conv1d_and_dilated_conv_packed_vs_unpacked_grad_parity(self):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    doc_a = jax.random.normal(k1, (1, 5, 8), dtype=jnp.float32)
    doc_b = jax.random.normal(k2, (1, 6, 8), dtype=jnp.float32)
    weight = jax.random.normal(k3, (4, 8), dtype=jnp.float32)

    packed_x = jnp.concatenate([doc_a, doc_b, jnp.zeros((1, 1, 8), jnp.float32)], axis=1)
    packed_seg = jnp.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0]], dtype=jnp.int32)

    for dilation in (1, 3):
      y_a = segment_causal_conv1d(doc_a, weight, jnp.ones((1, 5), jnp.int32), dilation=dilation)
      y_b = segment_causal_conv1d(doc_b, weight, jnp.ones((1, 6), jnp.int32), dilation=dilation)
      y_packed = segment_causal_conv1d(packed_x, weight, packed_seg, dilation=dilation)

      np.testing.assert_allclose(np.asarray(y_packed[:, :5]), np.asarray(y_a), atol=1e-6)
      np.testing.assert_allclose(np.asarray(y_packed[:, 5:11]), np.asarray(y_b), atol=1e-6)
      np.testing.assert_allclose(np.asarray(y_packed[:, 11:]), 0.0, atol=1e-6)

      grad_packed = jax.grad(lambda w, d=dilation: jnp.sum(segment_causal_conv1d(packed_x, w, packed_seg, dilation=d)))(
          weight
      )
      grad_unpacked = jax.grad(
          lambda w, d=dilation: jnp.sum(segment_causal_conv1d(doc_a, w, jnp.ones((1, 5), jnp.int32), dilation=d))
          + jnp.sum(segment_causal_conv1d(doc_b, w, jnp.ones((1, 6), jnp.int32), dilation=d))
      )(weight)
      np.testing.assert_allclose(np.asarray(grad_packed), np.asarray(grad_unpacked), atol=1e-5)

  def test_segment_ngram_hash_resets_at_document_boundaries(self):
    doc_a = jnp.array([[11, 12, 13]], dtype=jnp.int32)
    doc_b = jnp.array([[21, 22, 23, 24]], dtype=jnp.int32)
    packed = jnp.array([[11, 12, 13, 21, 22, 23, 24, 0]], dtype=jnp.int32)
    seg = jnp.array([[1, 1, 1, 2, 2, 2, 2, 0]], dtype=jnp.int32)

    h_a = segment_ngram_hash_ids(doc_a, jnp.ones_like(doc_a), ngram_size=3, num_buckets=1024, reset_token_id=2)
    h_b = segment_ngram_hash_ids(doc_b, jnp.ones_like(doc_b), ngram_size=3, num_buckets=1024, reset_token_id=2)
    h_packed = segment_ngram_hash_ids(packed, seg, ngram_size=3, num_buckets=1024, reset_token_id=2)

    np.testing.assert_array_equal(np.asarray(h_packed[:, :3]), np.asarray(h_a))
    np.testing.assert_array_equal(np.asarray(h_packed[:, 3:7]), np.asarray(h_b))
    np.testing.assert_array_equal(np.asarray(h_packed[:, 7:]), 0)

  def test_segment_linear_recurrence_and_chunk_decay_mask_isolation(self):
    key = jax.random.key(7)
    kq, kk, kv, kg = jax.random.split(key, 4)
    q_a = jax.random.normal(kq, (1, 4, 2, 6), dtype=jnp.float32)
    k_a = jax.random.normal(kk, (1, 4, 2, 6), dtype=jnp.float32)
    v_a = jax.random.normal(kv, (1, 4, 2, 6), dtype=jnp.float32)
    g_a = -jax.nn.softplus(jax.random.normal(kg, (1, 4, 2), dtype=jnp.float32))

    q_b = jax.random.normal(jax.random.fold_in(kq, 1), (1, 3, 2, 6), dtype=jnp.float32)
    k_b = jax.random.normal(jax.random.fold_in(kk, 1), (1, 3, 2, 6), dtype=jnp.float32)
    v_b = jax.random.normal(jax.random.fold_in(kv, 1), (1, 3, 2, 6), dtype=jnp.float32)
    g_b = -jax.nn.softplus(jax.random.normal(jax.random.fold_in(kg, 1), (1, 3, 2), dtype=jnp.float32))

    out_a = segment_linear_recurrence(q_a, k_a, v_a, g_a, jnp.ones((1, 4), jnp.int32))
    out_b = segment_linear_recurrence(q_b, k_b, v_b, g_b, jnp.ones((1, 3), jnp.int32))

    q_p = jnp.concatenate([q_a, q_b, jnp.zeros((1, 1, 2, 6))], axis=1)
    k_p = jnp.concatenate([k_a, k_b, jnp.zeros((1, 1, 2, 6))], axis=1)
    v_p = jnp.concatenate([v_a, v_b, jnp.zeros((1, 1, 2, 6))], axis=1)
    g_p = jnp.concatenate([g_a, g_b, jnp.zeros((1, 1, 2))], axis=1)
    seg_p = jnp.array([[1, 1, 1, 1, 2, 2, 2, 0]], dtype=jnp.int32)

    out_p = segment_linear_recurrence(q_p, k_p, v_p, g_p, seg_p)
    np.testing.assert_allclose(np.asarray(out_p[:, :4]), np.asarray(out_a), atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_p[:, 4:7]), np.asarray(out_b), atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_p[:, 7:]), 0.0, atol=1e-6)

    decay = segment_chunk_decay_mask(jnp.cumsum(g_p, axis=1), seg_p)
    # Cross-segment entry (token 5 in doc 2 attending to token 2 in doc 1) must be strictly zero.
    np.testing.assert_allclose(np.asarray(decay[:, :, 5, 2]), 0.0, atol=0.0)


if __name__ == "__main__":
  absltest.main()
