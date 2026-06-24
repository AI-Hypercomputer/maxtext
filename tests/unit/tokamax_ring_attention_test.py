# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for MaxText's Tokamax ring attention adapter."""

# pylint: disable=protected-access

from __future__ import annotations

import types
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.kernels.attention import tokamax_ring_attention


class TokamaxRingAttentionTest(absltest.TestCase):

  def test_is_context_parallel_ring_requested_accepts_case_insensitive_strategy(self):
    config = types.SimpleNamespace(context_parallel_strategy="Ring")

    self.assertTrue(tokamax_ring_attention.is_context_parallel_ring_requested(config))

  def test_make_causal_mask_uses_local_tokamax_causal_mask(self):
    mask = tokamax_ring_attention._make_causal_mask((16, 16), 4)

    self.assertEqual(mask.shape, (16, 16))
    self.assertEqual(mask.q_sequence.tolist(), list(range(16)))

  def test_runtime_validation_accepts_global_attention_type(self):
    tokamax_ring_attention.validate_tokamax_ring_runtime(model_mode=MODEL_MODE_TRAIN)

  def test_validate_ring_mesh_axis_requires_key_value_sequence_sharding(self):
    mesh = types.SimpleNamespace(shape={"context": 4})

    with self.assertRaisesRegex(ValueError, "K/V sequence sharding"):
      tokamax_ring_attention.validate_ring_mesh_axis(
          axis_names_q=(None, None, "context", None),
          axis_names_kv=(None, None, None, None),
          sequence_dim_q=2,
          sequence_dim_kv=2,
          mesh=mesh,
          ring_axis="context",
      )

  def test_validate_head_sharding_rejects_mismatched_gqa_axes(self):
    mesh = types.SimpleNamespace(shape={"tensor": 2})

    with self.assertRaisesRegex(ValueError, "Q and KV head sharding"):
      tokamax_ring_attention.validate_head_sharding(
          axis_names_q=(None, "tensor", "context", None),
          axis_names_kv=(None, None, "context", None),
          mesh=mesh,
          num_query_heads=8,
          num_kv_heads=4,
          head_dim_q=1,
          head_dim_kv=1,
      )

  def test_call_ring_attention_uses_ring_kernel_segment_ids(self):
    captured = {}

    class RingSegmentIds:

      def __init__(self, q, kv):
        self.q = q
        self.kv = kv

    def kernel(q, k, v, segment_ids):
      captured["segment_ids_type"] = type(segment_ids)
      captured["q_segment_shape"] = segment_ids.q.shape
      captured["kv_segment_shape"] = segment_ids.kv.shape
      return q + k + v

    query = jnp.ones((1, 2, 4, 2))
    key = jnp.ones((1, 2, 4, 2))
    value = jnp.ones((1, 2, 4, 2))
    segment_ids = jnp.ones((1, 4), dtype=jnp.int32)

    with mock.patch.object(tokamax_ring_attention.ring_attention_kernel, "SegmentIds", RingSegmentIds):
      out = tokamax_ring_attention.call_ring_attention(
          query,
          key,
          value,
          segment_ids,
          segment_ids,
          kernel,
      )

    self.assertEqual(out.shape, query.shape)
    self.assertIs(captured["segment_ids_type"], RingSegmentIds)
    self.assertEqual(captured["q_segment_shape"], (4,))
    self.assertEqual(captured["kv_segment_shape"], (4,))

  def test_with_sequence_axis_preserves_partition_spec_type(self):
    spec = jax.sharding.PartitionSpec("data", None, None, "model")

    out = tokamax_ring_attention.with_sequence_axis(spec, "context", sequence_dim=2)

    self.assertIsInstance(out, jax.sharding.PartitionSpec)
    self.assertEqual(tuple(out), ("data", None, "context", "model"))


if __name__ == "__main__":
  absltest.main()
