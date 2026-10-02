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
"""Tests for USP attention layout helpers."""

from __future__ import annotations

import types

from absl.testing import absltest
import jax

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.kernels.attention import ulysses_attention
from maxtext.kernels.attention import usp_attention


class UspAttentionTest(absltest.TestCase):

  def test_context_parallel_strategy_helper_identifies_usp(self):
    self.assertTrue(
        usp_attention.is_context_parallel_usp_requested(types.SimpleNamespace(context_parallel_strategy="usp"))
    )
    self.assertFalse(
        usp_attention.is_context_parallel_usp_requested(types.SimpleNamespace(context_parallel_strategy="ulysses"))
    )

  def test_validate_usp_runtime_allows_train_mode(self):
    usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN)

  def test_validate_usp_runtime_rejects_unsupported_runtime_features(self):
    with self.assertRaisesRegex(ValueError, "train mode"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_PREFILL)
    with self.assertRaisesRegex(ValueError, "ragged attention"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN, use_ragged_attention=True)
    with self.assertRaisesRegex(ValueError, "chunked prefill"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN, previous_chunk=object())
    with self.assertRaisesRegex(ValueError, "attention sinks"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN, sinks=object())
    with self.assertRaisesRegex(ValueError, "indexer"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN, indexer_mask=object())
    with self.assertRaisesRegex(ValueError, "bidirectional"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN, bidirectional_mask=object())
    with self.assertRaisesRegex(NotImplementedError, "record_max_logits"):
      usp_attention.validate_usp_runtime(model_mode=MODEL_MODE_TRAIN, record_max_logits=True)

  def test_with_usp_sequence_axes_preserves_partition_spec_type(self):
    spec = jax.sharding.PartitionSpec("data", None, None, "tensor")

    out = usp_attention.with_usp_sequence_axes(spec, "context", "context_usp_ulysses", sequence_dim=2)

    self.assertIsInstance(out, jax.sharding.PartitionSpec)
    self.assertEqual(tuple(out), ("data", None, ("context", "context_usp_ulysses"), "tensor"))

  def test_layout_validators_reject_invalid_shardings(self):
    mesh = types.SimpleNamespace(shape={"context": 2, "context_usp_ulysses": 2, "tensor": 2})
    pair = ("context", "context_usp_ulysses")
    cases = [
        (
            "sequence sharding dimension",
            lambda: usp_attention.with_usp_sequence_axes((None, None), *pair, sequence_dim=2),
        ),
        (
            "unsharded or exactly",
            lambda: usp_attention.with_usp_sequence_axes((None, None, "context", None), *pair, sequence_dim=2),
        ),
        (
            "to differ",
            lambda: usp_attention.validate_usp_mesh_axes(
                axis_names_q=(None, None, pair, None),
                axis_names_kv=(None, None, pair, None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=mesh,
                ring_axis="context",
                ulysses_axis="context",
            ),
        ),
        (
            "mesh axis 'context_usp_ulysses' to exist",
            lambda: usp_attention.validate_usp_mesh_axes(
                axis_names_q=(None, None, pair, None),
                axis_names_kv=(None, None, pair, None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=types.SimpleNamespace(shape={"context": 2}),
                ring_axis="context",
                ulysses_axis="context_usp_ulysses",
            ),
        ),
        (
            "only on the sequence dimension",
            lambda: usp_attention.validate_usp_mesh_axes(
                axis_names_q=(None, "context_usp_ulysses", pair, None),
                axis_names_kv=(None, None, pair, None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=mesh,
                ring_axis="context",
                ulysses_axis="context_usp_ulysses",
            ),
        ),
        (
            "Q sequence sharding to be exactly",
            lambda: usp_attention.validate_usp_mesh_axes(
                axis_names_q=(None, None, "context", None),
                axis_names_kv=(None, None, pair, None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=mesh,
                ring_axis="context",
                ulysses_axis="context_usp_ulysses",
            ),
        ),
        (
            "K/V sequence sharding to be exactly",
            lambda: usp_attention.validate_usp_mesh_axes(
                axis_names_q=(None, None, pair, None),
                axis_names_kv=(None, None, None, None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=mesh,
                ring_axis="context",
                ulysses_axis="context_usp_ulysses",
            ),
        ),
        (
            "TPU USP attention requires local query heads",
            lambda: ulysses_attention.validate_head_sharding(
                axis_names_q=(None, "tensor", pair, None),
                axis_names_kv=(None, "tensor", pair, None),
                mesh=mesh,
                num_query_heads=4,
                num_kv_heads=4,
                head_dim_q=1,
                head_dim_kv=1,
                ulysses_size=4,
                attention_label="TPU USP attention",
            ),
        ),
    ]
    for expected_regex, invoke in cases:
      with self.subTest(expected_regex=expected_regex):
        with self.assertRaisesRegex(ValueError, expected_regex):
          invoke()


if __name__ == "__main__":
  absltest.main()
