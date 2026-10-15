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
"""Tests for Ulysses attention layout helpers."""

from __future__ import annotations

import types
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.kernels.attention import ulysses_attention


class UlyssesAttentionTest(absltest.TestCase):

  def test_context_parallel_strategy_helper_identifies_ulysses(self):
    self.assertTrue(
        ulysses_attention.is_context_parallel_ulysses_requested(
            types.SimpleNamespace(context_parallel_strategy="ulysses")
        )
    )
    self.assertFalse(
        ulysses_attention.is_context_parallel_ulysses_requested(types.SimpleNamespace(context_parallel_strategy="ring"))
    )

  def test_validate_ulysses_runtime_allows_train_mode(self):
    ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN)

  def test_validate_ulysses_runtime_rejects_unsupported_runtime_features(self):
    with self.assertRaisesRegex(ValueError, "train mode"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_PREFILL)
    with self.assertRaisesRegex(ValueError, "ragged attention"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN, use_ragged_attention=True)
    with self.assertRaisesRegex(ValueError, "chunked prefill"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN, previous_chunk=object())
    with self.assertRaisesRegex(ValueError, "attention sinks"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN, sinks=object())
    with self.assertRaisesRegex(ValueError, "indexer"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN, indexer_mask=object())
    with self.assertRaisesRegex(ValueError, "bidirectional"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN, bidirectional_mask=object())
    with self.assertRaisesRegex(NotImplementedError, "record_max_logits"):
      ulysses_attention.validate_ulysses_runtime(model_mode=MODEL_MODE_TRAIN, record_max_logits=True)

  def test_with_sequence_axis_preserves_partition_spec_type(self):
    spec = jax.sharding.PartitionSpec("data", None, None, "tensor")

    out = ulysses_attention.with_sequence_axis(spec, "context", sequence_dim=2)

    self.assertIsInstance(out, jax.sharding.PartitionSpec)
    self.assertEqual(tuple(out), ("data", None, "context", "tensor"))

  def test_validate_ulysses_mesh_axis_requires_sequence_sharding(self):
    mesh = types.SimpleNamespace(shape={"context": 4})

    with self.assertRaisesRegex(ValueError, "K/V sequence"):
      ulysses_attention.validate_ulysses_mesh_axis(
          axis_names_q=(None, None, "context", None),
          axis_names_kv=(None, None, None, None),
          sequence_dim_q=2,
          sequence_dim_kv=2,
          mesh=mesh,
          ulysses_axis="context",
      )

  def test_layout_validators_reject_invalid_shardings(self):
    mesh = types.SimpleNamespace(shape={"context": 4, "tensor": 2})
    cases = [
        (
            "unsharded or exactly",
            lambda: ulysses_attention.with_sequence_axis((None, None, "tensor", None), "context", sequence_dim=2),
        ),
        (
            "mesh axis 'context' to exist",
            lambda: ulysses_attention.validate_ulysses_mesh_axis(
                axis_names_q=(None, None, "context", None),
                axis_names_kv=(None, None, "context", None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=types.SimpleNamespace(shape={"tensor": 2}),
                ulysses_axis="context",
            ),
        ),
        (
            "only on the sequence dimension",
            lambda: ulysses_attention.validate_ulysses_mesh_axis(
                axis_names_q=(None, "context", "context", None),
                axis_names_kv=(None, None, "context", None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=mesh,
                ulysses_axis="context",
            ),
        ),
        (
            "Q sequence sharding to be exactly",
            lambda: ulysses_attention.validate_ulysses_mesh_axis(
                axis_names_q=(None, None, None, None),
                axis_names_kv=(None, None, "context", None),
                sequence_dim_q=2,
                sequence_dim_kv=2,
                mesh=mesh,
                ulysses_axis="context",
            ),
        ),
        (
            "D_KV/head-dim",
            lambda: ulysses_attention.validate_dkv_sharding(
                axis_names_q=(None, None, "context", "tensor"),
                axis_names_kv=(None, None, "context", None),
                dkv_dim_q=3,
                dkv_dim_kv=3,
                attention_label="TPU Ulysses attention",
            ),
        ),
        (
            "divisible by Q head shards",
            lambda: ulysses_attention.validate_head_sharding(
                axis_names_q=(None, "tensor", "context", None),
                axis_names_kv=(None, "tensor", "context", None),
                mesh=mesh,
                num_query_heads=9,
                num_kv_heads=4,
                head_dim_q=1,
                head_dim_kv=1,
                ulysses_size=4,
                attention_label="TPU Ulysses attention",
            ),
        ),
        (
            "divisible by KV head shards",
            lambda: ulysses_attention.validate_head_sharding(
                axis_names_q=(None, "tensor", "context", None),
                axis_names_kv=(None, "tensor", "context", None),
                mesh=mesh,
                num_query_heads=64,
                num_kv_heads=9,
                head_dim_q=1,
                head_dim_kv=1,
                ulysses_size=4,
                attention_label="TPU Ulysses attention",
            ),
        ),
        (
            "divisible by local KV heads",
            lambda: ulysses_attention.validate_head_sharding(
                axis_names_q=(None, "tensor", "context", None),
                axis_names_kv=(None, "tensor", "context", None),
                mesh=mesh,
                num_query_heads=64,
                num_kv_heads=24,
                head_dim_q=1,
                head_dim_kv=1,
                ulysses_size=4,
                attention_label="TPU Ulysses attention",
            ),
        ),
        (
            r"local query heads \(8\) to be divisible by the Ulysses exchange size",
            lambda: ulysses_attention.validate_head_sharding(
                axis_names_q=(None, None, "context", None),
                axis_names_kv=(None, None, "context", None),
                mesh=mesh,
                num_query_heads=8,
                num_kv_heads=4,
                head_dim_q=1,
                head_dim_kv=1,
                ulysses_size=16,
                attention_label="TPU Ulysses attention",
            ),
        ),
    ]
    for expected_regex, invoke in cases:
      with self.subTest(expected_regex=expected_regex):
        with self.assertRaisesRegex(ValueError, expected_regex):
          invoke()

  def test_validate_head_sharding_uses_local_heads_after_tensor_sharding(self):
    mesh = types.SimpleNamespace(shape={"context": 4, "tensor": 2})

    ulysses_attention.validate_head_sharding(
        axis_names_q=(None, "tensor", "context", None),
        axis_names_kv=(None, "tensor", "context", None),
        mesh=mesh,
        num_query_heads=64,
        num_kv_heads=16,
        head_dim_q=1,
        head_dim_kv=1,
        ulysses_size=4,
        attention_label="TPU Ulysses attention",
    )

    with self.assertRaisesRegex(ValueError, "local KV heads"):
      ulysses_attention.validate_head_sharding(
          axis_names_q=(None, "tensor", "context", None),
          axis_names_kv=(None, "tensor", "context", None),
          mesh=mesh,
          num_query_heads=64,
          num_kv_heads=4,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
      )

  def test_validate_head_sharding_rejects_mqa(self):
    mesh = types.SimpleNamespace(shape={"context": 4})

    with self.assertRaisesRegex(ValueError, "MQA"):
      ulysses_attention.validate_head_sharding(
          axis_names_q=(None, None, "context", None),
          axis_names_kv=(None, None, "context", None),
          mesh=mesh,
          num_query_heads=16,
          num_kv_heads=1,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
      )

  def test_validate_head_sharding_requires_q_and_kv_head_axes_to_match(self):
    mesh = types.SimpleNamespace(shape={"context": 4, "tensor": 2})

    with self.assertRaisesRegex(ValueError, "head sharding to match"):
      ulysses_attention.validate_head_sharding(
          axis_names_q=(None, "tensor", "context", None),
          axis_names_kv=(None, None, "context", None),
          mesh=mesh,
          num_query_heads=64,
          num_kv_heads=16,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
      )

  def test_validate_head_sharding_accepts_kv_head_replication_only_when_allowed(self):
    """Two KV heads at Ulysses size 4 need replication, which only the Ulysses path (not USP) performs."""
    mesh = types.SimpleNamespace(shape={"context": 4})
    kwargs = {
        "axis_names_q": (None, None, "context", None),
        "axis_names_kv": (None, None, "context", None),
        "mesh": mesh,
        "num_query_heads": 32,
        "num_kv_heads": 2,
        "head_dim_q": 1,
        "head_dim_kv": 1,
        "ulysses_size": 4,
        "attention_label": "TPU Ulysses attention",
    }
    with self.assertRaisesRegex(
        ValueError, r"local KV heads \(2\) to be divisible by the Ulysses exchange size \(4\)\.$"
    ):
      ulysses_attention.validate_head_sharding(**kwargs)
    ulysses_attention.validate_head_sharding(**kwargs, allow_kv_head_replication=True)
    # One local KV head left by tensor sharding is repeated to every rank too.
    tensor_sharded = (None, "tensor", "context", None)
    ulysses_attention.validate_head_sharding(
        **{
            **kwargs,
            "axis_names_q": tensor_sharded,
            "axis_names_kv": tensor_sharded,
            "mesh": types.SimpleNamespace(shape={"context": 4, "tensor": 2}),
        },
        allow_kv_head_replication=True,
    )
    with self.assertRaisesRegex(ValueError, "or to divide it"):
      ulysses_attention.validate_head_sharding(
          **{**kwargs, "num_query_heads": 36, "num_kv_heads": 6}, allow_kv_head_replication=True
      )

  def test_kv_head_replication(self):
    cases = {(2, 4): 2, (2, 8): 4, (1, 4): 4, (4, 4): 1, (8, 4): 1, (16, 2): 1}
    for (local_kv_heads, ulysses_size), expected in cases.items():
      with self.subTest(local_kv_heads=local_kv_heads, ulysses_size=ulysses_size):
        self.assertEqual(ulysses_attention.kv_head_replication(local_kv_heads, ulysses_size), expected)
    for local_kv_heads, ulysses_size in ((3, 4), (6, 4), (4, 6)):
      with self.subTest(local_kv_heads=local_kv_heads, ulysses_size=ulysses_size):
        with self.assertRaisesRegex(ValueError, "divide one another"):
          ulysses_attention.kv_head_replication(local_kv_heads, ulysses_size)

  def test_repeat_kv_heads_pairs_every_query_head_with_its_kv_head(self):
    """After the exchange, every rank's query heads find their own KV head (32 query / 2 KV heads, Ulysses 4 and 8).

    Splash pairs local query head i with local KV head i // (local query heads / local KV heads). The exchange
    hands rank r the r-th contiguous block of heads, simulated here by splitting the head dimension.
    """
    num_query_heads, num_kv_heads = 32, 2
    key = jnp.arange(num_kv_heads, dtype=jnp.int32).reshape(1, num_kv_heads, 1, 1)
    for ulysses_size in (4, 8):
      with self.subTest(ulysses_size=ulysses_size):
        replication = ulysses_attention.kv_head_replication(num_kv_heads, ulysses_size)
        repeated = ulysses_attention.repeat_kv_heads_for_ulysses(key, replication)
        np.testing.assert_array_equal(repeated[0, :, 0, 0], np.repeat(np.arange(num_kv_heads), replication))
        query_heads = np.arange(num_query_heads)
        for rank, kv_block in enumerate(np.split(np.asarray(repeated[0, :, 0, 0]), ulysses_size)):
          rank_query_heads = np.split(query_heads, ulysses_size)[rank]
          local_group = len(rank_query_heads) // len(kv_block)
          paired = kv_block[np.arange(len(rank_query_heads)) // local_group]
          np.testing.assert_array_equal(paired, rank_query_heads // (num_query_heads // num_kv_heads))
    self.assertIs(ulysses_attention.repeat_kv_heads_for_ulysses(key, 1), key)

  def test_ulysses_all_to_all_moves_heads_to_sequence(self):
    tensor = jnp.ones((1, 16, 8, 2))

    with mock.patch.object(ulysses_attention.jax.lax, "all_to_all", return_value="out") as all_to_all:
      out = ulysses_attention.ulysses_all_to_all(tensor, "context")

    self.assertEqual(out, "out")
    all_to_all.assert_called_once_with(tensor, "context", split_axis=1, concat_axis=2, tiled=True)

  def test_inverse_ulysses_all_to_all_moves_sequence_to_heads(self):
    tensor = jnp.ones((1, 4, 32, 2))

    with mock.patch.object(ulysses_attention.jax.lax, "all_to_all", return_value="out") as all_to_all:
      out = ulysses_attention.inverse_ulysses_all_to_all(tensor, "context")

    self.assertEqual(out, "out")
    all_to_all.assert_called_once_with(tensor, "context", split_axis=2, concat_axis=1, tiled=True)


if __name__ == "__main__":
  absltest.main()
