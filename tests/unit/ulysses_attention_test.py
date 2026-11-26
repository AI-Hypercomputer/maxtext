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

  def test_validate_head_sharding_mqa_requires_replication_and_unsharded_kv_head(self):
    mesh = types.SimpleNamespace(shape={"context": 4, "tensor": 2})

    # Without KV-head replication (USP), MQA is rejected.
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
          attention_label="TPU USP attention",
      )

    # With KV-head replication (Ulysses), MQA is accepted when the single KV head is unsharded,
    # including when query heads are tensor-sharded.
    for q_axes in ((None, None, "context", None), (None, "tensor", "context", None)):
      ulysses_attention.validate_head_sharding(
          axis_names_q=q_axes,
          axis_names_kv=(None, None, "context", None),
          mesh=mesh,
          num_query_heads=16,
          num_kv_heads=1,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
          allow_kv_head_replication=True,
      )

    # Sharding the single MQA KV head or leaving local Q heads indivisible by Ulysses size is rejected.
    with self.assertRaisesRegex(ValueError, "divisible by KV head shards|single MQA KV head"):
      ulysses_attention.validate_head_sharding(
          axis_names_q=(None, "tensor", "context", None),
          axis_names_kv=(None, "tensor", "context", None),
          mesh=mesh,
          num_query_heads=16,
          num_kv_heads=1,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
          allow_kv_head_replication=True,
      )
    with self.assertRaisesRegex(ValueError, "single MQA KV head"):
      ulysses_attention.validate_head_sharding(
          axis_names_q=(None, "tensor", "context", None),
          axis_names_kv=(None, "tensor", "context", None),
          mesh=types.SimpleNamespace(shape={"context": 4, "tensor": 1}),
          num_query_heads=16,
          num_kv_heads=1,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
          allow_kv_head_replication=True,
      )
    with self.assertRaisesRegex(ValueError, r"local query heads \(6\) to be divisible by the Ulysses exchange size"):
      ulysses_attention.validate_head_sharding(
          axis_names_q=(None, None, "context", None),
          axis_names_kv=(None, None, "context", None),
          mesh=mesh,
          num_query_heads=6,
          num_kv_heads=1,
          head_dim_q=1,
          head_dim_kv=1,
          ulysses_size=4,
          attention_label="TPU Ulysses attention",
          allow_kv_head_replication=True,
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
        ValueError,
        r"local KV heads \(2\) to be divisible by the Ulysses exchange size \(4\)\.$",
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
          **{**kwargs, "num_query_heads": 36, "num_kv_heads": 6},
          allow_kv_head_replication=True,
      )

  def test_kv_head_replication(self):
    cases = {(2, 4): 2, (2, 8): 4, (1, 4): 4, (4, 4): 1, (8, 4): 1, (16, 2): 1}
    for (local_kv_heads, ulysses_size), expected in cases.items():
      with self.subTest(local_kv_heads=local_kv_heads, ulysses_size=ulysses_size):
        self.assertEqual(
            ulysses_attention.kv_head_replication(local_kv_heads, ulysses_size),
            expected,
        )
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
        np.testing.assert_array_equal(
            repeated[0, :, 0, 0],
            np.repeat(np.arange(num_kv_heads), replication),
        )
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

  def test_ulysses_all_to_all_qkv_packs_valid_inputs_and_falls_back_on_rank_or_shape_mismatch(self):
    """Packs 4D matching QKV into one exchange and falls back safely on rank < 4 or shape mismatch."""

    def fake_all_to_all(x, axis):
      self.assertEqual(axis, "context")
      if x.ndim != 4:
        return ("fallback", x.shape)
      return x.reshape(x.shape[0], x.shape[1] // 2, x.shape[2] * 2, x.shape[3])

    q = jnp.arange(1 * 8 * 4 * 2, dtype=jnp.float32).reshape(1, 8, 4, 2)
    k = jnp.arange(1 * 2 * 4 * 2, dtype=jnp.float32).reshape(1, 2, 4, 2)
    v = (k + 100.0).astype(jnp.float32)

    with mock.patch.object(ulysses_attention, "ulysses_all_to_all", side_effect=fake_all_to_all) as a2a:
      q_out, k_out, v_out = ulysses_attention.ulysses_all_to_all_qkv(q, k, v, "context", ulysses_size=2)
      self.assertEqual(a2a.call_count, 1)
      self.assertEqual(a2a.call_args[0][0].shape, (1, 12, 4, 2))
    self.assertEqual(q_out.shape, (1, 4, 8, 2))
    self.assertEqual(k_out.shape, (1, 1, 8, 2))
    self.assertEqual(v_out.shape, (1, 1, 8, 2))

    fallback_cases = (
        (q, jnp.ones((4,)), jnp.ones((4,))),
        (q, jnp.ones((1, 4)), jnp.ones((1, 4))),
        (q, jnp.ones((1, 2, 4)), jnp.ones((1, 2, 4))),
        (jnp.ones((1, 8, 4)), k, v),
        (q, jnp.ones((1, 2, 6, 2), dtype=jnp.float32), jnp.ones((1, 2, 6, 2), dtype=jnp.float32)),
    )
    for q_bad, k_bad, v_bad in fallback_cases:
      with self.subTest(q_shape=q_bad.shape, k_shape=k_bad.shape):
        with mock.patch.object(ulysses_attention, "ulysses_all_to_all", side_effect=fake_all_to_all) as a2a:
          out = ulysses_attention.ulysses_all_to_all_qkv(q_bad, k_bad, v_bad, "context", ulysses_size=2)
          self.assertEqual(a2a.call_count, 3)
          self.assertLen(out, 3)

  def test_attention_call_pins_row_major_layout_only_for_train_ulysses_without_shared_kv(self):
    """Attention.__call__ pins inputs_q, inputs_kv, and out (+ cotangents) iff train + ulysses + not use_shared_kv."""
    # pylint: disable=import-outside-toplevel
    import sys
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.layers import attentions
    from maxtext.models import qwen3
    from maxtext.utils import maxtext_utils
    from maxtext.utils import sharding
    from tests.utils.test_helpers import get_test_config_path

    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="ulysses_attention_layout_pin_test",
        enable_checkpointing=False,
        log_config=False,
        skip_jax_distributed_system=True,
        attention="dot_product",
        context_parallel_load_balance=False,
        base_emb_dim=32,
        base_num_query_heads=4,
        base_num_kv_heads=2,
        head_dim=8,
        max_target_length=16,
        max_prefill_predict_length=8,
        per_device_batch_size=1.0,
        dtype="float32",
        weight_dtype="float32",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    batch, seq_len, emb_dim = len(jax.devices()), 8, 32

    def make_attn(mode):
      with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(mesh):
        return attentions.Attention(
            config=cfg,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            max_target_length=16,
            mesh=mesh,
            attention_kernel="dot_product",
            inputs_q_shape=(batch, seq_len, emb_dim),
            inputs_kv_shape=(batch, seq_len, emb_dim),
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            max_prefill_predict_length=8,
            model_mode=mode,
            rngs=nnx.Rngs(params=0, dropout=1),
        )

    attn = make_attn(MODEL_MODE_TRAIN)
    attn_prefill = make_attn(MODEL_MODE_PREFILL)

    rng = np.random.default_rng(0)
    x_q = jnp.asarray(rng.standard_normal((batch, seq_len, emb_dim), dtype=np.float32))
    x_kv = jnp.asarray(rng.standard_normal((batch, seq_len, emb_dim), dtype=np.float32))
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
    seg_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
    expected_major_to_minor = (0, 1, 2)
    real_apply_layout = sharding.apply_layout_constraint

    # Verify both Attention and qwen3 route through the shared sharding.pin_row_major_with_cotangent helper.
    self.assertIs(attentions.pin_row_major_with_cotangent, sharding.pin_row_major_with_cotangent)
    self.assertIs(qwen3.pin_row_major_with_cotangent, sharding.pin_row_major_with_cotangent)
    gdn_layouts = []
    x_4d = jnp.ones((2, 4, 2, 8), dtype=jnp.float32)
    with mock.patch.object(
        sharding,
        "apply_layout_constraint",
        side_effect=lambda v, m2m: (gdn_layouts.append((v.shape, m2m)) or real_apply_layout(v, m2m)),
    ):
      _ = jax.jit(jax.grad(lambda z: jnp.sum(qwen3._pin_gdn_gated_norm_layout(z, "natural_fwd_bwd") ** 2)))(x_4d)  # pylint: disable=protected-access
    self.assertEqual(gdn_layouts, [((2, 4, 2, 8), (0, 1, 2, 3)), ((2, 4, 2, 8), (0, 1, 2, 3))])

    def run_case(
        strategy,
        model_mode,
        cross_attn=False,
        use_shared_kv=False,
        pin_mode="always",
        num_experts=1,
    ):
      target_attn = attn if model_mode == MODEL_MODE_TRAIN else attn_prefill
      recorded_layouts = []
      proj_inputs = {}
      real_q_proj = target_attn.query_projection
      real_kv_proj = target_attn.kv_projection

      def spy_layout(val, major_to_minor):
        recorded_layouts.append((val.shape, major_to_minor))
        return real_apply_layout(val, major_to_minor)

      def spy_q_proj(inp, *args, **kwargs):
        proj_inputs["q"] = inp
        return real_q_proj(inp, *args, **kwargs)

      def spy_kv_proj(inp, *args, **kwargs):
        if "kv" not in proj_inputs:
          proj_inputs["kv"] = inp
        return real_kv_proj(inp, *args, **kwargs)

      shared_k = jnp.ones((batch, seq_len, 2, 8), dtype=jnp.float32) if use_shared_kv else None
      shared_v = jnp.ones((batch, seq_len, 2, 8), dtype=jnp.float32) if use_shared_kv else None

      with (
          nn_partitioning.axis_rules(cfg.logical_axis_rules),
          jax.set_mesh(mesh),
          mock.patch.dict(
              cfg.get_keys(),
              {"ulysses_pin_row_major_layout": pin_mode, "num_experts": num_experts},
          ),
          mock.patch.object(attentions, "resolve_context_parallel_strategy", return_value=strategy),
          mock.patch.object(sharding, "apply_layout_constraint", side_effect=spy_layout),
          mock.patch.object(target_attn, "query_projection", side_effect=spy_q_proj),
          mock.patch.object(target_attn, "kv_projection", side_effect=spy_kv_proj),
      ):
        if model_mode != MODEL_MODE_TRAIN:
          gdef, state = nnx.split(target_attn)
          out, _ = jax.jit(
              lambda st, q: nnx.merge(gdef, st)(
                  q,
                  q,
                  positions,
                  seg_ids,
                  model_mode=model_mode,
                  deterministic=True,
              )
          )(state, x_q)
          return out, None, recorded_layouts, proj_inputs
        if cross_attn:
          loss, grads = jax.jit(
              jax.value_and_grad(
                  lambda q, kv: jnp.sum(
                      attn(q, kv, positions, seg_ids, model_mode=model_mode, deterministic=True)[0] ** 2
                  ),
                  argnums=(0, 1),
              )
          )(x_q, x_kv)
          return loss, grads, recorded_layouts, proj_inputs
        loss, grads = jax.jit(
            jax.value_and_grad(
                lambda q: jnp.sum(
                    attn(
                        q,
                        q,
                        positions,
                        seg_ids,
                        model_mode=model_mode,
                        deterministic=True,
                        shared_key=shared_k,
                        shared_value=shared_v,
                    )[0]
                    ** 2
                )
            )
        )(x_q)
        return loss, grads, recorded_layouts, proj_inputs

    # 1. Self-attention in train + ulysses: 2 forward pins (inputs_q, out) + 2 backward cotangent pins = 4.
    loss_u, grad_u, layouts_u, projs_u = run_case("ulysses", MODEL_MODE_TRAIN)
    self.assertLen(layouts_u, 4)
    for shape, m2m in layouts_u:
      self.assertEqual(shape, (batch, seq_len, emb_dim))
      self.assertEqual(m2m, expected_major_to_minor)
    self.assertIs(projs_u["q"], projs_u["kv"])

    # 2. Cross-attention in train + ulysses: 3 forward pins (inputs_q, inputs_kv, out) + 3 backward cotangent pins = 6.
    _, _, layouts_cross, projs_cross = run_case("ulysses", MODEL_MODE_TRAIN, cross_attn=True)
    self.assertLen(layouts_cross, 6)
    self.assertIsNot(projs_cross["q"], projs_cross["kv"])

    # 3. Ring strategy, prefill mode, and shared KV must not pin layouts, and ring matches ulysses values/grads.
    loss_ring, grad_ring, layouts_ring, _ = run_case("ring", MODEL_MODE_TRAIN)
    self.assertEmpty(layouts_ring)
    np.testing.assert_allclose(loss_u, loss_ring, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_u, grad_ring, rtol=1e-5, atol=1e-5)

    _, _, layouts_prefill, _ = run_case("ulysses", MODEL_MODE_PREFILL)
    self.assertEmpty(layouts_prefill)

    _, _, layouts_shared_kv, _ = run_case("ulysses", MODEL_MODE_TRAIN, use_shared_kv=True)
    self.assertEmpty(layouts_shared_kv)

    # 4. ulysses_pin_row_major_layout='auto' enables on MoE (num_experts > 1) and skips on dense (num_experts == 1),
    # while 'never' disables even on MoE and invalid values raise ValueError.
    _, _, layouts_auto_moe, _ = run_case("ulysses", MODEL_MODE_TRAIN, pin_mode="auto", num_experts=8)
    self.assertLen(layouts_auto_moe, 4)

    _, _, layouts_auto_dense, _ = run_case("ulysses", MODEL_MODE_TRAIN, pin_mode="auto", num_experts=1)
    self.assertEmpty(layouts_auto_dense)

    _, _, layouts_never_moe, _ = run_case("ulysses", MODEL_MODE_TRAIN, pin_mode="never", num_experts=8)
    self.assertEmpty(layouts_never_moe)

    with self.assertRaisesRegex(ValueError, "ulysses_pin_row_major_layout"):
      run_case("ulysses", MODEL_MODE_TRAIN, pin_mode="bogus", num_experts=8)

  def test_custom_remat_ulysses_qkv_device_and_offload_saves_post_exchange_qkv(self):
    """Custom remat with ulysses_qkv='device' or 'offload' skips replaying the forward QKV exchange."""
    # pylint: disable=import-outside-toplevel
    import functools
    import sys
    from jax.ad_checkpoint import checkpoint_name
    from maxtext.configs import pyconfig
    from maxtext.layers import nnx_decoders
    from maxtext.utils import maxtext_utils
    from tests.utils.test_helpers import get_test_config_path

    def make_decoder_policy(ulysses_qkv_loc: str):
      cfg = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name=f"ulysses_qkv_remat_{ulysses_qkv_loc}",
          enable_checkpointing=False,
          log_config=False,
          skip_jax_distributed_system=True,
          remat_policy="custom",
          ulysses_qkv=ulysses_qkv_loc,
      )
      save_names, offload_names = maxtext_utils.get_save_and_offload_names(cfg)
      dummy_decoder = types.SimpleNamespace(
          config=cfg,
          remat_policy_override=None,
          _effective_remat_policy=lambda: cfg.remat_policy,
      )
      policy = nnx_decoders.NNXDecoder.get_remat_policy(dummy_decoder)
      return save_names, offload_names, policy

    save_remat, off_remat, pol_remat = make_decoder_policy("remat")
    save_dev, off_dev, pol_dev = make_decoder_policy("device")
    save_off, off_off, pol_off = make_decoder_policy("offload")
    self.assertNotIn("ulysses_qkv", save_remat + off_remat)
    self.assertIn("ulysses_qkv", save_dev)
    self.assertNotIn("ulysses_qkv", off_dev)
    self.assertNotIn("ulysses_qkv", save_off)
    self.assertIn("ulysses_qkv", off_off)

    # Trace a checkpointed Ulysses attention block with KV-head replication (2 KV heads -> CP=4:
    # pre-exchange K/V [B, 2, S/4, D] vs saved post-exchange K/V [B, 1, S, D], 2x larger).
    b, hq, hkv_local, s_shard, d, ulysses_size = 1, 8, 2, 4, 8, 4
    q_in = jnp.linspace(-0.5, 0.5, b * hq * s_shard * d, dtype=jnp.float32).reshape(b, hq, s_shard, d)
    k_in = jnp.linspace(-0.3, 0.3, b * hkv_local * s_shard * d, dtype=jnp.float32).reshape(b, hkv_local, s_shard, d)
    v_in = jnp.linspace(-0.2, 0.4, b * hkv_local * s_shard * d, dtype=jnp.float32).reshape(b, hkv_local, s_shard, d)

    def simulated_a2a(x, _axis):
      # Simulates the forward Ulysses exchange [B, H, S/U, D] -> [B, H/U, S, D] using sin so we can count replays.
      h_new = x.shape[1] // ulysses_size
      s_full = x.shape[2] * ulysses_size
      return jnp.sin(x).reshape(x.shape[0], h_new, s_full, x.shape[3])

    def run_with_policy(policy):
      @functools.partial(jax.checkpoint, policy=policy)
      def block(q, k, v):
        rep = ulysses_attention.kv_head_replication(k.shape[1], ulysses_size)
        k_rep = ulysses_attention.repeat_kv_heads_for_ulysses(k, rep)
        v_rep = ulysses_attention.repeat_kv_heads_for_ulysses(v, rep)
        with mock.patch.object(ulysses_attention, "ulysses_all_to_all", side_effect=simulated_a2a):
          q_ex, k_ex, v_ex = ulysses_attention.ulysses_all_to_all_qkv(q, k_rep, v_rep, "context", ulysses_size)
        # Post-exchange K/V have shape [B, 1, S, D] = 2x the pre-exchange [B, 2, S/4, D] size at CP=4.
        self.assertEqual(k_ex.shape, (b, 1, s_shard * ulysses_size, d))
        self.assertEqual(k_ex.size, 2 * k.size)
        q_ex = checkpoint_name(q_ex, "ulysses_qkv")
        k_ex = checkpoint_name(k_ex, "ulysses_qkv")
        v_ex = checkpoint_name(v_ex, "ulysses_qkv")
        group = q_ex.shape[1] // k_ex.shape[1]
        k_b = jnp.repeat(k_ex, group, axis=1)
        v_b = jnp.repeat(v_ex, group, axis=1)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q_ex, k_b, precision="highest")
        return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, axis=-1), v_b, precision="highest")

      grad_fn = jax.grad(lambda q, k, v: jnp.sum(block(q, k, v) ** 2), argnums=(0, 1, 2))
      jaxpr = jax.make_jaxpr(grad_fn)(q_in, k_in, v_in)
      grads = jax.jit(grad_fn)(q_in, k_in, v_in)
      return jaxpr, grads

    jaxpr_remat, grads_remat = run_with_policy(pol_remat)
    jaxpr_dev, grads_dev = run_with_policy(pol_dev)
    jaxpr_off, grads_off = run_with_policy(pol_off)

    def count_remat_sins(closed_jaxpr):
      for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive.name == "remat2":
          return sum(1 for sub_eqn in eqn.params["jaxpr"].eqns if sub_eqn.primitive.name == "sin")
      return 0

    def count_host_offloads(closed_jaxpr):
      return sum(
          1
          for eqn in closed_jaxpr.jaxpr.eqns
          if eqn.primitive.name == "device_put" and "Host" in str(eqn.params.get("devices"))
      )

    # Under 'remat', remat2 replays simulated_a2a's sin + its JVP cos (1 sin inside remat2);
    # under 'device' and 'offload', post-exchange Q/K/V are saved/offloaded so remat2 replays 0 sins!
    self.assertGreater(count_remat_sins(jaxpr_remat), 0)
    self.assertEqual(count_remat_sins(jaxpr_dev), 0)
    self.assertEqual(count_remat_sins(jaxpr_off), 0)
    self.assertEqual(count_host_offloads(jaxpr_dev), 0)
    self.assertEqual(count_host_offloads(jaxpr_off), 3)  # Q, K, V offloaded to pinned_host

    for g_r, g_d, g_o in zip(grads_remat, grads_dev, grads_off):
      np.testing.assert_allclose(g_d, g_r, rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(g_o, g_r, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
