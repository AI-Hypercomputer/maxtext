# Copyright 2023–2026 Google LLC
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

"""Unit tests for lineage_adapter module."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import types as config_types
from maxtext.models import lineage_adapter
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops
import numpy as np


class LineageAdapterTest(parameterized.TestCase):
  """Tests for lineage_adapter functionality."""

  def test_fetch_lineage_sparse_weights_unfused(self):
    num_layers = 2
    emb_dim = 64
    cq_dim = 16
    ckv_dim = 16
    num_query_heads = 4
    qk_nope_head_dim = 16
    qk_rope_head_dim = 8
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 16
    num_experts = 4
    expert_hidden_dim = 32

    params = {
        "pre_self_attention_layer_norm": {
            "scale": jnp.ones((num_layers, emb_dim), dtype=jnp.float32),
        },
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((num_layers, emb_dim, cq_dim))},
            "wq_b": {"kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim))},
            "q_norm": {"scale": jnp.ones((num_layers, cq_dim))},
            "wkv_a": {"kernel": jnp.ones((num_layers, emb_dim, ckv_dim + qk_rope_head_dim))},
            "wkv_b": {
                "kernel": jnp.ones(
                    (
                        num_layers,
                        ckv_dim,
                        num_query_heads,
                        qk_nope_head_dim + v_head_dim,
                    )
                )
            },
            "kv_norm": {"scale": jnp.ones((num_layers, ckv_dim))},
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {
            "scale": jnp.ones((num_layers, emb_dim)),
        },
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {
                "gate": {
                    "kernel": jnp.ones((num_layers, emb_dim, num_experts)),
                    "bias": jnp.zeros((num_layers, num_experts)),
                },
                "wi_0": jnp.ones((num_layers, num_experts, emb_dim, expert_hidden_dim)),
                "wi_1": jnp.ones((num_layers, num_experts, emb_dim, expert_hidden_dim)),
                "wo": jnp.ones((num_layers, num_experts, expert_hidden_dim, emb_dim)),
            },
            "shared_experts": {
                "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, expert_hidden_dim))},
                "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, expert_hidden_dim))},
                "wo": {"kernel": jnp.ones((num_layers, expert_hidden_dim, emb_dim))},
            },
        },
    }

    weights = lineage_adapter.fetch_lineage_sparse_weights(params, dtype=jnp.bfloat16)

    self.assertIsInstance(weights, dsv3_types.DSv3SparseLayerWeightsPytree)
    self.assertIsNotNone(weights.pre_attn_norm_scale)
    assert weights.pre_attn_norm_scale is not None
    self.assertEqual(weights.pre_attn_norm_scale.shape, (num_layers, emb_dim))
    self.assertEqual(weights.pre_attn_norm_scale.dtype, jnp.bfloat16)

    self.assertIsNotNone(weights.mla.q_down)
    assert weights.mla.q_down is not None
    self.assertEqual(weights.mla.q_down.shape, (num_layers, emb_dim, cq_dim))

    self.assertIsNotNone(weights.mla.q_up)
    assert weights.mla.q_up is not None
    self.assertEqual(
        weights.mla.q_up.shape,
        (num_layers, cq_dim, num_query_heads, qk_head_dim),
    )

    self.assertIsNotNone(weights.mla.k_up)
    assert weights.mla.k_up is not None
    self.assertEqual(
        weights.mla.k_up.shape,
        (num_layers, ckv_dim, num_query_heads, qk_nope_head_dim),
    )

    self.assertIsNotNone(weights.mla.v_up)
    assert weights.mla.v_up is not None
    self.assertEqual(
        weights.mla.v_up.shape,
        (num_layers, ckv_dim, num_query_heads, v_head_dim),
    )

    self.assertIsNotNone(weights.moe.router.kernel)
    assert weights.moe.router.kernel is not None
    self.assertEqual(weights.moe.router.kernel.shape, (num_layers, emb_dim, num_experts))

    self.assertIsNotNone(weights.moe.router.bias)
    assert weights.moe.router.bias is not None
    self.assertEqual(weights.moe.router.bias.shape, (num_layers, num_experts))

    # Routed gate should concatenate wi_0 and wi_1 along axis -1
    self.assertIsNotNone(weights.moe.routed.gate)
    assert weights.moe.routed.gate is not None
    self.assertEqual(
        weights.moe.routed.gate.shape,
        (num_layers, num_experts, emb_dim, 2 * expert_hidden_dim),
    )

    self.assertIsNotNone(weights.moe.routed.linear)
    assert weights.moe.routed.linear is not None
    self.assertEqual(
        weights.moe.routed.linear.shape,
        (num_layers, num_experts, expert_hidden_dim, emb_dim),
    )

    self.assertIsNotNone(weights.moe.shared.gate_0)
    assert weights.moe.shared.gate_0 is not None
    self.assertEqual(
        weights.moe.shared.gate_0.shape,
        (num_layers, emb_dim, expert_hidden_dim),
    )

    self.assertIsNotNone(weights.moe.shared.linear)
    assert weights.moe.shared.linear is not None
    self.assertEqual(
        weights.moe.shared.linear.shape,
        (num_layers, expert_hidden_dim, emb_dim),
    )

  def test_fetch_lineage_sparse_weights_none_params(self):
    with self.assertRaises(ValueError):
      lineage_adapter.fetch_lineage_sparse_weights(None)

  def test_fetch_lineage_sparse_weights_param_scan_axis_1(self):
    num_layers = 2
    emb_dim = 32
    num_experts = 4
    expert_hidden_dim = 16

    # When param_scan_axis=1, layer axis is at index 1 instead of 0 in the
    # input.
    params = {
        "pre_self_attention_layer_norm": {"scale": jnp.ones((emb_dim, num_layers))},
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((emb_dim, num_layers, 8))},
            "wq_b": {"kernel": jnp.ones((8, num_layers, 2, 12))},
            "q_norm": {"scale": jnp.ones((8, num_layers))},
            "wkv_a": {"kernel": jnp.ones((emb_dim, num_layers, 12))},
            "wkv_b": {"kernel": jnp.ones((8, num_layers, 2, 16))},
            "kv_norm": {"scale": jnp.ones((8, num_layers))},
            "out": {"kernel": jnp.ones((2, num_layers, 8, emb_dim))},
        },
        "post_self_attention_layer_norm": {"scale": jnp.ones((emb_dim, num_layers))},
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {
                "gate": {
                    "kernel": jnp.ones((emb_dim, num_layers, num_experts)),
                    "bias": jnp.ones((num_experts, num_layers)),
                },
                "wi_0": jnp.ones((num_experts, num_layers, emb_dim, expert_hidden_dim)),
                "wi_1": jnp.ones((num_experts, num_layers, emb_dim, expert_hidden_dim)),
                "wo": jnp.ones((num_experts, num_layers, expert_hidden_dim, emb_dim)),
            },
            "shared_experts": {
                "wi_0": jnp.ones((emb_dim, num_layers, expert_hidden_dim)),
                "wi_1": jnp.ones((emb_dim, num_layers, expert_hidden_dim)),
                "wo": jnp.ones((expert_hidden_dim, num_layers, emb_dim)),
            },
        },
    }

    weights = lineage_adapter.fetch_lineage_sparse_weights(params, param_scan_axis=1)

    self.assertIsNotNone(weights.pre_attn_norm_scale)
    assert weights.pre_attn_norm_scale is not None
    self.assertEqual(weights.pre_attn_norm_scale.shape, (num_layers, emb_dim))

    self.assertIsNotNone(weights.moe.router.kernel)
    assert weights.moe.router.kernel is not None
    self.assertEqual(
        weights.moe.router.kernel.shape,
        (num_layers, emb_dim, num_experts),
    )

    self.assertIsNotNone(weights.moe.router.bias)
    assert weights.moe.router.bias is not None
    self.assertEqual(
        weights.moe.router.bias.shape,
        (num_layers, num_experts),
    )
    self.assertIsNotNone(weights.moe.routed.gate)
    assert weights.moe.routed.gate is not None
    self.assertEqual(
        weights.moe.routed.gate.shape,
        (num_layers, num_experts, emb_dim, 2 * expert_hidden_dim),
    )

  def test_fetch_lineage_sparse_weights_param_scan_axis_1_sharded(self):
    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(
        devices.reshape((1, 1)),
        ("data", "fsdp"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "fsdp"))
    arr = jax.device_put(jnp.ones((32, 2)), sharding)
    params = {
        "pre_self_attention_layer_norm": {
            "scale": arr,
        },
    }
    weights = lineage_adapter.fetch_lineage_sparse_weights(params, param_scan_axis=1)
    self.assertIsNotNone(weights.pre_attn_norm_scale)
    assert weights.pre_attn_norm_scale is not None
    self.assertEqual(weights.pre_attn_norm_scale.shape, (2, 32))
    scale_sharding = getattr(weights.pre_attn_norm_scale, "sharding", None)
    self.assertIsNotNone(scale_sharding)
    self.assertEqual(scale_sharding.spec, jax.sharding.PartitionSpec("fsdp", None))

  def test_fetch_lineage_sparse_weights_param_scan_axis_1_short_sharding(self):
    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(
        devices.reshape((1, 1)),
        ("data", "fsdp"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    # 1-element PartitionSpec on a 2D array: axis 0 is "fsdp", axis 1 is
    # implicitly None.
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
    arr = jax.device_put(jnp.ones((32, 2)), sharding)
    params = {
        "pre_self_attention_layer_norm": {
            "scale": arr,
        },
    }
    weights = lineage_adapter.fetch_lineage_sparse_weights(params, param_scan_axis=1)
    self.assertIsNotNone(weights.pre_attn_norm_scale)
    assert weights.pre_attn_norm_scale is not None
    self.assertEqual(weights.pre_attn_norm_scale.shape, (2, 32))
    scale_sharding = getattr(weights.pre_attn_norm_scale, "sharding", None)
    self.assertIsNotNone(scale_sharding)
    self.assertEqual(scale_sharding.spec, jax.sharding.PartitionSpec(None, "fsdp"))

  def test_fetch_lineage_sparse_weights_nested_mapping_value(self):
    class BoxedValue:

      def __init__(self, value):
        self.value = value

    params = {
        "pre_self_attention_layer_norm": {
            "scale": {"value": BoxedValue(jnp.ones((2, 16)))},
        },
    }
    weights = lineage_adapter.fetch_lineage_sparse_weights(params)
    self.assertIsNotNone(weights.pre_attn_norm_scale)
    assert weights.pre_attn_norm_scale is not None
    self.assertEqual(weights.pre_attn_norm_scale.shape, (2, 16))

  def test_build_axis_mapping(self):
    devices = np.array(jax.devices()[:1])
    mesh_physical = jax.sharding.Mesh(
        devices.reshape((1, 1, 1, 1)),
        ("x", "y", "z", "core"),
    )
    mapping_physical = lineage_adapter.build_axis_mapping(mesh_physical)
    self.assertEqual(mapping_physical["attention"], "core")
    self.assertEqual(mapping_physical["fsdp_attention"], ("x", "y", "z"))
    self.assertEqual(mapping_physical["expert"], ("x", "y", "core"))
    self.assertEqual(mapping_physical["fsdp_moe"], "z")
    self.assertEqual(mapping_physical["z"], "z")

  def test_ops_collect_along_axis_with_derived_mapping(self):
    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(
        devices.reshape((1, 1, 1, 1)),
        ("x", "y", "z", "core"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    axis_mapping = lineage_adapter.build_axis_mapping(mesh)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("z"))
    x = jax.device_put(jnp.ones((4, 4)), sharding)
    out = ops.collect_along_axis(x, "fsdp_moe", axis_mapping)
    self.assertIsNotNone(out)
    self.assertEqual(out.shape, (4, 4))

  def test_get_capacity_factor(self):
    cfg_standard = types.SimpleNamespace(capacity_factor=0.5)
    self.assertEqual(lineage_adapter.get_capacity_factor(cfg_standard), 0.5)

    cfg_legacy = types.SimpleNamespace(lineage_capacity_factor=0.5)
    self.assertEqual(lineage_adapter.get_capacity_factor(cfg_legacy), 0.5)

    # Standard capacity_factor takes precedence over legacy
    # lineage_capacity_factor.
    cfg_both = types.SimpleNamespace(capacity_factor=0.75, lineage_capacity_factor=0.25)
    self.assertEqual(lineage_adapter.get_capacity_factor(cfg_both), 0.75)

    with self.assertRaises(ValueError):
      lineage_adapter.get_capacity_factor(None)

    with self.assertRaises(ValueError):
      lineage_adapter.get_capacity_factor(types.SimpleNamespace())

    with self.assertRaises(ValueError):
      lineage_adapter.get_capacity_factor(types.SimpleNamespace(capacity_factor=-1.0))

    with self.assertRaises(ValueError):
      lineage_adapter.get_capacity_factor(types.SimpleNamespace(capacity_factor=0.0))

  def test_compute_effective_capacity_factor(self):
    devices = np.array([jax.devices()[0]])
    mesh = jax.sharding.Mesh(devices.reshape((1, 1, 1, 1)), ("data", "fsdp", "expert", "context"))
    axis_mapping = {
        "attention": "context",
        "fsdp_attention": ("data", "fsdp", "expert"),
        "expert": "expert",
        "fsdp_moe": "fsdp",
    }
    inputs = jnp.ones((1, 256, 64))
    # Base factor 0.5 with 256 tokens gives 128 raw, which should be raised
    # to min_capacity (1024).
    factor = lineage_adapter.compute_effective_capacity_factor(
        inputs, mesh, axis_mapping, base_capacity_factor=0.5, min_capacity=1024
    )
    self.assertGreaterEqual(factor * 256, 1024.0)
    self.assertEqual(int(factor * 256), 1024)

  def test_build_axis_mapping_logical_axis_rules_explicit(self):
    cfg_rule = types.SimpleNamespace(
        logical_axis_rules=[
            ["activation_length", ["core"]],
            ["activation_batch_attn", ["x", "y", "z"]],
            ["exp", ["x", "y", "core"]],
            ["embed_moe", ["z"]],
        ]
    )
    mapping = lineage_adapter.build_axis_mapping(None, cfg_rule)
    self.assertEqual(mapping["attention"], "core")
    self.assertEqual(mapping["fsdp_attention"], ("x", "y", "z"))
    self.assertEqual(mapping["expert"], ("x", "y", "core"))
    self.assertEqual(mapping["fsdp_moe"], "z")
    self.assertEqual(mapping["z"], "z")

  def test_compute_effective_capacity_factor_short_spec(self):
    devices = np.array([jax.devices()[0]])
    mesh = jax.sharding.Mesh(devices.reshape((1, 1, 1, 1)), ("x", "y", "z", "core"))
    axis_mapping = lineage_adapter.build_axis_mapping(mesh)
    # Test 1D array input with short PartitionSpec(None,)
    inputs_1d = jnp.ones((256,))
    factor = lineage_adapter.compute_effective_capacity_factor(
        inputs_1d,
        mesh,
        axis_mapping,
        base_capacity_factor=0.5,
        min_capacity=1024,
    )
    self.assertGreater(factor, 0.0)

  def test_compute_effective_capacity_factor_cp_as_ep(self):
    devices = np.array([jax.devices()[0]])
    mesh = jax.sharding.Mesh(devices.reshape((1, 1, 1, 1)), ("x", "y", "z", "core"))
    axis_mapping = lineage_adapter.build_axis_mapping(mesh)
    self.assertEqual(axis_mapping["expert"], ("x", "y", "core"))
    inputs = jnp.ones((1, 256, 64))
    factor = lineage_adapter.compute_effective_capacity_factor(
        inputs, mesh, axis_mapping, base_capacity_factor=0.5, min_capacity=1024
    )
    self.assertGreaterEqual(factor * 256, 1024.0)
    self.assertEqual(int(factor * 256), 1024)

  def test_ops_physical_pspec_physical_coordinates(self):
    mapping = {
        "attention": "core",
        "fsdp_attention": ("x", "y", "z"),
        "expert": ("x", "y", "core"),
        "fsdp_moe": "z",
    }
    logical_pspec = jax.sharding.PartitionSpec("fsdp_attention", "attention", None)
    expected_pspec = jax.sharding.PartitionSpec(("x", "y", "z"), "core", None)
    out_pspec = ops.physical_pspec(logical_pspec, mapping)
    self.assertEqual(out_pspec, expected_pspec)

  def test_ops_physical_pspec_single_axis_tuple(self):
    mapping = {"expert": ("expert", "context")}
    logical_pspec = jax.sharding.PartitionSpec("expert", None)
    expected_pspec = jax.sharding.PartitionSpec(("expert", "context"), None)
    out_pspec = ops.physical_pspec(logical_pspec, mapping)
    self.assertEqual(out_pspec, expected_pspec)

  def test_fetch_lineage_dense_weights_unfused(self):
    num_layers = 3
    emb_dim = 64
    cq_dim = 16
    ckv_dim = 16
    num_query_heads = 4
    qk_nope_head_dim = 16
    qk_rope_head_dim = 8
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 16
    mlp_dim = 32

    params = {
        "pre_self_attention_layer_norm": {
            "scale": jnp.ones((num_layers, emb_dim), dtype=jnp.float32),
        },
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((num_layers, emb_dim, cq_dim))},
            "wq_b": {"kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim))},
            "q_norm": {"scale": jnp.ones((num_layers, cq_dim))},
            "wkv_a": {"kernel": jnp.ones((num_layers, emb_dim, ckv_dim + qk_rope_head_dim))},
            "wkv_b": {
                "kernel": jnp.ones(
                    (
                        num_layers,
                        ckv_dim,
                        num_query_heads,
                        qk_nope_head_dim + v_head_dim,
                    )
                )
            },
            "kv_norm": {"scale": jnp.ones((num_layers, ckv_dim))},
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {
            "scale": jnp.ones((num_layers, emb_dim)),
        },
        "mlp": {
            "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wo": {"kernel": jnp.ones((num_layers, mlp_dim, emb_dim))},
        },
    }

    dense_weights = lineage_adapter.fetch_lineage_dense_weights(params, dtype=jnp.bfloat16)
    self.assertIsInstance(dense_weights, dsv3_types.DSv3DenseLayerWeightsPytree)
    self.assertIsNotNone(dense_weights.pre_attn_norm_scale)
    self.assertEqual(dense_weights.pre_attn_norm_scale.shape, (num_layers, emb_dim))
    self.assertEqual(dense_weights.pre_attn_norm_scale.dtype, jnp.bfloat16)
    self.assertIsNotNone(dense_weights.mla.q_down)
    self.assertEqual(dense_weights.mla.q_down.shape, (num_layers, emb_dim, cq_dim))
    self.assertIsNotNone(dense_weights.mla.q_up)
    self.assertEqual(
        dense_weights.mla.q_up.shape,
        (num_layers, cq_dim, num_query_heads, qk_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.k_up)
    self.assertEqual(
        dense_weights.mla.k_up.shape,
        (num_layers, ckv_dim, num_query_heads, qk_nope_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.v_up)
    self.assertEqual(
        dense_weights.mla.v_up.shape,
        (num_layers, ckv_dim, num_query_heads, v_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.out)
    self.assertEqual(
        dense_weights.mla.out.shape,
        (num_layers, num_query_heads, v_head_dim, emb_dim),
    )
    self.assertIsNotNone(dense_weights.mlp.gate_0)
    self.assertEqual(dense_weights.mlp.gate_0.shape, (num_layers, emb_dim, mlp_dim))
    self.assertIsNotNone(dense_weights.mlp.gate_1)
    self.assertEqual(dense_weights.mlp.gate_1.shape, (num_layers, emb_dim, mlp_dim))
    self.assertIsNotNone(dense_weights.mlp.linear)
    self.assertEqual(dense_weights.mlp.linear.shape, (num_layers, mlp_dim, emb_dim))

  def test_fetch_lineage_dense_weights_fused(self):
    num_layers = 3
    emb_dim = 64
    cq_dim = 16
    ckv_dim = 16
    num_query_heads = 4
    qk_nope_head_dim = 16
    qk_rope_head_dim = 8
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim = 16
    mlp_dim = 32

    params = {
        "pre_self_attention_layer_norm": {
            "scale": jnp.ones((num_layers, emb_dim)),
        },
        "self_attention": {
            "wq_a": {
                "kernel": jnp.ones((num_layers, emb_dim, cq_dim)),
            },
            "wq_b": {
                "kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim)),
            },
            "q_norm": {
                "scale": jnp.ones((num_layers, cq_dim)),
            },
            "wkv_a": {
                "kernel": jnp.ones((num_layers, emb_dim, ckv_dim + qk_rope_head_dim)),
            },
            "wkv_b": {
                "kernel": jnp.ones(
                    (
                        num_layers,
                        ckv_dim,
                        num_query_heads,
                        qk_nope_head_dim + v_head_dim,
                    )
                ),
            },
            "kv_norm": {
                "scale": jnp.ones((num_layers, ckv_dim)),
            },
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {
            "scale": jnp.ones((num_layers, emb_dim)),
        },
        "mlp": {
            "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wo": {"kernel": jnp.ones((num_layers, mlp_dim, emb_dim))},
        },
    }

    dense_weights = lineage_adapter.fetch_lineage_dense_weights(params, qk_head_dim=qk_nope_head_dim)
    self.assertIsInstance(dense_weights, dsv3_types.DSv3DenseLayerWeightsPytree)
    self.assertIsNotNone(dense_weights.mla.q_down)
    self.assertEqual(dense_weights.mla.q_down.shape, (num_layers, emb_dim, cq_dim))
    self.assertIsNotNone(dense_weights.mla.q_up)
    self.assertEqual(
        dense_weights.mla.q_up.shape,
        (num_layers, cq_dim, num_query_heads, qk_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.k_up)
    self.assertEqual(
        dense_weights.mla.k_up.shape,
        (num_layers, ckv_dim, num_query_heads, qk_nope_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.v_up)
    self.assertEqual(
        dense_weights.mla.v_up.shape,
        (num_layers, ckv_dim, num_query_heads, v_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.out)
    self.assertEqual(
        dense_weights.mla.out.shape,
        (num_layers, num_query_heads, v_head_dim, emb_dim),
    )
    self.assertIsNotNone(dense_weights.mlp.gate_0)
    self.assertEqual(dense_weights.mlp.gate_0.shape, (num_layers, emb_dim, mlp_dim))
    self.assertIsNotNone(dense_weights.mlp.gate_1)
    self.assertEqual(dense_weights.mlp.gate_1.shape, (num_layers, emb_dim, mlp_dim))
    self.assertIsNotNone(dense_weights.mlp.linear)
    self.assertEqual(dense_weights.mlp.linear.shape, (num_layers, mlp_dim, emb_dim))

  def test_fetch_lineage_dense_weights_none_params(self):
    with self.assertRaises(ValueError):
      lineage_adapter.fetch_lineage_dense_weights(None)

  def test_fetch_lineage_dense_weights_from_nnx_module(self):
    num_layers = 2
    emb_dim = 64
    cq_dim = 16
    ckv_dim = 16
    num_query_heads = 4
    qk_head_dim = 24
    v_head_dim = 16
    mlp_dim = 32

    class DummySubmodule(nnx.Module):

      def __init__(self, **kwargs):
        for k, v in kwargs.items():
          setattr(self, k, v)

    class DummyDenseStack(nnx.Module):
      """Dummy dense layer stack for testing."""

      def __init__(self):
        self.pre_self_attention_layer_norm = DummySubmodule(scale=nnx.Param(jnp.ones((num_layers, emb_dim))))
        self.self_attention = DummySubmodule(
            wq_a=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, emb_dim, cq_dim)))),
            wq_b=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim)))),
            q_norm=DummySubmodule(scale=nnx.Param(jnp.ones((num_layers, cq_dim)))),
            wkv_a=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, emb_dim, ckv_dim + 8)))),
            wkv_b=DummySubmodule(
                kernel=nnx.Param(
                    jnp.ones(
                        (
                            num_layers,
                            ckv_dim,
                            num_query_heads,
                            16 + v_head_dim,
                        )
                    )
                )
            ),
            kv_norm=DummySubmodule(scale=nnx.Param(jnp.ones((num_layers, ckv_dim)))),
            out=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim)))),
        )
        self.post_self_attention_layer_norm = DummySubmodule(scale=nnx.Param(jnp.ones((num_layers, emb_dim))))
        self.mlp = DummySubmodule(
            wi_0=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, emb_dim, mlp_dim)))),
            wi_1=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, emb_dim, mlp_dim)))),
            wo=DummySubmodule(kernel=nnx.Param(jnp.ones((num_layers, mlp_dim, emb_dim)))),
        )

    module = DummyDenseStack()
    dense_weights = lineage_adapter.fetch_lineage_dense_weights(module)
    self.assertIsInstance(dense_weights, dsv3_types.DSv3DenseLayerWeightsPytree)
    self.assertIsNotNone(dense_weights.mla.q_down)
    self.assertEqual(dense_weights.mla.q_down.shape, (num_layers, emb_dim, cq_dim))
    self.assertIsNotNone(dense_weights.mla.q_up)
    self.assertEqual(
        dense_weights.mla.q_up.shape,
        (num_layers, cq_dim, num_query_heads, qk_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.k_up)
    self.assertEqual(
        dense_weights.mla.k_up.shape,
        (num_layers, ckv_dim, num_query_heads, 16),
    )
    self.assertIsNotNone(dense_weights.mla.v_up)
    self.assertEqual(
        dense_weights.mla.v_up.shape,
        (num_layers, ckv_dim, num_query_heads, v_head_dim),
    )
    self.assertIsNotNone(dense_weights.mla.out)
    self.assertEqual(
        dense_weights.mla.out.shape,
        (num_layers, num_query_heads, v_head_dim, emb_dim),
    )
    self.assertIsNotNone(dense_weights.mlp.gate_0)

  def test_fetch_lineage_weights_combines_dense_and_sparse(self):
    num_layers = 2
    emb_dim = 64
    cq_dim = 16
    ckv_dim = 16
    num_query_heads = 4
    qk_head_dim = 24
    v_head_dim = 16
    mlp_dim = 32
    num_experts = 4
    expert_hidden_dim = 32

    dense_params = {
        "pre_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((num_layers, emb_dim, cq_dim))},
            "wq_b": {"kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim))},
            "q_norm": {"scale": jnp.ones((num_layers, cq_dim))},
            "wkv_a": {"kernel": jnp.ones((num_layers, emb_dim, ckv_dim + 8))},
            "wkv_b": {"kernel": jnp.ones((num_layers, ckv_dim, num_query_heads, 16 + v_head_dim))},
            "kv_norm": {"scale": jnp.ones((num_layers, ckv_dim))},
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "mlp": {
            "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wo": {"kernel": jnp.ones((num_layers, mlp_dim, emb_dim))},
        },
    }
    sparse_params = {
        "pre_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((num_layers, emb_dim, cq_dim))},
            "wq_b": {"kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim))},
            "q_norm": {"scale": jnp.ones((num_layers, cq_dim))},
            "wkv_a": {"kernel": jnp.ones((num_layers, emb_dim, ckv_dim + 8))},
            "wkv_b": {"kernel": jnp.ones((num_layers, ckv_dim, num_query_heads, 16 + v_head_dim))},
            "kv_norm": {"scale": jnp.ones((num_layers, ckv_dim))},
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {
                "gate": {
                    "kernel": jnp.ones((num_layers, emb_dim, num_experts)),
                    "bias": jnp.zeros((num_layers, num_experts)),
                },
                "wi_0": jnp.ones((num_layers, num_experts, emb_dim, expert_hidden_dim)),
                "wi_1": jnp.ones((num_layers, num_experts, emb_dim, expert_hidden_dim)),
                "wo": jnp.ones((num_layers, num_experts, expert_hidden_dim, emb_dim)),
            },
            "shared_experts": {
                "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, expert_hidden_dim))},
                "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, expert_hidden_dim))},
                "wo": {"kernel": jnp.ones((num_layers, expert_hidden_dim, emb_dim))},
            },
        },
    }

    full_weights = lineage_adapter.fetch_lineage_weights(dense_params, sparse_params)
    self.assertIsInstance(full_weights, dsv3_types.DSv3WeightsPytree)
    self.assertIsInstance(full_weights.dense, dsv3_types.DSv3DenseLayerWeightsPytree)
    self.assertIsInstance(full_weights.sparse, dsv3_types.DSv3SparseLayerWeightsPytree)

  def test_validate_lineage_config_success(self):
    cfg = config_types.MaxTextConfig(
        use_lineage=True,
        scan_layers=True,
        decoder_block=DecoderBlockType.DEEPSEEK,
    )
    self.assertTrue(cfg.use_lineage)

  def test_validate_lineage_config_scan_layers_false_raises(self):
    with self.assertRaisesRegex(ValueError, "use_lineage=True requires scan_layers=True."):
      config_types.MaxTextConfig(
          use_lineage=True,
          scan_layers=False,
          decoder_block=DecoderBlockType.DEEPSEEK,
      )

  def test_validate_lineage_config_invalid_decoder_block_raises(self):
    with self.assertRaisesRegex(ValueError, "use_lineage=True requires decoder_block='deepseek'"):
      config_types.MaxTextConfig(
          use_lineage=True,
          scan_layers=True,
          decoder_block=DecoderBlockType.LLAMA2,
      )

  def test_validate_lineage_config(self):
    valid_cfg = types.SimpleNamespace(
        use_lineage=True,
        decoder_block="deepseek",
        scan_layers=True,
        attention_type="mla",
        rope_type="yarn",
        param_scan_axis=0,
        capacity_factor=1.0,
    )
    # Valid config should not raise
    lineage_adapter.validate_lineage_config(valid_cfg)

    # Valid config with DecoderBlockType enum
    lineage_adapter.validate_lineage_config(
        types.SimpleNamespace(**{**vars(valid_cfg), "decoder_block": DecoderBlockType.DEEPSEEK})
    )

    # None config
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(None)

    # use_lineage=False
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "use_lineage": False}))

    # invalid decoder_block
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "decoder_block": "llama"}))

    # scan_layers=False
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "scan_layers": False}))

    # invalid attention_type
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "attention_type": "mha"}))

    # invalid rope_type
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "rope_type": "default"}))

    # invalid param_scan_axis
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "param_scan_axis": 2}))

    # invalid capacity_factor
    with self.assertRaises(ValueError):
      lineage_adapter.validate_lineage_config(types.SimpleNamespace(**{**vars(valid_cfg), "capacity_factor": -1.0}))

  def test_run_lineage_dsv3_invokes_dsv3_with_resharded_weights(self):
    devices = np.array(jax.devices()[:1])
    mesh = jax.sharding.Mesh(
        devices.reshape((1, 1, 1, 1)),
        ("x", "y", "z", "core"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    cfg = types.SimpleNamespace(
        use_lineage=True,
        decoder_block="deepseek",
        scan_layers=True,
        attention_type="mla",
        rope_type="yarn",
        param_scan_axis=0,
        capacity_factor=8.0,
        dtype=jnp.bfloat16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        num_query_heads=4,
        max_target_length=16,
        max_position_embeddings=64,
        original_max_position_embeddings=64,
        beta_fast=32,
        beta_slow=1,
        rope_factor=1,
        num_experts=4,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        normalization_layer_epsilon=1e-5,
        kv_lora_rank=16,
    )

    num_layers = 2
    emb_dim = 32
    cq_dim = 16
    ckv_dim = 16
    num_query_heads = 4
    qk_head_dim = 24
    v_head_dim = 16
    mlp_dim = 32
    num_experts = 4
    expert_hidden_dim = 16

    dense_params = {
        "pre_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((num_layers, emb_dim, cq_dim))},
            "wq_b": {"kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim))},
            "q_norm": {"scale": jnp.ones((num_layers, cq_dim))},
            "wkv_a": {"kernel": jnp.ones((num_layers, emb_dim, ckv_dim + 8))},
            "wkv_b": {"kernel": jnp.ones((num_layers, ckv_dim, num_query_heads, 16 + v_head_dim))},
            "kv_norm": {"scale": jnp.ones((num_layers, ckv_dim))},
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "mlp": {
            "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, mlp_dim))},
            "wo": {"kernel": jnp.ones((num_layers, mlp_dim, emb_dim))},
        },
    }
    sparse_params = {
        "pre_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "self_attention": {
            "wq_a": {"kernel": jnp.ones((num_layers, emb_dim, cq_dim))},
            "wq_b": {"kernel": jnp.ones((num_layers, cq_dim, num_query_heads, qk_head_dim))},
            "q_norm": {"scale": jnp.ones((num_layers, cq_dim))},
            "wkv_a": {"kernel": jnp.ones((num_layers, emb_dim, ckv_dim + 8))},
            "wkv_b": {"kernel": jnp.ones((num_layers, ckv_dim, num_query_heads, 16 + v_head_dim))},
            "kv_norm": {"scale": jnp.ones((num_layers, ckv_dim))},
            "out": {"kernel": jnp.ones((num_layers, num_query_heads, v_head_dim, emb_dim))},
        },
        "post_self_attention_layer_norm": {"scale": jnp.ones((num_layers, emb_dim))},
        "DeepSeekMoeBlock_0": {
            "MoeBlock_0": {
                "gate": {
                    "kernel": jnp.ones((num_layers, emb_dim, num_experts)),
                    "bias": jnp.zeros((num_layers, num_experts)),
                },
                "wi_0": jnp.ones((num_layers, num_experts, emb_dim, expert_hidden_dim)),
                "wi_1": jnp.ones((num_layers, num_experts, emb_dim, expert_hidden_dim)),
                "wo": jnp.ones((num_layers, num_experts, expert_hidden_dim, emb_dim)),
            },
            "shared_experts": {
                "wi_0": {"kernel": jnp.ones((num_layers, emb_dim, expert_hidden_dim))},
                "wi_1": {"kernel": jnp.ones((num_layers, emb_dim, expert_hidden_dim))},
                "wo": {"kernel": jnp.ones((num_layers, expert_hidden_dim, emb_dim))},
            },
        },
    }

    inputs = jnp.zeros((2, 16, emb_dim), dtype=jnp.bfloat16)
    decoder_positions = jnp.zeros((2, 16), dtype=jnp.int32)

    fake_out = jnp.zeros_like(inputs)
    with (
        mock.patch.object(lineage_adapter.dsv3, "dsv3", return_value=(fake_out, None)) as mock_dsv3,
        mock.patch.object(
            lineage_adapter.dsv3_mla,
            "init_splash_kernel",
            return_value="fake_kernel",
        ),
        mock.patch.object(
            lineage_adapter.dsv3_mla,
            "get_yarn_freqs",
            return_value=(jnp.ones((2, 16, 1, 8)), jnp.ones((2, 16, 1, 8))),
        ),
    ):
      out = lineage_adapter.run_lineage_dsv3(
          inputs=inputs,
          dense_params=dense_params,
          sparse_params=sparse_params,
          decoder_positions=decoder_positions,
          mesh=mesh,
          cfg=cfg,
      )
      self.assertEqual(out.shape, inputs.shape)
      mock_dsv3.assert_called_once()
      call_kwargs = mock_dsv3.call_args.kwargs
      self.assertNotIn("attention_sharding", call_kwargs)
      self.assertNotIn("q_down_out_pspec", call_kwargs)
      self.assertNotIn("kv_down_out_pspec", call_kwargs)

      # Verify weights passed to dsv3 have MLA, MoE, and MLP resharded correctly
      w = mock_dsv3.call_args.args[1]
      self.assertIsInstance(w, dsv3_types.DSv3WeightsPytree)
      for layer_w in (w.dense, w.sparse):
        assert layer_w is not None
        assert layer_w.pre_attn_norm_scale is not None
        assert layer_w.post_attn_norm_scale is not None
        assert layer_w.mla is not None
        assert layer_w.mla.q_down is not None
        assert layer_w.mla.q_up is not None
        assert layer_w.mla.q_norm_scale is not None
        assert layer_w.mla.kv_down is not None
        assert layer_w.mla.k_up is not None
        assert layer_w.mla.v_up is not None
        assert layer_w.mla.kv_norm_scale is not None
        assert layer_w.mla.out is not None
        self.assertEqual(
            layer_w.pre_attn_norm_scale.sharding.spec,
            jax.sharding.PartitionSpec(None, None),
        )
        self.assertEqual(
            layer_w.post_attn_norm_scale.sharding.spec,
            jax.sharding.PartitionSpec(None, None),
        )
        self.assertEqual(
            layer_w.mla.q_down.sharding.spec,
            jax.sharding.PartitionSpec(None, "z", None),
        )
        self.assertEqual(
            layer_w.mla.q_up.sharding.spec,
            jax.sharding.PartitionSpec(None, None, "core", "z"),
        )
        self.assertEqual(
            layer_w.mla.q_norm_scale.sharding.spec,
            jax.sharding.PartitionSpec(None, None),
        )
        self.assertEqual(
            layer_w.mla.kv_down.sharding.spec,
            jax.sharding.PartitionSpec(None, "z", None),
        )
        self.assertEqual(
            layer_w.mla.k_up.sharding.spec,
            jax.sharding.PartitionSpec(None, None, "core", "z"),
        )
        self.assertEqual(
            layer_w.mla.v_up.sharding.spec,
            jax.sharding.PartitionSpec(None, None, "core", "z"),
        )
        self.assertEqual(
            layer_w.mla.kv_norm_scale.sharding.spec,
            jax.sharding.PartitionSpec(None, None),
        )
        self.assertEqual(
            layer_w.mla.out.sharding.spec,
            jax.sharding.PartitionSpec(None, "core", None, "z"),
        )

      # Verify MoE resharding on sparse layer
      assert w.sparse is not None
      assert w.sparse.moe is not None
      assert w.sparse.moe.router is not None
      assert w.sparse.moe.router.kernel is not None
      assert w.sparse.moe.router.bias is not None
      assert w.sparse.moe.routed is not None
      assert w.sparse.moe.routed.gate is not None
      assert w.sparse.moe.routed.linear is not None
      assert w.sparse.moe.shared is not None
      assert w.sparse.moe.shared.gate_0 is not None
      assert w.sparse.moe.shared.gate_1 is not None
      assert w.sparse.moe.shared.linear is not None
      self.assertEqual(
          w.sparse.moe.router.kernel.sharding.spec,
          jax.sharding.PartitionSpec(None, None, None),
      )
      self.assertEqual(
          w.sparse.moe.router.bias.sharding.spec,
          jax.sharding.PartitionSpec(None, None),
      )
      self.assertEqual(
          w.sparse.moe.routed.gate.sharding.spec,
          jax.sharding.PartitionSpec(None, ("x", "y", "core"), None, "z"),
      )
      self.assertEqual(
          w.sparse.moe.routed.linear.sharding.spec,
          jax.sharding.PartitionSpec(None, ("x", "y", "core"), "z", None),
      )
      self.assertEqual(
          w.sparse.moe.shared.gate_0.sharding.spec,
          jax.sharding.PartitionSpec(None, None, "z"),
      )
      self.assertEqual(
          w.sparse.moe.shared.gate_1.sharding.spec,
          jax.sharding.PartitionSpec(None, None, "z"),
      )
      self.assertEqual(
          w.sparse.moe.shared.linear.sharding.spec,
          jax.sharding.PartitionSpec(None, "z", None),
      )

      # Verify MLP resharding on dense layer
      assert w.dense is not None
      assert w.dense.mlp is not None
      assert w.dense.mlp.gate_0 is not None
      assert w.dense.mlp.gate_1 is not None
      assert w.dense.mlp.linear is not None
      self.assertEqual(
          w.dense.mlp.gate_0.sharding.spec,
          jax.sharding.PartitionSpec(None, None, "z"),
      )
      self.assertEqual(
          w.dense.mlp.gate_1.sharding.spec,
          jax.sharding.PartitionSpec(None, None, "z"),
      )
      self.assertEqual(
          w.dense.mlp.linear.sharding.spec,
          jax.sharding.PartitionSpec(None, "z", None),
      )

  def test_fetch_lineage_weights_with_existing_pytree(self):
    dense_w = dsv3_types.DSv3DenseLayerWeightsPytree(
        pre_attn_norm_scale=jnp.ones((2, 16), dtype=jnp.bfloat16),
        mla=dsv3_types.DSv3MLAWeightsPytree(
            q_down=jnp.ones((2, 16, 8), dtype=jnp.bfloat16),
        ),
        post_attn_norm_scale=jnp.ones((2, 16), dtype=jnp.bfloat16),
        mlp=dsv3_types.DSv3MLPWeightsPytree(
            gate_0=jnp.ones((2, 16, 32), dtype=jnp.bfloat16),
        ),
    )
    existing_pytree = dsv3_types.DSv3WeightsPytree(dense=dense_w, sparse=dsv3_types.DSv3SparseLayerWeightsPytree())

    # Calling fetch_lineage_weights with existing pytree and dtype=float32
    # should transform dtype via _check_pytree.
    res = lineage_adapter.fetch_lineage_weights(
        dense_params=existing_pytree,
        sparse_params=None,
        dtype=jnp.float32,
    )
    self.assertIsInstance(res, dsv3_types.DSv3WeightsPytree)
    assert res.dense is not None
    assert res.dense.pre_attn_norm_scale is not None
    assert res.dense.mla is not None
    assert res.dense.mla.q_down is not None
    self.assertEqual(res.dense.pre_attn_norm_scale.dtype, jnp.float32)
    self.assertEqual(res.dense.mla.q_down.dtype, jnp.float32)


if __name__ == "__main__":
  absltest.main()
