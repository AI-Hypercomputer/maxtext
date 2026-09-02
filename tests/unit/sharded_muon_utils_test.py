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

"""Unit tests for sharded_muon_utils.py."""

# pylint: disable=protected-access

import collections.abc
import contextlib
import io
import unittest
from unittest import mock

from absl.testing import absltest
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxtext.optimizers.muon import ShardedMuonDimensionNumbers as smdn
from maxtext.utils import muon_utils
from maxtext.utils import sharded_muon_utils
import numpy as np


def _extract_axes(tree):
  """Recursively extracts (reduction_axis, output_axis) or None from dimension number trees."""
  if isinstance(tree, nnx.State):
    tree = nnx.to_pure_dict(tree)
  if isinstance(tree, (dict, collections.abc.Mapping)) or hasattr(tree, "items"):
    return {k: _extract_axes(v) for k, v in tree.items()}
  if tree is None:
    return None
  return (tree.reduction_axis, tree.output_axis)


class _AttentionSubModule(nnx.Module):
  """Placeholder NNX attention sub-module for testing parameter extraction."""

  def __init__(self):
    self.out = nnx.Param(jnp.ones((2, 4, 8)))


class _MoeLikeNNXModel(nnx.Module):
  """Placeholder NNX model for testing parameter extraction."""

  def __init__(self, rngs):
    del rngs  # Unused.
    self.w_standard = nnx.Param(jnp.ones((4, 8)))
    self.self_attention = _AttentionSubModule()
    self.scale = nnx.Param(jnp.ones((8,)))


class TestExtractSharding(unittest.TestCase):
  """Tests for _extract_sharding utility."""

  def setUp(self):
    super().setUp()
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    self.mesh = jax.sharding.Mesh(devices, ("data", "model"))
    self.named_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("data", "model"))
    self.partition_spec = jax.sharding.PartitionSpec("data", "model")

  def test_extracts_from_named_sharding(self):
    self.assertEqual(
        sharded_muon_utils._extract_sharding(self.named_sharding),
        self.named_sharding,
    )

  def test_extracts_from_partition_spec(self):
    self.assertEqual(
        sharded_muon_utils._extract_sharding(self.partition_spec),
        self.partition_spec,
    )

  def test_extracts_from_variable_with_get_value(self):
    leaf = mock.MagicMock()
    leaf.get_value.return_value = self.named_sharding
    self.assertEqual(
        sharded_muon_utils._extract_sharding(leaf),
        self.named_sharding,
    )

  def test_extracts_from_attribute(self):
    leaf = mock.MagicMock(spec=["sharding"])
    leaf.sharding = self.named_sharding
    self.assertEqual(
        sharded_muon_utils._extract_sharding(leaf),
        self.named_sharding,
    )

  def test_returns_none_for_raw_shape_struct(self):
    leaf = jax.ShapeDtypeStruct((4, 8), jnp.float32)
    self.assertIsNone(sharded_muon_utils._extract_sharding(leaf))

  def test_returns_none_for_non_sharding_value(self):
    leaf = mock.MagicMock()
    leaf.get_value.return_value = jnp.zeros((4, 8))
    del leaf.sharding
    self.assertIsNone(sharded_muon_utils._extract_sharding(leaf))


class TestGetShardedMuonWeightDimensionNumbersNNX(unittest.TestCase):
  """Tests for get_sharded_muon_weight_dimension_numbers with NNX models."""

  def setUp(self):
    super().setUp()
    self.model = _MoeLikeNNXModel(rngs=nnx.Rngs(0))
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    self.mesh = jax.sharding.Mesh(devices, ("data", "model"))

  def test_nnx_model_axes_match_base_muon(self):
    sharded_result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, mesh=self.mesh)
    base_result = muon_utils.get_muon_weight_dimension_numbers(self.model)
    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))

  def test_nnx_model_with_logical_axis_rules_axes_match_base_muon(self):
    config = mock.MagicMock()
    config.logical_axis_rules = (("embed", "fsdp"), ("mlp", "tensor"))
    sharded_result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(
        self.model, config=config, mesh=self.mesh
    )
    base_result = muon_utils.get_muon_weight_dimension_numbers(self.model, config=config)
    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))

  def test_nnx_model_populates_named_sharding(self):
    result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, mesh=self.mesh)
    self.assertIsNotNone(result["w_standard"].sharding)
    self.assertIsInstance(result["w_standard"].sharding, jax.sharding.NamedSharding)
    self.assertEqual(result["w_standard"].sharding.mesh, self.mesh)

  def test_nnx_model_resolves_mesh_from_config_when_mesh_is_none(self):
    fake_mesh = jax.sharding.Mesh(
        self.mesh.devices,
        ("data", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    config = mock.MagicMock()
    config.shard_mode = "explicit"
    config.mesh_axes = ("data", "model")
    config.logical_axis_rules = ()

    with mock.patch.object(
        sharded_muon_utils.maxtext_utils,
        "get_mesh_from_config",
        return_value=fake_mesh,
    ) as mock_get_mesh:
      sharded_result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, config=config, mesh=None)
      mock_get_mesh.assert_called_once_with(config)
    self.assertIsNotNone(sharded_result["w_standard"].sharding)
    self.assertEqual(sharded_result["w_standard"].sharding.mesh, fake_mesh)

  def test_raises_error_when_mesh_and_config_are_none(self):
    with self.assertRaisesRegex(ValueError, "Either mesh or config must be provided"):
      sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, config=None, mesh=None)

  def test_nnx_verbose_prints_debug_structure(self):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, mesh=self.mesh, verbose=True)
    self.assertIn("Model Structure", buf.getvalue())
    self.assertIn("Muon Dimension Numbers", buf.getvalue())


class TestGetShardedMuonWeightDimensionNumbersLinen(unittest.TestCase):
  """Tests for get_sharded_muon_weight_dimension_numbers with Linen models."""

  def setUp(self):
    super().setUp()
    patcher = mock.patch.object(
        sharded_muon_utils.sharding_lib,
        "remove_size_one_mesh_axis",
        side_effect=lambda spec, mesh: spec,
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_linen_model_axes_match_base_muon(self):
    class LinenStub(nn.Module):

      @nn.compact
      def __call__(self, x):
        return x

    model = LinenStub()

    fake_abstract_param = {
        "params": {
            "self_attention": {
                "out": nn.LogicallyPartitioned(
                    value=jax.ShapeDtypeStruct((8, 2, 4, 16), jnp.float32),
                    names=("heads", "kv", "embed"),
                )
            },
            "norm": {
                "scale": nn.LogicallyPartitioned(
                    value=jax.ShapeDtypeStruct((16,), jnp.float32),
                    names=("embed",),
                )
            },
        },
    }

    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    config = mock.MagicMock()
    config.logical_axis_rules = ()
    config.param_scan_axis = 1

    with (
        mock.patch.object(
            sharded_muon_utils.maxtext_utils,
            "get_abstract_param",
            return_value=fake_abstract_param,
        ),
        mock.patch.object(
            muon_utils.maxtext_utils,
            "get_abstract_param",
            return_value=fake_abstract_param,
        ),
    ):
      sharded_result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(model, config=config, mesh=mesh)
      base_result = muon_utils.get_muon_weight_dimension_numbers(model, config=config)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))

  def test_linen_get_transform_tree_creates_named_sharding(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    leaf = nn.LogicallyPartitioned(
        value=jax.ShapeDtypeStruct((4, 8), jnp.float32),
        names=("embed", "mlp"),
    )
    tree = {"params": {"mlp": {"kernel": leaf}}}
    with nn_partitioning.axis_rules((("embed", "data"), ("mlp", "model"))):
      sharded_result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh)
    base_result = muon_utils.get_transform_tree(tree)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))
    sharding = sharded_result["params"]["mlp"]["kernel"].sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertEqual(sharding.mesh, mesh)
    self.assertEqual(sharding.spec, jax.sharding.PartitionSpec("data", "model"))

  def test_linen_get_transform_tree_scanned_layers_param_scan_axis_1(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    # Standard MaxText MLP weight scanned at param_scan_axis=1: (in_features=4, num_layers=2, out_features=8)
    leaf = nn.LogicallyPartitioned(
        value=jax.ShapeDtypeStruct((4, 2, 8), jnp.float32),
        names=("embed", "mlp"),
    )
    tree = {"params": {"mlp": {"kernel": leaf}}}
    config = mock.MagicMock()
    config.param_scan_axis = 1
    with nn_partitioning.axis_rules((("embed", "data"), ("mlp", "model"))):
      sharded_result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh, config=config)
    base_result = muon_utils.get_transform_tree(tree)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))
    sharding = sharded_result["params"]["mlp"]["kernel"].sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertEqual(sharding.mesh, mesh)
    self.assertEqual(sharding.spec, jax.sharding.PartitionSpec("data", None, "model"))

  def test_linen_get_transform_tree_scanned_layers_param_scan_axis_0(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    # Layer scanned at param_scan_axis=0: (num_layers=2, in_features=4, out_features=8)
    leaf = nn.LogicallyPartitioned(
        value=jax.ShapeDtypeStruct((2, 4, 8), jnp.float32),
        names=("embed", "mlp"),
    )
    tree = {"params": {"mlp": {"kernel": leaf}}}
    config = mock.MagicMock()
    config.param_scan_axis = 0
    with nn_partitioning.axis_rules((("embed", "data"), ("mlp", "model"))):
      sharded_result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh, config=config)
    base_result = muon_utils.get_transform_tree(tree)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))
    sharding = sharded_result["params"]["mlp"]["kernel"].sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertEqual(sharding.mesh, mesh)
    self.assertEqual(sharding.spec, jax.sharding.PartitionSpec(None, "data", "model"))

  def test_linen_get_transform_tree_scanned_layers_with_list_names(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    # names provided as a list rather than a tuple
    leaf = nn.LogicallyPartitioned(
        value=jax.ShapeDtypeStruct((4, 2, 8), jnp.float32),
        names=["embed", "mlp"],
    )
    tree = {"params": {"mlp": {"kernel": leaf}}}
    config = mock.MagicMock()
    config.param_scan_axis = 1
    with nn_partitioning.axis_rules((("embed", "data"), ("mlp", "model"))):
      sharded_result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh, config=config)
    base_result = muon_utils.get_transform_tree(tree)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))
    sharding = sharded_result["params"]["mlp"]["kernel"].sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertEqual(sharding.mesh, mesh)
    self.assertEqual(sharding.spec, jax.sharding.PartitionSpec("data", None, "model"))

  def test_linen_get_transform_tree_attention_out_scanned(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    # Scanned Attention Out: (num_heads=8, num_layers=2, head_dim=4, out_features=16) with param_scan_axis=1
    leaf = nn.LogicallyPartitioned(
        value=jax.ShapeDtypeStruct((8, 2, 4, 16), jnp.float32),
        names=("heads", "kv", "embed"),
    )
    tree = {"params": {"self_attention": {"out": leaf}}}
    config = mock.MagicMock()
    config.param_scan_axis = 1
    with nn_partitioning.axis_rules((("heads", "model"), ("kv", None), ("embed", "data"))):
      sharded_result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh, config=config)
    base_result = muon_utils.get_transform_tree(tree)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))
    sharding = sharded_result["params"]["self_attention"]["out"].sharding
    self.assertIsInstance(sharding, jax.sharding.NamedSharding)
    self.assertEqual(sharding.mesh, mesh)
    self.assertEqual(sharding.spec, jax.sharding.PartitionSpec("model", None, None, "data"))

  def test_linen_get_transform_tree_unpartitioned_leaf(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    leaf = jax.ShapeDtypeStruct((4, 8), jnp.float32)
    tree = {"params": {"mlp": {"kernel": leaf}}}
    sharded_result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh)
    base_result = muon_utils.get_transform_tree(tree)

    self.assertEqual(_extract_axes(sharded_result), _extract_axes(base_result))
    self.assertIsNone(sharded_result["params"]["mlp"]["kernel"].sharding)

  def test_linen_get_transform_tree_raises_when_rank_diff_not_one(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    # Tensor rank 4 but names length 2 (diff = 2)
    leaf = nn.LogicallyPartitioned(
        value=jax.ShapeDtypeStruct((2, 4, 4, 8), jnp.float32),
        names=("embed", "mlp"),
    )
    tree = {"params": {"mlp": {"kernel": leaf}}}
    with self.assertRaisesRegex(ValueError, "differ from tensor rank .* by exactly 1"):
      sharded_muon_utils.get_transform_tree(tree, mesh=mesh)


class TestPrintStructureDebug(unittest.TestCase):
  """Tests for _print_structure_debug."""

  def test_prints_logically_partitioned_leaf_info(self):
    leaf = nn.LogicallyPartitioned(value=jax.ShapeDtypeStruct((4, 8), jnp.float32), names=("embed", "mlp"))
    tree = {"params": {"kernel": leaf}}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      sharded_muon_utils._print_structure_debug(
          tree,
          muon_weight_dimension_numbers={"params": {"kernel": smdn((0,), (-1,))}},
      )
    out = buf.getvalue()
    self.assertIn("(4, 8)", out)
    self.assertIn("embed", out)

  def test_prints_shape_dtype_struct_leaf_info(self):
    tree = {"kernel": jax.ShapeDtypeStruct((16, 32), jnp.float32)}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      sharded_muon_utils._print_structure_debug(tree, muon_weight_dimension_numbers={"kernel": smdn((0,), (-1,))})
    out = buf.getvalue()
    self.assertIn("(16, 32)", out)


if __name__ == "__main__":
  absltest.main()
