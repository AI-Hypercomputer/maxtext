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

import contextlib
import io
import unittest
from unittest import mock

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.optimizers.muon import ShardedMuonDimensionNumbers as smdn
from maxtext.utils import sharded_muon_utils
import numpy as np


class _AttentionSubModule(nnx.Module):

  def __init__(self):
    self.out = nnx.Param(jnp.ones((2, 4, 8)))


class _MoeLikeNNXModel(nnx.Module):
  """Small NNX model whose param paths exercise the NNX branch of get_sharded_muon_weight_dimension_numbers."""

  def __init__(self, rngs):
    self.w_standard = nnx.Param(jnp.ones((4, 8)))
    self.self_attention = _AttentionSubModule()
    self.scale = nnx.Param(jnp.ones((8,)))


class TestGetShardedMuonWeightDimensionNumbersNNX(unittest.TestCase):
  """Covers the NNX branch of get_sharded_muon_weight_dimension_numbers."""

  def setUp(self):
    self.model = _MoeLikeNNXModel(rngs=nnx.Rngs(0))
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    self.mesh = jax.sharding.Mesh(devices, ("data", "model"))

  def test_nnx_model_dispatches_to_tree_map_with_path(self):
    """NNX branch should produce an nnx.State tree with transform_logic applied per leaf."""
    result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, mesh=self.mesh)

    self.assertIn("w_standard", result)
    self.assertIn("self_attention", result)
    self.assertIn("out", result["self_attention"])
    self.assertIn("scale", result)

    self.assertIsNone(result["scale"])
    self.assertEqual(result["w_standard"].reduction_axis, (-2,))
    self.assertEqual(result["w_standard"].output_axis, (-1,))
    self.assertEqual(result["self_attention"]["out"].reduction_axis, (-3, -2))
    self.assertEqual(result["self_attention"]["out"].output_axis, (-1,))

  def test_nnx_model_with_logical_axis_rules(self):
    """Verifies that config.logical_axis_rules is active within get_sharded_muon_weight_dimension_numbers."""
    config = mock.MagicMock()
    config.logical_axis_rules = (("embed", "fsdp"), ("mlp", "tensor"))
    result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, config=config, mesh=self.mesh)
    self.assertEqual(result["w_standard"].reduction_axis, (-2,))
    self.assertEqual(result["w_standard"].output_axis, (-1,))
    self.assertEqual(result["self_attention"]["out"].reduction_axis, (-3, -2))
    self.assertEqual(result["self_attention"]["out"].output_axis, (-1,))
    self.assertIsNone(result["scale"])

  def test_nnx_model_with_mesh_populates_named_sharding(self):
    """Verifies that NamedSharding is properly attached to ShardedMuonDimensionNumbers when mesh is present."""
    result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, mesh=self.mesh)
    self.assertIsNotNone(result["w_standard"].sharding)
    self.assertIsInstance(result["w_standard"].sharding, jax.sharding.NamedSharding)

  def test_get_mesh_from_config_invoked_when_mesh_is_none(self):
    """Verifies that maxtext_utils.get_mesh_from_config is used when mesh is None."""
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
      result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, config=config, mesh=None)
      mock_get_mesh.assert_called_once_with(config)
    self.assertIsNotNone(result["w_standard"].sharding)

  def test_nnx_verbose_path_executes_print_debug(self):
    """verbose=True should execute _print_structure_debug without raising."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(self.model, mesh=self.mesh, verbose=True)
    self.assertIn("Model Structure", buf.getvalue())
    self.assertIn("Sharded Muon Dimension Numbers", buf.getvalue())


class TestGetShardedMuonWeightDimensionNumbersLinen(unittest.TestCase):
  """Covers the Linen branch of get_sharded_muon_weight_dimension_numbers."""

  def test_linen_branch_uses_get_abstract_param(self):
    """Linen models dispatch to maxtext_utils.get_abstract_param + get_transform_tree."""

    class LinenStub(nn.Module):

      @nn.compact
      def __call__(self, x):
        return x

    model = LinenStub()

    fake_abstract_param = {
        "params": {
            "self_attention": {"out": object()},
            "norm": {"scale": object()},
        },
    }

    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    with mock.patch.object(
        sharded_muon_utils.maxtext_utils,
        "get_abstract_param",
        return_value=fake_abstract_param,
    ):
      result = sharded_muon_utils.get_sharded_muon_weight_dimension_numbers(model, config=mock.MagicMock(), mesh=mesh)

    self.assertEqual(result["params"]["self_attention"]["out"].reduction_axis, (-3, -2))
    self.assertEqual(result["params"]["self_attention"]["out"].output_axis, (-1,))
    self.assertIsNone(result["params"]["norm"]["scale"])

  def test_linen_with_mesh_creates_named_sharding(self):
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    leaf = nn.LogicallyPartitioned(value=jax.ShapeDtypeStruct((4, 8), jnp.float32), names=("data", "model"))
    tree = {"params": {"mlp": {"kernel": leaf}}}
    result = sharded_muon_utils.get_transform_tree(tree, mesh=mesh)
    self.assertIsInstance(result["params"]["mlp"]["kernel"].sharding, jax.sharding.NamedSharding)


class TestPrintStructureDebug(unittest.TestCase):
  """Covers both branches of get_leaf_info inside _print_structure_debug."""

  def test_handles_logically_partitioned_leaf(self):
    leaf = nn.LogicallyPartitioned(value=jax.ShapeDtypeStruct((4, 8), jnp.float32), names=("embed", "mlp"))
    tree = {"params": {"kernel": leaf}}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      sharded_muon_utils._print_structure_debug(
          tree,
          muon_weight_dimension_numbers={"params": {"kernel": smdn((-2,), (-1,))}},
      )
    out = buf.getvalue()
    self.assertIn("(4, 8)", out)
    self.assertIn("embed", out)

  def test_handles_shape_dtype_struct_leaf(self):
    tree = {"kernel": jax.ShapeDtypeStruct((16, 32), jnp.float32)}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      sharded_muon_utils._print_structure_debug(tree, muon_weight_dimension_numbers={"kernel": smdn((-2,), (-1,))})
    out = buf.getvalue()
    self.assertIn("(16, 32)", out)


if __name__ == "__main__":
  unittest.main()
