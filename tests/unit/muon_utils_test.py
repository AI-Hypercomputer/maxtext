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

"""Unit tests for muon_utils.py."""

# pylint: disable=protected-access

import io
import contextlib
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from optax.contrib._muon import MuonDimensionNumbers as mdn

from maxtext.utils import muon_utils


class TestIsPathContainAny(unittest.TestCase):
  """Tests for _is_path_contain_any helper."""

  def test_returns_true_when_any_element_in_path(self):
    self.assertTrue(muon_utils._is_path_contain_any(("bias", "scale"), ("decoder", "bias")))

  def test_returns_false_when_no_element_in_path(self):
    self.assertFalse(muon_utils._is_path_contain_any(("bias", "scale"), ("decoder", "kernel")))

  def test_empty_tuples_returns_false(self):
    self.assertFalse(muon_utils._is_path_contain_any((), ("decoder", "kernel")))


class TestTransformLogic(unittest.TestCase):
  """Tests for transform_logic: covers every branch of the mapping."""

  # --- 1. Exclusions ---
  def test_scale_is_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "norm", "scale")))

  def test_bias_is_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "dense", "bias")))

  def test_embedding_is_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("token_embedder", "embedding")))

  def test_logits_dense_is_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "logits_dense", "kernel")))

  # --- 2.1 MoE ---
  def test_moe_wi_0_uses_last_two_axes(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "MoeBlock_0", "wi_0")), mdn((-2,), (-1,)))

  def test_moe_wi_1_uses_last_two_axes(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "MoeBlock_0", "wi_1")), mdn((-2,), (-1,)))

  def test_moe_wo_uses_last_two_axes(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "MoeBlock_0", "wo")), mdn((-2,), (-1,)))

  def test_moe_gate_falls_through_to_standard(self):
    # 'gate' is inside MoeBlock_0 but not one of (wi_0, wi_1, wo) → standard.
    self.assertEqual(muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate", "kernel")), mdn((0,), (-1,)))

  # --- 2.2 Self-attention ---
  def test_self_attention_out_projection(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "out")), mdn((0, -2), (-1,)))

  def test_self_attention_query_projection(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "query")), mdn((0,), (-2, -1)))

  def test_self_attention_key_projection(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "key")), mdn((0,), (-2, -1)))

  def test_self_attention_value_projection(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "value")), mdn((0,), (-2, -1)))

  def test_self_attention_wq_b_and_wkv_b(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "wq_b")), mdn((0,), (-2, -1)))
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "wkv_b")), mdn((0,), (-2, -1)))

  def test_self_attention_mla_wq_a_is_excluded_from_special(self):
    # wq_a / wkv_a are MLA down-projections; they fall through the self_attention branch
    # without matching anything, so the function returns the default standard mdn((0,), (-1,)).
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "wq_a")), mdn((0,), (-1,)))
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "wkv_a")), mdn((0,), (-1,)))

  # --- 3. Standard ---
  def test_standard_weight(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "mlp", "kernel")), mdn((0,), (-1,)))


class TestGetTransformTree(unittest.TestCase):
  """Tests for get_transform_tree: recursive dict walk that applies transform_logic."""

  def test_nested_dict_is_walked(self):
    tree = {"decoder": {"self_attention": {"out": 0}, "mlp": {"kernel": 0}}}
    result = muon_utils.get_transform_tree(tree)
    self.assertEqual(result["decoder"]["self_attention"]["out"], mdn((0, -2), (-1,)))
    self.assertEqual(result["decoder"]["mlp"]["kernel"], mdn((0,), (-1,)))

  def test_excluded_leaves_become_none(self):
    tree = {"decoder": {"norm": {"scale": 0}}}
    self.assertIsNone(muon_utils.get_transform_tree(tree)["decoder"]["norm"]["scale"])

  def test_non_dict_leaf_at_root_returns_transform(self):
    # If the tree itself is a leaf, path=() and transform_logic returns the standard mdn.
    self.assertEqual(muon_utils.get_transform_tree(0), mdn((0,), (-1,)))


class _MoeLikeNNXModel(nnx.Module):
  """Small NNX model whose param paths exercise the NNX branch of get_muon_weight_dimension_numbers."""

  def __init__(self, rngs):
    # Names are chosen so transform_logic matches each of the three meaningful branches:
    # - w_standard: default mdn
    # - self_attention_out: attention-out mdn
    # - scale: excluded (None)
    self.w_standard = nnx.Param(jnp.ones((4, 8)))
    self.self_attention_out = nnx.Param(jnp.ones((4, 8)))
    self.scale = nnx.Param(jnp.ones((8,)))


class TestGetMuonWeightDimensionNumbersNNX(unittest.TestCase):
  """Covers the NNX branch of get_muon_weight_dimension_numbers (isinstance(model, nnx.Module))."""

  def setUp(self):
    self.model = _MoeLikeNNXModel(rngs=nnx.Rngs(0))

  def test_nnx_model_dispatches_to_tree_map_with_path(self):
    """NNX branch should produce an nnx.State tree with transform_logic applied per leaf."""
    result = muon_utils.get_muon_weight_dimension_numbers(self.model, config=None)

    # Result is an nnx.State whose top-level keys mirror the model attributes.
    self.assertIn("w_standard", result)
    self.assertIn("self_attention_out", result)
    self.assertIn("scale", result)

    # NNX Variables are walked by jax.tree_util.tree_map_with_path, so the returned
    # tree replaces each Variable's value with transform_logic(path_strings).
    # 'scale' matches the exclusion branch → value is None.
    self.assertIsNone(result["scale"])
    # 'w_standard' does not trigger any special rule → standard mdn.
    self.assertEqual(result["w_standard"], mdn((0,), (-1,)))

  def test_nnx_verbose_path_executes_print_debug(self):
    """verbose=True should also execute _print_structure_debug without raising."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      muon_utils.get_muon_weight_dimension_numbers(self.model, config=None, verbose=True)
    self.assertIn("Model Structure", buf.getvalue())
    self.assertIn("Muon Dimension Numbers", buf.getvalue())


class TestGetMuonWeightDimensionNumbersLinen(unittest.TestCase):
  """Covers the Linen branch of get_muon_weight_dimension_numbers."""

  def test_linen_branch_uses_get_abstract_param(self):
    """Linen models dispatch to maxtext_utils.get_abstract_param + get_transform_tree."""
    # Build a Linen nn.Module so isinstance(model, nnx.Module) is False.

    class LinenStub(nn.Module):

      @nn.compact
      def __call__(self, x):
        return x

    model = LinenStub()

    # Mock the heavy get_abstract_param call with a pre-shaped dict that exercises
    # both a standard weight path and an excluded path.
    fake_abstract_param = {
        "params": {
            "self_attention": {"out": object()},
            "norm": {"scale": object()},
        },
    }

    with mock.patch.object(muon_utils.maxtext_utils, "get_abstract_param", return_value=fake_abstract_param):
      result = muon_utils.get_muon_weight_dimension_numbers(model, config=mock.MagicMock())

    self.assertEqual(result["params"]["self_attention"]["out"], mdn((0, -2), (-1,)))
    self.assertIsNone(result["params"]["norm"]["scale"])


class TestPrintStructureDebug(unittest.TestCase):
  """Covers both branches of get_leaf_info inside _print_structure_debug."""

  def test_handles_logically_partitioned_leaf(self):
    """Linen leaves are nn.LogicallyPartitioned; the helper should return {shape, names}."""
    leaf = nn.LogicallyPartitioned(value=jax.ShapeDtypeStruct((4, 8), jnp.float32), names=("embed", "mlp"))
    tree = {"params": {"kernel": leaf}}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      muon_utils._print_structure_debug(tree, muon_weight_dimension_numbers={"params": {"kernel": mdn((0,), (-1,))}})
    out = buf.getvalue()
    self.assertIn("(4, 8)", out)
    self.assertIn("embed", out)

  def test_handles_shape_dtype_struct_leaf(self):
    """NNX abstract leaves are ShapeDtypeStruct directly; the helper should return {shape}."""
    tree = {"kernel": jax.ShapeDtypeStruct((16, 32), jnp.float32)}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      muon_utils._print_structure_debug(tree, muon_weight_dimension_numbers={"kernel": mdn((0,), (-1,))})
    out = buf.getvalue()
    self.assertIn("(16, 32)", out)


if __name__ == "__main__":
  unittest.main()
