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

import contextlib
import io
import unittest
from unittest import mock

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.src.maxtext.optimizers.muon import MuonDimensionNumbers as mdn
from maxtext.src.maxtext.utils import muon_utils
import numpy as np


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

  def test_moe_gate_is_excluded(self):
    # 'gate' is excluded from Muon (optimized with standard AdamW).
    self.assertIsNone(muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate", "kernel")))

  def test_gate_up_proj_not_shadowed_by_gate(self):
    # 'gate_up_proj' and 'gate_proj' must not be excluded by 'gate' exact match.
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate_up_proj")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "gate_proj")),
        mdn((-2,), (-1,)),
    )

  def test_router_names_are_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "MoeBlock_0", "router", "kernel")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "MoeBlock_0", "moe_gate", "kernel")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "MoeBlock_0", "expert_gate", "kernel")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "MoeBlock_0", "router_weights")))

  def test_1d_tensor_is_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "mlp", "custom_param"), shape=(512,)))

  def test_2d_attention_projections_use_2d_axes(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "query"), shape=(512, 512)),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "out"), shape=(512, 512)),
        mdn((-2,), (-1,)),
    )

  # --- 2.2 Self-attention ---
  def test_self_attention_out_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "out")),
        mdn((-3, -2), (-1,)),
    )

  def test_self_attention_query_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "query")),
        mdn((-3,), (-2, -1)),
    )

  def test_self_attention_key_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "key")),
        mdn((-3,), (-2, -1)),
    )

  def test_self_attention_value_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "value")),
        mdn((-3,), (-2, -1)),
    )

  def test_self_attention_wq_b_and_wkv_b(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wq_b")),
        mdn((-3,), (-2, -1)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wkv_b")),
        mdn((-3,), (-2, -1)),
    )

  def test_self_attention_mla_wq_a_is_standard(self):
    # wq_a / wkv_a are MLA down-projections; they fall through the self_attention branch
    # to standard mdn((-2,), (-1,)).
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wq_a")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wkv_a")),
        mdn((-2,), (-1,)),
    )

  # --- 3. Standard ---
  def test_standard_weight(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "kernel")),
        mdn((-2,), (-1,)),
    )

  # --- 4. DeepSeek V4 Specific ---
  def test_deepseek_v4_exclusions(self):
    # Scale, beta, base, sinks, and tid2eid nested segments should be excluded (None)
    self.assertIsNone(muon_utils.transform_logic(("decoder", "hc_head", "hc_scale")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "hc_head", "hc_base")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "layers", "layers_0", "mhc_attention", "res_beta")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "layers", "layers_0", "self_attention", "sinks")))
    self.assertIsNone(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "mlp", "MoeBlock_0", "gate", "tid2eid"))
    )
    self.assertIsNone(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "self_attention", "rotary_embedding", "inv_freq"))
    )

  def test_deepseek_v4_self_attention_grouped_projection(self):
    # o_a_proj projects with reduction on in_features_per_group (-2)
    # and output on out_features_per_group (-1)
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "o_a_proj")), mdn((-2,), (-1,)))

  # --- 5. Qwen3-Next Specific ---
  def test_qwen3_next_moe_routed_experts(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "routed_experts", "wi_0")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "routed_experts", "wi_1")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "routed_experts", "wo")),
        mdn((-2,), (-1,)),
    )

  def test_qwen3_next_gdn_projections(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "gdn", "in_proj_qkvz")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "gdn", "in_proj_ba")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "gdn", "out_proj")),
        mdn((-2,), (-1,)),
    )

  def test_qwen3_next_exclusions(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "gdn", "A_log")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "gdn", "dt_bias")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "gdn", "conv1d")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "mlp", "routed_experts", "gate")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "mlp", "shared_expert_gate")))

class TestGetTransformTree(unittest.TestCase):
  """Tests for get_transform_tree: recursive dict walk that applies transform_logic."""

  def test_nested_dict_is_walked(self):
    tree = {"decoder": {"self_attention": {"out": 0}, "mlp": {"kernel": 0}}}
    result = muon_utils.get_transform_tree(tree)
    self.assertEqual(result["decoder"]["self_attention"]["out"], mdn((-3, -2), (-1,)))
    self.assertEqual(result["decoder"]["mlp"]["kernel"], mdn((-2,), (-1,)))

  def test_excluded_leaves_become_none(self):
    tree = {"decoder": {"norm": {"scale": 0}}}
    self.assertIsNone(muon_utils.get_transform_tree(tree)["decoder"]["norm"]["scale"])

  def test_non_dict_leaf_at_root_returns_transform(self):
    # If the tree itself is a leaf, path=() and transform_logic returns the standard mdn.
    self.assertEqual(muon_utils.get_transform_tree(0), mdn((-2,), (-1,)))


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
    self.assertEqual(result["w_standard"], mdn((-2,), (-1,)))
    self.assertEqual(result["self_attention_out"], mdn((-3, -2), (-1,)))

  def test_nnx_model_with_logical_axis_rules(self):
    """Verifies that config.logical_axis_rules is active within get_muon_weight_dimension_numbers."""
    config = mock.MagicMock()
    config.logical_axis_rules = (("embed", "fsdp"), ("mlp", "tensor"))
    result = muon_utils.get_muon_weight_dimension_numbers(self.model, config=config)
    self.assertEqual(result["w_standard"], mdn((-2,), (-1,)))
    self.assertEqual(result["self_attention_out"], mdn((-3, -2), (-1,)))
    self.assertIsNone(result["scale"])

  def test_nnx_model_with_mesh_populates_named_sharding(self):
    """Verifies that NamedSharding is properly attached to MuonDimensionNumbers when mesh is present."""
    devices = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    result = muon_utils.get_muon_weight_dimension_numbers(self.model, config=None, mesh=mesh)
    self.assertIsNotNone(result["w_standard"].sharding)
    self.assertIsInstance(result["w_standard"].sharding, jax.sharding.NamedSharding)

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

    with mock.patch.object(muon_utils.maxtext_utils, "get_abstract_param", return_value=fake_abstract_param):
      result = muon_utils.get_muon_weight_dimension_numbers(model, config=mock.MagicMock())

    self.assertEqual(result["params"]["self_attention"]["out"], mdn((-3, -2), (-1,)))
    self.assertIsNone(result["params"]["norm"]["scale"])


class TestPrintStructureDebug(unittest.TestCase):
  """Covers both branches of get_leaf_info inside _print_structure_debug."""

  def test_handles_logically_partitioned_leaf(self):
    """Linen leaves are nn.LogicallyPartitioned; the helper should return {shape, names}."""
    leaf = nn.LogicallyPartitioned(value=jax.ShapeDtypeStruct((4, 8), jnp.float32), names=("embed", "mlp"))
    tree = {"params": {"kernel": leaf}}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      muon_utils._print_structure_debug(
          tree,
          muon_weight_dimension_numbers={"params": {"kernel": mdn((-2,), (-1,))}},
      )
    out = buf.getvalue()
    self.assertIn("(4, 8)", out)
    self.assertIn("embed", out)

  def test_handles_shape_dtype_struct_leaf(self):
    """NNX abstract leaves are ShapeDtypeStruct directly; the helper should return {shape}."""
    tree = {"kernel": jax.ShapeDtypeStruct((16, 32), jnp.float32)}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      muon_utils._print_structure_debug(tree, muon_weight_dimension_numbers={"kernel": mdn((-2,), (-1,))})
    out = buf.getvalue()
    self.assertIn("(16, 32)", out)


class TestLayerScanInvariance(unittest.TestCase):
  """Verifies Muon dimension mapping is invariant to layer scanning."""

  def test_transform_tree_identical_with_and_without_layer_scan(self):
    """Parameter paths return identical dimension specs with/without scan."""
    unscanned_tree = {
        "decoder": {
            "self_attention": {
                "query": jax.ShapeDtypeStruct((512, 8, 64), jnp.float32),
                "out": jax.ShapeDtypeStruct((8, 64, 512), jnp.float32),
            },
            "mlp": {
                "wi_0": jax.ShapeDtypeStruct((512, 2048), jnp.float32),
                "wo": jax.ShapeDtypeStruct((2048, 512), jnp.float32),
            },
            "norm": {
                "scale": jax.ShapeDtypeStruct((512,), jnp.float32),
            },
        }
    }
    # Prepend layer dimension (num_layers=12) on axis 0 for all layers to
    # simulate layer scanning.
    scanned_tree = {
        "decoder": {
            "self_attention": {
                "query": jax.ShapeDtypeStruct((12, 512, 8, 64), jnp.float32),
                "out": jax.ShapeDtypeStruct((12, 8, 64, 512), jnp.float32),
            },
            "mlp": {
                "wi_0": jax.ShapeDtypeStruct((12, 512, 2048), jnp.float32),
                "wo": jax.ShapeDtypeStruct((12, 2048, 512), jnp.float32),
            },
            "norm": {
                "scale": jax.ShapeDtypeStruct((12, 512), jnp.float32),
            },
        }
    }

    unscanned_mdn = muon_utils.get_transform_tree(unscanned_tree)
    scanned_mdn = muon_utils.get_transform_tree(scanned_tree)

    self.assertEqual(unscanned_mdn, scanned_mdn)

  def test_relative_dimensions_resolve_consistent_matrix_features(self):
    """Negative dimensions index same feature axes regardless of scan axis."""
    # Standard MLP: 2D unscanned vs 3D scanned (leading layer axis = 12)
    mlp_unscanned_shape = (512, 2048)
    mlp_scanned_shape = (12, 512, 2048)
    mlp_mdn = muon_utils.transform_logic(("decoder", "mlp", "kernel"))

    # Reduction axis (-2) and Output axis (-1) point to identical feature sizes
    self.assertEqual(mlp_unscanned_shape[mlp_mdn.reduction_axis[0]], 512)
    self.assertEqual(mlp_scanned_shape[mlp_mdn.reduction_axis[0]], 512)
    self.assertEqual(mlp_unscanned_shape[mlp_mdn.output_axis[0]], 2048)
    self.assertEqual(mlp_scanned_shape[mlp_mdn.output_axis[0]], 2048)

    # Attention QKV: 3D unscanned vs 4D scanned
    qkv_unscanned_shape = (512, 8, 64)
    qkv_scanned_shape = (12, 512, 8, 64)
    qkv_mdn = muon_utils.transform_logic(("decoder", "self_attention", "query"))

    self.assertEqual(qkv_unscanned_shape[qkv_mdn.reduction_axis[0]], 512)
    self.assertEqual(qkv_scanned_shape[qkv_mdn.reduction_axis[0]], 512)
    self.assertEqual(tuple(qkv_unscanned_shape[ax] for ax in qkv_mdn.output_axis), (8, 64))
    self.assertEqual(tuple(qkv_scanned_shape[ax] for ax in qkv_mdn.output_axis), (8, 64))

    # Attention Out: 3D unscanned vs 4D scanned
    out_unscanned_shape = (8, 64, 512)
    out_scanned_shape = (12, 8, 64, 512)
    out_mdn = muon_utils.transform_logic(("decoder", "self_attention", "out"))

    self.assertEqual(
        tuple(out_unscanned_shape[ax] for ax in out_mdn.reduction_axis),
        (8, 64),
    )
    self.assertEqual(
        tuple(out_scanned_shape[ax] for ax in out_mdn.reduction_axis),
        (8, 64),
    )
    self.assertEqual(out_unscanned_shape[out_mdn.output_axis[0]], 512)
    self.assertEqual(out_scanned_shape[out_mdn.output_axis[0]], 512)


if __name__ == "__main__":
  unittest.main()
