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
import os
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from optax.contrib._muon import MuonDimensionNumbers as mdn

from maxtext.configs import pyconfig
from maxtext.layers import nnx_scan
from maxtext.utils import maxtext_muon_utils, maxtext_utils, model_creation_utils, muon_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR


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

  def test_bias_variant_is_excluded(self):
    # Bias names (e.g., in MoE with mlp_bias=True) are excluded across all expert projections.
    self.assertIsNone(muon_utils.transform_logic(("decoder", "moe_block", "wi_0_bias")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "GptOssMlp", "wi_0_bias")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "GptOssMlp", "wi_1_bias")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "GptOssMlp", "wo_bias")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "mlp", "mlp_bias")))

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

  def test_moe_prefused_wi_uses_last_two_axes(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "MoeBlock_0", "wi")), mdn((-2,), (-1,)))
    self.assertEqual(muon_utils.transform_logic(("decoder", "routed_experts", "wi")), mdn((-2,), (-1,)))

  def test_qwen3_moe_block(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "moe_block", "wi_0")), mdn((-2,), (-1,)))
    self.assertEqual(muon_utils.transform_logic(("decoder", "moe_block", "wo")), mdn((-2,), (-1,)))

  def test_gpt_oss_mlp_moe(self):
    self.assertEqual(muon_utils.transform_logic(("decoder", "GptOssMlp", "wi_0")), mdn((-2,), (-1,)))
    self.assertEqual(muon_utils.transform_logic(("decoder", "GptOssMlp", "wo")), mdn((-2,), (-1,)))

  def test_moe_gate_uses_standard_axes(self):
    # 'gate' is a 2D router matrix [in_features, (num_layers), num_experts] -> standard (0,), (-1,).
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate", "kernel")),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate", "kernel"), include_routers=True),
        mdn((0,), (-1,)),
    )

  def test_moe_gate_excluded_when_include_routers_false(self):
    # When include_routers=False, MoE router/gate weights return None and optimize with AdamW.
    self.assertIsNone(muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate", "kernel"), include_routers=False))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "moe_block", "gate", "kernel"), include_routers=False))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "GptOssMlp", "gate", "kernel"), include_routers=False))
    self.assertIsNone(
        muon_utils.transform_logic(
            ("decoder", "routed_experts", "gate", "kernel"),
            include_routers=False,
        )
    )
    self.assertIsNone(
        muon_utils.transform_logic(
            ("decoder", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "kernel"),
            include_routers=False,
        )
    )
    self.assertIsNone(
        muon_utils.transform_logic(
            ("decoder", "Llama4MoEBlock_0", "MoeBlock_0", "gate", "kernel"),
            include_routers=False,
        )
    )
    self.assertIsNone(muon_utils.transform_logic(("decoder", "routed_moe", "gate", "kernel"), include_routers=False))

  def test_deepseek_and_llama4_routed_experts_use_last_two_axes(self):
    # Routed experts inside DeepSeek / LLaMA 4 MoE blocks are 3D/4D and use last two axes.
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "DeepSeekMoeBlock_0", "MoeBlock_0", "wi_0")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "DeepSeekMoeBlock_0", "MoeBlock_0", "wo")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "Llama4MoEBlock_0", "MoeBlock_0", "wi_0")),
        mdn((-2,), (-1,)),
    )

  def test_moe_shared_experts_use_standard_axes(self):
    # Shared experts in DeepSeek/LLaMA 4 are standard 2D MLPs and must not use routed expert axes.
    self.assertEqual(
        muon_utils.transform_logic(
            (
                "decoder",
                "DeepSeekMoeBlock_0",
                "shared_experts",
                "wi_0",
                "kernel",
            ),
            include_routers=True,
        ),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(
            (
                "decoder",
                "DeepSeekMoeBlock_0",
                "shared_experts",
                "wi_0",
                "kernel",
            ),
            include_routers=False,
        ),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(
            ("decoder", "DeepSeekMoeBlock_0", "shared_experts", "wo", "kernel"),
            include_routers=True,
        ),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(
            ("decoder", "Llama4MoEBlock_0", "shared_experts", "wi_0", "kernel"),
            include_routers=True,
        ),
        mdn((0,), (-1,)),
    )

  def test_moe_expert_weights_unaffected_by_include_routers_false(self):
    # Expert weights (wi, wo, gate_up_proj) must remain mdn((-2,), (-1,)) even if include_routers=False.
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "wi_0"), include_routers=False),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "wi_1"), include_routers=False),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "wo"), include_routers=False),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate_up_proj"), include_routers=False),
        mdn((-2,), (-1,)),
    )

  def test_dense_mlp_gate_proj_unaffected_by_include_routers_false(self):
    # Dense MLP gate projections (e.g. SwiGLU) are not MoE routers and must remain standard mdn.
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "gate_proj"), include_routers=False),
        mdn((0,), (-1,)),
    )

  def test_gate_up_proj_uses_last_two_axes(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "MoeBlock_0", "gate_up_proj")),
        mdn((-2,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "gate_proj")),
        mdn((0,), (-1,)),
    )

  def test_1d_tensor_is_excluded(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "mlp", "custom_param"), shape=(512,)))

  def test_2d_attention_projections_use_2d_axes(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "query"), shape=(512, 512)),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "out"), shape=(512, 512)),
        mdn((0,), (-1,)),
    )

  # --- 2.2 Self-attention ---
  def test_self_attention_out_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "out")),
        mdn((0, -2), (-1,)),
    )

  def test_self_attention_query_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "query")),
        mdn((0,), (-2, -1)),
    )

  def test_self_attention_key_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "key")),
        mdn((0,), (-2, -1)),
    )

  def test_self_attention_value_projection(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "value")),
        mdn((0,), (-2, -1)),
    )

  def test_self_attention_wq_b_and_wkv_b(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wq_b")),
        mdn((0,), (-2, -1)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wkv_b")),
        mdn((0,), (-2, -1)),
    )

  def test_self_attention_mla_wq_a_is_standard(self):
    # wq_a / wkv_a are MLA down-projections; they fall through the self_attention branch
    # to standard mdn((0,), (-1,)).
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wq_a")),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "wkv_a")),
        mdn((0,), (-1,)),
    )

  def test_self_attention_mla_output_gate(self):
    # Head-expanded MLA g_proj (Kimi-K3): [emb, heads, v_head_dim] (unscanned) / [emb, layers, heads, v_head_dim]
    # (scanned) orthogonalizes [emb, heads * v_head_dim] like wq_b / wkv_b.
    for shape in ((256, 4, 32), (256, 2, 4, 32)):
      self.assertEqual(
          muon_utils.transform_logic(("decoder", "layers_3", "self_attention", "g_proj", "kernel"), shape=shape),
          mdn((0,), (-2, -1)),
      )
    # Flat 2D g_proj (e.g. KDA) keeps the standard mapping.
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "layers_0", "self_attention", "g_proj", "kernel"), shape=(256, 128)),
        mdn((0,), (-1,)),
    )
    # A flat g_proj under a scanned stack is 3D [emb, layers, out]: the layer axis stays a batch axis.
    for path in (
        ("decoder", "scanned_blocks", "layers_0", "self_attention", "g_proj", "kernel"),  # Kimi-K3 scanned KDA
        ("decoder", "layers", "self_attention", "g_proj", "kernel"),  # e.g. Ming scanned KDA / MLA
    ):
      self.assertEqual(muon_utils.transform_logic(path, shape=(256, 22, 128)), mdn((0,), (-1,)))
    # The scanned head-expanded MLA gate is 4D and keeps the head-aware mapping.
    self.assertEqual(
        muon_utils.transform_logic(
            ("decoder", "scanned_blocks", "layers_3", "self_attention", "g_proj", "kernel"), shape=(256, 22, 4, 32)
        ),
        mdn((0,), (-2, -1)),
    )

  def test_kimi_k3_scanned_kda_out_is_standard(self):
    # Kimi-K3 scanned KDA `out` is a flat [in, layers, out] projection, not a head-expanded [heads, head_dim, out].
    self.assertEqual(
        muon_utils.transform_logic(
            ("decoder", "scanned_blocks", "layers_0", "self_attention", "out", "kernel"), shape=(384, 22, 256)
        ),
        mdn((0,), (-1,)),
    )
    # Scanned MLA `out` is 4D [heads, layers, head_dim, out] and keeps the head-aware mapping.
    self.assertEqual(
        muon_utils.transform_logic(
            ("decoder", "scanned_blocks", "layers_3", "self_attention", "out", "kernel"), shape=(4, 22, 32, 256)
        ),
        mdn((0, -2), (-1,)),
    )

  # --- 3. Standard ---
  def test_standard_weight(self):
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "mlp", "kernel")),
        mdn((0,), (-1,)),
    )

  # --- 4. DeepSeek V4 Specific ---
  def test_deepseek_v4_exclusions(self):
    # Scale, beta, base, sinks, and tid2eid nested segments should be excluded (None)
    self.assertIsNone(muon_utils.transform_logic(("decoder", "hc_head", "hc_scale")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "hc_head", "hc_base")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "layers", "layers_0", "mhc_attention", "res_beta")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "layers", "layers_0", "mhc_attention", "res_alpha_scale")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "layers", "layers_0", "self_attention", "sinks")))
    self.assertIsNone(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "mlp", "MoeBlock_0", "gate", "tid2eid"))
    )
    self.assertIsNone(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "self_attention", "rotary_embedding", "inv_freq"))
    )

  def test_deepseek_v4_mhc_alpha_projections(self):
    # Alpha projection matrices in MHC are 2D dense linear projections
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "mhc_attention", "pre_alpha")),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "mhc_attention", "post_alpha")),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "layers", "layers_0", "mhc_attention", "res_alpha")),
        mdn((0,), (-1,)),
    )

  def test_deepseek_v4_self_attention_grouped_projection(self):
    # o_a_proj is a grouped linear layer [o_groups, (layers), in_features_per_group, out_features_per_group]
    # with reduction on in_features_per_group (-2) and output on out_features_per_group (-1)
    self.assertEqual(muon_utils.transform_logic(("decoder", "self_attention", "o_a_proj")), mdn((-2,), (-1,)))

  def test_deepseek_v4_position_bias_is_standard_weight(self):
    # position_bias in DeepSeek V4 compressed attention is a 2D weight matrix (not a 1D bias vector),
    # so it receives standard Muon dimension numbers mdn((0,), (-1,)).
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "csa_compressor", "position_bias")),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "self_attention", "hca_compressor", "position_bias")),
        mdn((0,), (-1,)),
    )

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
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "gdn", "in_proj_ba")),
        mdn((0,), (-1,)),
    )
    self.assertEqual(
        muon_utils.transform_logic(("decoder", "gdn", "out_proj")),
        mdn((0,), (-1,)),
    )

  def test_qwen3_next_exclusions(self):
    self.assertIsNone(muon_utils.transform_logic(("decoder", "gdn", "A_log")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "gdn", "dt_bias")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "gdn", "conv1d")))
    self.assertIsNone(muon_utils.transform_logic(("decoder", "mlp", "shared_expert_gate")))


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

  def test_get_transform_tree_with_include_routers_toggle(self):
    tree = {
        "decoder": {
            "MoeBlock_0": {
                "gate": {"kernel": 0},
                "wi_0": 0,
            },
            "mlp": {"gate_proj": 0},
        }
    }
    result_true = muon_utils.get_transform_tree(tree, include_routers=True)
    self.assertEqual(result_true["decoder"]["MoeBlock_0"]["gate"]["kernel"], mdn((0,), (-1,)))
    self.assertEqual(result_true["decoder"]["MoeBlock_0"]["wi_0"], mdn((-2,), (-1,)))
    self.assertEqual(result_true["decoder"]["mlp"]["gate_proj"], mdn((0,), (-1,)))

    result_false = muon_utils.get_transform_tree(tree, include_routers=False)
    self.assertIsNone(result_false["decoder"]["MoeBlock_0"]["gate"]["kernel"])
    self.assertEqual(result_false["decoder"]["MoeBlock_0"]["wi_0"], mdn((-2,), (-1,)))
    self.assertEqual(result_false["decoder"]["mlp"]["gate_proj"], mdn((0,), (-1,)))


class _AttentionSubModule(nnx.Module):

  def __init__(self):
    self.out = nnx.Param(jnp.ones((2, 4, 8)))


class _MoeSubModule(nnx.Module):

  def __init__(self):
    self.gate = nnx.Module()
    self.gate.kernel = nnx.Param(jnp.ones((4, 8)))
    self.wi_0 = nnx.Param(jnp.ones((8, 4, 8)))


class _MoeLikeNNXModel(nnx.Module):
  """Small NNX model whose param paths exercise the NNX branch of get_muon_weight_dimension_numbers."""

  def __init__(self, rngs):
    # Names are chosen so transform_logic matches each of the meaningful branches:
    # - w_standard: default mdn
    # - self_attention.out: attention-out mdn
    # - scale: excluded (None)
    # - MoeBlock_0: router gate + routed expert weights
    self.w_standard = nnx.Param(jnp.ones((4, 8)))
    self.self_attention = _AttentionSubModule()
    self.scale = nnx.Param(jnp.ones((8,)))
    self.MoeBlock_0 = _MoeSubModule()


class TestGetMuonWeightDimensionNumbersNNX(unittest.TestCase):
  """Covers the NNX branch of get_muon_weight_dimension_numbers (isinstance(model, nnx.Module))."""

  def setUp(self):
    self.model = _MoeLikeNNXModel(rngs=nnx.Rngs(0))

  def test_nnx_model_dispatches_to_tree_map_with_path(self):
    """NNX branch should produce an nnx.State tree with transform_logic applied per leaf."""
    result = muon_utils.get_muon_weight_dimension_numbers(self.model, config=None)

    # Result is an nnx.State whose top-level keys mirror the model attributes.
    self.assertIn("w_standard", result)
    self.assertIn("self_attention", result)
    self.assertIn("out", result["self_attention"])
    self.assertIn("scale", result)

    # NNX Variables are walked by jax.tree_util.tree_map_with_path, so the returned
    # tree replaces each Variable's value with transform_logic(path_strings).
    # 'scale' matches the exclusion branch → value is None.
    self.assertIsNone(result["scale"])
    # 'w_standard' does not trigger any special rule → standard mdn.
    self.assertEqual(result["w_standard"], mdn((0,), (-1,)))
    self.assertEqual(result["self_attention"]["out"], mdn((0, -2), (-1,)))
    self.assertEqual(result["MoeBlock_0"]["gate"]["kernel"], mdn((0,), (-1,)))
    self.assertEqual(result["MoeBlock_0"]["wi_0"], mdn((-2,), (-1,)))

  def test_nnx_model_respects_muon_include_routers_config(self):
    cfg_true = mock.MagicMock(muon_include_routers=True)
    result_true = muon_utils.get_muon_weight_dimension_numbers(self.model, config=cfg_true)
    self.assertEqual(result_true["MoeBlock_0"]["gate"]["kernel"], mdn((0,), (-1,)))
    self.assertEqual(result_true["MoeBlock_0"]["wi_0"], mdn((-2,), (-1,)))

    cfg_false = mock.MagicMock(muon_include_routers=False)
    result_false = muon_utils.get_muon_weight_dimension_numbers(self.model, config=cfg_false)
    self.assertIsNone(result_false["MoeBlock_0"]["gate"]["kernel"])
    self.assertEqual(result_false["MoeBlock_0"]["wi_0"], mdn((-2,), (-1,)))

  def test_nnx_verbose_path_executes_print_debug(self):
    """verbose=True should also execute _print_structure_debug without raising."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      muon_utils.get_muon_weight_dimension_numbers(self.model, config=None, verbose=True)
    self.assertIn("Model Structure", buf.getvalue())
    self.assertIn("Muon Dimension Numbers", buf.getvalue())


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
          muon_weight_dimension_numbers={"params": {"kernel": mdn((0,), (-1,))}},
      )
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
    # Place layer dimension (num_layers=12) on axis 1 (param_scan_axis=1) for
    # all layers to simulate layer scanning.
    scanned_tree = {
        "decoder": {
            "self_attention": {
                "query": jax.ShapeDtypeStruct((512, 12, 8, 64), jnp.float32),
                "out": jax.ShapeDtypeStruct((8, 12, 64, 512), jnp.float32),
            },
            "mlp": {
                "wi_0": jax.ShapeDtypeStruct((512, 12, 2048), jnp.float32),
                "wo": jax.ShapeDtypeStruct((2048, 12, 512), jnp.float32),
            },
            "norm": {
                "scale": jax.ShapeDtypeStruct((512, 12), jnp.float32),
            },
        }
    }

    unscanned_mdn = muon_utils.get_transform_tree(unscanned_tree)
    scanned_mdn = muon_utils.get_transform_tree(scanned_tree)

    self.assertEqual(unscanned_mdn, scanned_mdn)

  def test_relative_dimensions_resolve_consistent_matrix_features(self):
    """Dimension numbers index same feature axes regardless of scan axis."""
    # Standard MLP: 2D unscanned vs 3D scanned (param_scan_axis = 1, num_layers = 12)
    mlp_unscanned_shape = (512, 2048)
    mlp_scanned_shape = (512, 12, 2048)
    mlp_mdn = muon_utils.transform_logic(("decoder", "mlp", "kernel"))

    # Reduction axis (0) and Output axis (-1) point to identical feature sizes
    self.assertEqual(mlp_unscanned_shape[mlp_mdn.reduction_axis[0]], 512)
    self.assertEqual(mlp_scanned_shape[mlp_mdn.reduction_axis[0]], 512)
    self.assertEqual(mlp_unscanned_shape[mlp_mdn.output_axis[0]], 2048)
    self.assertEqual(mlp_scanned_shape[mlp_mdn.output_axis[0]], 2048)

    # Attention QKV: 3D unscanned vs 4D scanned (param_scan_axis = 1, num_layers = 12)
    qkv_unscanned_shape = (512, 8, 64)
    qkv_scanned_shape = (512, 12, 8, 64)
    qkv_mdn = muon_utils.transform_logic(("decoder", "self_attention", "query"))

    self.assertEqual(qkv_unscanned_shape[qkv_mdn.reduction_axis[0]], 512)
    self.assertEqual(qkv_scanned_shape[qkv_mdn.reduction_axis[0]], 512)
    self.assertEqual(tuple(qkv_unscanned_shape[ax] for ax in qkv_mdn.output_axis), (8, 64))
    self.assertEqual(tuple(qkv_scanned_shape[ax] for ax in qkv_mdn.output_axis), (8, 64))

    # Attention Out: 3D unscanned vs 4D scanned (param_scan_axis = 1, num_layers = 12)
    out_unscanned_shape = (8, 64, 512)
    out_scanned_shape = (8, 12, 64, 512)
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


def _scanned_param(shape, scan_axis, logical_axes, partition_name="layers"):
  """An nnx.Param carrying the scan metadata `nnx_scan.create_scanned_layers` attaches."""
  return nnx.Param(
      jnp.zeros(shape),
      sharding_names=logical_axes,
      **{nnx.PARTITION_NAME: partition_name, "param_scan_axis": scan_axis},
  )


class TestScanAxes(unittest.TestCase):
  """get_scan_axes reads the layer axes of a scanned parameter from its metadata."""

  def test_unscanned_param_has_no_scan_axes(self):
    self.assertEqual(muon_utils.get_scan_axes(nnx.Param(jnp.zeros((4, 8)))), ())

  def test_scan_axis_comes_from_metadata(self):
    self.assertEqual(muon_utils.get_scan_axes(_scanned_param((3, 4, 8), 0, ("layers", "embed", "mlp"))), (0,))
    self.assertEqual(muon_utils.get_scan_axes(_scanned_param((4, 3, 8), 1, ("embed", "layers", "mlp"))), (1,))

  def test_nested_block_scan_adds_the_local_layers_axis(self):
    param = _scanned_param((2, 3, 4, 8), 0, ("layers", "local_layers", "embed", "mlp"))
    self.assertEqual(muon_utils.get_scan_axes(param), (0, 1))
    param = _scanned_param((4, 2, 3, 8), 1, ("embed", "layers", "local_layers", "mlp"))
    self.assertEqual(muon_utils.get_scan_axes(param), (1, 2))

  def test_list_logical_axes(self):
    param = _scanned_param((4, 2, 3, 8), 1, ["embed", "layers", "local_layers", "mlp"])
    self.assertEqual(muon_utils.get_scan_axes(param), (1, 2))

  def test_nested_scan_of_list_named_layers(self):
    # create_scanned_layers normalizes list logical axes, so a nested stack still records its local_layers axis.
    class Layer(nnx.Module):

      def __init__(self, rngs):
        del rngs  # Unused.
        self.kernel = nnx.Param(jnp.zeros((4, 8)), sharding_names=["embed", "mlp"])

    class Block(nnx.Module):

      def __init__(self, rngs):
        self.local_layers = nnx_scan.create_scanned_layers(
            Layer, length=3, param_scan_axis=1, metadata_axis_name="local_layers", rngs=rngs
        )

    blocks = nnx_scan.create_scanned_layers(
        Block, length=2, param_scan_axis=1, metadata_axis_name="layers", rngs=nnx.Rngs(0)
    )
    self.assertEqual(blocks.local_layers.kernel.get_value().shape, (4, 2, 3, 8))
    self.assertEqual(muon_utils.get_scan_axes(blocks.local_layers.kernel), (1, 2))

  def test_inner_scan_alone_is_a_single_axis(self):
    param = _scanned_param((3, 4, 8), 0, ("local_layers", "embed", "mlp"), partition_name="local_layers")
    self.assertEqual(muon_utils.get_scan_axes(param), (0,))

  def test_stack_too_low_rank_for_the_scan_axis_keeps_it_in_front(self):
    self.assertEqual(muon_utils.get_scan_axes(_scanned_param((3,), 1, ("layers",))), (0,))


class TestToStackedLayout(unittest.TestCase):
  """to_stacked_layout maps feature-axis dimension numbers onto the stacked layout."""

  def test_identity_without_scan_axes_or_numbers(self):
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0,), (-1,)), 2, ()), mdn((0,), (-1,)))
    self.assertIsNone(muon_utils.to_stacked_layout(None, 3, (0,)))

  def test_scan_axis_1_leaves_the_numbers_unchanged(self):
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0,), (-1,)), 3, (1,)), mdn((0,), (-1,)))
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0, -2), (-1,)), 4, (1,)), mdn((0, -2), (-1,)))

  def test_scan_axis_0_shifts_the_leading_feature_axes(self):
    # [layers, in, out]: reduce over `in`, not over the layers.
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0,), (-1,)), 3, (0,)), mdn((1,), (-1,)))
    # [layers, heads, head_dim, out]
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0, -2), (-1,)), 4, (0,)), mdn((1, -2), (-1,)))
    # [layers, experts, in, out]
    self.assertEqual(muon_utils.to_stacked_layout(mdn((-2,), (-1,)), 4, (0,)), mdn((-2,), (-1,)))

  def test_stacked_vectors_are_left_alone(self):
    # [layers, features]: one feature axis, so reduction and output cannot both be feature axes.
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0,), (-1,)), 2, (0,)), mdn((0,), (-1,)))

  def test_nested_scan_axes(self):
    # [blocks, local, in, out] and [in, blocks, local, out]
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0,), (-1,)), 4, (0, 1)), mdn((2,), (-1,)))
    self.assertEqual(muon_utils.to_stacked_layout(mdn((0,), (-1,)), 4, (1, 2)), mdn((0,), (-1,)))


class _ScannedNNXModel(nnx.Module):
  """Layers stacked on axis 0, as `param_scan_axis=0` lays them out."""

  def __init__(self):
    self.layers = nnx.Module()
    self.layers.mlp = _scanned_param((3, 4, 8), 0, ("layers", "embed", "mlp"))
    self.layers.self_attention = nnx.Module()
    self.layers.self_attention.out = _scanned_param((3, 2, 4, 8), 0, ("layers", "heads", "kv", "embed"))


class TestScannedNNXModel(unittest.TestCase):
  """get_muon_weight_dimension_numbers never reduces over a scanned parameter's layer axis."""

  def test_layer_axis_is_skipped(self):
    result = muon_utils.get_muon_weight_dimension_numbers(_ScannedNNXModel(), config=None)
    self.assertEqual(result["layers"]["mlp"], mdn((1,), (-1,)))
    self.assertEqual(result["layers"]["self_attention"]["out"], mdn((1, -2), (-1,)))


def _dimension_numbers_by_path(model_name, param_scan_axis, sharded=False):
  """Returns {path: (shape, scan axes, Muon dimension numbers)} for every parameter of `model_name`.

  `sharded` selects MaxText's Muon (`ShardedMuonDimensionNumbers`) over Optax's (`MuonDimensionNumbers`).
  """
  config = pyconfig.initialize(
      [
          None,
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          f"model_name={model_name}",
          f"param_scan_axis={param_scan_axis}",
          "attention=dot_product",
          "skip_jax_distributed_system=True",
          # A layer count unlike any feature dim, so that indexing a layer axis cannot pass for a feature axis.
          "override_model_config=True",
          "base_num_decoder_layers=20",
      ]
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
  _, model = model_creation_utils.create_nnx_abstract_model(config, mesh)
  _, params, _ = nnx.split(model, nnx.Param, ...)
  if sharded:
    dimension_numbers = maxtext_muon_utils.get_maxtext_muon_weight_dimension_numbers(model, config, mesh=mesh)
  else:
    dimension_numbers = muon_utils.get_muon_weight_dimension_numbers(model, config)
  return {
      tuple(path): (variable.get_value().shape, muon_utils.get_scan_axes(variable), dim)
      for (path, variable), (_, dim) in zip(nnx.to_flat_state(params), nnx.to_flat_state(dimension_numbers), strict=True)
  }


class TestScannedLayoutDimensionNumbers(parameterized.TestCase):
  """Muon reduces over the same feature axes of a real model whatever its param_scan_axis."""

  @parameterized.product(
      model_name=("llama2-7b", "qwen3-30b-a3b", "qwen3-next-80b-a3b", "deepseek3-test"), sharded=(False, True)
  )
  def test_feature_axes_match_across_scan_axes(self, model_name, sharded):
    layer_major = _dimension_numbers_by_path(model_name, param_scan_axis=0, sharded=sharded)
    feature_major = _dimension_numbers_by_path(model_name, param_scan_axis=1, sharded=sharded)
    self.assertEqual(layer_major.keys(), feature_major.keys())

    def sizes(shape, axes):
      return tuple(shape[axis] for axis in axes)

    def feature_spec(shape, scan_axes, sharding):
      # The partitioning of each feature axis, with the layer-scan axes dropped.
      self.assertLessEqual(len(sharding.spec), len(shape))
      spec = tuple(sharding.spec) + (None,) * (len(shape) - len(sharding.spec))
      return tuple(entry for axis, entry in enumerate(spec) if axis not in scan_axes)

    remapped = 0
    for path, (shape, scan_axes, dim) in layer_major.items():
      reference_shape, reference_scan_axes, reference_dim = feature_major[path]
      with self.subTest(path=path):
        self.assertEqual(dim is None, reference_dim is None)
        if dim is None:
          continue
        self.assertEqual(sizes(shape, dim.reduction_axis), sizes(reference_shape, reference_dim.reduction_axis))
        self.assertEqual(sizes(shape, dim.output_axis), sizes(reference_shape, reference_dim.output_axis))
        if sharded:
          self.assertEqual(
              feature_spec(shape, scan_axes, dim.sharding),
              feature_spec(reference_shape, reference_scan_axes, reference_dim.sharding),
          )
        remapped += (dim.reduction_axis, dim.output_axis) != (reference_dim.reduction_axis, reference_dim.output_axis)
    # The scanned parameters' dimension numbers do move with the layer axis.
    self.assertGreater(remapped, 0)


if __name__ == "__main__":
  unittest.main()
