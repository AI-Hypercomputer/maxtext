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

"""Tests for selective-layer weight-only N:M sparsity."""

import re
import types

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import qarray
from qwix._src.core import sparsity

from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.utils import sparsity_utils
from tests.utils.test_helpers import get_test_config_path

_RULE_2_4 = sparsity.SparsityRule(weight_sparsity_n=2, weight_sparsity_m=4)
# Smallest valid fully-MoE overrides for the test config.
_MOE_OVERRIDES = ("num_experts=4", "num_experts_per_tok=2", "base_mlp_dim=32", "base_moe_mlp_dim=32")


def _config(*overrides):
  return pyconfig.initialize(
      ["", get_test_config_path(), *overrides],
      run_name="weight_sparsity_test",
      enable_checkpointing=False,
      skip_jax_distributed_system=True,
  )


def _sparsity_config(*overrides):
  return _config("weight_sparsity_n=2", "weight_sparsity_m=4", "use_qwix_quantization=true", *overrides)


def _fake_config(**kwargs):
  """A minimal config namespace for helpers that only read a few fields."""
  defaults = {
      "weight_sparsity_n": 2,
      "weight_sparsity_m": 4,
      "quantization": "",
      "scan_layers": False,
      "sparse_layers": "",
      "sparse_experts_only": False,
      "num_decoder_layers": 4,
  }
  defaults.update(kwargs)
  return types.SimpleNamespace(**defaults)


class ParseLayerSpecTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("single", "3", [3]),
      ("range", "0-2", [0, 1, 2]),
      ("mixed", "0-1, 3,5-6", [0, 1, 3, 5, 6]),
  )
  def test_valid(self, spec, expected):
    self.assertEqual(sparsity_utils.parse_layer_spec(spec, num_layers=8), expected)

  @parameterized.named_parameters(
      ("not_a_number", "a", "expected indices or ranges"),
      ("empty_element", "0,,2", "expected indices or ranges"),
      ("descending_range", "3-1", "descending"),
      ("unordered", "4,2", "ascending and non-overlapping"),
      ("overlapping", "0-3,2-5", "ascending and non-overlapping"),
      ("out_of_range", "0-8", "out of range"),
  )
  def test_invalid(self, spec, error):
    with self.assertRaisesRegex(ValueError, error):
      sparsity_utils.parse_layer_spec(spec, num_layers=8)

  def test_unset_selects_all_layers(self):
    self.assertIsNone(sparsity_utils.get_sparse_layer_indices(_fake_config()))

  def test_scanned_layers_rejected(self):
    with self.assertRaisesRegex(ValueError, "scan_layers=False"):
      sparsity_utils.get_sparse_layer_indices(_fake_config(sparse_layers="0", scan_layers=True))


class HfParamSelectionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("dense_mlp", "model.layers.1.mlp.down_proj.weight", None, False, True),
      ("gemma4_attention", "model.language_model.layers.1.self_attn.q_proj.weight", None, False, True),
      ("unselected_layer", "model.layers.2.mlp.down_proj.weight", {0, 1}, False, False),
      ("selected_layer", "model.layers.1.mlp.down_proj.weight", {0, 1}, False, True),
      ("embedding", "model.language_model.embed_tokens.weight", None, False, False),
      ("lm_head", "lm_head.weight", None, False, False),
      ("vision_tower", "model.vision_tower.encoder.layers.1.self_attn.q_proj.linear.weight", None, False, False),
      ("gemma4_router", "model.language_model.layers.1.router.proj.weight", None, False, False),
      ("mixtral_style_router", "model.layers.1.mlp.gate.weight", None, False, False),
      ("gate_proj_is_not_router", "model.layers.1.mlp.gate_proj.weight", None, False, True),
      ("experts_only_dense", "model.layers.1.mlp.down_proj.weight", None, True, False),
      ("experts_only_fused_experts", "model.language_model.layers.1.experts.gate_up_proj", None, True, True),
      ("experts_only_split_experts", "model.layers.1.mlp.experts.3.down_proj.weight", None, True, True),
  )
  def test_selection(self, hf_path, target_layers, experts_only, expected):
    self.assertEqual(sparsity_utils.is_sparsifiable_hf_param(hf_path, target_layers, experts_only), expected)


class WeightOnlySparsityRuleTest(absltest.TestCase):

  def test_no_rules_without_sparsity(self):
    config = _config("use_qwix_quantization=true")
    self.assertIsNone(quantizations.get_quantization_rule(config))
    self.assertIsNone(quantizations.get_qt_provider(config))

  def test_all_layers(self):
    config = _sparsity_config()
    router_rule, rule = quantizations.get_quantization_rule(config)
    self.assertEqual(router_rule.module_path, r".*/gate$")
    self.assertIsNone(router_rule.rhs_sparsity_rule)
    self.assertEqual(rule.module_path, "decoder/.*layers.*")
    self.assertEqual(rule.op_names, ("dot_general", "gmm"))
    self.assertIsNone(rule.weight_qtype)
    self.assertIsNone(rule.act_qtype)
    self.assertEqual(rule.rhs_sparsity_rule.weight_sparsity_n, 2)
    self.assertEqual(rule.rhs_sparsity_rule.weight_sparsity_m, 4)
    self.assertIsNotNone(quantizations.get_qt_provider(config))

  def test_sparse_layers(self):
    config = _sparsity_config("scan_layers=false", "base_num_decoder_layers=4", "sparse_layers=0,2-3")
    rule = quantizations.get_quantization_rule(config)[-1]
    self.assertEqual(rule.module_path, "decoder/layers_(0|2|3)/.*")
    self.assertTrue(re.fullmatch(rule.module_path, "decoder/layers_2/mlp/wi_0"))
    self.assertIsNone(re.fullmatch(rule.module_path, "decoder/layers_1/mlp/wi_0"))
    self.assertIsNone(re.fullmatch(rule.module_path, "decoder/layers_20/mlp/wi_0"))

  def test_experts_only(self):
    config = _sparsity_config(*_MOE_OVERRIDES, "sparse_experts_only=true")
    self.assertEqual(quantizations.get_quantization_rule(config)[-1].op_names, ("gmm",))


class SparsityConfigValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("sparse_layers_without_n_m", ("scan_layers=false", "sparse_layers=0"), "require weight_sparsity_n"),
      ("experts_only_without_n_m", (*_MOE_OVERRIDES, "sparse_experts_only=true"), "require weight_sparsity_n"),
      (
          "sparse_layers_with_quantization",
          ("weight_sparsity_n=2", "weight_sparsity_m=4", "quantization=fp8_full", "scan_layers=false", "sparse_layers=0"),
          "only supported for weight-only sparsity",
      ),
      (
          "sparse_layers_with_scan",
          ("weight_sparsity_n=2", "weight_sparsity_m=4", "use_qwix_quantization=true", "sparse_layers=0"),
          "requires scan_layers=False",
      ),
      (
          "bad_layer_spec",
          (
              "weight_sparsity_n=2",
              "weight_sparsity_m=4",
              "use_qwix_quantization=true",
              "scan_layers=false",
              "sparse_layers=3-1",
          ),
          "descending",
      ),
      (
          "layer_out_of_range",
          (
              "weight_sparsity_n=2",
              "weight_sparsity_m=4",
              "use_qwix_quantization=true",
              "scan_layers=false",
              "base_num_decoder_layers=4",
              "sparse_layers=0-4",
          ),
          "out of range",
      ),
      (
          "experts_only_dense_model",
          ("weight_sparsity_n=2", "weight_sparsity_m=4", "use_qwix_quantization=true", "sparse_experts_only=true"),
          "num_experts > 1",
      ),
      ("sparsity_without_qwix", ("weight_sparsity_n=2", "weight_sparsity_m=4"), "use_qwix_quantization=True"),
  )
  def test_invalid(self, overrides, error):
    with self.assertRaisesRegex(ValueError, error):
      _config(*overrides)


class _Dense(nnx.Module):

  def __init__(self, shape, axis, key):
    self.kernel = nnx.Param(jax.random.normal(key, shape))
    self.axis = axis


class _MoE(nnx.Module):

  def __init__(self, key):
    keys = jax.random.split(key, 4)
    self.gate = _Dense((8, 4), (-1,), keys[0])
    self.wi_0 = nnx.Param(jax.random.normal(keys[1], (4, 8, 12)))
    self.wi_1 = nnx.Param(jax.random.normal(keys[2], (4, 8, 12)))
    self.wo = nnx.Param(jax.random.normal(keys[3], (4, 12, 8)))


class _Layer(nnx.Module):

  def __init__(self, key):
    keys = jax.random.split(key, 3)
    self.query = _Dense((8, 2, 4), (-1,), keys[0])
    self.out = _Dense((2, 4, 8), (-2, -1), keys[1])
    self.moe = _MoE(keys[2])


class _Decoder(nnx.Module):

  def __init__(self, num_layers, key):
    for i, layer_key in enumerate(jax.random.split(key, num_layers)):
      setattr(self, f"layers_{i}", _Layer(layer_key))


class _Model(nnx.Module):

  def __init__(self, num_layers=3):
    self.decoder = _Decoder(num_layers, jax.random.key(0))


class InitialWeightSparsityTest(absltest.TestCase):

  def _assert_sparsified(self, actual, original, axis):
    np.testing.assert_array_equal(actual, qarray.sparsify(original, _RULE_2_4, axis=axis))

  def test_selected_layers_are_sparsified_along_contracting_axis(self):
    model = _Model()
    original = nnx.clone(model)
    count = quantizations.apply_initial_weight_sparsity(model, _fake_config(sparse_layers="0,2", num_decoder_layers=3))
    self.assertEqual(count, 2 * 5)
    for i in (0, 2):
      layer, orig = getattr(model.decoder, f"layers_{i}"), getattr(original.decoder, f"layers_{i}")
      self._assert_sparsified(layer.query.kernel.get_value(), orig.query.kernel.get_value(), axis=0)
      self._assert_sparsified(layer.out.kernel.get_value(), orig.out.kernel.get_value(), axis=1)
      for name in ("wi_0", "wi_1", "wo"):
        self._assert_sparsified(getattr(layer.moe, name).get_value(), getattr(orig.moe, name).get_value(), axis=1)
      # The router is never sparsified.
      np.testing.assert_array_equal(layer.moe.gate.kernel.get_value(), orig.moe.gate.kernel.get_value())
    # Unselected layers are untouched.
    jax.tree.map(np.testing.assert_array_equal, nnx.state(model.decoder.layers_1), nnx.state(original.decoder.layers_1))

  def test_experts_only(self):
    model = _Model(num_layers=1)
    original = nnx.clone(model)
    count = quantizations.apply_initial_weight_sparsity(model, _fake_config(sparse_experts_only=True))
    self.assertEqual(count, 3)
    layer, orig = model.decoder.layers_0, original.decoder.layers_0
    self._assert_sparsified(layer.moe.wo.get_value(), orig.moe.wo.get_value(), axis=1)
    np.testing.assert_array_equal(layer.query.kernel.get_value(), orig.query.kernel.get_value())

  def test_noop_without_sparsity_or_with_scanned_layers(self):
    for config in (_fake_config(weight_sparsity_n=None), _fake_config(scan_layers=True)):
      model = _Model(num_layers=1)
      original = nnx.clone(model)
      self.assertEqual(quantizations.apply_initial_weight_sparsity(model, config), 0)
      jax.tree.map(np.testing.assert_array_equal, nnx.state(model), nnx.state(original))


class ExportSparsityTest(absltest.TestCase):

  def test_export_masks_selected_decoder_weights(self):
    rng = np.random.default_rng(0)
    weights = {
        "model.language_model.layers.0.self_attn.q_proj.weight": rng.normal(size=(6, 8)).astype(np.float32),
        "model.language_model.layers.0.experts.gate_up_proj": rng.normal(size=(2, 6, 8)).astype(np.float32),
        "model.language_model.layers.0.router.proj.weight": rng.normal(size=(2, 8)).astype(np.float32),
        "model.language_model.layers.0.input_layernorm.weight": rng.normal(size=(8,)).astype(np.float32),
        "model.language_model.layers.1.self_attn.q_proj.weight": rng.normal(size=(6, 8)).astype(np.float32),
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": rng.normal(size=(6, 8)).astype(np.float32),
    }
    config = _fake_config(sparse_layers="0", num_decoder_layers=2)
    # pylint: disable-next=protected-access
    exported = dict(conversion_utils._maybe_sparsify_outputs(list(weights.items()), config))

    sparsified = (
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.experts.gate_up_proj",
    )
    for name, weight in weights.items():
      if name in sparsified:
        np.testing.assert_array_equal(exported[name], qarray.sparsify(jnp.asarray(weight), _RULE_2_4, axis=-1))
      else:
        np.testing.assert_array_equal(exported[name], weight)

  def test_export_noop_without_sparsity(self):
    weights = [("model.layers.0.mlp.down_proj.weight", np.ones((4, 8), np.float32))]
    config = _fake_config(weight_sparsity_n=None)
    # pylint: disable-next=protected-access
    self.assertIs(conversion_utils._maybe_sparsify_outputs(weights, config), weights)


if __name__ == "__main__":
  absltest.main()
