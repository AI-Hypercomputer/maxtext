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

"""Tests for linen_nnx_converter utilities."""

import unittest
import numpy as np

from maxtext.checkpoint_conversion.linen_nnx_converter import (
    detect_format,
    _has_params_in_opt_state,
    _has_value_wrappers,
    _strip_value_wrappers,
    _add_value_wrappers,
    _transpose_layers_axes,
    _stack_layers,
    convert_linen_to_nnx,
    convert_nnx_to_linen,
    _convert_opt_state_linen_to_nnx,
    _convert_opt_state_nnx_to_linen,
)


def _make_array(*shape):
  """Helper to create a numpy array with given shape."""
  return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


class TestDetectFormat(unittest.TestCase):

  def test_raises_when_no_params_key(self):
    with self.assertRaises(ValueError):
      detect_format({"step": 0})

  def test_detects_linen_format_double_nested(self):
    state = {"params": {"params": {"decoder": {"layers": {}}}}}
    self.assertEqual(detect_format(state), "linen")

  def test_detects_nnx_format_single_nested(self):
    state = {"params": {"decoder": {"layers": {}}}}
    self.assertEqual(detect_format(state), "nnx")

  def test_detects_linen_via_encoder(self):
    state = {"params": {"params": {"encoder": {"layers": {}}}}}
    self.assertEqual(detect_format(state), "linen")

  def test_detects_nnx_via_encoder(self):
    state = {"params": {"encoder": {"layers": {}}}}
    self.assertEqual(detect_format(state), "nnx")

  def test_detects_linen_via_opt_state(self):
    arr = _make_array(2, 2)
    state = {
        "params": {"something": arr},
        "opt_state": {"params": {"mu": {"decoder": {"kernel": arr}}}},
    }
    self.assertEqual(detect_format(state), "linen")

  def test_detects_nnx_via_opt_state_value_wrappers(self):
    arr = _make_array(2, 2)
    state = {
        "params": {"something": arr},
        "opt_state": {"mu": {"decoder": {"kernel": {"value": arr}}}},
    }
    self.assertEqual(detect_format(state), "nnx")

  def test_raises_on_undetectable_format(self):
    state = {"params": {"some_unknown_key": 42}}
    with self.assertRaises(ValueError):
      detect_format(state)


class TestHasParamsInOptState(unittest.TestCase):

  def test_returns_true_when_params_key_present(self):
    self.assertTrue(_has_params_in_opt_state({"params": {}}))

  def test_returns_true_when_params_nested(self):
    self.assertTrue(_has_params_in_opt_state({"mu": {"params": {}}}))

  def test_returns_false_when_no_params(self):
    self.assertFalse(_has_params_in_opt_state({"mu": {"decoder": {}}}))

  def test_returns_false_for_empty_dict(self):
    self.assertFalse(_has_params_in_opt_state({}))

  def test_returns_false_for_non_dict(self):
    self.assertFalse(_has_params_in_opt_state(42))


class TestHasValueWrappers(unittest.TestCase):

  def test_returns_true_for_value_wrapper(self):
    arr = _make_array(2, 2)
    self.assertTrue(_has_value_wrappers({"value": arr}))

  def test_returns_true_for_nested_value_wrapper(self):
    arr = _make_array(2, 2)
    self.assertTrue(_has_value_wrappers({"mu": {"value": arr}}))

  def test_returns_false_for_plain_array(self):
    # A plain array is not a {"value": ...} wrapper dict
    self.assertFalse(_has_value_wrappers(_make_array(2, 2)))

  def test_returns_false_for_multi_key_dict(self):
    arr = _make_array(2, 2)
    self.assertFalse(_has_value_wrappers({"value": arr, "extra": arr}))

  def test_returns_false_for_non_array_value(self):
    self.assertFalse(_has_value_wrappers({"value": "string"}))


class TestStripValueWrappers(unittest.TestCase):

  def test_strips_single_wrapper(self):
    arr = _make_array(3, 4)
    result = _strip_value_wrappers({"value": arr})
    np.testing.assert_array_equal(result, arr)

  def test_strips_nested_wrappers(self):
    arr = _make_array(2, 2)
    wrapped = {"decoder": {"layers": {"kernel": {"value": arr}}}}
    stripped = _strip_value_wrappers(wrapped)
    np.testing.assert_array_equal(stripped["decoder"]["layers"]["kernel"], arr)

  def test_passes_through_plain_array(self):
    arr = _make_array(2, 3)
    result = _strip_value_wrappers(arr)
    np.testing.assert_array_equal(result, arr)

  def test_handles_list_and_tuple(self):
    arr = _make_array(2)
    result_list = _strip_value_wrappers([{"value": arr}])
    result_tuple = _strip_value_wrappers(({"value": arr},))
    np.testing.assert_array_equal(result_list[0], arr)
    np.testing.assert_array_equal(result_tuple[0], arr)

  def test_passes_through_non_array_value(self):
    # A dict with key "value" but scalar content should not be unwrapped
    d = {"value": 42}
    result = _strip_value_wrappers(d)
    self.assertEqual(result, d)


class TestAddValueWrappers(unittest.TestCase):

  def test_wraps_array(self):
    arr = _make_array(3, 4)
    result = _add_value_wrappers(arr)
    self.assertIsInstance(result, dict)
    self.assertIn("value", result)
    np.testing.assert_array_equal(result["value"], arr)

  def test_wraps_nested_arrays(self):
    arr = _make_array(2, 2)
    nested = {"decoder": {"layers": {"kernel": arr}}}
    wrapped = _add_value_wrappers(nested)
    self.assertEqual(set(wrapped["decoder"]["layers"]["kernel"].keys()), {"value"})
    np.testing.assert_array_equal(wrapped["decoder"]["layers"]["kernel"]["value"], arr)

  def test_idempotent_on_already_wrapped(self):
    arr = _make_array(2)
    already_wrapped = {"value": arr}
    result = _add_value_wrappers(already_wrapped)
    # Should not double-wrap
    self.assertEqual(set(result.keys()), {"value"})
    np.testing.assert_array_equal(result["value"], arr)

  def test_handles_list_and_tuple(self):
    arr = _make_array(2)
    result_list = _add_value_wrappers([arr])
    result_tuple = _add_value_wrappers((arr,))
    self.assertEqual(set(result_list[0].keys()), {"value"})
    self.assertEqual(set(result_tuple[0].keys()), {"value"})

  def test_passes_through_non_array_scalars(self):
    result = _add_value_wrappers(42)
    self.assertEqual(result, 42)
    result_str = _add_value_wrappers("text")
    self.assertEqual(result_str, "text")


class TestTransposeLayersAxes(unittest.TestCase):

  def test_noop_when_same_axis(self):
    arr = _make_array(4, 2, 3)
    result = _transpose_layers_axes(arr, src_axis=0, dst_axis=0)
    np.testing.assert_array_equal(result, arr)

  def test_transposes_axis_0_to_1(self):
    arr = _make_array(4, 2, 3)
    result = _transpose_layers_axes(arr, src_axis=0, dst_axis=1)
    self.assertEqual(result.shape, (2, 4, 3))

  def test_transposes_axis_1_to_0(self):
    arr = _make_array(2, 4, 3)
    result = _transpose_layers_axes(arr, src_axis=1, dst_axis=0)
    self.assertEqual(result.shape, (4, 2, 3))

  def test_transposes_nested_dict(self):
    arr = _make_array(4, 2, 3)
    tree = {"decoder": {"layers": {"kernel": arr}}}
    result = _transpose_layers_axes(tree, src_axis=0, dst_axis=1)
    self.assertEqual(result["decoder"]["layers"]["kernel"].shape, (2, 4, 3))

  def test_passes_through_1d_array(self):
    arr = _make_array(5)
    result = _transpose_layers_axes(arr, src_axis=0, dst_axis=1)
    # 1D array has no axis 1, should be returned unchanged
    np.testing.assert_array_equal(result, arr)


class TestStackLayers(unittest.TestCase):

  def test_stacks_individual_layers(self):
    arr0 = _make_array(3, 4)
    arr1 = _make_array(3, 4)
    decoder = {
        "layers_0": {"mlp": {"kernel": arr0}},
        "layers_1": {"mlp": {"kernel": arr1}},
    }
    result, was_stacked = _stack_layers(decoder)
    self.assertTrue(was_stacked)
    self.assertIn("layers", result)
    stacked = result["layers"]["mlp"]["kernel"]
    self.assertEqual(stacked.shape, (2, 3, 4))
    np.testing.assert_array_equal(stacked[0], arr0)
    np.testing.assert_array_equal(stacked[1], arr1)

  def test_noop_when_no_layer_pattern(self):
    arr = _make_array(3, 4)
    decoder = {"layers": {"mlp": {"kernel": arr}}}
    result, was_stacked = _stack_layers(decoder)
    self.assertFalse(was_stacked)
    self.assertIs(result, decoder)

  def test_preserves_non_layer_keys(self):
    norm_weight = _make_array(4)
    arr0 = _make_array(3, 4)
    decoder = {
        "layers_0": {"mlp": {"kernel": arr0}},
        "final_norm": {"scale": norm_weight},
    }
    result, was_stacked = _stack_layers(decoder)
    self.assertTrue(was_stacked)
    self.assertIn("final_norm", result)
    np.testing.assert_array_equal(result["final_norm"]["scale"], norm_weight)

  def test_stacks_three_layers(self):
    arrays = [_make_array(2, 2) for _ in range(3)]
    decoder = {f"layers_{i}": {"w": arrays[i]} for i in range(3)}
    result, was_stacked = _stack_layers(decoder)
    self.assertTrue(was_stacked)
    stacked = result["layers"]["w"]
    self.assertEqual(stacked.shape, (3, 2, 2))


class TestConvertLinenToNNX(unittest.TestCase):

  def _make_linen_state(self, add_opt_state=False):
    """Creates a minimal Linen checkpoint structure."""
    arr = _make_array(2, 4, 3)  # (embed, layers, dim) at scan_axis=1
    state = {
        "step": 10,
        "params": {
            "params": {
                "decoder": {
                    "layers": {"mlp": {"wi": {"kernel": arr}}},
                    "decoder_norm": {"scale": _make_array(4)},
                }
            }
        },
    }
    if add_opt_state:
      state["opt_state"] = {"params": {"mu": {"decoder": {"layers": {"kernel": arr}}}}}
    return state

  def test_converts_step(self):
    state = self._make_linen_state()
    result = convert_linen_to_nnx(state)
    self.assertEqual(result["step"], 10)

  def test_removes_double_nesting(self):
    state = self._make_linen_state()
    result = convert_linen_to_nnx(state)
    # After conversion, params should have 'decoder' at top level, not 'params.decoder'
    self.assertIn("decoder", result["params"])
    self.assertNotIn("params", result["params"])

  def test_adds_value_wrappers(self):
    state = self._make_linen_state()
    result = convert_linen_to_nnx(state)
    # Arrays should be wrapped in {"value": array}
    kernel = result["params"]["decoder"]["layers"]["mlp"]["wi"]["kernel"]
    self.assertIsInstance(kernel, dict)
    self.assertIn("value", kernel)

  def test_converts_opt_state(self):
    state = self._make_linen_state(add_opt_state=True)
    result = convert_linen_to_nnx(state)
    self.assertIn("opt_state", result)
    # Linen opt_state had nested 'params' level; it should be removed
    self.assertNotIn("params", result["opt_state"])


class TestConvertNNXToLinen(unittest.TestCase):

  def _make_nnx_state(self, add_opt_state=False):
    """Creates a minimal NNX checkpoint structure."""
    arr = _make_array(2, 4, 3)
    state = {
        "step": 5,
        "params": {
            "decoder": {
                "layers": {"mlp": {"wi": {"kernel": {"value": arr}}}},
                "decoder_norm": {"scale": {"value": _make_array(4)}},
            }
        },
    }
    if add_opt_state:
      state["opt_state"] = {
          "mu": {"decoder": {"layers": {"kernel": {"value": arr}}}},
          "nu": {"decoder": {"layers": {"kernel": {"value": arr}}}},
      }
    return state

  def test_converts_step(self):
    state = self._make_nnx_state()
    result = convert_nnx_to_linen(state)
    self.assertEqual(result["step"], 5)

  def test_adds_double_nesting(self):
    state = self._make_nnx_state()
    result = convert_nnx_to_linen(state)
    # params should be double-nested: result["params"]["params"]["decoder"]
    self.assertIn("params", result["params"])
    self.assertIn("decoder", result["params"]["params"])

  def test_strips_value_wrappers(self):
    state = self._make_nnx_state()
    result = convert_nnx_to_linen(state)
    kernel = result["params"]["params"]["decoder"]["layers"]["mlp"]["wi"]["kernel"]
    self.assertIsInstance(kernel, np.ndarray)

  def test_converts_opt_state(self):
    state = self._make_nnx_state(add_opt_state=True)
    result = convert_nnx_to_linen(state)
    self.assertIn("opt_state", result)
    # mu/nu should get a 'params' level added
    self.assertIn("params", result["opt_state"]["mu"])
    self.assertIn("params", result["opt_state"]["nu"])


class TestRoundTrip(unittest.TestCase):
  """Verifies that linen->nnx->linen round-trip preserves data."""

  def test_linen_to_nnx_to_linen(self):
    arr = _make_array(2, 4, 3)
    linen_state = {
        "step": 42,
        "params": {
            "params": {
                "decoder": {
                    "layers": {"mlp": {"wi": {"kernel": arr}}},
                    "norm": {"scale": _make_array(4)},
                }
            }
        },
    }
    nnx_state = convert_linen_to_nnx(linen_state)
    recovered_state = convert_nnx_to_linen(nnx_state)

    self.assertEqual(recovered_state["step"], 42)
    recovered_kernel = recovered_state["params"]["params"]["decoder"]["layers"]["mlp"]["wi"]["kernel"]
    np.testing.assert_array_equal(recovered_kernel, arr)

  def test_nnx_to_linen_to_nnx(self):
    arr = _make_array(2, 4, 3)
    nnx_state = {
        "step": 7,
        "params": {
            "decoder": {
                "layers": {"mlp": {"wi": {"kernel": {"value": arr}}}},
            }
        },
    }
    linen_state = convert_nnx_to_linen(nnx_state)
    recovered_state = convert_linen_to_nnx(linen_state)

    self.assertEqual(recovered_state["step"], 7)
    recovered_kernel = recovered_state["params"]["decoder"]["layers"]["mlp"]["wi"]["kernel"]
    self.assertIn("value", recovered_kernel)
    np.testing.assert_array_equal(recovered_kernel["value"], arr)


class TestConvertOptState(unittest.TestCase):

  def test_linen_to_nnx_removes_params_level_and_wraps(self):
    arr = _make_array(3, 4)
    opt_state = {"mu": {"params": {"decoder": {"kernel": arr}}}}
    result = _convert_opt_state_linen_to_nnx(opt_state)
    # 'params' key removed; decoder promoted
    self.assertNotIn("params", result["mu"])
    self.assertIn("decoder", result["mu"])
    # Arrays wrapped
    self.assertEqual(set(result["mu"]["decoder"]["kernel"].keys()), {"value"})

  def test_linen_to_nnx_handles_list_input(self):
    arr = _make_array(2, 2)
    opt_state = [{"decoder": {"kernel": arr}}, {"decoder": {"kernel": arr}}]
    result = _convert_opt_state_linen_to_nnx(opt_state)
    self.assertIsInstance(result, list)
    # Arrays inside lists should be wrapped
    self.assertIn("value", result[0]["decoder"]["kernel"])

  def test_linen_to_nnx_handles_non_array_non_dict(self):
    # Scalars should be passed through unchanged
    result = _convert_opt_state_linen_to_nnx(42)
    self.assertEqual(result, 42)

  def test_linen_to_nnx_params_key_with_non_dict_value(self):
    # When k == "params" but converted value is not a dict, store it as-is
    opt_state = {"params": 99}
    result = _convert_opt_state_linen_to_nnx(opt_state)
    self.assertIn("params", result)
    self.assertEqual(result["params"], 99)

  def test_nnx_to_linen_adds_params_level_and_strips(self):
    arr = _make_array(3, 4)
    opt_state = {
        "mu": {"decoder": {"kernel": {"value": arr}}},
        "nu": {"decoder": {"kernel": {"value": arr}}},
    }
    result = _convert_opt_state_nnx_to_linen(opt_state)
    # mu/nu should have 'params' nested inside
    self.assertIn("params", result["mu"])
    self.assertIn("params", result["nu"])
    # Arrays unwrapped
    kernel = result["mu"]["params"]["decoder"]["kernel"]
    np.testing.assert_array_equal(kernel, arr)

  def test_nnx_to_linen_handles_list_input(self):
    arr = _make_array(2, 2)
    opt_state = [{"decoder": {"kernel": {"value": arr}}}]
    result = _convert_opt_state_nnx_to_linen(opt_state)
    self.assertIsInstance(result, list)
    np.testing.assert_array_equal(result[0]["decoder"]["kernel"], arr)

  def test_nnx_to_linen_passes_through_scalars(self):
    result = _convert_opt_state_nnx_to_linen("scalar_string")
    self.assertEqual(result, "scalar_string")

  def test_nnx_to_linen_value_wrapper_with_non_array_inner(self):
    # {"value": scalar} should NOT be unwrapped (only arrays get unwrapped)
    d = {"value": 42}
    result = _convert_opt_state_nnx_to_linen(d)
    # Since inner is not an array, it falls through to the regular dict processing
    # The "value" key gets recursively processed but 42 is a scalar -> returned as-is
    self.assertIn("value", result)
    self.assertEqual(result["value"], 42)


class TestAdditionalEdgeCases(unittest.TestCase):
  """Covers remaining uncovered branches."""

  def test_detect_format_params_has_params_but_no_decoder_encoder(self):
    # params["params"] exists but inner has no decoder/encoder -> falls through to NNX check
    state = {"params": {"params": {"some_other_key": {}}}}
    # Neither linen (no decoder/encoder in inner) nor nnx (no decoder/encoder at top)
    # and no opt_state -> should raise
    with self.assertRaises(ValueError):
      detect_format(state)

  def test_detect_format_opt_state_no_valid_pattern_raises(self):
    # opt_state present but neither linen nor nnx patterns match
    arr = _make_array(2)
    state = {
        "params": {"something": arr},
        "opt_state": {"mu": {"decoder": {"kernel": arr}}},  # no value wrappers, no params key
    }
    with self.assertRaises(ValueError):
      detect_format(state)

  def test_add_value_wrappers_value_key_with_non_array(self):
    # {"value": "text"} is not a wrapper (inner is not an array), should recurse and wrap nothing
    d = {"value": "not_an_array"}
    result = _add_value_wrappers(d)
    # Should recurse: "not_an_array" is a string -> passes through -> result = {"value": "not_an_array"}
    self.assertEqual(result, {"value": "not_an_array"})

  def test_transpose_layers_axes_handles_list(self):
    arr = _make_array(4, 2, 3)
    result = _transpose_layers_axes([arr], src_axis=0, dst_axis=1)
    self.assertIsInstance(result, list)
    self.assertEqual(result[0].shape, (2, 4, 3))

  def test_transpose_layers_axes_handles_tuple(self):
    arr = _make_array(4, 2, 3)
    result = _transpose_layers_axes((arr,), src_axis=0, dst_axis=1)
    self.assertIsInstance(result, tuple)
    self.assertEqual(result[0].shape, (2, 4, 3))

  def test_stack_layers_with_missing_key_in_some_layers(self):
    # Layer 0 has "bias", layer 1 does not -> "bias" key should be skipped
    arr = _make_array(3, 4)
    decoder = {
        "layers_0": {"mlp": {"kernel": arr, "bias": arr}},
        "layers_1": {"mlp": {"kernel": arr}},  # no "bias"
    }
    result, was_stacked = _stack_layers(decoder)
    self.assertTrue(was_stacked)
    # "kernel" should be stacked; "bias" might be skipped due to missing in layer 1
    self.assertIn("kernel", result["layers"]["mlp"])

  def test_convert_linen_to_nnx_no_step(self):
    arr = _make_array(2, 4, 3)
    state = {"params": {"params": {"decoder": {"layers": {"kernel": arr}}}}}
    result = convert_linen_to_nnx(state)
    self.assertNotIn("step", result)
    self.assertIn("params", result)

  def test_convert_linen_to_nnx_with_per_layer_params(self):
    # Linen checkpoint with layers_0, layers_1 (unscanned) -> should be stacked + transposed
    arr = _make_array(3, 4)
    state = {
        "params": {
            "params": {
                "decoder": {
                    "layers_0": {"mlp": {"kernel": arr}},
                    "layers_1": {"mlp": {"kernel": arr}},
                }
            }
        }
    }
    result = convert_linen_to_nnx(state)
    # After conversion: stacked layers should be at axis=1 (param_scan_axis)
    stacked = result["params"]["decoder"]["layers"]["mlp"]["kernel"]["value"]
    # Original shape (3, 4) stacked to (2, 3, 4), then transposed to (3, 2, 4)
    self.assertEqual(stacked.shape, (3, 2, 4))

  def test_convert_linen_to_nnx_no_double_nesting(self):
    # Linen state without double-nesting (unusual but handled)
    arr = _make_array(2, 4)
    state = {"params": {"decoder": {"layers": {"kernel": arr}}}}
    result = convert_linen_to_nnx(state)
    self.assertIn("decoder", result["params"])

  def test_convert_nnx_to_linen_no_step(self):
    arr = _make_array(2, 4)
    state = {"params": {"decoder": {"layers": {"kernel": {"value": arr}}}}}
    result = convert_nnx_to_linen(state)
    self.assertNotIn("step", result)
    self.assertIn("params", result)

  def test_convert_nnx_to_linen_already_has_params_nesting(self):
    # NNX state where stripped params already has a "params" key (unusual edge case)
    arr = _make_array(2, 4)
    state = {"params": {"params": {"decoder": {"layers": {"kernel": {"value": arr}}}}}}
    result = convert_nnx_to_linen(state)
    # Since "params" already exists in stripped, it's copied as-is
    self.assertIn("params", result)

  def test_convert_linen_to_nnx_no_params_key(self):
    # State without 'params' — only step is copied
    state = {"step": 3}
    result = convert_linen_to_nnx(state)
    self.assertEqual(result["step"], 3)
    self.assertNotIn("params", result)

  def test_convert_nnx_to_linen_no_params_key(self):
    # State without 'params' — only step is copied
    state = {"step": 8}
    result = convert_nnx_to_linen(state)
    self.assertEqual(result["step"], 8)
    self.assertNotIn("params", result)

  def test_stack_layers_non_array_non_dict_leaf(self):
    # Layer values that are scalars (neither array nor dict) — inner else branch
    decoder = {
        "layers_0": {"count": 1},
        "layers_1": {"count": 2},
    }
    result, was_stacked = _stack_layers(decoder)
    self.assertTrue(was_stacked)
    # The scalar value is not stackable; stack_arrays returns first element
    self.assertIn("layers", result)


if __name__ == "__main__":
  unittest.main()
