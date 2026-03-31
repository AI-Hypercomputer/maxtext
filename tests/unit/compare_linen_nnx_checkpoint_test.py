# Copyright 2023-2026 Google LLC
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

"""Tests for compare_linen_nnx_checkpoint utilities."""

import io
import unittest
from unittest.mock import patch
import numpy as np

from absl import flags as absl_flags
from maxtext.checkpoint_conversion.compare_linen_nnx_checkpoint import (
    is_rng_path,
    filter_rngs,
    detect_format,
    _has_value_wrappers,
    _strip_value_wrappers,
    _normalize_linen_params,
    _normalize_nnx_params,
    _extract_params,
    _normalize_params,
    get_tree_structure_info,
    print_structure_diff,
    compare_params,
    transform_nnx_params_for_comparison,
)


def _arr(*shape):
  """Helper: float32 array of given shape, values 0..prod(shape)-1."""
  return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)


def setUpModule():
  # Mark FLAGS as parsed so FLAGS.verbose etc. are accessible without a full
  # app.run(). Required flags (ckpt_path_1/2) are not needed in unit tests.
  absl_flags.FLAGS.mark_as_parsed()


# ---------------------------------------------------------------------------
# is_rng_path
# ---------------------------------------------------------------------------


class TestIsRngPath(unittest.TestCase):
  """Tests for is_rng_path."""

  def test_returns_true_for_rngs(self):
    self.assertTrue(is_rng_path("model/decoder/rngs/dropout"))

  def test_returns_true_for_rng(self):
    self.assertTrue(is_rng_path("model/rngs/params/key"))

  def test_returns_true_case_insensitive(self):
    self.assertTrue(is_rng_path("model/RNGs/state"))
    self.assertTrue(is_rng_path("model/RNG/state"))

  def test_returns_false_for_normal_path(self):
    self.assertFalse(is_rng_path("model/decoder/layers/kernel"))

  def test_returns_false_for_empty_string(self):
    self.assertFalse(is_rng_path(""))


# ---------------------------------------------------------------------------
# filter_rngs
# ---------------------------------------------------------------------------


class TestFilterRngs(unittest.TestCase):
  """Tests for filter_rngs."""

  def test_removes_top_level_rngs_key(self):
    tree = {"model": {"kernel": _arr(4)}, "rngs": {"dropout": _arr(2)}}
    result = filter_rngs(tree)
    self.assertNotIn("rngs", result)
    self.assertIn("model", result)

  def test_removes_nested_rngs_key(self):
    tree = {"model": {"kernel": _arr(4), "rngs": {"key": _arr(2)}}}
    result = filter_rngs(tree)
    self.assertNotIn("rngs", result["model"])
    self.assertIn("kernel", result["model"])

  def test_keeps_empty_parent_when_only_child_is_rng(self):
    # After filtering, the parent dict becomes empty and is dropped.
    tree = {"model": {"rngs": {"key": _arr(2)}}}
    result = filter_rngs(tree)
    self.assertNotIn("model", result)

  def test_passthrough_for_non_rng_tree(self):
    tree = {"params": {"kernel": _arr(4), "bias": _arr(2)}}
    result = filter_rngs(tree)
    self.assertEqual(set(result.keys()), {"params"})

  def test_passthrough_for_non_dict_input(self):
    arr = _arr(4)
    self.assertIs(filter_rngs(arr), arr)


# ---------------------------------------------------------------------------
# _has_value_wrappers
# ---------------------------------------------------------------------------


class TestHasValueWrappers(unittest.TestCase):
  """Tests for _has_value_wrappers."""

  def test_returns_true_for_direct_value_wrapper(self):
    tree = {"value": _arr(3, 4)}
    self.assertTrue(_has_value_wrappers(tree))

  def test_returns_true_for_nested_wrapper(self):
    tree = {"decoder": {"kernel": {"value": _arr(2, 2)}}}
    self.assertTrue(_has_value_wrappers(tree))

  def test_returns_false_for_plain_array(self):
    self.assertFalse(_has_value_wrappers(_arr(3)))

  def test_returns_false_for_multi_key_dict(self):
    tree = {"value": _arr(2), "extra": _arr(2)}
    self.assertFalse(_has_value_wrappers(tree))

  def test_returns_false_for_value_key_with_non_array(self):
    tree = {"value": 42}
    self.assertFalse(_has_value_wrappers(tree))


# ---------------------------------------------------------------------------
# _strip_value_wrappers
# ---------------------------------------------------------------------------


class TestStripValueWrappers(unittest.TestCase):
  """Tests for _strip_value_wrappers."""

  def test_strips_direct_wrapper(self):
    arr = _arr(3, 4)
    result = _strip_value_wrappers({"value": arr})
    np.testing.assert_array_equal(result, arr)

  def test_strips_nested_wrappers(self):
    arr = _arr(2, 2)
    tree = {"decoder": {"kernel": {"value": arr}}}
    result = _strip_value_wrappers(tree)
    np.testing.assert_array_equal(result["decoder"]["kernel"], arr)

  def test_passthrough_plain_array(self):
    arr = _arr(4)
    self.assertIs(_strip_value_wrappers(arr), arr)

  def test_handles_list(self):
    arr = _arr(2)
    result = _strip_value_wrappers([{"value": arr}])
    np.testing.assert_array_equal(result[0], arr)

  def test_handles_tuple(self):
    arr = _arr(2)
    result = _strip_value_wrappers(({"value": arr},))
    np.testing.assert_array_equal(result[0], arr)

  def test_passthrough_non_array_scalar(self):
    self.assertEqual(_strip_value_wrappers(42), 42)


# ---------------------------------------------------------------------------
# _normalize_linen_params
# ---------------------------------------------------------------------------


class TestNormalizeLinenParams(unittest.TestCase):
  """Tests for _normalize_linen_params."""

  def test_removes_double_nesting(self):
    inner = {"decoder": {"layers": {}}}
    params = {"params": inner}
    result = _normalize_linen_params(params)
    self.assertIs(result, inner)

  def test_removes_double_nesting_encoder(self):
    inner = {"encoder": {"layers": {}}}
    params = {"params": inner}
    result = _normalize_linen_params(params)
    self.assertIs(result, inner)

  def test_passthrough_when_no_double_nesting(self):
    params = {"decoder": {"layers": {}}}
    result = _normalize_linen_params(params)
    self.assertIs(result, params)

  def test_passthrough_when_inner_has_no_decoder_encoder(self):
    params = {"params": {"other_key": {}}}
    result = _normalize_linen_params(params)
    self.assertIs(result, params)


# ---------------------------------------------------------------------------
# _normalize_nnx_params
# ---------------------------------------------------------------------------


class TestNormalizeNnxParams(unittest.TestCase):
  """Tests for _normalize_nnx_params."""

  def test_strips_value_wrappers(self):
    arr = _arr(2, 3)
    params = {"decoder": {"kernel": {"value": arr}}}
    result = _normalize_nnx_params(params)
    np.testing.assert_array_equal(result["decoder"]["kernel"], arr)

  def test_passthrough_plain_tree(self):
    arr = _arr(4)
    params = {"decoder": {"kernel": arr}}
    result = _normalize_nnx_params(params)
    np.testing.assert_array_equal(result["decoder"]["kernel"], arr)


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat(unittest.TestCase):
  """Tests for detect_format."""

  def test_detects_nnx_via_model_key(self):
    state = {"model": {"decoder": {}}, "optimizer": {}}
    self.assertEqual(detect_format(state), "nnx")

  def test_detects_linen_via_double_nested_decoder(self):
    state = {"params": {"params": {"decoder": {}}}}
    self.assertEqual(detect_format(state), "linen")

  def test_detects_linen_via_double_nested_encoder(self):
    state = {"params": {"params": {"encoder": {}}}}
    self.assertEqual(detect_format(state), "linen")

  def test_detects_nnx_via_value_wrappers(self):
    arr = _arr(2, 2)
    state = {"params": {"decoder": {"kernel": {"value": arr}}}}
    self.assertEqual(detect_format(state), "nnx")

  def test_raises_when_no_params_or_model_key(self):
    with self.assertRaises(ValueError):
      detect_format({"step": 0})

  def test_raises_on_undetectable_format(self):
    with self.assertRaises(ValueError):
      detect_format({"params": {"unknown_key": {}}})


# ---------------------------------------------------------------------------
# _extract_params
# ---------------------------------------------------------------------------


class TestExtractParams(unittest.TestCase):
  """Tests for _extract_params."""

  def test_extracts_linen_params(self):
    params = {"params": {"decoder": {}}}
    state = {"params": params, "opt_state": {}}
    self.assertIs(_extract_params(state, "linen"), params)

  def test_extracts_nnx_params_from_model_key(self):
    model = {"decoder": {}}
    state = {"model": model, "optimizer": {}}
    self.assertIs(_extract_params(state, "nnx"), model)

  def test_extracts_nnx_params_falls_back_to_params_key(self):
    params = {"decoder": {}}
    state = {"params": params}
    self.assertIs(_extract_params(state, "nnx"), params)

  def test_returns_empty_dict_when_key_missing(self):
    state = {"optimizer": {}}
    result = _extract_params(state, "linen")
    self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# _normalize_params
# ---------------------------------------------------------------------------


class TestNormalizeParams(unittest.TestCase):
  """Tests for _normalize_params."""

  def test_dispatches_to_linen(self):
    inner = {"decoder": {}}
    params = {"params": inner}
    result = _normalize_params(params, "linen")
    self.assertIs(result, inner)

  def test_dispatches_to_nnx(self):
    arr = _arr(2, 2)
    params = {"decoder": {"kernel": {"value": arr}}}
    result = _normalize_params(params, "nnx")
    np.testing.assert_array_equal(result["decoder"]["kernel"], arr)


# ---------------------------------------------------------------------------
# get_tree_structure_info
# ---------------------------------------------------------------------------


class TestGetTreeStructureInfo(unittest.TestCase):
  """Tests for get_tree_structure_info."""

  def test_returns_shape_and_dtype(self):
    tree = {"kernel": _arr(3, 4), "bias": _arr(4)}
    info = get_tree_structure_info(tree)
    self.assertEqual(info["['kernel']"], ((3, 4), "float32"))
    self.assertEqual(info["['bias']"], ((4,), "float32"))

  def test_handles_nested_tree(self):
    tree = {"decoder": {"kernel": _arr(2, 2)}}
    info = get_tree_structure_info(tree)
    self.assertEqual(len(info), 1)
    shapes = [v[0] for v in info.values()]
    self.assertIn((2, 2), shapes)

  def test_handles_non_array_leaves(self):
    tree = {"step": 5}
    info = get_tree_structure_info(tree)
    self.assertEqual(len(info), 1)
    shape, _ = list(info.values())[0]
    self.assertEqual(shape, "N/A")


# ---------------------------------------------------------------------------
# print_structure_diff
# ---------------------------------------------------------------------------


class TestPrintStructureDiff(unittest.TestCase):
  """Tests for print_structure_diff."""

  def _make_params(self, keys_and_shapes):
    return {k: _arr(*s) for k, s in keys_and_shapes.items()}

  def test_returns_empty_tuples_when_identical(self):
    params = self._make_params({"kernel": (4, 4), "bias": (4,)})
    with patch("sys.stdout", new_callable=io.StringIO):
      only1, only2, shape_mm, dtype_mm = print_structure_diff(params, params)
    self.assertEqual(only1, [])
    self.assertEqual(only2, [])
    self.assertEqual(shape_mm, [])
    self.assertEqual(dtype_mm, [])

  def test_detects_key_only_in_first(self):
    p1 = self._make_params({"kernel": (4, 4), "bias": (4,)})
    p2 = self._make_params({"kernel": (4, 4)})
    with patch("sys.stdout", new_callable=io.StringIO):
      only1, only2, _, _ = print_structure_diff(p1, p2)
    self.assertEqual(len(only1), 1)
    self.assertEqual(only2, [])

  def test_detects_key_only_in_second(self):
    p1 = self._make_params({"kernel": (4, 4)})
    p2 = self._make_params({"kernel": (4, 4), "bias": (4,)})
    with patch("sys.stdout", new_callable=io.StringIO):
      only1, only2, _, _ = print_structure_diff(p1, p2)
    self.assertEqual(only1, [])
    self.assertEqual(len(only2), 1)

  def test_detects_shape_mismatch(self):
    p1 = {"kernel": _arr(4, 4)}
    p2 = {"kernel": _arr(4, 8)}
    with patch("sys.stdout", new_callable=io.StringIO):
      _, _, shape_mm, _ = print_structure_diff(p1, p2)
    self.assertEqual(len(shape_mm), 1)

  def test_detects_dtype_mismatch(self):
    p1 = {"kernel": np.zeros((4,), dtype=np.float32)}
    p2 = {"kernel": np.zeros((4,), dtype=np.float16)}
    with patch("sys.stdout", new_callable=io.StringIO):
      _, _, _, dtype_mm = print_structure_diff(p1, p2)
    self.assertEqual(len(dtype_mm), 1)


# ---------------------------------------------------------------------------
# compare_params
# ---------------------------------------------------------------------------


class TestCompareParams(unittest.TestCase):
  """Tests for compare_params."""

  def test_returns_true_for_identical_params(self):
    params = {"kernel": _arr(4, 4), "bias": _arr(4)}
    with patch("builtins.print"):
      result = compare_params(params, params)
    self.assertTrue(result)

  def test_returns_false_for_different_structures(self):
    p1 = {"kernel": _arr(4, 4)}
    p2 = {"kernel": _arr(4, 4), "bias": _arr(4)}
    with patch("builtins.print"):
      result = compare_params(p1, p2)
    self.assertFalse(result)

  def test_returns_false_for_shape_mismatch(self):
    p1 = {"kernel": _arr(4, 4)}
    p2 = {"kernel": _arr(4, 8)}
    with patch("builtins.print"):
      result = compare_params(p1, p2)
    self.assertFalse(result)

  def test_returns_false_for_dtype_mismatch(self):
    p1 = {"kernel": np.zeros((4,), dtype=np.float32)}
    p2 = {"kernel": np.zeros((4,), dtype=np.float16)}
    with patch("builtins.print"):
      result = compare_params(p1, p2)
    self.assertFalse(result)

  def test_value_comparison_passes_when_equal(self):
    arr = _arr(4)
    with patch("builtins.print"):
      result = compare_params({"w": arr}, {"w": arr.copy()}, compare_values=True)
    self.assertTrue(result)

  def test_value_comparison_fails_when_different(self):
    p1 = {"w": np.array([1.0, 2.0], dtype=np.float32)}
    p2 = {"w": np.array([1.0, 9.0], dtype=np.float32)}
    with patch("builtins.print"):
      result = compare_params(p1, p2, compare_values=True, atol=1e-5, rtol=1e-5)
    self.assertFalse(result)

  def test_value_comparison_passes_within_tolerance(self):
    p1 = {"w": np.array([1.0], dtype=np.float32)}
    p2 = {"w": np.array([1.0 + 1e-7], dtype=np.float32)}
    with patch("builtins.print"):
      result = compare_params(p1, p2, compare_values=True, atol=1e-5, rtol=1e-5)
    self.assertTrue(result)

  def test_verbose_mode_does_not_raise(self):
    params = {"kernel": _arr(2, 2)}
    with patch("builtins.print"):
      result = compare_params(params, params, verbose=True, compare_values=True)
    self.assertTrue(result)

  def test_nested_params(self):
    params = {"decoder": {"kernel": _arr(4, 4), "bias": _arr(4)}}
    with patch("builtins.print"):
      result = compare_params(params, params)
    self.assertTrue(result)


# ---------------------------------------------------------------------------
# transform_nnx_params_for_comparison
# ---------------------------------------------------------------------------


class TestTransformNnxParamsForComparison(unittest.TestCase):
  """Tests for transform_nnx_params_for_comparison."""

  def test_transposes_layer_array(self):
    # Shape (num_layers=3, d=4) -> (d=4, num_layers=3)
    arr = _arr(3, 4)
    tree = {"layers": {"kernel": arr}}
    with patch("builtins.print"):
      result = transform_nnx_params_for_comparison(tree)
    self.assertEqual(result["layers"]["kernel"].shape, (4, 3))

  def test_does_not_transpose_non_layer_array(self):
    arr = _arr(3, 4)
    tree = {"embedding": arr}
    with patch("builtins.print"):
      result = transform_nnx_params_for_comparison(tree)
    self.assertEqual(result["embedding"].shape, (3, 4))

  def test_does_not_transpose_1d_layer_array(self):
    arr = _arr(4)
    tree = {"layers": {"bias": arr}}
    with patch("builtins.print"):
      result = transform_nnx_params_for_comparison(tree)
    self.assertEqual(result["layers"]["bias"].shape, (4,))

  def test_transposes_higher_rank_layer_array(self):
    # Shape (num_layers=2, d1=3, d2=5) -> (d1=3, num_layers=2, d2=5)
    arr = _arr(2, 3, 5)
    tree = {"layers": {"kernel": arr}}
    with patch("builtins.print"):
      result = transform_nnx_params_for_comparison(tree)
    self.assertEqual(result["layers"]["kernel"].shape, (3, 2, 5))


if __name__ == "__main__":
  unittest.main()
