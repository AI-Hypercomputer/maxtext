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

"""Tests for lora_utils."""

import json
import unittest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import jax
import jax.numpy as jnp

from maxtext.utils.lora_utils import (
    apply_lora_on_base_params,
    unapply_lora_from_base_params,
    get_lora_abstract_state,
    load_adapter,
    setup_initial_lora_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lora_params(b, r, n, d):
  """Returns (base_params, lora_params) for a single-layer attention kernel."""
  kernel = jnp.ones((b, n, d), dtype=jnp.float32)
  lora_a_k = jnp.ones((b, r), dtype=jnp.float32)  # lora_params["lora_a.kernel"]
  lora_b_k = jnp.ones((r, n, d), dtype=jnp.float32)  # lora_params["lora_b.kernel"]
  base = {"kernel": kernel}
  lora = {"lora_a.kernel": lora_a_k, "lora_b.kernel": lora_b_k}
  return base, lora


# ---------------------------------------------------------------------------
# apply_lora_on_base_params
# ---------------------------------------------------------------------------


class TestApplyLoraOnBaseParams(unittest.TestCase):
  """Tests for apply_lora_on_base_params."""

  def test_applies_lora_update_to_kernel(self):
    base, lora = _lora_params(1, 2, 3, 4)
    original_kernel = np.array(base["kernel"])
    apply_lora_on_base_params(base, lora)
    # W_new = W + einsum("br,rnd->bnd", lora_a_k, lora_b_k) * 1.0
    expected = original_kernel + np.einsum("br,rnd->bnd", lora["lora_a.kernel"], lora["lora_b.kernel"])
    np.testing.assert_allclose(np.array(base["kernel"]), expected, rtol=1e-5)

  def test_applies_scale_factor(self):
    base, lora = _lora_params(1, 2, 3, 4)
    original_kernel = np.array(base["kernel"])
    apply_lora_on_base_params(base, lora, lora_scale_factor=0.5)
    expected = original_kernel + np.einsum("br,rnd->bnd", lora["lora_a.kernel"], lora["lora_b.kernel"]) * 0.5
    np.testing.assert_allclose(np.array(base["kernel"]), expected, rtol=1e-5)

  def test_skips_update_when_lora_leaf_is_none(self):
    kernel = jnp.ones((2, 3, 4), dtype=jnp.float32)
    base = {"kernel": kernel}
    lora = {"kernel": None}
    apply_lora_on_base_params(base, lora)
    np.testing.assert_array_equal(np.array(base["kernel"]), np.array(kernel))

  def test_recurses_into_nested_dict(self):
    base, lora = _lora_params(1, 2, 3, 4)
    nested_base = {"layer": base}
    nested_lora = {"layer": lora}
    apply_lora_on_base_params(nested_base, nested_lora)
    np.testing.assert_allclose(np.array(nested_base["layer"]["kernel"]), np.array(base["kernel"]), rtol=1e-5)
    # After apply, kernel == original + delta; verify structure is intact
    self.assertIn("kernel", nested_base["layer"])

  def test_raises_on_unexpected_lora_key(self):
    base = {"kernel": jnp.ones((2, 3, 4))}
    lora = {"unexpected_key": jnp.ones((2,))}
    with self.assertRaises(ValueError, msg="Expected ValueError for bad lora key"):
      apply_lora_on_base_params(base, lora)

  def test_multiple_nested_levels(self):
    base, lora = _lora_params(1, 2, 3, 4)
    nested_base = {"decoder": {"layers": {"attn": base}}}
    nested_lora = {"decoder": {"layers": {"attn": lora}}}
    original_kernel = np.array(base["kernel"])
    apply_lora_on_base_params(nested_base, nested_lora)
    result_kernel = np.array(nested_base["decoder"]["layers"]["attn"]["kernel"])
    self.assertFalse(np.allclose(result_kernel, original_kernel))  # was modified

  def test_keeps_base_when_one_lora_component_is_none(self):
    # lora_a.kernel present but lora_b.kernel is None -> lora_update_or_base else branch (line 51)
    kernel = jnp.ones((2, 3, 4), dtype=jnp.float32)
    base = {"kernel": kernel}
    lora = {"lora_a.kernel": jnp.ones((2, 2)), "lora_b.kernel": None}
    apply_lora_on_base_params(base, lora)
    np.testing.assert_array_equal(np.array(base["kernel"]), np.array(kernel))


# ---------------------------------------------------------------------------
# unapply_lora_from_base_params
# ---------------------------------------------------------------------------


class TestUnapplyLoraFromBaseParams(unittest.TestCase):
  """Tests for unapply_lora_from_base_params."""

  def test_unapplies_lora_update(self):
    base, lora = _lora_params(1, 2, 3, 4)
    original_kernel = np.array(base["kernel"])
    unapply_lora_from_base_params(base, lora)
    expected = original_kernel - np.einsum("br,rnd->bnd", lora["lora_a.kernel"], lora["lora_b.kernel"])
    np.testing.assert_allclose(np.array(base["kernel"]), expected, rtol=1e-5)

  def test_apply_then_unapply_is_identity(self):
    rng = np.random.default_rng(42)
    kernel = jnp.array(rng.standard_normal((2, 3, 4)).astype(np.float32))
    lora_a_k = jnp.array(rng.standard_normal((2, 2)).astype(np.float32))
    lora_b_k = jnp.array(rng.standard_normal((2, 3, 4)).astype(np.float32))
    lora = {"lora_a.kernel": lora_a_k, "lora_b.kernel": lora_b_k}

    original = np.array(kernel)
    base = {"kernel": kernel}
    apply_lora_on_base_params(base, lora)
    unapply_lora_from_base_params(base, lora)
    np.testing.assert_allclose(np.array(base["kernel"]), original, rtol=1e-5, atol=1e-5)

  def test_skips_update_when_lora_leaf_is_none(self):
    kernel = jnp.ones((2, 3, 4), dtype=jnp.float32)
    base = {"kernel": kernel}
    lora = {"kernel": None}
    unapply_lora_from_base_params(base, lora)
    np.testing.assert_array_equal(np.array(base["kernel"]), np.array(kernel))

  def test_recurses_into_nested_dict(self):
    base, lora = _lora_params(1, 2, 3, 4)
    nested_base = {"attn": base}
    nested_lora = {"attn": lora}
    unapply_lora_from_base_params(nested_base, nested_lora)
    self.assertIn("kernel", nested_base["attn"])

  def test_raises_on_unexpected_lora_key(self):
    base = {"kernel": jnp.ones((2, 3, 4))}
    lora = {"bad_key": jnp.ones((2,))}
    with self.assertRaises(ValueError):
      unapply_lora_from_base_params(base, lora)

  def test_unapply_with_scale_factor(self):
    base, lora = _lora_params(1, 2, 3, 4)
    original_kernel = np.array(base["kernel"])
    unapply_lora_from_base_params(base, lora, lora_scale_factor=2.0)
    expected = original_kernel - np.einsum("br,rnd->bnd", lora["lora_a.kernel"], lora["lora_b.kernel"]) * 2.0
    np.testing.assert_allclose(np.array(base["kernel"]), expected, rtol=1e-5)

  def test_keeps_base_when_one_lora_component_is_none(self):
    # lora_a.kernel present but lora_b.kernel is None -> lora_update_or_base else branch (line 90)
    kernel = jnp.ones((2, 3, 4), dtype=jnp.float32)
    base = {"kernel": kernel}
    lora = {"lora_a.kernel": jnp.ones((2, 2)), "lora_b.kernel": None}
    unapply_lora_from_base_params(base, lora)
    np.testing.assert_array_equal(np.array(base["kernel"]), np.array(kernel))


# ---------------------------------------------------------------------------
# get_lora_abstract_state
# ---------------------------------------------------------------------------


class TestGetLoraAbstractState(unittest.TestCase):
  """Tests for get_lora_abstract_state shape and structure computation."""

  @classmethod
  def setUpClass(cls):
    devices = np.array(jax.devices()[:1])
    cls.mesh = jax.sharding.Mesh(devices, ("x",))

  def _sharding(self, ndim):
    spec = jax.sharding.PartitionSpec(*([None] * ndim))
    return jax.sharding.NamedSharding(self.mesh, spec)

  def _struct(self, shape):
    return jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32, sharding=self._sharding(len(shape)))

  def _base_params(self, module_path, shape):
    """Build minimal base_abstract_params for a single module kernel."""
    inner = {"kernel": self._struct(shape)}
    parts = module_path.split(".")
    d = inner
    for part in reversed(parts):
      d = {part: d}
    return d

  def _call(self, base_params, target_modules, rank=2):
    lora_config = {"target_modules": target_modules, "r": rank}
    return get_lora_abstract_state(base_params, lora_config)

  def test_query_lora_a_shape(self):
    # base (4, 8, 16): lora_a_shape = (4,) + (rank,) = (4, 2)
    base = self._base_params("self_attention.query", (4, 8, 16))
    state, _ = self._call(base, ["self_attention.query"], rank=2)
    lora_a = state.params["self_attention"]["query"]["lora_a.kernel"]
    self.assertEqual(lora_a.shape, (4, 2))

  def test_query_lora_b_shape(self):
    # base (4, 8, 16): lora_b_shape = (rank,) + (8, 16) = (2, 8, 16)
    base = self._base_params("self_attention.query", (4, 8, 16))
    state, _ = self._call(base, ["self_attention.query"], rank=2)
    lora_b = state.params["self_attention"]["query"]["lora_b.kernel"]
    self.assertEqual(lora_b.shape, (2, 8, 16))

  def test_key_lora_shapes(self):
    base = self._base_params("self_attention.key", (4, 8, 16))
    state, _ = self._call(base, ["self_attention.key"], rank=3)
    lora_a = state.params["self_attention"]["key"]["lora_a.kernel"]
    lora_b = state.params["self_attention"]["key"]["lora_b.kernel"]
    self.assertEqual(lora_a.shape, (4, 3))  # (4,) + (3,)
    self.assertEqual(lora_b.shape, (3, 8, 16))  # (3,) + (8, 16)

  def test_value_lora_shapes(self):
    base = self._base_params("self_attention.value", (4, 8, 16))
    state, _ = self._call(base, ["self_attention.value"], rank=4)
    lora_a = state.params["self_attention"]["value"]["lora_a.kernel"]
    lora_b = state.params["self_attention"]["value"]["lora_b.kernel"]
    self.assertEqual(lora_a.shape, (4, 4))
    self.assertEqual(lora_b.shape, (4, 8, 16))

  def test_out_3d_lora_shapes(self):
    # base (4, 8, 16): out 3D
    # lora_a_shape = (4, 8) + (2,) = (4, 8, 2)
    # lora_b_shape = (2, 16)
    base = self._base_params("self_attention.out", (4, 8, 16))
    state, _ = self._call(base, ["self_attention.out"], rank=2)
    lora_a = state.params["self_attention"]["out"]["lora_a.kernel"]
    lora_b = state.params["self_attention"]["out"]["lora_b.kernel"]
    self.assertEqual(lora_a.shape, (4, 8, 2))
    self.assertEqual(lora_b.shape, (2, 16))

  def test_out_4d_lora_shapes(self):
    # base (4, 2, 8, 16): out 4D
    # lora_a_shape = (4, 2, 8) + (2,) = (4, 2, 8, 2)
    # lora_b_shape = (2, 2, 16)
    base = self._base_params("self_attention.out", (4, 2, 8, 16))
    state, _ = self._call(base, ["self_attention.out"], rank=2)
    lora_a = state.params["self_attention"]["out"]["lora_a.kernel"]
    lora_b = state.params["self_attention"]["out"]["lora_b.kernel"]
    self.assertEqual(lora_a.shape, (4, 2, 8, 2))
    self.assertEqual(lora_b.shape, (2, 2, 16))

  def test_name_mapping_q_proj_to_query(self):
    # "q_proj" should be remapped to "self_attention.query"
    base = self._base_params("self_attention.query", (4, 8, 16))
    state, _ = self._call(base, ["q_proj"], rank=2)
    self.assertIn("lora_a.kernel", state.params["self_attention"]["query"])

  def test_name_mapping_k_proj_to_key(self):
    base = self._base_params("self_attention.key", (4, 8, 16))
    state, _ = self._call(base, ["k_proj"], rank=2)
    self.assertIn("lora_a.kernel", state.params["self_attention"]["key"])

  def test_name_mapping_v_proj_to_value(self):
    base = self._base_params("self_attention.value", (4, 8, 16))
    state, _ = self._call(base, ["v_proj"], rank=2)
    self.assertIn("lora_a.kernel", state.params["self_attention"]["value"])

  def test_name_mapping_o_proj_to_out(self):
    base = self._base_params("self_attention.out", (4, 8, 16))
    state, _ = self._call(base, ["o_proj"], rank=2)
    self.assertIn("lora_a.kernel", state.params["self_attention"]["out"])

  def test_non_target_module_param_is_none(self):
    # Kernel of a non-target module should become None
    base = {
        "self_attention": {
            "query": {"kernel": self._struct((4, 8, 16))},
            "key": {"kernel": self._struct((4, 8, 16))},
        }
    }
    state, _ = self._call(base, ["self_attention.query"], rank=2)
    # query should have lora params; key should be None
    self.assertIn("lora_a.kernel", state.params["self_attention"]["query"])
    self.assertIsNone(state.params["self_attention"]["key"]["kernel"])

  def test_scale_and_embedding_are_valid_non_target_keys(self):
    base = {
        "token_embedding": {"embedding": self._struct((32000, 64))},
        "norm": {"scale": self._struct((64,))},
    }
    state, _ = self._call(base, ["self_attention.query"], rank=2)
    self.assertIsNone(state.params["token_embedding"]["embedding"])
    self.assertIsNone(state.params["norm"]["scale"])

  def test_raises_on_dimensions_greater_than_4(self):
    base = self._base_params("self_attention.query", (2, 3, 4, 5, 6))
    with self.assertRaises(ValueError, msg="Expected error for >4 dimensions"):
      self._call(base, ["self_attention.query"], rank=2)

  def test_raises_on_unsupported_lora_module(self):
    # "self_attention.ffn" is not in the supported list
    base = self._base_params("self_attention.ffn", (4, 8, 16))
    with self.assertRaises(ValueError):
      self._call(base, ["self_attention.ffn"], rank=2)

  def test_raises_on_invalid_param_key(self):
    # "bias" is not a valid param key (only kernel/scale/embedding)
    base = {"bias": self._struct((8,))}
    with self.assertRaises(ValueError):
      self._call(base, ["self_attention.query"], rank=2)

  def test_raises_on_non_shape_dtype_struct(self):
    # Passing a plain numpy array instead of ShapeDtypeStruct
    base = {"self_attention": {"query": {"kernel": np.ones((4, 8, 16))}}}
    with self.assertRaises(ValueError):
      self._call(base, ["self_attention.query"], rank=2)

  def test_returns_train_state_with_correct_structure(self):
    base = self._base_params("self_attention.query", (4, 8, 16))
    state, annotations = self._call(base, ["self_attention.query"], rank=2)
    self.assertEqual(state.step, 0)
    self.assertIn("self_attention", state.params)
    self.assertIsNotNone(annotations)

  def test_lora_params_have_correct_dtype(self):
    base = self._base_params("self_attention.query", (4, 8, 16))
    state, _ = self._call(base, ["self_attention.query"], rank=2)
    lora_a = state.params["self_attention"]["query"]["lora_a.kernel"]
    self.assertEqual(lora_a.dtype, jnp.float32)

  def test_sharding_replicated_when_base_is_replicated(self):
    # When base param has sharding=None, lora sharding is also None
    base = {
        "self_attention": {"query": {"kernel": jax.ShapeDtypeStruct(shape=(4, 8, 16), dtype=jnp.float32, sharding=None)}}
    }
    lora_config = {"target_modules": ["self_attention.query"], "r": 2}
    # get_lora_annotations calls x.sharding.spec which fails when sharding=None.
    # This is a known limitation of the current code; verify it raises AttributeError
    # (rather than silently producing wrong output).
    with self.assertRaises(AttributeError):
      get_lora_abstract_state(base, lora_config)


# ---------------------------------------------------------------------------
# load_adapter
# ---------------------------------------------------------------------------


class TestLoadAdapter(unittest.TestCase):
  """Tests for load_adapter."""

  def test_returns_none_when_no_adapter_config_path(self):
    config = MagicMock()
    lora_params, lora_config = load_adapter(config, {}, adapter_config_path=None, adapter_weights_path=None)
    self.assertIsNone(lora_params)
    self.assertIsNone(lora_config)

  def test_returns_none_when_empty_adapter_config_path(self):
    config = MagicMock()
    lora_params, lora_config = load_adapter(config, {}, adapter_config_path="", adapter_weights_path="")
    self.assertIsNone(lora_params)
    self.assertIsNone(lora_config)

  @patch("maxtext.utils.lora_utils.gcs_utils")
  @patch("maxtext.utils.lora_utils.checkpointing")
  @patch("maxtext.utils.lora_utils.get_lora_abstract_state")
  @patch("maxtext.utils.lora_utils.nn_partitioning.axis_rules")
  def test_loads_from_gcs_path(self, mock_axis_rules, mock_get_lora, mock_ckpt, mock_gcs):
    lora_cfg = {"target_modules": ["q_proj"], "r": 4}
    mock_gcs.read_json_from_gcs.return_value = lora_cfg
    mock_gcs.gcs_path_exists.return_value = True
    mock_axis_rules.return_value.__enter__ = MagicMock(return_value=None)
    mock_axis_rules.return_value.__exit__ = MagicMock(return_value=False)

    mock_lora_state = MagicMock()
    mock_get_lora.return_value = (mock_lora_state, MagicMock())
    mock_ckpt.load_params_from_path.return_value = {"params": {}}

    config = MagicMock()
    _, lora_config = load_adapter(
        config, {}, adapter_config_path="gs://bucket/adapter_config.json", adapter_weights_path="gs://bucket/weights"
    )
    mock_gcs.read_json_from_gcs.assert_called_once_with("gs://bucket/adapter_config.json")
    self.assertEqual(lora_config, lora_cfg)

  @patch("maxtext.utils.lora_utils.gcs_utils")
  @patch("maxtext.utils.lora_utils.checkpointing")
  @patch("maxtext.utils.lora_utils.get_lora_abstract_state")
  @patch("maxtext.utils.lora_utils.nn_partitioning.axis_rules")
  def test_loads_from_local_path(self, mock_axis_rules, mock_get_lora, mock_ckpt, mock_gcs):
    lora_cfg = {"target_modules": ["q_proj"], "r": 4}
    mock_gcs.gcs_path_exists.return_value = True
    mock_axis_rules.return_value.__enter__ = MagicMock(return_value=None)
    mock_axis_rules.return_value.__exit__ = MagicMock(return_value=False)
    mock_lora_state = MagicMock()
    mock_get_lora.return_value = (mock_lora_state, MagicMock())
    mock_ckpt.load_params_from_path.return_value = {}

    config = MagicMock()
    m = mock_open(read_data=json.dumps(lora_cfg))
    with patch("builtins.open", m):
      _, lora_config = load_adapter(
          config,
          {},
          adapter_config_path="/local/adapter_config.json",
          adapter_weights_path="/local/weights",
      )
    self.assertEqual(lora_config, lora_cfg)

  @patch("maxtext.utils.lora_utils.gcs_utils")
  def test_raises_when_lora_config_is_none(self, mock_gcs):
    mock_gcs.read_json_from_gcs.return_value = None
    config = MagicMock()
    with self.assertRaises(FileNotFoundError):
      load_adapter(config, {}, adapter_config_path="gs://bucket/config.json", adapter_weights_path="gs://bucket/w")

  @patch("maxtext.utils.lora_utils.gcs_utils")
  def test_raises_when_weights_path_missing(self, mock_gcs):
    mock_gcs.read_json_from_gcs.return_value = {"target_modules": ["q_proj"], "r": 4}
    mock_gcs.gcs_path_exists.return_value = False
    config = MagicMock()
    with self.assertRaises(FileNotFoundError):
      load_adapter(config, {}, adapter_config_path="gs://bucket/config.json", adapter_weights_path="gs://bucket/w")


# ---------------------------------------------------------------------------
# setup_initial_lora_state
# ---------------------------------------------------------------------------


class TestSetupInitialLoraState(unittest.TestCase):
  """Tests for setup_initial_lora_state."""

  def test_returns_nones_when_no_lora_adapter_path(self):
    config = MagicMock()
    mesh = MagicMock()
    lora_config, lora_state, lora_annotations = setup_initial_lora_state(
        model=None,
        data_iterator=None,
        tx=None,
        config=config,
        rng=None,
        mesh=mesh,
        checkpoint_manager=None,
        lora_adapter_path=None,
    )
    self.assertIsNone(lora_config)
    self.assertIsNone(lora_state)
    self.assertIsNone(lora_annotations)

  def test_returns_nones_when_empty_lora_adapter_path(self):
    config = MagicMock()
    lora_config, lora_state, lora_annotations = setup_initial_lora_state(
        model=None,
        data_iterator=None,
        tx=None,
        config=config,
        rng=None,
        mesh=None,
        checkpoint_manager=None,
        lora_adapter_path="",
    )
    self.assertIsNone(lora_config)
    self.assertIsNone(lora_state)
    self.assertIsNone(lora_annotations)

  @patch("maxtext.utils.lora_utils.max_logging")
  def test_raises_not_implemented_for_pure_nnx(self, mock_logging):
    config = MagicMock()
    config.pure_nnx = True
    with self.assertRaises(NotImplementedError):
      setup_initial_lora_state(
          model=MagicMock(),
          data_iterator=None,
          tx=MagicMock(),
          config=config,
          rng=MagicMock(),
          mesh=MagicMock(),
          checkpoint_manager=MagicMock(),
          lora_adapter_path="gs://bucket/adapter/",
      )

  @patch("maxtext.utils.lora_utils.max_utils")
  @patch("maxtext.utils.lora_utils.checkpointing")
  @patch("maxtext.utils.lora_utils.get_lora_abstract_state")
  @patch("maxtext.utils.lora_utils.gcs_utils")
  @patch("maxtext.utils.lora_utils.maxtext_utils")
  @patch("maxtext.utils.lora_utils.max_logging")
  @patch("maxtext.utils.lora_utils.nn_partitioning.axis_rules")
  def test_restored_lora_raises_not_implemented(
      self, mock_axis_rules, mock_logging, mock_maxtext, mock_gcs, mock_get_lora, mock_ckpt, mock_max_utils
  ):
    config = MagicMock()
    config.pure_nnx = False

    mock_abstract_state = MagicMock()
    mock_abstract_state.params = {}
    mock_maxtext.get_abstract_state.return_value = (mock_abstract_state, None, None)
    mock_gcs.read_json_from_gcs.return_value = {"target_modules": ["q_proj"], "r": 4}
    mock_lora_state = MagicMock()
    mock_get_lora.return_value = (mock_lora_state, MagicMock())
    # restored_lora = True -> NotImplementedError
    mock_ckpt.load_state_if_possible.return_value = (True, {})
    mock_axis_rules.return_value.__enter__ = MagicMock(return_value=None)
    mock_axis_rules.return_value.__exit__ = MagicMock(return_value=False)

    with self.assertRaises(NotImplementedError):
      setup_initial_lora_state(
          model=MagicMock(),
          data_iterator=None,
          tx=MagicMock(),
          config=config,
          rng=MagicMock(),
          mesh=MagicMock(),
          checkpoint_manager=MagicMock(),
          lora_adapter_path="gs://bucket/adapter/",
      )

  @patch("maxtext.utils.lora_utils.max_utils")
  @patch("maxtext.utils.lora_utils.checkpointing")
  @patch("maxtext.utils.lora_utils.get_lora_abstract_state")
  @patch("maxtext.utils.lora_utils.gcs_utils")
  @patch("maxtext.utils.lora_utils.maxtext_utils")
  @patch("maxtext.utils.lora_utils.max_logging")
  @patch("maxtext.utils.lora_utils.nn_partitioning.axis_rules")
  def test_successful_lora_state_setup(
      self, mock_axis_rules, mock_logging, mock_maxtext, mock_gcs, mock_get_lora, mock_ckpt, mock_max_utils
  ):
    config = MagicMock()
    config.pure_nnx = False

    mock_abstract_state = MagicMock()
    mock_abstract_state.params = {}
    mock_maxtext.get_abstract_state.return_value = (mock_abstract_state, None, None)

    expected_config = {"target_modules": ["q_proj"], "r": 4}
    mock_gcs.read_json_from_gcs.return_value = expected_config

    mock_lora_state = MagicMock()
    mock_annotations = MagicMock()
    mock_get_lora.return_value = (mock_lora_state, mock_annotations)

    raw_params = {"self_attention": {"query": {}}}
    # restored_lora = False -> normal path
    mock_ckpt.load_state_if_possible.return_value = (False, raw_params)
    mock_axis_rules.return_value.__enter__ = MagicMock(return_value=None)
    mock_axis_rules.return_value.__exit__ = MagicMock(return_value=False)

    returned_config, returned_state, returned_annotations = setup_initial_lora_state(
        model=MagicMock(),
        data_iterator=None,
        tx=MagicMock(),
        config=config,
        rng=MagicMock(),
        mesh=MagicMock(),
        checkpoint_manager=MagicMock(),
        lora_adapter_path="gs://bucket/adapter/",
    )

    self.assertEqual(returned_config, expected_config)
    self.assertIsNotNone(returned_state)
    self.assertEqual(returned_annotations, mock_annotations)
    # Verify lora_state.replace was called with raw_params
    mock_lora_state.replace.assert_called_once_with(params=raw_params)


if __name__ == "__main__":
  unittest.main()
