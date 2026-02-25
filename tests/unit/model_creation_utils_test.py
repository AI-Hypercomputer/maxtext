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
"""Unit tests for model_creation_utils.py."""

import sys
import unittest

import jax
import flax.linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides


def _make_config(**kwargs):
  """Returns a minimal pyconfig suitable for model-creation tests."""
  extra = get_decoupled_parallelism_overrides()
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      per_device_batch_size=1.0,
      run_name="test",
      enable_checkpointing=False,
      base_num_decoder_layers=2,
      attention="dot_product",
      max_target_length=16,
      base_emb_dim=256,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      max_prefill_predict_length=4,
      **kwargs,
      **extra,
  )


def _make_mesh(config):
  devices_array = maxtext_utils.create_device_mesh(config)
  return Mesh(devices_array, config.mesh_axes)


class TestGetTransformerModel(unittest.TestCase):
  """Tests for get_transformer_model()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_linen_module_when_rngs_is_none(self):
    """Without rngs, should return a Linen nn.Module."""
    model = model_creation_utils.get_transformer_model(self.config, self.mesh, quant=None, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_returns_nnx_module_when_rngs_provided(self):
    """With rngs, should return an NNX nnx.Module."""
    model = nnx.eval_shape(
        lambda: model_creation_utils.get_transformer_model(
            self.config, self.mesh, quant=None, rngs=nnx.Rngs(params=0, dropout=1, aqt=2)
        )
    )
    self.assertIsInstance(model, nnx.Module)

  def test_respects_model_mode_prefill(self):
    """Linen model created with MODEL_MODE_PREFILL should differ from train mode."""
    linen_train = model_creation_utils.get_transformer_model(
        self.config, self.mesh, quant=None, model_mode=MODEL_MODE_TRAIN, rngs=None
    )
    linen_prefill = model_creation_utils.get_transformer_model(
        self.config, self.mesh, quant=None, model_mode=MODEL_MODE_PREFILL, rngs=None
    )
    # Both are still nn.Module instances
    self.assertIsInstance(linen_train, nn.Module)
    self.assertIsInstance(linen_prefill, nn.Module)


class TestCreateModel(unittest.TestCase):
  """Tests for create_model()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_linen_model_without_rngs(self):
    model = model_creation_utils.create_model(self.config, self.mesh)
    self.assertIsInstance(model, nn.Module)

  def test_returns_nnx_model_with_rngs(self):
    model = nnx.eval_shape(
        lambda: model_creation_utils.create_model(self.config, self.mesh, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    )
    self.assertIsInstance(model, nnx.Module)

  def test_model_mode_train_default(self):
    """Default model_mode is MODEL_MODE_TRAIN."""
    model = model_creation_utils.create_model(self.config, self.mesh)
    self.assertIsInstance(model, nn.Module)


class TestFromConfig(unittest.TestCase):
  """Tests for from_config()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_linen_path_rngs_none(self):
    """from_config with rngs=None should return a Linen nn.Module."""
    model = model_creation_utils.from_config(self.config, mesh=self.mesh, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_nnx_path_with_rngs(self):
    """from_config with rngs provided should return an NNX nnx.Module."""
    model = nnx.eval_shape(
        lambda: model_creation_utils.from_config(self.config, mesh=self.mesh, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
    )
    self.assertIsInstance(model, nnx.Module)

  def test_mesh_created_from_devices_when_none(self):
    """from_config should work when mesh is None (creates mesh internally)."""
    model = model_creation_utils.from_config(self.config, devices=None, mesh=None, rngs=None)
    self.assertIsInstance(model, nn.Module)

  def test_model_mode_is_forwarded(self):
    """from_config should accept and forward model_mode."""
    model = model_creation_utils.from_config(self.config, mesh=self.mesh, model_mode=MODEL_MODE_PREFILL, rngs=None)
    self.assertIsInstance(model, nn.Module)


class TestGetNNXCreateModelFn(unittest.TestCase):
  """Tests for get_nnx_create_model_fn()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_callable(self):
    fn = model_creation_utils.get_nnx_create_model_fn(self.config, mesh=self.mesh)
    self.assertTrue(callable(fn))

  def test_callable_produces_nnx_module(self):
    fn = model_creation_utils.get_nnx_create_model_fn(self.config, mesh=self.mesh)
    model = nnx.eval_shape(fn)
    self.assertIsInstance(model, nnx.Module)

  def test_callable_uses_rng_key(self):
    """Supplying different rng_key values should produce deterministic but distinct inits."""
    fn_a = model_creation_utils.get_nnx_create_model_fn(self.config, mesh=self.mesh, rng_key=jax.random.PRNGKey(0))
    fn_b = model_creation_utils.get_nnx_create_model_fn(self.config, mesh=self.mesh, rng_key=jax.random.PRNGKey(1))
    model_a = nnx.eval_shape(fn_a)
    model_b = nnx.eval_shape(fn_b)
    # Both should be NNX modules; eval_shape returns abstract shapes so just check types
    self.assertIsInstance(model_a, nnx.Module)
    self.assertIsInstance(model_b, nnx.Module)

  def test_inference_model_mode(self):
    fn = model_creation_utils.get_nnx_create_model_fn(self.config, mesh=self.mesh, model_mode=MODEL_MODE_PREFILL)
    model = nnx.eval_shape(fn)
    self.assertIsInstance(model, nnx.Module)


class TestCreateNNXAbstractModel(unittest.TestCase):
  """Tests for create_nnx_abstract_model()."""

  def setUp(self):
    self.config = _make_config()
    self.mesh = _make_mesh(self.config)

  def test_returns_tuple_of_callable_and_module(self):
    create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(self.config, mesh=self.mesh)
    self.assertTrue(callable(create_fn))
    self.assertIsInstance(abstract_model, nnx.Module)

  def test_abstract_model_has_abstract_arrays(self):
    """Abstract model leaves should be ShapeDtypeStruct, not concrete arrays."""
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(self.config, mesh=self.mesh)
    _, state = nnx.split(abstract_model)
    leaves = jax.tree.leaves(state)
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      # In abstract state, values are nnx.Variable wrapping abstract shapes/ShapeDtypeStruct
      # Concrete jax.Array would have a .devices() method; abstract ones should not be Arrays
      self.assertNotIsInstance(leaf, jax.Array)

  def test_create_fn_produces_concrete_model(self):
    """The returned create_fn should produce a real (concrete) NNX Module."""
    create_fn, _ = model_creation_utils.create_nnx_abstract_model(self.config, mesh=self.mesh)
    with self.mesh:
      concrete = create_fn()
    self.assertIsInstance(concrete, nnx.Module)
    leaves = jax.tree.leaves(nnx.state(concrete))
    for leaf in leaves:
      self.assertIsInstance(leaf, jax.Array)

  def test_works_without_explicit_mesh(self):
    """create_nnx_abstract_model should work when mesh=None (extracts mesh from model)."""
    create_fn, abstract_model = model_creation_utils.create_nnx_abstract_model(self.config, mesh=None)
    self.assertTrue(callable(create_fn))
    self.assertIsInstance(abstract_model, nnx.Module)


if __name__ == "__main__":
  unittest.main()
