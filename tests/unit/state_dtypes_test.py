# Copyright 2023–2025 Google LLC
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

"""Test that all weights are expected dtype (default float32)"""
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from functools import partial
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.layers import train_state_nnx
from tests.utils.test_helpers import get_test_config_path

Transformer = models.transformer_as_linen


class StateDtypes(unittest.TestCase):
  """Tests that state has expected dtypes, e.g. weights default to float32"""

  def get_state(self, argv):
    """Gets model state including weights and optimizer state"""
    # Setup necessary inputs to build a model state
    config = pyconfig.initialize(argv)
    quant = quantizations.configure_quantization(config)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    if config.pure_nnx:
      _create_model_partial, model = model_creation_utils.create_nnx_abstract_model(config, mesh)
    else:
      model = Transformer(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    tx = optimizers.get_optimizer(config, learning_rate_schedule, model)
    _, example_rng = jax.random.split(jax.random.PRNGKey(0), 2)

    if config.pure_nnx:

      def create_train_state_fn():
        nnx_model = _create_model_partial()
        optimizer = nnx.Optimizer(nnx_model, tx, wrt=nnx.Param)
        return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

      init_state_fn = create_train_state_fn
    else:
      init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, True, example_rng)
    abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, True)
    return abstract_state, config.pure_nnx

  def get_weights(self, argv):
    state, is_nnx = self.get_state(argv)
    if is_nnx:
      return state.model
    return state.params

  def get_mu(self, argv):
    state, is_nnx = self.get_state(argv)
    if is_nnx:
      return state.optimizer.opt_state[0]["mu"]
    return state.opt_state[0].mu

  def assert_pytree_is_dtype(self, weights, expected_dtype):
    """Asserts that all valid parameter arrays within the PyTree match the expected dtype."""

    def check_dtype(path, leaf):
      # Support NNX Variable objects which wrap the array in `.value`
      if hasattr(getattr(leaf, "value", None), "dtype"):
        leaf_dtype = leaf.value.dtype
      elif hasattr(leaf, "dtype"):
        leaf_dtype = leaf.dtype
      else:
        return

      # Skip PRNG keys
      if type(leaf_dtype).__name__ == "KeyTy" or str(leaf_dtype).startswith("key<"):
        return

      if jnp.issubdtype(leaf_dtype, jnp.integer):
        # Skip integer fields like step counters
        return
      self.assertEqual(jnp.dtype(leaf_dtype), jnp.dtype(expected_dtype))

    jax.tree_util.tree_map_with_path(
        check_dtype, weights, is_leaf=lambda x: hasattr(x, "value") and hasattr(x.value, "dtype")
    )

  def test_default_float32(self):
    argv = [None, get_test_config_path(), "enable_checkpointing=False"]
    weights = self.get_weights(argv)
    self.assert_pytree_is_dtype(weights, jnp.float32)

  def test_set_bf16(self):
    argv = [
        None,
        get_test_config_path(),
        "enable_checkpointing=False",
        "weight_dtype=bfloat16",
    ]
    weights = self.get_weights(argv)
    self.assert_pytree_is_dtype(weights, jnp.bfloat16)

  def test_default_mu_float32(self):
    argv = [None, get_test_config_path(), "enable_checkpointing=False"]
    mu = self.get_mu(argv)
    self.assert_pytree_is_dtype(mu, jnp.float32)

  def test_set_mu_bf16(self):
    argv = [None, get_test_config_path(), "enable_checkpointing=False", "mu_dtype=bfloat16"]
    mu = self.get_mu(argv)
    self.assert_pytree_is_dtype(mu, jnp.bfloat16)
