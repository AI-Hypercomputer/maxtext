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

""" Test that all weights are expected dtype (default float32) """
from functools import partial
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

Transformer = models.transformer_as_linen


class StateDtypes(unittest.TestCase):
  """Tests that state has expected dtypes, e.g. weights default to float32"""

  def get_state(self, argv):
    """Gets model state including weights and optimizer state"""
    # Conditionally set ici_fsdp_parallelism to match device count in decoupled mode
    if is_decoupled():
      argv = list(argv) + [f"ici_fsdp_parallelism={jax.device_count()}"]

    # Setup necessary inputs to build a model state
    config = pyconfig.initialize(argv)
    quant = quantizations.configure_quantization(config)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    model = Transformer(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    tx = optimizers.get_optimizer(config, learning_rate_schedule)
    _, example_rng = jax.random.split(jax.random.PRNGKey(0), 2)

    if config.pure_nnx:
      # NNX has a different function to init the training state.
      raise NotImplementedError("Pure NNX support has not been implemented yet.")
    else:
      init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, True, example_rng)
    abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, True)
    return abstract_state

  def get_weights(self, argv):
    return self.get_state(argv).params

  def get_mu(self, argv):
    return self.get_state(argv).opt_state[0].mu

  def assert_pytree_is_dtype(self, weights, expected_dtype):
    jax.tree_util.tree_map_with_path(lambda x, y: self.assertEqual(y.dtype, expected_dtype), weights)

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
