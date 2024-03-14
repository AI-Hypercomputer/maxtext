"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Test that all weights are expected dtype (default float32) """
import unittest
from absl.testing import absltest

import pyconfig

import optimizers
from layers import models
from layers import quantizations
import max_utils
import jax
from jax.sharding import Mesh
import jax.numpy as jnp

Transformer = models.Transformer

class WeightDtypes(unittest.TestCase):
    """Test that all weights are expected dtype (default float32) """

    def get_weights(self, argv):
        """ Gets model weights """

        # Setup necessary inputs to build a model state
        pyconfig.initialize(argv)
        config = pyconfig.config
        quant = quantizations.configure_quantization(config)
        devices_array = max_utils.create_device_mesh(config)
        mesh = Mesh(devices_array, config.mesh_axes)
        model = Transformer(config, mesh, quant=quant)
        learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
        tx = optimizers.get_optimizer(config, learning_rate_schedule)
        _, example_rng = jax.random.split(jax.random.PRNGKey(0), 2)

        abstract_state, _ , _ =  max_utils.get_abstract_state(model, tx, config, example_rng, mesh)
        return abstract_state.params

    def assert_weights_are_dtype(self, weights, expected_dtype):
        jax.tree_util.tree_map_with_path(lambda x,y: self.assertEqual(y.dtype, expected_dtype), weights)

    def test_default_float32(self):
        argv = [None, "configs/base.yml", "enable_checkpointing=False"]
        weights = self.get_weights(argv)
        self.assert_weights_are_dtype(weights, jnp.float32)

    def test_set_bf16(self):
        argv = [None, "configs/base.yml", "enable_checkpointing=False", "weight_dtype=bfloat16"]
        weights = self.get_weights(argv)
        self.assert_weights_are_dtype(weights, jnp.bfloat16)


