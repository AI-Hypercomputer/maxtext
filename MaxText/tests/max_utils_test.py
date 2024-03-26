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

""" Tests for the common Max Utils """
import jax
import max_utils
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh
import optax
import pyconfig
import unittest
from layers import models
from layers import quantizations

Transformer = models.Transformer

class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""
  def test_l2norm_pytree(self):
    x = {'a': jax.numpy.array([0, 2, 0]), 'b': jax.numpy.array([0, 3, 6])}
    pytree_l2_norm = max_utils.l2norm_pytree(x)
    self.assertTrue(jax.numpy.allclose(pytree_l2_norm, 7, rtol=1e-05, atol=1e-08, equal_nan=False))

class MaxUtilsInitState(unittest.TestCase):
  """Tests initialization of training and decode states in max_utils.py"""
  def setUp(self):
    self.model = nn.Dense(features=5)
    self.key1, self.key2 = random.split(random.key(0))
    self.input = random.normal(self.key1, (10,)) # Dummy input data
    self.params = self.model.init(self.key2, self.input)
    self.output = self.model.apply(self.params, self.input)
    self.tx = optax.adam(learning_rate=0.001)

  def test_calculate_num_params_from_pytree(self):
    example_tree = [
      [1, 'a', object()],
      (1, (2, 3), ()),
      [1, {'k1': 2, 'k2': (3, 4)}, 5],
      {'a': 2, 'b': (2, 3)},
      jnp.array([1, 2, 3]),
      ]
    self.assertEqual(max_utils.calculate_num_params_from_pytree(example_tree), 17)
    # Model params
    self.assertEqual(max_utils.calculate_num_params_from_pytree(self.params), 55)

  def test_init_train_state(self):
    state = train_state.TrainState(
    step=0,
    apply_fn=self.model.apply,
    params=self.params,
    tx=None, # type: ignore
    opt_state={}
    )
    self.assertEqual(state.tx, None)
    self.assertEqual(state.step, 0)
    self.assertEqual(state.opt_state, {})
    self.assertEqual(state.apply_fn, self.model.apply)
    self.assertEqual(max_utils.calculate_num_params_from_pytree(state.params),
                     max_utils.calculate_num_params_from_pytree(self.params))


  def test_init_decode_state(self):
    decode_state = max_utils.init_decode_state(
      self.model.apply, self.params
    )
    self.assertEqual(decode_state.apply_fn, self.model.apply)
    output = decode_state.apply_fn(self.params, self.input)
    self.assertEqual(output.tolist(), self.output.tolist())
    self.assertEqual(decode_state.tx, None)
    self.assertEqual(decode_state.opt_state, {})
    self.assertEqual(decode_state.step, 0)
    self.assertEqual(
      max_utils.calculate_num_params_from_pytree(decode_state.params),
      max_utils.calculate_num_params_from_pytree(self.params)
    )

  def test_init_training_state(self):
    state = max_utils.init_training_state(self.model.apply, self.params, self.tx)
    self.assertEqual(state.apply_fn, self.model.apply)
    self.assertEqual(state.tx, self.tx)
    self.assertNotEqual(state.opt_state, {})
    self.assertEqual(
      max_utils.calculate_num_params_from_pytree(state.params),
      max_utils.calculate_num_params_from_pytree(self.params)
    )

class ModelWithMultipleCollections(nn.Module):
    """
      A simple model that has variables in multiple collections - "params" and "special_variables"
    """
    def setup(self):
      self.dense = nn.Dense(4)
      self.kernel = self.variable(
        "special_variables", "my_first_kernel", lambda: jnp.ones((4, 5))
      )
    
    def __call__(self, x, y):
      x = self.dense(x)
      x = x @ self.kernel.value
      return x 

class MaxUtilsInitStateWithMultipleCollections(unittest.TestCase):

  def setUp(self):
    pyconfig.initialize([None, "configs/base.yml"], enable_checkpointing=False)
    self.config = pyconfig.config
    self.model = ModelWithMultipleCollections()
    self.key1, self.key2, self.key3 = random.split(random.key(0), num=3)
    self.input = random.normal(self.key1, 
                               (self.config.global_batch_size_to_load, self.config.max_target_length))
    self.params = self.model.init(self.key2, self.input, self.input)
    self.tx = optax.adam(learning_rate=0.001)

  def _test_init_initial_state_driver(self, is_training):
    state_under_test = max_utils.init_initial_state(self.model, self.tx, self.config, is_training, self.key3)
    self.assertEqual(state_under_test.apply_fn, self.model.apply)
    if is_training:
      self.assertEqual(state_under_test.tx, self.tx)
      self.assertNotEqual(state_under_test.opt_state, {})
    else:
      self.assertIsNone(state_under_test.tx)
      self.assertEqual(state_under_test.opt_state, {})
    self.assertEqual(
      max_utils.calculate_num_params_from_pytree(state_under_test.params),
      max_utils.calculate_num_params_from_pytree(self.params)
    )
    self.assertEqual(
      len(self.params),
      len(state_under_test.params)
    )
    self.assertIn("special_variables", state_under_test.params)
    self.assertIn("params", state_under_test.params)
  
  def test_initial_train_state(self):
    self._test_init_initial_state_driver(True)
  
  def test_initial_decode_state(self):
    self._test_init_initial_state_driver(False)


class MaxUtilsInitTransformerState(unittest.TestCase):
  """Tests initialization of transformer states in max_utils.py"""

  def setUp(self):
    pyconfig.initialize([None, "configs/base.yml"], enable_checkpointing=False)
    self.config = pyconfig.config
    devices_array = max_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    quant = quantizations.configure_quantization(self.config)
    self.model = Transformer(self.config, mesh=self.mesh, quant=quant)

  def test_setup_decode_state(self):
    rng = random.PRNGKey(0)
    state, _ = max_utils.setup_decode_state(
      self.model, self.config, rng, self.mesh, None)
    self.assertEqual(state.tx, None)
    self.assertEqual(state.opt_state, {})

  def test_setup_initial_state(self):
    rng = random.PRNGKey(0)
    tx = optax.adam(learning_rate=0.001)
    state, _, _ = max_utils.setup_initial_state(
      self.model, None, tx, self.config, rng, self.mesh, None)
    self.assertEqual(state.tx, tx)
    self.assertNotEqual(state.opt_state, {})

class MaxUtilsT5XCrossEntropy(unittest.TestCase):
  """Tests for the cross entropy functions in max_utils.py"""
  def test_t5x_cross_entropy(self):
    # Generate random targets and logits
    key = jax.random.PRNGKey(0)
    targets = jax.random.randint(key, shape=(48, 2048),
                                        dtype=jax.numpy.int32, minval=1, maxval=10)
    logits = jax.random.uniform(key, shape=(48, 2048, 4096),
                                        dtype=jax.numpy.float32)

    # Calculate xent from optax implementation
    optax_xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    # Calculate xent from custom T5X implementation
    one_hot_targets = jax.nn.one_hot(targets, 4096)
    t5x_xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    t5x_xent = nn.with_logical_constraint(t5x_xent, ('activation_batch', 'activation_length'))

    # Compare results
    self.assertTrue(jax.numpy.allclose(optax_xent, t5x_xent, rtol=1e-05, atol=1e-08, equal_nan=False))

if __name__ == '__main__':
  unittest.main()
