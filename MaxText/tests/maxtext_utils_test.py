"""
Copyright 2024 Google LLC

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

from collections.abc import Callable

""" Tests for the common MaxText utilities """

from typing import Union, Any, Dict, Tuple
import os.path
import unittest

from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.core.scope import FrozenVariableDict
from flax.linen import Dense
from flax.training import train_state

import optax

from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.globals import PKG_DIR
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.maxtext_utils import assert_params_sufficiently_sharded, get_formatted_sharding_annotations

Transformer = models.Transformer


class TestGradientClipping(unittest.TestCase):
  """test class for gradient clipping"""

  def test_grad_clipping_with_no_fp8_stats(self):
    raw_grads = {"params": jnp.array([3.0, -4.0]), "wi_0": jnp.array([5.0, -6.0])}
    clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)
    for param_name, param_val in raw_grads.items():
      # The grads should all be clipped and not equal to what they were before
      self.assertFalse(jnp.array_equal(param_val, clipped_grads[param_name]))

  def test_fp8_stats_not_clipped_but_others_are(self):
    raw_grads = {"params": {"wi_0": jnp.array([5.0, -6.0]), "wi_1": jnp.array([7.0, -8.0])}}
    # Create the reference for how the params would be clipped if no fp8 stats were present
    expected_clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)

    raw_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT] = {
        "amax_history_wi_0": jnp.array([3.0, -4.0]),
        "scale_wi_0": jnp.array([13.2, -4.4]),
    }
    clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)

    # Check all non-fp8 parameters have been clipped in a manner as if the fp8 stats were not present at all
    for param_name in raw_grads["params"]:
      self.assertTrue(jnp.array_equal(expected_clipped_grads["params"][param_name], clipped_grads["params"][param_name]))

    # Then check all fp8 parameters were not clipped at all
    for param_name, raw_value in raw_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT].items():
      self.assertTrue(jnp.array_equal(raw_value, clipped_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT][param_name]))


class TestNestedValueRetrieval(unittest.TestCase):
  """test class for NestedValueRetrieval"""

  def setUp(self):
    self.test_dict = {
        "level1": {
            "level2": {
                "key": 0.1,
            }
        },
        "empty_level": {},
    }

  def test_valid_nested_key(self):
    nested_key = ("level1", "level2", "key")
    expected_value = 0.1
    result = maxtext_utils.get_nested_value(self.test_dict, nested_key, 0.0)
    self.assertEqual(result, expected_value)

  def test_invalid_nested_key(self):
    nested_key = ("level1", "nonexistent", "key")
    expected_value = 0.0
    result = maxtext_utils.get_nested_value(self.test_dict, nested_key, 0.0)
    self.assertEqual(result, expected_value)

  def test_empty_level(self):
    nested_key = ("empty_level", "key")
    expected_value = None
    result = maxtext_utils.get_nested_value(self.test_dict, nested_key)
    self.assertEqual(result, expected_value)


class MaxUtilsInitState(unittest.TestCase):
  """Tests initialization of training and decode states in maxtext_utils.py"""

  def setUp(self):
    self.model = nn.Dense(features=5)
    self.key1, self.key2 = random.split(random.key(0))
    self.input = random.normal(self.key1, (10,))  # Dummy input data
    self.params = self.model.init(self.key2, self.input)
    self.output = self.model.apply(self.params, self.input)
    self.tx = optax.adam(learning_rate=0.001)

  def test_init_train_state(self):
    state = train_state.TrainState(
        step=0, apply_fn=self.model.apply, params=self.params, tx=None, opt_state={}  # type: ignore
    )
    self.assertEqual(state.tx, None)
    self.assertEqual(state.step, 0)
    self.assertEqual(state.opt_state, {})
    self.assertEqual(state.apply_fn, self.model.apply)
    self.assertEqual(
        max_utils.calculate_num_params_from_pytree(state.params), max_utils.calculate_num_params_from_pytree(self.params)
    )

  def test_init_decode_state(self):
    decode_state = maxtext_utils.init_decode_state(self.model.apply, self.params)
    self.assertEqual(decode_state.apply_fn, self.model.apply)
    apply_fn: Callable = decode_state.apply_fn
    # pylint: disable=not-callable
    output: Union[Any, Tuple[Any, Union[FrozenVariableDict, Dict[str, Any]]]] = apply_fn(self.params, self.input)
    self.assertEqual(output.tolist(), self.output.tolist())
    self.assertEqual(decode_state.tx, None)
    self.assertEqual(decode_state.opt_state, {})
    self.assertEqual(decode_state.step, 0)
    self.assertEqual(
        max_utils.calculate_num_params_from_pytree(decode_state.params),
        max_utils.calculate_num_params_from_pytree(self.params),
    )

  def test_init_training_state(self):
    state = maxtext_utils.init_training_state(self.model.apply, self.params, self.tx)
    self.assertEqual(state.apply_fn, self.model.apply)
    self.assertEqual(state.tx, self.tx)
    self.assertNotEqual(state.opt_state, {})
    self.assertEqual(
        max_utils.calculate_num_params_from_pytree(state.params), max_utils.calculate_num_params_from_pytree(self.params)
    )


class ModelWithMultipleCollections(nn.Module):
  """
  A simple model that has variables in multiple collections - "params" and "special_variables"
  """

  dense: Dense = nn.Dense(4)

  def setup(self):
    self.kernel = self.variable("special_variables", "my_first_kernel", lambda: jnp.ones((4, 5)))

  def __call__(self, x, y, encoder_images=None):
    x = self.dense(x)
    x = x @ self.kernel.value
    return x


class MaxUtilsInitStateWithMultipleCollections(unittest.TestCase):
  """test class for multiple collection state in maxutils"""

  def setUp(self):
    self.config = pyconfig.initialize([None, os.path.join(PKG_DIR, "configs", "base.yml")], enable_checkpointing=False)
    self.model = ModelWithMultipleCollections()
    self.key1, self.key2, self.key3 = random.split(random.key(0), num=3)
    self.input = random.normal(self.key1, (self.config.global_batch_size_to_load, self.config.max_target_length))
    self.params = self.model.init(self.key2, self.input, self.input)
    self.tx = optax.adam(learning_rate=0.001)

  def _test_init_initial_state_driver(self, is_training):
    """test initiating of the initial state driver"""
    state_under_test = maxtext_utils.init_initial_state(self.model, self.tx, self.config, is_training, self.key3)
    self.assertEqual(state_under_test.apply_fn, self.model.apply)
    if is_training:
      self.assertEqual(state_under_test.tx, self.tx)
      self.assertNotEqual(state_under_test.opt_state, {})
    else:
      self.assertIsNone(state_under_test.tx)
      self.assertEqual(state_under_test.opt_state, {})
    self.assertEqual(
        max_utils.calculate_num_params_from_pytree(state_under_test.params),
        max_utils.calculate_num_params_from_pytree(self.params),
    )
    self.assertEqual(len(self.params), len(state_under_test.params))
    self.assertIn("special_variables", state_under_test.params)
    self.assertIn("params", state_under_test.params)

  def test_initial_train_state(self):
    self._test_init_initial_state_driver(True)

  def test_initial_decode_state(self):
    self._test_init_initial_state_driver(False)


class MaxUtilsInitTransformerState(unittest.TestCase):
  """Tests initialization of transformer states in max_utils.py"""

  def setUp(self):
    self.config = pyconfig.initialize([None, os.path.join(PKG_DIR, "configs", "base.yml")], enable_checkpointing=False)
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    quant = quantizations.configure_quantization(self.config)
    self.model = Transformer(self.config, mesh=self.mesh, quant=quant)

  def test_setup_decode_state(self):
    rng = random.PRNGKey(0)
    state, _ = maxtext_utils.setup_decode_state(self.model, self.config, rng, self.mesh, None)
    self.assertEqual(state.tx, None)
    self.assertEqual(state.opt_state, {})

  def test_setup_initial_state(self):
    rng = random.PRNGKey(0)
    tx = optax.adam(learning_rate=0.001)
    state, _, _, _ = maxtext_utils.setup_initial_state(self.model, None, tx, self.config, rng, self.mesh, None)
    self.assertEqual(state.tx, tx)
    self.assertNotEqual(state.opt_state, {})


class MaxUtilsPpAsDp(unittest.TestCase):
  """Tests logical_axis_rules_pp_act_as_dp converts rules so stage is added before data."""

  def test_stage_added_before_data(self):
    input_rules = (("activation_batch", ("data", "fsdp")),)
    expected_transform = (("activation_batch", ("stage", "data", "fsdp")),)
    transformed_rules = maxtext_utils.logical_axis_rules_pp_act_as_dp(input_rules)
    self.assertEqual(transformed_rules, expected_transform)

  def test_stage_removed(self):
    input_rules = (("layers", "stage"),)
    expected_transform = (
        (
            "layers",
            (),
        ),
    )
    transformed_rules = maxtext_utils.logical_axis_rules_pp_act_as_dp(input_rules)
    self.assertEqual(transformed_rules, expected_transform)

  def multiple_rules(self):
    """test multiple rules"""
    input_rules = (
        ("activation_batch", ("data", "fsdp")),
        ("layers", "stage"),
        ("experts", "expert"),
    )
    expected_transform = (
        ("activation_batch", ("stage", "data", "fsdp")),
        ("layers", ()),
        ("experts", "expert"),
    )
    transformed_rules = maxtext_utils.logical_axis_rules_pp_act_as_dp(input_rules)
    self.assertEqual(transformed_rules, expected_transform)


class TestRollAndMask(unittest.TestCase):
  """Test class for utility functions supporting Roll and Mask."""

  def test_mtp_roll_and_mask_shapes(self):
    """
    Validates that roll_and_mask works correctly on the specific tensor shapes
    that will be passed during training. The primary use case involves tensors
    with a [batch, sequence_length] shape.
    """
    batch_size = 4
    seq_len = 8
    # Create a dummy input tensor that mimics `input_ids` or `target_ids`.
    # The values are sequential for easy validation.
    # Shape: [4, 8]
    input_tensor = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape((batch_size, seq_len))

    # print(input_tensor)

    # --- Test Case 1: Default left shift by 1 ---
    # This is the most common operation inside the MTP loop.
    rolled_by_1 = maxtext_utils.roll_and_mask(input_tensor, shift=-1)

    # Manually construct the expected output using jnp
    expected_1 = jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 0],  # First row rolled left, last element masked
            [9, 10, 11, 12, 13, 14, 15, 0],  # Second row rolled left
            [17, 18, 19, 20, 21, 22, 23, 0],
            [25, 26, 27, 28, 29, 30, 31, 0],
        ],
        dtype=jnp.int32,
    )

    self.assertEqual(rolled_by_1.shape, (batch_size, seq_len), "Shape should be preserved after rolling.")
    self.assertTrue(jnp.array_equal(rolled_by_1, expected_1), "Array content is incorrect after shift by -1.")

    # --- Test Case 2: Larger left shift by 3 ---
    # This simulates a later step in a hypothetical MTP loop.
    rolled_by_3 = maxtext_utils.roll_and_mask(input_tensor, shift=-3)

    # Manually construct the expected output using jnp
    expected_3 = jnp.array(
        [
            [3, 4, 5, 6, 7, 0, 0, 0],  # First row rolled left by 3, last 3 masked
            [11, 12, 13, 14, 15, 0, 0, 0],
            [19, 20, 21, 22, 23, 0, 0, 0],
            [27, 28, 29, 30, 31, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    self.assertEqual(rolled_by_3.shape, (batch_size, seq_len), "Shape should be preserved after rolling.")
    self.assertTrue(jnp.array_equal(rolled_by_3, expected_3), "Array content is incorrect after shift by -3.")

    # --- Test Case 3: Shift of 0 (edge case) ---
    # This should result in no change to the tensor.
    rolled_by_0 = maxtext_utils.roll_and_mask(input_tensor, shift=0)
    self.assertTrue(jnp.array_equal(rolled_by_0, input_tensor), "A shift of 0 should be a no-op.")


class TestAssertParamsSufficientlySharded(unittest.TestCase):
  """
  Test suite for the sharding assertion utility function 'assert_params_sufficiently_sharded'.
  """

  def setUp(self):
    """
    Set up the test environment before each test method is run.
    This method initializes a device mesh required for sharding tests.
    """
    # Skip these tests if the environment has fewer than 4 devices, as the mesh requires them.
    if len(jax.devices()) < 4:
      self.skipTest("This test suite requires at least 4 TPU devices.")
    # Create a 2x2 device mesh from the first 4 available JAX devices.
    devices = np.array(jax.devices()[:4]).reshape((2, 2))
    # Define the non-trival mesh axes and a broader set of mesh axes.
    nonTrival_mesh_axes = ("fsdp", "tensor")
    self.mesh = Mesh(devices, nonTrival_mesh_axes)
    self.mesh_axes = ("fsdp", "sequence", "tensor", "stage", "context")

  def test_fully_sharded_2d(self):
    """
    Tests that a 2D tensor fully sharded across both mesh axes passes the assertion.
    """
    # Activate the mesh context.
    with self.mesh:
      # Define a sharding spec that shards the first tensor dimension by the 'fsdp' mesh axis
      # and the second dimension by the 'tensor' mesh axis.
      pspec = PartitionSpec("fsdp", "tensor")
      # Create a parameter and apply the sharding, ensuring it's distributed across all devices.
      params = {"layer1": jax.device_put(jnp.ones((8, 8)), NamedSharding(self.mesh, pspec))}

      # Assert that the parameters are sufficiently sharded; this should pass with no error.
      assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.1)

  def test_unsharded_fails(self):
    """
    Tests that a completely unsharded (fully replicated) parameter fails the assertion.
    """
    with self.mesh:
      # Create a parameter without any sharding specification. It will be replicated on all devices.
      params = {"layer1": jnp.ones((8, 8))}

      # Expect an AssertionError because 100% of params are unsharded, exceeding the 10% tolerance.
      with self.assertRaises(AssertionError):
        assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.1)

  def test_mixed_sharding_fails(self):
    """
    Tests that a mix of sharded and unsharded parameters fails when the unsharded
    portion exceeds the tolerance.
    """
    with self.mesh:
      sharded_param = jax.device_put(jnp.ones((8, 8)), NamedSharding(self.mesh, PartitionSpec("fsdp", "tensor")))
      unsharded_param = jnp.ones((8, 8))
      params = {"layer1": sharded_param, "layer2": unsharded_param}

      with self.assertRaises(AssertionError):
        assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.5)

  def test_3d_tensor_sharded_on_fsdp_axis(self):
    """
    Tests that a 3D tensor sharded only on a valid target axis ('fsdp') should fail.
    """
    with self.mesh:
      pspec = PartitionSpec("fsdp", None, None)
      params = {"conv3d_layer": jax.device_put(jnp.ones((8, 4, 4)), NamedSharding(self.mesh, pspec))}

      with self.assertRaises(AssertionError):
        assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.2)

  def test_multi_axis_sharding_pass(self):
    """
    Tests that a tensor sharded with a valid axis ('fsdp') on a complex,
    multi-dimensional mesh passes the assertion.
    """
    # Create a mesh shape for a 5D mesh.
    devices = np.array(jax.devices()).reshape((4, 1, 1, 1, 1))
    mesh = Mesh(devices, self.mesh_axes)

    with mesh:
      # Shard across multiple axes, including the valid 'fsdp' axis.
      pspec = PartitionSpec(("fsdp", "sequence"), "stage", ("tensor"), None)
      params = {"complex_layer": jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(mesh, pspec))}

      # This should pass because 'fsdp' is a valid sharding axis being used.
      assert_params_sufficiently_sharded(params, mesh, tolerance=0.05)

  def test_multi_axis_not_sharded_fails(self):
    """
    Tests that a tensor on a complex mesh fails if it's not sharded along any
    of the primary valid axes (like 'fsdp').
    """
    devices = np.array(jax.devices()).reshape((4, 1, 1, 1, 1))
    mesh = Mesh(devices, self.mesh_axes)
    with mesh:
      pspec = PartitionSpec(("sequence", "context"), "stage", "tensor", None)
      params = {"complex_layer": jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(mesh, pspec))}

      with self.assertRaises(AssertionError):
        assert_params_sufficiently_sharded(params, mesh, tolerance=0.05)

  def test_multi_axis_mixed_sharding_fails(self):
    """
    Tests that a mix of sharded (correctly) and unsharded tensors on a complex mesh fails.
    """
    devices = np.array(jax.devices()).reshape((4, 1, 1, 1, 1))
    mesh = Mesh(devices, self.mesh_axes)
    with mesh:
      sharded_pspec = PartitionSpec(("fsdp", "sequence"), "stage", ("tensor"), None)
      sharded_param = jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(mesh, sharded_pspec))
      unsharded_param = jnp.ones((8, 8, 2, 2))
      params = {
          "sharded_layer": sharded_param,
          "unsharded_layer": unsharded_param,
      }

      with self.assertRaises(AssertionError):
        assert_params_sufficiently_sharded(params, mesh, tolerance=0.5)


class TestAssert_Formatted_sharding_annotations(unittest.TestCase):
  """
  Test suite for sharding assertion formating functions.
  """

  def setUp(self):
    """
    Set up the common 2*2 mesh for sharding tests.
    """
    if len(jax.devices()) < 4:
      self.skipTest("This test suite requires at least 4 TPU devices")

    self.mesh_axes = ("fsdp", "sequence", "tensor", "stage", "context")
    devices = np.array(jax.devices()).reshape((4, 1, 1, 1, 1))
    self.mesh = Mesh(devices, self.mesh_axes)

  def test_multi_axis_mixed_formating(self):
    """
    Tests a mix of sharded and unsharded tensors on a complex mesh fails.
    """
    with self.mesh:
      sharded_pspec = PartitionSpec(("fsdp", "sequence"), "stage", ("tensor"), None)
      sharded_param = jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(self.mesh, sharded_pspec))
      unsharded_param = jnp.ones((8, 8, 2, 2))
      params = {
          "sharded_layer": sharded_param,
          "unsharded_layer": unsharded_param,
      }
      self.assertIsNotNone(get_formatted_sharding_annotations(params, self.mesh))


if __name__ == "__main__":
  unittest.main()
