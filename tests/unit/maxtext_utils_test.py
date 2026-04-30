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

"""Tests for the common MaxText utilities"""

import functools
from typing import Any, Sequence
from collections.abc import Callable
import unittest
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass, field

from flax import linen as nn
from flax import nnx
from flax.core.scope import FrozenVariableDict
from flax.training import train_state
import jax
from jax import random, vmap
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from maxtext.configs import pyconfig
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from maxtext.inference import inference_utils
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.utils.sharding import assert_params_sufficiently_sharded, get_formatted_sharding_annotations
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides
import numpy as np
import optax

Transformer = models.transformer_as_linen


class TestGradientClipping(unittest.TestCase):
  """test class for gradient clipping"""

  def test_grad_clipping_with_no_fp8_stats(self):
    raw_grads = {
        "params": jnp.array([3.0, -4.0]),
        "wi_0": jnp.array([5.0, -6.0]),
    }
    clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)
    for param_name, param_val in raw_grads.items():
      # The grads should all be clipped and not equal to what they were before
      self.assertFalse(jnp.array_equal(param_val, clipped_grads[param_name]))

  def test_fp8_stats_not_clipped_but_others_are(self):
    raw_grads = {
        "params": {
            "wi_0": jnp.array([5.0, -6.0]),
            "wi_1": jnp.array([7.0, -8.0]),
        }
    }
    # Create the reference for how the params would be clipped if no fp8 stats were present
    expected_clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)

    raw_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT] = {
        "amax_history_wi_0": jnp.array([3.0, -4.0]),
        "scale_wi_0": jnp.array([13.2, -4.4]),
    }
    clipped_grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, 1.0)

    # Check all non-fp8 parameters have been clipped in a manner as if the fp8 stats were not present at all
    for param_name in raw_grads["params"]:
      self.assertTrue(
          jnp.array_equal(
              expected_clipped_grads["params"][param_name],
              clipped_grads["params"][param_name],
          )
      )

    # Then check all fp8 parameters were not clipped at all
    for param_name, raw_value in raw_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT].items():
      self.assertTrue(
          jnp.array_equal(
              raw_value,
              clipped_grads[maxtext_utils.OVERWRITE_WITH_GRADIENT][param_name],
          )
      )


class TestIntermediateValueRetrieval(unittest.TestCase):
  """test class for IntermediateValueRetrieval"""

  def setUp(self):
    self.mock_model = MagicMock(name="Transformer")

    # 2. Create the Decoder Mock
    self.mock_decoder = MagicMock(name="Decoder")
    self.mock_model.decoder = self.mock_decoder
    self.mock_layers = {}
    self.mock_model.decoder.layers = self.mock_layers
    self.self_attention = {}
    self.mock_layers["self_attention"] = self.self_attention

  def test_valid_intermediate_key(self):
    expected_sowed_data = [0.1, 0.5, 0.9]
    mock_sowed_variable = Mock(name="out_projection_activations")
    mock_sowed_variable.get_value.return_value = (expected_sowed_data,)

    self.mock_decoder.layers["self_attention"]["out_projection_activations"] = mock_sowed_variable

    result = maxtext_utils.get_intermediate_value(self.mock_model, "out_projection_activations")

    self.assertEqual(result, expected_sowed_data)

  def test_returns_default_if_sow_did_not_happen(self):
    """
    Simulate a scenario where the model ran, but this specific key
    was NOT sowed (or the layer was skipped).
    """

    result = maxtext_utils.get_intermediate_value(self.mock_model, "out_projection_activations", default="MyDefault")

    self.assertEqual(result, "MyDefault")

  def test_unknown_key_raises_value_error(self):
    with self.assertRaises(ValueError) as cm:
      maxtext_utils.get_intermediate_value(self.mock_model, "some_random_layer_name")

    self.assertEqual(str(cm.exception), "Incorrect nested_key: some_random_layer_name")


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


class UpdateStateParamTest(unittest.TestCase):

  def setUp(self):
    self.model = nn.Dense(features=5)
    self.initial_params = {
        "layers": {
            "layer_0": {"bias": jnp.array([1.0, 1.0])},
            "layer_1": {"bias": jnp.array([2.0, 2.0])},
        },
        "decoder": {"gate": {"bias": jnp.array([0.5, 0.5])}},
    }
    self.state = train_state.TrainState(
        step=0,
        apply_fn=self.model.apply,
        params=self.initial_params,
        tx=None,
        opt_state={},
    )

  def test_update_mode_add(self):
    target_path = ("decoder", "gate", "bias")
    update_value = jnp.array([0.1, 0.2])
    new_state = maxtext_utils.update_state_param(self.state, target_path, update_value)

    expected = jnp.array([0.6, 0.7])
    actual = new_state.params["decoder"]["gate"]["bias"]
    self.assertTrue(jnp.allclose(actual, expected))

    # Other values are untouched
    original_layer_0 = self.state.params["layers"]["layer_0"]["bias"]
    new_layer_0 = new_state.params["layers"]["layer_0"]["bias"]
    self.assertTrue(jnp.array_equal(original_layer_0, new_layer_0))
    original_layer_1 = self.state.params["layers"]["layer_1"]["bias"]
    new_layer_1 = new_state.params["layers"]["layer_1"]["bias"]
    self.assertTrue(jnp.array_equal(original_layer_1, new_layer_1))

  def test_invalid_path_does_nothing(self):
    """If path doesn't exist (or is wrong), nothing should happen."""
    # Note: tree_map only iterates over EXISTING leaves. If path is wrong,
    # the if condition inside never triggers.
    target_path = ("decoder", "non_existent", "bias")
    new_state = maxtext_utils.update_state_param(self.state, target_path, jnp.array([1.0]))

    self.assertTrue(jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, new_state.params, self.state.params)))


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
        max_utils.calculate_num_params_from_pytree(state.params),
        max_utils.calculate_num_params_from_pytree(self.params),
    )

  def test_init_decode_state(self):
    decode_state = maxtext_utils.init_decode_state(self.model.apply, self.params)
    self.assertEqual(decode_state.apply_fn, self.model.apply)
    apply_fn: Callable = decode_state.apply_fn
    # pylint: disable=not-callable
    output: Any | tuple[Any, FrozenVariableDict | dict[str, Any]] = apply_fn(self.params, self.input)
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
        max_utils.calculate_num_params_from_pytree(state.params),
        max_utils.calculate_num_params_from_pytree(self.params),
    )


@nnx.register_variable_name("special_variables")
class SpecialVariables(nnx.Variable):
  pass


class ModelWithMultipleCollections(nnx.Module):
  """A simple model that has variables in multiple collections - "params" and "special_variables" """

  def __init__(self, input_dim: int, rngs: nnx.Rngs | None = None):
    self.dense = nnx.Linear(input_dim, 4, rngs=rngs)
    self.my_first_kernel = SpecialVariables(jnp.ones((4, 5)))

  def __call__(self, x, y, encoder_images=None, nnx_method=None, model_mode=None):
    x = self.dense(x)
    x = x @ self.my_first_kernel
    return x


class TrainState(train_state.TrainState):
  other_variables: nnx.State


class MaxUtilsInitStateWithMultipleCollections(unittest.TestCase):
  """test class for multiple collection state in maxutils"""

  def setUp(self):
    self.config = pyconfig.initialize([None, get_test_config_path()], enable_checkpointing=False)
    self.model = ModelWithMultipleCollections(self.config.max_target_length, nnx.Rngs(0))
    self.key = random.key(0)
    self.tx = optax.adam(learning_rate=0.001)

  def _test_init_initial_state_driver(self, is_training):
    """test initiating of the initial state driver"""
    if is_training:
      self.model.train()
    else:
      self.model.eval()

    graphdef, params, other_variables = nnx.split(self.model, nnx.Param, ...)

    state_under_test = None
    if is_training:
      state_under_test = TrainState.create(
          apply_fn=graphdef.apply,
          params=params,
          other_variables=other_variables,
          tx=self.tx,
      )
    else:
      state_under_test = TrainState(
          step=0,
          apply_fn=graphdef.apply,
          params=params,
          other_variables=other_variables,
          tx=None,
          opt_state={},
      )

    self.assertEqual(state_under_test.apply_fn, graphdef.apply)
    if is_training:
      self.assertEqual(state_under_test.tx, self.tx)
      self.assertNotEqual(state_under_test.opt_state, {})
    else:
      self.assertIsNone(state_under_test.tx)
      self.assertEqual(state_under_test.opt_state, {})
    self.assertEqual(
        max_utils.calculate_num_params_from_pytree(state_under_test.params),
        max_utils.calculate_num_params_from_pytree(params),
    )
    self.assertEqual(len(params), len(state_under_test.params))
    self.assertIsInstance(state_under_test.other_variables["my_first_kernel"], SpecialVariables)
    self.assertTrue(hasattr(state_under_test, "params"))

  def test_initial_train_state(self):
    self._test_init_initial_state_driver(True)

  def test_initial_decode_state(self):
    self._test_init_initial_state_driver(False)


class MaxUtilsInitTransformerState(unittest.TestCase):
  """Tests initialization of transformer states in max_utils.py"""

  def setUp(self):
    # Conditionally set ici_fsdp_parallelism to match device count in decoupled mode
    extra_args = get_decoupled_parallelism_overrides()
    self.config = pyconfig.initialize([None, get_test_config_path()], enable_checkpointing=False, **extra_args)
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    quant = quantizations.configure_quantization(self.config)
    if self.config.pure_nnx:
      raise NotImplementedError("Pure NNX support has not been implemented yet.")
    else:
      self.model = models.transformer_as_linen(self.config, mesh=self.mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  def test_setup_decode_state(self):
    rng = random.PRNGKey(0)
    if self.config.pure_nnx:
      # NNX has a different function to init the training state.
      raise NotImplementedError("Pure NNX support has not been implemented yet.")
    else:
      init_state_fn = functools.partial(maxtext_utils.init_initial_state, self.model, None, self.config, False, rng)
    state, _ = maxtext_utils.setup_decode_state(self.config, self.mesh, None, init_state_fn)
    self.assertEqual(state.tx, None)
    self.assertEqual(state.opt_state, {})

  def test_setup_initial_state(self):
    rng = random.PRNGKey(0)
    tx = optax.adam(learning_rate=0.001)
    if self.config.pure_nnx:
      # NNX has a different function to init the training state.
      raise NotImplementedError("Pure NNX support has not been implemented yet.")
    else:
      init_state_fn = functools.partial(maxtext_utils.init_initial_state, self.model, tx, self.config, True, rng)
    state, _, _, _ = maxtext_utils.setup_initial_state(None, self.config, self.mesh, None, init_state_fn)
    self.assertEqual(state.tx, tx)
    self.assertNotEqual(state.opt_state, {})


class MaxUtilsPpAsDp(unittest.TestCase):
  """Tests logical_axis_rules_pp_act_as_dp converts rules so stage is added before data."""

  def test_stage_added_before_data(self):
    input_rules = (("activation_batch", ("data", "fsdp")),)
    expected_transform = (("activation_batch", ("stage", "data", "fsdp")),)
    transformed_rules = sharding.logical_axis_rules_pp_act_as_dp(input_rules)
    self.assertEqual(transformed_rules, expected_transform)

  def test_stage_removed(self):
    input_rules = (("layers", "stage"),)
    expected_transform = (
        (
            "layers",
            (),
        ),
    )
    transformed_rules = sharding.logical_axis_rules_pp_act_as_dp(input_rules)
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
    transformed_rules = sharding.logical_axis_rules_pp_act_as_dp(input_rules)
    self.assertEqual(transformed_rules, expected_transform)


class TestAssertParamsSufficientlySharded(unittest.TestCase):
  """Test suite for the sharding assertion utility function 'assert_params_sufficiently_sharded'."""

  def setUp(self):
    """Set up the test environment before each test method is run.

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
    """Tests that a 2D tensor fully sharded across both mesh axes passes the assertion."""
    # Define a sharding spec that shards the first tensor dimension by the 'fsdp' mesh axis
    # and the second dimension by the 'tensor' mesh axis.
    pspec = PartitionSpec("fsdp", "tensor")
    # Create a parameter and apply the sharding, ensuring it's distributed across all devices.
    params = {"layer1": jax.device_put(jnp.ones((8, 8)), NamedSharding(self.mesh, pspec))}

    # Assert that the parameters are sufficiently sharded; this should pass with no error.
    assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.1)

  def test_unsharded_fails(self):
    """Tests that a completely unsharded (fully replicated) parameter fails the assertion."""
    # Create a parameter without any sharding specification. It will be replicated on all devices.
    params = {"layer1": jnp.ones((8, 8))}

    # Expect an AssertionError because 100% of params are unsharded, exceeding the 10% tolerance.
    with self.assertRaises(AssertionError):
      assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.1)

  def test_mixed_sharding_fails(self):
    """Tests that a mix of sharded and unsharded parameters fails when the unsharded

    portion exceeds the tolerance.
    """
    sharded_param = jax.device_put(
        jnp.ones((8, 8)),
        NamedSharding(self.mesh, PartitionSpec("fsdp", "tensor")),
    )
    unsharded_param = jnp.ones((8, 8))
    params = {"layer1": sharded_param, "layer2": unsharded_param}

    with self.assertRaises(AssertionError):
      assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.5)

  def test_3d_tensor_sharded_on_fsdp_axis(self):
    """Tests that a 3D tensor sharded only on a valid target axis ('fsdp') should fail."""
    pspec = PartitionSpec("fsdp", None, None)
    params = {"conv3d_layer": jax.device_put(jnp.ones((8, 4, 4)), NamedSharding(self.mesh, pspec))}

    with self.assertRaises(AssertionError):
      assert_params_sufficiently_sharded(params, self.mesh, tolerance=0.2)

  def test_multi_axis_sharding_pass(self):
    """Tests that a tensor sharded with a valid axis ('fsdp') on a complex,

    multi-dimensional mesh passes the assertion.
    """
    # Create a mesh shape for a 5D mesh.
    devices = np.array(jax.devices()).reshape((jax.device_count(), 1, 1, 1, 1))
    mesh = Mesh(devices, self.mesh_axes)

    # Shard across multiple axes, including the valid 'fsdp' axis.
    pspec = PartitionSpec(("fsdp", "sequence"), "stage", "tensor", None)
    params = {"complex_layer": jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(mesh, pspec))}

    # This should pass because 'fsdp' is a valid sharding axis being used.
    assert_params_sufficiently_sharded(params, mesh, tolerance=0.05)

  def test_multi_axis_not_sharded_fails(self):
    """Tests that a tensor on a complex mesh fails if it's not sharded along any

    of the primary valid axes (like 'fsdp').
    """
    devices = np.array(jax.devices()).reshape((jax.device_count(), 1, 1, 1, 1))
    mesh = Mesh(devices, self.mesh_axes)
    pspec = PartitionSpec(("sequence", "context"), "stage", "tensor", None)
    params = {"complex_layer": jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(mesh, pspec))}

    with self.assertRaises(AssertionError):
      assert_params_sufficiently_sharded(params, mesh, tolerance=0.05)

  def test_multi_axis_mixed_sharding_fails(self):
    """Tests that a mix of sharded (correctly) and unsharded tensors on a complex mesh fails."""
    devices = np.array(jax.devices()).reshape((jax.device_count(), 1, 1, 1, 1))
    mesh = Mesh(devices, self.mesh_axes)
    sharded_pspec = PartitionSpec(("fsdp", "sequence"), "stage", "tensor", None)
    sharded_param = jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(mesh, sharded_pspec))
    unsharded_param = jnp.ones((8, 8, 2, 2))
    params = {
        "sharded_layer": sharded_param,
        "unsharded_layer": unsharded_param,
    }

    with self.assertRaises(AssertionError):
      assert_params_sufficiently_sharded(params, mesh, tolerance=0.5)


class TestAssert_Formatted_sharding_annotations(unittest.TestCase):
  """Test suite for sharding assertion formatting functions."""

  def setUp(self):
    """Set up the common 2*2 mesh for sharding tests."""
    if len(jax.devices()) < 4:
      self.skipTest("This test suite requires at least 4 TPU devices")

    self.mesh_axes = ("fsdp", "sequence", "tensor", "stage", "context")
    devices = np.array(jax.devices()).reshape((jax.device_count(), 1, 1, 1, 1))
    self.mesh = Mesh(devices, self.mesh_axes)

  def test_multi_axis_mixed_formating(self):
    """Tests a mix of sharded and unsharded tensors on a complex mesh fails."""
    sharded_pspec = PartitionSpec(("fsdp", "sequence"), "stage", "tensor", None)
    sharded_param = jax.device_put(jnp.ones((8, 8, 2, 2)), NamedSharding(self.mesh, sharded_pspec))
    unsharded_param = jnp.ones((8, 8, 2, 2))
    params = {
        "sharded_layer": sharded_param,
        "unsharded_layer": unsharded_param,
    }
    self.assertIsNotNone(get_formatted_sharding_annotations(params, self.mesh))


class TestPromptLogprobsFromPrefill(unittest.TestCase):
  """Test suite for the inference utility function 'prompt_logprobs_from_prefill'."""

  def test_shift_and_masking(self):
    # B=1, S=5, V=4
    B, S, V = 1, 5, 4
    # tokens: valid up to true_length=4 (positions 0..3); last token is padding
    input_tokens = jnp.array([[1, 2, 3, 1, 0]], dtype=jnp.int32)
    true_length = 4

    # logits predict t+1 at index t.
    # Make steps t=0,1,2 strongly favor the actual next token (input_tokens[:, t+1])
    logits = jnp.zeros((B, S, V), dtype=jnp.float32)
    logits = logits.at[0, 0, input_tokens[0, 1]].set(10.0)  # predicts token at pos 1
    logits = logits.at[0, 1, input_tokens[0, 2]].set(10.0)  # predicts token at pos 2
    logits = logits.at[0, 2, input_tokens[0, 3]].set(10.0)  # predicts token at pos 3
    # logits[:, 3, :] would predict token at pos 4 (padding); won't be used after masking.

    out = inference_utils.prompt_logprobs_from_prefill(logits, input_tokens, true_length)
    out_np = np.asarray(out)

    # pos 0 must be NaN (no previous token)
    self.assertTrue(np.isnan(out_np[0, 0]))
    # positions 1..3 should be ~0 (log prob ~0 for correct, very confident prediction)
    for pos in (1, 2, 3):
      self.assertTrue(np.isfinite(out_np[0, pos]))
      self.assertGreater(out_np[0, pos], -1e-3)  # close to 0

    # positions >= true_length (>=4) masked to NaN
    self.assertTrue(np.isnan(out_np[0, 4]))

  def test_true_length_one_all_nan(self):
    # Only a single valid token => no predictable positions
    input_tokens = jnp.array([[2, 1, 1]], dtype=jnp.int32)
    logits = jnp.zeros((1, 3, 5), dtype=jnp.float32)
    out = inference_utils.prompt_logprobs_from_prefill(logits, input_tokens, true_length=1)
    out_np = np.asarray(out)
    # All NaN (pos 0 NaN by definition; others masked by true_length)
    self.assertTrue(np.all(np.isnan(out_np)))


class TestPromptLogprobsFromPackedPrefill(unittest.TestCase):
  """Test suite for the inference utility function 'prompt_logprobs_from_packed_prefill'."""

  def test_respects_segments_and_masking(self):
    # Build a packed sequence of two prompts.
    # Global S=8, V=5
    B, S, V = 1, 8, 5

    # Segment 0: start=0, L0=4, positions 0..3
    # Segment 1: start=4, L1=3, positions 0..2 (position 3 is padding in this segment)
    start0, L0 = 0, 4
    start1, L1 = 4, 3

    # Tokens per segment (last token of seg1 padding at pos 7)
    toks = np.array([1, 2, 3, 1, 4, 0, 2, 0], dtype=np.int32)  # shape [S]
    input_tokens = jnp.asarray(toks)[None, :]  # [B, S]

    # decoder_positions within each segment
    pos0 = np.arange(0, L0)  # [0,1,2,3]
    pos1 = np.array([0, 1, 2, 3])  # last is padding for seg1
    decoder_positions = jnp.asarray(np.concatenate([pos0, pos1])[None, :])  # [B, S]

    # segment ids: 0 for first 4, 1 for next 4
    decoder_segment_ids = jnp.asarray(np.concatenate([np.zeros(L0), np.ones(4)]).astype(np.int32)[None, :])  # [B, S]

    # true lengths per prompt
    true_lengths = jnp.asarray([L0, L1], dtype=jnp.int32)  # [num_prompts=2]

    # Construct logits so that for each segment:
    # logits[:, step, :] strongly favors the *next* token inside the same segment.
    logits = jnp.zeros((B, S, V), dtype=jnp.float32)

    # Segment 0: steps 0..2 predict tokens at positions 1..3
    logits = logits.at[0, start0 + 0, toks[start0 + 1]].set(10.0)
    logits = logits.at[0, start0 + 1, toks[start0 + 2]].set(10.0)
    logits = logits.at[0, start0 + 2, toks[start0 + 3]].set(10.0)
    # Step 3 would predict pos 4 (start of next segment) — must NOT be scored for seg0.

    # Segment 1: steps 4..5 predict tokens at positions 5..6 (pos 7 is padding)
    logits = logits.at[0, start1 + 0, toks[start1 + 1]].set(10.0)
    logits = logits.at[0, start1 + 1, toks[start1 + 2]].set(10.0)
    # Step start1+2 would predict pos 7 (padding) — must NOT be scored for seg1.

    out = inference_utils.prompt_logprobs_from_packed_prefill(
        logits=logits,
        input_tokens=input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        true_lengths=true_lengths,
    )
    out_np = np.asarray(out)

    # Segment 0 checks:
    # pos 0 (seg0) NaN
    self.assertTrue(np.isnan(out_np[0, 0]))
    # pos 1..3 finite and ~0
    for p in (1, 2, 3):
      self.assertTrue(np.isfinite(out_np[0, p]))
      self.assertGreater(out_np[0, p], -1e-3)
    # position 4 is *start of seg1* (pos==0 in its segment) -> NaN
    self.assertTrue(np.isnan(out_np[0, 4]))

    # Segment 1 checks:
    # pos 5..6 finite and ~0 (valid positions 1..2 of seg1)
    for p in (5, 6):
      self.assertTrue(np.isfinite(out_np[0, p]))
      self.assertGreater(out_np[0, p], -1e-3)
    # pos 7 >= true_length of seg1 -> NaN
    self.assertTrue(np.isnan(out_np[0, 7]))


class TestSamplingFunctions(unittest.TestCase):
  """Test suite for sampling utility functions."""

  def setUp(self):
    """Set up common logits and RNG for tests."""
    self.rng = jax.random.PRNGKey(0)
    # Logits with a clear ranking for a vocabulary of 10
    self.logits = jnp.array([[0.1, 0.5, 0.2, 1.5, 0.8, 2.5, 0.3, 1.8, 0.7, 0.4]])
    self.expected_order = jnp.argsort(self.logits, axis=None, descending=True)

  def test_topk_filtering(self):
    """Tests that sampling is restricted to the top-k tokens."""
    topk = 3
    # The indices with the 3 highest logits
    top_k_indices = set(self.expected_order[:topk].tolist())
    rngs = jax.random.split(self.rng, 100)

    for r in rngs:
      token = inference_utils.sample_topk_topp_weighted(self.logits, topk=topk, nucleus_topp=1.0, temperature=1.0, rng=r)
      self.assertIn(token.item(), top_k_indices)

  def test_topp_filtering(self):
    """Tests that nucleus sampling (top-p) is correctly applied."""
    # With nucleus_topp=0.8, we expect the top 6 indices.
    nucleus_topp = 0.8
    top_p_indices = set(self.expected_order[:6].tolist())
    rngs = jax.random.split(self.rng, 100)

    for r in rngs:
      token = inference_utils.sample_topk_topp_weighted(
          self.logits,
          topk=10,
          nucleus_topp=nucleus_topp,
          temperature=1.0,
          rng=r,
      )
      self.assertIn(token.item(), top_p_indices)

  def test_combined_filtering(self):
    """Tests the combination of top-k and top-p filtering."""
    # First, filter to top_k=5.
    # Then, apply nucleus_topp=0.8 to this smaller set.
    # The renormalized probabilities lead to a nucleus of the top 3 tokens.
    topk = 5
    nucleus_topp = 0.8
    valid_indices = set(self.expected_order[:3].tolist())
    rngs = jax.random.split(self.rng, 100)

    for r in rngs:
      token = inference_utils.sample_topk_topp_weighted(
          self.logits,
          topk=topk,
          nucleus_topp=nucleus_topp,
          temperature=1.0,
          rng=r,
      )
      self.assertIn(token.item(), valid_indices)

  def test_low_temperature_is_greedy(self):
    """Tests that a very low temperature results in greedy sampling."""
    low_temp = 1e-6
    greedy_token_index = self.expected_order[0]  # Index of the highest logit
    rngs = jax.random.split(self.rng, 10)

    for r in rngs:
      token = inference_utils.sample_topk_topp_weighted(
          self.logits, topk=10, nucleus_topp=1.0, temperature=low_temp, rng=r
      )
      self.assertEqual(token.item(), greedy_token_index)

  def test_invalid_args_raise_error(self):
    """Tests that invalid arguments for topk and nucleus_topp raise errors."""
    with self.assertRaises(ValueError):
      inference_utils.sample_topk_topp_weighted(self.logits, topk=0, nucleus_topp=1.0, temperature=1.0, rng=self.rng)
    with self.assertRaises(ValueError):
      inference_utils.sample_topk_topp_weighted(self.logits, topk=-1, nucleus_topp=1.0, temperature=1.0, rng=self.rng)
    with self.assertRaises(ValueError):
      inference_utils.sample_topk_topp_weighted(self.logits, topk=10, nucleus_topp=0.0, temperature=1.0, rng=self.rng)
    with self.assertRaises(ValueError):
      inference_utils.sample_topk_topp_weighted(self.logits, topk=10, nucleus_topp=1.1, temperature=1.0, rng=self.rng)

  def test_batch_dimension(self):
    """Tests that the function handles a batch of logits correctly."""
    batched_logits = jnp.vstack([self.logits, self.logits, self.logits])
    batch_size = batched_logits.shape[0]
    topk = 3
    top_k_indices = set(self.expected_order[:topk].tolist())
    rngs = jax.random.split(self.rng, batch_size)

    # CORRECTED: Use vmap to handle batching for JAX's random functions.
    # We map over the first axis (0) of logits and rngs.
    # Other arguments (topk, nucleus_topp, temperature) are fixed (None).
    vmapped_sample = vmap(
        inference_utils.sample_topk_topp_weighted,
        in_axes=(0, None, None, None, 0),
        out_axes=0,
    )

    tokens = vmapped_sample(batched_logits, topk, 1.0, 1.0, rngs)

    self.assertEqual(tokens.shape, (batch_size,))
    for token in tokens:
      self.assertIn(token.item(), top_k_indices)


class TestCalculateBytesFromPytree(unittest.TestCase):
  """Test suite for the byte calculation utility function."""

  def test_bytes_from_pytree_arrays(self):
    """Tests byte calculation with standard JAX and NumPy arrays."""
    params = {
        "a": jnp.zeros((2, 3), jnp.float32),  # 2 * 3 * 4 = 24 bytes
        "b": np.zeros((5,), np.int32),  # 5 * 4 = 20 bytes
    }
    expected_total_bytes = 44
    self.assertEqual(max_utils.calculate_bytes_from_pytree(params), expected_total_bytes)

  def test_bytes_from_pytree_shape_dtype_struct(self):
    """Tests byte calculation with a ShapeDtypeStruct."""
    s = jax.ShapeDtypeStruct(shape=(7, 11), dtype=jnp.bfloat16)
    params = {"s": s}
    # 7 * 11 * 2 (bfloat16 size) = 154 bytes
    expected_total_bytes = 154
    self.assertEqual(max_utils.calculate_bytes_from_pytree(params), expected_total_bytes)

  def test_bytes_from_pytree_mixed_and_none(self):
    """Tests a heterogeneous pytree with mixed types including None and scalars."""
    params = {
        "a": None,  # 0 bytes
        "b": 3,  # 8 bytes (int64)
        "c": 1.0,  # 8 bytes (float64)
        "d": jax.ShapeDtypeStruct((4,), jnp.int8),  # 4 * 1 = 4 bytes
    }
    expected_total_bytes = 20
    self.assertEqual(max_utils.calculate_bytes_from_pytree(params), expected_total_bytes)

  def test_bytes_from_pytree_empty_dict(self):
    """Tests that an empty pytree correctly returns 0 bytes."""
    self.assertEqual(max_utils.calculate_bytes_from_pytree({}), 0)


class TestLearningRateSchedules(unittest.TestCase):
  """Test suite for learning rate schedule functions."""

  def test_cosine_schedule(self):
    """Tests cosine learning rate schedule."""
    learning_rate = 1e-3
    learning_rate_schedule_steps = 1000
    steps = 1200
    warmup_steps_fraction = 0.1
    learning_rate_final_fraction = 0.1

    warmup_steps = int(learning_rate_schedule_steps * warmup_steps_fraction)

    config = pyconfig.initialize(
        [None, get_test_config_path()],
        enable_checkpointing=False,
        learning_rate=learning_rate,
        learning_rate_schedule_steps=learning_rate_schedule_steps,
        steps=steps,
        warmup_steps_fraction=warmup_steps_fraction,
        lr_schedule_type="cosine",
        learning_rate_final_fraction=learning_rate_final_fraction,
    )

    schedule_fn = maxtext_utils.create_learning_rate_schedule(config)

    # Warmup phase: 0 -> peak
    self.assertAlmostEqual(float(schedule_fn(0)), 0.0, places=6)
    self.assertAlmostEqual(float(schedule_fn(warmup_steps)), learning_rate, places=6)
    # Ensure delta is constant
    expected_slope = learning_rate / warmup_steps
    for i in range(1, warmup_steps + 1):
      current_lr = float(schedule_fn(i))
      self.assertAlmostEqual(current_lr - float(schedule_fn(i - 1)), expected_slope, places=6)

    # Cosine decay phase
    lr_end = schedule_fn(learning_rate_schedule_steps - 1)
    expected_final = learning_rate * learning_rate_final_fraction
    self.assertLess(float(lr_end), learning_rate)
    self.assertAlmostEqual(float(lr_end), expected_final, places=6)

    # Zero phase
    self.assertAlmostEqual(float(schedule_fn(steps - 1)), 0.0, places=6)

  def test_wsd_schedule(self):
    """Tests WSD learning rate schedule with both linear and cosine decay styles."""
    learning_rate = 1e-3
    learning_rate_schedule_steps = 1000
    steps = 1200
    warmup_steps_fraction = 0.1
    learning_rate_final_fraction = 0.1
    wsd_decay_steps_fraction = 0.1

    warmup_steps = int(learning_rate_schedule_steps * warmup_steps_fraction)
    decay_steps = int(learning_rate_schedule_steps * wsd_decay_steps_fraction)
    stable_steps = learning_rate_schedule_steps - warmup_steps - decay_steps
    decay_start = warmup_steps + stable_steps

    # Test both decay styles: linear and cosine
    for decay_style in ["linear", "cosine"]:
      config = pyconfig.initialize(
          [None, get_test_config_path()],
          enable_checkpointing=False,
          learning_rate=learning_rate,
          learning_rate_schedule_steps=learning_rate_schedule_steps,
          steps=steps,
          warmup_steps_fraction=warmup_steps_fraction,
          lr_schedule_type="wsd",
          learning_rate_final_fraction=learning_rate_final_fraction,
          wsd_decay_steps_fraction=wsd_decay_steps_fraction,
          wsd_decay_style=decay_style,
      )
      schedule_fn = maxtext_utils.create_learning_rate_schedule(config)

      # Warmup phase: 0 -> peak
      self.assertAlmostEqual(float(schedule_fn(0)), 0.0, places=6)
      self.assertAlmostEqual(float(schedule_fn(warmup_steps)), learning_rate, places=6)
      # Ensure delta is constant
      expected_slope = learning_rate / warmup_steps
      for i in range(1, warmup_steps + 1):
        current_lr = float(schedule_fn(i))
        self.assertAlmostEqual(current_lr - float(schedule_fn(i - 1)), expected_slope, places=6)

      # Stable phase: constant at peak
      self.assertAlmostEqual(float(schedule_fn(warmup_steps + 10)), learning_rate, places=6)
      self.assertAlmostEqual(
          float(schedule_fn(warmup_steps + stable_steps // 2)),
          learning_rate,
          places=6,
      )
      self.assertAlmostEqual(float(schedule_fn(decay_start - 1)), learning_rate, places=6)

      # Decay phase: peak -> final
      lr_mid_decay = schedule_fn(decay_start + decay_steps // 2)
      expected_final = learning_rate * learning_rate_final_fraction
      self.assertLess(float(lr_mid_decay), learning_rate)
      self.assertGreater(float(lr_mid_decay), expected_final)

      # End of decay phase: should reach expected_final
      lr_end = schedule_fn(learning_rate_schedule_steps - 1)
      self.assertAlmostEqual(float(lr_end), expected_final, places=6)

      # Zero phase
      self.assertAlmostEqual(float(schedule_fn(steps - 1)), 0.0, places=6)

    # Test invalid fractions - should raise during config initialization
    with self.assertRaises(ValueError) as cm:
      pyconfig.initialize(
          [None, get_test_config_path()],
          enable_checkpointing=False,
          learning_rate=learning_rate,
          learning_rate_schedule_steps=learning_rate_schedule_steps,
          steps=steps,
          warmup_steps_fraction=0.6,
          lr_schedule_type="wsd",
          learning_rate_final_fraction=learning_rate_final_fraction,
          wsd_decay_steps_fraction=0.5,  # Sum > 1.0
      )
    self.assertIn("warmup_steps_fraction", str(cm.exception))
    self.assertIn("wsd_decay_steps_fraction", str(cm.exception))


class TestMeshUtils(unittest.TestCase):
  """Test suite for the mesh creation utility function."""

  @dataclass
  class MockConfig:
    """Minimal mock for pyconfig.HyperParameters."""

    init_weights_seed: int = 42
    shard_mode: str = ShardMode.EXPLICIT
    mesh_axes: Sequence[str] = field(default_factory=lambda: ["data", "model"])

  def setUp(self):
    # Setup a dummy device array for the mock to return
    self.devices_array = np.array(jax.devices())

  @patch("MaxText.maxtext_utils.create_device_mesh")
  def test_get_mesh_explicit_mode(self, mock_create_device_mesh):
    """Tests that ShardMode.EXPLICIT sets axis_types to MANUAL."""
    # 1. Setup Mock
    mock_create_device_mesh.return_value = self.devices_array[:1].reshape((1,))
    config = self.MockConfig(shard_mode=ShardMode.EXPLICIT, mesh_axes=["data"])

    # 2. Run function
    mesh = maxtext_utils.get_mesh_from_config(config)

    # 3. Assertions
    # Check that the internal utility was called correctly
    mock_create_device_mesh.assert_called_once_with(config, None)

    # Verify Mesh properties
    self.assertEqual(mesh.axis_names, ("data",))
    # In JAX, AxisType.MANUAL is the equivalent for explicit control
    self.assertEqual(mesh.axis_types, (AxisType.Explicit,))

  @patch("MaxText.maxtext_utils.create_device_mesh")
  def test_get_mesh_auto_mode(self, mock_create_device_mesh):
    """Tests that ShardMode.AUTO sets axis_types to AUTO."""
    # 1. Setup Mock
    mock_create_device_mesh.return_value = self.devices_array[:2].reshape((2, 1))
    config = self.MockConfig(shard_mode=ShardMode.AUTO, mesh_axes=["data", "model"])

    # 2. Run function
    mesh = maxtext_utils.get_mesh_from_config(config)

    # 3. Assertions
    self.assertEqual(len(mesh.axis_types), 2)
    self.assertTrue(all(t == AxisType.Auto for t in mesh.axis_types))

  @patch("MaxText.maxtext_utils.create_device_mesh")
  def test_get_mesh_with_provided_devices(self, mock_create_device_mesh):
    """Tests that provided devices are passed through to the mesh creator."""
    config = self.MockConfig()
    specific_devices = self.devices_array[:2].reshape((1, 2))
    mock_create_device_mesh.return_value = specific_devices

    _ = maxtext_utils.get_mesh_from_config(config, devices=specific_devices)

    # Verify the second argument to create_device_mesh was our device list
    mock_create_device_mesh.assert_called_once_with(config, specific_devices)


class TestGetFunctionalTrainWithSignature(unittest.TestCase):
  """Tests for get_functional_train_with_signature."""

  def _make_mock_step(self):
    def train_step(_model, _config, _state_shardings, _params_shardings, state, _batch, _rng=None):
      return state, {}

    return train_step

  def test_returns_five_tuple(self):
    step = self._make_mock_step()
    result = maxtext_utils.get_functional_train_with_signature(
        step, "data_sharding", "state_shardings", "model", "config"
    )
    self.assertEqual(len(result), 5)

  def test_functional_train_has_correct_name(self):
    step = self._make_mock_step()
    fn, _, _, _, _ = maxtext_utils.get_functional_train_with_signature(
        step, "data_sharding", "state_shardings", "model", "config"
    )
    self.assertEqual(fn.__name__, "train_step")

  def test_in_shardings_structure(self):
    step = self._make_mock_step()
    _, in_shardings, _, _, _ = maxtext_utils.get_functional_train_with_signature(
        step, "data_sharding", "state_shardings", "model", "config"
    )
    # (state, batch, rng)
    self.assertEqual(len(in_shardings), 3)
    self.assertIsNone(in_shardings[2])  # rng sharding is None

  def test_donate_argnums_is_zero(self):
    step = self._make_mock_step()
    _, _, _, _, donate_argnums = maxtext_utils.get_functional_train_with_signature(
        step, "data_sharding", "state_shardings", "model", "config"
    )
    self.assertEqual(donate_argnums, 0)

  def test_functional_train_is_partial(self):
    """functional_train should partially apply model and config."""
    received = {}

    def train_step(model, config, _state_shardings, _params_shardings, state, _batch, _rng=None):
      received["model"] = model
      received["config"] = config
      return state, {}

    fn, _, _, _, _ = maxtext_utils.get_functional_train_with_signature(train_step, "ds", "ss", "my_model", "my_config")
    fn("state", "batch")
    self.assertEqual(received["model"], "my_model")
    self.assertEqual(received["config"], "my_config")


class TestGetFunctionalEvalWithSignature(unittest.TestCase):
  """Tests for get_functional_eval_with_signature."""

  def _make_mock_eval_step(self):
    def eval_step(_model, _config, _state, _batch, _rng=None):
      return {}

    return eval_step

  def test_returns_five_tuple(self):
    step = self._make_mock_eval_step()
    result = maxtext_utils.get_functional_eval_with_signature(step, "ds", "ss", "model", "config")
    self.assertEqual(len(result), 5)

  def test_functional_eval_has_correct_name(self):
    step = self._make_mock_eval_step()
    fn, _, _, _, _ = maxtext_utils.get_functional_eval_with_signature(step, "ds", "ss", "model", "config")
    self.assertEqual(fn.__name__, "eval_step")

  def test_out_shardings_is_none(self):
    step = self._make_mock_eval_step()
    _, _, out_shardings, _, _ = maxtext_utils.get_functional_eval_with_signature(step, "ds", "ss", "model", "config")
    self.assertIsNone(out_shardings)

  def test_donate_argnums_is_empty(self):
    step = self._make_mock_eval_step()
    _, _, _, _, donate_argnums = maxtext_utils.get_functional_eval_with_signature(step, "ds", "ss", "model", "config")
    self.assertEqual(donate_argnums, ())


class TestGetShapedBatch(unittest.TestCase):
  """Tests for get_shaped_batch."""

  def _make_cfg(self, *, enable_diloco=False, use_multimodal=False, use_audio=False):
    cfg = MagicMock()
    cfg.enable_diloco = enable_diloco
    cfg.global_batch_size_to_load = 4
    cfg.max_target_length = 16
    cfg.use_multimodal = use_multimodal
    cfg.use_audio = use_audio
    if enable_diloco:
      cfg.num_diloco_replicas = 2
    return cfg

  def test_standard_keys_present(self):
    batch = maxtext_utils.get_shaped_batch(self._make_cfg())
    for key in (
        "inputs",
        "inputs_position",
        "inputs_segmentation",
        "targets",
        "targets_position",
        "targets_segmentation",
    ):
      self.assertIn(key, batch)

  def test_standard_shape(self):
    cfg = self._make_cfg()
    batch = maxtext_utils.get_shaped_batch(cfg)
    expected_shape = (cfg.global_batch_size_to_load, cfg.max_target_length)
    self.assertEqual(batch["inputs"].shape, expected_shape)

  def test_diloco_shape(self):
    cfg = self._make_cfg(enable_diloco=True)
    batch = maxtext_utils.get_shaped_batch(cfg)
    expected_shape = (
        cfg.num_diloco_replicas,
        cfg.global_batch_size_to_load // cfg.num_diloco_replicas,
        cfg.max_target_length,
    )
    self.assertEqual(batch["inputs"].shape, expected_shape)

  def test_no_image_key_without_multimodal(self):
    batch = maxtext_utils.get_shaped_batch(self._make_cfg(use_multimodal=False))
    self.assertNotIn("images", batch)

  def test_no_audio_key_without_audio(self):
    batch = maxtext_utils.get_shaped_batch(self._make_cfg(use_audio=False))
    self.assertNotIn("audios", batch)

  def test_all_values_are_shape_dtype_struct(self):
    batch = maxtext_utils.get_shaped_batch(self._make_cfg())
    for v in batch.values():
      self.assertIsInstance(v, jax.ShapeDtypeStruct)


class TestShouldPreventCseInRemat(unittest.TestCase):
  """Tests for should_prevent_cse_in_remat."""

  def _make_cfg(self, scan_layers=False, gradient_accumulation_steps=1, hardware="tpu"):
    cfg = MagicMock()
    cfg.scan_layers = scan_layers
    cfg.gradient_accumulation_steps = gradient_accumulation_steps
    cfg.hardware = hardware
    return cfg

  def test_scan_layers_returns_false(self):
    self.assertFalse(maxtext_utils.should_prevent_cse_in_remat(self._make_cfg(scan_layers=True)))

  def test_gpu_with_grad_accum_returns_false(self):
    cfg = self._make_cfg(scan_layers=False, gradient_accumulation_steps=4, hardware="gpu")
    self.assertFalse(maxtext_utils.should_prevent_cse_in_remat(cfg))

  def test_gpu_multiprocess_with_grad_accum_returns_false(self):
    cfg = self._make_cfg(scan_layers=False, gradient_accumulation_steps=4, hardware="gpu_multiprocess")
    self.assertFalse(maxtext_utils.should_prevent_cse_in_remat(cfg))

  def test_tpu_with_grad_accum_returns_true(self):
    cfg = self._make_cfg(scan_layers=False, gradient_accumulation_steps=4, hardware="tpu")
    self.assertTrue(maxtext_utils.should_prevent_cse_in_remat(cfg))

  def test_default_case_returns_true(self):
    self.assertTrue(maxtext_utils.should_prevent_cse_in_remat(self._make_cfg()))


class TestCalculateTokensTrainingPerDevice(unittest.TestCase):
  """Tests for calculate_tokens_training_per_device."""

  def test_basic_calculation(self):
    cfg = MagicMock()
    cfg.max_target_length = 128
    cfg.per_device_batch_size = 2
    cfg.gradient_accumulation_steps = 4
    result = maxtext_utils.calculate_tokens_training_per_device(cfg)
    self.assertEqual(result, 128 * 2 * 4)


class TestCalculateIndexerMaskRatio(unittest.TestCase):
  """Tests for calculate_indexer_mask_ratio."""

  def test_half_topk(self):
    # K=T/2: ratio=0.5, mask = 0.5 - 0.5*0.25 = 0.375
    result = maxtext_utils.calculate_indexer_mask_ratio(indexer_topk=4, max_target_length=8)
    self.assertAlmostEqual(result, 0.375, places=6)

  def test_full_topk_equals_dense(self):
    # K=T: ratio=1, mask = 1 - 0.5 = 0.5 (same as causal)
    result = maxtext_utils.calculate_indexer_mask_ratio(indexer_topk=8, max_target_length=8)
    self.assertAlmostEqual(result, 0.5, places=6)

  def test_small_topk(self):
    # K=1, T=100: ratio=0.01, mask ≈ 0.01 - 0.5*0.0001 ≈ 0.00995
    result = maxtext_utils.calculate_indexer_mask_ratio(indexer_topk=1, max_target_length=100)
    expected = 0.01 - 0.5 * (0.01**2)
    self.assertAlmostEqual(result, expected, places=8)


class TestCalculateFfnMatmulTflops(unittest.TestCase):
  """Tests for calculate_ffn_mamtul_tflops_per_device."""

  def _make_cfg(self, num_activations=2):
    cfg = MagicMock()
    cfg.per_device_batch_size = 1
    cfg.max_target_length = 64
    cfg.emb_dim = 512
    cfg.mlp_activations = ["silu"] * num_activations
    return cfg

  def test_total_flops_positive(self):
    result = maxtext_utils.calculate_ffn_mamtul_tflops_per_device(self._make_cfg(), mlp_dim=2048)
    self.assertGreater(result, 0)

  def test_scales_with_mlp_dim(self):
    cfg = self._make_cfg()
    small = maxtext_utils.calculate_ffn_mamtul_tflops_per_device(cfg, mlp_dim=1024)
    large = maxtext_utils.calculate_ffn_mamtul_tflops_per_device(cfg, mlp_dim=4096)
    self.assertGreater(large, small)

  def test_single_activation(self):
    """With one activation, ffn1 uses 1x mlp_dim."""
    cfg = self._make_cfg(num_activations=1)
    result = maxtext_utils.calculate_ffn_mamtul_tflops_per_device(cfg, mlp_dim=2048)
    expected_ffn1 = 2 * 1 * 64 * 2048 * 512 * 1
    expected_ffn2 = 2 * 1 * 64 * 2048 * 512
    self.assertEqual(result, expected_ffn1 + expected_ffn2)


class TestGetDenseMoeLayers(unittest.TestCase):
  """Tests for get_dense_moe_layers."""

  def _make_cfg(self, decoder_block, num_decoder_layers=32, first_num_dense_layers=3, interleave_moe_layer_step=4):
    cfg = MagicMock()
    cfg.decoder_block = decoder_block
    cfg.num_decoder_layers = num_decoder_layers
    cfg.first_num_dense_layers = first_num_dense_layers
    cfg.interleave_moe_layer_step = interleave_moe_layer_step
    return cfg

  def test_deepseek_block(self):
    cfg = self._make_cfg(DecoderBlockType.DEEPSEEK, num_decoder_layers=32, first_num_dense_layers=3)
    dense, moe = maxtext_utils.get_dense_moe_layers(cfg)
    self.assertEqual(dense, 3)
    self.assertEqual(moe, 29)

  def test_llama4_block(self):
    cfg = self._make_cfg(DecoderBlockType.LLAMA4, num_decoder_layers=16, interleave_moe_layer_step=4)
    dense, moe = maxtext_utils.get_dense_moe_layers(cfg)
    self.assertEqual(moe, 4)  # 16 // 4
    self.assertEqual(dense, 12)  # 16 - 4

  def test_qwen3_next_block(self):
    cfg = self._make_cfg(DecoderBlockType.QWEN3_NEXT, num_decoder_layers=8)
    dense, moe = maxtext_utils.get_dense_moe_layers(cfg)
    self.assertEqual(dense, 0)
    self.assertEqual(moe, 8)

  def test_unsupported_block_raises(self):
    cfg = self._make_cfg(DecoderBlockType.DEFAULT)
    with self.assertRaises(ValueError):
      maxtext_utils.get_dense_moe_layers(cfg)


class TestCalculatePrefillTflops(unittest.TestCase):
  """Tests for calculate_prefill_tflops_per_device."""

  def _make_cfg(self, num_query_heads=8, num_decoder_layers=2, head_dim=64):
    cfg = MagicMock()
    cfg.num_query_heads = num_query_heads
    cfg.num_decoder_layers = num_decoder_layers
    cfg.head_dim = head_dim
    return cfg

  def test_returns_three_positive_values(self):
    cfg = self._make_cfg()
    total, lw, attn = maxtext_utils.calculate_prefill_tflops_per_device(
        num_model_parameters=1_000_000, prefill_length=128, config=cfg, log=False
    )
    self.assertGreater(total, 0)
    self.assertGreater(lw, 0)
    self.assertGreater(attn, 0)

  def test_total_is_sum_of_parts(self):
    cfg = self._make_cfg()
    total, lw, attn = maxtext_utils.calculate_prefill_tflops_per_device(
        num_model_parameters=500_000, prefill_length=64, config=cfg, log=False
    )
    self.assertAlmostEqual(total, lw + attn, places=10)

  def test_scales_with_prefill_length(self):
    cfg = self._make_cfg()
    _, _, attn_short = maxtext_utils.calculate_prefill_tflops_per_device(
        num_model_parameters=1_000_000, prefill_length=64, config=cfg, log=False
    )
    _, _, attn_long = maxtext_utils.calculate_prefill_tflops_per_device(
        num_model_parameters=1_000_000, prefill_length=128, config=cfg, log=False
    )
    # Attention scales quadratically with prefill length
    self.assertGreater(attn_long, attn_short * 3)


class TestSetupTrainingState(unittest.TestCase):
  """Tests for setup_training_state (thin wrapper over setup_initial_state)."""

  def setUp(self):
    extra_args = get_decoupled_parallelism_overrides()
    self.config = pyconfig.initialize([None, get_test_config_path()], enable_checkpointing=False, **extra_args)
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    quant = quantizations.configure_quantization(self.config)
    if self.config.pure_nnx:
      raise NotImplementedError("Pure NNX path not covered by this test.")
    self.model = Transformer(self.config, mesh=self.mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  def test_setup_training_state_returns_train_state(self):
    rng = jax.random.PRNGKey(0)
    tx = optax.adam(learning_rate=0.001)
    init_state_fn = functools.partial(maxtext_utils.init_initial_state, self.model, tx, self.config, True, rng)
    state, _, _, _ = maxtext_utils.setup_training_state(None, self.config, self.mesh, None, init_state_fn)
    self.assertEqual(state.tx, tx)
    self.assertNotEqual(state.opt_state, {})


class TestGetLogicalAnnotations(unittest.TestCase):
  """Tests for get_logical_annotations."""

  def setUp(self):
    extra_args = get_decoupled_parallelism_overrides()
    self.config = pyconfig.initialize([None, get_test_config_path()], enable_checkpointing=False, **extra_args)
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    quant = quantizations.configure_quantization(self.config)
    if self.config.pure_nnx:
      raise NotImplementedError("Pure NNX path not covered by this test.")
    self.model = Transformer(self.config, mesh=self.mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    self.rng = jax.random.PRNGKey(0)
    self.tx = optax.adam(learning_rate=0.001)

  def test_returns_partition_spec_tree(self):
    init_state_fn = functools.partial(maxtext_utils.init_initial_state, self.model, self.tx, self.config, True, self.rng)
    annotations = maxtext_utils.get_logical_annotations(self.config, self.mesh, init_state_fn)
    # Result should be a pytree with PartitionSpec leaves
    leaves = jax.tree_util.tree_leaves(annotations)
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      self.assertIsInstance(leaf, PartitionSpec)


class TestSaveQuantizedCheckpoint(unittest.TestCase):
  """Tests for save_quantized_checkpoint_if_configured."""

  def test_raises_when_no_quantization(self):
    cfg = MagicMock()
    cfg.quantization = ""
    with self.assertRaises(AssertionError):
      maxtext_utils.save_quantized_checkpoint_if_configured(cfg, params={})

  @unittest.mock.patch("maxtext.utils.maxtext_utils.checkpointing")
  def test_skips_save_when_path_empty(self, mock_ckpt):
    cfg = MagicMock()
    cfg.quantization = "int8"
    cfg.save_quantized_params_path = ""
    maxtext_utils.save_quantized_checkpoint_if_configured(cfg, params={})
    mock_ckpt.save_params_to_path.assert_not_called()

  @unittest.mock.patch("maxtext.utils.maxtext_utils.checkpointing")
  def test_calls_save_when_path_set(self, mock_ckpt):
    cfg = MagicMock()
    cfg.quantization = "int8"
    cfg.save_quantized_params_path = "/tmp/quantized"
    cfg.checkpoint_storage_use_ocdbt = True
    cfg.checkpoint_storage_use_zarr3 = True
    maxtext_utils.save_quantized_checkpoint_if_configured(cfg, params={"w": jnp.ones((2,))})
    mock_ckpt.save_params_to_path.assert_called_once()


class TestAddConfigToSummaryWriter(unittest.TestCase):
  """Tests for add_config_to_summary_writer."""

  def test_calls_add_text_for_each_key(self):
    cfg = MagicMock()
    cfg.get_keys.return_value = {"learning_rate": 0.001, "steps": 100}
    mock_writer = MagicMock()

    with unittest.mock.patch("maxtext.utils.max_utils.add_text_to_summary_writer") as mock_add:
      maxtext_utils.add_config_to_summary_writer(cfg, mock_writer)
      # Should have been called once per config key (process_index==0 in tests)
      if jax.process_index() == 0:
        self.assertEqual(mock_add.call_count, 2)


class TestMaybeDumpJaxpr(unittest.TestCase):
  """Tests for maybe_dump_jaxpr."""

  def test_early_return_when_disabled(self):
    cfg = MagicMock()
    cfg.dump_jaxpr = False
    # Should return immediately without calling any JAX tracing (no exception raised)
    maxtext_utils.maybe_dump_jaxpr(cfg, p_train_step=None, train_step_inputs=None)


class TestPrintShardingsParams(unittest.TestCase):
  """Tests for print_shardings_params — normalization branches."""

  def setUp(self):
    """Build a minimal mesh and sharded param for testing."""
    self.mesh = Mesh(np.array(jax.devices()), ("data",))

  def _make_simple_params(self):
    """Return (params, param_sharding, logical) without a .params attribute."""
    params = {"w": jnp.ones((4,))}
    param_sharding = {"w": NamedSharding(self.mesh, PartitionSpec(None))}
    logical = {"w": PartitionSpec(None)}
    return params, param_sharding, logical

  def test_runs_without_error_dict_inputs(self):
    """print_shardings_params should not raise with plain dict inputs."""
    params, param_sharding, logical = self._make_simple_params()
    # Should complete without raising
    maxtext_utils.print_shardings_params(params, param_sharding, logical)

  def test_runs_without_logical_annotations(self):
    """logical_annotations=None should be handled (no logical column)."""
    params, param_sharding, _ = self._make_simple_params()
    maxtext_utils.print_shardings_params(params, param_sharding, mesh=self.mesh, logical_annotations=None)


if __name__ == "__main__":
  unittest.main()
