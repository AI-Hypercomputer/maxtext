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

""" Tests for the common MaxText utilities """

from typing import Any
from collections.abc import Callable
import os.path
import unittest

from jax import random, vmap
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
from MaxText import inference_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.maxtext_utils import assert_params_sufficiently_sharded, get_formatted_sharding_annotations

Transformer = models.transformer_as_linen


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
        max_utils.calculate_num_params_from_pytree(state.params), max_utils.calculate_num_params_from_pytree(self.params)
    )


class ModelWithMultipleCollections(nn.Module):
  """
  A simple model that has variables in multiple collections - "params" and "special_variables"
  """

  dense: Dense = nn.Dense(4)

  def setup(self):
    self.kernel = self.variable("special_variables", "my_first_kernel", lambda: jnp.ones((4, 5)))

  def __call__(self, x, y, encoder_images=None, nnx_method=None):
    x = self.dense(x)
    x = x @ self.kernel.value
    return x


class MaxUtilsInitStateWithMultipleCollections(unittest.TestCase):
  """test class for multiple collection state in maxutils"""

  def setUp(self):
    self.config = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")], enable_checkpointing=False
    )
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
    self.config = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")], enable_checkpointing=False
    )
    devices_array = maxtext_utils.create_device_mesh(self.config)
    self.mesh = Mesh(devices_array, self.config.mesh_axes)
    quant = quantizations.configure_quantization(self.config)
    self.model = Transformer(self.config, mesh=self.mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

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
  Test suite for sharding assertion formatting functions.
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


class TestPromptLogprobsFromPrefill(unittest.TestCase):
  """
  Test suite for the inference utility function 'prompt_logprobs_from_prefill'.
  """

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
  """
  Test suite for the inference utility function 'prompt_logprobs_from_packed_prefill'.
  """

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
          self.logits, topk=10, nucleus_topp=nucleus_topp, temperature=1.0, rng=r
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
          self.logits, topk=topk, nucleus_topp=nucleus_topp, temperature=1.0, rng=r
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
    vmapped_sample = vmap(inference_utils.sample_topk_topp_weighted, in_axes=(0, None, None, None, 0), out_axes=0)

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


if __name__ == "__main__":
  unittest.main()
