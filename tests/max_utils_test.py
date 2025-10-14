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

""" Tests for the common Max Utils """
import os
import sys
import unittest
import time
import pytest

import jax
from jax import numpy as jnp
from jax import random

from flax import linen as nn

import optax

from MaxText import max_utils
from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.train_utils import setup_train_loop


class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""

  def test_l2norm_pytree(self):
    x = {"a": jax.numpy.array([0, 2, 0]), "b": jax.numpy.array([0, 3, 6])}
    pytree_l2_norm = max_utils.l2norm_pytree(x)
    self.assertTrue(jax.numpy.allclose(pytree_l2_norm, 7, rtol=1e-05, atol=1e-08, equal_nan=False))


class MaxUtilsPytree(unittest.TestCase):
  """Tests initialization of training and decode states in max_utils.py"""

  def setUp(self):
    self.model = nn.Dense(features=5)
    self.key1, self.key2 = random.split(random.key(0))
    self.input = random.normal(self.key1, (10,))  # Dummy input data
    self.params = self.model.init(self.key2, self.input)

  def test_calculate_num_params_from_pytree(self):
    example_tree = [
        [1, "a", object()],
        (1, (2, 3), ()),
        [1, {"k1": 2, "k2": (3, 4)}, 5],
        {"a": 2, "b": (2, 3)},
        jnp.array([1, 2, 3]),
    ]
    self.assertEqual(max_utils.calculate_num_params_from_pytree(example_tree), 17)
    # Model params
    self.assertEqual(max_utils.calculate_num_params_from_pytree(self.params), 55)


class MaxUtilsT5XCrossEntropy(unittest.TestCase):
  """Tests for the cross entropy functions in max_utils.py"""

  def test_t5x_cross_entropy(self):
    # Generate random targets and logits
    key = jax.random.PRNGKey(0)
    targets = jax.random.randint(key, shape=(48, 2048), dtype=jax.numpy.int32, minval=1, maxval=10)
    logits = jax.random.uniform(key, shape=(48, 2048, 4096), dtype=jax.numpy.float32)

    # Calculate xent from optax implementation
    optax_xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    # Calculate xent from custom T5X implementation
    one_hot_targets = jax.nn.one_hot(targets, 4096)
    t5x_xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    t5x_xent = nn.with_logical_constraint(t5x_xent, ("activation_batch", "activation_length"))

    # Compare results
    self.assertTrue(jax.numpy.allclose(optax_xent, t5x_xent, rtol=1e-05, atol=1e-08, equal_nan=False))


class MaxUtilsCustomMesh(unittest.TestCase):
  """Tests for the is_valid_custom_mesh function in max_utils.py"""

  def test_empty_value(self):
    self.assertFalse(max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 64, 4, 1], ""))

  def test_valid_64x4(self):
    self.assertTrue(max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 64, 4, 1], "hybrid_ring_64x4"))

  def test_valid_32x8(self):
    self.assertTrue(max_utils.is_valid_custom_mesh([1, 1, 32, 1, 1, 8, 1, 1], "hybrid_ring_32x8"))

  def test_invalid_64x4(self):
    with self.assertRaises(ValueError):
      max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 16, 16, 1], "hybrid_ring_64x4")

  def test_invalid_strategy(self):
    with self.assertRaises(ValueError):
      max_utils.is_valid_custom_mesh([1, 1, 1, 1, 1, 16, 16, 1], "invalid_strategy")


class UnscanTest(unittest.TestCase):
  """Test unscanning utility."""

  def init_pyconfig(self, **kwargs):
    """init pyconfig"""
    init_kwargs = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "dataset_type": "synthetic",
        "model_name": "llama3.1-8b",
    } | kwargs
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **init_kwargs,
    )
    return config
  
  @pytest.mark.tpu_only
  def test_unscan_train_state_params(self):
    """Test unscan_train_state_params logic and performance with a real model."""
    # Initialize a configuration for an 8B model.
    config = self.init_pyconfig()

    _, _, sharding, _, mesh, *_, state = setup_train_loop(config, None)

    scan_axis = config.param_scan_axis
    num_layers = config.base_num_decoder_layers

    # Make a copy to unscan, leaving the original state intact.
    params_to_unscan = jax.tree_util.tree_map(lambda x: x, state.params)
    sharding_to_unscan = jax.tree_util.tree_map(
        lambda x: x, sharding.params
    )

    # Time the unscan operation.
    start_time = time.time()
    max_utils.unscan_train_state_params(
        params_to_unscan,
        sharding_to_unscan,
        mesh,
        scan_axis,
        [("layers", num_layers)],
    )
    jax.block_until_ready(params_to_unscan)
    end_time = time.time()
    print(f"\nUnscanning 8B model took: {end_time - start_time:.4f} seconds.\n")

    # Assertions to verify correctness.
    decoder_params = params_to_unscan["params"]["decoder"]
    self.assertNotIn("layers", decoder_params)
    self.assertIn("layers_0", decoder_params)
    self.assertIn(f"layers_{num_layers-1}", decoder_params)

    # Check shape of one of the unstacked tensors.
    # The exact key might differ based on model implementation, adjust if needed.
    unstacked_shape = decoder_params["layers_5"]["mlp"]["wi_0"]["kernel"].shape
    expected_shape = (config.base_emb_dim, config.base_mlp_dim)
    self.assertEqual(unstacked_shape, expected_shape)

    # Check that the original state is unchanged.
    self.assertIn("layers", state.params["params"]["decoder"])
    self.assertNotIn("layers_0", state.params["params"]["decoder"])

if __name__ == "__main__":
  unittest.main()
