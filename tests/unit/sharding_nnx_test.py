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

"""Tests for MaxText NNX sharding."""

import unittest
from dataclasses import dataclass
import jax
from flax import nnx
import optax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from MaxText import sharding
from MaxText.layers import train_state_nnx
from maxtext.utils import maxtext_utils_nnx


class TestShardingNNX(unittest.TestCase):
  """Test NNX related sharding functions."""

  @dataclass
  class MockConfig:
    """Mock for the configuration object."""

    shard_optimizer_over_data: bool = False

  class MockModel(nnx.Module):
    """A simple model for testing sharding extraction logic."""

    def __init__(self, rngs: nnx.Rngs):
      # Use nnx.Dict to allow holding stateful JAX data (Arrays).
      self.layers = nnx.Dict(
          {
              "dense": nnx.Linear(
                  2,
                  4,
                  kernel_init=nnx.with_partitioning(nnx.initializers.ones, PartitionSpec("data", "model")),
                  bias_init=nnx.with_partitioning(nnx.initializers.zeros, PartitionSpec("data")),
                  rngs=rngs,
              )
          }
      )

  def setUp(self):
    """Sets up basic mesh and config."""
    devices = jax.local_devices()[:1]
    # Ensure all logical axis names used in PartitionSpecs are defined in the mesh.
    axis_names = ("data", "model", "extra", "custom_axis")
    self.mesh = Mesh(np.array(devices).reshape((1,) * len(axis_names)), axis_names=axis_names)
    self.config = self.MockConfig()

  def test_no_update_when_disabled(self):
    """Verifies that the state is unchanged if shard_optimizer_over_data is False."""

    def create_train_state():
      rngs = nnx.Rngs(0)
      model = self.MockModel(rngs)
      tx = optax.adam(1e-3)
      optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    # Get the abstract state structure
    _, abstract_state = nnx.get_abstract_model(create_train_state, self.mesh)

    # Extract "Shardings" from the abstract state.
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_sharding = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    config = self.MockConfig(shard_optimizer_over_data=False)
    # Call utility directly on raw NNX state
    prev, updated = sharding.maybe_update_params_sharding_with_opt_nnx(config, state_mesh_sharding)

    self.assertEqual(prev, state_mesh_sharding.model)
    self.assertEqual(updated, state_mesh_sharding)

  def test_update_with_direct_adam_state(self):
    """Verifies parameter sharding update when opt_state contains Adam momentum."""

    def create_train_state():
      rngs = nnx.Rngs(0)
      model = self.MockModel(rngs)
      tx = optax.adam(1e-3)
      optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    _, abstract_state = nnx.get_abstract_model(create_train_state, self.mesh)
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_sharding = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    # Tweak mu spec to be different from model spec to verify update
    new_mu_spec = PartitionSpec("data", "model", "extra")

    def update_mu_fn(path, spec):
      path_str = jax.tree_util.keystr(path)
      if "opt_state" in path_str and "mu" in path_str and "kernel" in path_str:
        return new_mu_spec
      return spec

    state_mesh_sharding = jax.tree.map_with_path(update_mu_fn, state_mesh_sharding)

    config = self.MockConfig(shard_optimizer_over_data=True)
    # Call utility directly; it should handle the nnx.State structure internally
    prev, updated = sharding.maybe_update_params_sharding_with_opt_nnx(config, state_mesh_sharding)

    self.assertEqual(updated.model.layers["dense"].kernel, new_mu_spec)
    self.assertEqual(prev.layers["dense"].kernel, PartitionSpec("data", "model"))

  def test_update_with_chained_optimizer_tuple(self):
    """Verifies logic when Adam is deep within a chained optimizer."""

    def create_train_state():
      rngs = nnx.Rngs(0)
      model = self.MockModel(rngs)
      tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
      optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    _, abstract_state = nnx.get_abstract_model(create_train_state, self.mesh)
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_sharding = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    mu_spec = PartitionSpec("data", "custom_axis")

    def update_mu_fn(path, spec):
      path_str = jax.tree_util.keystr(path)
      if "opt_state" in path_str and "mu" in path_str and "kernel" in path_str:
        return mu_spec
      return spec

    state_mesh_sharding = jax.tree.map_with_path(update_mu_fn, state_mesh_sharding)

    config = self.MockConfig(shard_optimizer_over_data=True)
    _, updated = sharding.maybe_update_params_sharding_with_opt_nnx(config, state_mesh_sharding)

    self.assertEqual(updated.model.layers["dense"].kernel, mu_spec)

  def test_raises_error_when_adam_missing_in_chain(self):
    """Ensures NotImplementedError is raised if Adam state isn't in the chain (stateless)."""

    def create_train_state():
      rngs = nnx.Rngs(0)
      model = self.MockModel(rngs)
      tx = optax.chain(optax.clip(1.0), optax.sgd(1e-3))
      optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    _, abstract_state = nnx.get_abstract_model(create_train_state, self.mesh)
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_sharding = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    config = self.MockConfig(shard_optimizer_over_data=True)

    # Assert that the function raises the error if Adam is missing from a stateless chain
    with self.assertRaisesRegex(NotImplementedError, "Could not find Adam optimizer state"):
      sharding.maybe_update_params_sharding_with_opt_nnx(config, state_mesh_sharding)

  def test_raises_error_with_other_stateful_optimizer(self):
    """Ensures NotImplementedError is raised for stateful optimizers that aren't Adam."""

    def create_train_state():
      rngs = nnx.Rngs(0)
      model = self.MockModel(rngs)
      # optax.trace creates a TraceState, which is stateful but lacks Adam's mu/nu buffers.
      tx = optax.trace(decay=0.9)
      optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    _, abstract_state = nnx.get_abstract_model(create_train_state, self.mesh)
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_sharding = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    config = self.MockConfig(shard_optimizer_over_data=True)

    # Should raise because TraceState is not ScaleByAdamState and doesn't have 'mu' keys
    with self.assertRaisesRegex(NotImplementedError, "Could not find Adam optimizer state"):
      sharding.maybe_update_params_sharding_with_opt_nnx(config, state_mesh_sharding)

  def test_nnx_state_immutability(self):
    """Confirms that the function produces a new State object (functional update)."""

    def create_train_state():
      rngs = nnx.Rngs(0)
      model = self.MockModel(rngs)
      optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    _, abstract_state = nnx.get_abstract_model(create_train_state, self.mesh)
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_sharding = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    # Introduce a difference in sharding to ensure the merge logic results in different values
    new_mu_spec = PartitionSpec("data", "custom_axis")

    def update_mu_fn(path, spec):
      path_str = jax.tree_util.keystr(path)
      if "opt_state" in path_str and "mu" in path_str:
        return new_mu_spec
      return spec

    state_mesh_sharding = jax.tree.map_with_path(update_mu_fn, state_mesh_sharding)

    config = self.MockConfig(shard_optimizer_over_data=True)
    _, updated = sharding.maybe_update_params_sharding_with_opt_nnx(config, state_mesh_sharding)

    # Verify functional update: new object, original remains unchanged
    self.assertIsNot(state_mesh_sharding, updated)
    # Kernels are now actually different (original 'data, model' vs updated 'data, custom_axis')
    self.assertNotEqual(state_mesh_sharding.model.layers["dense"].kernel, updated.model.layers["dense"].kernel)
    # Verify that the tree structure is preserved exactly
    # Convert to pure dictionaries before comparing tree structure.
    # This handles cases where one state uses standard dicts and the other uses nnx.State
    # wrappers for nested branches (e.g. 'layers'), ensuring we only compare the logical hierarchy.
    self.assertEqual(
        jax.tree_util.tree_structure(state_mesh_sharding.to_pure_dict()),
        jax.tree_util.tree_structure(updated.to_pure_dict()),
        "The PyTree structure was modified during the sharding update.",
    )


if __name__ == "__main__":
  unittest.main()
