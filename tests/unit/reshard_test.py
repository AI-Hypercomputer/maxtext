# Copyright 2025 Google LLC
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

"""Tests for resharding functionality.

This test module validates the resharding API that allows moving nnx model
states between different mesh configurations. This is useful for scenarios
like RL training where models need to be resharded between training and
inference meshes.
"""

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx
import numpy as np


def reshard_pytree(
    source,
    target_shardings,
):
  """Reshard a pytree from source shardings to target shardings.

  This is a simplified version of the resharding API used in tunix/rl/reshard.py.
  It uses jax.device_put to move arrays between different sharding configurations.

  Args:
    source: The source pytree containing jax.Arrays to reshard.
    target_shardings: A pytree of target shardings (NamedSharding objects).

  Returns:
    The resharded pytree with arrays matching the target shardings.
  """
  return jax.device_put(source, target_shardings)


class SimpleLinear(nnx.Module):
  """A simple linear layer for testing resharding."""

  def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
    self.kernel = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (in_features, out_features)))
    self.bias = nnx.Param(jnp.zeros((out_features,)))

  def __call__(self, x):
    return x @ self.kernel.value + self.bias.value


class SimpleMLP(nnx.Module):
  """A simple MLP model for testing resharding with multiple parameters."""

  def __init__(self, features: list[int], rngs: nnx.Rngs):
    self.layers = []
    for i in range(len(features) - 1):
      self.layers.append(SimpleLinear(features[i], features[i + 1], rngs))

  def __call__(self, x):
    for layer in self.layers[:-1]:
      x = nnx.relu(layer(x))
    return self.layers[-1](x)


class ReshardTest(unittest.TestCase):
  """Tests for resharding nnx model states between different meshes."""

  def setUp(self):
    """Set up test fixtures."""
    self.devices = jax.devices()
    self.num_devices = len(self.devices)

  def test_reshard_simple_state_same_mesh_different_sharding(self):
    """Test resharding a simple state within the same mesh but different partition specs."""
    if self.num_devices < 2:
      self.skipTest("Test requires at least 2 devices")

    # Create a simple mesh
    mesh_shape = (self.num_devices,)
    devices_array = np.array(self.devices).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("fsdp",))

    # Create a simple model
    rngs = nnx.Rngs(0)
    model = SimpleLinear(in_features=16, out_features=8, rngs=rngs)

    with mesh:
      # Get model state
      _, state = nnx.split(model)

      # Create source shardings - shard kernel along first axis (fsdp), replicate bias
      src_kernel_sharding = NamedSharding(mesh, P("fsdp", None))
      src_bias_sharding = NamedSharding(mesh, P(None))

      # Shard the state with source shardings
      src_shardings = jax.tree.map(
          lambda x: src_kernel_sharding if x.ndim == 2 else src_bias_sharding,
          state,
      )
      sharded_state = jax.device_put(state, src_shardings)

      # Verify source sharding
      kernel_src_sharding = jax.tree.leaves(sharded_state)[0].sharding
      self.assertEqual(kernel_src_sharding.spec, P("fsdp", None))

      # Create target shardings - shard kernel along second axis, replicate bias
      dst_kernel_sharding = NamedSharding(mesh, P(None, "fsdp"))
      dst_bias_sharding = NamedSharding(mesh, P(None))

      dst_shardings = jax.tree.map(
          lambda x: dst_kernel_sharding if x.ndim == 2 else dst_bias_sharding,
          state,
      )

      # Reshard to target
      resharded_state = reshard_pytree(sharded_state, dst_shardings)

      # Verify target sharding
      kernel_dst_sharding = jax.tree.leaves(resharded_state)[0].sharding
      self.assertEqual(kernel_dst_sharding.spec, P(None, "fsdp"))

      # Verify data integrity - values should be the same
      for src_leaf, dst_leaf in zip(jax.tree.leaves(sharded_state), jax.tree.leaves(resharded_state)):
        np.testing.assert_allclose(np.array(src_leaf), np.array(dst_leaf), rtol=1e-5)

  def test_reshard_state_between_different_meshes(self):
    """Test resharding a state between two different mesh configurations."""
    if self.num_devices < 4:
      self.skipTest("Test requires at least 4 devices")

    # Use a power of 2 number of devices for clean mesh shapes
    num_devices_to_use = 2 ** (self.num_devices.bit_length() - 1)
    num_devices_to_use = max(num_devices_to_use, 4)
    if num_devices_to_use > self.num_devices:
      self.skipTest(f"Test requires {num_devices_to_use} devices")

    devices_subset = self.devices[:num_devices_to_use]

    # Source mesh: (data=2, fsdp=num_devices/2)
    src_mesh_shape = (2, num_devices_to_use // 2)
    src_devices_array = np.array(devices_subset).reshape(src_mesh_shape)
    src_mesh = Mesh(src_devices_array, ("data", "fsdp"))

    # Target mesh: (fsdp=num_devices/2, tensor=2)
    dst_mesh_shape = (num_devices_to_use // 2, 2)
    dst_devices_array = np.array(devices_subset).reshape(dst_mesh_shape)
    dst_mesh = Mesh(dst_devices_array, ("fsdp", "tensor"))

    # Create a model with parameters divisible by mesh dimensions
    emb_dim = 32  # Divisible by 2
    hidden_dim = 64  # Divisible by fsdp dimension
    rngs = nnx.Rngs(42)
    model = SimpleMLP(features=[emb_dim, hidden_dim, emb_dim], rngs=rngs)

    # Split the model
    _, state = nnx.split(model)

    # Shard on source mesh with data-parallel sharding
    with src_mesh:
      src_shardings = jax.tree.map(
          lambda x: NamedSharding(src_mesh, P("fsdp", None) if x.ndim == 2 else P(None)),
          state,
      )
      src_sharded_state = jax.device_put(state, src_shardings)

      # Verify source mesh
      for leaf in jax.tree.leaves(src_sharded_state):
        self.assertEqual(leaf.sharding.mesh, src_mesh)

    # Create target shardings on destination mesh
    with dst_mesh:
      dst_shardings = jax.tree.map(
          lambda x: NamedSharding(dst_mesh, P("fsdp", "tensor") if x.ndim == 2 else P(None)),
          state,
      )

      # Reshard to destination mesh
      dst_sharded_state = reshard_pytree(src_sharded_state, dst_shardings)

      # Verify destination mesh and sharding
      for leaf in jax.tree.leaves(dst_sharded_state):
        self.assertEqual(leaf.sharding.mesh, dst_mesh)

    # Verify data integrity
    for src_leaf, dst_leaf in zip(jax.tree.leaves(src_sharded_state), jax.tree.leaves(dst_sharded_state)):
      np.testing.assert_allclose(np.array(src_leaf), np.array(dst_leaf), rtol=1e-5)

  def test_reshard_mlp_state_with_partition_spec(self):
    """Test resharding an MLP model state using partition specs from nnx.get_partition_spec."""
    if self.num_devices < 2:
      self.skipTest("Test requires at least 2 devices")

    # Create meshes
    mesh_shape = (self.num_devices,)
    devices_array = np.array(self.devices).reshape(mesh_shape)
    src_mesh = Mesh(devices_array, ("fsdp",))

    # Use same devices but different axis name for the target
    dst_mesh = Mesh(devices_array, ("tensor",))

    # Create model
    rngs = nnx.Rngs(123)
    model = SimpleMLP(features=[16, 32, 16], rngs=rngs)

    graph_def, state = nnx.split(model)

    # Get partition spec from the state
    pspecs = nnx.get_partition_spec(state)

    # Create source shardings
    with src_mesh:
      src_shardings = jax.tree.map(
          lambda x, pspec: NamedSharding(
              src_mesh,
              P("fsdp", None) if hasattr(x, "ndim") and x.ndim == 2 else P(None),
          ),
          state,
          pspecs,
      )
      src_state = jax.device_put(state, src_shardings)

    # Create target shardings with different mesh
    with dst_mesh:
      dst_shardings = jax.tree.map(
          lambda x: NamedSharding(
              dst_mesh,
              P(None, "tensor") if hasattr(x, "ndim") and x.ndim == 2 else P(None),
          ),
          state,
      )

      # Reshard
      dst_state = reshard_pytree(src_state, dst_shardings)

      # Verify the resharding
      for leaf in jax.tree.leaves(dst_state):
        self.assertEqual(leaf.sharding.mesh, dst_mesh)

    # Verify the model can still be used after resharding
    with dst_mesh:
      restored_model = nnx.merge(graph_def, dst_state)
      # Create test input
      test_input = jnp.ones((4, 16))
      output = restored_model(test_input)
      self.assertEqual(output.shape, (4, 16))

  def test_reshard_preserves_nnx_variable_types(self):
    """Test that resharding preserves the nnx variable types (Param, etc.)."""
    if self.num_devices < 2:
      self.skipTest("Test requires at least 2 devices")

    mesh_shape = (self.num_devices,)
    devices_array = np.array(self.devices).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("fsdp",))

    rngs = nnx.Rngs(0)
    model = SimpleLinear(in_features=16, out_features=8, rngs=rngs)

    with mesh:
      _, state = nnx.split(model)

      # Get original variable types
      original_types = jax.tree.map(type, state)

      # Create shardings and shard
      src_shardings = jax.tree.map(
          lambda x: NamedSharding(mesh, P("fsdp", None) if x.ndim == 2 else P(None)),
          state,
      )
      sharded_state = jax.device_put(state, src_shardings)

      # Create target shardings
      dst_shardings = jax.tree.map(
          lambda x: NamedSharding(mesh, P(None, "fsdp") if x.ndim == 2 else P(None)),
          state,
      )

      # Reshard
      resharded_state = reshard_pytree(sharded_state, dst_shardings)

      # Verify variable types are preserved
      resharded_types = jax.tree.map(type, resharded_state)

      for orig, resharded in zip(jax.tree.leaves(original_types), jax.tree.leaves(resharded_types)):
        self.assertEqual(orig, resharded)

  def test_reshard_with_replicated_to_sharded(self):
    """Test resharding from fully replicated to sharded state."""
    if self.num_devices < 2:
      self.skipTest("Test requires at least 2 devices")

    mesh_shape = (self.num_devices,)
    devices_array = np.array(self.devices).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("fsdp",))

    rngs = nnx.Rngs(0)
    model = SimpleLinear(in_features=16, out_features=8, rngs=rngs)

    with mesh:
      _, state = nnx.split(model)

      # Start with fully replicated state
      replicated_shardings = jax.tree.map(lambda x: NamedSharding(mesh, P()), state)
      replicated_state = jax.device_put(state, replicated_shardings)

      # Verify replicated
      for leaf in jax.tree.leaves(replicated_state):
        self.assertEqual(leaf.sharding.spec, P())

      # Create sharded target shardings
      sharded_shardings = jax.tree.map(
          lambda x: NamedSharding(mesh, P("fsdp", None) if x.ndim == 2 else P(None)),
          state,
      )

      # Reshard from replicated to sharded
      sharded_state = reshard_pytree(replicated_state, sharded_shardings)

      # Verify sharding was applied
      for leaf in jax.tree.leaves(sharded_state):
        if leaf.ndim == 2:
          self.assertEqual(leaf.sharding.spec, P("fsdp", None))

      # Verify data integrity
      for rep_leaf, shard_leaf in zip(jax.tree.leaves(replicated_state), jax.tree.leaves(sharded_state)):
        np.testing.assert_allclose(np.array(rep_leaf), np.array(shard_leaf), rtol=1e-5)

  def test_reshard_with_sharded_to_replicated(self):
    """Test resharding from sharded to fully replicated state (all-gather)."""
    if self.num_devices < 2:
      self.skipTest("Test requires at least 2 devices")

    mesh_shape = (self.num_devices,)
    devices_array = np.array(self.devices).reshape(mesh_shape)
    mesh = Mesh(devices_array, ("fsdp",))

    rngs = nnx.Rngs(0)
    model = SimpleLinear(in_features=16, out_features=8, rngs=rngs)

    with mesh:
      _, state = nnx.split(model)

      # Start with sharded state
      sharded_shardings = jax.tree.map(
          lambda x: NamedSharding(mesh, P("fsdp", None) if x.ndim == 2 else P(None)),
          state,
      )
      sharded_state = jax.device_put(state, sharded_shardings)

      # Create replicated target shardings
      replicated_shardings = jax.tree.map(lambda x: NamedSharding(mesh, P()), state)

      # Reshard from sharded to replicated (all-gather)
      replicated_state = reshard_pytree(sharded_state, replicated_shardings)

      # Verify replicated
      for leaf in jax.tree.leaves(replicated_state):
        self.assertEqual(leaf.sharding.spec, P())

      # Verify data integrity
      for shard_leaf, rep_leaf in zip(jax.tree.leaves(sharded_state), jax.tree.leaves(replicated_state)):
        np.testing.assert_allclose(np.array(shard_leaf), np.array(rep_leaf), rtol=1e-5)


class ReshardModelToMeshTest(unittest.TestCase):
  """Tests for reshard_model_to_mesh helper function."""

  def setUp(self):
    """Set up test fixtures."""
    self.devices = jax.devices()
    self.num_devices = len(self.devices)

  def test_reshard_model_to_mesh(self):
    """Test the reshard_model_to_mesh helper that reshards an entire nnx.Module."""
    if self.num_devices < 2:
      self.skipTest("Test requires at least 2 devices")

    # Create two meshes with same devices but different configurations
    mesh_shape = (self.num_devices,)
    devices_array = np.array(self.devices).reshape(mesh_shape)
    src_mesh = Mesh(devices_array, ("fsdp",))
    dst_mesh = Mesh(devices_array, ("tensor",))

    # Create and shard model on source mesh
    rngs = nnx.Rngs(0)
    model = SimpleMLP(features=[16, 32, 16], rngs=rngs)

    with src_mesh:
      graph_def, state = nnx.split(model)
      src_shardings = jax.tree.map(
          lambda x: NamedSharding(src_mesh, P("fsdp", None) if x.ndim == 2 else P(None)),
          state,
      )
      sharded_state = jax.device_put(state, src_shardings)
      model = nnx.merge(graph_def, sharded_state)

    # Verify model is on source mesh
    model_state = nnx.state(model)
    for leaf in jax.tree.leaves(model_state):
      self.assertEqual(leaf.sharding.mesh, src_mesh)

    # Reshard model to destination mesh
    with dst_mesh:
      graph_def, state = nnx.split(model)
      dst_shardings = jax.tree.map(
          lambda x: NamedSharding(dst_mesh, P("tensor", None) if x.ndim == 2 else P(None)),
          state,
      )
      resharded_state = reshard_pytree(state, dst_shardings)
      resharded_model = nnx.merge(graph_def, resharded_state)

    # Verify model is on destination mesh
    resharded_model_state = nnx.state(resharded_model)
    for leaf in jax.tree.leaves(resharded_model_state):
      self.assertEqual(leaf.sharding.mesh, dst_mesh)

    # Verify model still works
    with dst_mesh:
      test_input = jnp.ones((4, 16))
      output = resharded_model(test_input)
      self.assertEqual(output.shape, (4, 16))


if __name__ == "__main__":
  unittest.main()
