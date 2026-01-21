# Copyright 2023–2026 Google LLC
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

""" Tests for the common MaxText NNX utilities """
import unittest
from dataclasses import dataclass
from typing import Any
import jax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

from maxtext.utils import maxtext_utils_nnx


class TestMaxTextUtilsNNX(unittest.TestCase):
  """Test the functions for MaxText Utils."""

  @dataclass
  class MockConfig:
    """Minimal mock for pyconfig.HyperParameters."""

    init_weights_seed: int = 42

  class TinyModel(nnx.Module):
    """
    A tiny NNX model with logical annotations.
    Annotations are required to test that sharding extraction logic works.
    """

    def __init__(self, rngs: nnx.Rngs):
      self.linear = nnx.Linear(
          jax.device_count(),
          jax.device_count(),
          kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("data", None)),
          # FIX: Removed () from zeros. zeros is the initializer function itself,
          # not a factory like lecun_normal().
          bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("data",)),
          rngs=rngs,
      )

  def tiny_model_init_fn(self):
    """Factory function for model initialization."""
    return self.TinyModel(rngs=nnx.Rngs(0))

  def setUp(self):
    # Create a mesh for sharding tests.
    # NamedSharding requires an active Mesh to resolve logical names.
    self.devices = mesh_utils.create_device_mesh((jax.device_count(),))
    self.mesh = Mesh(self.devices, axis_names=("data",))

  def test_create_nnx_rngs_training(self):
    # Using Any to satisfy static type checkers for the MockConfig
    config: Any = self.MockConfig(init_weights_seed=123)
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, is_training=True)

    self.assertIsInstance(rngs, nnx.Rngs)
    # FIX: nnx.Rngs does not have a .streams attribute.
    # Check for stream attributes directly on the object.
    self.assertTrue(hasattr(rngs, "params"))
    self.assertTrue(hasattr(rngs, "dropout"))
    self.assertTrue(hasattr(rngs, "aqt"))

  def test_create_nnx_rngs_inference(self):
    config: Any = self.MockConfig(init_weights_seed=123)
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, is_training=False)

    self.assertIsInstance(rngs, nnx.Rngs)
    # Check that 'params' exists but 'dropout' and 'aqt' were excluded
    self.assertTrue(hasattr(rngs, "params"))
    self.assertFalse(hasattr(rngs, "dropout"))
    self.assertFalse(hasattr(rngs, "aqt"))

  def test_move_memory(self):
    sharding = NamedSharding(self.mesh, P("data"))
    self.assertNotEqual(sharding.memory_kind, "pinned_host")

    path = ("layers", "linear", "kernel")
    host_sharding = maxtext_utils_nnx.move_memory_to_host(path, sharding)

    self.assertEqual(host_sharding.memory_kind, "pinned_host")
    self.assertEqual(host_sharding.spec, P("data"))

    device_sharding = maxtext_utils_nnx.move_memory_to_device(path, sharding)

    self.assertEqual(device_sharding.memory_kind, "device")
    self.assertEqual(device_sharding.spec, P("data"))

  def test_get_set_named_sharding_nnx(self):
    # 1. Create the abstract state using standard NNX functional API
    _, abstract_state = nnx.get_abstract_model(self.tiny_model_init_fn, self.mesh)

    # 2. Test extraction
    extracted_shardings = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)

    # Verify kernel and bias match the P("data") annotations from TinyModel
    self.assertEqual(extracted_shardings.linear.kernel.get_value().spec, P("data", None))
    self.assertEqual(extracted_shardings.linear.bias.get_value().spec, P("data"))

    # Target kernel spec update
    new_kernel_spec = P(None, "data")

    def update_spec_fn(path, leaf_sharding):
      path_str = jax.tree_util.keystr(path)
      if "linear" in path_str and "kernel" in path_str:
        # Construct a new NamedSharding with the requested logical spec
        return NamedSharding(leaf_sharding.mesh, new_kernel_spec)
      return leaf_sharding

    # Apply the spec change to the extracted sharding tree
    extracted_shardings = jax.tree.map_with_path(update_spec_fn, extracted_shardings)

    # 3. Test setting new shardings
    # Transform the extracted shardings to host memory
    new_shardings = jax.tree_util.tree_map_with_path(maxtext_utils_nnx.move_memory_to_host, extracted_shardings)
    updated_abstract = maxtext_utils_nnx.set_named_sharding_nnx(abstract_state, new_shardings)

    # Verify the metadata inside the abstract state leaf has updated its sharding
    self.assertEqual(updated_abstract.linear.kernel.sharding.memory_kind, "pinned_host")
    # Also verify the spec was updated successfully
    self.assertEqual(updated_abstract.linear.kernel.sharding.spec, new_kernel_spec)

    # 4. Verify named sharding is preserved after NNX merge (update) and split (state)
    model = self.tiny_model_init_fn()
    nnx.update(model, updated_abstract)
    re_extracted_shardings = maxtext_utils_nnx.get_named_sharding_nnx(nnx.state(model))

    # Verify kernel and bias have expected sharding
    self.assertEqual(re_extracted_shardings.linear.kernel.get_value().spec, new_kernel_spec)
    self.assertEqual(re_extracted_shardings.linear.bias.get_value().spec, P("data"))

  def test_create_nnx_sharded_model(self):
    # 1. Create abstract model
    graphdef, abstract_state = nnx.get_abstract_model(self.tiny_model_init_fn, self.mesh)
    abstract_model = nnx.merge(graphdef, abstract_state)

    # 2. Modify shardings to trigger host offloading
    extracted_shardings = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    new_shardings = jax.tree_util.tree_map_with_path(maxtext_utils_nnx.move_memory_to_host, extracted_shardings)

    # 3. Run the sharded creation
    # We pass the abstract model and use the custom sharding for instantiation
    sharded_model = maxtext_utils_nnx.create_nnx_sharded_model(
        abstract_model, self.tiny_model_init_fn, mesh=self.mesh, named_sharding=new_shardings
    )

    # 4. Verify the model is concrete (contains Arrays) and sharded on host
    self.assertIsInstance(sharded_model.linear.kernel[...], jax.Array)
    self.assertEqual(sharded_model.linear.kernel[...].sharding.memory_kind, "pinned_host")

  def test_get_partition_spec_nnx(self):
    """Verifies extraction of PartitionSpecs from NamedShardings."""
    # 1. Create abstract state and get sharding
    _, abstract_state = nnx.get_abstract_model(self.tiny_model_init_fn, self.mesh)
    extracted_shardings = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)

    # 2. Execute extraction
    spec = maxtext_utils_nnx.get_partition_spec_nnx(extracted_shardings)

    # 3. Verify that the leaves are now raw PartitionSpecs
    # Expected values derived from TinyModel definition
    expected_spec_k = P("data", None)
    expected_spec_b = P("data")

    self.assertEqual(spec["linear"]["kernel"], expected_spec_k)
    self.assertEqual(spec["linear"]["bias"], expected_spec_b)
    self.assertNotIsInstance(spec["linear"]["kernel"], NamedSharding)


if __name__ == "__main__":
  unittest.main()
