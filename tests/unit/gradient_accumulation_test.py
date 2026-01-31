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

"""Test the gradient_accumulation."""

from dataclasses import dataclass
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

# Import the utilities
from maxtext.optimizers.gradient_accumulation import gradient_accumulation_loss_and_grad
from maxtext.utils import sharding
from maxtext.layers import train_state_nnx
from maxtext.utils import maxtext_utils_nnx


class TestNNXGradientAccumulation(unittest.TestCase):
  """Test the NNX gradient accumulation."""

  class DropoutModel(nnx.Module):
    """A model designed to consume RNGs to test advancement logic."""

    def __init__(self, rngs: nnx.Rngs):
      self.linear = nnx.Linear(2, 2, rngs=rngs)
      # Dropout itself doesn't hold state, but it will consume from our stateful rngs
      self.dropout = nnx.Dropout(rate=0.5)
      # Store the Rngs object so it is part of the Module's state PyTree
      self.rngs = rngs

    def __call__(self, x, is_train=True):
      x = self.linear(x)
      # Explicitly pass the stateful RNGs attribute to ensure they are advanced
      x = self.dropout(x, deterministic=not is_train, rngs=self.rngs)
      return x

  @dataclass
  class MockConfig:
    """Mock for the configuration object."""

    gradient_accumulation_steps: int = 2
    shard_optimizer_over_data: bool = False
    shard_mode: str = "auto"
    debug_sharding: bool = False
    ici_data_parallelism: int = 1
    pure_nnx: bool = True

  def setUp(self):
    """Sets up basic mesh and config."""
    self.config = self.MockConfig()
    devices = jax.local_devices()[:1]
    # Ensure logical axis names match what we expect in the tests
    axis_names = ("data", "model")
    self.mesh = Mesh(np.array(devices).reshape((1,) * len(axis_names)), axis_names=axis_names)

  def test_rng_advancement_logic(self):
    """
    Verifies that RNGs advance across microbatches and sync back to the instance.
    """
    # 1. Initialize model and capture initial RNG state
    rngs = nnx.Rngs(dropout=jax.random.key(42), params=jax.random.key(0))
    model = self.DropoutModel(rngs)

    # Get the abstract state structure
    _, abstract_state = nnx.get_abstract_model(lambda: train_state_nnx.TrainStateNNX(model, None), self.mesh)

    # Extract "Shardings" from the abstract state.
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_shardings = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    # Resolve sharding specs
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
        self.config, state_mesh_shardings
    )

    # We define a helper to safely convert both keys and counts to comparable lists.
    def to_comparable(tree):
      def _convert(leaf):
        # Check if it's a JAX PRNG key (dtype key<...>)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
          return jax.random.key_data(leaf).tolist()
        # Otherwise treat as a standard array/scalar (like the RNG count)
        return np.array(leaf).tolist()

      return jax.tree.map(_convert, tree)

    # Capture the "initial" state values
    initial_rng_state = to_comparable(nnx.state(model).to_pure_dict()["rngs"])

    # 2. Define a loss function that triggers dropout
    def mock_loss_fn(m, config, data, dr_rng, params, is_train=True):
      logits = m(data["inputs"], is_train=is_train)
      loss = jnp.mean(logits**2)
      return loss, {"total_loss": loss, "total_weights": 1.0, "moe_lb_loss": 0.0, "mtp_loss": 0.0}

    # 3. Create dummy data (2 microbatches)
    data = {"inputs": jnp.ones((2, 2))}

    # 4. Run Gradient Accumulation
    # FIX: Wrap the execution in the mesh context so PartitionSpecs can be resolved.
    with self.mesh:
      _, _, _ = gradient_accumulation_loss_and_grad(
          mock_loss_fn,
          self.config,
          model,
          None,  # params
          params_shardings,
          data,
          None,  # dropout_rng (Linen only)
          [],  # extra_dpo_args
      )

    # 5. VERIFICATION: Check RNG advancement
    # Capture the final state after advancement
    final_rng_state = to_comparable(nnx.state(model).to_pure_dict()["rngs"])

    # Verify that the state (either count or key) has changed.
    self.assertNotEqual(initial_rng_state, final_rng_state, "RNG state did not advance/sync back to the model instance.")

  def test_ga_consistency(self):
    """Checks that gradients are accumulated and averaged correctly."""
    rngs = nnx.Rngs(dropout=jax.random.key(1), params=jax.random.key(2))
    model = self.DropoutModel(rngs)

    # Get the abstract state structure
    _, abstract_state = nnx.get_abstract_model(lambda: train_state_nnx.TrainStateNNX(model, None), self.mesh)

    # Extract "Shardings" from the abstract state.
    named_sharding = maxtext_utils_nnx.get_named_sharding_nnx(abstract_state)
    state_mesh_shardings = maxtext_utils_nnx.get_partition_spec_nnx(named_sharding)

    # Resolve sharding specs
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
        self.config, state_mesh_shardings
    )

    # Included 'moe_lb_loss' and 'mtp_loss' in aux to avoid KeyErrors in the utility.
    def deterministic_loss(m, config, data, dr_rng, params, is_train=True):
      logits = m(data["inputs"], is_train=False)  # Dropout OFF
      loss = jnp.mean(logits**2)
      return loss, {"total_loss": loss, "total_weights": 1.0, "moe_lb_loss": 0.0, "mtp_loss": 0.0}

    data = {"inputs": jnp.ones((4, 2))}  # 4 steps total
    params_shardings = jax.tree.map(lambda _: None, nnx.state(model, nnx.Param))

    # Run with GA steps = 2
    self.config.gradient_accumulation_steps = 2

    # Even if shardings are None, it is safer to wrap in mesh context
    # to support the model application logic.
    with self.mesh:
      _, _, grads_ga = gradient_accumulation_loss_and_grad(
          deterministic_loss, self.config, model, None, params_shardings, data, None, []
      )

      # Run standard grad
      grad_fn = nnx.value_and_grad(deterministic_loss, argnums=0, has_aux=True)
      (_, _), grads_std = grad_fn(model, self.config, data, None, None, is_train=True)

    # Convert nnx.State to pure dicts before comparing values.
    jax.tree.map(
        lambda g1, g2: self.assertTrue(jnp.allclose(g1, g2, atol=1e-5)), grads_ga.to_pure_dict(), grads_std.to_pure_dict()
    )


if __name__ == "__main__":
  unittest.main()
