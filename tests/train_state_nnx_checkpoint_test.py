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

"""TrainStateNNX checkpoint tests."""

import pathlib
import tempfile
import shutil

import unittest
import jax
import jax.numpy as jnp
from flax import nnx, serialization
from flax import linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp

from MaxText.layers import train_state_nnx


class MockModel(nnx.Module):
  """A simple model for checkpoint testing."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


class LinenMockModel(nn.Module):
  """The Linen equivalent of the MockModel."""

  @nn.compact
  def __call__(self, x):
    # We name the layer 'linear' to match the attribute name in the NNX MockModel
    return nn.Dense(features=1, name="linear")(x)


class TestTrainStateNNXCheckpoint(unittest.TestCase):
  """Class to test NNX checkpoint."""

  def setUp(self):
    self.rngs = nnx.Rngs(0)
    self.model = MockModel(rngs=self.rngs)

    # Setup a chained optimizer: Gradient Clipping -> Adam
    # Note: optax.adam is also a chain (scale_by_adam + scale_by_learning_rate).
    # This creates a nested state structure: (EmptyState, (ScaleByAdamState, EmptyState))
    self.tx = optax.chain(
        optax.clip_by_global_norm(max_norm=1.0),
        optax.adam(1e-3),
    )

  def test_checkpoint_structure(self):
    """Ensures the state object contains both model and optimizer keys."""
    optimizer = nnx.Optimizer(self.model, self.tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(self.model, optimizer)

    # We use .to_pure_dict() to simulate the format stored in a checkpoint.
    # This converts nnx.Variable/State objects into raw arrays and dictionaries.
    full_state = nnx.state(state).to_pure_dict()

    # 1. Verify Top-level Keys
    self.assertIn("model", full_state)
    self.assertIn("optimizer", full_state)

    # 2. Verify Optimizer Internal Structure
    opt_inner_state = full_state["optimizer"]["opt_state"]

    # Because we used optax.chain(clip, adam), index 0 is clip, index 1 is adam.
    # Since adam is also a chain, index 1 is itself a dictionary/tuple representation.
    # Adam's momentum (mu/nu) is in the first element of its own sub-chain.
    adam_component = opt_inner_state[1][0]

    self.assertIn("mu", adam_component, "Adam 'mu' buffer not found in pure dict state.")
    self.assertIn("nu", adam_component, "Adam 'nu' buffer not found in pure dict state.")

    # In a pure dict, these are nested dictionaries containing arrays, not NNX objects.
    self.assertIsInstance(adam_component["mu"], dict)
    self.assertIsInstance(adam_component["nu"], dict)

    # To verify a specific leaf, we navigate the dictionary hierarchy:
    self.assertIsInstance(adam_component["mu"]["linear"]["kernel"], jax.Array)

  def test_checkpoint_and_restore(self):
    """Verifies that the full state can be captured and restored into a new instance."""
    # 1. Initialize original state and optimizer
    optimizer = nnx.Optimizer(self.model, self.tx, wrt=nnx.Param)
    state_original = train_state_nnx.TrainStateNNX(self.model, optimizer)

    # 2. Perform a training step to modify weights and optimizer buffers
    def loss_fn(m):
      return jnp.mean(m(jnp.ones((1, 2))) ** 2)

    grads = nnx.grad(loss_fn)(state_original.model)
    state_original.apply_gradients(grads)

    # Capture state after one step
    original_kernel_val = state_original.model.linear.kernel.value
    original_step_val = state_original.optimizer.step.value
    self.assertEqual(original_step_val, 1)

    # 3. Capture the "Checkpoint" as a pure dictionary
    checkpoint_state = nnx.state(state_original).to_pure_dict()

    # 4. Initialize a fresh, different instance
    new_rngs = nnx.Rngs(1)
    new_model = MockModel(rngs=new_rngs)
    new_optimizer = nnx.Optimizer(new_model, self.tx, wrt=nnx.Param)
    state_restored = train_state_nnx.TrainStateNNX(new_model, new_optimizer)

    # Check differences before restoration
    self.assertEqual(state_restored.optimizer.step.value, 0)
    self.assertFalse(jnp.allclose(state_restored.model.linear.kernel.value, original_kernel_val))

    # 5. Restore the state into the new instance.
    # nnx.update supports updating from a pure dictionary.
    nnx.update(state_restored, checkpoint_state)

    # 6. Verify restoration
    # Check step counter
    self.assertEqual(state_restored.optimizer.step.value, original_step_val)
    # Check model weights
    self.assertTrue(jnp.allclose(state_restored.model.linear.kernel.value, original_kernel_val))

    # Check that it can still be trained after restoration
    new_grads = nnx.grad(loss_fn)(state_restored.model)
    state_restored.apply_gradients(new_grads)
    self.assertEqual(state_restored.optimizer.step.value, 2)

  def test_restore_from_linen_state(self):
    """Verifies a multi-stage migration: Linen CKPT -> Migrate -> NNX CKPT -> Restore."""
    # 1. Setup Linen TrainState (Simulating original training)
    linen_model = LinenMockModel()
    dummy_input = jnp.ones((1, 2))
    variables = linen_model.init(jax.random.key(42), dummy_input)

    state_linen = train_state.TrainState.create(apply_fn=linen_model.apply, params=variables["params"], tx=self.tx)

    # Perform a step to populate optimizer buffers
    grads = jax.tree.map(jnp.ones_like, state_linen.params)
    state_linen = state_linen.apply_gradients(grads=grads)

    temp_dir = pathlib.Path(tempfile.mkdtemp())
    try:
      # --- PHASE 1: Save Legacy Linen Checkpoint ---
      linen_ckpt_dir = temp_dir / "linen_ckpt"
      mngr_linen = ocp.CheckpointManager(
          linen_ckpt_dir, options=ocp.CheckpointManagerOptions(create=True), item_handlers=ocp.StandardCheckpointHandler()
      )
      mngr_linen.save(0, args=ocp.args.StandardSave(state_linen))
      mngr_linen.wait_until_finished()

      # --- PHASE 2: Read Linen CKPT and Convert to NNX Structure ---
      # Load it back without knowing the blueprint (reading as a pure PyTree)
      restored_linen_obj = mngr_linen.restore(0)

      # Convert the restored object to a pure dictionary structure.
      restored_linen_dict = serialization.to_state_dict(restored_linen_obj)

      # Helper to recursively convert string keys back to integers
      # and filter out None values.
      def recursive_clean(obj):
        if isinstance(obj, dict):
          return {int(k) if k.isdigit() else k: recursive_clean(v) for k, v in obj.items() if v is not None}
        return obj

      # Converted dict - simple PyTree mapping, no NNX Module initialization needed here.
      # This simulates a situation where the conversion logic is blueprint-agnostic.
      linen_as_nnx_dict = {
          "model": restored_linen_dict["params"],
          "optimizer": {
              "step": jnp.array(restored_linen_dict["step"]),
              "opt_state": recursive_clean(restored_linen_dict["opt_state"]),
          },
      }

      # --- PHASE 3: Save as Native NNX Checkpoint ---
      nnx_ckpt_dir = temp_dir / "nnx_ckpt"
      mngr_nnx = ocp.CheckpointManager(
          nnx_ckpt_dir, options=ocp.CheckpointManagerOptions(create=True), item_handlers=ocp.StandardCheckpointHandler()
      )
      # We save the raw dictionary directly to disk.
      mngr_nnx.save(0, args=ocp.args.StandardSave(linen_as_nnx_dict))
      mngr_nnx.wait_until_finished()

      # --- PHASE 4: Restore from NNX Checkpoint to target Model ---
      nnx_model = MockModel(rngs=nnx.Rngs(0))
      nnx_optimizer = nnx.Optimizer(nnx_model, self.tx, wrt=nnx.Param)
      state_nnx = train_state_nnx.TrainStateNNX(nnx_model, nnx_optimizer)

      # We now restore using the nnx.State as a blueprint. This ensures Orbax
      # correctly maps the arrays on disk to the model's structural expectation.
      blueprint = nnx.state(state_nnx).to_pure_dict()
      restored_nnx_pytree = mngr_nnx.restore(0, args=ocp.args.StandardRestore(item=blueprint))
      nnx.update(state_nnx, restored_nnx_pytree)

      # --- PHASE 5: Verification ---
      # 1. Verify Step
      self.assertEqual(state_nnx.optimizer.step.value, 1)

      # 2. Verify Weights
      self.assertTrue(jnp.allclose(state_nnx.model.linear.kernel.value, state_linen.params["linear"]["kernel"]))

      # 3. Verify Chained Optimizer State (Clip at index 0, Adam at index 1)
      self.assertEqual(type(state_nnx.optimizer.opt_state[0]), type(state_linen.opt_state[0]))

      # state_linen.opt_state[1] is the Adam chain state.
      # state_linen.opt_state[1][0] is the ScaleByAdamState containing 'mu'.
      self.assertTrue(
          jnp.allclose(
              state_nnx.optimizer.opt_state[1][0].mu["linear"]["kernel"],
              state_linen.opt_state[1][0].mu["linear"]["kernel"],
          )
      )

    finally:
      # Cleanup temporary directory
      shutil.rmtree(temp_dir)

  def test_restore_from_checkpoint_model_params(self):
    """Verifies that model parameters can be restored from model params only."""
    # 1. Setup mocked parameters manually (no Linen model needed for setup)
    # This structure matches the path model.linear.kernel/bias in the NNX MockModel.
    mock_params = {"linear": {"kernel": jnp.ones((2, 1)) * 9.0, "bias": jnp.zeros((1,))}}

    # Simplified checkpoint dictionary using hardcoded mocked params as requested
    checkpoint_dict = {
        "model": mock_params,
    }

    temp_dir = pathlib.Path(tempfile.mkdtemp())
    try:
      # --- PHASE 1: Save the partial checkpoint ---
      mngr = ocp.CheckpointManager(
          temp_dir, options=ocp.CheckpointManagerOptions(create=True), item_handlers=ocp.StandardCheckpointHandler()
      )
      mngr.save(0, args=ocp.args.StandardSave(checkpoint_dict))
      mngr.wait_until_finished()

      # --- PHASE 2: Restore into a full TrainStateNNX ---
      nnx_model = MockModel(rngs=nnx.Rngs(0))
      nnx_optimizer = nnx.Optimizer(nnx_model, self.tx, wrt=nnx.Param)
      state_nnx = train_state_nnx.TrainStateNNX(nnx_model, nnx_optimizer)

      # We use nnx.state to get a full blueprint as a reference.
      full_nnx_pure_dict = nnx.state(state_nnx).to_pure_dict()
      blueprint = {"model": full_nnx_pure_dict["model"]}

      # If we don't know if the checkpoint on disk has 'optimizer' or not, we simulate
      # schema-agnostic restoration by calling restore without a blueprint.
      # This avoids Orbax structural mismatch errors while allowing us to see the data.
      restored_pytree = mngr.restore(0, args=ocp.args.StandardRestore(item=blueprint))

      # Use nnx.update to apply the restored data to the stateful NNX object.
      # nnx.update is naturally partial: it will update 'model' from the restored dict
      # and leave 'optimizer' untouched at its initialized value.
      nnx.update(state_nnx, restored_pytree)

      # --- PHASE 3: Verification ---
      # Check that weights were restored to the specific mock values
      self.assertTrue(jnp.allclose(state_nnx.model.linear.kernel.value, mock_params["linear"]["kernel"]))
      # Step remains at its initialized value (0) because it was not in the checkpoint
      self.assertEqual(state_nnx.optimizer.step.value, 0)

      # Verify that the optimizer state still exists in the object (initialized)
      # even though it was not provided in the checkpoint.
      # Adam's state is at index 1 of the chain, and it's a nested structure (tuple).
      # We verify that index 0 (ScaleByAdamState) contains the 'mu' State container.
      self.assertIsInstance(state_nnx.optimizer.opt_state[1][0].mu, nnx.State)

    finally:
      # Cleanup temporary directory
      shutil.rmtree(temp_dir)


if __name__ == "__main__":
  unittest.main()
