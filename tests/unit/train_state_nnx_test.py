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

"""TrainStateNNX tests."""

import unittest
import jax.numpy as jnp
from flax import nnx
import optax

from MaxText.layers import train_state_nnx


class MockModel(nnx.Module):
  """Mocked NNX model"""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


class TestTrainStateNNX(unittest.TestCase):
  """TrainStateNNX tests."""

  def setUp(self):
    self.rngs = nnx.Rngs(0)
    self.model = MockModel(rngs=self.rngs)
    self.tx = optax.adam(1e-3)

  def test_init_with_optimizer(self):
    """Test init with iptimizer."""
    optimizer = nnx.Optimizer(self.model, self.tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(self.model, optimizer)

    self.assertEqual(state.model, self.model)
    self.assertEqual(state.optimizer, optimizer)
    # Access step directly from optimizer
    self.assertEqual(state.optimizer.step.value, 0)

  def test_init_without_optimizer(self):
    """Test init without optimizer."""
    state = train_state_nnx.TrainStateNNX(self.model, None)

    self.assertEqual(state.model, self.model)
    self.assertIsNone(state.optimizer)

  def test_apply_gradients_success(self):
    """Test apply gradients can be called successfully."""
    optimizer = nnx.Optimizer(self.model, self.tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(self.model, optimizer)

    # Create dummy gradients matching the model state structure
    def loss_fn(m):
      return jnp.mean(m(jnp.ones((1, 2))) ** 2)

    grads = nnx.grad(loss_fn)(state.model)

    # Apply gradients
    state.apply_gradients(grads)

    # Verify step incremented (managed by nnx.Optimizer)
    self.assertEqual(state.optimizer.step.value, 1)

  def test_apply_gradients_raises_runtime_error(self):
    """Test apply gradients without a optimizer."""
    # Initialize without optimizer (inference mode)
    state = train_state_nnx.TrainStateNNX(self.model, None)

    dummy_grads = {}
    with self.assertRaises(RuntimeError) as cm:
      state.apply_gradients(dummy_grads)

    self.assertIn("inference only", str(cm.exception))


if __name__ == "__main__":
  unittest.main()
