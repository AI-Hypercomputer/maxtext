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
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
import optax


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


class TestStateReconstructionHelpers(unittest.TestCase):
  """default_for_sds / populate_pure_dict_from_partial / rebuild_nnx_with_values."""

  def test_default_for_sds_zeros_for_array(self):
    sds = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    out = train_state_nnx.default_for_sds(sds)
    self.assertEqual(out.shape, (2, 3))
    self.assertEqual(out.dtype, jnp.float32)
    self.assertTrue(jnp.array_equal(out, jnp.zeros((2, 3), jnp.float32)))

  def test_default_for_sds_key_for_key_dtype(self):
    key_dtype = jax.random.key(0).dtype  # str(dtype) contains "key"
    out = train_state_nnx.default_for_sds(jax.ShapeDtypeStruct((), key_dtype))
    self.assertIn("key", str(out.dtype))

  def test_default_for_sds_passthrough_non_sds(self):
    # A value without shape/dtype is returned unchanged.
    self.assertEqual(train_state_nnx.default_for_sds("not-an-array"), "not-an-array")

  def test_populate_takes_concrete_and_defaults_missing(self):
    abstract = {
        "restored": jax.ShapeDtypeStruct((2,), jnp.float32),
        "missing": jax.ShapeDtypeStruct((3,), jnp.float32),
    }
    partial = {"restored": jnp.ones((2,))}
    out = train_state_nnx.populate_pure_dict_from_partial(abstract, partial)
    self.assertTrue(jnp.array_equal(out["restored"], jnp.ones((2,))))  # taken from concrete
    self.assertTrue(jnp.array_equal(out["missing"], jnp.zeros((3,))))  # defaulted

  def test_rebuild_nnx_with_values_binds_arrays(self):
    abstract = {"w": nnx.Param(jnp.zeros((2,))), "b": nnx.Param(jnp.zeros(()))}
    concrete = {"w": jnp.ones((2,)), "b": jnp.asarray(5.0)}
    rebuilt = train_state_nnx.rebuild_nnx_with_values(abstract, concrete)
    # Read the bound array off each Variable leaf (version-robust vs. `.value`).
    w_val = jax.tree_util.tree_leaves(rebuilt["w"])[0]
    b_val = jax.tree_util.tree_leaves(rebuilt["b"])[0]
    self.assertTrue(jnp.array_equal(w_val, jnp.ones((2,))))
    self.assertEqual(float(b_val), 5.0)

  def test_rebuild_nnx_with_values_raises_on_count_mismatch(self):
    abstract = {"w": nnx.Param(jnp.zeros((2,))), "b": nnx.Param(jnp.zeros(()))}
    concrete = {"w": jnp.ones((2,))}  # one leaf vs two abstract Variables
    with self.assertRaises(ValueError):
      train_state_nnx.rebuild_nnx_with_values(abstract, concrete)


if __name__ == "__main__":
  unittest.main()
