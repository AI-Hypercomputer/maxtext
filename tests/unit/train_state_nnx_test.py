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
import numpy as np
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

  def test_apply_gradients_stochastic_rounding_matches_default_for_f32_params(self):
    """With f32 params, the stochastic-rounding path gives the same params and optimizer state as the default."""
    model_sr = MockModel(rngs=nnx.Rngs(0))
    state_ref = train_state_nnx.TrainStateNNX(self.model, nnx.Optimizer(self.model, self.tx, wrt=nnx.Param))
    state_sr = train_state_nnx.TrainStateNNX(model_sr, nnx.Optimizer(model_sr, self.tx, wrt=nnx.Param))

    def loss_fn(m):
      return jnp.mean(m(jnp.ones((1, 2))) ** 2)

    for _ in range(3):
      state_ref.apply_gradients(nnx.grad(loss_fn)(state_ref.model))
      state_sr.apply_gradients(nnx.grad(loss_fn)(state_sr.model), stochastic_rounding_seed=0)

    self.assertEqual(state_sr.optimizer.step.value, 3)
    for a, b in zip(jax.tree.leaves(nnx.state(state_ref.model)), jax.tree.leaves(nnx.state(state_sr.model))):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(jax.tree.leaves(state_ref.optimizer.opt_state), jax.tree.leaves(state_sr.optimizer.opt_state)):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_apply_gradients_stochastic_rounding_keeps_small_bf16_updates(self):
    """Updates far below the bf16 ulp are lost with round-to-nearest but kept on average with stochastic rounding."""

    class Bf16Model(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((4096,), jnp.bfloat16))

    lr = 1e-4  # bf16 ulp at 1.0 is 2**-7; an Adam step is ~lr per element, far below half an ulp.
    states = []
    for _ in range(2):
      model = Bf16Model()
      states.append(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)))
    grads = nnx.State({"w": nnx.Param(jnp.ones((4096,), jnp.bfloat16))})
    steps = 10
    for _ in range(steps):
      states[0].apply_gradients(grads)
      states[1].apply_gradients(grads, stochastic_rounding_seed=0)

    w_rtn = np.asarray(states[0].model.w[...], np.float32)
    w_sr = np.asarray(states[1].model.w[...], np.float32)
    np.testing.assert_array_equal(w_rtn, 1.0)  # every update was rounded away
    self.assertEqual(states[1].model.w[...].dtype, jnp.bfloat16)
    # Each Adam step here is -lr per element, so the expected value is 1 - steps * lr.
    np.testing.assert_allclose(w_sr.mean(), 1.0 - steps * lr, atol=steps * lr * 0.1)

  def test_apply_gradients_stochastic_rounding_with_muon(self):
    """Stochastic rounding works with a Muon transform (optax.contrib.muon): f32 parity and bf16 progress."""

    def make(dtype):
      model = nnx.Linear(64, 32, param_dtype=dtype, rngs=nnx.Rngs(0))
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.contrib.muon(1e-3), wrt=nnx.Param))

    def loss_fn(m):
      return jnp.mean(m(jnp.ones((4, 64), m.kernel[...].dtype)).astype(jnp.float32) ** 2)

    # f32 params: identical to the default update.
    ref, sr = make(jnp.float32), make(jnp.float32)
    for _ in range(3):
      ref.apply_gradients(nnx.grad(loss_fn)(ref.model))
      sr.apply_gradients(nnx.grad(loss_fn)(sr.model), stochastic_rounding_seed=0)
    for a, b in zip(jax.tree.leaves(nnx.state(ref.model)), jax.tree.leaves(nnx.state(sr.model))):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    self.assertEqual(sr.optimizer.step.value, 3)

    # bf16 params: the update runs, keeps the dtype and moves the weights.
    state = make(jnp.bfloat16)
    before = np.asarray(state.model.kernel[...], np.float32)
    state.apply_gradients(nnx.grad(loss_fn)(state.model), stochastic_rounding_seed=0)
    self.assertEqual(state.model.kernel[...].dtype, jnp.bfloat16)
    self.assertGreater(np.abs(np.asarray(state.model.kernel[...], np.float32) - before).max(), 0.0)

  def test_apply_gradients_stochastic_rounding_seed_controls_the_rounding(self):
    """The same seed reproduces the rounding exactly; a different seed rounds differently."""

    def run(seed):
      model = nnx.Linear(256, 256, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
      state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param))
      grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 256), jnp.bfloat16)).astype(jnp.float32) ** 2))(model)
      state.apply_gradients(grads, stochastic_rounding_seed=seed)
      return np.asarray(state.model.kernel[...], np.float32)

    np.testing.assert_array_equal(run(0), run(0))
    self.assertFalse(np.array_equal(run(0), run(1)))


class StochasticRoundToTest(unittest.TestCase):
  """stochastic_round_to: unbiased, only rounds to the two neighboring bf16 values, passes non-finite through."""

  def test_unbiased_between_neighbors(self):
    for value in (1.0 + 2**-9, -(3.0 + 2**-8), 1.0e-3 * (1 + 2**-10)):
      x = jnp.full((200_000,), value, jnp.float32)
      out = np.asarray(train_state_nnx.stochastic_round_to(x, jnp.bfloat16, jax.random.PRNGKey(0)), np.float32)
      lo = np.float32(jnp.asarray(value, jnp.float32).astype(jnp.bfloat16))
      neighbors = np.unique(out)
      self.assertLessEqual(len(neighbors), 2)
      self.assertTrue(np.all(np.abs(out - value) < abs(value) * 2**-7), (value, neighbors, lo))
      np.testing.assert_allclose(out.mean(), value, rtol=2**-13)

  def test_exact_and_non_finite_values_unchanged(self):
    x = jnp.array([0.0, 1.0, -2.5, jnp.inf, -jnp.inf, jnp.nan], jnp.float32)
    out = train_state_nnx.stochastic_round_to(x, jnp.bfloat16, jax.random.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(x.astype(jnp.bfloat16), np.float32))

  def test_largest_bf16_does_not_overflow(self):
    big = float(jnp.finfo(jnp.bfloat16).max)
    x = jnp.full((1024,), big * (1 + 2**-10), jnp.float32)
    out = train_state_nnx.stochastic_round_to(x, jnp.bfloat16, jax.random.PRNGKey(2))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_non_bf16_dtype_is_a_plain_cast(self):
    x = jnp.linspace(-1, 1, 7, dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(train_state_nnx.stochastic_round_to(x, jnp.float32, jax.random.PRNGKey(3))), np.asarray(x)
    )


class ApplyCheckpointAuxTest(unittest.TestCase):
  """Resuming must survive a model that stopped holding some transient state."""

  class _Holder(nnx.Module):

    def __init__(self, keep_rngs: bool):
      self.w = nnx.Param(jnp.zeros((2,)))
      if keep_rngs:
        self.rngs = nnx.Rngs(params=0, dropout=1)

  def _aux(self, module):
    return nnx.state(module, nnx.RngState)

  def test_extra_checkpoint_entries_are_skipped(self):
    """A checkpoint written before a module dropped its Rngs must still restore.

    nnx.replace_by_pure_dict raises on a key the state does not have, so without this
    an older checkpoint fails the resume outright.
    """
    written = nnx.to_pure_dict(self._aux(self._Holder(keep_rngs=True)))
    self.assertTrue(written, "the writer must actually have RNG state to drop")

    target = self._aux(self._Holder(keep_rngs=False))
    train_state_nnx.apply_checkpoint_aux(target, written)  # must not raise

  def test_matching_entries_are_restored(self):
    source = self._Holder(keep_rngs=True)
    source.rngs.dropout()  # advance so the saved count is distinguishable
    written = nnx.to_pure_dict(self._aux(source))

    # nnx.state() returns a copy; the real path merges it back afterwards, so assert
    # on the state object rather than the module it came from.
    target = self._aux(self._Holder(keep_rngs=True))
    train_state_nnx.apply_checkpoint_aux(target, written)

    self.assertEqual(
        nnx.to_pure_dict(target)["rngs"]["dropout"]["count"],
        written["rngs"]["dropout"]["count"],
        "entries the model still holds must be filled from the checkpoint",
    )

  def test_empty_aux_is_a_noop(self):
    target = self._aux(self._Holder(keep_rngs=True))
    train_state_nnx.apply_checkpoint_aux(target, {})


if __name__ == "__main__":
  unittest.main()
