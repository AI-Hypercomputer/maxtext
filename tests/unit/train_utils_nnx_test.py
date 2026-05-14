# Copyright 2025-2026 Google LLC
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

"""Unit tests for the NNX-specific helpers / patterns in train_utils.setup_train_loop.

setup_train_loop itself is integration territory (it touches data iterators,
checkpoint managers, and a real mesh), so we cover the NNX-only pieces that
have unit-testable contracts:

  1. The create_train_state_fn closure pattern: builds nnx.Optimizer + TrainStateNNX
     from a zero-arg model factory and a transform.
  2. nnx.split(state.model, nnx.Param, ...) returns Param-only state used to
     compute state_params / state_mesh_shardings_params.
  3. nnx.merge(state_graphdef, state) reconstitutes a TrainStateNNX from the
     pure-state form returned by setup_training_state.
"""

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from maxtext.layers import train_state_nnx


class _Model(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)


class TestCreateTrainStateFnClosure(unittest.TestCase):
  """Exercise the closure pattern in setup_train_loop:

  def create_train_state_fn():
    model = _create_model_partial()
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return train_state_nnx.TrainStateNNX(model, optimizer)
  """

  def test_returns_train_state_nnx_with_optimizer(self):
    tx = optax.sgd(0.01)

    def _create_model():
      return _Model(rngs=nnx.Rngs(0))

    def create_train_state_fn():
      model = _create_model()
      optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(model, optimizer)

    state = create_train_state_fn()
    self.assertIsInstance(state, train_state_nnx.TrainStateNNX)
    self.assertIsInstance(state.optimizer, nnx.Optimizer)
    self.assertEqual(int(state.optimizer.step.get_value()), 0)

  def test_two_invocations_produce_independent_states(self):
    """The lambda must call the factory each time (otherwise checkpoint init/restore would alias)."""
    tx = optax.sgd(0.01)
    counter = {"n": 0}

    def _create_model():
      counter["n"] += 1
      return _Model(rngs=nnx.Rngs(counter["n"]))

    def create_train_state_fn():
      model = _create_model()
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    s1 = create_train_state_fn()
    s2 = create_train_state_fn()
    self.assertEqual(counter["n"], 2)
    self.assertIsNot(s1.model, s2.model)


class TestSetupTrainLoopNNXTreeOps(unittest.TestCase):
  """Cover the nnx.split(state.model, nnx.Param, ...) and nnx.merge round-trip
  patterns that setup_train_loop uses to derive Param-only views and rebuild
  the full TrainStateNNX before returning."""

  def setUp(self):
    self.tx = optax.sgd(0.01)
    self.model = _Model(rngs=nnx.Rngs(0))
    self.state = train_state_nnx.TrainStateNNX(self.model, nnx.Optimizer(self.model, self.tx, wrt=nnx.Param))

  def test_nnx_split_yields_param_only_state(self):
    """state_params used for assert_params_sufficiently_sharded must contain only nnx.Param leaves."""
    _, state_params, _ = nnx.split(self.state.model, nnx.Param, ...)
    leaves = jax.tree.leaves(state_params, is_leaf=lambda x: isinstance(x, nnx.Variable))
    self.assertGreater(len(leaves), 0)
    for leaf in leaves:
      self.assertIsInstance(leaf, nnx.Param)

  def test_nnx_merge_reconstructs_train_state_nnx(self):
    """setup_train_loop ends with nnx.merge(state_graphdef, state) — verify that round-trips."""
    state_graphdef, state_pure = nnx.split(self.state)
    train_state = nnx.merge(state_graphdef, state_pure)
    self.assertIsInstance(train_state, train_state_nnx.TrainStateNNX)
    # Same numeric values.
    self.assertTrue(jnp.allclose(train_state.model.linear.kernel.value, self.state.model.linear.kernel.value))


class TestInitStateFnIsCallable(unittest.TestCase):
  """For the Linen path setup_train_loop builds init_state_fn = partial(...).

  The NNX path uses a closure instead — confirm both forms have the
  zero-argument call contract create_checkpoint_manager / setup_training_state expect.
  """

  def test_nnx_init_state_fn_callable_with_no_args(self):
    tx = optax.sgd(0.01)

    def _create_model():
      return _Model(rngs=nnx.Rngs(0))

    def init_state_fn():
      model = _create_model()
      return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, tx, wrt=nnx.Param))

    state = init_state_fn()  # must not raise / require args
    self.assertIsInstance(state, train_state_nnx.TrainStateNNX)

  def test_linen_init_state_fn_is_partial_callable_with_no_args(self):
    """Sanity: the Linen-side `partial(init_initial_state, model, tx, config, is_training, init_rng)` form."""

    def init_initial_state(model, tx, config, is_training, init_rng):
      del model, tx, config, is_training, init_rng
      return "linen-state"

    init_state_fn = partial(init_initial_state, "model", "tx", "config", True, "rng")
    self.assertEqual(init_state_fn(), "linen-state")


if __name__ == "__main__":
  unittest.main()
