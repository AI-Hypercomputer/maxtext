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

"""Unit tests for add_entropy_to_checkpoint on NNX state."""

import unittest

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from maxtext.layers import train_state_nnx
from maxtext.utils.standalone_checkpointer import add_entropy_to_checkpoint


class _TinyModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.lin = nnx.Linear(4, 4, rngs=rngs)


def _expected_cos_sin(params_state):
  mu = jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), params_state)
  nu = jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), params_state)
  return mu, nu


class AddEntropyNNXTest(unittest.TestCase):

  def test_overwrites_adam_mu_and_nu(self):
    model = _TinyModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)

    params_before = nnx.state(model, nnx.Param)
    expected_mu, expected_nu = _expected_cos_sin(params_before)

    new_state = add_entropy_to_checkpoint(state)

    self.assertIs(new_state, state)  # mutated in place
    actual_mu = new_state.optimizer.opt_state[0].mu
    actual_nu = new_state.optimizer.opt_state[0].nu

    expected_mu_leaves = jax.tree_util.tree_leaves(expected_mu)
    expected_nu_leaves = jax.tree_util.tree_leaves(expected_nu)
    actual_mu_leaves = jax.tree_util.tree_leaves(actual_mu)
    actual_nu_leaves = jax.tree_util.tree_leaves(actual_nu)
    self.assertEqual(len(expected_mu_leaves), len(actual_mu_leaves))
    for e, a in zip(expected_mu_leaves, actual_mu_leaves):
      self.assertTrue(jnp.allclose(e, a))
    for e, a in zip(expected_nu_leaves, actual_nu_leaves):
      self.assertTrue(jnp.allclose(e, a))

  def test_does_not_mutate_model_params(self):
    model = _TinyModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)

    params_before = jax.tree_util.tree_map(jnp.array, nnx.state(model, nnx.Param).to_pure_dict())
    add_entropy_to_checkpoint(state)
    params_after = nnx.state(model, nnx.Param).to_pure_dict()

    for path, before in jax.tree_util.tree_leaves_with_path(params_before):
      after = params_after
      for key in path:
        after = after[key.key]
      self.assertTrue(jnp.array_equal(before, after))

  def test_works_on_split_nnx_state(self):
    """`setup_training_state` returns a flat `nnx.State`, not a `TrainStateNNX`."""
    model = _TinyModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train_state = train_state_nnx.TrainStateNNX(model, optimizer)

    _, split_state = nnx.split(train_state)

    new_state = add_entropy_to_checkpoint(split_state)
    self.assertIs(new_state, split_state)

    # mu should now be cos(1000 * params); params for a freshly initialized
    # nnx.Linear bias is 0 so cos(0) = 1.
    mu_leaves = jax.tree_util.tree_leaves(new_state.optimizer.opt_state[0].mu)
    nu_leaves = jax.tree_util.tree_leaves(new_state.optimizer.opt_state[0].nu)
    self.assertTrue(any(jnp.allclose(leaf, 1.0) for leaf in mu_leaves))  # cos(0)=1
    self.assertTrue(any(jnp.allclose(leaf, 0.0) for leaf in nu_leaves))  # sin(0)=0


if __name__ == "__main__":
  unittest.main()
