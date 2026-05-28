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

"""Unit tests for the NNX-specific helpers in maxtext.utils.sharding."""

from dataclasses import dataclass
import unittest

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from maxtext.common import train_state_nnx
from maxtext.utils import sharding
import numpy as np
import optax


@dataclass
class _Cfg:
  pure_nnx: bool = True
  shard_optimizer_over_data: bool = False


class _LinearNNX(nnx.Module):
  """Tiny NNX model with a single Linear layer for sharding tests."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 4, rngs=rngs)


def _build_state_mesh_shardings(model, tx):
  """Build an nnx.State of NamedShardings mirroring the TrainStateNNX layout.

  This emulates what get_abstract_state_nnx returns: an nnx.State whose leaves
  are nnx.Variable wrappers around NamedSharding objects.
  """
  optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
  state_obj = train_state_nnx.TrainStateNNX(model, optimizer)
  state = nnx.state(state_obj)
  mesh = Mesh(
      np.array(jax.local_devices()[:1]).reshape(1, 1), ("data", "model")
  )

  def _to_sharding(var):
    val = var.get_value()
    if not hasattr(val, "shape") or val.ndim == 0:
      pspec = PartitionSpec()
    elif val.ndim == 1:
      pspec = PartitionSpec("model")
    else:
      pspec = PartitionSpec("data", "model")
    return var.replace(NamedSharding(mesh, pspec))

  return jax.tree.map(
      _to_sharding, state, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )


class TestMaybeUpdateParamsShardingWithOptNNX(unittest.TestCase):
  """Cover the NNX branches of maybe_update_params_sharding_with_opt."""

  def setUp(self):
    self.model = _LinearNNX(rngs=nnx.Rngs(0))

  def test_dispatch_from_main_helper_when_pure_nnx(self):
    """maybe_update_params_sharding_with_opt should dispatch to the NNX variant."""
    cfg = _Cfg(pure_nnx=True, shard_optimizer_over_data=False)
    state_mesh_shardings = _build_state_mesh_shardings(
        self.model, optax.adam(1e-3)
    )
    prev, updated = sharding.maybe_update_params_sharding_with_opt(
        cfg, state_mesh_shardings
    )
    # prev is the param-only view (no rngs / non-Param nodes)
    self.assertIsInstance(prev, nnx.State)
    self.assertIn("linear", prev)
    # updated is unchanged because shard_optimizer_over_data=False
    self.assertIs(updated, state_mesh_shardings)

  def test_extract_param_only_skips_non_param_variables(self):
    """prev_params_shardings must contain Params only — RngKey/RngCount/OptVariable filtered out."""
    cfg = _Cfg(shard_optimizer_over_data=False)
    state_mesh_shardings = _build_state_mesh_shardings(
        self.model, optax.adam(1e-3)
    )
    prev, _ = sharding.maybe_update_params_sharding_with_opt_nnx(
        cfg, state_mesh_shardings
    )
    leaves = jax.tree.leaves(
        prev, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )
    # Every surviving leaf is wrapped as an nnx.Param.
    self.assertTrue(all(isinstance(leaf, nnx.Param) for leaf in leaves))
    # The model has linear.kernel and linear.bias — exactly two Param leaves.
    self.assertEqual(len(leaves), 2)

  def test_returns_unchanged_when_shard_optimizer_over_data_false(self):
    """When shard_optimizer_over_data=False, the second return value must be the input object."""
    cfg = _Cfg(shard_optimizer_over_data=False)
    state_mesh_shardings = _build_state_mesh_shardings(
        self.model, optax.adam(1e-3)
    )
    _, updated = sharding.maybe_update_params_sharding_with_opt_nnx(
        cfg, state_mesh_shardings
    )
    self.assertIs(updated, state_mesh_shardings)

  def test_zero1_propagates_mu_sharding_to_model_params(self):
    """Zero-1: model param shardings must be replaced with the optimizer mu shardings."""
    cfg = _Cfg(shard_optimizer_over_data=True)
    state_mesh_shardings = _build_state_mesh_shardings(
        self.model, optax.adam(1e-3)
    )

    # Mutate the optimizer mu leaves in place so the function picks up a distinct PartitionSpec.
    mesh = Mesh(
        np.array(jax.local_devices()[:1]).reshape(1, 1), ("data", "model")
    )
    target_pspec = PartitionSpec(("data", "model"))
    new_mu_sharding = NamedSharding(mesh, target_pspec)

    # After _build_state_mesh_shardings, every leaf's .value is a NamedSharding (no .shape),
    # so we just override every Variable leaf in mu in place.
    # After _build_state_mesh_shardings, every leaf's value is a NamedSharding (no .shape),
    # so we just override every Variable leaf in mu in place via set_value (modern API).
    mu_state = state_mesh_shardings.optimizer.opt_state[0]["mu"]
    for var in jax.tree.leaves(
        mu_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
    ):
      if isinstance(var, nnx.Variable):
        var.set_value(new_mu_sharding)

    _, updated = sharding.maybe_update_params_sharding_with_opt_nnx(
        cfg, state_mesh_shardings
    )

    # All Param leaves under updated.model must now share the new mu sharding.
    param_leaves = jax.tree.leaves(
        updated.model, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )
    param_leaves = [v for v in param_leaves if isinstance(v, nnx.Param)]
    self.assertGreater(len(param_leaves), 0)
    for leaf in param_leaves:
      self.assertEqual(leaf.get_value().spec, target_pspec)

  def test_raises_when_no_adam_state_present(self):
    """Stateless optimizers (e.g., SGD) have no mu — function must raise NotImplementedError."""
    cfg = _Cfg(shard_optimizer_over_data=True)
    state_mesh_shardings = _build_state_mesh_shardings(
        self.model, optax.sgd(1e-3)
    )
    with self.assertRaises(NotImplementedError):
      sharding.maybe_update_params_sharding_with_opt_nnx(
          cfg, state_mesh_shardings
      )

  def test_chained_optimizer_recursion_finds_adam_mu(self):
    """A nested optax.chain(clip, adam) wraps mu under multiple containers — recursion must find it."""
    cfg = _Cfg(shard_optimizer_over_data=True)
    chained = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    state_mesh_shardings = _build_state_mesh_shardings(self.model, chained)

    # Should not raise; verify update happens (params replaced with mu shardings).
    prev, updated = sharding.maybe_update_params_sharding_with_opt_nnx(
        cfg, state_mesh_shardings
    )
    self.assertIsInstance(prev, nnx.State)
    self.assertIsInstance(updated, nnx.State)
    # Same number of Param leaves before and after.
    n_prev = len(
        jax.tree.leaves(prev, is_leaf=lambda x: isinstance(x, nnx.Variable))
    )
    n_after = len([
        v
        for v in jax.tree.leaves(
            updated.model, is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        if isinstance(v, nnx.Param)
    ])
    self.assertEqual(n_prev, n_after)


if __name__ == "__main__":
  unittest.main()
