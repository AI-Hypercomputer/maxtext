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
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
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


def _create_2d_test_mesh(axis_names=("data", "model")):
  devices = jax.local_devices()
  num_devices = len(devices)
  if num_devices >= 4:
    mesh_devices = np.array(devices[:4]).reshape(2, 2)
  elif num_devices >= 2:
    mesh_devices = np.array(devices[:2]).reshape(2, 1)
  else:
    mesh_devices = np.array(devices[:1]).reshape(1, 1)
  return Mesh(devices=mesh_devices, axis_names=axis_names)


def _build_state_mesh_shardings(model, tx):
  """Build an nnx.State of NamedShardings mirroring the TrainStateNNX layout.

  This emulates what get_abstract_state_nnx returns: an nnx.State whose leaves
  are nnx.Variable wrappers around NamedSharding objects.
  """
  optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
  state_obj = train_state_nnx.TrainStateNNX(model, optimizer)
  state = nnx.state(state_obj)
  mesh = _create_2d_test_mesh()

  def _to_sharding(var):
    val = var.get_value()
    if not hasattr(val, "shape") or val.ndim == 0:
      pspec = PartitionSpec()
    elif val.ndim == 1:
      pspec = PartitionSpec("model")
    else:
      pspec = PartitionSpec("data", "model")
    return var.replace(NamedSharding(mesh, pspec))

  return jax.tree.map(_to_sharding, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


class TestMaybeUpdateParamsShardingWithOptNNX(unittest.TestCase):
  """Cover the NNX branches of maybe_update_params_sharding_with_opt."""

  def setUp(self):
    self.model = _LinearNNX(rngs=nnx.Rngs(0))

  def test_dispatch_from_main_helper_when_pure_nnx(self):
    """maybe_update_params_sharding_with_opt should dispatch to the NNX variant."""
    cfg = _Cfg(pure_nnx=True, shard_optimizer_over_data=False)
    state_mesh_shardings = _build_state_mesh_shardings(self.model, optax.adam(1e-3))
    prev, updated = sharding.maybe_update_params_sharding_with_opt(cfg, state_mesh_shardings)
    # prev is the param-only view (no rngs / non-Param nodes)
    self.assertIsInstance(prev, nnx.State)
    self.assertIn("linear", prev)
    # updated is unchanged because shard_optimizer_over_data=False
    self.assertIs(updated, state_mesh_shardings)

  def test_extract_param_only_skips_non_param_variables(self):
    """prev_params_shardings must contain Params only — RngKey/RngCount/OptVariable filtered out."""
    cfg = _Cfg(shard_optimizer_over_data=False)
    state_mesh_shardings = _build_state_mesh_shardings(self.model, optax.adam(1e-3))
    prev, _ = sharding.maybe_update_params_sharding_with_opt_nnx(cfg, state_mesh_shardings)
    leaves = jax.tree.leaves(prev, is_leaf=lambda x: isinstance(x, nnx.Variable))
    # Every surviving leaf is wrapped as an nnx.Param.
    self.assertTrue(all(isinstance(leaf, nnx.Param) for leaf in leaves))
    # The model has linear.kernel and linear.bias — exactly two Param leaves.
    self.assertEqual(len(leaves), 2)

  def test_returns_unchanged_when_shard_optimizer_over_data_false(self):
    """When shard_optimizer_over_data=False, the second return value must be the input object."""
    cfg = _Cfg(shard_optimizer_over_data=False)
    state_mesh_shardings = _build_state_mesh_shardings(self.model, optax.adam(1e-3))
    _, updated = sharding.maybe_update_params_sharding_with_opt_nnx(cfg, state_mesh_shardings)
    self.assertIs(updated, state_mesh_shardings)

  def test_zero1_propagates_mu_sharding_to_model_params(self):
    """Zero-1: model param shardings must be replaced with the optimizer mu shardings."""
    cfg = _Cfg(shard_optimizer_over_data=True)
    state_mesh_shardings = _build_state_mesh_shardings(self.model, optax.adam(1e-3))

    # Mutate the optimizer mu leaves in place so the function picks up a distinct PartitionSpec.
    mesh = _create_2d_test_mesh()
    target_pspec = PartitionSpec(("data", "model"))
    new_mu_sharding = NamedSharding(mesh, target_pspec)

    # After _build_state_mesh_shardings, every leaf's .value is a NamedSharding (no .shape),
    # so we just override every Variable leaf in mu in place.
    # After _build_state_mesh_shardings, every leaf's value is a NamedSharding (no .shape),
    # so we just override every Variable leaf in mu in place via set_value (modern API).
    mu_state = state_mesh_shardings.optimizer.opt_state[0]["mu"]
    for var in jax.tree.leaves(mu_state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
      if isinstance(var, nnx.Variable):
        var.set_value(new_mu_sharding)

    _, updated = sharding.maybe_update_params_sharding_with_opt_nnx(cfg, state_mesh_shardings)

    # All Param leaves under updated.model must now share the new mu sharding.
    param_leaves = jax.tree.leaves(updated.model, is_leaf=lambda x: isinstance(x, nnx.Variable))
    param_leaves = [v for v in param_leaves if isinstance(v, nnx.Param)]
    self.assertGreater(len(param_leaves), 0)
    for leaf in param_leaves:
      self.assertEqual(leaf.get_value().spec, target_pspec)

  def test_raises_when_no_adam_state_present(self):
    """Stateless optimizers (e.g., SGD) have no mu — function must raise NotImplementedError."""
    cfg = _Cfg(shard_optimizer_over_data=True)
    state_mesh_shardings = _build_state_mesh_shardings(self.model, optax.sgd(1e-3))
    with self.assertRaises(NotImplementedError):
      sharding.maybe_update_params_sharding_with_opt_nnx(cfg, state_mesh_shardings)

  def test_chained_optimizer_recursion_finds_adam_mu(self):
    """A nested optax.chain(clip, adam) wraps mu under multiple containers — recursion must find it."""
    cfg = _Cfg(shard_optimizer_over_data=True)
    chained = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    state_mesh_shardings = _build_state_mesh_shardings(self.model, chained)

    # Should not raise; verify update happens (params replaced with mu shardings).
    prev, updated = sharding.maybe_update_params_sharding_with_opt_nnx(cfg, state_mesh_shardings)
    self.assertIsInstance(prev, nnx.State)
    self.assertIsInstance(updated, nnx.State)
    # Same number of Param leaves before and after.
    n_prev = len(jax.tree.leaves(prev, is_leaf=lambda x: isinstance(x, nnx.Variable)))
    n_after = len(
        [
            v
            for v in jax.tree.leaves(updated.model, is_leaf=lambda x: isinstance(x, nnx.Variable))
            if isinstance(v, nnx.Param)
        ]
    )
    self.assertEqual(n_prev, n_after)


class TestNnxConstructNamedSharding(unittest.TestCase):
  """Unit tests for nnx_construct_named_sharding covering every branch.

  The helper resolves a NamedSharding for each NNX Variable inside an nnx.State
  and
  — unlike flax.nnx.spmd.get_var_pspec — also inserts the `nnx.PARTITION_NAME`
  axis at
  `param_scan_axis` when scanned-layers metadata is present.
  """

  def setUp(self):
    # Mesh needs to contain every axis name the tests reference in partition specs.
    self.mesh = _create_2d_test_mesh(axis_names=("fsdp", "stage"))
    # In local test environments (e.g. single-device CPU), all mesh axes have size 1.
    # We stub remove_size_one_mesh_axis to act as a no-op so that resolved physical PartitionSpecs
    # are returned unreduced (e.g. retaining "fsdp", "stage", etc.), allowing us to verify naming
    # resolution. The actual size-one axis removal is tested separately in TestGetNNXNamedShardingSizeOneAxes.
    self._old_remove_size_one_mesh_axis = sharding.remove_size_one_mesh_axis
    sharding.remove_size_one_mesh_axis = lambda spec, mesh: spec

  def tearDown(self):
    sharding.remove_size_one_mesh_axis = self._old_remove_size_one_mesh_axis

  def _build_state(self, **variables):
    """Wrap a dict of {key: nnx.Variable} in an nnx.State for tree traversal."""
    return nnx.State(variables)

  def _run(self, state):
    return sharding.nnx_construct_named_sharding(state, self.mesh)

  def test_scan_axis_inserted_at_param_scan_axis(self):
    """When PARTITION_NAME is present, the partition name is inserted at `param_scan_axis`."""
    rules = (("layers", "stage"), ("fsdp", "fsdp"))
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      v = nnx.Param(
          jnp.zeros((3, 4, 8)),
          out_sharding=(None, "fsdp"),
          **{nnx.PARTITION_NAME: "layers", "param_scan_axis": 1},
      )
      out = self._run(self._build_state(w=v))
      result_sharding = out["w"].get_value()
      self.assertIsInstance(result_sharding, NamedSharding)
      # 'layers' resolves to physical axis 'stage' and is inserted at position 1 (param_scan_axis=1).
      self.assertEqual(result_sharding.spec, PartitionSpec(None, "stage", "fsdp"))

  def test_scan_axis_not_inserted_when_already_present(self):
    """Guard against double-insertion when partition_name is already in out_sharding."""
    rules = (("layers", "stage"), ("fsdp", "fsdp"))
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      v = nnx.Param(
          jnp.zeros((2, 2, 2)),
          out_sharding=("layers", None, "fsdp"),
          **{nnx.PARTITION_NAME: "layers", "param_scan_axis": 0},
      )
      out = self._run(self._build_state(w=v))
      result_sharding = out["w"].get_value()
      # 'stage' must appear exactly once — the same PartitionSpec we started with.
      self.assertEqual(result_sharding.spec, PartitionSpec("stage", None, "fsdp"))

  def test_masked_node_preserved_as_is(self):
    """Values without a .shape attribute (e.g., optax.MaskedNode) are returned unchanged."""
    masked = nnx.Variable(optax.MaskedNode())
    state = self._build_state(masked=masked)
    out = self._run(state)
    # The leaf must be the original Variable, not a NamedSharding wrapper.
    self.assertIs(out["masked"], masked)

  def test_empty_out_sharding_yields_empty_pspec(self):
    """A Variable without any sharding metadata should resolve to PartitionSpec()."""
    with jax.set_mesh(self.mesh):
      # No out_sharding/sharding_names/sharding metadata → falsy → PartitionSpec()
      v = nnx.Param(jnp.zeros((4,)))
    out = self._run(self._build_state(w=v))
    result_sharding = out["w"].get_value()
    self.assertIsInstance(result_sharding, NamedSharding)
    self.assertEqual(result_sharding.spec, PartitionSpec())

  def test_string_out_sharding_is_wrapped_into_tuple(self):
    """A single-string out_sharding value should still produce a valid PartitionSpec."""
    rules = (("layers", "stage"), ("fsdp", "fsdp"))
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      v = nnx.Param(
          jnp.zeros((4,)),
          out_sharding="fsdp",
          **{nnx.PARTITION_NAME: "layers", "param_scan_axis": 0},
      )
      out = self._run(self._build_state(w=v))
      result_sharding = out["w"].get_value()
      # The single string 'fsdp' is turned into a list, and 'layers' (resolving to 'stage') is prepended.
      self.assertEqual(result_sharding.spec, PartitionSpec("stage", "fsdp"))

  def test_sequential_matching_first_match_wins(self):
    """Multiple rules for the same logical axis are matched sequentially, first-match-wins."""
    # We define rules for 'embed' mapping to 'fsdp' (specific) then 'stage' (fallback)
    rules = (
        ("embed", "fsdp"),
        ("embed", "stage"),
    )
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      v = nnx.Param(
          jnp.zeros((3,)),
          out_sharding=("embed",),
      )
      out = self._run(self._build_state(w=v))
      result_sharding = out["w"].get_value()
      # 'embed' must match the first rule ('fsdp'), not the second ('stage').
      self.assertEqual(result_sharding.spec, PartitionSpec("fsdp"))

  def test_prevents_duplicate_physical_axes(self):
    """If multiple dimensions map to the same physical axis, the subsequent ones are skipped (mapped to None)."""
    # Setup rules where 'embed' maps to 'fsdp' and 'mlp' also maps to 'fsdp'.
    rules = (
        ("embed", "fsdp"),
        ("mlp", "fsdp"),
    )
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      v = nnx.Param(
          jnp.zeros((3, 4)),
          out_sharding=("embed", "mlp"),
      )
      out = self._run(self._build_state(w=v))
      result_sharding = out["w"].get_value()
      self.assertIsInstance(result_sharding, NamedSharding)
      # Expected: Dim 0 matches 'embed' -> 'fsdp'.
      # Dim 1 tries 'mlp' -> 'fsdp', but 'fsdp' is already assigned to Dim 0.
      # So it skips the rule and falls back to matching nothing -> None.
      self.assertEqual(result_sharding.spec, PartitionSpec("fsdp", None))

  def test_fallback_to_next_physical_axis_when_duplicated(self):
    """When a physical axis is already assigned, fallback priority rules should map to the next available physical option."""
    # Setup rules where 'embed' maps to 'fsdp', and 'mlp' maps to 'fsdp' (priority 1) or 'stage' (priority 2).
    rules = (
        ("embed", "fsdp"),
        ("mlp", "fsdp"),
        ("mlp", "stage"),
    )
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(rules):
      v = nnx.Param(
          jnp.zeros((3, 4)),
          out_sharding=("embed", "mlp"),
      )
      out = self._run(self._build_state(w=v))
      result_sharding = out["w"].get_value()
      self.assertIsInstance(result_sharding, NamedSharding)
      # Expected: Dim 0 matches 'embed' -> 'fsdp'.
      # Dim 1 tries 'mlp' -> 'fsdp', but 'fsdp' is already assigned to Dim 0.
      # So it falls to the next item in the list -> 'stage'.
      self.assertEqual(result_sharding.spec, PartitionSpec("fsdp", "stage"))

  def test_resolves_when_context_rules_is_none(self):
    """When context_rules is None but local_rules are defined, resolution should succeed."""
    # Ensure get_logical_axis_rules() returns None (which is the default outside axis_rules)
    # We define local rules on the variable metadata.
    with jax.set_mesh(self.mesh):
      v = nnx.Param(
          jnp.zeros((3,)),
          out_sharding=("embed",),
          sharding_rules=(("embed", "fsdp"),),
      )
    out = self._run(self._build_state(w=v))
    result_sharding = out["w"].get_value()
    # 'embed' must match the local rules even when context_rules is None.
    self.assertEqual(result_sharding.spec, PartitionSpec("fsdp"))

  def test_composite_pytree_variable_resolved_to_replicated_shardings(self):
    """A Variable holding a composite pytree (e.g.

    tuple of arrays) is resolved to replicated NamedShardings.
    """
    with jax.set_mesh(self.mesh):
      v = nnx.Variable((jnp.zeros((2, 2)), jnp.zeros((3, 3))))
    out = self._run(self._build_state(w=v))
    result = out["w"].get_value()
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)
    self.assertIsInstance(result[0], NamedSharding)
    self.assertIsInstance(result[1], NamedSharding)
    self.assertEqual(result[0].spec, PartitionSpec())
    self.assertEqual(result[1].spec, PartitionSpec())

  def test_rules_merged_when_both_context_and_local_rules_present(self):
    """When both local rules and context rules are present, they are concatenated in order of local then context."""
    # Local rules map 'embed' to 'stage'. Context rules map 'embed' to 'fsdp'.
    # Because local rules come first, 'embed' should resolve to 'stage'.
    context_rules = (("embed", "fsdp"),)
    with jax.set_mesh(self.mesh), nn_partitioning.axis_rules(context_rules):
      v = nnx.Param(
          jnp.zeros((3,)),
          out_sharding=("embed",),
          sharding_rules=(("embed", "stage"),),
      )
    out = self._run(self._build_state(w=v))
    result_sharding = out["w"].get_value()
    self.assertEqual(result_sharding.spec, PartitionSpec("stage"))

  def test_removes_size_one_mesh_axes_no_rules(self):
    """When no rules are defined but mesh is present, size-1 physical axes in the spec are removed."""
    # Temporarily restore the original remove_size_one_mesh_axis function
    sharding.remove_size_one_mesh_axis = self._old_remove_size_one_mesh_axis
    try:
      # Use an explicit 1x1 mesh so physical axis 'fsdp' has size 1 deterministically.
      mesh_1x1 = Mesh(np.array(jax.local_devices()[:1]).reshape(1, 1), ("fsdp", "stage"))
      with jax.set_mesh(mesh_1x1):
        v = nnx.Param(
            jnp.zeros((4,)),
            out_sharding=("fsdp",),
        )
      out_v = sharding.get_nnx_var_named_sharding_with_scan_axis(v, mesh_1x1)
      result_sharding = out_v.get_value()
      # 'fsdp' has size 1, so it gets reduced to None.
      self.assertEqual(result_sharding.spec, PartitionSpec(None))
    finally:
      sharding.remove_size_one_mesh_axis = lambda spec, mesh: spec

  def test_removes_size_one_mesh_axes(self):
    """When remove_size_one_mesh_axis is active, physical axes with size 1 are removed (mapped to None)."""
    # Temporarily restore the original remove_size_one_mesh_axis function
    sharding.remove_size_one_mesh_axis = self._old_remove_size_one_mesh_axis
    try:
      # Use an explicit 1x1 mesh so physical axes 'fsdp' and 'stage' have size 1 deterministically.
      mesh_1x1 = Mesh(np.array(jax.local_devices()[:1]).reshape(1, 1), ("fsdp", "stage"))
      rules = (("embed", "fsdp"), ("layers", "stage"))
      with jax.set_mesh(mesh_1x1), nn_partitioning.axis_rules(rules):
        v = nnx.Param(
            jnp.zeros((3, 4)),
            out_sharding=("embed",),
            **{nnx.PARTITION_NAME: "layers", "param_scan_axis": 1},
        )
        # Resolve sharding
        out_v = sharding.get_nnx_var_named_sharding_with_scan_axis(v, mesh_1x1)
        result_sharding = out_v.get_value()
        self.assertIsInstance(result_sharding, NamedSharding)
        # Expected: P("fsdp", "stage") gets reduced to P(None, None) since both fsdp and stage have size 1.
        self.assertEqual(result_sharding.spec, PartitionSpec(None, None))
    finally:
      # Re-apply the stub to keep other tests working
      sharding.remove_size_one_mesh_axis = lambda spec, mesh: spec


if __name__ == "__main__":
  unittest.main()
