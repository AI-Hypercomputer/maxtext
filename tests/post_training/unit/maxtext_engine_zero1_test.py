# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`shard_optimizer_over_data` (Zero-1) in `MaxTextTrainingEngine`.

The flag used to be read only by `gradient_accumulation.py`, which the engine does not go
through, so setting it here allocated a fully replicated optimizer and said nothing. It now
shards the parameter-shaped optimizer state over the data axis and does the update on those
slices, gathering the new parameters back at the end.

Like the deferral it pairs with, this is a change nothing functional depends on -- get it
wrong and the model still trains, just without the saving. So the tests assert on where the
arrays actually are (`sharding.spec`, and the shard each device holds) and on the compiled
HLO, and each such assertion is mirrored by the same probe run with the flag off.
"""

import os

# Must precede the first JAX import: a data-parallel mesh needs more than one device, and
# the CPU backend reads this only at initialization.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import re  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position

from absl.testing import absltest  # pylint: disable=wrong-import-position
from flax import nnx  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
from maxtext.training_engine import maxtext_engine  # pylint: disable=wrong-import-position
from maxtext.utils import maxtext_utils  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position

# The tiny-real-decoder rig this shares with the deferral it composes with: same model, same
# mesh, same batch. Reusing it is the point -- the two features have to hold on one config.
from tests.post_training.unit.maxtext_engine_deferred_all_reduce_test import (  # pylint: disable=wrong-import-position
    _REQUIRED_DEVICES,
    _KernelHlo,
    _array_all_reduces,
    _batch,
    _config,
    _no_deferral,
)

# training_engine imports tunix, so these tests need the post-training dependency bundle.
pytestmark = [pytest.mark.post_training]

_DATA = maxtext_engine._DATA_AXIS  # pylint: disable=protected-access

# The result shape of an all-gather in optimized HLO, in both the fused and the async form.
_ALL_GATHER = re.compile(r"=\s*(.+?)\s+all-gather(?:-start|-done)?\(")
# The dimensions inside one `f32[128,64]{1,0}`; a tupled result yields one match per element.
_SHAPE_DIMS = re.compile(r"\[([\d,]*)\]")


def _gathered_elements(hlo: str) -> int:
  """Total result size of every all-gather in `hlo`, as a stand-in for gathered volume.

  Counting instructions would be brittle -- XLA fuses and splits them freely -- and the
  absolute number here means little, since an async gather's start and done both count. Only
  the difference against the same kernel compiled without Zero-1 is ever asserted on, and
  both sides of that are counted the same way.
  """
  total = 0
  for line in hlo.splitlines():
    if match := _ALL_GATHER.search(line):
      for dims in _SHAPE_DIMS.findall(match.group(1)):
        if dims:
          total += int(np.prod([int(dim) for dim in dims.split(",")]))
  return total


def _axes(spec) -> list[str]:
  """The mesh axes a `PartitionSpec` names.

  A `PartitionSpec` is a pytree *leaf*, so flattening one gives back the spec itself; its
  entries have to be opened first, and they nest -- a dimension sharded over two axes is a
  tuple. The same trap the guard in `add_data_to_sharding` was written with.
  """
  return jax.tree.leaves(tuple(spec))


def _zero1_config(**overrides):
  """The shared config with Zero-1 on and a stateful optimizer to shard."""
  # SGD, the shared default, carries no parameter-shaped state at all, so Zero-1 would have
  # nothing to move and every assertion below would pass vacuously.
  overrides.setdefault("opt_type", "adamw")
  return _config(shard_optimizer_over_data=True, **overrides)


def _moments(engine):
  """`{path: array}` for every parameter-shaped optimizer moment in the engine's state."""
  _, state_pure = nnx.split(engine.state)
  return {
      jax.tree_util.keystr(path): leaf
      for path, leaf in jax.tree_util.tree_leaves_with_path(state_pure)
      if "['mu']" in jax.tree_util.keystr(path) or "['nu']" in jax.tree_util.keystr(path)
  }


def _params(engine):
  """`{path: array}` for the model's parameters."""
  return {
      jax.tree_util.keystr(path): leaf
      for path, leaf in jax.tree.flatten_with_path(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param)))[0]
  }


@unittest.skipIf(
    jax.device_count() < _REQUIRED_DEVICES,
    f"needs {_REQUIRED_DEVICES} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}",
)
class Zero1GateTest(absltest.TestCase):
  """`_zero1_active` decides whether the engine can honour the flag. It must decline widely."""

  def test_declines_when_the_flag_is_off(self):
    cfg = _config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertIsNotNone(maxtext_engine._zero1_active(cfg, mesh))  # pylint: disable=protected-access

  def test_opens_on_an_explicit_data_parallel_mesh(self):
    cfg = _zero1_config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertIsNone(maxtext_engine._zero1_active(cfg, mesh))  # pylint: disable=protected-access

  def test_declines_under_auto_shard_mode(self):
    """Under `auto` the reshards are hints GSPMD may ignore, which would replicate silently."""
    cfg = _zero1_config(shard_mode="auto")
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertIn("explicit", maxtext_engine._zero1_active(cfg, mesh))  # pylint: disable=protected-access

  def test_declines_on_an_auto_axis_mesh_even_in_explicit_mode(self):
    """A caller can hand the engine a bare `jax.sharding.Mesh` whatever `shard_mode` says."""
    cfg = _zero1_config()
    explicit_mesh = maxtext_utils.get_mesh_from_config(cfg)
    auto_mesh = jax.sharding.Mesh(explicit_mesh.devices, explicit_mesh.axis_names)

    self.assertIn("Explicit", maxtext_engine._zero1_active(cfg, auto_mesh))  # pylint: disable=protected-access

  def test_declines_when_there_are_no_data_replicas(self):
    cfg = _zero1_config(ici_data_parallelism=1, ici_tensor_parallelism=_REQUIRED_DEVICES)
    mesh = maxtext_utils.get_mesh_from_config(cfg)

    self.assertIn(_DATA, maxtext_engine._zero1_active(cfg, mesh))  # pylint: disable=protected-access

  def test_declines_without_a_mesh(self):
    self.assertIn("mesh", maxtext_engine._zero1_active(_zero1_config(), None))  # pylint: disable=protected-access


@unittest.skipIf(
    jax.device_count() < _REQUIRED_DEVICES,
    f"needs {_REQUIRED_DEVICES} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}",
)
class Zero1ShardingTest(absltest.TestCase):
  """`_zero1_sharding` places one leaf. Everything Zero-1 moves goes through it."""

  def setUp(self):
    super().setUp()
    self.mesh = maxtext_utils.get_mesh_from_config(_zero1_config())

  def _replicated(self, rank):
    return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*(None,) * rank))

  def _place(self, shape):
    return maxtext_engine._zero1_sharding(  # pylint: disable=protected-access
        self.mesh, jax.ShapeDtypeStruct(shape, jax.numpy.float32), self._replicated(len(shape))
    )

  def test_adds_the_data_axis_to_the_first_dimension_that_divides(self):
    self.assertEqual(self._place((128, 64)).spec, jax.sharding.PartitionSpec(_DATA, None))

  def test_skips_a_dimension_the_data_axis_does_not_divide(self):
    self.assertEqual(self._place((3, 64)).spec, jax.sharding.PartitionSpec(None, _DATA))

  def test_leaves_a_scalar_alone(self):
    """`adamw`'s step `count`, and every rng counter beside it. Nothing to slice."""
    self.assertIsNone(self._place(()))

  def test_leaves_a_shape_no_dimension_of_which_divides_alone(self):
    self.assertIsNone(self._place((3, 5)))

  def test_leaves_a_leaf_already_sharded_over_data_alone(self):
    already = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(_DATA, None))

    self.assertIsNone(
        maxtext_engine._zero1_sharding(  # pylint: disable=protected-access
            self.mesh, jax.ShapeDtypeStruct((128, 64), jax.numpy.float32), already
        )
    )


@pytest.mark.integration_test
@unittest.skipIf(
    jax.device_count() < _REQUIRED_DEVICES,
    f"needs {_REQUIRED_DEVICES} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_REQUIRED_DEVICES}",
)
class Zero1Test(absltest.TestCase):
  """End to end on a real decoder: where the optimizer state sits, and what the weights do."""

  def _run(self, micro_batches: int = 2, steps: int = 2, cfg=None, probe: bool = False):
    """Runs `steps` optimizer steps of `micro_batches` each.

    Returns `(engine, {kernel: optimized hlo})`, the HLO empty unless `probe`. Reading it
    means lowering the kernel again, and the `reduced` tag the fwd/bwd kernels apply needs
    the mesh to be set for that -- so it is read here, inside the context, not by the caller.
    """
    cfg = cfg if cfg is not None else _zero1_config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    kernels = {"first": "_compiled_fwd_bwd", "accum": "_compiled_fwd_bwd_accum", "update": "_compiled_update"}
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      probes = {name: _KernelHlo(engine, attr) for name, attr in kernels.items()} if probe else {}
      for step in range(steps):
        for micro in range(micro_batches):
          engine.fwd_bwd(_batch(cfg, step * micro_batches + micro))
        engine.update()
      return engine, {name: probe_for.text() for name, probe_for in probes.items()}

  def test_the_gate_opens_on_this_configuration(self):
    """Guards every other test in this class: without this they would all pass vacuously."""
    engine, _ = self._run(micro_batches=1, steps=1)

    self.assertIsNotNone(
        engine._zero1_params_shardings,  # pylint: disable=protected-access
        "Zero-1 never engaged, so nothing below is testing it",
    )

  def test_the_optimizer_moments_are_sharded_over_the_data_axis(self):
    """The saving itself: each replica holds and updates 1/N of every moment."""
    engine, _ = self._run()

    moments = _moments(engine)
    self.assertNotEmpty(moments, "adamw kept no parameter-shaped state, so there is nothing to shard")
    for path, leaf in moments.items():
      self.assertIn(_DATA, _axes(leaf.sharding.spec), f"{path} is not sharded over {_DATA!r}")
      shard = leaf.addressable_shards[0].data.shape
      self.assertEqual(
          np.prod(shard) * _REQUIRED_DEVICES,
          np.prod(leaf.shape),
          f"{path} claims to be sharded but each device still holds {shard} of {leaf.shape}",
      )

  def test_the_parameters_themselves_stay_replicated(self):
    """Zero-1, not Zero-2/3: only the optimizer state is stored sharded.

    The forward pass wants whole parameters and every kernel signature is unchanged, so the
    slicing lives entirely inside `update()`.
    """
    engine, _ = self._run()

    for path, leaf in _params(engine).items():
      self.assertNotIn(_DATA, _axes(leaf.sharding.spec), f"parameter {path} came back sharded")

  def test_without_the_flag_the_moments_stay_replicated(self):
    """Proves the probes above can fail. Same model, same optimizer, flag off."""
    engine, _ = self._run(cfg=_config(opt_type="adamw"))

    self.assertIsNone(engine._zero1_params_shardings)  # pylint: disable=protected-access
    for path, leaf in _moments(engine).items():
      self.assertNotIn(_DATA, _axes(leaf.sharding.spec), f"{path} is sharded with the flag off")

  def test_zero1_costs_one_all_gather_in_update_and_nothing_per_micro_batch(self):
    """Where the traffic Zero-1 adds is, and where it must not be.

    Each replica updates its own slice, so the new parameters have to be gathered before the
    next forward pass -- once per optimizer step, in `update()`. If that gather ever appears
    in a micro-batch kernel instead, Zero-1 has become a per-micro-batch cost.
    """
    baseline, baseline_probes = self._run(cfg=_config(opt_type="adamw"), probe=True)
    engine, probes = self._run(probe=True)

    added = {k: _gathered_elements(probes[k]) - _gathered_elements(baseline_probes[k]) for k in probes}
    self.assertGreater(added["update"], 0, "update() gathers nothing, so the parameters were never sharded")
    self.assertEqual(added["first"], 0, "Zero-1 added an all-gather to the first micro-batch")
    self.assertEqual(added["accum"], 0, "Zero-1 added an all-gather to the accumulating micro-batches")
    self.assertIsNotNone(engine._zero1_params_shardings)  # pylint: disable=protected-access
    self.assertIsNone(baseline._zero1_params_shardings)  # pylint: disable=protected-access

  def test_zero1_composes_with_the_deferred_all_reduce(self):
    """The pair is the point: one reduction per step, on 1/N of the optimizer.

    Zero-1 reshards the gradients onto the moments' layout inside `update()`, which is the
    same reshard that discharges the deferral's `unreduced` tag. So turning it on must not
    put parameter-sized traffic back into the micro-batches.
    """
    engine, probes = self._run(probe=True)

    self.assertIsNotNone(engine._plain_grad_shardings)  # pylint: disable=protected-access
    self.assertIsNotNone(engine._zero1_params_shardings)  # pylint: disable=protected-access
    self.assertEmpty(_array_all_reduces(probes["first"]))
    self.assertEmpty(_array_all_reduces(probes["accum"]))
    self.assertNotEmpty(_array_all_reduces(probes["update"]), "the gradients are never reduced at all")

  def test_zero1_does_not_change_the_weights(self):
    """Same optimizer arithmetic, run elementwise on disjoint slices instead of on all of it.

    Every operation `adamw` applies is elementwise in the parameter, so splitting the tensor
    across replicas changes nothing about the result -- on this CPU mesh, not even the last
    bit. The tolerance is for accelerators, where the gather is not exact.
    """
    zero1, _ = self._run(micro_batches=3, steps=3)
    baseline, _ = self._run(micro_batches=3, steps=3, cfg=_config(opt_type="adamw"))

    want, got = _params(baseline), _params(zero1)
    self.assertEqual(sorted(want), sorted(got))
    for path, expected in want.items():
      np.testing.assert_allclose(
          np.asarray(got[path]), np.asarray(expected), rtol=1e-6, atol=1e-6, err_msg=f"parameter {path}"
      )

  def test_gradient_clipping_still_sees_the_whole_gradient(self):
    """The one part of `update()` that is not elementwise.

    `l2norm_pytree` sums squares over every element, and under Zero-1 those elements are
    spread across replicas. If the sum stayed replica-local the norm would come out too
    small by a factor of N and clipping would barely bite; the weights would then differ.
    """
    clipped = {"gradient_clipping_threshold": 1e-4, "opt_type": "adamw"}
    zero1, _ = self._run(micro_batches=2, steps=2, cfg=_zero1_config(**clipped))
    baseline, _ = self._run(micro_batches=2, steps=2, cfg=_config(**clipped))

    want, got = _params(baseline), _params(zero1)
    for path, expected in want.items():
      np.testing.assert_allclose(
          np.asarray(got[path]), np.asarray(expected), rtol=1e-6, atol=1e-6, err_msg=f"parameter {path}"
      )

  def test_zero1_alone_is_enough_without_the_deferral(self):
    """The two are independent. With the deferral withheld, Zero-1 still shards the moments."""
    with _no_deferral():
      engine, _ = self._run()

    self.assertIsNone(engine._plain_grad_shardings)  # pylint: disable=protected-access
    self.assertIsNotNone(engine._zero1_params_shardings)  # pylint: disable=protected-access
    for path, leaf in _moments(engine).items():
      self.assertIn(_DATA, _axes(leaf.sharding.spec), f"{path} is not sharded over {_DATA!r}")

  def test_a_recompile_leaves_the_already_sharded_moments_where_they_are(self):
    """A second batch shape re-enters `_compile_for_batch`, which re-places the moments.

    They are already on the Zero-1 layout by then, so the placement has to be a no-op --
    adding the data axis a second time produces `P(('data', 'data'), ...)`, which
    `NamedSharding` rejects outright and which would take the whole engine down.
    """
    cfg = _zero1_config()
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      engine.fwd_bwd(_batch(cfg, 0))
      engine.update()
      # A shorter sequence: a different dynamic batch shape, so `fwd_bwd` recompiles.
      short = {name: value[:, : cfg.max_target_length // 2] for name, value in _batch(cfg, 1).items()}
      engine.fwd_bwd(short)
      engine.update()

    for path, leaf in _moments(engine).items():
      self.assertEqual(_axes(leaf.sharding.spec).count(_DATA), 1, f"{path} was sharded over {_DATA!r} twice")

  def test_the_moments_survive_a_checkpoint_round_trip_still_sharded(self):
    """Restoring does not recompile, so what Orbax hands back has to land where it left.

    `_compiled_update` was built against sharded moments. If they came back replicated the
    resumed step would die on an `in_shardings` mismatch -- and if it somehow did not, the
    optimizer would silently be replicated again for the rest of the run.
    """
    output_dir = self.create_tempdir().full_path
    cfg = _zero1_config(
        enable_checkpointing=True,
        base_output_directory=output_dir,
        async_checkpointing=False,
        checkpoint_period=1,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      engine.compile(_batch(cfg, 0))
      # One full step first, so the moments are non-zero and actually carry information.
      engine.fwd_bwd(_batch(cfg, 0))
      engine.update()
      # Then mid-step: one micro-batch in, no update() yet.
      engine.fwd_bwd(_batch(cfg, 1))
      engine.save_checkpoint(metadata={"marker": 1}, force=True)
      engine._checkpoint_manager.wait_until_finished()  # pylint: disable=protected-access

      engine.restore_checkpoint()
      restored = _moments(engine)
      engine.update()
      resumed = _params(engine)

    self.assertNotEmpty(restored, "no moments came back, so nothing below is checked")
    for path, leaf in restored.items():
      self.assertIn(_DATA, _axes(leaf.sharding.spec), f"{path} came back replicated from the checkpoint")

    uninterrupted, _ = self._run(micro_batches=1, steps=2)
    for path, expected in _params(uninterrupted).items():
      np.testing.assert_allclose(
          np.asarray(resumed[path]), np.asarray(expected), rtol=1e-6, atol=1e-6, err_msg=f"parameter {path}"
      )

  def test_a_request_the_engine_cannot_honour_is_reported_once(self):
    """The failure mode this replaces was silence: the flag set, and nothing done about it."""
    cfg = _zero1_config(shard_mode="auto")
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
      with self.assertLogs(level="WARNING") as logs:
        engine.compile(_batch(cfg, 0))
        engine.fwd_bwd(_batch(cfg, 0))
        engine.update()

    declined = [line for line in logs.output if "Zero-1" in line]
    self.assertLen(declined, 1, f"expected exactly one Zero-1 warning, got {declined}")
    self.assertIn("shard_mode=explicit", declined[0])
    self.assertIsNone(engine._zero1_params_shardings)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
