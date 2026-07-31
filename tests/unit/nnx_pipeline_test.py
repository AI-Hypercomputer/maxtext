# Copyright 2026 Google LLC
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

"""Unit tests for the NNX pipeline (pipeline.py NNXPipeline / NNXCircularPipeline).

The integration pipeline tests are all tpu_only, so the NNX pipeline __call__ paths had no unit
coverage. These need 4 devices (a single CPU split into 4 via XLA_FLAGS, or TPU chips in CI) and lock
the migration-parity fixes:
  - non_trainable handling: BOTH schedules carry it through the iteration scan -- non-circular via a
    5-way split plus the jax.lax.scan carry tuple, circular via the "carry_state" collection.
    It replaces the prior RngState-only assert, which crashed on any
    non_trainable variable, and a later broadcast-and-discard that silently lost mutations;
  - unconditional circular repeat-level remat (output transparency vs the iteration-remat flag).

Run standalone so the 4-device flag takes effect before JAX initializes:
  XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest tests/unit/nnx_pipeline_test.py
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.layers import pipeline
from maxtext.models import simple_layer
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

# jax.checkpoint lowers to this primitive. Match it by IDENTITY, never by substring-searching the
# printed jaxpr: JAX renames primitives' printed names for cosmetics (pjit_p went "pjit" -> "jit" in a
# commit that also touched ad_checkpoint.py), and "remat2" is a leftover marker from the 2021 remat
# rewrite. A rename would break present-checks loudly and make absent-checks pass vacuously forever.
from jax._src.ad_checkpoint import remat_p as _REMAT_PRIMITIVE  # pylint: disable=wrong-import-position
from jax._src import core as _jax_core  # pylint: disable=wrong-import-position


def _jaxpr_contains_primitive(jaxpr, primitive):
  """True if `primitive` appears anywhere in `jaxpr`, including nested sub-jaxprs (scan/cond bodies)."""
  inner = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
  if any(eqn.primitive is primitive for eqn in inner.eqns):
    return True
  return any(_jaxpr_contains_primitive(sub, primitive) for sub in _jax_core.subjaxprs(inner))


_NEEDS_4_DEVICES = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs 4 devices; run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)


def _make_pipeline_config(ag_per_repeat, num_layers, num_micro, **overrides):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      enable_checkpointing=False,
      enable_goodput_recording=False,
      run_name="nnx_pipeline_unit",
      max_target_length=64,
      base_emb_dim=28,
      ici_pipeline_parallelism=4,
      base_num_decoder_layers=num_layers,
      num_pipeline_microbatches=num_micro,
      per_device_batch_size=4,
      pipeline_fsdp_ag_per_repeat=ag_per_repeat,
      **overrides,
  )


def _inputs(config):
  bs = config.global_batch_size_to_train_on
  seq = config.max_target_length
  feat = config.emb_dim
  inputs = jax.random.normal(jax.random.PRNGKey(2), [bs, seq, feat], dtype=jnp.float32)
  positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (bs, seq))
  seg = jnp.ones((bs, seq), dtype=jnp.int32)
  return inputs, seg, positions


def _run_pipeline(config, stage_factory):
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  inputs, seg, positions = _inputs(config)
  my_pipeline = pipeline.create_pipeline(config=config, layers=stage_factory, mesh=mesh)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    params = my_pipeline.init(jax.random.PRNGKey(0), inputs, seg, positions, True, MODEL_MODE_TRAIN)
    out = my_pipeline.apply(params, inputs, seg, positions, True, MODEL_MODE_TRAIN)
  return out


def _simple_factory(config, mesh):
  def factory(stage_rngs):
    return simple_layer.SimpleDecoderLayer(config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=stage_rngs)

  return factory


def _pipeline_value_and_grad(config, mesh):
  """Builds the pipeline (fixed init seed) and returns (loss, grads) for a sum-of-squares loss
  differentiated wrt the pipeline params. Same seed across calls -> identical params, so two configs
  that differ only by a numerically-transparent flag (e.g. remat) must yield matching loss + grads."""
  inputs, seg, positions = _inputs(config)
  my_pipeline = pipeline.create_pipeline(config=config, layers=_simple_factory(config, mesh), mesh=mesh)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    params = my_pipeline.init(jax.random.PRNGKey(0), inputs, seg, positions, True, MODEL_MODE_TRAIN)

    def loss_fn(p):
      out = my_pipeline.apply(p, inputs, seg, positions, True, MODEL_MODE_TRAIN)
      return jnp.sum(out.astype(jnp.float32) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
  return loss, grads


def _assert_stage_chaining_intact(test_case, config, mesh):
  """Assert that the pipeline's inter-stage chaining is intact: stage 0 receives the fresh input, every
  other stage receives the previous stage's output (not the fresh input)."""
  raw_pipeline = pipeline.create_nnx_pipeline(
      config=config, stage_factory=_simple_factory(config, mesh), mesh=mesh, rngs=nnx.Rngs(params=0)
  )
  micro_size = config.micro_batch_size_to_train_on // config.num_pipeline_microbatches
  activation_shape = (micro_size, config.max_target_length, config.emb_dim)
  dummy_inputs = jnp.zeros((config.num_pipeline_microbatches,) + activation_shape, dtype=jnp.float32)
  with jax.set_mesh(mesh):
    loop_state = raw_pipeline.init_states(dummy_inputs)

    fresh_marker, shift_marker = 111.0, -222.0
    state_io = jnp.full_like(loop_state["state_io"], fresh_marker)
    shift = jnp.full_like(loop_state["shift"], shift_marker)
    circ_storage = jnp.zeros_like(loop_state["circ_storage"]) if loop_state["circ_storage"] is not None else None

    stages_in = raw_pipeline.get_iteration_inputs(
        loop_iteration=0, state_io=state_io, circ_storage=circ_storage, shift=shift
    )

  test_case.assertTrue(
      bool(jnp.all(stages_in[0] == fresh_marker)), "stage 0 must receive the fresh state_io-derived input"
  )
  test_case.assertTrue(
      bool(jnp.all(stages_in[1:] == shift_marker)),
      "every stage other than stage 0 must receive `shift` (the previous stage's rotated output), "
      "not the fresh input -- inter-stage chaining is broken",
  )


# A non_trainable variable type: not Param, not Intermediate, not RngState -> lands in the
# pipeline's catch-all partition (mirrors moe.Tid2EidVar, the DeepSeek-V4 hash-routing table).
class _NonTrainableVar(nnx.Variable):
  pass


@_NEEDS_4_DEVICES
class TestNNXPipelineForward(unittest.TestCase):
  """Smoke coverage: both schedules run and produce finite output of the right shape."""

  def _assert_ok(self, config):
    """Run the pipeline and assert the output is finite, right-shaped, and not trivially equal to the raw input"""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    out = _run_pipeline(config, _simple_factory(config, mesh))
    expected = (config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim)
    self.assertEqual(out.shape, expected)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    self.assertGreater(float(jnp.std(out)), 1e-2)

    raw_inputs, _, _ = _inputs(config)
    self.assertFalse(
        bool(jnp.allclose(np.array(out), np.array(raw_inputs), rtol=1e-3, atol=1e-3)),
        msg="pipeline output equals the raw input -> per-stage compute was not applied",
    )
    # Shape + finiteness alone cannot see a broken inter-stage chain (every stage still runs, just
    # on the wrong input), so directly check the stage-0-vs-rest routing too.
    _assert_stage_chaining_intact(self, config, mesh)

  def test_noncircular_forward(self):
    self._assert_ok(_make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4))

  def test_circular_forward(self):
    self._assert_ok(_make_pipeline_config(ag_per_repeat=True, num_layers=8, num_micro=8))


@_NEEDS_4_DEVICES
class TestNNXPipelineBackward(unittest.TestCase):
  """Backward coverage: value_and_grad through the non-circular NNX pipeline on 4 devices.

  The forward-only smoke tests never exercised autodiff through the schedule. This locks that a real
  backward runs end-to-end: the loss is finite, gradients are finite, and at least one gradient is
  nonzero (the cotangent actually reached the stage parameters, i.e. the backward was not DCE'd)."""

  def test_noncircular_pipeline_backward(self):
    config = _make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    loss, grads = _pipeline_value_and_grad(config, mesh)

    self.assertTrue(bool(jnp.isfinite(loss)))
    grad_leaves = jax.tree_util.tree_leaves(grads)
    self.assertGreater(len(grad_leaves), 0)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves), "pipeline gradient has non-finite entries")
    self.assertTrue(
        any(bool(jnp.any(g != 0)) for g in grad_leaves), "all pipeline gradients are zero -> backward did not run"
    )
    # A gradient can be finite and nonzero even if every stage is silently fed the same raw input
    # instead of the previous stage's output (each stage still has a live, differentiable path to
    # the loss) -- so also check the exact inter-stage wiring the backward pass depends on.
    _assert_stage_chaining_intact(self, config, mesh)


class TestNonTrainablePartitioning(unittest.TestCase):
  """Unit guards for the two building blocks of the pipeline's non_trainable handling (pipeline.py),
  tested directly without running a pipeline (no device mesh needed):
    1. the state split routes a non_trainable var to the catch-all partition (non-circular path),
       which is then threaded through the scan carry;
    2. advance_rng_state leaves non-RngState leaves untouched (circular carry path).
  Together these let a non_trainable collection (e.g. moe.Tid2EidVar, DeepSeek-V4 hash routing) flow
  through the pipeline scan. End-to-end forward+backward is covered by TestNonTrainablePipelineBackward.
  """

  def test_advance_rng_state_preserves_non_rng_leaves(self):
    """Circular carry safety: advance_rng_state must pass non-RngState leaves through unchanged."""
    from maxtext.utils import pipeline_utils as pu  # pylint: disable=import-outside-toplevel

    state = nnx.State({"nt": _NonTrainableVar(jnp.asarray(3.0, dtype=jnp.float32))})
    out = pu.advance_rng_state(state, jnp.int32(7))
    np.testing.assert_array_equal(np.array(out["nt"][...]), np.array(3.0, dtype=np.float32))

  def test_four_way_split_routes_non_trainable_to_catchall(self):
    """Non-circular split: a non_trainable var routes to the catch-all partition, NOT into the
    RngState partition; a Param routes to the static-param partition.

    This asserts BUCKET ROUTING only, deliberately not broadcast-vs-carry semantics — so it stays
    valid however the catch-all is subsequently threaded through the loop."""
    from maxtext.utils import pipeline_utils as pu  # pylint: disable=import-outside-toplevel

    state = nnx.State(
        {"p": nnx.Param(jnp.ones((2,), dtype=jnp.float32)), "nt": _NonTrainableVar(jnp.asarray(3.0, dtype=jnp.float32))}
    )
    _, params, _, rng, catchall = nnx.split(state, pu.is_static_param, nnx.Intermediate, nnx.RngState, ...)

    def _vars(s):
      return jax.tree.leaves(s, is_leaf=lambda x: isinstance(x, nnx.Variable))

    self.assertTrue(any(isinstance(v, nnx.Param) for v in _vars(params)))
    self.assertTrue(any(isinstance(v, _NonTrainableVar) for v in _vars(catchall)))
    self.assertFalse(any(isinstance(v, _NonTrainableVar) for v in _vars(rng)))


@_NEEDS_4_DEVICES
class TestNNXCircularRepeatRemat(unittest.TestCase):
  """Circular repeat-level remat is unconditional (Linen parity). It must be numerically
  transparent, and the pipeline must run regardless of the iteration-remat flag value."""

  def _assert_repeat_level_remat_applied(self, config, mesh):
    """The repeat-level `flax_lift.checkpoint(_stage_fn_for_scope, ...)` wrap is unconditional and,
    per inspection of NNXCircularPipeline.__call__, is completely independent of
    set_remat_policy_on_pipeline_iterations (that flag is only ever read by the NON-circular
    NNXPipeline; get_pipeline_remat_policy never references it either). So comparing "flag on" vs
    "flag off" outputs/gradients cannot catch the wrap being silently dropped: removing it changes
    neither which flag reaches this code path (neither did to begin with) nor, ordinarily, the
    computed values (remat is numerically transparent by construction -- that's the whole point of
    remat). Assert directly on the compiled jaxpr instead: flax_lift.checkpoint must lower to a
    real remat primitive; if the wrap is removed the primitive disappears entirely. Matched by
    primitive IDENTITY, not by substring-searching the printed jaxpr -- see _jaxpr_contains_primitive."""
    inputs, seg, positions = _inputs(config)
    my_pipeline = pipeline.create_pipeline(config=config, layers=_simple_factory(config, mesh), mesh=mesh)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
      params = my_pipeline.init(jax.random.PRNGKey(0), inputs, seg, positions, True, MODEL_MODE_TRAIN)

      def fwd(p):
        return my_pipeline.apply(p, inputs, seg, positions, True, MODEL_MODE_TRAIN)

      jaxpr = jax.make_jaxpr(fwd)(params)

    self.assertTrue(
        _jaxpr_contains_primitive(jaxpr, _REMAT_PRIMITIVE),
        "circular pipeline forward jaxpr has no remat primitive -- the unconditional repeat-level "
        "flax_lift.checkpoint(...) wrap around _stage_fn_for_scope appears to have been removed",
    )

  def test_repeat_remat_output_transparent(self):
    cfg_on = _make_pipeline_config(
        ag_per_repeat=True, num_layers=8, num_micro=8, set_remat_policy_on_pipeline_iterations=True
    )
    cfg_off = _make_pipeline_config(
        ag_per_repeat=True, num_layers=8, num_micro=8, set_remat_policy_on_pipeline_iterations=False
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_on)
    mesh = Mesh(devices_array, cfg_on.mesh_axes)
    out_on = _run_pipeline(cfg_on, _simple_factory(cfg_on, mesh))
    out_off = _run_pipeline(cfg_off, _simple_factory(cfg_off, mesh))
    np.testing.assert_allclose(np.array(out_on), np.array(out_off), rtol=1e-5, atol=1e-5)
    # The on-vs-off comparison above cannot see the remat wrap being dropped (see docstring on
    # _assert_repeat_level_remat_applied), so check its presence directly for both configs.
    self._assert_repeat_level_remat_applied(cfg_on, mesh)
    self._assert_repeat_level_remat_applied(cfg_off, mesh)

  def test_repeat_remat_grad_parity(self):
    """Remat must be transparent in the BACKWARD pass too, not just the forward output: the
    circular pipeline with repeat-level remat on vs off must produce matching loss AND gradients."""
    cfg_on = _make_pipeline_config(
        ag_per_repeat=True, num_layers=8, num_micro=8, set_remat_policy_on_pipeline_iterations=True
    )
    cfg_off = _make_pipeline_config(
        ag_per_repeat=True, num_layers=8, num_micro=8, set_remat_policy_on_pipeline_iterations=False
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_on)
    mesh = Mesh(devices_array, cfg_on.mesh_axes)

    loss_on, grads_on = _pipeline_value_and_grad(cfg_on, mesh)
    loss_off, grads_off = _pipeline_value_and_grad(cfg_off, mesh)

    np.testing.assert_allclose(np.array(loss_on), np.array(loss_off), rtol=1e-4, atol=1e-4)
    on_leaves = jax.tree_util.tree_leaves(grads_on)
    off_leaves = jax.tree_util.tree_leaves(grads_off)
    self.assertEqual(len(on_leaves), len(off_leaves))
    self.assertGreater(len(on_leaves), 0)
    for g_on, g_off in zip(on_leaves, off_leaves):
      np.testing.assert_allclose(np.array(g_on), np.array(g_off), rtol=1e-4, atol=1e-4)
    # Guard against a vacuous pass (all-zero grads would trivially match).
    self.assertTrue(any(bool(jnp.any(g != 0)) for g in on_leaves), "all gradients are zero -> backward did not run")
    # loss/grad parity between the two configs is a false signal here (see docstring on
    # _assert_repeat_level_remat_applied) -- confirm the remat wrap is actually present.
    self._assert_repeat_level_remat_applied(cfg_on, mesh)


class _StageWithNonTrainable(nnx.Module):
  """A pipeline stage carrying a non_trainable variable (like moe.Tid2EidVar, DeepSeek-V4 hash
  routing), added into the forward.

  Must return the SAME structure as the wrapped layer: SimpleDecoderLayer returns an (output, kv)
  tuple the pipeline loop-state relies on; collapsing it to a bare array trips the shard_map. So the
  non_trainable is folded into tuple[0] and the rest is passed through."""

  def __init__(self, config, mesh, value, *, rngs):
    self.inner = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
    self.nt = _NonTrainableVar(jnp.asarray(value, dtype=jnp.float32))

  def __call__(self, inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs):
    res = self.inner(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs)
    if isinstance(res, tuple):
      return (res[0] + self.nt[...],) + tuple(res[1:])
    return res + self.nt[...]


@_NEEDS_4_DEVICES
class TestNonTrainablePipelineBackward(unittest.TestCase):
  """A pipeline stage carrying a non_trainable variable must run forward AND backward, with finite,
  nonzero param gradients: the non_trainable collection is carried through the iteration scan on BOTH
  schedules and must not break autodiff to the trainable params.

  """

  def _assert_backward_ok(self, config):
    """Init the pipeline (non_trainable stage), value_and_grad a sum-of-squares loss, assert loss +
    param grads are finite and at least one is nonzero (backward ran with non_trainable present)."""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    inputs, seg, positions = _inputs(config)

    def factory(stage_rngs):
      return _StageWithNonTrainable(config, mesh, 0.5, rngs=stage_rngs)

    my_pipeline = pipeline.create_pipeline(config=config, layers=factory, mesh=mesh)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
      params = my_pipeline.init(jax.random.PRNGKey(0), inputs, seg, positions, True, MODEL_MODE_TRAIN)

      def loss_fn(p):
        out = my_pipeline.apply(p, inputs, seg, positions, True, MODEL_MODE_TRAIN)
        return jnp.sum(out.astype(jnp.float32) ** 2)

      loss, grads = jax.value_and_grad(loss_fn)(params)

    self.assertTrue(bool(jnp.isfinite(loss)))
    grad_leaves = jax.tree_util.tree_leaves(grads)
    self.assertGreater(len(grad_leaves), 0)
    self.assertTrue(
        all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves), "non_trainable pipeline grad has non-finite entries"
    )
    self.assertTrue(
        any(bool(jnp.any(g != 0)) for g in grad_leaves),
        "all grads zero -> backward did not run with a non_trainable variable present",
    )

  def test_noncircular_nontrainable_backward(self):
    self._assert_backward_ok(_make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4))

  def test_circular_nontrainable_backward(self):
    self._assert_backward_ok(_make_pipeline_config(ag_per_repeat=True, num_layers=8, num_micro=8))


class _MutatingNonTrainableStage(nnx.Module):
  """_StageWithNonTrainable, but it MUTATES the non_trainable on every invocation."""

  def __init__(self, config, mesh, value, *, rngs):
    self.inner = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
    self.nt = _NonTrainableVar(jnp.asarray(value, dtype=jnp.float32))

  def __call__(self, inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs):
    res = self.inner(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs)
    self.nt[...] = self.nt[...] + 1.0
    if isinstance(res, tuple):
      return (res[0] + self.nt[...],) + tuple(res[1:])
    return res + self.nt[...]


@_NEEDS_4_DEVICES
class TestNonTrainableMutationSurvivesLoop(unittest.TestCase):
  """A non_trainable mutated inside the iteration loop must still be mutated when the loop ends.
  The non_trainable is carried through the iteration scan on BOTH schedules, so the mutation must
  survive the loop on both schedules."""

  _INIT = 0.0

  def _final_non_trainable(self, config):
    """Run the pipeline with a mutating non_trainable stage, return the final value of the non_trainable."""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    inputs, seg, positions = _inputs(config)

    def factory(stage_rngs):
      return _MutatingNonTrainableStage(config, mesh, self._INIT, rngs=stage_rngs)

    my_pipeline = pipeline.create_pipeline(config=config, layers=factory, mesh=mesh)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
      variables = my_pipeline.init(jax.random.PRNGKey(0), inputs, seg, positions, True, MODEL_MODE_TRAIN)
      mutable = [k for k in variables if k != "params"]
      _, updated = my_pipeline.apply(variables, inputs, seg, positions, True, MODEL_MODE_TRAIN, mutable=mutable or True)
    leaves = jax.tree_util.tree_leaves(updated.get(_NonTrainableVar.__name__, {}))
    return [float(v) for leaf in leaves for v in jnp.ravel(leaf)]

  def _assert_mutation_kept(self, config):
    values = self._final_non_trainable(config)
    self.assertTrue(values, "no non_trainable leaf surfaced; test cannot conclude")
    self.assertTrue(
        all(v > self._INIT for v in values),
        f"non_trainable mutation was discarded by the iteration loop: {values} (init={self._INIT}). "
        "The stage incremented it on every invocation, so every entry must exceed the initial value.",
    )

  def test_noncircular_preserves_non_trainable_mutation(self):
    self._assert_mutation_kept(_make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4))

  def test_circular_preserves_non_trainable_mutation(self):
    self._assert_mutation_kept(_make_pipeline_config(ag_per_repeat=True, num_layers=8, num_micro=8))


class _FlagIndependentReturnStage(nnx.Module):
  """
  A stage whose return SHAPE deliberately does not track ``config.scan_layers``.
  """

  def __init__(self, config, mesh, *, rngs, always_tuple):
    self.inner = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh, model_mode=MODEL_MODE_TRAIN, rngs=rngs)
    self.always_tuple = always_tuple

  def __call__(self, inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs):
    res = self.inner(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs)
    out = res[0] if isinstance(res, tuple) else res
    return (out, None) if self.always_tuple else out


@_NEEDS_4_DEVICES
class TestStageOutputUnwrapIsFlagIndependent(unittest.TestCase):
  """The stage-output unwrap must key off the actual return shape, not off config.scan_layers.

  Both schedules must tolerate a stage whose return shape stops tracking the flag."""

  def _assert_pipeline_ok(self, config, always_tuple):
    """Run the pipeline with a stage whose return shape does not track config.scan_layers, assert the
    output is finite and right-shaped."""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    def factory(stage_rngs):
      return _FlagIndependentReturnStage(config, mesh, rngs=stage_rngs, always_tuple=always_tuple)

    out = _run_pipeline(config, factory)
    expected = (config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim)
    self.assertEqual(
        out.shape,
        expected,
        f"unwrap did not track the stage's real return shape (always_tuple={always_tuple}, "
        f"scan_layers={config.scan_layers})",
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_noncircular_tuple_return_when_scan_layers_off(self):
    config = _make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4, scan_layers=False)
    self._assert_pipeline_ok(config, always_tuple=True)

  def test_noncircular_bare_return_when_scan_layers_on(self):
    config = _make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4, scan_layers=True)
    self._assert_pipeline_ok(config, always_tuple=False)

  def test_circular_tuple_return_when_scan_layers_off(self):
    config = _make_pipeline_config(ag_per_repeat=True, num_layers=8, num_micro=8, scan_layers=False)
    self._assert_pipeline_ok(config, always_tuple=True)

  def test_circular_bare_return_when_scan_layers_on(self):
    config = _make_pipeline_config(ag_per_repeat=True, num_layers=8, num_micro=8, scan_layers=True)
    self._assert_pipeline_ok(config, always_tuple=False)


if __name__ == "__main__":
  unittest.main()
