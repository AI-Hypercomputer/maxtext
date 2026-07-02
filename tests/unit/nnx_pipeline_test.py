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

"""CPU unit tests for the NNX pipeline (pipeline.py NNXPipeline / NNXCircularPipeline).

The integration pipeline tests are all tpu_only, so the NNX pipeline __call__ paths had ZERO
CPU coverage. These run on 4 fake CPU devices and lock the migration-parity fixes:
  - non_trainable handling: non-circular broadcasts the catch-all (4-way split), circular carries
    it via "carry_state" (the prior RngState-only assert crashed any non_trainable variable);
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


# A non_trainable variable type: not Param, not Intermediate, not RngState -> lands in the
# pipeline's catch-all partition (mirrors moe.Tid2EidVar, the DeepSeek-V4 hash-routing table).
class _NonTrainableVar(nnx.Variable):
  pass


@_NEEDS_4_DEVICES
class TestNNXPipelineForward(unittest.TestCase):
  """Smoke coverage: both schedules run on CPU and produce finite output of the right shape."""

  def _assert_ok(self, config):
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    out = _run_pipeline(config, _simple_factory(config, mesh))
    expected = (config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim)
    self.assertEqual(out.shape, expected)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_noncircular_forward(self):
    self._assert_ok(_make_pipeline_config(ag_per_repeat=False, num_layers=4, num_micro=4))

  def test_circular_forward(self):
    self._assert_ok(_make_pipeline_config(ag_per_repeat=True, num_layers=8, num_micro=8))


@_NEEDS_4_DEVICES
class TestNNXPipelineBackward(unittest.TestCase):
  """Backward coverage: value_and_grad through the non-circular NNX pipeline on 4 fake CPU devices.

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


class TestNonTrainablePartitioning(unittest.TestCase):
  """Unit guards for the two building blocks of the pipeline's non_trainable handling (pipeline.py),
  tested directly without running a pipeline (no device mesh needed):
    1. the 4-way state split routes a non_trainable var to the broadcast catch-all (non-circular path);
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
    """Non-circular split: a non_trainable var routes to the catch-all (broadcast) partition,
    NOT into the RngState carry; a Param routes to the static-param (broadcast) partition."""
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
  nonzero param gradients: the non_trainable collection is broadcast (non-circular) / carried
  (circular) through the iteration scan and must not break autodiff to the trainable params.

  Runs on CPU: pipeline parallelism needs 1 device per stage (ici_pipeline_parallelism=4 -> 4), so a
  single CPU is split into 4 simulated XLA devices via XLA_FLAGS=--xla_force_host_platform_device_count=4
  (see the module header). @_NEEDS_4_DEVICES skips it when that flag is not set."""

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


if __name__ == "__main__":
  unittest.main()
