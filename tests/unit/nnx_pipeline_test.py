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


class TestNonTrainablePartitioning(unittest.TestCase):
  """Mechanism-level guards for the non_trainable migration fix (pipeline.py).

  The prior code asserted the pipeline's catch-all partition was ONLY nnx.RngState, crashing any
  model with a non_trainable variable (e.g. moe.Tid2EidVar). The fix relies on two invariants,
  tested directly here (no device mesh needed). End-to-end non_trainable-through-pipeline requires a
  real DeepSeek-V4 layer with sharding metadata and is exercised on TPU.
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


if __name__ == "__main__":
  unittest.main()
