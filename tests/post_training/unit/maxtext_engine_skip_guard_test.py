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

"""Pure-unittest tests for MaxTextTrainingEngine NaN/Inf and grad-norm spike guards."""

# pylint: disable=protected-access

import dataclasses
import tempfile
from typing import Any
import unittest
from unittest import mock

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.post_training]


class _DummyEngineModel(nnx.Module):

  def __init__(self):
    self.trainable_w = nnx.Param(jnp.array([1.0, 2.0], dtype=jnp.float32))
    self.frozen_w = nnx.Param(jnp.array([3.0, 4.0], dtype=jnp.float32))


@struct.dataclass(frozen=True, kw_only=True)
class _DummyPayload(abstract_engine.TrainerPayload):
  token_ids: Any = dataclasses.field(default_factory=lambda: np.ones((2, 2), dtype=np.int32))
  token_mask: Any = dataclasses.field(default_factory=lambda: np.ones((2, 2), dtype=np.int32))
  metadata: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)


class MaxTextEngineSkipGuardTest(unittest.TestCase):
  """Verifies wrapped, compiled/eager NaN-loss, unwrapped-fallback, and nested multi_transform skip guards."""

  def _make_engine(self, **overrides) -> maxtext_engine.MaxTextTrainingEngine:
    """Builds a test MaxTextTrainingEngine with a tiny two-parameter NNX model."""
    tmpdir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(tmpdir.cleanup)
    cfg_overrides = {
        "model_name": "llama3.1-8b",
        "run_name": "test_engine_skip_guard",
        "base_output_directory": tmpdir.name,
        "init_weights_seed": 42,
        "micro_batch_size_to_train_on": 2,
        "gradient_accumulation_steps": 1,
        "enable_dropout": False,
        "record_internal_nn_metrics": False,
        "enable_tensorboard": False,
        "tensorboard_dir": tmpdir.name,
        "skip_jax_distributed_system": True,
        "enable_checkpointing": False,
        "profiler_steps": 0,
        "opt_type": "sgd",
        "learning_rate": 0.1,
    }
    cfg_overrides.update(overrides)
    cfg = pyconfig.initialize([None, get_test_config_path()], **cfg_overrides)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = _DummyEngineModel()
    tx = optimizers.get_optimizer(cfg, lambda _: jnp.array(0.1, dtype=jnp.float32))
    with (
        mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=(model, mesh)),
        mock.patch.object(
            maxtext_engine.train_utils,
            "create_training_optimizer",
            return_value=(lambda _: jnp.array(0.1, dtype=jnp.float32), tx),
        ),
    ):
      return maxtext_engine.MaxTextTrainingEngine(cfg)

  def test_wrapped_spikes_and_max_grad_norm_spike_preserves_rolling_state(self):
    """Wrapped skip_step_on_spikes + max_grad_norm_spike must not clobber opt_state count/is_skipped/step."""
    engine = self._make_engine(
        skip_step_on_spikes=True,
        skip_step_interval=8,
        skip_step_scaling_factor=6.0,
        max_grad_norm_spike=0.5,
        gradient_clipping_threshold=0.01,
    )
    # Gradient wrt trainable_w is [2.0, 2.0] (raw norm = sqrt(8) > 0.5, clipped norm = 0.01 < 0.5).
    engine.with_loss_fn(
        lambda mdl, *_a, **_kw: abstract_engine.WeightedMetric(
            unreduced_sum=jnp.sum(mdl.trainable_w[...]) * 8.0,
            denominator=jnp.array(4.0, dtype=jnp.float32),
        ),
        has_aux=False,
    )
    before = np.asarray(engine.model.trainable_w[...]).copy()
    engine.fwd_bwd(_DummyPayload())
    engine.update()

    buf = engine.get_metrics(clear_cache=True)
    np.testing.assert_allclose(np.asarray(buf.scalar_metrics["step_skipped"]), [1.0])
    np.testing.assert_allclose(np.asarray(engine.state.model.trainable_w[...]), before)

    opt_pure = nnx.to_pure_dict(nnx.state(engine.state.optimizer))
    self.assertEqual(int(opt_pure["step"]), 1)
    self.assertTrue(bool(opt_pure["opt_state"]["is_skipped"]))
    self.assertEqual(int(opt_pure["opt_state"]["count"]), 1)

  def test_skip_step_on_nan_skips_nan_loss_with_finite_grad_norm_eager_and_compiled(self):
    """skip_step_on_nan=True (with skip_step_on_spikes=False) must feed mean_loss in eager and compiled modes."""
    for compiled in (False, True):
      with self.subTest(compiled=compiled):
        engine = self._make_engine(skip_step_on_nan=True, skip_step_on_spikes=False)
        # Finite gradient wrt trainable_w ([2.0, 2.0]), but primary loss sum carries an additive NaN constant.
        engine.with_loss_fn(
            lambda mdl, *_a, **_kw: abstract_engine.WeightedMetric(
                unreduced_sum=jnp.sum(mdl.trainable_w[...]) * 8.0 + jnp.float32(jnp.nan),
                denominator=jnp.array(4.0, dtype=jnp.float32),
            ),
            has_aux=False,
        )
        payload = _DummyPayload()
        if compiled:
          engine.compile(payload)
        before = np.asarray(engine.model.trainable_w[...]).copy()
        engine.fwd_bwd(payload)
        engine.update()

        buf = engine.get_metrics(clear_cache=True)
        np.testing.assert_allclose(np.asarray(buf.scalar_metrics["step_skipped"]), [1.0])
        np.testing.assert_allclose(np.asarray(engine.state.model.trainable_w[...]), before)

  def test_unwrapped_external_optimizer_fallback_reverts_full_state(self):
    """When engine.optimizer lacks a skip wrapper, elif use_skip_guard reverts params and moments cleanly."""
    engine = self._make_engine(
        skip_step_on_nan=True,
        max_grad_norm_spike=0.5,
        gradient_clipping_threshold=1.0,
    )
    engine.optimizer = nnx.Optimizer(engine.model, optax.adamw(0.1), wrt=nnx.Param)
    engine.with_loss_fn(
        lambda mdl, *_a, **_kw: abstract_engine.WeightedMetric(
            unreduced_sum=jnp.sum(mdl.trainable_w[...]) * 8.0,
            denominator=jnp.array(4.0, dtype=jnp.float32),
        ),
        has_aux=False,
    )
    before = np.asarray(engine.model.trainable_w[...]).copy()
    engine.fwd_bwd(_DummyPayload())
    engine.update()

    buf = engine.get_metrics(clear_cache=True)
    np.testing.assert_allclose(np.asarray(buf.scalar_metrics["step_skipped"]), [1.0])
    np.testing.assert_allclose(np.asarray(engine.state.model.trainable_w[...]), before)
    opt_pure = nnx.to_pure_dict(nnx.state(engine.state.optimizer))
    self.assertEqual(int(opt_pure["step"]), 0)

  def test_glm5_style_nested_multi_transform_with_skip_step_on_spikes_skips_spike(self):
    """glm5_train.py pattern wrapping engine._optimizer.tx in multi_transform({'train': tx, 'frozen': ...}) skips."""
    engine = self._make_engine(
        skip_step_on_spikes=True,
        skip_step_interval=4,
        skip_step_scaling_factor=1.0,
    )
    base_tx = engine._optimizer.tx
    masked_tx = optax.multi_transform(
        {"train": base_tx, "frozen": optax.set_to_zero()},
        param_labels=lambda params: jax.tree_util.tree_map_with_path(
            lambda kp, _: "train" if "trainable_w" in str(kp) else "frozen",
            params,
            is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array)),
        ),
    )
    engine._optimizer = engine._build_optimizer(masked_tx)
    engine._tx = masked_tx

    scale_holder = [1.0]
    engine.with_loss_fn(
        lambda mdl, *_a, **_kw: abstract_engine.WeightedMetric(
            unreduced_sum=jnp.sum(mdl.trainable_w[...] + mdl.frozen_w[...]) * scale_holder[0],
            denominator=jnp.array(1.0, dtype=jnp.float32),
        ),
        has_aux=False,
    )
    # Two finite warmup steps populate the nested skip_step_on_spikes rolling baseline.
    for _ in range(2):
      engine.fwd_bwd(_DummyPayload())
      engine.update()
      engine.get_metrics(clear_cache=True)

    np.testing.assert_allclose(np.asarray(engine.state.model.frozen_w[...]), [3.0, 4.0])
    before_spike = np.asarray(engine.state.model.trainable_w[...]).copy()

    # Step 3: huge loss/grad spike must be skipped by the nested skip_step_on_spikes wrapper.
    scale_holder[0] = 1000.0
    engine.fwd_bwd(_DummyPayload())
    engine.update()
    np.testing.assert_allclose(np.asarray(engine.state.model.trainable_w[...]), before_spike)
    np.testing.assert_allclose(np.asarray(engine.state.model.frozen_w[...]), [3.0, 4.0])


if __name__ == "__main__":
  unittest.main()
