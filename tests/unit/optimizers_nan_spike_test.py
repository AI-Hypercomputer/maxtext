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

"""Pure-unittest tests for optimizer NaN/Inf and max_grad_norm_spike guards."""

import sys
from types import SimpleNamespace
import unittest

from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from tests.utils.test_helpers import get_test_config_path


class OptimizerNanAndSpikeGuardTest(unittest.TestCase):
  """Verifies warmup NaN protection, rolling-buffer hygiene, and Tier-2 skip-step flags."""

  def test_skip_step_on_spikes_nan_loss_during_warmup(self):
    opt = optimizers.skip_step_on_spikes(optax.sgd(0.1), interval=4, scaling_factor=1.0)
    params = {"x": jnp.array([1.0])}
    opt_state = opt.init(params)

    updates, opt_state = opt.update(
        {"x": jnp.array([1.0])}, opt_state, params, loss=jnp.float32(jnp.nan), grad_norm=jnp.float32(1.0)
    )
    self.assertTrue(bool(jnp.all(updates["x"] == 0.0)))
    self.assertTrue(bool(opt_state["is_skipped"]))
    self.assertEqual(int(opt_state["count"]), 0)
    self.assertFalse(bool(jnp.any(jnp.isnan(opt_state["losses"]))))

  def test_skip_step_on_spikes_nan_grad_norm_during_warmup(self):
    opt = optimizers.skip_step_on_spikes(optax.sgd(0.1), interval=4, scaling_factor=1.0)
    params = {"x": jnp.array([1.0])}
    opt_state = opt.init(params)

    updates, opt_state = opt.update(
        {"x": jnp.array([1.0])}, opt_state, params, loss=jnp.float32(1.0), grad_norm=jnp.float32(jnp.nan)
    )
    self.assertTrue(bool(jnp.all(updates["x"] == 0.0)))
    self.assertTrue(bool(opt_state["is_skipped"]))
    self.assertEqual(int(opt_state["count"]), 0)
    self.assertFalse(bool(jnp.any(jnp.isnan(opt_state["grad_norms"]))))

  def test_nan_does_not_poison_rolling_buffer(self):
    opt = optimizers.skip_step_on_spikes(optax.sgd(0.1), interval=4, scaling_factor=1.0)
    params = {"x": jnp.array([1.0])}
    opt_state = opt.init(params)

    # Step 0: NaN occurs and is excluded from rolling history.
    _, opt_state = opt.update(
        {"x": jnp.array([1.0])}, opt_state, params, loss=jnp.float32(jnp.nan), grad_norm=jnp.float32(jnp.nan)
    )
    self.assertTrue(bool(opt_state["is_skipped"]))
    self.assertEqual(int(opt_state["count"]), 0)

    # Steps 1-2: Normal finite steps populate the rolling baseline.
    for expected_count in (1, 2):
      updates, opt_state = opt.update(
          {"x": jnp.array([1.0])}, opt_state, params, loss=jnp.float32(1.0), grad_norm=jnp.float32(1.0)
      )
      self.assertFalse(bool(opt_state["is_skipped"]))
      self.assertEqual(int(opt_state["count"]), expected_count)
      self.assertFalse(bool(jnp.all(updates["x"] == 0.0)))

    # Step 3: Subsequent finite spike is detected because rolling stats stayed finite.
    updates, opt_state = opt.update(
        {"x": jnp.array([1.0])}, opt_state, params, loss=jnp.float32(100.0), grad_norm=jnp.float32(1.0)
    )
    self.assertTrue(bool(opt_state["is_skipped"]))
    self.assertTrue(bool(jnp.all(updates["x"] == 0.0)))

  def test_skip_step_on_spikes_enforces_max_grad_norm_spike_during_warmup(self):
    opt = optimizers.skip_step_on_spikes(optax.sgd(0.1), interval=8, scaling_factor=6.0, max_grad_norm_spike=10.0)
    params = {"x": jnp.array([1.0])}
    opt_state = opt.init(params)

    updates, opt_state = opt.update(
        {"x": jnp.array([1.0])}, opt_state, params, loss=jnp.float32(1.0), grad_norm=jnp.float32(15.0)
    )
    self.assertTrue(bool(opt_state["is_skipped"]))
    self.assertTrue(bool(jnp.all(updates["x"] == 0.0)))

  def test_lightweight_nan_or_grad_spike_wrapper_and_default_unwrapped(self):
    base_cfg = SimpleNamespace(
        opt_type="sgd",
        skip_step_on_spikes=False,
        skip_step_on_nan=False,
        max_grad_norm_spike=0.0,
        trainable_parameters_mask=None,
    )
    default_opt = optimizers.get_optimizer(base_cfg, 0.1)
    params = {"x": jnp.array([2.0])}
    self.assertIsNone(optimizers.extract_is_skipped(default_opt.init(params)))

    guarded_cfg = SimpleNamespace(
        opt_type="sgd",
        skip_step_on_spikes=False,
        skip_step_on_nan=True,
        max_grad_norm_spike=5.0,
        trainable_parameters_mask=None,
    )
    guarded_opt = optimizers.get_optimizer(guarded_cfg, 0.1)
    self.assertIsInstance(guarded_opt, optax.GradientTransformationExtraArgs)

    state = guarded_opt.init(params)
    self.assertIsNotNone(optimizers.extract_is_skipped(state))
    self.assertNotIn("losses", state)

    # Normal step applies update.
    updates, state = guarded_opt.update(
        {"x": jnp.array([1.0])}, state, params, loss=jnp.float32(1.0), grad_norm=jnp.float32(2.0)
    )
    self.assertFalse(bool(optimizers.extract_is_skipped(state)))
    self.assertFalse(bool(jnp.all(updates["x"] == 0.0)))

    # Inf grad_norm skips update.
    updates, state = guarded_opt.update(
        {"x": jnp.array([1.0])}, state, params, loss=jnp.float32(1.0), grad_norm=jnp.float32(jnp.inf)
    )
    self.assertTrue(bool(optimizers.extract_is_skipped(state)))
    self.assertTrue(bool(jnp.all(updates["x"] == 0.0)))

    # Finite grad_norm > max_grad_norm_spike skips update.
    updates, state = guarded_opt.update(
        {"x": jnp.array([1.0])}, state, params, loss=jnp.float32(1.0), grad_norm=jnp.float32(6.0)
    )
    self.assertTrue(bool(optimizers.extract_is_skipped(state)))
    self.assertTrue(bool(jnp.all(updates["x"] == 0.0)))

  def test_pyconfig_tier2_defaults_and_overrides(self):
    default_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_skip_defaults",
        enable_checkpointing=False,
    )
    self.assertFalse(default_cfg.skip_step_on_nan)
    self.assertEqual(default_cfg.max_grad_norm_spike, 0.0)

    override_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_skip_overrides",
        enable_checkpointing=False,
        skip_step_on_nan=True,
        max_grad_norm_spike=25.0,
    )
    self.assertTrue(override_cfg.skip_step_on_nan)
    self.assertEqual(override_cfg.max_grad_norm_spike, 25.0)

  def test_get_optimizer_with_trainable_mask_and_nan_guard_nnx_pure_dict(self):
    """get_optimizer keeps skip_step_on_nan_or_grad_spike outermost under trainable_parameters_mask + NNX."""

    class _MaskedModel(nnx.Module):

      def __init__(self):
        self.indexer = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))
        self.kernel = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32) * 2.0)

    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test_mask_and_nan_guard",
        enable_checkpointing=False,
        trainable_parameters_mask=[".*indexer.*"],
        skip_step_on_spikes=False,
        skip_step_on_nan=True,
        max_grad_norm_spike=5.0,
    )
    tx = optimizers.get_optimizer(cfg, lambda _: 0.1)
    model = _MaskedModel()
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    state = train_state_nnx.TrainStateNNX(model, optimizer)

    raw_opt_state = tx.init({"indexer": jnp.ones((2, 2)), "kernel": jnp.ones((2, 2))})
    self.assertIsInstance(raw_opt_state["inner_state"], optax.PartitionState)
    self.assertIsInstance(
        raw_opt_state["inner_state"].inner_states["trainable"].inner_state[0].mu["kernel"], optax.MaskedNode
    )

    pure_opt_state = nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]
    self.assertIn("is_skipped", pure_opt_state)
    self.assertIsNotNone(optimizers.extract_is_skipped(pure_opt_state))

    grads = nnx.state(model, nnx.Param)
    # NaN step skips update and preserves both trainable and frozen weights.
    state.apply_gradients(grads, loss=jnp.float32(jnp.nan), grad_norm=jnp.float32(1.0))
    pure_opt_state = nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]
    self.assertTrue(bool(optimizers.extract_is_skipped(pure_opt_state)))
    np.testing.assert_allclose(np.asarray(model.indexer[...]), np.ones((2, 2)))
    np.testing.assert_allclose(np.asarray(model.kernel[...]), np.ones((2, 2)) * 2.0)

    # Pre-clipping grad_norm spike skips update and preserves weights.
    state.apply_gradients(grads, loss=jnp.float32(1.0), grad_norm=jnp.float32(10.0))
    pure_opt_state = nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]
    self.assertTrue(bool(optimizers.extract_is_skipped(pure_opt_state)))
    np.testing.assert_allclose(np.asarray(model.indexer[...]), np.ones((2, 2)))
    np.testing.assert_allclose(np.asarray(model.kernel[...]), np.ones((2, 2)) * 2.0)

    # Finite step within cap updates only the trainable parameter.
    state.apply_gradients(grads, loss=jnp.float32(1.0), grad_norm=jnp.float32(1.0))
    pure_opt_state = nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]
    self.assertFalse(bool(optimizers.extract_is_skipped(pure_opt_state)))
    self.assertFalse(np.allclose(np.asarray(model.indexer[...]), np.ones((2, 2))))
    np.testing.assert_allclose(np.asarray(model.kernel[...]), np.ones((2, 2)) * 2.0)


if __name__ == "__main__":
  unittest.main()
