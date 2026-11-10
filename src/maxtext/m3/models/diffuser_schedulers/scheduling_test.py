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

"""Parity tests of the JAX schedulers against the diffusers (torch) schedulers."""

import unittest

from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.models.diffuser_schedulers.scheduling_flow_match import FlaxFlowMatchScheduler
from maxtext.m3.models.diffuser_schedulers.scheduling_unipc_multistep import FlaxUniPCMultistepScheduler

try:
  import torch
except ImportError:
  torch = None

# Wan-AI/Wan2.1-T2V-14B-Diffusers scheduler/scheduler_config.json (also stored in converted checkpoints).
WAN21_UNIPC_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}

_SHAPE = (1, 4, 3, 8, 8)  # small [B, C, F, H, W] latent


def _fake_model(sample: np.ndarray, t: float) -> np.ndarray:
  """Deterministic stand-in for the transformer: a smooth function of the sample and the timestep."""
  return (0.5 * np.cos(t / 1000.0) * sample + 0.1 * np.sin(sample + t / 500.0)).astype(np.float32)


def _ref_unipc(**overrides):
  config = {k: v for k, v in WAN21_UNIPC_CONFIG.items() if not k.startswith("_")}
  return UniPCMultistepScheduler(**{**config, **overrides})


@unittest.skipIf(torch is None, "needs torch (diffusers reference schedulers)")
class UniPCParityTest(unittest.TestCase):
  """`FlaxUniPCMultistepScheduler` (Wan 2.1 config) vs `diffusers.UniPCMultistepScheduler`.

  Flow sigmas / `flow_prediction`: `set_timesteps` and a full sampling loop for solver orders 1-3. Manual run (needs
  torch; CPU, no network): `JAX_PLATFORMS=cpu python -m pytest src/maxtext/m3/models/diffuser_schedulers/`.
  """

  def test_from_config_ignores_diffusers_keys(self):
    scheduler, state = FlaxUniPCMultistepScheduler.from_config(WAN21_UNIPC_CONFIG, flow_shift=5.0)
    self.assertEqual(scheduler.config.flow_shift, 5.0)
    self.assertEqual(scheduler.config.prediction_type, "flow_prediction")
    self.assertIsNotNone(state)

  def test_set_timesteps(self):
    for flow_shift in (3.0, 5.0):
      for num_steps in (10, 50):
        with self.subTest(flow_shift=flow_shift, num_steps=num_steps):
          scheduler, state = FlaxUniPCMultistepScheduler.from_config(WAN21_UNIPC_CONFIG, flow_shift=flow_shift)
          state = scheduler.set_timesteps(state, num_inference_steps=num_steps, shape=_SHAPE)
          ref = _ref_unipc(flow_shift=flow_shift)
          ref.set_timesteps(num_steps)
          np.testing.assert_array_equal(np.asarray(state.timesteps), ref.timesteps.numpy())
          np.testing.assert_allclose(np.asarray(state.sigmas), ref.sigmas.numpy(), rtol=0, atol=1e-6)

  def test_sampling_loop(self):
    """Every step of a sampling loop (predictor + corrector, warmup and lower-order final steps)."""
    num_steps = 12
    for solver_order in (1, 2, 3):
      with self.subTest(solver_order=solver_order):
        scheduler, state = FlaxUniPCMultistepScheduler.from_config(
            WAN21_UNIPC_CONFIG, flow_shift=5.0, solver_order=solver_order
        )
        state = scheduler.set_timesteps(state, num_inference_steps=num_steps, shape=_SHAPE)
        ref = _ref_unipc(flow_shift=5.0, solver_order=solver_order)
        ref.set_timesteps(num_steps)

        x0 = np.random.default_rng(0).standard_normal(_SHAPE).astype(np.float32)
        ours, theirs = jnp.asarray(x0), torch.from_numpy(x0.copy())
        for i, t in enumerate(np.asarray(state.timesteps)):
          ours, state = scheduler.step(state, jnp.asarray(_fake_model(np.asarray(ours), float(t))), t, ours).to_tuple()
          theirs = ref.step(torch.from_numpy(_fake_model(theirs.numpy(), float(t))), ref.timesteps[i], theirs).prev_sample
          np.testing.assert_allclose(
              np.asarray(ours), theirs.numpy(), rtol=0, atol=2e-5, err_msg=f"solver_order={solver_order} step {i}"
          )


@unittest.skipIf(torch is None, "needs torch (diffusers reference schedulers)")
class FlowMatchParityTest(unittest.TestCase):
  """`FlaxFlowMatchScheduler` vs `diffusers.FlowMatchEulerDiscreteScheduler`.

  The JAX scheduler builds its unshifted sigmas as `linspace(sigma_max, sigma_min, n)` (DiffSynth convention); the
  diffusers reference is given the same unshifted sigmas and applies its own shift.
  """

  @staticmethod
  def _pair(shift, num_steps, extra_one_step=False):
    scheduler = FlaxFlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=extra_one_step)
    state = scheduler.set_timesteps(scheduler.create_state(), num_inference_steps=num_steps)
    n = num_steps + 1 if extra_one_step else num_steps
    unshifted = np.linspace(1.0, 0.0, n, dtype=np.float32)[:num_steps]
    ref = FlowMatchEulerDiscreteScheduler(shift=shift)
    ref.set_timesteps(sigmas=unshifted.tolist())
    return scheduler, state, ref

  def test_set_timesteps(self):
    for shift in (1.0, 3.0, 5.0):
      for extra_one_step in (False, True):
        with self.subTest(shift=shift, extra_one_step=extra_one_step):
          _, state, ref = self._pair(shift, 20, extra_one_step)
          np.testing.assert_allclose(np.asarray(state.sigmas), ref.sigmas.numpy()[:-1], rtol=0, atol=1e-6)
          np.testing.assert_allclose(np.asarray(state.timesteps), ref.timesteps.numpy(), rtol=0, atol=1e-3)

  def test_sampling_loop(self):
    num_steps = 10
    scheduler, state, ref = self._pair(5.0, num_steps, extra_one_step=True)
    x0 = np.random.default_rng(1).standard_normal(_SHAPE).astype(np.float32)
    ours, theirs = jnp.asarray(x0), torch.from_numpy(x0.copy())
    for i, t in enumerate(np.asarray(state.timesteps)):
      ours, state = scheduler.step(state, jnp.asarray(_fake_model(np.asarray(ours), float(t))), t, ours).to_tuple()
      theirs = ref.step(torch.from_numpy(_fake_model(theirs.numpy(), float(t))), ref.timesteps[i], theirs).prev_sample
      np.testing.assert_allclose(np.asarray(ours), theirs.numpy(), rtol=0, atol=1e-5, err_msg=f"step {i}")

  def test_add_noise(self):
    scheduler, state, ref = self._pair(3.0, 20)
    rng = np.random.default_rng(2)
    sample = rng.standard_normal((3,) + _SHAPE[1:]).astype(np.float32)
    noise = rng.standard_normal((3,) + _SHAPE[1:]).astype(np.float32)
    timesteps = np.asarray(state.timesteps)[[0, 7, 19]]
    ours = scheduler.add_noise(state, jnp.asarray(sample), jnp.asarray(noise), jnp.asarray(timesteps))
    theirs = ref.scale_noise(torch.from_numpy(sample), torch.from_numpy(timesteps), torch.from_numpy(noise))
    np.testing.assert_allclose(np.asarray(ours), theirs.numpy(), rtol=0, atol=1e-6)


class FlowMatchTrainingTest(unittest.TestCase):
  """Training helpers of `FlaxFlowMatchScheduler` (no diffusers counterpart: checked against the formulas)."""

  def test_apply_flow_match(self):
    scheduler = FlaxFlowMatchScheduler(shift=3.0, sigma_min=0.0, sigma_max=1.0)
    rng = np.random.default_rng(3)
    latents = rng.standard_normal((2, 4, 3, 8, 8)).astype(np.float32)
    noise = rng.standard_normal((2, 4, 3, 8, 8)).astype(np.float32)
    timesteps = np.array([250.0, 900.0], dtype=np.float32)
    noisy, target, weights = scheduler.apply_flow_match(jnp.asarray(noise), jnp.asarray(latents), jnp.asarray(timesteps))
    t = (timesteps / 1000.0).reshape((-1, 1, 1, 1, 1))
    np.testing.assert_allclose(np.asarray(noisy), (1 - t) * latents + t * noise, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(target), noise - latents, rtol=0, atol=1e-6)
    self.assertEqual(weights.shape, (2,))

  def test_training_weights_follow_the_schedule(self):
    """Weights are DiffSynth's schedule weights (normalized over the 1000-step training schedule, mean 1).

    Regression for #5418 B3: they used to be normalized over the batch's sampled timesteps, which gave NaN for
    batch 1, weight 0 for the smallest one in every batch, and a ~1000/B loss scale.
    """
    scheduler = FlaxFlowMatchScheduler(shift=3.0)
    schedule = scheduler.set_timesteps(scheduler.create_state(), num_inference_steps=1000, training=True)
    np.testing.assert_allclose(float(jnp.mean(schedule.linear_timesteps_weights)), 1.0, rtol=1e-5)

    def weights(timesteps):
      x = jnp.zeros((len(timesteps), 4, 1, 2, 2))
      return np.asarray(scheduler.apply_flow_match(x, x, jnp.asarray(timesteps, jnp.float32))[2])

    batch = np.array([981.9, 500.0, 120.0, 37.5], np.float32)
    w_batch = weights(batch)
    self.assertTrue(np.all(np.isfinite(w_batch)) and np.all(w_batch > 0))
    for i, t in enumerate(batch):  # each sample's weight does not depend on the rest of the batch
      np.testing.assert_allclose(weights(np.array([t])), w_batch[i : i + 1], rtol=1e-6)
    expected = np.asarray(scheduler.training_weight(schedule, jnp.asarray(batch)))
    np.testing.assert_allclose(w_batch, expected, rtol=1e-6)
    self.assertGreater(w_batch[1], w_batch[0])  # bell around the middle of the schedule

  def test_sample_timesteps_batch_one_is_1d(self):
    timesteps = FlaxFlowMatchScheduler(shift=3.0).sample_timesteps(jax.random.key(0), 1)
    self.assertEqual(timesteps.shape, (1,))

  def test_sample_timesteps_range(self):
    scheduler = FlaxFlowMatchScheduler(shift=3.0)
    timesteps = scheduler.sample_timesteps(jax.random.key(0), 64)
    self.assertEqual(timesteps.shape, (64,))
    self.assertTrue(bool(jnp.all((timesteps >= 0) & (timesteps <= 1000))))


if __name__ == "__main__":
  unittest.main()
