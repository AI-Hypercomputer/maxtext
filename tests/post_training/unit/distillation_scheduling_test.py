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


"""Unit tests for distillation dynamic loss weight scheduling."""

import unittest
from unittest import mock

import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.tpu_only, pytest.mark.post_training]

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from absl.testing import absltest

from maxtext.trainers.post_train.distillation import train_distill
from maxtext.trainers.post_train.distillation import distillation_utils


# Reusable test fixtures
def _make_dummy_outputs(vocab_size=4, seq_len=2, with_features=False, feature_shape=(2, 1, 2, 4)):
  """Creates matching student/teacher DistillationForwardOutput pairs."""
  logits = jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10
  features = jnp.ones(feature_shape) if with_features else None
  student = distillation_utils.DistillationForwardOutput(logits=logits, out_projection_activations=features)
  teacher = distillation_utils.DistillationForwardOutput(logits=logits, out_projection_activations=features)
  return student, teacher


def _make_divergent_outputs(vocab_size=4):
  """Creates student/teacher outputs that disagree, making soft_loss != hard_loss."""
  student = distillation_utils.DistillationForwardOutput(
      logits=jnp.array([[[10.0, -10.0, -10.0, -10.0], [-10.0, 10.0, -10.0, -10.0]]]),
      out_projection_activations=None,
  )
  teacher = distillation_utils.DistillationForwardOutput(
      logits=jnp.array([[[-10.0, -10.0, 10.0, -10.0], [-10.0, -10.0, -10.0, 10.0]]]),
      out_projection_activations=None,
  )
  return student, teacher


def _make_labels(vocab_size=4):
  return jax.nn.one_hot(jnp.array([[0, 1]]), vocab_size)


def _mean(pair):
  """Unpacks a (sum, count) metric tuple into a scalar mean value."""
  s, c = pair
  c_val = float(c)
  return float(s) / c_val if c_val > 0 else float(s)


# pylint: disable=protected-access
class ComputeScheduleTest(unittest.TestCase):
  """Tests for the compute_schedule utility function."""

  def test_constant_returns_start_value(self):
    """Constant schedule returns start_value regardless of step."""
    result = distillation_utils.compute_schedule(
        step=jnp.array(50), max_steps=100, start_value=0.5, end_value=0.1, schedule_type="constant"
    )
    np.testing.assert_allclose(result, 0.5)

  def test_none_end_value_returns_start_value(self):
    """When end_value is None, returns start_value even with a non-constant schedule type."""
    result = distillation_utils.compute_schedule(
        step=jnp.array(50), max_steps=100, start_value=0.5, end_value=None, schedule_type="linear"
    )
    np.testing.assert_allclose(result, 0.5)

  def test_linear_at_boundaries(self):
    """Linear schedule returns exact start/end values at step 0 and max_steps."""
    result_start = distillation_utils.compute_schedule(
        step=jnp.array(0), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result_start, 1.0, atol=1e-6)

    result_end = distillation_utils.compute_schedule(
        step=jnp.array(100), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result_end, 0.0, atol=1e-6)

  def test_linear_midpoint(self):
    """Linear schedule returns the arithmetic midpoint at 50% progress."""
    result = distillation_utils.compute_schedule(
        step=jnp.array(50), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result, 0.5, atol=1e-6)

  def test_cosine_at_boundaries(self):
    """Cosine schedule returns exact start/end values at step 0 and max_steps."""
    result_start = distillation_utils.compute_schedule(
        step=jnp.array(0), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_start, 1.0, atol=1e-6)

    result_end = distillation_utils.compute_schedule(
        step=jnp.array(100), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_end, 0.0, atol=1e-6)

  def test_cosine_midpoint(self):
    """Cosine schedule returns the midpoint at 50% progress (same as linear at this point)."""
    result = distillation_utils.compute_schedule(
        step=jnp.array(50), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="cosine"
    )
    np.testing.assert_allclose(result, 0.5, atol=1e-6)

  def test_cosine_differs_from_linear_at_non_boundary(self):
    """At 25% and 75% progress, cosine values differ significantly from linear."""
    # At 25%: linear=0.75, cosine≈0.854
    result_25 = distillation_utils.compute_schedule(
        step=jnp.array(25), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_25, 0.8535533, atol=1e-4)
    self.assertGreater(float(result_25), 0.80, "Cosine at 25% should be above 0.80, not linear 0.75")

    # At 75%: linear=0.25, cosine≈0.146
    result_75 = distillation_utils.compute_schedule(
        step=jnp.array(75), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_75, 0.1464466, atol=1e-4)
    self.assertLess(float(result_75), 0.20, "Cosine at 75% should be below 0.20, not linear 0.25")

  def test_clamps_beyond_max_steps(self):
    """Step > max_steps clamps to end_value for both linear and cosine."""
    result_linear = distillation_utils.compute_schedule(
        step=jnp.array(200), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result_linear, 0.0, atol=1e-6)

    result_cosine = distillation_utils.compute_schedule(
        step=jnp.array(200), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_cosine, 0.0, atol=1e-6)

  def test_increasing_direction_linear(self):
    """Linear schedule works correctly when end_value > start_value (ramp up)."""
    result_mid = distillation_utils.compute_schedule(
        step=jnp.array(50), max_steps=100, start_value=0.0, end_value=1.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result_mid, 0.5, atol=1e-6)

    result_end = distillation_utils.compute_schedule(
        step=jnp.array(100), max_steps=100, start_value=0.0, end_value=1.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result_end, 1.0, atol=1e-6)

  def test_increasing_direction_cosine(self):
    """Cosine schedule works correctly when end_value > start_value (ramp up)."""
    result_start = distillation_utils.compute_schedule(
        step=jnp.array(0), max_steps=100, start_value=0.2, end_value=0.8, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_start, 0.2, atol=1e-6)

    result_end = distillation_utils.compute_schedule(
        step=jnp.array(100), max_steps=100, start_value=0.2, end_value=0.8, schedule_type="cosine"
    )
    np.testing.assert_allclose(result_end, 0.8, atol=1e-6)

  def test_invalid_schedule_type_raises(self):
    """Unsupported schedule type raises ValueError."""
    with self.assertRaises(ValueError):
      distillation_utils.compute_schedule(
          step=jnp.array(0), max_steps=100, start_value=1.0, end_value=0.0, schedule_type="exponential"
      )


class StrategySchedulingTest(unittest.TestCase):
  """Tests for dynamic scheduling within CombinedDistillationStrategy."""

  def test_alpha_and_temperature_linear_schedule(self):
    """Alpha and temperature follow linear schedules at steps 0, 50, and 100."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        temperature=2.0,
        alpha=0.8,
        beta_feature=0.0,
        alpha_end=0.2,
        alpha_schedule="linear",
        temperature_end=1.0,
        temperature_schedule="linear",
        max_steps=100,
    )

    student, teacher = _make_dummy_outputs()
    labels = _make_labels()

    # Step 0
    _, m0 = strategy.compute_loss(student, teacher, labels, step=jnp.array(0))
    np.testing.assert_allclose(_mean(m0["distill/alpha"]), 0.8, atol=1e-5)
    np.testing.assert_allclose(_mean(m0["distill/temperature"]), 2.0, atol=1e-5)

    # Step 50
    _, m50 = strategy.compute_loss(student, teacher, labels, step=jnp.array(50))
    np.testing.assert_allclose(_mean(m50["distill/alpha"]), 0.5, atol=1e-5)
    np.testing.assert_allclose(_mean(m50["distill/temperature"]), 1.5, atol=1e-5)

    # Step 100
    _, m100 = strategy.compute_loss(student, teacher, labels, step=jnp.array(100))
    np.testing.assert_allclose(_mean(m100["distill/alpha"]), 0.2, atol=1e-5)
    np.testing.assert_allclose(_mean(m100["distill/temperature"]), 1.0, atol=1e-5)

  def test_cosine_alpha_at_strategy_level(self):
    """Cosine alpha at 25% matches the expected cosine value (not linear)."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        temperature=1.0,
        alpha=1.0,
        alpha_end=0.0,
        alpha_schedule="cosine",
        beta_feature=0.0,
        max_steps=100,
    )

    student, teacher = _make_dummy_outputs()
    labels = _make_labels()

    _, metrics = strategy.compute_loss(student, teacher, labels, step=jnp.array(25))
    np.testing.assert_allclose(_mean(metrics["distill/alpha"]), 0.8535533, atol=1e-4)

  def test_step_none_uses_fixed_values(self):
    """When step is None, fixed initial values are used even if schedules are configured."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        temperature=2.0,
        alpha=0.8,
        beta_feature=0.0,
        alpha_end=0.2,
        alpha_schedule="linear",
        max_steps=100,
    )

    student, teacher = _make_dummy_outputs()
    labels = _make_labels()

    _, metrics = strategy.compute_loss(student, teacher, labels)
    np.testing.assert_allclose(_mean(metrics["distill/alpha"]), 0.8, atol=1e-5)
    np.testing.assert_allclose(_mean(metrics["distill/temperature"]), 2.0, atol=1e-5)

  def test_beta_schedule_with_feature_loss(self):
    """Beta scheduling with actual L2 feature loss: feature_loss scales proportionally."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        temperature=1.0,
        alpha=0.5,
        beta_feature=1.0,
        beta_end=0.0,
        beta_schedule="linear",
        feature_loss_type="l2",
        max_steps=100,
    )

    # Divergent features so feature loss is non-trivial
    student = distillation_utils.DistillationForwardOutput(
        logits=jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10,
        out_projection_activations=jnp.zeros((2, 1, 2, 4)),
    )
    teacher = distillation_utils.DistillationForwardOutput(
        logits=jnp.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]) * 10,
        out_projection_activations=jnp.ones((2, 1, 2, 4)),
    )
    labels = _make_labels()

    # Step 0: beta=1.0
    _, m0 = strategy.compute_loss(student, teacher, labels, step=jnp.array(0))
    self.assertGreater(_mean(m0["distill/out_proj_feature_loss"]), 0.0)
    np.testing.assert_allclose(_mean(m0["distill/beta_feature"]), 1.0, atol=1e-5)

    # Step 50: beta=0.5, feature loss should be half
    _, m50 = strategy.compute_loss(student, teacher, labels, step=jnp.array(50))
    np.testing.assert_allclose(_mean(m50["distill/beta_feature"]), 0.5, atol=1e-5)
    np.testing.assert_allclose(
        _mean(m50["distill/out_proj_feature_loss"]),
        _mean(m0["distill/out_proj_feature_loss"]) * 0.5,
        atol=1e-5,
    )

    # Step 100: beta=0.0, feature loss should be zero
    _, m100 = strategy.compute_loss(student, teacher, labels, step=jnp.array(100))
    np.testing.assert_allclose(_mean(m100["distill/beta_feature"]), 0.0, atol=1e-5)
    np.testing.assert_allclose(_mean(m100["distill/out_proj_feature_loss"]), 0.0, atol=1e-5)

  def test_alpha_schedule_affects_total_loss(self):
    """Alpha=1.0 gives pure soft_loss; alpha=0.0 gives pure hard_loss; they differ."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        temperature=1.0,
        alpha=1.0,
        alpha_end=0.0,
        alpha_schedule="linear",
        beta_feature=0.0,
        max_steps=100,
    )

    student, teacher = _make_divergent_outputs()
    labels = _make_labels()

    # alpha=1.0 -> total = soft_loss
    loss_start, m_start = strategy.compute_loss(student, teacher, labels, step=jnp.array(0))
    np.testing.assert_allclose(float(loss_start), _mean(m_start["distill/soft_loss"]), atol=1e-5)

    # alpha=0.0 -> total = hard_loss
    loss_end, m_end = strategy.compute_loss(student, teacher, labels, step=jnp.array(100))
    np.testing.assert_allclose(float(loss_end), _mean(m_end["distill/hard_loss"]), atol=1e-5)

    # The two should differ
    self.assertNotAlmostEqual(float(loss_start), float(loss_end), places=2)


class StrategyValidationTest(unittest.TestCase):
  """Tests for configuration validation in CombinedDistillationStrategy."""

  def test_beta_zero_with_beta_end_raises(self):
    """beta_feature=0 with beta_end>0 is a misconfiguration and raises ValueError."""
    with self.assertRaisesRegex(ValueError, "distill_beta=0.0 but distill_beta_end="):
      distillation_utils.CombinedDistillationStrategy(
          student_forward_fn=lambda m, **k: None,
          teacher_forward_fn=lambda m, **k: None,
          vocab_size=4,
          beta_feature=0.0,
          beta_end=0.5,
          beta_schedule="linear",
          max_steps=100,
      )

  def test_beta_zero_with_beta_end_none_is_allowed(self):
    """beta_feature=0 with beta_end=None (no schedule) is valid."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        beta_feature=0.0,
        beta_end=None,
        max_steps=100,
    )
    self.assertEqual(strategy.beta_feature, 0.0)

  def test_beta_zero_with_beta_end_zero_is_allowed(self):
    """beta_feature=0 with beta_end=0 is valid (both are zero)."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        beta_feature=0.0,
        beta_end=0.0,
        beta_schedule="linear",
        max_steps=100,
    )
    self.assertEqual(strategy.beta_feature, 0.0)


class TrainerStepCounterTest(unittest.TestCase):
  """Tests for ModelBundle.training_step and its integration with _train_step."""

  def test_model_bundle_initializes_step_to_zero(self):
    """ModelBundle.training_step starts at 0."""

    class DummyModel(nnx.Module):

      def __init__(self):
        self.linear = nnx.Linear(in_features=2, out_features=2, rngs=nnx.Rngs(0))

      def __call__(self, x):
        return self.linear(x)

    bundle = train_distill.ModelBundle(teacher_model=DummyModel(), student_model=DummyModel())
    self.assertEqual(int(bundle.training_step[...]), 0)

  def test_model_bundle_step_increment(self):
    """ModelBundle.training_step can be incremented."""

    class DummyModel(nnx.Module):

      def __init__(self):
        self.linear = nnx.Linear(in_features=2, out_features=2, rngs=nnx.Rngs(0))

      def __call__(self, x):
        return self.linear(x)

    bundle = train_distill.ModelBundle(teacher_model=DummyModel(), student_model=DummyModel())
    bundle.training_step[...] = bundle.training_step[...] + 1
    self.assertEqual(int(bundle.training_step[...]), 1)
    bundle.training_step[...] = bundle.training_step[...] + 1
    self.assertEqual(int(bundle.training_step[...]), 2)

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.optax.global_norm")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.nnx.value_and_grad")
  def test_train_step_increments_and_passes_step(self, mock_value_and_grad, mock_global_norm):
    """_train_step passes pre-increment step to compute_loss and increments after."""
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()
    trainer.wrt_filter = lambda path, x: True  # type: ignore

    # Use a real DistillationForwardOutput so jax.tree.map(stop_gradient, ...) works
    # when loss_wrapper is invoked manually below.
    fake_teacher_output = distillation_utils.DistillationForwardOutput(
        logits=jnp.zeros((1, 2, 4)), out_projection_activations=None
    )
    mock_batch = {
        "input_tokens": mock.Mock(),
        "positions": mock.Mock(),
        "attention_mask": mock.Mock(),
        "decoder_segment_ids": mock.Mock(),
        "targets": mock.Mock(),
        "teacher_output": fake_teacher_output,
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=mock_batch)

    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)
    optimizer = mock.Mock()

    # Simulate resume from step 5
    model_bundle.training_step.set_value(jnp.array(5, dtype=jnp.int32))

    mock_grad_fn = mock.Mock(return_value=((mock.Mock(), {}), mock.Mock()))
    mock_value_and_grad.return_value = mock_grad_fn
    mock_global_norm.return_value = mock.Mock()

    trainer._train_step(model_bundle, optimizer, mock.Mock())

    # Step should have incremented to 6
    self.assertEqual(int(model_bundle.training_step[...]), 6)

    # Trigger loss_wrapper to verify step=5 was passed to compute_loss
    loss_wrapper = mock_value_and_grad.call_args[0][0]
    loss_wrapper(student_model, teacher_model, mock_batch)

    call_kwargs = trainer.strategy.compute_loss.call_args
    self.assertIn("step", call_kwargs.kwargs)
    self.assertEqual(int(call_kwargs.kwargs["step"]), 5)

  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.optax.global_norm")
  @mock.patch("maxtext.trainers.post_train.distillation.train_distill.nnx.value_and_grad")
  def test_consecutive_train_steps_increment(self, mock_value_and_grad, mock_global_norm):
    """training_step increments 0→1→2→3 across consecutive _train_step calls."""
    # pylint: disable=no-value-for-parameter
    trainer = train_distill.MaxTextDistillationTrainer.__new__(train_distill.MaxTextDistillationTrainer)
    trainer.strategy = mock.Mock()
    trainer.wrt_filter = lambda path, x: True  # type: ignore

    mock_batch = {
        "input_tokens": mock.Mock(),
        "positions": mock.Mock(),
        "targets": mock.Mock(),
        "teacher_output": mock.Mock(),
    }
    trainer.gen_model_input_fn = mock.Mock(return_value=mock_batch)

    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)
    optimizer = mock.Mock()

    mock_grad_fn = mock.Mock(return_value=((mock.Mock(), {}), mock.Mock()))
    mock_value_and_grad.return_value = mock_grad_fn
    mock_global_norm.return_value = mock.Mock()

    self.assertEqual(int(model_bundle.training_step[...]), 0)
    trainer._train_step(model_bundle, optimizer, mock.Mock())
    self.assertEqual(int(model_bundle.training_step[...]), 1)
    trainer._train_step(model_bundle, optimizer, mock.Mock())
    self.assertEqual(int(model_bundle.training_step[...]), 2)
    trainer._train_step(model_bundle, optimizer, mock.Mock())
    self.assertEqual(int(model_bundle.training_step[...]), 3)


class GetScheduledWeightsTest(unittest.TestCase):
  """Direct tests for _get_scheduled_weights dispatch logic."""

  def _make_strategy(self, **kwargs):
    """Create a CombinedDistillationStrategy with test defaults, overridden by kwargs."""
    defaults = {
        "student_forward_fn": lambda m, **k: None,
        "teacher_forward_fn": lambda m, **k: None,
        "vocab_size": 4,
        "temperature": 2.0,
        "alpha": 0.8,
        "beta_feature": 0.0,
        "max_steps": 100,
    }
    defaults.update(kwargs)
    return distillation_utils.CombinedDistillationStrategy(**defaults)

  def test_returns_fixed_when_step_is_none(self):
    """_get_scheduled_weights returns initial values when step is None."""
    strategy = self._make_strategy(alpha_end=0.0, alpha_schedule="linear")
    alpha, temperature, beta = strategy._get_scheduled_weights(step=None)
    np.testing.assert_allclose(float(alpha), 0.8)
    np.testing.assert_allclose(float(temperature), 2.0)
    np.testing.assert_allclose(float(beta), 0.0)

  def test_each_param_scheduled_independently(self):
    """Alpha uses cosine, temperature uses linear, beta stays constant — all in one strategy."""
    strategy = self._make_strategy(
        beta_feature=0.5,
        alpha_end=0.0,
        alpha_schedule="cosine",
        temperature_end=1.0,
        temperature_schedule="linear",
        # beta: no schedule configured → constant
    )
    alpha, temperature, beta = strategy._get_scheduled_weights(step=jnp.array(50))

    # alpha: cosine from 0.8 to 0.0 at 50% → 0.4
    np.testing.assert_allclose(float(alpha), 0.4, atol=1e-5)
    # temperature: linear from 2.0 to 1.0 at 50% → 1.5
    np.testing.assert_allclose(float(temperature), 1.5, atol=1e-5)
    # beta: no schedule → stays at 0.5
    np.testing.assert_allclose(float(beta), 0.5, atol=1e-5)


class ScheduleEdgeCasesTest(unittest.TestCase):
  """Edge cases for compute_schedule."""

  def test_max_steps_one(self):
    """max_steps=1 should jump to end_value at step 1."""
    result = distillation_utils.compute_schedule(
        step=jnp.array(0), max_steps=1, start_value=1.0, end_value=0.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result, 1.0, atol=1e-6)

    result = distillation_utils.compute_schedule(
        step=jnp.array(1), max_steps=1, start_value=1.0, end_value=0.0, schedule_type="linear"
    )
    np.testing.assert_allclose(result, 0.0, atol=1e-6)

  def test_start_equals_end(self):
    """start_value == end_value returns that value regardless of schedule type or step."""
    for stype in ("linear", "cosine"):
      result = distillation_utils.compute_schedule(
          step=jnp.array(50), max_steps=100, start_value=0.5, end_value=0.5, schedule_type=stype
      )
      np.testing.assert_allclose(result, 0.5, atol=1e-6, err_msg=f"Failed for {stype}")

  def test_step_zero_always_returns_start(self):
    """Step 0 returns start_value for all schedule types."""
    for stype in ("linear", "cosine"):
      result = distillation_utils.compute_schedule(
          step=jnp.array(0), max_steps=100, start_value=0.7, end_value=0.3, schedule_type=stype
      )
      np.testing.assert_allclose(result, 0.7, atol=1e-6, err_msg=f"Failed for {stype}")


class TemperatureScheduleEffectTest(unittest.TestCase):
  """Verifies temperature schedule actually affects loss values, not just metrics."""

  def test_temperature_schedule_changes_soft_loss(self):
    """Different temperatures produce different soft_loss values (not just metric logging)."""
    strategy = distillation_utils.CombinedDistillationStrategy(
        student_forward_fn=lambda m, **k: None,
        teacher_forward_fn=lambda m, **k: None,
        vocab_size=4,
        temperature=4.0,
        temperature_end=1.0,
        temperature_schedule="linear",
        alpha=1.0,  # pure soft loss
        beta_feature=0.0,
        max_steps=100,
    )

    # Student and teacher disagree so soft_loss is non-trivial
    student, teacher = _make_divergent_outputs()
    labels = _make_labels()

    _, m_high_temp = strategy.compute_loss(student, teacher, labels, step=jnp.array(0))  # temp=4.0
    _, m_low_temp = strategy.compute_loss(student, teacher, labels, step=jnp.array(100))  # temp=1.0

    # Different temperatures must produce different soft_loss values
    self.assertNotAlmostEqual(
        _mean(m_high_temp["distill/soft_loss"]),
        _mean(m_low_temp["distill/soft_loss"]),
        places=1,
        msg="Temperature schedule should change soft_loss, not just the metric value",
    )


class CheckpointStepSyncTest(unittest.TestCase):
  """Verifies the step counter sync after checkpoint restore."""

  def test_step_counter_sync_from_trainer(self):
    """ModelBundle.training_step is synced from trainer._train_steps after restore."""
    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)

    # Simulate what train_distill() does after setup_checkpoint_manager_and_restore
    restored_step = 42
    model_bundle.training_step.set_value(jnp.array(restored_step, dtype=jnp.int32))

    self.assertEqual(int(model_bundle.training_step[...]), 42)

  def test_step_counter_defaults_to_zero_without_restore(self):
    """Without restore, training_step stays at 0."""
    teacher_model, student_model = mock.Mock(), mock.Mock()
    model_bundle = train_distill.ModelBundle(teacher_model=teacher_model, student_model=student_model)

    self.assertEqual(int(model_bundle.training_step[...]), 0)


if __name__ == "__main__":
  absltest.main()
