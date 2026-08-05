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

"""Unit tests for train_sft.py."""

import unittest
from unittest import mock
from types import SimpleNamespace
import pytest

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from maxtext.trainers.post_train.sft import train_sft

pytestmark = [pytest.mark.post_training]


class _StatefulModel(nnx.Module):
  """Model that advances an RNG during the forward pass.

  The unscanned decoder writes back to each layer with `nnx.update`, which mutates the same
  variables. The scanned decoder carries its state through the scan and never writes to the
  module, which is why only unscanned hits this.
  """

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 2, rngs=rngs)
    self.rngs = rngs

  def __call__(self, x):
    self.rngs.default()
    return self.linear(x)


class TrainStepTraceLevelTest(unittest.TestCase):
  """The train step must not hand the loss function variables owned by an outer trace.

  `create_train_step_fn` captures the graphdef outside the transform and merges inside
  `jax.value_and_grad`. Without `copy=True` the merge returns the original variables, and
  writing to them raises `TraceContextError`.
  """

  def _train_step_over(self, model):
    """Builds the real train step over the trainer attributes it reads.

    Args:
      model: Model the step trains.

    Returns:
      The train step function.
    """
    trainer = SimpleNamespace(
        model=model,
        loss_fn=lambda m, x: (jnp.sum(m(x) ** 2), {"aux": 1}),
        _has_aux=True,
        gen_model_input_fn=lambda inputs: inputs,
        _lora_enabled=False,
    )
    return train_sft.MaxTextPeftTrainer.create_train_step_fn(trainer)

  def _build(self):
    model = _StatefulModel(nnx.Rngs(0))
    return model, nnx.Optimizer(model, optax.sgd(1e-2), wrt=nnx.Param)

  @pytest.mark.cpu_only
  def test_a_model_that_writes_to_its_own_state_trains(self):
    model, optimizer = self._build()
    before = jnp.asarray(model.linear.kernel[...])

    # The step returns (loss, aux) or (loss, aux, grad_norm), depending on the Tunix version.
    out = self._train_step_over(model)(model, optimizer, {"x": jnp.ones((1, 2))})

    self.assertTrue(jnp.isfinite(out[0]))
    self.assertFalse(jnp.array_equal(before, model.linear.kernel[...]), "weights did not move")

  @pytest.mark.cpu_only
  def test_the_same_model_trains_under_jit(self):
    """Tunix runs the step under `nnx.jit`, where the trace levels stack up."""
    model, optimizer = self._build()

    out = nnx.jit(self._train_step_over(model))(model, optimizer, {"x": jnp.ones((1, 2))})

    self.assertTrue(jnp.isfinite(out[0]))

  @pytest.mark.cpu_only
  def test_rng_state_updates_reach_the_caller_model(self):
    """RNG counters advance inside the traced loss and have to come back out."""
    model, optimizer = self._build()
    before = jax.tree.leaves(nnx.state(model, nnx.RngCount))

    self._train_step_over(model)(model, optimizer, {"x": jnp.ones((1, 2))})

    after = jax.tree.leaves(nnx.state(model, nnx.RngCount))
    self.assertNotEqual([int(c) for c in before], [int(c) for c in after])


class TrainSFTTest(unittest.TestCase):
  """Tests for train_sft.py."""

  def test_validate_config_valid(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=False,
    )
    # Should not raise any exception
    train_sft.validate_config(config)

  def test_validate_config_invalid_offload(self):
    config = SimpleNamespace(
        optimizer_memory_host_offload=True,
    )
    with self.assertRaisesRegex(ValueError, "optimizer_memory_host_offload=True is not supported"):
      train_sft.validate_config(config)

  def test_train_model_caching_moe(self):
    """Test that NNX graph caching is disabled for MoE models (num_experts > 1)."""
    mt_config = SimpleNamespace(
        logical_axis_rules=[],
        num_experts=8,
    )
    trainer = mock.MagicMock()
    trainer.data_hooks.train_data_iterator = "train_iter"
    trainer.data_hooks.eval_data_iterator = "eval_iter"
    mesh = mock.MagicMock()

    with mock.patch("jax.set_mesh"):
      train_sft.train_model(mt_config, trainer, mesh)

    trainer.train.assert_called_once_with(
        "train_iter",
        "eval_iter",
        cache_nnx_graph=False,
    )

  def test_train_model_caching_dense(self):
    """Test that NNX graph caching is enabled for dense models (num_experts <= 1)."""
    mt_config = SimpleNamespace(
        logical_axis_rules=[],
        num_experts=1,
    )
    trainer = mock.MagicMock()
    trainer.data_hooks.train_data_iterator = "train_iter"
    trainer.data_hooks.eval_data_iterator = "eval_iter"
    mesh = mock.MagicMock()

    with mock.patch("jax.set_mesh"):
      train_sft.train_model(mt_config, trainer, mesh)

    trainer.train.assert_called_once_with(
        "train_iter",
        "eval_iter",
        cache_nnx_graph=True,
    )


if __name__ == "__main__":
  unittest.main()
