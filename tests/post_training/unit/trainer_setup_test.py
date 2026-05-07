# Copyright 2023-2026 Google LLC
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

"""Unit tests for trainer_setup.py (CPU-only)."""

import unittest
import pytest
from types import SimpleNamespace

from maxtext.trainers.post_train.rl import trainer_setup

pytestmark = [pytest.mark.post_training]

class TestGetOptimizer(unittest.TestCase):
  """Tests for trainer_setup.get_optimizer."""

  def _make_optimizer_config(self, gradient_clipping_threshold=0.0):
    return SimpleNamespace(
        learning_rate=1e-4,
        warmup_steps_fraction=0.1,
        gradient_clipping_threshold=gradient_clipping_threshold,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_weight_decay=0.01,
    )

  @pytest.mark.cpu_only
  def test_returns_optimizer_without_clipping(self):
    """get_optimizer returns an optax optimizer when gradient clipping is disabled."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    config = self._make_optimizer_config(gradient_clipping_threshold=0.0)
    opt = trainer_setup.get_optimizer(config, max_train_steps=100)
    # Should be usable: init on a simple param tree
    params = {"w": jnp.ones(3)}
    state = opt.init(params)
    self.assertIn("learning_rate", state.hyperparams)

  @pytest.mark.cpu_only
  def test_returns_optimizer_with_clipping(self):
    """get_optimizer includes gradient clipping when threshold > 0."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    config = self._make_optimizer_config(gradient_clipping_threshold=1.0)
    opt = trainer_setup.get_optimizer(config, max_train_steps=100)
    params = {"w": jnp.ones(3)}
    state = opt.init(params)
    self.assertIn("learning_rate", state.hyperparams)

