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

"""Tests for Dynamic Fine-Tuning (DFT) loss.

DFT rescales per-token cross-entropy by the model's predicted probability:
  dft_loss_t = p(y_t).detach() * (-log p(y_t))
See: https://arxiv.org/abs/2508.05629
"""

import unittest

import jax
import jax.numpy as jnp

from maxtext.utils import max_utils


def _standard_xent(logits, targets_ids, vocab_size):
  """Standard per-token cross-entropy: -log p(y_t)."""
  one_hot = jax.nn.one_hot(targets_ids, vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot)
  return xent


def _dft_xent(logits, targets_ids, vocab_size):
  """DFT per-token cross-entropy: p(y_t).detach() * (-log p(y_t))."""
  xent = _standard_xent(logits, targets_ids, vocab_size)
  return jax.lax.stop_gradient(jnp.exp(-xent)) * xent


class DFTLossTest(unittest.TestCase):
  """Tests for the DFT loss rescaling."""

  def setUp(self):
    self.vocab_size = 8
    self.batch_size = 2
    self.seq_len = 4
    self.rng = jax.random.PRNGKey(42)
    self.logits = jax.random.normal(
        self.rng, (self.batch_size, self.seq_len, self.vocab_size)
    )
    self.targets = jax.random.randint(
        jax.random.PRNGKey(0), (self.batch_size, self.seq_len), 0, self.vocab_size
    )

  def test_dft_loss_less_than_or_equal_standard(self):
    """DFT loss per token should be <= standard cross-entropy since p(y_t) <= 1."""
    std = _standard_xent(self.logits, self.targets, self.vocab_size)
    dft = _dft_xent(self.logits, self.targets, self.vocab_size)
    # p(y_t) in [0, 1], so p(y_t) * (-log p(y_t)) <= (-log p(y_t))
    self.assertTrue(jnp.all(dft <= std + 1e-6))

  def test_dft_loss_non_negative(self):
    """DFT loss should be non-negative."""
    dft = _dft_xent(self.logits, self.targets, self.vocab_size)
    self.assertTrue(jnp.all(dft >= -1e-6))

  def test_dft_equals_standard_when_confident(self):
    """When model is maximally confident (p=1), both losses should be ~0."""
    # Create logits where the correct class has very high probability
    logits = jnp.full((1, 2, self.vocab_size), -1e6)
    targets = jnp.array([[0, 1]])
    logits = logits.at[0, 0, 0].set(100.0)
    logits = logits.at[0, 1, 1].set(100.0)

    std = _standard_xent(logits, targets, self.vocab_size)
    dft = _dft_xent(logits, targets, self.vocab_size)
    # Both should be near zero
    self.assertTrue(jnp.allclose(std, 0.0, atol=1e-3))
    self.assertTrue(jnp.allclose(dft, 0.0, atol=1e-3))

  def test_dft_down_weights_uncertain_tokens(self):
    """DFT should down-weight tokens where the model is uncertain (low p)."""
    # Uniform logits -> p(y_t) = 1/vocab_size for all tokens
    uniform_logits = jnp.zeros((1, 1, self.vocab_size))
    targets = jnp.array([[0]])

    std = _standard_xent(uniform_logits, targets, self.vocab_size)
    dft = _dft_xent(uniform_logits, targets, self.vocab_size)

    # p(y_t) = 1/vocab_size = 0.125, so DFT should be ~0.125 * std
    expected_ratio = 1.0 / self.vocab_size
    actual_ratio = float(dft[0, 0]) / float(std[0, 0])
    self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)

  def test_dft_gradient_differs_from_standard(self):
    """DFT and standard cross-entropy should produce different gradients."""

    def std_loss(logits):
      xent = _standard_xent(logits, self.targets, self.vocab_size)
      return jnp.sum(xent)

    def dft_loss(logits):
      xent = _dft_xent(logits, self.targets, self.vocab_size)
      return jnp.sum(xent)

    grad_std = jax.grad(std_loss)(self.logits)
    grad_dft = jax.grad(dft_loss)(self.logits)

    # Gradients should be different
    self.assertFalse(jnp.allclose(grad_std, grad_dft, atol=1e-6))

  def test_dft_stop_gradient_on_probability(self):
    """Verify stop_gradient is applied to p(y_t) by checking gradient structure.

    Without stop_gradient, the gradient of p*(-log p) = -p*(-1/p) + (-log p)*dp/d...
    = 1 + (-log p)*dp. With stop_gradient on p, it's just p * d(-log p), which
    is much simpler. We verify by comparing against a version without stop_gradient.
    """

    def dft_with_stop_grad(logits):
      xent = _standard_xent(logits, self.targets, self.vocab_size)
      return jnp.sum(jax.lax.stop_gradient(jnp.exp(-xent)) * xent)

    def dft_without_stop_grad(logits):
      xent = _standard_xent(logits, self.targets, self.vocab_size)
      return jnp.sum(jnp.exp(-xent) * xent)

    grad_with = jax.grad(dft_with_stop_grad)(self.logits)
    grad_without = jax.grad(dft_without_stop_grad)(self.logits)

    # These should be different, confirming stop_gradient matters
    self.assertFalse(jnp.allclose(grad_with, grad_without, atol=1e-6))


if __name__ == "__main__":
  unittest.main()
