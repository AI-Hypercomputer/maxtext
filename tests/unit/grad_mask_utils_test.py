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

"""Tests for per-layer per-token gradient mask (grad_mask_utils)."""

import unittest
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.utils.grad_mask_utils import _grad_mask, maybe_grad_mask


class GradMaskTest(unittest.TestCase):

  def setUp(self):
    self.rng = jax.random.PRNGKey(0)

  def test_forward_is_identity(self):
    """Forward pass must return input unchanged regardless of threshold."""
    x = jax.random.normal(self.rng, (2, 8, 16))
    for thr in [0.5, 1.0, 100.0]:
      y = _grad_mask(x, jnp.float32(thr))
      np.testing.assert_array_equal(np.asarray(y), np.asarray(x))

  def test_backward_below_threshold_passthrough(self):
    """When per-token RMS <= threshold, backward must return g unchanged."""
    x = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
    threshold = jnp.float32(1e6)  # huge → never clip

    def loss_fn(x):
      return jnp.sum(_grad_mask(x, threshold) * 0.5)

    g = jax.grad(loss_fn)(x)
    expected = jnp.full_like(x, 0.5)
    np.testing.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-3)

  def test_backward_outlier_tokens_are_masked(self):
    """Tokens with RMS > threshold get zero gradient; healthy tokens unchanged."""
    x = jnp.zeros((2, 3, 8), dtype=jnp.float32)
    threshold = jnp.float32(1.0)
    upstream = jnp.ones_like(x)
    # Make token (0, 0) an outlier (RMS = 10) and token (1, 2) an outlier (RMS = 100).
    upstream = upstream.at[0, 0].set(10.0)
    upstream = upstream.at[1, 2].set(100.0)

    def fn(x):
      return _grad_mask(x, threshold)

    _, vjp = jax.vjp(fn, x)
    (g_masked,) = vjp(upstream)
    g_masked = np.asarray(g_masked)
    # Outlier tokens zeroed.
    np.testing.assert_array_equal(g_masked[0, 0], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(g_masked[1, 2], np.zeros(8, dtype=np.float32))
    # Healthy tokens (RMS = 1.0 == threshold, passes through).
    np.testing.assert_array_equal(g_masked[0, 1], np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(g_masked[1, 0], np.ones(8, dtype=np.float32))

  def test_backward_threshold_grad_is_zero(self):
    """Threshold arg must receive a zero gradient (it's not differentiable)."""
    x = jnp.ones((2, 4, 8), dtype=jnp.float32)

    def fn(x, threshold):
      return _grad_mask(x, threshold)

    threshold = jnp.float32(1.0)
    _, vjp = jax.vjp(fn, x, threshold)
    upstream = jnp.ones_like(x)
    _, g_threshold = vjp(upstream)
    self.assertEqual(float(g_threshold), 0.0)

  def test_maybe_grad_mask_threshold_zero_is_noop(self):
    """maybe_grad_mask with threshold=0 returns input unchanged and inserts no boundary."""
    Cfg = namedtuple("Cfg", ["grad_mask_threshold"])
    cfg = Cfg(grad_mask_threshold=0.0)
    x = jax.random.normal(self.rng, (2, 4, 8))
    y = maybe_grad_mask(x, cfg)
    self.assertIs(y, x)  # exact identity, no jnp.array wrapping

  def test_maybe_grad_mask_threshold_positive_applies_mask(self):
    """maybe_grad_mask with threshold > 0 zeros tokens whose RMS exceeds threshold."""
    Cfg = namedtuple("Cfg", ["grad_mask_threshold"])
    cfg = Cfg(grad_mask_threshold=0.5)
    x = jnp.zeros((2, 4, 8), dtype=jnp.float32)

    def fn(x):
      return maybe_grad_mask(x, cfg)

    # All tokens have RMS = 10.0 (every element = 10.0); threshold = 0.5 → all masked.
    upstream = jnp.full_like(x, 10.0)
    _, vjp = jax.vjp(fn, x)
    (g,) = vjp(upstream)
    np.testing.assert_array_equal(np.asarray(g), np.zeros_like(np.asarray(x)))

  def test_dtype_preserved_in_backward(self):
    """Backward must preserve the gradient's dtype (bf16 in, bf16 out)."""
    x = jnp.zeros((2, 4, 8), dtype=jnp.bfloat16)
    threshold = jnp.float32(0.1)

    def fn(x):
      return _grad_mask(x, threshold)

    upstream = (jax.random.normal(self.rng, x.shape, dtype=jnp.float32) * 10.0).astype(jnp.bfloat16)
    _, vjp = jax.vjp(fn, x)
    (g,) = vjp(upstream)
    self.assertEqual(g.dtype, jnp.bfloat16)


if __name__ == "__main__":
  unittest.main()
