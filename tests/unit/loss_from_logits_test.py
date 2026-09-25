# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checks the loss boundary shared by standard and dual-pipe training."""

from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.trainers.pre_train import train


class LossFromLogitsTest(unittest.TestCase):
  """Keep token masking, gradient scaling, and z-loss metrics unchanged."""

  def setUp(self):
    super().setUp()
    self.config = SimpleNamespace(vocab_size=4, z_loss_multiplier=0.01, shard_mode=0, debug_sharding=False)
    self.logits = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10
    self.data = {
        "targets": jnp.asarray([[0, 1, 2], [3, 2, 1]]),
        "targets_segmentation": jnp.asarray([[1, 1, 0], [1, 0, 0]]),
    }
    patcher = mock.patch.object(train.sharding, "maybe_shard_with_logical", side_effect=lambda value, *a, **k: value)
    patcher.start()
    self.addCleanup(patcher.stop)

  def reference(self, logits, loss_mask=None):
    xent, z_loss = train.max_utils.cross_entropy_with_logits(
        logits, jax.nn.one_hot(self.data["targets"], self.config.vocab_size), z_loss=self.config.z_loss_multiplier
    )
    mask = self.data["targets_segmentation"] != 0 if loss_mask is None else loss_mask
    count = jnp.sum(mask)
    return jnp.sum(xent * mask), jnp.sum(z_loss * mask), count

  def test_matches_original_masked_loss_and_gradient(self):
    result = train.loss_from_logits(self.logits, self.data, self.config, None)
    expected = self.reference(self.logits)
    for actual, reference in zip(result, expected):
      np.testing.assert_allclose(actual, reference)
    self.assertEqual(int(result[2]), 3)
    actual_gradient = jax.grad(lambda logits: train.loss_from_logits(logits, self.data, self.config, None)[0])(self.logits)
    reference_gradient = jax.grad(lambda logits: self.reference(logits)[0])(self.logits)
    np.testing.assert_allclose(actual_gradient, reference_gradient)
    np.testing.assert_array_equal(actual_gradient[self.data["targets_segmentation"] == 0], 0)

  def test_explicit_loss_mask_preserves_sums_and_gradient(self):
    mask = jnp.asarray([[True, False, False], [True, False, False]])
    result = train.loss_from_logits(self.logits, self.data, self.config, None, loss_mask=mask)
    for actual, expected in zip(result, self.reference(self.logits, mask)):
      np.testing.assert_allclose(actual, expected)
    self.assertEqual(int(result[2]), 2)
    self.assertGreater(float(result[1]), 0.0)
    actual_gradient = jax.grad(
        lambda logits: train.loss_from_logits(logits, self.data, self.config, None, loss_mask=mask)[0]
    )(self.logits)
    reference_gradient = jax.grad(lambda logits: self.reference(logits, mask)[0])(self.logits)
    np.testing.assert_allclose(actual_gradient, reference_gradient)
    np.testing.assert_array_equal(actual_gradient[~mask], 0)

  def test_zero_tokens_return_zero_sums_and_gradient(self):
    self.data["targets_segmentation"] = jnp.zeros_like(self.data["targets_segmentation"])
    result = train.loss_from_logits(self.logits, self.data, self.config, None)
    for value in result:
      self.assertEqual(float(value), 0.0)
    gradient = jax.grad(lambda logits: train.loss_from_logits(logits, self.data, self.config, None)[0])(self.logits)
    np.testing.assert_array_equal(gradient, 0)


if __name__ == "__main__":
  unittest.main()
