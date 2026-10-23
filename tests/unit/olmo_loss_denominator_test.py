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

"""OLMo-core's loss divisor: filtered (all-masked) instances still count seq_len - 1 tokens."""

import unittest

import jax.numpy as jnp

from maxtext.trainers.pre_train.train import olmo_core_loss_denominator


class OlmoCoreLossDenominatorTest(unittest.TestCase):

  def test_counts_masked_instances(self):
    # 4 instances of 8 positions; the last target of each is ignored (7 trained tokens), rows 1 and 3 are filtered.
    seg = jnp.array([[1] * 7 + [0], [0] * 8, [2] * 7 + [0], [0] * 8])
    total_weights = jnp.sum(seg != 0)
    self.assertEqual(int(total_weights), 14)
    # OLMo-core: every instance contributes seq_len - 1 = 7, filtered or not.
    self.assertEqual(int(olmo_core_loss_denominator(seg, total_weights)), 4 * 7)

  def test_no_masked_instances_is_unchanged(self):
    seg = jnp.array([[1] * 7 + [0], [3] * 7 + [0]])
    total_weights = jnp.sum(seg != 0)
    self.assertEqual(int(olmo_core_loss_denominator(seg, total_weights)), int(total_weights))

  def test_hero_batch_is_constant(self):
    # 2048 instances of 8192, any number filtered: the divisor is always 2048 * 8191.
    for masked in (0, 2, 6):
      seg = jnp.concatenate([jnp.ones((2048 - masked, 8191), jnp.int32), jnp.zeros((2048 - masked, 1), jnp.int32)], 1)
      seg = jnp.concatenate([seg, jnp.zeros((masked, 8192), jnp.int32)])
      self.assertEqual(int(olmo_core_loss_denominator(seg, jnp.sum(seg != 0))), 2048 * 8191)


if __name__ == "__main__":
  unittest.main()
