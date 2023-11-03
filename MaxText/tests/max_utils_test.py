"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Tests for the common Max Utils """
import jax
import max_utils
import unittest
import optax
from flax import linen as nn


class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""
  def test_l2norm_pytree(self):
    x = {'a': jax.numpy.array([0, 2, 0]), 'b': jax.numpy.array([0, 3, 6])}
    pytree_l2_norm = max_utils.l2norm_pytree(x)
    self.assertTrue(jax.numpy.allclose(pytree_l2_norm, 7, rtol=1e-05, atol=1e-08, equal_nan=False))

class MaxUtilsT5XCrossEntropy(unittest.TestCase):
  """Tests for the cross entropy functions in max_utils.py"""
  def test_t5x_cross_entropy(self):
    # Generate random targets and logits
    key = jax.random.PRNGKey(0)
    targets = jax.random.randint(key, shape=(48, 2048),
                                        dtype=jax.numpy.int32, minval=1, maxval=10)
    logits = jax.random.uniform(key, shape=(48, 2048, 4096),
                                        dtype=jax.numpy.float32)

    # Calculate xent from optax implementation
    optax_xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    # Calculate xent from custom T5X implementation
    one_hot_targets = jax.nn.one_hot(targets, 4096)
    t5x_xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    t5x_xent = nn.with_logical_constraint(t5x_xent, ('activation_batch', 'activation_length'))

    # Compare results
    self.assertTrue(jax.numpy.allclose(optax_xent, t5x_xent, rtol=1e-05, atol=1e-08, equal_nan=False))

if __name__ == '__main__':
  unittest.main()
