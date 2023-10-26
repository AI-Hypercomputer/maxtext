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

class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""
  def test_l2norm_pytree(self):
    x = {'a': jax.numpy.array([0, 2, 0]), 'b': jax.numpy.array([0, 3, 6])}
    pytree_l2_norm = max_utils.l2norm_pytree(x)
    self.assertTrue(jax.numpy.allclose(pytree_l2_norm, 7, rtol=1e-05, atol=1e-08, equal_nan=False))

if __name__ == '__main__':
  unittest.main()

