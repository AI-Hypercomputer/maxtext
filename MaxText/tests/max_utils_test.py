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
import os
import unittest

jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""
  def test_l2norm_pytree(self):
    x = {'a': jax.numpy.array([0, 2, 0]), 'b': jax.numpy.array([0, 3, 6])}
    self.assertEqual(max_utils.l2norm_pytree(x), 7)

if __name__ == '__main__':
  unittest.main()

