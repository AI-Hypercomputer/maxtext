""" Tests for the common Max Utils """
import jax
import max_utils
import unittest

jax.config.update('jax_platform_name', 'cpu')

class MaxUtilsSummaryStats(unittest.TestCase):
  """Tests for the summary stats functions in max_utils.py"""
  def test_l2norm_pytree(self):
    x = {'a': jax.numpy.array([0, 2, 0]), 'b': jax.numpy.array([0, 3, 6])}
    self.assertEqual(max_utils.l2norm_pytree(x), 7)

if __name__ == '__main__':
  unittest.main()

