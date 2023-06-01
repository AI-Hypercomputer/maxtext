# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stats test base to be shared between TF and JAX AQTp ops.

Since both TF and JAX AQTp are supposed to have the exactly same logics and
APIs for all operations, it is desired to share their test cases. It will help
to make sure that both of them behave the same consistently and to maintain
and modify their test cases much easier whenever both the libraries evolve.
"""

from absl import flags
from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.test import aqt_test_shared_base
import numpy as np
import tensorflow.compat.v1 as tf


def f32(x):
  """Creates a float32 numpy array."""
  return np.array(x, dtype=np.float32)


class StatsTest(tf.test.TestCase, parameterized.TestCase):
  """Base class for testing Stats.

  As TF and JAX AQTp have the same API and logic for Stats class, all testing
  cases are put in this base class and are shared between TF and JAX through
  inheritance. On TF and JAX AQTp sides, only few methods which are not
  framework agnostic should be implemented.
  """

  _stats = None

  def set_stats(self, data_shape, config):
    """Sets self._stats with given data shape and stats config.

    Should be implemented in TF and JAX AQTp tests with their own AQT ops.

    Args:
      data_shape: Shape of data for which statistics are collected.
      config: Configuration fot Stats.
    """
    raise NotImplementedError

  def update(self, sample, weight):
    """Updates self._stats with a given tensor and its weight."""

    raise NotImplementedError

  def get_sum_of_ones(self):
    raise NotImplementedError

  def get_sum_of_vals(self):
    raise NotImplementedError

  def get_max_of_abs_vals(self):
    raise NotImplementedError

  def get_sum_of_l1_vals(self):
    raise NotImplementedError

  def get_sum_of_lp_vals(self):
    raise NotImplementedError

  def set_ema_update_count(self, ema_update_count):
    raise NotImplementedError

  def mean(self):
    raise NotImplementedError

  def max_dev(self):
    raise NotImplementedError

  def l1_dev(self):
    raise NotImplementedError

  def lp_dev(self):
    raise NotImplementedError

  def bound(self, calibration_config):
    raise NotImplementedError

  def check_mean(self, mean):
    self.assertAllEqual(self.mean(), f32(mean))

  def check_max_dev(self, expected_max):
    self.assertAllEqual(self.get_max_of_abs_vals(), f32(expected_max))

  def check_l1_dev(self, l1_dev):
    self.assertAllEqual(self.l1_dev(), f32(l1_dev))

  def check_lp_dev(self, lp_dev, approx=False):
    if approx:
      self.assertAllClose(self.lp_dev(), f32(lp_dev))
    else:
      self.assertAllEqual(self.lp_dev(), f32(lp_dev))

  def test_basics_carefully(self):
    """Checks stats against hand-computed results."""
    config = aqt_test_shared_base.test_stats_config()
    config.share_stats_axes[:] = [1]  # share only on rows

    with self.cached_session():
      rl = 4  # row length
      self.set_stats([2, rl], config)

      # update 1
      # ema_update_count = 1 completely overwrites the priors.
      wgt1 = f32([[2], [0.5]])
      self.update([[4, 0, -3, 0], [-40, 30, 0, 0]], wgt1)
      # We will take a peak inside first.
      # pylint: disable=protected-access
      self.assertAllEqual(self.get_sum_of_ones(), wgt1 * rl)
      self.assertAllEqual(self.get_sum_of_vals(),
                          f32([[4 - 3], [-40 + 30]]) * wgt1)
      self.assertAllEqual(self.get_sum_of_l1_vals(),
                          f32([[4 + 3], [40 + 30]]) * wgt1)
      squre_sum = f32([[4 * 4 + 3 * 3], [40 * 40 + 30 * 30]])
      self.assertAllEqual(self.get_sum_of_lp_vals(), squre_sum * wgt1)
      self.check_mean([[0.25], [-2.5]])
      self.check_l1_dev([[1.75], [17.5]])
      self.check_lp_dev([[2.5], [25]])
      self.check_max_dev([[4], [40]])

      # update 2
      # ema_update_count = 1 completely overwrites the update 1.
      # notice the change in weight and _sum_of_ones, but no change in stats.
      self.update([[4, 0, -3, 0], [-40, 30, 0, 0]], [[1], [0.5]])
      self.assertAllEqual(self.get_sum_of_ones(), f32([[1], [0.5]]) * rl)
      self.check_mean([[0.25], [-2.5]])
      self.check_l1_dev([[1.75], [17.5]])
      self.check_lp_dev([[2.5], [25]])
      self.check_max_dev([[4], [40]])

      # update 3
      # ema = 2 will make update 2 and 3 have equal weight.
      # This is the source of the '/ 2'.
      self.set_ema_update_count(2)
      self.update(f32([[10, 10, 0, 0], [50, 0, 0, 0]]), f32([[3], [0.5]]))
      # Functions could make them one-liners and improve readability.
      self.assertAllEqual(self.get_sum_of_ones(),
                          f32([[1 + 3], [0.5 + 0.5]]) / 2 * rl)
      self.assertAllEqual(self.get_sum_of_vals(),
                          f32([[1 * 1 + 20 * 3], [-10 * 0.5 + 50 * 0.5]]) / 2)
      self.assertAllEqual(self.get_sum_of_l1_vals(),
                          f32([[7 * 1 + 20 * 3], [70 * 0.5 + 50 * 0.5]]) / 2)
      self.assertAllEqual(
          self.get_sum_of_lp_vals(),
          f32([[25 * 1 + 200 * 3], [2500 * 0.5 + 2500 * 0.5]]) / 2)
      self.check_mean([[61 / 16], [5]])
      self.check_l1_dev([[67 / 16], [15]])
      self.check_lp_dev([[25 / 4], [25]])
      self.check_max_dev([[4 / 2 + 10 / 2], [50 / 2 + 40 / 2]])

  def test_filter_zeros_and_bound(self):
    """Makes sure that zero filter skips over zeros for statistics."""
    config = aqt_test_shared_base.test_stats_config()
    config.filter_zeros = True
    calibration_config = aqt_config.CalibrationConfig(
        const_bound_coeff=10, l1_dev_coeff=3, lp_dev_coeff=4, max_dev_coeff=100)

    with self.cached_session():
      self.set_stats([1, 5], config)
      self.update([[1, 1, 1, -1, 0]], [[42]])
      self.check_mean([[0.5]])
      self.check_l1_dev([[1]])
      self.check_lp_dev([[1]])
      self.check_max_dev([[1]])
      # bound
      self.assertAllEqual(
          self.bound(calibration_config), f32([[10 + 3 * 1 + 4 * 1 + 100 * 1]]))

  def test_p_norm(self):
    config = aqt_test_shared_base.test_stats_config()
    config.lp_order = 20  # close to max norm

    with self.cached_session():
      self.set_stats([1, 5], config)
      self.update([[1, -2, 3, -40, 0]], [[42]])
      self.check_lp_dev([[36.907234]])  # close to 40

  def test_max_skips_zero_weight(self):
    """Maximum calculation should elide zero-weight rows."""
    config = aqt_test_shared_base.test_stats_config()

    with self.cached_session():
      self.set_stats([1, 1], config)
      self.update([[100], [200]], [[1], [0]])
      self.check_max_dev([[100]])

  def test_odd_norm(self):
    """Validates that odd norms use absolute values."""
    config = aqt_test_shared_base.test_stats_config()
    config.lp_order = 3
    config.ema_update_count = 1

    with self.cached_session():
      acts = [1, -2, 3, -1, 0]
      self.set_stats([1, len(acts)], config)
      self.update([acts], [[1]])
      expected = (sum(abs(x)**3 for x in acts) / len(acts))**(1 / 3)
      self.check_lp_dev([[expected]], approx=True)

  # here we test some properties for a change, any dyadic params should work
  @parameterized.named_parameters(
      dict(op=10, mp=20, l1p=30, lpp=40, mxp=50, ema=8, testcase_name="1"),
      dict(op=10, mp=20, l1p=30, lpp=40, mxp=50, ema=1, testcase_name="2"),
      dict(op=40, mp=30, l1p=20, lpp=10, mxp=100, ema=8, testcase_name="3"),
  )
  def test_prior_and_ema(self, op, mp, l1p, lpp, mxp, ema):
    """Validates that the priors are used for the initial values of variables."""
    config = aqt_test_shared_base.test_stats_config()
    config.update_count_prior = op
    config.mean_prior = mp
    config.l1_dev_prior = l1p
    config.lp_dev_prior = lpp
    config.max_dev_prior = mxp
    config.ema_update_count = ema

    with self.cached_session():
      self.set_stats([1, 1], config)
      # Testing prior first
      self.check_mean([[config.mean_prior]])
      self.check_l1_dev([[config.l1_dev_prior]])
      self.check_lp_dev([[config.lp_dev_prior]])
      self.check_max_dev([[config.max_dev_prior]])
      # Testing decay of all stats given '0' update.
      r = 1 - 1 / (config.ema_update_count)
      self.update([[0]], [[config.update_count_prior]])
      self.check_mean([[config.mean_prior * r]])
      self.check_l1_dev([[config.l1_dev_prior * r]])
      self.check_lp_dev([[config.lp_dev_prior * r**(1. / config.lp_order)]])
      self.check_max_dev([[config.max_dev_prior * r]])

  def test_no_axis_share(self):
    """Tests the case where no axes have shared stats."""
    config = aqt_test_shared_base.test_stats_config()
    # We have only EMA averaging.
    config.share_stats_axes[:] = []

    with self.cached_session():
      self.set_stats([2, 3], config)
      self.update([[1, 2, -3], [4, 5, -6]], [[1, 1, 1], [1, 1, 1]])
      self.check_mean([[1, 2, -3], [4, 5, -6]])
      self.check_l1_dev([[1, 2, 3], [4, 5, 6]])
      self.check_lp_dev([[1, 2, 3], [4, 5, 6]], approx=True)

  # We average across both axis of 2-dim Tensor
  def test_axis_share_both(self):
    config = aqt_test_shared_base.test_stats_config()
    config.share_stats_axes[:] = [0, 1]

    with self.cached_session():
      self.set_stats([2, 2], config)
      self.update([[1, 2], [0, -2]], [[1, 1], [1, 1]])
      self.check_mean([[0.25]])
      self.check_l1_dev([[1.25]])
      self.check_lp_dev([[1.5]], approx=True)

  def test_cross_tpu(self):
    if not flags.FLAGS.use_tpu:
      return
    return  # TODO(lew): implement.
    # config = aqt_test_shared_base.test_stats_config()
    # config.tpu_cross_replica_sum = True
    # config.share_stats_axes[:] = []
    # with self.cached_session():
    #   self.set_stats([2, 3], config)
    #   self.update([[1, 2, -3], [4, 5, -6]], [[1, 1, 1], [1, 1, 1]])
    #   self.check_mean([[1, 2, -3], [4, 5, -6]])
    #   self.check_l1_dev([[1, 2, 3], [4, 5, 6]])
    #   self.check_lp_dev([[1, 2, 3], [4, 5, 6]], approx=True)
    #   # TODO(lew): How to create multi-replica Stats?
