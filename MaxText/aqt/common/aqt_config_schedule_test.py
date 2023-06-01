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

"""AQT operation config scheduling test.

Accurate Quantized Training (AQT) operations allow parameterizing their
configuration by an event count which may change over the course of training. As
a result, different quantization configurations may be applied depending on the
value of an event count tensor containing a scalar integer "time" value.

This test ensures that the appropriate configuration is applied at the
time that the config says it will be applied.
"""

import functools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from aqt.common import aqt_config
from aqt.tensorflow import aqt_ops
from aqt.tensorflow import aqt_tensor
import numpy as np
import tensorflow.compat.v1 as tf


def make_config_schedule_cases():
  # TODO(vladf): migrate this test to the class-based api once
  # conv2d and einsum have a class-based op version.
  matmul = {
      "lhs_share_stats": [0, 1],
      "rhs_share_stats": [0, 1],
      "lhs_shape": [3, 4],
      "rhs_shape": [4, 5],
      "op": aqt_ops.aqt_matmul,
      "testcase_name": "matmul"
  }

  conv2d = {
      "lhs_share_stats": [1, 2, 3],
      "rhs_share_stats": [0, 1, 2],
      "lhs_shape": [2, 5, 5, 1],
      "rhs_shape": [2, 3, 1, 1],
      "op": functools.partial(aqt_ops.aqt_conv2d, strides=1, padding="VALID"),
      "testcase_name": "conv2d"
  }

  einsum = {
      "lhs_share_stats": [0, 1, 2],
      "rhs_share_stats": [0, 2],
      "lhs_shape": [2, 1, 2, 2],
      "rhs_shape": [1, 1, 2],
      "op": functools.partial(aqt_ops.aqt_einsum, "ijkl,jmk->mil"),
      "testcase_name": "einsum"
  }

  return [matmul, conv2d, einsum]


class ConfigScheduleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(make_config_schedule_cases())
  def test_config_schedule(self, lhs_share_stats, rhs_share_stats, lhs_shape,
                           rhs_shape, op):
    """Validates that the config schedule is cased appropriately.

    Construct a schedule for quantization which differs across time
    (i.e., a value of `event_count`). At every given point in time `t`,
    construct an analogous timeless quantization config that's always
    active. Ensure that the resulting quantization is identical
    for when `event_count=t` in the time-varying config.

    Note this property only holds when not freezing scale. Since quantization
    scales are stateful and update as a function of the inputs they observe,
    this test also ensures that the observed calibration statistics are
    not modified as a result of the presence of other quantization
    configurations.

    Since this test only evaluates a behavioral equality between operations with
    different configurations (timeless vs time-varying), it's agnostic to the
    actual operation being performed. For this reason, we reuse the test
    across multiple bivariate AQT operations via test case parameterization.

    Args:
      lhs_share_stats: the axes for statistics sharing on the lhs.
      rhs_share_stats: the axes for statistics sharing on the rhs.
      lhs_shape: the shape for statistics sharing on the lhs.
      rhs_shape: the shape for statistics sharing on the rhs.
      op: a callable accepting lhs/rhs quantizers and the corresponding tensors.
    """

    # For lhs, use a boring constant configuration. Note constant bound
    # coeff should cover about 67% of lhs values as they'll be standard normal.
    lhs_stats_config = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=lhs_share_stats,
        update_count_prior=0,
        tpu_cross_replica_sum=False)
    lhs_calibration_config = aqt_config.CalibrationConfig(const_bound_coeff=1)

    # For rhs, we toggle statistics updates to include/exclude zeros and
    # generate rhs from a distribution which creates zeros.
    rhs_stats_config = aqt_config.StatsConfig(
        ema_update_count=100,
        share_stats_axes=rhs_share_stats,
        update_count_prior=0,
        tpu_cross_replica_sum=False,
        l1_dev_prior=1,
        filter_zeros=True)
    rhs_calibration_config = aqt_config.CalibrationConfig(
        const_bound_coeff=1, l1_dev_coeff=1)

    def tensor_configs(lo, hi, bits):
      """Generates lhs and rhs tensor configs with specified params."""
      int_quant_config = aqt_config.IntQuantConfig(bits, preserve_zero=True)
      common_params = {
          "quant_config": int_quant_config,
          "freeze_scale_at_begin": False,
          "begin_at_event": lo,
          "end_at_event": hi
      }

      lhs_config = aqt_config.AqtTensorConfig(
          calibration_config=lhs_calibration_config,
          **common_params)
      rhs_config = aqt_config.AqtTensorConfig(
          calibration_config=rhs_calibration_config,
          **common_params)
      return [lhs_config], [rhs_config]

    def tensor_quantizer(configs, lhs_or_rhs):
      assert lhs_or_rhs in ["lhs", "rhs"], lhs_or_rhs
      stats_config = (
          lhs_stats_config if lhs_or_rhs == "lhs" else rhs_stats_config)
      config = aqt_config.AqtScheduleConfig(stats_config, configs)
      shape = lhs_shape if lhs_or_rhs == "lhs" else rhs_shape
      return aqt_tensor.TensorQuantizer(shape, config)

    def quantizer_pairs(lhs_tensor_configs, rhs_tensor_configs):
      with tf.variable_scope(None, default_name="quantizers"):
        with tf.variable_scope("lhs"):
          lhs_quantizer = tensor_quantizer(lhs_tensor_configs, "lhs")
        with tf.variable_scope("rhs"):
          rhs_quantizer = tensor_quantizer(rhs_tensor_configs, "rhs")
      return lhs_quantizer, rhs_quantizer

    lhs_ph = tf.placeholder(tf.float32, lhs_shape)
    rhs_ph = tf.placeholder(tf.float32, rhs_shape)

    def make_op(quantizers, event_count=tf.constant(0, tf.int64)):
      lhs_q, rhs_q = quantizers
      lhs_update = lhs_q.update(lhs_ph, None, event_count)
      rhs_update = rhs_q.update(rhs_ph, None, event_count)
      with tf.control_dependencies([lhs_update, rhs_update]):
        return op(lhs_q, lhs_ph, rhs_q, rhs_ph)

    # Exercise no quantization before any events occur.
    test_event_counts = [1]
    tf_expecteds = [make_op(quantizer_pairs([], []))]

    # Exercise multiple different quantizations within the schedule.
    time_varying_lhs_config, time_varying_rhs_config = [], []
    intervals = [(3, 5), (6, 7), (8, 9)]
    bits = [16, 4, 8]
    for (lo, hi), bits in zip(intervals, bits):
      lhs, rhs = tensor_configs(lo, hi, bits)
      time_varying_lhs_config += lhs
      time_varying_rhs_config += rhs

      for event_count in range(lo, hi):
        test_event_counts.append(event_count)
        tf_expecteds.append(
            make_op(quantizer_pairs(*tensor_configs(None, None, bits))))

    # Exercise no quantization before after all events occur.
    test_event_counts.append(10)
    tf_expecteds.append(make_op(quantizer_pairs([], [])))

    event_count_ph = tf.placeholder(tf.int64)
    tf_actual = make_op(
        quantizer_pairs(time_varying_lhs_config, time_varying_rhs_config),
        event_count_ph)

    # Since ops are deterministic, any nontrivial inputs will do for teasing out
    # differences. Use a half-sparse rhs to tease out zero preservation
    # configuration differences.
    rng = np.random.default_rng(1234)

    with self.cached_session():
      tf.global_variables_initializer().run()

      for i, event_count in enumerate(test_event_counts):
        lhs_value = rng.standard_normal(size=lhs_shape, dtype=np.float32)
        rhs_value = rng.standard_normal(size=rhs_shape, dtype=np.float32)
        mask = rng.choice(2, size=rhs_shape)
        rhs_value *= mask

        feed_dict = {
            event_count_ph: event_count,
            lhs_ph: lhs_value,
            rhs_ph: rhs_value
        }
        logging.info("actual eval event_count=%s", event_count)
        actual = tf_actual.eval(feed_dict)
        # Make sure to run every expected test config to update their stats,
        # even though we should only equal the i-th.
        logging.info("expected eval event_count=%s", event_count)
        expecteds = [timeless.eval(feed_dict) for timeless in tf_expecteds]
        self.assertAllEqual(actual, expecteds[i], msg=f"at event {event_count}")

  @parameterized.named_parameters(make_config_schedule_cases())
  def test_inference_schedule(self, lhs_share_stats, rhs_share_stats, lhs_shape,
                              rhs_shape, op):
    """Like test_config_schedule, but validates inference scheduling."""

    # We intentionally use constant calibration, so that we can compare
    # quantizers with different numbers of updates but identically-configured
    # serving configs.

    def make_tensor_config(lo, const_bound):
      """Generates constant-bound tensor config."""
      # Just to make use of int8 ops more variable, choose bitwidth dynamically.
      bits = 8 if const_bound <= 7 else 10
      int_quant_config = aqt_config.IntQuantConfig(
          bits=bits, preserve_zero=True)
      calibration_config = aqt_config.CalibrationConfig(
          const_bound_coeff=const_bound)
      return aqt_config.AqtTensorConfig(
          calibration_config=calibration_config,
          quant_config=int_quant_config,
          freeze_scale_at_begin=False,
          begin_at_event=lo,
          end_at_event=lo + 1)

    def make_stats_config(share_stats):
      return aqt_config.StatsConfig(
          ema_update_count=1,
          share_stats_axes=share_stats,
          tpu_cross_replica_sum=False)

    def tensor_quantizer(configs, lhs_or_rhs, inference_config_index):
      assert lhs_or_rhs in ["lhs", "rhs"], lhs_or_rhs
      share_stats = (
          lhs_share_stats if lhs_or_rhs == "lhs" else rhs_share_stats)
      stats_config = make_stats_config(share_stats)
      use_quantized_variable = False
      config = aqt_config.AqtScheduleConfig(stats_config, configs,
                                            use_quantized_variable,
                                            inference_config_index)
      shape = lhs_shape if lhs_or_rhs == "lhs" else rhs_shape
      return aqt_tensor.TensorQuantizer(shape, config)

    def quantizer_pairs(lhs_configs, rhs_configs, inference_index):
      with tf.variable_scope(None, default_name="quantizers"):
        with tf.variable_scope("lhs"):
          lhs_quantizer = tensor_quantizer(lhs_configs, "lhs", inference_index)
        with tf.variable_scope("rhs"):
          rhs_quantizer = tensor_quantizer(rhs_configs, "rhs", inference_index)
      return lhs_quantizer, rhs_quantizer

    lhs_tensor_configs = [make_tensor_config(i, i) for i in range(1, 10)]
    rhs_tensor_configs = [make_tensor_config(i, 10 - i) for i in range(1, 10)]
    dynamic_quantizers = quantizer_pairs(lhs_tensor_configs, rhs_tensor_configs,
                                         None)
    static_quantizerss = [
        quantizer_pairs(lhs_tensor_configs, rhs_tensor_configs, i - 1)
        for i in range(1, 10)
    ]

    lhs_ph = tf.placeholder(tf.float32, lhs_shape)
    rhs_ph = tf.placeholder(tf.float32, rhs_shape)
    event_count_ph = tf.placeholder(tf.int64, [])

    def update_quantizers_op(quantizers):
      lhs_q, rhs_q = quantizers
      return tf.group([
          lhs_q.update(lhs_ph, None, event_count_ph),
          rhs_q.update(rhs_ph, None, event_count_ph)
      ])

    update_dynamic_quantizers = update_quantizers_op(dynamic_quantizers)
    update_static_quantizerss = [
        update_quantizers_op(static_quantizers)
        for static_quantizers in static_quantizerss
    ]

    def make_serving_op(quantizers):
      lhs_q, rhs_q = quantizers
      return op(lhs_q, lhs_ph, rhs_q, rhs_ph, train=False)

    dynamic_op = make_serving_op(dynamic_quantizers)
    static_ops = [
        make_serving_op(static_quantizers)
        for static_quantizers in static_quantizerss
    ]

    rng = np.random.default_rng(1234)

    with self.cached_session() as sess:
      tf.global_variables_initializer().run()
      lhs_value = rng.standard_normal(size=lhs_shape, dtype=np.float32)
      rhs_value = rng.standard_normal(size=rhs_shape, dtype=np.float32)

      # Initialize dynamic quantizer at time 0.
      feed_dict = {lhs_ph: lhs_value, rhs_ph: rhs_value, event_count_ph: 0}
      sess.run(update_dynamic_quantizers, feed_dict)

      for i in range(1, 10):
        lhs_value = rng.standard_normal(size=lhs_shape, dtype=np.float32)
        rhs_value = rng.standard_normal(size=rhs_shape, dtype=np.float32)
        feed_dict = {lhs_ph: lhs_value, rhs_ph: rhs_value, event_count_ph: i}
        sess.run(update_dynamic_quantizers, feed_dict)
        sess.run(update_static_quantizerss[i - 1], feed_dict)

        dynamic_result = sess.run(dynamic_op, feed_dict)
        static_result = sess.run(static_ops[i - 1], feed_dict)
        self.assertAllEqual(
            dynamic_result,
            static_result,
            msg=f"dynamic event {i} static event {i}")

        # Invasively update the quantizers' last update.
        for q in static_quantizerss[i - 1]:
          sess.run(q._last_update.assign(tf.zeros_like(q._last_update)))

        static_result = sess.run(static_ops[i - 1], feed_dict)
        self.assertAllEqual(
            dynamic_result,
            static_result,
            msg=f"dynamic event {i} static event 0")


if __name__ == "__main__":
  absltest.main()
