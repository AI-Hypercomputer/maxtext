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

"""Tests for matmul."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.tensorflow import aqt_matmul
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_matmul_test_base
import numpy as np
import tensorflow.compat.v1 as tf


def update_event_count(matmul, event_count_int: int):
  """Update the quantizer's event count without changing stats."""
  for quantizer in [matmul.lhs_quantizer, matmul.rhs_quantizer]:
    sample = tf.zeros(quantizer.data_shape)
    weights = tf.zeros([1] * len(quantizer.data_shape))
    event_count = tf.constant(event_count_int, tf.int64)
    quantizer.update(sample, weights, event_count).run()


def matmul_config(matmul):
  """Creates an AqtMatmulConfig corresponding to a Matmul."""
  return aqt_config.AqtMatmulConfig(matmul.lhs_quantizer.config,
                                    matmul.rhs_quantizer.config)


def _generate_grad_settings():
  scale = 10.0

  float_config_tc = aqt_config.AqtTensorConfig(
      freeze_scale_at_begin=True,
      quant_config=aqt_config.FloatConfig(),
      calibration_config=aqt_matmul_test_base.calibration_config(1))
  float_config = aqt_config.AqtScheduleConfig(
      aqt_matmul_test_base.test_stats_config(), [float_config_tc])
  int_config = aqt_matmul_test_base._schedule_config(8, scale, (0, 1))

  return (
      # Quantization in the forward direction only.
      # In this case we expect the relative error to be in the range
      # [0.01, 1.00] due to the int8 quantization of the LHS.
      dict(
          testcase_name="no_grad_quantization",
          config=aqt_config.AqtMatmulConfig(
              lhs=int_config, rhs=float_config, grad=None),
          check_individual_entries=True,
          minimum_rel_error=0.01,
          maximum_rel_error=1.0,
          scale=scale,
      ),
      # Grad quantization here should not degrade the error by much.
      # In addition to the quantization of the LHS, we also quantize the
      # gradient. But since the gradient has 16 bits we don't expect it to
      # impact the relative error by much.
      dict(
          testcase_name="16_bit_grad_quantization",
          config=aqt_config.AqtMatmulConfig(
              lhs=int_config,
              rhs=float_config,
              grad=aqt_matmul_test_base._schedule_config(
                  bits=16,
                  const_bound_coeff=100,
                  share_stats_axes=(0, 1),
                  freeze_scale_at_begin=False)),
          check_individual_entries=True,
          minimum_rel_error=0.01,
          maximum_rel_error=1.0,
          scale=scale,
      ),
      # Using low-bit quantization will degrade the error.
      # In addition to the quantization of the LHS, we apply a very coarse 1-bit
      # quantization to the gradient. Here we expect the error to be very bad.
      # In particular, we check that the minimum relative error is at least 1.0,
      # which was the upper bound of the cases above.
      dict(
          testcase_name="1_bit_grad_quantization",
          config=aqt_config.AqtMatmulConfig(
              lhs=int_config,
              rhs=float_config,
              grad=aqt_matmul_test_base._schedule_config(
                  bits=1,
                  const_bound_coeff=100,
                  share_stats_axes=(0, 1),
                  freeze_scale_at_begin=False)),
          check_individual_entries=False,
          minimum_rel_error=1.0,
          maximum_rel_error=None,
          scale=scale,
      ),
      # Quantize only the gradient.
      # We don't quantize the LHS, and the gradient is quantized using 16 bits.
      # So, the error we expect should be quite small. In particular, we can
      # upper bound it by 0.01, which was the lower bound for the
      # no_grad_quantization and 16_bit_grad_quantization cases above.
      dict(
          testcase_name="16_bit_grad_quantization_only",
          config=aqt_config.AqtMatmulConfig(
              lhs=float_config,
              rhs=float_config,
              grad=aqt_matmul_test_base._schedule_config(
                  bits=16,
                  const_bound_coeff=100,
                  share_stats_axes=(0, 1),
                  freeze_scale_at_begin=False)),
          check_individual_entries=True,
          minimum_rel_error=1e-5,
          maximum_rel_error=0.01,
          scale=scale,
      ),)


class IntNarrowedMatMulTest(tf.test.TestCase, parameterized.TestCase):

  def test_chooses_right_matmul(self):
    # Create a list of settings (left_bits, right_bits, expected_matmul)
    # and generate a schedule based on those settings.
    settings = [(8, 16, "default"), (4, 16, "default"), (4, 8, "int8"),
                (8, 9, "default"), (4, 4, "int8"), (3, 3, "int8"),
                (9, 7, "default")]

    lhs_schedule = []
    rhs_schedule = []
    expected_results = []
    for i, (l, r, expected) in enumerate(settings):
      lhs_schedule.append((i, i + 1, l))
      rhs_schedule.append((i, i + 1, r))
      expected_results.append(expected)

    lhs_config = aqt_matmul_test_base.config_from_schedule(lhs_schedule)
    rhs_config = aqt_matmul_test_base.config_from_schedule(rhs_schedule)

    shape = [1, 1]  # Any shape will do, we're mocking.
    lhs_quant = aqt_tensor.TensorQuantizer(shape, lhs_config, name="lhs")
    rhs_quant = aqt_tensor.TensorQuantizer(shape, rhs_config, name="rhs")

    module = "aqt.tensorflow.aqt_matmul"
    with mock.patch(f"{module}.default_matmul") as default_matmul, \
         mock.patch(f"{module}.int8_matmul") as int8_matmul:
      default_matmul.return_value = tf.constant("default")
      int8_matmul.return_value = tf.constant("int8")

      event_ph = tf.placeholder(tf.int64)
      lhs_quant._last_update = event_ph
      rhs_quant._last_update = event_ph
      tf_actual = aqt_matmul._matmul_case(lhs_quant, rhs_quant, None, None,
                                          True)

      with self.cached_session():
        tf.global_variables_initializer().run()
        for i, expected in enumerate(expected_results):
          actual = tf_actual.eval(feed_dict={event_ph: i})
          self.assertEqual(
              actual.decode("utf-8"), expected, msg=f"event_count {i}")


class MatmulTest(aqt_matmul_test_base.MatmulTest):

  def constant(self, x):
    return tf.constant(x, dtype=tf.float32)

  def matmul(self, config, lhs_shape, rhs_shape, name="aqt"):
    return aqt_matmul.Matmul(config, lhs_shape, rhs_shape, name)

  def matmul_apply(self, mm, lhs, rhs, train=True, keep_stats=False):
    event_count = tf.constant(0, tf.int64)
    lhs_sample = tf.zeros_like(lhs) if keep_stats else lhs
    lhs_weight = tf.ones_like(lhs) if keep_stats else None
    rhs_sample = tf.zeros_like(rhs) if keep_stats else rhs
    rhs_weight = tf.ones_like(rhs) if keep_stats else None
    updates = [
        mm.update_lhs(lhs_sample, lhs_weight, event_count),
        mm.update_rhs(rhs_sample, rhs_weight, event_count)
    ]
    with tf.control_dependencies(updates):
      result = mm.apply(lhs, rhs, train=train)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      return result.eval()

  def matmul_unquantized(self, lhs, rhs):
    result = tf.matmul(lhs, rhs)
    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      return result.eval()

  def gradients(self, fwd_func, x, w, use_reduce=False):
    if use_reduce:
      fwd_func = lambda x, w: tf.reduce_sum(fwd_func(x, w)**2)
    fwd = fwd_func(x, w)
    return tf.gradients([fwd], [x, w])

  def with_config(self, mm, config):
    """Returns new Matmul with the new config but otherwise the same."""
    with tf.variable_scope(None, default_name="uniqued"):
      return aqt_matmul.Matmul(config, mm.lhs_quantizer.data_shape,
                               mm.rhs_quantizer.data_shape, mm.name,
                               mm.lhs_name, mm.rhs_name)

  def test_validates_contraction(self):
    mm, _, _ = self.exact_int8_matmul_example()

    config = copy.deepcopy(matmul_config(mm))
    config.rhs.stats_config.share_stats_axes = [1]
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected rhs matmul contraction axis"):
      self.with_config(mm, config)

    config = copy.deepcopy(matmul_config(mm))
    config.lhs.stats_config.share_stats_axes = [0]
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected lhs matmul contraction axis"):
      self.with_config(mm, config)

  def test_validates_rank2(self):
    mm, lhs, rhs = self.exact_int8_matmul_example()

    mm.rhs_quantizer.data_shape.append(1)
    with self.assertRaisesRegex(aqt_config.ConfigError, "rhs data shape"):
      mm.apply(lhs, rhs)
    mm.rhs_quantizer.data_shape = mm.rhs_quantizer.data_shape[:-1]

    mm.lhs_quantizer.data_shape += (1,)
    with self.assertRaisesRegex(aqt_config.ConfigError, "lhs data shape"):
      mm.apply(lhs, rhs)

  @parameterized.named_parameters(*_generate_grad_settings())
  def test_grad_linearity(
      self,
      config,
      check_individual_entries,
      minimum_rel_error,
      maximum_rel_error,
      scale,
  ):
    """Validates gradients are correct on basic example."""
    contract_dim = 10
    lhs_shape = (1, contract_dim)
    rhs_shape = (contract_dim, 1)
    target_shape = lhs_shape[:1] + rhs_shape[1:]

    lhs_ph = tf.placeholder(tf.float32, shape=lhs_shape)
    rhs_ph = tf.placeholder(tf.float32, shape=rhs_shape)
    target_ph = tf.placeholder(tf.float32, shape=target_shape)

    mm = aqt_matmul.Matmul(config, lhs_shape, rhs_shape)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      if config.grad:
        with self.subTest("Gradient shape"):
          self.assertSequenceEqual(mm.grad_quantizer.data_shape, target_shape)

      event_count = tf.constant(0, tf.int64)
      updates = [
          mm.update_lhs(tf.ones(lhs_shape), None, event_count),
          mm.update_rhs(tf.ones(rhs_shape), None, event_count),
      ]
      with tf.control_dependencies(updates):
        aqt_mm = mm.apply(lhs_ph, rhs_ph)

      aqt_diff = aqt_mm - target_ph
      aqt_loss = tf.reduce_sum(aqt_diff**2) / 2
      aqt_mm_grad = tf.gradients([aqt_loss], [rhs_ph])[0]

      rng = np.random.default_rng(1234)
      for i in range(10):
        lhs = rng.standard_normal(lhs_shape).astype(np.float32)
        rhs = rng.standard_normal(rhs_shape).astype(np.float32)
        target = rng.standard_normal(target_shape).astype(np.float32)

        feed_dict = {lhs_ph: lhs, rhs_ph: rhs, target_ph: target}

        aqtd, aqt_grad = sess.run([aqt_diff, aqt_mm_grad], feed_dict=feed_dict)

        # Notice aqt gradient at position i is quantized(lhs)[i] * aqtd
        # assuming linearity of gradients.
        grad_factor = aqtd.ravel()
        float_grad = lhs.ravel() * grad_factor
        true_grad = aqt_grad.ravel()

        diff = np.abs(float_grad - true_grad)
        bucket_width = scale * 2 / 255
        for j, err in enumerate(diff):
          if check_individual_entries:
            with self.subTest("Individual entry errors"):
              self.assertLessEqual(
                  err,
                  bucket_width * abs(grad_factor),
                  msg=f"trial {i} position {j}")
        if minimum_rel_error:
          with self.subTest("Minimum overall error"):
            self.assertGreaterEqual(
                np.abs(np.sum(diff) / np.sum(float_grad)), minimum_rel_error)
        if maximum_rel_error:
          with self.subTest("Maximum overall error"):
            self.assertLessEqual(
                np.abs(np.sum(diff) / np.sum(float_grad)), maximum_rel_error)

  @parameterized.named_parameters(
      dict(testcase_name="no_grad", use_grad_quantization=False),
      dict(testcase_name="with_grad", use_grad_quantization=True))
  def test_diagnostics(self, use_grad_quantization):
    mm, lhs, rhs = self.exact_int8_matmul_example(
        use_grad_quantization=use_grad_quantization)

    with self.cached_session():
      tf.global_variables_initializer().run()
      update_event_count(mm, 0)

      # We just want to check if diagnostics works, so use a fake gradient.
      if use_grad_quantization:
        fake_grad = tf.zeros(shape=(lhs.shape[0], rhs.shape[1]))
      else:
        fake_grad = None
      d = mm.diagnostics(lhs, rhs, fake_grad)
      quantizers = {"lhs": mm.lhs_quantizer, "rhs": mm.rhs_quantizer}
      if use_grad_quantization:
        quantizers["grad"] = mm.grad_quantizer

      for qname, quantizer in quantizers.items():
        with self.subTest(f"qname {qname}."):
          for name, expected in quantizer.calibration_variables().items():
            actual = d[f"aqt/{qname}/{name}"]
            self.assertAllEqual(actual, expected)

          actual = d[f"aqt/{qname}/clipped_proportion"]
          expected = 0.0
          self.assertAllEqual(actual, expected)

          actual = d[f"aqt/{qname}/clip"]
          expected = quantizer.clip_range()
          self.assertAllEqual(actual, expected)

      with self.subTest("Out of range"):
        out_of_range_lhs, out_of_range_rhs = (
            tf.ones_like(x) * 512.0 for x in (lhs, rhs))
        d = mm.diagnostics(out_of_range_lhs, out_of_range_rhs)
        for arg in ["lhs", "rhs"]:
          actual = d[f"aqt/{arg}/clipped_proportion"]
          expected = 1.0
          self.assertAllEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
