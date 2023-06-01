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

from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.test import aqt_test_shared_base
import numpy as np
import tensorflow.compat.v1 as tf


def test_stats_config():
  """Config base for all the experiments."""
  return aqt_config.StatsConfig(
      ema_update_count=1,  # single update overwrites the stats
      update_count_prior=0,  # easier equations
      share_stats_axes=[0, 1],  # one stat per whole tensor
      tpu_cross_replica_sum=False,  # no cross-tpu reduction
      filter_zeros=False,  # on default zeros are a number like any other
  )


def calibration_config(const_coeff: float) -> aqt_config.CalibrationConfig:
  return aqt_config.CalibrationConfig(const_bound_coeff=const_coeff)


def _schedule_config(
    bits,
    const_bound_coeff,
    share_stats_axes,
    freeze_scale_at_begin=True,
) -> aqt_config.AqtScheduleConfig:
  """Creates schedule config with dynamic quantization."""
  iqc = aqt_config.IntQuantConfig(bits=bits)
  cc = aqt_config.CalibrationConfig(const_bound_coeff=const_bound_coeff)
  tc = aqt_config.AqtTensorConfig(
      quant_config=iqc,
      calibration_config=cc,
      freeze_scale_at_begin=freeze_scale_at_begin)
  sc = aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=list(share_stats_axes),
      update_count_prior=0,
      tpu_cross_replica_sum=False)
  return aqt_config.AqtScheduleConfig(sc, [tc])


def config_from_schedule(schedule):
  """Generates a schedule config from [(start, end, bits), ...]."""
  tensor_configs = []
  for start, end, bits in schedule:
    int_quant_config = aqt_config.IntQuantConfig(bits, preserve_zero=False)

    tensor_config = aqt_config.AqtTensorConfig(
        freeze_scale_at_begin=True,
        quant_config=int_quant_config,
        begin_at_event=start,
        end_at_event=end,
        calibration_config=calibration_config(1.0))

    tensor_configs.append(tensor_config)
  return aqt_config.AqtScheduleConfig(test_stats_config(), tensor_configs)


class MatmulTest(tf.test.TestCase, parameterized.TestCase):
  """Base class for testing aqt_matmul in TF and Jax."""

  def constant(self, x):
    raise NotImplementedError

  def matmul(self, config, lhs_shape, rhs_shape, name="aqt"):
    raise NotImplementedError

  def matmul_apply(self, module, lhs, rhs, train=True, keep_stats=False):
    raise NotImplementedError

  def matmul_unquantized(self, lhs, rhs):
    raise NotImplementedError

  def gradients(self, fwd_func, x, w, reduce_sum=False):
    raise NotImplementedError

  def exact_int8_matmul_example(
      self,
      lhs_use_quantized_variable=False,
      rhs_use_quantized_variable=False,
      name="aqt",
      use_float_config=False,
      use_grad_quantization=False,
  ):
    """Creates lhs/rhs tensors, their corresponding AQT configs, and matmul op."""
    lhs_config, lhs, rhs_config, rhs = aqt_test_shared_base.exact_int8_example(
        lhs_shape=(3, 2),
        rhs_shape=(2, 2),
        lhs_share_stats_axes=[0, 1],
        rhs_share_stats_axes=[0, 1],
        lhs_use_quantized_variable=lhs_use_quantized_variable,
        rhs_use_quantized_variable=rhs_use_quantized_variable)

    if use_float_config:
      lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    if use_grad_quantization:
      grad_config = copy.deepcopy(lhs_config)
      grad_config.tensor_configs[0].freeze_scale_at_begin = False
      grad_config.stats_config.ema_update_count = 1
    else:
      grad_config = None

    lhs = self.constant(lhs)
    rhs = self.constant(rhs)

    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config, grad_config)
    mm = self.matmul(config, lhs.shape, rhs.shape, name)

    return mm, lhs, rhs

  def test_matmul_none(self):
    """Ensures no quantization and float config give aqt_matmul = {tf, jnp}.matmul."""
    no_quant_config = aqt_config.AqtScheduleConfig(test_stats_config(), [])
    float_config_tc = aqt_config.AqtTensorConfig(
        freeze_scale_at_begin=True,
        quant_config=aqt_config.FloatConfig(),
        calibration_config=calibration_config(1))
    float_config = aqt_config.AqtScheduleConfig(test_stats_config(),
                                                [float_config_tc])

    lhs = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    lhs = self.constant(lhs)

    rhs = np.random.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float32)
    rhs = self.constant(rhs)

    config = aqt_config.AqtMatmulConfig(no_quant_config, no_quant_config)
    mm_no_quant = self.matmul(config, lhs.shape, rhs.shape, "no_quant")
    config = aqt_config.AqtMatmulConfig(float_config, float_config)
    mm_float = self.matmul(config, lhs.shape, rhs.shape, "float")

    no_quant_ret = self.matmul_apply(mm_no_quant, lhs, rhs)
    float_config_ret = self.matmul_apply(mm_float, lhs, rhs)
    expected_ret = self.matmul_unquantized(lhs, rhs)

    self.assertAllEqual(no_quant_ret, expected_ret)
    self.assertAllEqual(float_config_ret, expected_ret)

  def basic_quant_example(self):
    """Returns hand-computed examples."""
    lhs_config = _schedule_config(3, 7, [0, 1])
    lhs = self.constant(
        np.array(
            [
                [-8, 4.01, 4.01],  #
                [-5.99, 0.01, -4.01],
            ],))
    qlhs = self.constant(
        np.array(
            [
                [-6, 4, 4],  #
                [-6, 0, -4]
            ],))

    # Representable values: -1, 0, 1
    rhs_config = _schedule_config(2, 1.5, [0, 1])
    rhs = self.constant(
        np.array(
            [
                [-3, 0.99],  #
                [-0.99, 0],
                [-0.01, 2]
            ],))
    qrhs = self.constant(
        np.array(
            [
                [-1, 1],  #
                [-1, 0],
                [0, 1]
            ],))

    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config)

    return config, lhs, qlhs, rhs, qrhs

  def test_basic_matmul(self):
    """Test on a hand-computed example."""
    config, lhs, qlhs, rhs, qrhs = self.basic_quant_example()

    mm = self.matmul(config, lhs.shape, rhs.shape)

    actual = self.matmul_apply(mm, lhs, rhs)
    expected = self.matmul_unquantized(qlhs, qrhs)

    self.assertAllEqual(expected, actual)

  @parameterized.parameters([dict(lhs_float=True), dict(lhs_float=False)])
  def test_float_config_basic_matmul(self, lhs_float):
    """Float config test on hand-computed exmaple."""
    config, lhs, qlhs, rhs, qrhs = self.basic_quant_example()

    if lhs_float:
      config.lhs.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    else:
      config.rhs.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    mm = self.matmul(config, lhs.shape, rhs.shape)

    actual = self.matmul_apply(mm, lhs, rhs)
    if lhs_float:
      qlhs = lhs  # lhs is not quantized
    else:
      qrhs = rhs  # rhs is not quantized
    expected = self.matmul_unquantized(qlhs, qrhs)

    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(
      aqt_test_shared_base.generate_unaligned_schedule_intervals())
  def test_unaligned_schedule_intervals(self, lhs_intervals, rhs_intervals):
    """Raises error if lhs and rhs config schedules are not aligned."""
    bits = 8
    lhs_intervals = [(start, stop, bits) for start, stop in lhs_intervals]
    rhs_intervals = [(start, stop, bits) for start, stop in rhs_intervals]
    lhs_config = config_from_schedule(lhs_intervals)
    rhs_config = config_from_schedule(rhs_intervals)
    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config)

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "intervals do not match|config len"):
      self.matmul(config, (1, 1), (1, 1))

  def test_vars_dont_kill_grads(self):
    """Ensures quantized variables do not change gradients."""
    mm, lhs, rhs = self.exact_int8_matmul_example()
    unsaved_matmul = lambda lhs, rhs: self.matmul_apply(mm, lhs, rhs)

    mm_q, _, _ = self.exact_int8_matmul_example(True, True, "with_var")
    saved_matmul = lambda lhs, rhs: self.matmul_apply(mm_q, lhs, rhs)

    saved_grads = self.gradients(saved_matmul, lhs, rhs)
    unsaved_grads = self.gradients(unsaved_matmul, lhs, rhs)

    zipped_grads = zip(saved_grads, unsaved_grads)
    for actual_grad, expected_grad in zipped_grads:
      actual = actual_grad
      expected = expected_grad

      self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(lhs_use_quantized_variable=True),
      dict(lhs_use_quantized_variable=False)
  ])
  def test_vars_over_inputs_at_inference(self, lhs_use_quantized_variable):
    """Ensures quantized variables are used if TQ.use_quantized_variable=True at inference."""
    rhs_use_quantized_variable = not lhs_use_quantized_variable
    mm, tf_lhs, tf_rhs = self.exact_int8_matmul_example(
        lhs_use_quantized_variable, rhs_use_quantized_variable)
    mm_train, _, _ = self.exact_int8_matmul_example(
        lhs_use_quantized_variable=False,
        rhs_use_quantized_variable=False,
        name="no_quantize")

    actual = self.matmul_apply(
        mm, tf_lhs, tf_rhs, train=False, keep_stats=True)
    expected = np.zeros_like(actual)
    # Rely on zero initialization for variables as opposed to non-zero inputs.
    self.assertAllEqual(actual, expected)

    # But if train then use input instead.
    actual = self.matmul_apply(
        mm, tf_lhs, tf_rhs, train=True, keep_stats=True)

    expected = self.matmul_apply(
        mm_train, tf_lhs, tf_rhs, keep_stats=True)
    self.assertAllEqual(actual, expected)

  def test_float_config_not_save_quantized_var(self):
    mm, lhs, rhs = self.exact_int8_matmul_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True,
        use_float_config=True)

    actual = self.matmul_apply(mm, lhs, rhs, train=False)
    expected = np.zeros_like(actual)
    # Although lhs config sets use_quantized_variable to True, lhs has
    # a float config, and thus it uses zero-initialized quantized var.
    self.assertAllEqual(actual, expected)

  def test_exact_grads(self):
    """Ensures both quantized and unquantized conv op emit the same gradients."""
    mm, lhs, rhs = self.exact_int8_matmul_example()

    aqt_mm = lambda lhs, rhs: self.matmul_apply(mm, lhs, rhs)
    aqt_mm_grads = self.gradients(aqt_mm, lhs, rhs)

    mm_exact = self.matmul_unquantized
    mm_grads = self.gradients(mm_exact, lhs, rhs)

    for aqt_grad, tf_grad in zip(aqt_mm_grads, mm_grads):
      actual = aqt_grad
      expected = tf_grad

      self.assertAllEqual(actual, expected)
