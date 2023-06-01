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

"""Tests for einsum."""

import typing
from typing import Any, Dict, Optional, Sequence, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.common import aqt_common
from aqt.common import aqt_config
from aqt.tensorflow import aqt_einsum
from aqt.tensorflow import aqt_ops
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_test_shared_base
import numpy as np
import tensorflow.compat.v1 as tf


def _stats_config(share_stats_axes: Sequence[int]) -> aqt_config.StatsConfig:
  """Generates dynamic quantization stats configuration."""
  return aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=list(share_stats_axes),
      update_count_prior=0,
      tpu_cross_replica_sum=False)


def _schedule_config(
    bits: int, const_bound_coeff: float,
    share_stats_axes: Sequence[int]) -> aqt_config.AqtScheduleConfig:
  """Creates schedule config with dynamic quantization."""
  iqc = aqt_config.IntQuantConfig(bits=bits)
  cc = aqt_config.CalibrationConfig(const_bound_coeff=const_bound_coeff)
  tc = aqt_config.AqtTensorConfig(
      quant_config=iqc, calibration_config=cc, freeze_scale_at_begin=True)
  sc = _stats_config(share_stats_axes)
  return aqt_config.AqtScheduleConfig(sc, [tc])


def _schedule_config_emulation(
    share_stats_axes) -> aqt_config.AqtScheduleConfig:
  """Creates schedule config for emulated precision."""
  iqc = aqt_config.SmallFloatConfig(
      exponent_bits=5,
      mantissa_bits=2,
      min_exp=-14,
      max_exp=15,
      support_inf=False,
      rounding_mode=aqt_config.RoundingMode.ROUND_TO_NEAREST_EVEN)
  # Using the max number essentially disables scaling.
  cc = aqt_config.CalibrationConfig(
      const_bound_coeff=aqt_common._get_max_number_float(
          mantissa_bits=2, max_exp=15))
  tc = aqt_config.AqtTensorConfig(
      quant_config=iqc, calibration_config=cc, freeze_scale_at_begin=True)
  sc = aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=list(share_stats_axes),
      update_count_prior=0,
      tpu_cross_replica_sum=False)
  return aqt_config.AqtScheduleConfig(sc, [tc])


def _empty_config(
    share_stats_axes: Sequence[int]) -> aqt_config.AqtScheduleConfig:
  return aqt_config.AqtScheduleConfig(_stats_config(share_stats_axes), [])


def _einsum_op(
    eq: str,  #
    lhs: tf.Tensor,
    rhs: tf.Tensor,
    lhs_config: aqt_config.AqtScheduleConfig,
    rhs_config: aqt_config.AqtScheduleConfig,
    lhs_weights: Optional[tf.Tensor] = None,
    rhs_weights: Optional[tf.Tensor] = None,
    varscope_name: str = "einsum",
    train: bool = True,
    **einsum_kwargs) -> tf.Tensor:
  """Updates quantizers at event_count=0 and computes einsum."""
  aqtc = aqt_config.AqtEinsumConfig(
      lhs=lhs_config, rhs=rhs_config)
  einsum = aqt_einsum.Einsum(eq, aqtc, lhs.shape, rhs.shape, name=varscope_name)

  event_count = tf.constant(0, tf.int64)
  updates = [
      einsum.update_lhs(lhs, lhs_weights, event_count),
      einsum.update_rhs(rhs, rhs_weights, event_count)
  ]
  with tf.control_dependencies(updates):
    return einsum.apply(lhs, rhs, train, **einsum_kwargs)


def _generate_missing_shared_axes() -> Sequence[Dict[str, Any]]:
  """Cases where shared axes are missing."""

  keys = ["testcase_name", "eq", "lhs_share", "rhs_share"]
  cases: Sequence[Tuple[str, str, Sequence[int], Sequence[int]]] = [
      ("no_sharing_valid", "i,->i", [], []),
      ("contracting", "i,->", [], []),
      ("contracting_partial_rhs", "i,j->", [], [0]),
      ("contracting_partial_lhs", "i,j->", [0], []),
      ("contracting_both_valid", "i,j->", [0], [0]),
      ("contracting_one_valid", ",i->", [], [0]),
      ("repeated_contracting", "j,j->", [], []),
      ("repeated_contracting_partial_rhs", "j,j->", [], [0]),
      ("repeated_contracting_partial_lhs", "j,j->", [0], []),
      ("repeated_contracting_valid", "j,j->", [0], [0]),
  ]

  cases_dicts = []
  for vals in cases:
    case = dict(zip(keys, vals))
    case["is_valid"] = typing.cast(str, case["testcase_name"]).endswith("valid")
    cases_dicts.append(case)

  return cases_dicts


def _generate_bad_equation() -> Sequence[Dict[str, Any]]:
  """Cases where shared axes are missing."""

  keys = ["testcase_name", "eq", "lhs_rank", "rhs_rank"]
  cases = [
      ("single_arg", "ii->", 2, 0),
      ("single_arg_no_out", "ii", 2, 0),
      ("double_arg_no_out", "ii,ij", 2, 2),
      ("bad_out", "i,i>i", 2, 2),
      ("bad_out_dash", "i,i-i", 2, 2),
      ("nonstandard_axes", "i!j,ij->i", 3, 2),
      ("space", "i j,ij->i", 3, 2),
      ("newline", "ij,i\nj->i", 2, 3),
      ("ellipses", "...ij,jk->ik", 4, 2),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


def _generate_test_equations() -> Sequence[Dict[str, str]]:
  keys = ["testcase_name", "eq"]
  cases = [
      ("diag", "ii,->i"),
      ("sum", "i,->"),
      ("trace", "ii,->"),
      ("transpose", ",ij->ji"),
      ("matmul", "ij,jk->ik"),
      ("batch_matmul", "bij,bjk->bik"),
      ("dot", "i,i->"),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


class EinsumTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Seed random for deterministic but nontrivial inputs."""
    super().setUp()
    self.rng = np.random.default_rng(1234)

  def randn(self, *size):
    return self.rng.standard_normal(size=size, dtype=np.float32)

  @parameterized.named_parameters(_generate_missing_shared_axes())
  def test_missing_shared_axes(
      self,  #
      eq: str,
      lhs_share: Sequence[int],
      rhs_share: Sequence[int],
      is_valid: bool):

    def make_tensor(einsum_str):
      return tf.constant(np.ones([1] * len(einsum_str)), tf.float32)

    def make_op():
      lhs, rhs, _ = aqt_einsum._parse_equation(eq)
      lhs, rhs = make_tensor(lhs), make_tensor(rhs)
      return _einsum_op(eq, lhs, rhs, _empty_config(lhs_share),
                        _empty_config(rhs_share))

    if is_valid:
      make_op()
    else:
      with self.assertRaisesRegex(aqt_config.ConfigError,
                                  "axis .* of .* must be shared due to .*"):
        make_op()

  @parameterized.named_parameters(_generate_bad_equation())
  def test_bad_equation(self, eq: str, lhs_rank: int, rhs_rank: int):
    with self.assertRaisesRegex(aqt_config.ConfigError, "einsum equation"):

      def make_tensor(rank):
        return tf.constant(np.ones([1] * rank), tf.float32)

      lhs = make_tensor(lhs_rank)
      rhs = make_tensor(rhs_rank)
      lhs_config = _empty_config(list(range(lhs_rank)))
      rhs_config = _empty_config(list(range(rhs_rank)))
      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)

  def basic_quant_example(self):
    eq = "iji,jk->i"

    # Representable values: -6, -4, -2, 0, 2, 4, 6.
    lhs_config = _schedule_config(3, 7, [0, 1, 2])
    lhs = tf.constant(
        np.array(
            [
                [
                    [-8, -5.99],  #
                    [4.01, 0.01],
                    [4.01, -4.01]
                ],
                [
                    [-0.01, 2.01],  #
                    [4.01, 6.01],
                    [3.99, -3.99]
                ]
            ],
            dtype=np.float32))
    qlhs = tf.constant(
        np.array(
            [
                [
                    [-6, -6],  #
                    [4, 0],
                    [4, -4]
                ],
                [
                    [0, 2],  #
                    [4, 6],
                    [4, -4]
                ]
            ],
            dtype=np.float32))

    # Representable values: -1, 0, 1
    rhs_config = _schedule_config(2, 1.5, [0, 1])
    rhs = tf.constant(
        np.array(
            [
                [-3, 0.99],  #
                [-0.99, 0],
                [-0.01, 2]
            ],
            dtype=np.float32))
    qrhs = tf.constant(
        np.array(
            [
                [-1, 1],  #
                [-1, 0],
                [0, 1]
            ],
            dtype=np.float32))

    return eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs

  def basic_emulation_example(self):
    eq = "ij,jk->ik"
    lhs_config = _schedule_config_emulation([0, 1])
    lhs = tf.constant(
        np.array(
            [
                [-8.5, 4.3, 4.1],  #
                [-0.05, 0.01, -4.7],
            ],
            dtype=np.float32))
    qlhs = tf.constant(
        np.array(
            [
                [-8.0, 4.0, 4.0],  #
                [-0.046875, 0.00976562, -5.0]
            ],
            dtype=np.float32))

    rhs_config = _schedule_config_emulation([0, 1])
    rhs = tf.constant(
        np.array(
            [
                [-0.2, 0.02],  #
                [-1.1, 0],
                [-0.04, 2.3]
            ],
            dtype=np.float32))
    qrhs = tf.constant(
        np.array(
            [
                [-0.1875, 0.01953125],  #
                [-1.0, 0.0],
                [-0.0390625, 2.5]
            ],
            dtype=np.float32))

    return eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs

  def test_basic_einsum(self):
    with self.subTest("quant_example"):
      eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = (
          self.basic_quant_example())

    with self.subTest("emulation_example"):
      eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = (
          self.basic_emulation_example())

    actual = _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)
    expected = tf.einsum(eq, qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  def test_no_quantization(self):
    lhs = tf.constant(self.randn(3, 4))
    rhs = tf.constant(self.randn(4, 2))
    eq = "ij,jk->ik"
    lhs_config = _empty_config([1])
    rhs_config = _empty_config([0])

    lhs_float_config = _schedule_config(8, 1.0, [1])
    rhs_float_config = _schedule_config(8, 1.0, [0])

    lhs_float_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    rhs_float_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    tf_actual = _einsum_op(
        eq, lhs, rhs, lhs_config, rhs_config, varscope_name="no_quant_einsum")
    tf_float_config_actual = _einsum_op(eq, lhs, rhs, lhs_float_config,
                                        rhs_float_config,
                                        varscope_name="float_config_einsum")
    tf_expected = tf.einsum(eq, lhs, rhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(tf_expected, tf_actual)
      self.assertAllEqual(tf_expected, tf_float_config_actual)

  @parameterized.parameters([dict(lhs_float=True), dict(lhs_float=False)])
  def test_float_config_basic_einsum(self, lhs_float):
    eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = self.basic_quant_example(
    )
    if lhs_float:
      lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    else:
      rhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    actual = _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)
    if lhs_float:
      qlhs = lhs  # lhs is not quantized
    else:
      qrhs = rhs  # rhs is not quantized
    expected = tf.einsum(eq, qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  def test_passes_arguments_to_inner_einsum(self):
    module = "tensorflow.compat.v1"
    with mock.patch(f"{module}.einsum", side_effect=tf.einsum) as tfeinsum:
      lhs = tf.constant(self.randn(3, 4))
      rhs = tf.constant(self.randn(4, 2))
      eq = "ij,jk->ik"
      lhs_config = _schedule_config(8, 1.0, [1])
      rhs_config = _schedule_config(8, 1.0, [0])

      kwargs = {"optimize": "optimal", "name": "optimal_einsum"}

      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config, **kwargs)
      for (_, actual_kwargs) in tfeinsum.call_args_list:
        subset = {k: v for k, v in actual_kwargs.items() if k in kwargs}
        self.assertEqual(subset, kwargs)

  @parameterized.named_parameters(
      aqt_test_shared_base.generate_unaligned_schedule_intervals())
  def test_unaligned_schedule_intervals(self, lhs_intervals, rhs_intervals):

    def config_from_schedule(intervals):
      config = _empty_config([0, 1])
      for start, stop in intervals:
        config.tensor_configs += _schedule_config(8, 1.0, []).tensor_configs
        config.tensor_configs[-1].begin_at_event = start
        config.tensor_configs[-1].end_at_event = stop
      return config

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "intervals do not match|config len"):
      lhs = tf.constant(self.randn(3, 4))
      rhs = tf.constant(self.randn(4, 2))
      eq = "ij,jk->ik"
      lhs_config = config_from_schedule(lhs_intervals)
      rhs_config = config_from_schedule(rhs_intervals)
      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)

  def exact_int8_einsum_example(self,
                                eq,
                                quantize_lhs=False,
                                quantize_rhs=False):
    """Returns a pair of tensors and config to einsum exactly."""
    lhs, rhs, _ = aqt_einsum._parse_equation(eq)

    # A subset of the range of numbers which can be preserved exactly.
    bits = 8
    symmetric_uniform_range = 2**(bits - 1) - 1
    lo, hi = -symmetric_uniform_range, symmetric_uniform_range

    axis_labels = sorted(set(lhs + rhs))
    label_dims = {k: self.rng.integers(2, 10) for k in axis_labels}
    lhs_shape = [label_dims[k] for k in lhs]
    rhs_shape = [label_dims[k] for k in rhs]

    def make_tensor(shape):
      np_tensor = self.rng.integers(lo, hi, size=shape, dtype=np.int64)
      return tf.constant(np_tensor, dtype=tf.float32)

    lhs = make_tensor(lhs_shape)
    rhs = make_tensor(rhs_shape)

    iqc = aqt_config.IntQuantConfig(bits=8, preserve_zero=True)
    clip_bound = aqt_common.get_clip_bound(iqc)
    assert symmetric_uniform_range <= clip_bound

    lhs_config = _schedule_config(8, clip_bound, list(range(len(lhs_shape))))
    rhs_config = _schedule_config(8, clip_bound, list(range(len(rhs_shape))))

    lhs_config.use_quantized_variable = quantize_lhs
    rhs_config.use_quantized_variable = quantize_rhs

    return lhs_config, lhs, rhs_config, rhs

  @parameterized.named_parameters(_generate_test_equations())
  def test_vars_dont_kill_grads(self, eq):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_einsum_example(
        eq, True, True)
    lhs_config_novar, _, rhs_config_novar, _ = self.exact_int8_einsum_example(
        eq, True, True)

    expected_op = _einsum_op(
        eq, lhs, rhs, lhs_config_novar, rhs_config_novar, varscope_name="novar")
    actual_op = _einsum_op(
        eq, lhs, rhs, lhs_config, rhs_config, varscope_name="var")

    saved_grads = tf.gradients([expected_op], [lhs, rhs])
    unsaved_grads = tf.gradients([actual_op], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()

      zipped_grads = zip(saved_grads, unsaved_grads)
      for actual_grad, expected_grad in zipped_grads:
        self.assertAllEqual(actual_grad, expected_grad)

  @parameterized.named_parameters(_generate_test_equations())
  def test_vars_over_inputs_at_inference(self, eq):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_einsum_example(
        eq, True, True)

    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, rhs_config, name="rhs")

    # Update at least once to initialize scale, then grab the expected
    # value while in training mode.
    event_count = tf.constant(0, tf.int64)
    updates = [
        lhs_tq.update(lhs, weight=None, event_count=event_count),
        rhs_tq.update(rhs, weight=None, event_count=event_count)
    ]
    with tf.control_dependencies(updates):
      expected = aqt_ops.aqt_einsum(eq, lhs_tq, lhs, rhs_tq, rhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      expected = expected.eval()

      actual = aqt_ops.aqt_einsum(eq, lhs_tq, tf.zeros_like(lhs), rhs_tq,
                                  tf.zeros_like(rhs), train=False)

      self.assertAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_float_config_not_save_quantized_var(self, eq):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_einsum_example(
        eq, True, True)

    lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, rhs_config, name="rhs")

    event_count = tf.constant(0, tf.int64)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      lhs_tq.update(lhs, weight=None, event_count=event_count).run()
      rhs_tq.update(rhs, weight=None, event_count=event_count).run()

      actual = aqt_ops.aqt_einsum(eq, lhs_tq, lhs, rhs_tq, rhs, train=False)
      # Although the input tensors are non-zeros, the result of einsum with
      # inference mode should be zeros because lhs uses zero-initialized
      # quantized var while rhs can restore its updated quantized variable.
      expected = tf.zeros_like(actual)

      self.assertAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_exact(self, eq):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_einsum_example(eq)

    actual = _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)
    expected = tf.einsum(eq, lhs, rhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_exact_grads(self, eq):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_einsum_example(eq)

    actual_fwd = _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)
    expected_fwd = tf.einsum(eq, lhs, rhs)

    expected = tf.gradients([expected_fwd], [lhs, rhs])
    actual = tf.gradients([actual_fwd], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      for actual_grad, expected_grad in zip(actual, expected):
        self.assertAllEqual(actual_grad, expected_grad)


if __name__ == "__main__":
  absltest.main()
