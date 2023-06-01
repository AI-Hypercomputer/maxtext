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

"""Tests for conv2d."""

import itertools

from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.test import aqt_test_shared_base
import numpy as np
import tensorflow.compat.v1 as tf


def input_stats(filter_zeros=True):
  """Generates stats configuration for default NHWC inputs."""
  return aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=[1, 2, 3],
      update_count_prior=0,
      tpu_cross_replica_sum=False,
      filter_zeros=filter_zeros)


def filter_stats(filter_zeros=True):
  """Generates stats configuration for default HWIO filter."""
  sc = input_stats(filter_zeros)
  sc.share_stats_axes = [0, 1, 2]
  return sc


def int_quant_config(bits, preserve_zero=True):
  """Returns a zero-preserving int config with the given bits."""
  return aqt_config.IntQuantConfig(bits=bits, preserve_zero=preserve_zero)


def calibration_config(const_coeff, max_dev_coeff):
  """Returns a calibration config with the given constant coeff."""
  return aqt_config.CalibrationConfig(
      const_bound_coeff=const_coeff, max_dev_coeff=max_dev_coeff)


def stats_config(input_or_filter, filter_zeros=True):
  assert input_or_filter in ["input", "filter"]
  stats_config_fn = input_stats if input_or_filter == "input" else filter_stats
  return stats_config_fn(filter_zeros)


def schedule_config(input_or_filter,
                    *,
                    const_coeff,
                    bits,
                    max_dev_coeff=0,
                    filter_zeros=True,
                    preserve_zero=True):
  """Combines all helpers above to set schedule config."""
  tensor_config = aqt_config.AqtTensorConfig(
      quant_config=int_quant_config(bits, preserve_zero),
      calibration_config=calibration_config(const_coeff, max_dev_coeff),
      freeze_scale_at_begin=True)
  sc = stats_config(input_or_filter, filter_zeros)
  return aqt_config.AqtScheduleConfig(sc, [tensor_config])


def empty_config(input_or_filter):
  return aqt_config.AqtScheduleConfig(stats_config(input_or_filter), [])


def generate_missing_contraction_dims():
  """Cases where some spatial dims for conv are missing."""

  cases = []

  data_format = ["NHWC"]
  correct_filter_cases = [("correct", [0, 1, 2])]
  correct_input_cases = [("correct", [1, 2, 3])]
  incorrect_filter_cases = [("partial", [1]), ("wrong", [1, 3]),
                            ("partial2", [1, 2]), ("partial3", [0, 2]),
                            ("missing", [])]
  incorrect_input_cases = [("partial", [2]), ("wrong", [0, 1]),
                           ("partial2", [1, 2]), ("missing", [])]

  def add_cases(input_cases, filter_cases):
    case_tuples = itertools.product(data_format, input_cases, filter_cases)
    cases.extend(case_tuples)

  add_cases(correct_input_cases, incorrect_filter_cases)
  add_cases(incorrect_input_cases, correct_filter_cases)
  add_cases(incorrect_input_cases, incorrect_filter_cases)

  data_format = ["NCHW"]
  correct_input_cases = [("correct", [1, 2, 3])]
  incorrect_input_cases = [("partial", [2]), ("wrong", [0, 1]),
                           ("partial2", [1, 2]), ("missing", [])]

  add_cases(correct_input_cases, incorrect_filter_cases)
  add_cases(incorrect_input_cases, correct_filter_cases)
  add_cases(incorrect_input_cases, incorrect_filter_cases)

  case_dicts = []
  for data_format, (i_name, input_axes), (f_name, filter_axes) in cases:
    case_dicts.append({
        "data_format": data_format,
        "input_axes": input_axes,
        "filter_axes": filter_axes,
        "testcase_name": f"{data_format}_input_{i_name}_filter_{f_name}"
    })

  return case_dicts


class ConvTest(tf.test.TestCase, parameterized.TestCase):
  """Base class for testing aqt_conv2d in TF and aqt_conv_general in Jax."""

  def setUp(self):
    """Seed random for deterministic but nontrivial inputs."""
    super().setUp()
    self.rng = np.random.default_rng(1234)

  def create_random_input_and_filter(self,
                                     input_shape=(1, 3, 3, 1),
                                     filter_shape=(2, 2, 1, 2)):
    x = self.rng.standard_normal(size=input_shape, dtype=np.float32)
    w = self.rng.standard_normal(size=filter_shape, dtype=np.float32)
    return x, w

  def exact_int8_conv_example(self,
                              lhs_shape=(1, 3, 3, 1),
                              rhs_shape=(2, 2, 1, 2),
                              lhs_share_stats_axes=(1, 2, 3),
                              rhs_share_stats_axes=(0, 1, 2),
                              lhs_use_quantized_variable=False,
                              rhs_use_quantized_variable=False):
    return aqt_test_shared_base.exact_int8_example(
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        lhs_share_stats_axes=list(lhs_share_stats_axes),
        rhs_share_stats_axes=list(rhs_share_stats_axes),
        lhs_use_quantized_variable=lhs_use_quantized_variable,
        rhs_use_quantized_variable=rhs_use_quantized_variable)

  def conv_op_quantized(
      self,
      input,  # pylint: disable=redefined-builtin
      filter,  # pylint: disable=redefined-builtin
      input_config,
      filter_config,
      event_count,
      input_weights=None,
      train=True,
      **kwargs):
    """Performs quantized convolution op.

    For given input and filter configs, this method creates TensorQuantizers,
    update them to the event_count, and returns the result of quantized
    convolution. The reason to deal with TensorQuantizer inside thie method is
    that TensorQuantizer in Jax is nn.Module that should be initialized and
    updated inside another nn.Module. Thus, the arguments needed to manage
    TensorQuantizers are passed and then how they are handled needs to be
    defined in the inheriting classes.

    Args:
      input: The convolution input (lhs).
      filter: The convolution kernel (rhs).
      input_config: AqtScheduleConfig for the input.
      filter_config: AqtScheduleConfig for the filter.
      event_count: event_count to which input/filter TensorQuantizers should be
        updated
      input_weights: The weight of input for updating statistics.
      train: Indicates training mode if True, otherwise, inference mode.
      **kwargs: Keyword arguments to pass onto conv op.
    Returns:
      A tensor of quantized convolution result.
    """
    raise NotImplementedError

  def conv_op_unquantized(self, input, filter, **kwargs):  # pylint: disable=redefined-builtin
    """Performs unquantized convolution op such as tf.conv2d and lax.conv_general_dilated."""
    raise NotImplementedError

  def get_conv_kwargs(
      self,
      strides,
      padding,
      data_format="NHWC",
      dilations=None):
    """Returns a dict of keyword arguments for convolution.

    tf.conv2d and lax.conv_general_dilated have different argument sets,
    names, and their types. For example, the stride of a convolution is
    determined by `strides` in tf.conv2d but by `window_strides` in
    lax.conv_general_dilated. This wrapper should return a dict of such keyword
    arguments customized for each framework.

    Args:
      strides: Stride of the convolution.
      padding: Padding added to all four sides of the input.
      data_format: Data format of the input/output.
      dilations: Spacing between kernel elements.
    """
    raise NotImplementedError

  def get_module_and_side_effect(self):
    raise NotImplementedError

  def constant(self, x):
    raise NotImplementedError

  def gradients(self, fwd_func, x, w):
    raise NotImplementedError

  def test_basic_conv(self):
    """Test on a hand-computed example."""
    x = np.array([[1.3, -0.3, -1.1], [-2.3, 1.1, 0.0], [0.0, -0.5, 10]],
                 dtype=np.float32)[np.newaxis, :, :, np.newaxis]
    w0 = np.array([[1, 5.1], [-5.5, -3]], dtype=np.float32)
    w1 = np.array([[50, 6], [-1.1, 1.2]], dtype=np.float32)
    w = np.stack([w0, w1])[:, :, np.newaxis, :]

    # choose a const such that the clip is 2 for 3 buckets
    cutoff = 2 * 1.5
    input_config = schedule_config("input", const_coeff=cutoff, bits=2)

    cutoff = 10 * 1.5
    filter_config = schedule_config("filter", const_coeff=cutoff, bits=2)

    kwargs = self.get_conv_kwargs(strides=1, padding="VALID")
    actual = self.conv_op_quantized(
        x,
        w,
        input_config,
        filter_config,
        event_count=0,
        **kwargs,
    )

    x = np.array([[1, 0, -1], [-1, 1, 0], [0, 0, 1]],
                 dtype=np.float32)[np.newaxis, :, :, np.newaxis]
    w0 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    w1 = np.array([[1, 1], [0, 0]], dtype=np.float32)
    w = np.stack([w0, w1])[:, :, np.newaxis, :]

    expected = self.conv_op_unquantized(x, w, **kwargs)
    expected *= 2
    expected *= 10

    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters(generate_missing_contraction_dims())
  def test_missing_contraction_dims(self, data_format, filter_axes, input_axes):
    """Throw if spatial dimension stats are unshared."""

    input_config = schedule_config("input", bits=8, const_coeff=1.0)
    input_config.stats_config.share_stats_axes[:] = input_axes
    filter_config = schedule_config("filter", bits=8, const_coeff=1.0)
    filter_config.stats_config.share_stats_axes[:] = filter_axes

    x = np.zeros((1, 3, 3, 1), dtype=np.float32)
    w = np.zeros((2, 2, 1, 2), dtype=np.float32)

    kwargs = self.get_conv_kwargs(
        strides=1, padding="VALID", data_format=data_format)
    with self.assertRaises(aqt_config.ConfigError):
      self.conv_op_quantized(
          x,
          w,
          input_config,
          filter_config,
          event_count=0,
          **kwargs,
      )

  @parameterized.parameters(
      dict(strides=1, padding="SAME", dilations=[1, 2]),
      dict(strides=2, padding="SAME", dilations=[1, 2]),
      dict(strides=1, padding="VALID", dilations=[1, 2]),
      dict(strides=2, padding="VALID", dilations=[1, 2]),
      dict(strides=1, padding="SAME", dilations=[2, 1]),
      dict(strides=2, padding="SAME", dilations=[2, 1]),
      dict(strides=1, padding="VALID", dilations=[2, 1]),
      dict(strides=2, padding="VALID", dilations=[2, 1]),)
  def test_exact_int8_and_no_quant(self, strides, padding, dilations):
    """Validate conv results for exact_int8 and no quantization."""
    kwargs = self.get_conv_kwargs(
        strides=strides,
        padding=padding,
        data_format="NHWC",
        dilations=dilations)

    input_config, x, filter_config, w = self.exact_int8_conv_example()

    actual_exact_int8 = self.conv_op_quantized(
        x,
        w,
        input_config,
        filter_config,
        event_count=0,
        var_scope_name="exact_int8_conv",
        **kwargs,
    )
    expected_exact_int8 = self.conv_op_unquantized(x, w, **kwargs)

    self.assertAllEqual(actual_exact_int8, expected_exact_int8)

    x, w = self.create_random_input_and_filter()
    actual_no_quant = self.conv_op_quantized(
        x,
        w,
        empty_config("input"),
        empty_config("filter"),
        event_count=0,
        var_scope_name="no_quantization",
        **kwargs,
    )
    expected_no_quant = self.conv_op_unquantized(x, w, **kwargs)

    self.assertAllEqual(actual_no_quant, expected_no_quant)

  def test_exact_grads(self):
    """Ensures both quantized and unquantized conv op emit the same gradients."""
    input_config, x, filter_config, w = self.exact_int8_conv_example()

    kwargs = self.get_conv_kwargs(strides=1, padding="VALID")

    def actual_fwd(x, w):
      return self.conv_op_quantized(
          x, w, input_config, filter_config, event_count=0, **kwargs)

    def expected_fwd(x, w):
      return self.conv_op_unquantized(x, w, **kwargs)

    actual = self.gradients(actual_fwd, x, w)
    expected = self.gradients(expected_fwd, x, w)

    for actual_grad, expected_grad in zip(actual, expected):
      self.assertAllEqual(actual_grad, expected_grad)

  def test_filter_dilation(self):
    """Validates zero preservation on the filter when using filater dilation."""
    x = np.array([[1.3, -0.3, -1.1, 0.9], [-2.3, 1.1, 0.0, 1.3],
                  [0.0, -0.5, 10, -0.2], [1.2, 0.0, 5.4, -0.4]],
                 dtype=np.float32)[np.newaxis, :, :, np.newaxis]
    w0 = np.array([[1, 5.1], [-5.5, -3]], dtype=np.float32)
    w1 = np.array([[50, 6], [-1.1, 1.2]], dtype=np.float32)
    w = np.stack([w0, w1])[:, :, np.newaxis, :]
    kwargs = self.get_conv_kwargs(strides=1, padding="VALID", dilations=[2, 2])

    # choose a const such that the clip is 2 for 3 buckets
    cutoff = 2 * 1.5
    input_config = schedule_config("input", const_coeff=cutoff, bits=2)

    cutoff = 10 * 1.5
    filter_config = schedule_config("filter", const_coeff=cutoff, bits=2)

    actual_dilated = self.conv_op_quantized(
        x,
        w,
        input_config,
        filter_config,
        event_count=0,
        var_scope_name="dilated",
        **kwargs,
    )

    # Create the same filter as above but with zero values on the second row
    # and column, resulting in shape (3, 3, 1, 2). This is equivalent with the
    # (2, 2, 1, 2) shaped filter with a dilation factor of 2.
    w0 = np.array([[1, 5.1], [0.0, 0.0], [-5.5, -3]], dtype=np.float32)
    w1 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    w2 = np.array([[50, 6], [0.0, 0.0], [-1.1, 1.2]], dtype=np.float32)
    w = np.stack([w0, w1, w2])[:, :, np.newaxis, :]

    # No dilation.
    kwargs = self.get_conv_kwargs(strides=1, padding="VALID", dilations=None)
    actual_not_dilated = self.conv_op_quantized(
        x,
        w,
        input_config,
        filter_config,
        event_count=0,
        var_scope_name="not_dilated",
        **kwargs,
    )

    x = np.array([[1, 0, -1, 0], [-1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0]],
                 dtype=np.float32)[np.newaxis, :, :, np.newaxis]
    w0 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    w1 = np.array([[1, 1], [0, 0]], dtype=np.float32)
    w = np.stack([w0, w1])[:, :, np.newaxis, :]
    kwargs = self.get_conv_kwargs(strides=1, padding="VALID", dilations=[2, 2])

    expected = self.conv_op_unquantized(x, w, **kwargs)
    expected *= 2
    expected *= 10

    self.assertAllEqual(actual_dilated, actual_not_dilated)
    self.assertAllEqual(actual_dilated, expected)

  def test_zero_preservation_for_filter_dilation(self):
    """Validates preserve_zero is set to True when the filter is dilated."""
    input_config = schedule_config("input", bits=8, const_coeff=1.0)
    filter_config = schedule_config(
        "filter", bits=8, const_coeff=1.0, preserve_zero=False)

    x, w = self.create_random_input_and_filter()
    kwargs = self.get_conv_kwargs(strides=1, padding="VALID", dilations=[2, 2])

    with self.assertRaisesRegex(
        (aqt_config.ConfigError, tf.errors.InvalidArgumentError),
        "must be True if the filter is dilated"):
      self.conv_op_quantized(
          x,
          w,
          input_config,
          filter_config,
          event_count=0,
          **kwargs)
