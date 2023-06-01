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

"""Tests for conv_general."""

import itertools
from typing import Iterable, Optional

from absl.testing import absltest
from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.jax import aqt_conv_general
from aqt.jax import aqt_ops
from aqt.jax import aqt_tensor
from aqt.test import aqt_conv_test_base
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp


def generate_nondefault_conv_args():
  lhs_specs = ["NHWC", "HNWC", "HWNC", "HWCN"]
  rhs_specs = ["OHWI", "HOWI", "HWOI", "HWIO"]
  out_specs = ["CNHW", "NCHW", "NHCW", "NHWC"]

  case_dicts = []
  for lhs_spec, rhs_spec, out_spec in itertools.product(lhs_specs, rhs_specs,
                                                        out_specs):
    case_dicts.append(
        dict(
            dimension_numbers=(lhs_spec, rhs_spec, out_spec),
            window_strides=[1, 1],
            padding="VALID"))

  window_strides = [(1, 1), (2, 2), (1, 2), (2, 1)]
  paddings = ["SAME", "VALID", ((1, 2), (2, 1)), ((2, 2), (2, 2))]
  for strides, padding in itertools.product(window_strides, paddings):
    case_dicts.append(
        dict(
            dimension_numbers=("NHWC", "HWIO", "NWHC"),
            window_strides=strides,
            padding=padding))

  return case_dicts


class ConvGeneralModule(nn.Module):
  lhs_config: Optional[aqt_config.AqtScheduleConfig]
  rhs_config: Optional[aqt_config.AqtScheduleConfig]
  lhs_shape: Iterable[int]
  rhs_shape: Iterable[int]

  @nn.compact
  def __call__(self,
               lhs,
               rhs,
               weights,
               event_count,
               event_count_for_filter,
               train=True,
               **kwargs):
    lhs_quantizer = aqt_tensor.TensorQuantizer(
        list(self.lhs_shape), self.lhs_config)
    rhs_quantizer = aqt_tensor.TensorQuantizer(
        list(self.rhs_shape), self.rhs_config)

    lhs_quantizer.update(lhs, weights, event_count)
    rhs_quantizer.update(rhs, weights, event_count_for_filter)

    def conv_op(lhs, rhs):
      return aqt_ops.aqt_conv_general_dilated(
          lhs, rhs, lhs_quantizer, rhs_quantizer, train=train, **kwargs)

    return conv_op


class ConvGeneralTest(aqt_conv_test_base.ConvTest):

  def conv_op_quantized(
      self,
      input,  # pylint: disable=redefined-builtin
      filter,  # pylint: disable=redefined-builtin
      input_config,
      filter_config,
      event_count,
      event_count_for_filter=None,
      input_weights=None,
      train=True,
      var_scope_name=None,
      **kwargs):
    event_count_for_filter = event_count_for_filter if event_count_for_filter else event_count

    input = self.constant(input)
    filter = self.constant(filter)

    module = ConvGeneralModule(input_config, filter_config, input.shape,
                               filter.shape)
    conv_general, _ = module.init_with_output(
        jax.random.PRNGKey(0), input, filter, input_weights, event_count,
        event_count_for_filter, train, **kwargs)

    return conv_general(input, filter)

  def conv_op_unquantized(self, input, filter, **kwargs):  # pylint: disable=redefined-builtin
    input = self.constant(input)
    filter = self.constant(filter)
    return lax.conv_general_dilated(input, filter, **kwargs)

  def get_conv_kwargs(
      self,
      strides,
      padding,
      data_format="NHWC",
      dilations=None,
      lhs_dilation=None):
    return dict(
        window_strides=(strides, strides),
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=dilations,
        dimension_numbers=(data_format, "HWIO", data_format),
        feature_group_count=1,
        batch_group_count=1)

  def constant(self, x):
    return jnp.array(x)

  def gradients(self, fwd_func, x, w):
    y, vjp_fn = jax.vjp(fwd_func, x, w)
    return vjp_fn(jnp.ones(y.shape))[0]

  def test_vars_over_inputs_at_inference(self):
    input_config, x, filter_config, w = self.exact_int8_conv_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True)

    x = self.constant(x)
    w = self.constant(w)
    kwargs = self.get_conv_kwargs(strides=1, padding="VALID")

    module = ConvGeneralModule(input_config, filter_config, x.shape, w.shape)

    # Since quantized variables are not used at training, the conv_general with
    # zero inputs should produce zero values.
    conv_general, state = module.init_with_output(
        jax.random.PRNGKey(0),
        x,
        w,
        weights=None,
        event_count=0,
        event_count_for_filter=0,
        train=True,  # in training
        **kwargs)
    actual = conv_general(jnp.zeros_like(x), jnp.zeros_like(w))
    expected = jnp.zeros_like(actual)
    self.assertAllEqual(actual, expected)

    # Since quantized variables should be always used at inference, the
    # conv_general will rely on quantized variables.
    conv_general_infer, _ = module.apply(
        state,
        x,
        w,
        weights=None,
        event_count=0,
        event_count_for_filter=0,
        train=False,  # in inference
        **kwargs,
        mutable=True)
    actual = conv_general_infer(jnp.zeros_like(x), jnp.zeros_like(w))
    expected = conv_general(x, w)
    self.assertAllEqual(actual, expected)

  def test_zero_preservation_for_input_dilation(self):
    """Validates preserve_zero is set to True when the filter is dilated."""
    input_config = aqt_conv_test_base.schedule_config(
        "input", bits=8, const_coeff=1.0, preserve_zero=False)
    filter_config = aqt_conv_test_base.schedule_config(
        "filter", bits=8, const_coeff=1.0)

    x, w = self.create_random_input_and_filter()

    kwargs = self.get_conv_kwargs(
        strides=1, padding=((1, 1), (1, 1)), lhs_dilation=[2, 2])

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "must be True if the input is dilated"):
      self.conv_op_quantized(
          x,
          w,
          input_config,
          filter_config,
          event_count=0,
          **kwargs)

  def test_jvp(self):
    input_config, x, filter_config, w = self.exact_int8_conv_example()

    kwargs = self.get_conv_kwargs(strides=1, padding="VALID")

    def actual_fwd(x, w):
      return self.conv_op_quantized(
          x, w, input_config, filter_config, event_count=0, **kwargs)

    def expected_fwd(x, w):
      return self.conv_op_unquantized(x, w, **kwargs)

    actual = lambda x, w: jnp.sum(actual_fwd(x, w)**2)
    expected = lambda x, w: jnp.sum(expected_fwd(x, w)**2)

    _, y_dot_actual = jax.jvp(actual, (x, w),
                              (jnp.ones(x.shape), jnp.ones(w.shape)))
    _, y_dot_expected = jax.jvp(expected, (x, w),
                                (jnp.ones(x.shape), jnp.ones(w.shape)))
    self.assertAllEqual(y_dot_actual, y_dot_expected)

  @parameterized.parameters(generate_nondefault_conv_args())
  def test_nondefault_conv_args(self, dimension_numbers, window_strides,
                                padding):
    def get_share_stats_axes(dimension_numbers):
      # Based on the dimension number of string format, create share_stats_axes
      # which should be provided to stats config. Note that the first dimension
      # is always non-contracting one: "N" for LHS and "O" for RHS.
      return sorted(dimension_numbers[1:])

    # Create an input tensor of the shape (2, 3, 3, 1) and a filter tensor of
    # the shape (2, 2, 1, 2) with ("NHWC", "HWIO, "NHWC") dimension number.
    # In order to effectively see that the inv_scales are applied to conv result
    # correctly no matter which dimension numbers are specified, each of the
    # batch/feature dimensions should have different scales.
    x = jnp.array(
        [[[12.6, -23.1, -4.2], [-40.3, 255.0, 0.0], [0.0, -32.5, 55.1]],
         [[63.75, -47.3, -1.3], [-24.2, 14.4, 0.0], [0.0, -29.4, 30.1]]],
        dtype=jnp.float32)[:, :, :, jnp.newaxis]
    w0 = jnp.array([[1, 7.5], [-5.5, -3]], dtype=jnp.float32)
    w1 = jnp.array([[15, -2.4], [-1.1, 1.2]], dtype=jnp.float32)
    w = jnp.stack([w0, w1])[:, :, jnp.newaxis, :]

    # 8 bits quantization with max_dev_coeff=1
    input_config = aqt_conv_test_base.schedule_config(
        "input", const_coeff=0, bits=8, max_dev_coeff=1)
    # 2 bits quantization with max_dev_coeff=1
    filter_config = aqt_conv_test_base.schedule_config(
        "filter", const_coeff=0, bits=2, max_dev_coeff=1)

    dimension_numbers = lax.conv_dimension_numbers(x.shape, w.shape,
                                                   dimension_numbers)
    # Transpose the input and filter according to given dimension numbers.
    dim_numbers = {
        "NHWC": (0, 3, 1, 2),
        "HWIO": (3, 2, 0, 1),
    }
    x = aqt_conv_general._transpose_inv_scale(x, dim_numbers["NHWC"],
                                              dimension_numbers[0])
    w = aqt_conv_general._transpose_inv_scale(w, dim_numbers["HWIO"],
                                              dimension_numbers[1])
    input_config.stats_config.share_stats_axes = get_share_stats_axes(
        dimension_numbers[0])
    filter_config.stats_config.share_stats_axes = get_share_stats_axes(
        dimension_numbers[1])

    kwargs = dict(
        window_strides=window_strides,
        padding=padding,
        dimension_numbers=dimension_numbers)

    actual = self.conv_op_quantized(
        x, w, input_config, filter_config, event_count=0, **kwargs)

    # The quantized tensors for the examples created above.
    x = jnp.array([[[6, -12, -2], [-20, 127, 0], [0, -16, 28]],
                   [[127, -95, -3], [-48, 29, 0], [0, -59, 60]]],
                  dtype=jnp.float32)[:, :, :, jnp.newaxis]
    w0 = jnp.array([[0, 1], [-1, -1]], dtype=jnp.float32)
    w1 = jnp.array([[1, 0], [0, 0]], dtype=jnp.float32)
    w = jnp.stack([w0, w1])[:, :, jnp.newaxis, :]

    x = aqt_conv_general._transpose_inv_scale(x, dim_numbers["NHWC"],
                                              dimension_numbers[0])
    w = aqt_conv_general._transpose_inv_scale(w, dim_numbers["HWIO"],
                                              dimension_numbers[1])

    expected = self.conv_op_unquantized(x, w, **kwargs)

    batch_inv_scale = jnp.array([2, 0.5])[:, jnp.newaxis, jnp.newaxis,
                                          jnp.newaxis]
    batch_inv_scale = aqt_conv_general._transpose_inv_scale(
        batch_inv_scale, dim_numbers["NHWC"], dimension_numbers[2])
    feature_inv_scale = jnp.array([10, 5])[jnp.newaxis, jnp.newaxis,
                                           jnp.newaxis, :]
    feature_inv_scale = aqt_conv_general._transpose_inv_scale(
        feature_inv_scale, dim_numbers["NHWC"], dimension_numbers[2])
    expected *= (batch_inv_scale * feature_inv_scale)

    self.assertAllEqual(actual, expected)

  def test_weight_only_quantization(self):
    x, w = self.create_random_input_and_filter()
    input_config = aqt_conv_test_base.schedule_config(
        "input", bits=8, const_coeff=1.0)
    filter_config = aqt_conv_test_base.schedule_config(
        "filter", bits=8, const_coeff=1.0)
    input_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    kwargs = self.get_conv_kwargs(strides=1, padding="VALID")
    result_with_float_config = self.conv_op_quantized(
        x, w, input_config, filter_config, event_count=0, **kwargs)
    result_with_none_config = self.conv_op_quantized(
        x, w, None, filter_config, event_count=0, **kwargs)

    self.assertAllEqual(result_with_float_config, result_with_none_config)


if __name__ == "__main__":
  absltest.main()
