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

from absl.testing import absltest
from aqt.tensorflow import aqt_ops
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_conv_test_base
import tensorflow.compat.v1 as tf


class Conv2dTest(aqt_conv_test_base.ConvTest):

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
    event_count_for_filter = tf.constant(
        event_count_for_filter if event_count_for_filter else event_count,
        dtype=tf.int64)
    event_count = tf.constant(event_count, dtype=tf.int64)

    input = self.constant(input)
    filter = self.constant(filter)

    with tf.variable_scope(var_scope_name, default_name="default"):
      input_tq = aqt_tensor.TensorQuantizer(
          input.shape, input_config, name="input")
      filter_tq = aqt_tensor.TensorQuantizer(
          filter.shape, filter_config, name="filter")

    updates = [input_tq.update(input, input_weights, event_count),
               filter_tq.update(filter, None, event_count_for_filter)]

    with tf.control_dependencies(updates):
      result = aqt_ops.aqt_conv2d(input_tq, input, filter_tq, filter, **kwargs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      return result.eval()

  def conv_op_unquantized(self, input, filter, **kwargs):  # pylint: disable=redefined-builtin
    input = self.constant(input)
    filter = self.constant(filter)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      return tf.nn.conv2d(input, filter, **kwargs).eval()

  def get_conv_kwargs(
      self,
      strides,
      padding,
      data_format="NHWC",
      dilations=None):
    return dict(
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations)

  def constant(self, x):
    return tf.constant(x)

  def gradients(self, fwd_func, x, w):
    fwd = fwd_func(x, w)
    return tf.gradients([fwd], [x, w])

  def test_vars_over_inputs_at_inference(self):
    input_config, x, filter_config, w = self.exact_int8_conv_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True)

    x = self.constant(x)
    w = self.constant(w)
    kwargs = self.get_conv_kwargs(strides=1, padding="VALID")

    input_tq = aqt_tensor.TensorQuantizer(x.shape, input_config, name="input")
    filter_tq = aqt_tensor.TensorQuantizer(
        w.shape, filter_config, name="filter")
    event_count = tf.constant(0, dtype=tf.int64)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()

      input_tq.update(x, None, event_count).run()
      filter_tq.update(w, None, event_count).run()

      # Since quantized variables are not used at training, the conv2d with zero
      # inputs should produce zero values.
      actual_train = aqt_ops.aqt_conv2d(
          input_tq,
          tf.zeros_like(x),
          filter_tq,
          tf.zeros_like(w),
          train=True,
          **kwargs)
      expected = tf.zeros_like(actual_train)
      self.assertAllEqual(actual_train, expected)

      # Since quantized variables should be always used at inference, the conv2d
      # will rely on quantized variables.
      actual_infer = aqt_ops.aqt_conv2d(
          input_tq,
          tf.zeros_like(x),
          filter_tq,
          tf.zeros_like(w),
          train=False,
          **kwargs)
      expected = aqt_ops.aqt_conv2d(
          input_tq, x, filter_tq, w, train=True, **kwargs)
      self.assertAllEqual(actual_infer, expected)


class DepthwiseConv2dTest(Conv2dTest):

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
    event_count_for_filter = tf.constant(
        event_count_for_filter if event_count_for_filter else event_count,
        dtype=tf.int64)
    event_count = tf.constant(event_count, dtype=tf.int64)

    input = self.constant(input)
    filter = self.constant(filter)

    with tf.variable_scope(var_scope_name, default_name="default"):
      input_tq = aqt_tensor.TensorQuantizer(
          input.shape, input_config, name="input")
      filter_tq = aqt_tensor.TensorQuantizer(
          filter.shape, filter_config, name="filter")

    updates = [input_tq.update(input, input_weights, event_count),
               filter_tq.update(filter, None, event_count_for_filter)]

    # depthwise_conv2d expects a different stride format than conv2d
    if "strides" in kwargs:
      stride = kwargs["strides"]
      kwargs["strides"] = [1, stride, stride, 1]

    with tf.control_dependencies(updates):
      result = aqt_ops.aqt_depthwise_conv2d(input_tq, input, filter_tq,
                                            filter, **kwargs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      return result.eval()

  def conv_op_unquantized(self, input, filter, **kwargs):  # pylint: disable=redefined-builtin
    input = self.constant(input)
    filter = self.constant(filter)

    # tf.nn.depthwise_conv2d expects a different stride format than *_conv2d
    if "strides" in kwargs:
      stride = kwargs["strides"]
      kwargs["strides"] = [1, stride, stride, 1]

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      return tf.nn.depthwise_conv2d(input, filter, **kwargs).eval()

if __name__ == "__main__":
  absltest.main()
