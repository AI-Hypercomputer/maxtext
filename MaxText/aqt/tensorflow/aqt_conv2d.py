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
"""Quantized two-dimensional convolution.

Quantized convolution abstracts over the analogous :py:func:`tf.conv2d` and
:py:func:`tf.conv2d_transpose` APIs to provide their quantized analogues.

Both the filter and input arguments to the convolution are quantized.
One difference in the interface for AQTp here compared to the Jax one
is that filter and input dilations are relegated to separate functions
(`conv2d` and `conv2d_transpose`), whereas in Jax the general convolution
API provides both dilations simultaneously. This reflects the underlying
frameworks' differences.

Further, note that we do not allow different weighting for kernel parameters,
unlike the general :py:func:`aqt_ops.matmul`.
"""

from aqt.common import aqt_config
from aqt.common import aqt_config_utils
from aqt.tensorflow import aqt_tensor
import tensorflow.compat.v1 as tf

# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access


def _assert_dilation_argument(lhs_quantizer, rhs_quantizer, dilations):
  """Validates preserve_zero=True of AqtIntQuantConfig when the dilation is used."""
  lhs_configs = lhs_quantizer.config.tensor_configs
  rhs_configs = rhs_quantizer.config.tensor_configs
  event_count = lhs_quantizer._last_update

  is_valid = tf.constant(True)
  for lhs_config, rhs_config in zip(lhs_configs, rhs_configs):
    if (isinstance(lhs_config.quant_config, aqt_config.IntQuantConfig) and
        isinstance(rhs_config.quant_config, aqt_config.IntQuantConfig) and
        lhs_config.quant_config.bits <= 8 and
        rhs_config.quant_config.bits <= 8):
      is_config_active = aqt_tensor.is_config_active(rhs_config, event_count)

      is_valid &= (~is_config_active | (dilations is None)
                   | rhs_config.quant_config.preserve_zero)

  assert_op = tf.debugging.Assert(is_valid, [
      'rhs_config.quant_config.preserve_zero must be True if the filter is dilated.'   # pylint: disable=line-too-long
  ])

  return assert_op


def _validate_contraction(
    config: aqt_config.StatsConfig,  # Prevent python auto-formatting.
    config_path: str,
    data_format: str) -> None:
  """Validates that conv2d contraction axes have shared stats.

  The convolution operation can be viewed as a higher-order tensor contraction
  along the height, width, and input channel axes for its input and filter.
  When quantizing the convolution, we thus have to keep the same multiplicative
  scale during calibration along these axes. Otherwise, we'd be changing the
  implicit inner product the operation computes in a manner besides
  quantization.

  Args:
    config: The configuration for recording conv2d argument statistics.
    config_path: A string for clearer error messages, to refer to the config.
    data_format: the format of the tensor, with H denoting height, W width, and
      C input channels.

  Raises:
    aqt_config.ConfigError: `config.share_stats_axes` does not include any
      of the contraction axes.
  """

  stats_axes = config.share_stats_axes

  axis_names = ['height', 'width', 'channel']
  dims = ['H', 'W', 'C']

  for axis_name, dim in zip(axis_names, dims):
    axis = data_format.index(dim)
    if axis not in stats_axes:
      raise aqt_config.ConfigError(
          f'expected {axis_name} contraction axis ({axis}) to be in '
          f'{config_path}.share_stats_axes={stats_axes}')


def _validate_inputs(
    input_quantizer: aqt_tensor.TensorQuantizer,  #
    filter_quantizer: aqt_tensor.TensorQuantizer,
    data_format: str) -> None:
  """Validates configs and inputs for conv2d.

  Args:
    input_quantizer: the input tensor quantizer.
    filter_quantizer: the filter tensor quantizer.
    data_format: the conv2d input tensor argument axes format.

  Raises:
    aqt_config.ConfigError: The input or filter quantizer configurations
      do not share statistics along contraction axes as described by
      `_validate_contraction`, or their quantization schedules are not aligned
      per `aqt_config_utils._validate_alignment`.
  """

  aqt_config_utils._validate_alignment(
      'input_quantizer.config.tensor_configs',
      input_quantizer.config.tensor_configs,
      'filter_quantizer.config.tensor_configs',
      filter_quantizer.config.tensor_configs)

  _validate_contraction(input_quantizer.config.stats_config,
                        'input_quantizer.config.stats_config', data_format)

  # Filters are always assumed to be in HWCO format, where
  # H - height, W - width, O - output channels, C - input channels.
  # Note input is NHWC or NCHW where N - batch size.
  _validate_contraction(filter_quantizer.config.stats_config,
                        'filter_quantizer.config.stats_config', 'HWCO')


def conv2d(
    input_quantizer: aqt_tensor.TensorQuantizer,  #
    input: tf.Tensor,  # pylint: disable=redefined-builtin
    filter_quantizer: aqt_tensor.TensorQuantizer,
    filter: tf.Tensor,  # pylint: disable=redefined-builtin
    train: bool = True,
    **tf_conv2d_kwargs):
  r"""Quantized :py:func:`tf.nn.conv2d`.

  Quantization is symmetric and uniform, with range determined by the
  `config`. Otherwise, all arguments are passed on to
  :py:func:`tf.nn.conv2d`.

  Args:
    input_quantizer: TensorQuantizer for input
    input: A `tf.Tensor`. Must be `float32`. The convolution input.
    filter_quantizer: TensorQuantizer for filter
    filter: A `tf.Tensor`. Must have the same type as `input`. The convolution
      kernel.
    train: if False, allows static switching for inference quantization
      configuration, a performance optimization.
    **tf_conv2d_kwargs: Keyword arguments to pass onto `conv2d`.

  Returns:
    A tensor of the same type as `input` conformal to what `conv2d` would
    return.
  """
  data_format = tf_conv2d_kwargs.get('data_format', 'NHWC')
  _validate_inputs(input_quantizer, filter_quantizer, data_format)
  assert_op = _assert_dilation_argument(input_quantizer, filter_quantizer,
                                        tf_conv2d_kwargs.get('dilations', None))

  with tf.control_dependencies([assert_op]):
    with tf.name_scope('AqtConv2d'):
      with tf.name_scope('get_quant_scale'):
        with tf.name_scope('input'):
          input_scale, input_inv_scale = input_quantizer._get_quant_scale(
              train)  # [N, 1, 1, 1]
        with tf.name_scope('filter'):
          filter_scale, filter_inv_scale = filter_quantizer._get_quant_scale(
              train)  # [1, 1, 1, O]

      input = input_scale * input  # [N, 1, 1, 1] * [N, H, W, C]
      filter = filter_scale * filter  # [1, 1, 1, O] * [H, W, I, O]

      with tf.name_scope('to_quant'):
        with tf.name_scope('input'):
          input = input_quantizer._to_quant(input, train)  # [N, H, W, C]
        with tf.name_scope('filter'):
          filter = filter_quantizer._to_quant(filter, train)  # [H, W, I, O]

      # TODO(vladf): until tf.conv2d supports int8 arguments, we need to cast
      # the quantized variables to a floating point format.
      if input_quantizer.config.use_quantized_variable and not train:
        input = tf.cast(input_quantizer.quantized_variable.read_value(),
                        tf.float32)
      if filter_quantizer.config.use_quantized_variable and not train:
        filter = tf.cast(filter_quantizer.quantized_variable.read_value(),
                         tf.float32)

      with tf.name_scope('conv2d'):
        # TODO(vladf): implement calls to narrowed functions, e.g., int8_conv2d,
        # cased by the bitwidth settings for our tensor quantization
        # configurations, in preparation for quantization support in conv2d.
        conv = tf.nn.conv2d(input, filter, **tf_conv2d_kwargs)  # [N, H, W, O]

      with tf.name_scope('inv_scale'):
        # [N, H, W, O] * ([N, 1, 1, 1] * [1, 1, 1, O])
        return conv * (input_inv_scale * filter_inv_scale)


def depthwise_conv2d(
    input_quantizer: aqt_tensor.TensorQuantizer,  #
    input: tf.Tensor,  # pylint: disable=redefined-builtin
    filter_quantizer: aqt_tensor.TensorQuantizer,
    filter: tf.Tensor,  # pylint: disable=redefined-builtin
    train: bool = True,
    **tf_dw_conv2d_kwargs):
  r"""Quantized :py:func:`tf.nn.depthwise_conv2d`.

  Quantization is symmetric and uniform, with range determined by the
  `config`. Otherwise, all arguments are passed on to
  :py:func:`tf.nn.depthwise_conv2d`.

  Args:
    input_quantizer: TensorQuantizer for input
    input: A `tf.Tensor`. Must be `float32`. The convolution input.
    filter_quantizer: TensorQuantizer for filter
    filter: A `tf.Tensor`. Must have the same type as `input`. The convolution
      kernel.
    train: if False, allows static switching for inference quantization
      configuration, a performance optimization.
    **tf_dw_conv2d_kwargs: Keyword arguments to pass onto `depthwise_conv2d`.

  Returns:
    A tensor of the same type as `input` conformal to what
    `depthwise_conv2d` would return.
  """
  data_format = tf_dw_conv2d_kwargs.get('data_format', 'NHWC')
  _validate_inputs(input_quantizer, filter_quantizer, data_format)
  assert_op = _assert_dilation_argument(
      input_quantizer, filter_quantizer,
      tf_dw_conv2d_kwargs.get('dilations', None))

  with tf.control_dependencies([assert_op]):
    with tf.name_scope('AqtDepthwiseConv2d'):
      with tf.name_scope('get_quant_scale'):
        with tf.name_scope('input'):
          input_scale, input_inv_scale = input_quantizer._get_quant_scale(
              train)  # [N, 1, 1, 1]
        with tf.name_scope('filter'):
          filter_scale, filter_inv_scale = filter_quantizer._get_quant_scale(
              train)  # [1, 1, 1, O]

      input = input_scale * input  # [N, 1, 1, 1] * [N, H, W, C]
      filter = filter_scale * filter  # [1, 1, 1, O] * [H, W, I, O]

      with tf.name_scope('to_quant'):
        with tf.name_scope('input'):
          input = input_quantizer._to_quant(input, train)  # [N, H, W, C]
        with tf.name_scope('filter'):
          filter = filter_quantizer._to_quant(filter, train)  # [H, W, I, O]

      if input_quantizer.config.use_quantized_variable and not train:
        input = tf.cast(input_quantizer.quantized_variable.read_value(),
                        tf.float32)
      if filter_quantizer.config.use_quantized_variable and not train:
        filter = tf.cast(filter_quantizer.quantized_variable.read_value(),
                         tf.float32)

      with tf.name_scope('depthwise_conv2d'):
        dw_conv = tf.nn.depthwise_conv2d(input, filter, **tf_dw_conv2d_kwargs)

      with tf.name_scope('inv_scale'):
        # [N, H, W, O] * ([N, 1, 1, 1] * [1, 1, 1, O])
        return dw_conv * (input_inv_scale * filter_inv_scale)
