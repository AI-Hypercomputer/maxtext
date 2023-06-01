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

"""Tests for aqt.jax.flax_layers."""

import functools
import itertools
from typing import Any, Dict, Mapping
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax_legacy.jax import flax_layers
from aqt.jax_legacy.jax import fp_cast
from aqt.jax_legacy.jax import get_bounds
from aqt.jax_legacy.jax import primitives
from aqt.jax_legacy.jax import quant_config
from aqt.jax_legacy.jax import quantization
from aqt.jax_legacy.jax import shape_utils
from aqt.jax_legacy.jax import test_utils
from aqt.jax_legacy.jax.quantization import QuantOps
from aqt.jax_legacy.jax.quantization import QuantType
import flax
from flax import linen as nn
from flax import traverse_util
import jax
from jax import config
from jax import dtypes
from jax import lax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as onp


FLAGS = flags.FLAGS

# fp-1-4-3
#    1: sign
#    4: number of exponent-bits, (bias = 11), range: -11, ..., 4
#    3: number of significand-bits (excluding hidden-bit)
fp143_scaled = QuantOps.FloatQuant(
    is_scaled=True,
    fp_spec=QuantOps.FloatQuant.FloatPrec(
        exp_min=-11,
        exp_max=4,
        sig_bits=3,
    ),
)
fp143_unscaled = QuantOps.FloatQuant(
    is_scaled=False,
    fp_spec=QuantOps.FloatQuant.FloatPrec(
        exp_min=-11,
        exp_max=4,
        sig_bits=3,
    ),
)

METADATA_KEY = '__save_format_metadata__'


def filter_out_metadata(params: Mapping[str, Any]) -> Dict[str, Any]:
  """Removes "__save_format_metadata__" entries from a parameter tree."""
  result = {}
  for k, v in params.items():
    if k == METADATA_KEY:
      continue
    if isinstance(v, Mapping):
      v = filter_out_metadata(v)
      if not v:
        continue
    result[k] = v
  return result


def param_dtypes_shapes_axes(params: Mapping[str, Any],
                             params_axes: Mapping[str, Any]) -> Dict[str, Any]:
  """Construct a tree of param info including dtypes, shapes, and axis names.

  The leaf of the constructed dtree are of format [<dtype>, <axis_dim>, ...],
  where each <axis_dim> is of format <axis_name>=<dim>.

  Args:
    params: Model params.
    params_axes: Axis annotations, typically under state["params_axes"].

  Returns:
    A pytree with params info.
  """
  params = filter_out_metadata(params)
  params_axes = filter_out_metadata(params_axes)
  params = flax.core.unfreeze(params)  # pytype: disable=wrong-arg-types

  def remove_axes_suffix(ks):
    if not ks[-1].endswith('_axes'):
      raise ValueError(
          f'Param axes name should end with `_axes`, found {ks[-1]}')
    return tuple(ks[:-1]) + (ks[-1][:-len('_axes')],)

  params_axes = flax.core.unfreeze(params_axes)  # pytype: disable=wrong-arg-types
  flatten_axes = {
      remove_axes_suffix(ks): v
      for ks, v in traverse_util.flatten_dict(params_axes).items()
  }
  params_axes = traverse_util.unflatten_dict(flatten_axes)

  def _create_entry(param, param_axes):
    output = [str(param.dtype)]
    # The param axes should be paired with param dimension, so we check that.
    if param.ndim != len(param_axes.names):
      raise ValueError('Length of param dimension does not match axes, '
                       f'{param.shape} != {param_axes.names}.')
    for dim, axis_name in zip(param.shape, param_axes.names):
      output.append(f'{axis_name}={dim}')
    return output

  return jax.tree_map(_create_entry, params, params_axes)


class ConvAqtTest(parameterized.TestCase):
  """Tests for ConvAqt layer."""

  def setUp(self):
    super(ConvAqtTest, self).setUp()
    test_utils.configure_jax()
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = True
    self.rng_key = random.PRNGKey(0)

  def tearDown(self):
    config.update('jax_numpy_rank_promotion', 'warn')
    super(ConvAqtTest, self).tearDown()

  def init_model_with_1_layer(self,
                              inputs,
                              num_features,
                              kernel_size,
                              kernel_init=flax_layers.default_kernel_init,
                              weight_prec=None,
                              quant_act=None,
                              weight_half_shift=False):
    """Create and initialize a flax model with a single ConvAqt layer."""
    layer_kwargs = {
        'kernel_init': kernel_init,
        'features': num_features,
        'use_bias': False,
        'dynamic_context': quant_config.DynamicContext(update_bounds=False),
        'paxis_name': 'batch',
        'train': False,
        'kernel_size': kernel_size,
        'dtype': jnp.float32
    }
    layer_class = flax_layers.ConvAqt
    layer_kwargs['hparams'] = flax_layers.ConvAqt.HParams(
        weight_prec=weight_prec,
        quant_act=quant_act,
        quant_type=QuantType.FAKE_QUANT,
        weight_half_shift=weight_half_shift,
    )
    conv_module = layer_class(**layer_kwargs)
    initial_state = conv_module.init(self.rng_key, jnp.zeros(inputs.shape))
    return conv_module, initial_state

  # Following ConvAqt tests adapted from
  # Flax Conv tests.
  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
      dict(testcase_name='quant_2bit', weight_prec=2),
  )
  def test_conv(self, weight_prec=None):
    x = jnp.ones((1, 8, 8, 3))
    conv_module = flax_layers.ConvAqt(
        features=4,
        kernel_size=(3, 3),
        padding='VALID',
        paxis_name='batch',
        dynamic_context=quant_config.DynamicContext(update_bounds=False),
        train=False,
        hparams=flax_layers.ConvAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.FAKE_QUANT,
            weight_half_shift=False),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        dtype=jnp.float32)

    y, state = conv_module.init_with_output(self.rng_key, x)
    self.assertEqual(state['params']['kernel'].shape, (3, 3, 3, 4))
    test_utils.assert_all_close_prec(y, onp.full((1, 6, 6, 4), 28.),
                                     weight_prec)

  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
      dict(testcase_name='quant_2bit', weight_prec=2),
  )
  def test_group_conv(self, weight_prec=None):
    x = jnp.ones((1, 8, 8, 4))
    conv_module = flax_layers.ConvAqt(
        features=4,
        kernel_size=(3, 3),
        feature_group_count=2,
        padding='VALID',
        paxis_name='batch',
        dynamic_context=quant_config.DynamicContext(update_bounds=False),
        train=False,
        hparams=flax_layers.ConvAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.FAKE_QUANT,
            weight_half_shift=False),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        dtype=jnp.float32)
    y, state = conv_module.init_with_output(self.rng_key, x)
    self.assertEqual(state['params']['kernel'].shape, (3, 3, 2, 4))
    test_utils.assert_all_close_prec(y, onp.full((1, 6, 6, 4), 19.),
                                     weight_prec)

  @parameterized.named_parameters(
      dict(testcase_name='conv_quant_8bit', weight_prec=8),
      dict(testcase_name='conv_quant_4bit', weight_prec=4),
      dict(testcase_name='conv_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_should_give_precise_output(
      self, weight_prec):
    # If weights are ints (already quantized) and
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel,
    # no quantization error should be introduced.

    num_features = 256
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 16, 16, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, kernel_size, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(
        self.rng_key, kernel_size + (input_dim, num_features), minval,
        maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = (
        full_range_integer_weights.at[0, 0, :].set(maxval))
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = full_range_integer_weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)

    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint: disable=protected-access
    exp_outputs = lax.conv_general_dilated(
        inputs,
        jnp.asarray(state['params']['kernel'], jnp.float32), (1, 1),
        'SAME',
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=jax.lax.Precision.DEFAULT)

    onp.testing.assert_array_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      dict(testcase_name='conv_quant_4bit', weight_prec=4),
      dict(testcase_name='conv_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_with_float_scale_should_give_close_output(
      self, weight_prec):
    # If weights are ints (already quantized) with
    # max(abs(weights[..., ch])) == 2**(prec-1)-1 in each channel
    # and if these integer weights are multiplied by a float scale,
    # the resulting error should still be very small (just float rounding).

    num_features = 256
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 16, 16, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, kernel_size, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(
        self.rng_key, kernel_size + (input_dim, num_features), minval,
        maxval + 1)
    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = (
        full_range_integer_weights.at[0, 0, :].set(maxval))

    # (batch_size, spatial_dim, spatial_dim, num_features)
    float_scale = jax.random.uniform(self.rng_key, (1, 1, 1, num_features))
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = full_range_integer_weights * float_scale
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)
    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint: disable=protected-access
    exp_outputs = lax.conv_general_dilated(
        inputs,
        jnp.asarray(state['params']['kernel'], jnp.float32), (1, 1),
        'SAME',
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=jax.lax.Precision.DEFAULT)
    # We know that the noise should be proportional to the square root of
    # input_dim and inversely proportional to 2**weight_prec.
    # The following tol_const was obtained experimentally and should be derived
    # more systematically.
    tol_const = 5e-02
    onp.testing.assert_allclose(
        outputs,
        exp_outputs,
        rtol=jnp.sqrt(input_dim) * 2**(-weight_prec) * tol_const)

  @parameterized.named_parameters(
      dict(
          testcase_name='conv_quant_8bit',
          weight_prec=8,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='conv_quant_4bit',
          weight_prec=4,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='conv_quant_2bit',
          weight_prec=2,
          weight_scale=onp.array([1, 2, 4, 8])),
  )
  def test_weight_invariance_to_power_of_2_weight_scaling(
      self, weight_prec, weight_scale):
    # Scaling the weights before quantization by a power of 2 per channel should
    # also scale the output exactly by the same scale.

    num_features = 4
    assert num_features == weight_scale.shape[-1]
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 16, 16, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs,
        4,
        kernel_size,
        weight_prec=weight_prec,
        weight_half_shift=False)

    weights = random.uniform(
        self.rng_key, shape=kernel_size + (input_dim, num_features))
    weight_scale = weight_scale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = weights
    outputs_without_scaling = model.apply(flax.core.freeze(state), inputs)
    state['params']['kernel'] = jnp.multiply(weights, weight_scale)
    outputs_with_scaling = model.apply(flax.core.freeze(state), inputs)

    onp.testing.assert_array_equal(outputs_without_scaling * weight_scale,
                                   outputs_with_scaling)

  def test_1_bit_makes_all_weight_equal_to_zero(self):
    num_features = 4
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 32, 32, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, kernel_size, weight_prec=1)
    weights = random.uniform(
        self.rng_key, shape=kernel_size + (input_dim, num_features))
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = weights
    outputs = model.apply(flax.core.freeze(state), inputs)
    onp.testing.assert_array_equal(outputs, onp.zeros(
        (1, 32, 32, num_features)))

  @parameterized.named_parameters(
      dict(
          testcase_name='conv_quant_8bit',
          weight_prec=8,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='conv_quant_4bit',
          weight_prec=4,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='conv_quant_2bit',
          weight_prec=2,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='conv_signed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='conv_signed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='conv_signed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='conv_signed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
      dict(
          testcase_name='conv_signed_input_quant_2bit',
          weight_prec=None,
          acts_prec=2,
          fixed_bounds=True),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_weights_and_symmetrics_acts_should_call_clip_and_round(
      self, floor_with_gradient, round_with_gradient, weight_prec, acts_prec,
      fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.PER_TENSOR)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.SYMMETRIC,
        prec=acts_prec,
        bounds=bounds,
        half_shift=False)
    num_features = 4
    input_dim = 3
    inputs = jnp.ones((1, 32, 32, input_dim), dtype=jnp.float32)
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features,
        kernel_size,
        weight_prec=weight_prec,
        quant_act=quant_act,
        weight_half_shift=False)

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    outputs = model.apply(state, inputs)

    self.assertEqual(
        outputs.shape,
        (inputs.shape[0], inputs.shape[1], inputs.shape[2], num_features))
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_without_quantized_weights_should_not_call_quantization_ops(
      self, floor_with_gradient, round_with_gradient):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    inputs = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(inputs, 4, (3, 3))
    _ = model.apply(state, inputs)
    round_with_gradient.assert_not_called()
    floor_with_gradient.assert_not_called()


class DenseAqtTest(parameterized.TestCase):
  """Tests for DenseAqt layer."""

  def setUp(self):
    super(DenseAqtTest, self).setUp()
    test_utils.configure_jax()
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = True
    self.rng_key = random.PRNGKey(0)

  def tearDown(self):
    config.update('jax_numpy_rank_promotion', 'warn')
    super(DenseAqtTest, self).tearDown()

  def init_model_with_1_layer(self,
                              inputs,
                              num_features,
                              kernel_init=flax_layers.default_kernel_init,
                              weight_prec=None,
                              quant_act=None,
                              weight_half_shift=False,
                              kernel_axis_names=None,
                              train=False,
                              possibly_use_quantized_vars=False,
                              quant_type=QuantType.FAKE_QUANT):
    """Create and initialize a flax model with a single DenseAqt layer."""
    dynamic_context = quant_config.DynamicContext(
        update_bounds=False, collect_acts_stats=False)
    layer_kwargs = {
        'kernel_init': kernel_init,
        'features': num_features,
        'use_bias': False,
        'dynamic_context': dynamic_context,
        'paxis_name': 'batch',
        'train': train,
        'dtype': jnp.float32,
        'kernel_axis_names': kernel_axis_names,
        'possibly_use_quantized_vars': possibly_use_quantized_vars
    }
    layer_kwargs['hparams'] = flax_layers.DenseAqt.HParams(
        weight_prec=weight_prec,
        quant_act=quant_act,
        quant_type=quant_type,
        weight_quant_granularity=quant_config.QuantGranularity.PER_CHANNEL,
        weight_half_shift=weight_half_shift)

    dense_module = flax_layers.DenseAqt(**layer_kwargs)
    initial_state = dense_module.init(
        self.rng_key, jnp.zeros(inputs.shape), padding_mask=None)
    return dense_module, initial_state

  def test_padding(self):
    """Test that padding results in the right statistics being collected."""
    # Exact values don't matter here, we just need code to think it's using
    # dynamic bounds so it gathers activation statistics
    bounds = get_bounds.GetBounds.Hyper(
        initial_bound=0.0,
        stddev_coeff=1.0,
        absdev_coeff=0.0,
        mix_coeff=1.0,
        reset_stats=False,
        granularity=quant_config.QuantGranularity.PER_CHANNEL)
    quant_act = flax_layers.QuantOps.ActHParams(
        input_distribution=flax_layers.QuantOps.ActHParams.InputDistribution
        .SYMMETRIC,
        prec=8,
        bounds=bounds,
        half_shift=False)
    hparams = flax_layers.DenseAqt.HParams(
        quant_type=flax_layers.QuantType.FAKE_QUANT,
        weight_prec=8,
        quant_act=quant_act,
        weight_quant_granularity=quant_config.QuantGranularity.PER_CHANNEL,
        weight_half_shift=False)
    module = flax_layers.DenseAqt(
        hparams=hparams,
        features=1,
        paxis_name=None,
        dynamic_context=quant_config.DynamicContext(
            update_bounds=True, collect_acts_stats=False),
        train=True,
        dtype=jnp.float32)

    # Simulate an input with a batch size of 2, three tokens per example, two
    # channels per token
    x = jnp.arange(12).astype(jnp.float32).reshape((2, 3, 2))
    # Reshape it to have dimensions [batch, feature]
    x = x.reshape(6, 2)

    initial_state = module.init(self.rng_key, x, padding_mask=None)

    # Check that the per-channel activation statistics are as expected with no
    # padding
    _, state_nopadding = module.apply(
        initial_state, x, padding_mask=None, mutable='get_bounds')
    expected_means = onp.array([[(0 + 2 + 4 + 6 + 8 + 10) / 6,
                                 (1 + 3 + 5 + 7 + 9 + 11) / 6]])
    actual_means = state_nopadding['get_bounds']['GetBounds_0']['stats'].mean
    onp.testing.assert_allclose(actual_means, expected_means)

    # Now we pad out some of the tokens (chosen arbitrarily) and check that the
    # computed per-channel stats are the means of the non-padding tokens only
    # Exclude the second and third tokens from the first batch and the first
    # token from the second batch.
    padding_mask = jnp.array([[True, False, False], [False, True, True]])
    # Reshape it to have dimensions [batch, feature]
    padding_mask = padding_mask.reshape(6, 1)
    _, state_padding = module.apply(
        initial_state, x, padding_mask=padding_mask, mutable='get_bounds')
    expected_means = onp.array([[(0 + 8 + 10) / 3, (1 + 9 + 11) / 3]])
    actual_means = state_padding['get_bounds']['GetBounds_0']['stats'].mean
    onp.testing.assert_allclose(actual_means, expected_means)

  @parameterized.named_parameters(
      dict(testcase_name='dense_float', weight_prec=None),
      dict(
          testcase_name='dense_quant_fp143_scaled',
          weight_prec=fp143_scaled,
      ),
      dict(
          testcase_name='dense_quant_fp143',
          weight_prec=fp143_unscaled,
      ),
      dict(testcase_name='dense_quant_8bit', weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_ones_weights_should_give_precise_output(self, weight_prec):
    """If all weights are 1, no quantization error should be introduced."""
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        kernel_init=initializers.ones,
        weight_prec=weight_prec,
        weight_half_shift=False)
    outputs = model.apply(state, inputs, padding_mask=None)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    onp.testing.assert_array_equal(outputs, exp_outputs)

  def test_logical_axis_names(self):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    _, state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        kernel_init=initializers.ones,
        weight_prec=8,
        weight_half_shift=False,
        kernel_axis_names=('embed', 'mlp'))

    self.assertDictEqual(
        param_dtypes_shapes_axes(state['params'], state['params_axes']),
        {'kernel': ['float32', 'embed=3', 'mlp=4']})

  @parameterized.named_parameters(
      dict(testcase_name='dense_quant_8bit', weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_should_give_precise_output(
      self, weight_prec):
    # If weights are ints (already quantized) and
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel,
    # no quantization error should be introduced.
    num_features = 256
    input_dim = 1024
    inputs = random.uniform(self.rng_key, shape=(2, input_dim))
    model, state = self.init_model_with_1_layer(
        inputs, num_features, weight_prec=weight_prec, weight_half_shift=False)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(self.rng_key,
                                                (input_dim, num_features),
                                                minval, maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = full_range_integer_weights.at[0, :].set(maxval)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = full_range_integer_weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs, padding_mask=None)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    onp.testing.assert_array_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      # TODO(shivaniagrawal): this test is flaky and fails with rtol=0.0004
      # with given rtol=0.0001
      # dict(
      #     testcase_name='dense_quant_8bit',
      #     layer_class=flax_layers.DenseAqt,
      #     weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_with_float_scale_should_give_close_output(
      self, weight_prec):
    # If weights are ints (already quantized) with
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel
    # and if these integer weights are multiplied by a float scale,
    # the resulting error should still be very small (just float rounding).

    num_features = 256
    input_dim = 1024
    inputs = random.uniform(self.rng_key, shape=(2, input_dim))
    model, state = self.init_model_with_1_layer(
        inputs, num_features, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(self.rng_key,
                                                (input_dim, num_features),
                                                minval, maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = full_range_integer_weights.at[0, :].set(maxval)

    float_scale = jax.random.uniform(self.rng_key, (1, num_features))
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = full_range_integer_weights * float_scale
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs, padding_mask=None)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    # TODO(wanglisa): Determine how much noise is expected for following test.
    # We know that the noise should be proportional to the square root of
    # input_dim and inversely proportional to 2**weight_prec.
    # The following tol_const was obtained experimentally and should be derived
    # more systematically.
    tol_const = 8e-04
    onp.testing.assert_allclose(
        outputs,
        exp_outputs,
        rtol=jnp.sqrt(input_dim) * 2**(-weight_prec) * tol_const)

  @parameterized.named_parameters(
      # dict(
      #     testcase_name='dense_quant_8bit',
      #     weight_prec=8),
      # TODO(shivaniagrawal): fix the above test, test above doesn't follow
      # the expected tolerance. Expected absolute difference = 0.188386,
      # actual absolute difference: 0.20296225
      dict(
          testcase_name='dense_quant_fp143_scaled',
          weight_prec=fp143_scaled,
      ),
      dict(testcase_name='dense_quant_fp143', weight_prec=fp143_unscaled),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_float_weights_should_give_close_output(self, weight_prec):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=weight_prec)
    float_weights = jnp.linspace(-1 / 3, 1 / 3, num=12).reshape((3, 4))

    exp_output_without_quant = jnp.matmul(inputs, float_weights)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = float_weights
    state = flax.core.freeze(state)
    outputs_with_quant = model.apply(state, inputs, padding_mask=None)
    onp.testing.assert_raises(AssertionError, onp.testing.assert_array_equal,
                              outputs_with_quant, exp_output_without_quant)
    test_utils.assert_all_close_prec(exp_output_without_quant,
                                     outputs_with_quant, weight_prec)

  # TODO(wanglisa): Add tests with bigger matrices.

  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_fp143_scaled',
          weight_prec=fp143_scaled,
          weight_scale=onp.array([1, 2, 4, 8]),
      ),
      dict(
          testcase_name='dense_quant_fp143',
          weight_prec=fp143_unscaled,
          weight_scale=onp.array([1, 2, 4, 8]),
      ),
      dict(
          testcase_name='dense_quant_8bit',
          weight_prec=8,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='dense_quant_4bit',
          weight_prec=4,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='dense_quant_2bit',
          weight_prec=2,
          weight_scale=onp.array([1, 2, 4, 8])),
  )
  def test_weight_invariance_to_power_of_2_weight_scaling(
      self, weight_prec, weight_scale):
    # Scaling the weights before quantization by a power of 2 per channel should
    # also scale the output exactly by the same scale.

    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=weight_prec)
    weights = random.uniform(self.rng_key, shape=(3, 4))
    weight_scale = weight_scale[jnp.newaxis, :]
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = weights
    state = flax.core.freeze(state)
    outputs_without_scaling = model.apply(state, inputs, padding_mask=None)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = jnp.multiply(weights, weight_scale)
    state = flax.core.freeze(state)
    outputs_with_scaling = model.apply(state, inputs, padding_mask=None)

    onp.testing.assert_array_equal(outputs_without_scaling * weight_scale,
                                   outputs_with_scaling)

  def test_1_bit_makes_all_weight_equal_to_zero(self):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=1)
    weights = random.uniform(
        self.rng_key, shape=state['params']['kernel'].shape)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs, padding_mask=None)
    onp.testing.assert_array_equal(outputs, onp.zeros((2, 4)))

  # TODO(shivaniagrawal): change mock tests to check for QuantOps than
  # primitives.
  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_8bit',
          weight_prec=8,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='dense_quant_4bit',
          weight_prec=4,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='dense_quant_2bit',
          weight_prec=2,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='dense_signed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='dense_signed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='dense_signed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='dense_signed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
      dict(
          testcase_name='dense_signed_input_quant_2bit',
          weight_prec=None,
          acts_prec=2,
          fixed_bounds=True),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_weights_and_symmetrics_acts_should_call_clip_and_round(
      self, floor_with_gradient, round_with_gradient, weight_prec, acts_prec,
      fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.PER_TENSOR)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.SYMMETRIC,
        prec=acts_prec,
        bounds=bounds,
        half_shift=False)
    num_features = 4
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features,
        weight_prec=weight_prec,
        quant_act=quant_act,
        weight_half_shift=False)

    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    outputs = model.apply(state, inputs, padding_mask=None)

    self.assertEqual(outputs.shape, (inputs.shape[0], num_features))
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

  # TODO(shivaniagrawal): change mock tests to check for QuantOps than
  # primitives.
  @parameterized.named_parameters(
      dict(
          testcase_name='dense_pos_quant_8bit',
          pos_inputs_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='dense_pos_quant_4bit',
          pos_inputs_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='dense_pos_quant_8bit_auto_clip',
          pos_inputs_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='dense_pos_quant_4bit_aut_clip',
          pos_inputs_prec=4,
          fixed_bounds=False),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_inputs_should_call_clip_and_round(self,
                                                       floor_with_gradient,
                                                       round_with_gradient,
                                                       pos_inputs_prec,
                                                       fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.PER_TENSOR)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.POSITIVE,
        prec=pos_inputs_prec,
        bounds=bounds,
        half_shift=False)
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, init_state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        weight_prec=None,
        quant_act=quant_act,
        weight_half_shift=False)
    floor_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(floor_with_gradient.call_count, 1)
    round_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    model.apply(init_state, inputs, padding_mask=None)

    floor_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(floor_with_gradient.call_count, 1)
    round_with_gradient.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_fp143_scaled',
          inputs_prec=fp143_scaled,
          fixed_bounds=True,
      ),)
  @mock.patch.object(fp_cast, 'downcast_sat_ftz')
  def test_fp_quantized_inputs_should_call_downcast_sat_ftz(
      self, downcast_mock, inputs_prec, fixed_bounds):

    downcast_mock.side_effect = lambda x, *_: x
    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.PER_TENSOR)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.POSITIVE,
        prec=inputs_prec,
        bounds=bounds,
        half_shift=False)
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, init_state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        weight_prec=None,
        quant_act=quant_act,
        weight_half_shift=False)
    downcast_mock.assert_called_once_with(
        mock.ANY,
        inputs_prec.fp_spec.exp_min,
        inputs_prec.fp_spec.exp_max,
        inputs_prec.fp_spec.sig_bits,
    )
    downcast_mock.reset_mock()

    model.apply(init_state, inputs, padding_mask=None)

    downcast_mock.assert_called_once_with(
        mock.ANY,
        inputs_prec.fp_spec.exp_min,
        inputs_prec.fp_spec.exp_max,
        inputs_prec.fp_spec.sig_bits,
    )

  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_without_quantized_weights_should_not_call_quantization_ops(
      self, floor_with_gradient, round_with_gradient):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(inputs, num_features=4)
    _ = model.apply(state, inputs, padding_mask=None)
    round_with_gradient.assert_not_called()
    floor_with_gradient.assert_not_called()

  @parameterized.parameters(
      dict(granularity=quant_config.QuantGranularity.PER_CHANNEL, axis=(0,)),
      dict(granularity=quant_config.QuantGranularity.PER_TENSOR, axis=None))
  @mock.patch.object(quantization, 'quantized_dot_general')
  @mock.patch.object(shape_utils, 'assert_shapes_equal')
  def test_quant_granularity(self, _, mock_quantized_dot_general, granularity,
                             axis):
    hparams = flax_layers.DenseAqt.HParams(
        weight_prec=8,
        quant_act=None,
        quant_type=quantization.QuantType.FAKE_QUANT,
        weight_quant_granularity=granularity,
        weight_half_shift=False)
    layer = flax_layers.DenseAqt(
        features=2,
        hparams=hparams,
        dynamic_context=quant_config.DynamicContext(
            update_bounds=False, collect_acts_stats=False),
        paxis_name=None,
        train=False,
        dtype=jnp.float32)
    x = jnp.ones((2, 2))
    state = layer.init(self.rng_key, x, padding_mask=None)
    layer.apply(state, x, padding_mask=None)
    weight_params = mock_quantized_dot_general.call_args[1]['weight_params']
    self.assertEqual(weight_params.axis, axis)

  @parameterized.named_parameters(
      dict(
          testcase_name='train_quant',
          train=True,
          possibly_use_quantized_vars=True,
          param_info={
              'kernel': ['float32', 'embed=3', 'mlp=3'],
              'qkernel': ['float32', 'embed=3', 'mlp=3'],
              'qscale': ['float32', 'embed_qscale=1', 'mlp=3']
          }),
      dict(
          testcase_name='inference_quant',
          train=False,
          possibly_use_quantized_vars=True,
          param_info={
              'qkernel': ['int8', 'embed=3', 'mlp=3'],
              'qscale': ['float32', 'embed_qscale=1', 'mlp=3']
          }),
      dict(
          testcase_name='train_without_quant',
          train=True,
          possibly_use_quantized_vars=False,
          param_info={'kernel': ['float32', 'embed=3', 'mlp=3']}),
      dict(
          testcase_name='inference_without_quant',
          train=True,
          possibly_use_quantized_vars=False,
          param_info={'kernel': ['float32', 'embed=3', 'mlp=3']}),
  )
  def test_train_inference_differentiation(self, train,
                                           possibly_use_quantized_vars,
                                           param_info):
    num_features = 3
    inputs = random.uniform(self.rng_key, shape=(2, 3))

    _, state = self.init_model_with_1_layer(
        inputs,
        num_features=num_features,
        weight_prec=8,
        train=train,
        possibly_use_quantized_vars=possibly_use_quantized_vars,
        kernel_axis_names=('embed', 'mlp'),
        quant_type=QuantType.AQT)

    self.assertDictEqual(
        param_dtypes_shapes_axes(state['params'], state['params_axes']),
        param_info)


class EmbedLayerTest(parameterized.TestCase):
  """Tests for AQT Embed layer."""

  # TODO(shivaniagrawal): we are not raising error on jax rank
  # promotion. For EmbedAqt tests; in AQT style inputs and output are not be
  # of same shape; require more work to avoid rank promotion.
  @parameterized.named_parameters(
      dict(
          testcase_name='8_bit',
          weight_prec=8,
      ),
      dict(
          testcase_name='4_bit',
          weight_prec=4,
      ),
      dict(
          testcase_name='no_quantization',
          weight_prec=None,
      ),
  )
  def test_embed(self, weight_prec):
    # Since the dummy embedding matrix has a row of all zeros, we need 'epsilon'
    # to be added to it before calculating scale factors.
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = False
    rng = random.PRNGKey(0)
    x = jnp.arange(4)[None]
    dummy_embedding = jnp.broadcast_to(jnp.arange(4)[..., None],
                                       (4, 3)).astype(jnp.float32)
    embed_module = flax_layers.EmbedAqt(
        num_embeddings=4,
        features=3,
        dtype=jnp.float32,
        hparams=flax_layers.EmbedAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.FAKE_QUANT,
            weight_half_shift=False),
        embedding_init=lambda _rng, _shape: dummy_embedding,
        train=False,
        paxis_name=None,
        dynamic_context=quant_config.DynamicContext(update_bounds=False),
    )
    y, state = embed_module.init_with_output(rng, x)
    test_utils.assert_all_close_prec(dummy_embedding[None], y, weight_prec)

    z = embed_module.apply(
        state, jnp.ones((1, 3)), padding_mask=None, method=embed_module.attend)
    test_utils.assert_all_close_prec(3. * jnp.arange(4), z[0, ...], weight_prec)

  @parameterized.named_parameters(
      dict(
          testcase_name='8_bit',
          weight_prec=8,
      ),
      dict(
          testcase_name='4_bit',
          weight_prec=4,
      ),
      dict(
          testcase_name='no_quantization',
          weight_prec=None,
      ),
  )
  def test_embed_equality(self, weight_prec):
    rng = random.PRNGKey(0)
    x = 2 * jnp.ones(4, dtype=jnp.int32)[None]
    dummy_embedding = 2 * jnp.ones((4, 2)).astype(jnp.float32)
    embed_module = flax_layers.EmbedAqt(
        num_embeddings=4,
        features=2,
        dtype=jnp.float32,
        hparams=flax_layers.EmbedAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.FAKE_QUANT,
            weight_half_shift=False),
        embedding_init=lambda _rng, _shape: dummy_embedding,
        train=False,
        dynamic_context=quant_config.DynamicContext(update_bounds=False),
        paxis_name=None)
    y, init_state = embed_module.init_with_output(rng, x)
    onp.testing.assert_array_equal(dummy_embedding[None], y)

    z = embed_module.apply(
        init_state,
        jnp.ones((1, 2)),
        padding_mask=None,
        method=embed_module.attend)
    onp.testing.assert_array_equal(2. * (2 * jnp.ones(4)), z[0, ...])

  @parameterized.named_parameters(
      dict(
          testcase_name='embed_quant_8bit',
          weight_prec=8,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='embed_quant_4bit',
          weight_prec=4,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='embed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='embed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='embed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='embed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_embed_should_call_clip_and_round(self, floor_with_gradient,
                                            round_with_gradient, weight_prec,
                                            acts_prec, fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.PER_TENSOR)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.SYMMETRIC,
        prec=acts_prec,
        bounds=bounds,
        half_shift=False)
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))

    embed_module = flax_layers.EmbedAqt(
        num_embeddings=4,
        features=3,
        dtype=jnp.float32,
        hparams=flax_layers.EmbedAqt.HParams(
            weight_prec=weight_prec,
            quant_act=quant_act,
            quant_type=QuantType.FAKE_QUANT,
            weight_half_shift=False),
        dynamic_context=quant_config.DynamicContext(update_bounds=False),
        paxis_name=None,
        train=False)
    init_state = embed_module.init(
        rng, x, method=embed_module.attend, padding_mask=None)
    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()
    embed_module.apply(
        init_state, x, padding_mask=None, method=embed_module.attend)
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()


class LayerNormTest(parameterized.TestCase):

  @classmethod
  def make_hparams(cls, quantize_reductions, exp_min, exp_max, sig_bits):
    prec = QuantOps.FloatQuant.FloatPrec(
        exp_min=exp_min, exp_max=exp_max, sig_bits=sig_bits)
    reduction_prec = prec if quantize_reductions else None
    hparams = flax_layers.LayerNormAqt.HParams(
        quant_hparams=flax_layers.LayerNormAqt.QuantHParams(
            prec=prec,
            reduction_prec=reduction_prec,
        ))
    return hparams

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters(itertools.product((False, True), (False, True)))
  def test_quantized_layer_norm_matches_unquantized_in_fp32(
      self, quantize_acts, quantize_reductions):
    # We 'quantize' to a custom floating-point format that is approximately
    # equivalent to IEEE float32 and test that results are the same as using
    # Flax's upstream unquantized LayerNorm.
    hparams = self.make_hparams(
        exp_min=-2**7,
        exp_max=2**7,
        sig_bits=23,
        quantize_reductions=quantize_reductions)
    quantized_layer_norm = flax_layers.LayerNormAqt(
        hparams=hparams,
        dtype=jnp.float32,
        dynamic_context=quant_config.DynamicContext(
            update_bounds=False, quantize_acts=quantize_acts))
    x_rng, param_rng = jax.random.split(self.rng)
    x = jax.random.normal(x_rng, (3, 5))
    initial_params = quantized_layer_norm.init(param_rng, x)
    y_quantized = quantized_layer_norm.apply(initial_params, x)
    unquantized_layer_norm = nn.LayerNorm()
    y_unquantized = unquantized_layer_norm.apply(initial_params, x)
    onp.testing.assert_allclose(y_quantized, y_unquantized, rtol=2e-6)

  def test_epsilon_rounding(self):
    # We give LayerNorm a constant input. Since that input has a variance of
    # zero, we would expect layernorm to return NaN (0/0) unless the 'epsilon'
    # parameter which nudges the denominator away from zero was having an
    # effect. We test the case where the default epsilon value of 1e-6 would
    # ordinarily flush to zero after quantization with a high value of exp_min.
    # This test makes sure our code to round epsilon up to the smallest non-zero
    # representable value is wokring.
    hparams = self.make_hparams(
        exp_min=-2**2, exp_max=2**7, sig_bits=23, quantize_reductions=False)
    layer_norm = flax_layers.LayerNormAqt(
        hparams=hparams,
        use_bias=False,
        use_scale=False,
        epsilon=1e-6,
        dtype=jnp.float32,
        dynamic_context=quant_config.DynamicContext(
            update_bounds=False, quantize_acts=True))
    x = jnp.ones((2, 5))
    y = layer_norm.apply({}, x)
    onp.testing.assert_equal(onp.array(y), onp.zeros(x.shape))


def assert_same_tree(a, b):
  jax.tree_map(
      functools.partial(onp.testing.assert_allclose, atol=1e-6, rtol=1e-6), a,
      b)


class DenseGeneralAqtTest(parameterized.TestCase):
  """Tests for DenseGeneralAqt layer."""

  def setUp(self):
    super(DenseGeneralAqtTest, self).setUp()
    test_utils.configure_jax()
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = True
    self.rng_key = random.PRNGKey(0)

  def tearDown(self):
    config.update('jax_numpy_rank_promotion', 'warn')
    super(DenseGeneralAqtTest, self).tearDown()

  def init_model_with_1_layer(self,
                              inputs,
                              num_features,
                              axis=-1,
                              kernel_init=flax_layers.default_kernel_init,
                              weight_prec=None,
                              weight_half_shift=False,
                              quant_act=None,
                              train=False,
                              kernel_axis_names=None,
                              reshape_kernel=True,
                              possibly_use_quantized_vars=False):
    """Create and initialize a flax model with a single DenseAqt layer."""
    layer_kwargs = {
        'kernel_init': kernel_init,
        'features': num_features,
        'axis': axis,
        'use_bias': False,
        'train': train,
        'dtype': jnp.float32,
        'kernel_axis_names': kernel_axis_names,
        'reshape_kernel': reshape_kernel,
        'possibly_use_quantized_vars': possibly_use_quantized_vars,
    }
    layer_kwargs['hparams'] = flax_layers.DenseGeneralAqt.HParams(
        weight_prec=weight_prec,
        quant_act=quant_act,
        weight_quant_granularity=quant_config.QuantGranularity.PER_CHANNEL,
        weight_half_shift=weight_half_shift)

    dense_module = flax_layers.DenseGeneralAqt(**layer_kwargs)
    initial_state = dense_module.init(self.rng_key, jnp.zeros(inputs.shape))
    return dense_module, initial_state

  # DenseGeneral does not support fq version, hence fp quantization isn't
  # supported either
  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),

      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
      dict(testcase_name='quant_4bit_no_reshape', weight_prec=4, reshape=False),
      dict(testcase_name='quant_2bit', weight_prec=2),
  )
  def test_ones_weights_should_give_precise_output(self,
                                                   weight_prec,
                                                   reshape=True):
    """If all weights are 1, no quantization error should be introduced."""
    inputs = random.uniform(self.rng_key, shape=(2, 3, 4))
    axis = (1, 2)
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features=5,
        axis=axis,
        kernel_init=initializers.ones,
        weight_prec=weight_prec,
        weight_half_shift=False,
        reshape_kernel=reshape)
    outputs = model.apply(state, inputs)
    contract_ind = tuple(range(0, len(axis)))
    exp_outputs = lax.dot_general(
        inputs,
        jnp.reshape(state['params']['kernel'], (3, 4, 5)),
        dimension_numbers=((axis, contract_ind), ((), ())))
    onp.testing.assert_array_almost_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      dict(testcase_name='reshape_kernel', reshape_kernel=True),
      dict(testcase_name='no_reshape', reshape_kernel=False),
  )
  def test_logical_axis_names(self, reshape_kernel):
    inputs = random.uniform(self.rng_key, shape=(2, 3, 1, 4))
    _, state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        axis=(2, 3),
        kernel_init=initializers.ones,
        weight_prec=8,
        weight_half_shift=False,
        kernel_axis_names=('embed', 'head', 'mlp'),
        reshape_kernel=reshape_kernel)

    expected_shape_axes = [
        'float32', 'embed * head=4', 'mlp=4'
    ] if reshape_kernel else ['float32', 'embed=1', 'head=4', 'mlp=4']

    self.assertDictEqual(
        param_dtypes_shapes_axes(state['params'], state['params_axes']),
        {'kernel': expected_shape_axes})

  @parameterized.named_parameters(
      dict(testcase_name='dense_quant_8bit', weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_should_give_precise_output(
      self, weight_prec):
    # If weights are ints (already quantized) and
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel,
    # no quantization error should be introduced.
    num_features = 256
    input_dim = 1024
    inputs = random.uniform(self.rng_key, shape=(2, input_dim))
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features,
        axis=(1,),
        weight_prec=weight_prec,
        weight_half_shift=False)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(self.rng_key,
                                                (input_dim, num_features),
                                                minval, maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = full_range_integer_weights.at[0, :].set(maxval)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = full_range_integer_weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    onp.testing.assert_array_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      dict(
          testcase_name='densegeneral_quant_8bit',
          weight_prec=8,
          expected_output=[[-0.08770747, -0.00193817, 0.08321345, 0.16902271],
                           [-0.2838632, -0.1654416, -0.04855134, 0.06976502]]),
      dict(
          testcase_name='densegeneral_quant_4bit',
          weight_prec=4,
          expected_output=[[-0.09499891, -0.016027, 0.09432255, 0.17474438],
                           [-0.29012504, -0.17881386, -0.02733358,
                            0.08016007]]),
      dict(
          testcase_name='densegeneral_quant_2bit',
          weight_prec=2,
          expected_output=[[-0.19150016, 0.0456724, 0.0456724, 0.24732196],
                           [-0.29805943, -0.12025267, -0.12025267,
                            0.15108395]]),
  )
  def test_float_weights_regression_test(self, weight_prec, expected_output):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=weight_prec)
    float_weights = jnp.linspace(-1 / 3, 1 / 3, num=12).reshape((3, 4))

    exp_output_without_quant = jnp.matmul(inputs, float_weights)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = float_weights
    state = flax.core.freeze(state)
    outputs_with_quant = model.apply(state, inputs)
    onp.testing.assert_raises(AssertionError, onp.testing.assert_array_equal,
                              outputs_with_quant, exp_output_without_quant)
    onp.testing.assert_allclose(outputs_with_quant, expected_output, rtol=1e-5)
    onp.testing.assert_allclose(
        exp_output_without_quant,
        [[-0.08814345, -0.00231601, 0.0835114, 0.16933881],
         [-0.2836182, -0.16580023, -0.04798227, 0.06983571]],
        rtol=1e-5)
    test_utils.assert_all_close_prec(exp_output_without_quant,
                                     outputs_with_quant, weight_prec)

  @parameterized.named_parameters(
      # TODO(shivaniagrawal): this test is flaky and fails with rtol=0.0004
      # with given rtol=0.0001
      # dict(
      #     testcase_name='dense_quant_8bit',
      #     layer_class=flax_layers.DenseAqt,
      #     weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_with_float_scale_should_give_close_output(
      self, weight_prec):
    # If weights are ints (already quantized) with
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel
    # and if these integer weights are multiplied by a float scale,
    # the resulting error should still be very small (just float rounding).

    num_features = 256
    input_dim = 1024
    inputs = random.uniform(self.rng_key, shape=(2, input_dim))
    model, state = self.init_model_with_1_layer(
        inputs, num_features, axis=(1,), weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(self.rng_key,
                                                (input_dim, num_features),
                                                minval, maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = full_range_integer_weights.at[0, :].set(maxval)

    float_scale = jax.random.uniform(self.rng_key, (1, num_features))
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = full_range_integer_weights * float_scale
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    # TODO(wanglisa): Determine how much noise is expected for following test.
    # We know that the noise should be proportional to the square root of
    # input_dim and inversely proportional to 2**weight_prec.
    # The following tol_const was obtained experimentally and should be derived
    # more systematically.
    tol_const = 8e-04
    onp.testing.assert_allclose(
        outputs,
        exp_outputs,
        rtol=jnp.sqrt(input_dim) * 2**(-weight_prec) * tol_const)

  @parameterized.named_parameters(
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_float_weights_should_give_close_output(self, weight_prec):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=weight_prec)
    float_weights = jnp.linspace(-1 / 3, 1 / 3, num=12).reshape((3, 4))

    exp_output_without_quant = jnp.matmul(inputs, float_weights)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = float_weights
    state = flax.core.freeze(state)
    outputs_with_quant = model.apply(state, inputs)
    onp.testing.assert_raises(AssertionError, onp.testing.assert_array_equal,
                              outputs_with_quant, exp_output_without_quant)

    test_utils.assert_all_close_prec(exp_output_without_quant,
                                     outputs_with_quant, weight_prec)

  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_8bit',
          weight_prec=8,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='dense_quant_4bit',
          weight_prec=4,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='dense_quant_2bit',
          weight_prec=2,
          weight_scale=onp.array([1, 2, 4, 8])),
  )
  def test_weight_invariance_to_power_of_2_weight_scaling(
      self, weight_prec, weight_scale):
    # Scaling the weights before quantization by a power of 2 per channel should
    # also scale the output exactly by the same scale.

    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, axis=(1,), weight_prec=weight_prec)
    weights = random.uniform(self.rng_key, shape=(3, 4))
    weight_scale = weight_scale[jnp.newaxis, :]
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = weights
    state = flax.core.freeze(state)
    outputs_without_scaling = model.apply(state, inputs)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = jnp.multiply(weights, weight_scale)
    state = flax.core.freeze(state)
    outputs_with_scaling = model.apply(state, inputs)

    onp.testing.assert_array_equal(outputs_without_scaling * weight_scale,
                                   outputs_with_scaling)

  def test_1_bit_makes_all_weight_equal_to_zero(self):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, axis=(1,), weight_prec=1)
    weights = random.uniform(
        self.rng_key, shape=state['params']['kernel'].shape)
    state = flax.core.unfreeze(state)
    state['params']['kernel'] = weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)
    onp.testing.assert_array_equal(outputs, onp.zeros((2, 4)))

  # TODO(shivaniagrawal): change mock tests to check for QuantOps than
  # primitives.
  @parameterized.named_parameters(
      dict(
          testcase_name='dense_weight_quant_8bit',
          weight_prec=8,
          acts_prec=None),
      dict(
          testcase_name='dense_weight_quant_4bit',
          weight_prec=4,
          acts_prec=None),
      dict(
          testcase_name='dense_weight_quant_2bit',
          weight_prec=2,
          acts_prec=None),
      dict(
          testcase_name='dense_signed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8),
      dict(
          testcase_name='dense_signed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4),
      dict(
          testcase_name='dense_signed_input_quant_2bit',
          weight_prec=None,
          acts_prec=2),
      dict(
          testcase_name='dense_signed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='dense_signed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_weights_and_symmetrics_acts_should_call_clip_and_round(
      self,
      floor_with_gradient,
      round_with_gradient,
      weight_prec,
      acts_prec,
      fixed_bounds=False):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.DynamicBounds.Hyper(
          granularity=quant_config.QuantGranularity.PER_CHANNEL)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.SYMMETRIC,
        prec=acts_prec,
        bounds=bounds,
        half_shift=False)
    num_features = 4
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features,
        weight_prec=weight_prec,
        weight_half_shift=False,
        quant_act=quant_act,
    )

    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    outputs = model.apply(state, inputs)

    self.assertEqual(outputs.shape, (inputs.shape[0], num_features))
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_without_quantized_weights_should_not_call_quantization_ops(
      self, floor_with_gradient, round_with_gradient):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(inputs, num_features=4)
    _ = model.apply(state, inputs)
    round_with_gradient.assert_not_called()
    floor_with_gradient.assert_not_called()

  # TODO(shivaniagrawal): change mock tests to check for QuantOps than
  # primitives.
  @parameterized.named_parameters(
      dict(
          testcase_name='dense_pos_quant_8bit',
          pos_inputs_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='dense_pos_quant_4bit',
          pos_inputs_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='dense_pos_quant_8bit_auto_clip',
          pos_inputs_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='dense_pos_quant_4bit_auto_clip',
          pos_inputs_prec=4,
          fixed_bounds=False),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_inputs_should_call_clip_and_round(self,
                                                       floor_with_gradient,
                                                       round_with_gradient,
                                                       pos_inputs_prec,
                                                       fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.DynamicBounds.Hyper(
          granularity=quant_config.QuantGranularity.PER_CHANNEL)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.POSITIVE,
        prec=pos_inputs_prec,
        bounds=bounds,
        half_shift=False)
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, init_state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        weight_prec=None,
        quant_act=quant_act,
        weight_half_shift=False)
    floor_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(floor_with_gradient.call_count, 1)
    round_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    model.apply(init_state, inputs)

    floor_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(floor_with_gradient.call_count, 1)
    round_with_gradient.assert_not_called()

  @parameterized.parameters(
      dict(granularity=quant_config.QuantGranularity.PER_CHANNEL, axis=(0, 1)),
      dict(granularity=quant_config.QuantGranularity.PER_TENSOR, axis=None))
  @mock.patch.object(quantization, 'flaxformer_dot_general')
  @mock.patch.object(shape_utils, 'assert_shapes_equal')
  def test_weights_quant_granularity(self, _, mock_flaxformer_dot_general,
                                     granularity, axis):
    hparams = flax_layers.DenseGeneralAqt.HParams(
        weight_prec=8,
        quant_act=None,
        weight_quant_granularity=granularity,
        weight_half_shift=False)
    layer = flax_layers.DenseGeneralAqt(
        features=3,
        axis=(1, 2),
        hparams=hparams,
        train=False,
        use_bias=False,
        dtype=jnp.float32)
    x = jnp.ones((2, 3, 4))
    state = layer.init(self.rng_key, x)
    layer.apply(state, x)
    weight_params = mock_flaxformer_dot_general.call_args[1]['weight_params']
    self.assertEqual(weight_params.axis, axis)

  @parameterized.parameters(
      dict(
          granularity=quant_config.QuantGranularity.PER_CHANNEL,
          quant_axis=(1, 2)),
      dict(
          granularity=quant_config.QuantGranularity.PER_TENSOR,
          quant_axis=None))
  @mock.patch.object(quantization, 'flaxformer_dot_general')
  @mock.patch.object(shape_utils, 'assert_shapes_equal')
  def test_acts_quant_granularity(self, _, mock_flaxformer_dot_general,
                                  granularity, quant_axis):
    bounds = get_bounds.DynamicBounds.Hyper(granularity=granularity)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.POSITIVE,
        prec=8,
        bounds=bounds,
        half_shift=False)
    hparams = flax_layers.DenseGeneralAqt.HParams(
        weight_prec=None,
        quant_act=quant_act,
        weight_quant_granularity=quant_config.QuantGranularity.PER_TENSOR,
        weight_half_shift=False)
    layer = flax_layers.DenseGeneralAqt(
        features=2,
        axis=(1, 2),
        hparams=hparams,
        train=False,
        use_bias=False,
        dtype=jnp.float32)
    x = jnp.ones((2, 3, 4))
    state = layer.init(self.rng_key, x)
    layer.apply(state, x)
    bounds_params = mock_flaxformer_dot_general.call_args[1]['bounds_params']
    self.assertEqual(bounds_params.quant_axis, quant_axis)

  @parameterized.named_parameters(
      dict(
          testcase_name='train_quant',
          train=True,
          possibly_use_quantized_vars=True,
          param_info={
              'kernel': ['float32', 'embed=3', 'mlp=3'],
              'qkernel': ['float32', 'embed=3', 'mlp=3'],
              'qscale': ['float32', 'embed_qscale=1', 'mlp=3']
          }),
      dict(
          testcase_name='inference_quant',
          train=False,
          possibly_use_quantized_vars=True,
          param_info={
              'qkernel': ['int8', 'embed=3', 'mlp=3'],
              'qscale': ['float32', 'embed_qscale=1', 'mlp=3']
          }),
      dict(
          testcase_name='train_without_quant',
          train=True,
          possibly_use_quantized_vars=False,
          param_info={'kernel': ['float32', 'embed=3', 'mlp=3']}),
      dict(
          testcase_name='inference_without_quant',
          train=True,
          possibly_use_quantized_vars=False,
          param_info={'kernel': ['float32', 'embed=3', 'mlp=3']}),
  )
  def test_train_inference_differentiation(self, train,
                                           possibly_use_quantized_vars,
                                           param_info):
    num_features = 3
    inputs = random.uniform(self.rng_key, shape=(2, 3))

    _, state = self.init_model_with_1_layer(
        inputs,
        num_features=num_features,
        weight_prec=8,
        train=train,
        possibly_use_quantized_vars=possibly_use_quantized_vars,
        kernel_axis_names=('embed', 'mlp'))

    self.assertDictEqual(
        param_dtypes_shapes_axes(state['params'], state['params_axes']),
        param_info)


class DenseGeneralTest(parameterized.TestCase):

  # pylint: disable=unused-argument
  def _mock_initializer(self, key, shape, dtype=jnp.float_, val=1.0):  # pylint: disable=g-unreachable-test-method
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * val
  # pylint: enable=unused-argument

  def setUp(self):
    super(DenseGeneralTest, self).setUp()
    self.hparams = flax_layers.DenseGeneralAqt.HParams(
        weight_prec=None,
        quant_act=None,
        weight_quant_granularity=quant_config.QuantGranularity.PER_CHANNEL,
        weight_half_shift=False)

  def test_dense_general_no_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = flax_layers.DenseGeneralAqt(
        hparams=self.hparams,
        train=False,
        features=4,
        use_bias=False,
        kernel_init=initializers.ones,
    )
    y, _ = model.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    onp.testing.assert_allclose(y, onp.full((1, 4), 3.))

  def test_dense_general_with_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = flax_layers.DenseGeneralAqt(
        hparams=self.hparams,
        train=False,
        features=4,
        use_bias=True,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = model.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    onp.testing.assert_allclose(y, onp.full((1, 4), 4.))

  def test_dense_general_two_features(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    model = flax_layers.DenseGeneralAqt(
        hparams=self.hparams,
        train=False,
        features=(2, 2),
        use_bias=False,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        kernel_axis_names=('a', 'b', 'c'),
    )
    y, variables = model.init_with_output(rng, x)
    # We transform the last input dimension to two output dimensions (2, 2).
    onp.testing.assert_allclose(y, onp.full((1, 2, 2), 3.))


  def test_dense_general_two_axes(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 2))
    model = flax_layers.DenseGeneralAqt(
        hparams=self.hparams,
        train=False,
        features=3,
        use_bias=False,
        axis=(-2, 2),  # Note: this is the same as (1, 2).
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        kernel_axis_names=('a', 'b', 'c'),
    )
    y, variables = model.init_with_output(rng, x)
    # We transform the last two input dimensions (2, 2) to one output dimension.
    onp.testing.assert_allclose(y, onp.full((1, 3), 4.))



if __name__ == '__main__':
  absltest.main()
