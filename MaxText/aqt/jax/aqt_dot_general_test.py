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

"""Tests for dot_general."""
import copy
from typing import Any, Dict, Iterable, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.jax import aqt_ops
from aqt.jax import aqt_tensor
from aqt.test import aqt_test_shared_base
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


# pylint: disable=g-long-lambda


def test_stats_config(share_stats_axes):
  """Config base for all the experiments."""
  return aqt_config.StatsConfig(
      ema_update_count=1,  # single update overwrites the stats
      update_count_prior=0,  # easier equations
      share_stats_axes=share_stats_axes,  # one stat per whole tensor
      tpu_cross_replica_sum=False,  # no cross-tpu reduction
      filter_zeros=False,  # on default zeros are a number like any other
  )


def calibration_config(const_coeff: float) -> aqt_config.CalibrationConfig:
  return aqt_config.CalibrationConfig(const_bound_coeff=const_coeff)


def config_from_schedule(schedule, share_stats_axes):
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
  return aqt_config.AqtScheduleConfig(
      test_stats_config(share_stats_axes), tensor_configs)


class DotModule(nn.Module):
  lhs_config: Optional[aqt_config.AqtScheduleConfig]
  rhs_config: Optional[aqt_config.AqtScheduleConfig]
  lhs_shape: Iterable[int]
  rhs_shape: Iterable[int]

  @nn.compact
  def __call__(self):
    lhs_quantizer = aqt_tensor.TensorQuantizer(list(self.lhs_shape),
                                               self.lhs_config)
    rhs_quantizer = aqt_tensor.TensorQuantizer(list(self.rhs_shape),
                                               self.rhs_config)
    return lambda lhs, rhs: aqt_ops.aqt_dot(
        lhs,
        rhs,
        lhs_quantizer,
        rhs_quantizer)


class DotGeneralModule(nn.Module):
  lhs_config: Optional[aqt_config.AqtScheduleConfig]
  rhs_config: Optional[aqt_config.AqtScheduleConfig]
  lhs_shape: Iterable[int]
  rhs_shape: Iterable[int]

  @nn.compact
  def __call__(self, lhs, rhs, train=True):
    lhs_quantizer = aqt_tensor.TensorQuantizer(list(self.lhs_shape),
                                               self.lhs_config)
    rhs_quantizer = aqt_tensor.TensorQuantizer(list(self.rhs_shape),
                                               self.rhs_config)

    lhs_quantizer.update(lhs, None, 0)  # pytype: disable=wrong-arg-types  # jax-ndarray
    rhs_quantizer.update(rhs, None, 0)  # pytype: disable=wrong-arg-types  # jax-ndarray

    return lambda lhs, rhs, dimension_numbers: aqt_ops.aqt_dot_general(
        lhs,
        rhs,
        lhs_quantizer,
        rhs_quantizer,
        dimension_numbers=dimension_numbers,
        train=train)


def _generate_dimension_numbers() -> Sequence[Dict[str, Any]]:
  """Generates arbitrary dimension numbers for a tensor of shape (2, 2, 2)."""
  keys = ["testcase_name", "dimension_numbers"]
  # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
  # rhs_batch_dims))
  cases = [
      ("batch_matmul", (((2,), (1,)), ((0,), (0,)))),
      ("one_cont_two_batch_dims", (((2,), (2,)), ((0, 1,), (0, 1,)))),
      ("two_cont_one_batch_dims", (((1, 2), (1, 2)), ((0,), (0,)))),
      ("one_contracting_dims", (((2,), (1,)), ((), ()))),
      ("two_contracting_dims", (((1, 2), (1, 2)), ((), ()))),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


class AqtDotGeneralTest(parameterized.TestCase):

  def vgrad(self, f, lhs, rhs):
    y, vjp_fn = jax.vjp(f, lhs, rhs)
    return vjp_fn(jnp.ones(y.shape))

  def get_dot_module(self, lhs_config, rhs_config, lhs_shape, rhs_shape):
    module = DotModule(lhs_config, rhs_config, lhs_shape, rhs_shape)
    return module.init_with_output(jax.random.PRNGKey(0))

  def get_dot_general_module(self, lhs_config, rhs_config, lhs, rhs):
    module = DotGeneralModule(lhs_config, rhs_config, lhs.shape, rhs.shape)
    return module.init_with_output(jax.random.PRNGKey(0), lhs, rhs)

  def exact_int8_dot_general_example(self,
                                     lhs_use_quantized_variable=False,
                                     rhs_use_quantized_variable=False):
    lhs_config, lhs, rhs_config, rhs = aqt_test_shared_base.exact_int8_example(
        lhs_shape=(2, 2, 2),
        rhs_shape=(2, 2, 2),
        lhs_share_stats_axes=[0, 1, 2],
        rhs_share_stats_axes=[0, 1, 2],
        lhs_use_quantized_variable=lhs_use_quantized_variable,
        rhs_use_quantized_variable=rhs_use_quantized_variable)
    return lhs_config, jnp.array(lhs), rhs_config, jnp.array(rhs)

  def test_dot_none(self):
    no_quant_config = aqt_config.AqtScheduleConfig(
        test_stats_config([0, 1]), [])
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float32)
    dot, _ = self.get_dot_module(no_quant_config, no_quant_config, lhs.shape,
                                 rhs.shape)
    actual_ret = dot(lhs, rhs)
    expected_ret = lax.dot(lhs, rhs)
    np.testing.assert_array_equal(actual_ret, expected_ret)

  def test_dot_incompatible_shapes(self):
    config = aqt_config.AqtScheduleConfig(
        test_stats_config([0, 1]), [])
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, size=(2, 4)).astype(np.float32)
    dot, _ = self.get_dot_module(config, config, lhs.shape, rhs.shape)
    # the last dimension of lhs and the first dimension of rhs must be the same.
    with self.assertRaisesRegex(TypeError, "Incompatible shapes for dot"):
      dot(lhs, rhs)

    lhs_rank3 = np.random.uniform(-1.0, 1.0, size=(2, 3, 2)).astype(np.float32)
    dot, _ = self.get_dot_module(config, config, lhs_rank3.shape, rhs.shape)
    # lhs and rhs must be rank1 or rank2.
    with self.assertRaisesRegex(TypeError, "Incompatible shapes for dot"):
      dot(lhs_rank3, rhs)

    rhs_rank3 = np.random.uniform(-1.0, 1.0, size=(3, 4, 5)).astype(np.float32)
    dot, _ = self.get_dot_module(config, config, lhs.shape, rhs_rank3.shape)
    # lhs and rhs must be rank1 or rank2.
    with self.assertRaisesRegex(TypeError, "Incompatible shapes for dot"):
      dot(lhs, rhs_rank3)

  @parameterized.named_parameters(_generate_dimension_numbers())
  def test_dot_general_none(self, dimension_numbers):
    no_quant_config = aqt_config.AqtScheduleConfig(
        test_stats_config([0, 1, 2]), [])
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)

    # Creates a dot_general module with empty tensor configs.
    dot_general, _ = self.get_dot_general_module(no_quant_config,
                                                 no_quant_config, lhs, rhs)
    actual_ret = dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
    expected_ret = lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers)
    np.testing.assert_array_equal(actual_ret, expected_ret)

    # Creates a dot_general module with None schedule configs.
    dot_general, _ = self.get_dot_general_module(None, None, lhs, rhs)
    actual_ret = dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
    np.testing.assert_array_equal(actual_ret, expected_ret)

  @parameterized.named_parameters(_generate_dimension_numbers())
  def test_validates_contraction(self, dimension_numbers):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_dot_general_example()
    bad_rhs_config = copy.deepcopy(rhs_config)
    bad_rhs_config.stats_config.share_stats_axes = [0]
    dot_general, _ = self.get_dot_general_module(lhs_config, bad_rhs_config,
                                                 lhs, rhs)
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected rhs dot_general contraction axis"):
      dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
    bad_lhs_config = copy.deepcopy(lhs_config)
    bad_lhs_config.stats_config.share_stats_axes = [0]
    dot_general, _ = self.get_dot_general_module(bad_lhs_config, rhs_config,
                                                 lhs, rhs)
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected lhs dot_general contraction axis"):
      dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

  @parameterized.named_parameters(
      aqt_test_shared_base.generate_unaligned_schedule_intervals())
  def test_unaligned_schedule_intervals(self, lhs_intervals, rhs_intervals):
    bits = 8
    lhs_intervals = [(start, stop, bits) for start, stop in lhs_intervals]
    rhs_intervals = [(start, stop, bits) for start, stop in rhs_intervals]
    lhs_config = config_from_schedule(lhs_intervals, [0, 1])
    rhs_config = config_from_schedule(rhs_intervals, [0, 1])
    lhs = rhs = np.ones((1, 1)).astype(np.float32)
    lhs = jnp.array(lhs)
    rhs = jnp.array(lhs)
    dot_general, _ = self.get_dot_general_module(lhs_config, rhs_config, lhs,
                                                 rhs)
    dimension_numbers = (((1,), (0,)), ((), ()))
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "intervals do not match|config len"):
      dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

  def get_dot_general_for_grad(self, lhs_config, rhs_config, lhs, rhs,
                               dimension_numbers):
    # jax.vjp does not take a tuple of primals while `dimension_numbers` is in a
    # form of tuple. Thus, this method returns a lambda with `dimension_numbers`
    # argument which will be called for jax.vjp to get gradients.
    dot_general, _ = self.get_dot_general_module(lhs_config, rhs_config, lhs,
                                                 rhs)
    return lambda lhs, rhs: dot_general(lhs, rhs, dimension_numbers)

  @parameterized.named_parameters(_generate_dimension_numbers())
  def test_vars_dont_kill_grads(self, dimension_numbers):
    lhs_config_novar, lhs, rhs_config_novar, rhs = self.exact_int8_dot_general_example(
    )
    lhs_config, _, rhs_config, _ = self.exact_int8_dot_general_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True)

    unsaved_dot_general = self.get_dot_general_for_grad(lhs_config_novar,
                                                        rhs_config_novar, lhs,
                                                        rhs, dimension_numbers)
    saved_dot_general = self.get_dot_general_for_grad(lhs_config, rhs_config,
                                                      lhs, rhs,
                                                      dimension_numbers)

    saved_grads = self.vgrad(saved_dot_general, lhs, rhs)
    unsaved_grads = self.vgrad(unsaved_dot_general, lhs, rhs)
    zipped_grads = zip(saved_grads, unsaved_grads)
    for actual_grad, expected_grad in zipped_grads:
      np.testing.assert_array_equal(actual_grad, expected_grad)

  @parameterized.named_parameters(_generate_dimension_numbers())
  def test_vars_over_inputs_at_inference(self, dimension_numbers):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_dot_general_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True)
    module = DotGeneralModule(lhs_config, rhs_config, lhs.shape, rhs.shape)

    # Since quantized variables are not used at training, the dot_general with
    # zero inputs should produce zero values.
    dot_general, state = module.init_with_output(
        jax.random.PRNGKey(0), lhs, rhs)
    actual = dot_general(
        jnp.zeros_like(lhs), jnp.zeros_like(rhs), dimension_numbers)
    expected = jnp.zeros_like(actual)
    np.testing.assert_array_equal(actual, expected)

    # Since quantized variables should be always used at inference, the
    # dot_general will rely on quantized variables.
    dot_general_infer, _ = module.apply(state, lhs, rhs, False, mutable=True)
    actual = dot_general_infer(
        jnp.zeros_like(lhs), jnp.zeros_like(rhs), dimension_numbers)
    expected = dot_general(lhs, rhs, dimension_numbers)
    np.testing.assert_array_equal(actual, expected)

  @parameterized.named_parameters(_generate_dimension_numbers())
  def test_exact_grads(self, dimension_numbers):
    lhs_config, lhs, rhs_config, rhs = self.exact_int8_dot_general_example()

    aqt_dot_general = self.get_dot_general_for_grad(lhs_config, rhs_config, lhs,
                                                    rhs, dimension_numbers)

    aqt_dg = lambda lhs, rhs: jnp.sum(aqt_dot_general(lhs, rhs)**2)
    aqt_dg_grads = self.vgrad(aqt_dg, lhs, rhs)

    dg = lambda lhs, rhs: jnp.sum(lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers)**2)
    dg_grads = self.vgrad(dg, lhs, rhs)
    for aqt_grad, jax_grad in zip(aqt_dg_grads, dg_grads):
      actual = aqt_grad
      expected = jax_grad
      np.testing.assert_array_equal(actual, expected)

  def test_weight_only_quantization(self):
    lhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    lhs_config = config_from_schedule([(None, None, 8)], [0, 1, 2])
    rhs_config = config_from_schedule([(None, None, 8)], [0, 1, 2])
    lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    # dimension numbers for batch matmul
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))

    dot_general, _ = self.get_dot_general_module(lhs_config, rhs_config, lhs,
                                                 rhs)
    result_with_float_config = dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers)
    dot_general, _ = self.get_dot_general_module(None, rhs_config, lhs, rhs)
    result_with_none_config = dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers)
    np.testing.assert_array_equal(result_with_float_config,
                                  result_with_none_config)


if __name__ == "__main__":
  absltest.main()
