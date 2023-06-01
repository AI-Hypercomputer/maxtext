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
"""Accurate Quantized Training ops."""

from __future__ import annotations

from typing import List, Optional, Tuple

from aqt.common import aqt_common
from aqt.common import aqt_config
from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp


def pass_through(x: jnp.ndarray, fn) -> jnp.ndarray:
  # Create an exactly-zero expression with Sterbenz lemma that has an
  # exactly-one gradient.
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(fn(x))


class Stats(nn.Module):
  """Manages efficient gatherting of running statistics."""

  # This type annotation is what the constructor expects.
  # However nn.Module silently converts it into immutable Tuple, so
  # the methods will observe Tuple[Optional[int], ...]
  data_shape: List[Optional[int]] = struct.field(pytree_node=False)
  config: aqt_config.StatsConfig = struct.field(pytree_node=False)

  def setup(self):
    """Constructor to init a Stats instance.

    Returns:
      A new instance of Stats, with initialized statistics.
    """
    self.config.validate(self.data_shape)
    # TODO(jihwanlee): Move the check below in the config validator.
    if self.config.lp_order > 30:
      raise NotImplementedError('For higher norms we should add stabilization.')
    stats_shape = list(self.data_shape)  # nn.Module converts lists to tuples.
    for axis in self.config.share_stats_axes:
      stats_shape[axis] = 1
    self.stats_shape = stats_shape

    def init_var(init_val: float) -> jnp.ndarray:
      return jnp.full(self.stats_shape, init_val, dtype=jnp.float32)

    def mk_var(name: str, init_val: float) -> nn.Variable:
      return self.variable('aqt', name, init_var, init_val)

    self.ema_update_count = self.config.ema_update_count
    self.sum_of_ones = mk_var('sum_of_ones', self.config.update_count_prior)
    self.sum_of_vals = mk_var(
        'sum_of_vals', self.config.mean_prior * self.config.update_count_prior)
    self.max_of_abs_vals = mk_var('max_of_abs_vals', self.config.max_dev_prior)
    self.sum_of_l1_vals = mk_var(
        'sum_of_l1_vals',
        self.config.l1_dev_prior * self.config.update_count_prior)
    self.sum_of_lp_vals = mk_var(
        'sum_of_lp_vals', self.config.lp_dev_prior**self.config.lp_order *
        self.config.update_count_prior)

  def __call__(self):
    # Keep __call__() just not to require users to explicitly specify which
    # method should be called when invoking init().
    pass

  # TODO(lew): Remvoe override_ema_update_count.
  #   override_ema_update_count is used only in tests and is a hack.
  #   However replacing it with  self.ema_update_count does not work,
  #   because nn.Model will not allow to access it from outside.
  #   I'm not sure how this should be solved cleanly.
  def update(self,
             x: jnp.ndarray,
             weight: Optional[jnp.ndarray],
             override_ema_update_count=None):
    """Returns a new Stats object with updated statistics.

    Since flax.struct.dataclass objects are frozen, thie update method uses
    the dataclass.replace method to get a new object replacing the existing
    statistics with new ones and returns it.

    Args:
      x: input array to update the current statistics with.
      weight: weight for the input array x.
      override_ema_update_count: use this value instead of the one in config.

    Returns:
      Anew object of Stats with updated statistics.
    """
    aqt_common.check_shapes_conformal(x.shape, self.data_shape)
    if weight is not None and len(x.shape) != len(weight.shape):
      raise ValueError(
          f'expected rank(x)={len(x.shape)} == rank(weight)={weight.shape}')

    def update_var(var, s, reduce_max=False):
      assert len(s.shape) == len(self.data_shape), (s.shape, self.data_shape)
      assert s.size > 0
      if weight is not None:
        if reduce_max:
          # A maximum is a maximum, regardless of its nonzero weight.
          s = jnp.where(weight > 0, s, 0)
        else:
          s = s * weight
      reduce_fn = jnp.max if reduce_max else jnp.sum
      s = reduce_fn(s, axis=self.config.share_stats_axes, keepdims=True)
      if self.config.tpu_cross_replica_sum:
        raise NotImplementedError(
            'support for tpu_cross_replica_sum=True is not implemented')
      override = override_ema_update_count
      rate = 1.0 / (override if override is not None else self.ema_update_count)
      var.value = (1.0 - rate) * var.value + rate * s

    # Layers such as ReLU emit zeros often. In such cases, we can model
    # the non-sparse distribution of weights separately, resulting in
    # unbiased estimation of non-sparse mean l1 and lp.
    # This clips away less of the distribution of inputs.
    if self.config.filter_zeros:
      ones = jnp.where(x != 0, 1., 0.)
    else:
      ones = jnp.ones_like(x)

    px = x if self.config.lp_order % 2 == 0 else jnp.abs(x)

    update_var(self.sum_of_ones, ones)
    update_var(self.sum_of_vals, x)
    update_var(self.max_of_abs_vals, jnp.abs(x), reduce_max=True)
    update_var(self.sum_of_l1_vals, jnp.abs(x))
    update_var(self.sum_of_lp_vals, px**self.config.lp_order)

  def mean(self) -> jnp.ndarray:
    return self.sum_of_vals.value / self.sum_of_ones.value

  def max_dev(self) -> jnp.ndarray:
    return self.max_of_abs_vals.value

  def l1_dev(self) -> jnp.ndarray:
    return self.sum_of_l1_vals.value / self.sum_of_ones.value

  def lp_dev(self) -> jnp.ndarray:
    if self.config.lp_order == 2:
      # sqrt() is numerically more accurate
      return jnp.sqrt(self.sum_of_lp_vals.value / self.sum_of_ones.value)
    else:
      # TODO(b/205769820): Make sure if the output of pow op below is
      # numerically valid.
      return (self.sum_of_lp_vals.value /
              self.sum_of_ones.value)**(1.0 / self.config.lp_order)

  def bound(  #
      self, calibration_config: aqt_config.CalibrationConfig) -> jnp.ndarray:
    """Compute the upper bound on input tensor values, broadcastable to input."""
    return (calibration_config.l1_dev_coeff * self.l1_dev() +  #
            calibration_config.lp_dev_coeff * self.lp_dev() +  #
            calibration_config.max_dev_coeff * self.max_dev() +  #
            calibration_config.const_bound_coeff)


def is_config_active(config: aqt_config.AqtTensorConfig,
                     event_count: jnp.ndarray) -> bool:
  """Return whether the configuration is active at event_count."""
  config.validate()
  should_q = True
  if config.begin_at_event is not None:
    should_q &= config.begin_at_event <= event_count
  if config.end_at_event is not None:
    should_q &= event_count < config.end_at_event
  return should_q


class TensorQuantizer(nn.Module):
  """Maintains state associated with the quantization of an input tensor.

  A TensorQuantizer owns observed statistics for an input tensor, along with
  variables for the scale and event_count copy, recording the last time of then
  most recent update to this `TensorQuantizer` class.

  This class provides state-mutating methods for updating statistics for every
  observation of an input tensor and is used by AQT methods to derive
  calibration bounds.

  TensorQuantizer assumes the supplied `event_count` strictly monotonically
  increases across `TensorQuantizer.update` calls and starts out strictly
  greater than `int32.min`.

  Attributes:
    data_shape: the shape of input tensor. Some dimensions may be of unknown
      size, but these must have stats shared in `config.stats_config`. In this
      case, use of quanitzed variable is disallowed.
    config: A training quantization schedule.
    quantized_variable: If provided during initialization, stores the quantized
      version of the observed tensor from the most recent `update()`.
  """
  data_shape: List[Optional[int]]
  config: Optional[aqt_config.AqtScheduleConfig] = None

  def setup(self):
    if self.config is not None:
      self.config.fill_gaps_with_float_config()
      self.config.validate(self.data_shape)
      self._stats = Stats(self.data_shape, self.config.stats_config)
      self._scale = self.variable('TensorQuantizer', 'scale', jnp.zeros,
                                  self._stats.stats_shape, jnp.float32)
      self._inv_scale = self.variable('TensorQuantizer', 'inv_scale', jnp.zeros,
                                      self._stats.stats_shape, jnp.float32)
      if self.config.use_quantized_variable:
        self.quantized_variable = self.variable('TensorQuantizer',
                                                'quantized_variable', jnp.zeros,
                                                self.data_shape, jnp.int8)
      else:
        self.quantized_variable = self.variable('TensorQuantizer',
                                                'quantized_variable',
                                                lambda: None)
      self._last_update = self.variable('TensorQuantizer', 'last_update',
                                        lambda: jnp.iinfo(jnp.int32).min)

  def __call__(self):
    # Keep __call__() just not to require users to explicitly specify which
    # method should be called when invoking init().
    pass

  def _fresh_scale(
      self,
      config: aqt_config.AqtTensorConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns an updated scale provided by a config or the current one if None."""
    if isinstance(config.quant_config, aqt_config.FloatConfig):
      # We shouldn't update the scale if the given config contains FloatConfig;
      # fill with a poison value if we get into this situation.
      nan = jnp.full(self._stats.stats_shape, jnp.nan, dtype=jnp.float32)
      return nan, nan

    x_bound = self._stats.bound(config.calibration_config)
    clip_bound = aqt_common.get_clip_bound(config.quant_config)

    new_scale = clip_bound / x_bound
    inv_scale = x_bound / clip_bound
    return new_scale, inv_scale

  def clip_range(self) -> jnp.ndarray:
    """Returns the tensor clip range or zeros if no int config is active."""
    for config in self.config.tensor_configs:
      if (is_config_active(config, self._last_update.value) and
          isinstance(config.quant_config, aqt_config.IntQuantConfig)):
        clip_bound = aqt_common.get_clip_bound(config.quant_config)
        return self._inv_scale.value * clip_bound
    return jnp.zeros(self._stats.stats_shape)

  def update(
      self,  #
      sample: jnp.ndarray,
      weight: Optional[jnp.ndarray],
      event_count: jnp.ndarray):
    """Updates statistics, scale, and quantized variable."""

    if self.config is None:
      return

    active_configs = [
        c for c in self.config.tensor_configs
        if is_config_active(c, event_count)
    ]

    if len(active_configs) == 1:
      self._update_config(active_configs[0], sample, weight, event_count)
    else:
      raise ValueError('There must be exactly one active config.')

  def _update_config(
      self,  #,
      config: aqt_config.AqtTensorConfig,
      sample: jnp.ndarray,
      weight: Optional[jnp.ndarray],
      event_count: jnp.ndarray):
    """update(), but with active `config`."""

    should_update_scale = self._should_update_scale(config, event_count)
    self._last_update.value = event_count

    self._stats.update(sample, weight)

    new_scale, inv_scale = jax.lax.cond(
        should_update_scale,  #
        lambda: self._fresh_scale(config),
        lambda: (self._scale.value, self._inv_scale.value))
    self._scale.value = new_scale
    self._inv_scale.value = inv_scale

    if (self.config.use_quantized_variable and
        isinstance(config.quant_config, aqt_config.IntQuantConfig) and
        config.quant_config.compatible_with_int8()):
      scale, _ = self._get_quant_scale(train=True)
      sample = scale * sample
      new_var = self._to_quant(
          sample, train=True).astype(self.quantized_variable.value.dtype)
      self.quantized_variable.value = new_var

  def _should_update_scale(
      self,  #
      config: aqt_config.AqtTensorConfig,
      event_count: jnp.ndarray) -> bool:
    """Returns if scale should be updated for the config and event count."""
    if isinstance(config.quant_config, aqt_config.FloatConfig):
      return False

    if not config.freeze_scale_at_begin:
      return True

    # The first time a config is active, even if we freeze scale, we should
    # update the scale.
    was_previously_inactive = not is_config_active(config,
                                                   self._last_update.value)

    # We rely on jnp.int32.min being an illegal event count value, so that
    # even if is_config_active(config, jnp.int32.min), we still update scale.
    # This could happen if the very first update happens for a config which
    # has freeze_scale_at_begin and begin=None.
    assert event_count > jnp.iinfo(jnp.int32).min, ('event_count cannot be '
                                                    'int32.min')

    first_event = jnp.array(self._last_update.value == jnp.iinfo(jnp.int32).min)

    return was_previously_inactive | first_event

  def _to_quant(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Quantizes x with active quant config, if any, else act as identity."""

    def qparams(
        config: aqt_config.AqtTensorConfig,
        check_active: bool = True
    ) -> Tuple[bool, jnp.array, jnp.array, jnp.array]:
      """Returns parameters for AQT quantization routine."""

      should_quantize = False
      clip_bound = jnp.array(0.0)
      shift_before = jnp.array(0.0)
      shift_after = jnp.array(0.0)

      config_active = is_config_active(config, self._last_update.value)
      config_active = not check_active or config_active

      if isinstance(config.quant_config, aqt_config.FloatConfig):
        clip_bound = jnp.where(config_active, float('inf'), 0.0)
        return should_quantize, clip_bound, shift_before, shift_after
      elif isinstance(config.quant_config, aqt_config.IntQuantConfig):
        should_quantize |= config_active
        config_active = jnp.float32(config_active)

        clip_bound += config_active * aqt_common.safe_clip_bound(
            config.quant_config)

        if config.quant_config.preserve_zero:
          shift_before += config_active * 0.5
        else:
          shift_after += config_active * 0.5

        return should_quantize, clip_bound, shift_before, shift_after
      else:
        raise NotImplementedError(
            'Only supporting FloatConfig and IntQuantConfig for now.')

    if self.config is None:
      clip_bound = jnp.array(float('inf'))
      shift_before = shift_after = 0.0
      should_quantize = False
    elif not train and self.config.inference_config_index is not None:
      should_quantize, clip_bound, shift_before, shift_after = qparams(
          self.config.tensor_configs[self.config.inference_config_index],
          check_active=False)
    elif self.config.tensor_configs:
      should_quantize, clip_bound, shift_before, shift_after = zip(
          *map(qparams, self.config.tensor_configs))
      should_quantize = any(should_quantize)
      clip_bound = sum(clip_bound)
      shift_before = sum(shift_before)
      shift_after = sum(shift_after)
    else:
      clip_bound = shift_before = shift_after = 0.0
      should_quantize = False

    maybe_floor = (lambda y: jnp.where(should_quantize, jnp.floor(y), y))

    # Note that backprop does not depend directly on the value of _last_update
    # or any_config_active; only scales and constants need to be maintained
    # and there's no branching on ops including the input tensor x. This
    # results in significant memory reduction, see cl/415355150.
    x = jnp.clip(x, -clip_bound, clip_bound)
    x += shift_before
    x = pass_through(x, maybe_floor)
    # TODO(b/219778053): Add a test that validates the shift of values. Mutants
    # found that removing the line below doesn't affect any tests.
    x += shift_after

    return x

  def _get_quant_scale(self, train: bool) -> jnp.ndarray:
    """Returns scales to quantize/dequantize the active quant config, if any, else ones."""
    if self.config is None:
      scale = inv_scale = jnp.array(1.0, dtype=jnp.float32)
      return scale, inv_scale  # pytype: disable=bad-return-type  # jax-ndarray

    if not train and self.config.inference_config_index is not None:
      inference_config = self.config.tensor_configs[
          self.config.inference_config_index]
      should_quantize = isinstance(inference_config.quant_config,
                                   aqt_config.IntQuantConfig)
    else:
      should_quantize = False
      for config in self.config.tensor_configs:
        if isinstance(config.quant_config, aqt_config.FloatConfig):
          continue
        config_active = is_config_active(config, self._last_update.value)
        should_quantize |= config_active

    scale = jnp.where(should_quantize, self._scale.value,
                      jnp.ones_like(self._scale.value))
    inv_scale = jnp.where(should_quantize, self._inv_scale.value,
                          jnp.ones_like(self._inv_scale.value))

    return scale, inv_scale  # pytype: disable=bad-return-type  # jax-ndarray
