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

"""Functions to get bounds of activation/input using its statistics."""

import dataclasses
import typing
from typing import Iterable, Optional, Tuple, Union

from aqt.jax_legacy.jax import primitives
from aqt.jax_legacy.jax import quant_config
from aqt.jax_legacy.jax import shape_utils
from aqt.jax_legacy.jax.flax import struct as flax_struct
from aqt.jax_legacy.jax.stats import Stats
from flax import linen as nn
from jax import lax
import jax.numpy as jnp

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


class DynamicBounds(nn.Module):
  """Get Bounds of activation using statistics.

  Attributes:
    hyper: hyperparamater to compute bound from statistics.
  """

  @dataclass
  class Hyper:
    """Hyperparameters for DynamicBounds."""
    granularity: quant_config.QuantGranularity
    # Clipping coefficient applied during dynamic quantization.
    # E.g. if it is 0.9 then dynamic scale will be reduced by 10%.
    clipping_coeff: float = 1.0

  @dataclass
  class Params:
    """Parameters for act quantiztaion using get_bounds."""
    # Axis along which to quantize activations.
    quant_axis: Optional[Iterable[int]] = None
    # Optional shape to verify if bounds shape is expected. Defaults to None.
    expected_bounds_shape: Union[None, int, Tuple[int, ...]] = None
    # Optional name of the get_bounds module.
    module_name: Optional[str] = None

  hyper: Hyper

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      bounds_params: Params,
  ) -> jnp.ndarray:
    """Compute the input batch statistics.

    Args:
      x: the input to get bounds from using statistics.
      bounds_params: parameters to compute input's statistics and bounds.

    Returns:
      Bound value (same shape as inputs).
    """

    x = jnp.asarray(x, jnp.float32)

    hyper = self.hyper

    if bounds_params.quant_axis is not None:
      quant_axis = bounds_params.quant_axis
    else:
      if hyper.granularity == quant_config.QuantGranularity.PER_TENSOR:
        # Equivalently, this could be written as
        # quant_axis = tuple(range(x.ndim))
        quant_axis = None
      elif hyper.granularity == quant_config.QuantGranularity.PER_CHANNEL:
        # Quantize by aggregating activation statistics across all dimensions of
        # the activation tensor EXCEPT the batch dimension. Also known as per
        # example statistics
        quant_axis = tuple(range(1, x.ndim))
      else:
        raise ValueError(f'Unknown granularity {hyper.granularity}')

    abs_max_x = primitives.max_abs_weights(x, axis=quant_axis)

    return abs_max_x * self.hyper.clipping_coeff


class GetBounds(nn.Module):
  """Get Bounds of activation using statistics.

  Attributes:
    hyper: hyperparamater to compute bound from statistics.
  """

  @dataclass
  class Hyper:
    """Hyperparameters for GetBounds."""
    # bound =
    # stddev_coeff * mix_coeff * stddev + absdev_coeff * (1-mix_coeff) * absdev
    initial_bound: float  # Initial bounds value before bounds get updated for
    # the first time
    granularity: quant_config.QuantGranularity
    stddev_coeff: float  # param to weigh the stddev
    absdev_coeff: float  # param to weigh the absdev, not used in ucb
    mix_coeff: float  # 0 = only absdev, 1 = only stddev, not used in ucb
    reset_stats: bool = False  # whether to reset stats when bounds are updated
    # Exponential moving average coefficient for stats collection. 'None'
    # indicates to use a simple mean of a set of past samples, as governed by
    # the 'reset_stats' parameter.
    ema_coeff: Optional[float] = None

    # TODO(shivaniagrawal): Refactor these boolean flags to an enum since
    # they are mututally exclusive.

    # Whether to use |mean| + centered stddev formula to calculate bounds.
    # CAMS is an acronym for Centered Absolute Mean + Stddev.
    use_cams: bool = False
    # Whether to exclude zeros from statistics computation. This can be useful
    # when zeros were likely introduced by preceding ops, e.g. relu or dropout.
    exclude_zeros: bool = False
    # Whether to use a running mean of the maximum of the absolute value of
    # activation tensors to calculate bounds.
    use_mean_of_max: bool = False

    # below four variables are new get_bound coefficients
    # coefficients should be float, but set to None as default value
    # to avoid invisible bugs
    cams_coeff: Optional[float] = None
    cams_stddev_coeff: Optional[float] = None
    mean_of_max_coeff: Optional[float] = None
    fixed_bound: Optional[float] = None
    # whether to use old or new code to compute the bound
    use_old_code: bool = True

  @dataclass
  class Params:
    """Parameters for act quantiztaion using get_bounds."""
    update_stats: bool  # Whether to update statistics.
    update_bounds: bool  # Whether to update bounds.
    # The axis name used to combine batch statistics from multiple devices.
    # See `jax.pmap` for a description of axis names. Defaults to None.
    paxis_name: Optional[str] = None
    # Optional shape to verify if bounds shape is expected. Defaults to None.
    expected_bounds_shape: Union[None, int, Tuple[int, ...]] = None
    # Optional boolean tensor of the same shape as 'x' specifying which values
    # of 'x' to use as part of the bounds calculation. 'True' indicates the
    # corresponding value from 'x' should be used. If None, all values are used.
    mask: Optional[jnp.ndarray] = None
    # Optional name of the get_bounds module.
    module_name: Optional[str] = None

  hyper: Hyper

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      bounds_params: Params,
  ) -> jnp.ndarray:
    """Compute the input batch statistics.

    Args:
      x: the input to get bounds from using statistics.
      bounds_params: parameters to compute input's statistics and bounds.

    Returns:
      Bound value (same shape as inputs).
    """

    if bounds_params.mask is not None:
      shape_utils.assert_shapes_compatible(x.shape, bounds_params.mask.shape)

    x = jnp.asarray(x, jnp.float32)

    hyper = self.hyper

    is_initializing = not self.has_variable('get_bounds', 'stats')

    if hyper.granularity == quant_config.QuantGranularity.PER_TENSOR:
      # Equivalently, this could be written as
      # quant_axis = tuple(range(x.ndim))
      quant_axis = None
      stats_shape = (1,) * len(x.shape)
    elif hyper.granularity == quant_config.QuantGranularity.PER_CHANNEL:
      # Quantize by aggregating activation statistics across all dimensions of
      # the activation tensor EXCEPT the last dimension, which we interpret as
      # the channel dimension. For example, in a transformer context, x might
      # have a shape corresponding to [example, token, channel], in which case
      # this aggregates activation statistics separately for each feature, where
      # for each feature it aggregates over all unmasked tokens in all examples.
      quant_axis = tuple(range(x.ndim - 1))
      stats_shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
    else:
      raise ValueError(f'Unknown granularity {hyper.granularity}')

    stats_state = self.variable('get_bounds', 'stats', Stats.stats_initializer,
                                stats_shape)

    def bound_initializer(shape):
      return hyper.initial_bound * jnp.ones(shape)

    bounds = self.variable('get_bounds', 'bounds', bound_initializer,
                           stats_shape)

    if bounds_params.update_stats and not is_initializing:
      stats_state.value = Stats.create_updated_stats(
          stats_state.value,
          x,
          mask=bounds_params.mask,
          axis=quant_axis,
          paxis_name=bounds_params.paxis_name,
          alpha=hyper.ema_coeff,
          exclude_zeros=hyper.exclude_zeros)

    if bounds_params.update_bounds and not is_initializing:
      bounds.value = self._stats_to_bounds(stats_state.value)
      if hyper.reset_stats:
        stats_state.value = Stats.stats_initializer(stats_shape)
    return bounds.value

  def _stats_to_bounds(self, stats_value: Stats) -> jnp.ndarray:
    """Computes activation clipping bounds from activation statistics."""
    hyper = self.hyper
    maximum = stats_value.mean_batch_maximum
    minimum = stats_value.mean_batch_minimum
    mom = jnp.maximum(jnp.abs(maximum), jnp.abs(minimum))
    stddev_uncentered = lax.sqrt(stats_value.mean_sq)
    absdev_uncentered = stats_value.mean_abs
    stddev = lax.sqrt(stats_value.mean_sq - stats_value.mean**2)
    abs_mean = jnp.abs(stats_value.mean)
    if hyper.use_old_code:  # old code of computing the bound
      if hyper.use_cams:  # upper confidence bound formula
        return abs_mean + hyper.stddev_coeff * stddev
      elif hyper.use_mean_of_max:
        return mom
      else:
        return (hyper.mix_coeff * hyper.stddev_coeff * stddev_uncentered +
                (1 - hyper.mix_coeff) * hyper.absdev_coeff * absdev_uncentered)
    else:  # use new way of computing the bound
      cams = abs_mean + hyper.cams_stddev_coeff * stddev
      return (hyper.fixed_bound + hyper.mean_of_max_coeff * mom +
              hyper.stddev_coeff * stddev_uncentered +
              hyper.absdev_coeff * absdev_uncentered + hyper.cams_coeff * cams)
