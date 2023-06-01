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
"""Dataclass implementation of AQT configuration."""

import dataclasses
import enum
from typing import Any, Dict, List, Optional, Type, Union

from aqt.common.aqt_config_utils import _BaseConfig
from aqt.common.aqt_config_utils import _validate_intervals
from aqt.common.aqt_config_utils import ConfigError



@dataclasses.dataclass
class IntQuantConfig(_BaseConfig):
  # pyformat: disable
  """Config for integer quantization in aqt_ops.

  Attributes:
    bits:
      Number of bits to quantize to (e.g 4 for int4). Must be positive.
    preserve_zero:
      Whether or not zeros should be representable.
  """
  # pyformat: enable
  bits: int
  preserve_zero: bool = True

  def validate(self):
    if self.bits < 1:
      raise ConfigError(f'expected bits={self.bits} > 0')

  def compatible_with_int8(self) -> bool:
    """Checks if the config is compatible with int8."""
    return self.bits <= 8


@enum.unique
class RoundingMode(enum.Enum):
  """Supported rounding mode for SmallFloatConfig."""
  ROUND_AWAY_FROM_ZERO = 'round_away_from_zero'
  ROUND_TO_NEAREST_EVEN = 'round_to_nearest_even'
  ROUND_STOCHASTIC = 'round_stochastic'


@dataclasses.dataclass
class SmallFloatConfig(_BaseConfig):
  """Enum for emulated lower-precision formats."""
  exponent_bits: int
  mantissa_bits: int
  min_exp: int
  max_exp: int
  support_inf: bool  # Not yet supported
  rounding_mode: RoundingMode  # Currently only using ROUND_TO_NEAREST_EVEN


@dataclasses.dataclass
class FloatConfig(_BaseConfig):
  """Config for non quantization in aqt_ops.

  Attributes:
    use_bf16: Whether or not to use bfloat16.
  """
  use_bf16: bool = True


# Warning: Adding new oneof fields may cause from_dict to select the wrong
# value. The behavior of dataclasses is to select the first value in the
# union which has all values specified For example if FloatConfig were first,
# it would always be selected since its only field use_bf16 has a default
# value.
QuantConfig = Union[IntQuantConfig, SmallFloatConfig, FloatConfig]


@dataclasses.dataclass
class StatsConfig(_BaseConfig):
  # pyformat: disable
  """Config for aqt_ops.Stats.

  Attributes:
    ema_update_count:
      Scale calibration sample size.
      The weight of the newest batch will be 1 / ema_update_count.
      E.g. 1 means that only the current batch is used in calibration.
      Weights of examples in a batch are also correctly taken into account.
    share_stats_axes:
      Bound sharing. Required for all contraction axes. Each axis in this list
      must be in [0, rank) where rank is the rank of the tensor to quantize.
      This list must also be strictly sorted.
    filter_zeros:
      Ignores zeros when computing statistics. Useful for distributions with a
      zero spike like the output of Relu.
    lp_order:
      Defines moment used in lp_dev, needs to be even and positive.

    update_count_prior:
      Emulates a situation where some statistics have already been collected.
      Default of `1.0` prevents division by zero in prod, while setting it to
      `0` allows for precise logic in unit testing.
    mean_prior:
      Emulates a situation where some statistics have already been collected.
    l1_dev_prior:
      Emulates a situation where some statistics have already been collected.
    lp_dev_prior:
      Emulates a situation where some statistics have already been collected.
    max_dev_prior:
      Encode prior knowledge of the maximum absolute value observed; note that
      the `update_count_prior` does not affect the prior weight for the max so
      in practice this may be usually left at zero.

    tpu_cross_replica_sum:
      Setting it to false might improve performance, but note that checkpoint
      might contain wrong value.
  """
  # pyformat: enable
  ema_update_count: int
  share_stats_axes: List[int]
  filter_zeros: bool = True
  lp_order: int = 2

  update_count_prior: float = 1.0
  mean_prior: float = 0.0
  l1_dev_prior: float = 0.0
  lp_dev_prior: float = 0.0
  max_dev_prior: float = 0.0

  tpu_cross_replica_sum: bool = True

  def validate(self, data_shape: List[Optional[int]]):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Validates this StatsConfig for the provided data shape.

    Args:
      data_shape: the shape of the input tensor which will be quantized with
        self as the statistics configuration. If an entry is None, this
        indicates a dimension whose size is unknown at graph compilation time.

    Raises:
      ConfigError: if any of the specified share_stats_axes are not between
        [0, rank) where rank is len(data_shape), the share_stats_axes are
        not sorted, don't contain all unknown dimensions (None in data_shape),
        or if ema_update_count is not positive.
    """
    rank = len(data_shape)
    if any(ax < 0 or ax >= rank for ax in self.share_stats_axes):  # pylint: disable=not-an-iterable
      raise ConfigError(f'share_stats_axes ({self.share_stats_axes}) must be '
                        f'between 0 and rank ({rank}), not including rank')
    if not all(ax1 < ax2 for ax1, ax2 in zip(self.share_stats_axes[:-1],
                                             self.share_stats_axes[1:])):
      raise ConfigError(
          f'share_stats_axes ({self.share_stats_axes}) must be strictly sorted')

    unknown_axes = {i for i, dim in enumerate(data_shape) if dim is None}
    shared_axes = set(self.share_stats_axes)
    if not unknown_axes.issubset(shared_axes):
      raise ConfigError(f'expected share_stats_axes ({self.share_stats_axes}) '
                        'to contain unknown axes for given data shape '
                        f'({data_shape})')

    if self.ema_update_count < 1:
      raise ConfigError(
          f'expected ema_update_count={self.ema_update_count} >= 1')

    if self.lp_order < 1:
      raise ConfigError(f'expected lp_order={self.lp_order} >= 1')


@dataclasses.dataclass
class CalibrationConfig(_BaseConfig):
  """Config for aqt_ops.Stats.

  Attributes:
    const_bound_coeff: Bias used in bound calibration.
    l1_dev_coeff: Weight of the l1_dev in bound calibration.
    lp_dev_coeff: Weight of the lp_dev in bound calibration.
    max_dev_coeff: Weight of the max_dev in bound calibration.
  """
  const_bound_coeff: float = 0.0
  l1_dev_coeff: float = 0.0
  lp_dev_coeff: float = 0.0
  max_dev_coeff: float = 0.0


@dataclasses.dataclass
class AqtTensorConfig(_BaseConfig):
  """Config for aqt_ops.AqtTensor.

  Attributes:
    quant_config: Numerical format for quantization.
    calibration_config: Calibration config.
    freeze_scale_at_begin: Freezes the calibration when begin_at_event is
      reached. This breaks the feedback look between calibration scale and
      distribution observed by stats.
    begin_at_event: Start of this quantization interval. You need to make sure
      that there are enough updates before quantization is enabled with
      begin_at_event or set a proper prior or use const_bound. This is very
      important if freeze_scale_at_begin == true.
    end_at_event: End of this quantization interval.
  """
  quant_config: QuantConfig
  calibration_config: CalibrationConfig
  freeze_scale_at_begin: bool
  begin_at_event: Optional[int] = None
  end_at_event: Optional[int] = None

  def validate(self):
    """Validates this tensor config."""
    if (self.begin_at_event is not None and self.end_at_event is not None and
        self.begin_at_event > self.end_at_event):
      raise ConfigError(f'expected begin_at_event={self.begin_at_event} <= '
                        f'end_at_event={self.end_at_event}')

    if not isinstance(self.quant_config, IntQuantConfig) and not isinstance(
        self.quant_config, FloatConfig) and not isinstance(
            self.quant_config, SmallFloatConfig):
      raise ConfigError('quant_config must be one of '
                        '{int_quant_config, float_config, small_float_config}.')
    if isinstance(self.quant_config, SmallFloatConfig):
      small_float = self.quant_config
      if small_float.support_inf:
        raise NotImplementedError('support_inf is not yet supported.')
      if small_float.rounding_mode != RoundingMode.ROUND_TO_NEAREST_EVEN:
        raise NotImplementedError(
            'Using small_float.rounding_mode '
            'only currently supports ROUND_TO_NEAREST_EVEN.')

  def to_dict(self) -> Dict[str, Any]:
    # AqtTensorConfig dataclass does not have `int_quant_config` and
    # `float_config` which are present in proto. Instead, it comes with
    # `quant_config`, but it is not a valid field in the corresponding proto.
    # So, they are manually added/removed below for proper conversion.
    dataclass_dict = dataclasses.asdict(self)
    cfg_type = 'int_quant_config' if isinstance(
        self.quant_config, IntQuantConfig) else 'float_config'
    dataclass_dict[cfg_type] = dataclass_dict.pop('quant_config')
    return dataclass_dict


@dataclasses.dataclass
class AqtScheduleConfig(_BaseConfig):
  """Config for an `aqt_tensor.TensorQuantizer` configuration schedule.

  Attributes:
    stats_config: Statistics configs describing what statistics to track.
    tensor_configs: Tensor configs to apply to the input tensor. *_at_event must
      describe sorted and disjoint intervals. *_at_event must be identical in
      the filter configs.
    use_quantized_variable: If true and in training mode, saves intermediate
      quantizations to user-provided variables. During inference, quantized
      variables are read from but not written to with new input values.
    inference_config_index: If not None, then the index into `tensor_configs`
      which indicates which tensor configuration to use during inference. This
      allows for static switching logic at inference time, which improves
      throughput. If None, then the most recently trained-on config is the one
      that's used.
    allow_int_small_float: Whether Integer and SmallFloat configs can exist in
      the same schedule. This field allows later checks to be done at the Python
      level instead of introducing additional tf.where operations.
  """
  stats_config: StatsConfig
  tensor_configs: List[AqtTensorConfig]
  use_quantized_variable: bool = False
  inference_config_index: Optional[int] = None
  allow_int_small_float: Optional[bool] = None

  def quantization_mode(self) -> Type[QuantConfig]:
    """Returns which quantization to use.

    Raises: ConfigError if both IntQuantConfig and SmallFloatConfig are given.
    """
    has_quantization = False
    has_small_float = False
    for tensor_config in self.tensor_configs:
      quant_config = tensor_config.quant_config
      if isinstance(quant_config, IntQuantConfig):
        has_quantization = True
      elif isinstance(quant_config, SmallFloatConfig):
        has_small_float = True
      elif isinstance(quant_config, FloatConfig):
        continue
      else:
        raise ConfigError(
            'quant_config must be one of '
            '{int_quant_config, float_config, small_float_config}.')
    if has_quantization and has_small_float:
      raise ConfigError(
          'Found both IntQuantConfig and SmallFloatConfig. Only '
          'one of the two should be specified among all tensor_configs.')
    elif has_quantization:
      return IntQuantConfig
    elif has_small_float:
      return SmallFloatConfig
    else:
      return FloatConfig

  def validate(self, data_shape: List[Optional[int]]):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Validates this AqtScheduleConfig for the provided data shape."""
    # The output value of quantization_mode is unused.
    if not self.allow_int_small_float:
      self.quantization_mode()
    _validate_intervals(self.tensor_configs)
    if any(ax is None for ax in data_shape) and self.use_quantized_variable:
      raise ConfigError(
          'use of quantized variable with unknown dimensions is disallowed')
    if self.inference_config_index is not None and not (
        0 <= self.inference_config_index < len(self.tensor_configs)):
      raise ConfigError(
          f'inference_config_index ({self.inference_config_index}) must be '
          'at least 0 and less than len(tensor_configs) '
          f'({len(self.tensor_configs)})')

  def fill_gaps_with_float_config(self):
    """Fills gaps with FloatConfig to always have one active config."""

    def create_float_config(begin, end):
      tc = AqtTensorConfig(
          quant_config=FloatConfig(),
          calibration_config=CalibrationConfig(),
          freeze_scale_at_begin=True,
          begin_at_event=begin,
          end_at_event=end)
      return tc

    previous_end = None
    filled_configs = []
    inference_config_index = None
    for i, config in enumerate(self.tensor_configs):
      current_begin = config.begin_at_event

      if previous_end != current_begin and current_begin > 0:
        filled_configs.append(create_float_config(previous_end, current_begin))

      filled_configs.append(config)
      previous_end = config.end_at_event

      if self.inference_config_index == i:
        inference_config_index = len(filled_configs) - 1

    if previous_end is not None or not filled_configs:
      filled_configs.append(create_float_config(previous_end, None))

    self.tensor_configs = filled_configs
    # Note that self.inference_config_index is either None or in [0, len)
    # where len is the length of the previous self.tensor_configs. So the
    # update below either keeps it set to None or updates the index to the
    # reindexed value for the new self.tensor_configs.
    self.inference_config_index = inference_config_index


@dataclasses.dataclass
class AqtMatmulConfig(_BaseConfig):
  """Quantization config for a matmul.

  Attributes:
    lhs: quantization schedule for left-hand side argument
    rhs: quantization schedule for right-hand side argument
    grad: quantization schedule for gradients (optional).
  """
  lhs: AqtScheduleConfig
  rhs: AqtScheduleConfig
  grad: Optional[AqtScheduleConfig] = None


@dataclasses.dataclass
class AqtEinsumConfig(_BaseConfig):
  """Quantization config for a two-argument einsum.

  Attributes:
    lhs: quantization schedule for left-hand side argument
    rhs: quantization schedule for right-hand side argument
    grad: quantization schedule for gradients (optional).
  """
  lhs: AqtScheduleConfig
  rhs: AqtScheduleConfig

