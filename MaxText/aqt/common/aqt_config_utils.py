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

from typing import Any, Dict, List, Type, TypeVar

import dacite


class ConfigError(Exception):
  pass


T = TypeVar('T', bound='_BaseConfig')


@dataclasses.dataclass
class _BaseConfig:
  """Adds serialization and primitive type checking to dataclasses."""

  def to_dict(self) -> Dict[str, Any]:
    """Converts a dataclass to dict while preserving dataclass-specific logic."""
    dataclass_dict = {}
    for field in dataclasses.fields(self):
      attr = getattr(self, field.name)
      if isinstance(attr, _BaseConfig):
        dataclass_dict[field.name] = attr.to_dict()
      elif isinstance(attr, list):
        dataclass_dict[field.name] = [
            a.to_dict() if isinstance(a, _BaseConfig) else a for a in attr
        ]
      # For enums pass the name to use for making the proto.
      elif isinstance(attr, enum.Enum):
        dataclass_dict[field.name] = attr.name
      else:
        dataclass_dict[field.name] = attr

    return dataclass_dict

  @classmethod
  def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    return dacite.from_dict(cls, data)

  def validate(self):
    pass


# TODO(lew): Once the dependencies are untangled, we can uncomment this line.
# aqt_tensor_config_type = 'AqtTensorConfig'
aqt_tensor_config_type = Any


def _validate_intervals(configs: List[aqt_tensor_config_type]):
  """Ensures that the provided schedule specifies increasing time segments.

  We require configs to be in sorted order with respect to
  `(start_at_event, end_at_event)` intervals. This makes disjointness checking
  easier.

  Args:
    configs: a list of tensor quantization configs.

  Raises:
    ConfigError: if any of the described conditions are not met.
  """
  previous_end = None
  for i, config in enumerate(configs):
    current_begin = config.begin_at_event
    current_end = config.end_at_event

    neither_none = current_begin is not None and current_end is not None
    if neither_none and current_end < current_begin:
      raise ConfigError(f'config[{i}].end_at_event ({current_end}) < '
                        f'config[{i}].begin_at_event ({current_begin})')

    if i > 0:
      if current_begin is None:
        raise ConfigError('only the first config may omit begin_at_event, but '
                          f'config[{i}] does too')

      if previous_end is None:
        raise ConfigError('only the last config may omit end_at_event, but '
                          f'config[{i-1}] does too')

      if previous_end > current_begin:
        raise ConfigError(f'config[{i-1}].end_at_event ({previous_end}) > '
                          f'config[{i}].begin_at_event ({current_begin})')

    if previous_end is not None and previous_end != current_begin:
      raise ConfigError(
          f'There is a gap between config[{i-1}].end_at_event ({previous_end}) '
          f'and config[{i}].begin_at_event ({current_begin}). There must be '
          f'always one active config.')

    previous_end = current_end


def _validate_alignment(
    lhs_path: str,  #
    lhs_configs: List[aqt_tensor_config_type],
    rhs_path: str,
    rhs_configs: List[aqt_tensor_config_type]):
  """Ensures that quantization schedules for both arguments are aligned.

  For binary quantized operations, a quantization schedule for each argument
  specifies the type of quantization that should be applied at a given time.

  The configuration is assumed to specify equal-length lists of quantization
  configurations for LHS and RHS configs, for which parallel entries must
  have identical endpoints for quantization activation.

  Args:
    lhs_path: what to call the `lhs` in error strings.
    lhs_configs: the tensor quantization schedule for the left argument.
    rhs_path: what to call the `rhs` in error strings.
    rhs_configs: the tensor quantization schedule for the right argument.

  Raises:
    ConfigError: if any of the described conditions are not met.
  """
  if len(lhs_configs) != len(rhs_configs):
    raise ConfigError(f'lhs config len {len(lhs_configs)} must equal '
                      f'rhs config len {len(rhs_configs)}')

  for i, (lhs_config, rhs_config) in enumerate(zip(lhs_configs, rhs_configs)):

    def _interval(config):
      return config.begin_at_event, config.end_at_event

    if _interval(lhs_config) != _interval(rhs_config):
      raise ConfigError(
          '(begin_at_event, end_at_event) intervals do not match: '
          f'{lhs_path}[{i}] ({_interval(lhs_config)}) != '
          f'{rhs_path}[{i}] ({_interval(rhs_config)})')
