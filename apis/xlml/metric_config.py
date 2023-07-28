# Copyright 2023 Google LLC
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

"""Config file for a metric."""

import dataclasses
import enum


class AssertionType(enum.Enum):
  FIXED_VALUE = enum.auto()
  WITHIN_BOUNDS = enum.auto()
  STD_DEVS_FROM_MEAN = enum.auto()
  PERCENT_DIFFERENCE = enum.auto()


class AssertionStrategy(enum.Enum):
  LESS = enum.auto()
  GREATER = enum.auto()
  EQUAL = enum.auto()
  WITHIN = enum.auto()


class AggregationStrategy(enum.Enum):
  FINAL = enum.auto()
  MIN = enum.auto()
  MAX = enum.auto()
  AVERAGE = enum.auto()
  MEDIAN = enum.auto()


# TODO(ranran): refactor and add use cases for attributes during integration.
@dataclasses.dataclass
class MetricConfig:
  """This is a class to set up configs of a metric to validate ML model convergence.

  Attributes:
    metric_name: The name of a metric.
    aggreation_stragety: The aggregation strategy of a metrics, including less,
      greater, equal, and within.
    assertion_type: The assertion type of a metric to its boundary.
    assertion_strategy: The assertion strategy of a metric, including final,
      minimum, maximum, average, and median value.
    wait_for_n_data_points: The number of data points to discard before
      assertion.
    lower_bound: The lower bound of assertion.
    higher_bound: The higher bound of assertion.
  """

  metric_name: str
  aggreation_stragety: AggregationStrategy
  assertion_type: AssertionType
  assertion_strategy: AssertionStrategy
  wait_for_n_data_points: int
  lower_bound: float
  higher_bound: float
