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

"""Common utility functions used across different tests for both TF and JAX."""

import copy
from typing import List, Tuple, Union

from absl import flags
from aqt.common import aqt_common
from aqt.common import aqt_config
import numpy as np

flags.DEFINE_bool("use_tpu", False, "Set it to enable TPU-only tests.")


def test_stats_config():
  """Config base for all the experiments."""
  return aqt_config.StatsConfig(
      ema_update_count=1,  # single update overwrites the stats
      update_count_prior=0,  # easier equations
      share_stats_axes=[0, 1],  # one stat per whole tensor
      tpu_cross_replica_sum=False,  # no cross-tpu reduction
      filter_zeros=False,  # on default zeros are a number like any other
  )


def generate_unaligned_schedule_intervals():
  """Returns [start, stop) intervals for unaligned quantization schedules."""

  i64min = np.iinfo(np.int64).min
  i64max = np.iinfo(np.int64).max

  return [{
      "lhs_intervals": [(-1, 5)],
      "rhs_intervals": [(-1, 6)],
      "testcase_name": "ne_end_at_event"
  }, {
      "lhs_intervals": [(-1, 5)],
      "rhs_intervals": [(-2, 5)],
      "testcase_name": "ne_begin_at_event"
  }, {
      "lhs_intervals": [(None, None)],
      "rhs_intervals": [(None, i64max)],
      "testcase_name": "max_edge"
  }, {
      "lhs_intervals": [(None, None)],
      "rhs_intervals": [(i64min, None)],
      "testcase_name": "min_edge"
  }, {
      "lhs_intervals": [(0, 10)],
      "rhs_intervals": [(0, 5), (5, 10)],
      "testcase_name": "unequal_length"
  }]


def exact_int8_example(
    lhs_shape: Union[int, Tuple[int, ...]],  #
    rhs_shape: Union[int, Tuple[int, ...]],
    lhs_share_stats_axes: List[int],
    rhs_share_stats_axes: List[int],
    lhs_use_quantized_variable: bool,
    rhs_use_quantized_variable: bool
) -> Tuple[aqt_config.AqtScheduleConfig, np.ndarray,
           aqt_config.AqtScheduleConfig, np.ndarray]:
  """Returns a pair of matrices and config to multiply exactly.

  Exact int8 testing examples are used across different tests in AQTp with
  different specs required for creating examples, in order to validate the
  correctness of quantization ops. This method creates examples tensors and
  their configs as specified by the arguments depending on use cases.

  Args:
    lhs_shape: Shape of a tensor to be created for LHS.
    rhs_shape: Shape of a tensor to be created for RHS.
    lhs_share_stats_axes: Contraction axes for LHS.
    rhs_share_stats_axes: Contraction axes for RHS.
    lhs_use_quantized_variable:
      For LHS, if true and in training mode, saves intermediate quantizations to
      user-provided variables. During inference, quantized variables
      are read from but not written to with new input values.
    rhs_use_quantized_variable:
      For RHS, same as above.
  Returns:
    AqtScheduleConfigs for both LHS and RHS, and their corresponding tensors.
  """

  rng = np.random.default_rng(1234)
  bits = 8

  # A subset of the range of numbers which can be preserved exactly.
  symmetric_uniform_range = 2**(bits - 1) - 1
  lo, hi = -symmetric_uniform_range, symmetric_uniform_range

  lhs = rng.integers(lo, hi, size=lhs_shape, dtype=np.int64)
  rhs = rng.integers(lo, hi, size=rhs_shape, dtype=np.int64)

  iqc = aqt_config.IntQuantConfig(bits=8)
  clip_bound = aqt_common.get_clip_bound(iqc)
  assert symmetric_uniform_range <= clip_bound

  sc = aqt_config.StatsConfig(
      ema_update_count=10,
      share_stats_axes=lhs_share_stats_axes,
      tpu_cross_replica_sum=False)
  tc = aqt_config.AqtTensorConfig(
      freeze_scale_at_begin=True,
      quant_config=iqc,
      calibration_config=aqt_config.CalibrationConfig(
          const_bound_coeff=clip_bound))

  lhs_config = aqt_config.AqtScheduleConfig(sc, [tc],
                                            lhs_use_quantized_variable)
  rhs_config = copy.deepcopy(lhs_config)
  rhs_config.stats_config.share_stats_axes = rhs_share_stats_axes
  rhs_config.use_quantized_variable = rhs_use_quantized_variable

  lhs = np.array(lhs, np.float32)
  rhs = np.array(rhs, np.float32)

  return lhs_config, lhs, rhs_config, rhs
