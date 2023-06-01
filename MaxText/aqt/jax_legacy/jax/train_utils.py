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

"""Util functions for training, which can be shared across models."""
from absl import logging
from aqt.jax_legacy.jax import quant_config


def should_quantize_weights(weight_quant_start_step: int, step: int) -> bool:
  return step >= weight_quant_start_step


def should_update_bounds(activation_bound_update_freq: int,
                         activation_bound_start_step: int, step: int) -> bool:
  """Returns whether activation bounds should be updated.

  Args:
    activation_bound_update_freq: How frequently to update bounds after the
      initial bounds update. A value of '-1' indicates to not update the bounds
      after the first update.
    activation_bound_start_step: The first step to update bounds on. '-1'
      indicates to never update bounds.
    step: The current training step.

  Returns:
    Boolean indicating whether to update the bounds on the current step.
  """
  if activation_bound_start_step < -1:
    raise ValueError("Start step must be >= -1.")
  if activation_bound_update_freq < -1 or activation_bound_update_freq == 0:
    raise ValueError("Update frequency must be a positive integer or -1.")
  steps_since_start = step - activation_bound_start_step
  if activation_bound_start_step == -1 or steps_since_start < 0:
    return False
  if activation_bound_update_freq == -1:
    return steps_since_start == 0
  else:
    return steps_since_start % activation_bound_update_freq == 0


def update_sparsity_mask(sparsity_start_step: int, sparsity_update_freq: int,
                         step: int) -> bool:
  """Returns whether sparsity mask should be updated.

  Args:
    sparsity_start_step: The first training step to start applying sparsity.
      Setting start step -1 indicates to not apply sparsity at any step.
    sparsity_update_freq: How frequently to update sparsity. Setting frequency 0
      indicates not to update again after the given step.
    step: The current training step.

  Returns:
    Boolean indicating whether to update the sparsity mask on the current step.
  """
  if sparsity_start_step < -1:
    raise ValueError("Start step must be >= -1.")
  if sparsity_update_freq < 0:
    raise ValueError("Update frequency must be a positive integer or 0.")
  steps_since_start = step - sparsity_start_step
  if sparsity_start_step == -1 or steps_since_start < 0:
    return False
  if sparsity_update_freq == 0:
    return steps_since_start == 0
  else:
    return steps_since_start % sparsity_update_freq == 0


# pylint: disable=g-doc-args
def get_dynamic_context_for_step(
    *,
    activation_bound_update_freq: int,
    activation_bound_start_step: int,
    step: int,
    collect_acts_stats: bool,
    prefer_int8_to_int32_dot: bool,
    sparsity_start_step: int,
    sparsity_update_freq: int,
) -> quant_config.DynamicContext:
  """Returns correct quantization context for a given step.

  Args:
    activation_bound_update_freq: How frequently to update bounds after the
      initial bounds update. A value of '-1' indicates to not update the bounds
      after the first update.
    activation_bound_start_step: The first step to update bounds on. '-1'
      indicates to never update bounds.
    step: The current training step.
    collect_acts_stats: Whether to collect activation statistics.
    prefer_int8_to_int32_dot: Whether to feed lax.dot inputs with an int8 dtype
      and accumulate to int32.

  Returns:
    A quant_config.DynamicContext instance.
  """
  update_bounds = should_update_bounds(
      activation_bound_start_step=activation_bound_start_step,
      activation_bound_update_freq=activation_bound_update_freq,
      step=step)
  apply_sparsity = False
  if sparsity_start_step >= 0:
    apply_sparsity = step >= sparsity_start_step
  update_weight_sparsity = update_sparsity_mask(sparsity_start_step,
                                                sparsity_update_freq, step)
  update_act_sparsity = update_sparsity_mask(sparsity_start_step,
                                             sparsity_update_freq, step)
  num_update_sparsity = 0.0
  if sparsity_update_freq == 0:
    logging.warning("Sparsity mask updates only once.")
  else:
    if step >= sparsity_start_step:
      num_update_sparsity = int(
          (step - sparsity_start_step) / sparsity_update_freq)
  # TODO(ayazdan): Relax this parameter to enable more iterations of decaying.
  num_update_sparsity = min(16, num_update_sparsity)
  quantize_acts = step >= activation_bound_start_step
  return quant_config.DynamicContext(
      update_bounds=update_bounds,
      quantize_acts=quantize_acts,
      apply_sparsity=apply_sparsity,
      update_weight_sparsity=update_weight_sparsity,
      update_act_sparsity=update_act_sparsity,
      num_update_sparsity=num_update_sparsity,
      collect_acts_stats=collect_acts_stats,
      prefer_int8_to_int32_dot=prefer_int8_to_int32_dot,
  )
