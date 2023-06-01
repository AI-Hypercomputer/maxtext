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
"""Quantized einsum.

Quantized einsum for the analogous :py:func:`tf.einsum`.

Note that only 2-argument einsum is supported. For details, see go/aqtp-einsum.
In a two argument einsum, we have an equation `lhs,rhs->out` where `lhs`, `rhs`,
and `out` are strings containing single-character axes labels. Note that a
tensor axis is a distinct notion from einsum axes labels (henceforth, "labels"),
since a tensor axis is a natural number indexing from 0 to one less than that
tensor's dimension, but labels can be repeated within a single tensor.

We thus define an important notion for quantization of einsum: contracting axes,
which are tensor axes whose labels are not present in the output.

For instance, in matrix multiply (ij,jk->ik), axes with label j are contracting.
In scaled diag (ii,->i), there are no contracting axes.
In sum of sums (i,j->), both inputs have their only axes contract.

Of course, if an axis is contracting, all axes across all inputs with the same
label are contracting.
"""

import string
from typing import Dict, Iterable, Optional, Tuple

from aqt.common import aqt_config
from aqt.common import aqt_config_utils
from aqt.tensorflow import aqt_ops_util
from aqt.tensorflow import aqt_tensor
import tensorflow.compat.v1 as tf

# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access


def _parse_equation(eq: str) -> Tuple[str, str, str]:
  """Parses a two-argument einsum equation.

  Args:
    eq: the two-argument einsum equation.

  Returns:
    A tuple `(lhs, rhs, out)` for the separate parts of the einsum equation.

  Raises:
    aqt_config.ConfigError: `eq` is not strictly in the form `lhs,rhs->out`
    where
    `lhs`, `rhs`, and `out` are strings of characters `[a-zA-Z]`. Thus, ellipses
    and whitspace in equations is not supported.
  """
  args, sep, out = eq.partition('->')
  if not sep:
    raise aqt_config.ConfigError(
        f'einsum equation ("{eq}") expected to have in/out delimiter "->"')

  num_commas = args.count(',')
  if num_commas != 1:
    raise aqt_config.ConfigError(
        f'einsum equation ("{eq}") expected to have 2 arguments, but '
        f'{num_commas+1} found (consider splitting your expression into '
        'multiple two-argument einsums, see go/aqtp-einsum)')

  lhs, sep, rhs = args.partition(',')
  assert sep

  for labels in [lhs, rhs, out]:
    for char in labels:
      if char not in string.ascii_letters:
        raise aqt_config.ConfigError(
            f'einsum equation ("{eq}") has illegal character "{char}"')

  return lhs, rhs, out


def _validate_shared_axes(
    lhs_config: aqt_config.StatsConfig,  # Prevent python auto-formatting.
    rhs_config: aqt_config.StatsConfig,
    lhs: str,
    rhs: str,
    out: str) -> None:
  """Validates that contracting axes have shared statistics."""
  contracting_labels = set(lhs + rhs) - set(out)

  axes_name_configs = [(lhs, 'lhs', lhs_config), (rhs, 'rhs', rhs_config)]
  for axes, name, config in axes_name_configs:
    shared_indices = set(config.share_stats_axes)
    for i, label in enumerate(axes):
      if label in contracting_labels and i not in shared_indices:
        raise aqt_config.ConfigError(
            f'axis {i} of {name} must be shared due to contraction')


def einsum(
    eq: str,  #
    lhs_quantizer: aqt_tensor.TensorQuantizer,
    lhs: tf.Tensor,
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    rhs: tf.Tensor,
    train: bool = True,
    **tf_einsum_kwargs) -> tf.Tensor:
  """Performs a quantized two-argument :py:func:`tf.einsum`.

  Args:
    eq: The einsum equation.
    lhs_quantizer: TensorQuantizer for lhs
    lhs: A `tf.Tensor`. Must be `float32`. The convolution lhs.
    rhs_quantizer: TensorQuantizer for rhs
    rhs: A `tf.Tensor`. Must have the same type as `lhs`. The convolution
      kernel.
    train: If false and `use_quantized_variable` in lhs_quantizer or
      rhs_quantizer, then this indicates `aqt_einsum` should use the quantized
      variable with the latest quantized, memorized from the most recent
      `TensorQuantizer.update()` in quantized operations rather than the float
      tensor input `lhs` or `rhs` provided to those operations at inference
      time.
    **tf_einsum_kwargs: Keyword arguments to pass onto `einsum`.

  Returns:
    A `float32` tensor conformal to what `tf.einsum` would return.

  Raises:
    aqt_config.ConfigError: if the equation is not a two-argument einsum,
    the quantization schedules between arguments are misaligned, or a
    contracting axis does not have shared statistics.
  """
  lhs_labels, rhs_labels, out_labels = _parse_equation(eq)
  _validate_shared_axes(
      lhs_quantizer.config.stats_config,  #
      rhs_quantizer.config.stats_config,
      lhs_labels,
      rhs_labels,
      out_labels)
  aqt_config_utils._validate_alignment(
      'lhs_quantizer.config.tensor_configs',  #
      lhs_quantizer.config.tensor_configs,
      'rhs_quantizer.config.tensor_configs',
      rhs_quantizer.config.tensor_configs)

  with tf.name_scope('AqtEinsum'):
    with tf.name_scope('get_quant_scale'):
      with tf.name_scope('lhs'):
        lhs_scale, lhs_inv_scale = lhs_quantizer._get_quant_scale(train)
      with tf.name_scope('rhs'):
        rhs_scale, rhs_inv_scale = rhs_quantizer._get_quant_scale(train)

    lhs = lhs_scale * lhs
    rhs = rhs_scale * rhs

    with tf.name_scope('to_quant'):
      with tf.name_scope('lhs'):
        lhs = lhs_quantizer._to_quant(lhs, train)
      with tf.name_scope('rhs'):
        rhs = rhs_quantizer._to_quant(rhs, train)

    # TODO(vladf): until tf.einsum supports int8 arguments, we need to cast
    # the quantized variables to a floating point format.
    if lhs_quantizer.config.use_quantized_variable and not train:
      lhs = tf.cast(lhs_quantizer.quantized_variable.read_value(), tf.float32)
    if rhs_quantizer.config.use_quantized_variable and not train:
      rhs = tf.cast(rhs_quantizer.quantized_variable.read_value(), tf.float32)

    with tf.name_scope('einsum'):
      out = tf.einsum(eq, lhs, rhs, **tf_einsum_kwargs)

    with tf.name_scope('inv_scale'):
      assert len(lhs_inv_scale.shape) == len(lhs.shape)
      assert len(rhs_inv_scale.shape) == len(rhs.shape)
      inv_scale = tf.einsum(eq, lhs_inv_scale, rhs_inv_scale,
                            **tf_einsum_kwargs)
      return out * inv_scale


class Einsum:
  """Encapsulates a quantized einsum op and calibration state."""

  def __init__(
      self,
      eq: str,
      config: aqt_config.AqtEinsumConfig,
      lhs_shape: Iterable[Optional[int]],
      rhs_shape: Iterable[Optional[int]],
      name: str = 'einsum',
      lhs_name: str = 'lhs',
      rhs_name: str = 'rhs',
  ):
    """Creates a Einsum instance.

    This encapsulates the state necessary for quantizing both
    of the einsum arguments.

    Args:
      eq: the einsum equation.
      config: an AqtEinsumConfig
      lhs_shape: shape of the lhs tensor
      rhs_shape: shape of the rhs tensor
      name: variable scope for all variables to be created.
      lhs_name: scope for left hand side variables only.
      rhs_name: scope for right hand side variables only.
    """
    self.eq = eq
    self.name = name
    self.lhs_name = lhs_name
    self.rhs_name = rhs_name

    with tf.variable_scope(name):
      self.lhs_quantizer = aqt_tensor.TensorQuantizer(
          lhs_shape, config.lhs, name=lhs_name)
      self.rhs_quantizer = aqt_tensor.TensorQuantizer(
          rhs_shape, config.rhs, name=rhs_name)

  def update_lhs(
      self,
      x: tf.Tensor,
      weights: tf.Tensor,
      event_count: tf.Tensor,
  ) -> tf.Operation:
    """Updates variables for an observation of the lhs.

    Updating variables for an argument updates the statistics
    for a new input to account for incremental observations of
    a tensor's entries' magnitudes.

    This also updates the scales, if they were set according to
    a previous calibration config and now we've moved on to
    a new event associated with a different calibration configuration
    in the schedule.

    Args:
      x: a tensor for the observation of an lhs input
      weights: a weight matrix broadcastable to x, representing how much weight
        the corresponding axes should have on statistics associated with
        quantizing that dimension.
      event_count: the event of the observation

    Returns:
      The tf.Operation corresponding to the update
    """
    return self.lhs_quantizer.update(x, weights, event_count)

  def update_rhs(
      self,
      x: tf.Tensor,
      weights: tf.Tensor,
      event_count: tf.Tensor,
  ) -> tf.Operation:
    """Computes analogue of update_lhs, but for rhs."""
    return self.rhs_quantizer.update(x, weights, event_count)

  def apply(
      self,
      lhs: tf.Tensor,
      rhs: tf.Tensor,
      train: bool = True,
      **tf_einsum_kwargs
  ) -> tf.Tensor:
    """Generates a pure quantized einsum op.

    Make sure that `apply` is called within the context of any updates
    to statistics used for calibration you'd like to happen before the
    op.

    Args:
      lhs: a float32 tensor for the left hand side
      rhs: a float32 tensor for the right hand side
      train: whether to generate the training or serving graph
      **tf_einsum_kwargs: Keyword arguments to pass onto `einsum`.

    Returns:
      A tf.Tensor generated from possibly quantizing lhs and rhs
      with clip bounds derived from the current quantizer statistics.
    """

    return einsum(
        self.eq,
        self.lhs_quantizer,
        lhs,
        self.rhs_quantizer,
        rhs,
        train,
        **tf_einsum_kwargs)

  def diagnostics(
      self,
      lhs: tf.Tensor,
      rhs: tf.Tensor,
  ) -> Dict[str, tf.Tensor]:
    """Returns a dictionary from keys to diagnostic tensors.

    Args:
      lhs: lhs argument to self.Apply, used for deriving diangostics relative to
        a given input.
      rhs: as above, but for rhs

    Returns:
      A dictionary with various quantization-related diagnostics,
      whose string keys are prefixed by self.name/self.{lhs,rhs}_name.
    """
    return aqt_ops_util.diagnostics(self, lhs, rhs)
