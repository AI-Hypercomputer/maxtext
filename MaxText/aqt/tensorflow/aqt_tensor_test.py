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
"""Tests for aqt_tensor."""

from typing import Set

from absl.testing import absltest
from aqt.common import aqt_config
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_stats_test_base
from aqt.test import aqt_tensor_test_base
import numpy as np
import tensorflow.compat.v1 as tf


def f32(x):
  """Cast input array to f32."""
  return tf.cast(x, dtype=tf.float32)


def i64(x):
  """Cast input to i64."""
  return tf.cast(x, dtype=tf.int64)


class StatsTest(aqt_stats_test_base.StatsTest):
  """Tests for Stats class.

  Refer to aqt_stats_test_base.StatsTest for more details.
  """

  def set_stats(self, data_shape, config):
    self._stats = aqt_tensor.Stats(
        data_shape=data_shape,
        config=config,
        get_variable=aqt_tensor.default_get_variable)
    tf.global_variables_initializer().run()

  def update(self, sample, weight):
    self._stats.update(f32(sample), f32(weight)).run()

  def get_sum_of_ones(self):
    return self._stats.calibration_variables()['sum_of_ones']

  def get_sum_of_vals(self):
    return self._stats.calibration_variables()['sum_of_vals']

  def get_max_of_abs_vals(self):
    with self.cached_session() as sess, sess.as_default():
      return self._stats.max_dev().eval()

  def get_sum_of_l1_vals(self):
    return self._stats.calibration_variables()['sum_of_l1_vals']

  def get_sum_of_lp_vals(self):
    return self._stats.calibration_variables()['sum_of_lp_vals']

  def set_ema_update_count(self, ema_update_count):
    self._stats._ema_update_count = ema_update_count

  def mean(self):
    return self._stats.mean()

  def max_dev(self):
    return self._stats.max_dev()

  def l1_dev(self):
    return self._stats.l1_dev()

  def lp_dev(self):
    return self._stats.lp_dev()

  def bound(self, calibration_config):
    return self._stats.bound(calibration_config)


class AqtTensorQuantizerTest(aqt_tensor_test_base.AqtTensorQuantizerTest):
  """Tests for AqtTensorQuantizer class.

  Refer to aqt_tensor_test_base.AqtTensorQuantizerTest for more details.
  """

  def make_tensor_quantizer(self, data_shape, config, name='tq'):
    return aqt_tensor.TensorQuantizer(
        data_shape=data_shape, config=config, name=name)

  def update_quantizer(self, quant, sample, weight, event_count):
    with self.cached_session() as sess, sess.as_default():
      quant.update(f32(sample), weight, i64(event_count)).run()

  def to_quant(self, quant, x, train=True):
    with self.cached_session() as sess, sess.as_default():
      return quant._to_quant(x, train).eval()

  def get_quant_scale(self, quant, train=True):
    with self.cached_session() as sess, sess.as_default():
      scale, inv_scale = quant._get_quant_scale(train)
      return scale.eval(), inv_scale.eval()

  def init(self):
    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()

  def get_scale(self, quant):
    with self.cached_session() as sess, sess.as_default():
      return quant._scale.eval()

  def get_clip_range(self, quant):
    with self.cached_session() as sess, sess.as_default():
      return quant.clip_range().eval()

  def get_last_update(self, quant):
    with self.cached_session() as sess, sess.as_default():
      return quant._last_update.eval()

  def get_quantized_variable(self, quant):
    with self.cached_session() as sess, sess.as_default():
      return quant.quantized_variable.eval()

  def test_pure_quant_methods(self):
    """Validates the to_quant methods are pure functions of scale and input."""
    # Check the generated tensorflow graph to make sure that statistics are not
    # used and that the output is deterministic.

    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False,
    )
    config = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.IntQuantConfig(bits=8),
        calibration_config=aqt_config.CalibrationConfig(const_bound_coeff=1),
        freeze_scale_at_begin=False)
    config = aqt_config.AqtScheduleConfig(sc, [config])

    rng = np.random.default_rng(1234)
    x = rng.integers(-10, 10, size=(4, 4)).astype(np.float32)
    with tf.Graph().as_default():
      quant = aqt_tensor.TensorQuantizer(data_shape=x.shape, config=config)
      event_count = tf.constant(0, dtype=tf.int64)
      with self.cached_session():
        tf.global_variables_initializer().run()
        quant.update(f32(x), None, event_count).run()

      repeats = []
      for _ in range(10):
        scale, inv_scale = quant._get_quant_scale(train=True)
        x = scale * x
        ix = quant._to_quant(x, train=True)
        qx = inv_scale * ix
        repeats.append(qx)

      with self.cached_session():
        for l, r in zip(repeats, repeats[1:]):
          self.assertAllEqual(l, r)

    # Extract subgraph attached to `qx`.
    actual_varnames = extract_referenced_variables(qx)
    self.assertLen(actual_varnames, 2)
    expected_varnames = {quant._inv_scale.name, quant._scale.name}
    self.assertEqual(actual_varnames, expected_varnames)

  def test_alternative_variable_creation(self):
    """Makes sure that tensor quantization works with different GetVariable."""
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False,
    )
    config = aqt_config.AqtTensorConfig(
        aqt_config.IntQuantConfig(bits=8,),
        aqt_config.CalibrationConfig(l1_dev_coeff=2),
        freeze_scale_at_begin=False)
    config = aqt_config.AqtScheduleConfig(sc, [config])

    def alt_get_var(name, shape, dtype, init):
      init_fn = lambda: tf.constant(init, shape=shape, dtype=dtype)
      return tf.Variable(
          initial_value=init_fn,
          trainable=False,
          shape=shape,
          name=name,
          use_resource=True)

    rng = np.random.default_rng(1234)
    x = rng.integers(-10, 10, size=(4, 4)).astype(np.float32)
    with tf.Graph().as_default():
      var_quant = aqt_tensor.TensorQuantizer(
          x.shape, config, name='default_vars')
      alt_quant = aqt_tensor.TensorQuantizer(
          x.shape, config, name='new_vars', get_variable=alt_get_var)

      var_update, alt_update = tf.no_op(), tf.no_op()

      for i in range(1, 5):
        # Simulate multiple updates in a row.
        event_count = tf.constant(0, dtype=tf.int64)
        y = x * i / 4

        with tf.control_dependencies([var_update]):
          var_update = var_quant.update(f32(y), None, event_count)
        with tf.control_dependencies([alt_update]):
          alt_update = alt_quant.update(f32(y), None, event_count)

        with tf.control_dependencies([var_update]):
          scale, inv_scale = var_quant._get_quant_scale(train=True)
          y_scaled = scale * y
          ix = var_quant._to_quant(y_scaled, train=True)
          var_qx = inv_scale * ix

        with tf.control_dependencies([alt_update]):
          scale, inv_scale = alt_quant._get_quant_scale(train=True)
          y_scaled = scale * y
          ix = alt_quant._to_quant(y_scaled, train=True)
          alt_qx = inv_scale * ix

        with self.cached_session():
          tf.global_variables_initializer().run()
          self.assertAllEqual(var_qx, alt_qx)


def extract_referenced_variables(t: tf.Tensor) -> Set[str]:
  """Returns a set of all referenced variable names in a tensor's graph."""

  tensors = {t}

  def dfs(op):
    for tensor in op.inputs:
      if tensor not in tensors:
        tensors.add(tensor)
        dfs(tensor.op)
    for control_op in op.control_inputs:
      dfs(control_op)

  dfs(t.op)

  variables = [
      v for v in tensors
      if v.op.type == 'VarHandleOp' or v.op.type == 'VariableV2'
  ]

  return {v.name for v in variables}


if __name__ == '__main__':
  absltest.main()
