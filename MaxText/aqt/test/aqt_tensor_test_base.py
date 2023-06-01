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

"""AqtTensorQuantizer test base to be shared between TF and JAX."""

import copy
import itertools

from absl import logging
from absl.testing import parameterized
from aqt.common import aqt_common
from aqt.common import aqt_config
import numpy as np
import tensorflow.compat.v1 as tf


def f32(x):
  """Cast input array to f32."""
  return np.array(x, dtype=np.float32)


def make_quant_cases():
  for const, freeze in itertools.product([False, True], repeat=2):
    name = f"const_{const}_freeze_{freeze}"
    yield dict(
        const_calibration=const,
        freeze_scale_at_begin=freeze,
        testcase_name=name)


class AqtTensorQuantizerTest(tf.test.TestCase, parameterized.TestCase):
  """Base class for testing AqtTensorQuantizer.

  As TF and JAX AQTp have the same API and logic for AqtTensorQuantizer class,
  all testing cases are put in this base class and are shared between TF and JAX
  through inheritance. On TF and JAX AQTp sides, only few methods which are not
  framework agnostic should be implemented.
  """

  def make_tensor_quantizer(self, data_shape, config, name="tq"):
    raise NotImplementedError

  def update_quantizer(self, quant, sample, weight, event_count):
    raise NotImplementedError

  def to_quant(self, quant, x, train=True):
    raise NotImplementedError

  def get_quant_scale(self, quant, train=True):
    raise NotImplementedError

  def init(self):
    raise NotImplementedError

  def get_scale(self, quant):
    raise NotImplementedError

  def get_clip_range(self, quant):
    raise NotImplementedError

  def get_last_update(self, quant):
    raise NotImplementedError

  def get_quantized_variable(self, quant):
    raise NotImplementedError

  def quantize(self, x, quant, train):
    scale, inv_scale = self.get_quant_scale(quant, train)
    x = scale * x
    ix = self.to_quant(quant, x)
    return inv_scale * ix

  def test_validates_shape_config(self):
    """Checks that initializer validates the shape against config."""
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[1],
        tpu_cross_replica_sum=False,
    )
    config = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.IntQuantConfig(bits=8),
        calibration_config=aqt_config.CalibrationConfig(const_bound_coeff=1),
        freeze_scale_at_begin=True)
    config = aqt_config.AqtScheduleConfig(sc, [config])

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "share_stats_axes .* contain unknown"):
      self.make_tensor_quantizer(data_shape=[None, 3], config=config)

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "quantized variable with unknown"):
      config.use_quantized_variable = True
      self.make_tensor_quantizer(data_shape=[3, None], config=config)

  def test_validates_shape_update(self):
    """Check that update() validates input tensor shapes."""
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0],
        tpu_cross_replica_sum=False,
    )
    config = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.IntQuantConfig(bits=8),
        calibration_config=aqt_config.CalibrationConfig(const_bound_coeff=1),
        freeze_scale_at_begin=True)
    config = aqt_config.AqtScheduleConfig(sc, [config])

    quant = self.make_tensor_quantizer(data_shape=[None, 3], config=config)

    self.init()

    for good_shape in [(1, 3), (3, 3)]:
      x = np.ones(good_shape, dtype=np.float32)
      self.update_quantizer(quant, x, None, 0)

    for bad_shape in [(3,), (1, 3, 2)]:
      with self.assertRaisesRegex(ValueError, "shape .* compatible with"):
        x = np.ones(bad_shape, dtype=np.float32)
        self.update_quantizer(quant, x, None, 0)

  def test_assert_min_event(self):
    """Validates that event_count supplied cannot be int64.min."""
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False,
    )
    config = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.IntQuantConfig(bits=8),
        calibration_config=aqt_config.CalibrationConfig(const_bound_coeff=1),
        freeze_scale_at_begin=True)
    config = aqt_config.AqtScheduleConfig(sc, [config])

    x = np.ones((4, 4), dtype=np.float32)
    quant = self.make_tensor_quantizer(data_shape=x.shape, config=config)
    event_count = np.iinfo(np.int64).min
    with self.assertRaisesRegex(Exception,
                                "event_count cannot be"):
      self.update_quantizer(quant, x, None, event_count)

  def test_single_quant_simple(self):
    """Compares quantization to hand-computed example."""
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False,
    )
    config = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.IntQuantConfig(bits=3),
        calibration_config=aqt_config.CalibrationConfig(const_bound_coeff=7),
        freeze_scale_at_begin=True,
    )
    # representable values: -6, -4, -2, 0, 2, 4, 6

    x = f32([
        [0.99, 1.01, 1.99, 2.01],  #
        [2.99, 3.01, 3.99, 4.01],  #
        [4.99, 5.01, 5.99, 6.01],  #
        [6.99, 7.01, 7.99, 8.01],  #
        [-0.99, -1.01, -1.99, -2.01],  #
        [-2.99, -3.01, -3.99, -4.01],  #
        [-4.99, -5.01, -5.99, -6.01],  #
        [-6.99, -7.01, -7.99, -8.01],  #
    ])
    expected_output = f32([
        [0.00, 2.00, 2.00, 2.00],  #
        [2.00, 4.00, 4.00, 4.00],  #
        [4.00, 6.00, 6.00, 6.00],  #
        [6.00, 6.00, 6.00, 6.00],  #
        [-0.00, -2.00, -2.00, -2.00],  #
        [-2.00, -4.00, -4.00, -4.00],  #
        [-4.00, -6.00, -6.00, -6.00],  #
        [-6.00, -6.00, -6.00, -6.00],  #
    ])

    config = aqt_config.AqtScheduleConfig(sc, [config])
    quant = self.make_tensor_quantizer(data_shape=[8, 4], config=config)

    self.init()
    event_count = np.array(0, dtype=np.int64)
    self.update_quantizer(quant, x, np.full((1, 1), 1, dtype=np.float32),
                          event_count)
    qx = self.quantize(x, quant, True)
    self.assertAllEqual(self.get_scale(quant), np.full((1, 1), 0.5,
                                                       dtype=np.float32))
    self.assertAllEqual(qx, expected_output)

  @parameterized.named_parameters(make_quant_cases())
  def test_quant(self, const_calibration, freeze_scale_at_begin):
    """Tests basic TensorQuantizer behavior between update and to_quant."""
    bits = 8
    x_bound = 16.0

    calibration_config = aqt_config.CalibrationConfig(lp_dev_coeff=3)
    if const_calibration:
      calibration_config = aqt_config.CalibrationConfig(
          const_bound_coeff=x_bound)

    sc = aqt_config.StatsConfig(
        ema_update_count=1, share_stats_axes=[1], tpu_cross_replica_sum=False)
    config = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.IntQuantConfig(bits),
        calibration_config=calibration_config,
        freeze_scale_at_begin=freeze_scale_at_begin,
        begin_at_event=10,
        end_at_event=20,
    )
    config = aqt_config.AqtScheduleConfig(sc, [config])

    data_shape = [128, 256]
    has_quantized = False
    quant = self.make_tensor_quantizer(data_shape, config, "tq")
    self.init()
    last_scale = self.get_scale(quant)
    logging.info("scale l1 before loop: %s", np.abs(last_scale).sum())
    for (new_event_count, should_quantize) in [(9, False), (10, True),
                                               (19, True), (20, False)]:
      logging.info("loop: %s", (new_event_count, should_quantize))
      x = f32(np.random.uniform(low=-x_bound, high=x_bound, size=data_shape))
      bucket_count = 2.0**bits - 1
      bucket_size = 2 * x_bound / bucket_count
      new_event_count = np.array(new_event_count, dtype=np.int64)
      self.update_quantizer(quant,
                            x,
                            np.full((data_shape[0], 1), 1, dtype=np.float32),
                            new_event_count)
      logging.info("  scale l1: %s", np.abs(self.get_scale(quant)).sum())

      self.assertAllEqual(self.get_last_update(quant), new_event_count)

      scale, inv_scale = self.get_quant_scale(quant)
      x_scaled = scale * x

      ix = self.to_quant(quant, x_scaled)
      has_quantized |= should_quantize

      self.assertAllEqual(
          self.get_scale(quant).shape,
          self.get_clip_range(quant).shape)

      if should_quantize:
        # all quantized values are integers in a proper range
        self.assertAllEqual(ix, np.round(ix))
        self.assertAllLessEqual(np.abs(ix), 2**(bits - 1) - 1)
        bound = self.get_clip_range(quant)
        if const_calibration:
          self.assertAllEqual(bound, np.ones_like(bound) * x_bound)
        self.assertAllClose(bound, 127.5 / self.get_scale(quant))
      else:
        self.assertAllEqual(ix, x)
        bound = self.get_clip_range(quant)
        self.assertAllEqual(bound, np.zeros_like(bound))

      if const_calibration and has_quantized:
        self.assertAllEqual(self.get_scale(quant),
                            np.full((data_shape[0], 1), 127.5 / x_bound,
                                    dtype=np.float32))
      if not const_calibration:
        # does freeze_scale_at_begin has a proper effect
        curr_scale = self.get_scale(quant)
        scale_not_changed = (curr_scale == last_scale).all()
        frozen = should_quantize and freeze_scale_at_begin
        not_quantized = not should_quantize
        first_frozen_event = new_event_count == 10
        frozen &= not first_frozen_event  # First frozen event requires update.
        logging.info(
            "  scale_not_changed=%s,not_quantized=%s,frozen=%s,"
            "first_frozen_event=%s", scale_not_changed, not_quantized, frozen,
            first_frozen_event)
        self.assertEqual(scale_not_changed, not_quantized or frozen)
        last_scale = curr_scale

      qx = inv_scale * ix
      quant_noise = np.abs(qx - np.clip(x, -x_bound, x_bound))
      if should_quantize:
        if const_calibration:
          # Checking quant_noise is the core of this test.
          # its properties should hold in all configurations.
          # 1. max error should be half the bucket.
          self.assertAllLessEqual(
              np.max(quant_noise), bucket_size / 2)
          # 2. error size should be half of max error
          self.assertAllClose(
              np.mean(quant_noise),
              bucket_size / 4,
              atol=bucket_size / 200)
      else:
        self.assertAllEqual(quant_noise, np.zeros_like(quant_noise))

  def test_update_saves_variable(self):
    """Makes sure that update saves to quantized_variable."""
    iqc = aqt_config.IntQuantConfig(bits=8, preserve_zero=True)
    clip_bound = aqt_common.get_clip_bound(iqc)
    # Use a half of clip_bound for const_bound_coeff, which results in
    # scale=2.0. It will help to easily compute expected quantized values and
    # check if they are equal to actual quantized variable in this test.
    cc = aqt_config.CalibrationConfig(const_bound_coeff=clip_bound * 0.5)
    sc = aqt_config.StatsConfig(
        ema_update_count=1, share_stats_axes=[], tpu_cross_replica_sum=False)
    yes_q_config = aqt_config.AqtTensorConfig(
        freeze_scale_at_begin=True,
        quant_config=iqc,
        calibration_config=cc,
        end_at_event=1)

    # Create a quantizer with the first config quantized, the second not.
    iqc = aqt_config.IntQuantConfig(bits=9, preserve_zero=True)
    clip_bound = aqt_common.get_clip_bound(iqc)

    no_q_config = copy.deepcopy(yes_q_config)
    yes_q_config.end_at_event = 1
    no_q_config.quant_config = iqc
    no_q_config.calibration_config.const_bound_coeff = clip_bound
    no_q_config.begin_at_event = 1
    no_q_config.end_at_event = 2
    tensor_configs = [yes_q_config, no_q_config]
    config = aqt_config.AqtScheduleConfig(sc, tensor_configs, True)

    data_shape = (4, 2)
    quantizer = self.make_tensor_quantizer(data_shape=data_shape, config=config,
                                           name="quantizer")

    def update(sample, event_count):
      event_count = np.array(event_count, dtype=np.int64)
      self.update_quantizer(quantizer, sample, None, event_count)

    rng = np.random.default_rng(1234)
    x = rng.integers(-10, 10, size=data_shape, dtype=np.int64)
    x = np.array(x, dtype=np.float32)
    # Since all values in x are integers between -10 and 10, their quantized
    # values are just the result of scaling (scale=2.0), without clipping
    # (clip_bound [-127.5, 127.5]) and rounding.
    x_quantized = 2.0 * x

    self.init()

    z = np.zeros_like(self.get_quantized_variable(quantizer))
    self.assertAllEqual(self.get_quantized_variable(quantizer), z)

    # First config with 8 bits quantization active, variable should be updated.
    update(x, 0)
    self.assertAllEqual(self.get_quantized_variable(quantizer), x_quantized)

    # Second config with 9 bits quantization active, variable should not be
    # updated, since it is not compatible with int8.
    update(x * 2, 1)
    self.assertAllEqual(self.get_quantized_variable(quantizer), x_quantized)

    # No config, variable should not be updated.
    update(x * 2, 2)
    self.assertAllEqual(self.get_quantized_variable(quantizer), x_quantized)

  def test_none_weights(self):
    """Ensures semantics of None weights equal those of weights=1."""
    iqc = aqt_config.IntQuantConfig(bits=8, preserve_zero=True)
    cc = aqt_config.CalibrationConfig(l1_dev_coeff=2)
    sc = aqt_config.StatsConfig(
        ema_update_count=10,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False)
    tc = aqt_config.AqtTensorConfig(
        quant_config=iqc,
        calibration_config=cc,
        freeze_scale_at_begin=False)
    config = aqt_config.AqtScheduleConfig(sc, [tc], use_quantized_variable=True)

    rng = np.random.default_rng(1234)
    x = rng.normal(size=(3, 4)).astype(np.float32)
    q1 = self.make_tensor_quantizer(data_shape=x.shape, config=config,
                                    name="q1")
    q2 = self.make_tensor_quantizer(data_shape=x.shape, config=config,
                                    name="q2")
    weight = np.full(x.shape, 1.0, dtype=np.float32)

    self.init()

    ec = np.array(0, dtype=np.int64)
    self.update_quantizer(q1, x, None, ec)
    self.update_quantizer(q2, x, weight, ec)

    self.assertAllEqual(self.get_quantized_variable(q1),
                        self.get_quantized_variable(q2))

  def test_float_config_not_save_quantized_var(self):
    """Ensures FloatConfig does not save quantized variables."""
    float_tc = aqt_config.AqtTensorConfig(
        quant_config=aqt_config.FloatConfig(),
        calibration_config=aqt_config.CalibrationConfig(),
        freeze_scale_at_begin=True,
        begin_at_event=0,
        end_at_event=1)
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False)
    use_quantized_var = True
    config = aqt_config.AqtScheduleConfig(sc, [float_tc], use_quantized_var)

    rng = np.random.default_rng(1234)
    x = rng.normal(size=(3, 4)).astype(np.float32)
    quantizer = self.make_tensor_quantizer(data_shape=x.shape, config=config)

    self.init()

    event_count = np.array(0, dtype=np.int64)
    self.update_quantizer(quantizer, x, None, event_count)

    actual = self.get_quantized_variable(quantizer)
    expected = np.zeros_like(actual)
    self.assertAllEqual(actual, expected)

  def test_float_config_not_quantized(self):
    """Ensures an input is not quantized when it comes with FloatConfig."""
    iqc = aqt_config.IntQuantConfig(bits=3, preserve_zero=True)
    cc = aqt_config.CalibrationConfig(const_bound_coeff=7)
    int_tc = aqt_config.AqtTensorConfig(
        quant_config=iqc,
        calibration_config=cc,
        freeze_scale_at_begin=True,
        begin_at_event=1,
        end_at_event=2)
    float_tc = copy.deepcopy(int_tc)
    float_tc.quant_config = aqt_config.FloatConfig()
    float_tc.begin_at_event = 0
    float_tc.end_at_event = 1

    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False)

    tensor_configs = [float_tc, int_tc]
    config = aqt_config.AqtScheduleConfig(sc, tensor_configs, True)

    x = f32([
        [0.99, 1.01, 1.99, 2.01],  #
        [2.99, 3.01, 3.99, 4.01],  #
        [4.99, 5.01, 5.99, 6.01],  #
        [6.99, 7.01, 7.99, 8.01],  #
        [-0.99, -1.01, -1.99, -2.01],  #
        [-2.99, -3.01, -3.99, -4.01],  #
        [-4.99, -5.01, -5.99, -6.01],  #
        [-6.99, -7.01, -7.99, -8.01],  #
    ])
    x_quantized = f32([
        [0.00, 2.00, 2.00, 2.00],  #
        [2.00, 4.00, 4.00, 4.00],  #
        [4.00, 6.00, 6.00, 6.00],  #
        [6.00, 6.00, 6.00, 6.00],  #
        [-0.00, -2.00, -2.00, -2.00],  #
        [-2.00, -4.00, -4.00, -4.00],  #
        [-4.00, -6.00, -6.00, -6.00],  #
        [-6.00, -6.00, -6.00, -6.00],  #
    ])

    quantizer = self.make_tensor_quantizer(data_shape=x.shape, config=config,
                                           name="quantizer")

    self.init()

    def update_and_quantize(event_count):
      event_count = np.array(event_count, dtype=np.int64)
      self.update_quantizer(quantizer, x, None, event_count)
      qx = self.quantize(x, quantizer, True)
      return qx

    # At event_count=0, the input should not be quantized since the active
    # config is FloatConfig.
    qx = update_and_quantize(0)
    self.assertAllEqual(x, qx)

    # At event_count=1, the input should be quantized since the active config is
    # IntQuantConfig.
    qx = update_and_quantize(1)
    self.assertAllEqual(x_quantized, qx)

  @parameterized.parameters([{
      "use_quantized_variable": True
  }, {
      "use_quantized_variable": False
  }])
  def test_inference_config_index(self, use_quantized_variable):
    """Validate that the inference config index is used."""

    # Create a schedule where quantization behavior at times 0 and 1
    # differ. Then make sure that a dynamically switched inference
    # (with the inference index set to None) matches the appropriate
    # value for a statistically switched inference quantizer.

    iqc = aqt_config.IntQuantConfig(bits=8, preserve_zero=True)
    cc = aqt_config.CalibrationConfig(const_bound_coeff=2)
    int_tc = aqt_config.AqtTensorConfig(
        quant_config=iqc,
        calibration_config=cc,
        freeze_scale_at_begin=True,
        begin_at_event=1,
        end_at_event=2)
    float_tc = copy.deepcopy(int_tc)
    float_tc.quant_config = aqt_config.FloatConfig()
    float_tc.begin_at_event = 0
    float_tc.end_at_event = 1
    tensor_configs = [float_tc, int_tc]
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=[0, 1],
        tpu_cross_replica_sum=False)
    config_no_inference_index = aqt_config.AqtScheduleConfig(
        sc, tensor_configs, use_quantized_variable)
    config_inference0 = copy.deepcopy(config_no_inference_index)
    config_inference0.inference_config_index = 0
    config_inference1 = copy.deepcopy(config_no_inference_index)
    config_inference1.inference_config_index = 1

    data_shape = (3, 4)
    q_index_none = self.make_tensor_quantizer(
        data_shape, config=config_no_inference_index)
    q_index_0 = self.make_tensor_quantizer(
        data_shape, config=config_inference0, name="0")
    q_index_1 = self.make_tensor_quantizer(
        data_shape, config=config_inference1, name="1")

    self.init()

    # Note that because we only use constant coeff, updates don't
    # affect true scale values; only the config choice does.
    def update(event_count, quantizer):
      event_count = np.array(event_count, dtype=np.int64)
      x = np.ones(data_shape).astype(np.float32)
      self.update_quantizer(quantizer, x, None, event_count)

    # Update the indexed quantizers at least once to initialize.
    update(0, q_index_0)
    update(0, q_index_1)

    rng = np.random.default_rng(1234)

    def check_quant(q_index, train):
      """Checks q_index has the same behavior as q_index_none."""
      x = rng.normal(size=(3, 4)).astype(np.float32)
      scale_unindexed, inv_scale_unindexed = self.get_quant_scale(
          q_index_none, train=train)
      x_unindexed = scale_unindexed * x
      unindexed = self.to_quant(q_index_none, x_unindexed, train=train)
      scale_indexed, inv_scale_indexed = self.get_quant_scale(
          q_index, train=train)
      x_indexed = scale_indexed * x
      indexed = self.to_quant(q_index, x_indexed, train=train)

      self.assertAllEqual(unindexed, indexed)
      self.assertAllEqual(inv_scale_unindexed, inv_scale_indexed)

    update(0, q_index_none)
    check_quant(q_index_0, train=True)
    check_quant(q_index_0, train=False)

    # Even if we update the statically indexed one its behavior is the same at
    # inference time.
    update(1, q_index_0)
    check_quant(q_index_0, train=False)

    # Now update our dynamically indexed quantizer to time 1.
    update(1, q_index_none)
    check_quant(q_index_0, train=True)  # recall this is still at time 1.

    update(1, q_index_1)
    check_quant(q_index_1, train=True)
    check_quant(q_index_1, train=False)

  @parameterized.parameters([{
      "shared_stats_axes": [0],
  }, {
      "shared_stats_axes": [0, 1],
  }, {
      "shared_stats_axes": [0, 1, 2],
  }])
  def test_scale_and_inv_scale(self, shared_stats_axes):
    """Validates scale * inv_scale close to an identity tensor."""
    iqc = aqt_config.IntQuantConfig(bits=8, preserve_zero=True)
    cc = aqt_config.CalibrationConfig(l1_dev_coeff=2)
    sc = aqt_config.StatsConfig(
        ema_update_count=1,
        share_stats_axes=shared_stats_axes,
        tpu_cross_replica_sum=False)
    tc = aqt_config.AqtTensorConfig(
        quant_config=iqc,
        calibration_config=cc,
        freeze_scale_at_begin=False)
    config = aqt_config.AqtScheduleConfig(sc, [tc], use_quantized_variable=True)

    rng = np.random.default_rng(1234)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    quantizer = self.make_tensor_quantizer(x.shape, config)

    self.init()
    self.update_quantizer(quantizer, x, None, 0)

    scale, inv_scale = self.get_quant_scale(quantizer, train=True)

    actual = scale * inv_scale
    expected = np.ones_like(actual)

    self.assertAllClose(actual, expected)
