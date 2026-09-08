# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MaxText convert_utils functions."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.integration.vllm.convert_utils import (
    _interleave_moe_weights,
    get_host_rss_mb,
    pad_to_tpu_lanes,
    resolve_rollout_tp,
)
from maxtext.integration.vllm.moe_padding import TPU_V5P_SUBCORE_LANE_SIZE

pytestmark = [pytest.mark.post_training]


class ConvertUtilsTest(unittest.TestCase):
  """Unit tests for convert_utils utilities."""

  @pytest.mark.cpu_only
  def test_pad_to_tpu_lanes(self):
    self.assertEqual(pad_to_tpu_lanes(0), 0)
    self.assertEqual(pad_to_tpu_lanes(1), 128)
    self.assertEqual(pad_to_tpu_lanes(127), 128)
    self.assertEqual(pad_to_tpu_lanes(128), 128)
    self.assertEqual(pad_to_tpu_lanes(129), 256)
    self.assertEqual(pad_to_tpu_lanes(256), 256)

  @pytest.mark.cpu_only
  def test_interleave_moe_weights_128_lane(self):
    # Shape: (1, 2, 256) -> wi_0 and wi_1 each have 2 shards of size 128 along axis 2
    # target shape: (1, 2, 512), n_shards = 2
    lane_size = 128
    gate_shard0 = np.arange(1000, 1000 + lane_size, dtype=np.float32)
    gate_shard1 = np.arange(2000, 2000 + lane_size, dtype=np.float32)
    up_shard0 = np.arange(3000, 3000 + lane_size, dtype=np.float32)
    up_shard1 = np.arange(4000, 4000 + lane_size, dtype=np.float32)

    wi_0 = np.concatenate([gate_shard0, gate_shard1]).reshape(1, 1, 256)
    wi_1 = np.concatenate([up_shard0, up_shard1]).reshape(1, 1, 256)

    tgt_shape = (1, 1, 512)
    interleaved = _interleave_moe_weights(
        jnp.array(wi_0),
        jnp.array(wi_1),
        tgt_shape=tgt_shape,
        n_shards=2,
        axis=2,
        lane_size=lane_size,
    )

    out = np.array(interleaved).reshape(-1)
    # Shard 0: gate_shard0 (128) followed by up_shard0 (128)
    np.testing.assert_array_equal(out[:128], gate_shard0)
    np.testing.assert_array_equal(out[128:256], up_shard0)
    # Shard 1: gate_shard1 (128) followed by up_shard1 (128)
    np.testing.assert_array_equal(out[256:384], gate_shard1)
    np.testing.assert_array_equal(out[384:512], up_shard1)

  @pytest.mark.cpu_only
  def test_interleave_moe_weights_fallback(self):
    # When target_chunk_size % lane_size != 0, fallback to standard concatenation
    gate = np.arange(100, dtype=np.float32).reshape(1, 1, 100)
    up = np.arange(100, 200, dtype=np.float32).reshape(1, 1, 100)

    tgt_shape = (1, 1, 200)
    result = _interleave_moe_weights(
        jnp.array(gate),
        jnp.array(up),
        tgt_shape=tgt_shape,
        n_shards=2,
        axis=2,
        lane_size=128,  # 50 % 128 != 0
    )

    out = np.array(result).reshape(-1)
    # Shard 0 (size 50 from gate, 50 from up)
    np.testing.assert_array_equal(out[:50], np.arange(0, 50))
    np.testing.assert_array_equal(out[50:100], np.arange(100, 150))
    # Shard 1 (size 50 from gate, 50 from up)
    np.testing.assert_array_equal(out[100:150], np.arange(50, 100))
    np.testing.assert_array_equal(out[150:200], np.arange(150, 200))

  @pytest.mark.cpu_only
  def test_resolve_rollout_tp_conflict_raises(self):
    cfg = SimpleNamespace(
        cluster=SimpleNamespace(rollout_tensor_parallelism=4)
    )
    with mock.patch.dict(os.environ, {"ROLLOUT_TENSOR_PARALLELISM": "2"}):
      with self.assertRaisesRegex(ValueError, "Rollout TP mismatch"):
        resolve_rollout_tp(cfg)

  @pytest.mark.cpu_only
  def test_resolve_rollout_tp_success(self):
    cfg = SimpleNamespace(
        cluster=SimpleNamespace(rollout_tensor_parallelism=4)
    )
    with mock.patch.dict(os.environ, {"ROLLOUT_TENSOR_PARALLELISM": "4"}):
      self.assertEqual(resolve_rollout_tp(cfg), 4)

  @pytest.mark.cpu_only
  def test_get_host_rss_mb(self):
    rss = get_host_rss_mb()
    self.assertIsInstance(rss, float)
    self.assertGreater(rss, 0.0)


if __name__ == "__main__":
  unittest.main()
