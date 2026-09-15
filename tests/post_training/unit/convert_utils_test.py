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
import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.integration.vllm.convert_utils import (
    _interleave_moe_weights,
    compute_padded_moe_mlp_dim,
    DEFAULT_TPU_NUM_LANES,
    is_verify_weights_enabled,
    pad_to_tpu_lanes,
    resolve_prefuse_moe_weights,
    resolve_rollout_tp,
)

pytestmark = [pytest.mark.post_training]


class ConvertUtilsTest(unittest.TestCase):
  """Unit tests for convert_utils utilities."""

  def test_pad_to_tpu_lanes(self):
    self.assertEqual(pad_to_tpu_lanes(0), 0)
    self.assertEqual(pad_to_tpu_lanes(1), 128)
    self.assertEqual(pad_to_tpu_lanes(127), 128)
    self.assertEqual(pad_to_tpu_lanes(128), 128)
    self.assertEqual(pad_to_tpu_lanes(129), 256)
    self.assertEqual(pad_to_tpu_lanes(256), 256)

  def test_interleave_moe_weights_128_lane(self):
    # Shape: (1, 2, 256) -> wi_0 and wi_1 each have 2 shards of size 128 along axis 2
    # target shape: (1, 2, 512), n_shards = 2
    lane_size = 128
    gate_shard0 = np.arange(1000, 1000 + lane_size, dtype=np.float32)
    gate_shard1 = np.arange(2000, 2000 + lane_size, dtype=np.float32)
    up_shard0 = np.arange(3000, 3000 + lane_size, dtype=np.float32)
    up_shard1 = np.arange(4000, 4000 + lane_size, dtype=np.float32)

    wi_0 = np.concatenate([gate_shard0, gate_shard1]).reshape((1, 1, 256))
    wi_1 = np.concatenate([up_shard0, up_shard1]).reshape((1, 1, 256))

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

  def test_interleave_moe_weights_alternates_multiple_lanes(self):
    """The case that distinguishes the lane layout from plain concatenation.

    With `target_chunk_size == lane_size` there is one lane per shard, so the output
    is byte-identical to the old `concatenate` path and the test proves nothing. Two
    lanes per shard is the first shape where GMM_v2's `deinterleave_lane` ordering
    (`gate_c0, up_c0, gate_c1, up_c1`) actually differs from (`gate_all, up_all`).
    """
    lane_size = 4  # Stands in for 128; the layout logic is lane-size agnostic.
    n_shards = 2
    num_lanes = 2
    chunk = lane_size * num_lanes  # 8 per shard, per half
    half = chunk * n_shards  # 16
    gate = np.arange(0, half, dtype=np.float32).reshape(1, 1, half)
    up = np.arange(100, 100 + half, dtype=np.float32).reshape(1, 1, half)

    out = np.array(
        _interleave_moe_weights(
            jnp.array(gate),
            jnp.array(up),
            tgt_shape=(1, 1, 2 * half),
            n_shards=n_shards,
            axis=2,
            lane_size=lane_size,
        )
    ).reshape(-1)

    g, u = gate.reshape(-1), up.reshape(-1)
    expected = []
    for shard in range(n_shards):
      base = shard * chunk
      for lane in range(num_lanes):
        lo = base + lane * lane_size
        expected.extend(g[lo : lo + lane_size])
        expected.extend(u[lo : lo + lane_size])
    np.testing.assert_array_equal(out, np.array(expected, dtype=np.float32))

    # And it is genuinely not the concatenated layout.
    concatenated = np.concatenate([g[:chunk], u[:chunk], g[chunk:], u[chunk:]])
    self.assertFalse(np.array_equal(out, concatenated))

  def test_compute_padded_moe_mlp_dim_result_is_always_kernel_valid(self):
    """The padded dim must satisfy the divisibility rule that triggered padding.

    The predecessor (`next_power_of_two` growth in adapter.py) grew until
    `padded // tp >= 2 * lanes` while the *trigger* was `(dim // tp) % (2 * lanes)`.
    Those predicates diverge whenever `moe_mlp_tp_size` is not a power of two, so it
    returned dims the GMM_v2 kernel rejects. Assert the invariant directly.
    """
    lanes = DEFAULT_TPU_NUM_LANES
    for tp in (1, 2, 3, 4, 5, 6, 8, 12):
      for dim in range(256, 6000, 64):
        padded = compute_padded_moe_mlp_dim(dim, tp, lanes)
        self.assertGreaterEqual(padded, dim, f"shrank dim={dim} tp={tp}")
        self.assertEqual(
            (padded // tp) % (2 * lanes),
            0,
            f"dim={dim} tp={tp} -> {padded} violates the GMM_v2 alignment rule",
        )

  def test_compute_padded_moe_mlp_dim_leaves_aligned_dims_alone(self):
    lanes = DEFAULT_TPU_NUM_LANES
    self.assertEqual(compute_padded_moe_mlp_dim(1024, 4, lanes), 1024)
    self.assertEqual(compute_padded_moe_mlp_dim(2560, 2, lanes), 2560)
    self.assertIsNone(compute_padded_moe_mlp_dim(None, 4, lanes))

  def test_is_verify_weights_enabled(self):
    orig = os.environ.get("VERIFY_WEIGHTS")
    try:
      os.environ["VERIFY_WEIGHTS"] = "true"
      self.assertTrue(is_verify_weights_enabled())
      os.environ["VERIFY_WEIGHTS"] = "false"
      self.assertFalse(is_verify_weights_enabled())
      os.environ.pop("VERIFY_WEIGHTS", None)
      self.assertFalse(is_verify_weights_enabled())
    finally:
      if orig is not None:
        os.environ["VERIFY_WEIGHTS"] = orig
      else:
        os.environ.pop("VERIFY_WEIGHTS", None)

  def test_resolve_prefuse_moe_weights(self):
    self.assertTrue(resolve_prefuse_moe_weights(None, prefuse_moe_weights=True))
    self.assertFalse(resolve_prefuse_moe_weights(None, prefuse_moe_weights=False))

    class DummyConfig:
      prefuse_moe_weights = False
      rollout_backend = "vllm"

    self.assertFalse(resolve_prefuse_moe_weights(DummyConfig()))

  def test_resolve_rollout_tp(self):
    self.assertEqual(resolve_rollout_tp(None, tp=4), 4)

    class DummyConfig:
      rollout_tensor_parallelism = 2

    self.assertEqual(resolve_rollout_tp(DummyConfig()), 2)


if __name__ == "__main__":
  unittest.main()
