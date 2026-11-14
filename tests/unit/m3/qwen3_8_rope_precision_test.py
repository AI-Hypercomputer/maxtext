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

"""Precision tests for `Qwen3_8RotaryEmbedding` high-theta `inv_freq` precomputation."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.models.qwen3_8_flash_next.modeling_qwen3_8_flash_next import Qwen3_8RotaryEmbedding


def _float32_ulp_diff(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
  """Returns elementwise ULP distance between two finite float32 arrays of identical sign."""
  a_bits = np.asarray(actual, dtype=np.float32).view(np.int32)
  b_bits = np.asarray(expected, dtype=np.float32).view(np.int32)
  return np.abs(a_bits.astype(np.int64) - b_bits.astype(np.int64))


class Qwen3_8RotaryEmbeddingPrecisionTest(absltest.TestCase):
  """Verifies host `float64 -> float32` `inv_freq` precision for `rope_theta = 10_000_000.0`."""

  def test_inv_freq_and_long_positions_match_fp64_reference_within_one_ulp(self):
    rope_theta = 10_000_000.0
    head_dim = 256
    partial_rotary_factor = 0.25  # rotary_dim = 64
    rotary_dim = 64
    emb = Qwen3_8RotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        rope_theta=rope_theta,
        mrope_section=(11, 11, 10),
    )

    # Static `_inv_freq` tuple must add zero `nnx.Variable` leaves to module state.
    self.assertEmpty(jax.tree.leaves(nnx.state(emb)))

    inv_freq_ref64 = 1.0 / (
        np.float64(rope_theta) ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / np.float64(rotary_dim))
    )
    inv_freq_ref32 = inv_freq_ref64.astype(np.float32)
    actual_inv_freq = np.asarray(emb.inv_freq, dtype=np.float32)

    inv_freq_ulps = _float32_ulp_diff(actual_inv_freq, inv_freq_ref32)
    self.assertLessEqual(int(inv_freq_ulps.max()), 1)

    for pos in (2047, 8191):
      angles_actual = np.float32(pos) * actual_inv_freq
      angles_ref32 = (np.float64(pos) * inv_freq_ref64).astype(np.float32)
      angle_ulps = _float32_ulp_diff(angles_actual, angles_ref32)
      self.assertLessEqual(int(angle_ulps.max()), 1, f"pos={pos} max_ulp={angle_ulps.max()}")

      dummy_x = jnp.ones((1, 1, 1, rotary_dim), dtype=jnp.float32)
      pos_3d = jnp.full((3, 1, 1), pos, dtype=jnp.int32)
      cos_out, sin_out = emb(dummy_x, pos_3d)
      # PyTorch stores `inv_freq` as a float32 buffer cast from float64, then computes `pos.float() * inv_freq`.
      pt_angles32 = np.float32(pos) * inv_freq_ref32
      full_pt_angles32 = np.concatenate([pt_angles32, pt_angles32], axis=-1)
      cos_ulps = _float32_ulp_diff(np.asarray(cos_out[0, 0]), np.cos(full_pt_angles32).astype(np.float32))
      sin_ulps = _float32_ulp_diff(np.asarray(sin_out[0, 0]), np.sin(full_pt_angles32).astype(np.float32))
      self.assertLessEqual(int(cos_ulps.max()), 2, f"pos={pos} cos_max_ulp={cos_ulps.max()}")
      self.assertLessEqual(int(sin_ulps.max()), 2, f"pos={pos} sin_max_ulp={sin_ulps.max()}")
      np.testing.assert_allclose(
          np.asarray(cos_out[0, 0]), np.cos(full_pt_angles32).astype(np.float32), rtol=0, atol=1e-6
      )
      np.testing.assert_allclose(
          np.asarray(sin_out[0, 0]), np.sin(full_pt_angles32).astype(np.float32), rtol=0, atol=1e-6
      )

    # Negative controls:
    # (1) Naive float32 `theta ** (arange / dim)` double-rounds on host, and
    # (2) TPU XLA's `exp2(exponent * log2(theta))` float32 power lowering drifts
    #     by multiple ULPs on high `rope_theta = 1e7`, amplifying phase error at long positions.
    exponents32 = np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    naive_host_f32 = np.float32(1.0) / (np.float32(rope_theta) ** exponents32)
    tpu_exp2_log2_f32 = np.float32(1.0) / np.exp2(exponents32 * np.log2(np.float32(rope_theta))).astype(np.float32)

    naive_ulps = _float32_ulp_diff(naive_host_f32, inv_freq_ref32)
    tpu_ulps = _float32_ulp_diff(tpu_exp2_log2_f32, inv_freq_ref32)
    self.assertGreater(max(int(naive_ulps.max()), int(tpu_ulps.max())), 1)

    angles_ref32_8191 = (np.float64(8191) * inv_freq_ref64).astype(np.float32)
    naive_angle_ulps_8191 = _float32_ulp_diff(np.float32(8191) * naive_host_f32, angles_ref32_8191)
    tpu_angle_ulps_8191 = _float32_ulp_diff(np.float32(8191) * tpu_exp2_log2_f32, angles_ref32_8191)
    self.assertGreater(int(naive_angle_ulps_8191.max()), 1)
    self.assertGreaterEqual(int(tpu_angle_ulps_8191.max()), 5)

    tpu_rel_err_8191 = np.max(
        np.abs(np.float64(8191) * tpu_exp2_log2_f32 - np.float64(8191) * inv_freq_ref64)
        / (np.float64(8191) * inv_freq_ref64)
    )
    fixed_rel_err_8191 = np.max(
        np.abs(np.float64(8191) * actual_inv_freq - np.float64(8191) * inv_freq_ref64)
        / (np.float64(8191) * inv_freq_ref64)
    )
    self.assertGreater(tpu_rel_err_8191, fixed_rel_err_8191 * 5.0)


if __name__ == "__main__":
  absltest.main()
