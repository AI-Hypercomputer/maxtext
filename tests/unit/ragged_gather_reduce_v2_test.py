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

"""Tests for the bfloat16 output of the SparseCore ragged gather-reduce kernel."""

import unittest
from unittest import mock

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from maxtext.kernels.ragged import ragged_gather_reduce_v2 as rgr
import numpy as np
import pytest


def _bits(x):
  return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))


class Bf16RoundingTest(unittest.TestCase):
  """The in-kernel float32 -> bfloat16 rounding must match astype(bfloat16) bit for bit."""

  def test_matches_astype(self):
    rng = np.random.default_rng(0)
    random_bits = rng.integers(0, 2**32, size=1 << 20, dtype=np.uint64).astype(np.uint32)
    special = np.array(
        [
            0x00000000,  # +0
            0x80000000,  # -0
            0x00000001,  # smallest subnormal
            0x807FFFFF,  # largest negative subnormal
            0x3F808000,  # tie, even -> round down
            0x3F818000,  # tie, odd -> round up
            0x3F80FFFF,
            0x7F7FFFFF,  # max finite -> inf
            0xFF7FFFFF,  # -max finite -> -inf
            0x7F800000,  # inf
            0xFF800000,  # -inf
        ],
        dtype=np.uint32,
    )
    bits = np.concatenate([random_bits, special])
    x = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.float32)
    got = np.asarray(rgr._round_f32_bits_to_bf16_bits(jnp.asarray(bits))).astype(np.uint16)  # pylint: disable=protected-access
    want = _bits(x.astype(jnp.bfloat16))
    not_nan = ~np.isnan(np.asarray(x))
    np.testing.assert_array_equal(got[not_nan], want[not_nan])
    # NaN stays NaN with its sign (payload is not compared).
    got_bf16 = jax.lax.bitcast_convert_type(jnp.asarray(got), jnp.bfloat16)
    self.assertTrue(np.all(np.isnan(np.asarray(got_bf16, np.float32)[~not_nan])))
    np.testing.assert_array_equal(got[~not_nan] >> 15, want[~not_nan] >> 15)


class UnpackColumnsTest(unittest.TestCase):
  """_unpack_bf16_columns inverts the kernel's layout: word j of a chunk = (col j, col j + chunk // 2)."""

  def test_round_trip(self):
    rows, col_chunk_size, num_chunks = 5, 512, 3
    half = col_chunk_size // 2
    x = jax.random.normal(jax.random.PRNGKey(0), (rows, num_chunks * col_chunk_size), jnp.bfloat16)
    b = _bits(x).astype(np.uint32).reshape(rows, num_chunks, col_chunk_size)
    packed = (b[..., :half] | (b[..., half:] << 16)).reshape(rows, num_chunks * half)
    out = rgr._unpack_bf16_columns(jnp.asarray(packed), col_chunk_size)  # pylint: disable=protected-access
    self.assertEqual(out.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(_bits(out), _bits(x))


class PackedKernelLoweringTest(unittest.TestCase):
  """Lowers the SparseCore kernel for TPU7x from CPU (no TPU needed); SparseCore compilation is not covered."""

  def _lower(self, bf16_output):
    """Returns the TPU StableHLO text of the kernel call at DeepSeek-V3 MoE shapes."""
    amesh = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"))
    args = (
        jax.ShapeDtypeStruct((4096, 7168), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096 * 8,), jnp.int32),
        jax.ShapeDtypeStruct((4096 * 8,), jnp.float32),
        jax.ShapeDtypeStruct((4096 * 8,), jnp.bool_),
    )
    fn = jax.jit(lambda *a: rgr.ragged_gather_reduce(*a, reduce_group_size=8, bf16_output=bf16_output))
    with mock.patch.object(rgr.pltpu, "is_tpu_device", return_value=True), jax.sharding.use_abstract_mesh(amesh):
      return fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text()

  def test_bf16_output_lowers_to_packed_kernel(self):
    # Hidden 7168 as bfloat16 packs into 3584 uint32 words per row (plus the kernel's extra garbage row).
    self.assertNotIn("x3584xui32>", self._lower(bf16_output=False))
    self.assertIn("tensor<4097x3584xui32>", self._lower(bf16_output=True))


class RaggedGatherReduceBf16OutputTest(parameterized.TestCase):
  """bf16_output=True gives bitwise the same result as the float32-output kernel plus a cast."""

  def _check(self, shape, reduce_group_size, topk_dtype=jnp.float32, scale=1.0):
    """Runs the kernel with and without bf16_output on ``shape = (num_rows, hidden, input_size)``."""
    num_rows, hidden, input_size = shape
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    x = (jax.random.normal(k[0], (num_rows, hidden), jnp.float32) * scale).astype(jnp.bfloat16)
    indices = jax.random.randint(k[1], (input_size,), 0, num_rows, jnp.int32)
    weights = jax.random.uniform(k[2], (input_size,), jnp.float32).astype(topk_dtype)
    mask = jax.random.bernoulli(k[3], 0.8, (input_size,))

    def run(bf16_output):
      return jax.jit(rgr.ragged_gather_reduce, static_argnames=("reduce_group_size", "bf16_output"))(
          x, indices, weights, mask, reduce_group_size=reduce_group_size, bf16_output=bf16_output
      )

    ref, got = run(False), run(True)
    self.assertEqual(got.dtype, ref.dtype)
    self.assertEqual(got.shape, ref.shape)
    np.testing.assert_array_equal(_bits(got), _bits(ref))

  def test_cpu_fallback_accepts_flag(self):
    # Off TPU both calls take the JAX fallback; this only checks the flag is accepted end to end.
    self._check((64, 256, 128), reduce_group_size=8)

  @parameterized.named_parameters(
      ("topk8_dsv3_hidden", 4096, 7168, 4096 * 8, 8),
      ("topk8_hidden2048", 2048, 2048, 2048 * 8, 8),
      ("group1", 4096, 7168, 4096, 1),
      ("ragged_rows", 3000, 7168, 3000 * 8, 8),
  )
  @pytest.mark.tpu_only
  def test_matches_f32_output(self, num_rows, hidden, input_size, reduce_group_size):
    self._check((num_rows, hidden, input_size), reduce_group_size)

  @pytest.mark.tpu_only
  def test_matches_f32_output_bf16_weights(self):
    self._check((4096, 7168, 4096 * 8), 8, topk_dtype=jnp.bfloat16)

  @pytest.mark.tpu_only
  def test_matches_f32_output_tiny_values(self):
    # Sums near the float32 subnormal range: checks the in-kernel rounding against XLA's convert there too.
    self._check((4096, 7168, 4096 * 8), 8, scale=1e-37)

  @pytest.mark.tpu_only
  def test_matches_f32_output_large_values(self):
    # Sums near the bfloat16 max: rounding to inf must match.
    self._check((4096, 7168, 4096 * 8), 8, scale=3e37)


if __name__ == "__main__":
  unittest.main()
