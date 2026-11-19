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

"""TPU tests: gmm_pair_merged_dlhs (moe_wi_dlhs_merged_k) matches two gmm_v2 GMMs."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.megablox import ops
import pytest

pytestmark = pytest.mark.tpu_only  # Pallas TPU kernels.

M, D, F, E = 2048, 1024, 704, 4  # F = 704 is not a multiple of 128, like Gemma 4's expert MLP.
GROUP_SIZES = np.array([300, 0, 517, 700], np.int32)  # sum 1517 < M: padding rows.


def _rand(key, shape, scale=1.0):
  return (jax.random.normal(key, shape, jnp.float32) * scale).astype(jnp.bfloat16)


def _f32(x):
  return np.asarray(x.astype(jnp.float32))


class GmmPairMergedDlhsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.devices()[0].platform, "tpu")
    keys = jax.random.split(jax.random.key(0), 5)
    self.gs = jnp.asarray(GROUP_SIZES)
    self.lhs = _rand(keys[0], (M, D))
    self.rhs0 = _rand(keys[1], (E, D, F), 0.05)
    self.rhs1 = _rand(keys[2], (E, D, F), 0.05)
    self.dout0 = _rand(keys[3], (M, F))
    self.dout1 = _rand(keys[4], (M, F))

  @parameterized.product(dlhs_tile_k=(768, 256), transpose_rhs=(True, False))
  def test_matches_two_gmms(self, dlhs_tile_k, transpose_rhs):
    """Forward and weight gradients are bit-identical; the input gradient is one accumulation instead of two."""
    tiling = (512, D, 768, 512, dlhs_tile_k, D, 512, D, 768)

    def two_gmms(lhs, rhs0, rhs1):
      return tuple(
          ops.gmm(
              lhs,
              rhs,
              self.gs,
              jnp.bfloat16,
              tiling,
              use_tokamax_backend=True,
              use_gmm_v2=True,
              use_dlhs_transpose_rhs=transpose_rhs,
          )
          for rhs in (rhs0, rhs1)
      )

    def merged(lhs, rhs0, rhs1):
      return ops.gmm_pair_merged_dlhs(
          lhs, rhs0, rhs1, self.gs, jnp.bfloat16, tiling, use_dlhs_transpose_rhs=transpose_rhs
      )

    args = (self.lhs, self.rhs0, self.rhs1)
    cot = (self.dout0, self.dout1)
    want_out, want_vjp = jax.vjp(jax.jit(two_gmms), *args)
    got_out, got_vjp = jax.vjp(jax.jit(merged), *args)
    want_dlhs, want_drhs0, want_drhs1 = want_vjp(cot)
    got_dlhs, got_drhs0, got_drhs1 = got_vjp(cot)

    for got, want in zip((*got_out, got_drhs0, got_drhs1), (*want_out, want_drhs0, want_drhs1)):
      np.testing.assert_array_equal(_f32(got), _f32(want))

    # f32 reference for dlhs = dout0 @ rhs0^T + dout1 @ rhs1^T on the valid rows; padding rows stay zero.
    valid = int(GROUP_SIZES.sum())
    dout = jnp.concatenate(cot, axis=-1).astype(jnp.float32)
    rhs = jnp.concatenate((self.rhs0, self.rhs1), axis=-1).astype(jnp.float32)
    ref = jax.lax.ragged_dot(dout[:valid], rhs.swapaxes(1, 2), self.gs, precision=jax.lax.Precision.HIGHEST)
    ref = np.asarray(ref)
    got_dlhs, want_dlhs = _f32(got_dlhs), _f32(want_dlhs)
    np.testing.assert_array_equal(got_dlhs[valid:], 0.0)
    np.testing.assert_allclose(got_dlhs[:valid], ref, rtol=2e-2, atol=2e-2 * float(np.max(np.abs(ref))))
    # One f32 accumulation and one bf16 rounding instead of two bf16 partials added in bf16.
    self.assertLessEqual(np.max(np.abs(got_dlhs[:valid] - ref)), np.max(np.abs(want_dlhs[:valid] - ref)))


if __name__ == "__main__":
  absltest.main()
